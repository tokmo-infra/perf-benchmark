"""Architecture dependency rules for Stage Plugin locality."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "perf_benchmark"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    values: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            values.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            values.add(node.module)
    return values


def test_core_modules_do_not_import_concrete_plugins() -> None:
    violations = []
    for area in ("core", "pipeline", "runtime"):
        for path in (PACKAGE / area).rglob("*.py"):
            for imported in _imports(path):
                if imported.startswith("perf_benchmark.plugins."):
                    violations.append((str(path.relative_to(ROOT)), imported))
    assert violations == []


def test_concrete_plugins_do_not_import_other_concrete_plugins() -> None:
    violations = []
    plugin_root = PACKAGE / "plugins"
    for path in plugin_root.glob("*/*/**/*.py"):
        relative = path.relative_to(plugin_root)
        own_prefix = f"perf_benchmark.plugins.{relative.parts[0]}.{relative.parts[1]}"
        for imported in _imports(path):
            if imported.startswith("perf_benchmark.plugins.") and not (
                imported == "perf_benchmark.plugins" or imported.startswith(own_prefix)
            ):
                violations.append((str(path.relative_to(ROOT)), imported))
    assert violations == []


def test_result_stages_do_not_decode_scheduler_private_settings() -> None:
    forbidden = {
        "arrival_distribution",
        "continue_on_cancel",
        "control",
        "max_active_sessions",
        "max_concurrency",
        "max_pending_requests",
        "max_pending_sessions",
        "on_exhausted",
        "repetitions",
        "request_rate",
        "retain_partial",
        "retry",
        "seed",
        "session_rate",
        "stop",
        "sweep_point_id",
        "expected_repetitions",
        "demand_limited",
        "drain_duration_ms",
    }
    violations = []
    for area in (
        PACKAGE / "pipeline" / "results" / "metrics.py",
        PACKAGE / "plugins" / "metrics",
        PACKAGE / "plugins" / "saver",
        PACKAGE / "plugins" / "exporter",
    ):
        paths = area.rglob("*.py") if area.is_dir() else (area,)
        for path in paths:
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Subscript):
                    owner, key = node.value, node.slice
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get"
                    and node.args
                ):
                    owner, key = node.func.value, node.args[0]
                else:
                    continue
                if not (
                    isinstance(owner, ast.Attribute)
                    and owner.attr == "parameters"
                    and isinstance(key, ast.Constant)
                    and key.value in forbidden
                ):
                    continue
                violations.append((str(path.relative_to(ROOT)), key.value))
    assert violations == []


def test_stage_contract_methods_do_not_hide_capabilities_in_any_or_getattr() -> None:
    methods = {
        "loader.py": ("LoaderPlugin", "open"),
        "normalization.py": ("NormalizationPlugin", "normalize_batch"),
        "selector.py": ("SelectorPlugin", "select"),
        "executor.py": ("ExecutorPlugin", "create_runtime"),
        "scheduler.py": ("SchedulerPlugin", "schedule"),
        "metrics.py": ("MetricsPlugin", "create_backend"),
        "saver.py": ("SaverPlugin", "create_backend"),
        "exporter.py": ("ExporterPlugin", "export"),
    }
    contracts = PACKAGE / "pipeline" / "contracts"
    violations = []
    for filename, (class_name, method_name) in methods.items():
        tree = ast.parse((contracts / filename).read_text(), filename=filename)
        owner = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        method = next(
            node for node in owner.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == method_name
        )
        if any(isinstance(node, ast.Name) and node.id == "Any" for node in ast.walk(method)):
            violations.append((filename, class_name, method_name, "Any"))
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            for node in ast.walk(method)
        ):
            violations.append((filename, class_name, method_name, "getattr"))
    assert violations == []
