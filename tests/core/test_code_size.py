"""Responsibility-based source classification and physical line statistics."""

from collections import Counter
from io import BytesIO
from pathlib import Path
import subprocess
import tarfile

import pytest

from perf_benchmark.plugins import create_registry


# Exact files keep new modules/directories core until their responsibility is reviewed.
NON_CORE = {
    "application": (
        "CLI parsing, configuration loading, entrypoints, read-only result views and package version",
        ("__init__.py", "__main__.py", "app/__init__.py", "app/main.py", "app/config.py", "app/report.py",
         "app/results/__init__.py", "app/results/http.py", "app/results/service.py",
         "app/results/queries.py", "app/results/projections.py", "app/results/statistics.py",
         "app/results/web.py",  # Static UI assets/navigation; no measurement or result lifecycle.
         "app/control.py", "app/tui.py", "app/live_tui.py", "app/experiments.py",
         "app/results/compact.py",  # Offline derived-file maintenance; native facts stay intact.
         "static_benchmark/__init__.py", "static_benchmark/config.py", "static_benchmark/cli.py",
         "static_benchmark/runner.py",  # Orchestration of existing native Runs, no HTTP/metric implementation.
         "static_benchmark/metrics.py", "static_benchmark/excel_exporter.py",
         "static_benchmark/prompts.py", "static_benchmark/token_sampler.py"),
    ),
    "integration": (
        "Prometheus/OTLP conversion, standalone observation consumer and offline Nsight import",
        ("app/observe.py", "integrations/__init__.py", "integrations/observation_export.py",
         "app/nsight.py", "app/results/observation.py", "app/evidence.py", "app/preparation.py",
         "app/service_capacity.py", "app/reasoning.py"),  # Capability preparation stays outside measured requests.
    ),
    "hook": (
        "Concrete request-parameters Hook and its package marker",
        ("hooks/__init__.py", "hooks/request_parameters.py"),
    ),
    "corpus": (
        "Independent offline dataset tools, outside the Run execution lifecycle",
        ("corpus/__init__.py", "corpus/cache.py", "corpus/common.py", "corpus/compare.py", "corpus/generation.py",
         "corpus/inspect.py",
         "corpus/profile.py", "corpus/requests.py", "corpus/service.py", "corpus/source.py", "corpus/synthetic.py",
         "corpus/text_workloads.py",  # Offline embedding/rerank inputs; no Run lifecycle or formal metrics.
         "corpus/trace_prepare.py", "corpus/trace_sources.py", "corpus/trace_servegen.py",
         "corpus/trace_text.py",
         "corpus/rust/src/lib.rs"),
    ),
}

# HEAD may still use the previous paths. Classify their original responsibilities
# when reporting a move, without accepting those paths in the current tree.
HEAD_PATHS = {
    "app/results.py": "application",
    "app/results_statistics.py": "application",
    "app/results_observation.py": "integration",
    "runtime/observation_export.py": "integration",
}


def _source(path):
    return path.suffix in {".py", ".pyi", ".sql", ".go", ".rs"} and not path.name.endswith("_test.go") and "target" not in path.parts and "tests" not in path.parts


def _category(path, concrete):
    for category, (_, paths) in NON_CORE.items():
        if path.as_posix() in paths:
            return category
    if any(parent in concrete for parent in path.parents):
        return "plugin"
    return "core"


def _counts(files, concrete):
    counts = Counter()
    for path, lines in files.items():
        counts[_category(path, concrete)] += lines
    return counts


def _head_files(root):
    result = subprocess.run(
        ["git", "archive", "HEAD", "perf_benchmark"], cwd=root.parent,
        capture_output=True, check=False,
    )
    if result.returncode:
        return None  # Source archives without Git history have no comparison baseline.
    with tarfile.open(fileobj=BytesIO(result.stdout)) as archive:
        files = {}
        for member in archive.getmembers():
            path = Path(member.name).relative_to("perf_benchmark")
            if member.isfile() and _source(path):
                source = archive.extractfile(member)
                assert source is not None
                with source:
                    files[path] = len(source.read().decode().splitlines())
        return files


def test_source_classification_and_line_statistics(pytestconfig):
    root = Path(__file__).resolve().parents[2] / "perf_benchmark"
    registry = create_registry()
    concrete = {
        Path(*registry.resolve(stage, name).factory.__module__.split(".")[1:4])
        for stage in registry.STAGES
        for name in registry.names(stage)
    }
    files = {
        path.relative_to(root): len(path.read_text().splitlines())
        for path in root.rglob("*") if path.is_file() and _source(path)
    }
    exclusions = [path for _, paths in NON_CORE.values() for path in paths]
    assert len(exclusions) == len(set(exclusions)), "overlapping non-core classifications"
    assert all(Path(path) in files for path in exclusions), "stale non-core file classification"
    counts = _counts(files, concrete)
    head = _head_files(root)
    baseline = _counts(head, concrete) if head is not None else None
    if head is not None:
        for path, category in HEAD_PATHS.items():
            lines = head.get(Path(path), 0)
            baseline["core"] -= lines
            baseline[category] += lines
    report = ["physical source line statistics"]
    for category in ("core", *NON_CORE, "plugin"):
        delta = f"{counts[category] - baseline[category]:+d}" if baseline is not None else "unavailable"
        report.append(f"{category}={counts[category]} (delta vs HEAD, current rules: {delta})")
    report.append(f"all production source={sum(counts.values())}")
    reporter = pytestconfig.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line("\n" + "\n".join(report))


@pytest.mark.parametrize("path", [
    "app/new_ui/widget.py", "hooks/new_hook.py", "runtime/new_transport.go",
    "plugins/shared.py", "plugins/executor/unregistered/client.py", "new_area/worker.sql",
    "runtime/observation.py", "core/observation.py", "pipeline/contracts/observation.py",
    "runtime/http/rusthttp/src/new.rs", "new_area/worker.rs", "_native.pyi",
])
def test_unclassified_source_and_observation_lifecycle_remain_core(path):
    assert _category(Path(path), {Path("plugins/executor/openai_chat")}) == "core"


def test_concrete_implementations_have_separate_counts():
    assert _category(Path("plugins/executor/openai_chat/rust/src/lib.rs"),
                     {Path("plugins/executor/openai_chat")}) == "plugin"
    assert _category(Path("integrations/observation_export.py"), set()) == "integration"
    assert _category(Path("hooks/request_parameters.py"), set()) == "hook"


def test_rust_production_and_test_source_boundaries():
    assert _source(Path("runtime/http/rusthttp/src/http.rs"))
    assert _source(Path("plugins/executor/openai_chat/rust/src/decoder.rs"))
    assert not _source(Path("runtime/http/rusthttp/tests/http.rs"))
    assert not _source(Path("target/debug/build/generated.rs"))
