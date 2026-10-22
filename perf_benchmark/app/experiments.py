"""Serial Run YAML composition and derived cross-Run reports."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
from uuid import uuid4
from typing import cast
from perf_benchmark.pipeline.contracts import MetricsPlugin, SaverPlugin

from jinja2 import Environment
import yaml

from perf_benchmark.app.config import load_config
from perf_benchmark.app.main import execute_config
from perf_benchmark.plugins import create_registry
from perf_benchmark.runtime.cancellation import CancellationToken


def compare_files(baseline, candidate, specification):
    registry = create_registry()
    saver = cast(SaverPlugin, registry.bind("saver", "sqlite", {"path": "unused.db"}).plugin)
    metrics = cast(MetricsPlugin, registry.bind("metrics", "standard", {}).plugin)
    return metrics.compare(saver.open_reader(str(baseline)), saver.open_reader(str(candidate)), specification)


def comparison_html(result, destination):
    template = Environment(autoescape=True).from_string('''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>实验比较</title><style>body{font:16px system-ui;max-width:1100px;margin:3rem auto;padding:1rem}table{border-collapse:collapse;width:100%;margin:1rem 0}td,th{text-align:left;border-bottom:1px solid #ccc;padding:.6rem}pre{white-space:pre-wrap}</style>
<h1>跨 Run 比较</h1><p>{{ result.baseline_run }} → {{ result.candidate_run }}</p>
<p>{{ result.source_content }}；直接可比：{{ result.directly_comparable }}；负载有效：{{ result.load_qualified }}</p>
<h2>重复资格</h2><pre>{{ repetitions }}</pre>
<table><tr><th>指标</th><th>基线</th><th>候选</th><th>差值</th><th>判定</th></tr>
{% for row in result.metrics %}<tr><td>{{ row.metric }} / {{ row.statistic }} / {{ row.window }}</td><td>{{ row.baseline }} {{ row.unit }}</td><td>{{ row.candidate }}</td><td>{{ row.delta }}</td><td>{{ row.decision }} {{ row.reason or '' }}</td></tr>{% endfor %}</table>
<h2>条件</h2><table><tr><th>条件</th><th>基线</th><th>候选</th><th>已核实</th><th>允许变化</th></tr>{% for row in result.conditions %}<tr><td>{{ row.condition }}</td><td>{{ row.baseline }}</td><td>{{ row.candidate }}</td><td>{{ row.verified }}</td><td>{{ row.allowed_change }}</td></tr>{% endfor %}</table>
<h2>质量关联</h2><pre>{{ quality }}</pre></html>''')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name('.' + destination.name + '.' + uuid4().hex)
    try:
        temporary.write_text(template.render(result=result, quality=json.dumps(result["quality"], ensure_ascii=False, indent=2),
                                         repetitions=json.dumps(result["repetitions"], ensure_ascii=False, indent=2)))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


async def execute(path):
    path = Path(path).resolve(strict=True)
    manifest = yaml.safe_load(await asyncio.to_thread(path.read_text))
    if not isinstance(manifest, dict) or set(manifest) - {"runs", "comparisons", "output"}:
        raise ValueError("experiment supports runs, comparisons and output")
    items = manifest.get("runs")
    if not isinstance(items, list) or not items:
        raise ValueError("experiment requires a nonempty runs list")
    prepared, names = [], set()
    for item in items:
        if not isinstance(item, dict) or set(item) != {"name", "config"} or not isinstance(item["name"], str) or not item["name"] or item["name"] in names:
            raise ValueError("each experimental Run needs a unique name and config")
        names.add(item["name"])
        config_path = (path.parent / item["config"]).resolve(strict=True)
        config, base = await asyncio.to_thread(load_config, config_path)
        prepared.append((item["name"], config_path, config, base))
    comparisons = manifest.get("comparisons", [])
    if not isinstance(comparisons, list) or any(
        not isinstance(rule, dict) or rule.get("baseline") not in names or rule.get("candidate") not in names
        for rule in comparisons
    ):
        raise ValueError("comparison must refer to experiment Run names")
    root = (path.parent / manifest.get("output", "experiments") / uuid4().hex).resolve()
    await asyncio.to_thread(root.mkdir, parents=True, exist_ok=False)
    index = [{"name": name, "config": str(config_path), "sqlite": str(root / f"run-{i+1}.sqlite")}
             for i, (name, config_path, _, _) in enumerate(prepared)]
    # Locations only: pending/completed/failed state is derived from existing SQLite files.
    await asyncio.to_thread((root / "index.yaml").write_text, yaml.safe_dump({"runs": index, "comparisons": manifest.get("comparisons", [])}, allow_unicode=True))
    cancellation = CancellationToken()
    for row, (_, _, config, base) in zip(index, prepared):
        config = deepcopy(config)
        config["pipeline"]["saver"] = {"plugin": "sqlite", "settings": {"path": row["sqlite"]}}
        # A separate output location prevents one Run's HTML from overwriting another's.
        exporters = config["pipeline"].get("exporter", {}).get("plugins", [])
        for binding in exporters:
            if binding.get("plugin") == "html":
                binding.setdefault("settings", {})["output"] = row["sqlite"] + ".html"
        await execute_config(config, base, cancellation)
        if cancellation.is_cancelled():
            break
    if not cancellation.is_cancelled():
        locations = {item["name"]: item["sqlite"] for item in index}
        for i, rule in enumerate(comparisons):
            spec = {k: v for k, v in rule.items() if k not in {"baseline", "candidate"}}
            result = await asyncio.to_thread(compare_files, locations[rule["baseline"]], locations[rule["candidate"]], spec)
            await asyncio.to_thread(comparison_html, result, root / f"comparison-{i+1}.html")
    return root / "index.yaml"
