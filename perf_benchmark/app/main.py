from __future__ import annotations

import argparse
import asyncio
import signal
import json
import sys
from pathlib import Path

import yaml
if sys.platform == "win32":
    from asyncio import run as run_async
else:
    from uvloop import run as run_async

from perf_benchmark import __version__
from perf_benchmark.app.config import load_config
from perf_benchmark.core.errors import PipelineError
from perf_benchmark.pipeline.coordinator import Coordinator


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="perf-benchmark")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("config", type=Path)
    run.add_argument("--output-database", type=Path)
    run.add_argument("--tui", action="store_true")
    run.add_argument("--prometheus-url", default="http://127.0.0.1:9090")
    tui = commands.add_parser("tui")
    tui.add_argument("--url", default="http://127.0.0.1:8088")
    experiment = commands.add_parser("experiment")
    experiment.add_argument("config", type=Path)
    compare = commands.add_parser("compare")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--config", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    report = commands.add_parser("report")
    report.add_argument("database", type=Path)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--grafana-url", default="")
    export = commands.add_parser("export")
    export.add_argument("database", type=Path)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--formats", nargs="+", choices=("csv", "xlsx"), default=["xlsx"])
    export.add_argument("--tables", nargs="+", choices=("summary", "minutes", "requests", "kv_capacity"),
                        help="Tables to export (default: summary minutes)")
    observe = commands.add_parser("observe")
    observe.add_argument("database", type=Path)
    observe.add_argument("--config", type=Path)
    results = commands.add_parser("results")
    result_commands = results.add_subparsers(dest="results_command", required=True)
    cleanup = result_commands.add_parser("cleanup")
    cleanup.add_argument("--root", type=Path, required=True)
    compact = result_commands.add_parser("compact")
    compact.add_argument("--root", type=Path, required=True)
    compact.add_argument("--apply", action="store_true")
    compact.add_argument("--compress-metrics", action="store_true")
    archive = result_commands.add_parser("archive")
    archive.add_argument("database", type=Path)
    analyze = result_commands.add_parser("analyze")
    analyze.add_argument("archives", nargs="+", type=Path)
    analyze.add_argument("--sql", required=True)
    analyze.add_argument("--details", action="store_true")
    serve = result_commands.add_parser("serve")
    serve.add_argument("--config-root", type=Path)
    serve.add_argument("--root", type=Path, required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8088)
    serve.add_argument("--prometheus-url", default="http://127.0.0.1:9090")
    serve.add_argument("--loki-url", default="http://127.0.0.1:3100")
    serve.add_argument("--tempo-url", default="http://127.0.0.1:3200")
    serve.add_argument("--grafana-url", help="Browser-visible Grafana URL for diagnostic navigation")
    dataset = commands.add_parser("dataset")
    operations = dataset.add_subparsers(dest="dataset_command", required=True)
    for name in ("stats", "generate", "compare", "prepare-trace", "inspect"):
        operation = operations.add_parser(name)
        operation.add_argument("config", type=Path)
        if name not in {"compare", "inspect"}:
            operation.add_argument("--output", type=Path, required=True)
    static = commands.add_parser("static", help="Run static prefill and decode benchmark")
    static.add_argument("config", type=Path, help="Path to static benchmark YAML config")
    static.add_argument("--only-prefill", action="store_true", help="Run only prefill tests")
    static.add_argument("--only-decode", action="store_true", help="Run only decode tests")
    static.add_argument("--output", type=Path, help="Override output Excel path (.xlsx)")
    static.add_argument("--base-url", help="Override service base URL")
    static.add_argument("--model", help="Override model name")
    static.add_argument("--dry-run", action="store_true", help="Validate and print execution plan without sending requests")
    return root


async def run_config(path: Path, output_database: Path | None = None, *, tui=False,
                     prometheus_url="http://127.0.0.1:9090") -> int:
    config, base_dir = load_config(path)
    display = None
    if tui:
        from perf_benchmark.app.live_tui import LiveDisplay
        from perf_benchmark.core.observation import ObservationSettings
        from perf_benchmark.runtime.observation import ObservationRuntime
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            raise ValueError("--tui requires an interactive terminal")
        config["observation"] = {**config.get("observation", {}), "enabled": True}
        ObservationRuntime.validate(ObservationSettings.parse(config["observation"], base_dir=base_dir))
        display = LiveDisplay(prometheus_url)
    if output_database is not None:
        config["pipeline"]["saver"] = {"plugin": "sqlite", "settings": {"path": str(output_database.resolve())}}
        for binding in config["pipeline"].get("exporter", {}).get("plugins", []):
            if binding.get("plugin") == "html":
                binding.setdefault("settings", {})["output"] = str(output_database.resolve()) + ".html"
    try:
        if display is not None:
            await display.start()
        run = await execute_config(config, base_dir, live_display=display)
    finally:
        if display is not None:
            try:
                await display.stop()
            except Exception as exc:
                print(f"TUI cleanup failed: {exc}", file=sys.stderr)
    print(json.dumps({
        "run_id": run.id,
        "status": run.status,
        "metrics": run.metrics,
        "artifacts": run.artifacts,
    }, ensure_ascii=False))
    return 0 if run.status == "succeeded" else 1


async def execute_config(config, base_dir, cancellation=None, live_display=None):
    from perf_benchmark.app.config import case_description
    from perf_benchmark.app.evidence import conditions, quality
    from perf_benchmark.app.preparation import ServicePreparation, ApplicationPreparation
    from perf_benchmark.runtime.cancellation import CancellationToken

    case = case_description(config)
    application = config.get("application", {})
    if set(application) - {"conditions", "quality", "preparation", "service_capacity"}:
        raise ValueError("unknown application setting")
    token = cancellation or CancellationToken()
    loop = asyncio.get_running_loop()
    previous = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        if sys.platform == "win32":
            previous[sig] = signal.signal(
                sig, lambda *_: loop.call_soon_threadsafe(token.cancel, "user_cancelled"))
        else:
            loop.add_signal_handler(sig, token.cancel, "user_cancelled")
    try:
        from perf_benchmark.app.reasoning import resolve
        config, reasoning = await resolve(config, token)
        evidence = await asyncio.to_thread(conditions, config, application.get("conditions", {}), base_dir)
        evidence["case"] = case
        if reasoning is not None:
            evidence["reasoning"] = reasoning
        evidence["quality"] = await asyncio.to_thread(quality, application.get("quality", {}), config, base_dir)
        from perf_benchmark.app.service_capacity import collect
        evidence["service_capacity"] = await collect(config, application.get("service_capacity", {}), base_dir)
        prepare = ServicePreparation(application["preparation"], config["pipeline"]["executor"]) if "preparation" in application else None
        if prepare is not None or "quality" in application or evidence["quality"]["status"] == "not_applicable":
            prepare = ApplicationPreparation(prepare, evidence["quality"])
        return await Coordinator().execute(config, base_dir=base_dir, cancellation=token, evidence=evidence,
                                           preparation=prepare, live_display=live_display)
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            if sys.platform == "win32":
                signal.signal(sig, previous[sig])
            else:
                loop.remove_signal_handler(sig)


def console_main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "run":
            return run_async(run_config(args.config, args.output_database, tui=args.tui,
                                        prometheus_url=args.prometheus_url))
        if args.command == "tui":
            from perf_benchmark.app.tui import run
            return run_async(run(args.url))
        if args.command == "experiment":
            from perf_benchmark.app.experiments import execute
            print(run_async(execute(args.config)))
            return 0
        if args.command == "compare":
            from perf_benchmark.app.experiments import compare_files, comparison_html
            spec = yaml.safe_load(args.config.read_text())
            result = compare_files(args.baseline, args.candidate, spec)
            comparison_html(result, args.output)
            print(args.output)
            return 0
        if args.command == "report":
            from perf_benchmark.app.report import generate

            result = run_async(generate(args.database, args.output, args.grafana_url))
            print(result.outputs[0].location)
            return 0
        if args.command == "observe":
            from perf_benchmark.app.observe import main

            return main([str(args.database), *(["--config", str(args.config)] if args.config else [])])
        if args.command == "export":
            from perf_benchmark.app.report import export_tables

            result = run_async(export_tables(args.database, args.output, args.formats, tables=args.tables))
            for output in result.outputs:
                print(output.location)
            return 0
        if args.command == "results":
            if args.results_command == "compact":
                from perf_benchmark.app.results.compact import compact

                print(json.dumps(compact(args.root, apply=args.apply,
                                         compress=args.compress_metrics), ensure_ascii=False))
                return 0
            if args.results_command != "serve":
                from perf_benchmark.plugins.saver.sqlite.archive import cleanup, seal
                from perf_benchmark.plugins.saver.sqlite.parquet_reader import analyze
                if args.results_command == "archive":
                    print(seal(args.database))
                elif args.results_command == "analyze":
                    print(json.dumps(analyze(args.archives, args.sql, details=args.details), ensure_ascii=False))
                else:
                    root = args.root.resolve(strict=True)
                    outcomes = []
                    for path in sorted(root.rglob("*.archive")):
                        if not path.is_dir():
                            continue
                        try:
                            outcomes.append(cleanup(path))
                        except (OSError, ValueError) as exc:
                            outcomes.append({"archive": str(path), "state": "protected", "error": type(exc).__name__})
                    print(json.dumps(outcomes, ensure_ascii=False))
                return 0
            from perf_benchmark.app.results import serve

            serve(args.root, args.host, args.port, config_root=args.config_root, prometheus_url=args.prometheus_url, loki_url=args.loki_url, tempo_url=args.tempo_url, grafana_url=args.grafana_url)
            return 0
        if args.command == "dataset":
            if args.dataset_command == "inspect":
                from perf_benchmark.corpus.inspect import inspect
                result = run_async(inspect(args.config))
                print(yaml.safe_dump(result, allow_unicode=True, sort_keys=True))
                return 0
            from perf_benchmark.corpus.service import execute

            result = run_async(execute(args.dataset_command, args.config, getattr(args, "output", None)))
            print(yaml.safe_dump(result, allow_unicode=True, sort_keys=True))
            return 0
        if args.command == "static":
            from perf_benchmark.static_benchmark.cli import run_static_cli
            return run_async(run_static_cli(args))
    except (OSError, TypeError, ValueError, PipelineError, yaml.YAMLError) as exc:
        print(f"perf-benchmark: {exc}", file=sys.stderr)
        return 2
    return 2
