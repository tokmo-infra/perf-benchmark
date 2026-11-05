from io import StringIO
from threading import current_thread, main_thread

import pytest
from rich.console import Console

from perf_benchmark.core.models import Run
from perf_benchmark.pipeline.contracts.results import RunSummary, TrialRecord, MetricWindowRecord, MetricRecord, ComparisonRecord
from perf_benchmark.plugins.exporter.console import ConsoleExporter, parse_settings


def _point(unit, statistics, *, availability="available", reason=None, metadata=None) -> MetricRecord:
    return {
        "unit": unit,
        "formula_version": "test",
        "availability": availability,
        "unavailable_reason": reason,
        "source": "test",
        "statistics": statistics,
        "statistic_metadata": metadata or {},
    }


def test_console_exporter_rejects_settings():
    with pytest.raises(ValueError, match="unknown console exporter setting"):
        parse_settings({"color": True})


def test_console_marks_partial_numeric_totals_and_keeps_unknown_distinct():
    from perf_benchmark.plugins.exporter.console.render import _cell

    assert _cell(_point("tokens", {"value": 25}, availability="partial"), "value", "tokens") == "25*"
    assert _cell(_point("CNY", {"value": 0}, availability="partial"), "value", "CNY") == "0.000*"
    assert _cell(_point("CNY", {"value": None}, availability="unavailable"), "value", "CNY") == "unavailable"
    assert _cell(_point("CNY", {"value": 1}), "value", "CNY") == "1.000"


class Reader:
    location = "/tmp/results.db"

    def run_record(self, run_id):
        raise AssertionError("unexpected full Run read")

    def measurements(self, run_id, *, scope, scope_id):
        return {}

    def request_page(self, run_id, **kwargs):
        raise AssertionError("unexpected request read")

    def grouped_metrics(self, run_id, trial_id):
        raise AssertionError("unexpected grouped metric read")

    def metric_series(self, run_id, **kwargs):
        raise AssertionError("unexpected time series read")

    def run_ids(self):
        return ("run-console",)

    def iter_turn_records(self, run_id):
        raise AssertionError("console must read aggregate views")

    def iter_attempt_records(self, run_id):
        raise AssertionError("console must read aggregate views")

    def read_updates(self, cursor=None, limit=500):
        raise AssertionError("console must read aggregate views")

    def summary(self, run_id) -> RunSummary:
        return {
            "run_id": run_id,
            "name": "table-test",
            "status": "succeeded",
            "error": None,
            "trial_count": 1,
            "session_count": 10,
            "turn_count": 10,
            "metric_count": 100,
        }

    def trials(self, run_id) -> tuple[TrialRecord, ...]:
        return (
            {
                "trial_id": "trial-1",
                "name": "batch-128-repetition-1",
                "ordinal": 0,
                "status": "succeeded",
                "started_at": None,
                "finished_at": None,
                "error": None,
                "parameters": {"plugin_private": True},
                "metadata": {},
                "session_count": 10,
                "turn_count": 10,
                "load_profile": {
                    "mode": "closed-loop",
                    "target_concurrency": 2,
                    "target_session_rate": None,
                    "target_request_rate": None,
                },
                "resource_limits": {
                    "ready_capacity": 2,
                    "max_active_sessions": 2,
                    "max_concurrency": 2,
                    "max_pending_sessions": 2,
                    "max_pending_requests": 2,
                },
                "comparison": {
                    "group_id": "batch-128",
                    "axis_name": "batch_size",
                    "axis_value": 128,
                    "axis_unit": "requests",
                },
                "repetition": {"index": 1, "expected": 3},
                "scheduling": {},
            },
        )

    def metric_windows(self, run_id, *, scope, scope_id) -> tuple[MetricWindowRecord, ...]:
        return tuple(
            {
                "window": window,
                "start_ns": 0 if window == "overall" else None,
                "end_ns": 1_000_000_000 if window == "overall" else None,
                "duration_s": 1.0 if window == "overall" else None,
                "availability": "available" if window == "overall" else "unavailable",
                "unavailable_reason": None if window == "overall" else "target_not_reached",
            }
            for window in (
                "overall",
                "last30s",
                "drop_first_20pct",
                "steady_occupancy",
                "steady_strip_10pct",
            )
        )

    def metric_points(self, run_id, *, scope, scope_id, window, names=None):
        if scope == "comparison":
            return {
                "workload_rps": _point(
                    "req/s",
                    {"value_median": 3.1},
                    metadata={
                        "value_median": {
                            "availability": "available",
                            "valid_sample_count": 3,
                            "expected_sample_count": 3,
                        }
                    },
                ),
                "ttft": _point("s", {"p50_median": 0.1}),
                "tpot": _point("ms", {"p50_median": 5.0}),
                "workload_completion": _point("tok/s", {"value_median": 40.0}),
            }
        if window != "overall":
            return {
                "ttft": _point("s", {}, availability="unavailable", reason="target_not_reached")
            }
        points = {
            "succeeded": _point("requests", {"value": 10.0}),
            "workload_rps": _point("req/s", {"value": 3.25}),
            "workload_completion": _point("tok/s", {"value": 42.0}),
            "ttft": _point(
                "s",
                {"n": 10.0, "eligible_n": 10.0, "am": 0.11, "p50": 0.1, "p95": 0.15},
            ),
            "tpot": _point("ms", {"p50": 5.0}),
            "target_concurrency": _point("requests", {"value": 2.0}),
        }
        return points if names is None else {name: points[name] for name in names if name in points}

    def sweep_records(self, run_id):
        return ()

    def comparison_points(self, run_id) -> tuple[ComparisonRecord, ...]:
        return (
            {
                "group_id": "batch-128",
                "axis_name": "batch_size",
                "axis_value": 128,
                "axis_unit": "requests",
            },
        )


async def test_console_renders_trial_window_and_sweep_tables(stage_context) -> None:
    output = StringIO()
    console = Console(file=output, width=180, force_terminal=False, color_system=None)
    run = Run(id="run-console", name="console")

    result = await ConsoleExporter({}, console=console).export(
        run.id,
        Reader(),
        stage_context("exporter", run=run),
    )

    text = output.getvalue()
    assert "Benchmark Summary" in text
    assert "Trial Comparison" in text
    assert "Load Overview" in text
    assert "Latency & Throughput Statistics" in text
    assert "Request, Session, Cache & Concurrency" in text
    assert "Window Availability" in text
    assert "Comparison Robust Summary" in text
    assert "Batch size" in text
    assert "128 requests" in text
    assert "TTFT" in text
    assert "0.100" in text
    assert "target_not_reached" in text
    assert "/tmp/results.db" in text
    assert result.outputs[0].location == "stdout"


async def test_console_reads_and_renders_plugin_metrics_off_event_loop(stage_context) -> None:
    threads = []

    class PluginReader(Reader):
        def summary(self, run_id) -> RunSummary:
            threads.append(current_thread())
            return super().summary(run_id)

        def metric_windows(self, run_id, *, scope, scope_id) -> tuple[MetricWindowRecord, ...]:
            return (
                {
                    "window": "custom_window", "start_ns": 0, "end_ns": 1_000_000_000,
                    "duration_s": 1.0,
                    "availability": "available",
                    "unavailable_reason": None,
                },
            )

        def metric_points(self, run_id, *, scope, scope_id, window, names=None):
            return {"custom_score": _point("points", {"value": 123.0})}

        def comparison_points(self, run_id) -> tuple[ComparisonRecord, ...]:
            return ()

    output = StringIO()
    await ConsoleExporter({}, console=Console(file=output, width=180)).export(
        "run",
        PluginReader(),
        stage_context("exporter"),
    )
    assert threads and all(thread is not main_thread() for thread in threads)
    assert "Custom score" in output.getvalue()
    assert "custom_window" in output.getvalue()
    assert "123" in output.getvalue()


async def _export_persisted_evidence(tmp_path, stage_context, evidence):
    from perf_benchmark.pipeline.results.saver import SaverRuntime
    from perf_benchmark.plugins.saver.sqlite import SQLiteSaver

    run = Run(id="run-evidence", name="cancelled evidence")
    run.prepare()
    run.start()
    sink = await SaverRuntime.create(
        SQLiteSaver({"path": str(tmp_path / "evidence.db")}),
        run, stage_context("saver", run=run),
    )
    try:
        await sink.write_run_measurement("evidence", evidence)
        run.finish("safety_stop")
        reader = await sink.finalize(run)
        before = reader.summary(run.id)
        output = StringIO()
        result = await ConsoleExporter(
            {}, console=Console(file=output, width=240, color_system=None),
        ).export(run.id, reader, stage_context("exporter", run=run))
        assert result.outputs[0].location == "stdout"
        assert reader.measurements(run.id, scope="run", scope_id=run.id)["evidence"] == evidence
        assert reader.summary(run.id) == before
        assert before["status"] == "cancelled"
        assert before["error"] == "safety_stop"
        return output.getvalue()
    finally:
        await sink.close()


async def test_console_renders_bare_persisted_conditions_without_claiming_verification(
    tmp_path, stage_context,
):
    # Coordinator accepts JSON metadata, not only the application's status/value rows.
    conditions = {
        "config_sha256": "a" * 64,
        "dataset_sha256": "b" * 64,
        "policy": {"start_percent": 40},
        "task": "[bold red]cpu fixture[/bold red]",
        "enabled": True,
        "retries": 0,
        "cache": None,
        "pools": ["p", "d"],
        "missing_status": {"value": True},
        "missing_value": {"status": "observed"},
        "invalid_status": {"status": 0, "value": True},
    }
    text = await _export_persisted_evidence(
        tmp_path, stage_context, {"conditions": conditions},
    )
    for key, value in conditions.items():
        line = next(line for line in text.splitlines() if key in line)
        assert "unverified" in line
        assert str(value) in line
    assert "原始数据内容一致性未验证" in text


@pytest.mark.parametrize("status,value", [("observed", True), ("declared", False), ("unknown", None)])
async def test_console_preserves_structured_persisted_condition_status(
    tmp_path, stage_context, status, value,
):
    text = await _export_persisted_evidence(
        tmp_path, stage_context,
        {"conditions": {"stream": {"status": status, "value": value}}},
    )
    line = next(line for line in text.splitlines() if "stream" in line)
    assert status in line
    assert str(value) in line
