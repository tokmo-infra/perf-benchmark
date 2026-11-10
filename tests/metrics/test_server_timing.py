from dataclasses import replace

import pytest

from perf_benchmark.plugins.metrics.standard.backend import StandardMetricsBackend
from perf_benchmark.plugins.metrics.standard.formulas import request_metrics, request_values
from perf_benchmark.plugins.metrics.standard.maxtpm import latency_points
from tests.metrics.test_groups import fact
from tests.metrics.test_maxtpm_billing import session, turn, trial, occupancy


def make_turn(identifier, generation=90, completion=4, **changes):
    base = fact(identifier, completion=completion)
    return replace(base, server_metrics={"generation_time_ms": generation}, **changes)


def test_per_request_server_metrics_do_not_change_client_metrics():
    one = make_turn("one", output_event_count=1)
    values = request_values(one)
    metrics = request_metrics(one, values)
    assert metrics["tpot"].availability == "unavailable"
    assert metrics["server_generation_time"].statistics["value"] == 90
    assert metrics["server_tpot"].statistics["value"] == 30
    assert metrics["server_tpot"].unit == "ms"
    assert metrics["server_decode_throughput"].statistics["value"] == pytest.approx(1000 / 30)
    assert metrics["server_decode_throughput"].unit == "tok/s"
    assert metrics["server_tpot"].source == "server_generation_time_ms+server_usage"
    client = request_metrics(make_turn("same"), request_values(make_turn("same")))["tpot"]
    no_server = replace(make_turn("same"), server_metrics=None)
    assert request_metrics(no_server, request_values(no_server))["tpot"] == client


@pytest.mark.parametrize("bad", [None, 0, -1, True, False, "90", [], {}, float("nan"), float("inf"), float("-inf"), 10**1000])
def test_invalid_generation_never_becomes_zero_or_mean_itl(bad):
    item = replace(make_turn("bad"), server_metrics={"generation_time_ms": bad, "mean_itl_ms": 7})
    metrics = request_metrics(item, request_values(item))
    for name in ("server_generation_time", "server_tpot", "server_decode_throughput"):
        assert metrics[name].availability == "unavailable"
        assert metrics[name].statistics.get("value") is None


@pytest.mark.parametrize("completion", [None, 0, 1, True, "4", -1])
def test_missing_or_invalid_completion_is_unavailable(completion):
    item = make_turn("bad", completion=completion)
    metrics = request_metrics(item, request_values(item))
    for name in ("server_generation_time", "server_tpot", "server_decode_throughput"):
        assert metrics[name].availability == "unavailable"
        assert metrics[name].statistics.get("value") is None


def point(points, window, name, statistic):
    return next(p for p in points if p.window == window and p.metric == name and p.statistic == statistic)


def test_window_distributions_are_success_cohort_and_do_not_replace_client():
    backend = StandardMetricsBackend({}, seed=0)
    backend.open()
    try:
        good = make_turn("good", output_event_count=1)
        missing = replace(make_turn("missing"), server_metrics=None)
        failed = make_turn("failed", status="failed")
        updates = backend.consume((good, missing, failed))
        assert updates[0].metrics["server_tpot"].statistics["value"] == 30
        backend._flush_pending(force=True)
        points = backend._distribution_points("trial", "overall", 0, 1000, 2, 0)
        assert point(points, "overall", "server_tpot", "n").value == 1
        assert point(points, "overall", "server_tpot", "eligible_n").value == 2
        assert point(points, "overall", "server_tpot", "am").value == 30
        assert point(points, "overall", "server_tpot", "am").unit == "ms"
        assert point(points, "overall", "server_tpot", "am").source == "server_generation_time_ms+server_usage"
        assert point(points, "overall", "tpot", "n").value == 1
        assert point(points, "overall", "tpot", "eligible_n").value == 2
    finally:
        backend.close()


def test_maxtpm_server_metrics_use_exact_completion_cohort():
    backend = StandardMetricsBackend({"maxtpm": True, "maxtpm_start_percent": 0}, seed=0)
    backend.open()
    try:
        a = replace(turn("a", 10, completion=4), server_metrics={"generation_time_ms":90})
        b = replace(turn("b", 20, completion=4), server_metrics=None)
        outside = replace(turn("outside", 70, completion=4), server_metrics={"generation_time_ms":300})
        backend.consume((session(0, 100), *occupancy((1, 1)), a, b, outside))
        # Test the native filter directly: backend's compatibility fallback must
        # not hide missing names from LATENCY_NAMES.
        backend._flush_pending(force=True)
        points = latency_points(backend, "trial", 0, 1, 1, None)
        assert point(points, "maxtpm", "server_tpot", "n").value == 1
        assert point(points, "maxtpm", "server_tpot", "eligible_n").value == 2
        assert point(points, "maxtpm", "server_tpot", "am").value == 30
        assert point(points, "maxtpm", "server_generation_time", "am").value == 90
        assert point(points, "maxtpm", "server_decode_throughput", "am").value == pytest.approx(1000 / 30)
    finally:
        backend.close()


async def test_official_maxtpm_and_overall_reader_metrics(tmp_path, stage_context):
    from perf_benchmark.core.models import Run, Trial
    from perf_benchmark.pipeline.results.saver import SaverRuntime
    from perf_benchmark.plugins.saver.sqlite import SQLiteSaver

    backend = StandardMetricsBackend({"maxtpm": True, "maxtpm_start_percent": 0}, seed=0)
    backend.open()
    try:
        a = replace(turn("a", 10, completion=4), server_metrics={"generation_time_ms": 90})
        b = replace(turn("b", 20, completion=4), server_metrics=None)
        outside = replace(turn("outside", 70, completion=4), server_metrics={"generation_time_ms": 300})
        backend.consume((session(0, 100), *occupancy((1, 1)), a, b, outside))
        report = backend.finalize_trial(trial(3))
    finally:
        backend.close()
    path = tmp_path / "server-timing.db"
    run = Run(id="run", name="server timing")
    run.prepare()
    run.start()
    saved_trial = Trial(id="trial", run_id=run.id)
    sink = await SaverRuntime.create(SQLiteSaver({"path": str(path)}), run,
                                     stage_context("saver", run=run))
    try:
        await sink.write_trial(saved_trial)
        await sink.write_metric_points(report.points)
        await sink.flush()
    finally:
        await sink.close()
    reader = SQLiteSaver.open_reader(str(path))
    for window, expected_n, expected_eligible, expected_values in (
        ("overall", 2, 3, {"server_generation_time": 195, "server_tpot": 65,
                           "server_decode_throughput": (1000 / 30 + 10) / 2}),
        ("maxtpm", 1, 2, {"server_generation_time": 90, "server_tpot": 30,
                          "server_decode_throughput": 1000 / 30}),
    ):
        metrics = reader.metric_points("run", scope="trial", scope_id="trial", window=window)
        for name, expected_am in expected_values.items():
            assert metrics[name]["statistics"]["n"] == expected_n
            assert metrics[name]["statistics"]["eligible_n"] == expected_eligible
            assert metrics[name]["statistics"]["am"] == pytest.approx(expected_am)
            assert metrics[name]["availability"] == "available"
            assert metrics[name]["statistic_metadata"]["am"]["expected_sample_count"] == expected_eligible
            assert metrics[name]["statistic_metadata"]["am"]["valid_sample_count"] == expected_n
        server = metrics["server_tpot"]
        assert server["unit"] == "ms"
        assert server["source"] == "server_generation_time_ms+server_usage"


async def test_finalized_runtime_to_unified_reader_server_cohorts(tmp_path, stage_context):
    from perf_benchmark.core.models import (LifecycleMeasurements, RequestMeasurements,
        Run, Session, SessionMeasurements, Trial, Turn, TurnMeasurements)
    from perf_benchmark.pipeline.preparation.payloads import PayloadRef
    from perf_benchmark.pipeline.results.metrics import MetricsRuntime
    from perf_benchmark.pipeline.results.saver import SaverRuntime
    from perf_benchmark.plugins.metrics.standard import StandardMetrics
    from perf_benchmark.plugins.saver.sqlite import SQLiteSaver

    run = Run(id="pipeline-run", name="server timing full pipeline")
    model_trial = Trial(id="trial", run_id=run.id, status="succeeded")
    model_session = Session(id="session-0", trial_id="trial", status="succeeded", turns=[],
                            measurements=SessionMeasurements(lifecycle=LifecycleMeasurements(
                                started_ns=10**9, completed_ns=100 * 10**9)))
    for index, (completed, generation) in enumerate(((10, 90), (20, None), (70, 300))):
        request = RequestMeasurements(
            attempts=1, response_mode="stream", started_ns=10**9,
            first_output_ns=2 * 10**9, last_output_ns=2 * 10**9,
            response_completed_ns=completed * 10**9, output_event_count=1,
            server_metrics=None if generation is None else {"generation_time_ms": generation})
        model_session.turns.append(Turn(
            id=str(index), session_id=model_session.id, ordinal=index, status="succeeded",
            request_payload_ref=PayloadRef(0, 1), usage={"prompt_tokens":8,"completion_tokens":4},
            measurements=TurnMeasurements(lifecycle=LifecycleMeasurements(
                started_ns=10**9, completed_ns=completed * 10**9), request=request)))

    path=tmp_path / "pipeline.db"
    sink=await SaverRuntime.create(SQLiteSaver({"path":str(path)}),run,stage_context("saver",run=run))
    metrics=await MetricsRuntime.create(StandardMetrics({"maxtpm":True,"maxtpm_start_percent":0}),
        stage_context("metrics",run=run),publish_results=sink.write_results,wait_for_results=sink.wait_for_headroom)
    collector=None
    try:
        await sink.write_trial(model_trial)
        collector=await metrics.open_trial(model_trial)
        collector.record_occupancy("in_flight",10**9,1)
        for item in model_session.turns:
            collector.record_turn(item)
        collector.record_session(model_session)
        trial_fact=trial(3)
        report=await collector.finalize(model_trial,trial_fact)
        await sink.write_trial(model_trial,trial_fact)
        await sink.write_metric_points(report.points)
        await collector.close()
        collector=None
        await sink.flush()
    finally:
        if collector is not None:
            await collector.close()
        await metrics.close()
        await sink.close()

    reader=SQLiteSaver.open_reader(str(path))
    windows=reader.metric_windows(run.id,scope="trial",scope_id="trial")
    assert next(w for w in windows if w["window"]=="maxtpm")["duration_s"]==60
    for window,n,eligible,mean in (("overall",2,3,65),("maxtpm",1,2,30)):
        points=reader.metric_points(run.id,scope="trial",scope_id="trial",window=window)
        server=points["server_tpot"]
        assert server["unit"]=="ms"
        assert (server["statistics"]["n"],server["statistics"]["eligible_n"],server["statistics"]["am"])==(n,eligible,mean)
        # One output event per request remains insufficient for client TPOT.
        assert points["tpot"]["statistics"]["n"]==0
        assert points["tpot"]["statistics"]["eligible_n"]==eligible


def test_maxtpm_server_metrics_unavailable_without_full_minute():
    backend = StandardMetricsBackend({"maxtpm": True, "maxtpm_start_percent": 0}, seed=0)
    backend.open()
    try:
        a = replace(turn("a", 10, completion=4), server_metrics={"generation_time_ms":90})
        backend.consume((session(0, 20), *occupancy((1, 1)), a))
        report = backend.finalize_trial(trial(1))
        assert point(report.points, "maxtpm", "server_tpot", "am").availability == "unavailable"
    finally:
        backend.close()
