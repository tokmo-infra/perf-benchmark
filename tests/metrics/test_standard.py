
from perf_benchmark.pipeline.results.metrics import MetricsRuntime
import asyncio
import pytest

from perf_benchmark.core.models import (
    LifecycleMeasurements,
    RequestMeasurements,
    Run,
    Session,
    SessionMeasurements,
    Turn,
    Trial,
    TurnMeasurements,
)
from perf_benchmark.pipeline.results.measurement import ResponseTrace
from perf_benchmark.pipeline.contracts.facts import (
    ComparisonPoint,
    LoadProfile,
    RepetitionPlan,
    ResourceLimits,
    TrialFact,
    TrialSchedulingFact,
)
from perf_benchmark.pipeline.preparation.payloads import PayloadRef
from perf_benchmark.plugins.metrics.standard import StandardMetrics, parse_settings
from perf_benchmark.plugins.metrics.standard.distributions import (
    repetition_statistics,
)
from perf_benchmark.plugins.metrics.standard.windows import (
    WINDOW_ORDER,
    measurement_windows,
)
from perf_benchmark.pipeline.contracts.executor import AttemptResult


def _fact(
    trial: Trial,
    *,
    target_concurrency: int | None = 1,
    target_session_rate: float | None = None,
    target_request_rate: float | None = None,
    comparison: ComparisonPoint | None = None,
    repetition: RepetitionPlan | None = None,
) -> TrialFact:
    return TrialFact(
        trial.id,
        trial.status,
        LoadProfile(
            "test",
            target_concurrency=target_concurrency,
            target_session_rate=target_session_rate,
            target_request_rate=target_request_rate,
            max_active_sessions=target_concurrency,
            max_concurrency=target_concurrency,
        ),
        ResourceLimits(1, 1, 1),
        comparison,
        repetition or RepetitionPlan(),
        TrialSchedulingFact(),
    )


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"unknown": True}, "unknown standard metrics setting"),
        ({"steady": {"unknown": 1}}, "unknown metrics steady setting"),
        ({"memory_limit": "512"}, "memory_limit"),
        ({"goodput": {}}, "goodput.rules"),
        (
            {"goodput": {"rules": [{"metric": "itl", "op": "le", "value": "1ms"}]}},
            "unsupported goodput metric",
        ),
        (
            {"goodput": {"rules": [{"metric": "ttft", "op": "lt", "value": "1s"}]}},
            "goodput rule op",
        ),
    ],
)
def test_standard_metrics_settings_are_strict(settings, message):
    with pytest.raises(ValueError, match=message):
        parse_settings(settings)


def test_standard_metrics_normalizes_goodput_threshold_units():
    settings = parse_settings(
        {
            "goodput": {
                "rules": [
                    {"metric": "ttft", "op": "le", "value": "500ms"},
                    {"metric": "tpot", "op": "le", "value": "0.05s"},
                ]
            }
        }
    )

    assert settings["goodput"] == {
        "rules": [
            {"metric": "ttft", "op": "le", "value": "500ms", "threshold": 0.5, "unit": "s"},
            {"metric": "tpot", "op": "le", "value": "0.05s", "threshold": 50.0, "unit": "ms"},
        ]
    }


async def test_standard_metrics_builds_complete_five_window_matrix(
    stage_context,
) -> None:
    run = Run(name="metrics")
    ctx = stage_context(
        "metrics",
        {"steady": {"min_ratio": 0.9, "min_duration_s": 0.01}},
        run,
    )
    trial = Trial(id="trial")
    trial.status = "succeeded"
    batches = []

    def publish(batch):
        if batch.turns:
            batches.append(batch.turns)

    runtime = await MetricsRuntime.create(
        StandardMetrics({"steady": {"min_ratio": 0.9, "min_duration_s": 0.01}}),
        ctx,
        publish_results=publish, wait_for_results=lambda: asyncio.sleep(0),
    )
    collector = await runtime.open_trial(trial)
    turn = Turn(
        id="turn",
        request_payload_ref=PayloadRef(0, 1),
        status="succeeded",
        estimated_input_tokens=20,
        usage={
            "prompt_tokens": 20,
            "completion_tokens": 5,
            "cached_tokens": 2,
        },
        measurements=TurnMeasurements(
            lifecycle=LifecycleMeasurements(started_ns=1_000_000_000),
            request=RequestMeasurements(
                attempts=1,
                response_mode="stream",
                started_ns=1_000_000_000,
                first_output_ns=1_100_000_000,
                last_output_ns=1_300_000_000,
                response_completed_ns=1_350_000_000,
                output_event_count=3,
            ),
        ),
    )
    collector.record_occupancy("in_flight", 1_000_000_000, 1)
    trace = ResponseTrace(turn.id, ((1, 1100000000, 1), (2, 1120000000, 1), (3, 1300000000, 1)))
    collector.record_turn(turn, trace)
    collector.record_occupancy("in_flight", 1_350_000_000, 0)
    collector.record_session(
        Session(
            id="session",
            trial_id=trial.id,
            turns=[turn],
            status="succeeded",
            measurements=SessionMeasurements(
                lifecycle=LifecycleMeasurements(
                    started_ns=1_000_000_000,
                    completed_ns=1_350_000_000,
                ),
            ),
        )
    )
    points = (await collector.finalize(trial, _fact(trial))).points
    await collector.close()
    assert await runtime.finalize(run) == ()
    await runtime.close()

    assert tuple(item.window for item in trial.measurements.metric_windows if not item.window.startswith("timeline:")) == WINDOW_ORDER
    assert len(batches) == 1
    computed_turn = batches[0][0]
    assert computed_turn.metrics["ttft"].statistics["value"] == 0.1
    assert computed_turn.metrics["tpot"].statistics["value"] == 50.0
    assert computed_turn.metrics["e2e"].statistics["value"] == 0.35
    assert computed_turn.metrics["time_to_last_output"].statistics["value"] == 300.0
    assert computed_turn.metrics["response_drain"].statistics["value"] == 50.0
    assert len([point for point in points if not point.window.startswith("timeline:")]) == sum(
        len(definition.statistics) for definition in StandardMetrics({}).describe().metrics
    ) * len(WINDOW_ORDER)
    overall = {
        (point.metric, point.statistic): point for point in points if point.window == "overall"
    }
    assert overall[("ttft", "p50")].value == 0.1
    assert overall[("tpot", "p50")].value == 50.0
    assert overall[("tpot", "twm")].value == 50.0
    assert overall[("itl", "p50")].value == 20.0
    assert overall[("itl", "p50")].source == "protocol_output_event"
    assert overall[("itl", "p50")].formula_version == "standard-window-v8"
    assert overall[("e2e", "p50")].value == 0.35
    assert overall[("time_to_last_output", "p50")].value == 300.0
    assert overall[("response_drain", "p50")].value == 50.0
    assert overall[("prefill_throughput", "hm")].value == 200.0
    assert overall[("ttft", "tm")].value == 0.1
    assert overall[("workload_rps", "value")].value == pytest.approx(1 / 0.350000001)
    for window in trial.measurements.metric_windows:
        if window.window.startswith("timeline:") or window.availability != "available":
            continue
        duration = next(point for point in points if point.window == window.window
                        and point.metric == "window_duration" and point.statistic == "value")
        assert duration.unit == "s"
        assert duration.value == pytest.approx((window.end_ns - window.start_ns) / 1e9)
        rate = next(point for point in points if point.window == window.window
                    and point.metric == "workload_rps" and point.statistic == "value")
        completed_in_window = int(window.start_ns <= 1_350_000_000 < window.end_ns)
        assert rate.value * duration.value == pytest.approx(completed_in_window)
    assert trial.measurements.diagnostics["itl_contract"] == ("output_event_interval")
    assert overall[("goodput_rate", "value")].unavailable_reason == "slo_not_configured"


async def test_actual_arrival_rates_use_arrival_span_instead_of_response_drain(
    stage_context,
) -> None:
    run = Run(name="arrival-rates")
    ctx = stage_context("metrics", run=run)
    trial = Trial(id="trial", status="succeeded")
    def publish(batch):
        pass

    runtime = await MetricsRuntime.create(StandardMetrics({}), ctx, publish_results=publish, wait_for_results=lambda: asyncio.sleep(0))
    collector = await runtime.open_trial(trial)
    for index in range(2):
        request_at = 1_100_000_000 + index * 500_000_000
        session_at = 1_000_000_000 + index * 1_000_000_000
        completed_at = 8_000_000_000 + index * 1_000_000_000
        request = RequestMeasurements(
            attempts=1,
            response_mode="stream",
            target_ns=request_at - 2_000_000,
            offered_ns=request_at - 1_000_000,
            admitted_ns=request_at,
            started_ns=request_at,
            first_output_ns=request_at + 1_000_000,
            last_output_ns=request_at + 2_000_000,
            response_completed_ns=completed_at,
            output_event_count=2,
        )
        turn = Turn(
            id=f"turn-{index}",
            request_payload_ref=PayloadRef(index, 1),
            status="succeeded",
            measurements=TurnMeasurements(
                lifecycle=LifecycleMeasurements(request_at, completed_at),
                request=request,
            ),
        )
        session = Session(
            id=f"session-{index}",
            trial_id=trial.id,
            turns=[turn],
            status="succeeded",
            measurements=SessionMeasurements(
                lifecycle=LifecycleMeasurements(session_at, completed_at),
                target_ns=session_at - 2_000_000,
                offered_ns=session_at,
                admitted_ns=session_at + 1_000_000,
            ),
        )
        session.link()
        result = AttemptResult(turn.id, 1, status="succeeded")
        result.request = request
        collector.record_attempt(turn, result)
        collector.record_turn(turn)
        collector.record_session(session)

    points = (await collector.finalize(
        trial,
        _fact(trial, target_concurrency=None, target_session_rate=1.0, target_request_rate=2.0),
    )).points
    await collector.close()
    await runtime.close()
    overall = {
        point.metric: point for point in points
        if point.window == "overall" and point.statistic == "value"
    }
    assert overall["actual_session_rate"].value == 1.0
    assert overall["actual_request_rate"].value == 2.0
    assert overall["workload_rps"].value < 1.0


async def test_non_streaming_metrics_expose_only_observable_latencies(
    stage_context,
) -> None:
    run = Run(name="metrics-non-stream")
    ctx = stage_context("metrics", run=run)
    trial = Trial(id="trial")
    trial.status = "succeeded"
    batches = []

    def publish(batch):
        if batch.turns:
            batches.append(batch.turns)

    runtime = await MetricsRuntime.create(StandardMetrics({}), ctx, publish_results=publish, wait_for_results=lambda: asyncio.sleep(0))
    collector = await runtime.open_trial(trial)
    collector.record_turn(
        Turn(
            id="non-stream",
            request_payload_ref=PayloadRef(0, 1),
            status="succeeded",
            usage={"prompt_tokens": 4, "completion_tokens": 3},
            measurements=TurnMeasurements(
                lifecycle=LifecycleMeasurements(started_ns=100_000_000),
                request=RequestMeasurements(
                    attempts=1,
                    response_mode="non_stream",
                    started_ns=100_000_000,
                    response_completed_ns=350_000_000,
                ),
            ),
        )
    )
    await collector.finalize(trial, _fact(trial))
    await collector.close()
    await runtime.finalize(run)
    await runtime.close()

    metrics = batches[0][0].metrics
    assert metrics["e2e"].statistics["value"] == 0.25
    assert metrics["time_to_last_output"].statistics["value"] == 250.0
    for name in (
        "ttft",
        "tpot",
        "response_drain",
        "prefill_throughput",
        "decode_throughput",
    ):
        assert metrics[name].availability == "unavailable"
        assert metrics[name].unavailable_reason == "non_streaming_response"


async def test_same_timestamp_output_events_keep_zero_itl_but_no_tpot(
    stage_context,
) -> None:
    run = Run(name="metrics-zero-itl")
    ctx = stage_context("metrics", run=run)
    trial = Trial(id="trial")
    trial.status = "succeeded"
    batches = []

    def publish(batch):
        if batch.turns:
            batches.append(batch.turns)

    runtime = await MetricsRuntime.create(StandardMetrics({}), ctx, publish_results=publish, wait_for_results=lambda: asyncio.sleep(0))
    collector = await runtime.open_trial(trial)
    turn = Turn(
        id="same-timestamp",
        request_payload_ref=PayloadRef(0, 1),
        status="succeeded",
        usage={"prompt_tokens": 2, "completion_tokens": 2},
        measurements=TurnMeasurements(
            lifecycle=LifecycleMeasurements(started_ns=1_000_000),
            request=RequestMeasurements(
                attempts=1,
                response_mode="stream",
                started_ns=1_000_000,
                first_output_ns=2_000_000,
                last_output_ns=2_000_000,
                response_completed_ns=3_000_000,
                output_event_count=2,
            ),
        ),
    )
    trace = ResponseTrace(turn.id, ((1, 2000000, 1), (2, 2000000, 1)))
    collector.record_turn(turn, trace)
    points = (await collector.finalize(trial, _fact(trial))).points
    await collector.close()
    await runtime.finalize(run)
    await runtime.close()

    computed = batches[0][0].metrics
    assert computed["tpot"].availability == "unavailable"
    assert computed["tpot"].unavailable_reason == ("decode_interval_not_observable")
    assert computed["decode_throughput"].unavailable_reason == ("decode_interval_not_observable")
    overall = {
        (point.metric, point.statistic): point for point in points if point.window == "overall"
    }
    assert overall[("itl", "p50")].value == 0.0
    assert overall[("tpot", "p50")].availability == "unavailable"
    assert overall[("tpot", "p50")].unavailable_reason == ("decode_interval_not_observable")


async def test_tpot_reports_single_event_and_missing_usage_reasons(
    stage_context,
) -> None:
    run = Run(name="metrics-tpot-reasons")
    ctx = stage_context("metrics", run=run)
    trial = Trial(id="trial")
    trial.status = "succeeded"
    batches = []

    def publish(batch):
        if batch.turns:
            batches.append(batch.turns)

    runtime = await MetricsRuntime.create(StandardMetrics({}), ctx, publish_results=publish, wait_for_results=lambda: asyncio.sleep(0))
    collector = await runtime.open_trial(trial)
    cases = (
        (
            "missing-usage",
            {},
            2,
            (
                (1, 2_000_000, 1),
                (2, 3_000_000, 1),
            ),
        ),
        (
            "single-event",
            {"completion_tokens": 2},
            1,
            ((1, 6_000_000, 1),),
        ),
    )
    for index, (identifier, usage, event_count, samples) in enumerate(cases):
        first_ns = samples[0][1]
        last_ns = samples[-1][1]
        turn = Turn(
            id=identifier,
            request_payload_ref=PayloadRef(index, 1),
            status="succeeded",
            usage=usage,
            measurements=TurnMeasurements(
                lifecycle=LifecycleMeasurements(started_ns=first_ns - 1_000_000),
                request=RequestMeasurements(
                    attempts=1,
                    response_mode="stream",
                    started_ns=first_ns - 1_000_000,
                    first_output_ns=first_ns,
                    last_output_ns=last_ns,
                    response_completed_ns=last_ns + 1_000_000,
                    output_event_count=event_count,
                ),
            ),
        )
        trace = ResponseTrace(turn.id, samples)
        collector.record_turn(turn, trace)
    await collector.finalize(trial, _fact(trial))
    await collector.close()
    await runtime.finalize(run)
    await runtime.close()

    by_id = {turn.id: turn for batch in batches for turn in batch}
    assert by_id["missing-usage"].metrics["tpot"].unavailable_reason == ("usage_not_reported")
    assert by_id["single-event"].metrics["tpot"].unavailable_reason == (
        "insufficient_output_events"
    )


async def test_window_metrics_use_separate_start_and_completion_cohorts(
    stage_context,
) -> None:
    settings = parse_settings(
        {
            "steady": {"min_ratio": 0.9, "min_duration_s": 0.01},
            "goodput": {
                "rules": [
                    {"metric": "ttft", "op": "le", "value": "1s"},
                    {"metric": "e2e", "op": "le", "value": "20s"},
                ]
            },
        }
    )
    run = Run(name="metrics-cohorts")
    ctx = stage_context("metrics", settings, run)
    trial = Trial(id="trial")
    trial.status = "succeeded"
    batches = []

    def publish(batch):
        if batch.turns:
            batches.append(batch.turns)

    runtime = await MetricsRuntime.create(
        StandardMetrics(settings),
        ctx,
        publish_results=publish, wait_for_results=lambda: asyncio.sleep(0),
    )
    collector = await runtime.open_trial(trial)
    turns = (
        Turn(
            id="started-in-window",
            request_payload_ref=PayloadRef(0, 1),
            status="succeeded",
            usage={"prompt_tokens": 2, "completion_tokens": 2},
            measurements=TurnMeasurements(
                lifecycle=LifecycleMeasurements(started_ns=12_000_000_000),
                    request=RequestMeasurements(
                        response_mode="stream",
                        target_ns=12_000_000_000,
                        offered_ns=12_000_000_000,
                        admitted_ns=12_000_000_000,
                        started_ns=12_000_000_000,
                        first_output_ns=12_500_000_000,
                        last_output_ns=12_500_000_000,
                        response_completed_ns=30_000_000_000,
                        output_event_count=1,
                ),
            ),
        ),
        Turn(
            id="completed-in-window",
            request_payload_ref=PayloadRef(1, 1),
            status="succeeded",
            usage={"prompt_tokens": 3, "completion_tokens": 4},
            measurements=TurnMeasurements(
                lifecycle=LifecycleMeasurements(started_ns=5_000_000_000),
                    request=RequestMeasurements(
                        response_mode="non_stream",
                        target_ns=5_000_000_000,
                        offered_ns=5_000_000_000,
                        admitted_ns=5_000_000_000,
                        started_ns=5_000_000_000,
                    response_completed_ns=15_000_000_000,
                ),
            ),
        ),
        Turn(
            id="started-in-window-over-slo",
            request_payload_ref=PayloadRef(2, 1),
            status="succeeded",
            usage={"prompt_tokens": 1, "completion_tokens": 2},
            measurements=TurnMeasurements(
                lifecycle=LifecycleMeasurements(started_ns=13_000_000_000),
                request=RequestMeasurements(
                    response_mode="stream",
                    target_ns=13_000_000_000,
                    offered_ns=13_000_000_000,
                    admitted_ns=13_000_000_000,
                    started_ns=13_000_000_000,
                    first_output_ns=13_800_000_000,
                    last_output_ns=13_800_000_000,
                    response_completed_ns=34_000_000_000,
                    output_event_count=1,
                ),
            ),
        ),
        Turn(
            id="started-in-window-missing-ttft",
            request_payload_ref=PayloadRef(3, 1),
            status="succeeded",
            usage={"prompt_tokens": 1, "completion_tokens": 2},
            measurements=TurnMeasurements(
                lifecycle=LifecycleMeasurements(started_ns=14_000_000_000),
                request=RequestMeasurements(
                    response_mode="non_stream",
                    target_ns=14_000_000_000,
                    offered_ns=14_000_000_000,
                    admitted_ns=14_000_000_000,
                    started_ns=14_000_000_000,
                    response_completed_ns=15_000_000_000,
                ),
            ),
        ),
        Turn(
            id="failed-in-window",
            request_payload_ref=PayloadRef(4, 1),
            status="failed",
            error="service error",
            measurements=TurnMeasurements(
                lifecycle=LifecycleMeasurements(16_000_000_000, 17_000_000_000),
                request=RequestMeasurements(
                    response_mode="stream",
                    target_ns=16_000_000_000,
                    offered_ns=16_000_000_000,
                    admitted_ns=16_000_000_000,
                    started_ns=16_000_000_000,
                    first_output_ns=16_100_000_000,
                    last_output_ns=16_100_000_000,
                    response_completed_ns=17_000_000_000,
                    output_event_count=1,
                ),
            ),
        ),
    )
    collector.record_occupancy("in_flight", 10_000_000_000, 1)
    collector.record_occupancy("in_flight", 20_000_000_000, 0)
    for turn in turns:
        collector.record_turn(turn)
    points = (await collector.finalize(trial, _fact(trial))).points
    await collector.close()
    await runtime.finalize(run)
    await runtime.close()

    steady = {
        (point.metric, point.statistic): point
        for point in points
        if point.window == "steady_occupancy"
    }
    assert steady[("e2e", "n")].value == 3
    assert steady[("e2e", "p50")].value == 18.0
    assert steady[("workload_rps", "value")].value == pytest.approx(0.2)
    assert steady[("goodput_qualified", "value")].value == 1
    assert steady[("goodput_rate", "value")].value == pytest.approx(0.1)
    assert steady[("goodput_percentage", "value")].value == pytest.approx(0.25)
    assert steady[("total_completion_tokens", "value")].value == 6
    assert steady[("mean_output_tokens", "value")].value == 2
    cohorts = trial.measurements.diagnostics["window_cohorts"]
    assert isinstance(cohorts, dict)
    assert cohorts["steady_occupancy"] == {
        "offered": 4,
        "started": 4,
        "completed": 3,
        "latency_eligible": 3,
        "latency_missing": 0,
        "cross_window_completions": 2,
        "goodput_denominator": 4,
        "goodput_qualified": 1,
        "goodput_missing_required_metric": 1,
    }


async def test_standard_metrics_streams_turn_results_in_bounded_batches(
    stage_context,
) -> None:
    run = Run(name="metrics-batches")
    ctx = stage_context("metrics", run=run)
    trial = Trial(id="trial")
    trial.status = "succeeded"
    batches = []

    def publish(batch):
        if batch.turns:
            batches.append(batch.turns)

    runtime = await MetricsRuntime.create(StandardMetrics({}), ctx, publish_results=publish, wait_for_results=lambda: asyncio.sleep(0))
    collector = await runtime.open_trial(trial)
    for index in range(257):
        collector.record_turn(
            Turn(
                id=f"turn-{index:03d}",
                session_id="session",
                request_payload_ref=PayloadRef(index, 1),
                status="succeeded",
                measurements=TurnMeasurements(
                    lifecycle=LifecycleMeasurements(started_ns=index + 1),
                    request=RequestMeasurements(
                        response_mode="non_stream",
                        started_ns=index + 1,
                        response_completed_ns=index + 2,
                    ),
                ),
            )
        )
    await collector.finalize(trial, _fact(trial))
    await collector.close()
    await runtime.finalize(run)
    await runtime.close()

    assert sum(map(len, batches)) == 257
    assert all(len(batch) <= 100 for batch in batches)
    assert len({turn.id for batch in batches for turn in batch}) == 257


@pytest.mark.parametrize("trace", [ResponseTrace("other-turn", ()), ResponseTrace("turn", (), 2)])
async def test_standard_metrics_rejects_mismatched_response_trace(stage_context, trace):
    def publish(batch):
        pytest.fail("mismatched trace must not publish a result")

    runtime = await MetricsRuntime.create(
        StandardMetrics({}), stage_context("metrics"), publish_results=publish, wait_for_results=lambda: asyncio.sleep(0)
    )
    collector = await runtime.open_trial(Trial(id="trial"))
    turn = Turn(id="turn", request_payload_ref=PayloadRef(0, 1))
    turn.measurements.request.attempts = 1
    try:
        with pytest.raises(ValueError, match="another Turn or Attempt"):
            collector.record_turn(turn, trace)
    finally:
        await collector.close()
        await runtime.close()


async def test_failed_repetition_keeps_expected_count_and_unavailable_summaries(stage_context, monkeypatch):
    from perf_benchmark.pipeline.contracts.sweep import TrialValidity
    monkeypatch.setattr("perf_benchmark.plugins.metrics.standard.backend.assess",
                        lambda backend, fact, points, windows: TrialValidity(fact.trial_id, fact.status, "pass" if fact.status == "succeeded" else "indeterminate"))
    run = Run(name="failed repetition")

    def publish(batch):
        pass

    runtime = await MetricsRuntime.create(
        StandardMetrics({}), stage_context("metrics", run=run), publish_results=publish, wait_for_results=lambda: asyncio.sleep(0)
    )
    try:
        for index, status in enumerate(("succeeded", "failed"), 1):
            trial = Trial(id=f"trial-{index}", status=status)
            collector = await runtime.open_trial(trial)
            turn = Turn(
                id=f"turn-{index}", status="succeeded", request_payload_ref=PayloadRef(0, 1)
            )
            turn.measurements.request = RequestMeasurements(
                response_mode="non_stream", started_ns=100, response_completed_ns=200
            )
            collector.record_turn(turn)
            await collector.finalize(
                trial,
                _fact(
                    trial,
                    comparison=ComparisonPoint("c1", "concurrency", 1, "requests"),
                    repetition=RepetitionPlan(index, 2),
                ),
            )
            await collector.close()
        points = await runtime.finalize(run)
    finally:
        await runtime.close()
    overall = {
        (point.metric, point.statistic): point for point in points if point.window == "overall"
    }
    assert points and {point.scope for point in points} == {"comparison"}
    assert {point.scope_id for point in points} == {"c1"}
    assert overall[("repetition_valid", "value")].value == 1
    assert overall[("repetition_expected", "value")].value == 2
    assert overall[("repetition_qualified", "value")].value == 0
    summary = overall[("e2e", "p50_median")]
    assert summary.value is None
    assert summary.unavailable_reason == "insufficient_metric_samples"
    assert (summary.valid_sample_count, summary.expected_sample_count) == (1, 2)


def test_measurement_windows_are_time_windows() -> None:
    windows, diagnostics = measurement_windows(
        (0, 100_000_000_000),
        [
            (0, 0),
            (10_000_000_000, 4),
            (80_000_000_000, 0),
        ],
        4,
        min_ratio=0.9,
        min_duration_s=1.0,
    )
    by_name = {window.name: window for window in windows}
    assert by_name["last30s"].start_ns == 70_000_000_000
    assert by_name["drop_first_20pct"].start_ns == 20_000_000_000
    assert by_name["steady_occupancy"].start_ns == 10_000_000_000
    assert by_name["steady_occupancy"].end_ns == 80_000_000_000
    assert by_name["steady_strip_10pct"].start_ns == 17_000_000_000
    assert by_name["steady_strip_10pct"].end_ns == 73_000_000_000
    assert diagnostics["steady_occupancy_threshold"] == 4


def test_repetition_statistics_cover_all_nine_summaries() -> None:
    summary = repetition_statistics([1.0, 2.0, 3.0], "point", seed=0)
    assert set(summary) == {
        "median",
        "mean",
        "mad",
        "cv",
        "min",
        "max",
        "trimmed_mean_10pct",
        "bootstrap_median_ci_lower",
        "bootstrap_median_ci_upper",
    }
    assert summary["median"] == 2.0
    assert summary["trimmed_mean_10pct"] == 2.0


async def test_audio_progress_never_manufactures_text_token_latencies(stage_context):
    trial = Trial(id="audio-trial", status="succeeded")
    batches = []

    def publish(batch):
        batches.append(batch)

    runtime = await MetricsRuntime.create(
        StandardMetrics({}), stage_context("metrics"), publish_results=publish, wait_for_results=lambda: asyncio.sleep(0)
    )
    collector = await runtime.open_trial(trial)
    turn = Turn(
        id="audio-turn",
        request_payload_ref=PayloadRef(0, 1),
        status="succeeded",
        measurements=TurnMeasurements(
            lifecycle=LifecycleMeasurements(started_ns=1_000_000_000, completed_ns=1_400_000_000),
            request=RequestMeasurements(
                attempts=1,
                response_mode="stream",
                output_kind="audio",
                started_ns=1_000_000_000,
                first_output_ns=1_100_000_000,
                last_output_ns=1_300_000_000,
                response_completed_ns=1_400_000_000,
                output_event_count=2,
            ),
        ),
    )
    try:
        trace = ResponseTrace(turn.id, ((1, 1100000000, 0), (2, 1300000000, 0)))
        collector.record_turn(turn, trace)
        points = (await collector.finalize(trial, _fact(trial))).points
        measured = batches[0].turns[0]
        for name in ("ttft", "tpot", "prefill_throughput", "decode_throughput"):
            assert measured.metrics[name].availability == "unavailable"
            assert all(value is None for value in measured.metrics[name].statistics.values())
            assert measured.metrics[name].unavailable_reason == "non_text_output"
        itl = next(
            point
            for point in points
            if point.window == "overall" and point.metric == "itl" and point.statistic == "p50"
        )
        assert itl.value is None and itl.unavailable_reason == "non_text_output"
        assert measured.metrics["time_to_last_output"].statistics["value"] == 300
        assert measured.metrics["output_tokens"].availability == "unavailable"
    finally:
        await collector.close()
        await runtime.close()


def test_bootstrap_replays_same_samples_and_responds_to_seed():
    samples = [float(i * i) for i in range(100)]
    first = repetition_statistics(samples, "point:overall:ttft:p95", seed=42)
    replay = repetition_statistics(list(reversed(samples)), "point:overall:ttft:p95", seed=42)
    changed = repetition_statistics(samples, "point:overall:ttft:p95", seed=43)
    assert first == replay
    assert first != changed
    assert all(first[key] == changed[key] for key in first if not key.startswith("bootstrap_"))


@pytest.mark.parametrize("invalid", ["missing_usage", "zero_interval", "single_token", "non_stream"])
def test_time_weighted_speed_handles_bursts_and_preserves_sample_qualification(invalid):
    from dataclasses import replace
    from tests.metrics.test_groups import fact

    backend = StandardMetrics({}).create_backend(seed=0)
    backend.open()
    try:
        normal = replace(fact("normal", completion=101, start=1_000_000_000),
                         first_ns=2_000_000_000, last_ns=6_000_000_000,
                         response_completed_ns=7_000_000_000)
        burst = replace(normal, id="burst", usage={"prompt_tokens": 8, "completion_tokens": 25},
                        last_ns=2_000_020_000)
        missing = replace(normal, id="invalid", **{
            "missing_usage": {"usage": {"prompt_tokens": 8}},
            "zero_interval": {"last_ns": normal.first_ns},
            "single_token": {"usage": {"prompt_tokens": 8, "completion_tokens": 1}},
            "non_stream": {"response_mode": "non_stream"},
        }[invalid])
        backend.consume((normal, burst, missing, replace(normal, id="failed", status="failed")))
        backend._flush_pending(force=True)
        points = {(p.metric, p.statistic): p for p in backend._distribution_points(
            "trial", "overall", 0, 10_000_000_000, 3, 0)}
        speed = points["decode_throughput", "time_wm"]
        assert speed.value == pytest.approx(124 / 4.00002)
        assert (speed.valid_sample_count, speed.expected_sample_count, speed.missing_sample_count) == (2, 3, 1)
        assert points["decode_throughput", "am"].value == pytest.approx(600012.5)
        assert points["decode_throughput", "p99"].value == pytest.approx(1_200_000)
        empty = next(p for p in backend._distribution_points(
            "trial", "empty", 10_000_000_000, 20_000_000_000, 0, 0)
                     if p.metric == "decode_throughput" and p.statistic == "time_wm")
        assert empty.value is None and empty.valid_sample_count == 0
    finally:
        backend.close()


def test_v5_units_weights_and_repetition_statistics_are_not_double_converted(monkeypatch):
    from perf_benchmark.pipeline.contracts.sweep import TrialValidity
    monkeypatch.setattr("perf_benchmark.plugins.metrics.standard.backend.assess",
                        lambda backend, fact, points, windows: TrialValidity(fact.trial_id, fact.status, "pass"))
    from dataclasses import asdict
    from perf_benchmark.pipeline.results.measurement import TurnFact, AttemptFact, SessionFact

    backend = StandardMetrics({}).create_backend(seed=0)
    backend.open()
    try:
        for repetition in (1, 2):
            trial_id = f"trial-{repetition}"
            facts = []
            for index, tokens in enumerate((3, 9)):
                start = (1 + index * 10) * 10**9
                first = start + (index + 1) * repetition * 10**9
                last = first + (index + 1) * 10**9
                fact = TurnFact(
                    trial_id=trial_id, id=f"turn-{repetition}-{index}", session_id=f"session-{repetition}",
                    ordinal=index, status="succeeded", started_at=None, finished_at=None, error=None,
                    estimated_input_tokens=None, reference_output_tokens=None, requested_output_tokens=None,
                    usage={"prompt_tokens": 10, "completion_tokens": tokens}, metadata={},
                    response_samples=((1, first, 1), (2, last, 1)), target_ns=start, offered_ns=start,
                    admitted_ns=start, turn_started_ns=start, turn_completed_ns=last, attempts=1,
                    response_mode="stream", request_ns=start, first_ns=first, last_ns=last,
                    response_completed_ns=last, http_status=200, output_event_count=2, output_chars=2,
                    finish_reason="stop", total_started_ns=start,
                )
                facts.extend((AttemptFact(**asdict(fact)), fact))
            facts.append(SessionFact(f"session-{repetition}", trial_id, 0, "succeeded", None, None,
                                     None, {}, 2, 10**9, 10**9, 10**9, 10**9, 20*10**9, 2))
            updates = backend.consume(facts)
            request_updates = [u for u in updates if u.scope in {"attempt", "turn"}]
            assert request_updates[0].metrics["ttft"].statistics["value"] == repetition
            assert request_updates[0].metrics["ttft"].unit == "s"
            assert request_updates[0].metrics["e2e"].statistics["value"] == repetition + 1
            assert request_updates[0].metrics["tpot"].statistics["value"] == 500
            assert request_updates[0].metrics["tpot"].unit == "ms"
            assert request_updates[0].metrics["prefill_throughput"].unit == "tok/s"
            report = backend.finalize_trial(TrialFact(
                trial_id, "succeeded", LoadProfile("test", target_concurrency=1), ResourceLimits(1, 1, 1),
                ComparisonPoint("point", "max_concurrency", 1, "requests"),
                RepetitionPlan(repetition, 2), TrialSchedulingFact(),
            ))
            points = {(p.metric, p.statistic): p for p in report.points if p.window == "overall"}
            assert points["ttft", "am"].value == pytest.approx(1.5 * repetition)
            assert points["ttft", "p50"].value == repetition  # existing discrete percentile
            assert points["ttft", "n"].value == 2
            assert points["ttft", "n"].unit == "samples"
            assert points["session_first_turn_ttft", "am"].value == repetition
            assert points["session_subsequent_turn_ttft", "am"].value == 2 * repetition
            assert points["tpot", "am"].value == 375
            assert points["tpot", "twm"].value == 300
            assert points["prefill_throughput", "hm"].value == pytest.approx(20 / (3 * repetition))
            assert points["prefill_throughput", "time_wm"].value == pytest.approx(20 / (3 * repetition))
            decode = points["decode_throughput", "time_wm"]
            assert decode.value == pytest.approx(10 / 3)
            assert decode.valid_sample_count == decode.expected_sample_count == 2
            assert decode.formula_version == "client-speed-time-weighted-v1"
            assert points["decode_throughput", "am"].value == 3
            assert points["decode_throughput", "twm"].value == 3.6
            assert points["itl", "am"].value == 1500
            assert points["itl", "am"].unit == "ms"
            assert ("tpot", "hm") not in points
            backend.release_trial(trial_id)
        aggregate = {(p.metric, p.statistic): p for p in backend.finalize_run("run") if p.window == "overall"}
        assert aggregate["ttft", "am_median"].value == 2.25
        assert aggregate["ttft", "am_median"].unit == "s"
        assert aggregate["ttft", "am_cv"].value == pytest.approx(1 / 3)
        assert aggregate["ttft", "am_cv"].unit == "ratio"
    finally:
        backend.close()
