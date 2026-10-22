from dataclasses import asdict, replace

import pytest

from perf_benchmark.pipeline.contracts import (
    ComparisonPoint, LoadProfile, RepetitionPlan, ResourceLimits, TrialFact,
    TrialSchedulingFact, ValidityRequirements,
)
from perf_benchmark.pipeline.results.measurement import AttemptFact, SessionFact, OccupancyFact
from perf_benchmark.plugins.metrics.standard.backend import StandardMetricsBackend
from tests.metrics.test_groups import fact as turn_fact


def measured_trial(backend, identity="trial", *, target=100, count=101, lag_ns=0, latency_ns=1_000_000,
                   requirements=None, status="succeeded", repetition=1, expected=1, closed=False, request_failure=False):
    origin, end = 1_000_000_000, 2_000_000_001
    for i in range(count):
        start = origin + i * 10_000_000
        turn = replace(turn_fact(f"{identity}-turn-{i}", start=start),
                       status="failed" if request_failure and i == 0 else "succeeded",
                       trial_id=identity, session_id=f"{identity}-session-{i}",
                       target_ns=start-lag_ns, offered_ns=start, admitted_ns=start,
                       request_ns=start, first_ns=start+latency_ns, last_ns=start+latency_ns,
                       response_completed_ns=start+latency_ns, turn_started_ns=start,
                       turn_completed_ns=start+latency_ns, response_samples=((1, start+latency_ns, 1),))
        assert turn.session_id is not None
        backend.consume((AttemptFact(**asdict(turn)), turn,
                         SessionFact(turn.session_id, identity, 0, turn.status, None, None, None, {}, 1,
                                     start, start, start, start, start+latency_ns, 1)))
    backend.consume((OccupancyFact(identity, "in_flight", origin, 1),
                     OccupancyFact(identity, "in_flight", end, 0)))
    scheduling = TrialSchedulingFact(offered_sessions=count, admitted_sessions=count, completed_sessions=count,
                                     offered_attempts=count, admitted_attempts=count, completed_attempts=count,
                                     offering_started_ns=origin, session_offering_finished_ns=end,
                                     request_offering_finished_ns=end)
    return backend.finalize_trial(TrialFact(
        identity, "failed" if request_failure else status, LoadProfile("test", target_concurrency=target if closed else None,
                                     target_request_rate=None if closed else target),
        ResourceLimits(1, 1, 1), ComparisonPoint("point", "max_concurrency" if closed else "request_rate", target),
        RepetitionPlan(repetition, expected), scheduling,
        requirements or ValidityRequirements(min_samples=50, min_duration_s=0.9, max_scheduler_lag_ms=5, load_tolerance_ratio=0.1), execution_status=status,
    ))


@pytest.fixture
def backend():
    result = StandardMetricsBackend({}, seed=7)
    result.open()
    try:
        yield result
    finally:
        result.close()


def test_pressure_excludes_response_drain_and_uses_offered_not_completion_rate(backend):
    report = measured_trial(backend, latency_ns=5_000_000_000)
    assert report.validity.status == "pass"
    observed = {key: value for key, value, unit in report.validity.observations}
    assert observed["request_offering_duration"] == pytest.approx(1)
    assert observed["request_load_ratio"] == pytest.approx(1)
    assert next(w for w in report.windows if w.window == "overall").duration_s > 5


@pytest.mark.parametrize("kwargs,reason", [
    ({"target": 200}, "request_pressure_outside_tolerance"),
    ({"target": 40}, "request_pressure_outside_tolerance"),
    ({"count": 10}, "insufficient_samples"),
    ({"lag_ns": 6_000_000}, "max_scheduler_lag_ms_violated"),
    ({"closed": True, "target": 2}, "concurrency_pressure_outside_tolerance"),
    ({"status": "failed"}, "execution_failed"),
    ({"requirements": ValidityRequirements()}, "load_tolerance_ratio_not_configured"),
    ({"requirements": ValidityRequirements(window="missing")}, "measurement_window_unavailable"),
])
def test_invalid_pressure_or_evidence_is_indeterminate(backend, kwargs, reason):
    validity = measured_trial(backend, **kwargs).validity
    assert validity.status == "indeterminate"
    assert reason in validity.reasons


def test_closed_loop_uses_actual_occupancy(backend):
    assert measured_trial(backend, closed=True, target=1).validity.status == "pass"


def test_scalar_slo_carries_real_request_denominator(backend):
    report = measured_trial(backend)
    point = next(p for p in report.points if p.window == "overall" and p.metric == "success_rate")
    assert point.valid_sample_count == point.expected_sample_count == 101
    assert point.missing_sample_count == 0


def test_valid_slo_failure_stays_in_repetitions_but_invalid_load_does_not(backend):
    for index, latency in enumerate((1_000_000, 50_000_000, 100_000_000), 1):
        assert measured_trial(backend, f"trial-{index}", latency_ns=latency,
                              repetition=index, expected=3).validity.status == "pass"
        backend.release_trial(f"trial-{index}")
    points = {(p.metric, p.statistic): p for p in backend.finalize_run("run") if p.window == "overall"}
    assert points["e2e", "p50_median"].value == pytest.approx(0.05)
    assert points["repetition_valid", "value"].value == 3
    assert points["repetition_qualified", "value"].value == 1
    # Recompute the same identity with insufficient evidence, as a failed finalization would do.
    measured_trial(backend, "trial-3", target=200, repetition=3, expected=3)
    points = {(p.metric, p.statistic): p for p in backend.finalize_run("run") if p.window == "overall"}
    assert points["repetition_valid", "value"].value == 2
    assert points["e2e", "p50_median"].value is None
    assert points["e2e", "p50_median"].valid_sample_count == 2


def test_single_repetition_does_not_claim_a_confidence_interval(backend):
    measured_trial(backend, expected=1)
    backend.release_trial('trial')
    points = {(p.metric, p.statistic): p for p in backend.finalize_run('run') if p.window == 'overall'}
    assert points == {}
    from perf_benchmark.plugins.metrics.standard.distributions import repetition_statistics
    assert repetition_statistics([1.0], 'single', seed=7)['bootstrap_median_ci_lower'] is None


def test_service_failures_do_not_turn_successful_tool_execution_into_missing_evidence(backend):
    report = measured_trial(backend, request_failure=True, expected=3)
    assert report.validity.status == 'pass'
    assert report.validity.execution_status == 'succeeded'
    success = next(p for p in report.points if p.metric == 'success_rate' and p.window == 'overall')
    assert success.value < 1 and success.valid_sample_count == 101
