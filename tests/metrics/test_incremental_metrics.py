"""Incremental measurement facts and end-of-Trial distributions."""

from dataclasses import replace

import pytest

from perf_benchmark.core.models import Trial
from perf_benchmark.pipeline.contracts.metrics import MetricUpdate
from perf_benchmark.pipeline.results.measurement import OccupancyFact
from perf_benchmark.plugins.metrics.standard import parse_settings
from perf_benchmark.plugins.metrics.standard.backend import StandardMetricsBackend
from perf_benchmark.plugins.metrics.standard.formulas import request_metrics, request_values
from perf_benchmark.plugins.metrics.standard.windows import WINDOW_ORDER
from tests.metrics.test_groups import fact
from tests.metrics.test_standard import _fact


def test_request_facts_are_recorded_without_live_window_queries(monkeypatch):
    assert "timeline" not in parse_settings({})
    with pytest.raises(ValueError, match="unknown standard metrics setting: timeline"):
        parse_settings({"timeline": False})
    backend = StandardMetricsBackend({}, seed=0)
    backend.open()

    def unexpected_query(*args, **kwargs):
        pytest.fail("window aggregation ran during request processing")

    try:
        with monkeypatch.context() as guards:
            for name in ("_distribution_points", "_usage_totals", "_time_weighted", "_one"):
                guards.setattr(backend, name, unexpected_query)
            # Span twelve former live buckets, including completed request usage
            # and the occupancy facts needed by the final maxTPM/window analysis.
            for index in range(12):
                start = (10 * index + 1) * 1_000_000_000
                item = replace(
                    fact(str(index)), request_ns=start, first_ns=start + 1_000_000_000,
                    last_ns=start + 1_110_000_000, response_completed_ns=start + 2_000_000_000,
                    output_event_count=3,
                    response_samples=((1, start + 1_000_000_000, 1),
                                      (2, start + 1_010_000_000, 1),
                                      (3, start + 1_110_000_000, 1)),
                )
                updates = backend.consume((
                    OccupancyFact("trial", "in_flight", start, 1), item,
                    OccupancyFact("trial", "in_flight", start + 2_000_000_000, 0),
                ))
                assert len(updates) == 1 and isinstance(updates[0], MetricUpdate)
                assert updates[0].id == item.id
                assert updates[0].metrics["ttft"].statistics["value"] == 1
        report = backend.finalize_trial(_fact(Trial(id="trial", status="succeeded")))
        assert tuple(window.window for window in report.windows) == WINDOW_ORDER
        overall = {(p.metric, p.statistic): p for p in report.points if p.window == "overall"}
        assert report.counts["succeeded"] == 12
        assert overall["ttft", "am"].value == 1
        assert overall["ttft", "n"].value == 12
        assert overall["itl", "am"].value == 55
        assert overall["itl", "p50"].value == 10
        assert overall["itl", "p99"].value == 100
        assert overall["decode_throughput", "time_wm"].value == pytest.approx(36 / 1.32)
    finally:
        backend.close()


@pytest.mark.parametrize(('completion','events','mode','kind','reason'), [
    (None,3,'stream','text','usage_not_reported'),
    (1,1,'stream','text','insufficient_output_tokens'),
    (4,1,'stream','text','insufficient_output_events'),
    (2,3,'stream','text','output_events_exceed_tokens'),
    (4,3,'non_stream','text','non_streaming_response'),
    (4,3,'stream','non_text','non_text_output'),
])
def test_protocol_estimates_have_explicit_unavailability(completion, events, mode, kind, reason):
    item = replace(fact('a',completion=completion,mode=mode), output_event_count=events, output_kind=kind)
    metrics = request_metrics(item, request_values(item))
    for name in ('output_tokens_per_event_estimate','speculative_acceptance_proxy'):
        assert all(value is None for value in metrics[name].statistics.values())
        assert metrics[name].availability == 'unavailable'
        assert metrics[name].unavailable_reason == reason


def test_protocol_estimate_is_not_server_acceptance_and_has_distribution_counts():
    backend = StandardMetricsBackend({},seed=0)
    backend.open()
    try:
        backend.consume((replace(fact('a',completion=5),output_event_count=3),fact('missing',completion=None)))
        report = backend.finalize_trial(_fact(Trial(id="trial", status="succeeded")))
        point = next(p for p in report.points if p.window == 'overall'
                     and p.metric == 'speculative_acceptance_proxy' and p.statistic == 'am')
        assert point.value == .5
        assert point.source == 'estimated_protocol_events'
        assert (point.valid_sample_count,point.expected_sample_count,point.missing_sample_count)==(1,2,1)
    finally:
        backend.close()
