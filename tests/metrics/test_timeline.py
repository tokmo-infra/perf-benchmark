from dataclasses import FrozenInstanceError, replace

import pytest

from perf_benchmark.pipeline.results.measurement import OccupancyFact
from perf_benchmark.plugins.metrics.standard.backend import StandardMetricsBackend
from perf_benchmark.plugins.metrics.standard.formulas import request_metrics, request_values
from tests.metrics.test_groups import fact


def test_late_completion_replaces_start_bucket_and_itl_uses_individual_intervals():
    backend = StandardMetricsBackend({}, seed=0)
    backend.open()
    try:
        first = replace(fact('a'), request_ns=1_000_000_000, first_ns=2_000_000_000,
                        last_ns=2_110_000_000, response_completed_ns=3_000_000_000,
                        output_event_count=3,
                        response_samples=((1,2_000_000_000,1),(2,2_010_000_000,1),(3,2_110_000_000,1)))
        backend.consume((first, OccupancyFact('trial','in_flight',1_000_000_000,1),
                         OccupancyFact('trial','in_flight',3_000_000_000,0)))
        initial = backend.timeline.emit(backend, force=True)[0]
        def values(series, metric):
            return {p.statistic:p.value for p in series.points if p.metric==metric}
        assert values(initial, 'itl') == {'am':55, 'p50':10, 'p90':100, 'p95':100, 'p99':100, 'min':10, 'max':100}
        concurrency = next(p for p in initial.points if p.metric=='actual_in_flight_concurrency')
        assert concurrency.value is None and concurrency.unavailable_reason == 'bucket_in_progress'
        backend.consume((OccupancyFact('trial','in_flight',12_000_000_000,0),))
        closed = backend.timeline.emit(backend, force=True)[0]
        assert values(closed, 'actual_in_flight_concurrency')['value'] == pytest.approx(.2)
        with pytest.raises(FrozenInstanceError):
            initial.window.start_ns = 7
        late = replace(first, id='late', response_completed_ns=21_000_000_000,
                       first_ns=3_000_000_000, last_ns=3_050_000_000, output_event_count=2,
                       response_samples=((1,3_000_000_000,1),(2,3_050_000_000,1)))
        backend.consume((late,))
        series = backend.timeline.emit(backend, force=True)
        assert [s.window.window for s in series] == ['timeline:0','timeline:20000000000']
        updated = series[0]
        assert values(updated, 'itl')['am'] == pytest.approx(160/3)
        assert values(updated, 'itl')['p50'] == 50
        assert values(updated, 'e2e')['max'] == 20
        assert values(updated, 'e2e')['min'] == 2
        assert values(updated, 'workload_completion')['value'] == .4
        assert values(updated, 'decode_throughput')['time_wm'] == pytest.approx(6 / .16)
        assert values(series[1], 'workload_completion')['value'] == .4
        assert all(p.value is None for p in series[1].points if p.metric=='e2e')
        assert backend.timeline.emit(backend, force=True) == ()
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
        series = backend.timeline.emit(backend,force=True)[0]
        point = next(p for p in series.points if p.metric=='speculative_acceptance_proxy' and p.statistic=='am')
        assert point.value == .5
        assert point.source == 'estimated_protocol_events'
        assert (point.valid_sample_count,point.expected_sample_count,point.missing_sample_count)==(1,2,1)
    finally:
        backend.close()


@pytest.mark.parametrize("mode,completion", [("stream", 4), ("non_stream", 4), ("stream", None)])
def test_selected_distribution_statistics_equal_full_calculation(mode, completion):
    from perf_benchmark.plugins.metrics.standard.timeline import STATISTICS

    backend = StandardMetricsBackend({}, seed=0)
    backend.open()
    try:
        backend.consume(tuple(replace(fact(str(i), mode=mode, completion=completion), first_ns=150 + i, last_ns=250 + i,
                                     response_samples=((1, 150 + i, 1), (2, 250 + i, 1)))
                              for i in range(25)))
        backend._flush_pending(force=True)
        full = backend._distribution_points('trial', 'test', 0, 1000, 25, 0, request_only=True)
        selected = backend._distribution_points('trial', 'test', 0, 1000, 25, 0,
                                                request_only=True, statistics=STATISTICS)
        expected = [p for p in full if p.statistic in STATISTICS]
        assert len(selected) == len(expected)
        for actual, reference in zip(selected, expected):
            assert replace(actual, value=reference.value) == reference
            if reference.value is None:
                assert actual.value is None
            else:
                assert actual.value == pytest.approx(reference.value, rel=1e-12, abs=1e-15)
    finally:
        backend.close()
