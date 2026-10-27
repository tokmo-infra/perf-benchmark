from dataclasses import asdict, replace

import pytest

from perf_benchmark.pipeline.contracts.facts import (
    LoadProfile,
    RepetitionPlan,
    ResourceLimits,
    TrialFact,
    TrialSchedulingFact,
)
from perf_benchmark.pipeline.results.measurement import AttemptFact, OccupancyFact, SessionFact
from perf_benchmark.plugins.metrics.standard import StandardMetrics, parse_settings
from perf_benchmark.plugins.metrics.standard.maxtpm import _concurrency_end
from tests.metrics.test_groups import fact

NANO = 10**9
PRICES = dict(
    input_price=2, cached_input_price=0.2, output_price=8, per_tokens=1000, currency="CNY"
)


@pytest.mark.parametrize("source", [None, "", " \t", 0, False, [], {}])
def test_pricing_source_requires_nonempty_text(source):
    with pytest.raises(ValueError, match="pricing.source"):
        parse_settings({"pricing": {**PRICES, "source": source}})


@pytest.mark.parametrize("key,value", [("input_price", True), ("output_price", "8"),
    ("cached_input_price", -1), ("input_price", float("nan")), ("per_tokens", False), ("unknown", "value")])
def test_pricing_source_preserves_strict_numeric_and_field_validation(key, value):
    with pytest.raises(ValueError):
        parse_settings({"pricing": {**PRICES, "source": "configured price list", key: value}})


def test_pricing_source_is_frozen_metadata_and_keeps_billing_points_identical():
    reports = []
    for pricing in (PRICES, {**PRICES, "source": "https://example.org/prices#served-model"}):
        backend = StandardMetrics({"maxtpm": True, "pricing": pricing}).create_backend(seed=0)
        backend.open()
        try:
            known = turn("known", 20, prompt=100, completion=10, cached=25)
            backend.consume((*occupancy((1, 1)), session(0, 9999),
                             turn("before", 5, prompt=0, completion=0, cached=0),
                             turn("start", 10, prompt=0, completion=0, cached=0),
                             AttemptFact(**asdict(known)), known, turn("end", 140)))
            reports.append(backend.finalize_trial(trial(4)))
        finally:
            backend.close()
    assert reports[0].points == reports[1].points
    assert reports[0].diagnostics["pricing"] == PRICES
    assert reports[1].diagnostics["pricing"] == {**PRICES, "source": "https://example.org/prices#served-model"}


def trial(count=4):
    return TrialFact(
        "trial",
        "succeeded",
        LoadProfile("test", target_concurrency=1),
        ResourceLimits(1, 1, 1),
        None,
        RepetitionPlan(),
        TrialSchedulingFact(dataset_session_count=4, dataset_turn_count=count),
    )


def session(index, finished):
    return SessionFact(
        f"session-{index}",
        "trial",
        index,
        "succeeded",
        None,
        None,
        None,
        {},
        1,
        NANO,
        NANO,
        NANO,
        NANO,
        finished * NANO,
        1,
    )


def turn(name, end, **kwargs):
    return replace(
        fact(name, start=NANO, **kwargs),
        session_id="session-0",
        response_completed_ns=end * NANO,
        turn_completed_ns=end * NANO,
        first_ns=2 * NANO,
        last_ns=3 * NANO,
    )


def by_name(points, window="overall"):
    return {p.metric: p for p in points if p.window == window and p.statistic == "value"}


def occupancy(*rows):
    return tuple(OccupancyFact("trial", "in_flight", stamp * NANO, value) for stamp, value in rows)


@pytest.mark.parametrize("value", [
    True, False, None, "40", {}, [], -1, 100, 101, 10**1000,
    float("nan"), float("inf"), float("-inf"),
])
def test_maxtpm_start_percent_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="maxtpm_start_percent"):
        parse_settings({"maxtpm_start_percent": value})


def test_maxtpm_start_percent_yaml_and_default():
    import yaml

    config = yaml.safe_load("""
pipeline:
  metrics:
    plugin: standard
    settings:
      maxtpm: true
      maxtpm_start_percent: 12.5
""")
    parsed = parse_settings(config["pipeline"]["metrics"]["settings"])
    assert parsed["maxtpm"] is True
    assert parsed["maxtpm_start_percent"] == 12.5
    assert parse_settings({})["maxtpm_start_percent"] == 40
    assert parse_settings({})["maxtpm"] is False


@pytest.mark.parametrize("percent,start_turns,start_s,minutes,tpm", [
    (0, 0, 1, 2, 300),
    (0.1, 1, 10, 2, 300),
    (25, 3, 30, 2, 250),
    (40, 4, 40, 2, 200),
    (50, 5, 60, 1, 200),
    (60, 6, 100, 1, 200),
    (80, 8, 180, 0, None),
])
def test_maxtpm_configurable_start_uses_same_concurrency_cutoff_and_cohort(
    percent, start_turns, start_s, minutes, tpm
):
    backend = StandardMetrics(parse_settings({
        "maxtpm": True, "maxtpm_start_percent": percent, "pricing": PRICES,
    })).create_backend(seed=0)
    backend.open()
    try:
        backend.consume((session(0, 301), *occupancy((1, 192), (175, 191)),
            *(turn(str(i), stamp, prompt=90, completion=10, cached=25)
              for i, stamp in enumerate((10, 20, 30, 40, 60, 100, 140, 180, 240, 300)))))
        report = backend.finalize_trial(replace(trial(10), load_profile=LoadProfile(
            "test", target_concurrency=192, max_active_sessions=384,
        )))
        diagnostics = report.diagnostics
        assert diagnostics["maxtpm_start_percent"] == percent
        assert diagnostics["maxtpm_progress_start_fraction"] == percent / 100
        assert diagnostics["maxtpm_start_turns"] == start_turns
        assert diagnostics["maxtpm_start_ns"] == start_s * NANO
        cutoff = max(start_s, 175)
        assert diagnostics["maxtpm_end_ns"] == cutoff * NANO
        assert diagnostics["maxtpm_underfill_confirmed_ns"] == (cutoff + 60) * NANO
        assert diagnostics["maxtpm_end_reason"] == "concurrency_under_target_60s"
        summary = by_name(report.points, "maxtpm")
        assert summary["maxtpm_minutes"].value == minutes
        assert summary["maxtpm"].value == tpm
        assert by_name(report.points)["succeeded"].value == 10
        if minutes:
            assert summary["maxtpm_cache_hit_rate"].value == pytest.approx(25 / 90)
            assert summary["maxtpm_input_tpm"].value == pytest.approx(tpm * .9)
            assert summary["maxtpm_output_tpm"].value == pytest.approx(tpm * .1)
            assert summary["revenue_per_minute"].value == pytest.approx(tpm * .00215)
            succeeded = next(p.value for p in report.points if p.window == "maxtpm"
                             and p.metric == "succeeded" and p.statistic == "value")
            assert succeeded == tpm * minutes / 100
        else:
            assert summary["maxtpm"].unavailable_reason == "no_complete_minute_in_maxtpm_window"
    finally:
        backend.close()


@pytest.mark.parametrize("percent,count,expected", [
    (28, 25, 7), (12.5, 9, 2), (40, 6, 3), (99.9, 10, 10), (0, 9, 0),
])
def test_maxtpm_start_percent_rounds_up_exact_decimal_progress(percent, count, expected):
    backend = StandardMetrics({"maxtpm": True, "maxtpm_start_percent": percent}).create_backend(seed=0)
    backend.open()
    try:
        backend.consume((session(0, 300), *occupancy((1, 1)),
                         *(turn(str(i), 10 * (i + 1)) for i in range(count))))
        report = backend.finalize_trial(trial(count))
        assert report.diagnostics["maxtpm_start_turns"] == expected
        assert report.diagnostics["maxtpm_start_ns"] == (10 * expected if expected else 1) * NANO
    finally:
        backend.close()


def test_maxtpm_default_matches_explicit_40_percent():
    reports = []
    for settings in ({"maxtpm": True}, {"maxtpm": True, "maxtpm_start_percent": 40}):
        backend = StandardMetrics(parse_settings(settings)).create_backend(seed=0)
        backend.open()
        try:
            backend.consume((session(0, 300), *occupancy((1, 1)),
                             *(turn(str(i), stamp) for i, stamp in enumerate((10, 20, 80, 140)))))
            reports.append(backend.finalize_trial(trial(4)))
        finally:
            backend.close()
    assert reports[0] == reports[1]


def test_maxtpm_custom_start_is_unavailable_until_configured_progress():
    backend = StandardMetrics({"maxtpm": True, "maxtpm_start_percent": 80}).create_backend(seed=0)
    backend.open()
    try:
        backend.consume((session(0, 300), *occupancy((1, 1)),
                         turn("one", 10), turn("two", 20)))
        report = backend.finalize_trial(trial(10))
        assert report.diagnostics["maxtpm_start_turns"] == 8
        assert report.diagnostics["maxtpm_start_ns"] is None
        assert by_name(report.points, "maxtpm")["maxtpm"].unavailable_reason == "dataset_start_percent_not_completed"
        assert by_name(report.points)["succeeded"].value == 2
    finally:
        backend.close()


@pytest.mark.parametrize("rows,start,end,expected,confirmed,reason", [
    ([(1, 192), (100, 191), (160, 192)], 20, 300, 100, 160, "concurrency_under_target_60s"),
    ([(1, 192), (100, 191), (159, 192), (200, 191)], 20, 300, 200, 260, "concurrency_under_target_60s"),
    ([(1, 192), (100, 191), (159, 192)], 20, 300, 300, None, None),
    ([(1, 192), (200, 191)], 20, 250, 250, None, None),
    ([(1, 192), (200, 191)], 20, 260, 200, 260, "concurrency_under_target_60s"),
    ([(1, 191), (300, 192)], 20, 250, 20, 80, "concurrency_under_target_60s"),
    ([(1, 192), (100, 191), (120, 190)], 20, 300, 100, 160, "concurrency_under_target_60s"),
    ([(1, 192), (100, 191), (100, 192)], 20, 300, 300, None, None),
    ([(1, 192), (100, 191), (150, 192), (150, 191)], 20, 300, 100, 160, "concurrency_under_target_60s"),
    ([(1, 191), (20, 192)], 20, 300, 300, None, None),
    ([(1, 193), (100, 192)], 20, 300, 300, None, None),
    ([], 20, 300, None, None, "occupancy_not_observed_at_maxtpm_start"),
    ([(21, 192)], 20, 300, None, None, "occupancy_not_observed_at_maxtpm_start"),
])
def test_maxtpm_concurrency_end_uses_continuous_time_and_exact_target(
    rows, start, end, expected, confirmed, reason
):
    cutoff, confirmation, actual_reason = _concurrency_end(
        [(stamp * NANO, value) for stamp, value in rows], start * NANO, end * NANO, 192
    )
    assert cutoff == (expected * NANO if expected is not None else None)
    assert confirmation == (confirmed * NANO if confirmed is not None else None)
    assert actual_reason == reason


@pytest.mark.parametrize("target,rows,reason", [
    (None, ((1, 192),), "target_concurrency_not_configured"),
    (192, (), "occupancy_not_observed_at_maxtpm_start"),
    (192, ((21, 192),), "occupancy_not_observed_at_maxtpm_start"),
])
def test_maxtpm_missing_concurrency_keeps_global_metrics_available(target, rows, reason):
    backend = StandardMetrics({"maxtpm": True}).create_backend(seed=0)
    backend.open()
    try:
        backend.consume((session(0, 300), *occupancy(*rows),
                         turn("before", 10), turn("start", 20), turn("after", 200)))
        spec = replace(trial(3), load_profile=LoadProfile("test", target_concurrency=target))
        report = backend.finalize_trial(spec)
        assert by_name(report.points, "maxtpm")["maxtpm"].unavailable_reason == reason
        assert by_name(report.points)["succeeded"].value == 3
    finally:
        backend.close()


@pytest.mark.parametrize("count,end_reason", [(5, "first_pass_completed"), (6, "trial_ended")])
def test_maxtpm_keeps_observed_end_when_load_holds_and_caps_first_pass(count, end_reason):
    backend = StandardMetrics({"maxtpm": True}).create_backend(seed=0)
    backend.open()
    try:
        backend.consume((session(0, 300), session(8, 9999), *occupancy((1, 192)),
                         *(turn(str(i), stamp) for i, stamp in enumerate((10, 20, 80, 140, 260))),
                         replace(turn("cycle", 9000, prompt=99999), session_id="session-8")))
        spec = replace(trial(count), load_profile=LoadProfile("test", target_concurrency=192))
        report = backend.finalize_trial(spec)
        assert report.diagnostics["maxtpm_start_turns"] == (2 if count == 5 else 3)
        assert report.diagnostics["maxtpm_end_reason"] == end_reason
        assert report.diagnostics["maxtpm_underfill_confirmed_ns"] is None
        expected_end = 260 * NANO + 1 if count == 5 else 9999 * NANO + 1
        assert report.diagnostics["maxtpm_end_ns"] == expected_end
        if count == 5:
            assert by_name(report.points, "maxtpm")["maxtpm_minutes"].value == 4
    finally:
        backend.close()


def test_maxtpm_dynamic_end_uses_192_requests_and_preserves_global_statistics():
    reports = []
    for enabled in (False, True):
        backend = StandardMetrics({"maxtpm": enabled, "pricing": PRICES}).create_backend(seed=0)
        backend.open()
        try:
            backend.consume((session(0, 301), *occupancy((1, 192), (175, 191)),
                *(turn(str(i), stamp, prompt=90, completion=10, cached=25)
                  for i, stamp in enumerate((10, 20, 30, 40, 60, 100, 140, 180, 240, 300)))))
            spec = replace(trial(10), load_profile=LoadProfile(
                "test", target_concurrency=192, max_active_sessions=384
            ))
            reports.append(backend.finalize_trial(spec))
        finally:
            backend.close()
    assert [p for p in reports[0].points if p.window == "overall"] == [
        p for p in reports[1].points if p.window == "overall"
    ]
    report = reports[1]
    assert report.diagnostics["maxtpm_start_ns"] == 40 * NANO
    assert report.diagnostics["maxtpm_end_ns"] == 175 * NANO
    assert report.diagnostics["maxtpm_underfill_confirmed_ns"] == 235 * NANO
    assert report.diagnostics["maxtpm_end_turns"] == 7
    summary = by_name(report.points, "maxtpm")
    assert summary["maxtpm_minutes"].value == 2
    assert summary["maxtpm"].value == 200
    assert summary["maxtpm_input_tpm"].value == 180
    assert summary["maxtpm_output_tpm"].value == 20
    assert summary["maxtpm_cache_hit_rate"].value == pytest.approx(25 / 90)
    assert summary["revenue_per_minute"].value == pytest.approx(0.43)
    assert summary["succeeded"].value == 4


def test_maxtpm_terminal_clock_covers_cancelled_turns_without_transport_completion():
    backend = StandardMetrics({"maxtpm": True}).create_backend(seed=0)
    backend.open()
    try:
        backend.consume((session(0, 5), *occupancy((1, 1)),
            *(replace(turn(str(i), stamp, status="cancelled"), response_completed_ns=None)
              for i, stamp in enumerate((10, 20, 80, 140, 200)))))
        report = backend.finalize_trial(trial(5))
        assert report.diagnostics["maxtpm_start_ns"] == 20 * NANO
        assert report.diagnostics["maxtpm_end_ns"] == 200 * NANO + 1
        assert report.diagnostics["maxtpm_excluded_tail_ns"] == 1
        assert by_name(report.points, "maxtpm")["maxtpm_minutes"].value == 3
        assert by_name(report.points, "maxtpm")["maxtpm"].value == 0
    finally:
        backend.close()


@pytest.mark.parametrize('mode', ['stream', 'non_stream'])
def test_maxtpm_latencies_use_exact_successful_completion_cohort_and_pooled_quantiles(mode):
    backend=StandardMetrics({'maxtpm':True}).create_backend(seed=0)
    backend.open()
    try:
        first=turn('first',20,prompt=90,completion=10,mode=mode)
        cross=replace(turn('cross',30,prompt=90,completion=20,mode=mode),
                      request_ns=5*NANO,first_ns=7*NANO,last_ns=8*NANO)
        normal=replace(turn('normal',100,prompt=90,completion=10,mode=mode),
                       request_ns=25*NANO,first_ns=28*NANO,last_ns=31*NANO)
        after=replace(turn('after',200,mode=mode),request_ns=26*NANO,
                      first_ns=27*NANO,last_ns=29*NANO)
        backend.consume((*occupancy((1, 1), (140, 0)),
            session(0,9999),session(8,9999),turn('before-2',17,mode=mode),
            turn('before-3',18,mode=mode),turn('before',19,mode=mode),first,cross,
            turn('failed',80,status='failed',mode=mode),normal,turn('boundary',140,mode=mode),after,
            turn('after2',201,mode=mode),
            replace(turn('cycle',40,prompt=999,completion=999,mode=mode),session_id='session-8')))
        report=backend.finalize_trial(trial(10))
        points={(p.metric,p.statistic):p for p in report.points if p.window=='maxtpm'}
        assert points['succeeded','value'].value==3
        assert points['workload_rps','value'].value==pytest.approx(3/120)
        assert points['workload_completion','value'].value==pytest.approx(40/120)
        assert points['e2e','am'].value==pytest.approx((19+25+75)/3)
        assert points['e2e','n'].value==points['e2e','eligible_n'].value==3
        assert points['e2e','am'].formula_version=='dataset-turn-maxtpm-latency-v1-completion-cohort'
        if mode=='stream':
            assert points['ttft','am'].value==2
            assert points['ttft','p50'].value==2 and points['ttft','p99'].value==3
            assert points['tpot','am'].value==pytest.approx((1000/9+1000/19+3000/9)/3)
            assert points['decode_throughput','am'].value==pytest.approx((9+19+3)/3)
            assert points['decode_throughput','time_wm'].value==pytest.approx(37/5)
            assert points['prefill_throughput','time_wm'].value==45
            speed=points['decode_throughput','time_wm']
            assert speed.valid_sample_count==speed.expected_sample_count==3
            assert speed.formula_version=='dataset-turn-maxtpm-speed-time-weighted-v1-completion-cohort'
            assert points['ttft','am'].valid_sample_count==points['ttft','am'].expected_sample_count==3
        else:
            assert points['ttft','am'].value is None
            assert points['ttft','am'].unavailable_reason=='non_streaming_response'
            assert points['decode_throughput','am'].value is None
            assert points['decode_throughput','time_wm'].value is None
            assert points['decode_throughput','time_wm'].valid_sample_count==0
            assert points['decode_throughput','time_wm'].unavailable_reason=='non_streaming_response'
        assert by_name(report.points,'maxtpm')['maxtpm_output_tpm'].value==20
    finally:
        backend.close()


def test_maxtpm_starts_at_40_percent_and_ends_at_sustained_underfill_with_zero_minute():
    backend = StandardMetrics({"maxtpm": True}).create_backend(seed=0)
    backend.open()
    try:
        backend.consume((
            *occupancy((1, 1), (155, 0)),
            session(0, 9999), session(8, 5),
            replace(turn("cycle", 5, prompt=999), session_id="session-8"),
            turn("before-2", 17, prompt=999), turn("before-3", 18, prompt=999),
            turn("before", 19, prompt=999),
            turn("first", 20, prompt=90, completion=10),
            replace(turn("cycle-in-window", 21, prompt=999), session_id="session-8"),
            turn("failed1", 80, status="failed"),
            turn("failed2", 90, status="failed"),
            turn("tail", 150, prompt=999),
            turn("last", 155, prompt=999),
            turn("after", 300), turn("after2", 301),
        ))
        report = backend.finalize_trial(trial(10))
        summary = by_name(report.points, "maxtpm")
        assert summary["maxtpm"].value == 50
        assert summary["maxtpm_minutes"].value == 2
        assert [p.value for p in report.points if p.window.startswith("maxtpm:minute:")
                and p.metric == "maxtpm"] == [100, 0]
        assert report.diagnostics["maxtpm_start_ns"] == 20 * NANO
        assert report.diagnostics["maxtpm_end_ns"] == 155 * NANO
        assert report.diagnostics["maxtpm_excluded_tail_ns"] == 15 * NANO
        assert report.diagnostics["maxtpm_start_turns"] == 4
        assert report.diagnostics["maxtpm_end_turns"] == 8
        assert report.diagnostics["maxtpm_progress_start_fraction"] == 0.4
        assert "maxtpm_progress_end_fraction" not in report.diagnostics
        assert report.diagnostics["maxtpm_underfill_confirmed_ns"] == 215 * NANO
        assert report.diagnostics["maxtpm_end_reason"] == "concurrency_under_target_60s"
        assert summary["maxtpm"].formula_version == "dataset-turn-maxtpm-v7-configurable-start-underfill-60s-first-pass"
    finally:
        backend.close()


@pytest.mark.parametrize("count,turns,reason", [
    (None, (), "dataset_size_not_reported"),
    (4, (), "dataset_start_percent_not_completed"),
    (4, (turn("one", 10),), "dataset_start_percent_not_completed"),
    (2, (turn("one", 10), turn("two", 30)), "no_complete_minute_in_maxtpm_window"),
    (1, (turn("one", 10),), "no_complete_minute_in_maxtpm_window"),
])
def test_maxtpm_short_or_partial_data_is_unavailable(count, turns, reason):
    backend = StandardMetrics({"maxtpm": True}).create_backend(seed=0)
    backend.open()
    try:
        backend.consume((*occupancy((1, 1)), session(0, 9999), *turns))
        points=backend.finalize_trial(trial(count)).points
        point = by_name(points, "maxtpm")["maxtpm"]
        assert point.value is None and point.unavailable_reason == reason
        latency=next(p for p in points if p.window=='maxtpm' and p.metric=='ttft' and p.statistic=='am')
        assert latency.value is None and latency.unavailable_reason==reason
    finally:
        backend.close()


def test_maxtpm_rounds_40_percent_counts_terminal_turns_and_not_retry_attempts():
    backend = StandardMetrics({"maxtpm": True}).create_backend(seed=0)
    backend.open()
    try:
        cancelled = replace(turn("cancel", 80, status="cancelled"), response_completed_ns=None)
        backend.consume((*occupancy((1, 1), (140, 0)), session(0, 9999),
            turn("before", 10),
            AttemptFact(**asdict(turn("retry", 11, status="failed"))),
            turn("start", 20, prompt=90, completion=10),
            cancelled, turn("end", 140, status="failed"), turn("after", 200),
        ))
        report = backend.finalize_trial(trial(5))
        assert report.diagnostics["maxtpm_start_turns"] == 2
        assert report.diagnostics["maxtpm_end_turns"] == 4
        assert report.diagnostics["maxtpm_end_ns"] == 140 * NANO
        assert by_name(report.points, "maxtpm")["maxtpm"].value == 50
    finally:
        backend.close()


def test_maxtpm_missing_usage_does_not_drop_a_minute_or_request():
    backend = StandardMetrics({"maxtpm": True}).create_backend(seed=0)
    backend.open()
    try:
        backend.consume((*occupancy((1, 1), (140, 0)), session(0, 9999),
            turn("start", 10, prompt=0, completion=0),
            turn("unknown", 20, prompt=None), turn("last", 140), turn("after", 200)))
        report = backend.finalize_trial(trial(4))
        assert by_name(report.points, "maxtpm")["maxtpm"].unavailable_reason == "usage_not_reported"
        assert by_name(report.points, "maxtpm")["maxtpm_minutes"].value == 2
        assert [p.value for p in report.points if p.window.startswith("maxtpm:minute:")
                and p.metric == "maxtpm"] == [None, 0]
    finally:
        backend.close()


def test_billing_counts_retries_and_failed_usage_once_not_again_as_turn():
    backend = StandardMetrics({"pricing": PRICES}).create_backend(seed=0)
    backend.open()
    try:
        failed = turn("a", 10, prompt=100, completion=10, cached=25, status="failed")
        final = replace(turn("a", 20, prompt=100, completion=10, cached=25), attempts=2)
        updates = backend.consume(
            (AttemptFact(**asdict(failed)), AttemptFact(**asdict(final)), final)
        )
        for update in updates:
            if hasattr(update, "metrics"):
                assert update.metrics["total_cost"].statistics["value"] == pytest.approx(0.235)
                assert update.metrics["total_cost"].unit == "CNY"
        summary = by_name(backend.finalize_trial(trial()).points)
        assert summary["total_cost"].value == pytest.approx(0.47)
        assert summary["input_cost"].value == 0.3
        assert summary["cached_input_cost"].value == 0.01
        assert summary["output_cost"].value == 0.16
        assert summary["cost_observed_attempts"].value == 2
        assert summary["cost_unavailable_attempts"].value == 0
        assert summary["total_cost"].unit == "CNY"
    finally:
        backend.close()


def test_maxtpm_explains_missing_cached_usage_without_changing_throughput_cohort():
    backend=StandardMetrics({'maxtpm':True,'pricing':PRICES}).create_backend(seed=0)
    backend.open()
    try:
        backend.consume((*occupancy((1, 1), (140, 0)),
                         session(0,9999),turn('start',10,prompt=100,completion=10,cached=0),
                         turn('missing-cache',20,prompt=100,completion=10),
                         turn('end',140,prompt=100,completion=10,cached=0),
                         turn('after',200,prompt=100,completion=10,cached=0)))
        report=backend.finalize_trial(trial(4))
        points=by_name(report.points,'maxtpm')
        assert points['maxtpm'].value is not None
        for name in ('maxtpm_cache_hit_rate','revenue_per_minute'):
            assert points[name].value is None
            assert points[name].unavailable_reason=='cached_usage_missing_or_invalid'
        assert report.diagnostics['maxtpm_missing_cached_requests']==1
    finally:
        backend.close()


def test_billing_preserves_partial_cost_and_does_not_assume_missing_cache_zero():
    backend = StandardMetrics({"pricing": PRICES}).create_backend(seed=0)
    backend.open()
    try:
        known = AttemptFact(**asdict(turn("known", 10, prompt=100, completion=10, cached=25)))
        unknown = AttemptFact(**asdict(turn("unknown", 20, prompt=100, completion=10)))
        backend.consume((known, unknown))
        summary = by_name(backend.finalize_trial(trial()).points)
        assert summary["total_cost"].value == pytest.approx(0.315)
        assert summary["total_cost"].availability == "partial"
        assert summary["total_cost"].unavailable_reason == "attempt_usage_incomplete"
        assert summary["input_cost"].value == 0.15
        assert summary["input_cost"].availability == "partial"
        assert summary["cached_input_cost"].value == 0.005
        assert summary["cached_input_cost"].availability == "partial"
        assert summary["observed_cost"].value == pytest.approx(0.235)
        assert summary["output_cost"].value == 0.16
        assert summary["output_cost"].availability == "available"
        assert summary["cost_unavailable_attempts"].value == 1
    finally:
        backend.close()


@pytest.mark.parametrize("known", [False, True])
def test_billing_distinguishes_no_observed_cost_from_observed_zero(known):
    backend = StandardMetrics({"pricing": PRICES}).create_backend(seed=0)
    backend.open()
    try:
        rows = [AttemptFact(**asdict(turn("unknown", 20, prompt=None, completion=None)))]
        if known:
            rows.append(AttemptFact(**asdict(turn("zero", 10, prompt=0, completion=0, cached=0))))
        backend.consume(tuple(rows))
        summary = by_name(backend.finalize_trial(trial()).points)
        assert summary["total_cost"].value == (0 if known else None)
        assert summary["total_cost"].availability == ("partial" if known else "unavailable")
    finally:
        backend.close()


def test_billing_partial_total_includes_all_known_components_without_guessing_input():
    backend = StandardMetrics({"pricing": PRICES}).create_backend(seed=0)
    backend.open()
    try:
        backend.consume((
            AttemptFact(**asdict(turn("only-output", 10, prompt=100, completion=10))),
            AttemptFact(**asdict(turn("only-input", 20, prompt=100, completion=None, cached=25))),
        ))
        summary = by_name(backend.finalize_trial(trial()).points)
        assert summary["total_cost"].value == pytest.approx(0.235)
        assert summary["total_cost"].availability == "partial"
        assert summary["input_cost"].value == 0.15
        assert summary["cached_input_cost"].value == 0.005
        assert summary["output_cost"].value == 0.08
        assert summary["observed_cost"].value == 0
    finally:
        backend.close()


@pytest.mark.parametrize("cached,status,value", [
    (None, "unavailable", None), (0, "partial", 0), (25, "partial", 25),
])
def test_cache_total_sums_reported_usage_without_filling_missing_requests(cached, status, value):
    backend = StandardMetrics({}).create_backend(seed=0)
    backend.open()
    try:
        backend.consume((turn("known", 10, prompt=100, cached=cached),
                         turn("unknown", 20, prompt=100)))
        summary = by_name(backend.finalize_trial(trial()).points)
        assert summary["total_cached_tokens"].value == value
        assert summary["total_cached_tokens"].availability == status
        assert summary["token_cache_token_hit_rate"].value is None
    finally:
        backend.close()


def test_billing_equal_input_prices_need_no_cache_split():
    backend = StandardMetrics({"pricing": {**PRICES, "cached_input_price": 2}}).create_backend(
        seed=0
    )
    backend.open()
    try:
        updates = backend.consume(
            (AttemptFact(**asdict(turn("a", 10, prompt=100, completion=10))),)
        )
        update = next(u for u in updates if getattr(u, "scope", None) == "attempt")
        assert update.metrics["total_cost"].statistics["value"] == 0.28
        assert update.metrics["cached_input_cost"].availability == "unavailable"
        assert by_name(backend.finalize_trial(trial()).points)["total_cost"].value == 0.28
    finally:
        backend.close()


@pytest.mark.parametrize(
    "settings",
    [
        {"maxtpm": 1},
        {"pricing": {}},
        {"pricing": {**PRICES, "input_price": -1}},
        {"pricing": {**PRICES, "input_price": float("inf")}},
        {"pricing": {**PRICES, "per_tokens": 0}},
        {"pricing": {**PRICES, "typo": 1}},
        {"pricing": {**PRICES, "currency": ""}},
    ],
)
def test_optional_statistics_reject_invalid_settings(settings):
    with pytest.raises(ValueError):
        parse_settings(settings)


def test_minute_revenue_uses_same_cohort_including_empty_minutes():
    backend = StandardMetrics({"maxtpm": True, "pricing": PRICES}).create_backend(seed=0)
    backend.open()
    try:
        # Start 20s, end 215s: three whole minutes plus a 15s tail.
        final = turn("first", 20, prompt=100, completion=10, cached=25)
        retry = replace(AttemptFact(**asdict(final)), status="failed")
        backend.consume(
            (
                session(0, 9999),
                *occupancy((1, 1), (215, 0)),
                turn("before-2", 17, prompt=999, cached=999),
                turn("before-3", 18, prompt=999, cached=999),
                turn("before", 19, prompt=999, cached=999),
                retry,
                final,
                turn("second", 80, prompt=300, completion=30, cached=75),
                turn("failed", 140, prompt=999, cached=999, status="failed"),
                turn("tail", 200, prompt=999, cached=999),
                turn("end", 215, status="failed"),
                turn("after", 300), turn("after2", 400),
            )
        )
        report = backend.finalize_trial(trial(10))
        summary = by_name(report.points, "maxtpm")
        assert summary["maxtpm_minutes"].value == 3
        assert summary["maxtpm"].value == pytest.approx(440 / 3)
        assert summary["maxtpm_input_tpm"].value == pytest.approx(400 / 3)
        assert summary["maxtpm_output_tpm"].value == pytest.approx(40 / 3)
        assert summary["maxtpm_cached_tpm"].value == pytest.approx(100 / 3)
        assert summary["maxtpm_uncached_tpm"].value == 100
        assert summary["maxtpm_cache_hit_rate"].value == 0.25  # weighted, empty minute retained
        assert summary["revenue_per_minute"].value == pytest.approx(0.94 / 3)
        assert summary["input_revenue_per_minute"].value == pytest.approx(0.6 / 3)
        assert summary["cached_input_revenue_per_minute"].value == pytest.approx(0.02 / 3)
        assert summary["output_revenue_per_minute"].value == pytest.approx(0.32 / 3)
        assert summary["revenue_per_minute"].unit == "CNY/min"
        assert [
            by_name(report.points, f"maxtpm:minute:{i:06d}")["revenue_per_minute"].value
            for i in (1, 2, 3)
        ] == pytest.approx([0.235, 0.705, 0])
    finally:
        backend.close()


@pytest.mark.parametrize("cache", [None, -1, 101])
def test_minute_revenue_keeps_unknown_cache_unknown(cache):
    backend = StandardMetrics({"maxtpm": True, "pricing": PRICES}).create_backend(seed=0)
    backend.open()
    try:
        backend.consume(
            (
                session(0, 9999),
                *occupancy((1, 1), (140, 0)),
                turn("start", 10, prompt=0, completion=0, cached=0),
                turn("unknown", 20, prompt=100, completion=10, cached=cache),
                turn("end", 140), turn("after", 200),
            )
        )
        summary = by_name(backend.finalize_trial(trial(4)).points, "maxtpm")
        assert summary["maxtpm"].value == 55
        assert summary["maxtpm_cached_tpm"].value is None
        assert summary["maxtpm_cache_hit_rate"].value is None
        assert summary["revenue_per_minute"].value is None
        assert summary["output_revenue_per_minute"].value == 0.04
    finally:
        backend.close()


def test_minute_revenue_equal_input_prices_can_price_unknown_cache():
    backend = StandardMetrics(
        {"maxtpm": True, "pricing": {**PRICES, "cached_input_price": 2}}
    ).create_backend(seed=0)
    backend.open()
    try:
        backend.consume(
            (*occupancy((1, 1), (140, 0)),
             session(0, 9999), turn("start", 10, prompt=0, completion=0, cached=0),
             turn("unknown", 20, prompt=100, completion=10), turn("end", 140), turn("after", 200))
        )
        summary = by_name(backend.finalize_trial(trial(4)).points, "maxtpm")
        assert summary["revenue_per_minute"].value == 0.14
        assert summary["cached_input_revenue_per_minute"].value is None
    finally:
        backend.close()


@pytest.mark.parametrize(
    "settings,has_tpm,has_revenue",
    [
        ({}, False, False),
        ({"pricing": PRICES}, False, False),
        ({"maxtpm": True}, True, False),
        ({"maxtpm": True, "pricing": PRICES}, True, True),
    ],
)
def test_optional_revenue_capabilities_have_configured_units(settings, has_tpm, has_revenue):
    capabilities = {m.name: m for m in StandardMetrics(parse_settings(settings)).describe().metrics}
    assert ("maxtpm_input_tpm" in capabilities) == has_tpm
    assert ("revenue_per_minute" in capabilities) == has_revenue
    if has_revenue:
        assert capabilities["revenue_per_minute"].unit == "CNY/min"
        assert capabilities["cached_input_revenue_per_minute"].unit == "CNY/min"
