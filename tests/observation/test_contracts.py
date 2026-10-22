from dataclasses import FrozenInstanceError

import pytest

from perf_benchmark.pipeline.contracts import ClockAnchor, ObservationSettings, TraceContext


@pytest.mark.parametrize("values", [
    {"unknown": 1}, {"enabled": 1}, {"trace_mode": True}, {"trace_mode": "full"}, {"trace_export": 1},
    {"sample_rate": -1}, {"sample_rate": float("nan")},
    {"poll_seconds": 0}, {"slow_seconds": float("inf")}, {"page_size": True},
    {"slow_spans_per_second": 0}, {"slow_spans_per_second": float("nan")},
    {"port": -1}, {"host": ""}, {"otlp_endpoint": "https://secret:password@host"},
    {"otlp_endpoint": "https://:password@host"}, {"otlp_endpoint": "https://host?token=secret"},
    {"otlp_endpoint": "http://host:bad"}, {"otlp_endpoint": "http://host:0"},
])
def test_strict_settings(values):
    with pytest.raises((ValueError, TypeError)):
        ObservationSettings.parse(values)


def test_defaults_and_frozen_configuration():
    settings = ObservationSettings.parse({})
    assert settings.trace_mode == "benchmark"
    assert settings.trace_export
    assert not settings.enabled
    assert (settings.poll_seconds, settings.sample_rate, settings.page_size) == (2, .01, 500)
    with pytest.raises(FrozenInstanceError):
        setattr(settings, "enabled", True)


@pytest.mark.parametrize("mode", ["benchmark", "observe", "debug"])
def test_trace_mode_preserves_observer_and_export_configuration(mode):
    settings = ObservationSettings.parse({"enabled": True, "trace_mode": mode, "sample_rate": .25,
                                          "trace_export": False})
    assert settings.trace_mode == mode and settings.enabled
    assert settings.sample_rate == .25
    assert not settings.trace_export


def test_retired_trace_boolean_is_rejected():
    with pytest.raises(ValueError, match="unknown observation settings"):
        ObservationSettings.parse({"trace_enabled": True})


def test_trace_identity_is_stable_across_retries_and_replay():
    first = TraceContext.create("run", "turn", 1, .01, seed=0, session_ordinal=0, turn_ordinal=0)
    retry = TraceContext.create("run", "turn", 2, .01, seed=0, session_ordinal=0, turn_ordinal=0)
    assert first == TraceContext.create("run", "turn", 1, .01, seed=0, session_ordinal=0, turn_ordinal=0)
    assert (first.trace_id, first.parent_span_id, first.sampled) == (
        retry.trace_id, retry.parent_span_id, retry.sampled,
    )
    assert first.span_id != retry.span_id
    assert first.trace_id != TraceContext.create("other", "turn", 1, .01, seed=0, session_ordinal=0, turn_ordinal=0).trace_id
    assert not TraceContext.create("run", "turn", 1, 0, seed=0, session_ordinal=0, turn_ordinal=0).sampled
    assert TraceContext.create("run", "turn", 1, 1, seed=0, session_ordinal=0, turn_ordinal=0).sampled
    assert first.traceparent == f"00-{first.trace_id}-{first.span_id}-{'01' if first.sampled else '00'}"


def test_clock_conversion_does_not_follow_wall_clock_jumps(monkeypatch):
    import perf_benchmark.core.observation as clock

    samples = iter([100, 110])
    monkeypatch.setattr(clock, "perf_counter_ns", lambda: next(samples))
    monkeypatch.setattr(clock, "time_ns", lambda: 1000)
    anchor = ClockAnchor.capture()
    assert anchor == ClockAnchor(105, 1000, 10)
    monkeypatch.setattr(clock, "time_ns", lambda: -100000)
    assert anchor.unix(120) - anchor.unix(110) == 10
    assert anchor.unix(120) == 1015


def test_trace_sampling_replays_across_runs_and_uses_global_seed():
    def traces(run, seed):
        return [TraceContext.create(run, f"{run}-turn-{i}", 1, .5,
                                   seed=seed, session_ordinal=i, turn_ordinal=0)
                for i in range(100)]
    first, replay, changed = traces("first", 42), traces("replay", 42), traces("first", 43)
    assert [trace.sampled for trace in first] == [trace.sampled for trace in replay]
    assert [trace.sampled for trace in first] != [trace.sampled for trace in changed]
    assert all(a.trace_id != b.trace_id for a, b in zip(first, replay))
    assert all(a.trace_id == b.trace_id for a, b in zip(first, changed))
