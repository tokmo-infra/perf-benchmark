"""CPU-only coverage diagnostics over real temporary standard-metrics facts."""

from dataclasses import replace
import json

import pytest

from perf_benchmark.pipeline.contracts.facts import (
    LoadProfile, RepetitionPlan, ResourceLimits, TrialFact, TrialSchedulingFact,
)
from perf_benchmark.pipeline.results.measurement import OccupancyFact, SessionFact, TurnFact
from perf_benchmark.plugins.metrics.standard.backend import StandardMetricsBackend
from perf_benchmark.plugins.metrics.standard.cohort_diagnostics import completion_diagnostics
from perf_benchmark.plugins.metrics.standard.maxtpm import latency_points

SECOND = 10**9
START, END = 10 * SECOND, 70 * SECOND
WINDOW = {"window": "maxtpm", "availability": "available", "start_ns": START, "end_ns": END}


def turn(identifier="private-request-id", **changes):
    item = TurnFact(
        trial_id="trial", id=identifier, session_id="private-session-id", ordinal=0,
        status="succeeded", started_at=None, finished_at=None, error=None,
        estimated_input_tokens=None, reference_output_tokens=None, requested_output_tokens=None,
        usage={"prompt_tokens": 100, "completion_tokens": 4, "cached_tokens": 0},
        metadata={}, response_samples=(), target_ns=SECOND, offered_ns=SECOND,
        admitted_ns=SECOND, turn_started_ns=SECOND, turn_completed_ns=20 * SECOND,
        attempts=1, response_mode="stream", request_ns=SECOND, first_ns=2 * SECOND,
        last_ns=3 * SECOND, response_completed_ns=20 * SECOND, http_status=200,
        output_event_count=2, output_chars=8, finish_reason="stop",
    )
    return replace(item, **changes)


def trial(count):
    return TrialFact(
        "trial", "succeeded", LoadProfile("test", target_concurrency=1),
        ResourceLimits(1, 1, 1), None, RepetitionPlan(),
        TrialSchedulingFact(dataset_session_count=1, dataset_turn_count=count),
    )


@pytest.fixture
def backend():
    owner = StandardMetricsBackend({"maxtpm": True, "maxtpm_start_percent": 0}, seed=0)
    owner.open()
    owner.consume((
        SessionFact("private-session-id", "trial", 0, "succeeded", None, None, None, {}, 1,
                    SECOND, SECOND, SECOND, SECOND, 121 * SECOND, 1),
        OccupancyFact("trial", "in_flight", SECOND, 1),
    ))
    try:
        yield owner
    finally:
        owner.close()


def coverage(backend, *items):
    backend.consume(items)
    backend._flush_pending(force=True)
    before = latency_points(backend, "trial", START, 1, 1, None)
    result = completion_diagnostics(backend, trial(max(1, len(items))), [WINDOW],
                                    {"maxtpm_end_ns": END}, [(SECOND, 1)])
    # Diagnostics are read-only, with exactly the native completion cohort/counts.
    assert before == latency_points(backend, "trial", START, 1, 1, None)
    report = result["slo_cohort"]["sample_coverage"]
    for name, metric in report["metrics"].items():
        native = {p.statistic: p for p in before if p.metric == name}
        assert metric["valid_sample_count"] == native["n"].value
        assert metric["expected_sample_count"] == native["eligible_n"].value
        assert metric["missing_sample_count"] == native["n"].missing_sample_count
        assert metric["expected_sample_count"] == report["successful_sample_count"]
        assert metric["valid_sample_count"] + metric["missing_sample_count"] == metric["expected_sample_count"]
        assert sum(b["count"] for b in metric["missing_buckets"].values()) == metric["missing_sample_count"]
        for bucket in metric["missing_buckets"].values():
            assert sum(bucket["reasons"].values()) == bucket["count"]
    return report


@pytest.mark.parametrize("completion,reason", [(0, "insufficient_output_tokens"),
                                            (1, "insufficient_output_tokens"),
                                            (4, "insufficient_output_events")])
def test_natural_single_event_is_structural_without_invented_decode_samples(backend, completion, reason):
    result = coverage(backend, turn(output_event_count=1, last_ns=2 * SECOND,
                                   usage={"prompt_tokens": 100, "completion_tokens": completion}))
    assert result["metrics"]["ttft"]["valid_sample_count"] == 1
    for name in ("tpot", "decode_throughput"):
        metric = result["metrics"][name]
        assert metric["valid_sample_count"] == 0
        assert metric["missing_buckets"]["structurally_undefined"] == {"count": 1, "reasons": {reason: 1}}


def test_multiple_events_have_full_native_coverage(backend):
    result = coverage(backend, turn())
    assert result["available"] is True
    for metric in result["metrics"].values():
        assert metric["valid_sample_count"] == 1
        assert metric["missing_sample_count"] == 0


@pytest.mark.parametrize("events", [1, 2])
def test_missing_completion_usage_is_not_certified_structural(backend, events):
    result = coverage(backend, turn(usage={"prompt_tokens": 100}, output_event_count=events))
    assert result["metrics"]["ttft"]["valid_sample_count"] == 1
    for name in ("tpot", "decode_throughput"):
        assert result["metrics"][name]["missing_buckets"]["missing_metadata"] == {
            "count": 1, "reasons": {"usage_not_reported": 1},
        }


@pytest.mark.parametrize("changes,reason", [
    ({"output_kind": "non_text"}, "non_text_output"),
    ({"response_mode": "non_stream"}, "non_streaming_response"),
])
def test_protocol_structural_inapplicability(backend, changes, reason):
    result = coverage(backend, turn(**changes))
    for metric in result["metrics"].values():
        assert metric["missing_buckets"]["structurally_undefined"] == {
            "count": 1, "reasons": {reason: 1},
        }


@pytest.mark.parametrize("changes,names,kind,reason", [
    ({"request_ns": None}, ("ttft",), "missing_metadata", "request_timestamp_not_reported"),
    ({"first_ns": None}, ("ttft", "tpot", "decode_throughput"),
     "missing_metadata", "first_output_timestamp_not_reported"),
    ({"last_ns": None}, ("tpot", "decode_throughput"),
     "missing_metadata", "last_output_timestamp_not_reported"),
    ({"last_ns": 2 * SECOND}, ("tpot", "decode_throughput"),
     "invalid_interval", "decode_interval_not_observable"),
    ({"last_ns": SECOND}, ("tpot", "decode_throughput"),
     "invalid_interval", "decode_interval_not_observable"),
    ({"usage": {"completion_tokens": -1}}, ("tpot", "decode_throughput"),
     "invalid_metadata", "invalid_completion_tokens"),
    ({"output_event_count": -1}, ("tpot", "decode_throughput"),
     "invalid_metadata", "invalid_output_event_count"),
    ({"response_mode": None}, ("ttft", "tpot", "decode_throughput"),
     "missing_metadata", "response_mode_not_reported"),
    ({"output_kind": None}, ("ttft", "tpot", "decode_throughput"),
     "missing_metadata", "output_kind_not_reported"),
])
def test_missing_metadata_and_invalid_intervals_are_not_structural(backend, changes, names, kind, reason):
    result = coverage(backend, turn(**changes))
    for name in names:
        assert result["metrics"][name]["missing_buckets"][kind] == {"count": 1, "reasons": {reason: 1}}


def test_exact_completion_cohort_and_disjoint_mixed_counts(backend):
    backend.consume((SessionFact(
        "other-session", "trial", 1, "succeeded", None, None, None, {}, 1,
        SECOND, SECOND, SECOND, SECOND, 121 * SECOND, 1),))
    result = coverage(backend,
        turn("valid-at-start", response_completed_ns=START),
        turn("natural", output_event_count=1, last_ns=2 * SECOND, usage={"completion_tokens": 1}),
        turn("unknown-usage", usage={}),
        turn("invalid-interval", last_ns=2 * SECOND),
        turn("non-text", output_kind="non_text"),
        turn("failed", status="failed"), turn("cancelled", status="cancelled"),
        turn("before", response_completed_ns=START - 1),
        turn("at-end", response_completed_ns=END),
        turn("terminal-only", response_completed_ns=None, turn_completed_ns=20 * SECOND),
        turn("not-first-pass", session_id="other-session"),
        turn("not-this-trial", trial_id="other-trial"),
    )
    assert result["successful_sample_count"] == 5
    assert result["metrics"]["ttft"]["valid_sample_count"] == 4
    for name in ("tpot", "decode_throughput"):
        metric = result["metrics"][name]
        assert (metric["valid_sample_count"], metric["missing_sample_count"]) == (1, 4)
        assert {k: v["count"] for k, v in metric["missing_buckets"].items()} == {
            "structurally_undefined": 2, "missing_metadata": 1, "invalid_metadata": 0,
            "invalid_interval": 1, "unexplained": 0,
        }


def test_unexplained_stored_null_is_not_certified_structural(backend):
    backend.consume((turn(),))
    backend._flush_pending(force=True)
    backend.db.execute("UPDATE turns SET tpot=NULL,decode=NULL")
    result = coverage(backend)
    for name in ("tpot", "decode_throughput"):
        assert result["metrics"][name]["missing_buckets"]["unexplained"] == {
            "count": 1, "reasons": {"stored_sample_missing": 1},
        }


def test_observed_invalid_ttft_is_flagged_without_changing_native_samples(backend):
    result = coverage(backend, turn(first_ns=0))
    metric = result["metrics"]["ttft"]
    assert metric["valid_sample_count"] == 1
    assert metric["missing_sample_count"] == 0
    assert metric["observed_invalid_interval_sample_count"] == 1


def test_zero_success_keeps_no_samples_and_zero_counts(backend):
    result = coverage(backend, turn(status="failed"))
    assert result["available"] is False
    assert result["reason"] == "no_successful_completions"
    assert result["successful_sample_count"] == 0
    points = latency_points(backend, "trial", START, 1, 1, None)
    for point in points:
        if point.metric in result["metrics"] and point.statistic == "am":
            assert point.value is None and point.availability == "unavailable"
    assert all(metric["missing_sample_count"] == 0 for metric in result["metrics"].values())


def test_unavailable_window_preserves_existing_no_cohort_semantics(backend):
    result = completion_diagnostics(backend, trial(1), [], {}, [])
    assert result == {"slo_cohort": {"available": False, "reason": "native_window_unavailable"}}


def test_finalized_diagnostics_are_aggregate_only_and_do_not_change_points(backend, monkeypatch):
    backend.consume((turn(), turn("single", output_event_count=1, last_ns=2 * SECOND),
                     turn("excluded-tail", response_completed_ns=121 * SECOND + 1,
                          turn_completed_ns=121 * SECOND + 1)))
    report = backend.finalize_trial(trial(3))
    coverage = report.diagnostics["slo_cohort"]["sample_coverage"]
    assert coverage["successful_sample_count"] == 2
    import perf_benchmark.plugins.metrics.standard.cohort_diagnostics as module
    with monkeypatch.context() as patch:
        patch.setattr(module, "completion_diagnostics", lambda *args: {})
        baseline = backend.finalize_trial(trial(3))
    assert report.points == baseline.points
    assert report.counts == baseline.counts
    assert report.windows == baseline.windows
    serialized = json.dumps(coverage, allow_nan=False)
    assert "private-request-id" not in serialized and "private-session-id" not in serialized
    assert "response_samples" not in serialized and "turn_id" not in serialized
    # Summary diagnostics survive the usual release without retaining fact rows.
    backend.release_trial("trial")
    for table in ("turns", "attempts", "sessions", "itl", "occupancy"):
        assert backend.db.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
    assert json.dumps(coverage, allow_nan=False) == serialized
