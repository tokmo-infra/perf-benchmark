"""Additional diagnostics over unchanged native completion/window definitions.

Computed on the native owner before temporary facts are released. Nothing is
inferred from logs, target C, reference token counts or summary_only details.
"""
from itertools import groupby


_MISSING_REASONS = {
    "structurally_undefined": {
        "non_streaming_response", "non_text_output", "no_output_events",
        "insufficient_output_tokens", "insufficient_output_events",
    },
    "missing_metadata": {
        "response_mode_not_reported", "output_kind_not_reported", "usage_not_reported",
        "output_event_count_not_reported", "request_timestamp_not_reported",
        "first_output_timestamp_not_reported", "last_output_timestamp_not_reported",
    },
    "invalid_metadata": {
        "invalid_response_mode", "invalid_completion_tokens", "invalid_output_event_count",
    },
    "invalid_interval": {
        "invalid_ttft_interval", "decode_interval_not_observable",
    },
    "unexplained": {"stored_sample_missing"},
}


def _sample_coverage(backend, trial, sessions, start, end):
    """Bounded aggregates from the exact native latency_points success cohort.

    Native eligibility still includes ALL successful completions, including
    structurally undefined samples. Valid means stored non-NULL, exactly as in
    native distributions, not a new timing-validation gate. Missing samples get
    one primary reason: known non-stream/non-text first, then unknown/invalid
    metadata, then short outputs, then absent/invalid timestamps. Thus a short
    event trace with unknown completion usage is NOT certified structural.
    Bucket counts sum to missing; valid + missing equals expected. No request
    identities, traces or detail rows leave the temporary metrics owner.
    """
    rows = backend.db.execute(
        """
        WITH cohort AS (
            SELECT * FROM turns WHERE trial_id=? AND status='succeeded'
              AND response_completed_ns>=? AND response_completed_ns<?
              AND session_id IN (SELECT session_id FROM sessions
                                 WHERE trial_id=? AND ordinal<?)
        ), samples AS (
            SELECT *, unnest(['ttft','tpot','decode_throughput']) AS metric,
                   unnest([ttft,tpot,decode]) AS value FROM cohort
        ), classified AS (
            SELECT *, CASE
                WHEN value IS NOT NULL THEN NULL
                WHEN response_mode='non_stream' THEN 'non_streaming_response'
                WHEN output_kind IS NOT NULL AND output_kind<>'text' THEN 'non_text_output'
                WHEN response_mode IS NULL THEN 'response_mode_not_reported'
                WHEN response_mode<>'stream' THEN 'invalid_response_mode'
                WHEN output_kind IS NULL THEN 'output_kind_not_reported'
                WHEN metric<>'ttft' AND completion IS NULL THEN 'usage_not_reported'
                WHEN metric<>'ttft' AND completion<0 THEN 'invalid_completion_tokens'
                WHEN output_event_count IS NULL THEN 'output_event_count_not_reported'
                WHEN output_event_count<0 THEN 'invalid_output_event_count'
                WHEN metric<>'ttft' AND completion<=1 THEN 'insufficient_output_tokens'
                WHEN metric<>'ttft' AND output_event_count<2 THEN 'insufficient_output_events'
                WHEN metric='ttft' AND output_event_count=0 THEN 'no_output_events'
                WHEN metric='ttft' AND request_ns IS NULL THEN 'request_timestamp_not_reported'
                WHEN first_ns IS NULL THEN 'first_output_timestamp_not_reported'
                WHEN metric<>'ttft' AND last_ns IS NULL THEN 'last_output_timestamp_not_reported'
                WHEN metric='ttft' AND first_ns<request_ns THEN 'invalid_ttft_interval'
                WHEN metric<>'ttft' AND last_ns<=first_ns THEN 'decode_interval_not_observable'
                ELSE 'stored_sample_missing' END AS reason
            FROM samples
        )
        SELECT metric, reason, count(*), count(*) FILTER (WHERE value IS NOT NULL AND
            ((metric='ttft' AND first_ns<request_ns) OR
             (metric<>'ttft' AND last_ns<=first_ns)))
        FROM classified GROUP BY metric, reason ORDER BY metric, reason
        """, [trial, start, end, trial, sessions],
    ).fetchall()
    metrics = {name: {
        "expected_sample_count": 0, "valid_sample_count": 0, "missing_sample_count": 0,
        "observed_invalid_interval_sample_count": 0,
        "missing_buckets": {kind: {"count": 0, "reasons": {}} for kind in _MISSING_REASONS},
    } for name in ("ttft", "tpot", "decode_throughput")}
    reason_kinds = {reason: kind for kind, reasons in _MISSING_REASONS.items() for reason in reasons}
    for name, reason, count, invalid in rows:
        metric = metrics[name]
        metric["expected_sample_count"] += int(count)
        metric["observed_invalid_interval_sample_count"] += int(invalid)
        if reason is None:
            metric["valid_sample_count"] += int(count)
        else:
            metric["missing_sample_count"] += int(count)
            bucket = metric["missing_buckets"][reason_kinds[reason]]
            bucket["count"] += int(count)
            bucket["reasons"][reason] = int(count)
    count = metrics["ttft"]["expected_sample_count"]
    return {"available": count > 0, "reason": None if count else "no_successful_completions",
            "basis": "native_maxtpm_first_pass_successful_response_completion_cohort",
            "native_eligibility_unchanged": True, "successful_sample_count": count,
            "metrics": metrics}


def occupancy_summary(rows, start, end, target):
    if start is None or end is None or end <= start:
        return {"available": False, "reason": "no_positive_window"}
    value, at, segments = None, start, []
    for stamp, changes in groupby(rows, key=lambda r: r[0]):
        final = list(changes)[-1][1]
        if stamp <= start:
            value = final
            continue
        if value is None:
            return {"available": False, "reason": "missing_start_anchor"}
        until = min(stamp, end)
        if until > at:
            segments.append((until-at, value))
        if stamp >= end:
            at = end
            break
        at, value = stamp, final
    if value is None:
        return {"available": False, "reason": "missing_start_anchor"}
    if at < end:
        segments.append((end-at, value))
    duration = end-start
    covered = sum(d for d, _ in segments)
    return {"available": covered == duration, "basis": "native_in_flight_transitions",
            "start_ns": start, "end_ns": end, "covered_ns": covered,
            "time_weighted_mean": sum(d*v for d, v in segments)/duration,
            "min": min(v for _, v in segments), "max": max(v for _, v in segments),
            "at_or_above_target_fraction": sum(d for d, v in segments if v >= target)/duration,
            "transition_count": sum(start <= t < end for t, _ in rows),
            "not_server_running_or_active_sessions": True}


def completion_diagnostics(backend, fact, windows, diagnostics, occupancy):
    window = next((w for w in windows if w["window"] == "maxtpm"), None)
    if not window or window["availability"] != "available":
        return {"slo_cohort": {"available": False, "reason": "native_window_unavailable"}}
    start, end = window["start_ns"], window["end_ns"]
    trial, sessions = fact.trial_id, fact.scheduling.dataset_session_count
    terminal = backend._one(
        "SELECT count(*), count(*) FILTER (status='succeeded'), "
        "count(*) FILTER (status='failed'), count(*) FILTER (status='cancelled') "
        "FROM turns WHERE trial_id=? AND session_id IN "
        "(SELECT session_id FROM sessions WHERE trial_id=? AND ordinal<?) "
        "AND coalesce(response_completed_ns,turn_completed_ns)>=? "
        "AND coalesce(response_completed_ns,turn_completed_ns)<?",
        [trial, trial, sessions, start, end])
    attempt = backend._one(
        "SELECT count(*), count(*) FILTER (http_status=500), "
        "count(*) FILTER (http_status IS NULL), count(*) FILTER (status='failed'), "
        "count(*) FILTER (status='cancelled') FROM attempts WHERE trial_id=? "
        "AND turn_id IN (SELECT turn_id FROM turns WHERE trial_id=? AND session_id IN "
        "(SELECT session_id FROM sessions WHERE trial_id=? AND ordinal<?)) "
        "AND completed_ns>=? AND completed_ns<?",
        [trial, trial, trial, sessions, start, end])
    n, succeeded, failed, cancelled = map(int, terminal)
    an, http500, unknown, afailed, acancelled = map(int, attempt)
    target = fact.load_profile.target_concurrency
    transport = backend.db.execute(
        "SELECT transport_sent_ns,completed_ns,transport_observed FROM attempts WHERE trial_id=?",
        [trial]).fetchall()
    transport_rows, transport_complete = transport_timeline(transport)
    actual_sent = occupancy_summary(transport_rows, start, end, target)
    actual_sent['available'] = actual_sent.get('available',False) and transport_complete
    actual_sent['basis'] = 'all_typed_attempt_transport_dispatch_to_response_completion; not_server_running'
    actual_sent['all_attempt_timings_observed'] = transport_complete
    return {"slo_cohort": {"available": True, "start_ns": start, "end_ns": end,
        "basis": "first_pass_terminal_completion_cohort",
        "terminal": n, "succeeded": succeeded, "failed": failed, "cancelled": cancelled,
        "failed_rate": failed/n if n else None, "cancelled_rate": cancelled/n if n else None,
        "non_success_rate": (failed+cancelled)/n if n else None,
        "attempts": an, "attempt_failed": afailed, "attempt_cancelled": acancelled,
        "http500_attempts": http500, "http_status_unknown_attempts": unknown,
        "http500_count_is_exact": unknown == 0,
        "http500_rate_is_lower_bound": unknown > 0,
        "http500_rate": http500/an if an else None,
        "attempt_basis": "typed_attempt_completion_clock; not_final_turn_status",
        "sample_coverage": _sample_coverage(backend, trial, sessions, start, end),
        "occupancy": occupancy_summary(occupancy, start, end, target),
        "actual_sent_concurrency": actual_sent,
        "raw_interval_occupancy": occupancy_summary(occupancy, start,
            diagnostics.get("maxtpm_end_ns"), target)}}


def transport_timeline(attempts):
    events, complete = [], bool(attempts)
    for sent, end, observed in attempts:
        complete = complete and observed is True
        if sent is None:
            continue  # proven not dispatched, e.g. connect failure or queue cancellation
        if end is None or end < sent:
            complete = False
            continue
        events.extend(((sent,1),(end,-1)))
    value, rows = 0, [(0,0)]
    for at, group in groupby(sorted(events), key=lambda p:p[0]):
        value += sum(change for _,change in group)
        complete = complete and value >= 0
        rows.append((at,value))
    return rows, bool(complete and value == 0)
