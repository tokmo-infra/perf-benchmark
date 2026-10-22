"""Trial load qualification from the same committed measurement cohorts."""
from __future__ import annotations

import math

from perf_benchmark.pipeline.contracts.sweep import TrialValidity


def assess(backend, fact, points, windows):
    requirements = fact.validity_requirements
    if requirements is None:
        return TrialValidity(fact.trial_id, fact.execution_status, "not_applicable", ("validity_not_requested",))
    reasons, observations = [], []
    scheduling, profile = fact.scheduling, fact.load_profile
    if fact.execution_status != "succeeded":
        reasons.append("execution_" + fact.execution_status)
    for entity in ("attempts", "sessions"):
        offered, admitted, completed = (getattr(scheduling, prefix + entity)
                                       for prefix in ("offered_", "admitted_", "completed_"))
        if not offered or offered != admitted or admitted != completed:
            reasons.append("incomplete_" + entity + "_coverage")
    if scheduling.demand_exhausted and scheduling.termination_reason not in {"source_exhausted", "session_count_reached"}:
        reasons.append("premature_demand_exhaustion")
    for name in ("min_duration_s", "max_scheduler_lag_ms", "load_tolerance_ratio"):
        if getattr(requirements, name) is None:
            reasons.append(name + "_not_configured")
    window = next((w for w in windows if w.name == requirements.window), None)
    if window is None or not window.available:
        reasons.append("measurement_window_unavailable")
        return TrialValidity(fact.trial_id, fact.execution_status, "indeterminate", tuple(reasons))
    start, stop = scheduling.offering_started_ns, scheduling.request_offering_finished_ns
    if start is None or stop is None:
        reasons.append("offering_interval_unavailable")
        return TrialValidity(fact.trial_id, fact.execution_status, "indeterminate", tuple(reasons))
    start, stop = max(start, window.start_ns), min(stop, window.end_ns)
    duration = max(0, stop - start) / 1e9
    observations.append(("request_offering_duration", duration, "s"))
    if duration <= 0 or (requirements.min_duration_s is not None and duration < requirements.min_duration_s):
        reasons.append("insufficient_offering_duration")
    lag = backend._one("SELECT max(offered_ns-target_ns)/1000000.0 FROM attempts "
                       "WHERE trial_id=? AND offered_ns>=? AND offered_ns<?", [fact.trial_id, start, stop])[0]
    protocol_failures = backend._one("SELECT count(*) FROM attempts WHERE trial_id=? AND origin IN ('protocol','tool') "
                                     "AND offered_ns>=? AND offered_ns<?", [fact.trial_id, start, stop])[0]
    if protocol_failures:
        reasons.append("protocol_or_tool_contract_failure")
    observations.append(("scheduler_lag_max", lag, "ms"))
    if lag is None:
        reasons.append("scheduler_lag_unavailable")
    elif requirements.max_scheduler_lag_ms is not None and lag > requirements.max_scheduler_lag_ms:
        reasons.append("max_scheduler_lag_ms_violated")
    count = backend._one("SELECT count(*) FROM turns WHERE trial_id=? AND request_ns>=? AND request_ns<?",
                         [fact.trial_id, start, stop])[0]
    observations.append(("started_requests", count, "requests"))
    if count < requirements.min_samples:
        reasons.append("insufficient_samples")
    if profile.ramp_duration_s is not None:
        reasons.append("nonstationary_load")
    targets = [("request", profile.target_request_rate), ("session", profile.target_session_rate)]
    if any(target is not None for _, target in targets):
        for kind, target in targets:
            if target is None:
                continue
            end = (scheduling.session_offering_finished_ns if kind == "session" else stop)
            if end is None:
                reasons.append(kind + "_offering_interval_unavailable")
                continue
            end = min(end, stop, window.end_ns)
            elapsed = (end - start) / 1e9
            table = "sessions" if kind == "session" else "attempts"
            offered = backend._one(f"SELECT count(*) FROM {table} WHERE trial_id=? AND offered_ns>=? AND offered_ns<?",
                                   [fact.trial_id, start, end])[0]
            expected = target * elapsed + (start == scheduling.offering_started_ns)
            ratio = offered / expected if elapsed > 0 and expected > 0 else None
            observations.extend(((kind + "_offered", offered, kind + "s"),
                                 (kind + "_load_ratio", ratio, "ratio")))
            _pressure(ratio, requirements.load_tolerance_ratio, kind, reasons)
            admitted = backend._one(f"SELECT count(*) FROM {table} WHERE trial_id=? AND admitted_ns>=? AND admitted_ns<?",
                                    [fact.trial_id, start, end])[0]
            admitted_ratio = admitted / expected if elapsed > 0 and expected > 0 else None
            observations.append((kind + "_admitted_ratio", admitted_ratio, "ratio"))
            _pressure(admitted_ratio, requirements.load_tolerance_ratio, "client_" + kind + "_admission", reasons)
    else:
        sessions = fact.comparison is not None and fact.comparison.axis_name == "max_active_sessions"
        target = profile.max_active_sessions if sessions else profile.target_concurrency
        actual = backend._time_weighted(fact.trial_id, "active_sessions" if sessions else "in_flight", start, stop) if duration > 0 else None
        ratio = actual / target if actual is not None and target else None
        observations.extend((("actual_pressure", actual, "sessions" if sessions else "requests"),
                             ("load_ratio", ratio, "ratio")))
        _pressure(ratio, requirements.load_tolerance_ratio, "concurrency", reasons)
    return TrialValidity(fact.trial_id, fact.execution_status, "indeterminate" if reasons else "pass",
                         tuple(reasons), observations=tuple(observations))


def _pressure(ratio, tolerance, name, reasons):
    if ratio is None or not math.isfinite(ratio):
        reasons.append(name + "_pressure_unavailable")
    elif tolerance is not None and abs(ratio - 1) > tolerance:
        reasons.append(name + "_pressure_outside_tolerance")
