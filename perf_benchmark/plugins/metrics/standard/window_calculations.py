"""Window aggregation for the standard metrics backend."""

from __future__ import annotations

from perf_benchmark.pipeline.contracts.facts import TrialFact
from perf_benchmark.pipeline.contracts.results import MetricPoint
from perf_benchmark.plugins.metrics.standard.catalog import METRIC_CATALOG, OPTIONAL_METRICS
from perf_benchmark.plugins.metrics.standard.windows import MeasurementWindow
from perf_benchmark.plugins.metrics.standard.distribution_calculations import _finite

_GOODPUT_COLUMNS = {
    "ttft": "ttft / 1000.0",
    "tpot": "tpot",
    "e2e": "e2e / 1000.0",
}

class WindowCalculations:
    def _window_points(
        self,
        trial_id: str,
        window: MeasurementWindow,
        target: int | None,
        fact: TrialFact,
    ) -> list[MetricPoint]:
        if not window.available:
            return [
                self._point(
                    trial_id,
                    window.name,
                    metric,
                    statistic,
                    None,
                    "unavailable",
                    window.unavailable_reason,
                    None,
                )
                for metric, definition in METRIC_CATALOG.items() if metric not in OPTIONAL_METRICS
                for statistic in definition.statistics
            ]
        start, end = window.start_ns, window.end_ns
        if start is None or end is None:
            raise RuntimeError("available metric window has no bounds")
        duration = (end - start) / 1e9
        params = [trial_id, start, end]
        offered_where = "trial_id=? AND offered_ns>=? AND offered_ns<?"
        started_where = "trial_id=? AND request_ns>=? AND request_ns<?"
        completed_where = "trial_id=? AND response_completed_ns>=? AND response_completed_ns<?"
        offered = self._one(f"SELECT count(*) FROM turns WHERE {offered_where}", params)[0]
        admitted, succeeded, failed, cancelled, attempts, crossed = self._one(
            "SELECT count(*), count(*) FILTER (status='succeeded'), "
            "count(*) FILTER (status='failed'), "
            "count(*) FILTER (status='cancelled'), coalesce(sum(attempts),0), "
            "count(*) FILTER (response_completed_ns IS NOT NULL AND "
            "(response_completed_ns<? OR response_completed_ns>=?)) "
            f"FROM turns WHERE {started_where}",
            [start, end, *params],
        )
        completed, completed_succeeded = self._one(
            "SELECT count(*), count(*) FILTER (status='succeeded') "
            "FROM turns WHERE "
            f"{completed_where}",
            params,
        )
        session_started_where = "trial_id=? AND started_ns>=? AND started_ns<?"
        session_completed_where = "trial_id=? AND finished_ns>=? AND finished_ns<?"
        session_started = self._one(
            "SELECT count(*), count(*) FILTER (status='succeeded'), "
            "coalesce(sum(turns),0), coalesce(sum(completed_turns),0), "
            "count(*) FILTER (completed_turns < turns) FROM sessions WHERE "
            f"{session_started_where}",
            params,
        )
        sessions_started, started_sessions_succeeded = session_started[:2]
        expected_turns, completed_turns, incomplete_sessions = session_started[2:]
        sessions_completed, sessions_failed = self._one(
            "SELECT count(*) FILTER (status='succeeded'), "
            "count(*) FILTER (status='failed') FROM sessions WHERE "
            f"{session_completed_where}",
            params,
        )
        points: list[MetricPoint] = []

        def add_scalar(
            name: str,
            value,
            source: str = "measurement",
            reason: str = "not_observed",
        ) -> None:
            available = value is not None and _finite(value)
            points.append(
                self._point(
                    trial_id,
                    window.name,
                    name,
                    "value",
                    value if available else None,
                    "available" if available else "unavailable",
                    None if available else reason,
                    source if available else None,
                    valid=(admitted if name in {"success_rate", "cancellation_rate", "goodput_qualified", "goodput_rate", "goodput_percentage"}
                           else completed_succeeded if name == "workload_rps"
                           else sessions_started if name == "session_success_rate" else None),
                    expected=(admitted if name in {"success_rate", "cancellation_rate", "goodput_qualified", "goodput_rate", "goodput_percentage"}
                              else completed_succeeded if name == "workload_rps"
                              else sessions_started if name == "session_success_rate" else None),
                    missing=(0 if name in {"success_rate", "cancellation_rate", "goodput_qualified", "goodput_rate", "goodput_percentage", "workload_rps", "session_success_rate"} else None),
                )
            )

        attempt_count, failed_attempts, cancelled_attempts, policy_cancelled = self._one(
            "SELECT count(*), count(*) FILTER (status='failed'), "
            "count(*) FILTER (status='cancelled'), "
            "count(*) FILTER (status='cancelled' AND origin='scenario') "
            "FROM attempts WHERE trial_id=? AND started_ns>=? AND started_ns<?",
            params,
        )
        request_rate, scheduler_lag, scheduler_lag_max, request_queue_delay = self._one(
            "SELECT CASE WHEN count(*)>1 AND max(admitted_ns)>min(admitted_ns) "
            "THEN (count(*)-1)*1000000000.0/(max(admitted_ns)-min(admitted_ns)) END, "
            "avg(offered_ns-target_ns)/1000000.0, "
            "max(offered_ns-target_ns)/1000000.0, "
            "avg(admitted_ns-offered_ns)/1000000.0 FROM attempts "
            "WHERE trial_id=? AND admitted_ns>=? AND admitted_ns<?",
            params,
        )
        session_queue_delay = self._one(
            "SELECT avg(admitted_ns-offered_ns)/1000000.0 FROM sessions "
            "WHERE trial_id=? AND admitted_ns>=? AND admitted_ns<?",
            params,
        )[0]
        session_rate = self._one(
            "SELECT CASE WHEN count(*)>1 AND max(offered_ns)>min(offered_ns) "
            "THEN (count(*)-1)*1000000000.0/(max(offered_ns)-min(offered_ns)) END "
            "FROM sessions "
            "WHERE trial_id=? AND offered_ns>=? AND offered_ns<?",
            params,
        )[0]
        scalar_values = {
            "offered": offered,
            "admitted": admitted,
            "succeeded": succeeded,
            "failed": failed,
            "cancelled": cancelled,
            "cancellation_rate": cancelled / admitted if admitted else 0.0,
            "policy_cancelled": policy_cancelled,
            "policy_cancellation_rate": policy_cancelled / attempt_count if attempt_count else 0.0,
            "rejected": 0,
            "expired": 0,
            "success_rate": succeeded / admitted if admitted else 0.0,
            "workload_rps": completed_succeeded / duration,
            "target_concurrency": target,
            "attempts": attempts,
            "retries": max(0, attempts - admitted),
            "failed_attempts": failed_attempts,
            "cancelled_attempts": cancelled_attempts,
            "successful_concurrency": self._successful_concurrency(trial_id, start, end),
            "actual_in_flight_concurrency": self._time_weighted(trial_id, "in_flight", start, end),
            "actual_active_sessions": self._time_weighted(trial_id, "active_sessions", start, end),
            "reserved_sessions": self._time_weighted(trial_id, "reserved_sessions", start, end),
            "sessions_started": sessions_started,
            "sessions_completed": sessions_completed,
            "sessions_failed": sessions_failed,
            "session_completed_turns_total": completed_turns,
            "session_expected_turns_total": expected_turns,
            "session_incomplete_count": incomplete_sessions,
            "target_session_rate": fact.load_profile.target_session_rate,
            "actual_session_rate": session_rate,
            "target_request_rate": fact.load_profile.target_request_rate,
            "actual_request_rate": request_rate,
            "scheduler_lag": scheduler_lag,
            "scheduler_lag_max": scheduler_lag_max,
            "window_duration": duration * 1000.0,
            "session_queue_delay": session_queue_delay,
            "request_queue_delay": request_queue_delay,
            "session_pending_depth": self._time_weighted(
                trial_id, "session_pending", start, end
            ),
            "request_pending_depth": self._time_weighted(
                trial_id, "request_pending", start, end
            ),
            "drain_duration": fact.scheduling.drain_duration_ms,
            "partial_sessions": incomplete_sessions,
            "demand_limited": int(fact.scheduling.demand_limited),
            "session_success_rate": (
                started_sessions_succeeded / sessions_started if sessions_started else 0.0
            ),
        }
        for name, value in scalar_values.items():
            add_scalar(
                name,
                value,
                reason=(
                    "unbounded_by_scheduler"
                    if name == "target_concurrency" and target is None
                    else "closed_loop"
                    if name in {"target_session_rate", "target_request_rate"} and value is None
                    else "insufficient_arrivals"
                    if name in {"actual_session_rate", "actual_request_rate"} and value is None
                    else "not_observed"
                ),
            )
        active_target = fact.load_profile.max_active_sessions
        add_scalar(
            "target_active_sessions",
            active_target,
            reason="unbounded_by_scheduler" if active_target is None else "not_observed",
        )

        prompt_completed = self._usage_totals(
            completed_where, params, "prompt", int(completed_succeeded)
        )
        completion_completed = self._usage_totals(
            completed_where, params, "completion", int(completed_succeeded)
        )
        prompt_started = self._usage_totals(started_where, params, "prompt", int(succeeded))
        completion_started = self._usage_totals(started_where, params, "completion", int(succeeded))
        self._add_token_scalars(
            points,
            trial_id,
            window.name,
            duration,
            prompt_completed,
            completion_completed,
            int(completed_succeeded),
            prompt_started,
            completion_started,
            int(succeeded),
        )
        points.extend(
            self._cache_points(
                trial_id,
                window.name,
                completed_where,
                params,
                duration,
                int(completed_succeeded),
            )
        )
        goodput_qualified = goodput_missing = None
        if self.goodput_rules:
            predicates = []
            thresholds = []
            missing = []
            for rule in self.goodput_rules:
                column = _GOODPUT_COLUMNS[str(rule["metric"])]
                predicates.append(
                    f"({column}) {'<=' if rule['op'] == 'le' else '>='} ?"
                )
                thresholds.append(float(rule["threshold"]))
                missing.append(f"({column}) IS NULL")
            goodput_qualified = int(
                self._one(
                    "SELECT count(*) FROM turns WHERE "
                    f"{started_where} AND status='succeeded' AND "
                    + " AND ".join(predicates),
                    [*params, *thresholds],
                )[0]
            )
            goodput_missing = int(
                self._one(
                    "SELECT count(*) FROM turns WHERE "
                    f"{started_where} AND status='succeeded' AND ("
                    + " OR ".join(missing)
                    + ")",
                    params,
                )[0]
            )
            add_scalar("goodput_qualified", goodput_qualified, "goodput_slo")
            add_scalar("goodput_rate", goodput_qualified / duration, "goodput_slo")
            add_scalar(
                "goodput_percentage",
                goodput_qualified / admitted if admitted else 0.0,
                "goodput_slo",
            )
        else:
            for name in ("goodput_qualified", "goodput_rate", "goodput_percentage"):
                points.append(
                    self._point(
                        trial_id,
                        window.name,
                        name,
                        "value",
                        None,
                        "unavailable",
                        "slo_not_configured",
                        None,
                    )
                )

        points.extend(
            self._distribution_points(
                trial_id,
                window.name,
                start,
                end,
                succeeded,
                sessions_started,
            )
        )
        diagnostics = {
            "offered": int(offered),
            "started": int(admitted),
            "completed": int(completed),
            "latency_eligible": int(succeeded),
            "latency_missing": max(
                0,
                int(succeeded)
                - int(
                    self._one(
                        "SELECT count(e2e) FROM turns WHERE "
                        f"{started_where} AND status='succeeded'",
                        params,
                    )[0]
                ),
            ),
            "cross_window_completions": int(crossed),
        }
        if self.goodput_rules:
            assert goodput_qualified is not None and goodput_missing is not None
            diagnostics.update(
                goodput_denominator=int(admitted),
                goodput_qualified=goodput_qualified,
                goodput_missing_required_metric=goodput_missing,
            )
        self._window_diagnostics[window.name] = diagnostics
        return points
