"""Distribution statistics and common metric point construction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any
import math

from perf_benchmark.plugins.metrics.standard.catalog import METRIC_CATALOG, public_value
from perf_benchmark.pipeline.contracts.results import MetricPoint

FORMULA_VERSION = "standard-window-v8"
_DISTRIBUTION_SOURCES = {
    "input_tokens": "server_usage",
    "output_tokens": "server_usage",
    "ttft": "protocol_output_event",
    "tpot": "protocol_output_event+server_usage",
    "e2e": "client_clock",
    "turn_total_duration": "client_clock",
    "time_to_last_output": "response_output_clock",
    "response_drain": "client_clock",
    "prefill_throughput": "server_usage",
    "decode_throughput": "server_usage",
    "itl": "protocol_output_event",
    "output_tokens_per_event_estimate": "estimated_protocol_events",
    "speculative_acceptance_proxy": "estimated_protocol_events",
    "session_duration": "scheduler",
    "session_turns": "scheduler",
    "session_first_turn_ttft": "protocol_output_event",
    "session_subsequent_turn_ttft": "protocol_output_event",
}

class DistributionCalculations:
    def _distribution_points(self, trial_id, window, start, end, succeeded, sessions_started, *, request_only=False, statistics=None, cohort=None):
        # One ranked aggregate over the latency/usage/session families. Cohorts stay
        # separate: request start for Turn statistics, Session start for Session stats.
        assert self.db is not None
        source = ("SELECT *, ordinal AS first_ordinal FROM turns WHERE trial_id=? "
                  "AND request_ns>=? AND request_ns<?" if request_only else
                  "SELECT *, min(ordinal) OVER (PARTITION BY session_id) AS first_ordinal "
                  "FROM turns WHERE trial_id=?")
        parameters = [trial_id, start, end] if request_only else [trial_id]
        where, cohort_parameters = cohort if cohort is not None else (
            "request_ns>=? AND request_ns<?", [start, end])
        # Only trimmed means need a per-metric rank. Live timelines publish no
        # trimmed means, so avoid sorting every sample for a discarded result.
        ranking = (
            "row_number() OVER (PARTITION BY metric ORDER BY value) AS rn, "
            "count(*) OVER (PARTITION BY metric) AS n"
            if statistics is None or "tm" in statistics else
            "NULL::BIGINT AS rn, NULL::BIGINT AS n"
        )
        rows = self.db.execute(
            f"""
            WITH trial AS (
                {source}
            ), cohort AS (
                SELECT * FROM trial WHERE {where} AND status='succeeded'
            ), samples AS (
                SELECT unnest(['input_tokens','output_tokens','ttft','tpot','e2e',
                               'time_to_last_output','response_drain',
                               'prefill_throughput','decode_throughput','turn_total_duration',
                               'output_tokens_per_event_estimate','speculative_acceptance_proxy']) AS metric,
                       unnest([prompt,completion,ttft,tpot,e2e,time_to_last_output,
                               response_drain,prefill,decode,turn_total_duration,event_yield,acceptance_proxy]) AS value FROM cohort
                UNION ALL
                SELECT 'itl', i.value FROM itl i JOIN cohort t USING(trial_id,turn_id)
                UNION ALL
                SELECT unnest(['session_duration','session_turns']), unnest([duration,turns])
                FROM sessions WHERE trial_id=? AND started_ns>=? AND started_ns<?
                UNION ALL
                SELECT CASE WHEN ordinal=first_ordinal THEN 'session_first_turn_ttft'
                            ELSE 'session_subsequent_turn_ttft' END, ttft
                FROM cohort WHERE session_id IS NOT NULL
            ), ranked AS (
                SELECT *, {ranking}
                FROM samples WHERE value IS NOT NULL
            )
            SELECT metric, count(*), avg(value), stddev_samp(value), min(value), max(value),
                   quantile_disc(value, [0.01,0.05,0.10,0.50,0.90,0.95,0.99]),
                   avg(value) FILTER (WHERE rn>floor(n*0.1) AND rn<=n-floor(n*0.1)),
                   count(*) FILTER (WHERE value>0), sum(1.0/value) FILTER (WHERE value>0)
            FROM ranked GROUP BY metric
        """,
            [*parameters, *cohort_parameters, trial_id, start, end],
        ).fetchall()
        by_name = {row[0]: row[1:] for row in rows}
        facts = self._one(
            f"""
            SELECT count(*), count(*) FILTER (response_mode='stream'),
                   count(first_ns), count(*) FILTER (output_event_count>=2),
                   count(completion), count(*) FILTER (completion>1),
                   count(*) FILTER (last_ns>first_ns), count(prompt),
                   sum(decode_ns) FILTER (decode_tokens>0 AND decode_ns IS NOT NULL)
                     /1000000.0/sum(decode_tokens) FILTER (decode_tokens>0 AND decode_ns IS NOT NULL),
                   sum(prefill*prompt) FILTER (prompt>0 AND prefill IS NOT NULL)
                     /sum(prompt) FILTER (prompt>0 AND prefill IS NOT NULL),
                   sum(decode*decode_tokens) FILTER (decode_tokens>0 AND decode IS NOT NULL)
                     /sum(decode_tokens) FILTER (decode_tokens>0 AND decode IS NOT NULL),
                   count(*) FILTER (output_kind='text'),
                   sum(prompt) FILTER (prompt>0 AND prefill IS NOT NULL AND ttft>0)
                     / (sum(ttft) FILTER (prompt>0 AND prefill IS NOT NULL AND ttft>0) / 1000.0),
                   sum(decode_tokens) FILTER (decode_tokens>0 AND decode IS NOT NULL AND decode_ns>0)
                     / (sum(decode_ns) FILTER (decode_tokens>0 AND decode IS NOT NULL AND decode_ns>0) / 1000000000.0),
                   count(*) FILTER (prompt>0 AND prefill IS NOT NULL AND ttft>0),
                   count(*) FILTER (decode_tokens>0 AND decode IS NOT NULL AND decode_ns>0)
            FROM turns WHERE trial_id=? AND {where} AND status='succeeded'
        """,
            [trial_id, *cohort_parameters],
        )
        weighted = dict(zip(("tpot", "prefill_throughput", "decode_throughput"), facts[8:11]))
        time_weighted = dict(zip(("prefill_throughput", "decode_throughput"), facts[12:14]))
        time_weighted_n = dict(zip(("prefill_throughput", "decode_throughput"), facts[14:16]))
        output = []
        for name, source in _DISTRIBUTION_SOURCES.items():
            eligible = (
                sessions_started
                if name in {"session_duration", "session_turns", "session_first_turn_ttft"}
                else max(0, succeeded - sessions_started)
                if name == "session_subsequent_turn_ttft"
                else succeeded
            )
            row = by_name.get(name)
            count = row[0] if row else 0
            definition = METRIC_CATALOG[name]
            stats: dict[str, tuple[float | None, str | None]] = {"n": (count, None), "eligible_n": (eligible, None)}
            if row:
                stats.update(
                    {
                        "am": (row[1], None),
                        "std": (row[2], "insufficient_samples"),
                        "min": (row[3], None),
                        "max": (row[4], None),
                        "tm": (row[6], None),
                        **{
                            key: (value, None)
                            for key, value in zip(
                                ("p1", "p5", "p10", "p50", "p90", "p95", "p99"), row[5]
                            )
                        },
                    }
                )
                if "hm" in definition.statistics:
                    positive, reciprocal = row[7:9]
                    stats["hm"] = (
                        positive / reciprocal if positive == count and reciprocal else None,
                        "non_positive_samples",
                    )
                if "twm" in definition.statistics:
                    stats["twm"] = (weighted[name], "insufficient_weight")
                if "time_wm" in definition.statistics:
                    stats["time_wm"] = (time_weighted[name], "insufficient_weight")
            reason = self._metric_empty_reason(name, facts[:8])
            if not row and name in {"output_tokens_per_event_estimate", "speculative_acceptance_proxy"}:
                reason = self._one(
                    "SELECT CASE WHEN count(DISTINCT event_yield_reason)=1 THEN min(event_yield_reason) "
                    f"ELSE 'no_eligible_samples' END FROM turns WHERE trial_id=? AND {where} "
                    "AND status='succeeded'", [trial_id, *cohort_parameters],
                )[0] if succeeded else "no_samples"
            if (
                facts[0]
                and not facts[11]
                and name
                in {
                    "ttft",
                    "tpot",
                    "itl",
                    "prefill_throughput",
                    "decode_throughput",
                    "session_first_turn_ttft",
                    "session_subsequent_turn_ttft",
                }
            ):
                reason = "non_text_output"
            for statistic in definition.statistics:
                if statistics is not None and statistic not in statistics:
                    continue
                value, unavailable = stats.get(statistic, (None, reason))
                available = value is not None and _finite(value)
                output.append(
                    self._point(
                        trial_id,
                        window,
                        name,
                        statistic,
                        value if available else None,
                        "available" if available else "unavailable",
                        None if available else unavailable,
                        source,
                        time_weighted_n[name] if statistic == "time_wm" else count,
                        eligible,
                        max(0, eligible - (time_weighted_n[name] if statistic == "time_wm" else count)),
                    )
                )
        return output

    def _usage_totals(
        self,
        where: str,
        params: Sequence[object],
        column: str,
        succeeded: int,
    ) -> tuple[int | None, int]:
        total, observed = self._one(
            f"SELECT sum({column}), count({column}) FROM turns "
            f"WHERE {where} AND status='succeeded'",
            params,
        )
        if succeeded == 0:
            return 0, 0
        return (int(total) if total is not None else None, int(observed))

    def _add_token_scalars(
        self,
        points: list[MetricPoint],
        trial_id: str,
        window: str,
        duration: float,
        prompt: tuple[int | None, int],
        completion: tuple[int | None, int],
        completed_succeeded: int,
        started_prompt: tuple[int | None, int],
        started_completion: tuple[int | None, int],
        started_succeeded: int,
    ) -> None:
        prompt_total, prompt_n = prompt
        completion_total, completion_n = completion
        started_prompt_total, started_prompt_n = started_prompt
        started_completion_total, started_completion_n = started_completion
        values = {
            "total_prompt_tokens": prompt_total,
            "total_completion_tokens": completion_total,
            "mean_input_tokens": (
                started_prompt_total / started_succeeded
                if started_succeeded
                and started_prompt_n == started_succeeded
                and started_prompt_total is not None
                else 0.0
                if not started_succeeded
                else None
            ),
            "mean_output_tokens": (
                started_completion_total / started_succeeded
                if started_succeeded
                and started_completion_n == started_succeeded
                and started_completion_total is not None
                else 0.0
                if not started_succeeded
                else None
            ),
            "workload_prompt": (
                prompt_total / duration
                if prompt_n == completed_succeeded and prompt_total is not None
                else None
            ),
            "workload_completion": (
                completion_total / duration
                if completion_n == completed_succeeded and completion_total is not None
                else None
            ),
            "workload_total": (
                (prompt_total + completion_total) / duration
                if prompt_n == completed_succeeded
                and completion_n == completed_succeeded
                and prompt_total is not None
                and completion_total is not None
                else None
            ),
        }
        for name, value in values.items():
            available = value is not None
            points.append(
                self._point(
                    trial_id,
                    window,
                    name,
                    "value",
                    value,
                    "available" if available else "unavailable",
                    None if available else "usage_not_reported",
                    "server_usage" if available else None,
                )
            )

    def _cache_points(
        self,
        trial_id: str,
        window: str,
        where: str,
        params: Sequence[object],
        duration: float,
        succeeded: int,
    ) -> list[MetricPoint]:
        observable, hits, cached, prompt = self._one(
            "SELECT count(cached), count(*) FILTER (cached>0), "
            "sum(cached), sum(prompt) FILTER (cached IS NOT NULL) "
            f"FROM turns WHERE {where} AND status='succeeded'",
            params,
        )
        observation_rate = observable / succeeded if succeeded else 0.0
        values = {
            "token_cache_observable_requests": observable,
            "token_cache_observation_rate": observation_rate,
        }
        output = [
            self._point(trial_id, window, name, "value", value, "available", None, "server_usage")
            for name, value in values.items()
        ]
        fully_observed = succeeded == 0 or observable == succeeded
        output.append(replace(
            self._point(
                trial_id, window, "total_cached_tokens", "value",
                int(cached or 0) if fully_observed or observable else None,
                "available" if fully_observed else "partial" if observable else "unavailable",
                None if fully_observed else "partial_usage" if observable else "usage_not_reported",
                "server_usage" if fully_observed or observable else None,
            ),
            formula_version="observed-cache-total-v1",
        ))
        if fully_observed:
            cached = int(cached or 0)
            prompt = int(prompt or 0)
            new_prompt = max(prompt - cached, 0)
            cache_values = {
                "total_new_prompt_tokens": new_prompt,
                "token_cache_hit_requests": int(hits),
                "token_cache_request_hit_rate": hits / observable if observable else 0.0,
                "token_cache_token_hit_rate": cached / prompt if prompt else 0.0,
                "workload_cached_prompt": cached / duration,
                "workload_new_prompt": new_prompt / duration,
            }
            for name, value in cache_values.items():
                output.append(
                    self._point(
                        trial_id, window, name, "value", value, "available", None, "server_usage"
                    )
                )
        else:
            reason = "usage_not_reported" if not observable else "partial_usage"
            for name in (
                "total_new_prompt_tokens",
                "token_cache_hit_requests",
                "token_cache_request_hit_rate",
                "token_cache_token_hit_rate",
                "workload_cached_prompt",
                "workload_new_prompt",
            ):
                output.append(
                    self._point(trial_id, window, name, "value", None, "unavailable", reason, None)
                )
        return output

    def _successful_concurrency(
        self,
        trial_id: str,
        start: int,
        end: int,
    ) -> float:
        total = self._one(
            "SELECT coalesce(sum(greatest(0, "
            "least(response_completed_ns,?)-greatest(request_ns,?))),0) "
            "FROM turns "
            "WHERE trial_id=? AND status='succeeded' "
            "AND request_ns<? AND response_completed_ns>?",
            [end, start, trial_id, end, start],
        )[0]
        return float(total) / max(1, end - start)

    @staticmethod
    def _metric_empty_reason(name: str, row: Sequence[int]) -> str:
        if name not in {
            "ttft",
            "tpot",
            "itl",
            "response_drain",
            "prefill_throughput",
            "decode_throughput",
            "session_first_turn_ttft",
            "session_subsequent_turn_ttft",
        }:
            return "no_samples"
        eligible, streaming, first, event_pairs = map(int, row[:4])
        usage, multi_token, progressing, prompt = map(int, row[4:])
        if not eligible:
            return "no_samples"
        if not streaming:
            return "non_streaming_response"
        if name in {
            "ttft",
            "prefill_throughput",
            "session_first_turn_ttft",
            "session_subsequent_turn_ttft",
        }:
            if not first:
                return "no_output_events"
            if name == "prefill_throughput" and not prompt:
                return "usage_not_reported"
            return "no_samples"
        if name == "response_drain":
            return "no_output_events" if not first else "no_samples"
        if not event_pairs:
            return "insufficient_output_events"
        if name == "itl":
            return "no_samples"
        if not usage:
            return "usage_not_reported"
        if not multi_token:
            return "insufficient_output_tokens"
        if not progressing:
            return "decode_interval_not_observable"
        return "no_samples"

    def _time_weighted(
        self,
        trial_id: str,
        kind: str,
        start: int,
        end: int,
    ) -> float:
        return self._one(
            """
            SELECT coalesce(sum(value::HUGEINT * greatest(
                0, least(next_ns, ?)-greatest(at_ns, ?)
            )), 0) / greatest(1, ?-?)
            FROM (
                SELECT at_ns, value, lead(at_ns, 1, ?) OVER (ORDER BY at_ns, rowid) AS next_ns
                FROM occupancy WHERE trial_id=? AND kind=? AND at_ns<=?
            )
        """,
            [end, start, end, start, end, trial_id, kind, end],
        )[0]

    def _one(
        self,
        query: str,
        params: Sequence[object] = (),
    ) -> tuple[Any, ...]:
        assert self.db is not None
        row = self.db.execute(query, params).fetchone()
        if row is None:
            raise RuntimeError("metrics aggregate query returned no row")
        return row

    @staticmethod
    def _point(
        scope_id: str,
        window: str,
        metric: str,
        statistic: str,
        value,
        availability: str,
        reason: str | None,
        source: str | None,
        valid: int | None = None,
        expected: int | None = None,
        missing: int | None = None,
    ) -> MetricPoint:
        return MetricPoint(
            "trial",
            scope_id,
            window,
            metric,
            statistic,
            public_value(METRIC_CATALOG[metric], statistic, float(value)) if value is not None else None,
            "samples" if statistic in {"n", "eligible_n"} else METRIC_CATALOG[metric].unit,
            ("dataset-turn-maxtpm-speed-time-weighted-v1-completion-cohort"
             if window == "maxtpm" else "client-speed-time-weighted-v1")
            if statistic == "time_wm" else FORMULA_VERSION,
            availability,
            reason,
            source,
            valid,
            expected,
            missing,
        )


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
