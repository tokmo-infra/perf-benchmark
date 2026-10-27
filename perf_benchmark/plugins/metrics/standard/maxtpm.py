"""Complete minute throughput and priced revenue in the same completion cohort."""

from dataclasses import replace
from itertools import groupby
from decimal import Decimal
import math
from perf_benchmark.plugins.metrics.standard.catalog import METRIC_CATALOG

MINUTE_NS = 60_000_000_000
FORMULA_VERSION = "dataset-turn-maxtpm-v7-configurable-start-underfill-60s-first-pass"
TOKEN_NAMES = (
    "maxtpm",
    "maxtpm_input_tpm",
    "maxtpm_output_tpm",
    "maxtpm_cached_tpm",
    "maxtpm_uncached_tpm",
)
REVENUE_NAMES = (
    "revenue_per_minute",
    "input_revenue_per_minute",
    "cached_input_revenue_per_minute",
    "output_revenue_per_minute",
)
LATENCY_NAMES = ("ttft", "tpot", "e2e", "itl", "prefill_throughput", "decode_throughput",
                 "time_to_last_output", "response_drain", "input_tokens", "output_tokens")


def parse_start_percent(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not 0 <= value < 100 or not math.isfinite(value)):
        raise ValueError("metrics maxtpm_start_percent must be finite and in [0, 100)")
    return float(value)


def latency_points(backend, trial, start, minutes, session_count, reason):
    """Whole-request metrics for the exact successful TPM completion cohort."""
    if not minutes:
        return [backend._point(trial, 'maxtpm', name, statistic, None, 'unavailable', reason, None)
                for name in (*LATENCY_NAMES, 'workload_completion', 'workload_rps', 'succeeded')
                for statistic in METRIC_CATALOG[name].statistics]
    end=start+minutes*MINUTE_NS
    where=("response_completed_ns>=? AND response_completed_ns<? AND session_id IN "
           "(SELECT session_id FROM sessions WHERE trial_id=? AND ordinal<?)")
    params=[start,end,trial,session_count]
    succeeded=int(backend._one(
        f"SELECT count(*) FROM turns WHERE trial_id=? AND {where} AND status='succeeded'",
        [trial,*params])[0])
    points=[p if p.statistic == 'time_wm' else replace(p,formula_version='dataset-turn-maxtpm-latency-v1-completion-cohort')
            for p in backend._distribution_points(trial,'maxtpm',start,end,succeeded,0,
                cohort=(where,params)) if p.metric in LATENCY_NAMES]
    duration=minutes*60.0
    total, observed=backend._usage_totals('trial_id=? AND '+where,[trial,*params],
                                         'completion',succeeded)
    for name,value in (('succeeded',succeeded),('workload_rps',succeeded/duration),
                       ('workload_completion',total/duration if observed==succeeded and total is not None else None)):
        points.append(backend._point(trial,'maxtpm',name,'value',value,
            'available' if value is not None else 'unavailable',
            None if value is not None else 'output_usage_missing_or_invalid',
            'server_usage+completion_clock',succeeded,succeeded,0))
    return points


def usage_reason(metric, missing):
    prompt, output, cached = missing
    if prompt and metric != 'maxtpm_output_tpm' and metric != 'output_revenue_per_minute':
        return 'prompt_usage_missing_or_invalid'
    if output and metric in {'maxtpm','maxtpm_output_tpm','revenue_per_minute','output_revenue_per_minute'}:
        return 'output_usage_missing_or_invalid'
    if cached and metric in {'maxtpm_cached_tpm','maxtpm_uncached_tpm','maxtpm_cache_hit_rate',
                             'revenue_per_minute','input_revenue_per_minute','cached_input_revenue_per_minute'}:
        return 'cached_usage_missing_or_invalid'
    return 'usage_missing_or_invalid'


def minute_values(row, pricing):
    # Missing usage affects only the metrics requiring that field; keep the cohort intact.
    _, _, bad_prompt, bad_output, bad_cache, prompt, output, cached = row
    prompt = None if bad_prompt else int(prompt or 0)
    output = None if bad_output else int(output or 0)
    cached = None if bad_cache or bad_prompt else int(cached or 0)
    uncached = prompt - cached if prompt is not None and cached is not None else None
    values = dict(
        zip(
            TOKEN_NAMES,
            (
                prompt + output if prompt is not None and output is not None else None,
                prompt,
                output,
                cached,
                uncached,
            ),
        )
    )
    values["maxtpm_cache_hit_rate"] = cached / prompt if prompt and cached is not None else None
    if pricing is not None:

        def price(tokens, key):
            return (
                (Decimal(tokens) * Decimal(str(pricing[key])) / Decimal(pricing["per_tokens"]))
                if tokens is not None
                else None
            )

        raw = (
            price(uncached, "input_price"),
            price(cached, "cached_input_price"),
            price(output, "output_price"),
        )
        input_total = (
            raw[0] + raw[1]
            if raw[0] is not None and raw[1] is not None
            else (
                price(prompt, "input_price")
                if pricing["input_price"] == pricing["cached_input_price"]
                else None
            )
        )
        total = input_total + raw[2] if input_total is not None and raw[2] is not None else None
        values.update(
            {
                name: float(value) if value is not None else None
                for name, value in zip(REVENUE_NAMES, (total, *raw))
            }
        )
    return values


def _concurrency_end(rows, start, end, target):
    """Exclude the first underfilled spell lasting at least one full minute.

    Equal-time transitions have zero duration; only their final value applies.
    A recovery resets the timer. The observation horizon is never extended.
    """
    value = None
    under_since = None
    for stamp, changes in groupby(rows, key=lambda row: row[0]):
        final = None
        for _, final in changes:
            pass
        if stamp <= start:
            value = final
            under_since = start if value < target else None
            continue
        if value is None:
            return None, None, "occupancy_not_observed_at_maxtpm_start"
        if under_since is not None and min(stamp, end) - under_since >= MINUTE_NS:
            return under_since, under_since + MINUTE_NS, "concurrency_under_target_60s"
        if stamp >= end:
            break
        value = final
        if value >= target:
            under_since = None
        elif under_since is None:
            under_since = stamp
    if value is None:
        return None, None, "occupancy_not_observed_at_maxtpm_start"
    if under_since is not None and end - under_since >= MINUTE_NS:
        return under_since, under_since + MINUTE_NS, "concurrency_under_target_60s"
    return end, None, None


def dataset_maxtpm(backend, fact, bounds, occupancy):
    trial = fact.trial_id
    count = fact.scheduling.dataset_turn_count
    percent = parse_start_percent(backend.settings.get("maxtpm_start_percent", 40))
    numerator, denominator = (Decimal(str(percent)) / 100).as_integer_ratio()
    lower = (count * numerator + denominator - 1) // denominator if count else None
    target = fact.load_profile.target_concurrency
    start = end_boundary = None
    confirmed_ns = end_reason = None
    end_turns = None
    reason = "dataset_size_not_reported"
    pricing = backend.billing.pricing
    names = (*TOKEN_NAMES, "maxtpm_cache_hit_rate", *(REVENUE_NAMES if pricing else ()))
    if count:
        row = backend._one(
            "SELECT count(*), max(completed_ns) FILTER (WHERE position=?), "
            "max(completed_ns) FROM ("
            "SELECT coalesce(t.response_completed_ns,t.turn_completed_ns) AS completed_ns, "
            "row_number() OVER (ORDER BY coalesce(t.response_completed_ns,t.turn_completed_ns), "
            "s.ordinal,t.ordinal,t.turn_id) AS position "
            "FROM turns t JOIN sessions s ON s.trial_id=t.trial_id AND s.session_id=t.session_id "
            "WHERE t.trial_id=? AND s.ordinal<? "
            "AND coalesce(t.response_completed_ns,t.turn_completed_ns) IS NOT NULL "
            ")",
            [lower, trial, fact.scheduling.dataset_session_count],
        )
        if lower == 0:
            start = bounds[0]
        elif row[0] >= lower:
            start = int(row[1])
        if start is None:
            reason = "dataset_start_percent_not_completed"
        elif target is None:
            reason = "target_concurrency_not_configured"
        else:
            # Keep repeated sessions and post-first-pass cleanup out of the horizon.
            # Cancelled Turns can finish after their last transport timestamp.
            terminal_end = int(row[2]) + 1 if row[2] is not None else bounds[1]
            end = terminal_end if row[0] >= count else max(bounds[1], terminal_end)
            end_boundary, confirmed_ns, end_reason = _concurrency_end(occupancy, start, end, target)
            reason = end_reason if end_boundary is None else None
            if end_boundary is not None:
                end_reason = end_reason or ("first_pass_completed" if row[0] >= count else "trial_ended")
                end_turns = backend._one(
                    "SELECT count(*) FROM turns t JOIN sessions s "
                    "ON s.trial_id=t.trial_id AND s.session_id=t.session_id "
                    "WHERE t.trial_id=? AND s.ordinal<? "
                    "AND coalesce(t.response_completed_ns,t.turn_completed_ns)<=?",
                    [trial, fact.scheduling.dataset_session_count, end_boundary],
                )[0]
    minutes = (
        max(0, (end_boundary - start) // MINUTE_NS)
        if start is not None and end_boundary is not None
        else 0
    )
    if end_boundary is not None and not minutes:
        reason = "no_complete_minute_in_maxtpm_window"
    points, windows, cohorts = [], [], []
    missing = [0,0,0]

    def point(window, name, value, unavailable=None):
        return replace(
            backend._point(
                trial,
                window,
                name,
                "value",
                value,
                "available" if value is not None else "unavailable",
                unavailable if value is None else None,
                "configured_price+server_usage+completion_clock"
                if name in REVENUE_NAMES
                else "server_usage+completion_clock",
            ),
            unit=pricing["currency"] + "/min"
            if name in REVENUE_NAMES
            else METRIC_CATALOG[name].unit,
            formula_version=FORMULA_VERSION,
        )

    if minutes:
        end = start + minutes * MINUTE_NS
        rows = backend.db.execute(
            "SELECT (response_completed_ns-?) // ? AS bucket, count(*), "
            "count(*) FILTER (WHERE prompt IS NULL OR prompt<0), "
            "count(*) FILTER (WHERE completion IS NULL OR completion<0), "
            "count(*) FILTER (WHERE cached IS NULL OR cached<0 OR cached>prompt), "
            "sum(prompt::HUGEINT), sum(completion::HUGEINT), sum(cached::HUGEINT) "
            "FROM turns t JOIN sessions s ON s.trial_id=t.trial_id AND s.session_id=t.session_id WHERE t.trial_id=? AND s.ordinal<? AND t.status='succeeded' "
            "AND response_completed_ns>=? AND response_completed_ns<? GROUP BY bucket",
            [start, MINUTE_NS, trial, fact.scheduling.dataset_session_count, start, end],
        ).fetchall()
        buckets = {int(row[0]): row for row in rows}
        missing = [sum(int(row[i]) for row in rows) for i in (2,3,4)]
        for index in range(minutes):
            bucket = buckets.get(index, (index, 0, 0, 0, 0, 0, 0, 0))
            values = minute_values(bucket, pricing)
            cohorts.append(values)
            name = f"maxtpm:minute:{index + 1:06d}"
            for metric, value in values.items():
                unavailable = (
                    "no_prompt_tokens"
                    if metric == "maxtpm_cache_hit_rate" and values["maxtpm_input_tpm"] == 0
                    else usage_reason(metric,bucket[2:5])
                )
                points.append(point(name, metric, value, unavailable))
            windows.append(
                dict(
                    window=name,
                    start_ns=start + index * MINUTE_NS,
                    end_ns=start + (index + 1) * MINUTE_NS,
                    duration_s=60.0,
                    availability="available",
                    unavailable_reason=None,
                )
            )
    for metric in names:
        values = [cohort[metric] for cohort in cohorts]
        value = sum(values) / minutes if minutes and all(v is not None for v in values) else None
        unavailable = reason or usage_reason(metric,missing)
        if metric == "maxtpm_cache_hit_rate" and minutes:
            prompts = [c["maxtpm_input_tpm"] for c in cohorts]
            caches = [c["maxtpm_cached_tpm"] for c in cohorts]
            if all(v is not None for v in (*prompts, *caches)):
                value = sum(caches) / sum(prompts) if sum(prompts) else None
                unavailable = "no_prompt_tokens" if not sum(prompts) else None
        # Preserve the existing availability reason for the TPM metric.
        if metric == "maxtpm" and reason is None and value is None:
            unavailable = "usage_not_reported"
        points.append(point("maxtpm", metric, value, unavailable))
    points.append(point("maxtpm", "maxtpm_minutes", minutes))
    windows.append(
        dict(
            window="maxtpm",
            start_ns=start,
            end_ns=start + minutes * MINUTE_NS if start is not None else None,
            duration_s=minutes * 60.0 if start is not None else None,
            availability="available" if minutes else "unavailable",
            unavailable_reason=None if minutes else reason,
        )
    )
    points.extend(latency_points(backend,trial,start,minutes,fact.scheduling.dataset_session_count,reason))
    diagnostics = {
        "maxtpm_latency_cohort": "same_first_pass_successful_completion_cohort; full_request_latencies; pooled_samples",
        "maxtpm_missing_prompt_requests": missing[0],
        "maxtpm_missing_output_requests": missing[1],
        "maxtpm_missing_cached_requests": missing[2],
        "maxtpm_dataset_sessions": fact.scheduling.dataset_session_count,
        "maxtpm_dataset_turns": count,
        "maxtpm_progress_basis": "first_pass_terminal_turns",
        "maxtpm_start_percent": percent,
        "maxtpm_progress_start_fraction": percent / 100,
        "maxtpm_end_basis": "first_continuous_60s_below_target_concurrency",
        "maxtpm_target_concurrency": target,
        "maxtpm_underfill_grace_s": 60,
        "maxtpm_underfill_confirmed_ns": confirmed_ns,
        "maxtpm_end_reason": end_reason,
        "maxtpm_start_turns": lower,
        "maxtpm_end_turns": end_turns,
        "maxtpm_start_ns": start,
        "maxtpm_end_ns": end_boundary,
        "maxtpm_complete_minutes": minutes,
        "maxtpm_token_basis": "prompt_tokens_including_cache+completion_tokens",
        "revenue_basis": "same_maxtpm_successful_completion_cohort; average_of_all_complete_minutes",
        "maxtpm_excluded_tail_ns": end_boundary - start - minutes * MINUTE_NS
        if start is not None and end_boundary is not None
        else None,
    }
    return points, windows, diagnostics
