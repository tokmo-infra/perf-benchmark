from __future__ import annotations

from dataclasses import dataclass


BASE_STATISTICS = (
    "n",
    "eligible_n",
    "am",
    "std",
    "tm",
    "p1",
    "p5",
    "p10",
    "p50",
    "p90",
    "p95",
    "p99",
    "min",
    "max",
)
TPOT_STATISTICS = BASE_STATISTICS + ("twm",)
THROUGHPUT_STATISTICS = BASE_STATISTICS + ("hm", "twm", "time_wm")
SCALAR_STATISTICS = ("value",)


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    name: str
    unit: str
    statistics: tuple[str, ...]


def _definitions() -> tuple[MetricDefinition, ...]:
    scalar = SCALAR_STATISTICS
    base = BASE_STATISTICS
    throughput = THROUGHPUT_STATISTICS
    return (
        MetricDefinition("maxtpm", "tok/min", scalar),
        MetricDefinition("maxtpm_minutes", "minutes", scalar),
        *(MetricDefinition(name, "tok/min", scalar) for name in
          ("maxtpm_input_tpm", "maxtpm_output_tpm", "maxtpm_cached_tpm", "maxtpm_uncached_tpm")),
        MetricDefinition("maxtpm_cache_hit_rate", "ratio", scalar),
        *(MetricDefinition(name, "currency/min", scalar) for name in
          ("revenue_per_minute", "input_revenue_per_minute", "cached_input_revenue_per_minute",
           "output_revenue_per_minute")),
        *(MetricDefinition(name, "currency", scalar) for name in
          ("input_cost", "cached_input_cost", "output_cost", "total_cost", "observed_cost")),
        MetricDefinition("cost_observed_attempts", "attempts", scalar),
        MetricDefinition("cost_unavailable_attempts", "attempts", scalar),
        MetricDefinition("offered", "requests", scalar),
        MetricDefinition("admitted", "requests", scalar),
        MetricDefinition("succeeded", "requests", scalar),
        MetricDefinition("failed", "requests", scalar),
        MetricDefinition("cancelled", "requests", scalar),
        MetricDefinition("cancellation_rate", "ratio", scalar),
        MetricDefinition("policy_cancelled", "attempts", scalar),
        MetricDefinition("policy_cancellation_rate", "ratio", scalar),
        MetricDefinition("rejected", "requests", scalar),
        MetricDefinition("expired", "requests", scalar),
        MetricDefinition("success_rate", "ratio", scalar),
        MetricDefinition("goodput_qualified", "requests", scalar),
        MetricDefinition("workload_rps", "req/s", scalar),
        MetricDefinition("goodput_rate", "req/s", scalar),
        MetricDefinition("goodput_percentage", "ratio", scalar),
        MetricDefinition("total_prompt_tokens", "tokens", scalar),
        MetricDefinition("total_new_prompt_tokens", "tokens", scalar),
        MetricDefinition("total_cached_tokens", "tokens", scalar),
        MetricDefinition("total_completion_tokens", "tokens", scalar),
        MetricDefinition("mean_input_tokens", "tokens/request", scalar),
        MetricDefinition("mean_output_tokens", "tokens/request", scalar),
        MetricDefinition("workload_prompt", "tok/s", scalar),
        MetricDefinition("workload_new_prompt", "tok/s", scalar),
        MetricDefinition("workload_cached_prompt", "tok/s", scalar),
        MetricDefinition("workload_completion", "tok/s", scalar),
        MetricDefinition("workload_total", "tok/s", scalar),
        MetricDefinition("input_tokens", "tokens", base),
        MetricDefinition("output_tokens", "tokens", base),
        MetricDefinition("ttft", "s", base),
        MetricDefinition("tpot", "ms", TPOT_STATISTICS),
        MetricDefinition("itl", "ms", base),
        MetricDefinition("e2e", "s", base),
        MetricDefinition("turn_total_duration", "ms", base),
        MetricDefinition("time_to_last_output", "ms", base),
        MetricDefinition("response_drain", "ms", base),
        MetricDefinition("prefill_throughput", "tok/s", throughput),
        MetricDefinition("decode_throughput", "tok/s", throughput),
        MetricDefinition("output_tokens_per_event_estimate", "tok/event", base),
        MetricDefinition("speculative_acceptance_proxy", "ratio", base),
        MetricDefinition("session_duration", "ms", base),
        MetricDefinition("session_turns", "turns", base),
        MetricDefinition("session_first_turn_ttft", "s", base),
        MetricDefinition("session_subsequent_turn_ttft", "s", base),
        MetricDefinition("target_concurrency", "requests", scalar),
        MetricDefinition("target_active_sessions", "sessions", scalar),
        MetricDefinition("attempts", "attempts", scalar),
        MetricDefinition("retries", "attempts", scalar),
        MetricDefinition("failed_attempts", "attempts", scalar),
        MetricDefinition("cancelled_attempts", "attempts", scalar),
        MetricDefinition("token_cache_observable_requests", "requests", scalar),
        MetricDefinition("token_cache_hit_requests", "requests", scalar),
        MetricDefinition("token_cache_observation_rate", "ratio", scalar),
        MetricDefinition("token_cache_request_hit_rate", "ratio", scalar),
        MetricDefinition("token_cache_token_hit_rate", "ratio", scalar),
        MetricDefinition("successful_concurrency", "requests", scalar),
        MetricDefinition("actual_in_flight_concurrency", "requests", scalar),
        MetricDefinition("actual_active_sessions", "sessions", scalar),
        MetricDefinition("reserved_sessions", "sessions", scalar),
        MetricDefinition("sessions_started", "sessions", scalar),
        MetricDefinition("sessions_completed", "sessions", scalar),
        MetricDefinition("sessions_failed", "sessions", scalar),
        MetricDefinition("session_completed_turns_total", "turns", scalar),
        MetricDefinition("session_expected_turns_total", "turns", scalar),
        MetricDefinition("session_incomplete_count", "sessions", scalar),
        MetricDefinition("target_session_rate", "sessions/s", scalar),
        MetricDefinition("actual_session_rate", "sessions/s", scalar),
        MetricDefinition("target_request_rate", "req/s", scalar),
        MetricDefinition("actual_request_rate", "req/s", scalar),
        MetricDefinition("scheduler_lag", "ms", scalar),
        MetricDefinition("scheduler_lag_max", "ms", scalar),
        MetricDefinition("window_duration", "s", scalar),
        MetricDefinition("session_queue_delay", "ms", scalar),
        MetricDefinition("request_queue_delay", "ms", scalar),
        MetricDefinition("session_pending_depth", "sessions", scalar),
        MetricDefinition("request_pending_depth", "requests", scalar),
        MetricDefinition("drain_duration", "ms", scalar),
        MetricDefinition("partial_sessions", "sessions", scalar),
        MetricDefinition("demand_limited", "boolean", scalar),
        MetricDefinition("session_success_rate", "ratio", scalar),
    )


MAXTPM_METRICS = frozenset({"maxtpm", "maxtpm_minutes", "maxtpm_input_tpm", "maxtpm_output_tpm",
                           "maxtpm_cached_tpm", "maxtpm_uncached_tpm", "maxtpm_cache_hit_rate"})
REVENUE_METRICS = frozenset({"revenue_per_minute", "input_revenue_per_minute",
                            "cached_input_revenue_per_minute", "output_revenue_per_minute"})
OPTIONAL_METRICS = MAXTPM_METRICS | REVENUE_METRICS | frozenset({"input_cost", "cached_input_cost",
                              "output_cost", "total_cost", "observed_cost",
                              "cost_observed_attempts", "cost_unavailable_attempts"})

METRIC_CATALOG = {definition.name: definition for definition in _definitions()}

__all__ = ["METRIC_CATALOG", "MetricDefinition"]


def public_value(definition, statistic, value):
    """Convert internal millisecond samples once at the public metric boundary."""
    if value is not None and definition.unit == "s" and statistic not in {"n", "eligible_n"}:
        return value / 1000
    return value
