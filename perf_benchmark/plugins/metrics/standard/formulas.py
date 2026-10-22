from __future__ import annotations

import math

from perf_benchmark.plugins.metrics.standard.catalog import METRIC_CATALOG, public_value

from perf_benchmark.pipeline.contracts.results import ScalarMetricRecord
from perf_benchmark.plugins.metrics.standard.catalog import MetricDefinition


def unavailable(definition: MetricDefinition, reason: str) -> ScalarMetricRecord:
    return ScalarMetricRecord(
        unit=definition.unit,
        availability="unavailable",
        unavailable_reason=reason,
        source=None,
        statistics={name: None for name in definition.statistics},
    )


def scalar(
    definition: MetricDefinition,
    value: float | int | None,
    *,
    source: str | None = None,
    reason: str = "not_observed",
) -> ScalarMetricRecord:
    if value is None or not math.isfinite(float(value)):
        return unavailable(definition, reason)
    return ScalarMetricRecord(
        unit=definition.unit,
        availability="available",
        unavailable_reason=None,
        source=source,
        statistics={"value": public_value(definition, "value", value)},
    )


def integer(value):
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def request_values(fact):
    stream = fact.response_mode == "stream"
    text = fact.output_kind == "text"
    start, first, last, end = (
        fact.request_ns,
        fact.first_ns,
        fact.last_ns,
        fact.response_completed_ns,
    )
    prompt, completion = (
        integer(fact.usage.get("prompt_tokens")),
        integer(fact.usage.get("completion_tokens")),
    )
    ttft = (
        (first - start) / 1e6
        if stream and text and start is not None and first is not None
        else None
    )
    e2e = (end - start) / 1e6 if start is not None and end is not None else None
    decode_ns = (
        last - first
        if stream
        and text
        and fact.output_event_count >= 2
        and first is not None
        and last is not None
        and last > first
        else None
    )
    decode_tokens = (
        completion - 1 if stream and text and completion is not None and completion > 1 else None
    )
    events = fact.output_event_count
    event_yield = ((completion - 1) / (events - 1)
                   if stream and text and completion is not None and completion >= events > 1 else None)
    return {
        "ttft": ttft,
        "tpot": decode_ns / 1e6 / decode_tokens
        if decode_ns is not None and decode_tokens is not None
        else None,
        "e2e": e2e,
        "time_to_last_output": (
            (last - start) / 1e6
            if stream and start is not None and last is not None
            else e2e
            if fact.response_mode == "non_stream"
            else None
        ),
        "response_drain": (end - last) / 1e6
        if stream and last is not None and end is not None and end >= last
        else None,
        "prefill_throughput": prompt / (ttft / 1000.0)
        if stream and prompt is not None and ttft is not None and ttft > 0
        else None,
        "decode_throughput": decode_tokens / (decode_ns / 1e9)
        if decode_tokens is not None and decode_ns is not None and decode_ns > 0
        else None,
        "turn_total_duration": (fact.turn_completed_ns - fact.total_started_ns) / 1e6
        if fact.turn_completed_ns is not None and fact.total_started_ns is not None
        else None,
        "decode_ns": decode_ns,
        "decode_tokens": decode_tokens,
        "output_tokens_per_event_estimate": event_yield,
        "speculative_acceptance_proxy": 1 - 1 / event_yield if event_yield is not None else None,
    }


def itl_values(fact):
    if fact.response_mode != "stream" or fact.output_kind != "text":
        return
    samples = sorted(fact.response_samples, key=lambda sample: sample[0])
    for previous, current in zip(samples, samples[1:]):
        yield (current[1] - previous[1]) / 1e6


def request_metrics(turn, derived):
    values = (
        ("input_tokens", integer(turn.usage.get("prompt_tokens")), "server_usage"),
        ("output_tokens", integer(turn.usage.get("completion_tokens")), "server_usage"),
        ("ttft", derived["ttft"], "protocol_output_event"),
        ("tpot", derived["tpot"], "protocol_output_event+server_usage"),
        ("e2e", derived["e2e"], "client_clock"),
        ("turn_total_duration", derived["turn_total_duration"], "client_clock"),
        (
            "time_to_last_output",
            derived["time_to_last_output"],
            "protocol_output_event" if turn.response_mode == "stream" else "response_body_complete",
        ),
        ("response_drain", derived["response_drain"], "client_clock"),
        ("prefill_throughput", derived["prefill_throughput"], "server_usage"),
        ("decode_throughput", derived["decode_throughput"], "protocol_output_event+server_usage"),
        ("output_tokens_per_event_estimate", derived["output_tokens_per_event_estimate"], "estimated_protocol_events"),
        ("speculative_acceptance_proxy", derived["speculative_acceptance_proxy"], "estimated_protocol_events"),
    )
    return {
        name: scalar(
            METRIC_CATALOG[name],
            value,
            source=source,
            reason=(
                "usage_not_reported"
                if name in {"input_tokens", "output_tokens"}
                else _turn_unavailable_reason(
                    name,
                    turn.response_mode,
                    turn.output_kind,
                    turn.output_event_count,
                    integer(turn.usage.get("completion_tokens")),
                    turn.first_ns,
                    turn.last_ns,
                )
            ),
        )
        for name, value, source in values
    }


def _turn_unavailable_reason(
    name: str,
    mode: str | None,
    output_kind: str,
    events: int,
    completion: int | None,
    first_ns: object,
    last_ns: object,
) -> str:
    if name in {"output_tokens_per_event_estimate", "speculative_acceptance_proxy"}:
        if mode != "stream":
            return "non_streaming_response" if mode == "non_stream" else "response_mode_not_observed"
        if output_kind != "text":
            return "non_text_output"
        if completion is None:
            return "usage_not_reported"
        if completion <= 1:
            return "insufficient_output_tokens"
        if events <= 1:
            return "insufficient_output_events"
        return "output_events_exceed_tokens"
    if (
        name
        in {
            "ttft",
            "tpot",
            "response_drain",
            "prefill_throughput",
            "decode_throughput",
        }
        and mode == "non_stream"
    ):
        return "non_streaming_response"
    if output_kind != "text" and name in {
        "ttft",
        "tpot",
        "prefill_throughput",
        "decode_throughput",
    }:
        return "non_text_output"
    if name in {"ttft", "response_drain", "prefill_throughput"}:
        return "no_output_events" if not events else "usage_not_reported"
    if name in {"tpot", "decode_throughput"}:
        if events < 2:
            return "insufficient_output_events"
        if completion is None:
            return "usage_not_reported"
        if completion <= 1:
            return "insufficient_output_tokens"
        if not isinstance(first_ns, int) or not isinstance(last_ns, int):
            return "decode_interval_not_observable"
        if last_ns <= first_ns:
            return "decode_interval_not_observable"
    return "not_observed"
