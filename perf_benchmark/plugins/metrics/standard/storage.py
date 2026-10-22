"""DuckDB fact rows and temporary storage for standard metrics."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re

import duckdb
import pyarrow as pa
from perf_benchmark.plugins.metrics.standard.formulas import integer

_TURN_COLUMNS = (
    "trial_id",
    "turn_id",
    "session_id",
    "ordinal",
    "status",
    "offered_ns",
    "target_ns",
    "admitted_ns",
    "request_ns",
    "response_mode",
    "response_completed_ns",
    "turn_completed_ns",
    "first_ns",
    "last_ns",
    "output_event_count",
    "prompt",
    "completion",
    "cached",
    "attempts",
    "input_estimate",
    "ttft",
    "tpot",
    "e2e",
    "time_to_last_output",
    "response_drain",
    "prefill",
    "decode",
    "decode_tokens",
    "decode_ns",
    "turn_total_duration",
    "output_kind",
    "system_tokens",
    "ttft_reason",
    "tpot_reason",
    "e2e_reason",
    "event_yield",
    "acceptance_proxy",
    "event_yield_reason",
)
TURN_SCHEMA = pa.schema(
    [
        (
            name,
            pa.string()
            if name
            in {"trial_id", "turn_id", "session_id", "status", "response_mode", "output_kind",
                "ttft_reason", "tpot_reason", "e2e_reason", "event_yield_reason"}
            else pa.float64()
            if name
            in {
                "ttft",
                "tpot",
                "e2e",
                "time_to_last_output",
                "response_drain",
                "prefill",
                "decode",
                "turn_total_duration",
                "event_yield",
                "acceptance_proxy",
            }
            else pa.int64(),
        )
        for name in _TURN_COLUMNS
    ]
)
ITL_SCHEMA = pa.schema(
    [("trial_id", pa.string()), ("turn_id", pa.string()), ("value", pa.float64())]
)
ATTEMPT_SCHEMA = pa.schema(
    [
        ("trial_id", pa.string()),
        ("turn_id", pa.string()),
        ("attempt", pa.int64()),
        ("status", pa.string()),
        ("origin", pa.string()),
        ("target_ns", pa.int64()),
        ("offered_ns", pa.int64()),
        ("admitted_ns", pa.int64()),
        ("started_ns", pa.int64()),
    ]
)


def turn_row(fact, values, metrics):
    return (
        fact.trial_id,
        fact.id,
        fact.session_id,
        fact.ordinal,
        fact.status,
        fact.offered_ns,
        fact.target_ns,
        fact.admitted_ns,
        fact.request_ns,
        fact.response_mode,
        fact.response_completed_ns,
        fact.turn_completed_ns,
        fact.first_ns,
        fact.last_ns,
        fact.output_event_count,
        integer(fact.usage.get("prompt_tokens")),
        integer(fact.usage.get("completion_tokens")),
        integer(fact.usage.get("cached_tokens")),
        fact.attempts,
        fact.estimated_input_tokens,
        values["ttft"],
        values["tpot"],
        values["e2e"],
        values["time_to_last_output"],
        values["response_drain"],
        values["prefill_throughput"],
        values["decode_throughput"],
        values["decode_tokens"],
        values["decode_ns"],
        values["turn_total_duration"],
        fact.output_kind,
        fact.profile.system_tokens_estimate if fact.profile else None,
        *(metrics[name].unavailable_reason for name in ("ttft", "tpot", "e2e")),
        values["output_tokens_per_event_estimate"],
        values["speculative_acceptance_proxy"],
        metrics["output_tokens_per_event_estimate"].unavailable_reason,
    )


_SESSION_SCHEMA = pa.schema(
    [
        ("trial_id", pa.string()),
        ("session_id", pa.string()),
        ("ordinal", pa.int64()),
        ("status", pa.string()),
        ("target_ns", pa.int64()),
        ("offered_ns", pa.int64()),
        ("admitted_ns", pa.int64()),
        ("started_ns", pa.int64()),
        ("finished_ns", pa.int64()),
        ("duration", pa.float64()),
        ("turns", pa.int64()),
        ("completed_turns", pa.int64()),
    ]
)
_OCCUPANCY_SCHEMA = pa.schema(
    [
        ("trial_id", pa.string()),
        ("kind", pa.string()),
        ("at_ns", pa.int64()),
        ("value", pa.int64()),
    ]
)


def _duckdb_connection(settings: Mapping, temporary: str) -> duckdb.DuckDBPyConnection:
    memory = str(settings.get("memory_limit", "512MB")).upper()
    if not re.fullmatch(r"[1-9][0-9]*(KB|MB|GB)", memory):
        raise ValueError("metrics memory_limit must use KB, MB or GB")
    path = Path(str(settings.get("temp_directory", temporary))).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    # Aggregation state is disposable; only SQLite needs durable transactions.
    # DuckDB temporary tables spill to temp_directory under memory pressure.
    connection = duckdb.connect(
        config={
            "memory_limit": memory,
            "threads": 1,
            "preserve_insertion_order": True,
            "temp_directory": str(path),
        }
    )
    connection.execute("SET enable_progress_bar = false")
    return connection

def _arrow_table(rows, schema):
    columns = zip(*rows) if rows else (() for _ in schema)
    return pa.Table.from_arrays(
        [pa.array(column, type=field.type) for column, field in zip(columns, schema)],
        schema=schema,
    )
