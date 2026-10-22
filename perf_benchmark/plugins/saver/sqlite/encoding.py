"""Canonical, validated JSON for SQLite extension fields."""
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from functools import cache
from types import MappingProxyType
import math
import orjson
from perf_benchmark.pipeline.contracts.results import (
    ScalarMetricRecord, SessionResultRecord, TurnResultRecord, MetricWindowResultRecord,
    FORBIDDEN_FACT_FIELDS,
)

def _json_text(value: object) -> str:
    _validate_value(value)
    return _json_dump(value)


def _json_dump(value):
    # Adapt only the container currently visited by orjson; do not materialize
    # another recursive dict/list tree before encoding immutable records.
    return orjson.dumps(value, default=_json_container,
                        option=orjson.OPT_SORT_KEYS | orjson.OPT_PASSTHROUGH_DATACLASS).decode()


def _json_container(value):
    if isinstance(value, tuple):
        return list(value)
    if type(value) is MappingProxyType:
        return value.copy()
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return {name: getattr(value, name) for name in _field_names(type(value))}
    raise TypeError(f"fact contains unsupported {type(value).__name__}")


@cache
def _field_names(kind):
    # SQL column order is explicit, independent of dataclass inheritance order.
    if kind is MetricWindowResultRecord:
        return ("run_id", "scope", "scope_id", "window", "start_ns", "end_ns",
                "duration_s", "availability", "unavailable_reason")
    if kind is SessionResultRecord:
        return ("id", "trial_id", "ordinal", "status", "started_at", "finished_at", "error",
                "metadata", "measurements", "metrics", "turn_count")
    if kind is TurnResultRecord:
        return ("id", "session_id", "ordinal", "status", "started_at", "finished_at", "error",
                "estimated_input_tokens", "reference_output_tokens", "requested_output_tokens",
                "usage", "measurements", "metrics", "metadata", "scenario_passed")
    return tuple(item.name for item in fields(kind))


def _validate_value(value):
    """Validate facts without allocating a second JSON container tree."""
    kind = type(value)
    if kind is ScalarMetricRecord:
        return True
    if value is None or kind is bool or kind is int:
        return
    if isinstance(value, str):
        if len(value) > 4096:
            raise ValueError("fact string exceeds 4096 characters")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("fact contains a non-finite number")
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_value(item)
        return True
    if kind is dict or kind is MappingProxyType or isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("fact keys must be strings")
            if key.lower() in FORBIDDEN_FACT_FIELDS:
                raise ValueError(f"fact contains body field or credential: {key.lower()}")
            _validate_value(item)
        return True
    if is_dataclass(value):
        for name in _field_names(type(value)):
            _validate_value(getattr(value, name))
        return True
    if isinstance(value, int):
        return
    raise TypeError(f"fact contains unsupported {type(value).__name__}")
