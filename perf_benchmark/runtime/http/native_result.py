"""Internal typed value returned once by the native terminal batch."""
from typing import NamedTuple
from collections.abc import Sequence


class NativeResult(NamedTuple):
    status: str
    error: str | None
    error_origin: str | None
    retryable: bool
    started_ns: int | None
    completed_ns: int | None
    http_status: int | None
    samples: Sequence[tuple[int, int, int]]
    usage: dict[str, int]
    output_chars: int
    finish_reason: str | None
    continuation: bytes | None
    continuation_valid: bool
    output_kind: str
    trigger_ns: int | None
    decision_ns: int | None
    applied_ns: int | None
