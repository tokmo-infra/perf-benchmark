"""Public observation values shared with the domain model."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from perf_benchmark.core.observation import (
    ClockAnchor as ClockAnchor,
    ObservationSettings as ObservationSettings,
    TraceContext as TraceContext,
)


@dataclass(frozen=True, slots=True)
class WriterSnapshot:
    queued: int
    processing: int
    processed: int
    batch_seconds: float


DiagnosticSource = Callable[[], WriterSnapshot | Mapping[str, int | str]]


@dataclass(frozen=True, slots=True)
class SampledOutputTrace:
    run_id: str
    turn_id: str
    attempt: int
    trace: TraceContext
    timestamps_ns: tuple[int, ...]
    output_event_count: int
    truncated: bool
