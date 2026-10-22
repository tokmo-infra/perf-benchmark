from __future__ import annotations

from perf_benchmark.core.profiles import RequestProfile

from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence
from typing import Any
from typing import TypeAlias
from perf_benchmark.core.observation import TraceContext
from perf_benchmark.core.models import ResponseMode


# (event index, raw receive timestamp in ns, output characters).
# Exact tuples of integers can be untracked by GC; tuple subclasses cannot.
OutputEventSample: TypeAlias = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class ResponseTrace:
    turn_id: str
    samples: Sequence[OutputEventSample]
    attempt: int = 1


@dataclass(frozen=True, slots=True)
class TurnFact:
    trial_id: str
    id: str
    session_id: str | None
    ordinal: int
    status: str
    started_at: str | None
    finished_at: str | None
    error: str | None
    estimated_input_tokens: int | None
    reference_output_tokens: int | None
    requested_output_tokens: int | None
    usage: Mapping[str, int]
    metadata: Mapping[str, Any]
    response_samples: Sequence[OutputEventSample]
    target_ns: int | None
    offered_ns: int | None
    admitted_ns: int | None
    turn_started_ns: int | None
    turn_completed_ns: int | None
    attempts: int
    response_mode: ResponseMode | None
    request_ns: int | None
    first_ns: int | None
    last_ns: int | None
    response_completed_ns: int | None
    http_status: int | None
    output_event_count: int
    output_chars: int
    finish_reason: str | None
    scenario_passed: bool | None = None
    total_started_ns: int | None = None
    eligibility_started_ns: int | None = None
    eligibility_finished_ns: int | None = None
    eligibility_target_ns: int | None = None
    dependency_ready_ns: int | None = None
    output_kind: str = "text"
    result_ready_ns: int | None = None
    trace_context: TraceContext | None = None
    profile: RequestProfile | None = None



@dataclass(frozen=True, slots=True)
class AttemptFact(TurnFact):
    error_origin: str | None = None
    trigger_ns: int | None = None
    decision_ns: int | None = None
    applied_ns: int | None = None
    plan: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SessionFact:
    id: str
    trial_id: str | None
    ordinal: int
    status: str
    started_at: str | None
    finished_at: str | None
    error: str | None
    metadata: Mapping[str, Any]
    turn_count: int
    target_ns: int | None
    offered_ns: int | None
    admitted_ns: int | None
    started_ns: int
    completed_ns: int
    succeeded_turns: int


@dataclass(frozen=True, slots=True)
class OccupancyFact:
    trial_id: str
    kind: str
    at_ns: int
    value: int


MeasurementFact = TurnFact | AttemptFact | SessionFact | OccupancyFact
