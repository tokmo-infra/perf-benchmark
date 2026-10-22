"""Executor Plugin interface and immutable Attempt inputs."""

from __future__ import annotations

from perf_benchmark.core.profiles import Continuation, RequestProfile

from perf_benchmark.core.validation import positive_int, positive_number

from abc import ABC, abstractmethod
from collections.abc import Mapping, Callable, Awaitable
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Protocol

from perf_benchmark.core.models import (
    ContextMode,
    InputKind,
    JsonValue,
    RequestMeasurements,
    Session,
    Turn,
    freeze_snapshot,
)
from perf_benchmark.pipeline.context import ExecutorContext
from perf_benchmark.pipeline.results.measurement import ResponseTrace
from perf_benchmark.pipeline.preparation.payloads import PayloadRef
from perf_benchmark.pipeline.contracts.observation import TraceContext


@dataclass(frozen=True, slots=True)
class AttemptPlan:
    after_outputs: int | None = None
    after_first_output_ms: float | None = None
    action: str = "cancel"
    expected_status: str | None = None
    expected_http_status: int | None = None

    def __post_init__(self) -> None:
        positive_int(self.after_outputs, "after_outputs", optional=True)
        positive_number(self.after_first_output_ms, "after_first_output_ms", optional=True, allow_zero=True)
        if self.after_outputs is not None and self.after_first_output_ms is not None:
            raise ValueError("choose one output trigger")
        if self.action not in {"cancel", "fail"}:
            raise ValueError("control action must be cancel or fail")
        if self.expected_status not in {None, "succeeded", "failed", "cancelled"}:
            raise ValueError("invalid expected status")
        if self.expected_http_status is not None and (
            not isinstance(self.expected_http_status, int)
            or not 100 <= self.expected_http_status <= 599
        ):
            raise ValueError("expected_http_status must be an HTTP status code")


class PayloadReader(Protocol):
    def view(self, ref: PayloadRef) -> memoryview: ...


@dataclass(frozen=True, slots=True)
class ExecutorTrial:
    id: str
    request_parameters: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )
    metadata: Mapping[str, JsonValue] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("ExecutorTrial.id must not be empty")
        object.__setattr__(self, "request_parameters", freeze_snapshot(self.request_parameters))
        object.__setattr__(self, "metadata", freeze_snapshot(self.metadata))


@dataclass(frozen=True, slots=True)
class ExecutorTurn:
    id: str
    ordinal: int
    input_kind: InputKind
    context_mode: ContextMode
    request_payload_ref: PayloadRef
    recorded_response_payload_ref: PayloadRef | None
    requested_output_tokens: int | None
    reference_output_tokens: int | None
    status: str
    metadata: Mapping[str, JsonValue]

    profile: RequestProfile | None = None
    recorded_profile: RequestProfile | None = None
    extra_headers_payload_ref: PayloadRef | None = field(default=None, repr=False)

    @classmethod
    def capture(cls, turn: Turn) -> "ExecutorTurn":
        return cls(**{name: getattr(turn, name) for name in _TURN_FIELDS},
                   metadata=freeze_snapshot(turn.metadata))


_TURN_FIELDS = tuple(f.name for f in fields(ExecutorTurn) if f.name != "metadata")



@dataclass(frozen=True, slots=True)
class ExecutorSession:
    id: str
    ordinal: int
    metadata: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class AttemptInput:
    session: ExecutorSession
    turn: ExecutorTurn
    previous_turns: tuple[ExecutorTurn, ...]
    live_history: Mapping[str, Continuation]
    attempt: int
    is_final_turn: bool
    payloads: PayloadReader = field(repr=False, compare=False)
    trace_context: TraceContext | None = None

    @classmethod
    def capture(
        cls,
        session: Session,
        turn: Turn,
        live_history: Mapping[str, Continuation],
        *,
        attempt: int,
        payloads: PayloadReader,
        trace_context: TraceContext | None = None,
    ) -> "AttemptInput":
        positive_int(attempt, "attempt")
        try:
            index = next(index for index, item in enumerate(session.turns) if item is turn)
        except StopIteration:
            raise ValueError(f"Turn is not part of Session: {turn.id}") from None
        history = dict(live_history)
        return cls(
            session=ExecutorSession(session.id, session.ordinal, freeze_snapshot(session.metadata)),
            turn=ExecutorTurn.capture(turn),
            previous_turns=tuple(ExecutorTurn.capture(item) for item in session.turns[:index]),
            live_history=MappingProxyType(history),
            attempt=attempt,
            is_final_turn=index == len(session.turns) - 1,
            payloads=payloads,
            trace_context=trace_context,
        )


@dataclass(frozen=True, slots=True)
class TurnRequestFacts:
    requested_output_tokens: int | None
    metadata: Mapping[str, JsonValue] = field(default_factory=lambda: MappingProxyType({}))

    profile: RequestProfile | None = None

    def __post_init__(self) -> None:
        positive_int(self.requested_output_tokens, "requested_output_tokens", optional=True)
        object.__setattr__(self, "metadata", freeze_snapshot(self.metadata))


@dataclass(slots=True)
class AttemptResult:
    turn_id: str
    attempt: int
    status: str = "succeeded"
    request: RequestMeasurements = field(default_factory=RequestMeasurements)
    usage: dict[str, int] = field(default_factory=dict)
    trace: ResponseTrace | None = None
    error: str | None = None
    error_origin: str | None = None
    fatal_error: BaseException | None = None
    retryable: bool = False
    recoverable: bool = True
    continuation_payload: bytes | None = None
    continuation_profile: RequestProfile | None = None
    continuation_valid: bool = True
    scenario_passed: bool | None = None
    trigger_ns: int | None = None
    decision_ns: int | None = None
    applied_ns: int | None = None
    request_facts: TurnRequestFacts | None = None
    plan: AttemptPlan | None = None


class AttemptControl(Protocol):
    plan: AttemptPlan
    action: str
    origin: str
    trigger_ns: int | None
    decision_ns: int | None
    applied_ns: int | None
    reason: str | None
    closed: bool
    started_callback: Callable[[int], None] | None

    def bind(self, callback) -> None: ...
    def observe(self, count: int, received_ns: int) -> None: ...
    def trigger(
        self, reason: str, at_ns: int | None = None, *, action: str | None = None
    ) -> None: ...
    def close(self) -> None: ...


class PreparedAttempt(Protocol):
    input: AttemptInput
    request_facts: TurnRequestFacts


class ExecutorRuntime(Protocol):
    async def prepare(self) -> None: ...
    async def prepare_attempt(self, value: AttemptInput) -> PreparedAttempt: ...
    async def execute_attempt(
        self, prepared: PreparedAttempt, signal: AttemptControl
    ) -> AttemptResult: ...
    async def release(self, prepared: PreparedAttempt) -> Awaitable[None] | None: ...
    async def close_session(self, session_id: str) -> None: ...
    async def close(self) -> None: ...


class ExecutorPlugin(ABC):
    supports_start_notifications = False

    def validate_observation(self) -> None:
        """Validate managed tracing before any requests are admitted."""

    @abstractmethod
    async def create_runtime(
        self, trial: ExecutorTrial, ctx: ExecutorContext
    ) -> ExecutorRuntime: ...
