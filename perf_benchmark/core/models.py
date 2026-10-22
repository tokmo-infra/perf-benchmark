from __future__ import annotations

from perf_benchmark.core.profiles import RequestProfile
from perf_benchmark.core.causal import CausalNode

from collections.abc import Mapping
from types import MappingProxyType
from dataclasses import dataclass, field, fields, is_dataclass, replace
from datetime import datetime, timezone
from time import perf_counter_ns
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, overload
from uuid import uuid4

if TYPE_CHECKING:
    from perf_benchmark.pipeline.preparation.payloads import PayloadRef
    from perf_benchmark.pipeline.contracts.observation import TraceContext
    from perf_benchmark.pipeline.contracts.results import ScalarMetricRecord


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12]}"


def freeze_snapshot(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): freeze_snapshot(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_snapshot(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(freeze_snapshot(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        params = getattr(type(value), "__dataclass_params__", None)
        if params is None or not params.frozen:
            raise TypeError("parsed dataclass settings must be frozen")
        updates = {}
        for item in fields(value):
            current = getattr(value, item.name)
            frozen = freeze_snapshot(current)
            if item.init:
                updates[item.name] = frozen
            elif frozen is not current:
                raise TypeError(
                    f"parsed dataclass setting {item.name!r} must already be immutable"
                )
        return replace(value, **updates)
    return value


@overload
def thaw_snapshot(value: Mapping) -> dict[str, Any]: ...
@overload
def thaw_snapshot(value: list | tuple) -> list[Any]: ...
@overload
def thaw_snapshot(value: object) -> object: ...
def thaw_snapshot(value):
    """Make a mutable copy of a frozen configuration or fact mapping."""
    if isinstance(value, Mapping):
        return {str(key): thaw_snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw_snapshot(item) for item in value]
    return value


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


JsonValue: TypeAlias = bool | int | float | str | None | list["JsonValue"] | dict[str, "JsonValue"]
Metadata: TypeAlias = dict[str, JsonValue]
InputKind: TypeAlias = str
ContextMode: TypeAlias = Literal["snapshot", "recorded", "live"]
ResponseMode: TypeAlias = Literal["stream", "non_stream"]


def require_metadata(value: object, *, event: str) -> Metadata:
    if not isinstance(value, dict):
        raise TypeError(f"{event} metadata must be an object")
    metadata: Metadata = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{event} metadata keys must be strings")
        metadata[key] = _require_json_value(item, event=event)
    return metadata


def _require_json_value(value: object, *, event: str) -> JsonValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_require_json_value(item, event=event) for item in value]
    if isinstance(value, dict):
        return require_metadata(value, event=event)
    raise TypeError(f"{event} metadata values must be JSON-compatible")


@dataclass(slots=True)
class LifecycleMeasurements:
    started_ns: int | None = None
    completed_ns: int | None = None


@dataclass(slots=True)
class StageMeasurements(LifecycleMeasurements):
    plugin: str = ""
    status: str = "started"
    started_at: str | None = None
    completed_at: str | None = None
    duration_s: float | None = None


@dataclass(slots=True)
class RequestMeasurements:
    attempts: int = 0
    response_mode: ResponseMode | None = None
    target_ns: int | None = None
    offered_ns: int | None = None
    admitted_ns: int | None = None
    started_ns: int | None = None
    first_output_ns: int | None = None
    last_output_ns: int | None = None
    response_completed_ns: int | None = None
    http_status: int | None = None
    output_event_count: int = 0
    output_chars: int = 0
    finish_reason: str | None = None
    output_kind: str = "text"
    result_ready_ns: int | None = None
    trace_context: TraceContext | None = None
    profile: RequestProfile | None = None


@dataclass(slots=True)
class TurnMeasurements:
    lifecycle: LifecycleMeasurements = field(default_factory=LifecycleMeasurements)
    request: RequestMeasurements = field(default_factory=RequestMeasurements)
    total_started_ns: int | None = None
    eligibility_started_ns: int | None = None
    eligibility_finished_ns: int | None = None
    eligibility_target_ns: int | None = None
    dependency_ready_ns: int | None = None


@dataclass(slots=True)
class SessionMeasurements:
    lifecycle: LifecycleMeasurements = field(default_factory=LifecycleMeasurements)
    target_ns: int | None = None
    offered_ns: int | None = None
    admitted_ns: int | None = None


@dataclass(frozen=True, slots=True)
class MetricWindowMeasurement:
    window: str
    start_ns: int | None
    end_ns: int | None
    duration_s: float | None
    availability: str
    unavailable_reason: str | None


@dataclass(slots=True)
class TrialMeasurements:
    lifecycle: LifecycleMeasurements = field(default_factory=LifecycleMeasurements)
    metric_windows: tuple[MetricWindowMeasurement, ...] = ()
    metric_counts: dict[str, int] = field(default_factory=dict)
    diagnostics: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class RunMeasurements:
    preparation: LifecycleMeasurements = field(default_factory=LifecycleMeasurements)
    lifecycle: LifecycleMeasurements = field(default_factory=LifecycleMeasurements)
    stages: dict[str, StageMeasurements] = field(default_factory=dict)



def _start(entity: Turn | Session | Trial | Run) -> None:
    entity.status = "running"
    entity.started_at = now()
    entity.measurements.lifecycle.started_ns = perf_counter_ns()


@dataclass(slots=True)
class Turn:
    request_payload_ref: PayloadRef
    source_created_at: datetime | None = None
    causal: CausalNode | None = None
    input_kind: InputKind = "chat"
    context_mode: ContextMode = "snapshot"
    recorded_response_payload_ref: PayloadRef | None = None
    id: str = field(kw_only=True)
    session_id: str | None = None
    ordinal: int = 0
    estimated_input_tokens: int | None = None
    reference_output_tokens: int | None = None
    requested_output_tokens: int | None = None
    usage: dict[str, int] = field(default_factory=dict)
    scenario_passed: bool | None = None
    status: str = "pending"
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    measurements: TurnMeasurements = field(default_factory=TurnMeasurements)
    metrics: dict[str, ScalarMetricRecord] = field(default_factory=dict)
    metadata: Metadata = field(default_factory=dict)
    profile: RequestProfile | None = None
    recorded_profile: RequestProfile | None = None
    extra_headers_payload_ref: PayloadRef | None = field(default=None, repr=False)

    start = _start

    def succeed(self) -> None:
        self._finish("succeeded", self.error)

    def fail(self, error: str) -> None:
        self._finish("failed", error)

    def cancel(self, reason: str) -> None:
        self._finish("cancelled", reason)

    def _finish(self, status: str, error: str | None) -> None:
        self.error, self.status = error, status
        self.finished_at = now()
        self.measurements.lifecycle.completed_ns = perf_counter_ns()


@dataclass(slots=True)
class Session:
    turns: list[Turn]
    id: str = field(kw_only=True)
    trial_id: str | None = None
    ordinal: int = 0
    status: str = "pending"
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    measurements: SessionMeasurements = field(default_factory=SessionMeasurements)
    metrics: dict[str, ScalarMetricRecord] = field(default_factory=dict)
    metadata: Metadata = field(default_factory=dict)

    def link(self) -> None:
        for turn in self.turns:
            turn.session_id = self.id

    start = _start

    def finish(self) -> None:
        terminals = {}
        for turn in self.turns:
            if ((turn.status == "failed" and turn.scenario_passed is not True)
                    or turn.scenario_passed is False):
                terminals.setdefault("failed", turn)
            elif turn.status == "cancelled" and turn.scenario_passed is not True:
                terminals.setdefault("cancelled", turn)
        self.status = next((status for status in ("failed", "cancelled") if status in terminals), "succeeded")
        terminal = terminals.get(self.status)
        self.error = (terminal.error or "scenario expectation not met") if terminal else None
        self.finished_at = now()
        self.measurements.lifecycle.completed_ns = perf_counter_ns()


@dataclass(slots=True)
class Trial:
    id: str = field(kw_only=True)
    name: str = "trial"
    run_id: str | None = None
    ordinal: int = 0
    status: str = "pending"
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    session_count: int = 0
    turn_count: int = 0
    succeeded_turns: int = 0
    failed_turns: int = 0
    cancelled_turns: int = 0
    failed_sessions: int = 0
    cancelled_sessions: int = 0
    measurements: TrialMeasurements = field(default_factory=TrialMeasurements)
    metrics: dict[str, ScalarMetricRecord] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: Metadata = field(default_factory=dict)

    start = _start

    def observe(self, session: Session) -> None:
        self.session_count += 1
        self.turn_count += sum(turn.status != "pending" for turn in session.turns)
        self.succeeded_turns += sum(turn.status == "succeeded" for turn in session.turns)
        self.failed_turns += sum(turn.status == "failed" for turn in session.turns)
        self.cancelled_turns += sum(turn.status == "cancelled" for turn in session.turns)
        self.failed_sessions += session.status == "failed"
        self.cancelled_sessions += session.status == "cancelled"
        if session.status == "cancelled" and not self.failed_sessions:
            self.error = session.error or "session cancelled"
        elif session.status == "failed" and self.failed_sessions == 1:
            self.error = session.error or "session failed"

    def finish(
        self,
        *,
        cancellation_reason: str | None = None,
        failure_reason: str | None = None,
    ) -> None:
        if cancellation_reason is not None:
            self.status = "cancelled"
            self.error = cancellation_reason
        elif failure_reason is not None:
            self.status = "failed"
            self.error = failure_reason
        else:
            self.status = next((status for status, count in (
                ("failed", self.failed_sessions), ("cancelled", self.cancelled_sessions),
            ) if count), "succeeded")
            if self.status != "succeeded" and self.error is None:
                self.error = f"one or more turns {self.status}"
        self.finished_at = now()
        self.measurements.lifecycle.completed_ns = perf_counter_ns()


@dataclass(slots=True)
class Run:
    name: str
    id: str = field(default_factory=lambda: new_id("run"))
    status: str = "pending"
    error: str | None = None
    created_at: str = field(default_factory=now)
    preparing_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    trial_ids: list[str] = field(default_factory=list)
    trial_outcomes: list[tuple[str, str | None]] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    event_counts: dict[str, int] = field(default_factory=dict)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    handler_failure_count: int = 0
    dropped_hook_event_count: int = 0
    measurements: RunMeasurements = field(default_factory=RunMeasurements)
    metrics: dict[str, ScalarMetricRecord] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    metadata: Metadata = field(default_factory=dict)

    def prepare(self) -> None:
        if self.status != "pending":
            raise RuntimeError("Run can only prepare from pending")
        self.status = "preparing"
        self.preparing_at = now()
        self.measurements.preparation.started_ns = perf_counter_ns()

    def start(self) -> None:
        if self.status != "preparing":
            raise RuntimeError("Run can only start from preparing")
        self.measurements.preparation.completed_ns = perf_counter_ns()
        _start(self)

    def register_trial(self, trial: Trial) -> None:
        trial.run_id = self.id
        self.trial_ids.append(trial.id)

    def observe_trial(self, trial: Trial) -> None:
        self.trial_outcomes.append((trial.status, trial.error))
        self.counts["trials"] = len(self.trial_outcomes)
        for key, value in (
            ("sessions", trial.session_count), ("turns", trial.turn_count),
            ("succeeded_turns", trial.succeeded_turns), ("failed_turns", trial.failed_turns),
            ("cancelled_turns", trial.cancelled_turns),
        ):
            self.counts[key] = self.counts.get(key, 0) + value

    def finish(self, cancellation_reason: str | None = None) -> None:
        timing = self.measurements.preparation
        if timing.started_ns is not None and timing.completed_ns is None:
            timing.completed_ns = perf_counter_ns()
        if cancellation_reason is not None:
            self.status = "cancelled"
            self.error = cancellation_reason
        else:
            first_errors = dict(reversed(self.trial_outcomes))
            failed, cancelled = first_errors.get("failed"), first_errors.get("cancelled")
            self.status = "failed" if failed else "cancelled" if cancelled else "succeeded"
            self.error = failed or cancelled
        self.finished_at = now()
        self.measurements.lifecycle.completed_ns = perf_counter_ns()

    def fail(self, error: str) -> None:
        self.finish()
        self.status = "failed"
        self.error = error

    def record_event(self, event: dict[str, Any]) -> None:
        name = str(event["name"])
        self.event_counts[name] = self.event_counts.get(name, 0) + 1
        if name.endswith((".failed", ".cancelled")) or name == "handler.failed":
            self.diagnostics.append(event)
            del self.diagnostics[:-100]
