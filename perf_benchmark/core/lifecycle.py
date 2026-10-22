from __future__ import annotations

from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any, Mapping

RUN_STARTED = "run.started"
RUN_PREPARING = "run.preparing"
RUN_BEFORE_START = "run.before_start"
RUN_COMPLETED = "run.completed"
RUN_FAILED = "run.failed"
RUN_CANCELLED = "run.cancelled"
EXECUTOR_BOUND = "executor.bound"
RUNTIME_CREATED = "runtime.created"
RUNTIME_PREPARE_STARTED = "runtime.prepare.started"
RUNTIME_PREPARE_COMPLETED = "runtime.prepare.completed"
RUNTIME_PREPARE_FAILED = "runtime.prepare.failed"
RUNTIME_CLOSE_STARTED = "runtime.close.started"
RUNTIME_CLOSE_COMPLETED = "runtime.close.completed"
RUNTIME_CLOSE_FAILED = "runtime.close.failed"
LOADER_BATCH_LOADED = "loader.batch.loaded"
LOADER_RECORD_INVALID = "loader.record.invalid"
NORMALIZATION_BATCH_COMPLETED = "normalization.batch.completed"
NORMALIZATION_RECORD_FAILED = "normalization.record.failed"
SELECTOR_BATCH_COMPLETED = "selector.batch.completed"
SELECTOR_COMPLETED = "selector.completed"
SCHEDULER_QUEUE_CHANGED = "scheduler.queue.changed"
SCHEDULER_CONCURRENCY_CHANGED = "scheduler.concurrency.changed"
SCHEDULER_WORKER_STARTED = "scheduler.worker.started"
SCHEDULER_WORKER_STOPPED = "scheduler.worker.stopped"
TRIAL_STARTED = "trial.started"
TRIAL_BEFORE_START = "trial.before_start"
TRIAL_COMPLETED = "trial.completed"
SESSION_STARTED = "session.started"
SESSION_COMPLETED = "session.completed"
TURN_STARTED = "turn.started"
TURN_COMPLETED = "turn.completed"
TURN_FAILED = "turn.failed"
TURN_CANCELLED = "turn.cancelled"
TURN_ATTEMPT_STARTED = "turn.attempt.started"
TURN_ATTEMPT_COMPLETED = "turn.attempt.completed"
TURN_ATTEMPT_CANCELLED = "turn.attempt.cancelled"
TURN_ATTEMPT_FAILED = "turn.attempt.failed"
REQUEST_QUEUED = "request.queued"
REQUEST_STARTED = "request.started"
REQUEST_FIRST_OUTPUT = "request.first_output"
REQUEST_COMPLETED = "request.completed"
REQUEST_FAILED = "request.failed"
REQUEST_CANCELLED = "request.cancelled"
USAGE_RECEIVED = "usage.received"
METRICS_CALCULATED = "metrics.calculated"
METRICS_UNAVAILABLE = "metrics.unavailable"
SAVER_TRANSACTION_STARTED = "saver.transaction.started"
SAVER_TRANSACTION_COMMITTED = "saver.transaction.committed"
SAVER_TRANSACTION_ROLLED_BACK = "saver.transaction.rolled_back"
SAVER_ARTIFACT_CREATED = "saver.artifact.created"
EXPORTER_OUTPUT_COMPLETED = "exporter.output.completed"
HANDLER_FAILED = "handler.failed"

EVENT_NAMES = frozenset(
    value for name, value in tuple(globals().items()) if name.isupper() and isinstance(value, str)
)

INTERCEPTOR_FIELDS = MappingProxyType(
    {
        RUN_BEFORE_START: frozenset({"name", "metadata"}),
        TRIAL_BEFORE_START: frozenset({"name", "metadata", "request_parameters"}),
    }
)

INTERCEPTOR_EVENTS = frozenset(INTERCEPTOR_FIELDS)


@dataclass(frozen=True, slots=True)
class LifecycleEvent:
    name: str
    at: str
    at_ns: int
    run_id: str
    stage: str | None = None
    trial_id: str | None = None
    session_id: str | None = None
    turn_id: str | None = None
    attempt: int | None = None
    data: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def to_dict(self) -> dict[str, Any]:
        return {
            item.name: dict(self.data) if item.name == "data" else getattr(self, item.name)
            for item in fields(self)
        }
