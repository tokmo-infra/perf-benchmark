from collections.abc import Awaitable, Callable
from dataclasses import asdict, fields
from contextlib import AsyncExitStack
import logging
from types import MappingProxyType

from perf_benchmark.core.models import (
    RequestMeasurements,
    freeze_snapshot,
)
from perf_benchmark.pipeline.results.async_writer import AsyncBatchWriter
from perf_benchmark.pipeline.context import MetricsContext
from perf_benchmark.pipeline.contracts.observation import SampledOutputTrace
from perf_benchmark.pipeline.contracts.facts import TrialFact
from perf_benchmark.pipeline.contracts.metrics import MetricsBackend, MetricsPlugin, ResultPublisher
from perf_benchmark.pipeline.contracts.results import (
    AttemptResultRecord,
    ComputedResultBatch,
    SessionResultRecord,
    TurnResultRecord,
    MetricSeries,
    ScalarMetricRecord,
)
from perf_benchmark.pipeline.results.measurement import AttemptFact, OccupancyFact, SessionFact, TurnFact

logger = logging.getLogger("perf_benchmark.metrics")


def _key(fact):
    if isinstance(fact, OccupancyFact):
        return None
    if isinstance(fact, AttemptFact):
        return "attempt", fact.id, fact.attempts
    return ("session" if isinstance(fact, SessionFact) else "turn"), fact.id, 0


_RESULT_FIELDS = {
    cls: tuple(f.name for f in fields(cls) if f.name not in {"metrics", "measurements", "metadata", "usage"})
    for cls in (TurnResultRecord, SessionResultRecord)
}

_REQUEST_FIELDS = {
    item.name: {"started_ns": "request_ns", "first_output_ns": "first_ns",
                "last_output_ns": "last_ns"}.get(item.name, item.name)
    for item in fields(RequestMeasurements)
}


def _computed(fact: TurnFact | SessionFact, metrics):
    # The public scalar constructor owns validation/freezing. Preserve those
    # immutable values and isolate only the Backend's collection of metrics.
    if any(not isinstance(name, str) or type(value) is not ScalarMetricRecord
           for name, value in metrics.items()):
        raise TypeError("metric result must contain named ScalarMetricRecord values")
    metrics = MappingProxyType(dict(metrics))
    if isinstance(fact, SessionFact):
        return SessionResultRecord(
            **{name: getattr(fact, name) for name in _RESULT_FIELDS[SessionResultRecord]},
            measurements=MappingProxyType(dict(
                lifecycle=MappingProxyType(dict(started_ns=fact.started_ns, completed_ns=fact.completed_ns)),
                target_ns=fact.target_ns, offered_ns=fact.offered_ns, admitted_ns=fact.admitted_ns,
            )), metadata=fact.metadata, metrics=metrics,
        )
    request = {name: getattr(fact, source) for name, source in _REQUEST_FIELDS.items()}
    for name in ("profile", "trace_context"):
        value = request[name]
        if value is not None:
            request[name] = MappingProxyType({field.name: getattr(value, field.name) for field in fields(value)})
    request = MappingProxyType(request)
    if isinstance(fact, AttemptFact):
        assert fact.session_id is not None
        return AttemptResultRecord(
            fact.id, fact.session_id, fact.attempts, fact.status, fact.error, fact.error_origin,
            request, fact.usage, metrics, fact.scenario_passed, fact.trigger_ns,
            fact.decision_ns, fact.applied_ns, fact.plan,
        )
    return TurnResultRecord(
        **{name: getattr(fact, name) for name in _RESULT_FIELDS[TurnResultRecord]},
        metadata=fact.metadata, usage=fact.usage,
        measurements=MappingProxyType(dict(
            lifecycle=MappingProxyType(dict(started_ns=fact.turn_started_ns, completed_ns=fact.turn_completed_ns)),
            request=request, total_started_ns=fact.total_started_ns,
            eligibility_started_ns=fact.eligibility_started_ns, eligibility_finished_ns=fact.eligibility_finished_ns,
            eligibility_target_ns=fact.eligibility_target_ns, dependency_ready_ns=fact.dependency_ready_ns,
        )), metrics=metrics,
    )


class _MetricProcessor:
    def __init__(self, backend, ctx=None):
        self.backend, self.ctx = backend, ctx
        self.open = backend.open
        self.close = backend.close

    def write_batch(self, facts):
        keyed = [(key, fact) for fact in facts if (key := _key(fact)) is not None]
        expected = {key for key, _ in keyed}
        updates = {}
        series = []
        trials = {fact.trial_id for fact in facts}
        for update in self.backend.consume(tuple(facts)):
            if isinstance(update, MetricSeries):
                if update.trial_id not in trials or any(
                    p.scope != "trial" or p.scope_id != update.trial_id or p.window != update.window.window
                    for p in update.points
                ):
                    raise ValueError("metric series belongs to another Trial or window")
                series.append(update)
                continue
            key = update.scope, update.id, update.attempt
            if key not in expected or key in updates:
                raise ValueError(f"unknown or duplicate metric result: {key}")
            if not isinstance(update.metrics, dict):
                raise TypeError("metric result must contain a metrics dictionary")
            updates[key] = update.metrics
        if updates.keys() != expected:
            raise ValueError(f"missing metric results: {expected - updates.keys()}")
        turns, sessions, attempts = [], [], []
        groups = {"turn": turns, "session": sessions, "attempt": attempts}
        for key, fact in keyed:
            groups[key[0]].append(_computed(fact, updates[key]))
        if self.ctx is not None and self.ctx.sampled_output is not None:
            for _, fact in keyed:
                if isinstance(fact, AttemptFact) and fact.trace_context is not None and fact.trace_context.sampled:
                    stamps = tuple(sample[1] for sample in fact.response_samples[:512])
                    if stamps:
                        try:
                            self.ctx.sampled_output(SampledOutputTrace(self.ctx.run_id, fact.id, fact.attempts,
                                fact.trace_context, stamps, fact.output_event_count, len(stamps) < fact.output_event_count))
                        except Exception:
                            logger.exception("sampled output diagnostic delivery failed")
        return ComputedResultBatch(tuple(turns), tuple(sessions), tuple(attempts), tuple(series)) if expected or series else None


class MetricsRuntime:
    def __init__(
        self,
        backend: MetricsBackend,
        ctx: MetricsContext,
        publish_results: ResultPublisher,
        wait_for_results: Callable[[], Awaitable[None]],
    ) -> None:
        self.backend, self.ctx = backend, ctx
        self.wait_for_results = wait_for_results
        self.writer = AsyncBatchWriter(
            _MetricProcessor(backend, ctx), publish=publish_results, on_error=ctx.cancellation.cancel,
        )
        self.enable_diagnostics = self.writer.enable_diagnostics
        self.close = self.writer.close
        self.active_trial = None

    @classmethod
    async def create(
        cls,
        plugin: MetricsPlugin,
        ctx: MetricsContext,
        *,
        publish_results: ResultPublisher,
        wait_for_results: Callable[[], Awaitable[None]],
    ):
        runtime = cls(plugin.create_backend(seed=ctx.seed), ctx, publish_results, wait_for_results)
        await runtime.writer.start()
        return runtime

    async def open_trial(self, trial):
        if self.active_trial is not None:
            raise RuntimeError("only one metrics Trial may be active")
        self.active_trial = trial.id
        return MetricsCollector(self, trial.id)

    async def finalize(self, run):
        if self.active_trial is not None:
            raise RuntimeError("cannot finalize metrics with an active Trial")
        return await self.writer.call(self.backend.finalize_run, run.id)



class MetricsCollector:
    def __init__(self, runtime, trial_id):
        self.runtime, self.trial_id = runtime, trial_id
        self.closed = self.finalized = False

    def _submit(self, fact):
        if self.closed or self.finalized:
            raise RuntimeError("metrics Trial is closed")
        self.runtime.writer.submit(fact)

    async def wait_for_headroom(self):
        if self.closed or self.finalized:
            raise RuntimeError("metrics Trial is closed")
        await self.runtime.wait_for_results()
        await self.runtime.writer.wait_for_headroom()

    def _snapshot(self, turn, trace=None, result=None):
        request = result.request if result is not None else turn.measurements.request
        trace = result.trace if result is not None else trace
        for source in (trace, result):
            if source is not None and (source.turn_id, source.attempt) != (turn.id, request.attempts):
                raise ValueError("response trace or result belongs to another Turn or Attempt")
        extras = {}
        if result is not None:
            extras = {name: getattr(result, name) for name in (
                "error_origin", "trigger_ns", "decision_ns", "applied_ns",
            )}
            extras["plan"] = freeze_snapshot(asdict(result.plan) if result.plan else {})
        values = {name: getattr(turn, name) for name in _RESULT_FIELDS[TurnResultRecord]}
        values.update({name: getattr(result if result is not None else turn, name)
                       for name in ("status", "error", "scenario_passed")})
        return (AttemptFact if result is not None else TurnFact)(
            **values, **{target: getattr(request, name) for name, target in _REQUEST_FIELDS.items()},
            trial_id=self.trial_id,
            usage=freeze_snapshot(result.usage if result is not None else turn.usage),
            metadata=freeze_snapshot(turn.metadata), response_samples=trace.samples if trace else (),
            turn_started_ns=turn.measurements.lifecycle.started_ns,
            turn_completed_ns=(request.response_completed_ns if result is not None
                               else turn.measurements.lifecycle.completed_ns),
            total_started_ns=turn.measurements.total_started_ns,
            eligibility_started_ns=turn.measurements.eligibility_started_ns,
            eligibility_finished_ns=turn.measurements.eligibility_finished_ns,
            eligibility_target_ns=turn.measurements.eligibility_target_ns,
            dependency_ready_ns=turn.measurements.dependency_ready_ns, **extras,
        )


    def record_turn(self, turn, trace=None):
        self._submit(self._snapshot(turn, trace))

    def record_attempt(self, turn, result):
        self._submit(self._snapshot(turn, result=result))

    def record_session(self, session):
        lifecycle = session.measurements.lifecycle
        if lifecycle.started_ns is None or lifecycle.completed_ns is None:
            raise RuntimeError("session lifecycle measurements are incomplete")
        self._submit(SessionFact(
            session.id, session.trial_id, session.ordinal, session.status, session.started_at,
            session.finished_at, session.error, freeze_snapshot(session.metadata), len(session.turns),
            session.measurements.target_ns, session.measurements.offered_ns,
            session.measurements.admitted_ns,
            lifecycle.started_ns, lifecycle.completed_ns,
            sum(turn.status == "succeeded" for turn in session.turns),
        ))

    def record_occupancy(self, kind, at_ns, value):
        self._submit(OccupancyFact(self.trial_id, kind, at_ns, value))

    async def finalize(self, trial, fact: TrialFact):
        if self.closed or self.finalized or trial.id != self.trial_id:
            raise ValueError("metrics collector finalized for a closed or different Trial")
        if fact.trial_id != trial.id or fact.status != trial.status:
            raise ValueError("TrialFact belongs to another Trial state")
        self.finalized = True
        report = await self.runtime.writer.call(self.runtime.backend.finalize_trial, fact)
        trial.measurements.metric_windows = report.windows
        trial.measurements.metric_counts = report.counts
        trial.measurements.diagnostics.update(report.diagnostics)
        logger.info(
            "stage=metrics event=metrics.calculated trial_id=%s status=%s points=%s",
            trial.id,
            trial.status,
            len(report.points),
        )
        return report

    async def close(self):
        if not self.closed:
            self.closed = True
            async with AsyncExitStack() as cleanup:
                cleanup.callback(setattr, self.runtime, "active_trial", None)
                # Release the Trial even when flush reports a failed consumer or publisher.
                cleanup.push_async_callback(
                    self.runtime.writer.call, self.runtime.backend.release_trial, self.trial_id, after_error=True,
                )
                await self.runtime.writer.flush()
