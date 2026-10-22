from __future__ import annotations


from contextlib import AsyncExitStack, asynccontextmanager
from copy import deepcopy
from dataclasses import asdict, replace
from functools import partial
import asyncio
import hashlib
import json
from types import MappingProxyType

from perf_benchmark.core.lifecycle import (
    METRICS_CALCULATED,
    TRIAL_BEFORE_START,
    TRIAL_COMPLETED,
    TRIAL_STARTED,
)
from perf_benchmark.core.models import Trial, require_metadata, thaw_snapshot
from perf_benchmark.pipeline.context import ExecutorContext, RunContext
from perf_benchmark.pipeline.contracts.facts import TrialFact, TrialSchedulingFact
from perf_benchmark.pipeline.contracts.scheduler import (
    TrialOutcome,
    EffectiveTrial,
    StopPlan,
    TrialExecution,
    TrialPlan,
    TurnStrategy,
)
from perf_benchmark.pipeline.execution.engine import ExecutionEngine
from perf_benchmark.pipeline.hooks import require_request_parameters
from perf_benchmark.pipeline.resources import own
from perf_benchmark.pipeline.registry import StageBinding



def _plan_parameters(plan: TrialPlan, *, stop_source: str) -> dict:
    stop = (
        {"session_count": plan.stop.session_count}
        if plan.stop.session_count is not None
        else {"duration_s": plan.stop.duration_s}
    )
    return {
        "stop": stop,
        "stop_source": stop_source,
        "on_exhausted": plan.stop.on_exhausted,
        "load_profile": asdict(plan.load_profile),
        "resource_limits": asdict(plan.resources),
        "comparison": asdict(plan.comparison) if plan.comparison is not None else None,
        "repetition": asdict(plan.repetition),
        "validity_requirements": asdict(plan.validity_requirements) if plan.validity_requirements else None,
        "request_parameters": thaw_snapshot(plan.request_parameters),
    }


class TrialController:
    def __init__(
        self, run, streaming_input, executor_plugin, metrics_runtime, sink, ctx: RunContext, preparation=None,
        *, progress=None,
    ):
        self.preparation = preparation
        self.progress = progress
        self.run = run
        self.streaming_input = streaming_input
        self.executor_plugin = executor_plugin
        self.metrics_runtime = metrics_runtime
        self.sink = sink
        self.ctx = ctx
        self.active = False
        self.diagnostics = lambda: {}

    async def record_sweep(self, record):
        await self.sink.write_sweep(replace(record, run_id=self.run.id))
        await self.sink.flush()

    @asynccontextmanager
    async def open(self, plan: TrialPlan, strategy: TurnStrategy):
        if self.active:
            raise RuntimeError("only one Trial may execute at a time")
        if self.ctx.cancellation.is_cancelled():
            raise RuntimeError("cannot open Trial after cancellation")
        if self.streaming_input.max_graph_branches:
            if plan.load_profile.mode != "trace-causal":
                raise ValueError("causal plans require trace-causal Scheduler")
            if plan.load_profile.max_active_sessions is None or self.streaming_input.max_graph_branches > plan.load_profile.max_active_sessions:
                raise ValueError("one graph exceeds max_active_sessions")
            if self.streaming_input.requires_start_notifications and not self.executor_plugin.supports_start_notifications:
                raise ValueError("causal replay requires reliable execution start notifications")
        self.active = True
        stop_source = "scheduler"
        if plan.stop.session_count is None and plan.stop.duration_s is None:
            plan = replace(
                plan,
                stop=StopPlan(
                    session_count=self.streaming_input.session_count,
                    on_exhausted="error",
                ),
            )
            stop_source = "selector"
        ordinal = len(self.run.trial_ids)
        trial = Trial(
            id=f"{self.run.id}:trial-{ordinal + 1}",
            name=plan.name,
            ordinal=ordinal,
            parameters=_plan_parameters(plan, stop_source=stop_source),
            metadata=thaw_snapshot(plan.metadata),
        )
        emit = partial(self.ctx.events.emit, stage="scheduler", trial_id=trial.id)
        metrics = close_engine = close_metrics = close_stream = engine = None
        error = None
        execution = None
        try:
            async with AsyncExitStack() as resources:
                try:
                    payload = await emit(
                        TRIAL_BEFORE_START,
                        {
                            "name": trial.name,
                            "metadata": dict(trial.metadata),
                            "request_parameters": deepcopy(
                                trial.parameters["request_parameters"]
                            ),
                        },
                    )
                    if (
                        not isinstance(payload, dict)
                        or not isinstance(payload["name"], str)
                        or not payload["name"]
                    ):
                        raise TypeError("trial.before_start requires a non-empty name")
                    trial.name = payload["name"]
                    trial.metadata = require_metadata(
                        payload["metadata"], event=TRIAL_BEFORE_START
                    )
                    request_parameters = require_request_parameters(
                        payload["request_parameters"]
                    )
                    trial.parameters["request_parameters"] = request_parameters
                    scheduler = self.ctx.binding("scheduler")
                    assert isinstance(scheduler, StageBinding)
                    trial.metadata["scheduler_plugin"] = scheduler.name
                    trial.metadata["scheduler_fingerprint"] = hashlib.sha256(
                        json.dumps(
                            {
                                "plugin": scheduler.name,
                                "settings": thaw_snapshot(scheduler.settings),
                                "seed": self.ctx.seed,
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode()
                    ).hexdigest()
                    effective = EffectiveTrial(
                        trial.id,
                        trial.name,
                        plan,
                        request_parameters,
                        trial.metadata,
                    )
                    self.run.register_trial(trial)
                    if self.preparation is not None:
                        try:
                            prepared = await self.preparation.prepare(
                                effective, self.ctx.cancellation, self.streaming_input, self.executor_plugin
                            )
                        except BaseException as exc:
                            if hasattr(exc, "fact"):
                                trial.metadata["preparation"] = asdict(exc.fact)
                            raise
                        trial.metadata["preparation"] = asdict(prepared)
                        if self.ctx.cancellation.is_cancelled():
                            raise asyncio.CancelledError(self.ctx.cancellation.reason)
                    sessions = self.streaming_input.session_stream(
                        trial,
                        session_count=plan.stop.session_count,
                        on_exhausted=plan.stop.on_exhausted,
                    )
                    close_stream = own(resources, sessions.aclose)
                    metrics = await self.metrics_runtime.open_trial(trial)
                    close_metrics = own(resources, metrics.close)
                    trial_ctx = self.ctx.for_trial(self.run, trial, metrics)
                    runtime = await self.executor_plugin.create_runtime(
                        effective.executor_trial(),
                        ExecutorContext(self.ctx.cancellation),
                    )
                    engine = ExecutionEngine(
                        self.run,
                        runtime,
                        metrics,
                        self.sink,
                        trial_ctx,
                        strategy,
                        resource_concurrency=plan.load_profile.max_concurrency,
                    )
                    close_engine = own(resources, engine.close)
                    self.diagnostics = engine.diagnostics
                    ready = await engine.open()
                    if ready:
                        engine.open_ready_window(plan.resources.ready_capacity)
                        trial.start()
                        if self.progress is not None:
                            self.progress.begin_trial(trial)
                        await self.sink.write_trial(trial)
                        await emit(
                            TRIAL_STARTED,
                            data={
                                "name": trial.name,
                                "parameters": MappingProxyType(
                                    deepcopy(trial.parameters)
                                ),
                            },
                        )
                    execution = TrialExecution(sessions, engine)
                    yield execution
                except BaseException as exc:
                    error = exc
                    raise
                finally:
                    if trial.run_id is not None:
                        outcome = await self._finish(
                            trial,
                            plan,
                            engine,
                            metrics,
                            (close_stream, close_engine),
                            close_metrics,
                            error,
                        )
                        if execution is not None and error is None:
                            execution._outcome = outcome
        finally:
            self.active = False

    async def _finish(
        self,
        trial,
        plan,
        engine,
        metrics,
        execution_resources,
        close_metrics,
        error,
    ):
        emit = partial(self.ctx.events.emit, stage="scheduler", trial_id=trial.id)
        failure = None if isinstance(error, asyncio.CancelledError) else error
        points, validity = (), None
        for close in execution_resources:
            if close is not None:
                try:
                    await close()
                except BaseException as exc:
                    failure = failure or exc
        trial.finish(
            failure_reason=f"{type(failure).__name__}: {failure}" if failure else None,
            cancellation_reason=(
                self.ctx.cancellation.reason
                if failure is None and self.ctx.cancellation.is_cancelled()
                else "task_cancelled" if isinstance(error, asyncio.CancelledError) else None
            ),
        )
        scheduling = (
            engine.scheduling_fact
            if engine is not None and engine.scheduling_fact is not None
            else TrialSchedulingFact(
                termination_reason=(
                    self.ctx.cancellation.reason
                    if self.ctx.cancellation.is_cancelled()
                    else "trial_setup_failed"
                    if failure is not None
                    else "scheduler_returned_without_facts"
                )
            )
        )
        scheduling = replace(
            scheduling, dataset_session_count=self.streaming_input.session_count,
            dataset_turn_count=self.streaming_input.turn_count,
        )
        fact = TrialFact(
            trial.id, trial.status, plan.load_profile, plan.resources,
            plan.comparison, plan.repetition, scheduling, plan.validity_requirements,
            execution_status="failed" if failure is not None else "cancelled" if self.ctx.cancellation.is_cancelled() or isinstance(error, asyncio.CancelledError) else "succeeded",
        )
        if metrics is not None:
            try:
                report = await metrics.finalize(trial, fact)
                points, validity = report.points, report.validity
                fact = replace(fact, validity=validity)
                await self.sink.write_metric_points(points)
                await self.sink.write_metric_points(report.groups)
                await emit(
                    METRICS_CALCULATED,
                    stage="metrics",
                    data={
                        "metrics": len({point.metric for point in points}),
                        "windows": len(trial.measurements.metric_windows),
                        "points": len(points),
                    },
                )
            except BaseException as exc:
                failure = failure or exc
            finally:
                try:
                    await close_metrics()
                except BaseException as exc:
                    failure = failure or exc
        if failure is not None:
            trial.finish(failure_reason=f"{type(failure).__name__}: {failure}")
            if validity is not None:
                validity = replace(validity, execution_status="failed", status="indeterminate", reasons=(*validity.reasons, "execution_failed"))
            fact = replace(fact, status=trial.status, execution_status="failed", validity=validity)
        try:
            await self.sink.write_trial(trial, fact)
            await self.sink.flush()
            self.run.observe_trial(trial)
            if self.progress is not None:
                self.progress.finish_trial(trial)
            await emit(
                TRIAL_COMPLETED,
                data={
                    "status": trial.status,
                    "name": trial.name,
                    "parameters": deepcopy(trial.parameters),
                    "sessions": trial.session_count,
                    "turns": trial.turn_count,
                },
            )
        except BaseException as exc:
            failure = failure or exc
        if failure is not None and (error is None or isinstance(error, asyncio.CancelledError)):
            raise failure
        return TrialOutcome(trial.id, trial.status, scheduling, tuple(points), validity)
