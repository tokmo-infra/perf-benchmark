from __future__ import annotations

from contextlib import AsyncExitStack
import logging

from perf_benchmark.core.errors import ResultArchiveError
import asyncio
from pathlib import Path
from time import perf_counter_ns
from typing import cast

from perf_benchmark.core.logging import configure_logging, shutdown_logging
from perf_benchmark.core.models import Run, StageMeasurements, now, require_metadata
from perf_benchmark.core.observation import ObservationSettings
from perf_benchmark.core.validation import random_seed
from perf_benchmark.pipeline.context import (
    ExporterContext, MetricsContext, RunContext, SaverContext, SchedulerContext,
)
from perf_benchmark.pipeline.contracts import (
    ExecutorPlugin, ExporterPlugin, ExportResult, MetricsPlugin, SaverPlugin, SchedulerPlugin,
)
from perf_benchmark.pipeline.hooks import (
    HookContext, HookRegistry, create_registry as create_hook_registry,
    load_configured as load_configured_hooks,
)
from perf_benchmark.pipeline.results.metrics import MetricsRuntime
from perf_benchmark.pipeline.results.progress import ProgressLog
from perf_benchmark.pipeline.preparation.payloads import BlobStore
from perf_benchmark.pipeline.registry import StageRegistry
from perf_benchmark.pipeline.resources import own
from perf_benchmark.pipeline.results.saver import SaverRuntime
from perf_benchmark.pipeline.preparation.streaming import PreparedCorpus
from perf_benchmark.pipeline.execution.trials import TrialController
from perf_benchmark.pipeline.contracts.preparation import TrialPreparation
from perf_benchmark.plugins import create_registry as create_stage_registry
from perf_benchmark.runtime.cancellation import CancellationToken
from perf_benchmark.runtime.events import EventDispatcher
from perf_benchmark.runtime.observation import ObservationRuntime

logger = logging.getLogger("perf_benchmark.coordinator")


class _StageLifecycle:
    """Own terminal exclusivity and the common lifecycle event envelope."""

    def __init__(self, ctx: RunContext, timings):
        self.ctx = ctx
        self.timings = timings
        self.sink: SaverRuntime | None = None

    async def emit(self, event: str, *, at_ns=None, **data):
        stage = event.split(".")[0]
        await self.ctx.events.emit(
            event, stage="coordinator" if stage == "run" else stage, data=data, at_ns=at_ns,
        )

    async def publish(self):
        if self.sink is not None:
            await self.sink.write_run_measurement("stages", self.timings)

    async def start(self, stage: str):
        binding = self.ctx.binding(stage)
        timing = self.timings[stage] = StageMeasurements(
            plugin=binding.name, started_ns=perf_counter_ns(), started_at=now(),
        )
        await self.publish()
        await self.emit(f"{stage}.started", plugin=binding.name, at_ns=timing.started_ns)
        return binding.plugin

    async def finish(self, *stages, outcome="completed", **data):
        data.setdefault("status", "succeeded" if outcome == "completed" else outcome)
        if outcome == "cancelled":
            data.setdefault("error", self.ctx.cancellation.reason)
        for stage in stages:
            timing = self.timings.get(stage)
            if timing is not None and timing.status == "started":
                timing.status = outcome
                timing.completed_ns, timing.completed_at = perf_counter_ns(), now()
                assert timing.started_ns is not None
                timing.duration_s = (timing.completed_ns - timing.started_ns) / 1e9
                await self.publish()
                await self.emit(f"{stage}.{outcome}", plugin=timing.plugin, at_ns=timing.completed_ns,
                                duration_s=timing.duration_s, **data)


class _Results:
    """One result cluster shared by success and partial-result finalization."""

    def __init__(self, stages: _StageLifecycle, observation: ObservationRuntime):
        self.stages, self.observation = stages, observation
        self.sink: SaverRuntime | None = None
        self.metrics: MetricsRuntime | None = None
        self.metrics_finalized = False

    async def finalize_metrics(self, run):
        if self.metrics is not None and not self.metrics_finalized:
            assert self.sink is not None
            points = await self.metrics.finalize(run)
            await self.sink.write_metric_points(points)
            self.metrics_finalized = True

    async def finalize_saver(self, run, status):
        assert self.sink is not None
        reader = await self.sink.finalize(run)
        await self.stages.emit("saver.transaction.committed", location=self.sink.artifact.location)
        await self.stages.finish("saver", status=status)
        await self.sink.flush()
        await self.observation.close()
        if self.observation.settings.enabled:
            await self.sink.write_run_measurement('observation_summary', {
                'output_messages_sent': self.observation.outputs_sent,
                'output_messages_dropped': self.observation.outputs_dropped,
                'observer': self.observation.summary,
                'availability': 'observed' if self.observation.summary is not None else 'unknown',
            })
            await self.sink.flush()
        return reader

    async def preserve_partial(self, run):
        try:
            await self.finalize_metrics(run)
        except BaseException:
            logger.exception("partial metrics finalization failed")
        try:
            if self.sink is not None:
                await self.finalize_saver(run, "partial")
                return
        except BaseException:
            logger.exception("partial result finalization failed")
        if "saver" in self.stages.timings and self.stages.timings["saver"].status == "started":
            try:
                data = ({"location": self.sink.artifact.location} if self.sink else
                        {"plugin": self.stages.ctx.binding("saver").name})
                await self.stages.emit("saver.transaction.rolled_back", **data)
            except Exception:
                logger.exception("saver rollback hook failed")


class Coordinator:
    def __init__(self, registry: StageRegistry | None = None, hook_registry: HookRegistry | None = None):
        self.registry = registry or create_stage_registry()
        self.hook_registry = hook_registry or create_hook_registry()

    async def execute(self, config: dict, *, base_dir: Path | None = None,
                      cancellation: CancellationToken | None = None,
                      evidence: dict | None = None, preparation: TrialPreparation | None = None,
                      live_display=None) -> Run:
        self._validate(config)
        evidence = require_metadata(evidence or {}, event="run.evidence")
        seed = random_seed(config.get("seed", 0))
        bindings = self.registry.bind_pipeline(config["pipeline"])
        cast(SchedulerPlugin, bindings["scheduler"].plugin).validate_metrics(bindings["metrics"].plugin)
        observation = ObservationRuntime(ObservationSettings.parse(config.get("observation", {}), base_dir=base_dir))
        ObservationRuntime.validate(observation.settings)
        if observation.settings.enabled and observation.settings.trace_mode != "benchmark":
            cast(ExecutorPlugin, bindings["executor"].plugin).validate_observation()
        configure_logging(config.get("logging"))
        async with AsyncExitStack() as resources:
            resources.callback(shutdown_logging)
            progress = ProgressLog(config.get("logging"))
            own(resources, progress.close)
            root = base_dir or Path.cwd()
            load_configured_hooks(self.hook_registry, config.get("hooks", {}), root)
            run = Run(name=str(config.get("name", "run")))
            dispatcher = EventDispatcher(
                run, self.hook_registry, config.get("hooks", {}), HookContext(run.id, run.name, root),
            )
            own(resources, dispatcher.close)
            own(resources, observation.close)
            before = await dispatcher.emit(
                "run.before_start", {"name": run.name, "metadata": dict(run.metadata)}, stage="coordinator",
            )
            if not isinstance(before, dict) or not isinstance(before["name"], str) or not before["name"]:
                raise TypeError("run.before_start requires a non-empty name")
            run.name = before["name"]
            run.metadata = require_metadata(before["metadata"], event="run.before_start")
            run.metadata["seed"] = seed
            ctx = RunContext(bindings, root, dispatcher, cancellation or CancellationToken(),
                             BlobStore(), observation.settings, seed)
            corpus = PreparedCorpus(ctx.payloads)
            own(resources, corpus.close)
            stages = _StageLifecycle(ctx, run.measurements.stages)
            results = _Results(stages, observation)
            try:
                run.prepare()
                progress.start(run)
                await stages.emit("run.preparing", name=run.name)
                await corpus.prepare(run, ctx, stages)
                if ctx.cancellation.is_cancelled():
                    run.finish(ctx.cancellation.reason)
                else:
                    run.start()
                    await stages.emit("run.started", name=run.name)

                executor = cast(ExecutorPlugin, await stages.start("executor"))
                await stages.emit("executor.bound", plugin=type(executor).__name__)
                saver = cast(SaverPlugin, await stages.start("saver"))
                await stages.emit("saver.transaction.started", plugin=ctx.binding("saver").name)
                sink = results.sink = await SaverRuntime.create(saver, run, SaverContext(ctx.base_dir, ctx.cancellation, observation.committed))
                close_sink = own(resources, sink.close)
                stages.sink = sink
                fingerprints = sorted({t.profile.tokenizer_fingerprint for p in corpus.plans for t in p.turns
                                       if t.profile is not None and t.profile.tokenizer_fingerprint is not None})
                conditions = evidence.get("conditions")
                if isinstance(conditions, dict):
                    conditions["tokenizer_fingerprint"] = require_metadata({"value": fingerprints or None,
                        "status": "observed" if fingerprints else "unknown"}, event="run.conditions")
                await sink.write_run_measurement("evidence", evidence)
                await sink.write_run_measurement("observation_mode", {"enabled": observation.settings.enabled,
                    "trace_mode": observation.settings.trace_mode, "trace_export": observation.settings.trace_export})
                await stages.publish()
                await stages.emit("saver.artifact.created", kind=sink.artifact.kind, location=sink.artifact.location)
                await observation.start(sink, config["pipeline"]["saver"], run,
                                        snapshot_socket=live_display.socket_path if live_display else None,
                                        prometheus_url=live_display.prometheus_url if live_display else None,
                                        display_pid=live_display.process.pid if live_display else None)

                metrics_plugin = cast(MetricsPlugin, await stages.start("metrics"))
                def publish(batch):
                    sink.write_results(batch)
                    progress.consume(batch)
                metrics = results.metrics = await MetricsRuntime.create(
                    metrics_plugin, MetricsContext(ctx.cancellation, ctx.seed, run.id, observation.sampled_output if observation.settings.enabled and observation.settings.trace_mode == "debug" else None), publish_results=publish, wait_for_results=sink.wait_for_headroom,
                )
                close_metrics = own(resources, metrics.close)
                if observation.settings.enabled:
                    observation.watch("metrics", metrics.enable_diagnostics())
                scheduler = cast(SchedulerPlugin, await stages.start("scheduler"))
                trials = TrialController(run, corpus, executor, metrics, sink, ctx, preparation, progress=progress)
                progress.load = lambda: trials.diagnostics()
                observation.watch("load", lambda: trials.diagnostics())
                if not ctx.cancellation.is_cancelled():
                    await scheduler.schedule(trials, SchedulerContext(ctx.cancellation, ctx.seed))
                run.finish(ctx.cancellation.reason if ctx.cancellation.is_cancelled() else None)
                await stages.finish("scheduler", "executor", status=run.status,
                                    outcome="cancelled" if ctx.cancellation.is_cancelled() else "completed")

                await results.finalize_metrics(run)
                await close_metrics()
                await stages.finish("metrics", status=run.status)
                reader = await results.finalize_saver(run, run.status)
                if live_display is not None:
                    try:
                        await live_display.stop()
                    except Exception:
                        logger.exception("TUI shutdown failed; committed benchmark results remain authoritative")
                exporters = cast(tuple[ExporterPlugin, ...], await stages.start("exporter"))
                exporter_ctx = ExporterContext(ctx.base_dir, ctx.cancellation)
                for exporter in exporters:
                    exported = await exporter.export(run.id, reader, exporter_ctx)
                    if not isinstance(exported, ExportResult):
                        raise TypeError("exporter must return ExportResult")
                    for output in exported.outputs:
                        if output.kind in run.artifacts and run.artifacts[output.kind] != output.location:
                            raise ValueError(f"duplicate exporter output kind: {output.kind}")
                        run.artifacts[output.kind] = output.location
                        await stages.emit("exporter.output.completed", kind=output.kind,
                                          location=output.location, format=output.format)
                await stages.finish("exporter", status=run.status)
                await close_sink()
                terminal = "completed" if run.status == "succeeded" else run.status
                await stages.emit(f"run.{terminal}", status=run.status)
                return run
            except ResultArchiveError:
                logger.exception("result archive delivery failed run_id=%s", run.id)
                raise
            except BaseException as exc:
                if live_display is not None:
                    try:
                        await live_display.stop()
                    except Exception:
                        logger.exception("TUI shutdown failed while preserving benchmark results")
                if isinstance(exc, asyncio.CancelledError) and ctx.cancellation.is_cancelled():
                    run.finish(ctx.cancellation.reason)
                    await results.preserve_partial(run)
                    await stages.finish(*StageRegistry.STAGES, outcome="cancelled", status=run.status)
                    await stages.emit("run.cancelled", status=run.status)
                    return run
                run.fail(f"{type(exc).__name__}: {exc}")
                logger.exception("run failed run_id=%s", run.id)
                await results.preserve_partial(run)
                try:
                    await stages.finish(*StageRegistry.STAGES, outcome="failed", error=run.error)
                    await stages.emit("run.failed", status=run.status, error=run.error)
                except Exception:
                    logger.exception("run.failed hook failed")
                raise

    @staticmethod
    def _validate(config: dict) -> None:
        if "plugins" in config:
            raise ValueError("plugins configuration is not supported; register Stage Plugins in perf_benchmark.plugins")
        if not isinstance(config.get("pipeline"), dict):
            raise ValueError("pipeline must be an object")
        if "streaming" in config:
            raise ValueError("streaming configuration is not supported")
