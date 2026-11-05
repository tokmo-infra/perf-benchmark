"""Synthetic CPU-only coverage. No executor, HTTP, dataset, model, or services.

These tests are intentionally not marked live and never use wall-clock sleeps.
"""
import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
import pytest

from perf_benchmark.core.models import Run, Trial
from perf_benchmark.pipeline.contracts.facts import (
    LoadProfile, RepetitionPlan, ResourceLimits, TrialFact, TrialSchedulingFact,
)
from perf_benchmark.pipeline.execution.engine import ExecutionEngine
from perf_benchmark.pipeline.execution.trials import TrialController
from perf_benchmark.pipeline.registry import PluginSpec, StageBinding
from perf_benchmark.pipeline.results.metrics import MetricsCollector
from perf_benchmark.plugins.scheduler.multi_turn_closed_loop import (
    MultiTurnClosedLoopScheduler, validate,
)
from perf_benchmark.runtime.cancellation import CancellationToken

from perf_benchmark.pipeline.execution.underfill import UnderfillStop
from perf_benchmark.pipeline.results.measurement import AttemptFact
from tests.pipeline.test_underfill_stop import SECOND, complete, occupancy, started, terminal


class FakeWriter:
    def __init__(self):
        self.items = []
        self.error = None

    def submit(self, fact):
        if self.error:
            raise self.error
        self.items.append(fact)

    async def flush(self):
        if self.error:
            raise self.error


@pytest.mark.parametrize("outcome", ["ttft_safety", "http500"])
async def test_native_result_observer_keeps_domain_ids_and_http500_does_not_stop(
    trial_context, tmp_path, monkeypatch, outcome,
):
    # Inject the native terminal value, not the observer itself: exercise the
    # actual runtime -> engine -> durable notice path without sleeping 60 s,
    # network requests, models or a replacement transport implementation.
    from perf_benchmark.runtime.attempt import AttemptSignal
    from perf_benchmark.pipeline.contracts.executor import AttemptPlan
    from perf_benchmark.runtime.http.native_result import NativeResult
    from tests.runtime.test_request import setup_request
    from tests.pipeline.test_execution import _strategy, Sink

    ctx, runtime, attempt = await setup_request(trial_context)
    safety = outcome == "ttft_safety"
    value = NativeResult(
        status="failed", error="TTFTWatchdogExceeded" if safety else "HTTP500",
        error_origin="safety" if safety else "http", retryable=False,
        started_ns=100, completed_ns=60_000_000_201, http_status=None if safety else 500,
        samples=(), usage={}, output_chars=0, finish_reason=None, continuation=None,
        continuation_valid=False, output_kind="none", trigger_ns=None,
        decision_ns=None, applied_ns=None, transport_sent_ns=200,
    )
    handle = SimpleNamespace(result=AsyncMock(return_value=value), settled_done=True)
    submissions = []

    def submit(body, command, *, cancelled):
        submissions.append(command)
        assert not cancelled
        return handle

    runtime.native = SimpleNamespace(submit=submit, close=AsyncMock())
    writer = FakeWriter()
    metrics = MetricsCollector(SimpleNamespace(writer=writer), ctx.trial.id)
    engine = ExecutionEngine(ctx.domain_run, runtime, metrics, Sink(), ctx, _strategy())
    monkeypatch.setenv("PERF_CONTROL_DIR", str(tmp_path))
    engine.enable_underfill_stop(1, 2, 1, start_percent=40)
    try:
        prepared = await runtime.prepare_attempt(attempt)
        result = await runtime.execute_attempt(prepared, AttemptSignal(AttemptPlan()))
        assert callable(engine.run)
        assert len(submissions) == 1  # No automatic POST retry, including 500.
        assert result.error == value.error and result.error_origin == value.error_origin
        assert result.request.http_status == value.http_status
        assert result.transport_sent_ns == value.transport_sent_ns
        assert not runtime._prepared and prepared.release_done
        notice = tmp_path / "safety-stop.json"
        assert ctx.cancellation.is_cancelled() is safety
        assert notice.exists() is safety
        if safety:
            packet = json.loads(notice.read_text())
            assert packet["run_id"] == ctx.domain_run.id
            assert packet["trial_id"] == ctx.trial.id
            assert packet["request"]["turn_id"] == "turn"
            assert packet["request"]["request_id"] is None
            assert "X-Request-ID" not in submissions[0]["headers"]
            assert packet["request"]["sent_ns"] == value.transport_sent_ns
            assert packet["request"]["first_token_pending"] is True
            assert ctx.cancellation.reason == "ttft_watchdog_safety_abort"
            assert engine.underfill.hold_reason == "ttft_watchdog_safety_abort"
    finally:
        engine.underfill_watcher.cancel()
        await asyncio.gather(engine.underfill_watcher, return_exceptions=True)
        await runtime.close()
        await ctx.events.close()


class CollectorCounterTests(unittest.IsolatedAsyncioTestCase):
    async def test_typed500_attempts_not_error_text_final_turn_or_unknown(self):
        writer = FakeWriter()
        metrics = MetricsCollector(SimpleNamespace(writer=writer), "trial")
        seen = []
        metrics.observe_facts(seen.append)
        for i, status in enumerate((500, 503, None, "500", 500.0, True, 500)):
            metrics._submit(terminal(i, i + 1, cls=AttemptFact, http_status=status))
        metrics._submit(terminal(0, 1, status="failed", http_status=500))
        metrics._submit(replace(terminal(99, 2, cls=AttemptFact), error="HTTP 500 fake"))
        await metrics.flush()
        self.assertEqual(metrics.http500_attempts, 2)
        self.assertEqual(metrics.http_status_unknown_attempts, 4)
        self.assertEqual(metrics.attempts_observed, 8)
        self.assertEqual(seen, writer.items)

    async def test_rejected_writer_does_not_increment_count_or_notify_observer(self):
        writer = FakeWriter()
        writer.error = RuntimeError("synthetic writer rejection")
        metrics = MetricsCollector(SimpleNamespace(writer=writer), "trial")
        seen = []
        metrics.observe_facts(seen.append)
        with self.assertRaises(RuntimeError):
            metrics._submit(terminal(0, 1, cls=AttemptFact, http_status=500))
        self.assertEqual(metrics.http500_attempts, 0)
        self.assertEqual(seen, [])

    async def test_finalize_persists_counts_and_failed_ack_is_incomplete(self):
        report = SimpleNamespace(windows=(), counts={}, diagnostics={}, points=())
        for broken in (False, True):
            writer = FakeWriter()
            writer.call = AsyncMock(return_value=report)
            if broken:
                writer.call.side_effect = RuntimeError("synthetic finalize error")
            backend = SimpleNamespace(finalize_trial=object())
            metrics = MetricsCollector(SimpleNamespace(writer=writer, backend=backend), "trial")
            metrics._submit(terminal(0, 1, cls=AttemptFact, http_status=500))
            trial = Trial(id="trial")
            trial.finish()
            fact = TrialFact("trial", trial.status, LoadProfile("test"), ResourceLimits(1, 1, 1),
                             None, RepetitionPlan(), TrialSchedulingFact(), None)
            if broken:
                with self.assertRaises(RuntimeError):
                    await metrics.finalize(trial, fact)
            else:
                await metrics.finalize(trial, fact)
            self.assertEqual(trial.measurements.diagnostics["http500_attempts"], 1)
            self.assertEqual(trial.measurements.diagnostics["http_status_counts_complete"], not broken)


class SchedulerConfigurationTests(unittest.TestCase):
    def settings(self, **extra):
        return {"max_active_sessions": 3072, "max_concurrency": 2048,
                "stop": {"session_count": 3072}, "on_exhausted": "error", **extra}

    def test_default_off_and_explicit_finite2048_3072(self):
        for enabled in (False, True):
            settings = validate(self.settings(stop_on_underfill=enabled))
            plan, = MultiTurnClosedLoopScheduler(settings).trial_plans()
            self.assertEqual(plan.stop_on_underfill, enabled)
            self.assertEqual(plan.load_profile.target_concurrency, 2048)
            self.assertEqual(plan.stop.session_count, 3072)
            self.assertEqual(settings["retry"]["max_attempts"], 1)

    def test_unsafe_policy_is_rejected_not_silently_overwritten(self):
        cases = (
            {"retry": {"max_attempts": 2}}, {"cancel": {"after_outputs": 1}},
            {"repetitions": 2}, {"on_exhausted": "cycle"},
            {"stop": {"duration_s": 120}}, {"control": {"after_outputs": 1}},
            {"stop_on_underfill": "yes"},
            {"sweep": {"axis": "max_concurrency", "values": [2048]}},
        )
        for extra in cases:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                validate(self.settings(**{"stop_on_underfill": True, **extra}))


class EngineHandoffTests(unittest.IsolatedAsyncioTestCase):
    def engine(self, control):
        writer = FakeWriter()
        metrics = MetricsCollector(SimpleNamespace(writer=writer), "trial")
        metrics.observe_facts(control.observe)
        ctx = SimpleNamespace(events=SimpleNamespace(emit=AsyncMock()),
                              trial=Trial(id="trial"), cancellation=CancellationToken())
        engine = ExecutionEngine(Run(name="underfill-fixture"), SimpleNamespace(),
                                 metrics, SimpleNamespace(), ctx, None)
        engine.underfill = control
        return engine

    def test_only_undispatched_controls_advance_fences(self):
        control = UnderfillStop("trial", 4, 20, 1)
        engine = self.engine(control)
        for i, name in enumerate(("queued", "native", "returned")):
            control.begin_turn(name, 0, i, 0)
            engine.controls[name] = SimpleNamespace(
                turn=SimpleNamespace(id=name), transport_invoked=name == "native",
                result=object() if name == "returned" else None)
        engine._advance_undispatched_fences(100 * SECOND)
        self.assertEqual(control.active["queued"][0], 100 * SECOND)
        self.assertEqual(control.active["native"][0], 0)
        self.assertEqual(control.active["returned"][0], 0)

    async def test_capture_before_headroom_ties_native_order_and_cancel_does_not_drop(self):
        control = UnderfillStop("trial", 3072, 20, 2048)
        engine = self.engine(control)
        entered, release = asyncio.Event(), asyncio.Event()

        async def headroom():
            entered.set()
            await release.wait()

        engine.metrics.wait_for_headroom = headroom
        with patch("perf_benchmark.pipeline.execution.engine.perf_counter_ns", return_value=100):
            first = asyncio.create_task(engine._occupancy("in_flight", 1))
            await entered.wait()
            self.assertEqual(control.pending_occupancy, {1: (100, 1)})
            second = asyncio.create_task(engine._occupancy("in_flight", -1))
            first.cancel()
            release.set()
            results = await asyncio.gather(first, second, return_exceptions=True)
        self.assertIsInstance(results[0], asyncio.CancelledError)
        self.assertEqual(control.pending_occupancy, {})
        # The observer compresses predicate segments; native rows stay exact.
        self.assertEqual(list(control.occupancy), [(100, True)])
        rows = engine.metrics.runtime.writer.items
        self.assertEqual([(f.at_ns, f.value) for f in rows], [(100, 1), (100, 0)])
        self.assertIsNone(control.hold_reason)

    async def test_owner_observer_stops_only_after_metrics_and_saver_ack(self):
        control = started()
        occupancy(control, 20, 2047)
        complete(control, 19, 80, floor=12 * SECOND)
        engine = self.engine(control)
        engine.ctx = SimpleNamespace(cancellation=CancellationToken())
        engine.trial = Trial(id="trial")
        events = []

        async def metrics_ack():
            events.append("metrics")

        async def saver_ack():
            self.assertFalse(engine.ctx.cancellation.is_cancelled())
            events.append("saver")

        engine.metrics.flush = metrics_ack
        engine.sink = SimpleNamespace(flush=saver_ack)
        with patch("perf_benchmark.pipeline.execution.engine.asyncio.sleep", new=AsyncMock()), \
             patch("perf_benchmark.pipeline.execution.engine.perf_counter_ns",
                   return_value=81 * SECOND), \
             patch("perf_benchmark.pipeline.execution.stop_notice.emit_stop_notice") as notice:
            await engine._watch_underfill()
        self.assertEqual(events, ["metrics", "saver"])
        self.assertTrue(engine.ctx.cancellation.is_cancelled())
        self.assertEqual(control.requested_ns, 81 * SECOND)
        notice.assert_called_once_with(engine.domain_run.id, engine.trial.id, control.snapshot())
        self.assertIsNone(control.hold_reason)

    async def test_saver_ack_failure_holds_without_sending_cancel(self):
        control = started()
        occupancy(control, 20, 2047)
        complete(control, 19, 80, floor=12 * SECOND)
        engine = self.engine(control)
        engine.ctx = SimpleNamespace(cancellation=CancellationToken())
        engine.trial = Trial(id="trial")
        engine.sink = SimpleNamespace(flush=AsyncMock(side_effect=RuntimeError("synthetic ack")))
        with patch("perf_benchmark.pipeline.execution.engine.asyncio.sleep", new=AsyncMock()), \
             patch("perf_benchmark.pipeline.execution.engine.perf_counter_ns",
                   return_value=81 * SECOND):
            await engine._watch_underfill()
        self.assertFalse(engine.ctx.cancellation.is_cancelled())
        self.assertIsNone(control.requested_ns)
        self.assertIn("observer_failed", control.hold_reason)

    async def test_host_rejects_selected_denominator_mismatch_before_executor_creation(self):
        settings = validate(SchedulerConfigurationTests().settings(stop_on_underfill=True))
        plan, = MultiTurnClosedLoopScheduler(settings).trial_plans()
        for metric_settings, selected_count in (
                ({"maxtpm": True, "maxtpm_start_percent": 55}, 4000),
                ({"maxtpm": True, "maxtpm_start_percent": 0}, 3072)):
            binding = StageBinding(PluginSpec("standard", "metrics", object, "fake", dict),
                                   metric_settings, object())
            ctx = SimpleNamespace(cancellation=CancellationToken(), binding=lambda stage: binding)
            host = TrialController(
                Run(name="synthetic"),
                SimpleNamespace(max_graph_branches=0, session_count=selected_count),
                None, None, None, ctx,
            )
            with self.assertRaises(ValueError):
                async with host.open(plan, None):
                    self.fail("unsafe host opened")
            self.assertFalse(host.active)


async def test_cancel_at_fenced_send_never_executes_http_or_leaks_occupancy(trial_context):
    from tests.pipeline.test_execution import _admission, _session, _strategy, Sink
    runtime = SimpleNamespace(
        prepare=AsyncMock(), prepare_attempt=AsyncMock(side_effect=lambda value: value),
        execute_attempt=AsyncMock(), release=AsyncMock(),
        close_session=AsyncMock(), close=AsyncMock(),
    )
    ctx = trial_context("scheduler")
    ctx.trial.start()
    writer = FakeWriter()
    metrics = MetricsCollector(SimpleNamespace(writer=writer), ctx.trial.id)
    engine = ExecutionEngine(ctx.domain_run, runtime, metrics, Sink(), ctx,
                             _strategy(concurrency=1))
    engine.underfill = UnderfillStop(ctx.trial.id, 1, 2, 1)
    metrics.observe_facts(engine.underfill.observe)
    blocked, resume = asyncio.Event(), asyncio.Event()

    async def headroom():
        if engine.occupancy["in_flight"] == 1:
            blocked.set()
            await resume.wait()

    metrics.wait_for_headroom = headroom
    task = None
    try:
        await engine.open()
        engine.open_ready_window(1)
        task = asyncio.create_task(engine.run(_session("first", "pending"), _admission()))
        await asyncio.wait_for(blocked.wait(), 1)
        ctx.cancellation.cancel("before-http")
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        resume.set()
        await asyncio.wait_for(task, 1)
        runtime.execute_attempt.assert_not_awaited()
        attempts = [f for f in writer.items if isinstance(f, AttemptFact)]
        assert len(attempts) == 1
        assert attempts[0].status == "cancelled"
        assert attempts[0].http_status is None
        assert engine.occupancy["in_flight"] == engine.occupancy["active_sessions"] == 0
        assert not engine.underfill.pending_occupancy
    finally:
        resume.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        await engine.close()
        await ctx.events.close()


async def test_watcher_rechecks_native_horizon_after_flush_before_cancelling():
    control = started()
    occupancy(control, 20, 2047)
    for i in range(11, 19):
        control.begin_turn(f"t-{i}", 0, i, 12 * SECOND)
    engine = EngineHandoffTests().engine(control)
    engine.ctx = SimpleNamespace(cancellation=CancellationToken())
    engine.trial = Trial(id="trial")
    engine.metrics.flush = AsyncMock()

    async def saver_ack():
        control.begin_turn("t-19", 0, 19, 13 * SECOND)

    engine.sink = SimpleNamespace(flush=saver_ack)
    # End after the watcher has rejected the raced proof and resumed polling.
    with patch("perf_benchmark.pipeline.execution.engine.asyncio.sleep",
               new=AsyncMock(side_effect=[None, asyncio.CancelledError()])), \
         patch("perf_benchmark.pipeline.execution.engine.perf_counter_ns",
               return_value=81 * SECOND):
        try:
            await engine._watch_underfill()
        except asyncio.CancelledError:
            pass
    assert engine.metrics.flush.await_count == 1
    assert not engine.ctx.cancellation.is_cancelled()
    assert control.requested_ns is None
