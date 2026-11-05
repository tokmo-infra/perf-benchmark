"""CPU loopback HTTP + this tree's native executor + real metrics/SQLite writers.

No GPU/model/external service. Cancellation is injected only after the actual
HTTP response has returned, with barriers outside the transport timing window.
"""
import asyncio

from aiohttp import web
from aiohttp.test_utils import TestServer
import pytest

from perf_benchmark.core.models import Session
from perf_benchmark.pipeline.contracts.facts import (
    LoadProfile, RepetitionPlan, ResourceLimits, TrialFact, TrialSchedulingFact,
)
from perf_benchmark.pipeline.execution.engine import ExecutionEngine
from perf_benchmark.pipeline.execution.underfill import UnderfillStop
from perf_benchmark.pipeline.results.measurement import AttemptFact, OccupancyFact
from perf_benchmark.pipeline.results.metrics import MetricsRuntime
from perf_benchmark.pipeline.results.saver import SaverRuntime
from perf_benchmark.plugins.metrics.standard import StandardMetrics
from perf_benchmark.plugins.saver.sqlite import SQLiteResultReader, SQLiteSaver
from tests.executor.test_openai_chat import _runtime, _turn
from tests.pipeline.test_execution import _admission, _strategy


async def test_native_watchdog_stops_other_requests_before_python_notification(monkeypatch):
    from perf_benchmark import _native
    from tests.runtime.native_support import command

    monkeypatch.setenv("PERF_TTFT_LIMIT_MS", "500")
    active = asyncio.Event()
    release = asyncio.Event()
    count = 0

    async def stalled(request):
        nonlocal count
        await request.read()
        response = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        count += 1
        if count == 2:
            active.set()
        await release.wait()
        return response

    app = web.Application()
    app.router.add_post("/", stalled)
    async with TestServer(app) as server:
        trial = _native.create_trial("openai-chat")
        spec = command(url=str(server.make_url("/")), stream=True, timeout=5.0)
        first = trial.submit(_native.PreparedBody((), 0), spec)
        second = trial.submit(_native.PreparedBody((), 0), spec)
        try:
            await asyncio.wait_for(active.wait(), 3)
            outcomes = await asyncio.wait_for(asyncio.gather(first.result(), second.result()), 3)
            assert any(value.error_origin == "safety" for value in outcomes)
            assert all(value.status != "succeeded" for value in outcomes)
            assert all(value.transport_sent_ns is not None for value in outcomes)
            with pytest.raises(RuntimeError):
                trial.submit(_native.PreparedBody((), 0), spec)
        finally:
            release.set()
            await first.settled()
            await second.settled()
            await trial.close()


@pytest.mark.parametrize("delay_headers", [False, True])
async def test_slow_http500_with_guard_never_stops_native_trial(monkeypatch, delay_headers):
    from perf_benchmark import _native
    from tests.runtime.native_support import command

    monkeypatch.setenv("PERF_TTFT_LIMIT_MS", "100")
    release = asyncio.Event()

    async def fail(request):
        await request.read()
        if delay_headers:
            await asyncio.sleep(0.2)
        response = web.StreamResponse(status=500)
        await response.prepare(request)
        await release.wait()  # a slow or stalled error body must not affect status handling
        return response

    app = web.Application()
    app.router.add_post("/", fail)
    async with TestServer(app) as server:
        trial = _native.create_trial("openai-chat")
        spec = command(url=str(server.make_url("/")), stream=True, timeout=2.0)
        first = trial.submit(_native.PreparedBody((), 0), spec)
        second = None
        try:
            result = await asyncio.wait_for(first.result(), 3)
            assert result.status == "failed"
            assert result.http_status == 500 and result.error_origin == "service"
            # Submission after the observed 500 proves the native trial was not latched.
            second = trial.submit(_native.PreparedBody((), 0), spec)
            other = await asyncio.wait_for(second.result(), 3)
            assert other.http_status == 500 and other.error_origin == "service"
        finally:
            release.set()
            await first.settled()
            if second is not None:
                await second.settled()
            await trial.close()


@pytest.mark.parametrize("code", [200, 500])
@pytest.mark.parametrize("phase", ["release", "headroom", "observation"])
@pytest.mark.parametrize("cancels", [1, 3])
async def test_completed_http_attempt_survives_cleanup_cancellation(
    trial_context, stage_context, tmp_path, monkeypatch, code, phase, cancels,
):
    if code == 500:
        monkeypatch.setenv("PERF_TTFT_LIMIT_MS", "100")
    ctx = trial_context("scheduler")
    run, trial = ctx.domain_run, ctx.trial
    run.prepare()
    run.start()
    trial.start()
    turn = await _turn(ctx, "http-turn", "local fixture")
    skipped = await _turn(ctx, "never-sent", "must remain pending")
    skipped.ordinal = 1
    ctx.payloads.seal()
    session = Session(id="http-session", trial_id=trial.id, turns=[turn, skipped])
    session.link()
    requests, facts, returned = [], [], []
    blocked, resume = asyncio.Event(), asyncio.Event()

    async def response(request):
        requests.append(await request.json())
        if code == 500:
            await asyncio.sleep(0.2)  # Unknown status past the guard deadline.
            return web.json_response({"error": {"message": "real loopback failure"}}, status=500)
        return web.json_response({
            "choices": [{"message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        })

    app = web.Application()
    app.router.add_post("/v1/chat/completions", response)
    async with TestServer(app) as server:
        runtime = _runtime({"base_url": str(server.make_url("/")),
                            "model": "cpu-fixture", "stream": False}, ctx)
        database = tmp_path / "attempt.db"
        sink = await SaverRuntime.create(SQLiteSaver({"path": str(database)}), run,
                                        stage_context("saver", run=run))
        metrics_runtime = await MetricsRuntime.create(
            StandardMetrics({}), stage_context("metrics", run=run),
            publish_results=sink.write_results, wait_for_results=sink.wait_for_headroom,
        )
        metrics = await metrics_runtime.open_trial(trial)
        engine = ExecutionEngine(run, runtime, metrics, sink, ctx, _strategy(concurrency=1),
                                 resource_concurrency=1)
        # Deliberately no polling: this regression isolates the stop's cancel
        # boundary rather than spending a real minute in a CPU HTTP test.
        engine.underfill = UnderfillStop(trial.id, 1, 2, 1)

        def observed(fact):
            facts.append(fact)
            engine.underfill.observe(fact)

        metrics.observe_facts(observed)
        execute, release = runtime.execute_attempt, runtime.release
        headroom, scheduler_event = metrics.wait_for_headroom, engine._scheduler_event

        async def execute_observed(*args):
            result = await execute(*args)
            returned.append(result)
            return result

        async def release_observed(prepared):
            if phase == "release" and returned:
                blocked.set()
                await resume.wait()
            return await release(prepared)

        async def wait_headroom():
            if phase == "headroom" and returned and engine.occupancy["in_flight"] == 0:
                blocked.set()
                await resume.wait()
            await headroom()

        async def observe_event(name, **kwargs):
            if (phase == "observation" and returned
                    and kwargs.get("data") == {"kind": "in_flight", "value": 0}):
                blocked.set()
                await resume.wait()
            await scheduler_event(name, **kwargs)

        runtime.execute_attempt, runtime.release = execute_observed, release_observed
        metrics.wait_for_headroom, engine._scheduler_event = wait_headroom, observe_event
        task = None
        try:
            await sink.write_trial(trial)
            assert await engine.open()
            engine.open_ready_window(1)
            task = asyncio.create_task(engine.run(session, _admission()))
            await asyncio.wait_for(blocked.wait(), 5)
            assert len(returned) == 1
            original = returned[0]
            assert original.request.http_status == code
            assert not ctx.cancellation.is_cancelled()
            # Exercise the same token->watcher boundary as automatic stop, then
            # repeat task cancellation to ensure owned cleanup survives it too.
            ctx.cancellation.cancel("cpu-regression-stop")
            for _ in range(cancels):
                await asyncio.sleep(0)
                task.cancel()
                await asyncio.sleep(0)
            resume.set()
            await asyncio.wait_for(task, 5)
            await metrics.flush()
            await sink.flush()

            attempts = [f for f in facts if isinstance(f, AttemptFact)]
            assert len(attempts) == 1
            fact = attempts[0]
            assert (fact.id, fact.attempts, fact.http_status) == (turn.id, 1, code)
            assert fact.status == ("succeeded" if code == 200 else "failed")
            assert fact.response_completed_ns == original.request.response_completed_ns
            assert fact.usage == original.usage
            assert turn.measurements.request is original.request
            assert turn.status == session.status == "cancelled"
            assert skipped.status == "pending"
            assert metrics.http500_attempts == (code == 500)
            assert len(requests) == 1  # NO retry of HTTP 500, or anything else
            rows = [f.value for f in facts if isinstance(f, OccupancyFact)
                    and f.kind == "in_flight"]
            assert rows == [1, 0]
            assert engine.occupancy["in_flight"] == engine.occupancy["active_sessions"] == 0
            assert engine.strategy.attempts.semaphore._value == 1
            assert not engine.underfill.pending_occupancy
            reader = SQLiteResultReader(database)
            stored = tuple(reader.iter_attempt_records(run.id))
            assert len(stored) == 1
            assert stored[0]["status"] == fact.status
            assert stored[0]["measurements"]["http_status"] == code
            assert stored[0]["measurements"]["response_completed_ns"] == fact.response_completed_ns
            trial.finish()
            final = TrialFact(trial.id, trial.status, LoadProfile("test"), ResourceLimits(1, 1, 1),
                              None, RepetitionPlan(), TrialSchedulingFact())
            await metrics.finalize(trial, final)
            assert trial.measurements.diagnostics["http_status_counts_complete"] is True
            assert trial.measurements.diagnostics["http500_attempts"] == (code == 500)
        finally:
            resume.set()
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            await engine.close()
            await metrics.close()
            await metrics_runtime.close()
            await sink.close()
            await ctx.events.close()
