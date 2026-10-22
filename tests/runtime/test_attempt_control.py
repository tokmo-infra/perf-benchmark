
from perf_benchmark.pipeline.results.metrics import MetricsRuntime
from perf_benchmark.pipeline.results.saver import SaverRuntime
import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
import json
from time import perf_counter_ns

from aiohttp import web
from aiohttp.test_utils import TestServer
import pytest

from perf_benchmark.core.models import ContextMode, Run, Session, Trial, Turn
from perf_benchmark.pipeline.context import (
    ExecutorContext,
    MetricsContext,
    SaverContext,
)
from perf_benchmark.pipeline.contracts.executor import AttemptPlan, ExecutorTrial
from perf_benchmark.pipeline.contracts.facts import (
    LoadProfile,
    ResourceLimits,
    TrialFact,
    TrialSchedulingFact,
)
from perf_benchmark.pipeline.contracts.scheduler import SessionAdmission, StopPlan, TrialPlan
from perf_benchmark.pipeline.execution.engine import ExecutionEngine
from perf_benchmark.pipeline.scheduling.mechanics import StandardTurnStrategy, parse_common_settings
from perf_benchmark.plugins.executor.openai_chat import OpenAIChatExecutor, validate
from perf_benchmark.plugins.metrics.standard import StandardMetrics, parse_settings as parse_metrics
from perf_benchmark.plugins.saver.sqlite import SQLiteSaver, parse_settings as parse_saver


def _trial_plan() -> TrialPlan:
    return TrialPlan(
        "attempt control",
        StopPlan(session_count=1),
        ResourceLimits(
            ready_capacity=1,
            max_pending_sessions=1,
            max_pending_requests=8,
        ),
        LoadProfile(
            mode="multi-turn-closed-loop",
            target_concurrency=1,
            max_active_sessions=1,
            max_concurrency=1,
        )
    )


class SessionScheduler(StandardTurnStrategy):
    def __init__(self, settings=None):
        parsed = parse_common_settings(settings or {}, allowed=set())
        super().__init__(parsed, _trial_plan(), perf_counter_ns(), seed=0, reuse_session_arrival=False)


@asynccontextmanager
async def execution(
    run_context,
    url,
    settings,
    policy=None,
    executor=None,
    count=2,
    mode: ContextMode = "live",
    executor_options=None,
):
    run = Run(name="attempts")
    run.prepare()
    run.start()
    ctx = run_context("scheduler", settings, run)
    trial = Trial(parameters={"load_profile": {"mode": "multi-turn-closed-loop"}}, id="trial-fixture-73")
    run.register_trial(trial)
    trial.start()
    turns = []
    for i in range(count):
        payload = {"role": "user", "content": f"question-{i}"}
        if mode == "snapshot":
            payload = {"messages": [payload]}
        turns.append(
            Turn(
                id=f"turn-{i}",
                ordinal=i,
                context_mode=mode,
                requested_output_tokens=3,
                request_payload_ref=ctx.payloads.append(json.dumps(payload).encode()),
            )
        )
    ctx.payloads.seal()
    session = Session(id="session", trial_id=trial.id, turns=turns)
    session.link()
    async with AsyncExitStack() as scope:
        scope.callback(ctx.payloads.close)
        scope.push_async_callback(ctx.events.close)
        sink = await SaverRuntime.create(
            SQLiteSaver(parse_saver({"path": "results.db"})),
            run,
            SaverContext(ctx.base_dir, ctx.cancellation),
        )
        scope.push_async_callback(sink.close)
        metrics = await MetricsRuntime.create(
            StandardMetrics(parse_metrics({"memory_limit": "64MB"})),
            MetricsContext(ctx.cancellation),
            publish_results=sink.write_results, wait_for_results=sink.wait_for_headroom,
        )
        scope.push_async_callback(metrics.close)
        collector = await metrics.open_trial(trial)
        scope.push_async_callback(collector.close)
        await sink.write_trial(trial)
        executor_settings = validate({"base_url": url, "model": "normal", **(executor_options or {})})
        plugin = (executor or OpenAIChatExecutor)(executor_settings)
        runtime = await plugin.create_runtime(
            ExecutorTrial(trial.id),
            ExecutorContext(ctx.cancellation),
        )
        trial_ctx = ctx.for_trial(run, trial, collector)
        selected_policy = (policy or SessionScheduler)(settings)
        engine = ExecutionEngine(
            run,
            runtime,
            collector,
            sink,
            trial_ctx,
            selected_policy,
        )
        scope.push_async_callback(engine.close)
        await engine.open()
        engine.open_ready_window(1)
        yield engine, session, ctx, sink, collector
        trial.finish(
            cancellation_reason=ctx.cancellation.reason if ctx.cancellation.is_cancelled() else None
        )
        gate = selected_policy.attempts
        fact = TrialFact(
            trial.id,
            trial.status,
            _trial_plan().load_profile,
            _trial_plan().resources,
            None,
            _trial_plan().repetition,
            TrialSchedulingFact(
                offered_sessions=1,
                admitted_sessions=1,
                completed_sessions=1,
                offered_attempts=gate.offered,
                admitted_attempts=gate.admitted,
                completed_attempts=gate.completed,
                demand_exhausted=True,
                termination_reason="session_count_reached",
            ),
        )
        points = (await collector.finalize(trial, fact)).points
        await sink.write_trial(trial, fact)
        await sink.write_metric_points(points)
        run.observe_trial(trial)
        run.finish()
        await sink.finalize(run)


async def send_stream(request, texts, *, disconnect=False):
    response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await response.prepare(request)
    for text in texts:
        await response.write(
            f"data: {json.dumps({'choices': [{'delta': {'content': text}}]})}\n\n".encode()
        )
    await response.write(b'data: {"usage":{"prompt_tokens":1,"completion_tokens":3}}\n\n')
    if disconnect:
        request.transport.close()
    else:
        await response.write(b"data: [DONE]\n\n")
    return response


@pytest.mark.parametrize("last_fails", [False, True])
async def test_success_resend_keeps_history_and_final_attempt_facts(last_fails, run_context):
    requests = []

    async def handler(request):
        requests.append(await request.json())
        if last_fails and len(requests) == 2:
            return web.Response(status=503)
        return await send_stream(request, [f"answer-{len(requests)}"])

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    async with TestServer(app) as server:
        async with execution(run_context, str(server.make_url("")), {
            "retry": {"model": "constant", "on_status": ["succeeded"], "probability": 1,
                      "max_attempts": 2, "delay_s": 0},
        }) as (engine, session, ctx, sink, collector):
            now = perf_counter_ns()
            await engine.run(session, SessionAdmission(now, now))
            assert requests[0] == requests[1]
            assert session.turns[0].status == ("failed" if last_fails else "succeeded")
            if last_fails:
                assert len(requests) == 2
                assert session.turns[1].status == "pending"
            else:
                assert len(requests) == 4
                assert requests[2] == requests[3]
                assert requests[2]["messages"][1]["content"] == "answer-2"
                assert "answer-1" not in json.dumps(requests[2])
            await collector.runtime.writer.flush()
            await sink.flush()
            reader = await sink.finalize(engine.domain_run)
            attempts = await asyncio.to_thread(lambda: tuple(reader.iter_attempt_records(engine.domain_run.id)))
            turns = await asyncio.to_thread(lambda: tuple(reader.iter_turn_records(engine.domain_run.id)))
            assert len(attempts) == len(requests)
            assert [(row["attempt"], row["status"]) for row in attempts[:2]] == [
                (1, "succeeded"), (2, "failed" if last_fails else "succeeded"),
            ]
            assert turns[0]["metrics"]["e2e"] == attempts[1]["metrics"]["e2e"]
            assert turns[0]["measurements"]["request"]["attempts"] == 2


@pytest.mark.parametrize("probability,threshold", [(0, 2), (1, 2), (1, {"min": 2, "max": 3})])
async def test_probability_cancel_reaches_rust_transport_and_sqlite(probability, threshold, run_context):
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        await request.read()
        return await send_stream(request, ["a", "b", "c", "d", "e"])

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    async with TestServer(app) as server:
        async with execution(run_context, str(server.make_url("")), {
            "retry": {"model": "constant", "on_status": ["failed"], "max_attempts": 3},
            "cancel": {"model": "constant", "probability": probability, "after_outputs": threshold},
        }, count=1) as (engine, session, ctx, sink, collector):
            now = perf_counter_ns()
            await engine.run(session, SessionAdmission(now, now))
            assert calls == 1
            assert session.turns[0].status == ("cancelled" if probability else "succeeded")
            await collector.runtime.writer.flush()
            await sink.flush()
            reader = await sink.finalize(engine.domain_run)
            rows = await asyncio.to_thread(lambda: tuple(reader.iter_attempt_records(engine.domain_run.id)))
            assert len(rows) == 1 and rows[0]["status"] == session.turns[0].status
            outputs = rows[0]["measurements"]["output_event_count"]
            assert outputs in (2, 3) if probability else outputs == 5


async def test_timeout_remains_a_failed_retryable_attempt(run_context):
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        await request.read()
        if calls == 1:
            await asyncio.sleep(0.3)
            return web.Response()
        return await send_stream(request, ["recovered"])

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    async with TestServer(app) as server:
        async with execution(run_context, str(server.make_url("")), {
            "retry": {"model": "constant", "on_status": ["failed"], "probability": 1,
                      "max_attempts": 2, "delay_s": 0},
        }, count=1, executor_options={"timeout_seconds": 0.1}) as (
            engine, session, ctx, sink, collector,
        ):
            now = perf_counter_ns()
            await engine.run(session, SessionAdmission(now, now))
            assert calls == 2 and session.turns[0].status == "succeeded"
            await collector.runtime.writer.flush()
            await sink.flush()
            reader = await sink.finalize(engine.domain_run)
            rows = await asyncio.to_thread(lambda: tuple(reader.iter_attempt_records(engine.domain_run.id)))
            assert [row["status"] for row in rows] == ["failed", "succeeded"]
            assert rows[0]["error"] == "TimeoutError"


@pytest.mark.parametrize("final_succeeds", [True, False])
async def test_expected_first_failure_does_not_mask_later_unexpected_failure(
    final_succeeds, run_context
):
    calls = 0

    class Policy(SessionScheduler):
        def plan_attempt(self, turn, attempt, *, session_ordinal):
            return AttemptPlan(expected_status="failed") if attempt == 1 else AttemptPlan()

    async def handler(request):
        nonlocal calls
        calls += 1
        await request.read()
        if calls == 1 or not final_succeeds:
            return web.Response(status=503)
        return await send_stream(request, ["valid"])

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    async with TestServer(app) as server:
        async with execution(
            run_context,
            str(server.make_url("")),
            {"retry": {"max_attempts": 2, "delay_s": 0}},
            Policy,
            count=1,
        ) as (engine, session, ctx, sink, collector):
            now = perf_counter_ns()
            await engine.run(session, SessionAdmission(now, now))
            assert calls == 2
            assert session.status == ("succeeded" if final_succeeds else "failed")
        assert engine.domain_run.status == session.status


@pytest.mark.parametrize("fault", ["model", "partial"])
async def test_retry_rebuilds_request_and_persists_each_attempt(fault, run_context):
    calls, requests = [], []

    class Executor(OpenAIChatExecutor):
        def build_request(self, value, settings):
            spec = super().build_request(value, settings)
            calls.append((value.turn.id, value.attempt, spec.parameters["model"]))
            if fault == "model" and value.turn.id == "turn-0" and value.attempt == 1:
                spec.parameters["model"] = "invalid"
            return spec

    class Policy(SessionScheduler):
        def retry_delay(self, turn, result, *, session_ordinal):
            if result.status == "failed" and result.attempt == 1:
                assert result.recoverable
                return 0.02
            return None

    async def handler(request):
        payload = await request.json()
        requests.append(payload)
        if payload["model"] == "invalid":
            return web.Response(status=400)
        return await send_stream(
            request,
            ["failed-prefix" if len(requests) == 1 else "valid"],
            disconnect=fault == "partial" and len(requests) == 1,
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    async with TestServer(app) as server:
        async with execution(run_context, str(server.make_url("")), {}, Policy, Executor) as (
            engine,
            session,
            ctx,
            sink,
            collector,
        ):
            now = perf_counter_ns()
            await engine.run(session, SessionAdmission(now, now))
            assert session.status == "succeeded"
            assert calls == [
                ("turn-0", 1, "normal"),
                ("turn-0", 2, "normal"),
                ("turn-1", 1, "normal"),
            ]
            assert requests[-1]["messages"][1]["content"] == "valid"
            assert "failed-prefix" not in json.dumps(requests[-1])
            await collector.runtime.writer.flush()
            await sink.flush()
            reader = await sink.finalize(engine.domain_run)
            rows = await asyncio.to_thread(lambda: tuple(reader.iter_attempt_records(engine.domain_run.id)))
            assert [(x["turn_id"], x["attempt"], x["status"]) for x in rows] == [
                ("turn-0", 1, "failed"),
                ("turn-0", 2, "succeeded"),
                ("turn-1", 1, "succeeded"),
            ]
            turns = await asyncio.to_thread(lambda: tuple(reader.iter_turn_records(engine.domain_run.id)))
            first = turns[0]
            assert first["measurements"]["request"]["attempts"] == 2
            assert first["metrics"]["turn_total_duration"]["statistics"]["value"] >= 20
            assert first["metrics"]["e2e"] == rows[1]["metrics"]["e2e"]
            if fault == "partial":
                assert rows[0]["measurements"]["output_event_count"] == 1


@pytest.mark.parametrize("action, retain", [("cancel", False), ("cancel", True), ("fail", False)])
async def test_scene_keeps_request_failure_separate_from_execution_result(
    action, retain, run_context
):
    requests, release = [], asyncio.Event()

    class Policy(SessionScheduler):
        def plan_attempt(self, turn, attempt, *, session_ordinal):
            if turn.ordinal == 0:
                return AttemptPlan(
                    after_outputs=2,
                    action=action,
                    expected_status="failed" if action == "fail" else "cancelled",
                )
            return AttemptPlan()

    async def handler(request):
        requests.append(await request.json())
        if len(requests) > 1:
            return await send_stream(request, ["done"])
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(
            b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
            b'data: {"usage":{"completion_tokens":9}}\n\n'
            b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"b"}}]}\n\n'
        )
        await release.wait()
        return response

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    async with TestServer(app) as server:
        try:
            async with execution(
                run_context,
                str(server.make_url("")),
                {"continue_on_cancel": True, "retain_partial": retain},
                Policy,
            ) as (engine, session, ctx, sink, collector):
                now = perf_counter_ns()
                await asyncio.wait_for(
                    engine.run(session, SessionAdmission(now, now)), 3
                )
                assert session.status == "succeeded"
                assert session.turns[0].scenario_passed is True
                assert session.turns[0].status == ("failed" if action == "fail" else "cancelled")
                if action == "cancel":
                    history = requests[1]["messages"]
                    assert [x["content"] for x in history] == (
                        ["question-0", "ab", "question-1"]
                        if retain
                        else ["question-0", "question-1"]
                    )
                else:
                    assert len(requests) == 1
                await collector.runtime.writer.flush()
                reader = await sink.finalize(engine.domain_run)
                rows = await asyncio.to_thread(lambda: tuple(reader.iter_attempt_records(engine.domain_run.id)))
                first = rows[0]
                assert first["error_origin"] == "scenario"
                assert first["measurements"]["output_event_count"] == 2
                assert first["trigger_ns"] <= first["decision_ns"] <= first["applied_ns"]
                assert first["plan"]["after_outputs"] == 2
        finally:
            release.set()


@pytest.mark.parametrize(
    "control,actual_status",
    [
        ({"after_outputs": 3, "expected_status": "cancelled"}, "succeeded"),
        ({"after_outputs": 1, "expected_status": "failed"}, "cancelled"),
    ],
)
async def test_unmet_scene_fails_run_without_rewriting_request(control, actual_status, run_context):
    async def handler(request):
        await request.read()
        return await send_stream(request, ["done"])

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    async with TestServer(app) as server:
        async with execution(
            run_context,
            str(server.make_url("")),
            {"control": control},
            count=1,
        ) as (engine, session, ctx, sink, collector):
            now = perf_counter_ns()
            await engine.run(session, SessionAdmission(now, now))
            assert session.status == "failed"
            assert session.turns[0].status == actual_status
            assert session.turns[0].scenario_passed is False
        assert engine.domain_run.status == "failed"
