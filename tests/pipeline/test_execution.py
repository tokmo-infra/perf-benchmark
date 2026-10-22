import asyncio
from time import perf_counter_ns

import pytest

from perf_benchmark.core.models import Run, Session, Turn
from perf_benchmark.pipeline.contracts.executor import AttemptResult, TurnRequestFacts
from perf_benchmark.pipeline.contracts.facts import LoadProfile, ResourceLimits
from perf_benchmark.pipeline.contracts.scheduler import SessionAdmission, StopPlan, TrialPlan
from perf_benchmark.pipeline.execution.engine import ExecutionEngine
from perf_benchmark.pipeline.preparation.payloads import PayloadRef
from perf_benchmark.pipeline.scheduling.mechanics import StandardTurnStrategy, parse_common_settings


def _strategy(*, concurrency=8, pending=64, settings=None):
    plan = TrialPlan(
        "execution",
        StopPlan(session_count=100),
        ResourceLimits(concurrency, pending, pending),
        LoadProfile(
            "multi-turn-closed-loop",
            target_concurrency=concurrency,
            max_active_sessions=concurrency,
            max_concurrency=concurrency,
        ),
    )
    return StandardTurnStrategy(
        parse_common_settings({
            "retry": {"max_attempts": 1, "delay_s": 0.2},
            "control": {},
            "continue_on_cancel": False,
            "retain_partial": False,
            **(settings or {}),
        }, allowed=set()),
        plan,
        perf_counter_ns(), seed=0, reuse_session_arrival=False,
    )


def _admission():
    now = perf_counter_ns()
    return SessionAdmission(now, now)


class Metrics:
    def __init__(self):
        self.completed = []

    def record_attempt(self, turn, result):
        pass

    def record_turn(self, turn, trace=None):
        self.completed.append(turn.id)

    def record_session(self, session):
        pass

    def record_occupancy(self, kind, at_ns, value):
        pass

    async def wait_for_headroom(self):
        pass


class Sink:
    def __init__(self):
        self.sessions = []

    async def write_session(self, session):
        self.sessions.append((session.id, session.status))


def _session(*ids):
    turns = [
        Turn(id=identifier, ordinal=index, request_payload_ref=PayloadRef(index, 1))
        for index, identifier in enumerate(ids)
    ]
    session = Session(id="session", turns=turns)
    session.link()
    return session


async def test_execution_engine_owns_runtime_and_domain_lifecycle(trial_context):
    calls = []
    timestamps = {}

    class Runtime:
        async def release(self, prepared):
            timestamps["release_started"] = perf_counter_ns()
            await asyncio.sleep(.01)
            timestamps["release_finished"] = perf_counter_ns()

        async def close_session(self, session_id):
            calls.append(f"close-{session_id}")

        async def prepare(self):
            calls.append("prepare")

        async def prepare_attempt(self, value):
            assert value.live_history == {}
            calls.append(f"prepare-{value.turn.id}")
            return value

        async def execute_attempt(self, prepared, signal):
            calls.append(prepared.turn.id)
            result = AttemptResult(
                prepared.turn.id,
                prepared.attempt,
                continuation_payload=b'{"role":"assistant","content":"ok"}',
                request_facts=TurnRequestFacts(7, {"execution_fingerprint": "abc"}),
            )
            result.request.started_ns = perf_counter_ns()
            result.request.response_completed_ns = perf_counter_ns()
            timestamps["response_completed"] = result.request.response_completed_ns
            await asyncio.sleep(.01)  # decoding, outside the transport's E2E boundary
            timestamps["executor_return"] = perf_counter_ns()
            return result

        async def close(self):
            calls.append("close")

    run = Run(name="engine")
    ctx = trial_context("scheduler", {}, run)
    trial = ctx.trial
    trial.start()
    metrics, sink = Metrics(), Sink()
    engine = ExecutionEngine(run, Runtime(), metrics, sink, ctx, _strategy(concurrency=1))
    session = _session("turn")

    assert await engine.open()
    engine.open_ready_window(1)
    await engine.run(session, _admission())
    await engine.close()

    assert calls == ["prepare", "prepare-turn", "turn", "close-session", "close"]
    assert metrics.completed == ["turn"]
    assert session.status == "succeeded"
    assert trial.succeeded_turns == 1
    assert session.turns[0].requested_output_tokens == 7
    assert session.turns[0].metadata["execution_fingerprint"] == "abc"
    request = session.turns[0].measurements.request
    assert request.response_completed_ns == timestamps["response_completed"]
    assert request.response_completed_ns < timestamps["executor_return"] <= request.result_ready_ns
    assert request.result_ready_ns <= timestamps["release_started"] < timestamps["release_finished"]


async def test_result_releases_concurrency_before_body_budget(trial_context):
    settled = []
    started = asyncio.Queue()

    class Runtime:
        async def prepare(self):
            pass

        async def prepare_attempt(self, value):
            return value

        async def execute_attempt(self, prepared, signal):
            started.put_nowait(prepared.session.id)
            return AttemptResult(prepared.turn.id, prepared.attempt)

        async def release(self, prepared):
            future = asyncio.get_running_loop().create_future()
            settled.append(future)
            return future

        async def close_session(self, session_id):
            pass

        async def close(self):
            pass

    ctx = trial_context("scheduler")
    ctx.trial.start()
    engine = ExecutionEngine(
        ctx.domain_run, Runtime(), Metrics(), Sink(), ctx,
        _strategy(concurrency=1), resource_concurrency=1,
    )
    await engine.open()
    engine.open_ready_window(1)
    sessions = [_session(f"turn-{n}") for n in range(3)]
    for index, session in enumerate(sessions):
        session.id = f"session-{index}"
        session.link()
    try:
        first = asyncio.create_task(engine.run(sessions[0], _admission()))
        assert await asyncio.wait_for(started.get(), 2) == "session-0"
        await first
        second = asyncio.create_task(engine.run(sessions[1], _admission()))
        assert await asyncio.wait_for(started.get(), 2) == "session-1"
        await second
        third = asyncio.create_task(engine.run(sessions[2], _admission()))
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(started.get(), .02)
        settled[0].set_result(None)
        assert await asyncio.wait_for(started.get(), 2) == "session-2"
        await third
    finally:
        for future in settled:
            if not future.done():
                future.set_result(None)
        await engine.close()


async def test_background_body_cleanup_failure_fails_trial(trial_context):
    settled = asyncio.get_running_loop().create_future()

    class Runtime:
        async def prepare(self):
            pass

        async def prepare_attempt(self, value):
            return value

        async def execute_attempt(self, prepared, signal):
            return AttemptResult(prepared.turn.id, prepared.attempt)

        async def release(self, prepared):
            return settled

        async def close_session(self, session_id):
            pass

        async def close(self):
            pass

    ctx = trial_context("scheduler")
    ctx.trial.start()
    engine = ExecutionEngine(
        ctx.domain_run, Runtime(), Metrics(), Sink(), ctx,
        _strategy(concurrency=1), resource_concurrency=1,
    )
    await engine.open()
    engine.open_ready_window(1)
    await engine.run(_session("turn"), _admission())
    settled.set_exception(RuntimeError("body cleanup failed"))
    await asyncio.sleep(0)
    assert ctx.cancellation.is_cancelled()
    with pytest.raises(RuntimeError, match="body cleanup failed"):
        await engine.close()


async def test_executor_receives_frozen_live_history_snapshots(trial_context):
    observed = []

    class Runtime:
        async def release(self, prepared):
            pass

        async def close_session(self, session_id):
            pass

        async def prepare(self):
            pass

        async def prepare_attempt(self, value):
            observed.append(value.live_history)
            return value

        async def execute_attempt(self, prepared, signal):
            continuation = (
                b'{"role":"assistant","content":"answer"}'
                if prepared.turn.ordinal == 0
                else None
            )
            return AttemptResult(
                prepared.turn.id,
                prepared.attempt,
                continuation_payload=continuation,
            )

        async def close(self):
            pass

    ctx = trial_context("scheduler")
    engine = ExecutionEngine(
        ctx.domain_run, Runtime(), Metrics(), Sink(), ctx, _strategy(concurrency=1)
    )
    session = _session("first", "second")
    await engine.open()
    engine.open_ready_window(1)
    await engine.run(session, _admission())
    await engine.close()

    assert observed[0] == {}
    assert observed[1]["first"].payload == b'{"role":"assistant","content":"answer"}'


async def test_execution_engine_skips_prepare_when_already_cancelled(trial_context):
    calls = []

    class Runtime:
        async def prepare(self):
            calls.append("prepare")

        async def close(self):
            calls.append("close")

    ctx = trial_context("scheduler")
    ctx.cancellation.cancel("stop")
    engine = ExecutionEngine(
        ctx.domain_run, Runtime(), Metrics(), Sink(), ctx, _strategy(concurrency=1)
    )

    assert not await engine.open()
    await engine.close()
    assert calls == ["close"]


@pytest.mark.parametrize("initially_enabled", [False, True])
async def test_execution_engine_suspends_gc_only_for_the_execution_window(
    trial_context, monkeypatch, initially_enabled
):
    calls = []

    class Runtime:
        async def prepare(self):
            calls.append("prepare")

        async def close(self):
            calls.append("close")

    monkeypatch.setattr("perf_benchmark.pipeline.execution.engine.gc.isenabled", lambda: initially_enabled)
    monkeypatch.setattr("perf_benchmark.pipeline.execution.engine.gc.disable", lambda: calls.append("disable"))
    monkeypatch.setattr("perf_benchmark.pipeline.execution.engine.gc.enable", lambda: calls.append("enable"))
    ctx = trial_context("scheduler")
    engine = ExecutionEngine(
        ctx.domain_run, Runtime(), Metrics(), Sink(), ctx, _strategy(concurrency=1)
    )

    assert await engine.open()
    await engine.close()

    assert calls == ["prepare", "disable", "close", *(["enable"] if initially_enabled else [])]


async def test_execution_engine_bounds_ready_requests_to_concurrency(trial_context):
    outstanding = maximum = 0
    release_execution = asyncio.Event()

    class Runtime:
        async def prepare(self):
            pass

        async def close(self):
            pass

        async def release(self, prepared):
            nonlocal outstanding
            outstanding -= 1

        async def close_session(self, session_id):
            pass

        async def prepare_attempt(self, value):
            nonlocal outstanding, maximum
            outstanding += 1
            maximum = max(maximum, outstanding)
            return value

        async def execute_attempt(self, prepared, signal):
            await release_execution.wait()
            return AttemptResult(prepared.turn.id, prepared.attempt)

    ctx = trial_context("scheduler")
    engine = ExecutionEngine(
        ctx.domain_run,
        Runtime(),
        Metrics(),
        Sink(),
        ctx,
        _strategy(concurrency=2, pending=8),
    )
    sessions = [
        Session(
            id=f"session-{index}",
            turns=[
                Turn(
                    id=f"turn-{index}",
                    request_payload_ref=PayloadRef(index, 1),
                )
            ],
        )
        for index in range(5)
    ]
    await engine.open()
    engine.open_ready_window(2)

    tasks = [asyncio.create_task(engine.run(session, _admission())) for session in sessions]
    for _ in range(100):
        if maximum == 4:
            break
        await asyncio.sleep(0.001)
    assert maximum == 4
    # Two executing Attempts plus two ready bodies must not count queued preparation as in-flight.
    assert engine.diagnostics()["in_flight"] == 2
    assert engine.diagnostics()["request_pending"] == 3
    release_execution.set()
    await asyncio.gather(*tasks)
    await engine.close()


@pytest.mark.parametrize("phase", ["prepare", "admission", "active", "retry"])
async def test_cancellation_closes_each_attempt_phase(phase, trial_context):
    entered = asyncio.Event()
    released = []

    class Runtime:
        async def prepare(self):
            pass

        async def prepare_attempt(self, value):
            if phase == "prepare":
                entered.set()
                await asyncio.Event().wait()
            return value

        async def execute_attempt(self, prepared, signal):
            if phase == "active":
                entered.set()
                await asyncio.Event().wait()
            return AttemptResult(
                prepared.turn.id,
                prepared.attempt,
                status="failed",
                error="transient",
                retryable=True,
            )

        async def release(self, prepared):
            if prepared not in released:
                released.append(prepared)

        async def close_session(self, session_id):
            pass

        async def close(self):
            pass

    strategy = _strategy(
        concurrency=1,
        settings={"retry": {"max_attempts": 2, "delay_s": 60}},
    )
    if phase == "admission":
        original_admit = strategy.attempts.admit

        async def admit(ticket):
            entered.set()
            await asyncio.Event().wait()
            return await original_admit(ticket)

        strategy.attempts.admit = admit
    elif phase == "retry":

        async def retry(turn, control):
            await control.send()
            entered.set()
            return await control.retry(60)

        strategy.run_turn = retry

    ctx = trial_context("scheduler")
    metrics = Metrics()
    engine = ExecutionEngine(ctx.domain_run, Runtime(), metrics, Sink(), ctx, strategy)
    session = _session("turn-0", "turn-1")
    await engine.open()
    engine.open_ready_window(1)
    task = asyncio.create_task(engine.run(session, _admission()))
    await asyncio.wait_for(entered.wait(), 1)
    await engine.cancel(session, "stop")
    await asyncio.wait_for(task, 1)

    assert session.turns[1].status == "pending"
    assert engine.occupancy["in_flight"] == engine.occupancy["active_sessions"] == 0
    assert metrics.completed == ["turn-0"]
    await engine.close()


@pytest.mark.parametrize("outcome", ["release", "cancel", "failure"])
async def test_session_waits_for_result_headroom_before_producing_facts(trial_context, outcome):
    entered, available = asyncio.Event(), asyncio.Event()
    recorded = []

    class WaitingMetrics(Metrics):
        async def wait_for_headroom(self):
            entered.set()
            await available.wait()
            if outcome == "failure":
                raise RuntimeError("result writer failed")

        def record_occupancy(self, *args):
            recorded.append(args)

    class Runtime:
        async def prepare(self):
            pass

        async def prepare_attempt(self, value):
            return value

        async def execute_attempt(self, prepared, signal):
            return AttemptResult(prepared.turn.id, prepared.attempt)

        async def release(self, prepared):
            pass

        async def close_session(self, identifier):
            pass

        async def close(self):
            pass

    ctx = trial_context("scheduler")
    sink = Sink()
    engine = ExecutionEngine(ctx.domain_run, Runtime(), WaitingMetrics(), sink, ctx,
                             _strategy(concurrency=1), resource_concurrency=1)
    session = _session("turn")
    await engine.open()
    engine.open_ready_window(1)
    task = asyncio.create_task(engine.run(session, _admission()))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        assert session.status == "pending"
        assert not sink.sessions and not recorded
        if outcome == "cancel":
            ctx.cancellation.cancel("cancel while waiting for results")
        else:
            available.set()
        if outcome == "failure":
            with pytest.raises(RuntimeError, match="result writer failed"):
                await task
        else:
            await asyncio.wait_for(task, 1)
        if outcome != "release":
            assert session.status == "pending"
            assert not sink.sessions and not recorded
        else:
            assert session.status == "succeeded"
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await engine.close()
