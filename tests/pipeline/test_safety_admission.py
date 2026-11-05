"""CPU loopback/native safety race through the real Coordinator and writers."""
import asyncio
from collections import Counter
from dataclasses import replace
import json

from aiohttp import web
from aiohttp.test_utils import TestServer
import pytest

from perf_benchmark import _native
from perf_benchmark.pipeline.coordinator import Coordinator
from perf_benchmark.pipeline.execution.engine import ExecutionEngine
from perf_benchmark.pipeline.results.measurement import AttemptFact
from perf_benchmark.pipeline.results.metrics import MetricsCollector
from perf_benchmark.plugins.saver.sqlite import SQLiteResultReader
from perf_benchmark.runtime.cancellation import CancellationToken
from perf_benchmark.runtime.http.rust_executor import RustHttpExecutorRuntime
from tests.runtime.native_support import command


@pytest.mark.parametrize("header_name", [None, "x-request-id", "X-ReQuEsT-Id"])
async def test_safety_admission_race_keeps_real_and_unsent_facts(
    tmp_path, tokenizer_path, monkeypatch, header_name,
):
    monkeypatch.setenv("PERF_TTFT_LIMIT_MS", "500")
    monkeypatch.setenv("PERF_CONTROL_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_BENCHMARK_PREPARATION_WORKERS", "1")
    token = CancellationToken()
    overdue_buffered, release_overdue, server_release = (asyncio.Event() for _ in range(3))
    rejected, requests, identities, runtimes, workers, accepted = [], [], {}, [], [], []
    original_prepare = RustHttpExecutorRuntime.prepare
    original_prepare_attempt = RustHttpExecutorRuntime.prepare_attempt
    original_worker_stopped = ExecutionEngine.worker_stopped
    original_submit = MetricsCollector._submit

    class DelayedAttempt:
        """Delay only Python consumption, never replace transport or outcomes."""
        def __init__(self, handle):
            self.handle = handle

        def __getattr__(self, name):
            return getattr(self.handle, name)

        async def result(self):
            value = await self.handle.result()
            assert value.error_origin == "safety" and value.transport_sent_ns is not None
            overdue_buffered.set()
            await release_overdue.wait()
            assert token.is_cancelled()  # Rejection wins before the real overdue is consumed.
            return value

    class DeliveryBarrier:
        def __init__(self, native):
            self.native = native
            self.overdue_session = None

        def __getattr__(self, name):
            return getattr(self.native, name)

        def submit(self, body, spec, *, cancelled):
            try:
                handle = self.native.submit(body, spec, cancelled=cancelled)
            except _native.SafetyAdmissionRejected:
                assert overdue_buffered.is_set() and not token.is_cancelled()
                with pytest.raises(ValueError):
                    self.native.submit(body, {**spec, "body_length": -1})
                with pytest.raises(TypeError):
                    self.native.submit(body, {**spec, "body_length": None})
                rejected.append(spec)
                release_overdue.set()
                raise
            return DelayedAttempt(handle) if spec["session_id"] == self.overdue_session else handle

    async def prepare(runtime):
        await original_prepare(runtime)
        runtime.native = DeliveryBarrier(runtime.native)
        runtimes.append(runtime)

    async def prepare_attempt(runtime, value):
        ordinal = value.session.ordinal
        identities[ordinal] = value.turn.id
        if ordinal == 1:
            runtime.native.overdue_session = value.session.id
        if ordinal == 2:
            await asyncio.wait_for(overdue_buffered.wait(), 5)
            assert runtime.native.closed and not token.is_cancelled()
        return await original_prepare_attempt(runtime, value)

    async def worker_stopped(engine, index):
        workers.append(index)
        await original_worker_stopped(engine, index)

    def submit_fact(collector, fact):
        original_submit(collector, fact)
        if isinstance(fact, AttemptFact):
            accepted.append(fact)

    monkeypatch.setattr(MetricsCollector, "_submit", submit_fact)
    monkeypatch.setattr(RustHttpExecutorRuntime, "prepare", prepare)
    monkeypatch.setattr(RustHttpExecutorRuntime, "prepare_attempt", prepare_attempt)
    monkeypatch.setattr(ExecutionEngine, "worker_stopped", worker_stopped)

    async def handle(request):
        body = await request.json()
        content = body["messages"][-1]["content"]
        requests.append((content, request.headers.getall("X-Request-ID", [])))
        if content == "overdue":
            await server_release.wait()  # No headers; the native sent clock must fire.
            return web.Response(status=500)
        assert content == "success"
        data = (
            'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}\n\n'
            'data: [DONE]\n\n'
        )
        return web.Response(text=data, content_type="text/event-stream")

    source = tmp_path / "source.jsonl"
    source.write_text("\n".join(json.dumps({
        "id": str(i), "token_name": str(i), "created": 1700000000 + i,
        "messages": [{"role": "user", "content": content}],
    }) for i, content in enumerate(("success", "overdue", "unsent"))))
    app = web.Application()
    app.router.add_post("/v1/chat/completions", handle)
    async with TestServer(app) as server:
        settings = {"model": "cpu-fixture", "base_url": str(server.make_url("/")),
                    "stream": True, "timeout_seconds": None}
        if header_name is not None:
            settings["extra_headers"] = {header_name: "caller-correlation"}
        config = {"pipeline": {
            "loader": {"plugin": "hf-datasets", "settings": {
                "dataset": "json", "split": "train", "data_files": [str(source)],
                "cache_dir": str(tmp_path / "cache")}},
            "normalization": {"plugin": "openai-messages", "settings": {
                "tokenizer_path": str(tokenizer_path)}},
            "selector": {"plugin": "all"},
            "executor": {"plugin": "openai-chat", "settings": settings},
            "scheduler": {"plugin": "multi-turn-closed-loop", "settings": {
                "max_active_sessions": 3, "max_concurrency": 3, "stop_on_underfill": True}},
            "metrics": {"plugin": "standard", "settings": {"maxtpm": True, "maxtpm_start_percent": 40}},
            "saver": {"plugin": "sqlite", "settings": {"path": str(tmp_path / "run.sqlite")}},
            "exporter": {"plugins": []},
        }}
        try:
            run = await asyncio.wait_for(
                Coordinator().execute(config, base_dir=tmp_path, cancellation=token), 15)
        finally:
            release_overdue.set()
            server_release.set()
    assert run.status == "cancelled" and token.reason == "ttft_watchdog_safety_abort"
    assert len(rejected) == 1 and sorted(workers) == [0, 1, 2]
    assert len(requests) == 2 and {item[0] for item in requests} == {"success", "overdue"}
    for content, headers in requests:
        ordinal = 0 if content == "success" else 1
        expected = "caller-correlation" if header_name else f"{identities[ordinal]}:a1"
        assert headers == [expected]
    for runtime in runtimes:
        assert not runtime._prepared and not runtime._background
        assert runtime.native.active_count == 0 and runtime.native.finished
        body = _native.PreparedBody((), 0)
        try:
            with pytest.raises(RuntimeError):
                runtime.native.submit(body, command(), cancelled=False)
        finally:
            body.release()
    reader = SQLiteResultReader(tmp_path / "run.sqlite")
    saved_run = reader.run_record(run.id)
    assert saved_run.status == "cancelled"
    assert saved_run.counts == run.counts
    for key, expected in {"trials": 1, "sessions": 3, "turns": 3,
                          "succeeded_turns": 1, "failed_turns": 0, "cancelled_turns": 2}.items():
        assert run.counts[key] == expected
    trial, = reader.trials(run.id)
    assert trial["status"] == trial["execution_status"] == "cancelled"
    assert trial["scheduling"]["admitted_attempts"] == trial["scheduling"]["completed_attempts"] == 3
    assert trial["scheduling"]["admitted_sessions"] == trial["scheduling"]["completed_sessions"] == 3
    attempts = {row["turn_id"]: row for row in reader.iter_attempt_records(run.id)}
    turns = tuple(reader.iter_turn_records(run.id))
    assert len(accepted) == len(attempts) == len(turns) == 3
    accepted_by_id = {fact.id: fact for fact in accepted}
    assert Counter(row["status"] for row in attempts.values()) == {"succeeded": 1, "failed": 1, "cancelled": 1}
    assert Counter(row["status"] for row in turns) == {"succeeded": 1, "cancelled": 2}
    overdue = attempts[identities[1]]
    assert overdue["error"] == "TTFTWatchdogExceeded"
    assert overdue["measurements"]["started_ns"] is not None
    assert overdue["measurements"]["first_output_ns"] is None
    assert accepted_by_id[identities[1]].metadata["_perf_transport_sent_ns"] is not None
    unsent = attempts[identities[2]]
    assert unsent["error"] == "native_safety_admission_rejected"
    assert accepted_by_id[identities[2]].metadata["_perf_transport_timing_contract"] == "not-dispatched-v1"
    assert accepted_by_id[identities[2]].metadata["_perf_transport_sent_ns"] is None
    assert unsent["usage"] == {}
    for name in ("started_ns", "response_completed_ns", "first_output_ns", "last_output_ns", "http_status"):
        assert unsent["measurements"][name] is None
    diagnostics = reader.measurements(run.id, scope="trial", scope_id=trial["trial_id"])
    assert diagnostics["metric_counts"] == {"offered": 3, "admitted": 2, "succeeded": 1, "failed": 0, "cancelled": 2}
    assert diagnostics["diagnostics"]["attempts_observed"] == 3
    assert diagnostics["diagnostics"]["http_status_counts_complete"] is True
    notice = json.loads((tmp_path / "safety-stop.json").read_text())
    proof = notice["request"]
    assert proof["turn_id"] == identities[1] and proof["sent_ns"] == accepted_by_id[identities[1]].metadata["_perf_transport_sent_ns"]
    assert proof["request_id"] == ("caller-correlation" if header_name else f"{identities[1]}:a1")
    assert proof["first_token_ns"] is None and proof["first_token_pending"]
    assert not (tmp_path / "native-stop.json").exists()


async def test_explicit_native_close_is_not_safety_rejection():
    trial = _native.create_trial("openai-chat")
    await trial.close()
    body = _native.PreparedBody((), 0)
    try:
        with pytest.raises(RuntimeError) as failure:
            trial.submit(body, command())
        assert not isinstance(failure.value, _native.SafetyAdmissionRejected)
    finally:
        body.release()


async def test_generated_request_ids_are_unique_for_each_attempt(trial_context):
    from perf_benchmark.core.models import Session
    from perf_benchmark.pipeline.contracts.executor import AttemptPlan
    from perf_benchmark.runtime.attempt import AttemptSignal
    from tests.executor.test_openai_chat import _attempt_input, _runtime, _turn

    ids = []

    async def respond(request):
        await request.read()
        ids.append(request.headers.getall("X-Request-ID"))
        return web.json_response({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", respond)
    ctx = trial_context("executor")
    turn = await _turn(ctx, "unique-turn", "fixture")
    session = Session(id="fixture-session", turns=[turn])
    ctx.payloads.seal()
    async with TestServer(app) as server:
        runtime = _runtime({"model": "fixture", "base_url": str(server.make_url("/")), "stream": False}, ctx)
        try:
            await runtime.prepare()
            for attempt in (1, 2):
                value = _attempt_input(session, turn, ctx, attempt=attempt)
                prepared = await runtime.prepare_attempt(value)
                result = await runtime.execute_attempt(prepared, AttemptSignal(AttemptPlan()))
                assert result.status == "succeeded"
                assert "X-Request-ID" not in prepared.spec.headers
            assert ids == [["unique-turn:a1"], ["unique-turn:a2"]]
        finally:
            await runtime.close()
            await ctx.events.close()


@pytest.mark.parametrize("error", ["closed", "body_length"])
async def test_runtime_keeps_native_programming_errors_visible(trial_context, error):
    from perf_benchmark.pipeline.contracts.executor import AttemptPlan
    from perf_benchmark.runtime.attempt import AttemptSignal
    from tests.runtime.test_request import setup_request

    ctx, runtime, value = await setup_request(trial_context)
    try:
        await runtime.prepare()
        prepared = await runtime.prepare_attempt(value)
        if error == "closed":
            await runtime.native.close()
            expected = RuntimeError
        else:
            prepared.body = replace(prepared.body, content_length=prepared.body.content_length + 1)
            expected = ValueError
        with pytest.raises(expected) as failure:
            await runtime.execute_attempt(prepared, AttemptSignal(AttemptPlan()))
        assert not isinstance(failure.value, _native.SafetyAdmissionRejected)
        assert prepared.native_attempt is None and prepared.release_done
        assert ctx.cancellation.reason == "native execution failure"
    finally:
        await runtime.close()
        await ctx.events.close()
