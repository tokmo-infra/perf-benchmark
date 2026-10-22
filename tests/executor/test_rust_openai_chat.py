"""The production OpenAI executor uses an in-process Rust extension and real HTTP."""
import asyncio
import json
from types import SimpleNamespace
from time import perf_counter_ns

from aiohttp import web
from aiohttp.test_utils import TestServer
import pytest

from perf_benchmark.core.models import Session, Turn
from perf_benchmark.pipeline.contracts.executor import AttemptInput, AttemptPlan, ExecutorTrial
from perf_benchmark.plugins.executor.openai_chat import OpenAIChatExecutor, validate
from perf_benchmark.runtime.attempt import AttemptSignal
from perf_benchmark.runtime.http.request import PreparedBody
from perf_benchmark import _native
from tests.runtime.native_support import request as _request


@pytest.mark.parametrize("stream", [True, False])
async def test_rust_extension_runs_real_http_and_preserves_native_timestamps(trial_context, stream):
    received = []
    async def complete(request):
        received.append(await request.json())
        if not stream:
            return web.json_response({"choices": [{"message": {"content": "你好"}}],
                                      "usage": {"completion_tokens": 2}})
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b': keepalive\r\n\r\n')
        await asyncio.sleep(.01)
        await response.write('data: {"choices":[{"delta":{"content":"你好"}}]}\r\n\r\n'.encode())
        await response.write(b'data: {"usage":{"completion_tokens":2}}\n\ndata: [DONE]\n\n')
        return response
    app = web.Application()
    app.router.add_post('/v1/chat/completions', complete)
    async with TestServer(app) as server:
        ctx = trial_context("executor")
        payload = ctx.payloads.append(b'{"messages":[{"role":"user","content":"hi"}]}')
        ctx.payloads.seal()
        turn = Turn(id="turn", request_payload_ref=payload, requested_output_tokens=2)
        session = Session(id="session", turns=[turn])
        plugin = OpenAIChatExecutor(validate({"base_url": str(server.make_url("")), "model": "test", "stream": stream}))
        runtime = await plugin.create_runtime(ExecutorTrial("trial"), ctx)
        before = perf_counter_ns()
        try:
            await runtime.prepare()
            prepared = await runtime.prepare_attempt(AttemptInput.capture(session, turn, {}, attempt=1, payloads=ctx.payloads))
            result = await runtime.execute_attempt(prepared, AttemptSignal(AttemptPlan()))
        finally:
            await runtime.close()
            await ctx.events.close()
        after = perf_counter_ns()
    assert result.status == "succeeded", result
    assert result.request.started_ns is not None and result.request.response_completed_ns is not None
    assert before <= result.request.started_ns <= result.request.response_completed_ns <= after
    assert result.request.output_chars == 2
    assert result.usage == {"completion_tokens": 2}
    assert result.continuation_payload is None
    assert result.request.output_event_count == (1 if stream else 0)
    assert received[0]["messages"] == [{"role": "user", "content": "hi"}]
    assert runtime.native.closed
    assert runtime.native.active_count == 0
    if stream:
        # Native storage survives Trial/body cleanup and exposes no mutation API.
        samples = result.trace.samples
        assert samples[0] == samples[-1]
        assert samples[:] == (samples[0],)
        assert samples[::-1] == samples[:]
        with pytest.raises(TypeError):
            samples[0] = (9, 9, 9)
        with pytest.raises(IndexError):
            samples[len(samples)]
        with pytest.raises(IndexError):
            samples[-len(samples) - 1]
        with pytest.raises(ValueError):
            samples[::0]


@pytest.mark.parametrize("mode", ["repeat_cancel", "trial_close"])
async def test_pending_attempt_settles_before_releasing_body(trial_context, mode):
    entered, release = asyncio.Event(), asyncio.Event()

    async def hold(request):
        await request.read()
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b'data: {"choices":[{"delta":{"content":"prefix"}}]}\n\n')
        entered.set()
        await release.wait()
        return response

    app = web.Application()
    app.router.add_post('/v1/chat/completions', hold)
    async with TestServer(app) as server:
        ctx = trial_context("executor")
        payload = ctx.payloads.append(b'{"messages":[{"role":"user","content":"hi"}]}')
        ctx.payloads.seal()
        turn = Turn(id="turn", request_payload_ref=payload, requested_output_tokens=2)
        session = Session(id="session", turns=[turn])
        plugin = OpenAIChatExecutor(validate({"base_url": str(server.make_url("")), "model": "test"}))
        runtime = await plugin.create_runtime(ExecutorTrial("trial"), ctx)
        await runtime.prepare()
        prepared = await runtime.prepare_attempt(AttemptInput.capture(session, turn, {}, attempt=1, payloads=ctx.payloads))
        assert isinstance(prepared.body, PreparedBody)
        task = asyncio.create_task(runtime.execute_attempt(prepared, AttemptSignal(AttemptPlan())))
        await asyncio.wait_for(entered.wait(), 3)
        try:
            if mode == "repeat_cancel":
                task.cancel()
                await asyncio.sleep(0)
                task.cancel()
            else:
                await runtime.close()
            result = await asyncio.wait_for(task, 3)
            assert result.status == "cancelled"
            assert result.request.output_chars in (0, len("prefix"))
            await runtime.close()
            assert runtime.native.active_count == 0
            with pytest.raises(ValueError, match="released memoryview"):
                bytes(prepared.body.segments[1])
            ctx.payloads.close()
        finally:
            release.set()
            await ctx.events.close()


@pytest.mark.parametrize('server_default',[False,True])
async def test_null_output_cap_survives_native_http(trial_context,server_default):
    received = []
    async def complete(request):
        received.append(await request.json())
        return web.json_response({'choices':[{'message':{'content':'ok'},'finish_reason':'stop'}],
            'usage':{'prompt_tokens':1,'completion_tokens':8192 if server_default else 2}})
    app = web.Application()
    app.router.add_post('/v1/chat/completions',complete)
    async with TestServer(app) as http:
        ctx = trial_context('executor')
        ref = ctx.payloads.append(b'{"messages":[{"role":"user","content":"hi"}]}')
        ctx.payloads.seal()
        turn = Turn(id='null-cap',request_payload_ref=ref,metadata={'dataset_output_policy':'natural'})
        session = Session(id='null-cap',turns=[turn])
        settings = {'base_url':str(http.make_url('')),'model':'test','stream':False}
        if server_default:
            settings['max_tokens'] = None
        plugin = OpenAIChatExecutor(validate(settings))
        runtime = await plugin.create_runtime(ExecutorTrial('trial'),ctx)
        try:
            await runtime.prepare()
            prepared = await runtime.prepare_attempt(AttemptInput.capture(session,turn,{},attempt=1,payloads=ctx.payloads))
            assert prepared.request_facts.requested_output_tokens == (None if server_default else 4096)
            result = await runtime.execute_attempt(prepared,AttemptSignal(AttemptPlan()))
        finally:
            await runtime.close()
            await ctx.events.close()
            ctx.payloads.close()
    assert result.status == 'succeeded',result
    assert ('max_tokens' not in received[0]) if server_default else received[0]['max_tokens'] == 4096
    assert result.usage['completion_tokens'] == (8192 if server_default else 2)
    assert result.request.finish_reason == 'stop'




@pytest.mark.parametrize("scope,expected", [("session", 2), ("request", 6), ("trial", 1)])
async def test_connection_scope_across_turns_retries_and_sessions(trial_context, scope, expected):
    connections = []

    async def complete(request):
        await request.read()
        connections.append(request.transport)
        return web.json_response({"choices": [{"message": {"content": "ok"}}]})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", complete)
    async with TestServer(app) as server:
        ctx = trial_context("executor")
        payload = ctx.payloads.append(b'{"messages":[{"role":"user","content":"hi"}]}')
        ctx.payloads.seal()
        plugin = OpenAIChatExecutor(validate({
            "base_url": str(server.make_url("")), "model": "test",
            "stream": False, "connection_scope": scope,
        }))
        runtime = await plugin.create_runtime(ExecutorTrial("trial"), ctx)
        try:
            await runtime.prepare()
            for session_id in ("one", "two"):
                turns = [Turn(id=f"{session_id}-{n}", ordinal=n, request_payload_ref=payload, requested_output_tokens=2) for n in range(2)]
                session = Session(id=session_id, turns=turns)
                for turn, attempt in ((turns[0], 1), (turns[0], 2), (turns[1], 1)):
                    prepared = await runtime.prepare_attempt(AttemptInput.capture(
                        session, turn, {}, attempt=attempt, payloads=ctx.payloads,
                    ))
                    result = await runtime.execute_attempt(prepared, AttemptSignal(AttemptPlan()))
                    assert result.status == "succeeded", result
                await runtime.close_session(session.id)
            assert len(set(connections)) == expected
            if scope == "session":
                assert connections[0] is connections[1] is connections[2]
                assert connections[3] is connections[4] is connections[5]
                for _ in range(100):
                    if all(conn.is_closing() for conn in connections):
                        break
                    await asyncio.sleep(.001)
                assert all(conn.is_closing() for conn in connections)
        finally:
            await runtime.close()
            await ctx.events.close()


@pytest.mark.parametrize("settings", [
    {"connection_scope": "turn"}, {"connection_scope": None},
    {"verify_ssl": "false"}, {"base_url": "ftp://localhost"},
    {"base_url": "localhost:8000"},
])
def test_invalid_http_settings_fail_at_configuration(settings):
    with pytest.raises((ValueError, TypeError)):
        validate({"base_url": "http://localhost", "model": "test", **settings})

@pytest.mark.parametrize("tail,error", [
    (b'data: [DONE]\n\n', None), (b'', 'IncompleteResponse'),
    (b'data: [DONE]', 'IncompleteResponse'), (b'data: [DONE]\n', 'IncompleteResponse'),
    (b'data: {secret}\n\n', 'ProtocolError'),
    (b'data: {"error":{"message":"secret"}}\n\n', 'ProtocolError'),
    (b'data: {} garbage\n\n', 'ProtocolError'),
    (b'data: ' + b'x' * (1024 * 1024) + b'\n\n', 'ProtocolError'),
], ids=['done', 'eof', 'partial-done', 'partial-line', 'invalid-json', 'error', 'trailing-json', 'oversized'])
async def test_stream_completion_and_failure_preserve_partial_facts(tail, error):

    async def complete(request):
        await request.read()
        return web.Response(body=b'data: {"choices":[{"delta":{"content":"ok"}}],"usage":{"completion_tokens":2}}\n\n' + tail)

    app = web.Application()
    app.router.add_post('/', complete)
    async with TestServer(app) as server:
        native = _native.create_trial('openai-chat')
        try:
            value = await _request(native, dict( url=str(server.make_url('/')), method='POST',
                connection_scope='trial', body_length=0, stream=True, timeout=2))
            assert value.status == ('failed' if error else 'succeeded')
            assert value.error == error
            assert value.usage['completion_tokens'] == 2
            assert len(value.samples) == 1
            assert value.output_chars == 2
            assert value.started_ns <= value.samples[0][1] <= value.completed_ns
        finally:
            await native.close()


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("finish_reason", ["length", "stop", "content_filter"])
@pytest.mark.parametrize("completion_tokens", [0, 4096])
@pytest.mark.parametrize("cached_tokens", [0, 126720])
async def test_completed_empty_response_keeps_usage_and_missing_output_timing(
    stream, finish_reason, completion_tokens, cached_tokens,
):
    from perf_benchmark.plugins.metrics.standard.formulas import request_metrics, request_values
    usage = {"prompt_tokens": 133930, "completion_tokens": completion_tokens,
             "prompt_tokens_details": {"cached_tokens": cached_tokens}}
    async def complete(request):
        await request.read()
        if not stream:
            return web.json_response({"choices": [{"message": {"role": "assistant", "content": None},
                "finish_reason": finish_reason}], "usage": usage})
        events = [
            {"choices": [{"delta": {"role": "assistant", "content": ""}}]},
            {"choices": [{"delta": {}, "finish_reason": finish_reason}]},
            {"choices": [], "usage": usage},
        ]
        body = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
        return web.Response(body=body + b"data: [DONE]\n\n", content_type="text/event-stream")
    app = web.Application()
    app.router.add_post("/", complete)
    async with TestServer(app) as server:
        trial = _native.create_trial("openai-chat")
        try:
            value = await _request(trial, {"url": str(server.make_url("/")), "stream": stream})
            assert value.status == "succeeded" and value.error is None
            assert value.finish_reason == finish_reason
            assert value.output_chars == 0 and len(value.samples) == 0
            assert value.usage["completion_tokens"] == completion_tokens
            assert value.usage["cached_tokens"] == cached_tokens
            fact = SimpleNamespace(response_mode="stream" if stream else "non_stream", output_kind="text",
                request_ns=value.started_ns, first_ns=None, last_ns=None, response_completed_ns=value.completed_ns,
                turn_completed_ns=value.completed_ns, total_started_ns=value.started_ns,
                usage=value.usage, output_event_count=0)
            metrics = request_metrics(fact, request_values(fact))
            for name in ("ttft", "tpot", "prefill_throughput", "decode_throughput"):
                assert metrics[name].availability == "unavailable"
                assert all(value is None for value in metrics[name].statistics.values())
            assert metrics["e2e"].availability == "available"
            assert metrics["output_tokens"].statistics["value"] == completion_tokens
        finally:
            await trial.close()


@pytest.mark.parametrize("events,tail,error", [
    ([{"choices": [{"delta": {"role": "assistant"}}]}], b"data: [DONE]\n\n", "ProtocolError"),
    ([{"choices": [], "usage": {"completion_tokens": 4096}}], b"data: [DONE]\n\n", "ProtocolError"),
    ([{"choices": [{"delta": None, "finish_reason": "length"}]}], b"data: [DONE]\n\n", "ProtocolError"),
    ([{"choices": [{"delta": {}, "finish_reason": "unknown"}]}], b"data: [DONE]\n\n", "ProtocolError"),
    ([{"choices": [{"delta": {}, "finish_reason": "length"}]}], b"", "IncompleteResponse"),
    ([{"choices": [{"delta": {}, "finish_reason": "length"}]}], b"data: [DONE]", "IncompleteResponse"),
    ([{"choices": [{"delta": {}, "finish_reason": "length"}]}], b"data: {secret}\n\ndata: [DONE]\n\n", "ProtocolError"),
    ([{"choices": [{"delta": {}, "finish_reason": "length"}]}], b'data: {"error":{}}\n\ndata: [DONE]\n\n', "ProtocolError"),
])
async def test_empty_stream_invalid_terminal_and_truncation_remain_failures(events, tail, error):
    async def complete(request):
        await request.read()
        body = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
        return web.Response(body=body + tail, content_type="text/event-stream")
    app = web.Application()
    app.router.add_post("/", complete)
    async with TestServer(app) as server:
        trial = _native.create_trial("openai-chat")
        try:
            value = await _request(trial, {"url": str(server.make_url("/")), "stream": True})
            assert value.status == "failed" and value.error_origin == "protocol"
            assert value.error == error and value.output_chars == 0 and len(value.samples) == 0
        finally:
            await trial.close()


@pytest.mark.parametrize('verify', [False, True])
async def test_https_certificate_verification(tmp_path, verify):
    import ssl
    import subprocess

    key, cert = tmp_path / 'key.pem', tmp_path / 'cert.pem'
    # macOS LibreSSL otherwise generates a v1 certificate rejected by rustls.
    certificate_config = tmp_path / 'openssl.cnf'
    certificate_config.write_text(
        '[req]\ndistinguished_name=dn\nx509_extensions=v3\n'
        '[dn]\nCN=localhost\n[v3]\nbasicConstraints=critical,CA:FALSE\n'
        'subjectAltName=DNS:localhost,IP:127.0.0.1\n'
    )
    subprocess.run(['openssl', 'req', '-config', str(certificate_config), '-x509', '-newkey', 'rsa:2048', '-nodes', '-sha256', '-days', '1',
                    '-subj', '/CN=localhost', '-keyout', str(key), '-out', str(cert)],
                   check=True, capture_output=True)
    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.load_cert_chain(cert, key)

    async def complete(request):
        await request.read()
        return web.json_response({'choices': [{'message': {'content': 'ok'}}]})

    app = web.Application()
    app.router.add_post('/', complete)
    server = TestServer(app, scheme='https')
    await server.start_server(ssl=tls)
    native = _native.create_trial('openai-chat')
    try:
        value = await _request(native, dict( url=str(server.make_url('/')), method='POST',
            connection_scope='trial', body_length=0, stream=False, timeout=2, verify_ssl=verify))
        assert value.status == ('failed' if verify else 'succeeded'), value
        assert value.error_origin == ('transport' if verify else None)
    finally:
        await native.close()
        await server.close()




@pytest.mark.parametrize('fragment', [1, 17, 65536])
async def test_control_counts_business_events_across_fragmented_uploads(fragment):
    await _check_native_control(fragment, {'after_outputs': 2})


async def test_control_timer_begins_at_first_output():
    await _check_native_control(65536, {'after_first_output_ms': 20})


async def _check_native_control(fragment, plan):

    release = asyncio.Event()
    async def complete(request):
        await request.read()
        response = web.StreamResponse()
        await response.prepare(request)
        # Longer than the configured timer: no business output has arrived yet.
        await response.write(b'data: {"usage":{"completion_tokens":100}}\n\n')
        await asyncio.sleep(.06)
        body = b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n' * 2
        for offset in range(0, len(body), fragment):
            await response.write(body[offset:offset + fragment])
            await asyncio.sleep(0)
        await release.wait()
        return response

    app = web.Application()
    app.router.add_post('/', complete)
    async with TestServer(app) as server:
        native = _native.create_trial('openai-chat')
        try:
            result = await _request(native, dict( url=str(server.make_url('/')),
                method='POST', connection_scope='trial', body_length=0, stream=True,
                timeout=2, plan=plan))
            assert result.status == 'cancelled'
            assert result.error_origin == 'scenario'
            assert len(result.samples) == 2
            assert result.usage['completion_tokens'] == 100
            first = result.samples[0][1]
            assert first - result.started_ns >= 50_000_000
            expected = result.samples[-1][1] if 'after_outputs' in plan else first + 20_000_000
            assert result.trigger_ns == expected
            assert expected <= result.decision_ns <= result.applied_ns
        finally:
            release.set()
            await native.close()


@pytest.mark.parametrize('action', ['cancel', 'fail'])
async def test_external_signal_preserves_cause_and_partial_output(trial_context, action):
    from tests.executor.test_openai_chat import _attempt_input, _runtime, _turn

    entered, release = asyncio.Event(), asyncio.Event()
    async def complete(request):
        await request.read()
        response = web.StreamResponse()
        await response.prepare(request)
        await response.write(b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n')
        entered.set()
        await release.wait()
        return response

    app = web.Application()
    app.router.add_post('/v1/chat/completions', complete)
    async with TestServer(app) as server:
        ctx = trial_context('executor')
        turn = await _turn(ctx, 'turn', 'hello')
        ctx.payloads.seal()
        runtime = _runtime({'base_url': str(server.make_url('')), 'model': 'test'}, ctx)
        signal = AttemptSignal(AttemptPlan())
        try:
            await runtime.prepare()
            prepared = await runtime.prepare_attempt(_attempt_input(Session(turns=[turn], id="session-fixture-319"), turn, ctx))
            running = asyncio.create_task(runtime.execute_attempt(prepared, signal))
            await asyncio.wait_for(entered.wait(), 2)
            # Allow the first response bytes to arrive before externally stopping.
            await asyncio.sleep(.03)
            trigger = perf_counter_ns()
            await asyncio.to_thread(signal.trigger, 'operator stop', trigger, action=action)
            signal.trigger('duplicate ignored', action=action)
            result = await asyncio.wait_for(running, 2)
            assert result.status == ('failed' if action == 'fail' else 'cancelled')
            assert result.error == 'operator stop' and result.error_origin == 'client'
            assert result.retryable == (action == 'fail')
            assert result.request.output_chars == 7
            assert result.trigger_ns == trigger
            assert trigger <= result.decision_ns <= result.applied_ns
            signal.trigger('closed ignored', action=action)
            assert signal.reason == 'operator stop'
        finally:
            release.set()
            await runtime.close()
            await ctx.events.close()


@pytest.mark.parametrize("_iteration", range(20))
async def test_native_started_is_delivered_once_before_result(_iteration):

    async def complete(request):
        await request.read()
        return web.json_response({'choices': [{'message': {'content': 'ok'}}]})

    app = web.Application()
    app.router.add_post('/', complete)
    async with TestServer(app) as server:
        native = _native.create_trial('openai-chat')
        notifications = []
        try:
            result = await _request(native, dict( url=str(server.make_url('/')),
                method='POST', connection_scope='trial', body_length=0, timeout=2, notify_start=True), on_started=notifications.append)
            assert notifications == [result.started_ns]
            assert result.status == 'succeeded'
            assert result.started_ns <= result.completed_ns
            # Result delivery intentionally precedes transport/body cleanup.
            # Only the explicit close/settlement boundary guarantees no tasks.
            await asyncio.wait_for(native.close(), 2)
            assert native.closed and native.finished
            assert native.active_count == 0
        finally:
            await native.close()
