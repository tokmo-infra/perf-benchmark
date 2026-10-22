from perf_benchmark.core.profiles import Continuation
from typing import Any
import json
from copy import deepcopy

from aiohttp import web
from aiohttp.test_utils import TestServer
import pytest

from perf_benchmark.core.models import ContextMode, Run, Session, Turn
from perf_benchmark.pipeline.contracts.executor import (
    AttemptInput,
    AttemptPlan,
    ExecutorTrial,
)
from perf_benchmark.plugins.executor.openai_chat import OpenAIChatExecutor, validate
from perf_benchmark.runtime.http.rust_executor import RustHttpExecutorRuntime
from perf_benchmark.runtime.attempt import AttemptSignal


def _plugin(settings):
    return OpenAIChatExecutor(validate(settings))


def _runtime(settings, ctx):
    plugin = _plugin(settings)
    return RustHttpExecutorRuntime(plugin, plugin.settings, ctx.cancellation, "openai-chat")


def _attempt_input(session, turn, ctx, *, live_history=None, attempt=1):
    return AttemptInput.capture(
        session,
        turn,
        live_history if live_history is not None else {},
        attempt=attempt,
        payloads=ctx.payloads,
    )


async def _prepare(runtime, session, turn, ctx, *, live_history=None, attempt=1):
    return await runtime.prepare_attempt(
        _attempt_input(
            session,
            turn,
            ctx,
            live_history=live_history,
            attempt=attempt,
        )
    )


async def _turn(
    ctx,
    identifier: str,
    content: str,
    *,
    context_mode: ContextMode = "snapshot",
) -> Turn:
    value = (
        {"messages": [{"role": "user", "content": content}]}
        if context_mode == "snapshot"
        else {"role": "user", "content": content}
    )
    payload = json.dumps(value, separators=(",", ":")).encode()
    ref = ctx.payloads.append(payload)
    return Turn(
        id=identifier,
        request_payload_ref=ref,
        reference_output_tokens=3,
        requested_output_tokens=3,
        context_mode=context_mode,
    )


async def _execute(runtime, session, turn, ctx, live_history=None):
    prepared = await _prepare(
        runtime, session, turn, ctx, live_history=live_history, attempt=1
    )
    return await runtime.execute_attempt(
        prepared, AttemptSignal(AttemptPlan())
    )






@pytest.mark.parametrize("policy", [None, "fixed", "natural"])
async def test_trial_request_parameters_are_applied_and_isolated(
    trial_context, stage_context, policy,
):
    settings = {
        "base_url": "http://127.0.0.1:1", "model": "default", "stream": True,
        "max_tokens": 50, "extra_body": {"temperature": 0.9, "top_p": 0.8},
    }
    overrides = {"temperature": 0.1, "thinking": {"enabled": False}}
    original = deepcopy(overrides)
    trial = ExecutorTrial("trial", overrides)
    executor_ctx = stage_context("executor", settings)
    plugin = _plugin(settings)
    runtime = await plugin.create_runtime(trial, executor_ctx)
    following = await plugin.create_runtime(ExecutorTrial("following"), executor_ctx)
    ctx = trial_context("executor")
    turn = await _turn(ctx, "turn", "input")
    if policy:
        turn.metadata["dataset_output_policy"] = policy
    ctx.payloads.seal()
    try:
        overrides["thinking"]["enabled"] = True
        prepared = await _prepare(runtime, Session(turns=[turn], id="session-fixture-110"), turn, ctx)
        body = json.loads(b"".join(prepared.body.segments))
        assert body["model"] == "default"
        assert body["stream"] is True
        assert body["temperature"] == 0.1 and body["top_p"] == 0.8
        assert body["thinking"] == {"enabled": False}
        assert body["max_tokens"] == 3  # The dataset target still wins over Executor default.
        assert prepared.spec.streaming is True
        assert body["stream_options"]["include_usage"] is True
        if policy == "fixed":
            assert body["ignore_eos"] is True
        assert following.settings["base_url"] == settings["base_url"]
        assert following.settings["model"] == settings["model"]
        assert runtime.settings["extra_body"]["thinking"] == original["thinking"]
    finally:
        await runtime.close()
        await following.close()
        ctx.payloads.close()


@pytest.mark.parametrize("policy,overrides,match", [
    ("fixed", {"ignore_eos": False}, "fixed output conflicts"),
    ("fixed", {"stop": ["done"]}, "fixed output conflicts"),
    ("natural", {"ignore_eos": True}, "natural output conflicts"),
])
async def test_trial_request_parameters_preserve_output_contract(
    trial_context, stage_context, policy, overrides, match,
):
    ctx = trial_context("executor")
    turn = await _turn(ctx, "turn", "input")
    turn.metadata["dataset_output_policy"] = policy
    ctx.payloads.seal()
    settings: dict[str, Any] = {"base_url": "http://127.0.0.1:1", "model": "test"}
    runtime = await _plugin(settings).create_runtime(
        ExecutorTrial("trial", overrides),
        stage_context("executor", {"base_url": "http://127.0.0.1:1", "model": "test"}),
    )
    try:
        with pytest.raises(ValueError, match=match):
            await _prepare(runtime, Session(turns=[turn], id="session-fixture-149"), turn, ctx)
    finally:
        await runtime.close()
        ctx.payloads.close()


@pytest.mark.parametrize('policy,extra', [('natural', {}), ('fixed', {}), ('', {'ignore_eos': True})])
async def test_server_default_output_requires_natural_stop(trial_context, policy, extra):
    ctx = trial_context('executor')
    turn = await _turn(ctx, 'turn', 'input')
    turn.requested_output_tokens = None
    turn.metadata['dataset_output_policy'] = policy
    ctx.payloads.seal()
    session = Session(turns=[turn], id='server-default-session')
    runtime = _runtime({'base_url':'http://127.0.0.1:1', 'model':'test', 'extra_body':extra}, ctx)
    try:
        if policy != 'natural':
            with pytest.raises(ValueError, match='explicit output target'):
                await _prepare(runtime, session, turn, ctx)
        else:
            prepared = await _prepare(runtime, session, turn, ctx)
            body = json.loads(b''.join(prepared.body.segments))
            assert body['max_tokens'] == 4096
            assert body['temperature'] == 0.1
            assert prepared.request_facts.profile.temperature == 0.1
            assert not {'max_completion_tokens','max_output_tokens','ignore_eos'} & body.keys()
            assert prepared.request_facts.requested_output_tokens == 4096
    finally:
        await runtime.close()
        ctx.payloads.close()


@pytest.mark.parametrize(
    "requested, default, expected", [(7, 11, 7), (None, 11, 11), (None, None, 4096)]
)
async def test_request_target_is_independent_of_reference(
    trial_context, requested, default, expected
):
    ctx = trial_context("executor")
    turn = await _turn(ctx, "turn", "input")
    turn.reference_output_tokens = 999
    turn.requested_output_tokens = requested
    ctx.payloads.seal()
    session = Session(turns=[turn], id="session-fixture-166")
    defaults = {"max_tokens": default} if default is not None else {}
    runtime = _runtime(
        {"base_url": "http://127.0.0.1:1", "model": "test", **defaults},
        ctx,
    )
    try:
        prepared = await _prepare(runtime, session, turn, ctx)
        body = json.loads(b"".join(prepared.body.segments))
        assert body.get("max_tokens") == expected
        if expected is None:
            assert not {"max_tokens", "max_completion_tokens", "max_output_tokens", "ignore_eos"} & body.keys()
        assert prepared.request_facts.requested_output_tokens == expected
        assert turn.requested_output_tokens == requested
        assert turn.reference_output_tokens == 999
    finally:
        await runtime.close()
        ctx.payloads.close()


@pytest.mark.parametrize('stream',[True,False])
@pytest.mark.parametrize('requested,expected',[(None,None),(7,7)])
async def test_explicit_null_uses_server_default_without_output_cap(trial_context,stream,requested,expected):
    ctx = trial_context('executor')
    turn = await _turn(ctx,'server-default','input')
    turn.requested_output_tokens = requested
    turn.reference_output_tokens = 999
    turn.metadata['dataset_output_policy'] = 'natural'
    ctx.payloads.seal()
    runtime = _runtime({'base_url':'http://127.0.0.1:1','model':'test',
                        'max_tokens':None,'stream':stream},ctx)
    try:
        prepared = await _prepare(runtime,Session(turns=[turn],id='server-default'),turn,ctx)
        body = json.loads(b''.join(prepared.body.segments))
        assert body.get('max_tokens') == expected
        if expected is None:
            assert not {'max_tokens','max_completion_tokens','max_output_tokens','ignore_eos'} & body.keys()
        assert prepared.request_facts.requested_output_tokens == expected
        assert prepared.request_facts.profile.max_tokens == expected
        assert turn.reference_output_tokens == 999
    finally:
        await runtime.close()
        ctx.payloads.close()


@pytest.mark.parametrize(
    "defaults", [{"stop": ["done"]}, {"stop_token_ids": [2]}, {"ignore_eos": False}]
)
async def test_fixed_dataset_rejects_conflicting_stops(trial_context, defaults):
    ctx = trial_context("executor")
    turn = await _turn(ctx, "turn", "input")
    turn.metadata["dataset_output_policy"] = "fixed"
    ctx.payloads.seal()
    runtime = _runtime(
        {"base_url": "http://127.0.0.1:1", "model": "test", "extra_body": defaults},
        ctx,
    )
    try:
        with pytest.raises(ValueError, match="fixed output conflicts"):
            await _prepare(runtime, Session(turns=[turn], id="session-fixture-201"), turn, ctx)
    finally:
        await runtime.close()
        ctx.payloads.close()


























async def test_executor_creates_unprepared_runtime(trial_context, stage_context):
    settings = {"base_url": "http://127.0.0.1:11434", "model": "qwen"}
    ctx = stage_context("executor", settings)
    runtime = await _plugin(settings).create_runtime(ExecutorTrial("trial"), ctx)
    assert runtime.settings["model"] == "qwen"
    await runtime.close()


@pytest.mark.parametrize(
    "settings,parameters,match",
    [
        ({"cache_salt_enabled": "yes"}, {}, "must be a boolean"),
        ({}, {"request_parameters": {"cache_salt_enabled": False}}, "generated fields"),
        ({"cache_salt_on_cycle": True}, {}, "unknown"),
        ({"extra_body": {"cache_salt": "fixed"}}, {}, "generated fields"),
        ({}, {"request_parameters": {"cache_salt": "fixed"}}, "generated fields"),
    ],
)
async def test_cycle_cache_salt_configuration_is_strict(stage_context, settings, parameters, match):
    with pytest.raises((TypeError, ValueError), match=match):
        base_settings = {"base_url": "http://127.0.0.1:1", "model": "test", **settings}
        await _plugin(base_settings).create_runtime(
            ExecutorTrial("trial", parameters.get("request_parameters", {})),
            stage_context(
                "executor",
                base_settings,
            ),
        )


@pytest.mark.parametrize("count", [0, -1, True, "16"])
def test_dp_rank_count_requires_positive_integer(count):
    with pytest.raises(ValueError, match="dp_rank_count"):
        _plugin({"base_url": "http://127.0.0.1:1", "model": "test", "dp_rank_count": count})


def test_dp_rank_count_rejects_static_rank_header():
    with pytest.raises(ValueError, match="conflicts"):
        _plugin({"base_url": "http://127.0.0.1:1", "model": "test", "dp_rank_count": 16,
                 "extra_headers": {"x-data-parallel-rank": "0"}})


@pytest.mark.parametrize("urls", [[], ["http://127.0.0.1:1", "not-a-url"], "http://127.0.0.1:1"])
def test_base_urls_requires_nonempty_absolute_url_list(urls):
    with pytest.raises(ValueError, match="base_urls|base_url"):
        _plugin({"base_urls": urls, "model": "test"})


def test_base_urls_rejects_ambiguous_base_url():
    with pytest.raises(ValueError, match="choose base_url or base_urls"):
        _plugin({"base_url": "http://127.0.0.1:1", "base_urls": ["http://127.0.0.1:2"],
                 "model": "test"})


@pytest.mark.asyncio
async def test_dp_rank_is_sticky_per_session(trial_context):
    ctx = trial_context("executor")
    turn = await _turn(ctx, "turn", "input")
    ctx.payloads.seal()
    plugin = _plugin({"base_url": "http://127.0.0.1:1", "model": "test",
                      "max_tokens": 3, "dp_rank_count": 16})
    try:
        ranks = []
        for ordinal in (0, 16, 17):
            session = Session(id=f"session-{ordinal}", turns=[turn], ordinal=ordinal)
            spec = plugin.build_request(_attempt_input(session, turn, ctx), plugin.settings)
            ranks.append(spec.headers["X-Data-Parallel-Rank"])
        assert ranks == ["0", "0", "1"]
    finally:
        ctx.payloads.close()


@pytest.mark.asyncio
async def test_base_urls_are_sticky_per_session(trial_context):
    ctx = trial_context("executor")
    turn = await _turn(ctx, "turn", "input")
    ctx.payloads.seal()
    plugin = _plugin({"base_urls": ["http://127.0.0.1:1", "http://127.0.0.1:2"],
                      "model": "test", "max_tokens": 3})
    try:
        urls = []
        for ordinal in (0, 1, 2):
            session = Session(id=f"session-{ordinal}", turns=[turn], ordinal=ordinal)
            spec = plugin.build_request(_attempt_input(session, turn, ctx), plugin.settings)
            urls.append(spec.url)
        assert urls == ["http://127.0.0.1:1/v1/chat/completions",
                        "http://127.0.0.1:2/v1/chat/completions",
                        "http://127.0.0.1:1/v1/chat/completions"]
    finally:
        ctx.payloads.close()


@pytest.mark.parametrize("size", [None, 0, -1, True, "16", 1.5])
def test_base_url_session_block_size_requires_positive_integer(size):
    with pytest.raises(ValueError, match="base_url_session_block_size"):
        _plugin({"base_urls": ["http://127.0.0.1:1"], "model": "test",
                 "base_url_session_block_size": size})


def test_base_url_session_block_size_requires_multiple_url_setting():
    with pytest.raises(ValueError, match="requires base_urls"):
        _plugin({"base_url": "http://127.0.0.1:1", "model": "test",
                 "base_url_session_block_size": 16})


@pytest.mark.parametrize("engines,dp", [(1, 8), (2, 4), (4, 4), (2, 8)])
async def test_block_routing_preserves_original_32_user_pool_and_dp_ranks(
    trial_context, engines, dp,
):
    ctx = trial_context("executor")
    first = await _turn(ctx, "first", "cold")
    deep = await _turn(ctx, "deep", "deep history")
    reset = await _turn(ctx, "reset", "new prefix")
    ctx.payloads.seal()
    urls = [f"http://127.0.0.1:{30001 + i * 1000}" for i in range(engines)]
    block_size = 32 // engines
    # The frozen generator creates all sessions for URL0, then all for URL1.
    expected_routes = [url for url in urls for _ in range(block_size)]
    plugin = _plugin({"base_urls": urls, "base_url_session_block_size": block_size,
                      "model": "test", "dp_rank_count": dp})
    try:
        for sid, expected in enumerate(expected_routes):
            session = Session(id=f"session-{sid}", turns=[first, deep, reset], ordinal=sid)
            for turn in session.turns:
                spec = plugin.build_request(_attempt_input(session, turn, ctx), plugin.settings)
                assert spec.url == expected + "/v1/chat/completions"
                assert spec.headers["X-Data-Parallel-Rank"] == str(sid % dp)
        for engine in range(engines):
            pool = range(engine * block_size, (engine + 1) * block_size)
            assert {sid % dp for sid in pool} == set(range(dp))
    finally:
        ctx.payloads.close()


async def test_cycle_cache_salt_lifecycle(trial_context, stage_context):
    ctx = trial_context("executor", run=Run(name="cycle-salt"))
    original_turn = await _turn(ctx, "original", "same prompt")
    first_turn = await _turn(ctx, "first-1", "same prompt")
    second_turn = await _turn(ctx, "first-2", "same prompt")
    peer_turn = await _turn(ctx, "peer", "same prompt")
    disabled_turn = await _turn(ctx, "disabled", "same prompt")
    warmup_turn = await _turn(ctx, "warmup", "same prompt")
    ctx.payloads.seal()

    original = Session(id="original-session", turns=[original_turn])
    replay = Session(id="replay-session", turns=[first_turn, second_turn], metadata={"cycle": 1})
    peer = Session(id="peer-session", turns=[peer_turn], metadata={"cycle": 1})
    disabled = Session(id="disabled-session", turns=[disabled_turn], metadata={"cycle": 1})
    warmup = Session(id="warmup-session", turns=[warmup_turn], metadata={"warmup": True})
    for session in (original, replay, peer, disabled, warmup):
        session.link()

    settings: dict[str, Any] = {"base_url": "http://127.0.0.1:1", "model": "test", "cache_salt_enabled": True}
    disabled_settings = {**settings, "cache_salt_enabled": False}
    runtime = await _plugin(settings).create_runtime(
        ExecutorTrial("trial"), stage_context("executor", settings)
    )
    disabled_runtime = await _plugin(disabled_settings).create_runtime(
        ExecutorTrial("disabled"), stage_context("executor", disabled_settings)
    )
    prepared = []
    try:
        for session, turn, attempt in (
            (original, original_turn, 1),
            (replay, first_turn, 1),
            (replay, first_turn, 2),
            (replay, second_turn, 1),
            (peer, peer_turn, 1),
            (warmup, warmup_turn, 1),
        ):
            prepared.append(await _prepare(runtime, session, turn, ctx, attempt=attempt))
        disabled_prepared = await _prepare(
            disabled_runtime, disabled, disabled_turn, ctx, attempt=1
        )
        bodies = [json.loads(b"".join(value.body.segments)) for value in prepared]
        disabled_body = json.loads(b"".join(disabled_prepared.body.segments))

        assert "cache_salt" in bodies[0]
        assert "cache_salt" not in disabled_body
        replay_salt = bodies[1]["cache_salt"]
        assert len(replay_salt) == 43
        assert bodies[2]["cache_salt"] == bodies[3]["cache_salt"] == replay_salt
        assert bodies[4]["cache_salt"] == replay_salt
        assert bodies[0]["cache_salt"] != replay_salt
        assert bodies[5]["cache_salt"] not in {bodies[0]["cache_salt"], replay_salt}
        assert all(
            body["messages"] == [{"role": "user", "content": "same prompt"}] for body in bodies
        )
        assert "cache_salt_applied" not in disabled_prepared.request_facts.metadata
        assert all(value.request_facts.metadata["cache_salt_applied"] for value in prepared)
        fingerprints = {
            value.request_facts.metadata["execution_fingerprint"] for value in prepared
        }
        assert len(fingerprints) == 1
        assert disabled_prepared.request_facts.metadata["execution_fingerprint"] not in fingerprints
        with ctx.payloads.view(original_turn.request_payload_ref) as payload:
            assert json.loads(bytes(payload)) == {
                "messages": [{"role": "user", "content": "same prompt"}]
            }
    finally:
        await disabled_runtime.close()
        await runtime.close()
        ctx.payloads.close()


async def test_prepared_body_uses_read_only_mmap_segment(trial_context) -> None:
    run = Run(name="mmap-body")
    ctx = trial_context("executor", run=run)
    turn = await _turn(ctx, "turn", "question")
    ctx.payloads.seal()
    session = Session(turns=[turn], id="session-fixture-337")
    session.link()
    runtime = _runtime(
        {
            "base_url": "http://127.0.0.1:1",
            "model": "debug",
        },
        ctx,
    )

    prepared = await _prepare(runtime, session, turn, ctx)

    mmap_segment = prepared.body.segments[1]
    assert isinstance(mmap_segment, memoryview)
    assert mmap_segment.readonly
    assert b"question" in bytes(mmap_segment)
    with pytest.raises(BufferError):
        mmap_segment.release()
    await runtime.release(prepared)


async def test_openai_stream_records_usage_without_retaining_final_body(
    trial_context,
) -> None:
    received = {}

    async def complete(request):
        received.update(await request.json())
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for event in (
            {
                "choices": [
                    {
                        "delta": {
                            "content": "",
                            "reasoning": "thinking",
                        }
                    }
                ]
            },
            {"choices": [{"delta": {"content": "hello "}}]},
            {"choices": [{"delta": {"content": "world"}, "finish_reason": "stop"}]},
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 3,
                    "total_tokens": 10,
                    "prompt_tokens_details": {"cached_tokens": 2},
                },
            },
        ):
            await response.write(f"data: {json.dumps(event)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        return response

    app = web.Application()
    app.router.add_post("/v1/chat/completions", complete)
    server = TestServer(app)
    await server.start_server()
    run = Run(name="runtime")
    ctx = trial_context("executor", run=run)
    runtime = _runtime(
        {
            "base_url": str(server.make_url("")).rstrip("/"),
            "model": "debug",
        },
        ctx,
    )
    turn = await _turn(ctx, "turn", "question")
    ctx.payloads.seal()
    session = Session(turns=[turn], id="session-fixture-409")
    session.link()
    try:
        await runtime.prepare()
        result = await _execute(runtime, session, turn, ctx)
    finally:
        await runtime.close()
        await server.close()

    assert received["messages"] == [{"role": "user", "content": "question"}]
    assert received["stream_options"] == {"include_usage": True}
    assert result.continuation_payload is None
    assert result.request.output_chars == len("hello worldthinking")
    assert result.usage == {
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "total_tokens": 10,
        "cached_tokens": 2,
    }
    assert result.request.response_mode == "stream"
    assert result.request.output_event_count == 3
    started_ns = result.request.started_ns
    first_output_ns = result.request.first_output_ns
    assert started_ns is not None and first_output_ns is not None
    assert started_ns <= first_output_ns
    assert not hasattr(turn, "response_messages")


@pytest.mark.parametrize("stream", [True, False])
async def test_executor_settings_build_body_and_select_response_mode(
    stream, trial_context
):
    requests = []
    parameters: dict[str, Any] = {
        "ignore_eos": True,
        "thinking": True,
        "temperature": 0.1,
    }

    async def complete(request):
        body = await request.json()
        requests.append(body)
        if not body["stream"]:
            return web.json_response(
                {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
            )
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n')
        if body["stream_options"]["include_usage"]:
            await response.write(b'data: {"usage":{"completion_tokens":1}}\n\n')
        await response.write(b"data: [DONE]\n\n")
        return response

    app = web.Application()
    app.router.add_post("/v1/chat/completions", complete)
    async with TestServer(app) as server:
        ctx = trial_context("executor")
        turn = await _turn(ctx, "turn", "question")
        turn.requested_output_tokens = None
        ctx.payloads.seal()
        runtime = _runtime(
            {
                "base_url": str(server.make_url("")),
                "model": "configured-model",
                "stream": stream,
                "max_tokens": 5,
                "extra_body": parameters,
            },
            ctx,
        )
        try:
            await runtime.prepare()
            result = await _execute(runtime, Session(turns=[turn], id="session-fixture-482"), turn, ctx)
        finally:
            await runtime.close()
            await ctx.events.close()

    (body,) = requests
    assert body["messages"] == [{"role": "user", "content": "question"}]
    assert body["model"] == "configured-model" and body["max_tokens"] == 5
    assert body["ignore_eos"] is True and body["thinking"] is True
    assert body["temperature"] == 0.1 and body["stream"] is stream
    assert result.status == "succeeded"
    assert result.request.response_mode == ("stream" if stream else "non_stream")
    assert parameters.get("stream_options") is None
    if not stream:
        assert "stream_options" not in body
        assert result.request.first_output_ns is None and not result.trace.samples
    else:
        assert body["stream_options"] == {"include_usage": True}
        assert result.request.first_output_ns is not None
        assert result.usage == {"completion_tokens": 1}


async def test_openai_stream_submits_one_terminal_trace(
    trial_context,
) -> None:
    async def complete(request):
        await request.json()
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        # Exercise many semantic events, not an unpaced transport burst at C=1.
        await response.write(
            b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n' * 65 + b"data: [DONE]\n\n"
        )
        return response

    class Recorder:
        def __init__(self):
            self.traces = []

        def record_turn(self, turn, trace=None):
            self.traces.append(trace)

    app = web.Application()
    app.router.add_post("/v1/chat/completions", complete)
    server = TestServer(app)
    await server.start_server()
    run = Run(name="runtime")
    ctx = trial_context("executor", run=run)
    runtime = _runtime(
        {
            "base_url": str(server.make_url("")).rstrip("/"),
            "model": "debug",
        },
        ctx,
    )
    recorder = Recorder()
    ctx.measurements = recorder
    turn = await _turn(ctx, "turn", "question")
    ctx.payloads.seal()
    turn.requested_output_tokens = 65
    session = Session(turns=[turn], id="session-fixture-542")
    session.link()
    try:
        await runtime.prepare()
        result = await _execute(runtime, session, turn, ctx)
    finally:
        await runtime.close()
        await server.close()

    assert recorder.traces == []
    assert len(result.trace.samples) == result.request.output_event_count
    assert result.trace.samples[0][0] == 1
    assert result.trace.samples[-1][0] == 65


async def test_openai_stream_retains_only_required_multiturn_history(
    trial_context,
) -> None:
    requests = []

    async def complete(request):
        requests.append(await request.json())
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b'data: {"choices":[{"delta":{"content":"answer"}}]}\n\n')
        await response.write(b"data: [DONE]\n\n")
        return response

    app = web.Application()
    app.router.add_post("/v1/chat/completions", complete)
    server = TestServer(app)
    await server.start_server()
    run = Run(name="runtime")
    ctx = trial_context("executor", run=run)
    runtime = _runtime(
        {
            "base_url": str(server.make_url("")).rstrip("/"),
            "model": "debug",
        },
        ctx,
    )
    first = await _turn(ctx, "turn-1", "first", context_mode="live")
    second = await _turn(ctx, "turn-2", "second", context_mode="live")
    ctx.payloads.seal()
    session = Session(id="session", turns=[first, second])
    session.link()
    try:
        await runtime.prepare()
        live_history = {}
        first_result = await _execute(runtime, session, first, ctx, live_history)
        first.succeed()
        live_history[first.id] = Continuation(first_result.continuation_payload, first_result.continuation_profile)
        second_result = await _execute(runtime, session, second, ctx, live_history)
    finally:
        await runtime.close()
        await server.close()

    assert json.loads(first_result.continuation_payload) == {"role": "assistant", "content": "answer"}
    assert second_result.continuation_payload is None
    assert requests[1]["messages"] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "second"},
    ]


async def test_recorded_multiturn_uses_dataset_answer_not_measured_answer(
    trial_context,
) -> None:
    requests = []

    async def complete(request):
        requests.append(await request.json())
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b'data: {"choices":[{"delta":{"content":"measured answer"}}]}\n\n')
        await response.write(b"data: [DONE]\n\n")
        return response

    app = web.Application()
    app.router.add_post("/v1/chat/completions", complete)
    server = TestServer(app)
    await server.start_server()
    run = Run(name="runtime")
    ctx = trial_context("executor", run=run)
    runtime = _runtime(
        {
            "base_url": str(server.make_url("")).rstrip("/"),
            "model": "debug",
        },
        ctx,
    )
    first = await _turn(ctx, "turn-1", "first", context_mode="recorded")
    second = await _turn(ctx, "turn-2", "second", context_mode="recorded")
    recorded = json.dumps(
        {"role": "assistant", "content": "recorded answer"},
        separators=(",", ":"),
    ).encode()
    first.recorded_response_payload_ref = ctx.payloads.append(recorded)
    second.recorded_response_payload_ref = ctx.payloads.append(recorded)
    ctx.payloads.seal()
    session = Session(id="session", turns=[first, second])
    session.link()
    try:
        await runtime.prepare()
        result = await _execute(runtime, session, first, ctx)
        first.succeed()
        await _execute(runtime, session, second, ctx)
    finally:
        await runtime.close()
        await server.close()

    assert result.continuation_payload is None
    assert requests[1]["messages"] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "recorded answer"},
        {"role": "user", "content": "second"},
    ]


async def test_openai_chat_rejects_non_chat_input(trial_context) -> None:
    run = Run(name="runtime")
    ctx = trial_context("executor", run=run)
    turn = await _turn(ctx, "turn", "input")
    ctx.payloads.seal()
    turn.input_kind = "embedding"
    session = Session(turns=[turn], id="session-fixture-668")
    session.link()
    runtime = _runtime({"base_url": "http://127.0.0.1:1", "model": "debug"}, ctx)
    try:
        with pytest.raises(TypeError, match="requires chat input"):
            await _execute(runtime, session, turn, ctx)
    finally:
        await runtime.close()


async def test_openai_non_streaming_parses_response_in_request_memory(
    trial_context,
) -> None:
    received = {}

    async def complete(request):
        received.update(await request.json())
        return web.json_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "non-stream answer",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 3,
                    "total_tokens": 7,
                },
            }
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", complete)
    server = TestServer(app)
    await server.start_server()
    run = Run(name="runtime")
    ctx = trial_context("executor", run=run)
    runtime = _runtime(
        {
            "base_url": str(server.make_url("")).rstrip("/"),
            "model": "debug",
            "stream": False,
        },
        ctx,
    )
    turn = await _turn(ctx, "turn", "question")
    ctx.payloads.seal()
    session = Session(turns=[turn], id="session-fixture-720")
    session.link()
    try:
        await runtime.prepare()
        result = await _execute(runtime, session, turn, ctx)
    finally:
        await runtime.close()
        await server.close()

    assert received["stream"] is False
    assert "stream_options" not in received
    assert result.continuation_payload is None
    assert result.request.output_chars == len("non-stream answer")
    assert result.usage == {
        "prompt_tokens": 4,
        "completion_tokens": 3,
        "total_tokens": 7,
    }
    request = result.request
    assert request.response_mode == "non_stream"
    assert request.output_event_count == 0
    assert request.first_output_ns is None
    assert request.last_output_ns is None
    assert request.response_completed_ns is not None


@pytest.mark.parametrize("send", [False, True])
async def test_runtime_releases_unsent_body_with_retained_preparation(trial_context, send):
    ctx = trial_context("executor", run=Run(name="release"))
    turn = await _turn(ctx, "turn", "question")
    ctx.payloads.seal()
    session = Session(turns=[turn], id="session-fixture-751")
    session.link()
    runtime = _runtime({"base_url": "http://127.0.0.1:1", "model": "test",
                        "extra_body": {"temperature": 0.2}}, ctx)
    prepared = await _prepare(runtime, session, turn, ctx)
    retained = []
    if send:
        try:
            # Failure before entering the aiohttp request context still owns a body.
            result = await runtime.execute_attempt(
                prepared, AttemptSignal(AttemptPlan())
            )
            if result.fatal_error:
                retained.append(result.fatal_error)
        except RuntimeError as exc:
            retained.append(exc)
    await runtime.close()
    await runtime.close()
    ctx.payloads.close()
    assert len(retained) == int(send)
    with pytest.raises(ValueError, match="released memoryview"):
        bytes(prepared.body.segments[1])


@pytest.mark.parametrize("failure", ["snapshot", "live", "recorded", "parameters"])
async def test_body_construction_failure_releases_partial_views(trial_context, failure):
    ctx = trial_context("executor", run=Run(name="partial-body"))
    mode = "snapshot" if failure == "snapshot" else "recorded" if failure == "recorded" else "live"
    first = await _turn(ctx, "first", "question", context_mode=mode)
    turns = [first]
    if failure in ("live", "recorded"):
        turns.append(await _turn(ctx, "second", "follow up", context_mode=mode))
    if failure == "snapshot":
        first.request_payload_ref = ctx.payloads.append(b"[]")
    ctx.payloads.seal()
    session = Session(turns=turns, id="session-fixture-786")
    session.link()
    settings: dict[str, Any] = {"base_url": "http://127.0.0.1:1", "model": "test"}
    if failure == "parameters":
        settings["extra_body"] = {"extra": "x" * 65500}
    runtime = _runtime(settings, ctx)
    with pytest.raises((ValueError, RuntimeError)) as retained:
        await _prepare(runtime, session, turns[-1], ctx)
    await runtime.close()
    ctx.payloads.close()
    assert retained.value.__traceback__ is not None


@pytest.mark.parametrize("live", [False, True])
async def test_cancelled_preparation_is_joined_before_mmap_close(trial_context, monkeypatch, live):
    import asyncio
    import threading
    from perf_benchmark.plugins.executor.openai_chat import request as runtime_module

    ctx = trial_context("executor", run=Run(name="cancel-prepare"))
    mode = "live" if live else "snapshot"
    first = await _turn(ctx, "first", "question", context_mode=mode)
    turns = [first]
    history = {}
    if live:
        first.succeed()
        history[first.id] = Continuation(b'{"role":"assistant","content":"retained history"}')
        turns.append(await _turn(ctx, "second", "follow up", context_mode=mode))
    ctx.payloads.seal()
    session = Session(turns=turns, id="session-fixture-815")
    session.link()
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    original = runtime_module._build_body
    bodies, encoded = [], []

    def blocked(*args):
        loop.call_soon_threadsafe(entered.set)
        if not release.wait(5):
            raise TimeoutError("test did not release preparation")
        body = original(*args)
        bodies.append(body)
        encoded.append(b"".join(body.segments))
        return body

    monkeypatch.setattr(runtime_module, "_build_body", blocked)
    runtime = _runtime({"base_url": "http://127.0.0.1:1", "model": "test",
                        "extra_body": {"temperature": 0.2}}, ctx)
    preparing = asyncio.create_task(
        _prepare(runtime, session, turns[-1], ctx, live_history=history)
    )
    await asyncio.wait_for(entered.wait(), 2)
    preparing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await preparing
    history.clear()
    closing = asyncio.create_task(runtime.close())
    try:
        await asyncio.sleep(0)
        closing.cancel()
        await asyncio.sleep(0)
        closing.cancel()
        await asyncio.sleep(0)
        assert not closing.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(closing, 3)
    await runtime.close()
    ctx.payloads.close()
    assert len(bodies) == 1
    if live:
        assert b"retained history" in encoded[0]
    with pytest.raises(RuntimeError, match="closed"):
        await _prepare(runtime, session, turns[-1], ctx)




async def test_multimodal_snapshot_preserves_protocol_content_without_copying_body(trial_context):
    ctx = trial_context("executor")
    content = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
                ],
            }
        ]
    }
    turn = Turn(
        request_payload_ref=ctx.payloads.append(json.dumps(content).encode()),
        requested_output_tokens=3,
        id="turn-fixture-879",
    )
    ctx.payloads.seal()
    runtime = _runtime({"base_url": "http://unused", "model": "test"}, ctx)
    try:
        prepared = await _prepare(runtime, Session(turns=[turn], id="session-fixture-886"), turn, ctx)
        assert isinstance(prepared.body.segments[1], memoryview)
        assert json.loads(b"".join(prepared.body.segments))["messages"] == content["messages"]
    finally:
        await runtime.close()
        ctx.payloads.close()
