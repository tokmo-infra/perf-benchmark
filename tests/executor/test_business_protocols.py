import json

from aiohttp import web
from aiohttp.test_utils import TestServer
import pytest

from perf_benchmark.core.models import Session, Turn
from perf_benchmark.core.profiles import RequestProfile
from perf_benchmark.pipeline.contracts.executor import AttemptInput, AttemptPlan, ExecutorTrial
from perf_benchmark.plugins.executor import openai_embeddings, openai_responses, rerank
from perf_benchmark.runtime.attempt import AttemptSignal


@pytest.mark.parametrize('plugin,cls,kind,path,payload,reply', [
    (openai_embeddings, openai_embeddings.EmbeddingsExecutor, 'embedding', '/v1/embeddings',
     {'input': ['one', 'two']}, {'data': [{'index': 1, 'embedding': [0.1, 0.2]}, {'index': 0, 'embedding': [0.3, 0.4]}],
                               'usage': {'prompt_tokens': 2, 'total_tokens': 2}}),
    (rerank, rerank.RerankExecutor, 'rerank', '/v1/rerank', {'query': 'q', 'documents': ['a', 'b']},
     {'results': [{'index': 1, 'relevance_score': 0.9}, {'index': 0, 'relevance_score': 0.1}],
      'usage': {'prompt_tokens': 3, 'total_tokens': 3}}),
])
@pytest.mark.parametrize('bad', [False, True])
async def test_non_text_protocol_validation(trial_context, stage_context, plugin, cls, kind, path, payload, reply, bad):
    received = []
    if bad:
        reply = json.loads(json.dumps(reply))
        reply['data' if kind == 'embedding' else 'results'][1]['index'] = 1
    async def handle(request):
        received.append(await request.json())
        return web.json_response(reply)
    app = web.Application()
    app.router.add_post(path, handle)
    async with TestServer(app) as server:
        instance = cls(plugin.validate({'base_url': str(server.make_url('')).rstrip('/'), 'model': 'mock'}))
        result = await execute(instance, trial_context, stage_context, kind, payload)
    assert result.status == ('failed' if bad else 'succeeded')
    assert result.request.output_kind == 'non_text'
    assert result.request.response_mode == 'non_stream'
    assert not result.trace.samples
    assert result.continuation_payload is None
    assert 'completion_tokens' not in result.usage
    assert received[0]['model'] == 'mock'
    if not bad:
        assert result.usage == reply['usage']


@pytest.mark.parametrize('usage,expected,status', [
    ({'prompt_tokens': 5, 'total_tokens': 5}, {'prompt_tokens': 5, 'total_tokens': 5}, 'succeeded'),
    ({'prompt_tokens': 0}, {'prompt_tokens': 0}, 'succeeded'),
    ({'total_tokens': 7}, {'total_tokens': 7}, 'succeeded'),
    ({'prompt_tokens': None, 'total_tokens': None}, {}, 'succeeded'),
    ({'completion_tokens': 4}, {}, 'succeeded'),
    ({}, {}, 'succeeded'),
    (None, {}, 'succeeded'),
    ({'prompt_tokens': -1}, {}, 'failed'),
    ({'total_tokens': -1}, {}, 'failed'),
    ({'prompt_tokens': 1.5}, {}, 'failed'),
    ({'total_tokens': True}, {}, 'failed'),
    ({'prompt_tokens': '3'}, {}, 'failed'),
])
async def test_rerank_server_usage(trial_context, stage_context, usage, expected, status):
    async def handle(request):
        body = await request.json()
        assert body['query'] == 'q' and body['documents'] == ['a', 'b']
        return web.json_response({
            'results': [{'index': 1, 'relevance_score': 0.9}, {'index': 0, 'relevance_score': 0.1}],
            'usage': usage,
        })

    app = web.Application()
    app.router.add_post('/v1/rerank', handle)
    async with TestServer(app) as server:
        plugin = rerank.RerankExecutor(rerank.validate({
            'base_url': str(server.make_url('')).rstrip('/'), 'model': 'mock',
        }))
        result = await execute(plugin, trial_context, stage_context, 'rerank', {
            'query': 'q', 'documents': ['a', 'b'],
        })
    assert result.status == status
    assert result.usage == expected
    assert result.request.output_kind == 'non_text'
    assert result.request.response_mode == 'non_stream'
    assert not result.trace.samples


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('tool_only', [False, True])
@pytest.mark.parametrize('terminal', ['completed', 'incomplete'])
async def test_responses_text_tools_and_partial_usage(trial_context, stage_context, stream, tool_only, terminal):
    final = {'status': terminal, 'output': [{'type': 'function_call', 'name': 'lookup', 'arguments': '{}'}] if tool_only else
             [{'type': 'message', 'content': [{'type': 'output_text', 'text': '答案'}]}],
             'usage': {'input_tokens': 3, 'output_tokens': 2, 'total_tokens': 5}}
    async def handle(request):
        body = await request.json()
        assert body['input'] == 'question' and body['store'] is False
        assert body['max_output_tokens'] == 4096
        assert body['temperature'] == 0.1
        if not stream:
            return web.json_response(final)
        response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
        await response.prepare(request)
        events = [dict(type='response.function_call_arguments.delta' if tool_only else 'response.output_text.delta',
                       delta='{}' if tool_only else '答案'), dict(type='response.' + terminal, response=final)]
        data = ''.join('data: '+json.dumps(event, ensure_ascii=False)+'\n\n' for event in events).encode()
        for i in range(0, len(data), 5):
            await response.write(data[i:i+5])
        await response.write_eof()
        return response
    app = web.Application()
    app.router.add_post('/v1/responses', handle)
    async with TestServer(app) as server:
        instance = openai_responses.ResponsesExecutor(openai_responses.validate({
            'base_url': str(server.make_url('')).rstrip('/'), 'model': 'mock', 'stream': stream}))
        result = await execute(instance, trial_context, stage_context, 'responses', {'input': 'question'})
    assert result.status == ('succeeded' if terminal == 'completed' else 'failed')
    assert result.usage == {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5}
    assert result.request.output_kind == ('non_text' if tool_only else 'text')
    assert result.request.output_chars == (0 if tool_only else 2)
    assert len(result.trace.samples) == (1 if stream and not tool_only else 0)
    assert result.continuation_payload is None


async def execute(plugin, trial_context, stage_context, kind, payload):
    ctx = trial_context('executor')
    turn = Turn(id='t', request_payload_ref=ctx.payloads.append(json.dumps(payload).encode()),
                input_kind=kind, profile=RequestProfile(input_items=2))
    session = Session(turns=[turn], id="session-fixture-83")
    session.link()
    ctx.payloads.seal()
    runtime = await plugin.create_runtime(ExecutorTrial('trial'), stage_context('executor'))
    try:
        await runtime.prepare()
        prepared = await runtime.prepare_attempt(AttemptInput.capture(session, turn, {}, attempt=1, payloads=ctx.payloads))
        assert any(isinstance(s, memoryview) for s in prepared.body.segments)
        return await runtime.execute_attempt(prepared, AttemptSignal(AttemptPlan()))
    finally:
        await runtime.close()
        ctx.payloads.close()
