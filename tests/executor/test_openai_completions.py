import json

from aiohttp import web
from aiohttp.test_utils import TestServer
import pytest

from perf_benchmark.core.models import Session, Turn
from perf_benchmark.pipeline.contracts.executor import AttemptInput, AttemptPlan, ExecutorTrial
from perf_benchmark.plugins.executor.openai_completions import OpenAICompletionsExecutor, validate
from perf_benchmark.runtime.attempt import AttemptSignal
from perf_benchmark.runtime.http.request import PreparedBody


@pytest.mark.parametrize('stream', [True, False])
@pytest.mark.parametrize('target', [None, 2])
async def test_completions_transmits_text_and_records_transport_facts(trial_context, stage_context, stream, target):
    received = []
    async def complete(request):
        body = await request.json()
        received.append(body)
        response = {'choices': [{'index': 0, 'text': '答案', 'finish_reason': 'length' if target else 'stop'}],
                    'usage': {'prompt_tokens': 3, 'completion_tokens': 2}}
        if not stream:
            return web.json_response(response)
        result = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
        await result.prepare(request)
        data = ('data: ' + json.dumps(response) + '\n\ndata: [DONE]\n\n').encode()
        for offset in range(0, len(data), 7):
            await result.write(data[offset:offset + 7])
        await result.write_eof()
        return result
    app = web.Application()
    app.router.add_post('/v1/completions', complete)
    async with TestServer(app) as server:
        plugin = OpenAICompletionsExecutor(validate({'base_url': str(server.make_url('')).rstrip('/'),
                                                    'model': 'target', 'stream': stream}))
        ctx = trial_context('executor')
        turn = Turn(id='t', request_payload_ref=ctx.payloads.append(b'{"prompt":"sample text"}'),
                    input_kind='completion', requested_output_tokens=target)
        session = Session(turns=[turn], id="session-fixture-39")
        session.link()
        ctx.payloads.seal()
        runtime = await plugin.create_runtime(ExecutorTrial('trial'), stage_context('executor'))
        try:
            await runtime.prepare()
            prepared = await runtime.prepare_attempt(AttemptInput.capture(session, turn, {}, attempt=1, payloads=ctx.payloads))
            assert isinstance(prepared.body, PreparedBody)
            assert any(isinstance(s, memoryview) for s in prepared.body.segments)
            result = await runtime.execute_attempt(prepared, AttemptSignal(AttemptPlan()))
            assert result.status == 'succeeded'
            assert result.usage['completion_tokens'] == 2
            assert result.request.started_ns is not None
            assert result.request.response_mode == ('stream' if stream else 'non_stream')
            assert result.continuation_payload is None
            await runtime.release(prepared)
        finally:
            await runtime.close()
            ctx.payloads.close()
    assert received[0]['prompt'] == 'sample text'
    assert received[0]['temperature'] == 0.1
    if target is None:
        assert received[0]['max_tokens'] == 4096
        assert not {'max_completion_tokens', 'max_output_tokens', 'ignore_eos'} & received[0].keys()
    else:
        assert received[0]['max_tokens'] == 2 and received[0]['ignore_eos'] is True
    assert 'messages' not in received[0]


@pytest.mark.parametrize('defaults', [{'ignore_eos': False}, {'stop': ['end']}, {'n': 2},
                                      {'echo': True}, {'prompt': [1]}, {'stream_options': {'include_usage': False}}])
def test_completions_rejects_conflicting_output_contract(defaults):
    with pytest.raises(ValueError):
        validate({'base_url': 'http://localhost:8000', 'model': 'target', 'extra_body': defaults})
