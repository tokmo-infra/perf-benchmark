"""Per-record headers cross the HTTP boundary without entering request facts."""

import json

import pytest

from perf_benchmark.core.models import Session, Turn
from perf_benchmark.core.observation import TraceContext
from perf_benchmark.core.profiles import RequestProfile
from perf_benchmark.pipeline.contracts.executor import AttemptInput, ExecutorTrial
from perf_benchmark.plugins.executor import (
    openai_chat, openai_completions, openai_embeddings, openai_responses, rerank,
)


PROTOCOLS = [
    (openai_chat, openai_chat.OpenAIChatExecutor, 'chat'),
    (openai_completions, openai_completions.OpenAICompletionsExecutor, 'completion'),
    (openai_embeddings, openai_embeddings.EmbeddingsExecutor, 'embedding'),
    (openai_responses, openai_responses.ResponsesExecutor, 'responses'),
    (rerank, rerank.RerankExecutor, 'rerank'),
]
BASE = {'base_url': 'http://localhost:8000', 'model': 'target'}
PAYLOADS = {
    'chat': {'messages': [{'role': 'user', 'content': 'q'}]},
    'completion': {'prompt': 'q'},
    'embedding': {'input': ['q']},
    'responses': {'input': 'q'},
    'rerank': {'query': 'q', 'documents': ['d']},
}


def _turn(ctx, kind, headers):
    return Turn(id='turn', input_kind=kind,
                request_payload_ref=ctx.payloads.append(json.dumps(PAYLOADS[kind]).encode()),
                extra_headers_payload_ref=ctx.payloads.append(json.dumps(headers).encode()),
                requested_output_tokens=2, profile=RequestProfile(input_items=1))


@pytest.mark.parametrize('protocol,cls,kind', PROTOCOLS, ids=[p[2] for p in PROTOCOLS])
async def test_every_protocol_merges_dataset_and_global_headers_without_body_leaks(
    protocol, cls, kind, trial_context,
):
    ctx = trial_context('executor')
    turn = _turn(ctx, kind, {'X-Request-Token': 'PRIVATE_DATASET_HEADER', 'X-Row': 'row-1'})
    session = Session(id='session', turns=[turn])
    ctx.payloads.seal()
    plugin = cls(protocol.validate({**BASE, 'extra_headers': {'X-Global': 'global'}}))
    runtime = await plugin.create_runtime(ExecutorTrial('trial'), ctx)
    prepared = None
    try:
        value = AttemptInput.capture(session, turn, {}, attempt=1, payloads=ctx.payloads)
        prepared = await runtime.prepare_attempt(value)
        assert prepared.spec.headers == {
            'Content-Type': 'application/json', 'X-Global': 'global',
            'X-Request-Token': 'PRIVATE_DATASET_HEADER', 'X-Row': 'row-1',
        }
        body = json.loads(b''.join(prepared.body.segments))
        assert 'headers' not in body and 'extra_headers' not in body
        for safe in (body, plugin.settings, runtime.settings, prepared.request_facts,
                     turn.metadata, value.turn.metadata, value):
            assert 'PRIVATE_DATASET_HEADER' not in str(safe)
    finally:
        if prepared is not None:
            await runtime.release(prepared)
        await runtime.close()
        await ctx.events.close()


@pytest.mark.parametrize('global_name,dataset_name', [
    ('X-Token', 'X-Token'), ('X-Token', 'x-token'), ('x-token', 'X-TOKEN'),
])
async def test_dataset_headers_cannot_override_global_header_values(
    trial_context, global_name, dataset_name,
):
    ctx = trial_context('executor')
    turn = _turn(ctx, 'chat', {dataset_name: 'PRIVATE_DATASET_HEADER'})
    ctx.payloads.seal()
    plugin = openai_chat.OpenAIChatExecutor(openai_chat.validate(
        {**BASE, 'extra_headers': {global_name: 'PRIVATE_GLOBAL_HEADER'}}))
    value = AttemptInput.capture(Session(id='s', turns=[turn]), turn, {}, attempt=1, payloads=ctx.payloads)
    try:
        with pytest.raises(ValueError) as failure:
            plugin.build_request(value, plugin.settings)
        assert 'PRIVATE_' not in str(failure.value)
    finally:
        await ctx.events.close()


@pytest.mark.parametrize('headers', [
    {'Host': 'PRIVATE_HEADER'}, {'content-type': 'PRIVATE_HEADER'},
    {'CONTENT-LENGTH': 'PRIVATE_HEADER'}, {'X-A': 'PRIVATE_HEADER', 'x-a': 'PRIVATE_HEADER'},
    {'X-A': 'PRIVATE_HEADER\r\nInjected'}, {'Bad Name': 'PRIVATE_HEADER'},
    {'X-A': 1}, ['PRIVATE_HEADER'],
])
async def test_executor_revalidates_header_payloads_without_disclosing_values(trial_context, headers):
    ctx = trial_context('executor')
    turn = _turn(ctx, 'chat', headers)
    ctx.payloads.seal()
    plugin = openai_chat.OpenAIChatExecutor(openai_chat.validate(BASE))
    value = AttemptInput.capture(Session(id='s', turns=[turn]), turn, {}, attempt=1, payloads=ctx.payloads)
    try:
        with pytest.raises((TypeError, ValueError)) as failure:
            plugin.build_request(value, plugin.settings)
        assert 'PRIVATE_HEADER' not in str(failure.value)
    finally:
        await ctx.events.close()


@pytest.mark.parametrize('managed_auth', [False, True])
async def test_dataset_authorization_only_allowed_without_managed_api_keys(
    trial_context, monkeypatch, managed_auth,
):
    ctx = trial_context('executor')
    turn = _turn(ctx, 'chat', {'authorization': 'Bearer PRIVATE_DATASET_KEY'})
    ctx.payloads.seal()
    settings = dict(BASE)
    if managed_auth:
        monkeypatch.setenv('PERF_MANAGED_KEY', 'PRIVATE_MANAGED_KEY')
        settings['api_keys'] = [{'env': 'PERF_MANAGED_KEY'}]
    plugin = openai_chat.OpenAIChatExecutor(openai_chat.validate(settings))
    value = AttemptInput.capture(Session(id='s', turns=[turn]), turn, {}, attempt=1, payloads=ctx.payloads)
    try:
        if managed_auth:
            with pytest.raises(ValueError, match='Authorization') as failure:
                plugin.build_request(value, plugin.settings)
            assert 'PRIVATE_' not in str(failure.value)
        else:
            spec = plugin.build_request(value, plugin.settings)
            assert spec.headers['authorization'] == 'Bearer PRIVATE_DATASET_KEY'
    finally:
        await ctx.events.close()


@pytest.mark.parametrize('header', ['traceparent', 'TraceState'])
@pytest.mark.parametrize('tracing', [False, True])
async def test_dataset_trace_headers_conflict_only_when_managed_trace_is_active(
    trial_context, header, tracing,
):
    ctx = trial_context('executor')
    turn = _turn(ctx, 'chat', {header: 'PRIVATE_DATASET_TRACE'})
    ctx.payloads.seal()
    plugin = openai_chat.OpenAIChatExecutor(openai_chat.validate(BASE))
    runtime = await plugin.create_runtime(ExecutorTrial('trial'), ctx)
    trace = TraceContext('1' * 32, '2' * 16, '3' * 16, False) if tracing else None
    value = AttemptInput.capture(Session(id='s', turns=[turn]), turn, {}, attempt=1,
                                 payloads=ctx.payloads, trace_context=trace)
    try:
        if tracing:
            with pytest.raises(ValueError, match='tracing') as failure:
                await runtime.prepare_attempt(value)
            assert 'PRIVATE_DATASET_TRACE' not in str(failure.value)
        else:
            prepared = await runtime.prepare_attempt(value)
            try:
                assert prepared.spec.headers[header] == 'PRIVATE_DATASET_TRACE'
            finally:
                await runtime.release(prepared)
    finally:
        await runtime.close()
        await ctx.events.close()
