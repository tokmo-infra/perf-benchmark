"""Reject invalid configuration and payloads before creating native execution resources."""
import json

import pytest

from perf_benchmark.core.models import Session, Turn
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


@pytest.mark.parametrize('protocol,cls,kind', PROTOCOLS, ids=[p[2] for p in PROTOCOLS])
@pytest.mark.parametrize('change', [
    {'unknown': True}, {'base_url': ''}, {'model': ''}, {'base_url': 'file:///tmp/model'},
    {'connection_scope': 'shared'}, {'verify_ssl': 'false'}, {'timeout_seconds': 0},
    {'extra_headers': []}, {'api_key_env': 'PERF_TARGET_API_KEY'}, {'extra_body': []},
    {'headers': {}}, {'request_defaults': {}},
])
def test_invalid_settings_fail_before_runtime(protocol, cls, kind, change):
    with pytest.raises((TypeError, ValueError)):
        protocol.validate({**BASE, **change})


@pytest.mark.parametrize('protocol,cls,kind', PROTOCOLS[1:], ids=[p[2] for p in PROTOCOLS[1:]])
@pytest.mark.parametrize('change', [
    {'base_url': 'http://user:secret@localhost'}, {'base_url': 'http://localhost?secret=x'},
    {'extra_headers': {'bad': 1}},
])
def test_snapshot_settings_reject_url_credentials_and_nonstring_headers(protocol, cls, kind, change):
    with pytest.raises(ValueError):
        protocol.validate({**BASE, **change})


@pytest.mark.parametrize('protocol,change', [
    (openai_chat, {'endpoint': ''}), (openai_chat, {'stream': 1}),
    (openai_completions, {'stream': 1}), (openai_responses, {'stream': 1}),
    (openai_completions, {'max_tokens': 0}),
    (openai_completions, {'extra_body': {'max_tokens': 5}}),
    (openai_completions, {'extra_body': {'temperature': -1}}),
    (openai_embeddings, {'extra_body': {'dimensions': 0}}),
    (openai_embeddings, {'extra_body': {'encoding_format': 'base64'}}),
    (openai_responses, {'extra_body': {'max_output_tokens': 0}}),
    (openai_responses, {'extra_body': {'temperature': -1}}),
    (openai_responses, {'extra_body': {'parallel_tool_calls': 1}}),
    (openai_responses, {'extra_body': {'tool_choice': 'unknown'}}),
    (openai_responses, {'extra_body': {'store': True}}),
    (rerank, {'extra_body': {'top_n': 0}}),
    (rerank, {'extra_body': {'return_documents': True}}),
])
def test_protocol_parameters_keep_their_owned_fields_and_types(protocol, change):
    with pytest.raises((TypeError, ValueError)):
        protocol.validate({**BASE, **change})


@pytest.mark.parametrize('protocol,cls,kind', PROTOCOLS[1:], ids=[p[2] for p in PROTOCOLS[1:]])
@pytest.mark.parametrize('fault', ['wrong_kind', 'non_object', 'oversized_parameters', 'auth'])
async def test_snapshot_preparation_contracts_and_auth(protocol, cls, kind, fault, trial_context, monkeypatch):
    ctx = trial_context('executor')
    payload = b'[]' if fault == 'non_object' else b'{"input":"x","prompt":"sample"}'
    turn = Turn(id='turn', input_kind='chat' if fault == 'wrong_kind' else kind,
                request_payload_ref=ctx.payloads.append(payload), requested_output_tokens=2,
                profile=RequestProfile(input_items=2))
    session = Session(turns=[turn], id="session-fixture-74")
    ctx.payloads.seal()
    monkeypatch.setenv('PERF_TARGET_API_KEY', 'test-secret')
    instance = cls(protocol.validate({**BASE, 'api_keys': [{'env': 'PERF_TARGET_API_KEY'}]}))
    runtime = await instance.create_runtime(ExecutorTrial('trial'), ctx)
    value = AttemptInput.capture(session, turn, {}, attempt=1, payloads=ctx.payloads)
    try:
        if fault == 'oversized_parameters':
            build = instance.build_request
            def oversized(*args):
                spec = build(*args)
                spec.parameters['test_padding'] = 'x' * 65536
                return spec
            instance.build_request = oversized
        if fault == 'auth':
            prepared = await runtime.prepare_attempt(value)
            assert prepared.spec.headers['Authorization'] == 'Bearer test-secret'
            assert json.loads(b''.join(prepared.body.segments))['model'] == 'target'
        else:
            with pytest.raises(ValueError, match='snapshot|JSON object|64 KiB'):
                await runtime.prepare_attempt(value)
        assert runtime.native is None
    finally:
        await runtime.close()
        await ctx.events.close()


async def test_rerank_top_n_cannot_exceed_input(trial_context):
    ctx = trial_context('executor')
    turn = Turn(request_payload_ref=ctx.payloads.append(b'{}'), input_kind='rerank', profile=RequestProfile(input_items=1), id="turn-fixture-103")
    instance = rerank.RerankExecutor(rerank.validate({**BASE, 'extra_body': {'top_n': 2}}))
    value = AttemptInput.capture(Session(turns=[turn], id="session-fixture-105"), turn, {}, attempt=1, payloads=ctx.payloads)
    with pytest.raises(ValueError, match='exceeds document count'):
        instance.build_request(value, instance.settings)
    await ctx.events.close()


@pytest.mark.parametrize('protocol,cls,kind', PROTOCOLS, ids=[p[2] for p in PROTOCOLS])
async def test_extra_fields_are_added_to_all_protocols(protocol, cls, kind, trial_context):
    ctx = trial_context('executor')
    payloads = {
        'chat': {'messages': [{'role': 'user', 'content': 'q'}]},
        'completion': {'prompt': 'q'},
        'embedding': {'input': ['q']},
        'responses': {'input': 'q'},
        'rerank': {'query': 'q', 'documents': ['d']},
    }
    turn = Turn(input_kind=kind, request_payload_ref=ctx.payloads.append(
        json.dumps(payloads[kind]).encode()), requested_output_tokens=2,
        profile=RequestProfile(input_items=1), id="turn-fixture-121")
    ctx.payloads.seal()
    settings = {**BASE, 'extra_headers': {'X-Custom': 'value'},
                'extra_body': {'vendor_option': {'enabled': True}}}
    plugin = cls(protocol.validate(settings))
    trial_settings = plugin.settings_for_trial(ExecutorTrial('trial'))
    spec = plugin.build_request(AttemptInput.capture(
        Session(turns=[turn], id="session-fixture-130"), turn, {}, attempt=1, payloads=ctx.payloads), trial_settings)
    body = await spec.build_body(spec)
    try:
        sent = json.loads(b''.join(body.segments))
        assert sent['vendor_option'] == {'enabled': True}
        assert sent['model'] == 'target'
        if kind in {'embedding','rerank'}:
            assert 'temperature' not in sent
        assert spec.headers['X-Custom'] == 'value'
        assert spec.headers['Content-Type'] == 'application/json'
    finally:
        body.release()
        ctx.payloads.close()
        await ctx.events.close()


@pytest.mark.parametrize('protocol,cls,kind', PROTOCOLS, ids=[p[2] for p in PROTOCOLS])
@pytest.mark.parametrize('extra_headers', [
    {'content-type': 'text/plain'}, {'X-Token': 'a', 'x-token': 'b'},
])
def test_extra_header_conflicts_fail_at_configuration(protocol, cls, kind, extra_headers):
    with pytest.raises(ValueError, match='extra_headers'):
        protocol.validate({**BASE, 'extra_headers': extra_headers})


@pytest.mark.parametrize('protocol,cls,kind', PROTOCOLS, ids=[p[2] for p in PROTOCOLS])
def test_extra_header_cannot_override_generated_authorization(
    protocol, cls, kind, monkeypatch,
):
    monkeypatch.setenv('PERF_TARGET_API_KEY', 'secret')
    with pytest.raises(ValueError, match='Authorization'):
        protocol.validate({**BASE, 'api_keys': [{'env': 'PERF_TARGET_API_KEY'}],
                           'extra_headers': {'authorization': 'other'}})


@pytest.mark.parametrize('protocol,cls,kind', PROTOCOLS, ids=[p[2] for p in PROTOCOLS])
def test_generated_body_field_cannot_be_overridden(protocol, cls, kind):
    with pytest.raises(ValueError, match='generated fields'):
        protocol.validate({**BASE, 'extra_body': {'model': 'other'}})


async def test_dataset_body_field_cannot_be_overridden(trial_context):
    ctx = trial_context('executor')
    turn = Turn(input_kind='responses', request_payload_ref=ctx.payloads.append(
        b'{"input":"q","tools":[]}'), profile=RequestProfile(input_items=1), id="turn-fixture-172")
    ctx.payloads.seal()
    plugin = openai_responses.ResponsesExecutor(openai_responses.validate(
        {**BASE, 'extra_body': {'tools': [{'type': 'function'}]}}))
    settings = plugin.settings_for_trial(ExecutorTrial('trial'))
    spec = plugin.build_request(AttemptInput.capture(
        Session(turns=[turn], id="session-fixture-179"), turn, {}, attempt=1, payloads=ctx.payloads), settings)
    with pytest.raises(ValueError, match='dataset fields: tools'):
        await spec.build_body(spec)
    ctx.payloads.close()
    await ctx.events.close()


@pytest.mark.parametrize('requested,expected', [(None, 11), (7, 7)])
async def test_completions_output_budget_prefers_turn(requested, expected, trial_context):
    ctx = trial_context('executor')
    turn = Turn(input_kind='completion', request_payload_ref=ctx.payloads.append(b'{"prompt":"q"}'),
                requested_output_tokens=requested, id="turn-fixture-189")
    ctx.payloads.seal()
    plugin = openai_completions.OpenAICompletionsExecutor(openai_completions.validate(
        {**BASE, 'max_tokens': 11}))
    settings = plugin.settings_for_trial(ExecutorTrial('trial'))
    spec = plugin.build_request(AttemptInput.capture(
        Session(turns=[turn], id="session-fixture-196"), turn, {}, attempt=1, payloads=ctx.payloads), settings)
    body = await spec.build_body(spec)
    try:
        assert json.loads(b''.join(body.segments))['max_tokens'] == expected
        assert spec.request_facts.requested_output_tokens == expected
    finally:
        body.release()
        ctx.payloads.close()
        await ctx.events.close()


@pytest.mark.parametrize('requested,configured,expected', [
    (None, None, 4096), (None, 12000, 12000), (7000, 12000, 7000),
])
async def test_responses_output_cap_keeps_explicit_precedence(requested, configured, expected, trial_context):
    ctx = trial_context('executor')
    turn = Turn(input_kind='responses', request_payload_ref=ctx.payloads.append(b'{"input":"q"}'),
                requested_output_tokens=requested, id='responses-cap')
    ctx.payloads.seal()
    extra = {} if configured is None else {'max_output_tokens': configured}
    plugin = openai_responses.ResponsesExecutor(openai_responses.validate({**BASE, 'extra_body': extra}))
    spec = plugin.build_request(AttemptInput.capture(Session(turns=[turn], id='session-cap'),
        turn, {}, attempt=1, payloads=ctx.payloads), plugin.settings_for_trial(ExecutorTrial('trial')))
    body = await spec.build_body(spec)
    try:
        payload = json.loads(b''.join(body.segments))
        assert payload['max_output_tokens'] == expected
        assert 'ignore_eos' not in payload
        assert spec.request_facts.requested_output_tokens == expected
    finally:
        body.release()
        ctx.payloads.close()
        await ctx.events.close()


@pytest.mark.parametrize('protocol,cls,kind', [p for p in PROTOCOLS if p[2] in {'chat','completion','responses'}])
@pytest.mark.parametrize('configured', [None,0.0,0.7])
@pytest.mark.parametrize('override', [None,0.0,0.2])
async def test_generation_temperature_default_and_explicit_precedence(
    protocol,cls,kind,configured,override,trial_context,
):
    ctx=trial_context('executor')
    payload={'chat':{'messages':[{'role':'user','content':'q'}]},
             'completion':{'prompt':'q'},'responses':{'input':'q'}}[kind]
    turn=Turn(id='temperature-turn',input_kind=kind,
              request_payload_ref=ctx.payloads.append(json.dumps(payload).encode()))
    ctx.payloads.seal()
    extra={} if configured is None else {'temperature':configured}
    plugin=cls(protocol.validate({**BASE,'extra_body':extra}))
    overrides={} if override is None else {'temperature':override}
    settings=plugin.settings_for_trial(ExecutorTrial('temperature-trial',overrides))
    spec=plugin.build_request(AttemptInput.capture(Session(turns=[turn],id='temperature-session'),
        turn,{},attempt=1,payloads=ctx.payloads),settings)
    body=await spec.build_body(spec)
    try:
        expected=override if override is not None else configured if configured is not None else 0.1
        assert json.loads(b''.join(body.segments))['temperature']==expected
        assert spec.request_facts.profile.temperature==expected
        assert extra==({} if configured is None else {'temperature':configured})
    finally:
        body.release()
        ctx.payloads.close()
        await ctx.events.close()
