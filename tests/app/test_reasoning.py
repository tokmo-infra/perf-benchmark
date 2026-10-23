"""Pre-measurement reasoning negotiation and explicit workload precedence."""
import asyncio
from copy import deepcopy
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from perf_benchmark.app.reasoning import resolve, _unsupported
from perf_benchmark.runtime.cancellation import CancellationToken


def config(url, *, extra=None, protocol='openai-chat', **settings):
    return {'pipeline': {'executor': {'plugin': protocol, 'settings': {
        'base_url': url, 'model': 'mock', 'extra_body': extra or {}, **settings,
    }}}}


@pytest.mark.parametrize('accepted,expected', [
    ('max', ['max']), ('xhigh', ['max', 'xhigh']), ('high', ['max', 'xhigh', 'high']),
])
@pytest.mark.parametrize('protocol', ['openai-chat', 'openai-responses'])
async def test_default_negotiates_once_and_freezes_actual_effort(accepted, expected, protocol):
    requests = []

    async def handle(request):
        body = await request.json()
        effort = body.get('reasoning_effort') or body['reasoning']['effort']
        requests.append(effort)
        assert body['stream'] is False
        if effort != accepted:
            return web.json_response({'error': {'param': 'reasoning_effort',
                'message': f'Unsupported reasoning_effort: {effort}'}}, status=400)
        return web.json_response({'choices': []})

    app = web.Application()
    app.router.add_post('/v1/chat/completions', handle)
    app.router.add_post('/v1/responses', handle)
    async with TestServer(app) as server:
        original = config(str(server.make_url('')).rstrip('/'), protocol=protocol)
        snapshot = deepcopy(original)
        resolved, evidence = await resolve(original)
        assert original == snapshot
        extra = resolved['pipeline']['executor']['settings']['extra_body']
        assert (extra['reasoning_effort'] if protocol == 'openai-chat'
                else extra['reasoning']['effort']) == accepted
        assert evidence['selected'] == accepted and evidence['mode'] == 'auto'
        assert requests == expected
        assert [p['effort'] for p in evidence['probes']] == expected
        # Reusing resolved configuration needs no capability traffic.
        same, explicit = await resolve(resolved)
        assert same == resolved and explicit['mode'] == 'explicit'
        assert requests == expected


@pytest.mark.parametrize('extra', [
    {'reasoning_effort': 'low'}, {'reasoning_effort': 'max'},
    {'reasoning_effort': None}, {'thinking': {'enabled': False}},
    {'enable_thinking': True}, {'chat_template_kwargs': {'enable_thinking': False}},
])
async def test_explicit_configuration_avoids_network(extra):
    original = config('http://127.0.0.1:1', extra=extra)
    resolved, fact = await resolve(original)
    assert fact['mode'].startswith('explicit') and fact['probes'] == []
    actual = resolved['pipeline']['executor']['settings']['extra_body']
    assert actual == ({} if extra == {'reasoning_effort': None} else extra)
    assert original['pipeline']['executor']['settings']['extra_body'] == extra


@pytest.mark.parametrize('effort', ['low', None])
async def test_explicit_responses_configuration_preserves_other_reasoning_options(effort):
    value, fact = await resolve(config('http://127.0.0.1:1', protocol='openai-responses',
        extra={'reasoning': {'effort': effort, 'summary': 'auto'}}))
    expected = {'summary': 'auto', **({'effort': effort} if effort else {})}
    assert value['pipeline']['executor']['settings']['extra_body']['reasoning'] == expected
    assert fact['mode'] == 'explicit'


@pytest.mark.parametrize('protocol', ['openai-completions', 'embedding', 'rerank', None])
async def test_other_protocols_keep_original_settings(protocol):
    original = config('http://127.0.0.1:1', protocol=protocol)
    value, fact = await resolve(original)
    assert value is original and fact is None


@pytest.mark.parametrize('status', [400, 401, 403, 429, 500, 503])
async def test_unrelated_service_errors_do_not_downgrade(status):
    calls = []

    async def handle(request):
        calls.append(await request.json())
        return web.json_response({'error': {'message': 'unrelated service failure'}}, status=status)

    app = web.Application()
    app.router.add_post('/v1/chat/completions', handle)
    async with TestServer(app) as server:
        with pytest.raises(ValueError, match='capability probe failed'):
            await resolve(config(str(server.make_url('')).rstrip('/')))
    assert len(calls) == 1


async def test_multiple_endpoints_choose_highest_common_effort_and_pass_auth(monkeypatch):
    calls = []
    monkeypatch.setenv('TEST_REASONING_KEY', 'private-test-credential')

    async def handle(request):
        body = await request.json()
        effort = body['reasoning_effort']
        calls.append((request.path, effort))
        assert request.headers['Authorization'] == 'Bearer private-test-credential'
        if request.path.startswith('/second') and effort != 'high':
            return web.json_response({'error': {'message': f'reasoning_effort {effort} is not supported'}}, status=422)
        return web.json_response({'choices': []})

    app = web.Application()
    app.router.add_post('/first/v1/chat/completions', handle)
    app.router.add_post('/second/v1/chat/completions', handle)
    async with TestServer(app) as server:
        base = str(server.make_url('')).rstrip('/')
        original = config(base, api_keys=[{'env': 'TEST_REASONING_KEY'}])
        settings = original['pipeline']['executor']['settings']
        settings.pop('base_url')
        settings['base_urls'] = [base + '/first', base + '/second']
        resolved, fact = await resolve(original)
    assert resolved['pipeline']['executor']['settings']['extra_body']['reasoning_effort'] == 'high'
    assert len(calls) == 6
    assert 'private-test-credential' not in json.dumps(fact)


async def test_all_three_unsupported_stops_preparation():
    calls = []

    async def handle(request):
        body = await request.json()
        calls.append(body['reasoning_effort'])
        return web.json_response({'detail': [{'loc': ['body', 'reasoning_effort'],
            'type': 'literal_error', 'msg': 'Input should be low', 'input': body['reasoning_effort']}]}, status=422)

    app = web.Application()
    app.router.add_post('/v1/chat/completions', handle)
    async with TestServer(app) as server:
        with pytest.raises(ValueError, match='max, xhigh and high'):
            await resolve(config(str(server.make_url('')).rstrip('/')))
    assert calls == ['max', 'xhigh', 'high']


async def test_capability_probe_is_bounded_and_cancellable():
    started = asyncio.Event()

    async def handle(request):
        started.set()
        await asyncio.sleep(.1)
        return web.Response(body=b'x' * (64 * 1024 + 1))

    app = web.Application()
    app.router.add_post('/v1/chat/completions', handle)
    async with TestServer(app) as server:
        original = config(str(server.make_url('')).rstrip('/'))
        token = CancellationToken()
        token.cancel('already-cancelled')
        with pytest.raises(asyncio.CancelledError):
            await resolve(original, token)
        assert not started.is_set()
        token = CancellationToken()
        task = asyncio.create_task(resolve(original, token))
        await started.wait()
        token.cancel('cancel-probe')
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(ValueError, match='64 KiB'):
            await resolve(original)


@pytest.mark.parametrize('status', [200, 400, 422, 500])
def test_explicit_effort_rejections_are_recognized(status):
    assert _unsupported(status, json.dumps({'error': {
        'message': 'Invalid reasoning_effort max; supported values: high'}}).encode(), 'max')
    assert not _unsupported(401, b'{"error":{"message":"unsupported reasoning_effort max"}}', 'max')
    assert not _unsupported(200, b'{"choices":[{"message":{"content":"invalid reasoning_effort max"}}]}', 'max')
    assert not _unsupported(400, b'{"error":{"param":"max_tokens","message":"invalid max_tokens","request":{"reasoning_effort":"max"}}}', 'max')
    assert not _unsupported(400, b'{"error":{"message":"max_tokens must be at least 16 for this reasoning model"}}', 'max')


async def test_application_freezes_selected_effort_in_conditions_and_excel_parameters(monkeypatch, tmp_path):
    from perf_benchmark.app import main
    from perf_benchmark.app.evidence import conditions

    async def fake_execute(self, config, **kwargs):
        extra = config['pipeline']['executor']['settings']['extra_body']
        assert extra['reasoning_effort'] == 'xhigh'
        assert kwargs['evidence']['conditions']['generation']['value']['reasoning_effort'] == 'xhigh'
        assert kwargs['evidence']['reasoning']['selected'] == 'xhigh'
        return 'finished'

    async def handle(request):
        body = await request.json()
        if body['reasoning_effort'] == 'max':
            return web.json_response({'error': {'message': 'unsupported reasoning_effort max'}}, status=400)
        return web.json_response({'choices': []})

    app = web.Application()
    app.router.add_post('/v1/chat/completions', handle)
    monkeypatch.setattr(main.Coordinator, 'execute', fake_execute)
    async with TestServer(app) as server:
        original = config(str(server.make_url('')).rstrip('/'))
        original.update(case='default reasoning preparation', application={'service_capacity': {'enabled': False}})
        original['pipeline']['normalization'] = {'plugin': 'sharegpt'}
        assert conditions(original, {})['conditions']['generation']['value'] == {}
        assert await main.execute_config(original, tmp_path) == 'finished'
        assert original['pipeline']['executor']['settings']['extra_body'] == {}
