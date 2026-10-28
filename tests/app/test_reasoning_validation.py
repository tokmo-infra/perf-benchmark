"""Negotiate reasoning effort from protocol validation across model templates."""
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from perf_benchmark.app.reasoning import _unsupported, resolve


@pytest.mark.parametrize('status,message', [
    (200, 'Unexpected reasoning effort max. Supported types are xhigh (default), medium, and low.'),
    (400, 'Unexpected reasoning effort max. Supported types are xhigh (default), medium, and low.'),
    (422, "Unexpected value 'max' for reasoning_effort. Supported values: xhigh, high."),
    (500, 'reasoning.effort: unexpected max; expected one of xhigh, high.'),
])
def test_template_validation_rejects_only_the_attempted_effort(status, message):
    data = json.dumps({'error': {'message': message}}).encode()
    assert _unsupported(status, data, 'max')
    assert not _unsupported(status, data, 'xhigh')
    assert not _unsupported(401, data, 'max')
    assert not _unsupported(429, data, 'max')
    assert not _unsupported(500, b'{"error":{"message":"CUDA illegal memory access for reasoning effort max"}}', 'max')


@pytest.mark.parametrize('protocol,accepted', [
    ('openai-chat', 'xhigh'), ('openai-responses', 'high'),
])
async def test_template_validation_negotiates_supported_effort(accepted, protocol):
    efforts = []

    async def handle(request):
        body = await request.json()
        effort = body.get('reasoning_effort') or body['reasoning']['effort']
        efforts.append(effort)
        if effort != accepted:
            return web.json_response({'error': {'message':
                f"Unexpected value '{effort}' for reasoning_effort. Supported values: {accepted}."}}, status=400)
        return web.json_response({'choices': []})

    app = web.Application()
    app.router.add_post('/v1/chat/completions', handle)
    app.router.add_post('/v1/responses', handle)
    async with TestServer(app) as server:
        config = {'pipeline': {'executor': {'plugin': protocol, 'settings': {
            'base_url': str(server.make_url('')).rstrip('/'), 'model': 'arbitrary-model',
        }}}}
        resolved, evidence = await resolve(config)
    expected = ['max', 'xhigh'] if accepted == 'xhigh' else ['max', 'xhigh', 'high']
    assert efforts == expected
    extra = resolved['pipeline']['executor']['settings']['extra_body']
    assert (extra['reasoning_effort'] if protocol == 'openai-chat'
            else extra['reasoning']['effort']) == accepted
    assert evidence['selected'] == accepted
    assert [probe['accepted'] for probe in evidence['probes']] == [False] * (len(expected) - 1) + [True]


@pytest.mark.parametrize('data,effort', [
    (b'{"error":{"message":"Unexpected reasoning_effort maximum. Supported values: high"}}', 'max'),
    (b'{"error":{"message":"Unexpected reasoning_effort xhigh. Supported types: medium"}}', 'high'),
    (b'{"error":{"message":"Unexpected CUDA error with reasoning_effort max"}}', 'max'),
    (b'{"error":{"param":"max_tokens","message":"Unexpected max for reasoning_effort. Supported values: high"}}', 'max'),
])
def test_negotiation_preserves_unrelated_errors_and_effort_boundaries(data, effort):
    assert not _unsupported(500, data, effort)
