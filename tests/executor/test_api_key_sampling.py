"""API key configuration, per-Attempt sampling and credential ownership."""

from collections import Counter
import json
import random
from types import SimpleNamespace

import pytest

from perf_benchmark.core.models import Session, Turn
from perf_benchmark.core.profiles import RequestProfile
from perf_benchmark.pipeline.contracts.executor import AttemptInput, ExecutorTrial
from perf_benchmark.plugins.executor import (
    openai_chat, openai_completions, openai_embeddings, openai_responses, rerank,
)
from perf_benchmark.runtime.http import request


PROTOCOLS = [
    (openai_chat, openai_chat.OpenAIChatExecutor, 'chat'),
    (openai_completions, openai_completions.OpenAICompletionsExecutor, 'completion'),
    (openai_embeddings, openai_embeddings.EmbeddingsExecutor, 'embedding'),
    (openai_responses, openai_responses.ResponsesExecutor, 'responses'),
    (rerank, rerank.RerankExecutor, 'rerank'),
]
BASE = {'base_url': 'http://localhost:8000', 'model': 'target'}
KEYS = [{'env': 'PERF_KEY_A'}, {'env': 'PERF_KEY_B'}]


@pytest.mark.parametrize('protocol,cls,kind', PROTOCOLS, ids=[p[2] for p in PROTOCOLS])
def test_default_is_unauthenticated_even_when_old_default_env_is_set(
    protocol, cls, kind, monkeypatch,
):
    monkeypatch.setenv('PERF_TARGET_API_KEY', 'unused-private-token')
    plugin = cls(protocol.validate(BASE))
    assert plugin.settings['api_keys'] == []
    assert plugin.settings['api_key_sampling'] == 'uniform'
    assert plugin.request_headers(plugin.settings) == {'Content-Type': 'application/json'}
    custom = cls(protocol.validate({**BASE, 'extra_headers': {'Authorization': 'custom'}}))
    assert custom.request_headers(custom.settings)['Authorization'] == 'custom'


@pytest.mark.parametrize('change', [
    {'api_key_sampling': 'round-robin'},
    {'api_key_sampling': None},
    {'api_keys': 'PERF_KEY_A'},
    {'api_keys': {}},
    {'api_keys': [None]},
    {'api_keys': ['PERF_KEY_A']},
    {'api_keys': [{}]},
    {'api_keys': [{'env': ''}]},
    {'api_keys': [{'env': '   '}]},
    {'api_keys': [{'env': 1}]},
    {'api_keys': [{'env': 'BAD=NAME'}]},
    {'api_keys': [{'env': 'BAD\0NAME'}]},
    {'api_keys': [{'env': 'PERF_KEY_A', 'key': 'raw-secret'}]},
    {'api_keys': [{'env': 'PERF_KEY_A'}, {'env': 'PERF_KEY_A'}]},
    {'api_keys': [{'env': 'PERF_KEY_A', 'weight': 1}]},
    {'api_key_sampling': 'weighted', 'api_keys': []},
    {'api_key_sampling': 'weighted', 'api_keys': [{'env': 'PERF_KEY_A', 'weight': 0}]},
    {'api_key_sampling': 'weighted', 'api_keys': [
        {'env': 'PERF_KEY_A', 'weight': 1e308}, {'env': 'PERF_KEY_B', 'weight': 1e308},
    ]},
])
def test_invalid_api_key_settings_fail_at_validation(change):
    with pytest.raises(ValueError):
        request.validate_api_keys(dict(change))


@pytest.mark.parametrize('weight', [True, False, '1', None, -1, float('nan'), float('inf'), -float('inf'), 10 ** 1000])
def test_weights_must_be_finite_nonnegative_real_numbers(weight):
    with pytest.raises(ValueError, match='weight'):
        request.validate_api_keys({'api_key_sampling': 'weighted',
                           'api_keys': [{'env': 'PERF_KEY_A', 'weight': weight}]})


@pytest.mark.parametrize('header', ['Authorization', 'authorization', 'aUtHoRiZaTiOn'])
def test_managed_credentials_reject_custom_authorization(header):
    with pytest.raises(ValueError, match='Authorization'):
        request.validate_api_keys({'api_keys': KEYS,
                           'extra_headers': {header: 'raw-secret'}})


@pytest.mark.parametrize('protocol,cls,kind', PROTOCOLS, ids=[p[2] for p in PROTOCOLS])
def test_protocols_apply_shared_credential_validation(protocol, cls, kind):
    with pytest.raises(ValueError, match='api_key_sampling'):
        protocol.validate({**BASE, 'api_key_sampling': 'round-robin'})


@pytest.mark.parametrize('key', [None, '', ' \t ', 'PRIVATE_TOKEN\rBAD', 'PRIVATE_TOKEN\nBAD'])
def test_missing_or_invalid_environment_values_fail_without_disclosure(
    key, monkeypatch,
):
    monkeypatch.setenv('PERF_KEY_A', 'valid-private-token')
    if key is None:
        monkeypatch.delenv('PERF_KEY_B', raising=False)
    else:
        monkeypatch.setenv('PERF_KEY_B', key)
    parsed = openai_chat.validate({**BASE, 'api_keys': KEYS})
    with pytest.raises(ValueError) as failure:
        openai_chat.OpenAIChatExecutor(parsed)
    assert 'valid-private-token' not in str(failure.value)
    assert 'PRIVATE_TOKEN' not in str(failure.value)


@pytest.mark.parametrize('sampling,weights,expected', [
    ('uniform', None, [1 / 3, 1 / 3, 1 / 3]),
    ('weighted', [1, 3, 0], [1 / 4, 3 / 4, 0]),
])
def test_seeded_sampling_follows_uniform_and_weighted_probabilities(
    sampling, weights, expected, monkeypatch,
):
    entries = []
    for index in range(3):
        name = f'PERF_KEY_{index}'
        monkeypatch.setenv(name, f'private-token-{index}')
        entry = {'env': name}
        if weights is not None:
            entry['weight'] = weights[index]
        entries.append(entry)
    plugin = openai_chat.OpenAIChatExecutor(openai_chat.validate(
        {**BASE, 'api_keys': entries, 'api_key_sampling': sampling}))
    monkeypatch.setattr(request, 'random', random.Random(20261014))
    draws = 12000
    counts = Counter(plugin.request_headers(plugin.settings)['Authorization'] for _ in range(draws))
    for index, probability in enumerate(expected):
        count = counts[f'Bearer private-token-{index}']
        if probability == 0:
            assert count == 0
        else:
            assert count / draws == pytest.approx(probability, abs=0.015)


def test_weighted_sampling_defaults_weights_and_accepts_small_positive_totals(monkeypatch):
    monkeypatch.setenv('PERF_KEY_A', 'private-token-a')
    monkeypatch.setenv('PERF_KEY_B', 'private-token-b')
    settings = openai_chat.validate({**BASE, 'api_key_sampling': 'weighted', 'api_keys': [
        {'env': 'PERF_KEY_A'}, {'env': 'PERF_KEY_B', 'weight': 0.5},
    ]})
    assert [entry['weight'] for entry in settings['api_keys']] == [1.0, 0.5]
    plugin = openai_chat.OpenAIChatExecutor(openai_chat.validate({**BASE,
        'api_key_sampling': 'weighted', 'api_keys': [{'env': 'PERF_KEY_A', 'weight': 5e-324}]}))
    assert plugin.request_headers(plugin.settings)['Authorization'] == 'Bearer private-token-a'


def test_credentials_are_resolved_once_and_never_inserted_into_settings(
    monkeypatch,
):
    monkeypatch.setenv('PERF_KEY_A', 'private-original-a')
    monkeypatch.setenv('PERF_KEY_B', 'private-original-b')
    plugin = openai_chat.OpenAIChatExecutor(openai_chat.validate({**BASE, 'api_keys': KEYS}))
    monkeypatch.setenv('PERF_KEY_A', 'private-replacement-a')
    monkeypatch.delenv('PERF_KEY_B')
    selections = iter([0, 1])
    monkeypatch.setattr(request, 'random', SimpleNamespace(choice=lambda keys: keys[next(selections)]))
    assert plugin.request_headers(plugin.settings)['Authorization'] == 'Bearer private-original-a'
    assert plugin.request_headers(plugin.settings)['Authorization'] == 'Bearer private-original-b'
    settings = plugin.settings_for_trial(ExecutorTrial('trial'))
    for secret in ('private-original-a', 'private-original-b', 'private-replacement-a'):
        assert secret not in str(plugin.settings)
        assert secret not in str(settings)
    with pytest.raises(ValueError, match='Authorization'):
        plugin.request_headers({**settings, 'extra_headers': {'authorization': 'override'}})


@pytest.mark.parametrize('protocol,cls,kind', PROTOCOLS, ids=[p[2] for p in PROTOCOLS])
async def test_every_protocol_samples_each_attempt_independently(
    protocol, cls, kind, trial_context, monkeypatch,
):
    monkeypatch.setenv('PERF_KEY_A', 'private-token-a')
    monkeypatch.setenv('PERF_KEY_B', 'private-token-b')
    selections = iter([0, 1, 0])
    monkeypatch.setattr(request, 'random', SimpleNamespace(choice=lambda keys: keys[next(selections)]))
    ctx = trial_context('executor')
    payload = {
        'chat': {'messages': [{'role': 'user', 'content': 'q'}]},
        'completion': {'prompt': 'q'},
        'embedding': {'input': ['q']},
        'responses': {'input': 'q'},
        'rerank': {'query': 'q', 'documents': ['d']},
    }[kind]
    turn = Turn(id='turn', input_kind=kind,
                request_payload_ref=ctx.payloads.append(json.dumps(payload).encode()),
                requested_output_tokens=2, profile=RequestProfile(input_items=1))
    session = Session(id='session', turns=[turn])
    ctx.payloads.seal()
    plugin = cls(protocol.validate({**BASE, 'api_keys': KEYS}))
    runtime = await plugin.create_runtime(ExecutorTrial('trial'), ctx)
    try:
        headers = []
        for attempt in (1, 2, 3):
            value = AttemptInput.capture(session, turn, {}, attempt=attempt, payloads=ctx.payloads)
            prepared = await runtime.prepare_attempt(value)
            try:
                headers.append(prepared.spec.headers['Authorization'])
                assert prepared.spec.headers['Content-Type'] == 'application/json'
                assert all(secret not in str(prepared.request_facts)
                           for secret in ('private-token-a', 'private-token-b'))
            finally:
                await runtime.release(prepared)
        assert headers == ['Bearer private-token-a', 'Bearer private-token-b', 'Bearer private-token-a']
    finally:
        await runtime.close()
        await ctx.events.close()
