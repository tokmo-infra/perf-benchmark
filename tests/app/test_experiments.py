import asyncio
import json
from types import SimpleNamespace

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest
import yaml

from perf_benchmark.app.evidence import conditions, quality
from perf_benchmark.app.preparation import ServicePreparation, PreparationCancelled
from perf_benchmark.app import experiments
from perf_benchmark.app.results import create_app
from perf_benchmark.core.models import freeze_snapshot, Session, Turn
from perf_benchmark.pipeline.preparation.payloads import PayloadRef
from perf_benchmark.pipeline.contracts.executor import AttemptResult, ExecutorTrial
from perf_benchmark.runtime.cancellation import CancellationToken


def config():
    return {'pipeline': {'loader': {'plugin': 'hf-datasets', 'settings': {'dataset': 'json', 'data_files': ['local.jsonl'], 'token': 'PRIVATE'}},
        'normalization': {'plugin': 'openqa', 'settings': {}}, 'selector': {'plugin': 'all', 'settings': {}},
        'executor': {'plugin': 'openai-completions', 'settings': {'model': 'model', 'base_url': 'http://localhost',
                    'extra_headers': {'X-Private': 'PRIVATE'}, 'extra_body': {'temperature': 0}}}},
        'application': {'conditions': {'quantization': 'none', 'template': 'template-v1'}}}


def test_conditions_never_copy_credentials_or_source_contents():
    result = conditions(config(), {'deployment': 'deployment-v1'})
    assert 'PRIVATE' not in json.dumps(result)
    assert result['conditions']['cache']['status'] == 'unknown'
    assert result['conditions']['deployment']['status'] == 'declared'
    assert result['source_content'] == '原始数据内容一致性未验证'


def test_local_source_identity_tracks_bytes_and_order(tmp_path):
    first = tmp_path / 'data1.jsonl'
    second = tmp_path / 'data2.jsonl'
    first.write_text('{"text":"one"}\n')
    second.write_text('{"text":"two"}\n')
    value = config()
    value['pipeline']['loader']['settings'] = {'dataset': 'json', 'split': 'train',
        'data_files': ['data1.jsonl', 'data2.jsonl']}
    source = conditions(value, {}, tmp_path)['conditions']['source_identity']
    assert source['status'] == 'observed'
    assert [item['name'] for item in source['value']['files']] == ['data1.jsonl', 'data2.jsonl']
    second.write_text('{"text":"changed"}\n')
    changed = conditions(value, {}, tmp_path)['conditions']['source_identity']
    assert changed['value'] != source['value']
    value['pipeline']['loader']['settings']['data_files'].reverse()
    reordered = conditions(value, {}, tmp_path)['conditions']['source_identity']
    assert reordered['value'] != changed['value']


def test_hub_source_requires_commit_revision(tmp_path):
    value = config()
    value['pipeline']['loader']['settings'] = {'dataset': 'org/dataset', 'split': 'train'}
    assert conditions(value, {}, tmp_path)['conditions']['source_identity']['status'] == 'unknown'
    value['pipeline']['loader']['settings']['load_kwargs'] = {'revision': 'a' * 40}
    assert conditions(value, {}, tmp_path)['conditions']['source_identity']['value']['revision'] == 'a' * 40


def test_evalscope_selects_metric_and_rejects_identity_mismatch(tmp_path):
    report = tmp_path/'report.json'
    report.write_text(json.dumps({'model_name': 'model', 'dataset_name': 'task', 'score': 1,
        'metrics': [{'name': 'wrong', 'score': 1}, {'name': 'accuracy', 'score': .6}]}))
    cfg = tmp_path/'eval.yaml'
    cfg.write_text(yaml.safe_dump({'model': 'model', 'model_args': {'quantization': 'none'}, 'generation_config': {'temperature': 0}}))
    settings = {'report': 'report.json', 'config': 'eval.yaml', 'tool_version': '1.5', 'metric': 'accuracy',
                'threshold': .8, 'template_id': 'template-v1'}
    result = quality(settings, config(), tmp_path)
    assert result['score'] == .6 and result['status'] == 'fail'
    settings['threshold'] = .5
    assert quality(settings, config(), tmp_path)['status'] == 'pass'
    settings['template_id'] = 'different'
    assert quality(settings, config(), tmp_path)['identity_checks']['template'] == 'mismatch'
    assert quality(settings, config(), tmp_path)['status'] == 'fail'
    del settings['template_id']
    assert quality(settings, config(), tmp_path)['status'] == 'unverified'
    settings['metric'] = 'absent'
    with pytest.raises(ValueError, match='exactly once'):
        quality(settings, config(), tmp_path)


@pytest.mark.parametrize('adapter', ['vllm', 'sglang'])
async def test_service_prepare_order_cache_confirmation_and_repetition(adapter, monkeypatch):
    from perf_benchmark.plugins.executor.openai_completions import OpenAICompletionsExecutor, validate

    monkeypatch.setenv('PERF_TEST_PREPARATION_KEY', 'preparation-secret')
    calls = []
    authorization = []
    async def models(request):
        calls.append('models')
        authorization.append(request.headers.get('Authorization'))
        return web.json_response({'data': [{'id': 'model'}]})
    app = web.Application()
    app.router.add_get('/v1/models', models)
    class Corpus:
        payloads = object()
        def warmup_sessions(self, trial, count):
            turns = [Turn(PayloadRef(0, 1), id=f'turn-{index}') for index in range(count)]
            session = Session(turns=turns, id='warmup', metadata={'warmup': True})
            session.link()
            yield session
    class Runtime:
        async def prepare(self): calls.append('prepare')
        async def prepare_attempt(self, value):
            calls.append(value.turn.id)
            return value
        async def execute_attempt(self, prepared, signal):
            return AttemptResult(prepared.turn.id, 1)
        async def release(self, prepared): return None
        async def close_session(self, session_id): calls.append('close_session')
        async def close(self): calls.append('close')
    class Executor(OpenAICompletionsExecutor):
        async def create_runtime(self, trial, ctx): return Runtime()
    async with TestServer(app) as server:
        executor = config()['pipeline']['executor']
        executor['settings']['base_url'] = str(server.make_url('')).rstrip('/')
        executor['settings']['cache_salt_enabled'] = True
        executor['settings']['api_keys'] = [{'env': 'PERF_TEST_PREPARATION_KEY'}]
        prepare = ServicePreparation({'adapter': adapter, 'exclusive': True, 'warmup_requests': 2,
                                      }, executor)
        trial = SimpleNamespace(request_parameters=freeze_snapshot({'temperature': .2}),
                                id='trial', executor_trial=lambda: ExecutorTrial('trial'))
        for _ in range(2):
            fact = await prepare.prepare(trial, CancellationToken(), Corpus(), Executor(validate(executor['settings'])))
            assert fact.cache_state == 'salt_sent_backend_unverified' and fact.succeeded_requests == 2
        assert calls == ['models', 'prepare', 'turn-0', 'turn-1', 'close_session', 'close'] * 2
        assert authorization == ['Bearer preparation-secret'] * 2
        assert 'preparation-secret' not in repr(fact)


async def test_service_prepare_cancellation_closes_runtime_and_session():
    started = asyncio.Event()
    closed = []
    class Corpus:
        payloads = object()
        def warmup_sessions(self, trial, count):
            session = Session(turns=[Turn(PayloadRef(0, 1), id="turn-fixture-132")], metadata={'warmup': True}, id="session-fixture-132")
            session.link()
            yield session
    class Runtime:
        async def prepare(self): pass
        async def prepare_attempt(self, value): return value
        async def execute_attempt(self, prepared, signal):
            started.set()
            await asyncio.Event().wait()
        async def release(self, prepared):
            closed.append('release')
            return None
        async def close_session(self, session_id): closed.append('session')
        async def close(self): closed.append('runtime')
    class Executor:
        def request_headers(self, settings): return {}
        async def create_runtime(self, trial, ctx): return Runtime()
    async def models(request):
        return web.json_response({'data': [{'id': 'model'}]})
    app = web.Application()
    app.router.add_get('/v1/models', models)
    async with TestServer(app) as server:
        settings = {'plugin': 'openai-chat', 'settings': {'base_url': str(server.make_url('')).rstrip('/'),
                    'model': 'model', 'cache_salt_enabled': True}}
        preparation = ServicePreparation({'adapter': 'vllm', 'exclusive': True, 'warmup_requests': 1}, settings)
        trial = SimpleNamespace(id='trial', executor_trial=lambda: ExecutorTrial('trial'))
        token = CancellationToken()
        task = asyncio.create_task(preparation.prepare(trial, token, Corpus(), Executor()))
        await asyncio.wait_for(started.wait(), 5)
        token.cancel('user_cancelled')
        with pytest.raises(PreparationCancelled) as stopped:
            await asyncio.wait_for(task, 5)
    assert closed == ['release', 'session', 'runtime']
    assert stopped.value.fact.succeeded_requests == 0
    assert stopped.value.fact.failure_reason == 'user_cancelled'


async def test_experiment_failure_preserves_locations_and_rerun_is_new(tmp_path, monkeypatch):
    run = tmp_path/'run.yaml'
    run.write_text(yaml.safe_dump({'case':'experiment failure recovery','pipeline': {'exporter': {}}}))
    plan = tmp_path/'experiment.yaml'
    plan.write_text(yaml.safe_dump({'runs': [{'name': 'first', 'config': 'run.yaml'}, {'name': 'second', 'config': 'run.yaml'}]}))
    calls = []
    async def execute(config, base, token):
        calls.append(config['pipeline']['saver']['settings']['path'])
        raise RuntimeError('preparation failed')
    monkeypatch.setattr(experiments, 'execute_config', execute)
    for _ in range(2):
        with pytest.raises(RuntimeError, match='preparation failed'):
            await experiments.execute(plan)
    assert len(calls) == 2 and calls[0] != calls[1]
    indexes = list((tmp_path/'experiments').rglob('index.yaml'))
    assert len(indexes) == 2
    assert all(len(yaml.safe_load(p.read_text())['runs']) == 2 for p in indexes)


async def test_experiment_rejects_missing_comparison_reference_before_running(tmp_path, monkeypatch):
    (tmp_path/'run.yaml').write_text(yaml.safe_dump({'case':'experiment validation','pipeline': {'exporter': {}}}))
    plan = tmp_path/'experiment.yaml'
    plan.write_text(yaml.safe_dump({'runs': [{'name': 'first', 'config': 'run.yaml'}],
                                   'comparisons': [{'baseline': 'first', 'candidate': 'missing'}]}))
    async def unexpected(*args):
        pytest.fail('invalid experiment launched a Run')
    monkeypatch.setattr(experiments, 'execute_config', unexpected)
    with pytest.raises(ValueError, match='Run names'):
        await experiments.execute(plan)
    assert not (tmp_path/'experiments').exists()


async def test_control_is_bounded_and_only_launches_existing_yaml(tmp_path, monkeypatch):
    import perf_benchmark.app.control as module
    from perf_benchmark.app.control import RunControl
    path = tmp_path/'run.yaml'
    path.write_text('pipeline: {}')
    class Process:
        returncode = None
        def send_signal(self, sig): self.returncode = 0
        async def wait(self): return self.returncode
    launched = []
    async def spawn(*args, **kwargs):
        launched.append(args)
        return Process()
    monkeypatch.setattr(module.asyncio, 'create_subprocess_exec', spawn)
    control = RunControl(tmp_path, tmp_path, limit=1)
    first = await control.start('run.yaml')
    with pytest.raises(web.HTTPTooManyRequests):
        await control.start('run.yaml')
    await control.cancel(first)
    second = await control.start('run.yaml')
    assert first != second and len(control.jobs) == 1
    assert '--output-database' in launched[0]
    with pytest.raises((ValueError, FileNotFoundError)):
        await control.start('../outside.yaml')
    await control.close()


async def test_control_http_rejects_cross_origin(tmp_path, monkeypatch):
    from perf_benchmark.app.results import web as workbench
    (tmp_path / 'index.html').write_text('<div id="root"></div>')
    monkeypatch.setattr(workbench, 'ASSETS', tmp_path)
    async with TestClient(TestServer(create_app(tmp_path, config_root=tmp_path))) as client:
        response = await client.post('/api/control/start', json={'config': 'run.yaml'}, headers={'Origin': 'https://other.test'})
        assert response.status == 403
        response = await client.get('/api/control/configs')
        assert await response.json() == {'configs': []}
        assert (await client.get('/')).status == 200


async def test_changed_effective_trial_generation_cannot_reuse_quality_pass():
    from perf_benchmark.app.preparation import ApplicationPreparation
    prepare = ApplicationPreparation(None, {'status': 'pass', 'identity': {'generation': {'temperature': 0}}})
    trial = SimpleNamespace(request_parameters=freeze_snapshot({'temperature': .8}))
    result = await prepare.prepare(trial, CancellationToken(), None, None)
    assert result.quality_status == 'unverified'
    trial = SimpleNamespace(request_parameters=freeze_snapshot({'temperature': 0}))
    assert (await prepare.prepare(trial, CancellationToken(), None, None)).quality_status == 'pass'


def test_source_urls_do_not_persist_authentication_or_signed_queries():
    value = config()
    value['pipeline']['loader']['settings']['data_files'] = ['https://user:PRIVATE@example.test/data?token=PRIVATE']
    result = conditions(value, {})
    assert 'PRIVATE' not in json.dumps(result)
    assert result['conditions']['loader.settings']['value']['data_files'] == ['https://example.test/data']
    assert result['conditions']['loader.source_parameters']['status'] == 'unknown'
