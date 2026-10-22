import json
from pathlib import Path

from aiohttp import web
from aiohttp.test_utils import TestServer
import pytest

from perf_benchmark.app.main import execute_config
from perf_benchmark.plugins.saver.sqlite import SQLiteResultReader


@pytest.mark.parametrize('normalizer,executor,path,body,response', [
    ('embedding', 'openai-embeddings', '/v1/embeddings', {'input': ['one','two']},
     {'data': [{'index': 0, 'embedding': [0.1]}, {'index': 1, 'embedding': [0.2]}], 'usage': {'prompt_tokens': 2}}),
    ('rerank', 'rerank', '/v1/rerank', {'query': 'q', 'documents': ['a','b']},
     {'results': [{'index': 1, 'relevance_score': .8}, {'index': 0, 'relevance_score': .4}]}),
    ('openai-responses', 'openai-responses', '/v1/responses', {'input': 'question'},
     {'status': 'completed', 'output': [{'type': 'function_call', 'name': 'lookup', 'arguments': '{}'}], 'usage': {'input_tokens': 3, 'output_tokens': 2}}),
])
async def test_business_protocol_full_pipeline(tmp_path, tokenizer_path, normalizer, executor, path, body, response):
    source = tmp_path/'source.jsonl'
    source.write_text(json.dumps({'id': 'one', **body}))
    async def handle(request):
        assert (await request.json())['model'] == 'model'
        return web.json_response(response)
    app = web.Application()
    app.router.add_post(path, handle)
    async with TestServer(app) as server:
        settings: dict = {'model': 'model', 'base_url': str(server.make_url('')).rstrip('/')}
        if executor == 'openai-responses':
            settings['stream'] = False
        config = {'case':'business protocol end-to-end','pipeline': {
            'loader': {'plugin': 'hf-datasets', 'settings': {'dataset': 'json', 'split': 'train', 'data_files': [str(source)]}},
            'normalization': {'plugin': normalizer, 'settings': {'tokenizer_path': str(tokenizer_path)}},
            'selector': {'plugin': 'all', 'settings': {}}, 'executor': {'plugin': executor, 'settings': settings},
            'scheduler': {'plugin': 'single-turn-closed-loop', 'settings': {'max_concurrency': 1}},
            'metrics': {'plugin': 'standard', 'settings': {}},
            'saver': {'plugin': 'sqlite', 'settings': {'path': str(tmp_path/'run.sqlite')}},
            'exporter': {'plugins': [{'plugin': 'html', 'settings': {'output': str(tmp_path/'report.html')}}]},
        }}
        run = await execute_config(config, tmp_path)
    reader = SQLiteResultReader(Path(run.artifacts['database']))
    assert run.status == 'succeeded'
    turn = tuple(reader.iter_turn_records(run.id))[0]
    assert turn['measurements']['request']['output_kind'] == 'non_text'
    assert turn['metrics']['e2e']['availability'] == 'available'
    assert turn['metrics']['ttft']['availability'] == 'unavailable'
    evidence = reader.measurements(run.id, scope='run', scope_id=run.id)['evidence']
    assert evidence['quality']['status'] == 'unverified'
    assert evidence['conditions']['tokenizer_fingerprint']['status'] == 'observed'
    assert '原始数据内容一致性未验证' in (tmp_path/'report.html').read_text()
    page = reader.request_page(run.id, session_id=turn['session_id'], limit=1)
    assert page['total'] == 1 and len(page['items']) == 1
