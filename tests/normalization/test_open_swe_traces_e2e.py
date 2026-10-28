import json
import asyncio
from aiohttp import web
from aiohttp.test_utils import TestServer
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from perf_benchmark.pipeline.coordinator import Coordinator
from zipfile import ZipFile
from tests.exporter.test_tabular import xlsx_rows, without_header_units


@pytest.mark.parametrize("tool_format,tool_id_policy,reject", [
    ("openai", "preserve", False), ("openai", "preserve", True),
    ("openai", "empty", False), ("openai", "empty", True),
    ("openai", "match-single", False), ("openai", "match-single", True),
    ("anthropic", "preserve", False),
], ids=["recorded", "service-rejection", "empty-id", "empty-id-rejected",
        "matched-id", "matched-id-rejected", "anthropic-tools"])
async def test_complete_recorded_swe_replay_uses_natural_outputs(
    tmp_path, monkeypatch, reject, tool_id_policy, tool_format,
):
    requests = []
    reasoning_only = tool_format == 'anthropic'
    output_tokens = 4096 if reasoning_only else 2
    def forbidden(*args, **kwargs):
        raise AssertionError("SWE preparation must not initialize a tokenizer")
    monkeypatch.setattr("perf_benchmark.core.tokenization.ModelTokenizer.__init__", forbidden)

    async def complete(request):
        body = await request.json()
        requests.append(body)
        if body.get('tools') != expected_tools:
            return web.json_response({'error': {'message': 'invalid tools schema'}}, status=422)
        assert body['max_tokens'] == 4096
        assert body['temperature'] == 0.1
        assert not {'max_completion_tokens', 'max_output_tokens', 'ignore_eos'} & body.keys()
        if tool_id_policy == 'empty':
            assert all(isinstance(m.get('tool_call_id'), str)
                       for m in body['messages'] if m['role'] == 'tool')
        if tool_id_policy == 'match-single':
            assert all(m['tool_call_id'] == 'call-a'
                       for m in body['messages'] if m['role'] == 'tool')
        if reject and len(body['messages']) == 4:
            return web.json_response({'error': {'message': 'input exceeds service context'}}, status=400)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        field = 'reasoning' if reasoning_only else 'content'
        await response.write(b'data: ' + json.dumps({
            'choices': [{'delta': {field: 'measured'}}]}).encode() + b'\n\n')
        await asyncio.sleep(0.001)
        await response.write(b'data: ' + json.dumps({
            'choices': [{'delta': {field: ' output'}}]}).encode() + b'\n\n')
        await response.write(b'data: ' + json.dumps({
            'choices': [{'delta': {}, 'finish_reason': 'length' if reasoning_only else 'stop'}],
            'usage': {'prompt_tokens': 100, 'completion_tokens': output_tokens,
                      'total_tokens': 100 + output_tokens,
                      'prompt_tokens_details': {'cached_tokens': 64}}}).encode() + b'\n\n')
        await response.write(b'data: [DONE]\n\n')
        return response

    app = web.Application()
    app.router.add_post('/v1/chat/completions', complete)
    server = TestServer(app)
    await server.start_server()
    record = {'benchmark_session_id': 's', 'trajectory_id': 't',
              'messages': [{'role': 'system', 'content': 'system'},
                           {'role': 'user', 'content': 'fix'},
                           {'role': 'assistant', 'content': 'recorded',
                            'tool_calls': [{'id': 'call-a', 'type': 'function',
                                            'function': {'name': 'bash', 'arguments': '{}'}}]},
                           {'role': 'tool', 'content': 'tool-result'},
                           {'role': 'assistant', 'content': 'final'}],
              'tools': [json.dumps({'type': 'function', 'function': {'name': 'bash',
                       'parameters': {'type': 'object', 'properties': {}}}})],
              'benchmark_input_tokens': [30, 100], 'benchmark_output_tokens': [10, 20]}
    expected_tools = [json.loads(record['tools'][0])]
    if tool_format == 'anthropic':
        function = expected_tools[0]['function']
        record['tools'] = [json.dumps({'name': function['name'],
                                      'input_schema': function['parameters']})]
    original = json.loads(json.dumps(record))
    source = tmp_path / 'swe.parquet'
    pq.write_table(pa.Table.from_pylist([record]), source)
    config = {'name': 'mock-swe', 'seed': 42, 'pipeline': {
        'loader': {'plugin': 'hf-datasets', 'settings': {'dataset': 'parquet',
                   'data_files': str(source), 'split': 'train', 'batch_rows': 1}},
        'normalization': {'plugin': 'open-swe-traces',
                          'settings': {'missing_tool_call_id': tool_id_policy}},
        'selector': {'plugin': 'all', 'settings': {}},
        'executor': {'plugin': 'openai-chat', 'settings': {'model': 'mock',
                     'base_url': str(server.make_url('')), 'stream': True}},
        'scheduler': {'plugin': 'multi-turn-closed-loop', 'settings': {
                      'max_concurrency': 1, 'max_active_sessions': 1,
                      'stop': {'session_count': 1}, 'on_exhausted': 'error'}},
        'metrics': {'plugin': 'standard', 'settings': {}},
        'saver': {'plugin': 'sqlite', 'settings': {'path': str(tmp_path / 'run.db')}},
        'exporter': {}}}
    try:
        run = await Coordinator().execute(config, base_dir=tmp_path)
    finally:
        await server.close()
    assert run.status == ('failed' if reject else 'succeeded')
    assert len(requests) == 2
    expected = [dict(message) for message in record['messages'][:4]]
    if tool_id_policy == 'empty':
        expected[-1]['tool_call_id'] = ''
    elif tool_id_policy == 'match-single':
        expected[-1]['tool_call_id'] = 'call-a'
    assert requests[1]['messages'] == expected
    assert requests[1]['max_tokens'] == 4096
    assert 'ignore_eos' not in requests[1]
    assert all(body['tools'] == expected_tools for body in requests)
    assert record == original
    if reasoning_only:
        from perf_benchmark.plugins.saver.sqlite import SQLiteResultReader
        reader = SQLiteResultReader(tmp_path / 'run.db')
        turns = tuple(reader.iter_turn_records(run.id))
        assert len(turns) == 2
        assert all(turn['status'] == 'succeeded' and
                   turn['measurements']['request']['finish_reason'] == 'length' for turn in turns)
    if reject:
        from perf_benchmark.plugins.saver.sqlite import SQLiteResultReader
        reader = SQLiteResultReader(tmp_path / 'run.db')
        assert reader.request_page(run.id, status='failed', limit=1, detail=False)['total'] == 1
    excel = tmp_path / 'run-tables' / 'results.xlsx'
    assert run.artifacts['tabular:results.xlsx'] == str(excel)
    assert not list(excel.parent.glob('*.csv'))
    with ZipFile(excel) as workbook:
        row = without_header_units(xlsx_rows(workbook, 1))[0]
        assert list(row)[:7] == ['case','model', 'input_tokens_mean', 'output_tokens_mean',
                                'concurrency', 'active_users', 'generation_parameters']
        assert 'stream' not in row
        assert json.loads(row['generation_parameters']) == {'temperature':0.1}
        assert row['model'] == 'mock' and row['concurrency'] == row['active_users'] == '1'
        assert row['input_tokens_mean'] == '100.0'
        assert row['output_tokens_mean'] == str(float(output_tokens))
        assert row['ttft_mean'] != '' and row['ttft_p90'] != ''
        assert row['decode_throughput_p99'] != ''
        assert 'maxtpm_start_ns' not in row
        assert len(workbook.read('xl/workbook.xml').split(b'<sheet ')) - 1 == 3
        assert b'name="metrics"' not in workbook.read('xl/workbook.xml')
