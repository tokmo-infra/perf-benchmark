import gzip
import hashlib
import json
import os
import sqlite3
from zipfile import ZipFile

import pytest

from perf_benchmark.app.results.compact import compact, compress_metrics, digest, shared_input


def result(root, name='case', status='succeeded'):
    case = root / name
    case.mkdir()
    with sqlite3.connect(case / 'run.db') as db:
        db.execute('CREATE TABLE runs(status TEXT,finished_at TEXT)')
        db.execute('INSERT INTO runs VALUES (?,?)', (status, '2026-10-16T00:00:00+00:00'))
        db.execute('CREATE TABLE preserved_facts(token_count INTEGER)')
        db.execute('INSERT INTO preserved_facts VALUES (123456)')
    tables = case / 'run-tables'
    tables.mkdir()
    with ZipFile(tables / 'results.xlsx', 'w') as workbook:
        workbook.writestr('xl/workbook.xml', '<workbook/>')
    (tables / 'metrics.csv').write_text('derived metric rows')
    (case / 'report.html').write_text('derived html')
    (case / 'engine-observation.prom.log').write_text('# scrape_utc fixed\nmetric 123456\n')
    (case / 'selected.jsonl').write_text('{"fixed_input":true}\n')
    (case / 'selection-manifest.json').write_text(json.dumps({
        'selected_sha256': digest(case / 'selected.jsonl')}))
    return case


def test_compact_preserves_native_facts_workbook_plain_logs_and_shares_inputs(tmp_path):
    a, b = result(tmp_path, 'a'), result(tmp_path, 'b')
    before = {p: digest(p) for case in (a, b) for p in
              (case / 'run.db', case / 'run-tables/results.xlsx', case / 'engine-observation.prom.log')}
    assert all(row['state'] == 'planned' for row in compact(tmp_path))
    assert (a / 'report.html').exists()
    assert all(row['state'] == 'compacted' for row in compact(tmp_path, apply=True))
    assert os.path.samefile(a / 'selected.jsonl', b / 'selected.jsonl')
    assert all(digest(p) == sha for p, sha in before.items())
    assert not (a / 'report.html').exists() and not (b / 'run-tables/metrics.csv').exists()


@pytest.mark.parametrize('protection', ['active', 'running', 'wal', 'bad_workbook'])
def test_protected_results_cannot_lose_reports(tmp_path, protection):
    case = result(tmp_path, status='running' if protection == 'running' else 'succeeded')
    if protection == 'active':
        (tmp_path / 'state.json').write_text(json.dumps({'current': {'name': 'case'}}))
    elif protection == 'wal':
        (case / 'run.db-wal').write_bytes(b'pending')
    elif protection == 'bad_workbook':
        (case / 'run-tables/results.xlsx').write_bytes(b'corrupt')
    assert compact(tmp_path, apply=True)[0]['state'] == 'protected'
    assert (case / 'report.html').exists()


def test_manifest_mismatch_prevents_all_deletion_then_valid_manifest_is_amended(tmp_path):
    case = result(tmp_path)
    entries = {p.name: {'bytes': p.stat().st_size, 'sha256': digest(p)}
               for p in (case / 'run.db', case / 'report.html')}
    manifest = case / 'artifact-manifest.json'
    manifest.write_text(json.dumps({'files': entries, 'sqlite_writer_closed': True}))
    original_manifest_sha = digest(manifest)
    manifest.chmod(0o644)
    (case / 'mac-sync-receipt.json').write_text(json.dumps({
        'artifact_manifest_sha256': original_manifest_sha, 'all_sizes_and_sha256_match': True}))
    (case / 'report.html').write_text('tampered')
    assert compact(tmp_path, apply=True)[0]['state'] == 'protected'
    assert (case / 'run-tables/metrics.csv').exists()
    (case / 'report.html').write_text('derived html')
    assert compact(tmp_path, apply=True)[0]['state'] == 'compacted'
    assert set(json.loads(manifest.read_text())['files']) == {'run.db'}
    receipt = json.loads((case / 'mac-sync-receipt.json').read_text())
    assert receipt['artifact_manifest_sha256'] == digest(manifest)
    assert receipt['prior_artifact_manifest_sha256'] == original_manifest_sha
    assert manifest.stat().st_mode & 0o777 == 0o644
    assert (case / 'storage-compaction.json').stat().st_mode & 0o777 == 0o644


def test_shared_input_hash_failure_never_replaces_source(tmp_path):
    path = tmp_path / 'selected.jsonl'
    path.write_bytes(b'original')
    with pytest.raises(ValueError, match='SHA mismatch'):
        shared_input(path, tmp_path / 'shared', hashlib.sha256(b'wrong').hexdigest())
    assert path.read_bytes() == b'original'


def test_closed_metrics_compression_round_trips_all_timestamps_and_samples(tmp_path):
    path = tmp_path / 'engine-observation.prom.log'
    data = (b'# scrape_utc 2026-10-16T00:00:00Z\ncounter{dp="0"} 123\n' * 100)
    path.write_bytes(data)
    evidence = compress_metrics(path)
    assert not path.exists()
    assert gzip.decompress((tmp_path / evidence['path']).read_bytes()) == data
