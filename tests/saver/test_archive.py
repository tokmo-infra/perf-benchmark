"""Real Standard Metrics -> committed SQLite -> independent permanent Parquet."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
import pyarrow.parquet as pq

from benchmarks.result_storage import prepared_batches
from perf_benchmark.core.errors import ResultArchiveError
from perf_benchmark.core.models import Run, Trial
from perf_benchmark.pipeline.context import SaverContext
from perf_benchmark.pipeline.contracts.results import MetricPoint, RunRecord, MeasurementRecord, MetricWindowResultRecord
from perf_benchmark.pipeline.results.saver import _snapshot, _trial_records
from perf_benchmark.plugins.saver.sqlite import SQLiteSaver, SQLiteResultReader
from perf_benchmark.plugins.saver.sqlite import archive
from perf_benchmark.plugins.saver.sqlite.parquet_reader import SummaryReader, analyze
from perf_benchmark.runtime.cancellation import CancellationToken

FINISHED = datetime.now(timezone.utc).isoformat()


@pytest.fixture(autouse=True)
def private_sealing_root(tmp_path, monkeypatch):
    root = tmp_path / 'sealing-coordination'
    root.mkdir()
    monkeypatch.setattr(archive, '_sealing_root', lambda: root)


def fixture(path, *, run_id='run', finished=FINISHED, status='succeeded', unit='s', settings=None):
    run = Run(id=run_id, name='正式事实一致性')
    trial = Trial(id='trial', run_id=run_id)
    backend = SQLiteSaver({'path': str(path), **(settings or {})}).create_backend(_snapshot(RunRecord, run), SaverContext(path.parent, CancellationToken()))
    backend.open()
    backend.write_batch((MeasurementRecord(run_id, "run", run_id, "observation_mode", {"enabled": False, "trace_mode": "benchmark"}),))
    backend.write_batch(tuple(_trial_records(trial, None)))
    # Deliberately collide dictionary IDs across Runs with different definitions.
    backend.write_batch((MetricPoint('run', run_id, 'overall', 'fixture', 'value', 2., unit, 'fixture-v1', 'available', None, 'fixture'),))
    for batch in prepared_batches(16, 'prepared'):
        backend.write_batch(tuple(replace(record, run_id=run_id) if hasattr(record, 'run_id') else record for record in batch))
    anchor = 2**53 + 7
    backend.write_batch((MetricWindowResultRecord(window='timeline:fixture', start_ns=anchor,
        end_ns=anchor+10, duration_s=1e-8, availability='available', unavailable_reason=None,
        run_id=run_id, scope='trial', scope_id='trial'),
        MetricPoint('trial', 'trial', 'timeline:fixture', 'workload_rps', 'value', None, 'requests/s',
                    'fixture-v1', 'unavailable', 'fixture missing sample', 'fixture', 0, 16, 16)))
    backend.finalize(replace(_snapshot(RunRecord, run), status=status, finished_at=finished, counts={'turns': 16, 'sessions': 16}))
    backend.close()
    return archive.archive_path(path)


def test_archive_matches_facts_and_independent_formal_views(tmp_path):
    path = tmp_path / 'run.sqlite'
    directory = fixture(path)
    sql, permanent = SQLiteResultReader(path), SummaryReader(directory)
    assert permanent.summary('run') == sql.summary('run')
    assert permanent.run_record('run') == sql.run_record('run')
    assert permanent.trials('run') == sql.trials('run')
    assert tuple(permanent.iter_summary_records('run')) == tuple(sql.iter_summary_records('run'))
    assert permanent.grouped_metrics('run', 'trial') == sql.grouped_metrics('run', 'trial')
    for scope, identity in [('run', 'run'), ('trial', 'trial')]:
        assert permanent.metric_points('run', scope=scope, scope_id=identity, window='overall') == sql.metric_points('run', scope=scope, scope_id=identity, window='overall')
    params = dict(trial_id=None, metric='workload_rps', start_ns=0, end_ns=10**18)
    assert permanent.metric_series('run', **params) == sql.metric_series('run', **params)
    routed = SQLiteSaver.open_reader(str(path))
    assert routed.request_page('run') == sql.request_page('run')
    assert routed.request_page('run', scope='attempt') == sql.request_page('run', scope='attempt')
    assert tuple(routed.iter_turn_records('run')) == tuple(sql.iter_turn_records('run'))
    info = archive.read_info(directory, verify=True)
    assert info['observation_enabled'] is False
    assert info['detail_expires_at'] == (datetime.fromisoformat(FINISHED) + timedelta(days=7)).isoformat()
    requests = analyze([directory], 'SELECT count(*) AS n,sum(output_chars) AS chars FROM requests', details=True)
    assert requests['items'] == [{'n': 16, 'chars': 512}]
    assert routed.read_updates(limit=50) == sql.read_updates(limit=50)


def test_both_details_expire_summary_and_cross_run_remain(tmp_path):
    old = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
    paths = [tmp_path / name for name in ('a.sqlite', 'b.sqlite')]
    directories = [fixture(path, run_id=name, finished=old, unit=unit) for path, name, unit in zip(paths, ('a', 'b'), ('s', 'ms'))]
    from perf_benchmark.app.experiments import compare_files
    spec = {'metrics': [{'metric': 'ttft', 'statistic': 'p99'}]}
    comparison = compare_files(paths[0], paths[1], spec)
    before = [tuple(SummaryReader(directory).iter_summary_records(name)) for directory, name in zip(directories, ('a', 'b'))]
    for directory, path, name in zip(directories, paths, ('a', 'b')):
        with pytest.raises(ValueError, match='detail_expired'):
            SQLiteSaver.open_reader(str(path)).request_page(name)
        with pytest.raises(ValueError, match='detail_expired'):
            SQLiteResultReader(path).request_page(name)
        assert archive.cleanup(directory)['state'] == 'detail_expired'
        assert not path.exists() and not (directory / 'details').exists()
        assert tuple(SQLiteSaver.open_reader(str(directory)).iter_summary_records(name)) == before[0 if name == 'a' else 1]
        assert archive.cleanup(directory)['state'] == 'detail_expired'
    assert compare_files(directories[0], directories[1], spec) == comparison
    result = analyze(directories, "SELECT run_id,unit,value FROM metrics WHERE metric='fixture' ORDER BY run_id")
    assert result['items'] == [{'run_id': 'a', 'unit': 's', 'value': 2.}, {'run_id': 'b', 'unit': 'ms', 'value': 2.}]
    with pytest.raises(ValueError, match='detail_expired'):
        analyze(directories, 'SELECT count(*) FROM requests', details=True)
    from perf_benchmark.app.results.service import Catalog
    catalog = Catalog(tmp_path)
    assert catalog.runs({})['total'] == 2
    assert catalog.errors == ()
    assert catalog.reader('a').summary('a')['turn_count'] == 16


def test_live_lease_defers_cleanup_and_missing_sqlite_does_not_use_parquet(tmp_path):
    path = tmp_path / 'run.sqlite'
    directory = fixture(path)
    expiry = datetime.fromisoformat(archive.read_info(directory)['detail_expires_at'])
    with archive.detail_lease(path):
        assert archive.cleanup(directory, now=expiry)['state'] == 'reader_active'
    path.unlink()
    with pytest.raises(ValueError, match='detail_unavailable'):
        SQLiteSaver.open_reader(str(directory)).request_page('run')
    assert (directory / 'details' / 'turns.parquet').exists()
    assert archive.cleanup(directory, now=expiry)['state'] == 'detail_expired'


def test_request_reads_only_write_shared_coordination_not_formal_result_directory(tmp_path):
    root = tmp_path / 'formal-results'
    root.mkdir()
    path = root / 'run.sqlite'
    directory = fixture(path)
    original = {p: p.stat().st_mode for p in root.rglob('*')}
    original[root] = root.stat().st_mode
    try:
        for p in original:
            p.chmod(0o555 if p.is_dir() else 0o444)
        assert SQLiteSaver.open_reader(str(path)).request_page('run')['total'] == 16
        assert tuple(SummaryReader(directory).iter_summary_records('run'))
        assert not list(root.glob('*.lock')) and not list(root.glob('*.leases'))
        leases = archive._coordination(path) / 'leases'
        with archive.detail_lease(path):
            assert len(list(leases.glob('*.lock'))) == 1
        assert not list(leases.glob('*.lock'))
    finally:
        for p, mode in original.items():
            p.chmod(mode)


@pytest.mark.parametrize('status', ['failed', 'cancelled'])
def test_partial_terminal_run_is_sealed(tmp_path, status):
    path = tmp_path / 'partial.sqlite'
    directory = fixture(path, status=status)
    reader = SummaryReader(directory)
    assert reader.run_record('run').status == status
    assert reader.summary('run')['turn_count'] == 16


def test_unknown_expiry_and_corrupt_summary_protect_source(tmp_path):
    path = tmp_path / 'unknown.sqlite'
    directory = fixture(path, finished=None)
    assert archive.cleanup(directory)['state'] == 'expiry_unknown'
    target = directory / 'summary' / 'trials.parquet'
    content = bytearray(target.read_bytes())
    content[20] ^= 1
    target.write_bytes(content)
    with pytest.raises((ValueError, OSError), match='corrupt|Invalid|invalid'):
        archive.cleanup(directory, now=datetime.now(timezone.utc) + timedelta(days=100))
    assert path.exists() and (directory / 'details').exists()


def test_failed_seal_preserves_sqlite_and_publishes_nothing(tmp_path):
    path = tmp_path / 'failed.sqlite'
    with patch.object(pq, 'ParquetWriter', side_effect=OSError('disk full')):
        with pytest.raises(ResultArchiveError, match='disk full'):
            fixture(path)
    assert path.exists() and not archive.archive_path(path).exists()
    assert SQLiteResultReader(path).summary('run')['turn_count'] == 16
    assert not list(tmp_path.glob('.*.archive-*'))
    archive.seal(path)
    with pytest.raises(ValueError, match='frozen'):
        SQLiteSaver({'path': str(path)}).create_backend(_snapshot(RunRecord, Run(name="frozen")), SaverContext(tmp_path, CancellationToken())).open()


def test_catalog_counts_a_sealed_run_once_and_rejects_archive_revision_drift(tmp_path):
    path = tmp_path / 'run.sqlite'
    directory = fixture(path)
    from perf_benchmark.app.results.service import Catalog
    assert Catalog(tmp_path).runs({})['total'] == 1
    import sqlite3
    with sqlite3.connect(path) as db:
        db.execute('UPDATE result_state SET revision=revision+1')
    with pytest.raises(ValueError, match='different source revision'):
        archive.seal(path)
    with pytest.raises(ValueError, match='detail_source_changed'):
        SQLiteSaver.open_reader(str(directory)).request_page('run')


def test_analysis_rejects_mutations_and_duplicate_runs(tmp_path):
    directory = fixture(tmp_path / 'run.sqlite')
    with pytest.raises(ValueError, match='one SELECT'):
        analyze([directory], 'CREATE TABLE x AS SELECT 1')
    with pytest.raises(ValueError, match='distinct'):
        analyze([directory, directory], 'SELECT count(*) FROM runs')



def test_sealing_waits_for_other_writer_and_then_drains_pending_runs(tmp_path):
    live = tmp_path / 'live.sqlite'
    run = Run(name='other active Run', id='other')
    backend = SQLiteSaver({'path': str(live)}).create_backend(_snapshot(RunRecord, run), SaverContext(tmp_path, CancellationToken()))
    backend.open()
    path = tmp_path / 'finished.sqlite'
    directory = fixture(path)
    assert not directory.exists()
    assert SQLiteResultReader(path).summary('run')['turn_count'] == 16
    backend.finalize(replace(_snapshot(RunRecord, run), status='cancelled', finished_at=FINISHED))
    backend.close()
    assert directory.is_dir() and archive.archive_path(live).is_dir()
    assert not list(archive._sealing_root().glob('*.pending'))


def test_corrupt_details_rejected_without_changing_permanent_summary(tmp_path):
    path = tmp_path / 'run.sqlite'
    directory = fixture(path)
    target = directory / 'details' / 'turns.parquet'
    target.write_bytes(target.read_bytes() + b'corrupt')
    with pytest.raises((ValueError, OSError)):
        analyze([directory], 'SELECT count(*) FROM requests', details=True)
    assert SummaryReader(directory).summary('run')['turn_count'] == 16


def test_unfinished_stage_blocks_manual_sealing(tmp_path):
    import sqlite3
    from perf_benchmark.plugins.saver.sqlite.encoding import _json_text
    path = tmp_path / 'run.sqlite'
    directory = fixture(path)
    import shutil
    shutil.rmtree(directory)
    with sqlite3.connect(path) as db:
        db.execute('INSERT INTO measurements VALUES (?,?,?,?,?,?)',
                   ('run','run','run','stages',_json_text({'exporter': {'status':'started'}}), 1))
    with pytest.raises(ValueError, match='incomplete_finalization'):
        archive.seal(path)
    assert path.exists() and not directory.exists()



def test_edge_facts_retry_non_stream_non_text_and_null_survive_archive(tmp_path):
    from benchmarks.sqlite_parquet_comparison import create
    from perf_benchmark.plugins.saver.sqlite.archive_tools import restore
    directory = tmp_path / 'edges'
    create(directory, 64, 'edges')
    path = directory / 'results.db'
    sql = SQLiteResultReader(path)
    permanent = SummaryReader(archive.archive_path(path))
    assert tuple(sql.iter_summary_records('edges')) == tuple(permanent.iter_summary_records('edges'))
    requests = analyze([archive.archive_path(path)],
        'SELECT status,count(*) AS n FROM requests GROUP BY status ORDER BY status', details=True)
    expected = {}
    for row in sql.iter_turn_records('edges'):
        expected[row['status']] = expected.get(row['status'], 0) + 1
    assert requests['items'] == [{'status': status, 'n': count} for status, count in sorted(expected.items())]
    assert analyze([archive.archive_path(path)], 'SELECT count(*) AS n FROM attempts', details=True)['items'] == [{'n': 72}]
    turns, attempts = tuple(sql.iter_turn_records('edges')), tuple(sql.iter_attempt_records('edges'))
    archive.delete_sqlite(archive.archive_path(path))
    restored = SQLiteResultReader(restore(archive.archive_path(path), path))
    assert tuple(restored.iter_turn_records('edges')) == turns
    assert tuple(restored.iter_attempt_records('edges')) == attempts



def test_one_writer_per_source_and_fresh_run_files(tmp_path):
    path = tmp_path / 'one.sqlite'
    plugin = SQLiteSaver({'path': str(path)})
    ctx = SaverContext(tmp_path, CancellationToken())
    first = plugin.create_backend(_snapshot(RunRecord, Run(name='first')), ctx)
    second = plugin.create_backend(_snapshot(RunRecord, Run(name='second')), ctx)
    first.open()
    with pytest.raises(ValueError, match='already_active'):
        second.open()
    second.close()
    first.close()
    third = plugin.create_backend(_snapshot(RunRecord, Run(name='third')), ctx)
    try:
        with pytest.raises(ValueError, match='already contains'):
            third.open()
    finally:
        third.close()


async def test_compressed_storage_restores_exact_requests_and_preserves_excel(tmp_path):
    from perf_benchmark.app.report import export_tables
    from perf_benchmark.plugins.saver.sqlite.archive_tools import restore, verify
    from openpyxl import load_workbook

    path = tmp_path / 'run.sqlite'
    directory = fixture(path)
    original = SQLiteResultReader(path)
    turns = tuple(original.iter_turn_records('run'))
    attempts = tuple(original.iter_attempt_records('run'))
    page = original.request_page('run', scope='attempt')
    summary = tuple(original.iter_summary_records('run'))
    await export_tables(path, tmp_path / 'before', ['xlsx'])
    before = path.stat().st_size
    assert archive.delete_sqlite(directory)['removed_bytes'] >= before
    assert not path.exists()
    assert archive.delete_sqlite(directory)['removed_bytes'] == 0
    assert verify(path)['state'] == 'verified'
    assert analyze([directory], 'SELECT count(*) AS n FROM requests', details=True)['items'] == [{'n': 16}]
    assert tuple(SQLiteSaver.open_reader(str(path)).iter_summary_records('run')) == summary
    restored = restore(directory, tmp_path / 'restored.sqlite')
    await export_tables(restored, tmp_path / 'after', ['xlsx'])

    def excel_values(folder):
        workbook = load_workbook(folder / 'results.xlsx', read_only=True)
        try:
            return {sheet.title: tuple(sheet.values) for sheet in workbook}
        finally:
            workbook.close()

    assert excel_values(tmp_path / 'before') == excel_values(tmp_path / 'after')
    reader = SQLiteResultReader(restored)
    assert tuple(reader.iter_turn_records('run')) == turns
    assert tuple(reader.iter_attempt_records('run')) == attempts
    assert reader.request_page('run', scope='attempt') == page
    assert tuple(reader.iter_summary_records('run')) == summary
    with pytest.raises(FileExistsError):
        restore(directory, restored)
    assert not list(tmp_path.glob('.restore-*'))


@pytest.mark.parametrize('days', [None, 30])
def test_automatic_db_deletion_and_configurable_detail_retention(tmp_path, days):
    from perf_benchmark.plugins.saver.sqlite.archive_tools import restore, verify
    path = tmp_path / 'run.sqlite'
    old = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
    directory = fixture(path, finished=old, settings={
        'delete_db_after_archive': True, 'detail_retention_days': days})
    assert not path.exists()
    assert verify(directory)['state'] == 'verified'
    info = archive.read_info(directory)
    assert info['detail_retention_days'] == days
    assert info['detail_expires_at'] == archive.expires_at(old, days)
    assert archive.cleanup(directory)['state'] == 'retained'
    restored = restore(directory, path)
    assert SQLiteSaver.open_reader(str(restored)).request_page('run')['total'] == 16
    if days is None:
        future = datetime.now(timezone.utc) + timedelta(days=3650)
        assert archive.cleanup(directory, now=future)['state'] == 'retained'
        with archive.detail_lease(path, now=future):
            pass
    else:
        assert archive.cleanup(directory, now=datetime.fromisoformat(info['detail_expires_at']))['state'] == 'detail_expired'


def test_delete_sqlite_respects_readers_source_identity_and_corrupt_details(tmp_path):
    import sqlite3
    path = tmp_path / 'run.sqlite'
    directory = fixture(path)
    with archive.detail_lease(path):
        assert archive.delete_sqlite(directory)['state'] == 'reader_active'
    assert path.exists()
    with sqlite3.connect(path) as db:
        db.execute('UPDATE result_state SET revision=revision+1')
    with pytest.raises(ValueError, match='source_changed'):
        archive.delete_sqlite(directory)
    with sqlite3.connect(path) as db:
        db.execute('UPDATE result_state SET revision=revision-1')
    target = directory / 'details' / 'request_facts.parquet'
    target.write_bytes(target.read_bytes() + b'corrupt')
    with pytest.raises((ValueError, OSError)):
        archive.delete_sqlite(directory)
    assert path.exists()


def test_automatic_compression_failure_preserves_sqlite(tmp_path):
    path = tmp_path / 'failed.sqlite'
    with patch.object(pq, 'ParquetWriter', side_effect=OSError('disk full')):
        with pytest.raises(ResultArchiveError, match='disk full'):
            fixture(path, settings={'delete_db_after_archive': True})
    assert SQLiteResultReader(path).summary('run')['turn_count'] == 16
    assert not archive.archive_path(path).exists()


def test_deferred_sealing_keeps_automatic_compression_policy(tmp_path):
    live = tmp_path / 'live.sqlite'
    run = Run(name='active Run', id='other')
    backend = SQLiteSaver({'path': str(live)}).create_backend(
        _snapshot(RunRecord, run), SaverContext(tmp_path, CancellationToken()))
    backend.open()
    path = tmp_path / 'finished.sqlite'
    directory = fixture(path, settings={'delete_db_after_archive': True})
    assert path.exists() and not directory.exists()
    backend.finalize(replace(_snapshot(RunRecord, run), status='cancelled', finished_at=FINISHED))
    backend.close()
    assert not path.exists() and directory.exists()
    assert live.exists()
    assert analyze([directory], 'SELECT count(*) AS n FROM requests', details=True)['items'] == [{'n': 16}]


def test_storage_settings_validate_only_supported_types():
    from perf_benchmark.plugins.saver.sqlite import parse_settings
    assert parse_settings({'path': 'run.db'})['delete_db_after_archive'] is False
    for settings in ({'delete_db_after_archive': 1}, {'detail_retention_days': True},
                     {'detail_retention_days': -1}, {'detail_retention_days': 0.5}):
        with pytest.raises(ValueError):
            parse_settings({'path': 'run.db', **settings})


def test_archive_verify_restore_cli(tmp_path, capsys):
    import json
    from perf_benchmark.app.main import console_main as main
    path = tmp_path / 'run.sqlite'
    directory = fixture(path)
    assert main(['results', 'archive', str(path), '--delete-db']) == 0
    assert json.loads(capsys.readouterr().out)['state'] == 'sqlite_removed'
    assert main(['results', 'verify', str(directory)]) == 0
    assert json.loads(capsys.readouterr().out)['state'] == 'verified'
    restored = tmp_path / 'recovered.sqlite'
    assert main(['results', 'restore', str(directory), '--output', str(restored)]) == 0
    assert capsys.readouterr().out.strip() == str(restored)
    assert SQLiteResultReader(restored).request_page('run')['total'] == 16
