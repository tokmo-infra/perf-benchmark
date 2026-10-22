"""Offline, lossless SQLite/Parquet comparison; never changes a production path.

Run with ``uv run python -m benchmarks.sqlite_parquet_comparison --output report.json.gz``.
Each phase runs in a fresh process. Parquet mirrors ALL committed SQLite tables,
including dictionaries, compact metric encoding and revision metadata. It is an
experimental archive layout, not a Saver Plugin or a replacement ResultReader.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import closing
from dataclasses import fields, replace
import gc
import gzip
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import resource
import sqlite3
import statistics
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event, Thread
from time import perf_counter
from unittest.mock import patch

import duckdb
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from benchmarks import result_storage
from benchmarks.result_pipeline import events
from perf_benchmark.core.models import Run, Trial
from perf_benchmark.pipeline.context import SaverContext
from perf_benchmark.pipeline.contracts.results import RunRecord, TrialFactRecord, TrialResultRecord
from perf_benchmark.pipeline.contracts import facts as fact_contract
from perf_benchmark.pipeline.results.measurement import AttemptFact, SessionFact, TurnFact
from perf_benchmark.pipeline.results.saver import _snapshot, _trial_records
from perf_benchmark.plugins.saver.sqlite import SQLiteSaver, entities
from perf_benchmark.runtime.cancellation import CancellationToken


def quoted(name):
    return '"' + name.replace('"', '""') + '"'


def files_size(directory):
    logical = allocated = 0
    for path in directory.rglob('*'):
        try:
            if path.is_file():
                stat = path.stat()
                logical += stat.st_size
                allocated += stat.st_blocks * 512
        except FileNotFoundError:
            pass  # A writer may rename its temporary file during this sample.
    return logical, allocated


class Peak:
    """Sample task-owned files and RSS; include explicit start/end observations."""

    def __init__(self, directory):
        self.directory = directory
        self.stop = Event()
        self.logical = self.allocated = self.rss = 0
        self.samples = 0
        self.process = psutil.Process()

    def sample(self):
        logical, allocated = files_size(self.directory)
        self.logical = max(self.logical, logical)
        self.allocated = max(self.allocated, allocated)
        self.rss = max(self.rss, self.process.memory_info().rss)
        self.samples += 1

    def __enter__(self):
        self.sample()
        self.started = perf_counter()
        def observe():
            while not self.stop.wait(0.02):
                self.sample()
        self.thread = Thread(target=observe, name='archive-space-sampler')
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.sample()
        self.seconds = perf_counter() - self.started

    def report(self):
        return dict(seconds=self.seconds, sampled_peak_logical_bytes=self.logical,
                    sampled_peak_allocated_bytes=self.allocated, sampled_peak_rss_bytes=self.rss,
                    process_high_water_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * (1 if sys.platform == 'darwin' else 1024),
                    sample_interval_seconds=0.02, samples=self.samples)


def fixture_events(count, identity_order, variant):
    """Change facts BEFORE Standard Metrics, so final metrics describe these facts."""
    for event in events(count, identity_order):
        if isinstance(event, TurnFact):
            index = int(event.id.split('turn-')[1].split('-')[0])
            if variant == 'varied':
                event = replace(event, metadata={**event.metadata,
                    'dataset_row': index, 'plugin_label': sha256(str(index).encode()).hexdigest()},
                    profile=replace(event.profile, model=f'model-{index % 16}'),
                    usage={'prompt_tokens': 8 + index % 23, 'completion_tokens': 4,
                           'total_tokens': 12 + index % 23})
            elif variant == 'edges':
                kind = index % 8
                if kind in (1, 2):
                    event = replace(event, status='failed' if kind == 1 else 'cancelled',
                        error='fixture failure' if kind == 1 else 'fixture cancellation',
                        response_samples=(), first_ns=None, last_ns=None, usage={},
                        output_event_count=0, output_chars=0, finish_reason=None,
                        http_status=500 if kind == 1 else None, scenario_passed=False)
                    if isinstance(event, AttemptFact):
                        event = replace(event, error_origin='executor' if kind == 1 else 'cancelled',
                            trigger_ns=event.request_ns, decision_ns=event.request_ns + 1,
                            applied_ns=event.response_completed_ns)
                elif kind == 3:
                    event = replace(event, response_mode='non_stream', response_samples=(),
                                    first_ns=None, last_ns=None)
                elif kind == 4:
                    event = replace(event, output_kind='non_text', response_samples=(),
                                    first_ns=None, last_ns=None, usage={})
                elif kind == 5:
                    event = replace(event, usage={}, requested_output_tokens=None,
                        metadata={**event.metadata, 'null': None, 'unicode': '准确性',
                                  'nested': {'values': [1, 1.0, -0.0, None]}})
                elif kind == 6:
                    if isinstance(event, AttemptFact):
                        yield replace(event, status='failed', error='retry first attempt',
                            error_origin='executor', response_samples=(), first_ns=None,
                            last_ns=None, usage={}, http_status=503, output_chars=0,
                            output_event_count=0, response_completed_ns=event.request_ns + 10_000_000,
                            turn_completed_ns=event.request_ns + 10_000_000)
                    event = replace(event, attempts=2, request_ns=event.request_ns + 10_000_000,
                                    total_started_ns=event.request_ns)
        elif isinstance(event, SessionFact) and variant == 'edges':
            kind = event.ordinal % 8
            if kind in (1, 2):
                event = replace(event, status='failed' if kind == 1 else 'cancelled',
                                error='fixture terminal', succeeded_turns=0)
        yield event


def namespace(record, tag):
    changes = {}
    for field in fields(record):
        if field.name not in {'id', 'run_id', 'trial_id', 'session_id', 'turn_id', 'scope_id'}:
            continue
        value = getattr(record, field.name)
        if value == 'run':
            changes[field.name] = tag
        elif value == 'trial':
            changes[field.name] = tag + ':trial'
        elif isinstance(value, str) and value.startswith('trial:'):
            changes[field.name] = tag + ':' + value
    if isinstance(record, TrialFactRecord):
        changes['fact'] = replace(record.fact, trial_id=tag + ':trial')
    return replace(record, **changes) if changes else record


def create(directory, count, variant):
    directory.mkdir(parents=True)
    tag = directory.name
    run, trial = Run(id=tag, name=tag), Trial(id=tag + ':trial', name='fixture')
    run.register_trial(trial)
    backend = SQLiteSaver({'path': str(directory / 'results.db')}).create_backend(
        _snapshot(RunRecord, run), SaverContext(directory, CancellationToken()))
    diagnostics = {}
    with Peak(directory) as peak:
        backend.open()
        try:
            backend.write_batch(tuple(_trial_records(trial, None)))
            def source(n, identity_order):
                return fixture_events(n, identity_order, variant)
            original_trial_fact = fact_contract.TrialFact
            def trial_fact(*args, **kwargs):
                fact = original_trial_fact(*args, **kwargs)
                if variant == 'edges':
                    scheduling = fact.scheduling
                    extra = count // 8
                    fact = replace(fact, scheduling=replace(scheduling,
                        offered_attempts=scheduling.offered_attempts + extra,
                        admitted_attempts=scheduling.admitted_attempts + extra,
                        completed_attempts=scheduling.completed_attempts + extra))
                return fact
            # These two scoped substitutions affect only the reused synthetic
            # generator; production sources and execution paths are unchanged.
            with patch.object(result_storage, 'events', source), patch.object(fact_contract, 'TrialFact', trial_fact):
                for batch in result_storage.prepared_batches(count, 'prepared', 100, diagnostics):
                    backend.write_batch(tuple(namespace(r, tag) for r in batch))
            terminal = next(r for r in _trial_records(trial, None) if isinstance(r, TrialResultRecord))
            succeeded = backend.connection.execute("SELECT count(*) FROM turns WHERE status='succeeded'").fetchone()[0]
            backend.write_batch((replace(terminal, status='succeeded', session_count=count,
                turn_count=count, succeeded_turns=succeeded, failed_or_cancelled_turns=count-succeeded),))
            backend.finalize(replace(_snapshot(RunRecord, run), status='succeeded',
                                    counts={'sessions': count, 'turns': count}))
            checkpoint = perf_counter()
            receipt = backend.connection.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()
            assert receipt[0] == 0, receipt
            checkpoint = perf_counter() - checkpoint
        finally:
            backend.close()
    return dict(case=tag, requests=count, variant=variant, source=peak.report(),
                checkpoint_seconds=checkpoint, sqlite_bytes=(directory / 'results.db').stat().st_size,
                metrics_diagnostics=diagnostics)


def table_schema(db, table):
    columns = db.execute(f'PRAGMA table_info({quoted(table)})').fetchall()
    types = {'INTEGER': pa.int64(), 'REAL': pa.float64(), 'TEXT': pa.string(), 'BLOB': pa.binary()}
    schema = pa.schema([pa.field(c[1], types[c[2]], nullable=not c[3]) for c in columns])
    primary = [c[1] for c in sorted(columns, key=lambda c: c[5]) if c[5]]
    return schema, primary


def tables(db):
    return [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' "
                                    "AND name NOT LIKE 'sqlite_%' ORDER BY name")]


def seal(directory, repetition, rows):
    path = directory / 'results.db'
    staging = directory / f'.parquet-{repetition}'
    staging.mkdir()
    output = directory / 'parquet'
    with Peak(directory) as peak, closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as db:
        db.execute('BEGIN')
        layout = {}
        started = perf_counter()
        for table in tables(db):
            schema, primary = table_schema(db, table)
            order = ','.join(map(quoted, primary)) if primary else 'rowid'
            cursor = db.execute(f'SELECT * FROM {quoted(table)} ORDER BY {order}')
            count = 0
            with pq.ParquetWriter(staging / (table + '.parquet'), schema,
                                  compression='zstd', compression_level=3,
                                  use_dictionary=True, write_statistics=True) as writer:
                while batch := cursor.fetchmany(rows):
                    columns = [pa.array(values, type=field.type) for values, field in zip(zip(*batch), schema)]
                    writer.write_batch(pa.RecordBatch.from_arrays(columns, schema=schema), row_group_size=rows)
                    count += len(batch)
            layout[table] = dict(rows=count, bytes=(staging / (table + '.parquet')).stat().st_size,
                                 primary_key=primary, arrow_schema=str(schema))
        db.rollback()
        # Finish the closed files before atomic directory publication. Record
        # the archive fsync cost; it is not the source SQLite NORMAL durability.
        for file in staging.iterdir():
            with file.open('rb') as handle:
                os.fsync(handle.fileno())
        staging.rename(output)
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        write_seconds = perf_counter() - started
    for table in layout:
        digest = sha256()
        with (output / (table + '.parquet')).open('rb') as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        layout[table]['file_sha256'] = digest.hexdigest()
    return dict(repetition=repetition, row_group_rows=rows, compression='zstd:3',
                seal_write_close_fsync_publish_seconds=write_seconds,
                peak=peak.report(), parquet_bytes=files_size(output)[0], tables=layout)


def values_hash(rows):
    digest = sha256()
    count = 0
    for row in rows:
        count += 1
        for value in row:
            if value is None:
                data = b'n'
            elif isinstance(value, int):
                data = b'i' + str(value).encode()
            elif isinstance(value, float):
                data = b'f' + struct.pack('>d', value)
            elif isinstance(value, str):
                data = b's' + value.encode()
            elif isinstance(value, bytes):
                data = b'b' + value
            else:
                raise TypeError(type(value))
            digest.update(struct.pack('>Q', len(data)))
            digest.update(data)
        digest.update(b'\xff')
    return dict(rows=count, sha256=digest.hexdigest())


def parquet_rows(path):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8192):
        yield from zip(*(column.to_pylist() for column in batch.columns))


def validate(directory):
    started = perf_counter()
    report = {}
    with closing(sqlite3.connect(directory / 'results.db')) as db:
        assert db.execute('PRAGMA integrity_check').fetchall() == [('ok',)]
        assert not db.execute('PRAGMA foreign_key_check').fetchall()
        for table in tables(db):
            _, primary = table_schema(db, table)
            order = ','.join(map(quoted, primary)) if primary else 'rowid'
            source = values_hash(db.execute(f'SELECT * FROM {quoted(table)} ORDER BY {order}'))
            archive = values_hash(parquet_rows(directory / 'parquet' / (table + '.parquet')))
            assert source == archive, (table, source, archive)
            report[table] = source
    return dict(seconds=perf_counter()-started, exact_tables=report, mismatches=0,
                comparison='every row/column; int64, float64 bits, null, UTF-8, BLOB and encoded metric JSON')


class Cursor:
    def __init__(self, columns, rows):
        self.rows = [dict(zip(columns, row)) for row in rows]

    def __iter__(self):
        return iter(self.rows)


class QueryDB:
    """Experimental DB-API projection to reuse the SAME existing entity decoder."""

    def __init__(self, directory, kind):
        self.kind = kind
        if kind == 'sqlite':
            self.db = sqlite3.connect((directory / 'results.db').as_uri() + '?mode=ro', uri=True)
        else:
            self.db = duckdb.connect(config={'threads': 1, 'memory_limit': '512MB'})
            for file in sorted((directory / 'parquet').glob('*.parquet')):
                self.db.read_parquet(str(file)).create_view(file.stem)

    def execute(self, sql, params=()):
        cursor = self.db.execute(sql, params)
        return Cursor([c[0] for c in cursor.description], cursor.fetchall())

    def close(self):
        self.db.close()


def canonical(value):
    if isinstance(value, Mapping):
        return {key: canonical(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if isinstance(value, float):
        return {'float64_bits': struct.pack('>d', value).hex()}
    return value


def result_hash(value):
    return sha256(json.dumps(canonical(value), sort_keys=True, ensure_ascii=False,
                             separators=(',', ':')).encode()).hexdigest()


def request_query(db, table, identity=None, offset=0, detail=True):
    columns, source = entities.entity_select(table)
    where = 'WHERE k.turn_id=?' if identity is not None else ''
    params = [identity] if identity is not None else []
    order = 'r.attempt' if identity is not None and table == 'attempts' else 'k.id'
    params.extend([50, offset])
    rows = db.execute(f'SELECT {columns} {source} {where} ORDER BY {order} LIMIT ? OFFSET ?', params)
    return entities.decode_entities(db, table, rows, detail=detail)


def stats(samples):
    ordered = sorted(samples)
    return dict(samples_seconds=samples, median_seconds=statistics.median(samples),
                p95_nearest_rank_seconds=ordered[max(0, math.ceil(len(samples)*.95)-1)],
                max_seconds=max(samples))


def queries(directory, count, repetitions):
    tag = directory.name
    identities = [f'{tag}:trial:turn-{i}-0' for i in dict.fromkeys(
        [0, 1, 2, 3, 4, 5, 6, count//10, count//2, count-1])]
    operations = [('turn_detail', 'turns', identity, 0, True) for identity in identities]
    operations += [('attempt_detail', 'attempts', identity, 0, True) for identity in identities]
    operations += [(f'page_{offset}', 'turns', None, offset, False)
                   for offset in (0, count//2, max(0, count-50))]
    expected = {}
    output = {kind: {'connection_first': [], 'warm': {}} for kind in ('sqlite','parquet')}
    for iteration in range(repetitions):
        for kind in (('sqlite','parquet') if iteration % 2 == 0 else ('parquet','sqlite')):
            start = perf_counter()
            db = QueryDB(directory, kind)
            opened = perf_counter()-start
            try:
                querying = perf_counter()
                value = request_query(db, 'turns', identities[-1])
                elapsed = perf_counter()-querying
                fingerprint = result_hash(value)
                assert value, identities[-1]
                key = 'first'
                expected.setdefault(key, fingerprint)
                assert expected[key] == fingerprint
                output[kind]['connection_first'].append(dict(open_seconds=opened,
                    query_decode_seconds=elapsed, total_seconds=opened+elapsed))
                for label, table, identity, offset, detail in operations:
                    querying = perf_counter()
                    value = request_query(db, table, identity, offset, detail)
                    elapsed = perf_counter()-querying
                    assert value, (kind, table, identity, offset)
                    key = (label, identity, offset)
                    fingerprint = result_hash(value)
                    expected.setdefault(key, fingerprint)
                    assert expected[key] == fingerprint, key
                    output[kind]['warm'].setdefault(label, []).append(elapsed)
            finally:
                db.close()
    for kind in output:
        output[kind]['warm'] = {key: stats(value) for key, value in output[kind]['warm'].items()}
    return dict(results=output, public_query_result_hashes={str(k):v for k,v in expected.items()},
                public_fact_metric_mismatches=0, decoder='production entities.decode_entities for both formats',
                cache_scope='fresh connections / repeated connections; OS file cache NOT evicted',
                page_scope='50 rows with compact production list decoder; excludes total-count query')


def cross_run(root, repetitions):
    directories = sorted(p for p in root.iterdir() if (p / 'results.db').exists())
    sql = {
        'formal_summary': "SELECT r.run_id,m.scope,m.scope_id,m.window,m.metric,m.statistic,m.value,"
            "m.availability,m.unavailable_reason,m.valid_sample_count,m.expected_sample_count,"
            "m.missing_sample_count,d.payload FROM {prefix}metric_points m JOIN {prefix}runs r "
            "ON r.run_id=m.run_id JOIN {prefix}metric_definitions d ON d.id=m.definition_id",
        'request_scan': "SELECT t.run_id,u.status,count(*) AS requests,min(f.started_ns) AS first_ns,"
            "max(f.response_completed_ns) AS last_ns,sum(CAST(json_extract(f.usage_json,'$.completion_tokens') AS BIGINT)) AS completion_tokens "
            "FROM {prefix}turns u JOIN {prefix}turn_keys k ON k.id=u.id "
            "JOIN {prefix}sessions s ON s.id=k.session_key JOIN {prefix}trials t ON t.trial_id=s.trial_id "
            "JOIN {prefix}request_facts f ON f.id=u.fact_id GROUP BY t.run_id,u.status",
    }
    expected, output = {}, {kind: {} for kind in ('sqlite','parquet')}
    for iteration in range(repetitions):
        for kind in (('sqlite','parquet') if iteration%2 == 0 else ('parquet','sqlite')):
            start = perf_counter()
            db = sqlite3.connect(':memory:') if kind == 'sqlite' else duckdb.connect(
                config={'threads': 1, 'memory_limit': '512MB'})
            try:
                for index, directory in enumerate(directories):
                    alias = f'r{index}'
                    if kind == 'sqlite':
                        db.execute(f'ATTACH DATABASE ? AS {alias}', (str(directory / 'results.db'),))
                    else:
                        db.execute(f'CREATE SCHEMA {alias}')
                        db.execute(f"SET schema='{alias}'")
                        for file in sorted((directory / 'parquet').glob('*.parquet')):
                            db.read_parquet(str(file)).create_view(file.stem)
                opened = perf_counter()-start
                for label, statement in sql.items():
                    query = ' UNION ALL '.join(statement.format(prefix=f'r{i}.') for i in range(len(directories)))
                    query += ' ORDER BY ALL' if kind == 'parquet' else ''
                    reading = perf_counter()
                    rows = db.execute(query).fetchall()
                    elapsed = perf_counter()-reading
                    rows.sort(key=lambda r: json.dumps(canonical(r), sort_keys=True))
                    fingerprint = result_hash(rows)
                    expected.setdefault(label, fingerprint)
                    assert fingerprint == expected[label], label
                    output[kind].setdefault(label, []).append(dict(query_seconds=elapsed,
                                                                  setup_seconds=opened, rows=len(rows)))
            finally:
                db.close()
    return dict(runs=[p.name for p in directories], results=output, result_hashes=expected, mismatches=0,
                semantics='formal points copied without recomputation; request scan is analysis, not a new formal metric')


def child(phase, directory, **kwargs):
    args = [sys.executable, '-m', 'benchmarks.sqlite_parquet_comparison', '--phase', phase,
            '--directory', str(directory)]
    for key, value in kwargs.items():
        args.extend(['--' + key.replace('_','-'), str(value)])
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'{phase} failed ({result.returncode}): ' + result.stderr)
    if result.stderr:
        raise RuntimeError('Unexpected child stderr: ' + result.stderr)
    return json.loads(result.stdout)


def main(args):
    if args.phase:
        directory = args.directory.resolve()
        if args.phase == 'create':
            output = create(directory, args.requests, args.variant)
        elif args.phase == 'seal':
            output = seal(directory, args.repetition, args.row_group_rows)
        elif args.phase == 'validate':
            output = validate(directory)
        elif args.phase == 'queries':
            output = queries(directory, args.requests, args.repetitions)
        else:
            output = cross_run(directory, args.repetitions)
        print(json.dumps(output, sort_keys=True))
        return
    report = dict(environment={'platform': platform.platform(), 'python': sys.version,
        'duckdb': duckdb.__version__, 'pyarrow': pa.__version__,
        'git_head': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        'script_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
        'uv_lock_sha256': sha256(Path('uv.lock').read_bytes()).hexdigest()}, cases=[])
    with TemporaryDirectory(prefix='sqlite-parquet-comparison-') as temporary:
        root = Path(temporary)
        cases = [(f'standard-{args.requests}', args.requests, 'standard'),
                 (f'varied-{args.requests}', args.requests, 'varied'), ('edges-64', 64, 'edges')]
        if args.large_requests:
            cases.append((f'standard-{args.large_requests}', args.large_requests, 'standard'))
        for tag, count, variant in cases:
            print(f'Creating {tag}', file=sys.stderr, flush=True)
            directory = root / tag
            case = child('create', directory, requests=count, variant=variant)
            case['sealing'] = []
            for repetition in range(args.repetitions):
                if repetition:
                    import shutil
                    shutil.rmtree(directory / 'parquet')  # Only this experiment's disposable output.
                print(f'Sealing {tag} #{repetition+1}', file=sys.stderr, flush=True)
                case['sealing'].append(child('seal', directory, repetition=repetition,
                                             row_group_rows=args.row_group_rows))
                assert case['sealing'][-1]['tables'] == case['sealing'][0]['tables'], 'archive changed between repetitions'
            print(f'Validating/querying {tag}', file=sys.stderr, flush=True)
            case['validation'] = child('validate', directory)
            case['queries'] = child('queries', directory, requests=count, repetitions=args.repetitions)
            report['cases'].append(case)
        print('Cross-Run queries', file=sys.stderr, flush=True)
        report['cross_run'] = child('cross', root, repetitions=args.repetitions)
    report['limitations'] = [
        'No production path changes, no HTTP capacity claim; synthetic body-free formal fixtures.',
        'Parquet mirrors current compact SQLite tables; not an alternate wide-table schema or new metric formula.',
        'Source data generated once; repeats seal/query SAME committed facts in fresh processes.',
        '20ms sampled file/RSS peaks plus endpoints; OS file cache not flushed.',
        'Parquet fsync is included; source SQLite synchronous=NORMAL remains unchanged.',
        'Indexes/schema/triggers belong to SQLite file; Parquet contains all table content, no secondary indexes.',
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, 'wt', encoding='utf-8') as handle:
        json.dump(report, handle, sort_keys=True)
    print(str(args.output.resolve()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('benchmarks/reports/sqlite-parquet-comparison-2026-10-12.json.gz'))
    parser.add_argument('--requests', type=int, default=100_000)
    parser.add_argument('--large-requests', type=int, default=1_000_000)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--row-group-rows', type=int, default=65_536)
    parser.add_argument('--phase', choices=('create','seal','validate','queries','cross'))
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--variant', choices=('standard','varied','edges'), default='standard')
    parser.add_argument('--repetition', type=int, default=0)
    args = parser.parse_args()
    if args.requests < 8 or args.large_requests < 0 or args.repetitions < 1 or args.row_group_rows < 1:
        parser.error('invalid experiment size')
    if args.phase and args.directory is None:
        parser.error('--phase requires --directory')
    gc.disable()
    main(args)
