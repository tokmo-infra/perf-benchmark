"""Permanent formal views and explicit seven-day SQLite drilldown routing."""
from contextlib import contextmanager
from pathlib import Path

import duckdb
import orjson
import pyarrow.parquet as pq

from . import SQLiteResultReader, _result_record, summaries
from .archive import SUMMARY_TABLES, archive_path, detail_lease, read_info
from .schema import RESULT_TABLES
from .layout import ENTITY_TABLES


class _Row(dict):
    """Mapping with the positional access used by the existing formal decoder."""
    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            return tuple(self.values())[key]
        return super().__getitem__(key)


class SummaryReader(SQLiteResultReader):
    """Reads only permanent files, even while SQLite/details still exist."""
    def __init__(self, directory):
        self.path = Path(directory).resolve()
        self.info = read_info(self.path, verify=True)

    @contextmanager
    def _connection(self, query=None):
        with duckdb.connect(config={'threads': 1}) as db:
            for table in SUMMARY_TABLES:
                # These are our fixed Reader queries, not arbitrary user SQL.
                if query is not None and table not in query:
                    continue
                path = self.path / 'summary' / (table + '.parquet')
                db.sql('SELECT row_number() OVER () AS __archive_order,* '
                       'FROM read_parquet(?, hive_partitioning=false)', params=[str(path)]).create_view(table)
            yield db

    def _rows(self, query, parameters=()):
        prefix = summaries.query_prefix(query).replace('json_extract(', 'json_extract_string(')
        # rowid is a local ordering aid, never an identity in the archive.
        prefix = prefix.replace('p.rowid AS rowid,', '')
        with self._connection(prefix + query) as db:
            cursor = db.execute(prefix + query, parameters)
            names = [column[0] for column in cursor.description]
            return [_Row((name, value) for name, value in zip(names, row) if name != "__archive_order")
                    for row in cursor.fetchall()]

    def run_ids(self):
        return (self.info['run_id'],)

    def summary(self, run_id):
        record = self.run_record(run_id)
        counts = pq.ParquetFile(self.path / 'summary' / 'run_totals.parquet').read().to_pylist()[0]
        return {'run_id': record.id, 'name': record.name, 'status': record.status, 'error': record.error,
                'trial_count': counts['trials'], 'session_count': counts['sessions'],
                'turn_count': counts['turns'], 'metric_count': counts['metric_points']}

    def iter_summary_records(self, run_id):
        with self._connection() as db:
            for table in RESULT_TABLES:
                if table in ENTITY_TABLES:
                    continue
                select, source = summaries.select(table)
                select = select.replace('json_extract(', 'json_extract_string(').replace('r.*', 'r.* EXCLUDE(__archive_order)')
                where = 'r.run_id=?'
                if table == 'trial_facts':
                    select += ',t.run_id,t.status'
                    source += ' JOIN trials t USING(trial_id)'
                    where = 't.run_id=?'
                cursor = db.execute(f'SELECT {select} {source} WHERE {where} ORDER BY r.__archive_order', (run_id,))
                names = [column[0] for column in cursor.description]
                while rows := cursor.fetchmany(500):
                    for row in rows:
                        yield _result_record(table, _Row(zip(names, row)))[0]

    def metric_series(self, run_id, *, trial_id, metric, start_ns, end_ns, limit=10000):
        if type(limit) is not int or not 1 <= limit <= 10000:
            raise ValueError('invalid metric series limit')
        if type(start_ns) is not int or type(end_ns) is not int or start_ns > end_ns:
            raise ValueError('invalid metric series time range')
        where = "p.run_id=? AND p.scope='trial' AND p.window LIKE 'timeline:%' AND p.metric=? AND w.end_ns>? AND w.start_ns<?"
        params = [run_id, metric, start_ns, end_ns]
        if trial_id is not None:
            where += " AND p.scope_id=?"
            params.append(trial_id)
        join = 'metric_points p JOIN metric_windows w USING(run_id,scope,scope_id,"window")'
        total = self._rows(f'SELECT count(*) FROM {join} WHERE {where}', params)[0][0]
        rows = self._rows(f'SELECT p.*,w.start_ns,w.end_ns FROM {join} WHERE {where} '
                          'ORDER BY w.start_ns,p.scope_id,p.statistic LIMIT ?', [*params, limit])
        return {'total': total, 'items': tuple({k: v for k, v in row.items()
                                             if k not in {'revision', 'definition_id'}} for row in rows)}

    def request_page(self, *args, **kwargs):
        raise ValueError('detail_unavailable_in_summary')

    def _iter_entities(self, *args, **kwargs):
        raise ValueError('detail_unavailable_in_summary')
        yield  # A summary never carries request rows.

    def read_updates(self, *args, **kwargs):
        raise ValueError('archive_is_frozen')


class ArchiveReader(SummaryReader):
    """Formal summaries use Parquet; point reads use the original SQLite only."""
    def _live(self):
        source = self.path.parent / self.info['source_name']
        if archive_path(source) != self.path:
            raise ValueError('archive source location mismatch')
        if not source.exists():
            raise ValueError('detail_unavailable')
        reader = SQLiteResultReader(source)
        identity = reader._rows('SELECT database_id FROM result_identity')[0][0]
        revision = reader._rows('SELECT revision FROM result_state')[0][0]
        if (identity, revision) != (self.info['database_id'], self.info['revision']):
            raise ValueError('detail_source_changed')
        return reader

    def read_updates(self, *args, **kwargs):
        source = self.path.parent / self.info['source_name']
        with detail_lease(source):
            return self._live().read_updates(*args, **kwargs)

    def request_page(self, *args, **kwargs):
        source = self.path.parent / self.info['source_name']
        with detail_lease(source):
            return self._live().request_page(*args, **kwargs)

    def _iter_entities(self, *args, **kwargs):
        source = self.path.parent / self.info['source_name']
        with detail_lease(source):
            yield from self._live()._iter_entities(*args, **kwargs)


def analyze(locations, query, *, details=False):
    """One read-only query over fixed Run snapshots; dictionaries stay Run-local."""
    from contextlib import ExitStack
    from .archive import DETAIL_TABLES
    directories = []
    for location in locations:
        path = Path(location).resolve()
        directories.append(path if path.is_dir() else archive_path(path))
    if not directories or len(set(directories)) != len(directories):
        raise ValueError('analysis requires distinct archives')
    with ExitStack() as resources:
        infos = [(directory, read_info(directory, verify=True)) for directory in directories]
        if len({info['run_id'] for _, info in infos}) != len(infos):
            raise ValueError('analysis requires unique Run IDs')
        if details:
            for directory, info in sorted(infos, key=lambda item: item[1]['run_id']):
                resources.enter_context(detail_lease(directory.parent / info['source_name']))
                for entry in orjson.loads(info['files_json']):
                    if entry['path'].startswith('details/') and not (directory / entry['path']).is_file():
                        raise ValueError('detail_unavailable')
                read_info(directory, verify=True, verify_details=True)
        db = resources.enter_context(duckdb.connect(config={'threads': 1}))
        statements = db.extract_statements(query)
        if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
            raise ValueError('analysis accepts one SELECT statement')
        views = {name: [] for name in ('runs', 'trials', 'trial_facts', 'metrics', 'windows', 'groups')}
        if details:
            views.update(requests=[], attempts=[], sessions=[])
        for index, (directory, info) in enumerate(infos):
            namespace = 'r' + str(index)
            db.execute(f'CREATE SCHEMA {namespace}')
            db.execute(f'USE {namespace}')
            for tier, tables in (('summary', SUMMARY_TABLES), ('details', DETAIL_TABLES if details else ())):
                for table in tables:
                    if tier == 'details' and table == 'metric_definitions':
                        # Request metrics use the complete dictionary, in a different view.
                        table_name = 'request_metric_definitions'
                    else:
                        table_name = table
                    db.from_parquet(str(directory / tier / (table + '.parquet')), hive_partitioning=False).create_view(table_name)
            db.execute('USE main')
            for name, table in (('runs', 'runs'), ('trials', 'trials'), ('trial_facts', 'trial_facts'), ('windows', 'metric_windows')):
                if name == 'trial_facts':
                    views[name].append(f'SELECT t.run_id,f.* FROM {namespace}.trial_facts f JOIN {namespace}.trials t USING(trial_id)')
                else:
                    views[name].append(f'SELECT * FROM {namespace}.{table}')
            for name, table in (('metrics', 'metric_points'), ('groups', 'grouped_metrics')):
                views[name].append(f'''SELECT p.*,json_extract_string(d.payload,'$.unit') AS unit,
                    json_extract_string(d.payload,'$.formula_version') AS formula_version,
                    json_extract_string(d.payload,'$.source') AS source
                    FROM {namespace}.{table} p JOIN {namespace}.metric_definitions d ON d.id=p.definition_id''')
            if details:
                views['sessions'].append(f'SELECT t.run_id,s.* FROM {namespace}.sessions s JOIN {namespace}.trials t USING(trial_id)')
                for name, table, key in (('requests', 'turns', 'id'), ('attempts', 'attempts', 'turn_key')):
                    views[name].append(f'''SELECT t.run_id,t.trial_id,s.session_id,k.turn_id,q.*,
                        f.* EXCLUDE(id) FROM {namespace}.{table} q
                        JOIN {namespace}.turn_keys k ON k.id=q.{key}
                        JOIN {namespace}.sessions s ON s.id=k.session_key
                        JOIN {namespace}.trials t ON t.trial_id=s.trial_id
                        LEFT JOIN {namespace}.request_facts f ON f.id=q.fact_id''')
        for name, selections in views.items():
            db.execute(f'CREATE VIEW {name} AS ' + ' UNION ALL '.join(selections))
        cursor = db.execute(query)
        names = [column[0] for column in cursor.description]
        return {'run_ids': [info['run_id'] for _, info in infos], 'content': 'details' if details else 'summary',
                'items': [dict(zip(names, row)) for row in cursor.fetchall()]}
