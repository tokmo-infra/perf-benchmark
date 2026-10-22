"""Shared summary definitions; typed point values stay in their own tables."""
from dataclasses import fields

from perf_benchmark.pipeline.contracts.results import GroupedMetricRecord, MetricPoint
from .encoding import _json_dump, _validate_value

TABLES = {'metric_points', 'grouped_metrics'}


def definition(record):
    return _json_dump({'kind': 'group' if isinstance(record, GroupedMetricRecord) else 'point',
        'metric': record.metric, 'unit': record.unit, 'formula_version': record.formula_version,
        'source': getattr(record, 'source', None),
        'statistics': tuple(record.statistics) if isinstance(record, GroupedMetricRecord) else ('value',)})


def rows(records, run_id, definitions):
    selected = [(r, definition(r)) for r in records if isinstance(r, (MetricPoint, GroupedMetricRecord))]
    ids = definitions.resolve('metric_definitions', [payload for _, payload in selected])
    output = {'point': [], 'group': []}
    for record, payload in selected:
        _validate_value(record)
        group = isinstance(record, GroupedMetricRecord)
        value = (run_id, *(getattr(record, f.name) for f in fields(record)
                           if f.name not in {'unit', 'formula_version', 'source'}), ids[payload])
        output['group' if group else 'point'].append(value)
    return output


def select(table, alias='r'):
    if table not in TABLES:
        return f'{alias}.*', f'FROM {table} {alias}'
    names = ('unit', 'formula_version', 'source') if table == 'metric_points' else ('unit', 'formula_version')
    extra = ','.join(f"json_extract(d.payload,'$.{name}') AS {name}" for name in names)
    return f'{alias}.*,{extra}', f'FROM {table} {alias} JOIN metric_definitions d ON d.id={alias}.definition_id'


def query_prefix(query):
    tables = sorted(table for table in TABLES if table in query)
    if not tables:
        return ''
    views = []
    for table in tables:
        columns, source = select(table, 'p')
        source = source.replace('FROM ' + table, 'FROM main.' + table)
        views.append(f'{table} AS (SELECT p.rowid AS rowid,{columns} {source})')
    return 'WITH ' + ','.join(views) + ' '
