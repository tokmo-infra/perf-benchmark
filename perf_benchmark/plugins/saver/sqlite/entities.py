"""Normalized entity persistence and lossless, batched read projections."""
from collections import OrderedDict, defaultdict
from hashlib import sha256
from itertools import chain

import sqlite3

import orjson

from perf_benchmark.pipeline.contracts.results import AttemptResultRecord, SessionResultRecord, TurnResultRecord
from perf_benchmark.plugins.saver.sqlite.encoding import _json_dump, _validate_value
from perf_benchmark.plugins.saver.sqlite.metric_codec import encode_metrics, metric_delta, metric_patch, restore_metrics
from perf_benchmark.plugins.saver.sqlite.layout import REQUEST_COLUMNS, REQUEST_TEXT, DEFINITIONS, ENTITY_TABLES, MEASUREMENT_PATHS, MEASUREMENT_COLUMNS


def digest(payload):
    return sha256(payload.encode()).digest()


def _chunks(values, size=400):
    for offset in range(0, len(values), size):
        yield values[offset:offset + size]


def _insert(db, table, rows, *, columns=None, suffix='', revision=None):
    if not rows:
        return []
    if table in ENTITY_TABLES:
        assert columns is not None and revision is not None
        columns += ',revision'
        rows = [(*row, revision) for row in rows]
    width = len(rows[0])
    maximum = min(400, db.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER) // width)
    output = []
    for batch in _chunks(rows, maximum):
        prefix = f'INSERT INTO {table}' + (f'({columns})' if columns else '') + ' VALUES '
        sql = prefix + ','.join('(' + ','.join('?' for _ in range(width)) + ')' for _ in batch)
        output.extend(db.execute(sql + suffix, tuple(chain.from_iterable(batch))).fetchall())
    return output


class Definitions:
    """One bounded exact-content cache; the database owns the full dictionary."""
    def __init__(self, db):
        self.db = db
        self.cache = OrderedDict()
        self.bytes = 0

    def clear(self):
        self.cache.clear()
        self.bytes = 0

    def _remember(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
            return
        size = len(key[1].encode()) + 128
        if size > 4 * 1024 * 1024:
            return
        while self.cache and (len(self.cache) >= 4096 or self.bytes + size > 4 * 1024 * 1024):
            old, _ = self.cache.popitem(last=False)
            self.bytes -= len(old[1].encode()) + 128
        self.cache[key] = value
        self.bytes += size

    def resolve(self, table, payloads):
        assert table in DEFINITIONS
        payloads = tuple(dict.fromkeys(payloads))
        output = {p: self.cache[(table, p)] for p in payloads if (table, p) in self.cache}
        missing = {p: digest(p) for p in payloads if p not in output}
        by_hash = defaultdict(set)
        for payload, fingerprint in missing.items():
            by_hash[fingerprint].add(payload)
        for batch in _chunks(tuple(by_hash)):
            for identity, fingerprint, payload in self.db.execute(
                f"SELECT id,digest,payload FROM {table} WHERE digest IN ({','.join('?' for _ in batch)})", batch
            ):
                if payload in by_hash[fingerprint]:  # Hash collision is never equality.
                    output[payload] = identity
        new = [(fingerprint, payload) for payload, fingerprint in missing.items() if payload not in output]
        for identity, payload in _insert(self.db, table, new, columns='digest,payload', suffix=' RETURNING id,payload'):
            output[payload] = identity
        for payload, identity in output.items():
            self._remember((table, payload), identity)
        return output



def request_row(request, usage, profiles):
    for name in REQUEST_COLUMNS:
        value = request.get(name)
        if value is not None and (not isinstance(value, str) if name in REQUEST_TEXT else type(value) is not int):
            raise TypeError(f'request fact {name} has an invalid scalar type')
    names = (*REQUEST_COLUMNS, 'profile', 'trace_context')
    present = sum(1 << i for i, name in enumerate(names) if name in request)
    profile = request.get('profile')
    return (present, *(request.get(name) for name in REQUEST_COLUMNS),
            profiles[_json_dump(profile)] if profile is not None else None,
            _json_dump(request['trace_context']) if request.get('trace_context') is not None else None,
            _json_dump({k: v for k, v in request.items() if k not in names}), _json_dump(usage))


def pack_measurements(table, measurement):
    extras = dict(measurement)
    if isinstance(extras.get('lifecycle'), dict) or hasattr(extras.get('lifecycle'), 'items'):
        extras['lifecycle'] = dict(extras['lifecycle'])
    mask, values = 0, []
    for i, path in enumerate(MEASUREMENT_PATHS[table]):
        parent, _, name = path.rpartition('.')
        source = extras.get(parent) if parent else extras
        present = isinstance(source, dict) and name in source
        value = source.pop(name) if present else None
        if value is not None and type(value) is not int:
            raise TypeError('measurement timestamps must be integer nanoseconds')
        mask |= int(present) << i
        values.append(value)
    return extras, (mask, *values)


def unpack_measurements(table, row, extras):
    for i, (path, column) in enumerate(zip(MEASUREMENT_PATHS[table], MEASUREMENT_COLUMNS[table])):
        if row['measure_mask'] & (1 << i):
            parent, _, name = path.rpartition('.')
            target = extras.setdefault(parent, {}) if parent else extras
            target[name] = row[column]
    return extras


class EntityStore:
    def __init__(self, db):
        self.db = db
        self.definitions = Definitions(db)
        self.next_fact_id = None
        self.revision = 0
        self.columns = {table: ','.join(c[1] for c in db.execute(f'PRAGMA table_info({table})')
                                        if c[1] != 'revision') for table in ENTITY_TABLES}

    def clear(self):
        self.definitions.clear()
        self.next_fact_id = None

    def write(self, records):
        if not records:
            return
        unique, ordered = {}, []
        for record in records:
            if isinstance(record, SessionResultRecord):
                ordered.append(record)
                continue
            key = (type(record), record.turn_id, record.attempt) if isinstance(record, AttemptResultRecord) else (type(record), record.id)
            previous = unique.get(key)
            if previous is not None:
                if _json_dump(previous) != _json_dump(record):
                    raise sqlite3.IntegrityError('conflicting terminal result')
            else:
                unique[key] = record
                ordered.append(record)
        records = ordered
        values = {table: [] for table in DEFINITIONS}
        encoded = {}
        for r in records:
            # Validate all metadata, extensions and metrics at the persistence boundary.
            _validate_value(r)
            encoded[id(r)] = encode_metrics(r.metrics)
            values['metric_definitions'].append(encoded[id(r)][0])
            if r.error is not None:
                values['error_definitions'].append(_json_dump(r.error))
            if isinstance(r, AttemptResultRecord):
                values['attempt_plans'].append(_json_dump(r.plan))
                request = r.measurements
            elif isinstance(r, TurnResultRecord):
                request = r.measurements.get('request', {})
            else:
                continue
            if request.get('profile') is not None:
                values['request_profiles'].append(_json_dump(request['profile']))
        definitions = {table: self.definitions.resolve(table, payloads) for table, payloads in values.items()}
        metric_defs = definitions['metric_definitions']
        errors = definitions['error_definitions']
        def error(r):
            return errors[_json_dump(r.error)] if r.error is not None else None

        def metrics(r):
            definition, numbers = encoded[id(r)]
            return [metric_defs[definition], numbers]
        sessions = [r for r in records if isinstance(r, SessionResultRecord)]
        session_rows = []
        for r in sessions:
            extras, measurements = pack_measurements('sessions', r.measurements)
            session_rows.append((r.id, r.trial_id, r.ordinal, r.status, r.started_at, r.finished_at, error(r),
                _json_dump(r.metadata), _json_dump(extras), _json_dump(metrics(r)), r.turn_count, *measurements))
        session_columns = ('session_id', 'trial_id', 'ordinal', 'status', 'started_at', 'finished_at',
                           'error_id', 'metadata_json', 'measurements_json', 'metrics_json', 'turn_count',
                           'measure_mask', *MEASUREMENT_COLUMNS['sessions'])
        _insert(self.db, 'sessions', session_rows, columns=','.join(session_columns), revision=self.revision,
            suffix=' ON CONFLICT(session_id) DO UPDATE SET revision=excluded.revision,' + ','.join(
                f'{name}=excluded.{name}' for name in session_columns if name != 'session_id') + ' WHERE ' + ' OR '.join(
                f'sessions.{name} IS NOT excluded.{name}' for name in session_columns if name != 'session_id'))
        requests = [r for r in records if not isinstance(r, SessionResultRecord)]
        session_ids = tuple(dict.fromkeys(r.session_id for r in requests))
        session_keys = {}
        for batch in _chunks(session_ids):
            session_keys.update((public, key) for key, public in self.db.execute(
                f"SELECT id,session_id FROM sessions WHERE session_id IN ({','.join('?' for _ in batch)})", batch))
        identities = {}
        for r in requests:
            public = r.turn_id if isinstance(r, AttemptResultRecord) else r.id
            key = session_keys[r.session_id]
            if public in identities and identities[public] != key:
                raise ValueError('conflicting Turn parent')
            identities[public] = key
        created_keys = _insert(self.db, 'turn_keys', list(identities.items()), columns='turn_id,session_key',
                               suffix=' ON CONFLICT(turn_id) DO NOTHING RETURNING id,turn_id,session_key')
        turn_keys = {public: key for key, public, _ in created_keys}
        new_keys = set(turn_keys.values())
        missing_keys = tuple(public for public in identities if public not in turn_keys)
        for batch in _chunks(missing_keys):
            for key, public, parent in self.db.execute(
                f"SELECT id,turn_id,session_key FROM turn_keys WHERE turn_id IN ({','.join('?' for _ in batch)})", batch
            ):
                if parent != identities[public]:
                    raise ValueError('conflicting Turn parent')
                turn_keys[public] = key
        # Fetch only these entities, never retain an unbounded identity/fact cache.
        existing_attempts, existing_turns, facts = {}, {}, {}
        for batch in _chunks(tuple(key for key in turn_keys.values() if key not in new_keys)):
            for row in self.db.execute(f"SELECT * FROM turns WHERE id IN ({','.join('?' for _ in batch)})", batch):
                existing_turns[row[0]] = tuple(row[:-1])
        wanted = set()
        for r in requests:
            if isinstance(r, AttemptResultRecord):
                wanted.add((turn_keys[r.turn_id], r.attempt))
            elif r.measurements.get('request', {}).get('attempts'):
                wanted.add((turn_keys[r.id], r.measurements['request']['attempts']))
        for batch in _chunks(tuple(pair for pair in wanted if pair[0] not in new_keys)):
            for row in self.db.execute(
                "SELECT * FROM attempts WHERE (turn_key,attempt) IN (" + ','.join('(?,?)' for _ in batch) + ')',
                tuple(chain.from_iterable(batch)),
            ):
                existing_attempts[(row[1], row[2])] = tuple(row[:-1])
        bases = {row[10] for row in existing_turns.values() if row[10] is not None}
        for batch in _chunks(tuple(bases)):
            for row in self.db.execute(f"SELECT * FROM attempts WHERE id IN ({','.join('?' for _ in batch)})", batch):
                existing_attempts[(row[1], row[2])] = tuple(row[:-1])
        fact_ids = {r[6] for r in existing_attempts.values()} | {r[9] for r in existing_turns.values() if r[9] is not None}
        for batch in _chunks(tuple(fact_ids)):
            facts.update((row[0], tuple(row[1:])) for row in self.db.execute(
                f"SELECT * FROM request_facts WHERE id IN ({','.join('?' for _ in batch)})", batch))
        # Reuse only request facts belonging to this Turn; equality is exact.
        candidates = defaultdict(dict)
        for (key, _), row in existing_attempts.items():
            candidates[key][facts[row[6]]] = row[6]
        for key, row in existing_turns.items():
            if row[9] is not None:
                candidates[key][facts[row[9]]] = row[9]
        pending_facts = []
        record_facts = {}
        for r in requests:
            public = r.turn_id if isinstance(r, AttemptResultRecord) else r.id
            key = turn_keys[public]
            request = r.measurements if isinstance(r, AttemptResultRecord) else r.measurements.get('request', {})
            fact = request_row(request, r.usage, definitions['request_profiles'])
            if fact not in candidates[key]:
                candidates[key][fact] = -(len(pending_facts) + 1)
                pending_facts.append((None, *fact))
            record_facts[id(r)] = candidates[key][fact]
        # SQLite does not guarantee RETURNING row order. The single writer
        # assigns these private identities before insertion, never by output order.
        if pending_facts and self.next_fact_id is None:
            self.next_fact_id = self.db.execute('SELECT coalesce(max(id),0)+1 FROM request_facts').fetchone()[0]
        first = self.next_fact_id or 1
        new_facts = range(first, first + len(pending_facts))
        _insert(self.db, 'request_facts', [(identity, *row[1:]) for identity, row in zip(new_facts, pending_facts)])
        if pending_facts:
            self.next_fact_id = new_facts.stop
        def fact_id(r):
            value = record_facts[id(r)]
            return new_facts[-value - 1] if value < 0 else value
        attempt_rows = []
        for r in requests:
            if not isinstance(r, AttemptResultRecord):
                continue
            key = turn_keys[r.turn_id]
            value = (key, r.attempt, r.status, error(r), r.error_origin, fact_id(r), _json_dump(metrics(r)),
                     r.scenario_passed, r.trigger_ns, r.decision_ns, r.applied_ns,
                     definitions['attempt_plans'][_json_dump(r.plan)])
            previous = existing_attempts.get((key, r.attempt))
            if previous is not None:
                if previous[1:] != value:
                    raise sqlite3.IntegrityError('conflicting terminal result')
            else:
                attempt_rows.append((None, *value))
        inserted = _insert(self.db, 'attempts', attempt_rows, columns=self.columns['attempts'],
                           revision=self.revision, suffix=' RETURNING *')
        existing_attempts.update(((row[1], row[2]), tuple(row[:-1])) for row in inserted)
        by_attempt_id = {row[0]: row for row in existing_attempts.values()}
        stored_metrics = {identity: orjson.loads(row[7]) for identity, row in by_attempt_id.items()}
        layouts = {identity: orjson.loads(payload) for payload, identity in metric_defs.items()}
        missing_layouts = tuple({value[0] for value in stored_metrics.values()} - layouts.keys())
        for batch in _chunks(missing_layouts):
            layouts.update((identity, orjson.loads(payload)) for identity, payload in self.db.execute(
                f"SELECT id,payload FROM metric_definitions WHERE id IN ({','.join('?' for _ in batch)})", batch))
        patches = {}
        turn_rows = []
        for r in requests:
            if not isinstance(r, TurnResultRecord):
                continue
            key = turn_keys[r.id]
            request = r.measurements.get('request', {})
            number = request.get('attempts')
            final = existing_attempts.get((key, number)) if number else None
            # A previous terminal row fixes its metric base, including when an
            # Attempt arrives in a later transaction. No retroactive mutations.
            previous = existing_turns.get(key)
            base_id = previous[10] if previous is not None else final[0] if final else None
            base = stored_metrics.get(base_id)
            definition, numbers = metrics(r)
            pair = definition, base[0] if base else None
            if pair not in patches:
                patches[pair] = metric_patch(layouts[definition], layouts[base[0]] if base else [])
            changes, removed = metric_delta(numbers, base[1] if base else [], patches[pair])
            encoded_metrics = [definition, changes, removed]
            measurement = {k: v for k, v in r.measurements.items() if k != 'request'}
            # Keep absence of request distinct from an explicitly empty request.
            extras, measurement_values = pack_measurements('turns', measurement)
            measurement_payload = _json_dump([int('request' in r.measurements), extras])
            lifecycle = measurement.get('lifecycle') or {}
            value = (key, r.ordinal, r.status, r.started_at, r.finished_at, error(r),
                     r.estimated_input_tokens, r.reference_output_tokens, r.requested_output_tokens,
                     fact_id(r), base_id, measurement_payload, _json_dump(encoded_metrics),
                     _json_dump(r.metadata), r.scenario_passed,
                     measurement.get('total_started_ns') if measurement.get('total_started_ns') is not None else lifecycle.get('started_ns'),
                     lifecycle.get('completed_ns'), *measurement_values)
            if previous is not None:
                if previous != value:
                    raise sqlite3.IntegrityError('conflicting terminal result')
            else:
                turn_rows.append(value)
        _insert(self.db, 'turns', turn_rows, columns=self.columns['turns'], revision=self.revision)


def _lookup(db, table, identities, columns="*"):
    output = {}
    for batch in _chunks(tuple(identities)):
        for row in db.execute(f"SELECT {columns} FROM {table} WHERE id IN ({','.join('?' for _ in batch)})", batch):
            output[row['id']] = dict(row)
    return output


def decode_entities(db, table, rows, *, detail=True):
    """Decode a bounded page with one batched lookup per relation."""
    rows = [dict(r) for r in rows]
    fact_columns = '*' if detail else ','.join(('id', 'present', *REQUEST_COLUMNS, 'trace_json', 'usage_json'))
    facts = _lookup(db, 'request_facts', {r['fact_id'] for r in rows if r.get('fact_id') is not None}, fact_columns)
    base = _lookup(db, 'attempts', {r['final_attempt_id'] for r in rows if r.get('final_attempt_id') is not None})
    references = {t: set() for t in DEFINITIONS}
    if detail:
        references['request_profiles'].update(f['profile_id'] for f in facts.values() if f['profile_id'] is not None)
    for r in rows:
        if detail and r['error_id'] is not None:
            references['error_definitions'].add(r['error_id'])
        if detail and table == 'attempts':
            references['attempt_plans'].add(r['plan_id'])
        values = orjson.loads(r['metrics_json'])
        r['_metrics'] = values
        references['metric_definitions'].add(values[0])
        if table == 'turns' and r['final_attempt_id'] is not None:
            b = base[r['final_attempt_id']]
            b['_metrics'] = orjson.loads(b['metrics_json'])
            references['metric_definitions'].add(b['_metrics'][0])
    definitions = {t: {key: orjson.loads(v['payload']) for key, v in _lookup(db, t, keys).items()}
                   for t, keys in references.items()}
    output = []
    for r in rows:
        value = {k: v for k, v in r.items() if k not in {
            'id', 'error_id', 'fact_id', 'final_attempt_id', 'plan_id', 'turn_key',
            'metrics_json', 'measurements_json', 'metadata_json',
            'start_ns', 'end_ns', '_metrics', 'measure_mask', 'revision',
            *MEASUREMENT_COLUMNS.get(table, ()),
        }}
        value['error'] = definitions['error_definitions'][r['error_id']] if detail and r['error_id'] is not None else None
        if 'metadata_json' in r:
            value['metadata'] = orjson.loads(r['metadata_json'])
        if table == 'sessions':
            value['measurements'] = unpack_measurements(table, r, orjson.loads(r['measurements_json']))
        else:
            f = facts[r['fact_id']]
            request = orjson.loads(f['extension_json']) if detail else {}
            for i, name in enumerate(REQUEST_COLUMNS):
                if f['present'] & (1 << i):
                    request[name] = f[name]
            if detail and f['present'] & (1 << len(REQUEST_COLUMNS)):
                request['profile'] = definitions['request_profiles'][f['profile_id']] if f['profile_id'] is not None else None
            if f['present'] & (1 << (len(REQUEST_COLUMNS) + 1)):
                request['trace_context'] = orjson.loads(f['trace_json']) if f['trace_json'] is not None else None
            value['usage'] = orjson.loads(f['usage_json'])
            if table == 'turns':
                present, value['measurements'] = orjson.loads(r['measurements_json']) if detail else (1, {})
                unpack_measurements(table, r, value['measurements'])
                if present:
                    value['measurements']['request'] = request
            else:
                value['measurements'] = request
                if detail:
                    value['plan'] = definitions['attempt_plans'][r['plan_id']]
        stored = r['_metrics']
        layout = definitions['metric_definitions'][stored[0]]
        if table == 'turns':
            b = base.get(r['final_attempt_id'])
            base_metrics = b['_metrics'] if b else None
            value['metrics'] = restore_metrics(layout, stored[1], removed=stored[2],
                base_layout=definitions['metric_definitions'][base_metrics[0]] if b else [],
                base_values=base_metrics[1] if b else [])
        else:
            value['metrics'] = restore_metrics(layout, stored[1])
        if not detail:
            value['metrics'] = {name: metric for name, metric in value['metrics'].items()
                                if name in {'ttft', 'tpot', 'e2e'}}
        output.append(value)
    return output


def entity_select(table, alias='r'):
    """Select public identities without materializing request facts in SQL."""
    if table == 'sessions':
        return f'{alias}.*,t.run_id', f'FROM sessions {alias} JOIN trials t ON t.trial_id={alias}.trial_id'
    key = f'{alias}.id' if table == 'turns' else f'{alias}.turn_key'
    return (f'{alias}.*,k.turn_id,s.session_id,s.trial_id,t.run_id',
            f'FROM {table} {alias} JOIN turn_keys k ON k.id={key} '
            'JOIN sessions s ON s.id=k.session_key JOIN trials t ON t.trial_id=s.trial_id')
