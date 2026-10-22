"""Schema 19 fixed request columns and normalized entity layout."""
REQUEST_COLUMNS = (
    'attempts', 'response_mode', 'target_ns', 'offered_ns', 'admitted_ns',
    'started_ns', 'first_output_ns', 'last_output_ns', 'response_completed_ns',
    'http_status', 'output_event_count', 'output_chars', 'finish_reason',
    'output_kind', 'result_ready_ns',
)
REQUEST_TEXT = {'response_mode', 'finish_reason', 'output_kind'}
DEFINITIONS = ('request_profiles', 'attempt_plans', 'metric_definitions', 'error_definitions')
ENTITY_TABLES = ('sessions', 'turns', 'attempts')
NORMALIZED_SCHEMA = '''
CREATE TABLE sessions(
 id INTEGER PRIMARY KEY, session_id TEXT NOT NULL UNIQUE,
 trial_id TEXT NOT NULL REFERENCES trials(trial_id), ordinal INTEGER NOT NULL,
 status TEXT NOT NULL, started_at TEXT, finished_at TEXT,
 error_id INTEGER REFERENCES error_definitions(id), metadata_json TEXT NOT NULL,
 measurements_json TEXT NOT NULL, metrics_json TEXT NOT NULL, turn_count INTEGER NOT NULL
);
CREATE INDEX sessions_trial_order ON sessions(trial_id,ordinal);
CREATE TABLE turn_keys(
 id INTEGER PRIMARY KEY, turn_id TEXT NOT NULL UNIQUE,
 session_key INTEGER NOT NULL REFERENCES sessions(id)
);
CREATE INDEX turn_keys_session ON turn_keys(session_key,id);
CREATE TABLE request_facts(
 id INTEGER PRIMARY KEY, present INTEGER NOT NULL,
 REQUEST_FIELDS,
 profile_id INTEGER REFERENCES request_profiles(id), trace_json TEXT,
 extension_json TEXT NOT NULL, usage_json TEXT NOT NULL
);
CREATE TABLE attempts(
 id INTEGER PRIMARY KEY, turn_key INTEGER NOT NULL REFERENCES turn_keys(id),
 attempt INTEGER NOT NULL, status TEXT NOT NULL,
 error_id INTEGER REFERENCES error_definitions(id), error_origin TEXT,
 fact_id INTEGER NOT NULL REFERENCES request_facts(id), metrics_json TEXT NOT NULL,
 scenario_passed INTEGER, trigger_ns INTEGER, decision_ns INTEGER, applied_ns INTEGER,
 plan_id INTEGER NOT NULL REFERENCES attempt_plans(id),
 UNIQUE(turn_key,attempt)
);
CREATE TABLE turns(
 id INTEGER PRIMARY KEY REFERENCES turn_keys(id), ordinal INTEGER NOT NULL,
 status TEXT NOT NULL, started_at TEXT, finished_at TEXT,
 error_id INTEGER REFERENCES error_definitions(id), estimated_input_tokens INTEGER,
 reference_output_tokens INTEGER, requested_output_tokens INTEGER,
 fact_id INTEGER REFERENCES request_facts(id), final_attempt_id INTEGER REFERENCES attempts(id),
 measurements_json TEXT NOT NULL, metrics_json TEXT NOT NULL,
 metadata_json TEXT NOT NULL, scenario_passed INTEGER,
 start_ns INTEGER, end_ns INTEGER
);
'''.replace('REQUEST_FIELDS', ',\n '.join(
    name + (' TEXT' if name in REQUEST_TEXT else ' INTEGER') for name in REQUEST_COLUMNS
)) + ''.join(f'''
CREATE TABLE {table}(id INTEGER PRIMARY KEY, digest BLOB NOT NULL, payload TEXT NOT NULL);
CREATE INDEX {table}_digest ON {table}(digest);
''' for table in DEFINITIONS)

MEASUREMENT_PATHS = {
    'sessions': ('lifecycle.started_ns', 'lifecycle.completed_ns', 'target_ns', 'offered_ns', 'admitted_ns'),
    'turns': ('lifecycle.started_ns', 'lifecycle.completed_ns', 'total_started_ns',
              'eligibility_started_ns', 'eligibility_finished_ns', 'eligibility_target_ns', 'dependency_ready_ns'),
}
MEASUREMENT_COLUMNS = {table: tuple('m_' + path.replace('.', '_') for path in paths)
                       for table, paths in MEASUREMENT_PATHS.items()}
for _table, _anchor in (('sessions', 'turn_count INTEGER NOT NULL'), ('turns', 'start_ns INTEGER, end_ns INTEGER')):
    NORMALIZED_SCHEMA = NORMALIZED_SCHEMA.replace(
        _anchor, _anchor + ', measure_mask INTEGER NOT NULL, ' +
        ', '.join(name + ' INTEGER' for name in MEASUREMENT_COLUMNS[_table]))
