from __future__ import annotations

from contextlib import closing

import sqlite3
from pathlib import Path

from perf_benchmark.plugins.saver.sqlite.layout import NORMALIZED_SCHEMA

ARTIFACT_SCHEMA_VERSION = 19
MEASUREMENT_CONTRACT = "standard-window-v8"
RESULT_TABLES = (
    "sweeps", "runs", "trials", "trial_facts", "sessions", "turns", "attempts",
    "measurements", "metric_windows", "metric_points", "grouped_metrics",
)

SCHEMA = """
CREATE TABLE artifact_meta(
  schema_version INTEGER NOT NULL,
  measurement_contract TEXT NOT NULL
);
INSERT INTO artifact_meta VALUES (19, 'standard-window-v8');
CREATE TABLE result_identity(database_id TEXT NOT NULL, observation_contract TEXT NOT NULL);
INSERT INTO result_identity VALUES (lower(hex(randomblob(16))), 'observation-v2');
CREATE TABLE result_state(id INTEGER PRIMARY KEY CHECK(id=1), revision INTEGER NOT NULL);
INSERT INTO result_state VALUES (1,0);
CREATE TABLE runs(
  run_id TEXT PRIMARY KEY, name TEXT NOT NULL, status TEXT NOT NULL,
  created_at TEXT NOT NULL, preparing_at TEXT, started_at TEXT,
  finished_at TEXT, error TEXT,
  counts_json TEXT NOT NULL
);
CREATE TABLE sweeps(
  run_id TEXT NOT NULL REFERENCES runs(run_id), sequence INTEGER NOT NULL,
  payload_json TEXT NOT NULL, PRIMARY KEY(run_id, sequence)
);
CREATE TABLE trials(
  trial_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
  name TEXT NOT NULL, ordinal INTEGER NOT NULL, status TEXT NOT NULL,
  parameters_json TEXT NOT NULL, metadata_json TEXT NOT NULL, started_at TEXT,
  finished_at TEXT, error TEXT, session_count INTEGER NOT NULL,
  turn_count INTEGER NOT NULL, succeeded_turns INTEGER NOT NULL,
  failed_or_cancelled_turns INTEGER NOT NULL
);
CREATE TABLE trial_facts(
  trial_id TEXT PRIMARY KEY REFERENCES trials(trial_id),
  load_profile_json TEXT NOT NULL, resource_limits_json TEXT NOT NULL,
  comparison_group_id TEXT, comparison_axis_name TEXT,
  comparison_axis_value_json TEXT, comparison_axis_unit TEXT,
  repetition_index INTEGER NOT NULL, expected_repetitions INTEGER NOT NULL,
  scheduling_json TEXT NOT NULL, validity_requirements_json TEXT, validity_json TEXT, execution_status TEXT NOT NULL
);
CREATE TABLE measurements(
  run_id TEXT NOT NULL REFERENCES runs(run_id), scope TEXT NOT NULL,
  scope_id TEXT NOT NULL, name TEXT NOT NULL, value_json TEXT NOT NULL,
  PRIMARY KEY(run_id, scope, scope_id, name)
);
CREATE TABLE metric_windows(
  run_id TEXT NOT NULL REFERENCES runs(run_id), scope TEXT NOT NULL,
  scope_id TEXT NOT NULL, window TEXT NOT NULL, start_ns INTEGER, end_ns INTEGER,
  duration_s REAL, availability TEXT NOT NULL, unavailable_reason TEXT,
  PRIMARY KEY(run_id, scope, scope_id, window)
);
CREATE TABLE metric_points(
  run_id TEXT NOT NULL REFERENCES runs(run_id), scope TEXT NOT NULL,
  scope_id TEXT NOT NULL, window TEXT NOT NULL, metric TEXT NOT NULL,
  statistic TEXT NOT NULL, value REAL, availability TEXT NOT NULL,
  unavailable_reason TEXT, valid_sample_count INTEGER,
  expected_sample_count INTEGER, missing_sample_count INTEGER,
  definition_id INTEGER NOT NULL REFERENCES metric_definitions(id),
  PRIMARY KEY(run_id, scope, scope_id, window, metric, statistic)
);
CREATE TABLE grouped_metrics(
  run_id TEXT NOT NULL REFERENCES runs(run_id),
  trial_id TEXT NOT NULL REFERENCES trials(trial_id), window TEXT NOT NULL,
  dimensions_json TEXT NOT NULL, metric TEXT NOT NULL, sample_count INTEGER NOT NULL,
  eligible_sample_count INTEGER NOT NULL, valid_sample_count INTEGER NOT NULL,
  statistics_json TEXT NOT NULL, unavailable_reasons_json TEXT NOT NULL,
  definition_id INTEGER NOT NULL REFERENCES metric_definitions(id),
  PRIMARY KEY(run_id, trial_id, window, dimensions_json, metric)
);
CREATE INDEX trials_run_order ON trials(run_id,ordinal);
CREATE INDEX trial_facts_comparison
  ON trial_facts(comparison_group_id, comparison_axis_name);
"""
SCHEMA += NORMALIZED_SCHEMA
SCHEMA += "".join(f"ALTER TABLE {table} ADD COLUMN revision INTEGER NOT NULL DEFAULT 0;"
                  f"CREATE INDEX {table}_revision ON {table}(revision);" for table in RESULT_TABLES)


class ResultDatabase:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as connection:
            # Set before WAL/schema creation; never rebuild an existing result file.
            connection.execute("PRAGMA page_size=8192")
            _configure(connection)
            _initialize(connection)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        try:
            _configure(connection)
        except BaseException:
            connection.close()
            raise
        return connection


def _configure(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=30000")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    # Retain the previous ~4 MiB automatic-checkpoint budget with 8 KiB pages.
    connection.execute("PRAGMA wal_autocheckpoint=500")


def _initialize(connection: sqlite3.Connection) -> None:
    actual = _signature(connection)
    if not actual:
        connection.executescript(SCHEMA)
        _immutability_constraints(connection)
        connection.commit()
        return
    _validate(connection, actual)


def _immutability_constraints(connection):
    for table in RESULT_TABLES:
        columns = connection.execute(f"PRAGMA table_info({table})").fetchall()
        changed = " OR ".join(f"OLD.{c[1]} IS NOT NEW.{c[1]}" for c in columns if c[1] != 'revision')
        if table in {'turns', 'attempts', 'sessions'}:
            terminal = "OLD.status IN ('succeeded','failed','cancelled') AND " if table == 'sessions' else ''
            connection.execute(f"CREATE TRIGGER {table}_immutable BEFORE UPDATE ON {table} "
                f"WHEN {terminal}({changed}) BEGIN SELECT RAISE(ABORT,'conflicting terminal result'); END")


def validate_result_database(path: str | Path) -> None:
    database = Path(path)
    if not database.is_file():
        raise FileNotFoundError(f"result database not found: {database}")
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        _validate(connection, _signature(connection))


def _validate(
    connection: sqlite3.Connection,
    actual: tuple[tuple[object, ...], ...],
) -> None:
    with closing(sqlite3.connect(":memory:")) as canonical:
        canonical.executescript(SCHEMA)
        _immutability_constraints(canonical)
        expected = _signature(canonical)
    if actual != expected:
        raise RuntimeError("incompatible result database structure")
    marker = connection.execute(
        "SELECT schema_version,measurement_contract FROM artifact_meta"
    ).fetchall()
    if marker != [(ARTIFACT_SCHEMA_VERSION, MEASUREMENT_CONTRACT)]:
        raise RuntimeError("incompatible result database contract")
    identity = connection.execute("SELECT database_id,observation_contract FROM result_identity").fetchall()
    if len(identity) != 1 or len(identity[0][0]) != 32 or identity[0][1] != "observation-v2":
        raise RuntimeError("incompatible observation database contract")


def _signature(connection: sqlite3.Connection) -> tuple[tuple[object, ...], ...]:
    return tuple(
        tuple(row)
        for row in connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
        )
    )


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "MEASUREMENT_CONTRACT",
    "ResultDatabase",
    "validate_result_database",
]
