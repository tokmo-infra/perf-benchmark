"""Opt-in aggregate call profile and SQLite phase timing, not performance scores.

CPython monitoring can include other threads; never attribute this call profile
to the thread that enabled it. SQL wall times include scheduling/GIL waits.
"""
import cProfile
import pstats
import sqlite3
from contextlib import closing
from collections import defaultdict
from time import perf_counter, thread_time


def database_layout(path):
    """Post-run allocated/payload/free bytes per SQLite table or index."""
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as connection:
        try:
            rows = connection.execute(
                'SELECT name,count(*),sum(pgsize),sum(payload),sum(unused) FROM dbstat GROUP BY name'
            ).fetchall()
        except sqlite3.OperationalError as error:
            return {'unavailable': str(error)}
        return {name: dict(pages=pages, allocated_bytes=size, payload_bytes=payload, unused_bytes=unused)
                for name, pages, size, payload, unused in rows}


def source_identity():
    """Fingerprint source and loaded native binary before the measured window."""
    from hashlib import sha256
    from pathlib import Path
    import subprocess
    from perf_benchmark import _native
    root = Path(__file__).resolve().parents[1]
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root,
    ).decode().split('\0')
    files = {}
    for name in sorted(set(names)):
        path = root / name
        if (name.startswith(('perf_benchmark/', 'benchmarks/')) or name in ('pyproject.toml', 'uv.lock')) and path.is_file() and path.suffix in ('.py', '.pyi', '.rs', '.sql', '.toml', '.lock', '.go'):
            files[name] = sha256(path.read_bytes()).hexdigest()
    return dict(git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                source_sha256=sha256(repr(files).encode()).hexdigest(), source_files=files,
                native_sha256=sha256(Path(_native.__file__).read_bytes()).hexdigest())


class TimedConnection:
    def __init__(self, connection):
        self.connection = connection
        self.seconds = defaultdict(float)
        self.calls = defaultdict(int)
        self.cpu_seconds = defaultdict(float)

    def __getattr__(self, name):
        return getattr(self.connection, name)

    def execute(self, sql, *args):
        key = sql.split()[0]
        started, cpu = perf_counter(), thread_time()
        try:
            return self.connection.execute(sql, *args)
        finally:
            self.seconds[key] += perf_counter() - started
            self.calls[key] += 1
            self.cpu_seconds[key] += thread_time() - cpu

    def executemany(self, sql, rows):
        started, cpu = perf_counter(), thread_time()
        try:
            return self.connection.executemany(sql, rows)
        finally:
            self.seconds['executemany'] += perf_counter() - started
            self.calls['executemany'] += 1
            self.cpu_seconds['executemany'] += thread_time() - cpu

    def __enter__(self):
        self.connection.__enter__()
        return self

    def __exit__(self, *args):
        started, cpu = perf_counter(), thread_time()
        try:
            return self.connection.__exit__(*args)
        finally:
            self.seconds['transaction_exit'] += perf_counter() - started
            self.calls['transaction_exit'] += 1
            self.cpu_seconds['transaction_exit'] += thread_time() - cpu


def start(runtime):
    from perf_benchmark.plugins.saver.sqlite import entities
    original = entities._json_dump
    encoding = dict(calls=0, seconds=0., cpu_seconds=0., scope="normalized entity JSON encoding")

    def encode(row):
        started, cpu = perf_counter(), thread_time()
        try:
            return original(row)
        finally:
            encoding["calls"] += 1
            encoding["seconds"] += perf_counter() - started
            encoding["cpu_seconds"] += thread_time() - cpu

    connection = TimedConnection(runtime.backend.connection)
    profile = cProfile.Profile()
    profile.enable()
    entities._json_dump = encode
    runtime.backend.connection = connection
    runtime.backend.entities.db = runtime.backend.entities.definitions.db = connection
    return profile, original, encoding


def stop(state, runtime):
    from perf_benchmark.plugins.saver.sqlite import entities
    profile, original, encoding = state
    profile.disable()
    entities._json_dump = original
    connection = runtime.backend.connection
    runtime.backend.connection = connection.connection
    runtime.backend.entities.db = runtime.backend.entities.definitions.db = connection.connection
    entries = pstats.Stats(profile).stats
    result = {'top': [dict(function=f'{key[0]}:{key[1]}:{key[2]}', calls=value[1],
                          self_seconds=value[2], cumulative_seconds=value[3])
                      for key, value in sorted(entries.items(), key=lambda entry: entry[1][2], reverse=True)[:25]]}
    result.update(encoding=encoding, sql_seconds=dict(connection.seconds), sql_cpu_seconds=dict(connection.cpu_seconds),
                  sql_calls=dict(connection.calls),
                  profile_scope="aggregate calls; no per-thread attribution")
    return result
