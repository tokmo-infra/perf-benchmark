from tests.result_records import frozen_record

from perf_benchmark.pipeline.results.metrics import MetricsRuntime
from perf_benchmark.pipeline.results.saver import SaverRuntime
import asyncio
import sqlite3

import pytest

from perf_benchmark.core.models import (
    MetricWindowMeasurement,
    Run,
    Session,
    Trial,
    TrialMeasurements,
    Turn,
)
from perf_benchmark.pipeline.contracts.results import (
    ComputedResultBatch,
    SessionResultRecord,
    TurnResultRecord,
    MetricPoint,
)
from perf_benchmark.pipeline.contracts.facts import (
    ComparisonPoint,
    LoadProfile,
    RepetitionPlan,
    ResourceLimits,
    TrialFact,
    TrialSchedulingFact,
)
from perf_benchmark.pipeline.preparation.payloads import PayloadRef
from perf_benchmark.plugins.saver.sqlite import SQLiteResultReader, SQLiteSaver, parse_settings
from perf_benchmark.plugins.saver.sqlite.schema import (
    ARTIFACT_SCHEMA_VERSION,
    MEASUREMENT_CONTRACT,
    ResultDatabase,
)
from perf_benchmark.plugins.metrics.standard import StandardMetrics


async def test_completed_turn_is_visible_before_trial_finishes(tmp_path, stage_context):
    run = Run(id="live-run", name="continuous persistence")
    run.prepare()
    run.start()
    ctx = stage_context("metrics", run=run)
    trial = Trial(id="trial", run_id=run.id)
    trial.start()
    session = Session(id="session", trial_id=trial.id, turns=[])
    session.start()
    turn = Turn(id="completed", session_id=session.id, request_payload_ref=PayloadRef(0, 1))
    turn.start()
    turn.measurements.request.response_mode = "non_stream"
    turn.measurements.request.started_ns = 1_000_000_000
    turn.measurements.request.response_completed_ns = 1_250_000_000
    turn.succeed()
    session.turns.append(turn)
    database = tmp_path / "live.db"
    sink = await SaverRuntime.create(
        SQLiteSaver({"path": str(database)}), run, stage_context("saver", run=run)
    )
    metrics = None
    try:
        await sink.write_trial(trial)
        await sink.write_session(session)
        metrics = await MetricsRuntime.create(StandardMetrics({}), ctx, publish_results=sink.write_results, wait_for_results=sink.wait_for_headroom)
        collector = await metrics.open_trial(trial)
        collector.record_turn(turn)
        session.finish()
        collector.record_session(session)

        async def visible():
            while True:
                reader = await asyncio.to_thread(SQLiteResultReader, database)
                if (await asyncio.to_thread(reader.summary, run.id))["turn_count"] == 1:
                    return
                await asyncio.sleep(0.01)

        await asyncio.wait_for(visible(), 5)
        assert trial.status == "running"

        reader = await asyncio.to_thread(SQLiteResultReader, database)
        values = (await asyncio.to_thread(lambda: tuple(reader.iter_turn_records(run.id))))[0]['metrics']
        assert values["e2e"]["statistics"]["value"] == 0.25
        trial.finish()
        trial.status = "succeeded"
        await collector.finalize(trial, _trial_fact(trial))
        await collector.close()
        await sink.flush()
        assert (await asyncio.to_thread(reader_summary, database, run.id))["turn_count"] == 1
    finally:
        if metrics is not None:
            await metrics.close()
        await sink.close()


@pytest.mark.parametrize("settings", [{}, {"path": ""}, {"path": "x", "unknown": 1}])
def test_sqlite_saver_settings_are_strict(settings):
    with pytest.raises(ValueError):
        parse_settings(settings)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
async def test_non_finite_metric_fails_without_losing_committed_results(
    tmp_path, stage_context, value
):
    run = Run(name="invalid-metric")
    database = tmp_path / "invalid.db"
    sink = await SaverRuntime.create(
        SQLiteSaver({"path": str(database)}), run, stage_context("saver", run=run)
    )
    trial = Trial(id="trial", run_id=run.id)
    await sink.write_trial(trial)
    await sink.flush()
    try:
        with pytest.raises(ValueError, match="non-finite"):
            await sink.write_metric_points(
                (
                    MetricPoint(
                        "trial",
                        trial.id,
                        "overall",
                        "ttft",
                        "value",
                        value,
                        "ms",
                        "test",
                        "available",
                        None,
                        "test",
                    ),
                )
            )
    finally:
        with pytest.raises(ValueError, match="non-finite"):
            await sink.close()
    summary = await asyncio.to_thread(reader_summary, database, run.id)
    assert summary["trial_count"] == 1
    assert summary["metric_count"] == 0


def reader_summary(database, run_id):
    return SQLiteResultReader(database).summary(run_id)


async def test_finalization_can_record_a_later_export_failure(tmp_path, stage_context):
    database = tmp_path / "export-failure.db"
    run = Run(name="late failure")
    run.prepare()
    run.start()
    sink = await SaverRuntime.create(
        SQLiteSaver({"path": str(database)}), run, stage_context("saver", run=run)
    )
    try:
        run.finish()
        await sink.finalize(run)
        run.fail("exporter failed")
        await sink.finalize(run)
    finally:
        await sink.close()
    summary = await asyncio.to_thread(reader_summary, database, run.id)
    assert summary["status"] == "failed"
    assert summary["error"] == "exporter failed"


async def test_terminal_sessions_do_not_scan_all_persisted_turns(
    tmp_path, stage_context, monkeypatch
):
    # A VM-work budget is deterministic; a wall-clock threshold depends on the host.
    checking, calls = False, 0
    connect = ResultDatabase.connect

    def bounded_connection(database):
        connection = connect(database)

        def progress():
            nonlocal calls
            if checking:
                calls += 1
            return calls > 10

        connection.set_progress_handler(progress, 10_000)
        return connection

    monkeypatch.setattr(ResultDatabase, "connect", bounded_connection)
    database = tmp_path / "many-turns.db"
    run = Run(name="session updates")
    sink = await SaverRuntime.create(
        SQLiteSaver({"path": str(database)}), run, stage_context("saver", run=run)
    )
    trial = Trial(id="trial", run_id=run.id)
    sessions = [Session(id=f"s{i}", trial_id=trial.id, turns=[]) for i in range(100)]
    try:
        await sink.write_trial(trial)
        for session in sessions:
            session.start()
            await sink.write_session(session)
        await sink.flush()

        def seed_turns():
            with sqlite3.connect(database) as db:
                parents = dict(db.execute('SELECT session_id,id FROM sessions'))
                db.executemany('INSERT INTO turn_keys(id,turn_id,session_key) VALUES(?,?,?)',
                               ((i + 1, f't{i}', parents[f's{i // 100}']) for i in range(10_000)))
                db.execute("INSERT INTO request_facts(id,present,extension_json,usage_json) VALUES(1,0,'{}','{}')")
                db.executemany(
                    "INSERT INTO turns(id,ordinal,status,fact_id,measurements_json,metrics_json,metadata_json,measure_mask) "
                    "VALUES(?,?,'succeeded',1,'[0,{}]','[[],[]]','{}',0)",
                    ((i + 1, i % 100) for i in range(10_000)),
                )

        await asyncio.to_thread(seed_turns)
        checking = True
        for session in sessions:
            session.finish()
        sink.write_results(
            ComputedResultBatch(
                sessions=tuple(frozen_record(SessionResultRecord,
                    id=session.id, trial_id=session.trial_id, ordinal=session.ordinal, status=session.status,
                    started_at=session.started_at, finished_at=session.finished_at, error=session.error,
                    metadata=session.metadata, measurements=session.measurements, metrics=session.metrics, turn_count=len(session.turns),
                ) for session in sessions)
            )
        )
        await sink.flush()
    finally:
        checking = False
        await sink.close()
    summary = await asyncio.to_thread(reader_summary, database, run.id)
    assert summary["session_count"] == 100
    assert summary["turn_count"] == 10_000


async def test_sqlite_saves_windowed_trial_and_generic_comparison_metrics(
    tmp_path,
    stage_context,
) -> None:
    database = tmp_path / "results.db"
    run = Run(id="run", name="saver")
    run.prepare()
    run.start()
    trial = Trial(
        id="trial",
        name="batch-128-repetition-1",
        run_id=run.id,
        parameters={"plugin_private": "preserved-but-not-interpreted"},
        metadata={"label": "small"},
        measurements=TrialMeasurements(
            metric_windows=(
                MetricWindowMeasurement(
                    window="overall",
                    start_ns=1,
                    end_ns=2,
                    duration_s=1e-9,
                    availability="available",
                    unavailable_reason=None,
                ),
            )
        ),
    )
    session = Session(id="session", trial_id=trial.id, turns=[])
    turn = Turn(
        id="turn",
        session_id=session.id,
        request_payload_ref=PayloadRef(0, 18),
        status="succeeded",
        usage={"prompt_tokens": 1},
    )
    ctx = stage_context("saver", {"path": str(database)}, run)
    sink = await SaverRuntime.create(SQLiteSaver({"path": str(database)}), run, ctx)
    trial.status = "succeeded"
    await sink.write_trial(
        trial,
        _trial_fact(
            trial,
            comparison=ComparisonPoint("batch-128", "batch_size", 128, "requests"),
            repetition=RepetitionPlan(1, 2),
        ),
    )
    await sink.write_session(session)
    sink.write_results(
        ComputedResultBatch(
            turns=(
                frozen_record(TurnResultRecord,
                    id=turn.id,
                    session_id=turn.session_id,
                    ordinal=turn.ordinal,
                    status=turn.status,
                    started_at=turn.started_at,
                    finished_at=turn.finished_at,
                    error=turn.error,
                    estimated_input_tokens=turn.estimated_input_tokens,
                    reference_output_tokens=turn.reference_output_tokens,
                    requested_output_tokens=turn.requested_output_tokens,
                    usage=turn.usage,
                    measurements=turn.measurements,
                    metrics=turn.metrics,
                    metadata=turn.metadata,
                ),
            )
        )
    )
    await sink.write_metric_points(
        (
            MetricPoint(
                "trial",
                trial.id,
                "overall",
                "workload_rps",
                "value",
                3.0,
                "req/s",
                "test",
                "available",
                None,
                "client_clock",
            ),
            MetricPoint(
                "comparison",
                "batch-128",
                "overall",
                "workload_rps",
                "value_median",
                3.0,
                "req/s",
                "test",
                "available",
                None,
                "repetition",
                2,
                2,
                0,
            ),
        )
    )
    reader = await sink.finalize(run)
    await sink.close()

    assert reader.summary(run.id)["turn_count"] == 1
    assert reader.metric_windows(run.id, scope="trial", scope_id=trial.id)[0]["window"] == "overall"
    metrics = reader.metric_points(
        run.id,
        scope="trial",
        scope_id=trial.id,
        window="overall",
    )
    assert metrics["workload_rps"]["statistics"]["value"] == 3.0
    stored_trial = reader.trials(run.id)[0]
    assert stored_trial["parameters"] == {"plugin_private": "preserved-but-not-interpreted"}
    assert stored_trial["load_profile"] is not None and stored_trial["scheduling"] is not None
    assert stored_trial["load_profile"]["target_concurrency"] == 2
    assert stored_trial["scheduling"]["demand_limited"] is False
    assert reader.comparison_points(run.id) == (
        {
            "group_id": "batch-128",
            "axis_name": "batch_size",
            "axis_value": 128,
            "axis_unit": "requests",
        },
    )
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT schema_version,measurement_contract FROM artifact_meta"
        ).fetchall() == [
            (
                ARTIFACT_SCHEMA_VERSION,
                MEASUREMENT_CONTRACT,
            )
        ]
        assert connection.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='schema_info'"
        ).fetchone() == (0,)
        columns = {row[1] for row in connection.execute("PRAGMA table_info(turns)")}
        assert not {"request_json", "response_json", "reference_output"}.intersection(columns)
    assert b'"content"' not in database.read_bytes()


def test_sqlite_reader_rejects_old_artifact_immediately(tmp_path) -> None:
    database = tmp_path / "old.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE runs(run_id TEXT PRIMARY KEY, name TEXT NOT NULL)")

    with pytest.raises(RuntimeError, match="incompatible result database"):
        SQLiteResultReader(database)


@pytest.mark.parametrize("version", [5, 6, 7, 16, 17])
def test_sqlite_reader_rejects_old_schema_without_migration(tmp_path, version) -> None:
    database = tmp_path / f"schema-{version}.db"
    ResultDatabase(database)
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE artifact_meta SET schema_version=?", (version,))

    with pytest.raises(RuntimeError, match="incompatible result database contract"):
        SQLiteResultReader(database)


def _trial_fact(
    trial: Trial,
    *,
    comparison: ComparisonPoint | None = None,
    repetition: RepetitionPlan | None = None,
) -> TrialFact:
    return TrialFact(
        trial.id,
        trial.status,
        LoadProfile(
            "test", target_concurrency=2, max_active_sessions=2, max_concurrency=2
        ),
        ResourceLimits(2, 2, 2),
        comparison,
        repetition or RepetitionPlan(),
        TrialSchedulingFact(),
    )


@pytest.mark.parametrize(
    'value, error, message',
    [
        ({'nested': [{'Authorization': 'secret'}]}, ValueError, 'credential'),
        ({'nested': [float('nan')]}, ValueError, 'non-finite'),
        ({'nested': ['x' * 4097]}, ValueError, '4096'),
        ({'nested': {1: 'value'}}, TypeError, 'keys must be strings'),
        ({'nested': b'body'}, TypeError, 'unsupported'),
    ],
)
async def test_invalid_nested_fact_preserves_prior_commit(
    tmp_path, stage_context, value, error, message
):
    run = Run(name='nested-validation')
    database = tmp_path / 'nested.db'
    sink = await SaverRuntime.create(
        SQLiteSaver({'path': str(database)}), run, stage_context('saver', run=run)
    )
    trial = Trial(id='trial', run_id=run.id)
    await sink.write_trial(trial)
    await sink.flush()
    from perf_benchmark.pipeline.contracts.results import MeasurementRecord

    sink.writer.submit(MeasurementRecord(run.id, 'trial', trial.id, 'diagnostic', value))
    try:
        with pytest.raises(error, match=message):
            await sink.flush()
    finally:
        with pytest.raises(error, match=message):
            await sink.close()
    assert (await asyncio.to_thread(reader_summary, database, run.id))['trial_count'] == 1


@pytest.mark.parametrize('value', [
    {'pools': [{'pool': f'dp:{index}', 'source': '/get_server_info', 'capacity_tokens': 1000000}
               for index in range(128)]},
    'x' * 4096,
], ids=['many-pools', 'boundary-string'])
async def test_measurement_json_size_is_distinct_from_fact_string_limit(tmp_path, stage_context, value):
    import json
    from perf_benchmark.pipeline.contracts.results import MeasurementRecord

    run = Run(name='large-structured-evidence')
    database = tmp_path / 'evidence.db'
    sink = await SaverRuntime.create(SQLiteSaver({'path': str(database)}), run,
                                    stage_context('saver', run=run))
    try:
        sink.writer.submit(MeasurementRecord(run.id, 'run', run.id, 'capacity', value))
        await sink.flush()
        with sqlite3.connect(database) as db:
            encoded = db.execute("SELECT value_json FROM measurements WHERE name='capacity'").fetchone()[0]
        assert len(encoded) > 4096
        assert json.loads(encoded) == value
        reader = SQLiteResultReader(database)
        assert reader.measurements(run.id, scope='run', scope_id=run.id)['capacity'] == value
    finally:
        await sink.close()


async def test_canonical_json_from_immutable_nested_containers(tmp_path, stage_context):
    from dataclasses import dataclass
    from types import MappingProxyType
    from typing import NamedTuple
    from perf_benchmark.pipeline.contracts.results import MeasurementRecord

    class Pair(NamedTuple):
        left: int
        right: int

    @dataclass(frozen=True)
    class Detail:
        z: object
        a: float

    value = MappingProxyType({'z': Detail(Pair(2, 3), 1.25), 'a': (None, True, '中文')})
    run = Run(name='canonical-json')
    path = tmp_path / 'canonical.db'
    sink = await SaverRuntime.create(SQLiteSaver({'path': str(path)}), run,
                                     stage_context('saver', run=run))
    try:
        sink.writer.submit(MeasurementRecord(run.id, 'run', run.id, 'test', value))
        await sink.flush()
        with sqlite3.connect(path) as db:
            encoded = db.execute("SELECT value_json FROM measurements WHERE name='test'").fetchone()[0]
        assert encoded == '{"a":[null,true,"中文"],"z":{"a":1.25,"z":[2,3]}}'
        assert isinstance(value['z'].z, Pair)
        assert value['a'] == (None, True, '中文')
    finally:
        await sink.close()
