from __future__ import annotations

from collections.abc import Mapping
import logging
from tempfile import TemporaryDirectory

import duckdb
from perf_benchmark.core.models import Metadata, MetricWindowMeasurement
from perf_benchmark.pipeline.contracts.facts import TrialFact
from perf_benchmark.pipeline.contracts.metrics import MetricUpdate, TrialMetricReport
from perf_benchmark.pipeline.contracts.results import MetricPoint, MetricWindowRecord
from perf_benchmark.pipeline.results.measurement import AttemptFact, SessionFact, OccupancyFact
from perf_benchmark.plugins.metrics.standard.catalog import METRIC_CATALOG
from perf_benchmark.plugins.metrics.standard.billing import Billing
from perf_benchmark.plugins.metrics.standard.maxtpm import dataset_maxtpm
from perf_benchmark.plugins.metrics.standard.distributions import (
    REPETITION_STATISTICS,
    repetition_statistics,
)
from perf_benchmark.plugins.metrics.standard.formulas import (
    scalar,
    request_values,
    request_metrics,
    itl_values,
)
from perf_benchmark.plugins.metrics.standard.windows import measurement_windows
from perf_benchmark.plugins.metrics.standard.groups import grouped_metrics
from perf_benchmark.plugins.metrics.standard.timeline import Timeline
from perf_benchmark.plugins.metrics.standard.validity import assess

from perf_benchmark.plugins.metrics.standard.storage import (
    TURN_SCHEMA, ITL_SCHEMA, ATTEMPT_SCHEMA, _SESSION_SCHEMA, _OCCUPANCY_SCHEMA,
    turn_row, _duckdb_connection, _arrow_table,
)
from perf_benchmark.plugins.metrics.standard.window_calculations import WindowCalculations
from perf_benchmark.plugins.metrics.standard.distribution_calculations import (
    DistributionCalculations, FORMULA_VERSION,
)

REPETITION_FORMULA_VERSION = "repetition-v2"

logger = logging.getLogger("perf_benchmark.metrics.standard")

class StandardMetricsBackend(WindowCalculations, DistributionCalculations):
    def __init__(self, settings: Mapping, *, seed: int) -> None:
        self.settings = settings
        self.seed = seed
        self.timeline = Timeline()
        self.billing = Billing(settings.get("pricing"))
        steady = settings.get("steady", {})
        if not isinstance(steady, Mapping):
            raise ValueError("metrics steady settings must be an object")
        self.min_ratio = float(steady.get("min_ratio", 0.9))
        self.min_duration_s = float(steady.get("min_duration_s", 1.0))
        if not 0 < self.min_ratio <= 1:
            raise ValueError("steady.min_ratio must be in (0, 1]")
        if self.min_duration_s <= 0:
            raise ValueError("steady.min_duration_s must be positive")
        goodput = settings.get("goodput")
        self.goodput_rules = (
            tuple(goodput["rules"])
            if isinstance(goodput, Mapping)
            else ()
        )
        self.db: duckdb.DuckDBPyConnection | None = None
        self.temporary: TemporaryDirectory[str] | None = None
        self._pending = {
            "turns": [],
            "itl": [],
            "attempts": [],
            "sessions": [],
            "occupancy": [],
        }

    def open(self) -> None:
        self.temporary = TemporaryDirectory(prefix="perf-benchmark-metrics-")
        self.db = _duckdb_connection(self.settings, self.temporary.name)
        self.db.execute("""
            CREATE TEMP TABLE turns(
              trial_id VARCHAR, turn_id VARCHAR, session_id VARCHAR,
              ordinal BIGINT, status VARCHAR, offered_ns BIGINT,
              target_ns BIGINT, admitted_ns BIGINT,
              request_ns BIGINT, response_mode VARCHAR,
              response_completed_ns BIGINT, turn_completed_ns BIGINT, first_ns BIGINT, last_ns BIGINT,
              output_event_count BIGINT, prompt BIGINT, completion BIGINT,
              cached BIGINT, attempts BIGINT, input_estimate BIGINT,
              ttft DOUBLE, tpot DOUBLE, e2e DOUBLE,
              time_to_last_output DOUBLE, response_drain DOUBLE,
              prefill DOUBLE, decode DOUBLE,
              decode_tokens BIGINT, decode_ns BIGINT, turn_total_duration DOUBLE, output_kind VARCHAR,
              system_tokens BIGINT, ttft_reason VARCHAR, tpot_reason VARCHAR, e2e_reason VARCHAR,
              event_yield DOUBLE, acceptance_proxy DOUBLE, event_yield_reason VARCHAR
            );
            CREATE TEMP TABLE attempts(
              trial_id VARCHAR, turn_id VARCHAR, attempt BIGINT, status VARCHAR,
              origin VARCHAR, target_ns BIGINT, offered_ns BIGINT,
              admitted_ns BIGINT, started_ns BIGINT
            );
            CREATE TEMP TABLE itl(
              trial_id VARCHAR, turn_id VARCHAR, value DOUBLE
            );
            CREATE TEMP TABLE sessions(
              trial_id VARCHAR, session_id VARCHAR, ordinal BIGINT, status VARCHAR, target_ns BIGINT, offered_ns BIGINT,
              admitted_ns BIGINT, started_ns BIGINT,
              finished_ns BIGINT, duration DOUBLE, turns BIGINT,
              completed_turns BIGINT
            );
            CREATE TEMP TABLE occupancy(
              trial_id VARCHAR, kind VARCHAR, at_ns BIGINT, value BIGINT
            );
            CREATE TEMP TABLE trial_meta(
              trial_id VARCHAR PRIMARY KEY, comparison_id VARCHAR,
              repetition BIGINT, status VARCHAR, expected BIGINT, validity VARCHAR
            );
            CREATE TEMP TABLE trial_points(
              trial_id VARCHAR, window_name VARCHAR, metric VARCHAR,
              statistic VARCHAR, value DOUBLE, unit VARCHAR,
              availability VARCHAR, unavailable_reason VARCHAR,
              source VARCHAR
            );
        """)

        # Initialize Arrow conversion and DuckDB ingestion on the owner thread,
        # before request admission. Lazy dependency imports can otherwise stall
        # the first occupancy flush while terminal facts fill the bounded queue.
        # An empty batch exercises the real path without introducing facts.
        self._append("occupancy", [], _OCCUPANCY_SCHEMA)

    def consume(self, facts):
        turns, attempts, sessions, occupancy, intervals, output = [], [], [], [], [], []
        for fact in facts:
            if isinstance(fact, OccupancyFact):
                occupancy.append((fact.trial_id, fact.kind, fact.at_ns, fact.value))
            elif isinstance(fact, SessionFact):
                duration = (fact.completed_ns - fact.started_ns) / 1e6
                sessions.append(
                    (
                        fact.trial_id,
                        fact.id,
                        fact.ordinal,
                        fact.status,
                        fact.target_ns,
                        fact.offered_ns,
                        fact.admitted_ns,
                        fact.started_ns,
                        fact.completed_ns,
                        duration,
                        fact.turn_count,
                        fact.succeeded_turns,
                    )
                )
                output.append(
                    MetricUpdate(
                        "session",
                        fact.id,
                        {
                            "session_duration": scalar(
                                METRIC_CATALOG["session_duration"], duration, source="scheduler"
                            ),
                            "session_turns": scalar(
                                METRIC_CATALOG["session_turns"],
                                fact.turn_count,
                                source="scheduler",
                            ),
                        },
                    )
                )
            else:
                values = request_values(fact)
                metrics = request_metrics(fact, values)
                metrics.update(self.billing.record(fact))
                if isinstance(fact, AttemptFact):
                    metrics.pop("turn_total_duration")
                    attempts.append(
                        (
                            fact.trial_id,
                            fact.id,
                            fact.attempts,
                            fact.status,
                            fact.error_origin,
                            fact.target_ns,
                            fact.offered_ns,
                            fact.admitted_ns,
                            fact.request_ns,
                        )
                    )
                    output.append(MetricUpdate("attempt", fact.id, metrics, fact.attempts))
                else:
                    turns.append(turn_row(fact, values, metrics))
                    intervals.extend(
                        (fact.trial_id, fact.id, value) for value in itl_values(fact)
                    )
                    output.append(MetricUpdate("turn", fact.id, metrics))
        for table, rows in (
            ("turns", turns),
            ("itl", intervals),
            ("attempts", attempts),
            ("sessions", sessions),
            ("occupancy", occupancy),
        ):
            self._pending[table].extend(rows)
        self._flush_pending()
        self.timeline.observe(facts)
        return (*output, *self.timeline.emit(self))

    def _flush_pending(self, *, force=False):
        assert self.db is not None
        schemas = {
            "turns": (TURN_SCHEMA, 1000),
            "itl": (ITL_SCHEMA, 10000),
            "attempts": (ATTEMPT_SCHEMA, 1000),
            "sessions": (_SESSION_SCHEMA, 1000),
            "occupancy": (_OCCUPANCY_SCHEMA, 4000),
        }
        ready = [
            (table, rows, schema)
            for table, rows in self._pending.items()
            for schema, threshold in (schemas[table],)
            if rows and (force or len(rows) >= threshold)
        ]
        if not ready:
            return
        self.db.begin()
        try:
            for table, rows, schema in ready:
                self._append(table, rows, schema)
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise
        for _, rows, _ in ready:
            rows.clear()

    def _append(self, table, rows, schema):
        assert self.db is not None
        self.db.register("batch_facts", _arrow_table(rows, schema))
        try:
            self.db.execute(f"INSERT INTO {table} SELECT * FROM batch_facts")
        finally:
            self.db.unregister("batch_facts")

    def finalize_trial(self, fact: TrialFact) -> TrialMetricReport:
        assert self.db is not None
        self._flush_pending(force=True)
        trial_id = fact.trial_id
        target = fact.load_profile.target_concurrency
        raw_bounds = self._one(
            "SELECT min(start_ns), max(end_ns) FROM ("
            "SELECT coalesce(offered_ns,request_ns,response_completed_ns) start_ns, "
            "coalesce(response_completed_ns,request_ns,offered_ns) end_ns "
            "FROM turns WHERE trial_id=? UNION ALL "
            "SELECT coalesce(target_ns,offered_ns,admitted_ns,started_ns), "
            "coalesce(finished_ns,started_ns,admitted_ns,offered_ns) "
            "FROM sessions WHERE trial_id=? UNION ALL "
            "SELECT coalesce(target_ns,offered_ns,admitted_ns,started_ns), "
            "coalesce(started_ns,admitted_ns,offered_ns,target_ns) "
            "FROM attempts WHERE trial_id=?)",
            [trial_id, trial_id, trial_id],
        )
        bounds = (
            int(raw_bounds[0]) if raw_bounds[0] is not None else None,
            int(raw_bounds[1]) + 1 if raw_bounds[1] is not None else None,
        )
        occupancy = [
            (int(row[0]), int(row[1]))
            for row in self.db.execute(
                "SELECT at_ns,value FROM occupancy "
                "WHERE trial_id=? AND kind='in_flight' ORDER BY at_ns,rowid",
                [trial_id],
            ).fetchall()
        ]
        windows, window_diagnostics = measurement_windows(
            bounds,
            occupancy,
            target,
            min_ratio=self.min_ratio,
            min_duration_s=self.min_duration_s,
        )
        points: list[MetricPoint] = []
        window_rows: list[MetricWindowRecord] = []
        self._window_diagnostics: dict[str, dict[str, int]] = {}
        for window in windows:
            window_rows.append(
                {
                    "window": window.name,
                    "start_ns": window.start_ns,
                    "end_ns": window.end_ns,
                    "duration_s": window.duration_s,
                    "availability": "available" if window.available else "unavailable",
                    "unavailable_reason": window.unavailable_reason,
                }
            )
            points.extend(self._window_points(trial_id, window, target, fact))
        extra_diagnostics = {}
        if self.settings.get("maxtpm"):
            extra_points, extra_windows, extra_diagnostics = dataset_maxtpm(self, fact, bounds, occupancy)
            points.extend(extra_points)
            window_rows.extend(extra_windows)
        points.extend(self.billing.finalize_trial(self, trial_id))
        validity = assess(self, fact, points, windows)
        self.db.execute(
            "INSERT OR REPLACE INTO trial_meta VALUES (?, ?, ?, ?, ?, ?)",
            [
                trial_id,
                fact.comparison.group_id if fact.comparison is not None else None,
                fact.repetition.index,
                validity.execution_status,
                fact.repetition.expected,
                validity.status,
            ],
        )
        self.db.execute("DELETE FROM trial_points WHERE trial_id=?", [trial_id])
        self.db.executemany(
            "INSERT INTO trial_points VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    point.scope_id,
                    point.window,
                    point.metric,
                    point.statistic,
                    point.value,
                    point.unit,
                    point.availability,
                    point.unavailable_reason,
                    point.source,
                )
                for point in points
            ],
        )
        offered, admitted, succeeded, failed, cancelled = self._one(
            "SELECT count(*), count(*) FILTER (request_ns IS NOT NULL), "
            "count(*) FILTER (status='succeeded'), "
            "count(*) FILTER (status='failed'), "
            "count(*) FILTER (status='cancelled') "
            "FROM turns WHERE trial_id=?",
            [trial_id],
        )
        diagnostics: Metadata = {
            **window_diagnostics,
            **extra_diagnostics,
            **({"pricing": dict(self.billing.pricing)} if self.billing.pricing else {}),
            "itl_contract": "output_event_interval",
            "window_cohorts": {name: {**values} for name, values in self._window_diagnostics.items()},
        }
        counts = {
            "offered": int(offered), "admitted": int(admitted),
            "succeeded": int(succeeded), "failed": int(failed), "cancelled": int(cancelled),
        }
        overall = {
            (p.metric, p.statistic): p.value
            for p in points
            if p.window == "overall" and p.availability == "available"
        }
        logger.info(
            "trial_id=%s workload_rps=%s ttft_p95_s=%s tpot_p95_ms=%s",
            trial_id,
            overall.get(("workload_rps", "value")),
            overall.get(("ttft", "p95")),
            overall.get(("tpot", "p95")),
        )
        series = self.timeline.emit(self, force=True, final_trial=fact.trial_id)
        return TrialMetricReport(
            (*points, *(point for bucket in series for point in bucket.points)),
            (*tuple(MetricWindowMeasurement(**row) for row in window_rows), *(bucket.window for bucket in series)),
            counts,
            diagnostics,
            grouped_metrics(self.db, trial_id, windows, FORMULA_VERSION),
            validity,
        )

    def release_trial(self, trial_id: str) -> None:
        self.timeline.release(trial_id)
        self.billing.release(trial_id)
        assert self.db is not None
        for rows in self._pending.values():
            rows[:] = (row for row in rows if row[0] != trial_id)
        for table in (
            "itl",
            "attempts",
            "turns",
            "sessions",
            "occupancy",
        ):
            self.db.execute(f"DELETE FROM {table} WHERE trial_id=?", [trial_id])

    def finalize_run(
        self,
        run_id: str,
    ) -> tuple[MetricPoint, ...]:
        assert self.db is not None
        groups = self.db.execute("""
            SELECT comparison_id, max(expected), count(*),
                   count(*) FILTER (WHERE status='succeeded' AND validity='pass')
            FROM trial_meta WHERE comparison_id IS NOT NULL
            GROUP BY comparison_id ORDER BY min(trial_id)
        """).fetchall()
        output: list[MetricPoint] = []
        for point_id, expected, trial_count, valid_count in groups:
            if expected <= 1:
                continue
            identities = self.db.execute(
                """
                SELECT p.window_name, p.metric, p.statistic, p.unit,
                       list(p.value ORDER BY p.rowid) FILTER (
                           WHERE m.status='succeeded' AND m.validity='pass' AND p.availability='available'
                             AND p.value IS NOT NULL
                       )
                FROM trial_points p JOIN trial_meta m USING(trial_id)
                WHERE m.comparison_id=?
                GROUP BY p.window_name, p.metric, p.statistic, p.unit
                ORDER BY p.window_name, p.metric, p.statistic
            """,
                [point_id],
            ).fetchall()
            for window, metric, statistic, unit, samples in identities:
                values = samples or []
                available = len(values) == expected
                summaries = (
                    repetition_statistics(
                        values,
                        f"{point_id}:{window}:{metric}:{statistic}",
                        seed=self.seed,
                    )
                    if available
                    else {name: None for name in REPETITION_STATISTICS}
                )
                for name in REPETITION_STATISTICS:
                    output.append(
                        MetricPoint(
                            "comparison",
                            str(point_id),
                            str(window),
                            str(metric),
                            f"{statistic}_{name}",
                            summaries[name],
                            "ratio" if name == "cv" else str(unit),
                            REPETITION_FORMULA_VERSION,
                            "available" if available and summaries[name] is not None else "unavailable",
                            None if available and summaries[name] is not None else "insufficient_repetitions" if expected < 2 else "insufficient_metric_samples",
                            "repetition",
                            len(values),
                            expected,
                            expected - len(values),
                        )
                    )
            qualified = trial_count == expected and valid_count == expected
            for metric, value, unit in (
                ("repetition_valid", valid_count, "repetitions"),
                ("repetition_expected", expected, "repetitions"),
                ("repetition_failed", expected - valid_count, "repetitions"),
                ("repetition_qualified", int(qualified), "boolean"),
            ):
                output.append(
                    MetricPoint(
                        "comparison",
                        str(point_id),
                        "overall",
                        metric,
                        "value",
                        float(value),
                        unit,
                        REPETITION_FORMULA_VERSION,
                        "available",
                        None,
                        "repetition",
                        valid_count,
                        expected,
                        expected - valid_count,
                    )
                )
        return tuple(output)

    def close(self) -> None:
        if self.db is not None:
            self.db.close()
        if self.temporary is not None:
            self.temporary.cleanup()
            self.temporary = None
