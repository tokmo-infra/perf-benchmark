"""Frozen pre-refactor outputs, generated before replacing the DuckDB scalar formulas."""

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import pytest

from perf_benchmark.pipeline.contracts.facts import (
    LoadProfile,
    RepetitionPlan,
    ResourceLimits,
    TrialFact,
    TrialSchedulingFact,
)
from perf_benchmark.pipeline.results.measurement import OccupancyFact, TurnFact
from perf_benchmark.plugins.metrics.standard import StandardMetrics


def assert_same(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_same(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert_same(a, e)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)
    else:
        assert actual == expected


def test_standard_coalesces_occupancy_rows_before_duckdb_append(monkeypatch):
    backend = StandardMetrics({}).create_backend(seed=0)
    backend.open()
    appended = []
    original = backend._append

    def observe(table, rows, schema):
        appended.append((table, len(rows)))
        original(table, rows, schema)

    monkeypatch.setattr(backend, "_append", observe)
    try:
        for batch in range(40):
            backend.consume(
                tuple(
                    OccupancyFact("trial", "in_flight", batch * 100 + index, index)
                    for index in range(100)
                )
            )
        assert appended == [("occupancy", 4000)]
        assert backend.db is not None
        assert backend.db.execute("SELECT count(*) FROM occupancy").fetchone() == (4000,)
    finally:
        backend.close()


def test_standard_matches_pre_refactor_scalar_and_window_results():
    fixture = json.loads(
        Path(__file__).with_name("fixtures").joinpath("standard-baseline.json").read_text()
    )
    # Convert the frozen millisecond baseline independently to the v6 public units.
    for metrics in fixture["metrics"].values():
        for name, metric in metrics.items():
            metric["unit"] = {"ms/token": "ms", "tokens/s": "tok/s"}.get(metric["unit"], metric["unit"])
            if name in {"ttft", "e2e"}:
                metric["unit"] = "s"
                metric["statistics"] = {k: v / 1000 if v is not None else None for k, v in metric["statistics"].items()}
    facts = []
    for row in fixture["facts"]:
        values: dict[str, Any] = {
            **row, "target_ns": row["offered_ns"], "admitted_ns": row["request_ns"],
            "turn_started_ns": row["offered_ns"],
            "response_samples": tuple(map(tuple, row["response_samples"])),
        }
        facts.append(TurnFact(**values))
    backend = StandardMetrics(
        {"steady": {"min_ratio": 0.9, "min_duration_s": 0.01}}
    ).create_backend(seed=0)
    backend.open()
    try:
        updates = backend.consume(
            (
                *facts,
                OccupancyFact("trial", "in_flight", 1000000000, 1),
                OccupancyFact("trial", "in_flight", 1350000009, 0),
            )
        )
        actual = {u.id: {name: {field: getattr(value, field) for field in
                  ("unit", "availability", "unavailable_reason", "source", "statistics")}
                  for name, value in u.metrics.items() if name in fixture["metrics"][u.id]} for u in updates}
        # The frozen baseline checks existing statistics; time_wm has focused tests.
        for metrics in actual.values():
            for metric in metrics.values():
                metric["statistics"] = {key: value for key, value in metric["statistics"].items()
                                        if key != "time_wm"}
        assert_same(actual, fixture["metrics"])
        report = backend.finalize_trial(
            TrialFact(
                "trial",
                "failed",
                LoadProfile("test", target_concurrency=1, max_concurrency=1),
                ResourceLimits(1, 1, 1),
                None,
                RepetitionPlan(),
                TrialSchedulingFact(),
            )
        )
        points = sorted(report.points, key=lambda p: (p.window, p.metric, p.statistic))
        expected = [
            ["standard-window-v8" if value == "standard-window-v3" else value for value in row]
            for row in fixture["points"]
            if row[3] != "session_arrival_rate"
        ]
        for row in expected:
            row[6] = {"ms/token": "ms", "tokens/s": "tok/s"}.get(row[6], row[6])
            if row[4] in {"n", "eligible_n"}:
                row[6] = "samples"
            elif row[3] in {"ttft", "e2e", "session_first_turn_ttft", "session_subsequent_turn_ttft"}:
                row[6] = "s"
                row[5] = float(row[5]) / 1000 if row[5] is not None else None
            if row[3] == "target_active_sessions":
                row[9] = "unbounded_by_scheduler"
            if row[3] == "total_cached_tokens":
                row[7] = "observed-cache-total-v1"
                if row[9] == "partial_usage":
                    window = next(w for w in report.windows if w.window == row[2])
                    row[5] = sum(f.usage["cached_tokens"] for f in facts
                                 if f.status == "succeeded" and f.response_completed_ns is not None
                                 and window.start_ns <= f.response_completed_ns < window.end_ns
                                 and "cached_tokens" in f.usage)
                    row[8], row[10] = "partial", "server_usage"
        # v8 adds explicit scalar sample eligibility; values/units remain the frozen baseline.
        for row in expected:
            window = next(w for w in report.windows if w.window == row[2])
            if window.start_ns is None or window.end_ns is None:
                continue
            if row[3] in {"success_rate", "cancellation_rate"}:
                n = sum(f.request_ns is not None and window.start_ns <= f.request_ns < window.end_ns for f in facts)
            elif row[3] == "workload_rps":
                n = sum(f.status == "succeeded" and f.response_completed_ns is not None and window.start_ns <= f.response_completed_ns < window.end_ns for f in facts)
            elif row[3] == "session_success_rate":
                n = 0  # The frozen baseline contains no Session facts.
            else:
                continue
            row[11:14] = [n, n, 0]
        legacy = {(row[2], row[3], row[4]) for row in expected}
        actual = [list(asdict(p).values()) for p in points if (p.window, p.metric, p.statistic) in legacy]
        assert_same(actual, expected)
        windows = [asdict(w) for w in report.windows]
        assert_same(windows, fixture["windows"])
        assert_same(
            {**report.diagnostics, "metric_windows": windows, "metric_counts": report.counts},
            fixture["diagnostics"],
        )
    finally:
        backend.close()
