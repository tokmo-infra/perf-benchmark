"""Contextual maxTPM defaults use selected samples, independently of user count."""

from dataclasses import replace

import pytest

from perf_benchmark.pipeline.contracts.facts import LoadProfile, TrialSchedulingFact
from perf_benchmark.plugins.metrics.standard import StandardMetrics, parse_settings
from perf_benchmark.plugins.metrics.standard.maxtpm import resolve_start_percent
from tests.metrics.test_maxtpm_billing import NANO, occupancy, session, trial, turn


@pytest.mark.parametrize("samples,concurrency", [(None, 2), (3, None)])
def test_missing_sampling_facts_use_regular_default(samples, concurrency):
    assert resolve_start_percent(parse_settings({}), samples, concurrency) == (40, "default")


@pytest.mark.parametrize(
    "samples,concurrency,users,configured,expected,source",
    [
        (3, 2, 99, None, 53, "samples_1_5x_concurrency"),
        (4, 2, 3, None, 40, "default"),
        (383, 256, 384, None, 40, "default"),
        (3, 2, 3, 40, 40, "configured"),
        (3, 2, 3, 0, 0, "configured"),
    ],
)
def test_effective_window_and_diagnostics_follow_selected_sample_count(
    samples,
    concurrency,
    users,
    configured,
    expected,
    source,
):
    settings = {"maxtpm": True}
    if configured is not None:
        settings["maxtpm_start_percent"] = configured
    backend = StandardMetrics(parse_settings(settings)).create_backend(seed=0)
    backend.open()
    try:
        backend.consume(
            (
                session(0, 301),
                *occupancy((1, concurrency)),
                *(
                    turn(str(i), stamp, prompt=90, completion=10, cached=25)
                    for i, stamp in enumerate((10, 20, 30, 40, 60, 100, 140, 180, 240, 300))
                ),
            )
        )
        fact = replace(
            trial(10),
            load_profile=LoadProfile(
                "test", target_concurrency=concurrency, max_active_sessions=users
            ),
            scheduling=TrialSchedulingFact(dataset_session_count=samples, dataset_turn_count=10),
        )
        report = backend.finalize_trial(fact)
        assert report.diagnostics["maxtpm_start_percent"] == expected
        assert report.diagnostics["maxtpm_start_percent_source"] == source
        assert report.diagnostics["maxtpm_start_turns"] == (10 * expected + 99) // 100
        assert report.diagnostics["maxtpm_start_ns"] == ({53: 100, 40: 40, 0: 1}[expected]) * NANO
    finally:
        backend.close()
