from dataclasses import fields, replace

import pytest

from perf_benchmark.core.profiles import RequestProfile
from perf_benchmark.core.models import ResponseMode
from perf_benchmark.pipeline.results.measurement import AttemptFact, TurnFact
from perf_benchmark.plugins.metrics.standard.backend import StandardMetricsBackend, FORMULA_VERSION
from perf_benchmark.plugins.metrics.standard.groups import grouped_metrics
from perf_benchmark.plugins.metrics.standard.windows import MeasurementWindow


def fact(
    identifier, *, prompt=8, completion=4, cached=None, status="succeeded", mode: ResponseMode="stream", start=100
):
    usage = {
        k: v
        for k, v in {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "cached_tokens": cached,
        }.items()
        if v is not None
    }
    return TurnFact(
        "trial",
        identifier,
        "session",
        0,
        status,
        None,
        None,
        None,
        999,
        None,
        5,
        usage,
        {},
        (),
        None,
        None,
        None,
        start,
        500,
        1,
        mode,
        start,
        200,
        400,
        500,
        200,
        2,
        8,
        "stop",
        profile=RequestProfile(system_tokens_estimate=2),
    )


def test_group_qualification_unknown_usage_boundaries_and_units():
    backend = StandardMetricsBackend({}, seed=0)
    backend.open()
    try:
        previous = AttemptFact(
            **{f.name: getattr(fact("a", status="failed"), f.name) for f in fields(TurnFact)}
        )
        backend.consume(
            (
                previous,
                replace(fact("a", cached=0), attempts=2),
                fact("b", prompt=16, cached=2),
                fact("c", prompt=None, completion=None),
                fact("d", status="failed"),
                fact("e", mode="non_stream"),
                fact("cancelled", status="cancelled"),
                fact("outside", start=1000),
            )
        )
        backend._flush_pending(force=True)
        window = MeasurementWindow("test", 100, 1000)
        groups = grouped_metrics(backend.db, "trial", [window], FORMULA_VERSION)
        inputs = [
            g
            for g in groups
            if len(g.dimensions) == 1
            and g.dimensions[0][0] == "input_tokens"
            and g.metric == "ttft"
        ]
        assert sum(g.sample_count for g in inputs) == 6
        group = next(g for g in inputs if g.dimensions[0][1] == "[8,16)")
        assert (group.sample_count, group.eligible_sample_count, group.valid_sample_count) == (
            4,
            2,
            1,
        )
        assert group.unit == "s"
        assert group.statistics["p95"] == pytest.approx(1e-7)
        assert group.unavailable_reasons == {"non_streaming_response": 1}
        assert next(g for g in inputs if g.dimensions[0][1] == "unknown").sample_count == 1
        caches = {g.dimensions[0][1] for g in groups if g.dimensions[0][0] == "cache"}
        assert caches == {"unknown", "reported_hit", "reported_miss"}
        assert all(g.trial_id == "trial" for g in groups)
    finally:
        backend.close()


def test_discrete_percentiles_and_explicit_overflow():
    backend = StandardMetricsBackend({}, seed=0)
    backend.open()
    try:
        backend.consume(
            tuple(replace(fact(str(i), prompt=2**40), first_ns=100 + i * 100) for i in range(1, 5))
        )
        backend._flush_pending(force=True)
        groups = grouped_metrics(
            backend.db, "trial", [MeasurementWindow("w", 0, 1000)], FORMULA_VERSION
        )
        group = next(
            g
            for g in groups
            if g.dimensions == (("input_tokens", "[1073741824,infinity)"),) and g.metric == "ttft"
        )
        assert group.statistics["p50"] == pytest.approx(2e-7)
        assert group.statistics["p95"] == pytest.approx(4e-7)
    finally:
        backend.close()
