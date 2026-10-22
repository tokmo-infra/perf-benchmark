"""Bounded final-Turn cohorts, computed on the Metrics owner thread."""

from perf_benchmark.pipeline.contracts.results import GroupedMetricRecord
from perf_benchmark.plugins.metrics.standard.catalog import METRIC_CATALOG, public_value


def length_bucket(column):
    # Exact integer boundaries, including an explicit overflow bucket.
    cases = [f"WHEN {column} IS NULL THEN 'unknown'", f"WHEN {column}=0 THEN '0'"]
    cases.extend(
        f"WHEN {column}<{2 ** (power + 1)} THEN '[{2**power},{2 ** (power + 1)})'"
        for power in range(30)
    )
    return "CASE " + " ".join(cases) + " ELSE '[1073741824,infinity)' END"


def grouped_metrics(db, trial_id, windows, formula_version):
    output = []
    for window in windows:
        if not window.available:
            # The persisted window carries its unavailable reason; it has no cohort.
            continue
        rows = db.execute(
            f"""
            WITH buckets AS (
                SELECT *, {length_bucket("prompt")} AS input_bin,
                       {length_bucket("completion")} AS output_bin,
                       {length_bucket("system_tokens")} AS system_bin,
                       CASE WHEN cached IS NULL THEN 'unknown'
                            WHEN cached=0 THEN 'reported_miss' ELSE 'reported_hit' END AS cache_bin
                FROM turns WHERE trial_id=? AND request_ns>=? AND request_ns<?
            ), samples AS (
                SELECT axis, bin, second_axis, second_bin, metric,
                       status='succeeded' AS eligible,
                       CASE WHEN status='succeeded' THEN value END AS value,
                       CASE WHEN status='succeeded' THEN reason END AS reason
                FROM buckets, LATERAL (VALUES
                    ('input_tokens',input_bin,'',''), ('output_tokens',output_bin,'',''),
                    ('system_tokens_estimate',system_bin,'',''), ('cache',cache_bin,'',''),
                    ('input_tokens',input_bin,'output_tokens',output_bin),
                    ('input_tokens',input_bin,'cache',cache_bin)
                ) AS dimensions(axis,bin,second_axis,second_bin), LATERAL (VALUES
                    ('ttft',ttft,ttft_reason), ('tpot',tpot,tpot_reason), ('e2e',e2e,e2e_reason)
                ) AS measurements(metric,value,reason)
            )
            SELECT axis,bin,second_axis,second_bin,metric,count(*),
                   count(*) FILTER (eligible),count(value),avg(value),
                   quantile_disc(value,[0.5,0.95,0.99]),
                   histogram(coalesce(reason, 'not_observed')) FILTER (eligible AND value IS NULL)
            FROM samples GROUP BY axis,bin,second_axis,second_bin,metric
            ORDER BY axis,bin,second_axis,second_bin,metric
        """,
            [trial_id, window.start_ns, window.end_ns],
        ).fetchall()
        for (
            axis,
            bucket,
            second,
            other,
            metric,
            total,
            eligible,
            valid,
            mean,
            quantiles,
            reasons,
        ) in rows:
            definition = METRIC_CATALOG[metric]
            values = [mean, *(quantiles or [None] * 3)]
            output.append(
                GroupedMetricRecord(
                    trial_id,
                    window.name,
                    ((axis, bucket), (second, other)) if second else ((axis, bucket),),
                    metric,
                    definition.unit,
                    formula_version,
                    total,
                    eligible,
                    valid,
                    {
                        stat: public_value(definition, stat, value) if value is not None else None
                        for stat, value in zip(("am", "p50", "p95", "p99"), values)
                    },
                    reasons or {},
                )
            )
    return tuple(output)
