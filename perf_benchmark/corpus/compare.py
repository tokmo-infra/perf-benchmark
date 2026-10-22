"""Read-only load/performance comparison. No server calls or cache manipulation."""

import math
from statistics import mean, stdev

import duckdb

from perf_benchmark.plugins.saver.sqlite import SQLiteResultReader

from .common import fingerprint, local_path, read_yaml


def _tv(db, left, right):
    return db.sql(f"""WITH a AS ({left}), b AS ({right}),
        p AS (SELECT key,n/sum(n) OVER () AS p FROM a),
        q AS (SELECT key,n/sum(n) OVER () AS q FROM b)
        SELECT CASE WHEN (SELECT count(*) FROM a)=0 AND (SELECT count(*) FROM b)=0 THEN 0
          WHEN (SELECT count(*) FROM a)=0 OR (SELECT count(*) FROM b)=0 THEN 1
          ELSE coalesce(sum(abs(coalesce(p,0)-coalesce(q,0)))/2,0) END
        FROM p FULL JOIN q USING(key)""").fetchone()[0]


def compare_profiles(real, generated):
    manifests = [read_yaml(path / "profile.yaml") for path in (real, generated)]
    if any(value.get("version") != 1 for value in manifests):
        raise ValueError("unsupported profile version")
    if any(value.get("algorithm") != "joint-session-unigram-bigram-v1" for value in manifests):
        raise ValueError("profile comparison requires joint-session-unigram-bigram-v1 Chat profiles")
    comparable = manifests[0]["tokenizer_fingerprint"] == manifests[1]["tokenizer_fingerprint"]
    report = {"tokenizer_comparable": comparable, "input_tokens": {
        "real": manifests[0]["input_tokens"], "generated": manifests[1]["input_tokens"],
    }}
    with duckdb.connect(config={"threads": 1}) as db:
        db.execute("SET memory_limit='512MB'")
        for side, path in (("a", real), ("b", generated)):
            for name in ("structures", "lengths", "tokens", "prefixes"):
                db.read_parquet(str(path / f"{name}.parquet")).create_view(f"{side}_{name}")
        distances = {}
        for name, expression in (
            ("input_length", "input"), ("reference_output_length", "reference_output"),
            ("requested_output_length", "requested_output"), ("prompt_category", "category"),
            ("input_requested_joint", "to_json([input,requested_output])"),
        ):
            distances[name] = _tv(db, *[
                f"SELECT coalesce(cast({expression} AS VARCHAR),'missing') AS key,count(*) AS n FROM {side}_lengths GROUP BY 1"
                for side in ("a", "b")
            ])
        for name, expression in (
            ("session_turn_count", "json_array_length(structure)"),
            ("session_joint_lengths", "json_extract(structure,'$[*].input_tokens')::VARCHAR || ':' || json_extract(structure,'$[*].output_tokens')::VARCHAR"),
            ("role_sequence", "json_extract(structure,'$[*].request.messages[*].role')"),
            ("tool_definitions", "json_extract(structure,'$[*].request.tools')"),
        ):
            distances[name] = _tv(db, *[
                f"SELECT cast({expression} AS VARCHAR) AS key,count(*) AS n FROM {side}_structures GROUP BY 1"
                for side in ("a", "b")
            ])
        for name, condition in (("unigram", "second=-1"), ("bigram", "second>=0")):
            distances[name] = _tv(db, *[
                f'''SELECT to_json(["group",first::VARCHAR,second::VARCHAR]) AS key,sum(count) AS n
                    FROM {side}_tokens WHERE {condition} GROUP BY 1'''
                for side in ("a", "b")
            ]) if comparable else None
        report["total_variation"] = distances
        report["shared_prefixes"] = {
            label: [dict(zip(("kind", "depth", "reuse_fraction"), row, strict=True)) for row in db.sql(
                f"SELECT kind,depth,sum(count-1)::DOUBLE/sum(count) FROM {side}_prefixes GROUP BY kind,depth ORDER BY kind,depth"
            ).fetchall()]
            for side, label in (("a", "real"), ("b", "generated"))
        }
    report["interpretation"] = "TV is 0 for equal empirical distributions and 1 for disjoint ones; no universal acceptance threshold. Prefix reuse is not a cache hit rate."
    return report


def validate_service_lengths(records):
    checked, failures, unavailable = 0, 0, 0
    for record in records:
        usage = record.get("usage", {})
        request = record.get("measurements", {}).get("request", {})
        metadata = record.get("metadata", {})
        if metadata.get("dataset_output_policy") != "fixed":
            unavailable += 1
            continue
        expected_input = record.get("estimated_input_tokens")
        expected_output = record.get("requested_output_tokens")
        if any(value is None for value in (usage.get("prompt_tokens"), usage.get("completion_tokens"), request.get("finish_reason"), expected_input, expected_output)):
            unavailable += 1
            continue
        counts = (usage["prompt_tokens"], usage["completion_tokens"], expected_input, expected_output)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
            unavailable += 1
            continue
        checked += 1
        if (record.get("status") != "succeeded" or usage["prompt_tokens"] != expected_input
                or usage["completion_tokens"] != expected_output or request["finish_reason"] != "length"):
            failures += 1
    return {"status": "passed" if checked and not failures and not unavailable else "not_passed",
            "checked": checked, "mismatched": failures, "unavailable": unavailable}


def _observations(reader, run_id, trial_id):
    metrics = reader.metric_points(run_id, scope="trial", scope_id=trial_id, window="overall") if trial_id is not None else {}
    result = {}
    for name, statistic in (("ttft", "am"), ("tpot", "am"), ("e2e", "am"), ("workload_rps", "value"), ("success_rate", "value")):
        metric = metrics.get(name, {})
        availability = metric.get("statistic_metadata", {}).get(statistic, {})
        result[name] = metric.get("statistics", {}).get(statistic) if availability.get("availability") == "available" else None
    success_rate = result.pop("success_rate")
    result["failure_rate"] = 1 - success_rate if success_rate is not None else None
    return result


def compare_performance(runs, base, thresholds=None):
    results, reasons, conditions = [], [], []
    observed_execution, observed_scheduler, observed_tokenizers = set(), set(), set()
    if len(runs) < 6:
        reasons.append("at least three paired repetitions are required")
    paths = set()
    for item in runs:
        if item.get("side") not in {"real", "generated"}:
            raise ValueError("comparison side must be real or generated")
        path = local_path(base, item.get("sqlite"))
        if path in paths:
            reasons.append("a SQLite artifact was reused as an independent repetition")
        paths.add(path)
        reader = SQLiteResultReader(path)
        ids = reader.run_ids()
        run_id = item.get("run_id")
        if run_id is None:
            if len(ids) != 1:
                raise ValueError("run_id required for an artifact with multiple runs")
            run_id = ids[0]
        if reader.summary(run_id)["status"] != "succeeded":
            reasons.append("a benchmark run did not succeed")
        condition = item.get("conditions", {})
        required = {"deployment", "scheduler", "max_concurrency", "output_policy", "cache_state"}
        if not required <= condition.keys() or item.get("conditions_verified") is not True:
            reasons.append("deployment/scheduler/max_concurrency/output/cache conditions are unverified")
        conditions.append(fingerprint(condition))
        records = reader.iter_turn_records(run_id)
        signatures, tokenizers = set(), set()
        output_matches = True
        def inspect_records():
            nonlocal output_matches
            for row in records:
                metadata = row.get('metadata', {})
                # Equality/unknown decisions need only two distinct witnesses,
                # not a retained fingerprint for every request.
                for seen, name in ((signatures, 'execution_fingerprint'),
                                   (tokenizers, 'dataset_tokenizer_fingerprint')):
                    value = metadata.get(name)
                    if value is None or len(seen) < 2:
                        seen.add(value)
                output_matches &= metadata.get('dataset_output_policy') == condition.get('output_policy')
                yield row
        if condition.get('output_policy') == 'fixed':
            lengths = validate_service_lengths(inspect_records())
            if lengths['status'] != 'passed':
                reasons.append('fixed service input/output lengths are not verified')
        else:
            for _ in inspect_records():
                pass
            lengths = {'status': 'natural_termination_not_fixed'}
            if condition.get('output_policy') != 'natural':
                reasons.append('output policy is unknown')
        if not signatures or None in signatures:
            reasons.append('actual execution settings were not recorded')
        observed_execution.update(signatures)
        observed_tokenizers.update(tokenizers)
        if None in tokenizers:
            reasons.append('workload tokenizer/template identity is unknown')
        if not output_matches:
            reasons.append('recorded output policy does not match the declared conditions')
        trial_parameters = reader.trials(run_id)
        if len(trial_parameters) != 1:
            reasons.append("compare requires one trial per run, not a mixed max_concurrency sweep")
        for trial in trial_parameters:
            metadata = trial.get("metadata", {})
            signature = metadata.get("scheduler_fingerprint")
            if signature is None or metadata.get("scheduler_plugin") != condition.get("scheduler") or trial["parameters"].get("max_concurrency") != condition.get("max_concurrency"):
                reasons.append("actual scheduler/max_concurrency does not match the declared conditions")
            observed_scheduler.add(signature)
        trial_id = trial_parameters[0]["trial_id"] if len(trial_parameters) == 1 else None
        results.append({"side": item["side"], "metrics": _observations(reader, run_id, trial_id), "length_validation": lengths})
    if len(set(conditions)) > 1:
        reasons.append("benchmark conditions differ")
    if len(observed_execution) > 1 or len(observed_scheduler) > 1 or len(observed_tokenizers) > 1:
        reasons.append("recorded execution or scheduler settings differ")
    for index in range(0, len(runs), 2):
        expected = ["real", "generated"] if index % 4 == 0 else ["generated", "real"]
        if [item["side"] for item in runs[index:index + 2]] != expected:
            reasons.append("paired execution order must alternate real/generated then generated/real")
    comparison = {}
    for name in ("ttft", "tpot", "e2e", "workload_rps", "failure_rate"):
        groups = {side: [row["metrics"][name] for row in results if row["side"] == side] for side in ("real", "generated")}
        if any(len(values) < 3 or any(v is None or not math.isfinite(v) for v in values) for values in groups.values()):
            comparison[name] = {"availability": "unavailable", "reason": "missing metrics or repetitions"}
            continue
        baseline, generated = mean(groups["real"]), mean(groups["generated"])
        comparison[name] = {
            "availability": "available", "real_mean": baseline, "generated_mean": generated,
            "real_std": stdev(groups["real"]), "generated_std": stdev(groups["generated"]),
            "absolute_delta": generated - baseline,
            "relative_delta": (generated - baseline) / abs(baseline) if baseline else None,
        }
    outcome = "not_comparable" if reasons else "reported_without_pass_threshold"
    if thresholds and not reasons:
        passed = True
        for name, limit in thresholds.items():
            if name not in comparison or isinstance(limit, bool) or not isinstance(limit, (float, int)) or not math.isfinite(limit) or limit < 0:
                raise ValueError("thresholds must map metric names to non-negative relative deviations")
            delta = comparison[name].get("relative_delta")
            passed &= delta is not None and abs(delta) <= limit
        outcome = "passed_user_thresholds" if passed else "not_passed_user_thresholds"
    return {"status": outcome, "reasons": sorted(set(reasons)), "metrics": comparison, "runs": results,
            "conditions_evidence": "Explicit operator attestations; the offline tool cannot inspect deployment or cache state."}


def compare(config, base):
    result = {}
    if "profiles" in config:
        value = config["profiles"]
        result["workload"] = compare_profiles(local_path(base, value["real"]), local_path(base, value["generated"]))
    if "runs" in config:
        result["performance"] = compare_performance(config["runs"], base, config.get("max_relative_deviation"))
    if not result:
        raise ValueError("compare requires profiles and/or runs")
    return result
