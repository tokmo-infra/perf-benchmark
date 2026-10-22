"""Cross-Run comparison of persisted formal metrics and declared experimental controls."""
import math
from perf_benchmark.core.profiles import GENERATION_KEYS


def compare(baseline, candidate, specification):
    if set(specification) - {"allowed_changes", "metrics", "baseline_trial", "candidate_trial"}:
        raise ValueError("unknown comparison setting")
    allowed = specification.get("allowed_changes", [])
    if not isinstance(allowed, list) or any(not isinstance(k, str) for k in allowed):
        raise ValueError("allowed_changes must be condition names")
    ids = [reader.run_ids() for reader in (baseline, candidate)]
    if any(len(items) != 1 for items in ids):
        raise ValueError("comparison requires one Run per SQLite")
    run_ids = [items[0] for items in ids]
    evidence = [reader.measurements(run_id, scope="run", scope_id=run_id).get("evidence", {})
                for reader, run_id in zip((baseline, candidate), run_ids)]
    conditions = [dict(value.get("conditions", {})) for value in evidence]
    for side in conditions:
        if not side:
            side["run_conditions"] = {"status": "unknown", "value": None}
        side.setdefault("source_identity", {"status": "unknown", "value": None})
    trials = [reader.trials(run_id) for reader, run_id in zip((baseline, candidate), run_ids)]
    selected = [specification.get(name) for name in ("baseline_trial", "candidate_trial")]
    for side, group, trial_id in zip(conditions, trials, selected):
        selected_group = [t for t in group if t["trial_id"] == trial_id] if trial_id else group
        preparations = [t.get("metadata", {}).get("preparation", {}) for t in selected_group]
        if preparations and all(p.get("cache_state") == "salt_sent_backend_unverified" for p in preparations):
            side["cache_salt"] = {"status": "observed", "value": "sent"}
            side["backend_cache_isolation"] = {"status": "unknown", "value": None}
        overrides = [t["parameters"].get("request_parameters", {}) for t in selected_group]
        side["effective_generation"] = {"status": "observed" if all(not set(row)-GENERATION_KEYS for row in overrides) else "unknown",
            "value": [{k: v for k, v in row.items() if k in GENERATION_KEYS} for row in overrides]}
        side["effective_load"] = {"status": "observed", "value": [
            t.get("load_profile") for t in selected_group]}
    keys = sorted(set(conditions[0]) | set(conditions[1]))
    if set(allowed) - set(keys):
        raise ValueError("allowed_changes contains an unknown condition")
    rows, comparable = [], bool(keys)
    for key in keys:
        a, b = (side.get(key, {"status": "unknown", "value": None}) for side in conditions)
        unknown = "unknown" in (a["status"], b["status"]) or (key.endswith("unverified_fields") and (a["value"] or b["value"]))
        changed = a["value"] != b["value"]
        accepted = key in allowed
        if unknown or (changed and not accepted):
            comparable = False
        rows.append({"condition": key, "baseline": a, "candidate": b, "changed": changed,
                     "allowed_change": accepted, "verified": not bool(unknown)})
    qualified = []
    repetition_evidence = []
    for group, trial_id in zip(trials, selected):
        group = [t for t in group if t["trial_id"] == trial_id] if trial_id else list(group)
        valid = [t for t in group if (t.get("validity") or {}).get("execution_status") == "succeeded"
                 and (t.get("validity") or {}).get("status") == "pass"]
        qualified.append(bool(group) and len(valid) == len(group))
        repetition_evidence.append({"selected": len(group), "valid": len(valid),
            "invalid": len(group) - len(valid),
            "declared_expected": sorted({t["repetition"]["expected"] for t in group if t.get("repetition")})})
    metric_rows = []
    metrics = specification.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("comparison requires selected metrics")
    for rule in metrics:
        if not isinstance(rule, dict) or set(rule) - {"metric", "statistic", "window", "direction", "max_regression_ratio"}:
            raise ValueError("invalid comparison metric rule")
        name, statistic, window = rule["metric"], rule.get("statistic", "p99"), rule.get("window", "overall")
        direction, threshold = rule.get("direction", "lower"), rule.get("max_regression_ratio")
        if direction not in {"higher", "lower"} or (threshold is not None and
            (isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold < 0)):
            raise ValueError("invalid regression direction or threshold")
        records = [reader.metric_points(run_id, scope="trial" if trial_id else "run", scope_id=trial_id or run_id,
                                       window=window, names=[name]).get(name)
                   for reader, run_id, trial_id in zip((baseline, candidate), run_ids, selected)]
        values = [record.get("statistics", {}).get(statistic) if record and record.get("availability") == "available" else None for record in records]
        reason = None
        if any(v is None for v in values):
            reason = "metric_unavailable"
        elif records[0].get("formula_version") != records[1].get("formula_version"):
            reason = "formula_version_mismatch"
        elif records[0]["unit"] != records[1]["unit"]:
            reason = "unit_mismatch"
        a, b = values
        delta = b - a if reason is None and a is not None and b is not None else None
        ratio = delta / abs(a) if delta is not None and a is not None and a != 0 else None
        decision = "difference_only"
        if threshold is not None:
            if reason or ratio is None or not comparable or not all(qualified):
                decision = "indeterminate"
            else:
                decision = "regression" if ratio * (1 if direction == "lower" else -1) > threshold else "within_threshold"
        metric_rows.append({"metric": name, "statistic": statistic, "window": window,
                            "baseline": values[0], "candidate": values[1], "delta": delta, "relative_delta": ratio,
                            "unit": records[0]["unit"] if records[0] else None, "decision": decision,
                            "reason": reason, "max_regression_ratio": threshold})
    source_verified = all(side.get("source_identity", {}).get("status") == "observed" for side in conditions)
    return {"baseline_run": run_ids[0], "candidate_run": run_ids[1], "directly_comparable": comparable,
            "load_qualified": qualified, "repetitions": repetition_evidence,
            "conditions": rows, "metrics": metric_rows,
            "quality": [{"association": value.get("quality", {"status": "unverified"}),
                         "effective_trial_statuses": [t.get("metadata", {}).get("preparation", {}).get("quality_status", "unverified")
                                                      for t in group]} for value, group in zip(evidence, trials)],
            "source_content": "已核实固定来源身份" if source_verified else "原始数据内容一致性未验证"}
