"""Presentation-only tables from persisted ResultReader records; never recompute metrics."""

DISTRIBUTION_COLUMNS = ("n", "am", "p50", "p90", "p95", "p99", "min", "max")
EXTRA_COLUMNS = ("eligible_n", "std", "p1", "p5", "p10")
MEAN_COLUMNS = ("time_wm", "am", "tm", "hm", "twm")
PRIMARY_MEANS = {"tpot", "prefill_throughput", "decode_throughput"}
REPETITIONS = ("median", "mean", "mad", "cv", "min", "max", "trimmed_mean_10pct",
               "bootstrap_median_ci_lower", "bootstrap_median_ci_upper")
VIEWS = {"distribution", "distribution_full", "means", "means_all", "scalar", "compare", "groups", "repetitions"}


def result_state(run_status):
    # A terminal Trial row can precede the last metric batch. Only the Run's
    # successful Saver finalization guarantees all final metric writes finished.
    if run_status == "succeeded":
        return "final"
    return "partial" if run_status in {"failed", "cancelled"} else "in_progress"


def result_summary(run_status):
    state = result_state(run_status)
    label, message = {
        "final": ("最终结果", "Run 已成功收尾；展示 SQLite 正式统计，图表框选不重算。"),
        "partial": ("保留结果", "Run 失败或取消；仅展示已提交结果，不能作为完整测试结论。"),
        "in_progress": ("未最终确定", "Run 尚未成功收尾；表格可能尚未收齐，历史趋势桶也可能回填。"),
    }[state]
    return {"state": state, "label": label, "message": message, "run_status": run_status}


def exact_value(value):
    """Round-trip numeric text, without the dashboard's significant-digit rounding."""
    return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)


def _display(record, statistic):
    if statistic not in record["statistics"]:
        return "不适用"
    metadata = record.get("statistic_metadata", {}).get(statistic, record)
    value = record["statistics"][statistic]
    if metadata.get("availability") != "available" or value is None:
        return "不可用：" + (metadata.get("unavailable_reason") or "not_observed")
    return exact_value(value)


def _unit(record):
    return next((meta["unit"] for stat, meta in record.get("statistic_metadata", {}).items()
                 if stat not in {"n", "eligible_n"}), record["unit"])


def window_records(reader, run_id, trials, clock=None):
    rows = []
    for trial in trials:
        for window in reader.metric_windows(run_id, scope="trial", scope_id=trial["trial_id"]):
            if window["window"].startswith("timeline:"):
                continue
            row = {"trial": trial["name"], "trial_id": trial["trial_id"], **window}
            for name in ("start", "end"):
                stamp = window[name + "_ns"]
                row[name + "_ns"] = str(stamp) if stamp is not None else None
                row[name + "_ms"] = ((stamp + clock["unix_ns"] - clock["monotonic_ns"]) // 1_000_000
                                     if stamp is not None and clock else None)
            rows.append(row)
    return rows


def statistics_table(reader, run_id, trials, query, run_status):
    view = query.get("view", "distribution")
    if view not in VIEWS:
        raise ValueError("unknown statistics view")
    window = query.get("window", "overall")
    statistic = query.get("statistic", "p90")
    if statistic not in {*DISTRIBUTION_COLUMNS, *EXTRA_COLUMNS, *MEAN_COLUMNS}:
        raise ValueError("unknown comparison statistic")
    windows = window_records(reader, run_id, trials)
    if window.startswith("timeline:") or (run_status == "succeeded" and windows
                                         and window not in {w["window"] for w in windows}):
        raise ValueError("unknown persisted statistics window")
    rows = []
    for trial in trials:
        identity = trial["trial_id"]
        base = {"trial": trial["name"], "trial_id": identity, "window": window}
        if view == "groups":
            for group in reader.grouped_metrics(run_id, identity):
                if group.window != window:
                    continue
                dimensions = " × ".join(axis for axis, _ in group.dimensions)
                selected = query.get("dimension", "input_tokens")
                if selected != "all" and dimensions != selected:
                    continue
                rows.append({**base, "dimensions": dimensions,
                             "group": " × ".join(value for _, value in group.dimensions),
                             "metric": group.metric, "unit": group.unit,
                             "total": group.sample_count, "eligible": group.eligible_sample_count,
                             "valid": group.valid_sample_count,
                             **{stat: exact_value(value) if value is not None else "不可用"
                                for stat, value in group.statistics.items()},
                             "unavailable_reasons": str(dict(group.unavailable_reasons)) if group.unavailable_reasons else "",
                             "formula_version": group.formula_version})
            continue
        if view == "repetitions":
            continue
        points = reader.metric_points(run_id, scope="trial", scope_id=identity, window=window)
        if view == "compare":
            by_window = {w["window"]: reader.metric_points(run_id, scope="trial", scope_id=identity, window=w["window"])
                         for w in windows if w["trial_id"] == identity}
            names = sorted({name for metrics in by_window.values() for name in metrics})
            for name in names:
                record = next(metrics[name] for metrics in by_window.values() if name in metrics)
                stat = "value" if "value" in record["statistics"] else statistic
                row = {"trial": trial["name"], "trial_id": identity, "metric": name,
                       "unit": _unit(record), "statistic": stat}
                for key, metrics in by_window.items():
                    row[key] = _display(metrics[name], stat) if name in metrics else "不可用：not_recorded"
                rows.append(row)
            continue
        for name, record in points.items():
            scalar = "value" in record["statistics"]
            if (view == "scalar") != scalar:
                continue
            if view == "means" and name not in PRIMARY_MEANS:
                continue
            columns = (("value",) if scalar else MEAN_COLUMNS if view.startswith("means") else
                       DISTRIBUTION_COLUMNS + (EXTRA_COLUMNS if view == "distribution_full" else ()))
            rows.append({**base, "metric": name, "unit": _unit(record),
                         **{stat: _display(record, stat) for stat in columns},
                         "formula_version": record["formula_version"]})
    if view == "repetitions":
        groups = {t["comparison"]["group_id"] for t in trials if t.get("comparison")}
        for comparison in reader.comparison_points(run_id):
            if comparison["group_id"] not in groups:
                continue
            points = reader.metric_points(run_id, scope="comparison", scope_id=comparison["group_id"], window=window)
            for name, record in points.items():
                # Match the longest suffix, e.g. trimmed_mean_10pct before mean.
                statistics = {next(stat[:-(len(suffix) + 1)] for suffix in sorted(REPETITIONS, key=len, reverse=True)
                                   if stat.endswith("_" + suffix)) for stat in record["statistics"]
                              if any(stat.endswith("_" + suffix) for suffix in REPETITIONS)}
                for stat in sorted(statistics):
                    unit = next((meta["unit"] for key, meta in record["statistic_metadata"].items()
                                 if key == stat + "_median"), record["unit"])
                    rows.append({"comparison": comparison["group_id"], "axis": comparison["axis_name"],
                                 "axis_value": comparison["axis_value"], "window": window,
                                 "metric": name, "statistic": stat, "unit": unit,
                                 **{label: _display(points[metric], "value") if metric in points else "不可用：not_recorded"
                                    for label, metric in (("valid_repetitions", "repetition_valid"),
                                                          ("expected_repetitions", "repetition_expected"),
                                                          ("qualified_repetitions", "repetition_qualified"))},
                                 **{suffix: _display(record, stat + "_" + suffix) for suffix in REPETITIONS},
                                 "formula_version": record["formula_version"]})
    search = query.get("search", "").casefold()
    if search:
        rows = [row for row in rows if search in row["metric"].casefold()]
    if len(trials) == 1:
        for row in rows:
            row.pop("trial", None)
    priority = ("ttft", "tpot", "itl", "e2e", "turn_total_duration", "time_to_last_output", "response_drain",
                "prefill_throughput", "decode_throughput", "input_tokens", "output_tokens",
                "session_duration", "session_turns", "session_first_turn_ttft", "session_subsequent_turn_ttft",
                "output_tokens_per_event_estimate", "speculative_acceptance_proxy")
    if view in {"distribution", "distribution_full", "means", "means_all"}:
        rows.sort(key=lambda row: (row["trial_id"], priority.index(row["metric"])
                                  if row["metric"] in priority else len(priority), row["metric"]))
    summary = result_summary(run_status)
    summary.update(window=window, rows=len(rows), data_state="recorded" if rows else "not_recorded")
    return {"items": rows, "windows": windows, "view": view, "window": window,
            "total": len(rows), "state": summary["state"], "summary": [summary],
            "time_range_applied": False}
