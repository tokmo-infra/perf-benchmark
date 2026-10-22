from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

from rich.console import Console
from rich.table import Table
from rich.text import Text

from perf_benchmark.pipeline.contracts.results import MetricRecord, ResultReader, TrialRecord

WINDOW_LABELS = {
    "overall": "Overall",
    "maxtpm": "Dataset 25%-75%",
    "last30s": "Last 30 s",
    "drop_first_20pct": "Drop first 20%",
    "steady_occupancy": "Steady occupancy",
    "steady_strip_10pct": "Steady strip 10%",
}
STATISTIC_ORDER = (
    "n",
    "am",
    "std",
    "tm",
    "hm",
    "twm",
    "time_wm",
    "p1",
    "p5",
    "p10",
    "p50",
    "p90",
    "p95",
    "p99",
    "min",
    "max",
)
STATISTIC_LABELS = {
    "value": "Value",
    "n": "N / eligible",
    "am": "AM",
    "std": "Std",
    "tm": "Trimmed AM",
    "hm": "HM",
    "twm": "Token-WM",
    "time_wm": "Time-WM",
    "p1": "P1",
    "p5": "P5",
    "p10": "P10",
    "p50": "P50",
    "p90": "P90",
    "p95": "P95",
    "p99": "P99",
    "min": "Min",
    "max": "Max",
}

_DISTRIBUTION_METRICS = (
    "input_tokens",
    "output_tokens",
    "ttft",
    "tpot",
    "itl",
    "e2e",
    "time_to_last_output",
    "response_drain",
    "prefill_throughput",
    "decode_throughput",
    "session_duration",
    "session_turns",
    "session_first_turn_ttft",
    "session_subsequent_turn_ttft",
)
_OPERATIONAL_METRICS = (
    "target_concurrency",
    "target_active_sessions",
    "attempts",
    "retries",
    "token_cache_observable_requests",
    "token_cache_hit_requests",
    "token_cache_observation_rate",
    "token_cache_request_hit_rate",
    "token_cache_token_hit_rate",
    "successful_concurrency",
    "actual_in_flight_concurrency",
    "actual_active_sessions",
    "sessions_started",
    "sessions_completed",
    "sessions_failed",
    "session_completed_turns_total",
    "session_expected_turns_total",
    "session_incomplete_count",
    "target_session_rate",
    "actual_session_rate",
    "target_request_rate",
    "actual_request_rate",
    "scheduler_lag",
    "session_queue_delay",
    "request_queue_delay",
    "session_pending_depth",
    "request_pending_depth",
    "drain_duration",
    "partial_sessions",
    "demand_limited",
    "session_success_rate",
)
_OTHER_GROUPS = frozenset(_DISTRIBUTION_METRICS + _OPERATIONAL_METRICS)

_LABEL_OVERRIDES = {
    "workload_rps": "Request throughput",
    "workload_prompt": "Input throughput",
    "workload_new_prompt": "New input throughput",
    "workload_cached_prompt": "Cached input throughput",
    "workload_completion": "Output throughput",
    "workload_total": "Total token throughput",
    "ttft": "TTFT",
    "tpot": "TPOT",
    "itl": "ITL",
    "e2e": "E2EL",
    "session_first_turn_ttft": "Session first-turn TTFT",
    "session_subsequent_turn_ttft": "Session subsequent-turn TTFT",
}


def render_report(console: Console, run_id: str, results: ResultReader) -> None:
    summary = results.summary(run_id)
    trials = results.trials(run_id)
    _render_run_summary(console, summary, results.location)
    _render_trial_comparison(console, run_id, trials, results)
    for trial in trials:
        _render_trial(console, run_id, trial, results)
    _render_comparison_summary(console, run_id, results)
    _render_sweep(console, results.sweep_records(run_id))
    _render_evidence(console, run_id, results, trials)


def _render_run_summary(
    console: Console,
    summary: Mapping[str, object],
    location: str,
) -> None:
    table = Table(title="Benchmark Summary", border_style="cyan", show_header=False)
    table.add_column("Field", style="dim")
    table.add_column("Value")
    for label, key in (
        ("Run", "run_id"),
        ("Name", "name"),
        ("Status", "status"),
        ("Trials", "trial_count"),
        ("Sessions", "session_count"),
        ("Turns", "turn_count"),
        ("Metric points", "metric_count"),
    ):
        table.add_row(label, str(summary.get(key, "-")))
    if summary.get("error"):
        table.add_row("Error", str(summary["error"]))
    table.add_row("Database", location)
    console.print(table)


def _render_trial_comparison(
    console: Console,
    run_id: str,
    trials: tuple[TrialRecord, ...],
    results: ResultReader,
) -> None:
    table = Table(title="Trial Comparison", border_style="green")
    for label, justify in (
        ("Trial", "left"),
        ("Load profile", "left"),
        ("Status", "left"),
        ("Succeeded", "right"),
        ("RPS", "right"),
        ("TTFT P50", "right"),
        ("TTFT P95", "right"),
        ("TPOT P50", "right"),
        ("Output tok/s", "right"),
    ):
        table.add_column(label, justify=justify)
    for trial in trials:
        metrics = results.metric_points(
            run_id,
            scope="trial",
            scope_id=trial["trial_id"],
            window="overall",
            names=("succeeded", "workload_rps", "ttft", "tpot", "workload_completion"),
        )
        table.add_row(
            trial["name"],
            _load_label(trial),
            trial["status"],
            _metric_value(metrics, "succeeded", "value"),
            _metric_value(metrics, "workload_rps", "value"),
            _metric_value(metrics, "ttft", "p50"),
            _metric_value(metrics, "ttft", "p95"),
            _metric_value(metrics, "tpot", "p50"),
            _metric_value(metrics, "workload_completion", "value"),
        )
    console.print(table)


def _render_trial(
    console: Console,
    run_id: str,
    trial: TrialRecord,
    results: ResultReader,
) -> None:
    heading = Text()
    heading.append("Trial ", style="bold")
    heading.append(trial["name"], style="bold cyan")
    heading.append(f"  status={trial['status']}")
    heading.append(f"  load={_load_label(trial)}")
    console.print(heading)
    matrix = {
        window: results.metric_points(
            run_id,
            scope="trial",
            scope_id=trial["trial_id"],
            window=window,
        )
        for window in (
            item["window"]
            for item in results.metric_windows(
                run_id,
                scope="trial",
                scope_id=trial["trial_id"],
            )
            if not item["window"].startswith(("timeline:", "maxtpm:minute:"))
        )
    }
    minute_windows = [item for item in results.metric_windows(
        run_id, scope="trial", scope_id=trial["trial_id"],
    ) if item["window"].startswith("maxtpm:minute:")]
    if minute_windows:
        table = Table(title="Dataset 25%-75% TPM / complete minutes")
        table.add_column("Minute")
        table.add_column("TPM", justify="right")
        table.add_column("Input / output tok", justify="right")
        table.add_column("Cached tok / hit rate", justify="right")
        table.add_column("Revenue / min", justify="right")
        for item in minute_windows:
            measured = results.metric_points(run_id, scope="trial", scope_id=trial["trial_id"],
                                             window=item["window"])
            table.add_row(str(int(item["window"].rsplit(":", 1)[1])),
                          _metric_value(measured, "maxtpm", "value"),
                          _metric_value(measured, "maxtpm_input_tpm", "value") + " / " +
                          _metric_value(measured, "maxtpm_output_tpm", "value"),
                          _metric_value(measured, "maxtpm_cached_tpm", "value") + " / " +
                          _metric_value(measured, "maxtpm_cache_hit_rate", "value"),
                          _metric_value(measured, "revenue_per_minute", "value"))
        console.print(table)
    names = dict.fromkeys(name for points in matrix.values() for name in points)
    distributions = tuple(
        name
        for name in names
        if name in _DISTRIBUTION_METRICS
        or any(
            set(points.get(name, {}).get("statistics", {})) - {"value", "eligible_n"}
            for points in matrix.values()
        )
    )
    load = tuple(name for name in names if name not in _OTHER_GROUPS and name not in distributions)
    _render_matrix(console, "Load Overview", load, matrix)
    _render_matrix(console, "Latency & Throughput Statistics", distributions, matrix)
    _render_matrix(
        console,
        "Request, Session, Cache & Concurrency",
        _OPERATIONAL_METRICS,
        matrix,
    )
    _render_window_diagnostics(console, run_id, trial["trial_id"], results, matrix)


def _render_matrix(
    console: Console,
    title: str,
    names: tuple[str, ...],
    matrix: Mapping[str, Mapping[str, MetricRecord]],
) -> None:
    table = Table(title=title, border_style="blue")
    table.add_column("Metric", style="dim", no_wrap=True)
    table.add_column("Statistic", style="dim", no_wrap=True)
    table.add_column("Unit", no_wrap=True)
    for window in matrix:
        table.add_column(WINDOW_LABELS.get(window, window), justify="right", overflow="fold")
    for name in names:
        records = [points[name] for points in matrix.values() if name in points]
        if not records:
            continue
        unit = records[0]["unit"]
        statistics = {key for point in records for key in point["statistics"]} - {"eligible_n"}
        ordered = tuple(key for key in ("value", *STATISTIC_ORDER) if key in statistics)
        ordered += tuple(sorted(statistics - set(ordered)))
        for statistic in ordered or ("value",):
            table.add_row(
                _label(name),
                STATISTIC_LABELS.get(statistic, statistic),
                "samples" if statistic == "n" else unit,
                *(_cell(points.get(name), statistic, unit) for points in matrix.values()),
            )
    console.print(table)


def _render_window_diagnostics(
    console: Console,
    run_id: str,
    trial_id: str,
    results: ResultReader,
    matrix: Mapping[str, Mapping[str, MetricRecord]],
) -> None:
    table = Table(title="Window Availability", border_style="yellow")
    for column in ("Window", "Status", "Duration", "Reason"):
        table.add_column(column)
    for window in results.metric_windows(run_id, scope="trial", scope_id=trial_id):
        duration = window["duration_s"]
        table.add_row(
            WINDOW_LABELS.get(window["window"], window["window"]),
            window["availability"],
            "-" if duration is None else f"{duration:.3f} s",
            window["unavailable_reason"] or "-",
        )
    console.print(table)

    reasons: Counter[tuple[str, str]] = Counter()
    for window, metrics in matrix.items():
        for point in metrics.values():
            reason = point.get("unavailable_reason")
            if point.get("availability") == "unavailable" and reason:
                reasons[(window, reason)] += 1
    if not reasons:
        return
    diagnostics = Table(title="Unavailable Metric Points", border_style="yellow")
    diagnostics.add_column("Window")
    diagnostics.add_column("Reason")
    diagnostics.add_column("Count", justify="right")
    for (window, reason), count in reasons.items():
        diagnostics.add_row(WINDOW_LABELS.get(window, window), reason, str(count))
    console.print(diagnostics)


def _render_comparison_summary(console: Console, run_id: str, results: ResultReader) -> None:
    comparisons = results.comparison_points(run_id)
    if not comparisons:
        return
    table = Table(title="Comparison Robust Summary", border_style="magenta")
    for label in (
        "Point",
        "Axis",
        "Value",
        "RPS median",
        "TTFT P50 median",
        "TPOT P50 median",
        "Output tok/s median",
        "Status",
        "Samples",
    ):
        table.add_column(label, justify="right" if label not in {"Point", "Status"} else "left")
    for comparison in comparisons:
        metrics = results.metric_points(
            run_id,
            scope="comparison",
            scope_id=comparison["group_id"],
            window="overall",
            names=("workload_rps", "ttft", "tpot", "workload_completion"),
        )
        coordinates = (
            ("workload_rps", "value_median"),
            ("ttft", "p50_median"),
            ("tpot", "p50_median"),
            ("workload_completion", "value_median"),
        )
        cells = [_metric_value(metrics, metric, statistic) for metric, statistic in coordinates]
        metadata = (
            metrics.get("workload_rps", {}).get("statistic_metadata", {}).get("value_median", {})
        )
        valid = metadata.get("valid_sample_count")
        expected = metadata.get("expected_sample_count")
        status = metrics.get("workload_rps", {}).get("availability", "unavailable")
        table.add_row(
            comparison["group_id"],
            _label(comparison["axis_name"]),
            _axis_value(comparison["axis_value"], comparison["axis_unit"]),
            *cells,
            status,
            "-" if valid is None else f"{valid}/{expected or 0}",
        )
    console.print(table)


def _load_label(trial: TrialRecord) -> str:
    profile = trial.get("load_profile")
    if not profile:
        return "-"
    values = [str(profile.get("mode", "-"))]
    for name in ("target_concurrency", "target_session_rate", "target_request_rate"):
        value = profile.get(name)
        if value is not None:
            values.append(f"{name.removeprefix('target_')}={value}")
    return "; ".join(values)


def _axis_value(value: object, unit: str | None) -> str:
    return f"{value} {unit}" if unit else str(value)


def _metric_value(
    metrics: Mapping[str, MetricRecord],
    name: str,
    statistic: str,
) -> str:
    point = metrics.get(name)
    unit = point["unit"] if point is not None else ""
    return _cell(point, statistic, unit)


def _cell(point: MetricRecord | None, statistic: str, unit: str) -> str:
    if point is None:
        return "-"
    metadata = point.get("statistic_metadata", {}).get(statistic, {})
    availability = metadata.get("availability", point.get("availability"))
    if availability not in {"available", "partial"}:
        return "unavailable"
    value = point.get("statistics", {}).get(statistic)
    if statistic == "n":
        if value is None:
            return "-"
        eligible = point.get("statistics", {}).get("eligible_n")
        return f"{int(value):,}" if eligible is None else f"{int(value):,}/{int(eligible):,}"
    formatted = _format_value(value, metadata.get("unit", unit))
    return formatted + "*" if value is not None and availability == "partial" else formatted


def _format_value(value: object, unit: str) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "-"
    numeric = float(value)
    if unit in {"requests", "attempts", "sessions", "samples", "tokens", "turns"}:
        return f"{int(numeric):,}"
    if unit == "ratio":
        return f"{numeric * 100:.2f}%"
    if numeric and (abs(numeric) >= 1e9 or abs(numeric) < 1e-3):
        return f"{numeric:.3e}"
    return f"{numeric:.3f}"


def _label(name: str) -> str:
    return _LABEL_OVERRIDES.get(name, name.replace("_", " ").capitalize())


__all__ = ["render_report"]


def _render_sweep(console, records):
    if not records:
        return
    final = records[-1]
    console.print(Text(
        f"Sweep {final.axis}: {final.status}; passing={final.lower}, failing={final.upper}; "
        f"reason={final.reason or '-'}", style="bold cyan",
    ))
    table = Table(title="SLO evidence", border_style="cyan")
    for label in ("Point", "Phase", "Trial", "Metric", "Rule", "Actual", "Result"):
        table.add_column(label)
    for record in records:
        for check in record.checks:
            rule = check.rule
            table.add_row(
                str(record.value), record.phase, check.trial_id,
                f"{rule.metric}.{rule.statistic}",
                f"{rule.op} {rule.original} ({rule.threshold:g} {rule.unit})",
                "-" if check.value is None else f"{check.value:g} {rule.unit}",
                check.status + (f": {check.reason}" if check.reason else ""),
            )
    console.print(table)

    validity = Table(title="Load validity (indeterminate points do not establish capacity)")
    for label in ("Point", "Trial", "Execution", "Validity", "Reasons"):
        validity.add_column(label)
    for record in records:
        for item in record.validity:
            validity.add_row(str(record.value), item.trial_id, item.execution_status,
                             item.status, "; ".join(item.reasons) or "-")
    console.print(validity)


def _render_evidence(console, run_id, results, trials):
    from rich.text import Text
    evidence = results.measurements(run_id, scope="run", scope_id=run_id).get("evidence", {})
    console.print(Text(evidence.get("source_content", "原始数据内容一致性未验证")))
    table = Table(title="实验条件与质量关联")
    table.add_column("条件")
    table.add_column("状态")
    table.add_column("值")
    for key, row in evidence.get("conditions", {}).items():
        table.add_row(Text(key), Text(row["status"]), Text(str(row["value"])))
    quality = evidence.get("quality", {"status": "unverified"})
    table.add_row("质量关联", Text(quality["status"]), Text(str(quality)))
    console.print(table)
    table = Table(title="Trial 有效性、压力、交付和图覆盖")
    table.add_column("Trial")
    table.add_column("工具执行")
    table.add_column("负载资格")
    table.add_column("目标 / 实际压力")
    table.add_column("offered / admitted / completed")
    table.add_column("原因 / 覆盖")
    for trial in trials:
        validity = trial.get("validity") or {}
        profile = trial.get("load_profile") or {}
        scheduling = trial.get("scheduling") or {}
        observations = {name: value for name, value, _unit in validity.get("observations", [])}
        target = profile.get("target_concurrency") or profile.get("max_active_sessions") or profile.get("target_request_rate") or profile.get("target_session_rate")
        actual = observations.get("actual_pressure", observations.get("request_offered", observations.get("session_offered")))
        delivery = " / ".join(str(scheduling.get(key, "?")) for key in
                              ("offered_attempts", "admitted_attempts", "completed_attempts"))
        coverage = (trial.get("scheduling") or {}).get("graph_coverage", [])
        table.add_row(Text(trial["trial_id"]), Text(str(trial.get("execution_status") or "unknown")), Text(validity.get("status", "unverified")),
                      Text(f"{target if target is not None else '?'} / {actual if actual is not None else '?'}"),
                      Text(delivery), Text(str({"reasons": validity.get("reasons", []), "observations": observations, "graph_coverage": coverage,
                                "preparation": trial["metadata"].get("preparation")})))
    console.print(table)
