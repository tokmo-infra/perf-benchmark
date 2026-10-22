"""Read-only per-Run result query projections."""

from collections.abc import Mapping
from dataclasses import asdict, fields, replace
from typing import cast

from aiohttp import web
import orjson
from perf_benchmark.core.models import thaw_snapshot
from perf_benchmark.pipeline.registry import StageRegistry
from .projections import PAGE_SIZE, choice, dashboard, integer, metric_rows, request_timeline, request_view, run_view, time_range, unix_ms
from .statistics import exact_value, result_state, result_summary, statistics_table, window_records

def query_run(reader, run_id, operation, query):
    run = reader.run_record(run_id)
    measurements = reader.measurements(run_id, scope="run", scope_id=run_id)
    observation = measurements.get("observation", {})
    if operation == "evidence":
        return {"evidence": measurements.get("evidence", {}), "trials": [
            {"trial_id": t["trial_id"], "status": t["status"], "execution_status": t.get("execution_status"), "validity": t.get("validity"),
             "repetition": t.get("repetition"), "load_profile": t.get("load_profile"), "scheduling": t.get("scheduling"),
             "preparation": t["metadata"].get("preparation"),
             "graph_coverage": (t.get("scheduling") or {}).get("graph_coverage", [])} for t in reader.trials(run_id)],
                "capacity": [asdict(replace(row, definition=cast(Mapping[str, object] | None, thaw_snapshot(row.definition))))
                             for row in reader.sweep_records(run_id)]}
    if operation == "stages":
        timings = measurements.get("stages")
        items = []
        for stage in StageRegistry.STAGES:
            timing = (timings or {}).get(stage)
            items.append({"stage": stage,
                          "plugin": timing["plugin"] if timing else None,
                          "status": timing["status"] if timing else "not_recorded" if timings is None else "not_started",
                          "start_ms": unix_ms(timing["started_at"]) if timing else None,
                          "end_ms": unix_ms(timing["completed_at"]) if timing else None,
                          "duration_s": timing["duration_s"] if timing else None,
                          "started_ns": str(timing["started_ns"]) if timing else None,
                          "completed_ns": str(timing["completed_ns"]) if timing and timing["completed_ns"] is not None else None})
        return {"items": items, "scope": "run_lifecycle", "overlapping": True,
                "state": "recorded" if timings is not None else "not_recorded"}
    if operation == "run":
        return {"items": [{**run_view(run), "trace_mode": observation.get("settings", {}).get("trace_mode", "benchmark"),
                           "trace_export": observation.get("settings", {}).get("trace_export", True),
                           "observation_enabled": observation.get("settings", {}).get("enabled", False),
                           "clock": {key: str(value) for key, value in observation.get("clock", {}).items()}}]}
    if operation == "trials":
        return {"items": [{key: row[key] for key in (
            "trial_id", "name", "ordinal", "status", "started_at", "finished_at", "turn_count",
        )} for row in reader.trials(run_id)]}
    trial = choice(query, "trial")
    if trial and trial not in {row["trial_id"] for row in reader.trials(run_id)}:
        raise web.HTTPNotFound(text="Trial not found in Run")
    if operation in {"statistics", "windows", "evidence"}:
        trials = [row for row in reader.trials(run_id) if trial is None or row["trial_id"] == trial]
        if operation == "windows":
            return {"items": window_records(reader, run_id, trials, observation.get("clock"))}
        return statistics_table(reader, run_id, trials, query, run.status)
    if operation == "metrics":
        scopes = [("trial", row["trial_id"], row["name"], row["status"]) for row in reader.trials(run_id)
                  if trial is None or row["trial_id"] == trial]
        if trial is None:
            scopes.insert(0, ("run", run_id, run.name, run.status))
        items = []
        for scope, identity, name, status in scopes:
            items.extend({**row, "scope": scope, "scope_id": identity, "name": name,
                          "state": result_state(run.status)}
                         for row in metric_rows(reader.metric_points(
                             run_id, scope=scope, scope_id=identity, window=query.get("window", "overall"),
                         )))
        return {"items": items, "state": result_state(run.status), "summary": [result_summary(run.status)]}
    if operation == "groups":
        trials = [trial] if trial else [row["trial_id"] for row in reader.trials(run_id)]
        return {"items": [{**{field.name: getattr(row, field.name) for field in fields(row)
                               if field.name not in {"dimensions", "statistics", "unavailable_reasons"}},
                           "dimensions": orjson.dumps(dict(row.dimensions)).decode(),
                           "statistics": orjson.dumps(dict(row.statistics)).decode(),
                           "unavailable_reasons": orjson.dumps(dict(row.unavailable_reasons)).decode()}
                          for key in trials for row in reader.grouped_metrics(run_id, key)]}
    if operation == "targets":
        targets = observation.get("settings", {}).get("targets", {})
        return {"items": [{"kind": kind, **target} for kind, group in targets.items() for target in group]}
    start, end = time_range(query, run)
    if operation == "series":
        clock = observation.get("clock")
        if not clock:
            return {"items": [], "state": "clock_not_observed", "truncated": False}
        metric = query.get("metric", "")
        if not metric or len(metric) > 128:
            raise ValueError("metric is required")
        offset = clock["unix_ns"] - clock["monotonic_ns"]
        result = reader.metric_series(run_id, trial_id=trial, metric=metric,
                                      start_ns=start * 1_000_000 - offset, end_ns=end * 1_000_000 - offset)
        if result["total"] > len(result["items"]):
            raise ValueError("time range exceeds 10000 metric points; select a shorter range or one Trial")
        statistics = query.get("statistics", "time_wm,am,p50,p90,p95,p99,min,max,value").split(",")
        basis = ("响应完成时间；token 总量 / 固定10秒" if metric.startswith("workload_") else
                 "桶内实际在途并发的时间加权值" if metric == "actual_in_flight_concurrency" else
                 "最终 Attempt 开始于桶内的成功 Turn；原始输出事件间隔" if metric == "itl" else
                 "最终 Attempt 开始于桶内的成功 Turn")
        items = [{**{k: v for k, v in row.items() if k not in {"start_ns", "end_ns"}},
                  "start_ns": str(row["start_ns"]), "end_ns": str(row["end_ns"]),
                  "time": (row["start_ns"] + offset) // 1_000_000,
                  "end_time": (row["end_ns"] + offset) // 1_000_000,
                  "value_text": exact_value(row["value"]) if row["value"] is not None else "不可用",
                  "basis": basis, "state": result_state(run.status),
                  "series": metric + " · " + row["scope_id"] + " · " + ("mean" if row["statistic"] == "am" else row["statistic"])}
                 for row in result["items"] if row["statistic"] in statistics]
        return {"items": items, "total": result["total"], "truncated": result["total"] > len(result["items"]),
                "state": result_state(run.status), "committed_only": True, "bucket_seconds": 10,
                "summary": [{**result_summary(run.status), "basis": basis,
                             "interval": "[开始, 结束)", "clock_uncertainty_ns": str(clock["uncertainty_ns"])}]}
    if operation == "timeline":
        result = request_timeline(reader, run_id, observation, query, trial, start, end)
        result["summary"] = [{key: value for key, value in result.items()
                              if key not in {"items", "outputs", "coverage"}}]
        return result
    offset = integer(query, "offset", 0)
    page_size = integer(query, "page_size", PAGE_SIZE, minimum=1)
    if page_size > 1000:
        raise ValueError("page_size must be at most 1000")
    turn_id = choice(query, "turn_id")
    if operation in {"turn", "attempts"} and not turn_id:
        return {"items": [], "metrics": [], "timings": [], "total": 0,
                "pagination": [{"total": 0, "offset": offset, "page_size": page_size}], "state": "select_turn"}
    scope = "attempt" if operation == "attempts" else "turn"
    result = reader.request_page(
        run_id, scope=scope, trial_id=trial, turn_id=turn_id, status=choice(query, "status"),
        offset=offset, limit=page_size, session_id=choice(query, "session_id"), detail=operation != "requests",
        from_ms=start if operation == "requests" else None, to_ms=end if operation == "requests" else None,
    )
    if operation == "turn" and not result["items"]:
        raise web.HTTPNotFound(text="Committed Turn not found")
    variables = {key: query.get(key, default) for key, default in (
        ("trial", "all"), ("status", "all"), ("turn_id", ""),
        ("request_offset", "0"), ("request_size", "50"),
    )}
    variables["trial"] = trial or "$__all"
    if operation == "requests":
        variables.update(request_offset=offset, request_size=page_size)
    else:
        variables["turn_id"] = ""
    items = []
    previous_ready_ns = None
    previous_attempt = None
    for row in result["items"]:
        item = request_view(row, observation.get("clock"),
                            observation.get("settings", {}).get("slow_seconds", 10))
        if operation == "attempts":
            admitted_ns = row["measurements"].get("admitted_ns")
            gap_ns = (admitted_ns - previous_ready_ns
                      if admitted_ns is not None and previous_ready_ns is not None
                      and previous_attempt == row["attempt"] - 1 else None)
            item["retry_wait_ms"] = gap_ns / 1_000_000 if gap_ns is not None and gap_ns >= 0 else None
            item["retry_wait_state"] = (
                "已观测" if item["retry_wait_ms"] is not None else
                "首个 Attempt" if row["attempt"] == 1 else
                "前一 Attempt 不在本页" if previous_attempt != row["attempt"] - 1 else "时间事实不足"
            )
            item["context_page_state"] = (
                "仍有 Attempt 未加载" if offset > 0 or offset + len(result["items"]) < result["total"]
                else "本 Turn 已提交 Attempt 均已加载"
            )
            previous_ready_ns = row["measurements"].get("result_ready_ns")
            previous_attempt = row["attempt"]
        item["detail_url"] = dashboard("perf-run", run_id, start, end, **{
            **variables, "selected_turn": row["turn_id"], "attempt_offset": 0, "output_page": 1,
        })
        items.append(item)
    return {"items": items, "total": result["total"],
            "pagination": [{"total": result["total"], "offset": offset, "page_size": page_size}],
            "metrics": [{"attempt": item["attempt"], **metric} for item in items for metric in item["metrics"]],
            "timings": [{"attempt": item["attempt"], **stamp} for item in items for stamp in item["timing"]],
            "from_ms": start, "to_ms": end, "scope": scope, "committed_only": True}
