"""Presentation projections over persisted result and diagnostic facts."""

from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime
from time import time
from typing import overload
from urllib.parse import urlencode

import orjson
from perf_benchmark.pipeline.contracts.results import RunRecord

PAGE_SIZE = 50
TERMINAL = {"succeeded", "failed", "cancelled"}
DIAGNOSTIC_RETENTION_SECONDS = 7 * 24 * 3600



@overload
def unix_ms(value: str) -> int: ...


@overload
def unix_ms(value: None) -> None: ...


def unix_ms(value: str | None) -> int | None:
    return None if value is None else int(datetime.fromisoformat(value).timestamp() * 1000)


def dashboard(uid, run_id, start, end, **variables):
    return "/d/" + uid + "?" + urlencode({
        "var-run": run_id, "from": start, "to": end,
        **{"var-" + name: value for name, value in variables.items() if value is not None},
    }, doseq=True)


def run_view(record: RunRecord):
    row: dict[str, object] = {key: value for key, value in asdict(record).items() if key not in {"error", "counts"}}
    start = unix_ms(record.started_at or record.preparing_at or record.created_at)
    end = unix_ms(record.finished_at)
    row.update(start_ms=start, end_ms=end, duration_s=None if end is None else (end - start) / 1000)
    row.update({key: record.counts.get(key) if record.status in TERMINAL else None
                for key in ("trials", "succeeded_turns", "failed_turns", "cancelled_turns")})
    row["trend_status"] = (
        "超出 7 天诊断保留期" if end is not None and end < int((time() - DIAGNOSTIC_RETENTION_SECONDS) * 1000)
        else "保留期内，见采集覆盖" if end is not None else "运行中，见采集覆盖"
    )
    row["run_url"] = dashboard("perf-run", record.id, start, end if end is not None else "now", trial="$__all")
    return row

def choice(query, name):
    value = query.get(name)
    return None if value in (None, "", "all", "$__all", ".*") else value


@overload
def integer(query: Mapping[str, str], name: str, default: int, *, minimum: int = 0) -> int: ...


@overload
def integer(query: Mapping[str, str], name: str, default: None = None, *, minimum: int = 0) -> int | None: ...


def integer(query: Mapping[str, str], name: str, default: int | None = None, *, minimum: int = 0) -> int | None:
    value = query.get(name, default)
    if value is None:
        return None
    try:
        number = int(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if number < minimum or number > 10**15:
        raise ValueError(f"{name} outside supported bounds")
    return number


def time_range(query: Mapping[str, str], run: RunRecord) -> tuple[int, int]:
    start = integer(query, "from", unix_ms(run.started_at or run.preparing_at or run.created_at))
    finished = unix_ms(run.finished_at)
    end = int(time() * 1000) if query.get("to") == "now" else integer(
        query, "to", finished if finished is not None else int(time() * 1000),
    )
    if start > end:
        raise ValueError("time range is reversed")
    return start, end


def metric_rows(metrics):
    return [
        {"metric": name, "statistic": statistic, "value": value,
         "unit": metric.get("statistic_metadata", {}).get(statistic, metric)["unit"],
         "availability": metric.get("statistic_metadata", {}).get(statistic, metric).get("availability"),
         "reason": metric.get("statistic_metadata", {}).get(statistic, metric).get("unavailable_reason"),
         "source": metric.get("statistic_metadata", {}).get(statistic, metric).get("source"),
         "formula_version": metric.get("formula_version")}
        for name, metric in metrics.items() for statistic, value in metric.get("statistics", {}).items()
    ]


def request_timeline(reader, run_id, observation, query, trial, start, end):
    settings = observation.get("settings", {})
    empty = {"items": [], "outputs": [], "coverage": [], "total": 0, "page": 1, "pages": 1}
    detail = query.get("detail") == "1"
    selected = choice(query, "selected_turn")
    if detail and not selected:
        return {**empty, "state": "select_turn"}
    if not settings.get("enabled") or settings.get("trace_mode", "benchmark") == "benchmark":
        return {**empty, "state": "trace_disabled"}
    clock = observation.get("clock")
    if not clock:
        return {**empty, "state": "clock_not_observed"}
    page = integer(query, "timeline_page", 1, minimum=1)
    result = reader.request_page(
        run_id, scope="attempt", trial_id=trial, turn_id=selected if detail else choice(query, "turn_id"),
        status=None if detail else choice(query, "status"),
        from_ms=None if detail else start, to_ms=None if detail else end,
        offset=(page - 1) * PAGE_SIZE, limit=PAGE_SIZE, sampled_only=not detail,
    )
    offset = clock["unix_ns"] - clock["monotonic_ns"]
    items, outputs, coverage = [], [], []
    for row in result["items"]:
        request = row["measurements"]
        trace = request.get("trace_context")
        sampled = bool(trace and trace["sampled"])
        turn_label = (str(row["turn_ordinal"] + 1) if row["turn_ordinal"] is not None
                      else f'未提交（{row["turn_id"]}）')
        lane = f'Trial {row["trial_ordinal"] + 1} / Session {row["session_ordinal"] + 1} / Turn {turn_label}'
        source_record = None if row["source_order"] is None else row["source_order"] + 1
        source_text = (f'来源记录 #{source_record}' if source_record is not None else '来源记录：未记录')
        for key in ("source_session_id", "source_turn_id"):
            if row[key] is not None:
                source_text += f' · {key}={row[key]}'
        captured = 0
        truncated = False
        common = {"request": lane, "turn_id": row["turn_id"], "attempt": row["attempt"],
                  "trial_id": row["trial_id"], "session_id": row["session_id"], "status": row["status"],
                  "source_record": source_record, "source_order": row["source_order"],
                  "source_session_id": row["source_session_id"], "source_turn_id": row["source_turn_id"],
                  "trace_id": trace["trace_id"] if trace else None,
                  "attempt_span_id": trace["span_id"] if trace else None,
                  "detail_url": dashboard("perf-run", run_id, start, end, trial=trial or "$__all",
                                          selected_turn=row["turn_id"], timeline_page=page)}
        tpot = row["metrics"].get("tpot", {})
        tpot_value = tpot.get("statistics", {}).get("value")
        tpot_text = (f'TPOT：{tpot_value:.6g} {tpot["unit"]} / token'
                     if tpot.get("availability") == "available" and tpot_value is not None
                     else 'TPOT：不可用（' + (tpot.get("unavailable_reason") or "未记录") + '）')

        def interval(begin, finish, state):
            if begin is not None and finish is not None and finish > begin:
                items.append({"request": lane,
                              "start": (begin + offset) / 1_000_000,
                              "end": (finish + offset) / 1_000_000,
                              "state": f'{state} · {(finish - begin) / 1_000_000:.6g} ms'
                              f' · Attempt {row["attempt"]}\n{tpot_text}\n{source_text}',
                              "duration_ms": (finish - begin) / 1_000_000})

        streaming_text = request["response_mode"] == "stream" and request["output_kind"] == "text"
        if streaming_text:
            interval(request["started_ns"], request["first_output_ns"], "客户端等待首输出")
        else:
            interval(request["started_ns"], request["response_completed_ns"], "非流式或非文本响应")
        if streaming_text and sampled:
            interval(request["first_output_ns"], request["last_output_ns"], "输出细节未获得或已过期")
        if streaming_text:
            interval(request["last_output_ns"], request["response_completed_ns"], "等待协议结束")
        if streaming_text and request["first_output_ns"] is None:
            interval(request["started_ns"], request["response_completed_ns"], "未观测到首输出")
        first, last, sent = request["first_output_ns"], request["last_output_ns"], request["started_ns"]
        coverage.append({**common, "captured_outputs": captured,
                         "total_outputs": request["output_event_count"], "truncated": truncated,
                         "output_evidence": ("未创建 Trace" if trace is None else
                                             "Observe：未采集逐输出细节" if settings.get("trace_mode") != "debug" else
                                             "Trace 导出关闭（开销校准）" if not settings.get("trace_export", True) else
                                             "未采样：无逐输出细节" if not sampled else
                                             "未获得或已过期"),
                         "tpot": tpot_value if tpot.get("availability") == "available" else None,
                         "tpot_unit": tpot.get("unit"), "tpot_reason": tpot.get("unavailable_reason"),
                         "first_output_wait_ms": (first - sent) / 1_000_000
                         if streaming_text and first is not None and sent is not None else None,
                         "output_receive_span_ms": (last - first) / 1_000_000
                         if streaming_text and first is not None and last is not None else None,
                         "phase_source": "client_observed" if streaming_text else "non_streaming_or_non_text",
                         "clock_uncertainty_ns": str(clock["uncertainty_ns"])})
    pages = max(1, (result["total"] + PAGE_SIZE - 1) // PAGE_SIZE)
    return {"items": sorted(items, key=lambda item: item["start"]),
            "outputs": outputs, "coverage": coverage,
            "state": "committed_turn_attempts" if detail else "sampled_committed_attempts",
            "trace_mode": settings.get("trace_mode"), "trace_export": settings.get("trace_export", True),
            "sample_rate": 1.0 if settings.get("trace_mode") == "debug" else settings.get("sample_rate"), "clock_offset_ns": str(offset),
            "total": result["total"], "page": page,
            "pages": pages, "remaining_attempts": detail and page * PAGE_SIZE < result["total"]}


def request_view(row, clock, slow_seconds=10):
    attempt = "attempt" in row
    measurement = row["measurements"]
    request = measurement if attempt else measurement["request"]
    profile = request.get("profile") or {}
    output = {key: row.get(key) for key in (
        "turn_id", "trial_id", "session_id", "status", "ordinal", "attempt", "scenario_passed", "error_origin",
        "estimated_input_tokens", "requested_output_tokens", "reference_output_tokens",
    )}
    output.update({"input_tokens": row["usage"].get("prompt_tokens"),
                   "output_tokens": row["usage"].get("completion_tokens"),
                   "cached_tokens": row["usage"].get("cached_tokens"),
                   "profile": profile, "metrics": metric_rows(row["metrics"])})
    output.update({key: row.get(key) for key in (
        "source_order", "source_session_id", "source_turn_id", "source_branch_id",
    )})
    output["source_record"] = row["source_order"] + 1 if row.get("source_order") is not None else None
    output["request"] = " / ".join(
        f"{label} {row[key] + 1}" if row.get(key) is not None else f"{label} 未提交"
        for label, key in (("Trial", "trial_ordinal"), ("Session", "session_ordinal"), ("Turn", "turn_ordinal"))
    )
    start = (request.get("admitted_ns") or request.get("started_ns")) if attempt else (
        measurement.get("total_started_ns") or measurement["lifecycle"]["started_ns"]
    )
    end = request.get("result_ready_ns") if attempt else measurement["lifecycle"]["completed_ns"]
    for key, stamp in (("start_ms", start), ("end_ms", end)):
        output[key] = (clock["unix_ns"] + stamp - clock["monotonic_ns"]) // 1_000_000 if clock and stamp is not None else (
            unix_ms(row.get("started_at" if key == "start_ms" else "finished_at")) if not attempt else None
        )
    output["clock_source"] = "monotonic_anchor" if clock else "recorded_wall_clock"
    trace = request.get("trace_context")
    output["trace_id"] = trace["trace_id"] if trace else None
    for name in ("ttft", "tpot", "e2e"):
        metric = row["metrics"].get(name, {})
        output[name] = metric.get("statistics", {}).get("value") if metric.get("availability") == "available" else None
        output[name + "_unit"] = metric.get("unit")
        output[name + "_reason"] = metric.get("unavailable_reason")
    if attempt:
        output["diagnostic_focus"] = (
            "失败或取消" if row["status"] in {"failed", "cancelled"} else
            "慢请求" if output["e2e"] is not None and output["e2e"] >= slow_seconds else "上下文"
        )
    eligibility = {} if attempt else {key: measurement.get(key) for key in (
        "eligibility_started_ns", "eligibility_finished_ns", "eligibility_target_ns", "dependency_ready_ns",
    )}
    output["timing"] = [{"event": key, "monotonic_ns": str(value),
                          "unix_ms": (clock["unix_ns"] + value - clock["monotonic_ns"]) // 1_000_000 if clock else None}
                         for key, value in {**request, **eligibility, **{key: row.get(key) for key in ("trigger_ns", "decision_ns", "applied_ns")}}.items()
                         if key.endswith("_ns") and value is not None]
    output["trace_url"] = None
    if trace:
        pane = {
            "datasource": "tempo",
            "queries": [{"refId": "A", "datasource": {"type": "tempo", "uid": "tempo"},
                         "queryType": "traceql", "query": output["trace_id"]}],
        }
        if output["start_ms"] is not None and output["end_ms"] is not None:
            pane["range"] = {"from": str(output["start_ms"]), "to": str(output["end_ms"])}
        output["trace_url"] = "/explore?" + urlencode({
            "schemaVersion": 1, "panes": orjson.dumps({"workbench": pane}).decode(),
        })
    return output
