"""Read external diagnostic backends for Grafana; never derive formal results."""

import asyncio
import base64
from time import time, monotonic

from aiohttp import ClientError, ClientResponseError, ClientSession, ClientTimeout, web
import orjson

from .service import SERVICE
from .projections import DIAGNOSTIC_RETENTION_SECONDS, choice, dashboard, time_range, unix_ms


BACKENDS = web.AppKey("diagnostic_backends", dict)


async def get_json(client, url, params):
    async with client.get(url, params=params, allow_redirects=False) as response:
        response.raise_for_status()
        # Backend limits also bound sample/entry counts; cap wire bytes before decoding.
        try:
            data = await response.content.readexactly(4 * 1024 * 1024 + 1)
        except asyncio.IncompleteReadError as exc:
            data = exc.partial
        else:
            raise ValueError("Diagnostic response exceeded 4 MiB")
        return orjson.loads(data)


def context(service, run_id):
    reader = service.catalog.reader(run_id)
    values = reader.measurements(run_id, scope="run", scope_id=run_id)
    return reader.run_record(run_id), {**values.get("observation", {}),
                                       "coverage_summary": values.get("observation_summary")}


async def coverage(request):
    run_id = request.match_info["run_id"]
    run, observation = await request.app[SERVICE].read(lambda: context(request.app[SERVICE], run_id))
    start, end = time_range(request.query, run)
    settings = observation.get("settings", {})
    targets = settings.get("targets", {})
    items = []
    query = f'up{{benchmark_run_id={orjson.dumps(run_id).decode()},observer_replay="0"}}'
    samples, error = [], None
    export_series, export_error = [], None
    if observation and end >= int((time() - DIAGNOSTIC_RETENTION_SECONDS) * 1000):
        try:
            async with ClientSession(timeout=ClientTimeout(total=4)) as client:
                payload = await get_json(client, request.app[BACKENDS]["prometheus"] + "/api/v1/query_range", {
                    "query": query, "start": start / 1000, "end": end / 1000,
                    "step": max(2, (end - start) / 200_000),
                })
                samples = payload["data"]["result"]
                if (settings.get("enabled") and settings.get("trace_mode", "benchmark") != "benchmark"
                        and settings.get("trace_export", True) and run.status in {"succeeded", "failed", "cancelled"}):
                    export_query = '{__name__=~"perf_benchmark_trace_(dropped|export_failures)_total",benchmark_run_id=' + orjson.dumps(run_id).decode() + '}'
                    finished = unix_ms(run.finished_at) or end
                    export_end = min(int(time() * 1000), finished + 15_000)
                    export_start = unix_ms(run.started_at or run.created_at) or start
                    try:
                        exported = await get_json(client, request.app[BACKENDS]["prometheus"] + "/api/v1/query_range", {
                            "query": export_query, "start": export_start / 1000, "end": export_end / 1000,
                            "step": max(2, (export_end - export_start) / 200_000),
                        })
                        export_series = exported["data"]["result"]
                    except (ClientError, TimeoutError, ValueError, KeyError):
                        export_error = "导出聚合计数不可用"
        except (ClientError, TimeoutError, ValueError, KeyError):
            error = "无可用数据：诊断后端不可达或响应无效"
    for kind, job, label in (("benchmark", "perf-benchmark", "客户端"), ("engines", "inference", "推理服务"), ("gpu", "gpu", "DCGM")):
        configured = bool(observation) if kind == "benchmark" else bool(targets.get(kind))
        series = [sample for sample in samples if sample["metric"].get("job") == job]
        selected = choice(request.query, "instance") if kind == "engines" else None
        if selected:
            series = [sample for sample in series if sample["metric"].get("instance") == selected]
        values = [float(value) for sample in series for _, value in sample.get("values", ())]
        state = (
            "未配置" if not configured else
            "超出 7 天诊断保留期" if end < int((time() - DIAGNOSTIC_RETENTION_SECONDS) * 1000) else
            error if error else
            "无样本" if not values else
            "采集失败" if all(value == 0 for value in values) else
            "部分采集失败" if any(value == 0 for value in values) else "采集正常"
        )
        items.append({"source": label, "state": state, "observed_targets": len(series)})
    try:
        drops = sum(float(series["values"][-1][1]) for series in export_series
                    if series["metric"].get("__name__") == "perf_benchmark_trace_dropped_total" and series.get("values"))
        failures = sum(float(series["values"][-1][1]) for series in export_series
                       if series["metric"].get("__name__") == "perf_benchmark_trace_export_failures_total" and series.get("values"))
    except (KeyError, IndexError, TypeError, ValueError):
        export_series, drops, failures = [], 0.0, 0.0
        export_error = "导出计数响应无效"
    trace_mode = settings.get("trace_mode", "benchmark")
    tracing = settings.get("enabled") and trace_mode != "benchmark"
    trace_export = settings.get("trace_export", True)
    export_state = (
        "未启用" if not tracing else
        "导出关闭（开销校准）" if not trace_export else
        "运行中：未查询聚合计数" if run.status not in {"succeeded", "failed", "cancelled"} else
        "超出保留期" if end < int((time() - DIAGNOSTIC_RETENTION_SECONDS) * 1000) else
        export_error or error or ("已取得聚合计数" if export_series else "未取得导出计数样本")
    )
    mode_label = {"benchmark": "Benchmark：仅正式结果", "observe": "Observe：采样", "debug": "Debug：全采样及有界输出事件"}[trace_mode]
    items.append({"source": "请求 Trace", "state": mode_label if settings.get("enabled") else "Observation 未启用",
                  "trace_mode": trace_mode, "trace_export": trace_export,
                  "observed_targets": None, "export_scope": "Run 聚合，不能归因单个请求",
                  "export_state": export_state, "dropped_spans": drops if export_series else None,
                  "export_failures": failures if export_series else None})
    return web.json_response({"items": items, "diagnostic_summary": observation.get("coverage_summary")})


def _span_id(value):
    if not value:
        return None
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, base64.binascii.Error):
        return None
    return raw.hex() if len(raw) == 8 else None


async def server_trace_evidence(request):
    run_id = request.match_info["run_id"]
    turn_id = choice(request.query, "turn_id")
    if not turn_id:
        return web.json_response({"items": [{"state": "请选择 Turn"}]})
    service = request.app[SERVICE]
    def read():
        reader = service.catalog.reader(run_id)
        run = reader.run_record(run_id)
        observation = reader.measurements(run_id, scope="run", scope_id=run_id).get("observation", {})
        page = reader.request_page(run_id, scope="attempt", turn_id=turn_id, limit=1000)
        return run, observation, page
    run, observation, page = await service.read(read)
    state = {"turn_id": turn_id, "trace_id": None, "server_spans": None,
             "services": None, "scope": "Tempo 诊断，不改变 SQLite 正式结果"}
    if page["total"] == 0:
        state["state"] = "未找到已提交 Attempt"
        return web.json_response({"items": [state]})
    settings = observation.get("settings", {})
    if not settings.get("enabled") or settings.get("trace_mode", "benchmark") == "benchmark":
        state["state"] = "Trace 未启用"
        return web.json_response({"items": [state]})
    if not settings.get("trace_export", True):
        state["state"] = "Trace 导出关闭（开销校准）"
        return web.json_response({"items": [state]})
    contexts = [row["measurements"].get("trace_context") for row in page["items"]]
    contexts = [value for value in contexts if value]
    if not contexts:
        state["state"] = "未取得客户端 Trace 身份"
        return web.json_response({"items": [state]})
    trace_ids = {value["trace_id"] for value in contexts}
    if len(trace_ids) != 1:
        state["state"] = "同一 Turn 的 Trace 身份不一致"
        return web.json_response({"items": [state]})
    state["trace_id"] = trace_id = next(iter(trace_ids))
    if page["total"] > len(page["items"]):
        state["state"] = "Attempt 超出单次查询上限，父子关系未验证"
        return web.json_response({"items": [state]})
    if run.status not in {"succeeded", "failed", "cancelled"}:
        state["state"] = "运行中：未查询服务端证据"
        return web.json_response({"items": [state]})
    try:
        async with ClientSession(timeout=ClientTimeout(total=4)) as client:
            payload = await get_json(client, request.app[BACKENDS]["tempo"] + "/api/v2/traces/" + trace_id, {})
    except ClientResponseError as exc:
        state["state"] = "未取得服务端证据" if exc.status == 404 else "Tempo 查询失败"
        return web.json_response({"items": [state]})
    except (ClientError, TimeoutError, ValueError):
        state["state"] = "Tempo 不可达或响应无效"
        return web.json_response({"items": [state]})
    try:
        spans = []
        for resource in payload["trace"]["resourceSpans"]:
            attributes = {item["key"]: item["value"].get("stringValue") for item in resource["resource"]["attributes"]}
            for scope in resource["scopeSpans"]:
                for span in scope["spans"]:
                    spans.append((_span_id(span["spanId"]), _span_id(span.get("parentSpanId")),
                                  attributes.get("service.name"), span.get("kind")))
        linked = {value["span_id"] for value in contexts}
        remaining = spans
        found = []
        while remaining:
            next_remaining = []
            for span_id, parent_id, name, kind in remaining:
                if span_id is not None and parent_id is not None and parent_id in linked:
                    linked.add(span_id)
                    if name != "perf-benchmark" and kind in {"SPAN_KIND_SERVER", 2}:
                        found.append(name or "未命名服务")
                else:
                    next_remaining.append((span_id, parent_id, name, kind))
            if len(next_remaining) == len(remaining):
                break
            remaining = next_remaining
    except (KeyError, TypeError, ValueError, IndexError):
        state["state"] = "Tempo Trace 响应无效"
        return web.json_response({"items": [state]})
    state.update(state="已取得服务端 Span" if found else "未取得服务端证据",
                 server_spans=len(found), services=", ".join(sorted(set(found))) if found else None)
    return web.json_response({"items": [state]})


async def diagnostics(request):
    run_id = request.match_info["run_id"]
    service = request.app[SERVICE]
    await service.read(lambda: service.catalog.reader(run_id).run_record(run_id))
    capture = choice(request.query, "capture")
    identity = orjson.dumps(run_id).decode()
    query = '{service_name="perf-nsight",mode="diagnostic"} | json | (benchmark_run_id = ' + identity + ' or diagnostic_run_id = ' + identity + ')'
    query += ' | diagnostic_id = ' + orjson.dumps(capture).decode() if capture else ' | kind = "summary"'
    try:
        async with ClientSession(timeout=ClientTimeout(total=4)) as client:
            payload = await get_json(client, request.app[BACKENDS]["loki"] + "/loki/api/v1/query_range", {
                "query": query, "start": str(int((time() - DIAGNOSTIC_RETENTION_SECONDS) * 1e9)),
                "end": str(int(time() * 1e9)), "limit": "1000", "direction": "backward",
            })
        entries = [orjson.loads(line) for stream in payload["data"]["result"] for _, line in stream["values"]]
    except (ClientError, TimeoutError, ValueError, KeyError):
        return web.json_response({"items": [], "state": "无可用数据：Nsight 诊断后端不可达或响应无效"})
    rows = []
    for entry in entries:
        own_run = entry.get("diagnostic_run_id")
        start, end = int(entry["started_unix_ns"]) // 1_000_000, int(entry["finished_unix_ns"]) // 1_000_000
        row = {key: entry.get(key) for key in (
            "diagnostic_id", "benchmark_run_id", "diagnostic_run_id", "host", "nsys_version",
            "report", "kind", "availability", "reason", "artifact_url", "rank", "name",
            "total_ns", "count", "share_percent",
        )}
        row.update(start_ms=start, end_ms=end, association="诊断 Run 未标识" if not own_run else "本 Run 采集" if own_run == run_id else "关联诊断 Run")
        row["artifact_status"] = "外部报告链接（可达性未验证）" if row["artifact_url"] else "报告链接不可用"
        row["capture_url"] = dashboard("perf-run", own_run or run_id, start, end, capture=entry["diagnostic_id"])
        row["diagnostic_url"] = None
        if own_run and own_run != run_id:
            try:
                record = await service.read(lambda: service.catalog.reader(own_run).run_record(own_run))
                row["diagnostic_url"] = dashboard("perf-run", own_run, unix_ms(record.started_at or record.created_at),
                                                  unix_ms(record.finished_at) if record.finished_at else "now", trial="$__all")
            except (web.HTTPNotFound, web.HTTPConflict):
                # Preserve the evidence even when that diagnostic's SQLite is not mounted.
                row["association"] = "关联诊断 Run（结果库未提供）"
                row["capture_url"] = dashboard("perf-run", run_id, start, end, capture=entry["diagnostic_id"])
        rows.append(row)
    return web.json_response({"items": rows, "state": "已导入" if rows else "未找到保留期内的采集记录", "truncated": len(entries) == 1000})


async def annotations(request):
    run_id = request.match_info["run_id"]
    service = request.app[SERVICE]
    run, trials = await service.read(lambda: (service.catalog.reader(run_id).run_record(run_id),
                                             service.catalog.reader(run_id).trials(run_id)))
    selected = choice(request.query, "trial")
    items = []
    for name, start, end in (
        ("Run preparation", run.preparing_at, run.started_at or run.finished_at),
        *((row["name"], row["started_at"], row["finished_at"]) for row in trials if selected is None or row["trial_id"] == selected),
    ):
        if start:
            items.append({"time": unix_ms(start), "timeEnd": unix_ms(end) if end else int(time() * 1000),
                          "text": name, "tags": "trial" if name != "Run preparation" else "preparation"})
    if request.query.get("diagnostics") == "true":
        body = (await diagnostics(request)).body
        assert isinstance(body, (bytes, bytearray))
        captures = orjson.loads(body)
        seen = set()
        for row in captures["items"]:
            key = row["diagnostic_id"], row["start_ms"], row["end_ms"]
            if row["diagnostic_run_id"] == run_id and key not in seen:
                seen.add(key)
                items.append({"time": row["start_ms"], "timeEnd": row["end_ms"],
                              "text": "Nsight " + row["diagnostic_id"], "tags": "diagnostic"})
        _, observation = await service.read(lambda: context(service, run_id))
        settings = observation.get("settings", {})
        if (settings.get("enabled") and settings.get("trace_mode", "benchmark") != "benchmark"
                and settings.get("trace_export", True)):
            start, end = time_range(request.query, run)
            query = '{ resource.service.name = "perf-benchmark" && span.benchmark.run_id = ' + orjson.dumps(run_id).decode()
            if selected:
                query += ' && span.benchmark.trial_id = ' + orjson.dumps(selected).decode()
            try:
                async with ClientSession(timeout=ClientTimeout(total=4)) as client:
                    payload = await get_json(client, request.app[BACKENDS]["tempo"] + "/api/search", {
                        "q": query + " }", "start": start // 1000, "end": end // 1000, "limit": 50,
                    })
                for trace in payload["traces"]:
                    items.append({"time": int(trace["startTimeUnixNano"]) // 1_000_000,
                                  "text": "采样 Trace " + trace["traceID"], "tags": "trace"})
            except (ClientError, TimeoutError, ValueError, KeyError):
                pass  # A diagnostic backend failure cannot erase persisted Trial boundaries.
    return web.json_response({"items": items})


def add_routes(app, prometheus_url, loki_url, tempo_url):
    app[BACKENDS] = {"prometheus": prometheus_url.rstrip("/"), "loki": loki_url.rstrip("/"), "tempo": tempo_url.rstrip("/")}
    app.router.add_get("/api/runs/{run_id}/coverage", coverage)
    app.router.add_get("/api/runs/{run_id}/server-evidence", server_trace_evidence)
    app.router.add_get("/api/runs/{run_id}/diagnostics", diagnostics)
    app.router.add_get("/api/runs/{run_id}/annotations", annotations)


async def enrich_timeline(request, result):
    """Fetch only selected sampled diagnostic events, with four bounded requests."""
    offset = int(result.pop('clock_offset_ns', '0'))
    coverage = result.get('coverage', [])
    by_trace = {}
    for row in coverage:
        if row.get('trace_id') and row['output_evidence'] == '未获得或已过期':
            by_trace.setdefault(row['trace_id'], []).append(row)
    deadline = monotonic() + 4
    async with ClientSession(timeout=ClientTimeout(total=4)) as client:
        async def fetch(identity):
            try:
                async with asyncio.timeout(max(.001, deadline - monotonic())):
                    payload = await get_json(client, request.app[BACKENDS]['tempo'] + '/api/v2/traces/' + identity, {})
                spans = [span for resource in payload['trace']['resourceSpans']
                         for scope in resource['scopeSpans'] for span in scope['spans']]
                return identity, spans, None
            except ClientResponseError as exc:
                return identity, [], '未获得或已过期' if exc.status == 404 else '后端不可达'
            except (ClientError, TimeoutError, ValueError, KeyError, TypeError):
                return identity, [], '后端不可达'
        identities = list(by_trace)
        for start in range(0, len(identities), 4):
            batch = identities[start:start+4]
            replies = (await asyncio.gather(*(fetch(identity) for identity in batch)) if monotonic() < deadline
                       else [(identity, [], '后端不可达') for identity in batch])
            for identity, spans, error in replies:
                for row in by_trace[identity]:
                    row['output_evidence'] = error or '未获得或已过期'
                    for span in spans:
                        if span.get('name') != 'benchmark.output_events' or _span_id(span.get('parentSpanId')) != row['attempt_span_id']:
                            continue
                        try:
                            attrs = {a['key']: next(iter(a['value'].values())) for a in span['attributes']}
                            if attrs.get('benchmark.turn_id') != row['turn_id'] or int(attrs['benchmark.attempt']) != row['attempt']:
                                continue
                            events = span.get('events', [])
                            if not 0 < len(events) <= 512 or any(e['name'] != 'output.event' for e in events):
                                raise ValueError('invalid output events')
                            stamps = [int(e['timeUnixNano']) - offset for e in events]
                            if stamps != sorted(stamps) or len(stamps) != int(attrs['benchmark.captured_outputs']):
                                raise ValueError('invalid output event order')
                            truncated = attrs['benchmark.outputs_truncated']
                            if (type(truncated) is not bool or int(attrs['benchmark.output_event_count']) != row['total_outputs']
                                    or truncated != (len(stamps) < row['total_outputs'])):
                                raise ValueError('invalid output coverage')
                            row.update(captured_outputs=len(stamps), truncated=truncated,
                                       output_evidence='已截断' if truncated else '可用')
                            missing = [item for item in result['items'] if item['request'] == row['request']
                                       and item['state'].startswith('输出细节未获得')
                                       and f"Attempt {row['attempt']}\n" in item['state']]
                            for item in missing:
                                result['items'].remove(item)
                                tail = (stamps[-1] + offset) / 1_000_000
                                if truncated and item['end'] > tail:
                                    result['items'].append({**item, 'start': tail,
                                        'duration_ms': item['end'] - tail, 'state': '输出细节已截断'})
                            for i, stamp in enumerate(stamps):
                                delta = None if i == 0 else (stamp-stamps[i-1])/1_000_000
                                result['outputs'].append({'request':row['request'],'turn_id':row['turn_id'],
                                    'attempt':row['attempt'],'output_index':i+1,'timestamp_ns':str(stamp),
                                    'time':(stamp+offset)/1_000_000,'interval_ms':delta})
                                if delta is not None and delta > 0 and row['phase_source'] == 'client_observed':
                                    result['items'].append({'request':row['request'],'start':(stamps[i-1]+offset)/1_000_000,
                                        'end':(stamp+offset)/1_000_000,'duration_ms':delta,
                                        'state':f'ITL · 输出 #{i} → #{i+1} · 诊断采样'})
                        except (KeyError, TypeError, ValueError, IndexError):
                            row['output_evidence'] = '后端响应无效'
                        break
    result['items'].sort(key=lambda item: item['start'])
