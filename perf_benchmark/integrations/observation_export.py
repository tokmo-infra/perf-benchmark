"""Observation-process consumers. Import this module only in the observer."""

import asyncio
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field as dataclass_field
from hashlib import blake2b
import math
from time import monotonic, time
from typing import TypeGuard

from prometheus_client import Counter, Gauge, Histogram

from perf_benchmark.core.observation import ClockAnchor, TraceContext

from perf_benchmark.pipeline.contracts.results import (
    AttemptResultRecord,
    MeasurementRecord,
    MetricPoint,
    RunRecord,
    SessionResultRecord,
    TrialResultRecord,
    TurnResultRecord,
)

PREFIX = "perf_benchmark_"
BUCKETS = tuple(0.0001 * 2**k for k in range(21))


@dataclass(slots=True)
class LiveSnapshot:
    """Bounded, observer-owned view of committed results and diagnostics."""

    run_id: str | None = None
    run_status: str = "pending"
    trials: dict[str, dict] = dataclass_field(default_factory=dict)
    counts: dict[str, dict[str, dict[str, int]]] = dataclass_field(default_factory=dict)
    final_metrics: dict[str, dict] = dataclass_field(default_factory=dict)
    preview: dict[str, dict] = dataclass_field(default_factory=dict)
    load: dict = dataclass_field(default_factory=dict)
    writers: dict = dataclass_field(default_factory=dict)
    resources: dict = dataclass_field(default_factory=dict)
    gpu: dict = dataclass_field(default_factory=lambda: {"state": "not_configured", "devices": []})
    revision: int = 0
    pending_commits: int = 0
    updated_at: float | None = None
    load_at: float | None = None
    resources_at: float | None = None
    gpu_at: float | None = None
    notifications_dropped: int = 0
    output_messages: dict = dataclass_field(default_factory=dict)

    def diagnostic(self, packet: dict, at: float) -> None:
        self.load = dict(packet.get("writers", {}).get("load", {}))
        self.writers = {name: dict(value) for name, value in packet.get("writers", {}).items() if name != "load"}
        self.load_at = at
        self.notifications_dropped = packet.get("dropped", 0)
        self.output_messages = dict(packet.get("output_messages", {}))

    def resource_sample(self, samples: dict, at: float) -> None:
        self.resources = {f"{group}.{metric}": value for (group, metric), value in samples.items()}
        self.resources_at = at

    def consume(self, change) -> None:
        row = change.record
        self.run_id = change.run_id
        if isinstance(row, RunRecord):
            self.run_status = row.status
        elif isinstance(row, TrialResultRecord):
            self.trials[row.id] = {"status": row.status, "name": row.name, "ordinal": row.ordinal}
        elif isinstance(row, (TurnResultRecord, AttemptResultRecord)):
            if change.trial_id is None:
                raise ValueError("request result requires Trial identity")
            scope = "turn" if isinstance(row, TurnResultRecord) else "attempt"
            statuses = self.counts.setdefault(change.trial_id, {}).setdefault(scope, {})
            statuses[row.status] = statuses.get(row.status, 0) + 1
            if scope == "turn" and row.status == "succeeded":
                for name, expected_unit in (("ttft", "s"), ("tpot", "ms"), ("e2e", "s")):
                    metric = row.metrics.get(name)
                    value = metric.statistics.get("value") if metric is not None else None
                    if metric is None or metric.availability != "available" or metric.unit != expected_unit or not _number(value):
                        continue
                    seconds = value / 1000 if name == "tpot" else value
                    if seconds < 0:
                        continue
                    key = f"{change.trial_id}:{name}"
                    item = self.preview.setdefault(key, {"n": 0, "buckets": [0] * (len(BUCKETS) + 1)})
                    item["n"] += 1
                    bucket = next((i for i, bound in enumerate(BUCKETS) if seconds <= bound), len(BUCKETS))
                    item["buckets"][bucket] += 1
        elif isinstance(row, MetricPoint) and row.scope in {"run", "trial"} and not row.window.startswith("timeline:"):
            key = f"{row.scope}:{row.scope_id}:{row.window}:{row.metric}:{row.statistic}"
            self.final_metrics[key] = {"value": row.value, "unit": row.unit, "availability": row.availability,
                                       "reason": row.unavailable_reason, "window": row.window,
                                       "source": row.source, "formula_version": row.formula_version,
                                       "valid_sample_count": row.valid_sample_count,
                                       "expected_sample_count": row.expected_sample_count}

    def view(self) -> dict:
        gpu = self.gpu.copy()
        gpu["devices"] = [device.copy() for device in gpu["devices"]]
        for device in gpu["devices"]:
            if device.get("sample_at") is not None and time() - device["sample_at"] > 15:
                device["state"] = "stale"
        if gpu["state"] == "available" and not any(device.get("state") == "available" for device in gpu["devices"]):
            gpu["state"] = "stale"
        previews = {}
        for key, item in self.preview.items():
            values = {}
            for label, fraction in (("p50", 0.50), ("p99", 0.99)):
                target, cumulative = math.ceil(item["n"] * fraction), 0
                for index, count in enumerate(item["buckets"]):
                    cumulative += count
                    if cumulative >= target:
                        values[label] = BUCKETS[index] if index < len(BUCKETS) else None
                        break
            previews[key] = {"n": item["n"], "unit": "s", **values, "kind": "approximate_committed_samples"}
        return {"run_id": self.run_id, "run_status": self.run_status, "trials": self.trials,
                "counts": self.counts, "final_metrics": self.final_metrics, "preview": previews,
                "load": self.load, "writers": self.writers, "resources": self.resources, "gpu": gpu,
                "revision": self.revision, "pending_commits": self.pending_commits, "updated_at": self.updated_at,
                "load_at": self.load_at, "resources_at": self.resources_at, "gpu_at": self.gpu_at,
                "notifications_dropped": self.notifications_dropped, "output_messages": self.output_messages}


class Telemetry:
    def __init__(self, registry):
        self.ready = Gauge(PREFIX + "observer_ready", "Caught up to committed results", registry=registry)
        self.replay = Gauge(PREFIX + "observer_replay", "Standalone replay, not a live completion clock", registry=registry)
        self.progress = Gauge(PREFIX + "observer_revision", "Committed revision read progress", registry=registry)
        self.backlog = Gauge(PREFIX + "observer_pending_commits", "Revision difference, not unread requests", registry=registry)
        self.updated = Gauge(PREFIX + "observer_updated_unixtime", "Last successful read", registry=registry)
        self.available = Gauge(PREFIX + "trace_available", "Recorded clock available for diagnostic Trace reconstruction", registry=registry)
        self.status = Gauge(PREFIX + "status", "Latest recorded lifecycle state", ["run", "trial", "scope", "status"], registry=registry)
        labels = ["run", "trial", "scope", "status"]
        self.requests = Counter(PREFIX + "completed", "Committed terminal records", labels, registry=registry)
        self.histograms = {name: Histogram(
            PREFIX + name + ("_seconds_per_token" if name == "tpot" else "_seconds"),
            "Completed available samples; arrival at observer, not request completion time", labels,
            buckets=BUCKETS, registry=registry,
        ) for name in ("ttft", "tpot", "e2e")}
        self.missing = Counter(PREFIX + "unavailable", "Unavailable request metric", labels + ["metric"], registry=registry)
        final_labels = [
            "run", "scope", "scope_id", "window", "metric", "statistic", "unit",
        ]
        self.final = Gauge(PREFIX + "metric_value", "Authoritative final metric point", final_labels, registry=registry)
        self.final_available = Gauge(PREFIX + "metric_available", "Final point availability", final_labels, registry=registry)
        self.dropped = Counter(PREFIX + "trace_dropped", "Lost or unconfirmed diagnostic spans; only summaries can be replayed", ["reason"], registry=registry)
        self.failures = Counter(PREFIX + "trace_export_failures", "Diagnostic sender or OTLP export/shutdown failures", registry=registry)
        self.exported = Counter(PREFIX + "trace_exported_spans", "Successfully exported diagnostic spans", registry=registry)
        self.processed = Counter(PREFIX + "trace_processed_spans", "Processed spans, including construction without network export", registry=registry)
        self.processed_requests = Counter(PREFIX + "trace_processed_request_spans", "Processed Attempt spans, including construction without network export", registry=registry)
        self.trace_missing = Counter(PREFIX + "trace_unavailable", "Records without reconstructable Trace boundaries", ["reason"], registry=registry)
        self.sampled_out = Counter(PREFIX + "trace_sampled_out", "Intentional summary sampling", ["reason"], registry=registry)
        self.resources = Gauge(PREFIX + "resource", "Process and writer diagnostic sample", ["process", "metric"], registry=registry)
        self.load = Gauge(PREFIX + "load", "Latest absolute execution occupancy; not a rate", ["trial", "metric"], registry=registry)
        self.load_labels = set()
        self.retire = {}
        self.groups = set()

    def update_load(self, snapshot):
        trial = snapshot.get("trial_id")
        values = {(trial, key): value for key, value in snapshot.items() if trial and key != "trial_id"}
        for labels in self.load_labels - values.keys():
            self.load.remove(*labels)
        for labels, value in values.items():
            self.load.labels(*labels).set(value)
        self.load_labels = set(values)

    def retire_trials(self):
        for trial, deadline in tuple(self.retire.items()):
            if deadline is None:
                self.retire[trial] = monotonic() + 10
                continue
            if monotonic() < deadline:
                continue
            for labels in tuple(self.groups):
                if labels[1] == trial:
                    for histogram in self.histograms.values():
                        histogram.remove(*labels)
                    self.requests.remove(*labels)
                    for metric in self.histograms:
                        self.missing.remove(*labels, metric)
                    self.groups.remove(labels)
            del self.retire[trial]


def _number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


class TraceBuilder:
    def __init__(self, slow_seconds, *, sample_rate=.01, reconstruct=False):
        from opentelemetry.sdk.resources import Resource

        self.clock = None
        self.formula_version = None
        self.slow_seconds = slow_seconds
        self.sample_rate, self.reconstruct = sample_rate, reconstruct
        self.resource = Resource.create({"service.name": "perf-benchmark"})

    def _sampled(self, run_id, identity):
        # Replay has no original ordinal/seed sampling decision. Its stable ID
        # selection is explicitly an offline choice and remains consistent across retries.
        digest = blake2b(f"{len(run_id)}:{run_id}{identity}".encode(), digest_size=16).digest()
        return int.from_bytes(digest) < int(self.sample_rate * (1 << 128))

    def _session_context(self, run_id, session_id):
        from opentelemetry.trace import SpanContext, TraceFlags

        identity = f"{len(run_id)}:{run_id}:session:{session_id}".encode()
        return SpanContext(int.from_bytes(blake2b(identity, digest_size=16).digest()),
                           int.from_bytes(blake2b(identity + b":workflow", digest_size=8).digest()) or 1,
                           False, TraceFlags(1))

    def session(self, row: SessionResultRecord, run_id: str, trial_id: str):
        from opentelemetry.sdk.trace import ReadableSpan
        from opentelemetry.trace import SpanKind, Status, StatusCode

        if self.clock is None or not self._sampled(run_id, row.id):
            return None
        lifecycle = row.measurements["lifecycle"]
        start, end = lifecycle["started_ns"], lifecycle["completed_ns"]
        if start is None or end is None or end < start:
            return None
        return ReadableSpan(name="benchmark.session", context=self._session_context(run_id, row.id),
                            resource=self.resource, kind=SpanKind.INTERNAL,
                            attributes={"benchmark.run_id": run_id, "benchmark.trial_id": trial_id,
                                        "benchmark.session_id": row.id, "benchmark.status": row.status,
                                        "benchmark.turn_count": row.turn_count,
                                        "benchmark.evidence": "committed_session_lifecycle",
                                        "sampling.reason": "workflow",
                                        "clock.uncertainty_ns": self.clock.uncertainty_ns},
                            status=Status(StatusCode.ERROR if row.status == "failed" else StatusCode.UNSET),
                            start_time=self.clock.unix(start), end_time=self.clock.unix(end)), False

    def build(self, row: TurnResultRecord | AttemptResultRecord, run_id: str, trial_id: str):
        from opentelemetry.sdk.trace import Event, ReadableSpan
        from opentelemetry.trace import Link, SpanContext, SpanKind, Status, StatusCode, TraceFlags

        scope = "turn" if isinstance(row, TurnResultRecord) else "attempt"
        request = row.measurements["request"] if isinstance(row, TurnResultRecord) else row.measurements
        if self.clock is None:
            return None
        recorded = request.get("trace_context")
        turn_id = row.id if isinstance(row, TurnResultRecord) else row.turn_id
        if recorded:
            trace = TraceContext(**recorded)
        elif self.reconstruct:
            # Identity exactly matches live TraceContext.create; offline sampling
            # never claims to reproduce a decision that was not recorded.
            identity = f"{len(run_id)}:{run_id}{turn_id}".encode()
            trace = TraceContext(blake2b(identity, digest_size=16).hexdigest(),
                                 blake2b(identity + f":attempt:{request['attempts']}".encode(), digest_size=8).hexdigest(),
                                 blake2b(identity + b":turn", digest_size=8).hexdigest(),
                                 self._sampled(run_id, turn_id))
        else:
            return None
        metrics = row.metrics
        e2e_metric = metrics.get("e2e")
        e2e = e2e_metric.statistics.get("value") if e2e_metric is not None else None
        reason = (
            "error" if row.status in {"failed", "cancelled"} else
            "retry" if request["attempts"] > 1 else "head" if trace.sampled or (
                self.reconstruct and self._sampled(run_id, turn_id)) else
            "slow" if e2e_metric is not None and e2e_metric.unit == "s" and _number(e2e) and e2e >= self.slow_seconds else None
        )
        if reason is None:
            return None
        start, end = request["started_ns"], request.get("result_ready_ns")
        if isinstance(row, TurnResultRecord):
            measurement = row.measurements
            start = measurement["total_started_ns"] or measurement["lifecycle"]["started_ns"]
            end = measurement["lifecycle"]["completed_ns"]
        else:
            start = start or request["admitted_ns"]
        if start is None or end is None or end < start:
            return None
        context = SpanContext(int(trace.trace_id, 16), int(
            trace.parent_span_id if scope == "turn" else trace.span_id, 16
        ), False, TraceFlags(1))
        parent = None if scope == "turn" else SpanContext(
            int(trace.trace_id, 16), int(trace.parent_span_id, 16), False, TraceFlags(1)
        )
        attributes = {
            "benchmark.run_id": run_id, "benchmark.trial_id": trial_id,
            "benchmark.turn_id": turn_id,
            "benchmark.attempt": request["attempts"], "benchmark.status": row.status,
            "benchmark.observation_version": "observation-v2",
            "sampling.reason": reason, "sampling.server_details_may_be_missing": not trace.sampled,
            "clock.uncertainty_ns": self.clock.uncertainty_ns,
            "benchmark.trace_origin": "recorded_context" if recorded else "offline_reconstructed",
            "benchmark.evidence": "committed_request_facts",
        }
        if not recorded:
            attributes["benchmark.trace_propagated"] = False
        if row.session_id is not None:
            attributes["benchmark.session_id"] = row.session_id
        for name in ("prompt_tokens", "completion_tokens", "cached_tokens"):
            value = row.usage.get(name)
            if _number(value):
                attributes["benchmark.usage." + name] = value
        profile = request.get("profile") or {}
        if scope == "attempt":
            model = profile.get("model")
            if isinstance(model, str) and model:
                attributes["gen_ai.request.model"] = model[:256]
            for source, target in (("prompt_tokens", "gen_ai.usage.input_tokens"),
                                   ("completion_tokens", "gen_ai.usage.output_tokens"),
                                   ("cached_tokens", "gen_ai.usage.cache_read.input_tokens")):
                tokens = row.usage.get(source)
                if type(tokens) is int and tokens >= 0:
                    attributes[target] = tokens
        for name in ("message_count", "history_message_count", "tool_count", "text_chars",
                     "system_chars", "developer_chars", "system_tokens_estimate",
                     "developer_tokens_estimate", "body_bytes", "model", "stream",
                     "temperature", "top_p", "max_tokens", "context_mode", "cache_isolation",
                     "tokenizer_fingerprint"):
            value = profile.get(name)
            if type(value) in (str, bool, int, float):
                attributes["benchmark.request." + name] = value
        attributes["benchmark.usage.cache_availability"] = (
            "reported" if "cached_tokens" in row.usage else "unknown")
        # Only numeric metric values and contract descriptions cross this seam;
        # metadata, errors, body, endpoint and arbitrary usage fields do not.
        for name, metric in sorted(metrics.items()):
            if len(attributes) >= 240:
                attributes["benchmark.metrics_truncated"] = True
                break
            if self.formula_version is not None:
                attributes[f"benchmark.metric.{name}.formula_version"] = self.formula_version
            for field in ("unit", "availability", "unavailable_reason", "source", "formula_version"):
                value = getattr(metric, field)
                if isinstance(value, str):
                    attributes[f"benchmark.metric.{name}.{field}"] = value[:256]
            value = metric.statistics.get("value") if metric is not None else None
            if _number(value):
                attributes[f"benchmark.metric.{name}.value"] = value
        events = [Event(name, timestamp=self.clock.unix(timestamp)) for name, timestamp in (
            ("request.start", request["started_ns"]), ("first_output", request["first_output_ns"]),
            ("last_output", request["last_output_ns"]), ("response.complete", request["response_completed_ns"]),
            ("result.ready", request.get("result_ready_ns")),
        ) if timestamp is not None and start <= timestamp <= end]
        return ReadableSpan(
            name="benchmark." + scope, context=context, parent=parent, resource=self.resource,
            attributes=attributes, events=events,
            links=[Link(self._session_context(run_id, row.session_id))] if scope == "turn" and row.session_id is not None else [],
            kind=SpanKind.CLIENT if scope == "attempt" else SpanKind.INTERNAL,
            status=Status(StatusCode.ERROR if row.status == "failed" else StatusCode.UNSET),
            start_time=self.clock.unix(start), end_time=self.clock.unix(end),
        ), reason != "head"


class SpanQueue:
    def __init__(self, exporter, telemetry, capacity=2048, byte_limit=8 * 1024 * 1024, slow_rate=20):
        self.exporter, self.telemetry = exporter, telemetry
        self.capacity, self.byte_limit = capacity, byte_limit
        self.normal, self.priority = deque(), deque()
        self.count = self.bytes = 0
        self.incomplete = False
        self.closing = False
        self.inflight = []
        self.wake = asyncio.Event()
        self.slow_rate, self.tokens = slow_rate, max(1, slow_rate)
        self.refilled = monotonic()

    def offer(self, span, priority):
        if self.closing:
            self.incomplete = True
            self.telemetry.dropped.labels("closed").inc()
            return False
        if span.attributes.get("sampling.reason") == "slow":
            now = monotonic()
            self.tokens = min(max(1, self.slow_rate), self.tokens + (now - self.refilled) * self.slow_rate)
            self.refilled = now
            if self.tokens < 1:
                self.telemetry.sampled_out.labels("slow_budget").inc()
                return False
            self.tokens -= 1
        # Conservative field budget, not a Python RSS bound. Do not serialize JSON before protobuf.
        size = 1024 + sum(64 + 4 * (len(key) + len(str(value))) for key, value in span.attributes.items())
        size += sum(128 + 4 * len(event.name) +
                    sum(64 + 4 * (len(k) + len(str(v))) for k, v in (event.attributes or {}).items())
                    for event in span.events)
        while self.count >= self.capacity or self.bytes + size > self.byte_limit:
            if not self.normal:
                self.incomplete = True
                self.telemetry.dropped.labels("overload").inc()
                return False
            _, victim_size = self.normal.popleft()
            self.incomplete = True
            self.count -= 1
            self.bytes -= victim_size
            self.telemetry.dropped.labels("evicted_normal").inc()
        (self.priority if priority else self.normal).append((span, size))
        self.count += 1
        self.bytes += size
        if self.count == 1 or self._batch_ready():
            self.wake.set()
        return True

    def close(self):
        self.closing = True
        self.wake.set()

    def abort(self, reason):
        """Account for queued and in-flight delivery with an unknown outcome once."""
        self.close()
        if self.count:
            self.incomplete = True
            self.telemetry.dropped.labels(reason).inc(self.count)
        self.normal.clear()
        self.priority.clear()
        self.inflight.clear()
        self.count = self.bytes = 0

    async def finish(self, sender, timeout=4):
        self.close()
        try:
            async with asyncio.timeout(timeout):
                await asyncio.shield(sender)
        except TimeoutError:
            self.abort("drain_timeout")
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
        except Exception:
            self.abort("sender_failed")
            self.telemetry.failures.inc()
        except asyncio.CancelledError:
            self.abort("drain_cancelled")
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
            raise
        finally:
            if self.exporter is not None:
                try:
                    await asyncio.to_thread(self.exporter.shutdown)
                except Exception:
                    self.incomplete = True
                    self.telemetry.failures.inc()

    def _batch_ready(self):
        return self.count >= min(256, self.capacity) or self.bytes * 2 >= self.byte_limit

    async def run(self):
        try:
            await self._run()
        except asyncio.CancelledError:
            self.abort("sender_cancelled")
            raise

    async def _run(self):
        if self.exporter is not None:
            from opentelemetry.sdk.trace.export import SpanExportResult

        while not self.closing or self.normal or self.priority:
            if not self.closing and not self.count:
                self.wake.clear()
                await self.wake.wait()
                continue
            if not self.closing and not self._batch_ready():
                self.wake.clear()
                try:
                    await asyncio.wait_for(self.wake.wait(), timeout=1)
                except TimeoutError:
                    pass
            if not self.normal and not self.priority:
                continue
            batch = []
            while len(batch) < 256 and (self.normal or self.priority):
                queue = self.normal if self.normal and (not self.priority or len(batch) % 4 == 3) else self.priority
                batch.append(queue.popleft())
            self.inflight = batch
            try:
                # The SDK owns short retry within its timeout; Alloy owns downstream retry.
                success = self.exporter is None or await asyncio.to_thread(
                    self.exporter.export, tuple(item[0] for item in batch)) == SpanExportResult.SUCCESS
            except Exception:
                success = False
            if success:
                self.telemetry.processed.inc(len(batch))
                self.telemetry.processed_requests.inc(sum(span.name == "benchmark.attempt" for span, _ in batch))
                if self.exporter is not None:
                    self.telemetry.exported.inc(len(batch))
            if not success:
                self.telemetry.failures.inc()
                self.incomplete = True
                self.telemetry.dropped.labels("export_failed").inc(len(batch))
            self.count -= len(batch)
            self.bytes -= sum(item[1] for item in batch)
            self.inflight = []


class ResultObserver:
    def __init__(self, telemetry, traces, settings, *, replay=False):
        self.telemetry, self.traces = telemetry, traces
        self.builder = TraceBuilder(settings.slow_seconds, sample_rate=1.0 if settings.trace_mode == "debug" else settings.sample_rate,
                                    reconstruct=replay) if settings.trace_mode != "benchmark" else None
        self.position = (0, -1, 0)
        self.snapshot = LiveSnapshot()

    async def consume(self, page):
        if self.builder is not None:
            self.builder.formula_version = page.measurement_contract
        for change in page.updates:
            position = (change.revision, change.entity_kind, change.internal_record_id)
            if position <= self.position:
                continue
            self.position = position
            row = change.record
            self.snapshot.consume(change)
            if isinstance(row, MeasurementRecord) and row.name == "observation":
                if not isinstance(row.value, Mapping):
                    raise TypeError("observation measurement must be a mapping")
                if self.builder is not None:
                    self.builder.clock = ClockAnchor(**row.value["clock"])
                self.telemetry.available.set(self.builder is not None)
            elif isinstance(row, (RunRecord, TrialResultRecord)):
                kind = "runs" if isinstance(row, RunRecord) else "trials"
                trial = change.trial_id or ""
                for status in ("pending", "preparing", "running", "succeeded", "failed", "cancelled"):
                    self.telemetry.status.labels(change.run_id, trial, kind, status).set(status == row.status)
                if isinstance(row, TrialResultRecord) and row.status in {"succeeded", "failed", "cancelled"}:
                    self.telemetry.retire.setdefault(trial, None)
            elif isinstance(row, SessionResultRecord) and self.builder is not None and self.traces is not None:
                if change.trial_id is None:
                    raise ValueError("session result requires Trial identity")
                trace = self.builder.session(row, change.run_id, change.trial_id)
                if trace:
                    self.traces.offer(*trace)
                    await asyncio.sleep(0)
            elif isinstance(row, (TurnResultRecord, AttemptResultRecord)):
                if change.trial_id is None:
                    raise ValueError("request result requires Trial identity")
                scope = "turn" if isinstance(row, TurnResultRecord) else "attempt"
                labels = (change.run_id, change.trial_id, scope, row.status)
                self.telemetry.groups.add(labels)
                self.telemetry.requests.labels(*labels).inc()
                trace = self.builder.build(row, change.run_id, change.trial_id) if self.builder is not None else None
                if self.builder is not None and self.builder.clock is None:
                    self.telemetry.trace_missing.labels("clock_not_recorded").inc()
                accepted = trace and self.traces is not None and self.traces.offer(*trace)
                context = trace[0].context if accepted and self.traces.exporter is not None else None
                exemplar = {"trace_id": context.trace_id.to_bytes(16).hex()} if context is not None else None
                for name, histogram in self.telemetry.histograms.items():
                    metric = row.metrics.get(name)
                    value = metric.statistics.get("value") if metric is not None else None
                    unit = "s" if name in {"ttft", "e2e"} else "ms"
                    if metric is not None and metric.availability == "available" and metric.unit == unit and _number(value):
                        histogram.labels(*labels).observe(value if unit == "s" else value / 1000, exemplar=exemplar)
                    else:
                        self.telemetry.missing.labels(*labels, name).inc()
                if trace:
                    # A committed page can contain more spans than the byte budget;
                    # let the sender run between records without awaiting network I/O.
                    await asyncio.sleep(0)
            elif isinstance(row, MetricPoint) and row.scope in {"run", "trial", "comparison"} and not row.window.startswith("timeline:"):
                labels = (change.run_id, row.scope, row.scope_id, row.window, row.metric, row.statistic, row.unit)
                valid = row.availability == "available" and _number(row.value)
                self.telemetry.final_available.labels(*labels).set(valid)
                if valid:
                    self.telemetry.final.labels(*labels).set(row.value)
                else:
                    self.telemetry.final.remove(*labels)


def sampled_output_span(packet, clock):
    """Decode a bounded diagnostic message; never infer formal request success."""
    from hashlib import sha256
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import Event, ReadableSpan
    from opentelemetry.trace import SpanContext, SpanKind, TraceFlags
    if clock is None:
        raise ValueError('output trace clock not observed')
    trace = TraceContext(**packet['trace'])
    stamps = packet['timestamps_ns']
    count = packet['output_event_count']
    if (not trace.sampled or not isinstance(stamps, (tuple, list)) or not stamps or len(stamps) > 512
            or any(type(t) is not int or t < 0 for t in stamps) or list(stamps) != sorted(stamps)
            or type(count) is not int or count < len(stamps)
            or type(packet['truncated']) is not bool or packet['truncated'] != (len(stamps) < count)
            or type(packet['attempt']) is not int or packet['attempt'] < 1
            or any(not isinstance(packet[k], str) or not packet[k] or len(packet[k]) > 256 for k in ('run_id','turn_id'))):
        raise ValueError('invalid sampled output trace')
    identity = int.from_bytes(sha256((trace.span_id + ':output-events').encode()).digest()[:8], 'big') or 1
    context = SpanContext(int(trace.trace_id,16), identity, False, TraceFlags(1))
    parent = SpanContext(int(trace.trace_id,16), int(trace.span_id,16), False, TraceFlags(1))
    attributes = {'benchmark.run_id':packet['run_id'],'benchmark.turn_id':packet['turn_id'],
        'benchmark.attempt':packet['attempt'],'benchmark.output_event_count':count,
        'benchmark.captured_outputs':len(stamps),'benchmark.outputs_truncated':packet['truncated'],
        'benchmark.evidence':'pre_commit_diagnostic','benchmark.event_semantics':'output_event_not_token',
        'benchmark.output_event_limit':512,
        'sampling.reason':'head'}
    return ReadableSpan(name='benchmark.output_events',context=context,parent=parent,
        resource=Resource.create({'service.name':'perf-benchmark'}),kind=SpanKind.INTERNAL,
        attributes=attributes,start_time=clock.unix(stamps[0]),end_time=clock.unix(stamps[-1]),
        events=[Event('output.event',timestamp=clock.unix(t),attributes={'output.index':i+1}) for i,t in enumerate(stamps)])
