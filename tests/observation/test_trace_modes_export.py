"""Trace modes preserve formal facts and bound optional diagnostic delivery."""

import asyncio
from dataclasses import asdict, replace
from threading import Event

import pytest

pytest.importorskip("opentelemetry.sdk")
pytest.importorskip("prometheus_client")
from opentelemetry.sdk.trace.export import SpanExportResult
from prometheus_client import CollectorRegistry

from perf_benchmark.core.models import RequestMeasurements
from perf_benchmark.core.observation import ClockAnchor, ObservationSettings, TraceContext
from perf_benchmark.integrations.observation_export import ResultObserver, SpanQueue, Telemetry, TraceBuilder
from perf_benchmark.pipeline.contracts.results import (
    AttemptResultRecord, MeasurementRecord, ResultUpdate, ResultUpdateCursor, ResultUpdatePage,
    ScalarMetricRecord, SessionResultRecord, TurnResultRecord,
)


CLOCK = ClockAnchor(0, 1_000_000_000_000, 3)


def attempt(*, number=1, context=None):
    request = RequestMeasurements(attempts=number, admitted_ns=10, started_ns=20, first_output_ns=30,
                                  last_output_ns=40, response_completed_ns=50, result_ready_ns=60,
                                  trace_context=context)
    return AttemptResultRecord("turn", "session", number, "succeeded", None, None,
                               asdict(request), {"prompt_tokens": 7, "completion_tokens": 2},
                               {"e2e": ScalarMetricRecord(unit="s", statistics={"value": .123},
                                                         availability="available", unavailable_reason=None,
                                                         source="transport", formula_version="test")},
                               None, None, None, None, {})


def turn(request):
    return TurnResultRecord(id="turn", ordinal=0, status="succeeded", started_at=None, finished_at=None,
                            error=None, measurements={"lifecycle": {"started_ns": 0, "completed_ns": 70},
                                                      "total_started_ns": 0, "request": request.measurements},
                            metrics=request.metrics, metadata={"prompt": "PRIVATE"}, session_id="session",
                            estimated_input_tokens=None, reference_output_tokens=None,
                            requested_output_tokens=None, usage=request.usage)


def page(*rows):
    updates = tuple(ResultUpdate(i + 1, 0, i + 1, (str(i),), row, "run", "trial")
                    for i, row in enumerate(rows))
    return ResultUpdatePage(updates, ResultUpdateCursor("db", len(rows)), len(rows), True, "test")


def builder(*, rate=1, replay=True):
    result = TraceBuilder(10, sample_rate=rate, reconstruct=replay)
    result.clock = CLOCK
    return result


async def test_benchmark_replay_reconstructs_original_ids_without_claiming_propagation():
    registry = CollectorRegistry()
    queue = SpanQueue(None, Telemetry(registry))
    observer = ResultObserver(queue.telemetry, queue, ObservationSettings(trace_mode="debug", trace_export=False),
                              replay=True)
    measurement = MeasurementRecord("run", "run", "run", "observation",
                                    {"clock": asdict(CLOCK), "settings": {"trace_mode": "benchmark"}})
    record = attempt()
    await observer.consume(page(measurement, record, turn(record)))
    spans = [item[0] for item in queue.normal]
    expected = TraceContext.create("run", "turn", 1, 1, seed=42, session_ordinal=5, turn_ordinal=8)
    assert len(spans) == 2
    assert spans[0].context.trace_id == int(expected.trace_id, 16)
    assert spans[0].context.span_id == int(expected.span_id, 16)
    assert spans[0].parent.span_id == spans[1].context.span_id == int(expected.parent_span_id, 16)
    assert spans[0].start_time == CLOCK.unix(20)
    assert spans[0].attributes["benchmark.metric.e2e.value"] == .123
    for span in spans:
        assert span.attributes["benchmark.trace_origin"] == "offline_reconstructed"
        assert span.attributes["benchmark.trace_propagated"] is False
        assert "PRIVATE" not in span.to_json()
    assert registry.get_sample_value("perf_benchmark_trace_available") == 1
    queue.close()
    await queue.run()
    assert queue.count == queue.bytes == 0
    assert registry.get_sample_value("perf_benchmark_trace_processed_spans_total") == 2
    assert registry.get_sample_value("perf_benchmark_trace_exported_spans_total") == 0
    assert registry.get_sample_value("perf_benchmark_trace_processed_request_spans_total") == 1


def test_offline_retry_ids_and_selection_are_stable_without_live_context():
    trace = builder(rate=.01)
    assert trace._sampled("run", "turn") == trace._sampled("run", "turn")
    # Error/retry summaries remain available when head sampling excludes the workflow.
    trace = builder(rate=0)
    one = trace.build(replace(attempt(), status="failed"), "run", "trial")[0]
    two = trace.build(attempt(number=2), "run", "trial")[0]
    assert one.context.trace_id == two.context.trace_id
    assert one.context.span_id != two.context.span_id
    assert one.parent == two.parent
    assert two.attributes["sampling.server_details_may_be_missing"] is True


def test_debug_replay_keeps_unsampled_recorded_context_and_marks_missing_server_details():
    context = TraceContext.create("run", "turn", 1, 0, seed=0, session_ordinal=0, turn_ordinal=0)
    span = builder().build(attempt(context=context), "run", "trial")[0]
    assert span.context.trace_id == int(context.trace_id, 16)
    assert span.attributes["sampling.reason"] == "head"
    assert span.attributes["sampling.server_details_may_be_missing"] is True
    assert span.attributes["benchmark.trace_origin"] == "recorded_context"
    assert "benchmark.trace_propagated" not in span.attributes


def test_live_observer_never_invents_context_and_missing_clock_has_no_span():
    assert builder(replay=False).build(attempt(), "run", "trial") is None
    trace = builder()
    trace.clock = None
    assert trace.build(attempt(), "run", "trial") is None


async def test_missing_replay_clock_reports_unavailable_and_keeps_formal_metrics():
    registry = CollectorRegistry()
    telemetry = Telemetry(registry)
    queue = SpanQueue(None, telemetry)
    observer = ResultObserver(telemetry, queue, ObservationSettings(trace_mode="debug"), replay=True)
    await observer.consume(page(attempt()))
    assert queue.count == 0
    assert registry.get_sample_value("perf_benchmark_trace_available") == 0
    assert registry.get_sample_value("perf_benchmark_trace_unavailable_total", {"reason": "clock_not_recorded"}) == 1
    assert registry.get_sample_value("perf_benchmark_e2e_seconds_sum",
                                     {"run": "run", "trial": "trial", "scope": "attempt", "status": "succeeded"}) == .123


def test_session_workflow_link_uses_actual_lifecycle_and_does_not_collect_turns():
    session = SessionResultRecord(id="session", ordinal=0, status="succeeded", started_at=None, finished_at=None,
                                 error=None, measurements={"lifecycle": {"started_ns": 0, "completed_ns": 70}},
                                 metrics={}, metadata={"response": "PRIVATE"}, trial_id="trial", turn_count=1)
    trace = builder()
    workflow = trace.session(session, "run", "trial")[0]
    child = trace.build(turn(attempt()), "run", "trial")[0]
    assert child.links[0].context == workflow.context
    assert child.context.trace_id != workflow.context.trace_id
    assert workflow.start_time == CLOCK.unix(0)
    assert workflow.end_time == CLOCK.unix(70)
    assert workflow.attributes["benchmark.evidence"] == "committed_session_lifecycle"
    assert "PRIVATE" not in workflow.to_json()
    assert builder(rate=0).session(session, "run", "trial") is None


async def test_drain_timeout_counts_inflight_once_and_closed_queue_rejects_new_work():
    entered, release = Event(), Event()

    class Exporter:
        def export(self, spans):
            entered.set()
            release.wait(2)
            return SpanExportResult.SUCCESS

        def shutdown(self):
            release.set()

    registry = CollectorRegistry()
    queue = SpanQueue(Exporter(), Telemetry(registry), capacity=1)
    sample = builder().build(attempt(), "run", "trial")
    assert queue.offer(*sample)
    sender = asyncio.create_task(queue.run())
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        assert queue.count == 1 and len(queue.inflight) == 1
        assert not queue.offer(*sample)
        await queue.finish(sender, timeout=.01)
        assert sender.cancelled()
        assert queue.count == queue.bytes == 0 and queue.incomplete
        assert not queue.offer(*sample)
        assert registry.get_sample_value("perf_benchmark_trace_dropped_total", {"reason": "drain_timeout"}) == 1
        assert registry.get_sample_value("perf_benchmark_trace_dropped_total", {"reason": "sender_cancelled"}) is None
        assert registry.get_sample_value("perf_benchmark_trace_exported_spans_total") == 0
    finally:
        release.set()
        sender.cancel()
        await asyncio.gather(sender, return_exceptions=True)


async def test_sender_cancellation_accounts_for_waiting_spans():
    registry = CollectorRegistry()
    queue = SpanQueue(None, Telemetry(registry))
    assert queue.offer(*builder().build(attempt(), "run", "trial"))
    sender = asyncio.create_task(queue.run())
    await asyncio.sleep(0)
    sender.cancel()
    await asyncio.gather(sender, return_exceptions=True)
    assert queue.count == queue.bytes == 0 and queue.incomplete
    assert registry.get_sample_value("perf_benchmark_trace_dropped_total", {"reason": "sender_cancelled"}) == 1


async def test_cancelled_drain_accounts_for_pending_work_and_closes_exporter():
    entered, release, closed = Event(), Event(), Event()

    class Exporter:
        def export(self, spans):
            entered.set()
            release.wait(2)
            return SpanExportResult.SUCCESS

        def shutdown(self):
            closed.set()
            release.set()

    registry = CollectorRegistry()
    queue = SpanQueue(Exporter(), Telemetry(registry), capacity=1)
    assert queue.offer(*builder().build(attempt(), "run", "trial"))
    sender = asyncio.create_task(queue.run())
    drain = asyncio.create_task(queue.finish(sender))
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        drain.cancel()
        await asyncio.gather(drain, return_exceptions=True)
        assert drain.cancelled() and sender.cancelled() and closed.is_set()
        assert queue.count == queue.bytes == 0 and queue.incomplete
        assert registry.get_sample_value("perf_benchmark_trace_dropped_total", {"reason": "drain_cancelled"}) == 1
    finally:
        release.set()
        drain.cancel()
        sender.cancel()
        await asyncio.gather(drain, sender, return_exceptions=True)


async def test_exporter_shutdown_failure_is_diagnostic_and_preserves_confirmed_delivery():
    class Exporter:
        def export(self, spans):
            return SpanExportResult.SUCCESS

        def shutdown(self):
            raise RuntimeError("collector close failed")

    registry = CollectorRegistry()
    queue = SpanQueue(Exporter(), Telemetry(registry))
    assert queue.offer(*builder().build(attempt(), "run", "trial"))
    await queue.finish(asyncio.create_task(queue.run()))
    assert queue.count == queue.bytes == 0 and queue.incomplete
    assert registry.get_sample_value("perf_benchmark_trace_exported_spans_total") == 1
    assert registry.get_sample_value("perf_benchmark_trace_processed_spans_total") == 1
    assert registry.get_sample_value("perf_benchmark_trace_export_failures_total") == 1
