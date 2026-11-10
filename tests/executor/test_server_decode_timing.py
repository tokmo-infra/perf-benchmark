"""Keep engine generation time separate from buffered client-visible output."""
import json

from aiohttp import web
from aiohttp.test_utils import TestServer
import pytest

from perf_benchmark import _native
from tests.runtime.native_support import request


@pytest.mark.parametrize("stream", [True, False])
async def test_engine_timing_survives_terminal_usage_without_fake_output(stream):
    timing = {"generation_time_ms": 1000.0, "mean_itl_ms": 1000 / 30,
              "queue_time_ms": 20.0, "time_to_first_token_ms": 10.0}
    async def complete(req):
        if not stream:
            return web.json_response({"choices": [{"message": {"content": "ok"},
                "finish_reason": "stop"}], "usage": {"completion_tokens": 31},
                "metrics": timing})
        # Both tool deltas arrive in one TCP body. Their arrival span is not
        # a measurement of the server's one-second generation interval.
        events = [
            {"choices": [{"delta": {"tool_calls": [{"index": 0,
                "function": {"name": "f", "arguments": "{"}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0,
                "function": {"arguments": "}"}}]}, "finish_reason": "tool_calls"}]},
            {"choices": [], "usage": {"completion_tokens": 31}, "metrics": timing},
        ]
        return web.Response(text="".join("data: " + json.dumps(x) + "\n\n" for x in events)
                            + "data: [DONE]\n\n", content_type="text/event-stream")
    app = web.Application()
    app.router.add_post("/", complete)
    async with TestServer(app) as server:
        native = _native.create_trial("openai-chat")
        try:
            result = await request(native, {"url": str(server.make_url("/")), "stream": stream})
            assert result.status == "succeeded"
            assert result.server_metrics == timing
            assert result.usage == {"completion_tokens": 31}
            if stream:
                assert len(result.samples) == 2  # No output event for the metrics-only frame.
                assert result.samples[0][1] == result.samples[-1][1]
            else:
                assert len(result.samples) == 0
            assert 30 * 1000 / result.server_metrics["generation_time_ms"] == 30
        finally:
            await native.close()


async def test_absent_or_invalid_server_metrics_are_not_zero():
    async def complete(req):
        return web.json_response({"choices": [{"message": {"content": "ok"}}],
            "metrics": {"generation_time_ms": -1, "mean_itl_ms": None,
                        "queue_time_ms": "1000", "tokens_per_second": 999999}})
    app = web.Application()
    app.router.add_post("/", complete)
    async with TestServer(app) as server:
        native = _native.create_trial("openai-chat")
        try:
            result = await request(native, {"url": str(server.make_url("/"))})
            assert result.status == "succeeded"
            assert result.server_metrics is None
        finally:
            await native.close()


async def test_one_buffered_tool_event_does_not_gain_fake_timing_samples():
    async def complete(req):
        events = [
            {"choices": [{"delta": {"tool_calls": [{"index": 0,
                "function": {"name": "run", "arguments": "{}"}}]},
                "finish_reason": "tool_calls"}]},
            {"choices": [], "usage": {"completion_tokens": 31},
             "metrics": {"generation_time_ms": 1000.0}},
        ]
        return web.Response(text="".join("data: " + json.dumps(x) + "\n\n" for x in events)
                            + "data: [DONE]\n\n", content_type="text/event-stream")
    app = web.Application()
    app.router.add_post("/", complete)
    async with TestServer(app) as server:
        native = _native.create_trial("openai-chat")
        try:
            result = await request(native, {"url": str(server.make_url("/")), "stream": True})
            assert result.status == "succeeded"
            assert len(result.samples) == 1
            assert result.finish_reason == "tool_calls"
            assert result.server_metrics == {"generation_time_ms": 1000.0}
        finally:
            await native.close()


async def test_engine_timing_roundtrips_metric_pipeline_and_sqlite(tmp_path, stage_context):
    from perf_benchmark.core.models import Run, Trial, Session, Turn
    from perf_benchmark.pipeline.preparation.payloads import PayloadRef
    from perf_benchmark.pipeline.results.metrics import MetricsRuntime
    from perf_benchmark.pipeline.results.saver import SaverRuntime
    from perf_benchmark.plugins.metrics.standard import StandardMetrics
    from perf_benchmark.plugins.saver.sqlite import SQLiteResultReader, SQLiteSaver

    run = Run(name="server-clock")
    trial = Trial(id="timing-trial", run_id=run.id)
    session = Session(id="timing-session", trial_id=trial.id, turns=[])
    turn = Turn(id="timing-turn", session_id=session.id, request_payload_ref=PayloadRef(0, 1))
    turn.start()
    turn.measurements.request.started_ns = 1_000_000_000
    turn.measurements.request.response_completed_ns = 2_000_000_000
    timing = {"generation_time_ms": 1000.0, "mean_itl_ms": 1000 / 30}
    turn.measurements.request.server_metrics = timing
    turn.succeed()
    database = tmp_path / "native.db"
    sink = await SaverRuntime.create(SQLiteSaver({"path": str(database)}), run, stage_context("saver", run=run))
    metrics = await MetricsRuntime.create(StandardMetrics({}), stage_context("metrics", run=run),
        publish_results=sink.write_results, wait_for_results=sink.wait_for_headroom)
    try:
        await sink.write_trial(trial)
        await sink.write_session(session)
        collector = await metrics.open_trial(trial)
        collector.record_turn(turn)
        # Snapshot facts must not retain a mutable server-response dictionary.
        timing["generation_time_ms"] = 99999
        await metrics.close()
        await sink.flush()
        result = tuple(SQLiteResultReader(database).iter_turn_records(run.id))[0]
        assert result["measurements"]["request"]["server_metrics"]["generation_time_ms"] == 1000
    finally:
        await metrics.close()
        await sink.close()
