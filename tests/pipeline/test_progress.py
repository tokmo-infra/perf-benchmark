import asyncio
import logging
from types import SimpleNamespace

import pytest

from perf_benchmark.core.models import Run, Trial
from perf_benchmark.pipeline.contracts.results import ComputedResultBatch, ScalarMetricRecord
from perf_benchmark.pipeline.results.progress import ProgressLog


def metric(value, unit):
    return ScalarMetricRecord(unit, "available", None, "fixture", {"value": value})


def turn(status="succeeded", *, usage=None, metrics=None):
    return SimpleNamespace(status=status, usage=usage or {}, metrics=metrics or {})


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, "30"])
def test_invalid_interval_is_rejected(value):
    with pytest.raises(ValueError, match="progress_interval_seconds"):
        ProgressLog({"progress_interval_seconds": value})


async def test_preview_counts_only_final_turns_keeps_units_and_missing_usage(caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger="perf_benchmark.progress")
    clock = [100.0]
    monkeypatch.setattr("perf_benchmark.pipeline.results.progress.monotonic", lambda: clock[0])
    run = Run(name="preview")
    progress = ProgressLog()
    progress.start(run)
    trial = Trial(id="trial-1", parameters={"stop": {"session_count": 3}})
    progress.begin_trial(trial)
    progress.load = lambda: {"active_sessions": 2, "in_flight": 1}
    progress.consume(ComputedResultBatch(
        turns=(turn(usage={"completion_tokens": 120}, metrics={
            "ttft": metric(0.5, "s"), "tpot": metric(20, "ms"),
            "e2e": metric(3, "s"), "decode_throughput": metric(50, "tok/s")}),
            turn("failed"), turn("cancelled")),
        sessions=(object(),), attempts=(turn("failed"), turn("succeeded"))))
    clock[0] += 30
    progress.emit("progress")
    text = caplog.records[-1].getMessage()
    for expected in ("sessions=1/3", "turns_ok=1 failed=1 cancelled=1", "in_flight=1",
                     "completed_output_tok_s=4.000 interval_s=30.0", "ttft_mean_s=0.500",
                     "tpot_mean_ms=20.000", "e2e_mean_s=3.000", "decode_time_weighted_tok_s=50.000",
                     "latency_samples=1/1/1/1", "preview=true"):
        assert expected in text
    assert len(caplog.records) == 3  # Two lifecycle lines, one explicitly requested preview.
    progress.consume(ComputedResultBatch(turns=(turn(),)))
    clock[0] += 30
    progress.emit("progress")
    assert "completed_output_tok_s=unknown" in caplog.records[-1].getMessage()
    progress.finish_trial(trial)
    progress.begin_trial(Trial(id="trial-2", parameters={"stop": {"duration_s": 60}}))
    assert "sessions=0/unknown turns_ok=0 failed=0 cancelled=0" in caplog.records[-1].getMessage()
    assert "ttft_mean_s=unknown" in caplog.records[-1].getMessage()
    await progress.close()
    assert progress.task is None


async def test_decode_preview_weights_receive_time_and_resets_each_trial(caplog):
    caplog.set_level(logging.INFO, logger="perf_benchmark.progress")
    progress = ProgressLog()
    progress.start(Run(name="decode"))
    progress.begin_trial(Trial(id="normal-and-burst"))
    progress.consume(ComputedResultBatch(turns=(
        turn(usage={"completion_tokens": 101}, metrics={"decode_throughput": metric(25, "tok/s")}),
        turn(usage={"completion_tokens": 25}, metrics={"decode_throughput": metric(1_200_000, "tok/s")}),
        turn(metrics={"decode_throughput": metric(1_000_000, "tok/s")}),
        turn(usage={"completion_tokens": 1}, metrics={"decode_throughput": metric(1_000_000, "tok/s")}),
        turn(usage={"completion_tokens": 25}, metrics={"decode_throughput": metric(0, "tok/s")}),
        turn(usage={"completion_tokens": 25}, metrics={"decode_throughput": metric(1e-320, "tok/s")}),
        turn("failed", usage={"completion_tokens": 999},
             metrics={"decode_throughput": metric(1_000_000, "tok/s")}),
    )))
    progress.emit("progress")
    assert "decode_time_weighted_tok_s=31.000" in caplog.records[-1].getMessage()
    assert progress.sums["decode_tokens"] == 124
    assert progress.sums["decode_seconds"] == pytest.approx(4.00002)
    assert progress.samples["decode_throughput"] == 2
    progress.begin_trial(Trial(id="fresh"))
    assert "decode_time_weighted_tok_s=unknown" in caplog.records[-1].getMessage()
    assert progress.sums["decode_seconds"] == 0
    await progress.close()


async def test_periodic_timer_and_shutdown_are_bounded(caplog):
    caplog.set_level(logging.INFO, logger="perf_benchmark.progress")
    progress = ProgressLog({"progress_interval_seconds": 0.01})
    emitted = asyncio.Event()
    original = progress.emit

    def emit(event):
        original(event)
        if event == "progress":
            emitted.set()

    progress.emit = emit
    progress.start(Run(name="timer"))
    await asyncio.wait_for(emitted.wait(), 1)
    await progress.close()
    count = len(caplog.records)
    await asyncio.sleep(0.03)
    assert len(caplog.records) == count
    await progress.close()
    assert len(caplog.records) == count


@pytest.mark.parametrize("settings,level", [({"progress_interval_seconds": 0}, logging.INFO),
                                           ({}, logging.WARNING)])
async def test_disabled_progress_adds_no_timer_or_preview(caplog, settings, level):
    caplog.set_level(level, logger="perf_benchmark.progress")
    progress = ProgressLog(settings)
    progress.start(Run(name="disabled"))
    progress.consume(ComputedResultBatch(turns=(turn(),)))
    await progress.close()
    assert progress.task is None and not progress.counts and not caplog.records


async def test_multiturn_real_pipeline_emits_sparse_logs_and_preserves_results(tmp_path, tokenizer_path, monkeypatch):
    import io
    import json
    import sqlite3
    import sys
    from aiohttp import web
    from aiohttp.test_utils import TestServer
    from perf_benchmark.pipeline.coordinator import Coordinator

    requests = []

    async def complete(request):
        requests.append(await request.json())
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n')
        await response.write(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":2}}\n\n')
        await response.write(b'data: [DONE]\n\n')
        return response

    app = web.Application()
    app.router.add_post("/v1/chat/completions", complete)
    server = TestServer(app)
    await server.start_server()
    source, database = tmp_path/"input.jsonl", tmp_path/"run.db"
    source.write_text(json.dumps({"id": "conversation", "conversations": [
        {"from": "human", "value": "first"}, {"from": "gpt", "value": "answer"},
        {"from": "human", "value": "second"}, {"from": "gpt", "value": "answer"}]})+"\n")
    output = io.StringIO()
    monkeypatch.setattr(sys, "stderr", output)
    config = {"logging": {"progress_interval_seconds": 30}, "pipeline": {
        "loader": {"plugin": "hf-datasets", "settings": {"dataset": "json", "data_files": str(source), "split": "train"}},
        "normalization": {"plugin": "sharegpt", "settings": {"multi_turn": True, "tokenizer_path": str(tokenizer_path)}},
        "selector": {"plugin": "all", "settings": {}},
        "executor": {"plugin": "openai-chat", "settings": {"base_url": str(server.make_url("")), "model": "mock", "stream": True}},
        "scheduler": {"plugin": "multi-turn-closed-loop", "settings": {"max_concurrency": 1, "max_active_sessions": 1, "stop": {"session_count": 3}, "on_exhausted": "cycle"}},
        "metrics": {"plugin": "standard", "settings": {}},
        "saver": {"plugin": "sqlite", "settings": {"path": str(database)}}, "exporter": {}}}
    try:
        run = await Coordinator().execute(config, base_dir=tmp_path)
    finally:
        await server.close()
    lines = [line for line in output.getvalue().splitlines() if "perf_benchmark.progress" in line]
    assert run.status == "succeeded" and len(requests) == 6
    assert len(lines) == 4 and all("event=progress " not in line for line in lines)
    assert any("sessions=3/3 turns_ok=6 failed=0 cancelled=0" in line for line in lines)
    with sqlite3.connect(database) as db:
        assert db.execute("select count(*) from turns where status='succeeded'").fetchone()[0] == 6
    assert all(body["max_tokens"] == 4096 and body["temperature"] == 0.1 for body in requests)
