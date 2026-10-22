import asyncio
import os
from pathlib import Path
import time

import multiprocessing as multiprocess
import pyarrow as pa
import pytest

from perf_benchmark.pipeline.contracts.normalization import RecordNormalizationPlugin, TurnDraft
from perf_benchmark.pipeline.contracts.loader import SOURCE_ORDER_COLUMN
from perf_benchmark.pipeline.preparation.parallel import normalize_batches


class Records(RecordNormalizationPlugin):
    def prepare(self, ctx):
        self.directory = str(ctx.base_dir)

    def normalize_record(self, record, source_order):
        directory = Path(self.directory)
        (directory / f"worker-{os.getpid()}").touch()
        mode = self.settings.get("mode")
        if mode == "die":
            os._exit(7)
        if mode == "fail":
            raise ValueError(f"invalid record {source_order}")
        if mode == "cancel":
            time.sleep(30)
        deadline = time.monotonic() + 10
        while len(list(directory.glob("worker-*"))) < 4:
            if time.monotonic() > deadline:
                raise RuntimeError("four preparation workers did not start")
            time.sleep(0.01)
        yield TurnDraft(str(source_order), str(source_order), 0,
                        {"messages": [{"role": "user", "content": record["text"]}]},
                        1, None, metadata={"pid": os.getpid()})


class Source:
    async def batches(self):
        # A loader using one row per batch must still use four workers.
        for index in range(8):
            yield pa.record_batch({"text": [str(index)], SOURCE_ORDER_COLUMN: [index]})


async def test_four_processes_preserve_source_order_and_full_requests(stage_context, monkeypatch):
    monkeypatch.delenv("PERF_BENCHMARK_PREPARATION_WORKERS", raising=False)
    ctx = stage_context("normalization")
    plugin = Records({})
    plugin.prepare(ctx)
    children = {p.pid for p in multiprocess.active_children()}
    rows = [row async for row in normalize_batches(plugin, Source(), ctx)]
    assert [batch.column(SOURCE_ORDER_COLUMN)[0].as_py() for batch, _ in rows] == list(range(8))
    drafts = [draft for _, records in rows for draft in records]
    assert [draft.request["messages"][0]["content"] for draft in drafts] == [str(i) for i in range(8)]
    assert len({draft.metadata["pid"] for draft in drafts}) == 4
    assert os.getpid() not in {draft.metadata["pid"] for draft in drafts}
    assert {p.pid for p in multiprocess.active_children()} == children


@pytest.mark.parametrize("mode,exception,match", [
    ("fail", Exception, "invalid record 0"),
    ("die", RuntimeError, "terminated abruptly"),
])
async def test_worker_errors_stop_preparation_and_join_children(stage_context, mode, exception, match):
    ctx = stage_context("normalization")
    plugin = Records({"mode": mode})
    plugin.prepare(ctx)
    children = {p.pid for p in multiprocess.active_children()}
    with pytest.raises(exception, match=match):
        async for _ in normalize_batches(plugin, Source(), ctx):
            pass
    assert {p.pid for p in multiprocess.active_children()} == children


async def test_cancellation_terminates_busy_workers(stage_context):
    ctx = stage_context("normalization")
    plugin = Records({"mode": "cancel"})
    plugin.prepare(ctx)
    children = {p.pid for p in multiprocess.active_children()}
    async def consume():
        return [row async for row in normalize_batches(plugin, Source(), ctx)]
    task = asyncio.create_task(consume())
    deadline = time.monotonic() + 10
    while not list(ctx.base_dir.glob("worker-*")):
        assert time.monotonic() < deadline
        await asyncio.sleep(0.02)
    ctx.cancellation.cancel("test preparation cancellation")
    assert await asyncio.wait_for(task, 5) == []
    assert {p.pid for p in multiprocess.active_children()} == children


@pytest.mark.parametrize("workers", ["0", "-1", "invalid"])
async def test_invalid_worker_count_fails_before_spawning(stage_context, monkeypatch, workers):
    monkeypatch.setenv("PERF_BENCHMARK_PREPARATION_WORKERS", workers)
    with pytest.raises(ValueError):
        async for _ in normalize_batches(Records({}), Source(), stage_context("normalization")):
            pass
