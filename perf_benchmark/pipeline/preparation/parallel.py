"""Bounded, ordered process workers for independent source records."""

import asyncio
from collections import deque
import os

import multiprocessing as multiprocess
from concurrent.futures import ProcessPoolExecutor
import dill

from perf_benchmark.pipeline.resources import finish_task


def _normalize(plugin, batch):
    return list(dill.loads(plugin).normalize_batch(batch))


async def normalize_batches(plugin, source, ctx):
    workers = int(os.environ.get("PERF_BENCHMARK_PREPARATION_WORKERS", "4"))
    if workers <= 0:
        raise ValueError("PERF_BENCHMARK_PREPARATION_WORKERS must be positive")
    plugin = dill.dumps(plugin)
    pool = ProcessPoolExecutor(workers, mp_context=multiprocess.get_context("spawn"))
    pending = deque()

    async def consume():
        batch, result = pending.popleft()
        while not result.done():
            if ctx.cancellation.is_cancelled():
                return None
            await asyncio.sleep(0.02)
        return batch, result.result()

    try:
        async for batch in source.batches():
            for index in range(batch.num_rows):
                if ctx.cancellation.is_cancelled():
                    return
                record = batch.slice(index, 1)
                pending.append((record, pool.submit(_normalize, plugin, record)))
                if len(pending) == workers:
                    value = await consume()
                    if value is None:
                        return
                    yield value
        while pending and not ctx.cancellation.is_cancelled():
            value = await consume()
            if value is None:
                return
            yield value
    finally:
        for process in (pool._processes or {}).values():
            process.terminate()
        await finish_task(asyncio.create_task(asyncio.to_thread(pool.shutdown, cancel_futures=True)))
