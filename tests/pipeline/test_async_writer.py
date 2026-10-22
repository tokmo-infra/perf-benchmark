import asyncio
from threading import Event
import time

import pytest

from perf_benchmark.pipeline.results.async_writer import (
    AsyncBatchWriter,
    BatchWriterOverloaded,
)


async def test_flush_waits_until_batch_results_are_published() -> None:
    published = []
    publishing = Event()
    release = Event()

    class Backend:
        def open(self):
            pass

        def write_batch(self, items):
            return tuple(items)

        def close(self):
            pass

    def publish(result):
        publishing.set()
        release.wait()
        published.extend(result)

    writer = AsyncBatchWriter(Backend(), capacity=2, publish=publish)
    await writer.start()
    writer.submit("completed turn")
    flushing = asyncio.create_task(writer.flush())
    try:
        assert await asyncio.to_thread(publishing.wait, 1)
        assert not flushing.done()
        release.set()
        await asyncio.wait_for(flushing, 1)
        assert published == ["completed turn"]
    finally:
        release.set()
        await flushing
        await writer.close()


async def test_batch_writer_keeps_database_work_off_event_loop() -> None:
    class Backend:
        def __init__(self):
            self.items = []

        def open(self):
            pass

        def write_batch(self, items):
            time.sleep(0.05)
            self.items.extend(items)

        def close(self):
            pass

    backend = Backend()
    writer = AsyncBatchWriter(backend, capacity=2)
    await writer.start()
    writer.submit("row")
    progressed = False

    async def ticker():
        nonlocal progressed
        await asyncio.sleep(0.01)
        progressed = True

    await asyncio.gather(writer.flush(), ticker())
    await writer.close()

    assert progressed
    assert backend.items == ["row"]


async def test_batch_writer_never_waits_when_queue_is_full() -> None:
    class Backend:
        def open(self):
            pass

        def write_batch(self, items):
            pass

        def close(self):
            pass

    writer = AsyncBatchWriter(Backend(), capacity=1)
    await writer.start()
    writer.submit("accepted")
    with pytest.raises(BatchWriterOverloaded, match="measurement is invalid"):
        writer.submit("rejected")
    with pytest.raises(BatchWriterOverloaded):
        await writer.close()


async def test_batch_writer_closes_while_worker_is_idle() -> None:
    class Backend:
        def open(self):
            pass

        def write_batch(self, items):
            pass

        def close(self):
            pass

    writer = AsyncBatchWriter(Backend(), capacity=2)
    await writer.start()
    await asyncio.wait_for(writer.close(), timeout=1)

    assert writer.closed
    with pytest.raises(RuntimeError, match="not running"):
        await asyncio.wait_for(writer.flush(), 1)


async def test_overload_counts_records_and_drains_already_accepted_facts() -> None:
    stored, failures = [], []

    class Backend:
        def open(self):
            pass

        def write_batch(self, items):
            stored.extend(items)

        def close(self):
            pass

    writer = AsyncBatchWriter(Backend(), capacity=3, on_error=failures.append)
    await writer.start()
    writer.submit_many((1, 2))
    with pytest.raises(BatchWriterOverloaded):
        writer.submit_many((3, 4))
    with pytest.raises(BatchWriterOverloaded):
        await writer.flush()
    with pytest.raises(BatchWriterOverloaded):
        await writer.close()
    assert stored == [1, 2]
    assert len(failures) == 1


@pytest.mark.parametrize("failure", ["backend", "publish"])
async def test_writer_failure_reaches_flush_and_close_without_hanging(failure):
    stored, closed, failures = [], [], []

    class Backend:
        def open(self):
            pass

        def write_batch(self, items):
            if failure == "backend":
                raise ValueError("backend failed")
            stored.extend(items)
            return items

        def close(self):
            closed.append(True)

    def publish(items):
        raise ValueError("publish failed")

    writer = AsyncBatchWriter(Backend(), publish=publish, on_error=failures.append)
    await writer.start()
    writer.submit_many((1, 2))
    with pytest.raises(ValueError, match=f"{failure} failed"):
        await asyncio.wait_for(writer.flush(), 1)
    with pytest.raises(ValueError, match=f"{failure} failed"):
        await asyncio.wait_for(writer.close(), 1)
    assert closed == [True]
    assert len(failures) == 1
    assert stored == ([] if failure == "backend" else [1, 2])


async def test_flush_finishes_when_partial_batch_reaches_its_deadline():
    stored = []

    class Backend:
        def open(self):
            pass

        def write_batch(self, items):
            stored.extend(items)

        def close(self):
            pass

    writer = AsyncBatchWriter(Backend())
    await writer.start()
    try:
        writer.submit("accepted fact")
        await asyncio.wait_for(writer.flush(), 0.5)
        assert stored == ["accepted fact"]
    finally:
        await writer.close()


async def test_public_diagnostic_source_is_detached_immutable_snapshot():
    from dataclasses import FrozenInstanceError
    from perf_benchmark.pipeline.contracts import WriterSnapshot

    class Backend:
        def open(self):
            pass

        def write_batch(self, records):
            pass

        def close(self):
            pass

    writer = AsyncBatchWriter(Backend())
    source = writer.enable_diagnostics()
    before = source()
    assert isinstance(before, WriterSnapshot)
    with pytest.raises(FrozenInstanceError):
        setattr(before, "processed", 99)
    await writer.start()
    try:
        writer.submit("fact")
        await writer.flush()
        after = source()
        assert isinstance(after, WriterSnapshot)
        assert before.processed == 0 and after.processed == 1
        assert after.queued == after.processing == 0
        assert after.batch_seconds >= 0
    finally:
        await writer.close()


@pytest.mark.parametrize("outcome", ["commit", "failure", "overload", "close"])
async def test_blocked_admissions_wake_on_commit_failure_overload_and_close(outcome):
    from contextlib import suppress

    entered, release = Event(), Event()
    closed, stored = [], []

    class Backend:
        def open(self):
            pass

        def write_batch(self, records):
            entered.set()
            assert release.wait(3)
            if outcome == "failure":
                raise ValueError("transaction failed")
            stored.extend(records)

        def close(self):
            closed.append(True)

    writer = AsyncBatchWriter(Backend(), capacity=200)
    await writer.start()
    waiting, closing = [], None
    try:
        writer.submit_many(tuple(range(200)))
        assert await asyncio.to_thread(entered.wait, 1)
        assert writer.queue.unfinished_tasks == 200
        waiting = [asyncio.create_task(writer.wait_for_headroom()) for _ in range(32)]
        await asyncio.sleep(0)
        assert not any(task.done() for task in waiting)
        cancelled = waiting.pop()
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        if outcome == "commit":
            release.set()
            expected = None
        elif outcome == "failure":
            release.set()
            expected = ValueError
        elif outcome == "overload":
            with pytest.raises(BatchWriterOverloaded):
                writer.submit("overflow")
            expected = BatchWriterOverloaded
        else:
            closing = asyncio.create_task(writer.close())
            expected = RuntimeError
        outcomes = await asyncio.wait_for(asyncio.gather(*waiting, return_exceptions=True), 2)
        assert all(value is None if expected is None else isinstance(value, expected) for value in outcomes)
    finally:
        release.set()
        with suppress(ValueError, BatchWriterOverloaded):
            await asyncio.wait_for(closing if closing is not None else writer.close(), 2)
    assert closed == [True]
    assert writer.queue.unfinished_tasks == 0
    if outcome != "failure":
        assert stored == list(range(200))
