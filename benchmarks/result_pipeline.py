"""Replay deterministic body-free facts through Metrics -> Saver, without HTTP.

Admission waits for both queues, so this measures drainable throughput rather than
proving HTTP concurrency capacity. The SQLite file is temporary diagnostic data.
"""

from __future__ import annotations

from perf_benchmark.pipeline.results.metrics import MetricsRuntime
from perf_benchmark.pipeline.results.saver import SaverRuntime


import argparse
import asyncio
from collections import Counter
from contextlib import AsyncExitStack
from dataclasses import fields
from heapq import merge
from hashlib import sha256
import json
import gc
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from time import perf_counter, process_time

from perf_benchmark.core.models import Run, Session, Trial, Turn, freeze_snapshot
from perf_benchmark.core.profiles import RequestProfile
from perf_benchmark.pipeline.preparation.payloads import PayloadRef
from perf_benchmark.pipeline.contracts.executor import AttemptPlan
from dataclasses import asdict
from perf_benchmark.pipeline.context import MetricsContext, SaverContext
from perf_benchmark.pipeline.results.measurement import AttemptFact, OccupancyFact, SessionFact, TurnFact
from perf_benchmark.plugins.metrics.standard import StandardMetrics
from perf_benchmark.plugins.saver.sqlite import SQLiteSaver
from perf_benchmark.runtime.cancellation import CancellationToken


def request_events(i, identity_order="random"):
    if identity_order not in ("random", "random-wide", "ordered", "prepared"):
        raise ValueError("unknown diagnostic identity order")
    """A body-free closed-loop request: 200 ms service, 5000 starts/s.

    Queue and active occupancies are absolute levels, as in the Engine. The
    finite fixture ramps up and drains; it does not claim HTTP qualification.
    """
    start = 1_000_000_000 + i * 200_000
    end = start + 200_000_000
    def identity(kind):
        if identity_order == "prepared":
            return f"trial:session-{i}" if kind == "session" else f"trial:turn-{i}-0"
        digest = sha256(f"{kind}:{i}".encode()).hexdigest()
        if identity_order == 'ordered':
            suffix = f"{i // 5:012x}" + digest[:20]
        else:
            suffix = digest[:32 if identity_order == 'random-wide' else 12]
        return kind + '-' + suffix
    session, turn = identity('session'), identity('turn')
    samples = tuple((j + 1, start + (j + 1) * 50_000_000, 8) for j in range(4))
    record = TurnFact(
        trial_id="trial", id=turn, session_id=session, ordinal=0, status="succeeded",
        started_at="2026-09-29T00:00:00+00:00", finished_at="2026-09-29T00:00:00.200000+00:00", error=None, estimated_input_tokens=None,
        reference_output_tokens=None, requested_output_tokens=4,
        usage=freeze_snapshot({"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12}), metadata=freeze_snapshot({"execution_fingerprint": "0" * 64}),
        profile=RequestProfile(body_bytes=4185, model="benchmark", stream=True, max_tokens=4,
                               context_mode="snapshot", cache_isolation="disabled"),
        response_samples=samples, target_ns=start, offered_ns=start, admitted_ns=start,
        turn_started_ns=start, turn_completed_ns=end, attempts=1, response_mode="stream",
        request_ns=start, first_ns=samples[0][1], last_ns=end, response_completed_ns=end,
        http_status=200, output_event_count=4, output_chars=32, finish_reason="stop",
    )
    return start, end, record, SessionFact(session, "trial", i, "succeeded", None, None,
                                         None, freeze_snapshot({}), 1, start, start, start, start, end, 1)


def events(count, identity_order="random"):
    """Chronological start/finish stream, including timed Session snapshots."""
    active = 0
    starts = ((1_000_000_000 + i * 200_000, 1, i) for i in range(count))
    finishes = ((1_200_000_000 + i * 200_000, 0, i) for i in range(count))
    for at, opening, i in merge(starts, finishes):
        start, end, record, session = request_events(i, identity_order)
        if opening:
            yield OccupancyFact("trial", "session_pending", at, 1)
            yield OccupancyFact("trial", "session_pending", at, 0)
            yield Session(id=session.id, trial_id="trial", ordinal=i, turns=[
                Turn(id=record.id, session_id=session.id, request_payload_ref=PayloadRef(0, 0))
            ])
            active += 1
            yield OccupancyFact("trial", "active_sessions", at, active)
            yield OccupancyFact("trial", "request_pending", at, 1)
            yield OccupancyFact("trial", "request_pending", at, 0)
            yield OccupancyFact("trial", "in_flight", at, active)
        else:
            active -= 1
            yield OccupancyFact("trial", "in_flight", at, active)
            yield AttemptFact(**{field.name: getattr(record, field.name) for field in fields(TurnFact)},
                              plan=freeze_snapshot(asdict(AttemptPlan())))
            yield record
            yield session
            yield OccupancyFact("trial", "active_sessions", at, active)


def facts(count):
    return (item for item in events(count) if not isinstance(item, Session))


async def replay(count, profile=False):
    # Build identical immutable facts before timing the consumers.
    records = tuple(events(count))
    kinds = dict(Counter(type(item).__name__ for item in records))
    token = CancellationToken()
    run = Run(id="run", name="result-pipeline-replay")
    trial = Trial(id="trial", name="replay")
    run.register_trial(trial)
    with TemporaryDirectory(prefix="result-pipeline-replay-") as directory:
        path = Path(directory) / "results.db"
        async with AsyncExitStack() as stack:
            saver = await SaverRuntime.create(SQLiteSaver({"path": str(path)}), run,
                                               SaverContext(Path(directory), token))
            stack.push_async_callback(saver.close)
            metrics = await MetricsRuntime.create(StandardMetrics({}), MetricsContext(token, 1),
                                                  publish_results=saver.write_results, wait_for_results=saver.wait_for_headroom)
            stack.push_async_callback(metrics.close)
            await saver.write_trial(trial)
            await saver.writer.flush()
            diagnostics = {}
            async with AsyncExitStack() as sampling:
                if profile:
                    from benchmarks.result_diagnostics import start, stop
                    profiler = await saver.writer.call(start, saver)

                    async def capture():
                        diagnostics.update(await saver.writer.call(stop, profiler, saver, after_error=True))
                    sampling.push_async_callback(capture)
                started, cpu = perf_counter(), process_time()
                peaks = {"metrics": 0, "saver": 0}
                for offset in range(0, len(records), 100):
                    await metrics.writer.wait_for_headroom()
                    await saver.writer.wait_for_headroom()
                    batch = records[offset:offset + 100]
                    for item in batch:
                        if isinstance(item, Session):
                            await saver.write_session(item)
                    metrics.writer.submit_many(tuple(item for item in batch if not isinstance(item, Session)))
                    for name, writer in (("metrics", metrics.writer), ("saver", saver.writer)):
                        peaks[name] = max(peaks[name], writer.queue.unfinished_tasks)
                submitted = perf_counter() - started
                await metrics.writer.flush()
                await metrics.writer.call(lambda: metrics.backend._flush_pending(force=True))
                consumed = perf_counter() - started
                await saver.writer.flush()
                elapsed, cpu = perf_counter() - started, process_time() - cpu
            persisted = await asyncio.to_thread(read_counts, path)
            assert persisted == {name: count for name in ("turns", "attempts", "sessions")}, persisted
        return {"requests": count, "facts": len(records) - count, "session_start_records": count, "fact_types": kinds,
                "seconds": elapsed, "cpu_seconds": cpu, "requests_s": count / elapsed,
                "facts_s": (len(records) - count) / elapsed,
                "submitted_seconds": submitted, "consumed_seconds": consumed, "committed_seconds": elapsed,
                "database_bytes": path.stat().st_size, "fixture_service_ms": 200, "sampled_queue_peaks": peaks,
                "persisted": persisted, "http_capacity_claim": False, "diagnostics": diagnostics}


def read_counts(path):
    with sqlite3.connect(path) as connection:
        return {table: connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                for table in ("turns", "attempts", "sessions")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=10_000)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.requests <= 0:
        parser.error("requests must be positive")
    gc.disable()  # Match ExecutionEngine.open; interpreter exit releases this diagnostic process.
    print(json.dumps(asyncio.run(replay(args.requests, args.profile)), sort_keys=True))
