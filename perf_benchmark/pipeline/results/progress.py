"""Low-frequency previews from computed final Turns, outside request timing."""

import asyncio
from collections import Counter
import logging
import math
from threading import Lock
from time import monotonic

from perf_benchmark.core.validation import positive_number

logger = logging.getLogger("perf_benchmark.progress")
_METRICS = {"ttft": "s", "tpot": "ms", "e2e": "s", "decode_throughput": "tok/s"}


class ProgressLog:
    def __init__(self, settings=None):
        self.interval = positive_number((settings or {}).get("progress_interval_seconds", 30),
                                        "logging.progress_interval_seconds", allow_zero=True)
        self.enabled = bool(self.interval) and logger.isEnabledFor(logging.INFO)
        self.lock = Lock()
        self.stop = asyncio.Event()
        self.task = None
        self.run = self.trial = None
        self.load = lambda: {}
        self.counts = Counter()
        self.sums = Counter()
        self.samples = Counter()
        self.phase = "preparing"
        self.started = self.previous = monotonic()

    def start(self, run):
        self.run = run
        if self.enabled:
            self.emit("run.preparing")
            self.task = asyncio.create_task(self._periodic())

    def begin_trial(self, trial):
        if not self.enabled:
            return
        with self.lock:
            self.trial = trial
            self.counts.clear()
            self.sums.clear()
            self.samples.clear()
            self.started = self.previous = monotonic()
            self.phase = "running"
        self.emit("trial.started")

    def finish_trial(self, trial):
        if self.enabled and self.trial is trial:
            self.phase = "finalizing"
            self.emit("trial.finished")

    def consume(self, batch):
        if not self.enabled:
            return
        with self.lock:
            self.counts["sessions"] += len(batch.sessions)
            for turn in batch.turns:
                self.counts[turn.status] += 1
                if turn.status != "succeeded":
                    continue
                value = turn.usage.get("completion_tokens")
                if type(value) is int and value >= 0:
                    self.counts["interval_output"] += value
                else:
                    self.counts["interval_missing_usage"] += 1
                for name, unit in _METRICS.items():
                    item = turn.metrics.get(name)
                    value = item.statistics.get("value") if item is not None else None
                    if item is not None and item.availability == "available" and item.unit == unit and value is not None:
                        if name == "decode_throughput":
                            tokens = turn.usage.get("completion_tokens")
                            if type(tokens) is not int or tokens <= 1 or not math.isfinite(value) or value <= 0:
                                continue
                            seconds = (tokens - 1) / value
                            if not math.isfinite(seconds) or seconds <= 0:
                                continue
                            self.sums["decode_tokens"] += tokens - 1
                            self.sums["decode_seconds"] += seconds
                        else:
                            self.sums[name] += value
                        self.samples[name] += 1

    def emit(self, event):
        if not self.enabled:
            return
        at = monotonic()
        with self.lock:
            elapsed, interval = at - self.started, at - self.previous
            counts = self.counts.copy()
            samples = self.samples.copy()
            means = {name: f"{self.sums[name] / self.samples[name]:.3f}" if self.samples[name] else "unknown"
                     for name in _METRICS}
            means["decode_throughput"] = (f"{self.sums['decode_tokens'] / self.sums['decode_seconds']:.3f}"
                                          if self.sums["decode_seconds"] else "unknown")
            rate = (f"{counts['interval_output'] / interval:.3f}"
                    if interval > 0 and not counts["interval_missing_usage"] else "unknown")
            self.counts["interval_output"] = self.counts["interval_missing_usage"] = 0
            self.previous = at
        load = self.load()
        target = self.trial.parameters.get("stop", {}).get("session_count") if self.trial else None
        logger.info(
            "event=%s run=%s trial=%s phase=%s status=%s elapsed_s=%.1f "
            "sessions=%s/%s turns_ok=%s failed=%s cancelled=%s active_sessions=%s in_flight=%s "
            "completed_output_tok_s=%s interval_s=%.1f "
            "ttft_mean_s=%s tpot_mean_ms=%s e2e_mean_s=%s decode_time_weighted_tok_s=%s "
            "latency_samples=%s/%s/%s/%s preview=true",
            event, self.run.id, self.trial.id if self.trial else "pending", self.phase,
            self.trial.status if self.trial and not event.startswith("run.") else self.run.status, elapsed,
            counts["sessions"], target if target is not None else "unknown",
            counts["succeeded"], counts["failed"], counts["cancelled"],
            load.get("active_sessions", "unknown"), load.get("in_flight", "unknown"), rate, interval,
            *(means[name] for name in _METRICS), *(samples[name] for name in _METRICS),
        )

    async def _periodic(self):
        while not self.stop.is_set():
            try:
                await asyncio.wait_for(self.stop.wait(), self.interval)
            except TimeoutError:
                self.emit("progress")

    async def close(self):
        self.stop.set()
        if self.task is not None:
            await self.task
            self.task = None
            self.phase = "finished"
            self.emit("run.finished")
