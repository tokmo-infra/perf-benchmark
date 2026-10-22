"""External Attempt cancellation and scenario result assessment."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from threading import Lock
from time import perf_counter_ns

from perf_benchmark.pipeline.contracts.executor import (
    AttemptPlan,
    AttemptResult,
)


def assess_result(result: AttemptResult, plan: AttemptPlan) -> None:
    result.plan = plan
    if plan.expected_status is not None or plan.expected_http_status is not None:
        result.scenario_passed = (
            result.fatal_error is None
            and (plan.expected_status is None or result.status == plan.expected_status)
            and (
                plan.expected_http_status is None
                or result.request.http_status == plan.expected_http_status
            )
        )


class AttemptSignal:
    trigger_ns: int | None
    decision_ns: int | None
    applied_ns: int | None
    """A worker can trigger once; the owning event loop performs the operation."""

    def __init__(self, plan: AttemptPlan):
        self.plan = plan
        self.action, self.origin = plan.action, "scenario"
        self.loop = asyncio.get_running_loop()
        self.lock = Lock()
        self.trigger_ns = self.decision_ns = self.applied_ns = None
        self.reason: str | None = None
        self.callback = None
        self.started_callback: Callable[[int], None] | None = None
        self.closed = False

    def bind(self, callback) -> None:
        self.callback = callback
        if self.reason is not None:
            self._apply()

    def trigger(
        self,
        reason: str,
        at_ns: int | None = None,
        *,
        action: str | None = None,
    ) -> None:
        with self.lock:
            if self.closed or self.reason is not None:
                return
            if action is not None:
                self.action, self.origin = action, "client"
            self.reason = reason
            self.trigger_ns = at_ns if at_ns is not None else perf_counter_ns()
            self.decision_ns = perf_counter_ns()
        self.loop.call_soon_threadsafe(self._apply)

    def _apply(self) -> None:
        if not self.closed and self.callback is not None and self.applied_ns is None:
            self.applied_ns = perf_counter_ns()
            self.callback()

    def close(self) -> None:
        with self.lock:
            self.closed = True
