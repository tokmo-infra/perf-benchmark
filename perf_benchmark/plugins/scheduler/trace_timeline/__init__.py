"""Absolute source-time arrivals using the shared bounded scheduler."""

import asyncio
from time import perf_counter_ns

from perf_benchmark.core.validation import positive_int, positive_number
from perf_benchmark.pipeline.contracts import LoadProfile, ResourceLimits, SessionAdmission, StopPlan, TrialPlan, SchedulingStopped
from perf_benchmark.pipeline.scheduling.mechanics import SchedulerBase, StandardTurnStrategy, parse_common_settings
from perf_benchmark.plugins import register_plugin


def parse_settings(settings):
    parsed = parse_common_settings(settings, allowed={"max_concurrency", "max_pending_requests", "time_scale"})
    positive_int(parsed.get("max_concurrency"), "max_concurrency")
    positive_int(parsed.get("max_pending_requests"), "max_pending_requests")
    positive_number(parsed.setdefault("time_scale", 1.0), "time_scale")
    if "stop" in parsed or "retry" in settings:
        raise ValueError("trace-timeline replays a finite source once without retry or cycling")
    return parsed


class TimelineStrategy(StandardTurnStrategy):
    def start_clock(self, started_ns):
        super().start_clock(started_ns)
        self.origin_ns, self.source_origin, self.previous = started_ns, None, None

    async def offer_session(self, session):
        created = session.turns[0].source_created_at
        if created is None:
            raise ValueError("trace-timeline requires source timestamps")
        if self.previous is not None and created < self.previous:
            raise ValueError("trace-timeline requires chronological selection")
        self.previous = created
        if self.source_origin is None:
            self.source_origin = created
        elapsed = (created - self.source_origin).total_seconds() / self.settings["time_scale"]
        target = self.origin_ns + round(elapsed * 1e9)
        if self.attempts.failed.is_set():
            raise SchedulingStopped("attempt_pending_overflow")
        async def arrival():
            await asyncio.sleep(max(0, (target - perf_counter_ns()) / 1e9))
            return SessionAdmission(target, perf_counter_ns())
        return await self.wait_arrival(arrival())


@register_plugin(stage="scheduler", name="trace-timeline", settings_parser=parse_settings,
                 description="Replay independent request snapshots on an absolute trace timeline")
class TraceTimelineScheduler(SchedulerBase):
    expected_turns = 1

    def trial_plan(self, settings):
        concurrency, pending = settings["max_concurrency"], settings["max_pending_requests"]
        return TrialPlan("trace-timeline", StopPlan(), ResourceLimits(concurrency, pending, pending),
                         LoadProfile("trace-timeline", max_active_sessions=concurrency,
                                     max_concurrency=concurrency, arrival_distribution="timeline"),
                         metadata={"time_scale": settings["time_scale"], "trace_mode": "timeline"})

    def create_strategy(self, plan, started_ns, *, seed):
        return TimelineStrategy(self.settings, plan, started_ns, seed=seed, reuse_session_arrival=True)
