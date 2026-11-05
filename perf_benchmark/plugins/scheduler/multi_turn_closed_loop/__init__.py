from perf_benchmark.core.validation import positive_int
from typing import cast

from perf_benchmark.pipeline.contracts.facts import LoadProfile, ResourceLimits
from perf_benchmark.pipeline.contracts.scheduler import TrialPlan
from perf_benchmark.pipeline.scheduling.mechanics import (
    SchedulerBase,
    StandardTurnStrategy,
    parse_common_settings,
    stop_plan,
)
from perf_benchmark.plugins import register_plugin
from perf_benchmark.pipeline.scheduling.sweep import parse_sweep


def validate(settings):
    if not isinstance(settings, dict):
        raise ValueError("scheduler settings must be an object")
    if settings.get("stop_on_underfill") and "sweep" in settings:
        raise ValueError("stop_on_underfill requires a single finite Trial, not sweep")
    return parse_sweep(settings, _validate, ['max_active_sessions', 'max_concurrency'])


def _validate(settings):
    settings = parse_common_settings(
        settings, allowed={"max_active_sessions", "max_concurrency", "stop_on_underfill"}
    )
    active = positive_int(settings.get("max_active_sessions"), "max_active_sessions")
    maximum = positive_int(settings.get("max_concurrency"), "max_concurrency")
    if active < maximum:
        raise ValueError("max_active_sessions must be greater than or equal to max_concurrency")
    enabled = settings.get("stop_on_underfill", False)
    if not isinstance(enabled, bool):
        raise ValueError("stop_on_underfill must be boolean")
    if enabled:
        if (settings["retry"]["max_attempts"] != 1 or "cancel" in settings
                or settings["repetitions"] != 1
                or set(settings["control"]) & {"after_outputs", "after_first_output_ms"}):
            raise ValueError("stop_on_underfill forbids retry, cancellation scenarios and repetitions")
        stop = stop_plan(settings)
        if stop.duration_s is not None or stop.on_exhausted == "cycle":
            raise ValueError("stop_on_underfill requires one finite first pass")
    return settings


@register_plugin(
    stage="scheduler",
    name="multi-turn-closed-loop",
    description="Closed-loop multi-Turn Sessions with separate Session and HTTP limits",
    settings_parser=validate,
)
class MultiTurnClosedLoopScheduler(SchedulerBase):
    expected_turns = None

    def create_strategy(self, plan: TrialPlan, started_ns: int, *, seed: int) -> StandardTurnStrategy:
        return StandardTurnStrategy(
            self.settings, plan, started_ns, seed=seed, reuse_session_arrival=False,
        )

    def trial_plan(self, settings):
        active = cast(int, settings["max_active_sessions"])
        maximum = cast(int, settings["max_concurrency"])
        return TrialPlan(
            "multi-turn-closed-loop",
            stop_plan(settings),
            ResourceLimits(maximum, active, active),
            LoadProfile(
                "multi-turn-closed-loop",
                target_concurrency=maximum,
                max_active_sessions=active,
                max_concurrency=maximum,
            ),
            stop_on_underfill=settings.get("stop_on_underfill", False),
        )


__all__ = ["MultiTurnClosedLoopScheduler"]
