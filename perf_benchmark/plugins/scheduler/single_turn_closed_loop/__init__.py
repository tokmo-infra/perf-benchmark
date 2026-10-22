from perf_benchmark.core.validation import positive_int
from typing import cast

from perf_benchmark.pipeline.contracts.facts import (
    ComparisonPoint,
    LoadProfile,
    ResourceLimits,
)
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
    return parse_sweep(settings, _validate, ['max_concurrency'])


def _validate(settings):
    settings = parse_common_settings(
        settings, allowed={"max_concurrency", "repetitions"}
    )
    positive_int(settings.get("max_concurrency"), "max_concurrency")
    return settings


@register_plugin(
    stage="scheduler",
    name="single-turn-closed-loop",
    description="Closed-loop single-Turn Sessions with bounded HTTP concurrency",
    settings_parser=validate,
)
class SingleTurnClosedLoopScheduler(SchedulerBase):
    expected_turns = 1

    def create_strategy(self, plan: TrialPlan, started_ns: int, *, seed: int) -> StandardTurnStrategy:
        return StandardTurnStrategy(
            self.settings, plan, started_ns, seed=seed, reuse_session_arrival=False,
        )

    def trial_plan(self, settings):
        maximum = cast(int, settings["max_concurrency"])
        point = f"max-concurrency-{maximum}"
        return TrialPlan(
            point, stop_plan(settings), ResourceLimits(maximum, maximum, maximum),
            LoadProfile("single-turn-closed-loop", target_concurrency=maximum,
                        max_active_sessions=maximum, max_concurrency=maximum),
            comparison=ComparisonPoint(point, "max_concurrency", maximum, "requests"),
        )


__all__ = ["SingleTurnClosedLoopScheduler"]
