from perf_benchmark.core.validation import positive_int, positive_number
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
    return parse_sweep(settings, _validate, ['max_concurrency', 'request_rate'])


def _validate(settings):
    settings = parse_common_settings(
        settings,
        allowed={"request_rate", "max_pending_requests", "max_concurrency"},
        open_loop=True,
    )
    positive_number(settings.get("request_rate"), "request_rate")
    positive_int(settings.get("max_pending_requests"), "max_pending_requests")
    if cast(int | None, settings.get("max_concurrency")) is not None:
        positive_int(settings["max_concurrency"], "max_concurrency")
    return settings


@register_plugin(
    stage="scheduler",
    name="single-turn-open-loop",
    description="Rate-driven single-Turn Sessions with an optional concurrency cap",
    settings_parser=validate,
)
class SingleTurnOpenLoopScheduler(SchedulerBase):
    expected_turns = 1

    def create_strategy(self, plan: TrialPlan, started_ns: int, *, seed: int) -> StandardTurnStrategy:
        return StandardTurnStrategy(
            self.settings, plan, started_ns, seed=seed, reuse_session_arrival=True,
        )

    def trial_plan(self, settings):
        maximum = cast(int | None, settings.get("max_concurrency"))
        pending = cast(int, settings["max_pending_requests"])
        return TrialPlan(
            "request-rate",
            stop_plan(settings),
            ResourceLimits(
                maximum or pending,
                pending,
                pending,
            ),
            LoadProfile(
                "single-turn-open-loop",
                arrival_distribution=settings["arrival_distribution"],
                gamma_shape=settings.get("gamma_shape"),
                ramp_duration_s=settings.get("arrival_ramp", {}).get("duration_s"),
                ramp_end_multiplier=settings.get("arrival_ramp", {}).get("end_multiplier"),
                target_session_rate=float(cast(int | float, settings["request_rate"])),
                target_request_rate=float(cast(int | float, settings["request_rate"])),
                max_active_sessions=maximum,
                max_concurrency=maximum,
            ),
        )


__all__ = ["SingleTurnOpenLoopScheduler"]
