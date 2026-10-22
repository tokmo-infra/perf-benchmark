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
    return parse_sweep(settings, _validate, ['max_active_sessions', 'max_concurrency', 'request_rate', 'session_rate'])


def _validate(settings):
    settings = parse_common_settings(
        settings,
        allowed={
            "session_rate",
            "request_rate",
            "max_pending_sessions",
            "max_pending_requests",
            "max_active_sessions",
            "max_concurrency",
        },
        open_loop=True,
    )
    for name in ("session_rate", "request_rate"):
        positive_number(settings.get(name), name)
    for name in ("max_pending_sessions", "max_pending_requests"):
        positive_int(settings.get(name), name)
    for name in ("max_active_sessions", "max_concurrency"):
        if settings.get(name) is not None:
            positive_int(settings[name], name)
    return settings


@register_plugin(
    stage="scheduler",
    name="multi-turn-open-loop",
    description="Independent Session and Request arrival clocks for multi-Turn Sessions",
    settings_parser=validate,
)
class MultiTurnOpenLoopScheduler(SchedulerBase):
    expected_turns = None

    def create_strategy(self, plan: TrialPlan, started_ns: int, *, seed: int) -> StandardTurnStrategy:
        return StandardTurnStrategy(
            self.settings, plan, started_ns, seed=seed, reuse_session_arrival=False,
        )

    def trial_plan(self, settings):
        pending_requests = cast(int, settings["max_pending_requests"])
        return TrialPlan(
            "multi-turn-open-loop",
            stop_plan(settings),
            ResourceLimits(
                cast(int | None, settings.get("max_concurrency")) or pending_requests,
                cast(int, settings["max_pending_sessions"]),
                pending_requests,
            ),
            LoadProfile(
                "multi-turn-open-loop",
                arrival_distribution=settings["arrival_distribution"],
                gamma_shape=settings.get("gamma_shape"),
                ramp_duration_s=settings.get("arrival_ramp", {}).get("duration_s"),
                ramp_end_multiplier=settings.get("arrival_ramp", {}).get("end_multiplier"),
                target_session_rate=float(cast(int | float, settings["session_rate"])),
                target_request_rate=float(cast(int | float, settings["request_rate"])),
                max_active_sessions=cast(int | None, settings.get("max_active_sessions")),
                max_concurrency=cast(int | None, settings.get("max_concurrency")),
            ),
        )


__all__ = ["MultiTurnOpenLoopScheduler"]
