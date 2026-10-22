"""Override small request body parameters once, before each Trial starts."""

from collections.abc import Mapping
from copy import deepcopy

from perf_benchmark.core.lifecycle import TRIAL_BEFORE_START, LifecycleEvent
from perf_benchmark.pipeline.hooks import (
    HookContext, HookRegistry, HookSpec, require_request_parameters,
)


def _settings_schema(settings: dict) -> dict:
    if set(settings) != {"parameters"}:
        raise ValueError("request-parameters requires a parameters object")
    require_request_parameters(settings["parameters"])
    return settings


class RequestParametersHook:
    def __init__(self, settings: dict) -> None:
        self.parameters = deepcopy(settings["parameters"])

    async def handle(
        self, event: LifecycleEvent, payload: object, ctx: HookContext
    ) -> dict[str, object]:
        if event.name != TRIAL_BEFORE_START:
            raise ValueError("request-parameters must use trial.before_start")
        if not isinstance(payload, Mapping):
            raise TypeError("trial.before_start payload must be a mapping")
        return {
            **payload,
            "request_parameters": {
                **payload["request_parameters"], **deepcopy(self.parameters),
            },
        }


def register(registry: HookRegistry) -> None:
    registry.register(HookSpec(
        "request-parameters", RequestParametersHook,
        "Override small request body parameters before each Trial starts",
        settings_schema=_settings_schema,
    ))
