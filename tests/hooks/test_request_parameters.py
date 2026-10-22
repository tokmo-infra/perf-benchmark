from copy import deepcopy

import pytest

from perf_benchmark.core.lifecycle import RUN_BEFORE_START, TRIAL_BEFORE_START
from perf_benchmark.core.models import Run
from perf_benchmark.pipeline.hooks import HookContext, create_registry
from perf_benchmark.runtime.events import EventDispatcher


async def test_request_parameters_merge_once_per_trial_without_sharing_values(tmp_path):
    settings = {"parameters": {"top_p": 0.5, "thinking": {"enabled": False}}}
    original_settings = deepcopy(settings)
    run = Run(name="hook")
    dispatcher = EventDispatcher(
        run, create_registry(), {"events": {TRIAL_BEFORE_START: [
            {"hook": "request-parameters", "settings": settings},
            {"hook": "request-parameters", "settings": {"parameters": {"temperature": 0.1}}},
        ]}}, HookContext(run.id, run.name, tmp_path),
    )
    original = {
        "name": "trial", "metadata": {"tag": "kept"},
        "request_parameters": {"top_p": 0.8},
    }
    first = await dispatcher.emit(TRIAL_BEFORE_START, original)
    assert first == {**original, "request_parameters": {
        "top_p": 0.5, "thinking": {"enabled": False}, "temperature": 0.1,
    }}
    assert isinstance(first, dict)
    first["request_parameters"]["thinking"]["enabled"] = True
    second = await dispatcher.emit(TRIAL_BEFORE_START, original)
    assert isinstance(second, dict)
    assert second["request_parameters"]["thinking"] == {"enabled": False}
    assert original["request_parameters"] == {"top_p": 0.8}
    assert settings == original_settings
    await dispatcher.close()


@pytest.mark.parametrize("parameters", [
    None, [], {"messages": []}, {"headers": {"Authorization": "secret"}},
    {"extra_headers": {"X-Test": "x"}}, {"model": "other"}, {"stream": False},
    {"max_tokens": 10},
    {"api_key": "secret"}, {"api_keys": [{"env": "TARGET_KEY"}]},
    {"api_key_sampling": "weighted"}, {"input": "body"}, {"temperature": float("nan")},
    {"thinking": object()}, {"extra": "x" * 65536}, {1: "invalid key"},
])
def test_request_parameters_reject_invalid_configuration(parameters):
    with pytest.raises((TypeError, ValueError)):
        create_registry().create("request-parameters", {"parameters": parameters})


async def test_request_parameters_cannot_modify_run(tmp_path):
    run = Run(name="hook")
    dispatcher = EventDispatcher(
        run, create_registry(), {"events": {RUN_BEFORE_START: [
            {"hook": "request-parameters", "settings": {"parameters": {"temperature": 0.1}}},
        ]}}, HookContext(run.id, run.name, tmp_path),
    )
    try:
        with pytest.raises(ValueError, match="must use trial.before_start"):
            await dispatcher.emit(RUN_BEFORE_START, {"name": run.name, "metadata": {}})
    finally:
        await dispatcher.close()
