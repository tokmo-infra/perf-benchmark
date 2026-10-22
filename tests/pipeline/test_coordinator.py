import pytest

from perf_benchmark.pipeline.coordinator import Coordinator
from perf_benchmark.pipeline.registry import StageRegistry
from perf_benchmark.plugins import create_registry


def _empty_plugin(settings):
    return object()


def test_builtin_registry_contains_all_stage_plugins() -> None:
    registry = create_registry()
    expected = {
        "loader": ("hf-datasets",),
        "normalization": (
            "agentx", "azure-2024", "burstgpt",
            "embedding",
            "gsm8k",
            "longalpaca",
            "longbench_v1",
            "mooncake",
            "open-swe-traces",
            "openai-messages",
            "openai-responses",
            "openqa",
            "qwen-bailian",
            "rerank",
            "servegen",
            "sharegpt",
        ),
        "selector": ("all", "created-turns", "token-length", "trace-causal", "trace-timeline"),
        "executor": ("openai-chat", "openai-completions", "openai-embeddings", "openai-responses", "rerank"),
        "scheduler": (
            "multi-turn-closed-loop", "multi-turn-open-loop",
            "single-turn-closed-loop", "single-turn-open-loop", "trace-causal", "trace-timeline",
        ),
        "metrics": ("standard",),
        "saver": ("sqlite",), "exporter": ("console", "html", "tabular"),
    }
    assert {stage: registry.names(stage) for stage in StageRegistry.STAGES} == expected


def test_plugin_replacement_is_explicit() -> None:
    registry = StageRegistry()
    def register(description, **values):
        return registry.register(
            stage="selector",
            name="sample",
            description=description,
            settings_parser=lambda value: value,
            **values,
        )
    register("first")(_empty_plugin)
    first = registry.resolve("selector", "sample")
    with pytest.raises(ValueError, match="already registered"):
        register("second")(_empty_plugin)
    assert registry.resolve("selector", "sample") is first
    register("second", replace=True)(_empty_plugin)
    assert registry.resolve("selector", "sample").description == "second"


async def test_loader_binding_is_required() -> None:
    with pytest.raises(ValueError, match="pipeline.loader.plugin is required"):
        await Coordinator().execute({"pipeline": {}})


async def test_removed_streaming_configuration_is_rejected() -> None:
    config = {
        "streaming": {"reader_batch_rows": 128},
        "pipeline": {
            stage: {"plugin": "fixture"}
            for stage in StageRegistry.STAGES
        },
    }
    with pytest.raises(ValueError, match="streaming configuration is not supported"):
        await Coordinator().execute(config)
