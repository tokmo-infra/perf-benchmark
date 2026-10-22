from perf_benchmark.pipeline.registry import StageRegistry

_catalog = StageRegistry()
register_plugin = _catalog.register


def create_registry() -> StageRegistry:
    """Load the explicit plugin catalog and return an isolated directory."""
    # These imports execute decorators, not plugin constructors. Keep them local
    # so register_plugin is available before an implementation imports it back.
    from perf_benchmark.plugins.executor import openai_chat  # noqa: F401
    from perf_benchmark.plugins.executor import openai_completions  # noqa: F401
    from perf_benchmark.plugins.executor import openai_embeddings, openai_responses as responses_executor, rerank as rerank_executor  # noqa: F401
    from perf_benchmark.plugins.exporter import console, html, tabular  # noqa: F401
    from perf_benchmark.plugins.loader import hf_datasets  # noqa: F401
    from perf_benchmark.plugins.metrics import standard  # noqa: F401
    from perf_benchmark.plugins.normalization import (  # noqa: F401
        embedding,
        gsm8k,
        longalpaca,
        longbench_v1,
        openai_messages,
        open_swe_traces,
        openai_responses,
        openqa,
        rerank,
        sharegpt,
        agentx,
        azure_2024,
        burstgpt,
        mooncake,
        qwen_bailian,
        servegen,
    )
    from perf_benchmark.plugins.saver import sqlite  # noqa: F401
    from perf_benchmark.plugins.scheduler import (  # noqa: F401
        multi_turn_closed_loop,
        multi_turn_open_loop,
        single_turn_closed_loop,
        single_turn_open_loop,
        trace_timeline,
        trace_causal,
    )
    from perf_benchmark.plugins.selector import all, created_turns, token_length  # noqa: F401
    from perf_benchmark.plugins.selector import trace_timeline as trace_selector  # noqa: F401

    from perf_benchmark.plugins.selector import trace_causal as causal_selector  # noqa: F401

    return _catalog.copy()


__all__ = ["create_registry", "register_plugin"]
