from collections.abc import Mapping
from typing import Any

from perf_benchmark.pipeline.contracts.normalization import (
    RecordNormalizationPlugin,
    TurnDraft,
    parse_record_settings,
    required_text,
    token_texts,
)
from perf_benchmark.plugins import register_plugin


def parse_settings(settings: dict[str, Any]) -> dict[str, Any]:
    parsed = parse_record_settings(
        settings,
        plugin="longbench_v1",
        extra=("output_tokens",),
    )
    output_tokens = parsed.get("output_tokens")
    if output_tokens is not None and (
        isinstance(output_tokens, bool)
        or not isinstance(output_tokens, int)
        or output_tokens <= 0
    ):
        raise ValueError("longbench_v1 output_tokens must be positive")
    return parsed


@register_plugin(
    stage="normalization",
    name="longbench_v1",
    description="Normalize LongBench v1 question rows",
    settings_parser=parse_settings,
)
class LongbenchV1Normalization(RecordNormalizationPlugin):
    def normalize_record(self, record: Mapping[str, Any], source_order: int):
        question = required_text(record, "question")
        output_tokens = self.settings.get("output_tokens")
        identifier = str(record.get("id") or f"longbench_v1-{source_order + 1}")
        yield TurnDraft(identifier, f"{identifier}:turn-1", 0,
                        {"messages": [{"role": "user", "content": question}]},
                        token_texts(question), None, requested_output_tokens=output_tokens)


__all__ = ["LongbenchV1Normalization"]
