from collections.abc import Mapping
from typing import Any

from perf_benchmark.pipeline.contracts.normalization import (
    RecordNormalizationPlugin,
    TurnDraft,
    optional_text,
    parse_record_settings,
    required_text,
    token_texts,
)
from perf_benchmark.plugins import register_plugin


def parse_settings(settings: dict[str, Any]) -> dict[str, Any]:
    return parse_record_settings(settings, plugin="longalpaca")


@register_plugin(
    stage="normalization",
    name="longalpaca",
    description="Normalize LongAlpaca instruction rows",
    settings_parser=parse_settings,
)
class LongAlpacaNormalization(RecordNormalizationPlugin):
    def normalize_record(self, record: Mapping[str, Any], source_order: int):
        instruction = required_text(record, "instruction")
        context = optional_text(record, "input")
        answer = required_text(record, "output")
        prompt = instruction if context is None else f"{instruction}\n{context}"
        identifier = str(record.get("id") or f"longalpaca-{source_order + 1}")
        yield TurnDraft(identifier, f"{identifier}:turn-1", 0,
                        {"messages": [{"role": "user", "content": prompt}]},
                        token_texts(prompt), token_texts(answer))


__all__ = ["LongAlpacaNormalization"]
