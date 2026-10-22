from collections.abc import Mapping
from typing import Any

from perf_benchmark.core.profiles import RequestProfile
from perf_benchmark.pipeline.contracts.normalization import (
    RecordNormalizationPlugin,
    TurnDraft,
    fixed_tokens,
    parse_record_settings,
    token_texts,
)
from perf_benchmark.plugins import register_plugin


def parse_settings(settings: dict[str, Any]) -> dict[str, Any]:
    return parse_record_settings(settings, plugin="embedding")


@register_plugin(
    stage="normalization",
    name="embedding",
    description="Normalize embedding input rows",
    settings_parser=parse_settings,
)
class EmbeddingNormalization(RecordNormalizationPlugin):
    def normalize_record(self, record: Mapping[str, Any], source_order: int):
        value = record.get("input")
        if isinstance(value, str) and value:
            texts = (value,)
        elif isinstance(value, list) and value and all(isinstance(item, str) and item for item in value):
            texts = tuple(value)
        else:
            raise ValueError("input must be non-empty text or a non-empty text array")
        identifier = str(record.get("id") or f"embedding-{source_order + 1}")
        yield TurnDraft(identifier, f"{identifier}:turn-1", 0, {"input": value},
                        token_texts(*texts), fixed_tokens(0), input_kind="embedding", profile=RequestProfile(input_items=len(texts)))


__all__ = ["EmbeddingNormalization"]
