from collections.abc import Mapping
from typing import Any

from perf_benchmark.core.profiles import RequestProfile
from perf_benchmark.pipeline.contracts.normalization import (
    RecordNormalizationPlugin,
    TurnDraft,
    fixed_tokens,
    parse_record_settings,
    required_text,
    token_texts,
)
from perf_benchmark.plugins import register_plugin


def parse_settings(settings: dict[str, Any]) -> dict[str, Any]:
    return parse_record_settings(settings, plugin="rerank")


@register_plugin(
    stage="normalization",
    name="rerank",
    description="Normalize rerank query and document rows",
    settings_parser=parse_settings,
)
class RerankNormalization(RecordNormalizationPlugin):
    def normalize_record(self, record: Mapping[str, Any], source_order: int):
        query = required_text(record, "query")
        documents = record.get("documents")
        if not isinstance(documents, list) or not documents or not all(
            isinstance(item, str) and item for item in documents
        ):
            raise ValueError("documents must be a non-empty text array")
        identifier = str(record.get("id") or f"rerank-{source_order + 1}")
        yield TurnDraft(identifier, f"{identifier}:turn-1", 0,
                        {"query": query, "documents": documents},
                        token_texts(query, *documents), fixed_tokens(0), input_kind="rerank", profile=RequestProfile(input_items=len(documents)))


__all__ = ["RerankNormalization"]
