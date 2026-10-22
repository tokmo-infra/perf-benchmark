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
    return parse_record_settings(settings, plugin="openqa")


@register_plugin(
    stage="normalization",
    name="openqa",
    description="Normalize OpenQA rows to canonical chat snapshots",
    settings_parser=parse_settings,
)
class OpenQANormalization(RecordNormalizationPlugin):
    def normalize_record(
        self,
        record: Mapping[str, Any],
        source_order: int,
    ):
        question = required_text(record, "question")
        answers = record.get("chatgpt_answers")
        if not isinstance(answers, list) or not answers or not isinstance(answers[0], str):
            raise TypeError("chatgpt_answers[0] must be text")
        identifier = str(record.get("id") or f"openqa-{source_order + 1}")
        yield TurnDraft(
            identifier,
            str(record.get("turn_id") or f"{identifier}:turn-1"),
            int(record.get("turn_ordinal", 0)),
            {"messages": [{"role": "user", "content": question}]},
            token_texts(question),
            token_texts(answers[0]),
        )


__all__ = ["OpenQANormalization"]
