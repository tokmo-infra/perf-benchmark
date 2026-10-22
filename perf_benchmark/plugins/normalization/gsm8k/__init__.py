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
    return parse_record_settings(settings, plugin="gsm8k")


@register_plugin(
    stage="normalization",
    name="gsm8k",
    description="Normalize GSM8K question and answer rows",
    settings_parser=parse_settings,
)
class GSM8KNormalization(RecordNormalizationPlugin):
    def normalize_record(self, record: Mapping[str, Any], source_order: int):
        question = required_text(record, "question")
        answer = required_text(record, "answer")
        prompt = f"{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        identifier = str(record.get("id") or f"gsm8k-{source_order + 1}")
        yield TurnDraft(identifier, f"{identifier}:turn-1", 0,
                        {"messages": [{"role": "user", "content": prompt}]},
                        token_texts(prompt), token_texts(answer))


__all__ = ["GSM8KNormalization"]
