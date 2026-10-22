from collections.abc import Mapping
from typing import Any

from perf_benchmark.pipeline.contracts.normalization import (
    RecordNormalizationPlugin,
    TurnDraft,
    combine_tokens,
    message_tokens,
    parse_record_settings,
    required_text,
)
from perf_benchmark.plugins import register_plugin


def parse_settings(settings: dict[str, Any]) -> dict[str, Any]:
    parsed = parse_record_settings(
        settings,
        plugin="sharegpt",
        extra=("multi_turn", "live_continuation"),
    )
    for name in ("multi_turn", "live_continuation"):
        value = parsed.get(name, False)
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be boolean")
        parsed[name] = value
    if parsed["live_continuation"] and not parsed["multi_turn"]:
        raise ValueError("live_continuation requires multi_turn")
    return parsed


@register_plugin(
    stage="normalization",
    name="sharegpt",
    description="Normalize ShareGPT single or multi-turn conversations",
    settings_parser=parse_settings,
)
class ShareGPTNormalization(RecordNormalizationPlugin):
    def normalize_record(self, record: Mapping[str, Any], source_order: int):
        del source_order
        identifier = required_text(record, "id")
        conversation = record.get("conversations")
        if not isinstance(conversation, list) or not conversation or len(conversation) % 2:
            raise ValueError("conversations must contain complete human/gpt pairs")
        pairs = []
        for index in range(0, len(conversation), 2):
            human, assistant = conversation[index:index + 2]
            if not isinstance(human, Mapping) or human.get("from") != "human":
                raise ValueError("conversations must alternate human then gpt")
            if not isinstance(assistant, Mapping) or assistant.get("from") != "gpt":
                raise ValueError("conversations must alternate human then gpt")
            pairs.append((
                human,
                assistant,
                required_text(human, "value"),
                required_text(assistant, "value"),
            ))
        multi_turn = self.settings["multi_turn"]
        live = self.settings["live_continuation"]
        if not multi_turn:
            human, assistant, question, answer = pairs[0]
            yield TurnDraft(identifier, f"{identifier}:turn-1", 0,
                            {"messages": [{"role": "user", "content": question}]},
                            message_tokens(human, question),
                            message_tokens(assistant, answer),
                            requested_output_tokens=assistant.get("requested_output_tokens"))
            return
        history_tokens = 0
        for ordinal, (human, assistant, question, answer) in enumerate(pairs):
            question_tokens = message_tokens(human, question)
            answer_tokens = message_tokens(assistant, answer)
            yield TurnDraft(
                identifier,
                f"{identifier}:turn-{ordinal + 1}",
                ordinal,
                {"role": "user", "content": question},
                combine_tokens(history_tokens, question_tokens),
                answer_tokens,
                requested_output_tokens=assistant.get("requested_output_tokens"),
                context_mode="live" if live else "recorded",
                recorded_response=(
                    None if live else {"role": "assistant", "content": answer}
                ),
            )
            history_tokens = combine_tokens(
                history_tokens,
                question_tokens,
                answer_tokens,
            )


__all__ = ["ShareGPTNormalization"]
