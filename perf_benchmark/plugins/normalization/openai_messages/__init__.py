from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from perf_benchmark.core.validation import positive_int
from perf_benchmark.pipeline.contracts.normalization import (
    RecordNormalizationPlugin,
    TurnDraft,
    combine_tokens,
    compact_json_text,
    message_tokens,
    parse_created,
    parse_record_settings,
    required_text,
    token_texts,
)
from perf_benchmark.plugins import register_plugin


def parse_settings(settings: dict[str, Any]) -> dict[str, Any]:
    parsed = parse_record_settings(
        settings,
        plugin="openai-messages",
        extra=("multi_turn", "live_continuation"),
    )
    for name in ("multi_turn", "live_continuation"):
        value = parsed.get(name, False)
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be boolean")
        parsed[name] = value
    if parsed["live_continuation"]:
        raise ValueError("openai-messages does not support live_continuation")
    return parsed


@register_plugin(
    stage="normalization",
    name="openai-messages",
    description="Normalize OpenAI message request snapshots",
    settings_parser=parse_settings,
)
class OpenAIMessagesNormalization(RecordNormalizationPlugin):
    def normalize_record(
        self,
        record: Mapping[str, Any],
        source_order: int,
    ):
        del source_order
        identifier = required_text(record, "id")
        source_session = required_text(record, "token_name")
        created = parse_created(record.get("created"))
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages or not all(
            isinstance(item, Mapping) for item in messages
        ):
            raise ValueError("messages must be a non-empty object array")
        request: dict[str, object] = {"messages": messages}
        tools = record.get("tools")
        if tools is not None:
            if not isinstance(tools, list) or not all(
                isinstance(item, Mapping) for item in tools
            ):
                raise TypeError("tools must be an object array")
            request["tools"] = tools
        if record.get("tool_choice") is not None:
            request["tool_choice"] = record["tool_choice"]
        if record.get("chat_template_kwargs") is not None:
            if not isinstance(record["chat_template_kwargs"], dict):
                raise TypeError("chat_template_kwargs must be an object")
            request["chat_template_kwargs"] = record["chat_template_kwargs"]
        if record.get("add_generation_prompt") is not None:
            if not isinstance(record["add_generation_prompt"], bool):
                raise TypeError("add_generation_prompt must be a boolean")
            request["add_generation_prompt"] = record["add_generation_prompt"]
        media_input_tokens = _media_input_tokens(record, messages)
        if media_input_tokens is not None:
            input_tokens = media_input_tokens
        elif any(message.get("token_count") is not None for message in messages):
            input_tokens = combine_tokens(*(
                message_tokens(message, _message_text(message))
                for message in messages
            ))
            request_extras = {
                name: request[name]
                for name in ("tools", "tool_choice")
                if name in request
            }
            if request_extras:
                input_tokens = combine_tokens(
                    input_tokens,
                    token_texts(compact_json_text(request_extras)),
                )
        else:
            input_tokens = token_texts(compact_json_text(request))
        response = record.get("response_body_content")
        output_tokens = (
            token_texts(
                response if isinstance(response, str) else compact_json_text(response)
            )
            if response is not None else None
        )
        multi_turn = self.settings["multi_turn"]
        dataset_metadata = {}
        for key in ("dataset_category", "dataset_output_policy", "dataset_tokenizer_fingerprint"):
            if record.get(key) is not None:
                value = record[key]
                if not isinstance(value, str) or len(value) > 256:
                    raise ValueError(f"{key} must be a short string")
                dataset_metadata[key] = value
        if dataset_metadata.get("dataset_output_policy") not in {None, "fixed", "natural"}:
            raise ValueError("dataset_output_policy must be fixed or natural")
        yield TurnDraft(
            source_session if multi_turn else identifier,
            identifier,
            _epoch_microseconds(created) if multi_turn else 0,
            request,
            input_tokens,
            output_tokens,
            created_at=created,
            metadata=dataset_metadata,
            source_session_id=source_session,
        )


def _epoch_microseconds(value: datetime) -> int:
    return int(value.timestamp()) * 1_000_000 + value.microsecond


def _media_input_tokens(record, messages):
    if not any(
        isinstance(message.get("content"), list)
        and any(
            not isinstance(part, Mapping)
            or part.get("type") not in {"text", "input_text", "output_text"}
            for part in message["content"]
        )
        for message in messages
    ):
        return None
    usage = record.get("usage")
    if usage is not None and not isinstance(usage, Mapping):
        raise TypeError("usage must be an object")
    value = usage.get("prompt_tokens") if usage is not None else None
    name = "usage.prompt_tokens"
    if value is None:
        value, name = record.get("input_tokens"), "input_tokens"
    if value is None:
        raise ValueError(
            "non-text message parts require explicit input_tokens or usage.prompt_tokens"
        )
    return positive_int(value, name, allow_zero=True)


def _message_text(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    return content if isinstance(content, str) else compact_json_text(message)


__all__ = ["OpenAIMessagesNormalization"]
