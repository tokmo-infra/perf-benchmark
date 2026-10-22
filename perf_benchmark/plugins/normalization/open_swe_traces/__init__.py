"""Replay complete SWE trajectories using dataset reference counts."""
import json
from perf_benchmark.pipeline.contracts.normalization import (
    FixedTokens, RecordNormalizationPlugin, TurnDraft, parse_record_settings,
    required_text,
)
from perf_benchmark.core.validation import positive_int
from perf_benchmark.plugins import register_plugin


def parse_settings(settings):
    parsed = parse_record_settings(
        settings, plugin="open-swe-traces", extra=("missing_tool_call_id",),
    )
    if "tokenizer_path" in parsed:
        raise ValueError("open-swe-traces uses dataset token counts, not tokenizer_path")
    policy = parsed.setdefault("missing_tool_call_id", "preserve")
    if policy not in ("preserve", "empty", "match-single"):
        raise ValueError("missing_tool_call_id must be preserve, empty or match-single")
    return parsed


def _match_single(value, pending):
    repaired = 0
    if value.get("role") == "assistant":
        if pending:
            raise ValueError("assistant reached before all tool_call_id results were supplied")
        for call in value.get("tool_calls") or []:
            identifier = call.get("id")
            if not isinstance(identifier, str) or not identifier:
                raise ValueError("match-single requires nonempty assistant tool call IDs")
            if identifier in pending:
                raise ValueError("duplicate pending tool_call_id")
            pending.add(identifier)
    elif value.get("role") == "tool":
        identifier = value.get("tool_call_id")
        if identifier is None or identifier == "":
            if len(pending) != 1:
                raise ValueError("match-single requires exactly one pending tool_call_id; association is ambiguous or missing")
            identifier, = pending
            value["tool_call_id"] = identifier
            repaired = 1
        elif not isinstance(identifier, str):
            raise TypeError("tool_call_id must be a string")
        if identifier not in pending:
            raise ValueError("tool_call_id does not match a pending assistant call")
        pending.remove(identifier)
    return repaired


def _messages(messages, policy):
    cleaned, placeholders, repairs, pending = [], [0], [0], set()
    for message in messages:
        value = {key: item for key, item in message.items() if item is not None}
        added = 0
        if policy == "empty" and value.get("role") == "tool":
            if "tool_call_id" not in value:
                # Schema placeholder only: parallel calls cannot safely be paired
                # by position or by guessing an ID from the preceding assistant.
                value["tool_call_id"] = ""
                added = 1
            elif not isinstance(value["tool_call_id"], str):
                raise TypeError("tool_call_id must be a string")
        cleaned.append(value)
        placeholders.append(placeholders[-1] + added)
        repaired = _match_single(value, pending) if policy == "match-single" else 0
        repairs.append(repairs[-1] + repaired)
    return cleaned, placeholders, repairs


@register_plugin(stage="normalization", name="open-swe-traces",
                 description="Complete recorded SWE tool trajectories without local tokenization",
                 settings_parser=parse_settings)
class OpenSWETracesNormalization(RecordNormalizationPlugin):
    def normalize_record(self, record, source_order):
        del source_order
        identifier = required_text(record, "benchmark_session_id")
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be non-empty")
        cleaned, placeholders, repairs = _messages(
            messages, self.settings["missing_tool_call_id"],
        )
        tools = [json.loads(tool) if isinstance(tool, str) else tool
                 for tool in record.get("tools") or []]
        if any(not isinstance(tool, dict) for tool in tools):
            raise ValueError("tools must contain objects")
        positions = [index for index, message in enumerate(cleaned)
                     if message.get("role") == "assistant"]
        inputs, outputs = record["benchmark_input_tokens"], record["benchmark_output_tokens"]
        if len(positions) != len(inputs) or len(inputs) != len(outputs):
            raise ValueError("precomputed counts must match every assistant turn")
        for ordinal, position in enumerate(positions):
            if position == 0:
                raise ValueError("assistant requires recorded input history")
            dataset_input_tokens = positive_int(inputs[ordinal], "input_tokens", allow_zero=True)
            output_tokens = positive_int(outputs[ordinal], "output_tokens", allow_zero=True)
            request = {"messages": cleaned[:position]}
            if tools:
                request["tools"] = tools
            declared_output = record.get("requested_output_tokens")
            if declared_output is not None:
                declared_output = positive_int(declared_output, "requested_output_tokens")
            yield TurnDraft(
                identifier, f"{identifier}:turn-{ordinal + 1}", ordinal,
                request, FixedTokens(dataset_input_tokens), FixedTokens(output_tokens),
                requested_output_tokens=declared_output,
                source_session_id=identifier,
                metadata={"dataset_output_policy": "natural",
                          "dataset_category": "coding",
                          "dataset_input_tokens": dataset_input_tokens,
                          "input_token_count_source": "dataset_reference",
                          "history_policy": "recorded_swe_tool_trace",
                          "source_trajectory_id": str(record.get("trajectory_id", "")),
                          "tool_call_id_policy": self.settings["missing_tool_call_id"],
                          "tool_call_id_placeholders": placeholders[position],
                          **({"tool_call_id_repairs": repairs[position]}
                             if self.settings["missing_tool_call_id"] == "match-single" else {})},
            )
