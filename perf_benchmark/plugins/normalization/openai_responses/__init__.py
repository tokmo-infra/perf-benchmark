"""Stateless text and function-call Responses snapshots."""
from collections.abc import Mapping

from perf_benchmark.core.profiles import RequestProfile
from perf_benchmark.pipeline.contracts.normalization import (
    RecordNormalizationPlugin, TurnDraft, compact_json_text,
    parse_record_settings, token_texts,
)
from perf_benchmark.plugins import register_plugin


def validate(settings):
    return parse_record_settings(settings, plugin="openai-responses")


@register_plugin(stage="normalization", name="openai-responses", description="Responses text and function snapshots",
                 settings_parser=validate)
class ResponsesNormalization(RecordNormalizationPlugin):
    def normalize_record(self, record: Mapping, source_order):
        value = record.get("input")
        if isinstance(value, str):
            if not value:
                raise ValueError("Responses input must not be empty")
        elif isinstance(value, list) and value:
            for item in value:
                if not isinstance(item, dict):
                    raise ValueError("Responses input items must be objects")
                kind = item.get("type", "message")
                if kind == "message":
                    if set(item) - {"type", "role", "content"} or item.get("role") not in {"user", "assistant", "system", "developer"}:
                        raise ValueError("invalid Responses message")
                    content = item.get("content")
                    if not isinstance(content, str):
                        if not isinstance(content, list) or not content or any(
                            not isinstance(part, dict) or set(part) != {"type", "text"}
                            or part["type"] not in {"input_text", "output_text"} or not isinstance(part["text"], str)
                            for part in content
                        ):
                            raise ValueError("Responses supports text content only")
                elif kind == "function_call":
                    if set(item) - {"type", "id", "call_id", "name", "arguments", "status"} or any(not isinstance(item.get(k), str) for k in ("call_id", "name", "arguments")):
                        raise ValueError("invalid recorded function call")
                elif kind == "function_call_output":
                    if set(item) - {"type", "id", "call_id", "output", "status"} or any(not isinstance(item.get(k), str) for k in ("call_id", "output")):
                        raise ValueError("invalid recorded function output")
                else:
                    raise ValueError(f"unsupported Responses input type: {kind}")
        else:
            raise ValueError("Responses input must be text or a nonempty item array")
        request = {"input": value}
        tools = record.get("tools", [])
        if not isinstance(tools, list) or any(not isinstance(t, dict) or t.get("type") != "function" or
            not isinstance(t.get("name"), str) or set(t)-{"type", "name", "description", "parameters", "strict"} for t in tools):
            raise ValueError("Responses supports function tool declarations only")
        if tools:
            request["tools"] = tools
        if "instructions" in record:
            if not isinstance(record["instructions"], str):
                raise ValueError("instructions must be text")
            request["instructions"] = record["instructions"]
        if any(k in record for k in ("previous_response_id", "conversation", "background")):
            raise ValueError("Responses input must be a stateless snapshot")
        identifier = str(record.get("id") or f"responses-{source_order + 1}")
        yield TurnDraft(identifier, identifier+":turn-1", 0, request,
                        token_texts(compact_json_text(request)), None, input_kind="responses",
                        profile=RequestProfile(tool_count=len(tools)))
