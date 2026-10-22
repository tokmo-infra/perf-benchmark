"""Offline snapshot validation and exact-length construction using core tokenization."""

import json

from perf_benchmark.plugins.executor.openai_chat.content import template_messages

from .common import positive


def validate_request(request):
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("a non-empty text chat snapshot is required")
    tools = request.get("tools") or []
    if not isinstance(tools, list):
        raise ValueError("tools must be a list")
    names = set()
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
            raise ValueError("only function tools are supported")
        function = tool["function"]
        if not isinstance(function.get("name"), str) or not function["name"] or not isinstance(function.get("parameters"), dict):
            raise ValueError("tools require a name and JSON parameter schema")
        if function["name"] in names:
            raise ValueError("tool function names must be unique")
        names.add(function["name"])
    pending, seen = set(), set()
    previous = None
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError("messages must be objects")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError("unsupported message role")
        if not isinstance(content, str) and not (content is None and role == "assistant" and message.get("tool_calls")):
            raise ValueError("only text content is supported")
        if role == "system" and index != 0:
            raise ValueError("only one leading system message is supported")
        if pending and role != "tool":
            raise ValueError("tool calls must be followed by all matching tool results")
        if role == "tool":
            identifier = message.get("tool_call_id")
            if identifier not in pending:
                raise ValueError("unmatched or duplicate tool result")
            pending.remove(identifier)
        elif role == "assistant":
            if previous not in {"user", "tool"}:
                raise ValueError("assistant must follow a user or tool result")
        elif role == "user" and previous not in {None, "system", "assistant", "tool"}:
            raise ValueError("user must follow a completed history")
        calls = message.get("tool_calls") or []
        if not isinstance(calls, list):
            raise ValueError("tool_calls must be a list")
        for call in calls:
            if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                raise ValueError("tool calls require function objects")
            identifier = call.get("id")
            function = call.get("function", {})
            if role != "assistant" or call.get("type") != "function" or not isinstance(identifier, str) or not identifier or identifier in seen:
                raise ValueError("invalid or duplicate tool call")
            if function.get("name") not in names:
                raise ValueError("tool call refers to an undefined function")
            try:
                arguments = json.loads(function["arguments"])
                json.dumps(arguments, allow_nan=False)
            except (KeyError, TypeError, ValueError):
                raise ValueError("tool arguments must be a JSON object string") from None
            if not isinstance(arguments, dict):
                raise ValueError("tool arguments must be a JSON object string")
            pending.add(identifier)
            seen.add(identifier)
        previous = role
    if pending or previous not in {"user", "tool"}:
        raise ValueError("snapshot must end at a user or completed tool result")


def request_token_ids(tokenizer, request):
    validate_request(request)
    if request.get("chat_template_kwargs", {}) != {key: value for key, value in tokenizer.options.items() if key != "add_generation_prompt"} and "chat_template_kwargs" in request:
        raise ValueError("snapshot chat_template_kwargs differs from the fixed model template options")
    if request.get("add_generation_prompt", tokenizer.options["add_generation_prompt"]) != tokenizer.options["add_generation_prompt"]:
        raise ValueError("snapshot generation prompt option differs from the fixed model options")
    return tokenizer.encode_chat(template_messages(request), tools=request.get("tools"))


def fit_input(tokenizer, request, index, prefix, suffix, target, output, content, *, context_tokens):
    positive(target, "input_tokens")
    positive(output, "output_tokens")
    if target + output > positive(context_tokens, "context_tokens"):
        raise ValueError("input plus output exceeds model context_tokens")
    message = request["messages"][index]
    message["content"] = prefix + suffix
    fixed = len(request_token_ids(tokenizer, request))
    if fixed > target:
        raise ValueError("fixed prompts/history/tools exceed target input length")
    tokens = []
    for attempt in range(32):
        if attempt == 0:
            actual = fixed
        else:
            message["content"] = prefix + tokenizer.decode(tokens) + suffix
            actual = len(request_token_ids(tokenizer, request))
        if actual == target:
            return actual
        delta = target - actual
        if delta > 0:
            tokens.extend(content(delta))
        elif tokens:
            estimate = round(len(tokens) * (target - fixed) / max(1, actual - fixed))
            del tokens[max(0, min(len(tokens) - 1, estimate)):]
        else:
            break
        if attempt % 4 == 3 and tokens:
            # A BPE boundary may jump over the target and oscillate forever.
            # Resample only the mutable tail; never touch fixed prompts/history.
            width = min(16, len(tokens))
            tokens[-width:] = content(width)
    raise ValueError("exact input length not reachable within 32 correction attempts")
