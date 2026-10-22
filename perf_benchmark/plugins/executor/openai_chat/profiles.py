"""Wire profile composition, using preparation summaries and retained live summaries."""

from dataclasses import replace

from perf_benchmark.core.profiles import RequestProfile

_COUNTS = (
    "message_count",
    "text_chars",
    "system_chars",
    "developer_chars",
    "system_tokens_estimate",
    "developer_tokens_estimate",
)


def request_profile(value, parameters, body_bytes, cache_isolation):
    current = value.turn.profile
    summary = current or RequestProfile()
    history = []
    if value.turn.context_mode != "snapshot":
        for previous in value.previous_turns:
            history.append(previous.profile)
            if value.turn.context_mode == "recorded":
                history.append(previous.recorded_profile)
            elif previous.id in value.live_history:
                history.append(value.live_history[previous.id].profile)
        profiles = [*history, current]
        counts = {
            name: sum(getattr(p, name) for p in profiles)
            if all(p is not None and getattr(p, name) is not None for p in profiles)
            else None
            for name in _COUNTS
        }
        history_count = (
            sum(p.message_count for p in history)
            if all(p is not None and p.message_count is not None for p in history)
            else None
        )
        fingerprints = {p.tokenizer_fingerprint for p in profiles if p and p.tokenizer_fingerprint}
        if len(fingerprints) > 1:
            counts["system_tokens_estimate"] = counts["developer_tokens_estimate"] = None
        summary = replace(
            summary,
            message_count=counts["message_count"], text_chars=counts["text_chars"],
            system_chars=counts["system_chars"], developer_chars=counts["developer_chars"],
            system_tokens_estimate=counts["system_tokens_estimate"],
            developer_tokens_estimate=counts["developer_tokens_estimate"],
            history_message_count=history_count,
            tokenizer_fingerprint=next(iter(fingerprints)) if len(fingerprints) == 1 else None,
        )
    tools = parameters.get("tools", [])
    tool_count = (
        current.tool_count
        if current and current.tool_count is not None
        else (len(tools) if isinstance(tools, (tuple, list)) else None)
    )
    return replace(
        summary,
        body_bytes=body_bytes,
        tool_count=tool_count,
        model=parameters.get("model"),
        stream=parameters.get("stream"),
        temperature=parameters.get("temperature"),
        top_p=parameters.get("top_p"),
        max_tokens=parameters.get("max_tokens"),
        context_mode=value.turn.context_mode,
        cache_isolation=cache_isolation,
    )
