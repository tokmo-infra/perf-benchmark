"""Body-free request profiles (None means unavailable) and private continuations."""

from dataclasses import dataclass

GENERATION_KEYS = frozenset({"temperature", "top_p", "top_k", "seed", "max_tokens", "max_completion_tokens",
                             "max_output_tokens", "ignore_eos", "repetition_penalty", "presence_penalty", "frequency_penalty", "n"})


@dataclass(frozen=True, slots=True)
class RequestProfile:
    input_items: int | None = None
    message_count: int | None = None
    history_message_count: int | None = None
    tool_count: int | None = None
    text_chars: int | None = None
    system_chars: int | None = None
    developer_chars: int | None = None
    system_tokens_estimate: int | None = None
    developer_tokens_estimate: int | None = None
    tokenizer_fingerprint: str | None = None
    body_bytes: int | None = None
    model: str | None = None
    stream: bool | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    context_mode: str | None = None
    cache_isolation: str | None = None


@dataclass(frozen=True, slots=True)
class Continuation:
    payload: bytes
    profile: RequestProfile | None = None
