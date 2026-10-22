from __future__ import annotations

import asyncio

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import orjson

from perf_benchmark.pipeline.preparation.payloads import BlobStore, PayloadRef
from perf_benchmark.pipeline.contracts.executor import TurnRequestFacts
from perf_benchmark.plugins.executor.openai_chat.content import attempt_message_segments
from perf_benchmark.plugins.executor.openai_chat.profiles import request_profile
from perf_benchmark.runtime.http.request import PreparedBody, HttpRequestSpec
from perf_benchmark.runtime.http.request import merge_extra_body

_MAX_INLINE_PARAMETERS_BYTES = 64 * 1024


def build_chat_request(value, settings, headers, cache_salt=None):
    turn = value.turn
    if turn.input_kind != "chat":
        raise TypeError(f"openai-chat requires chat input, got {turn.input_kind}")
    if not (settings.get("base_url") or settings.get("base_urls")) or not settings.get("model"):
        raise ValueError("base_url(s) and model are required")
    extra = deepcopy(settings.get("extra_body", {}))
    max_tokens = turn.requested_output_tokens
    if max_tokens is None:
        max_tokens = settings.get("max_tokens")
    if max_tokens is not None and (type(max_tokens) is not int or max_tokens <= 0):
        raise ValueError("max_tokens must be a positive integer when supplied")
    stream = bool(settings.get("stream", True))
    parameters = {
        "model": settings["model"],
        "stream": stream,
    }
    if max_tokens is not None:
        parameters["max_tokens"] = max_tokens
    if cache_salt is not None:
        parameters["cache_salt"] = cache_salt
    fixed = turn.metadata.get("dataset_output_policy") == "fixed"
    natural = turn.metadata.get("dataset_output_policy") == "natural"
    if natural and extra.get("ignore_eos") is True:
        raise ValueError("natural output conflicts with ignore_eos")
    if fixed:
        if max_tokens is None:
            raise ValueError("fixed output requires an explicit output target")
        if (
            any(extra.get(key) for key in ("stop", "stop_token_ids"))
            or extra.get("ignore_eos") is False
        ):
            raise ValueError("fixed output conflicts with explicit stop conditions")
        parameters["ignore_eos"] = True
    parameters = merge_extra_body(parameters, extra)
    headers = dict(headers)
    dp_rank_count = settings.get("dp_rank_count")
    dp_rank = None
    if dp_rank_count is not None:
        dp_rank = value.session.ordinal % dp_rank_count
        headers["X-Data-Parallel-Rank"] = str(dp_rank)

    def body(spec):
        parameters = spec.parameters
        parameters.setdefault("temperature", 0.1)
        effective = parameters.get("max_tokens")
        if effective is None and (fixed or parameters.get("ignore_eos") is True):
            raise ValueError("fixed output requires an explicit output target")
        if effective is None and not ('max_tokens' in settings and settings['max_tokens'] is None):
            effective = parameters["max_tokens"] = 4096
        if "max_completion_tokens" in parameters:
            raise ValueError("request parameters must use max_tokens for the output target")
        if "max_tokens" in parameters and (type(effective) is not int or effective <= 0):
            raise ValueError("request max_tokens must be a positive integer")
        if fixed and (effective != max_tokens or parameters.get("ignore_eos") is not True):
            raise ValueError("request parameters violate the fixed output contract")
        if natural and parameters.get("ignore_eos") is True:
            raise ValueError("request parameters violate the natural output contract")
        effective_stream = parameters.get("stream")
        if not isinstance(effective_stream, bool):
            raise TypeError("request stream must be a boolean")
        spec.streaming = effective_stream
        if effective_stream:
            parameters["stream_options"] = {"include_usage": True}
        metadata = {}
        if dp_rank is not None:
            metadata["dp_rank"] = dp_rank
        if cache_salt is not None:
            metadata["cache_salt_applied"] = True
            metadata["cache_phase"] = "warmup" if value.session.metadata.get("warmup") else "formal"
            metadata["cache_cycle"] = value.session.metadata.get("cycle", 0)
        metadata["execution_fingerprint"] = hashlib.sha256(
            json.dumps(
                [
                    spec.url,
                    {
                        key: value
                        for key, value in parameters.items()
                        if key not in {"cache_salt", "max_tokens"}
                    },
                    {"cache_salt_enabled": settings.get("cache_salt_enabled", False),
                     "dp_rank_count": dp_rank_count},
                ],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        body = _build_body(parameters, value, bool(extra))
        try:
            profile = request_profile(value, parameters, body.content_length,
                                      "salt_sent" if cache_salt else "disabled")
            spec.request_facts = TurnRequestFacts(effective, metadata, profile)
            return body
        except BaseException:
            body.release()
            raise

    async def prepare_body(spec):
        # Snapshot splicing does not parse, copy or scan the body. Only bounded
        # request parameters and already prepared profiles are processed here.
        if turn.context_mode == "snapshot" and not extra:
            return body(spec)
        return await asyncio.to_thread(body, spec)

    urls = settings.get("base_urls")
    block_size = settings.get("base_url_session_block_size", 1)
    base_url = urls[(value.session.ordinal // block_size) % len(urls)] if urls else settings["base_url"]
    return HttpRequestSpec(
        str(base_url).rstrip("/")
        + str(settings.get("endpoint", "/v1/chat/completions")),
        parameters,
        prepare_body,
        headers,
        streaming=stream,
    )


def _snapshot_body(
    parameters: Mapping[str, object],
    snapshot: PayloadRef,
    payloads: BlobStore,
    check_collision: bool = False,
) -> PreparedBody:
    encoded = json.dumps(dict(parameters), ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    if len(encoded) > _MAX_INLINE_PARAMETERS_BYTES:
        raise ValueError("inline request parameters exceed 64 KiB")
    prefix = b"{" if encoded == b"{}" else encoded[:-1] + b","
    value = payloads.view(snapshot)
    try:
        if len(value) < 2 or value[0] != ord("{") or value[-1] != ord("}"):
            raise ValueError("snapshot payload must be a JSON object")
        if check_collision and (collision := orjson.loads(value).keys() & parameters.keys()):
            raise ValueError(f"extra_body conflicts with dataset fields: {', '.join(sorted(collision))}")
        inner = value[1:-1]
        return PreparedBody(
            (prefix, inner, b"}"),
            len(prefix) + len(inner) + 1,
        )
    finally:
        value.release()


def _build_body(parameters, value, check_collision=False) -> PreparedBody:
    turn = value.turn
    if turn.context_mode == "snapshot":
        return _snapshot_body(parameters, turn.request_payload_ref, value.payloads, check_collision)
    messages = attempt_message_segments(value)
    try:
        return _request_body(parameters, messages)
    except BaseException:
        for value in messages:
            if isinstance(value, memoryview):
                value.release()
        raise


def _request_body(
    parameters: Mapping[str, object],
    messages: tuple[bytes | memoryview, ...],
) -> PreparedBody:
    encoded = json.dumps(
        dict(parameters),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > _MAX_INLINE_PARAMETERS_BYTES:
        raise ValueError("inline request parameters exceed 64 KiB")
    prefix = b'{"messages":[' if encoded == b"{}" else encoded[:-1] + b',"messages":['
    suffix = b"]}"
    segments: list[bytes | memoryview] = [prefix]
    for index, value in enumerate(messages):
        if index:
            segments.append(b",")
        segments.append(value)
    segments.append(suffix)
    return PreparedBody(
        tuple(segments),
        sum(len(segment) for segment in segments),
    )
