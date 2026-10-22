"""Segmented snapshot bodies and common HTTP configuration, without protocol policy."""

import asyncio
from collections.abc import Mapping
from dataclasses import replace
from urllib.parse import urlsplit

import orjson

from perf_benchmark.core.models import thaw_snapshot
from perf_benchmark.core.profiles import RequestProfile
from perf_benchmark.core.validation import positive_number, validate_extra_headers
from perf_benchmark.pipeline.contracts.executor import TurnRequestFacts
from perf_benchmark.runtime.http.request import (
    PreparedBody, HttpRequestSpec, validate_api_keys, validate_extra_body,
)


def http_settings(settings, *, stream=False, owned_body=None):
    parsed = dict(settings)
    allowed = {"base_url", "model", "extra_headers", "api_keys", "api_key_sampling", "timeout_seconds", "extra_body",
               "connection_scope", "verify_ssl", "cache_salt_enabled"} | ({"stream"} if stream else set())
    if unknown := set(parsed) - allowed:
        raise ValueError(f"unknown HTTP settings: {sorted(unknown)}")
    for key in ("base_url", "model"):
        if not isinstance(parsed.get(key), str) or not parsed[key]:
            raise ValueError(f"{key} is required")
    url = urlsplit(parsed["base_url"])
    if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError("base_url must be an HTTP URL without credentials, query or fragment")
    if parsed.setdefault("connection_scope", "session") not in {"session", "request", "trial"}:
        raise ValueError("invalid connection_scope")
    for key in ("verify_ssl", "cache_salt_enabled") + (("stream",) if stream else ()):
        if not isinstance(parsed.setdefault(key, key == "stream"), bool):
            raise ValueError(f"{key} must be boolean")
    positive_number(parsed.setdefault("timeout_seconds", 300.0), "timeout_seconds")
    validate_extra_headers(parsed.setdefault("extra_headers", {}))
    validate_api_keys(parsed)
    validate_extra_body(parsed.setdefault("extra_body", {}), owned=owned_body or set())
    return parsed


def trial_settings(settings, trial, validate):
    result = thaw_snapshot(settings)
    overrides = thaw_snapshot(trial.request_parameters)
    assert isinstance(result, dict) and isinstance(overrides, Mapping)
    result["extra_body"] = {**result["extra_body"], **overrides}
    validate(result["extra_body"])
    result["_trial_id"] = trial.id
    return result


def snapshot_request(value, settings, *, path, parameters, input_kind, headers, expectations=None):
    turn = value.turn
    if turn.input_kind != input_kind or turn.context_mode != "snapshot":
        raise ValueError(f"{path} requires a {input_kind} snapshot")

    def body(spec):
        encoded = orjson.dumps(spec.parameters)
        if len(encoded) > 65536:
            raise ValueError("inline request parameters exceed 64 KiB")
        with value.payloads.view(turn.request_payload_ref) as payload:
            if len(payload) < 2 or payload[0] != 123 or payload[-1] != 125:
                raise ValueError("snapshot must be a JSON object")
            snapshot = orjson.loads(payload) if settings.get("extra_body") else None
            if snapshot is not None and (conflict := snapshot.keys() & spec.parameters.keys()):
                raise ValueError(f"extra_body conflicts with dataset fields: {', '.join(sorted(conflict))}")
            prefix = payload[:-1]
            suffix = (b"," if len(payload) > 2 else b"") + encoded[1:] if encoded != b"{}" else b"}"
            result = PreparedBody((prefix, suffix), len(prefix) + len(suffix))
        profile = replace(turn.profile or RequestProfile(), model=settings["model"],
                          stream=bool(parameters.get("stream", False)), context_mode="snapshot",
                          temperature=parameters.get("temperature"), top_p=parameters.get("top_p"),
                          max_tokens=parameters.get("max_output_tokens"), body_bytes=result.content_length,
                          cache_isolation="salt_sent" if "cache_salt" in parameters else "disabled")
        metadata = ({"cache_salt_applied": True, "cache_phase": "warmup" if value.session.metadata.get("warmup") else "formal",
                     "cache_cycle": value.session.metadata.get("cycle", 0)} if "cache_salt" in parameters else {})
        spec.request_facts = TurnRequestFacts(profile.max_tokens, metadata, profile)
        return result

    async def prepare_body(spec):
        if not settings.get("extra_body"):
            return body(spec)
        return await asyncio.to_thread(body, spec)

    stream = bool(parameters.get("stream", False))
    return HttpRequestSpec(settings["base_url"].rstrip("/") + path, parameters, prepare_body, headers, streaming=stream,
                           response_expectations=expectations or {})
