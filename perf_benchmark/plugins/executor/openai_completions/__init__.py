"""Text completions; snapshots pass through the existing mmap/Rust path."""

import asyncio
from dataclasses import replace
from collections.abc import Mapping
from urllib.parse import urlsplit

import orjson

from perf_benchmark.core.models import thaw_snapshot
from perf_benchmark.core.profiles import RequestProfile
from perf_benchmark.core.validation import positive_number, positive_int, validate_extra_headers
from perf_benchmark.pipeline.contracts.executor import TurnRequestFacts
from perf_benchmark.plugins import register_plugin
from perf_benchmark.runtime.http.rust_executor import RustHttpExecutorRuntime
from perf_benchmark.runtime.http.request import (PreparedBody, HttpRequestPlugin, HttpRequestSpec,
    merge_extra_body, validate_api_keys, validate_extra_body)


def validate(settings):
    parsed = dict(settings)
    if unknown := set(parsed) - {"base_url", "model", "extra_headers", "api_keys", "api_key_sampling", "stream",
                                "timeout_seconds", "extra_body", "max_tokens", "connection_scope", "verify_ssl", "cache_salt_enabled"}:
        raise ValueError(f"unknown openai-completions settings: {sorted(unknown)}")
    for key in ("base_url", "model"):
        if not isinstance(parsed.get(key), str) or not parsed[key]:
            raise ValueError(f"{key} is required")
    url = urlsplit(parsed["base_url"])
    if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError("base_url must be an HTTP URL without credentials, query or fragment")
    if parsed.setdefault("connection_scope", "session") not in {"session", "request", "trial"}:
        raise ValueError("invalid connection_scope")
    for key, default in (("stream", True), ("verify_ssl", False), ("cache_salt_enabled", False)):
        if not isinstance(parsed.setdefault(key, default), bool):
            raise ValueError(f"{key} must be boolean")
    positive_number(parsed.setdefault("timeout_seconds", 300.0), "timeout_seconds")
    validate_extra_headers(parsed.setdefault("extra_headers", {}))
    validate_api_keys(parsed)
    positive_int(parsed.get("max_tokens"), "max_tokens", optional=True)
    _parameters(parsed.setdefault("extra_body", {}))
    return parsed


def _parameters(parameters):
    validate_extra_body(parameters, owned={"prompt", "messages", "model", "stream", "max_tokens",
        "max_completion_tokens", "cache_salt", "cache_salt_enabled", "ignore_eos", "stream_options"})
    if any(type(parameters.get(key, 1)) is not int or parameters.get(key, 1) != 1 for key in ("n", "best_of")) or parameters.get("echo", False) is not False:
        raise ValueError("completions requires n=best_of=1 and echo=false")
    if any(parameters.get(k) for k in ("stop", "stop_token_ids")):
        raise ValueError("fixed output requires ignore_eos and no stop conditions")
    for key in ("temperature", "top_p"):
        if key in parameters:
            positive_number(parameters[key], key, allow_zero=True)


@register_plugin(stage="executor", name="openai-completions",
                 description="OpenAI text completions with optional output targets", settings_parser=validate)
class OpenAICompletionsExecutor(HttpRequestPlugin):
    supports_start_notifications = True

    async def create_runtime(self, trial, ctx):
        return RustHttpExecutorRuntime(self, self.settings_for_trial(trial), ctx.cancellation,
                                     'openai-completions')

    def settings_for_trial(self, trial):
        settings = thaw_snapshot(self.settings)
        overrides = thaw_snapshot(trial.request_parameters)
        assert isinstance(overrides, Mapping)
        parameters = {**settings["extra_body"], **overrides}
        _parameters(parameters)
        settings["extra_body"] = parameters
        settings["_trial_id"] = trial.id
        return settings

    def build_request(self, value, settings):
        turn = value.turn
        if turn.input_kind != "completion" or turn.context_mode != "snapshot":
            raise ValueError("openai-completions requires a text completion snapshot")
        extra = settings["extra_body"]
        maximum = turn.requested_output_tokens
        if maximum is None:
            maximum = settings.get("max_tokens")
        positive_int(maximum, "requested_output_tokens", optional=True)
        explicit_target = maximum is not None
        if maximum is None:
            maximum = 4096
        parameters = dict(model=settings["model"], stream=settings["stream"])
        parameters["max_tokens"] = maximum
        if explicit_target:
            parameters["ignore_eos"] = True
        if salt := self.cache_salt(value, settings):
            parameters["cache_salt"] = salt
        if settings["stream"]:
            parameters["stream_options"] = {"include_usage": True}
        parameters = merge_extra_body(parameters, extra)
        parameters.setdefault("temperature", 0.1)
        headers = self.request_headers(settings, value)

        def body(spec):
            encoded = orjson.dumps(spec.parameters)
            if len(encoded) > 65536:
                raise ValueError("inline request parameters exceed 64 KiB")
            payload = value.payloads.view(turn.request_payload_ref)
            try:
                if len(payload) < 2 or payload[0] != 123 or payload[-1] != 125:
                    raise ValueError("completion snapshot must be a JSON object containing prompt")
                if extra and (conflict := orjson.loads(payload).keys() & parameters.keys()):
                    raise ValueError(f"extra_body conflicts with dataset fields: {', '.join(sorted(conflict))}")
                prefix, inner = encoded[:-1] + b",", payload[1:-1]
                result = PreparedBody((prefix, inner, b"}"), len(prefix) + len(inner) + 1)
            finally:
                payload.release()
            profile = replace(turn.profile or RequestProfile(), model=settings["model"],
                              stream=settings["stream"], max_tokens=maximum, context_mode="snapshot",
                              temperature=parameters.get("temperature"), top_p=parameters.get("top_p"),
                              body_bytes=result.content_length, cache_isolation="salt_sent" if "cache_salt" in parameters else "disabled")
            metadata = ({"cache_salt_applied": True, "cache_phase": "warmup" if value.session.metadata.get("warmup") else "formal",
                         "cache_cycle": value.session.metadata.get("cycle", 0)} if "cache_salt" in parameters else {})
            spec.request_facts = TurnRequestFacts(maximum, metadata, profile)
            return result

        async def prepare_body(spec):
            if not extra:
                return body(spec)
            return await asyncio.to_thread(body, spec)

        return HttpRequestSpec(settings["base_url"].rstrip("/") + "/v1/completions", parameters, prepare_body,
                               headers, streaming=settings["stream"],
                               response_expectations={"output_tokens": maximum} if explicit_target else {})
