import asyncio
import orjson
from copy import deepcopy
from urllib.parse import urlsplit

from perf_benchmark.core.models import thaw_snapshot
from perf_benchmark.pipeline.contracts.executor import ExecutorTrial
from perf_benchmark.plugins import register_plugin
from perf_benchmark.plugins.executor.openai_chat.request import build_chat_request
from perf_benchmark.runtime.http.request import HttpRequestPlugin
from perf_benchmark.runtime.http.request import validate_api_keys, validate_extra_body
from perf_benchmark.core.validation import positive_int, validate_extra_headers
from perf_benchmark.runtime.http.rust_executor import RustHttpExecutorRuntime
from perf_benchmark.pipeline.contracts.normalization import message_profile


_SETTING_KEYS = {
    "base_url",
    "base_urls",
    "base_url_session_block_size",
    "dp_rank_count",
    "model",
    "endpoint",
    "extra_headers",
    "api_keys",
    "api_key_sampling",
    "stream",
    "timeout_seconds",
    "extra_body",
    "max_tokens",
    "cache_salt_enabled",
    "connection_scope",
    "verify_ssl",
}


def reasoning_probe(settings, effort):
    """Minimal protocol capability request, separate from benchmark attempts."""
    for base in dict.fromkeys(settings.get('base_urls') or [settings['base_url']]):
        yield base.rstrip('/') + settings['endpoint'], {
            'model': settings['model'], 'messages': [{'role': 'user', 'content': 'Reply OK.'}],
            'stream': False, 'max_tokens': 1, 'reasoning_effort': effort,
        }


def validate(settings):
    if not isinstance(settings, dict):
        raise ValueError("executor settings must be an object")
    unknown = set(settings) - _SETTING_KEYS
    if unknown:
        raise ValueError(f"unknown openai-chat settings: {', '.join(sorted(unknown))}")
    parsed = deepcopy(settings)
    if not isinstance(parsed.get("model"), str) or not parsed["model"]:
        raise ValueError("model is required")
    if "base_url" in parsed and "base_urls" in parsed:
        raise ValueError("choose base_url or base_urls")
    urls = parsed.get("base_urls")
    if urls is not None:
        if not isinstance(urls, list) or not urls or any(not isinstance(item, str) for item in urls):
            raise ValueError("base_urls must be a non-empty list of URLs")
    else:
        urls = [parsed.get("base_url")]
    if any(not isinstance(item, str) or
           urlsplit(item).scheme not in ("http", "https") or not urlsplit(item).hostname
           for item in urls):
        raise ValueError("base_url(s) must be absolute http or https URLs")
    block_size = parsed.get("base_url_session_block_size")
    if "base_url_session_block_size" in parsed:
        positive_int(block_size, "base_url_session_block_size")
        if "base_urls" not in parsed:
            raise ValueError("base_url_session_block_size requires base_urls")
    scope = parsed.setdefault("connection_scope", "session")
    if scope not in ("session", "request", "trial"):
        raise ValueError("connection_scope must be session, request or trial")
    for name, default in (
        ("endpoint", "/v1/chat/completions"),
    ):
        value = parsed.setdefault(name, default)
        if not isinstance(value, str) or not value:
            raise TypeError(f"{name} must be a non-empty string")
    for name, default in (("stream", True), ("cache_salt_enabled", False), ("verify_ssl", False)):
        value = parsed.setdefault(name, default)
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a boolean")
    timeout = parsed.setdefault("timeout_seconds", 300.0)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("timeout_seconds must be positive")
    headers = parsed.setdefault("extra_headers", {})
    validate_extra_headers(headers)
    validate_api_keys(parsed)
    dp_rank_count = parsed.get("dp_rank_count")
    positive_int(dp_rank_count, "dp_rank_count", optional=True)
    if dp_rank_count is not None and any(key.lower() == "x-data-parallel-rank" for key in headers):
        raise ValueError("dp_rank_count conflicts with X-Data-Parallel-Rank header")
    validate_extra_body(parsed.setdefault("extra_body", {}), owned={
        "messages", "model", "stream", "max_tokens", "max_completion_tokens",
        "cache_salt", "cache_salt_enabled", "stream_options",
    })
    positive_int(parsed.get("max_tokens"), "max_tokens", optional=True)
    return parsed


@register_plugin(
    stage="executor",
    name="openai-chat",
    description="OpenAI-compatible chat protocol",
    settings_parser=validate,
)
class OpenAIChatExecutor(HttpRequestPlugin):
    supports_start_notifications = True

    async def create_runtime(self, trial, ctx):
        return _ChatRuntime(
            self, self.settings_for_trial(trial), ctx.cancellation, 'openai-chat',
        )

    def settings_for_trial(self, trial: ExecutorTrial):
        settings = thaw_snapshot(self.settings)
        parameters = thaw_snapshot(trial.request_parameters)
        validate_extra_body(parameters, owned={
            "messages", "model", "stream", "max_tokens", "max_completion_tokens",
            "cache_salt", "cache_salt_enabled", "stream_options",
        })
        settings["extra_body"] = {**settings["extra_body"], **parameters}
        settings["_trial_id"] = trial.id
        return settings

    def build_request(self, value, settings):
        return build_chat_request(value, settings, self.request_headers(settings, value), self.cache_salt(value, settings))


__all__ = ["OpenAIChatExecutor"]


class _ChatRuntime(RustHttpExecutorRuntime):
    async def execute_attempt(self, prepared, signal):
        result = await super().execute_attempt(prepared, signal)
        if result.continuation_payload is not None:
            # Decode one retained assistant message once, after transport facts
            # are frozen. SessionRunner owns its body-free summary until close.
            result.continuation_profile = await asyncio.to_thread(
                _continuation_profile, result.continuation_payload)
        return result


def _continuation_profile(payload):
    return message_profile([orjson.loads(payload)])[0]
