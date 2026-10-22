"""Request construction, bounded preparation and body ownership for Rust executors."""

from __future__ import annotations

from abc import abstractmethod
import asyncio
import base64
import hmac
import json
import os
import random
import secrets
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from itertools import accumulate
from typing import TYPE_CHECKING

import orjson

if TYPE_CHECKING:
    from perf_benchmark._native import PreparedBody as NativeBody

from perf_benchmark.pipeline.contracts.executor import (
    AttemptInput,
    ExecutorPlugin,
    ExecutorTrial,
    TurnRequestFacts,
)
from perf_benchmark.pipeline.resources import finish_task
from perf_benchmark.core import validation


def validate_api_keys(settings: dict) -> None:
    sampling = settings.setdefault("api_key_sampling", "uniform")
    if sampling not in ("uniform", "weighted"):
        raise ValueError("api_key_sampling must be uniform or weighted")
    entries = settings.setdefault("api_keys", [])
    if not isinstance(entries, list):
        raise ValueError("api_keys must be a list")
    normalized, names = [], set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) - {"env", "weight"}:
            raise ValueError(f"api_keys[{index}] must contain only env and optional weight")
        name = entry.get("env")
        if not isinstance(name, str) or not name.strip() or "\x00" in name or "=" in name:
            raise ValueError(f"api_keys[{index}].env must be a nonempty environment variable name")
        if name in names:
            raise ValueError("api_keys contains duplicate environment variable names")
        names.add(name)
        item = {"env": name}
        if sampling == "uniform":
            if "weight" in entry:
                raise ValueError("api_keys weights require api_key_sampling=weighted")
        else:
            item["weight"] = validation.positive_number(
                entry.get("weight", 1), f"api_keys[{index}].weight", allow_zero=True)
        normalized.append(item)
    if sampling == "weighted":
        validation.positive_number(sum(item["weight"] for item in normalized), "api_keys weights total")
    if normalized and any(name.lower() == "authorization" for name in settings.get("extra_headers", {})):
        raise ValueError("extra_headers conflicts with managed Authorization header")
    settings["api_keys"] = normalized


def validate_extra_body(value: object, *, owned: set[str]) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("extra_body must be a JSON object with string keys")
    if conflict := set(value) & owned:
        raise ValueError(f"extra_body conflicts with generated fields: {', '.join(sorted(conflict))}")
    try:
        encoded = json.dumps(value, allow_nan=False, ensure_ascii=False).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError("extra_body must contain finite JSON values") from exc
    if len(encoded) > 64 * 1024:
        raise ValueError("extra_body must not exceed 64 KiB")
    return value


def merge_extra_body(base: dict, extra: Mapping[str, object]) -> dict:
    if conflict := base.keys() & extra.keys():
        raise ValueError(f"extra_body conflicts with generated fields: {', '.join(sorted(conflict))}")
    return {**base, **extra}


@dataclass(frozen=True, slots=True)
class PreparedBody:
    segments: tuple[bytes | memoryview, ...]
    content_length: int
    content_type: str = "application/json"

    def release(self) -> None:
        for segment in self.segments:
            if isinstance(segment, memoryview):
                segment.release()


@dataclass(slots=True)
class HttpRequestSpec:
    url: str
    parameters: dict
    build_body: Callable[[HttpRequestSpec], Awaitable[PreparedBody]]
    headers: dict[str, str] = field(default_factory=dict)
    method: str = "POST"
    retain_output: bool = False
    streaming: bool = False
    request_facts: TurnRequestFacts | None = None
    response_expectations: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class PreparedRequest:
    input: AttemptInput
    spec: HttpRequestSpec
    body: PreparedBody
    request_facts: TurnRequestFacts

    native_body: NativeBody | None = None
    native_attempt: object | None = None
    release_task: asyncio.Task | None = None
    release_done: bool = False

    async def close(self):
        if self.native_body is not None:
            self.native_body.release()
        self.body.release()
        self.native_attempt = None


class HttpRequestPlugin(ExecutorPlugin):
    def __init__(self, settings: Mapping[str, object]) -> None:
        self.settings = settings
        self._cache_salt_key = secrets.token_bytes(32)
        keys = []
        for entry in settings.get("api_keys", ()):
            key = os.getenv(entry["env"])
            if key is None or not key.strip() or "\r" in key or "\n" in key:
                raise ValueError("configured API key environment variable must contain a nonempty single-line value")
            keys.append(key)
        self._api_keys = tuple(keys)
        self._api_key_cumulative_weights = (
            tuple(accumulate(entry["weight"] for entry in settings["api_keys"]))
            if settings.get("api_key_sampling") == "weighted" else None
        )

    def request_headers(
        self, settings: Mapping[str, object], value: AttemptInput | None = None,
    ) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **settings.get("extra_headers", {})}
        if value is not None and value.turn.extra_headers_payload_ref is not None:
            with value.payloads.view(value.turn.extra_headers_payload_ref) as payload:
                extra = orjson.loads(payload)
            validation.validate_extra_headers(extra)
            if {name.lower() for name in extra} & {name.lower() for name in headers}:
                raise ValueError("dataset headers conflict with executor extra_headers")
            headers.update(extra)
        if self._api_keys:
            if any(name.lower() == "authorization" for name in headers):
                raise ValueError("extra_headers conflicts with managed Authorization header")
            key = (
                random.choice(self._api_keys) if self._api_key_cumulative_weights is None
                else random.choices(self._api_keys, cum_weights=self._api_key_cumulative_weights, k=1)[0]
            )
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def cache_salt(self, value: AttemptInput, settings: Mapping[str, object]) -> str | None:
        if not settings.get("cache_salt_enabled", False):
            return None
        phase = "warmup" if value.session.metadata.get("warmup") is True else "formal"
        cycle = value.session.metadata.get("cycle", 0) if phase == "formal" else 0
        identity = f"{settings['_trial_id']}:{phase}:{cycle}".encode()
        digest = hmac.digest(self._cache_salt_key, identity, "sha256")
        return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")

    def validate_observation(self):
        headers = self.settings.get("extra_headers", {})
        if not isinstance(headers, Mapping):
            raise TypeError("Executor headers must be a mapping")
        if any(key.lower() in {"traceparent", "tracestate"} for key in headers):
            raise ValueError("managed tracing conflicts with Executor trace headers")

    @abstractmethod
    def settings_for_trial(self, trial: ExecutorTrial) -> Mapping[str, object]: ...

    @abstractmethod
    def build_request(
        self, value: AttemptInput, settings: Mapping[str, object]
    ) -> HttpRequestSpec: ...


class RequestRuntime:
    def __init__(self, plugin, settings, cancellation):
        self.plugin, self.settings = plugin, dict(settings)
        self.cancellation = cancellation
        self._preparations = set()
        self._preparation_error = None
        self._prepared = {}
        self._close_task = None

    async def prepare_attempt(self, value: AttemptInput):
        if self._close_task is not None:
            raise RuntimeError("HTTP runtime is closed")
        spec = self.plugin.build_request(value, self.settings)
        self.propagate_trace(value, spec)
        spec.retain_output = value.turn.context_mode == "live" and not value.is_final_turn
        task = asyncio.create_task(self._prepare(value, spec))
        self._preparations.add(task)
        task.add_done_callback(self._prepared_done)
        try:
            return await asyncio.shield(task)
        except Exception as exc:
            if self._preparation_error is exc:
                self._preparation_error = None
            raise

    def propagate_trace(self, value, spec):
        if value.trace_context is not None:
            if any(key.lower() in {"traceparent", "tracestate"} for key in spec.headers):
                raise ValueError("managed tracing conflicts with Executor trace headers")
            spec.headers = {**spec.headers, "traceparent": value.trace_context.traceparent}

    def _prepared_done(self, task):
        self._preparations.discard(task)
        if not task.cancelled():
            error = task.exception()
            self._preparation_error = self._preparation_error or error

    async def _prepare(self, value, spec):
        body = await spec.build_body(spec)
        if not isinstance(body, PreparedBody):
            raise TypeError("Rust execution requires a segmented PreparedBody")
        request_facts = spec.request_facts or TurnRequestFacts(
            value.turn.requested_output_tokens
        )
        prepared = PreparedRequest(value, spec, body, request_facts)
        self._prepared[id(prepared)] = prepared
        return prepared

    async def release(self, prepared):
        if self._prepared.pop(id(prepared), None) is not None:
            await prepared.close()

    async def close(self):
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        await finish_task(self._close_task)

    async def _close(self):
        await asyncio.gather(*tuple(self._preparations), return_exceptions=True)
        for prepared in tuple(self._prepared.values()):
            await self.release(prepared)
        if self._preparation_error is not None:
            raise self._preparation_error
