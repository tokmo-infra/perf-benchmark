"""Hook contracts, registration and startup loading."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import entry_points
import json
import logging
from pathlib import Path
from typing import Any, Callable, Protocol

import yaml

from perf_benchmark.core.errors import HookNotFound
from perf_benchmark.core.lifecycle import LifecycleEvent

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HookContext:
    run_id: str
    run_name: str
    base_dir: Path
    stage: str | None = None

    def for_stage(self, stage: str | None) -> HookContext:
        return HookContext(self.run_id, self.run_name, self.base_dir, stage)


class LifecycleHook(Protocol):
    async def handle(
        self,
        event: LifecycleEvent,
        payload: object,
        ctx: HookContext,
    ) -> object | None: ...


@dataclass(frozen=True, slots=True)
class HookSpec:
    name: str
    entrypoint: Callable[[dict], LifecycleHook]
    description: str
    settings_schema: Callable[[dict], dict] | None = None


class HookRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, HookSpec] = {}

    def register(self, spec: HookSpec, *, replace: bool = False) -> None:
        if spec.name in self._specs and not replace:
            raise ValueError(f"hook already registered: {spec.name}")
        self._specs[spec.name] = spec
        logger.debug("registered hook=%s replace=%s", spec.name, replace)

    def resolve(self, name: str) -> HookSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise HookNotFound(f"hook not found: {name}") from exc

    def create(self, name: str, settings: dict) -> LifecycleHook:
        spec = self.resolve(name)
        if spec.settings_schema is not None:
            spec.settings_schema(settings)
        hook = spec.entrypoint(dict(settings))
        logger.debug("created hook=%s implementation=%s", name, type(hook).__name__)
        return hook

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))


def create_registry() -> HookRegistry:
    from perf_benchmark.hooks.request_parameters import register

    registry = HookRegistry()
    register(registry)
    load_entrypoints(registry)
    return registry


def require_request_parameters(value: object) -> dict:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError("request_parameters must be an object with string keys")
    forbidden = set(value) & {
        "messages", "input", "prompt", "query", "documents", "audio", "file",
        "headers", "extra_headers", "extra_body", "api_key", "api_keys", "api_key_sampling",
        "base_url", "endpoint", "model", "stream", "max_tokens", "max_completion_tokens",
        "cache_salt", "cache_salt_enabled", "stream_options", "encoding_format",
        "return_documents", "store",
    }
    if forbidden:
        raise ValueError(f"request_parameters cannot contain: {', '.join(sorted(forbidden))}")
    encoded = json.dumps(value, allow_nan=False, ensure_ascii=False).encode()
    if len(encoded) > 64 * 1024:
        raise ValueError("request_parameters must not exceed 64 KiB")
    return json.loads(encoded)


def load_entrypoints(registry: HookRegistry) -> None:
    for entrypoint in entry_points(group="perf_benchmark.hooks"):
        entrypoint.load()(registry)


def load_module(registry: HookRegistry, module_name: str) -> None:
    register = getattr(import_module(module_name), "register", None)
    if not callable(register):
        raise TypeError(f"{module_name} must define register(registry)")
    register(registry)


def load_manifest(registry: HookRegistry, path: Path) -> None:
    with path.open("r", encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    modules = manifest.get("modules") if isinstance(manifest, dict) else None
    if not isinstance(modules, list) or not all(isinstance(item, str) for item in modules):
        raise TypeError(f"{path}: modules must be a list of import paths")
    for module_name in modules:
        load_module(registry, module_name)


def load_configured(registry: HookRegistry, value: Any, base_dir: Path) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise TypeError("hooks must be an object")
    modules = value.get("modules", [])
    manifests = value.get("manifests", [])
    if not isinstance(modules, list) or not all(isinstance(item, str) for item in modules):
        raise TypeError("hooks.modules must be a list of import paths")
    if not isinstance(manifests, list) or not all(isinstance(item, str) for item in manifests):
        raise TypeError("hooks.manifests must be a list of paths")
    for module_name in modules:
        load_module(registry, module_name)
    for manifest in manifests:
        path = Path(manifest).expanduser()
        load_manifest(registry, path if path.is_absolute() else base_dir / path)
