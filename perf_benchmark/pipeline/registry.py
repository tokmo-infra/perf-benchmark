from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
import logging
from types import MappingProxyType
from typing import Any, cast, TypeVar

from perf_benchmark.core.errors import PluginNotFound
from perf_benchmark.core.models import freeze_snapshot

logger = logging.getLogger("perf_benchmark.pipeline.registry")
FactoryT = TypeVar("FactoryT", bound=Callable[[object], object])
SettingsParser = Callable[[dict[str, Any]], object]


@dataclass(frozen=True, slots=True)
class PluginSpec:
    name: str
    stage: str
    factory: Callable[[object], object]
    description: str
    settings_parser: SettingsParser


@dataclass(frozen=True, slots=True)
class StageBinding:
    spec: PluginSpec
    settings: object
    plugin: object

    @property
    def name(self) -> str:
        return self.spec.name


@dataclass(frozen=True, slots=True)
class ExporterBindings:
    bindings: tuple[StageBinding, ...]

    @property
    def name(self) -> str:
        return ",".join(binding.name for binding in self.bindings)

    @property
    def plugin(self) -> tuple[object, ...]:
        return tuple(binding.plugin for binding in self.bindings)




class StageRegistry:
    STAGES = (
        "loader", "normalization", "selector", "executor",
        "scheduler", "metrics", "saver", "exporter",
    )

    def __init__(self) -> None:
        self._specs: dict[tuple[str, str], PluginSpec] = {}

    def register(
        self,
        *,
        stage: str,
        name: str,
        description: str,
        settings_parser: SettingsParser,
        replace: bool = False,
    ) -> Callable[[FactoryT], FactoryT]:
        """Register a class or factory without constructing or wrapping it."""
        def decorate(factory: FactoryT) -> FactoryT:
            if stage not in self.STAGES:
                raise ValueError(f"unknown stage: {stage}")
            key = (stage, name)
            if key in self._specs and not replace:
                raise ValueError(f"plugin already registered: {stage}:{name}")
            self._specs[key] = PluginSpec(name, stage, factory, description, settings_parser)
            logger.debug(
                "registered stage plugin stage=%s plugin=%s replace=%s", stage, name, replace,
            )
            return factory

        return decorate

    def copy(self) -> StageRegistry:
        """Copy the directory while sharing immutable plugin definitions."""
        registry = StageRegistry()
        registry._specs = self._specs.copy()
        return registry

    def resolve(self, stage: str, name: str) -> PluginSpec:
        try:
            return self._specs[(stage, name)]
        except KeyError as exc:
            raise PluginNotFound(f"plugin not found: {stage}:{name}") from exc

    def bind(self, stage: str, name: str, settings: dict[str, Any]) -> StageBinding:
        spec = self.resolve(stage, name)
        parsed = freeze_snapshot(spec.settings_parser(deepcopy(settings)))
        plugin = spec.factory(parsed)
        logger.debug(
            "bound stage plugin stage=%s plugin=%s implementation=%s",
            stage, name, type(plugin).__name__,
        )
        return StageBinding(spec, parsed, plugin)

    def bind_pipeline(
        self,
        pipeline: Mapping[str, object],
    ) -> Mapping[str, StageBinding | ExporterBindings]:
        unknown = set(pipeline) - set(self.STAGES)
        if unknown:
            raise ValueError(f"unknown pipeline stages: {', '.join(sorted(unknown))}")
        parsed: list[tuple[PluginSpec, object]] = []
        for stage in self.STAGES[:-1]:
            selected = pipeline.get(stage)
            if selected is None:
                raise ValueError(f"pipeline.{stage}.plugin is required")
            if not isinstance(selected, Mapping):
                raise ValueError(f"pipeline.{stage} must be an object")
            name = selected.get("plugin")
            if not isinstance(name, str) or not name:
                raise ValueError(f"pipeline.{stage}.plugin is required")
            raw_settings = selected.get("settings", {})
            if not isinstance(raw_settings, dict):
                raise ValueError(f"pipeline.{stage}.settings must be an object")
            spec = self.resolve(stage, name)
            parsed.append((spec, freeze_snapshot(spec.settings_parser(deepcopy(raw_settings)))))
        exporters = pipeline.get("exporter")
        if exporters is None:
            raise ValueError("pipeline.exporter is required")
        if not isinstance(exporters, Mapping):
            raise ValueError("pipeline.exporter must be an object")
        unknown = set(exporters) - {"plugins"}
        if unknown:
            raise ValueError(f"unknown pipeline.exporter fields: {', '.join(sorted(unknown))}")
        configured = exporters.get("plugins", [])
        if not isinstance(configured, list):
            raise ValueError("pipeline.exporter.plugins must be a list")
        selections: list[Mapping[str, object]] = [{"plugin": "console", "settings": {}}]
        seen = {"console"}
        for index, selected in enumerate(configured):
            if not isinstance(selected, Mapping):
                raise ValueError(f"pipeline.exporter.plugins[{index}] must be an object")
            unknown = set(selected) - {"plugin", "settings"}
            if unknown:
                raise ValueError(
                    f"unknown pipeline.exporter.plugins[{index}] fields: "
                    f"{', '.join(sorted(unknown))}"
                )
            name = selected.get("plugin")
            if not isinstance(name, str) or not name:
                raise ValueError(f"pipeline.exporter.plugins[{index}].plugin is required")
            if name in seen:
                raise ValueError(f"duplicate exporter plugin: {name}")
            seen.add(name)
            selections.append(selected)
        if "tabular" not in seen:
            selections.append({"plugin": "tabular", "settings": {}})
        parsed_exporters: list[tuple[PluginSpec, object]] = []
        for index, selected in enumerate(selections):
            name = cast(str, selected["plugin"])
            raw_settings = selected.get("settings", {})
            if not isinstance(raw_settings, dict):
                path = "console" if index == 0 else f"plugins[{index - 1}]"
                raise ValueError(f"pipeline.exporter.{path}.settings must be an object")
            spec = self.resolve("exporter", name)
            parsed_exporters.append(
                (spec, freeze_snapshot(spec.settings_parser(deepcopy(raw_settings))))
            )
        bound: dict[str, StageBinding | ExporterBindings] = {
            spec.stage: StageBinding(spec, settings, spec.factory(settings))
            for spec, settings in parsed
        }
        bound["exporter"] = ExporterBindings(tuple(
            StageBinding(spec, settings, spec.factory(settings))
            for spec, settings in parsed_exporters
        ))
        return MappingProxyType(bound)

    def names(self, stage: str) -> tuple[str, ...]:
        return tuple(sorted(name for item_stage, name in self._specs if item_stage == stage))
