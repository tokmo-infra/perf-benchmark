import math
import re
from collections.abc import Mapping

from perf_benchmark.plugins import register_plugin
from perf_benchmark.pipeline.contracts.metrics import MetricCapability, MetricsCapabilities
from perf_benchmark.plugins.metrics.standard.catalog import (
    METRIC_CATALOG, OPTIONAL_METRICS, MAXTPM_METRICS, REVENUE_METRICS,
)
from perf_benchmark.plugins.metrics.standard.windows import WINDOW_ORDER
from perf_benchmark.plugins.metrics.standard.backend import StandardMetricsBackend
from perf_benchmark.plugins.metrics.standard.billing import parse_pricing
from perf_benchmark.pipeline.scheduling.sweep import threshold


_GOODPUT_METRICS = frozenset({"ttft", "tpot", "e2e"})


def _parse_goodput(value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError("metrics goodput settings must be an object")
    unknown = set(value) - {"rules"}
    if unknown:
        raise ValueError(f"unknown metrics goodput setting: {sorted(unknown)[0]}")
    rules = value.get("rules")
    if not isinstance(rules, list) or not rules:
        raise ValueError("metrics goodput.rules must be a non-empty list")
    parsed = []
    for rule in rules:
        if not isinstance(rule, dict) or set(rule) != {"metric", "op", "value"}:
            raise ValueError("goodput rule requires exactly metric, op and value")
        metric = rule["metric"]
        if not isinstance(metric, str) or metric not in _GOODPUT_METRICS:
            raise ValueError(f"unsupported goodput metric: {metric}")
        op = rule["op"]
        if not isinstance(op, str) or op not in {"le", "ge"}:
            raise ValueError("goodput rule op must be le or ge")
        unit = METRIC_CATALOG[metric].unit
        parsed.append(
            {
                "metric": metric,
                "op": op,
                "value": rule["value"],
                "threshold": threshold(rule["value"], unit),
                "unit": unit,
            }
        )
    return {"rules": parsed}


def parse_settings(settings: dict) -> dict:
    unknown = set(settings) - {"memory_limit", "temp_directory", "steady", "goodput", "maxtpm", "pricing"}
    if unknown:
        raise ValueError(f"unknown standard metrics setting: {sorted(unknown)[0]}")
    memory = settings.get("memory_limit", "512MB")
    if not isinstance(memory, str) or not re.fullmatch(r"[1-9][0-9]*(?:KB|MB|GB)", memory.upper()):
        raise ValueError("metrics memory_limit must use KB, MB or GB")
    temporary = settings.get("temp_directory")
    if temporary is not None and (not isinstance(temporary, str) or not temporary):
        raise ValueError("metrics temp_directory must be a non-empty path")
    steady = settings.get("steady", {})
    if not isinstance(steady, dict):
        raise ValueError("metrics steady settings must be an object")
    unknown_steady = set(steady) - {"min_ratio", "min_duration_s"}
    if unknown_steady:
        raise ValueError(f"unknown metrics steady setting: {sorted(unknown_steady)[0]}")
    ratio = steady.get("min_ratio", 0.9)
    duration = steady.get("min_duration_s", 1.0)
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio):
        raise ValueError("steady.min_ratio must be in (0, 1]")
    if not 0 < ratio <= 1:
        raise ValueError("steady.min_ratio must be in (0, 1]")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or duration <= 0
    ):
        raise ValueError("steady.min_duration_s must be finite and positive")
    parsed = {"memory_limit": memory.upper(), "steady": {"min_ratio": float(ratio), "min_duration_s": float(duration)}}
    if temporary is not None:
        parsed["temp_directory"] = temporary
    if "goodput" in settings:
        parsed["goodput"] = _parse_goodput(settings["goodput"])
    maxtpm = settings.get("maxtpm", False)
    if not isinstance(maxtpm, bool):
        raise ValueError("metrics maxtpm must be boolean")
    parsed["maxtpm"] = maxtpm
    if "pricing" in settings:
        parsed["pricing"] = parse_pricing(settings["pricing"])
    return parsed


@register_plugin(
    stage="metrics",
    name="standard",
    description="Run-level five-window metrics and repetition aggregates",
    settings_parser=parse_settings,
)
class StandardMetrics:
    def __init__(self, settings: Mapping[str, object]) -> None:
        self.settings = settings

    def describe(self) -> MetricsCapabilities:
        pricing = self.settings.get("pricing")
        maxtpm = self.settings.get("maxtpm")
        def enabled(d):
            if d.name in REVENUE_METRICS:
                return maxtpm and pricing is not None
            if d.name in MAXTPM_METRICS:
                return maxtpm
            return d.name not in OPTIONAL_METRICS or pricing is not None

        def unit(d):
            if pricing and d.unit.startswith("currency"):
                return d.unit.replace("currency", pricing.get("currency", "USD"))
            return d.unit

        return MetricsCapabilities(
            tuple(MetricCapability(d.name, unit(d), d.statistics)
                  for d in METRIC_CATALOG.values() if enabled(d)),
            (*WINDOW_ORDER, *(("maxtpm",) if self.settings.get("maxtpm") else ())),
        )

    def compare(self, baseline, candidate, specification):
        from .comparison import compare
        return compare(baseline, candidate, specification)

    def create_backend(self, *, seed: int) -> StandardMetricsBackend:
        return StandardMetricsBackend(self.settings, seed=seed)


__all__ = ["StandardMetrics", "parse_settings"]
