"""Bounded, serial sweep selection over committed Trial outcomes."""
from __future__ import annotations

import asyncio
import math
import re
from dataclasses import replace, asdict

from perf_benchmark.core.models import freeze_snapshot
from perf_benchmark.core.validation import positive_int, positive_number
from perf_benchmark.pipeline.contracts.facts import ComparisonPoint, RepetitionPlan, ValidityRequirements
from perf_benchmark.pipeline.contracts.sweep import SloRule, SloCheck, SweepRecord, TrialValidity

_TIME = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1, "m": 60, "h": 3600}


def threshold(value, unit):
    original = value
    if isinstance(value, str):
        match = re.fullmatch(r"([+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)([a-z/]+)", value)
        if match is None:
            raise ValueError(f"invalid SLO threshold: {value!r}")
        number, suffix = match.groups()
        if suffix in _TIME and unit in _TIME:
            value = float(number) * _TIME[suffix] / _TIME[unit]
        elif suffix == unit and unit in {"tok/s", "req/s", "sessions/s"}:
            value = float(number)
        else:
            raise ValueError(f"SLO threshold {original!r} is incompatible with {unit}")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("SLO threshold must be finite and nonnegative")
    return float(value)


def _object(value, allowed, required=()):
    if not isinstance(value, dict) or set(value) - set(allowed) or set(required) - set(value):
        raise ValueError(f"expected settings with fields {sorted(allowed)}; required {sorted(required)}")
    return value


def parse_sweep(settings, parser, axes):
    settings = dict(settings)
    sweep = settings.pop("sweep", None)
    if sweep is None:
        return parser(settings)
    if "validity" in settings:
        raise ValueError("sweep validity must be configured inside sweep")
    sweep = dict(_object(sweep, {"axis", "values", "search", "slo", "validity"}, {"axis"}))
    validity = _object(sweep.setdefault("validity", {}), {"max_scheduler_lag_ms", "min_duration_s", "load_tolerance_ratio"})
    for key, value in validity.items():
        positive_number(value, key, allow_zero=key != "min_duration_s")
    ValidityRequirements(**validity)
    if "search" in sweep and "arrival_ramp" in settings:
        raise ValueError("capacity search requires stationary arrivals; use sweep.values for ramp")
    settings.setdefault("repetitions", 3)
    axis = sweep["axis"]
    if axis not in axes or axis in settings or ("values" in sweep) == ("search" in sweep):
        raise ValueError("sweep requires a supported, non-duplicated axis and exactly one of values/search")
    number = positive_int if axis.startswith("max_") else positive_number
    if "values" in sweep:
        values = sweep["values"]
        if not isinstance(values, list) or not values:
            raise ValueError("sweep.values must be a non-empty list")
        for value in values:
            number(value, axis)
        if len(set(values)) != len(values):
            raise ValueError("sweep.values must be unique")
    else:
        search = _object(sweep["search"], {"min", "max", "initial", "resolution", "max_trials", "timeout_s"},
                         {"min", "max", "initial", "resolution", "max_trials", "timeout_s"})
        for key in ("min", "max", "initial", "resolution"):
            number(search[key], key)
        positive_int(search["max_trials"], "max_trials")
        positive_number(search["timeout_s"], "timeout_s", optional=True)
        if not search["min"] <= search["initial"] <= search["max"]:
            raise ValueError("search requires min <= initial <= max")
        if number is positive_number and search["resolution"] < math.ulp(search["max"]):
            raise ValueError("search resolution is below representable precision")
        values = [search["min"], search["max"], search["initial"]]
        if "slo" not in sweep:
            raise ValueError("search requires SLO rules")
    if "slo" in sweep:
        slo = dict(_object(sweep["slo"], {"window", "min_samples", "rules"}, {"rules"}))
        slo.setdefault("window", "overall")
        positive_int(slo.setdefault("min_samples", 1), "min_samples")
        if not isinstance(slo["rules"], list) or not slo["rules"]:
            raise ValueError("slo.rules must be non-empty")
        for rule in slo["rules"]:
            _object(rule, {"metric", "statistic", "op", "value"}, {"metric", "statistic", "op", "value"})
            if rule["op"] not in {"le", "ge"}:
                raise ValueError("SLO op must be le or ge")
        sweep["slo"] = slo
    for value in values:
        parser({**settings, axis: value})
    return {**parser({**settings, axis: values[0]}), "sweep": sweep}


def resolve_rules(sweep, capabilities):
    slo = sweep["slo"]
    if slo["window"] not in capabilities.windows:
        raise ValueError(f"unsupported SLO window: {slo['window']}")
    catalog = {item.name: item for item in capabilities.metrics}
    rules = []
    for rule in slo["rules"]:
        metric = catalog.get(rule["metric"])
        if metric is None or rule["statistic"] not in metric.statistics:
            raise ValueError(f"unsupported SLO metric/statistic: {rule['metric']}.{rule['statistic']}")
        unit = "samples" if rule["statistic"] in {"n", "eligible_n"} else metric.unit
        rules.append(SloRule(rule["metric"], rule["statistic"], rule["op"], rule["value"],
                             threshold(rule["value"], unit), unit))
    return tuple(rules)


def evaluate(outcome, rules, window, minimum):
    points = {(p.metric, p.statistic): p for p in outcome.points if p.scope == "trial" and p.scope_id == outcome.trial_id and p.window == window}
    checks = []
    for rule in rules:
        point = points.get((rule.metric, rule.statistic))
        reason = "metric_not_reported" if point is None else point.unavailable_reason
        if point is not None:
            if point.availability != "available" or point.value is None or not math.isfinite(point.value):
                reason = reason or "metric_unavailable"
            elif point.unit != rule.unit:
                reason = "unit_mismatch"
            elif rule.statistic not in {"n", "eligible_n"} and (point.valid_sample_count or 0) < minimum:
                reason = "insufficient_samples"
        status = "indeterminate" if reason or point is None or point.value is None else "pass" if (
            point.value <= rule.threshold if rule.op == "le" else point.value >= rule.threshold
        ) else "fail"
        checks.append(SloCheck(rule, outcome.trial_id, point.value if point else None, status, reason,
                               point.valid_sample_count if point else None, point.expected_sample_count if point else None,
                               point.missing_sample_count if point else None))
    return tuple(checks)


def verdict(checks):
    statuses = {check.status for check in checks}
    return "fail" if "fail" in statuses else "indeterminate" if "indeterminate" in statuses else "pass"



class Sweep:
    def __init__(self, scheduler, trials, ctx):
        self.scheduler, self.trials, self.ctx = scheduler, trials, ctx
        self.spec = scheduler.settings["sweep"]
        self.search = self.spec.get("search")
        self.rules = scheduler.slo_rules
        if "slo" in self.spec and not self.rules:
            raise RuntimeError("SLO metrics must be validated before scheduling")
        self.repetitions = scheduler.settings["repetitions"]
        self.used = self.sequence = 0
        self.lower: int | float | None = None
        self.upper: int | float | None = None
        self.samples: dict[int | float, tuple[str, tuple[SloCheck, ...]]] = {}

    async def record(self, phase, status, **kwargs):
        await self.trials.record_sweep(SweepRecord("", self.sequence, self.spec["axis"], phase, status,
                                                 lower=self.lower, upper=self.upper, **kwargs))
        self.sequence += 1

    async def probe(self, value, phase="probe"):
        if self.search and self.used + self.repetitions > self.search["max_trials"]:
            return "budget_exhausted"
        checks, validity = [], []
        group = f"{self.spec['axis']}-{value}-{self.sequence}"
        await self.record("started", "running", value=value, reason=group)
        for index in range(1, self.repetitions + 1):
            if self.ctx.cancellation.is_cancelled():
                return "cancelled"
            plan = self.scheduler.trial_plan({**self.scheduler.settings, self.spec["axis"]: value})
            slo = self.spec.get("slo", {})
            plan = replace(plan, validity_requirements=ValidityRequirements(
                               window=slo.get("window", "overall"), min_samples=slo.get("min_samples", 1), **self.spec["validity"]), name=f"{group}-repetition-{index}",
                           comparison=ComparisonPoint(group, self.spec["axis"], value,
                                                      {"max_concurrency": "requests", "max_active_sessions": "sessions",
                                                       "request_rate": "req/s", "session_rate": "sessions/s"}[self.spec["axis"]]),
                           repetition=RepetitionPlan(index, self.repetitions),
                           metadata={**plan.metadata, "sweep_group": group, "sweep_phase": phase})
            self.used += 1
            execution = await self.scheduler.execute_plan(plan, self.trials, self.ctx)
            outcome = execution.outcome
            if (outcome.validity is not None and outcome.validity.execution_status == "cancelled") or self.ctx.cancellation.is_cancelled():
                return "cancelled"
            slo = self.spec.get("slo", {})
            current = evaluate(outcome, self.rules, slo.get("window", "overall"), slo.get("min_samples", 1))
            checks.extend(current)
            evidence = outcome.validity or TrialValidity(outcome.trial_id, "unknown", "indeterminate", ("validity_not_reported",))
            if evidence.execution_status != "succeeded":
                evidence = replace(evidence, status="indeterminate", reasons=(*evidence.reasons, "execution_" + evidence.execution_status))
            validity.append(evidence)
            status = verdict(current) if evidence.status == "pass" and all(c.status != "indeterminate" for c in current) else "indeterminate"
            await self.record(phase, status, value=value, trial_ids=(outcome.trial_id,), checks=current, validity=(evidence,))
        status = verdict(checks) if all(item.status == "pass" for item in validity) and all(c.status != "indeterminate" for c in checks) else "indeterminate"
        if self.search and any(
            (other < value and old[0] == "fail" and status == "pass") or
            (other > value and old[0] == "pass" and status == "fail")
            for other, old in self.samples.items()
        ):
            return "unstable"
        self.samples[value] = status, tuple(checks)
        return status

    async def run(self):
        task = asyncio.current_task()
        assert task is not None
        expired = False
        def expire():
            nonlocal expired
            expired = True
            self.ctx.cancellation.cancel("slo_search_timeout")
            task.cancel()
        timeout_s = self.search["timeout_s"] if self.search else None
        timer = asyncio.get_running_loop().call_later(timeout_s, expire) if timeout_s is not None else None
        reason = None
        try:
            await self.record("definition", "running", definition=freeze_snapshot({**dict(self.spec), "repetitions": self.repetitions, "normalized_rules": [asdict(r) for r in self.rules]}))
            status = await self.select()
        except asyncio.CancelledError:
            if not expired or task.uncancel():
                await self.record("finished", "cancelled", reason="task_cancelled")
                raise
            status, reason = "budget_exhausted", "timeout"
        except Exception as exc:
            await self.record("finished", "failed", reason=type(exc).__name__)
            raise
        finally:
            if timer is not None:
                timer.cancel()
        await self.record("finished", status, reason=reason or ("max_trials" if status == "budget_exhausted" else status))

    async def select(self):
        if not self.search:
            for value in self.spec["values"]:
                status = await self.probe(value)
                if status == "cancelled":
                    return status
                if status == "pass" and self.rules:
                    self.lower = max(self.lower or value, value)
            return "completed"
        value = self.search["initial"]
        while True:
            status = await self.probe(value)
            if status not in {"pass", "fail"}:
                return status
            if status == "pass":
                self.lower = value
            else:
                self.upper = value
            terminal = (
                "upper_limit_passed" if self.lower == self.search["max"] else
                "no_feasible_point" if self.upper == self.search["min"] else
                "boundary_confirmed" if self.lower is not None and self.upper is not None
                and self.upper - self.lower <= self.search["resolution"] else None
            )
            if terminal:
                endpoints = [(self.lower, "pass"), (self.upper, "fail")]
                endpoints = [(point, expected) for point, expected in endpoints if point is not None]
                if self.used + len(endpoints) * self.repetitions > self.search["max_trials"]:
                    return "budget_exhausted"
                for point, expected in endpoints:
                    result = await self.probe(point, "confirm")
                    if result not in {"pass", "fail"}:
                        return result
                    if result != expected:
                        return "unstable"
                return terminal
            if self.upper is None:
                value = min(value * 2, self.search["max"])
            elif self.lower is None:
                half = value // 2 if self.spec["axis"].startswith("max_") else value / 2
                value = max(half, self.search["min"])
            else:
                value = self.interpolate()

    def interpolate(self):
        low, high = self.lower, self.upper
        assert low is not None and high is not None
        candidates = []
        for rule in self.rules:
            a = [c.value for c in self.samples[low][1] if c.rule == rule and c.status != "indeterminate" and c.value is not None]
            b = [c.value for c in self.samples[high][1] if c.rule == rule and c.status != "indeterminate" and c.value is not None]
            if not a or not b:
                continue
            worst = max if rule.op == "le" else min
            left, right = worst(a), worst(b)
            if left != right and min(left, right) <= rule.threshold <= max(left, right):
                candidates.append(low + (rule.threshold - left) / (right - left) * (high - low))
        value = min(candidates) if candidates else (low + high) / 2
        value = max(low + (high-low)/4, min(high - (high-low)/4, value))
        if self.spec["axis"].startswith("max_"):
            value = min(high - 1, max(low + 1, round(value)))
        if not low < value < high:
            raise ValueError("search resolution is below representable precision")
        return value
