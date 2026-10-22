"""Optional configured token charges; server usage is the only billing evidence."""

from collections import defaultdict
from collections.abc import Mapping
from decimal import Decimal
from dataclasses import replace
import math

from perf_benchmark.pipeline.results.measurement import AttemptFact
from perf_benchmark.plugins.metrics.standard.catalog import METRIC_CATALOG
from perf_benchmark.plugins.metrics.standard.formulas import integer, scalar

COST_NAMES = ("input_cost", "cached_input_cost", "output_cost", "total_cost")


def parse_pricing(value):
    required = {"input_price", "output_price", "cached_input_price"}
    if not isinstance(value, Mapping) or set(value) - required - {"currency", "per_tokens", "source"}:
        raise ValueError(
            "pricing accepts input_price, output_price, cached_input_price, currency, per_tokens, source"
        )
    if not required <= set(value):
        raise ValueError("pricing requires input_price, output_price and cached_input_price")
    parsed = dict(value)
    for name in required:
        rate = parsed[name]
        if (
            isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not math.isfinite(rate)
            or rate < 0
        ):
            raise ValueError(f"pricing.{name} must be finite and nonnegative")
    size = parsed.setdefault("per_tokens", 1_000_000)
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("pricing.per_tokens must be a positive integer")
    currency = parsed.setdefault("currency", "USD")
    if not isinstance(currency, str) or not currency.strip():
        raise ValueError("pricing.currency must be a non-empty string")
    if "source" in parsed and (not isinstance(parsed["source"], str) or not parsed["source"].strip()):
        raise ValueError("pricing.source must be a non-empty string")
    return parsed


class Billing:
    def __init__(self, pricing):
        self.pricing = parse_pricing(pricing) if pricing is not None else None
        self.totals = defaultdict(lambda: {name: Decimal(0) for name in COST_NAMES})
        self.missing = defaultdict(lambda: {name: 0 for name in COST_NAMES})
        self.counts = defaultdict(lambda: [0, 0])
        self.known_cost = defaultdict(lambda: Decimal(0))
        self.known_cost_attempts = defaultdict(int)

    def record(self, fact):
        if self.pricing is None:
            return {}
        p = self.pricing
        prompt, cached, output = (
            integer(fact.usage.get(k))
            for k in ("prompt_tokens", "cached_tokens", "completion_tokens")
        )
        valid_prompt = prompt is not None and prompt >= 0
        valid_output = output is not None and output >= 0
        valid_cache = valid_prompt and cached is not None and 0 <= cached <= prompt

        def charge(tokens, key):
            return Decimal(tokens) * Decimal(str(p[key])) / Decimal(p["per_tokens"])

        values = {
            "input_cost": charge(prompt - cached, "input_price") if valid_cache else None,
            "cached_input_cost": charge(cached, "cached_input_price") if valid_cache else None,
            "output_cost": charge(output, "output_price") if valid_output else None,
        }
        input_total = (
            values["input_cost"] + values["cached_input_cost"]
            if valid_cache
            else charge(prompt, "input_price")
            if valid_prompt and cached is None and p["input_price"] == p["cached_input_price"]
            else None
        )
        values["total_cost"] = (
            input_total + values["output_cost"]
            if input_total is not None and valid_output
            else None
        )
        reason = (
            "cache_usage_not_reported"
            if valid_prompt and valid_output and cached is None
            else "usage_missing_or_invalid"
        )
        if isinstance(fact, AttemptFact) and fact.request_ns is not None:
            state = self.counts[fact.trial_id]
            state[values["total_cost"] is None] += 1
            components = [value for value in (input_total, values["output_cost"]) if value is not None]
            if components:
                self.known_cost[fact.trial_id] += sum(components)
                self.known_cost_attempts[fact.trial_id] += 1
            for name, value in values.items():
                if value is None:
                    self.missing[fact.trial_id][name] += 1
                else:
                    self.totals[fact.trial_id][name] += value
        return {
            name: replace(
                scalar(
                    METRIC_CATALOG[name],
                    float(value) if value is not None else None,
                    source="configured_price+server_usage",
                    reason=reason,
                ),
                unit=p["currency"],
            )
            for name, value in values.items()
        }

    def finalize_trial(self, backend, trial_id):
        if self.pricing is None:
            return ()
        totals, missing = self.totals[trial_id], self.missing[trial_id]
        observed, unknown = self.counts[trial_id]
        attempts = observed + unknown
        values = {
            name: float(totals[name]) if not missing[name] or missing[name] < attempts else None
            for name in COST_NAMES
        }
        if missing["total_cost"] and self.known_cost_attempts[trial_id]:
            values["total_cost"] = float(self.known_cost[trial_id])
        values.update(
            observed_cost=float(totals["total_cost"]),
            cost_observed_attempts=observed,
            cost_unavailable_attempts=unknown,
        )
        return tuple(
            replace(
                backend._point(
                    trial_id,
                    "overall",
                    name,
                    "value",
                    value,
                    "unavailable" if value is None else "partial"
                    if name in COST_NAMES and missing[name] else "available",
                    "attempt_usage_incomplete" if value is None
                    or name in COST_NAMES and missing[name] else None,
                    "configured_price+server_usage",
                ),
                formula_version="token-price-v2-observed-totals",
                unit=self.pricing["currency"]
                if name in {*COST_NAMES, "observed_cost"}
                else "attempts",
            )
            for name, value in values.items()
        )

    def release(self, trial_id):
        for mapping in (self.totals, self.missing, self.counts, self.known_cost, self.known_cost_attempts):
            mapping.pop(trial_id, None)
