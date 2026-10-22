"""Global Scheduler probability policies; no dataset or protocol configuration."""

from collections.abc import Mapping
import math
import random

from perf_benchmark.core.models import Turn
from perf_benchmark.core.validation import positive_int, positive_number


def parse_scenarios(settings: dict) -> None:
    retry = settings.setdefault("retry", {})
    cancel = settings.get("cancel")
    for name, policy, extra in (
        ("retry", retry, {"on_status", "max_attempts", "delay_s"}),
        ("cancel", cancel, {"after_outputs", "after_first_output_ms"}),
    ):
        if name == "cancel" and "cancel" not in settings:
            continue
        if not isinstance(policy, dict):
            raise ValueError(f"{name} must be an object")
        if unknown := set(policy) - {"model", "probability"} - extra:
            raise ValueError(f"unknown {name} settings: {sorted(unknown)}")
        if policy.setdefault("model", "constant") != "constant":
            raise ValueError(f"{name}.model must be constant")
        probability = policy.setdefault("probability", 1.0)
        if (type(probability) not in (int, float) or not math.isfinite(probability)
                or not 0 <= probability <= 1):
            raise ValueError(f"{name}.probability must be between 0 and 1")
    positive_int(retry.setdefault("max_attempts", 1), "retry.max_attempts")
    positive_number(retry.setdefault("delay_s", 0.2), "retry.delay_s", allow_zero=True)
    statuses = retry.setdefault("on_status", ["failed"])
    if (not isinstance(statuses, list) or not statuses
            or any(status not in ("succeeded", "failed") for status in statuses)
            or len(set(statuses)) != len(statuses)):
        raise ValueError("retry.on_status must list succeeded and/or failed without duplicates")
    if cancel is None:
        return
    triggers = set(cancel) & {"after_outputs", "after_first_output_ms"}
    if len(triggers) != 1:
        raise ValueError("cancel requires exactly one output trigger")
    if set(settings.get("control", {})) & {"after_outputs", "after_first_output_ms"}:
        raise ValueError("cancel and control cannot both configure output triggers")
    if "after_outputs" in cancel:
        threshold = cancel["after_outputs"]
        if isinstance(threshold, dict):
            if set(threshold) != {"min", "max"}:
                raise ValueError("cancel.after_outputs range requires min and max")
            for key in ("min", "max"):
                positive_int(threshold[key], f"cancel.after_outputs.{key}")
            if threshold["min"] > threshold["max"]:
                raise ValueError("cancel.after_outputs.min must not exceed max")
        else:
            positive_int(threshold, "cancel.after_outputs")
    else:
        positive_number(cancel["after_first_output_ms"], "cancel.after_first_output_ms",
                        allow_zero=True)


def scenario_random(
    seed: int, turn: Turn, attempt: int, purpose: str, session_ordinal: int,
) -> random.Random:
    # Runtime ordinals distinguish cycle occurrences without Run IDs or shared RNG state.
    return random.Random(f"{seed}:{purpose}:{session_ordinal}:{turn.ordinal}:{attempt}")


def selected(
    policy: Mapping, turn: Turn, attempt: int, purpose: str, session_ordinal: int, seed: int,
) -> bool:
    probability = policy["probability"]
    if probability == 0 or probability == 1:
        return bool(probability)
    return scenario_random(seed, turn, attempt, purpose, session_ordinal).random() < probability


def cancellation_trigger(
    policy: Mapping, turn: Turn, attempt: int, session_ordinal: int, seed: int,
) -> dict:
    if not selected(policy, turn, attempt, "cancel", session_ordinal, seed):
        return {}
    if "after_first_output_ms" in policy:
        return {"after_first_output_ms": policy["after_first_output_ms"], "action": "cancel"}
    threshold = policy["after_outputs"]
    if isinstance(threshold, Mapping):
        threshold = scenario_random(seed, turn, attempt, "cancel-threshold", session_ordinal).randint(
            threshold["min"], threshold["max"],
        )
    return {"after_outputs": threshold, "action": "cancel"}
