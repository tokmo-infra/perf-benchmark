from __future__ import annotations

import hashlib
import random
from statistics import fmean, median, pstdev, quantiles

REPETITION_STATISTICS = (
    "median",
    "mean",
    "mad",
    "cv",
    "min",
    "max",
    "trimmed_mean_10pct",
    "bootstrap_median_ci_lower",
    "bootstrap_median_ci_upper",
)


def repetition_statistics(
    samples: list[float],
    identity: str,
    *, seed: int,
) -> dict[str, float | None]:
    ordered = sorted(samples)
    center = median(ordered)
    mean = fmean(ordered)
    cut = int(len(ordered) * 0.1)
    retained = ordered[cut:len(ordered) - cut] or ordered
    result: dict[str, float | None] = {
        "median": center,
        "mean": mean,
        "mad": median(abs(value - center) for value in ordered),
        "cv": pstdev(ordered) / abs(mean) if mean else 0.0,
        "min": ordered[0],
        "max": ordered[-1],
        "trimmed_mean_10pct": fmean(retained),
    }
    if len(ordered) < 2:
        result["bootstrap_median_ci_lower"] = result["bootstrap_median_ci_upper"] = None
        return result
    rng = random.Random(hashlib.sha256(f"{seed}:bootstrap:{identity}".encode()).digest())
    estimates = sorted(
        median(rng.choice(ordered) for _ in ordered)
        for _ in range(2000)
    )
    confidence = quantiles(estimates, n=40, method="inclusive")
    result["bootstrap_median_ci_lower"] = confidence[0]
    result["bootstrap_median_ci_upper"] = confidence[-1]
    return result


__all__ = ["REPETITION_STATISTICS", "repetition_statistics"]
