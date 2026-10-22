from __future__ import annotations

from dataclasses import dataclass
import math

WINDOW_ORDER = (
    "overall",
    "last30s",
    "drop_first_20pct",
    "steady_occupancy",
    "steady_strip_10pct",
)


@dataclass(frozen=True, slots=True)
class MeasurementWindow:
    name: str
    start_ns: int | None
    end_ns: int | None
    unavailable_reason: str | None = None

    @property
    def available(self) -> bool:
        return (
            self.start_ns is not None
            and self.end_ns is not None
            and self.end_ns > self.start_ns
        )

    @property
    def duration_s(self) -> float | None:
        if self.start_ns is None or self.end_ns is None:
            return None
        return (self.end_ns - self.start_ns) / 1e9


def measurement_windows(
    bounds: tuple[int | None, int | None],
    occupancy_rows: list[tuple[int, int]],
    target_concurrency: int | None,
    *,
    min_ratio: float,
    min_duration_s: float,
) -> tuple[tuple[MeasurementWindow, ...], dict[str, float | int | str | None]]:
    start_ns, end_ns = bounds
    if start_ns is None or end_ns is None or end_ns <= start_ns:
        windows = tuple(
            MeasurementWindow(name, None, None, "measurement_window_unavailable")
            for name in WINDOW_ORDER
        )
        return windows, {
            "steady_occupancy_threshold": (
                math.ceil(target_concurrency * min_ratio) if target_concurrency is not None else None
            ),
            "max_observed_occupancy": 0,
            "underfill_duration_s": 0.0,
            "steady_reason": "measurement_window_unavailable",
        }

    span_ns = end_ns - start_ns
    if target_concurrency is None:
        steady = MeasurementWindow("steady_occupancy", None, None, "unbounded_by_scheduler")
        diagnostics = {
            "steady_occupancy_threshold": None,
            "max_observed_occupancy": max((value for _, value in occupancy_rows), default=0),
            "underfill_duration_s": 0.0,
            "steady_reason": "unbounded_by_scheduler",
        }
    else:
        steady, diagnostics = _steady_window(
            start_ns, end_ns, occupancy_rows, target_concurrency,
            min_ratio=min_ratio, min_duration_s=min_duration_s,
        )
    last30_start = max(start_ns, end_ns - 30_000_000_000)
    drop_start = start_ns + int(span_ns * 0.2)
    if steady.start_ns is not None and steady.end_ns is not None:
        steady_span = steady.end_ns - steady.start_ns
        strip_start = steady.start_ns + int(steady_span * 0.1)
        strip_end = steady.end_ns - int(steady_span * 0.1)
        stripped = (
            MeasurementWindow("steady_strip_10pct", strip_start, strip_end)
            if strip_end > strip_start
            else MeasurementWindow(
                "steady_strip_10pct",
                None,
                None,
                "steady_strip_10pct_invalid",
            )
        )
    else:
        stripped = MeasurementWindow(
            "steady_strip_10pct",
            None,
            None,
            steady.unavailable_reason,
        )
    return (
        MeasurementWindow("overall", start_ns, end_ns),
        MeasurementWindow("last30s", last30_start, end_ns),
        MeasurementWindow("drop_first_20pct", drop_start, end_ns),
        steady,
        stripped,
    ), diagnostics


def _steady_window(
    start_ns: int,
    end_ns: int,
    rows: list[tuple[int, int]],
    target: int,
    *,
    min_ratio: float,
    min_duration_s: float,
) -> tuple[MeasurementWindow, dict[str, float | int | str | None]]:
    threshold = math.ceil(target * min_ratio)
    minimum_ns = int(min_duration_s * 1e9)
    occupancy = 0
    maximum = 0
    previous_ns = start_ns
    candidate_start: int | None = None
    best: tuple[int, int] | None = None
    underfill_ns = 0

    for raw_timestamp, raw_value in rows:
        timestamp = min(end_ns, max(start_ns, raw_timestamp))
        if raw_timestamp <= start_ns:
            occupancy = raw_value
            maximum = max(maximum, occupancy)
            continue
        if timestamp > previous_ns:
            if occupancy >= threshold:
                candidate_start = (
                    previous_ns if candidate_start is None else candidate_start
                )
            else:
                underfill_ns += timestamp - previous_ns
                if candidate_start is not None:
                    best = _longer(best, (candidate_start, previous_ns))
                    candidate_start = None
        next_occupancy = raw_value
        if (
            occupancy >= threshold
            and next_occupancy < threshold
            and candidate_start is not None
        ):
            best = _longer(best, (candidate_start, timestamp))
            candidate_start = None
        occupancy = next_occupancy
        maximum = max(maximum, occupancy)
        previous_ns = timestamp
        if raw_timestamp >= end_ns:
            break

    if previous_ns < end_ns:
        if occupancy >= threshold:
            candidate_start = previous_ns if candidate_start is None else candidate_start
        else:
            underfill_ns += end_ns - previous_ns
    if candidate_start is not None:
        best = _longer(best, (candidate_start, end_ns))

    reason = None
    if best is None:
        reason = "target_not_reached"
    elif best[1] - best[0] < minimum_ns:
        reason = "steady_too_short"
    if reason is None and best is not None:
        window = MeasurementWindow("steady_occupancy", best[0], best[1])
    else:
        window = MeasurementWindow("steady_occupancy", None, None, reason)
    return window, {
        "steady_occupancy_threshold": threshold,
        "max_observed_occupancy": maximum,
        "underfill_duration_s": underfill_ns / 1e9,
        "steady_reason": reason,
    }


def _longer(
    current: tuple[int, int] | None,
    candidate: tuple[int, int],
) -> tuple[int, int]:
    if current is None or candidate[1] - candidate[0] > current[1] - current[0]:
        return candidate
    return current


__all__ = ["MeasurementWindow", "WINDOW_ORDER", "measurement_windows"]
