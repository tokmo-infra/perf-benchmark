"""Immutable cross-stage facts produced by scheduling and execution."""

from __future__ import annotations

from perf_benchmark.core.validation import positive_int, positive_number

from dataclasses import dataclass
import math

from .sweep import TrialValidity


@dataclass(frozen=True, slots=True)
class LoadProfile:
    mode: str
    target_concurrency: int | None = None
    target_session_rate: float | None = None
    target_request_rate: float | None = None
    max_active_sessions: int | None = None
    max_concurrency: int | None = None
    arrival_distribution: str = "constant"
    gamma_shape: float | None = None
    ramp_duration_s: float | None = None
    ramp_end_multiplier: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str) or not self.mode:
            raise ValueError("load profile mode must not be empty")
        positive_int(self.target_concurrency, "target_concurrency", optional=True)
        positive_number(self.target_session_rate, "target_session_rate", optional=True)
        positive_number(self.target_request_rate, "target_request_rate", optional=True)
        positive_int(self.max_active_sessions, "max_active_sessions", optional=True)
        positive_int(self.max_concurrency, "max_concurrency", optional=True)
        if self.arrival_distribution not in {"constant", "poisson", "gamma", "timeline"}:
            raise ValueError("unsupported arrival distribution")
        positive_number(self.gamma_shape, "gamma_shape", optional=self.arrival_distribution != "gamma")
        if self.gamma_shape is not None and self.arrival_distribution != "gamma":
            raise ValueError("gamma_shape requires gamma arrivals")
        if (self.ramp_duration_s is None) != (self.ramp_end_multiplier is None):
            raise ValueError("ramp requires both duration and end multiplier")
        positive_number(self.ramp_duration_s, "ramp_duration_s", optional=True)
        positive_number(self.ramp_end_multiplier, "ramp_end_multiplier", optional=True)


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    ready_capacity: int
    max_pending_sessions: int
    max_pending_requests: int

    def __post_init__(self) -> None:
        positive_int(self.ready_capacity, "ready_capacity")
        positive_int(self.max_pending_sessions, "max_pending_sessions")
        positive_int(self.max_pending_requests, "max_pending_requests")


@dataclass(frozen=True, slots=True)
class ComparisonPoint:
    group_id: str
    axis_name: str
    axis_value: int | float | str
    axis_unit: str | None = None

    def __post_init__(self) -> None:
        if not self.group_id:
            raise ValueError("comparison group_id must not be empty")
        if not self.axis_name:
            raise ValueError("comparison axis_name must not be empty")
        if isinstance(self.axis_value, bool) or not isinstance(
            self.axis_value, (int, float, str)
        ):
            raise TypeError("comparison axis_value must be a number or string")
        if isinstance(self.axis_value, float) and not math.isfinite(self.axis_value):
            raise ValueError("comparison axis_value must be finite")


@dataclass(frozen=True, slots=True)
class RepetitionPlan:
    index: int = 1
    expected: int = 1

    def __post_init__(self) -> None:
        positive_int(self.index, "repetition index")
        positive_int(self.expected, "expected repetitions")
        if self.index > self.expected:
            raise ValueError("repetition index cannot exceed expected repetitions")


@dataclass(frozen=True, slots=True)
class ValidityRequirements:
    window: str = "overall"
    min_samples: int = 1
    min_duration_s: float | None = None
    max_scheduler_lag_ms: float | None = None
    load_tolerance_ratio: float | None = None

    def __post_init__(self) -> None:
        if not self.window:
            raise ValueError("validity window must not be empty")
        positive_int(self.min_samples, "min_samples")
        positive_number(self.min_duration_s, "min_duration_s", optional=True)
        positive_number(self.max_scheduler_lag_ms, "max_scheduler_lag_ms", optional=True, allow_zero=True)
        positive_number(self.load_tolerance_ratio, "load_tolerance_ratio", optional=True, allow_zero=True)
        if self.load_tolerance_ratio is not None and self.load_tolerance_ratio >= 1:
            raise ValueError("load_tolerance_ratio must be less than 1")


@dataclass(frozen=True, slots=True)
class GraphCoverage:
    graph_id: str
    selected: int
    started: int
    completed: int
    blocked: int
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class TrialSchedulingFact:
    offered_sessions: int = 0
    admitted_sessions: int = 0
    completed_sessions: int = 0
    offered_attempts: int = 0
    admitted_attempts: int = 0
    completed_attempts: int = 0
    offering_started_ns: int | None = None
    session_offering_finished_ns: int | None = None
    request_offering_finished_ns: int | None = None
    drain_duration_ms: float = 0.0
    demand_limited: bool = False
    demand_exhausted: bool = False
    termination_reason: str | None = None
    graph_coverage: tuple[GraphCoverage, ...] = ()
    dataset_session_count: int | None = None
    dataset_turn_count: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "offered_sessions",
            "admitted_sessions",
            "completed_sessions",
            "offered_attempts",
            "admitted_attempts",
            "completed_attempts",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        positive_int(self.dataset_session_count, "dataset_session_count", optional=True)
        positive_int(self.dataset_turn_count, "dataset_turn_count", optional=True)
        positive_number(self.drain_duration_ms, "drain_duration_ms", allow_zero=True)


@dataclass(frozen=True, slots=True)
class TrialFact:
    trial_id: str
    status: str
    load_profile: LoadProfile
    resource_limits: ResourceLimits
    comparison: ComparisonPoint | None
    repetition: RepetitionPlan
    scheduling: TrialSchedulingFact
    validity_requirements: ValidityRequirements | None = None
    validity: TrialValidity | None = None
    execution_status: str = "unknown"

    def __post_init__(self) -> None:
        if not self.trial_id:
            raise ValueError("trial_id must not be empty")
        if self.execution_status not in {"unknown", "succeeded", "failed", "cancelled"}:
            raise ValueError("invalid Trial execution status")
        if self.status not in {"succeeded", "failed", "cancelled"}:
            raise ValueError("TrialFact requires a terminal status")
