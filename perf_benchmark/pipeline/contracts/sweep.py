"""Immutable SLO evidence exchanged by Scheduler, Host and result readers."""
from dataclasses import dataclass
from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class SloRule:
    metric: str
    statistic: str
    op: str
    original: str | float | int
    threshold: float
    unit: str


@dataclass(frozen=True, slots=True)
class SloCheck:
    rule: SloRule
    trial_id: str
    value: float | None
    status: str
    reason: str | None
    valid_samples: int | None
    expected_samples: int | None
    missing_samples: int | None


@dataclass(frozen=True, slots=True)
class TrialValidity:
    trial_id: str
    execution_status: str
    status: str
    reasons: tuple[str, ...] = ()
    observations: tuple[tuple[str, float | None, str], ...] = ()


@dataclass(frozen=True, slots=True)
class SweepRecord:
    run_id: str
    sequence: int
    axis: str
    phase: str
    status: str
    value: int | float | None = None
    trial_ids: tuple[str, ...] = ()
    checks: tuple[SloCheck, ...] = ()
    lower: int | float | None = None
    upper: int | float | None = None
    reason: str | None = None
    definition: Mapping[str, object] | None = None
    validity: tuple[TrialValidity, ...] = ()
