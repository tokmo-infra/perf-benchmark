from __future__ import annotations

from collections.abc import Collection, Mapping, Iterator
from dataclasses import dataclass
from functools import lru_cache
import math
from types import MappingProxyType
from typing import Any, Protocol, TypedDict, TypeAlias, NotRequired

from perf_benchmark.core.models import (
    MetricWindowMeasurement,
    ResponseMode,
)

from perf_benchmark.pipeline.contracts.facts import TrialFact
from perf_benchmark.pipeline.contracts.sweep import SweepRecord


class RunSummary(TypedDict):
    run_id: str
    name: str
    status: str
    error: str | None
    trial_count: int
    session_count: int
    turn_count: int
    metric_count: int


class TrialRecord(TypedDict):
    trial_id: str
    name: str
    ordinal: int
    status: str
    error: str | None
    started_at: str | None
    finished_at: str | None
    parameters: dict[str, Any]
    metadata: dict[str, Any]
    session_count: int
    turn_count: int
    load_profile: dict[str, Any] | None
    resource_limits: dict[str, Any] | None
    comparison: ComparisonRecord | None
    repetition: dict[str, int] | None
    scheduling: dict[str, Any] | None
    validity_requirements: NotRequired[dict[str, Any] | None]
    validity: NotRequired[dict[str, Any] | None]
    execution_status: NotRequired[str | None]


class MetricWindowRecord(TypedDict):
    window: str
    start_ns: int | None
    end_ns: int | None
    duration_s: float | None
    availability: str
    unavailable_reason: str | None


class StatisticMetadata(TypedDict):
    unit: str
    availability: str
    unavailable_reason: str | None
    source: str | None
    valid_sample_count: int | None
    expected_sample_count: int | None
    missing_sample_count: int | None


class MetricRecord(TypedDict):
    unit: str
    formula_version: str
    availability: str
    unavailable_reason: str | None
    source: str | None
    statistics: dict[str, float | None]
    statistic_metadata: dict[str, StatisticMetadata]


class ComparisonRecord(TypedDict):
    group_id: str
    axis_name: str
    axis_value: int | float | str
    axis_unit: str | None


@dataclass(frozen=True, slots=True)
class MetricPoint:
    scope: str
    scope_id: str
    window: str
    metric: str
    statistic: str
    value: float | None
    unit: str
    formula_version: str
    availability: str
    unavailable_reason: str | None
    source: str | None
    valid_sample_count: int | None = None
    expected_sample_count: int | None = None
    missing_sample_count: int | None = None


@dataclass(frozen=True, slots=True)
class MetricSeries:
    """A replaceable time bucket computed by Metrics, scoped to one Trial."""
    trial_id: str
    window: MetricWindowMeasurement
    points: tuple[MetricPoint, ...]


@dataclass(frozen=True, slots=True)
class GroupedMetricRecord:
    """Final Turn cohort; values use the same public units as MetricPoint."""
    trial_id: str
    window: str
    dimensions: tuple[tuple[str, str], ...]
    metric: str
    unit: str
    formula_version: str
    sample_count: int
    eligible_sample_count: int
    valid_sample_count: int
    statistics: Mapping[str, float | None]
    unavailable_reasons: Mapping[str, int]


@dataclass(frozen=True, slots=True, kw_only=True)
class _EntityResult:
    id: str
    ordinal: int
    status: str
    started_at: str | None
    finished_at: str | None
    error: str | None
    measurements: Mapping[str, object]
    metrics: Mapping[str, ScalarMetricRecord]
    metadata: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class TurnResultRecord(_EntityResult):
    session_id: str | None
    estimated_input_tokens: int | None
    reference_output_tokens: int | None
    requested_output_tokens: int | None
    usage: Mapping[str, int]
    scenario_passed: bool | None = None


@dataclass(frozen=True, slots=True)
class SessionResultRecord(_EntityResult):
    trial_id: str | None
    turn_count: int


@dataclass(frozen=True, slots=True)
class AttemptResultRecord:
    turn_id: str
    session_id: str
    attempt: int
    status: str
    error: str | None
    error_origin: str | None
    measurements: Mapping[str, object]
    usage: Mapping[str, int]
    metrics: Mapping[str, ScalarMetricRecord]
    scenario_passed: bool | None
    trigger_ns: int | None
    decision_ns: int | None
    applied_ns: int | None
    plan: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ComputedResultBatch:
    turns: tuple[TurnResultRecord, ...] = ()
    sessions: tuple[SessionResultRecord, ...] = ()
    attempts: tuple[AttemptResultRecord, ...] = ()
    series: tuple[MetricSeries, ...] = ()


class RecordPage(TypedDict):
    items: tuple[dict[str, Any], ...]
    total: int


class ResultReader(Protocol):
    def run_record(self, run_id: str) -> RunRecord: ...

    def measurements(self, run_id: str, *, scope: str, scope_id: str) -> Mapping[str, Any]: ...

    def request_page(self, run_id: str, *, scope: str = "turn", trial_id: str | None = None,
                     turn_id: str | None = None, status: str | None = None, offset: int = 0,
                     limit: int = 50, from_ms: int | None = None, to_ms: int | None = None,
                     sampled_only: bool = False, session_id: str | None = None, detail: bool = True) -> RecordPage: ...

    def grouped_metrics(self, run_id: str, trial_id: str) -> tuple[GroupedMetricRecord, ...]: ...

    def metric_series(self, run_id: str, *, trial_id: str | None, metric: str,
                      start_ns: int, end_ns: int, limit: int = 10000) -> RecordPage: ...

    def sweep_records(self, run_id: str) -> tuple[SweepRecord, ...]: ...

    def read_updates(self, cursor: ResultUpdateCursor | None = None, limit: int = 500) -> ResultUpdatePage: ...

    @property
    def location(self) -> str: ...

    def summary(self, run_id: str) -> RunSummary: ...

    def trials(self, run_id: str) -> tuple[TrialRecord, ...]: ...

    def run_ids(self) -> tuple[str, ...]: ...

    def iter_summary_records(self, run_id: str) -> Iterator[ResultRecord]: ...

    def iter_turn_records(self, run_id: str) -> Iterator[dict[str, Any]]: ...

    def iter_attempt_records(self, run_id: str) -> Iterator[dict[str, Any]]: ...

    def metric_windows(
        self,
        run_id: str,
        *,
        scope: str,
        scope_id: str,
    ) -> tuple[MetricWindowRecord, ...]: ...

    def metric_points(
        self,
        run_id: str,
        *,
        scope: str,
        scope_id: str,
        window: str,
        names: Collection[str] | None = None,
    ) -> Mapping[str, MetricRecord]: ...

    def comparison_points(self, run_id: str) -> tuple[ComparisonRecord, ...]: ...


class TraceSnapshot(TypedDict):
    trace_id: str
    parent_span_id: str
    span_id: str
    sampled: bool


class LifecycleSnapshot(TypedDict):
    started_ns: int | None
    completed_ns: int | None


class RequestSnapshot(TypedDict):
    attempts: int
    response_mode: ResponseMode | None
    target_ns: int | None
    offered_ns: int | None
    admitted_ns: int | None
    started_ns: int | None
    first_output_ns: int | None
    last_output_ns: int | None
    response_completed_ns: int | None
    http_status: int | None
    output_event_count: int
    output_chars: int
    finish_reason: str | None
    output_kind: str
    result_ready_ns: int | None
    trace_context: TraceSnapshot | None
    profile: Mapping[str, object] | None


class TurnSnapshot(TypedDict):
    lifecycle: LifecycleSnapshot
    request: RequestSnapshot
    total_started_ns: int | None


class SessionSnapshot(TypedDict):
    lifecycle: LifecycleSnapshot
    target_ns: int | None
    offered_ns: int | None
    admitted_ns: int | None


FORBIDDEN_FACT_FIELDS = frozenset({
    "messages", "request_messages", "response_messages", "body", "content",
    "authorization", "api_key", "access_token", "api-key", "x-api-key",
})


@lru_cache(maxsize=256)
def _metric_attributes(unit, availability, reason, source, formula_version):
    for index, value in enumerate((unit, availability, reason, source, formula_version)):
        if index >= 2 and value is None:
            continue
        if not isinstance(value, str):
            raise TypeError("metric attributes must be text")
        if len(value) > 4096:
            raise ValueError("fact string exceeds 4096 characters")


@dataclass(frozen=True, slots=True)
class ScalarMetricRecord:
    """Validated immutable scalar result; Saver need not validate it again."""
    unit: str
    availability: str
    unavailable_reason: str | None
    source: str | None
    statistics: Mapping[str, int | float | None]
    formula_version: str | None = None

    def __post_init__(self):
        _metric_attributes(self.unit, self.availability, self.unavailable_reason, self.source, self.formula_version)
        statistics = dict(self.statistics)
        for key, value in statistics.items():
            if not isinstance(key, str) or type(value) not in (int, float, type(None)):
                raise TypeError("metric result must contain named scalar statistics")
            if key.lower() in FORBIDDEN_FACT_FIELDS:
                raise ValueError("fact contains body field or credential")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("fact contains a non-finite number")
        object.__setattr__(self, "statistics", MappingProxyType(statistics))


@dataclass(frozen=True, slots=True)
class RunRecord:
    id: str
    name: str
    status: str
    created_at: str
    preparing_at: str | None
    started_at: str | None
    finished_at: str | None
    error: str | None
    counts: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class TrialResultRecord:
    id: str
    run_id: str
    name: str
    ordinal: int
    status: str
    parameters: Mapping[str, object]
    metadata: Mapping[str, object]
    started_at: str | None
    finished_at: str | None
    error: str | None
    session_count: int
    turn_count: int
    succeeded_turns: int
    failed_or_cancelled_turns: int


@dataclass(frozen=True, slots=True)
class TrialFactRecord:
    fact: TrialFact


@dataclass(frozen=True, slots=True)
class MeasurementRecord:
    run_id: str
    scope: str
    scope_id: str
    name: str
    value: object


@dataclass(frozen=True, slots=True)
class MetricWindowResultRecord(MetricWindowMeasurement):
    run_id: str
    scope: str
    scope_id: str

ResultRecord: TypeAlias = (
    SweepRecord
    | RunRecord
    | TrialResultRecord
    | TrialFactRecord
    | SessionResultRecord
    | TurnResultRecord
    | AttemptResultRecord
    | MeasurementRecord
    | MetricWindowResultRecord
    | MetricPoint
    | GroupedMetricRecord
)


@dataclass(frozen=True, slots=True)
class ResultUpdateCursor:
    database_id: str
    revision: int = 0
    entity_kind: int = -1
    internal_record_id: int = 0


@dataclass(frozen=True, slots=True)
class ResultUpdate:
    revision: int
    entity_kind: int
    internal_record_id: int
    key: tuple[str | int, ...]
    record: ResultRecord
    run_id: str
    trial_id: str | None = None


@dataclass(frozen=True, slots=True)
class ResultUpdatePage:
    updates: tuple[ResultUpdate, ...]
    cursor: ResultUpdateCursor
    commit_revision: int
    caught_up: bool
    measurement_contract: str


@dataclass(frozen=True, slots=True)
class ResultCommit:
    revision: int
