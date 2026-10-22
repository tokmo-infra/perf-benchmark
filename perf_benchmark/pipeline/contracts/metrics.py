"""Metrics Stage contracts.

The public backend consumes immutable measurement and scheduling facts.  It does
not receive Scheduler settings or mutable domain objects.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from perf_benchmark.core.models import Metadata, MetricWindowMeasurement, Run, Trial
from perf_benchmark.pipeline.context import MeasurementRecorder
from perf_benchmark.pipeline.contracts.facts import TrialFact
from perf_benchmark.pipeline.contracts.sweep import TrialValidity
from perf_benchmark.pipeline.contracts.results import ScalarMetricRecord, ComputedResultBatch, GroupedMetricRecord, MetricPoint, MetricSeries
from perf_benchmark.pipeline.results.measurement import MeasurementFact


@dataclass(frozen=True, slots=True)
class MetricUpdate:
    scope: str
    id: str
    metrics: dict[str, ScalarMetricRecord]
    attempt: int = 0


@dataclass(frozen=True, slots=True)
class TrialMetricReport:
    points: tuple[MetricPoint, ...]
    windows: tuple[MetricWindowMeasurement, ...]
    counts: dict[str, int]
    diagnostics: Metadata
    groups: tuple[GroupedMetricRecord, ...] = ()
    validity: TrialValidity | None = None


ResultPublisher = Callable[[ComputedResultBatch], None]


class MetricsBackend(Protocol):
    def open(self) -> None: ...
    def consume(self, facts: Sequence[MeasurementFact]) -> Sequence[MetricUpdate | MetricSeries]: ...
    def finalize_trial(self, fact: TrialFact) -> TrialMetricReport: ...
    def release_trial(self, trial_id: str) -> None: ...
    def finalize_run(self, run_id: str) -> tuple[MetricPoint, ...]: ...
    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class MetricCapability:
    name: str
    unit: str
    statistics: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MetricsCapabilities:
    metrics: tuple[MetricCapability, ...]
    windows: tuple[str, ...]


class MetricsPlugin(Protocol):
    def compare(self, baseline, candidate, specification) -> dict: ...

    def describe(self) -> MetricsCapabilities: ...

    def create_backend(self, *, seed: int) -> MetricsBackend: ...


class MetricsCollector(MeasurementRecorder, Protocol):
    async def wait_for_headroom(self) -> None: ...
    async def finalize(
        self, trial: Trial, fact: TrialFact
    ) -> TrialMetricReport: ...
    async def close(self) -> None: ...


class MetricsRuntime(Protocol):
    async def open_trial(self, trial: Trial) -> MetricsCollector: ...
    async def finalize(self, run: Run) -> tuple[MetricPoint, ...]: ...
    async def close(self) -> None: ...
