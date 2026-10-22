from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping, Callable
from typing import Protocol, TYPE_CHECKING

from perf_benchmark.core.models import Run, Session, Trial, Turn
from perf_benchmark.runtime.events import EventDispatcher
from perf_benchmark.pipeline.results.measurement import ResponseTrace
from perf_benchmark.pipeline.preparation.payloads import BlobStore
from perf_benchmark.runtime.cancellation import CancellationToken
from perf_benchmark.pipeline.registry import ExporterBindings, StageBinding
from perf_benchmark.core.observation import ObservationSettings
if TYPE_CHECKING:
    from perf_benchmark.pipeline.contracts.observation import SampledOutputTrace
    from perf_benchmark.pipeline.contracts.results import ResultCommit


class MeasurementRecorder(Protocol):
    def record_turn(self, turn: Turn, trace: ResponseTrace | None = None) -> None: ...
    def record_attempt(self, turn: Turn, result) -> None: ...
    def record_session(self, session: Session) -> None: ...
    def record_occupancy(self, kind: str, at_ns: int, value: int) -> None: ...


@dataclass(frozen=True, slots=True)
class LoaderContext:
    base_dir: Path
    cancellation: CancellationToken


@dataclass(frozen=True, slots=True)
class NormalizationContext:
    base_dir: Path
    cancellation: CancellationToken
    seed: int = 0


@dataclass(frozen=True, slots=True)
class SelectorContext:
    cancellation: CancellationToken


@dataclass(frozen=True, slots=True)
class ExecutorContext:
    cancellation: CancellationToken


@dataclass(frozen=True, slots=True)
class SchedulerContext:
    cancellation: CancellationToken
    seed: int = 0


@dataclass(frozen=True, slots=True)
class MetricsContext:
    cancellation: CancellationToken
    seed: int = 0
    run_id: str = ""
    sampled_output: Callable[[SampledOutputTrace], None] | None = None


@dataclass(frozen=True, slots=True)
class SaverContext:
    base_dir: Path
    cancellation: CancellationToken
    committed: Callable[[ResultCommit], None] | None = None


@dataclass(frozen=True, slots=True)
class ExporterContext:
    base_dir: Path
    cancellation: CancellationToken


@dataclass(slots=True)
class RunContext:
    bindings: Mapping[str, StageBinding | ExporterBindings]
    base_dir: Path
    events: EventDispatcher
    cancellation: CancellationToken
    payloads: BlobStore
    observation: ObservationSettings = ObservationSettings()
    seed: int = 0

    def binding(self, stage: str) -> StageBinding | ExporterBindings:
        try:
            return self.bindings[stage]
        except KeyError as exc:
            raise ValueError(f"unbound pipeline stage: {stage}") from exc

    def for_trial(self, domain_run: Run, trial: Trial, measurements=None) -> TrialContext:
        return TrialContext(
            domain_run, trial, self.events, self.cancellation, self.payloads, measurements,
            self.observation, self.seed,
        )


@dataclass(slots=True)
class TrialContext:
    domain_run: Run
    trial: Trial
    events: EventDispatcher
    cancellation: CancellationToken
    payloads: BlobStore
    measurements: MeasurementRecorder | None = None
    observation: ObservationSettings = ObservationSettings()
    seed: int = 0
