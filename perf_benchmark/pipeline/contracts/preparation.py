"""Application-owned preparation, awaited by the Trial Host before measurement."""
from dataclasses import dataclass
from typing import Protocol

from perf_benchmark.pipeline.contracts.scheduler import EffectiveTrial
from perf_benchmark.runtime.cancellation import CancellationToken


@dataclass(frozen=True, slots=True)
class PreparationFact:
    adapter: str
    planned_requests: int
    succeeded_requests: int
    cache_state: str
    quality_status: str = "unverified"
    failure_reason: str | None = None


class TrialPreparation(Protocol):
    async def prepare(self, trial: EffectiveTrial, cancellation: CancellationToken,
                      corpus: object, executor: object) -> PreparationFact: ...
