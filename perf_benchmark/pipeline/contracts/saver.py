"""Saver Stage backend contract and immutable persistence records."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from perf_benchmark.pipeline.context import SaverContext

from perf_benchmark.pipeline.contracts.results import ResultReader, ResultRecord, RunRecord, ResultCommit


@dataclass(frozen=True, slots=True)
class SaverArtifact:
    kind: str
    location: str


class SaverBackend(Protocol):
    @property
    def artifact(self) -> SaverArtifact: ...
    def open(self) -> None: ...
    def write_batch(self, records: Sequence[ResultRecord]) -> ResultCommit | None: ...
    def finalize(self, run: RunRecord) -> ResultReader: ...
    def close(self) -> None: ...


class SaverPlugin(Protocol):
    def open_reader(self, location: str) -> ResultReader: ...
    def create_backend(self, run: RunRecord, ctx: SaverContext) -> SaverBackend: ...
