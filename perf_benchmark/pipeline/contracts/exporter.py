"""Exporter Stage contract values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from perf_benchmark.pipeline.context import ExporterContext
from perf_benchmark.pipeline.contracts.results import ResultReader


@dataclass(frozen=True, slots=True)
class ExportOutput:
    kind: str
    location: str
    format: str


@dataclass(frozen=True, slots=True)
class ExportResult:
    outputs: tuple[ExportOutput, ...] = ()


class ExporterPlugin(Protocol):
    async def export(
        self, run_id: str, results: ResultReader, ctx: ExporterContext
    ) -> ExportResult: ...
