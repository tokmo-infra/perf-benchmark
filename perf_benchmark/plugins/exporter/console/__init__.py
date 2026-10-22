from __future__ import annotations

import asyncio

from rich.console import Console

from perf_benchmark.pipeline.context import ExporterContext
from perf_benchmark.pipeline.contracts.exporter import ExportOutput, ExportResult
from perf_benchmark.pipeline.contracts.results import ResultReader
from perf_benchmark.plugins import register_plugin
from perf_benchmark.plugins.exporter.console.render import render_report


def parse_settings(settings: dict) -> dict:
    if settings:
        raise ValueError(f"unknown console exporter setting: {sorted(settings)[0]}")
    return {}


@register_plugin(
    stage="exporter",
    name="console",
    description="Render benchmark and comparison metric tables in the terminal",
    settings_parser=parse_settings,
)
class ConsoleExporter:
    def __init__(self, settings, *, console: Console | None = None) -> None:
        self.settings = settings
        self._console = console

    async def export(
        self,
        run_id: str,
        results: ResultReader,
        ctx: ExporterContext,
    ) -> ExportResult:
        console = self._console
        if console is None:
            terminal = Console()
            console = Console(width=max(terminal.size.width, 140))
        await asyncio.to_thread(render_report, console, run_id, results)
        return ExportResult(
            (ExportOutput(kind="stream", location="stdout", format="rich-table"),)
        )


__all__ = ["ConsoleExporter", "parse_settings"]
