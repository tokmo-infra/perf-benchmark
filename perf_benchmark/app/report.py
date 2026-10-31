"""Offline application entrypoint; uses the same Exporter as the pipeline."""

import asyncio
from pathlib import Path

from perf_benchmark.pipeline.context import ExporterContext
from perf_benchmark.pipeline.contracts import ExportResult
from perf_benchmark.plugins.exporter.html import HtmlExporter, parse_settings
from perf_benchmark.plugins.saver.sqlite import SQLiteSaver
from perf_benchmark.runtime.cancellation import CancellationToken


async def generate(database: Path, output: Path, grafana_url: str = '') -> ExportResult:
    reader = await asyncio.to_thread(SQLiteSaver.open_reader, str(database))
    run_ids = await asyncio.to_thread(reader.run_ids)
    if len(run_ids) != 1:
        raise ValueError('report requires exactly one Run per database')
    exporter = HtmlExporter(parse_settings({'output': str(output), 'grafana_url': grafana_url}))
    return await exporter.export(run_ids[0], reader, ExporterContext(Path.cwd(), CancellationToken()))


async def export_tables(
    database: Path, output: Path, formats: list[str], *, tables: list[str] | None = None
) -> ExportResult:
    from perf_benchmark.plugins.exporter.tabular import TabularExporter, parse_settings

    reader = await asyncio.to_thread(SQLiteSaver.open_reader, str(database))
    run_ids = await asyncio.to_thread(reader.run_ids)
    if len(run_ids) != 1:
        raise ValueError('export requires exactly one Run per database')
    settings = {'output': str(output), 'formats': formats}
    if tables is not None:
        settings['tables'] = tables
    exporter = TabularExporter(parse_settings(settings))
    return await exporter.export(run_ids[0], reader, ExporterContext(Path.cwd(), CancellationToken()))
