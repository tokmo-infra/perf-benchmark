"""CSV and Excel exports of stored measurements, including maxtpm minute revenue."""

import asyncio
from pathlib import Path

from perf_benchmark.pipeline.contracts import ExportOutput, ExportResult
from perf_benchmark.plugins import register_plugin
from perf_benchmark.plugins.exporter.tabular.render import DEFAULT_TABLES, TABLE_NAMES, render_tables


def parse_settings(settings):
    if set(settings) - {"output", "formats", "excel_path", "tables"}:
        raise ValueError("tabular accepts output, formats, excel_path and tables")
    output = settings.get("output")
    formats = settings.get("formats", ["xlsx"])
    if "output" in settings and (not isinstance(output, str) or not output.strip()):
        raise ValueError("tabular.output must be a nonempty directory path")
    if (
        not isinstance(formats, list)
        or not formats
        or any(f not in ("csv", "xlsx") for f in formats)
    ):
        raise ValueError("tabular.formats must be a nonempty list of csv/xlsx")
    if len(formats) != len(set(formats)):
        raise ValueError("tabular.formats must not repeat a format")
    tables = settings.get("tables", list(DEFAULT_TABLES))
    if not isinstance(tables, list) or not tables or any(name not in TABLE_NAMES for name in tables):
        raise ValueError("tabular.tables must be a nonempty list of summary/minutes/requests/kv_capacity")
    if len(tables) != len(set(tables)):
        raise ValueError("tabular.tables must not repeat a table")
    excel = settings.get("excel_path")
    if excel is not None and (not isinstance(excel,str) or not excel.strip() or Path(excel).suffix.lower()!='.xlsx'):
        raise ValueError("tabular.excel_path must be an .xlsx file path")
    if excel is not None and 'xlsx' not in formats:
        raise ValueError("tabular.excel_path requires xlsx format")
    return {"output": output, "formats": formats, "excel_path": excel, "tables": tables}


@register_plugin(
    stage="exporter",
    name="tabular",
    description="Export CSV and Excel measurements",
    settings_parser=parse_settings,
)
class TabularExporter:
    def __init__(self, settings):
        self.settings = settings

    async def export(self, run_id, results, ctx):
        output = self.settings["output"]
        if output is None:
            database = Path(results.location).expanduser()
            output = database.with_name(database.stem + "-tables")
        directory = (ctx.base_dir / Path(output).expanduser()).resolve()
        paths = await asyncio.to_thread(
            render_tables, run_id, results, directory, self.settings["formats"],
            excel_path=(ctx.base_dir / Path(self.settings['excel_path']).expanduser()).resolve()
            if self.settings.get('excel_path') else None,
            table_names=self.settings["tables"],
        )
        return ExportResult(tuple(
            ExportOutput(f"tabular:{p.name}", str(p), p.suffix[1:]) for p in paths
        ))
