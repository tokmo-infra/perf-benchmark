"""Summary and trend report, using only the ResultReader contract."""

import asyncio
from pathlib import Path
from urllib.parse import urlsplit

from perf_benchmark.pipeline.contracts import ExportOutput, ExportResult
from perf_benchmark.plugins import register_plugin
from perf_benchmark.plugins.exporter.html.render import render_report


def parse_settings(settings):
    if unknown := set(settings) - {"output", "grafana_url"}:
        raise ValueError(f"unknown html exporter settings: {sorted(unknown)}")
    output = settings.get("output", "report.html")
    if not isinstance(output, str) or not output.strip():
        raise ValueError("html.output must be a nonempty path")
    url = settings.get("grafana_url", "")
    if not isinstance(url, str):
        raise ValueError("html.grafana_url must be a string")
    if url:
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"https", "http"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "html.grafana_url must be an HTTP base URL without credentials or query"
            )
    return {"output": output, "grafana_url": url.rstrip("/")}


@register_plugin(
    stage="exporter",
    name="html",
    description="Export formal summaries, cohorts and trends",
    settings_parser=parse_settings,
)
class HtmlExporter:
    def __init__(self, settings):
        self.settings = settings

    async def export(self, run_id, results, ctx):
        path = (ctx.base_dir / Path(self.settings["output"]).expanduser()).resolve()
        database = str(Path(results.location).resolve())
        if str(path) in {database, database + "-wal", database + "-shm"}:
            raise ValueError("HTML output must not replace the result database")
        await asyncio.to_thread(render_report, run_id, results, path, self.settings["grafana_url"])
        return ExportResult((ExportOutput("report", str(path), "html"),))
