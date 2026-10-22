"""Bounded formal result readers and ephemeral Run location index."""

import asyncio
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import cast

from aiohttp import web
from perf_benchmark.pipeline.contracts.saver import SaverPlugin
from perf_benchmark.plugins import create_registry
from .projections import PAGE_SIZE, choice, integer, run_view
from .queries import query_run

SERVICE = web.AppKey("results", object)

class Catalog:
    """An ephemeral location index, rebuilt from live SQLite and sealed summaries after service restart."""

    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("results root must be a directory")
        self.plugin = cast(SaverPlugin, create_registry().bind("saver", "sqlite", {"path": "unused.db"}).plugin)
        self.lock = Lock()
        self.refreshed = float("-inf")
        self.locations = {}
        self.rows = ()
        self.errors = ()

    def refresh(self):
        with self.lock:
            if monotonic() - self.refreshed < 5:
                return
            locations, rows, errors = {}, [], []
            seen = set()
            for path in sorted(self.root.rglob("*")):
                if path.name == "artifact_info.parquet" and path.parent.name == "summary" and path.parent.parent.name.endswith(".archive"):
                    path = path.parent.parent
                elif path.suffix not in {".db", ".sqlite", ".sqlite3"} or not path.is_file():
                    continue
                if not path.resolve().is_relative_to(self.root):
                    continue
                try:
                    reader = self.plugin.open_reader(str(path))
                    if reader.location in seen:
                        continue
                    seen.add(reader.location)
                    for run_id in reader.run_ids():
                        locations.setdefault(run_id, []).append(reader.location)
                        rows.append(run_view(reader.run_record(run_id)))
                except (OSError, ValueError, RuntimeError) as exc:
                    errors.append({"file": str(path.relative_to(self.root)), "error": type(exc).__name__})
                except Exception as exc:
                    # A corrupt/incomplete file cannot hide other Runs. No exception text or credentials.
                    errors.append({"file": str(path.relative_to(self.root)), "error": type(exc).__name__})
            self.locations, self.rows, self.errors = locations, tuple(rows), tuple(errors)
            self.refreshed = monotonic()

    def reader(self, run_id):
        self.refresh()
        paths = self.locations.get(run_id, ())
        if not paths:
            raise web.HTTPNotFound(text="Run not found")
        if len(paths) != 1:
            raise web.HTTPConflict(text="Duplicate Run ID in results root; select a root with unique Runs")
        return self.plugin.open_reader(paths[0])

    def runs(self, query):
        self.refresh()
        search = query.get("search", "").casefold()
        status = choice(query, "status")
        rows = sorted((row for row in self.rows
                       if search in (row["id"] + " " + row["name"]).casefold()
                       and (status is None or row["status"] == status)),
                      key=lambda row: (row["start_ms"], row["id"]), reverse=True)
        page = integer(query, "page", 1, minimum=1)
        start = (page - 1) * PAGE_SIZE
        return {"items": rows[start:start + PAGE_SIZE], "total": len(rows), "page": page,
                "errors": self.errors, "duplicate_run_ids": [key for key, paths in self.locations.items() if len(paths) > 1]}
class ResultsService:
    def __init__(self, root):
        self.catalog = Catalog(root)
        self.slots = asyncio.Semaphore(4)
        self.pending = 0

    async def read(self, operation):
        if self.pending >= 32:
            raise web.HTTPServiceUnavailable(text="Result readers busy; retry later", headers={"Retry-After": "1"})
        self.pending += 1
        try:
            async with self.slots:
                task = asyncio.create_task(asyncio.to_thread(operation))
                try:
                    return await asyncio.shield(task)
                except asyncio.CancelledError:
                    # Do not release a read slot while its SQLite thread is still running.
                    await task
                    raise
        finally:
            self.pending -= 1

    def query(self, run_id, operation, query):
        return query_run(self.catalog.reader(run_id), run_id, operation, query)
