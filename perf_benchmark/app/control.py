"""Bounded local child-process handles; SQLite remains the result authority."""
import asyncio
from pathlib import Path
import signal
import sqlite3
import sys
from uuid import uuid4

import orjson
from aiohttp import web


CONTROL = web.AppKey("control", object)


class RunControl:
    def __init__(self, config_root, result_root, limit=8):
        self.config_root = Path(config_root).resolve(strict=True)
        self.result_root = Path(result_root).resolve(strict=True)
        self.limit, self.jobs = limit, {}
        self.lock = asyncio.Lock()

    def configs(self):
        return sorted(str(path.relative_to(self.config_root)) for path in self.config_root.rglob("*")
                      if path.suffix in {".yaml", ".yml"} and path.is_file() and path.resolve().is_relative_to(self.config_root))

    async def start(self, name):
        path = (self.config_root / name).resolve(strict=True)
        if not path.is_relative_to(self.config_root) or path.suffix not in {".yaml", ".yml"}:
            raise ValueError("select an existing YAML inside config root")
        async with self.lock:
            while len(self.jobs) >= self.limit:
                completed = next((key for key, value in self.jobs.items() if value["process"].returncode is not None), None)
                if completed is None:
                    raise web.HTTPTooManyRequests(text="local running process limit reached")
                self.jobs.pop(completed)
            identifier = uuid4().hex
            database = self.result_root / (identifier + ".sqlite")
            process = await asyncio.create_subprocess_exec(sys.executable, "-m", "perf_benchmark", "run", str(path),
                "--output-database", str(database), stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
            self.jobs[identifier] = {"process": process, "config": name, "sqlite": str(database)}
            return identifier

    def status(self):
        return [{"id": key, "config": value["config"], "sqlite": value["sqlite"],
                 "process_state": "running" if value["process"].returncode is None else "exited",
                 "exit_code": value["process"].returncode} for key, value in self.jobs.items()]

    async def cancel(self, identifier):
        async with self.lock:
            value = self.jobs.get(identifier)
            if value is None:
                raise web.HTTPNotFound(text="process handle not owned by this service")
            if value["process"].returncode is None:
                try:
                    value["process"].send_signal(signal.SIGINT)
                except ProcessLookupError:
                    pass

    async def close(self):
        for identifier in tuple(self.jobs):
            await self.cancel(identifier)
        # Await reliable writer cleanup. There is no forced kill or stale-PID adoption.
        await asyncio.gather(*(value["process"].wait() for value in self.jobs.values()))
        self.jobs.clear()


def same_origin(request):
    origin = request.headers.get("Origin")
    if origin is not None and origin != f"{request.scheme}://{request.host}":
        raise web.HTTPForbidden(text="cross-origin control is not permitted")
    if request.content_type != "application/json":
        raise web.HTTPUnsupportedMediaType()


async def endpoint(request):
    control = request.app[CONTROL]
    action = request.match_info["action"]
    if request.method == "GET":
        value = {"configs": await asyncio.to_thread(control.configs)} if action == "configs" else {"jobs": control.status()}
        if action == "jobs":
            from .results.service import SERVICE
            service = request.app[SERVICE]
            for job in value["jobs"]:
                def identity(path=job["sqlite"]):
                    try:
                        reader = service.catalog.plugin.open_reader(path)
                        ids = reader.run_ids()
                        return ids[0] if len(ids) == 1 else None
                    except (OSError, ValueError, RuntimeError, sqlite3.Error):
                        return None  # Saver has not published a readable identity yet.
                job["run_id"] = await service.read(identity)
    else:
        same_origin(request)
        value = await request.json()
        if action == "start":
            if not isinstance(value, dict) or set(value) != {"config"} or not isinstance(value["config"], str):
                raise ValueError("start requires config filename")
            value = {"id": await control.start(value["config"])}
        else:
            if not isinstance(value, dict) or set(value) != {"id"}:
                raise ValueError("cancel requires process handle id")
            await control.cancel(value["id"])
            value = {"cancel_requested": True}
    return web.Response(body=orjson.dumps(value), content_type="application/json")


def add_routes(app, config_root, result_root):
    app[CONTROL] = RunControl(config_root, result_root)
    app.router.add_get("/api/control/{action:configs|jobs}", endpoint)
    app.router.add_post("/api/control/{action:start|cancel}", endpoint)
    async def cleanup(app):
        await app[CONTROL].close()
    app.on_cleanup.append(cleanup)
