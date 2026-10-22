"""Optional observation Host: no SDK; sampled-only diagnostic IPC in the benchmark process."""

import asyncio
from contextlib import suppress
from dataclasses import asdict, is_dataclass
from importlib.util import find_spec
import logging
import os
import socket
import sys
from time import perf_counter

import orjson

from perf_benchmark.core.observation import ClockAnchor, ObservationSettings
from perf_benchmark.pipeline.contracts.observation import DiagnosticSource
from perf_benchmark.runtime.observation_discovery import DiscoveryRegistration

logger = logging.getLogger(__name__)


class ObservationRuntime:
    def __init__(self, settings: ObservationSettings):
        self.settings = settings
        self.port: int | None = None
        self.registration = DiscoveryRegistration(settings.discovery_dir)
        self.process: asyncio.subprocess.Process | None = None
        self.channel: socket.socket | None = None
        self.task: asyncio.Task[None] | None = None
        self.sources: dict[str, DiagnosticSource] = {}
        self.dropped = 0
        self.degraded = False
        self.closed = False
        self.outputs_sent = self.outputs_dropped = 0
        self.summary = None

    @staticmethod
    def validate(settings):
        if not settings.enabled:
            return
        try:
            modules = ["prometheus_client", "psutil"]
            if settings.trace_mode != "benchmark":
                modules.append("opentelemetry.sdk")
                if settings.trace_export:
                    modules.append("opentelemetry.exporter.otlp.proto.http.trace_exporter")
            installed = all(find_spec(name) is not None for name in modules)
        except ModuleNotFoundError:
            installed = False
        if not installed:
            raise ValueError("Observation requires: uv sync --extra observation")

    async def start(self, sink, saver_config, run, *, snapshot_socket=None, prometheus_url=None, display_pid=None):
        await sink.write_run_measurement("observation", {
            "clock": asdict(ClockAnchor.capture()), "settings": asdict(self.settings),
        })
        if not self.settings.enabled:
            return
        await sink.flush()
        parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        for channel in (parent, child):
            channel.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 32768)
            channel.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
        parent.setblocking(False)
        self.channel = parent
        try:
            extra = (["--snapshot-socket", snapshot_socket, "--prometheus-url", prometheus_url,
                      "--display-pid", str(display_pid)]
                     if snapshot_socket is not None else [])
            self.process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "perf_benchmark.app.observe", sink.artifact.location,
                "--registration-id", self.registration.identity,
                "--saver-config", orjson.dumps(saver_config).decode(), "--control-fd", str(child.fileno()),
                "--parent-pid", str(os.getpid()), "--settings", orjson.dumps(asdict(self.settings)).decode(),
                *extra, pass_fds=(child.fileno(),), stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            assert self.process.stdout is not None
            async with asyncio.timeout(10):
                response = await self.process.stdout.readline()
            try:
                ready = orjson.loads(response)
                assert ready["status"] == "ready" and ready["run_id"] == run.id
                assert type(ready["port"]) is int and 1 <= ready["port"] <= 65535
            except (ValueError, KeyError, TypeError, AssertionError) as exc:
                raise RuntimeError(f"Observer startup failed: {response.decode().strip()}") from exc
            self.port = ready["port"]
            logger.info("Observer ready: run=%s port=%s discovery=%s", run.id, self.port, self.settings.discovery_dir)
            self.watch("saver", sink.enable_diagnostics())
            self.task = asyncio.create_task(self._sample(run))
        except BaseException:
            await self.close()
            raise
        finally:
            child.close()

    def watch(self, name: str, source: DiagnosticSource) -> None:
        if self.settings.enabled:
            self.sources[name] = source

    def committed(self, receipt):
        self.notify({'commit_revision': receipt.revision})

    def sampled_output(self, sample):
        if not self.settings.enabled or self.settings.trace_mode != "debug":
            return
        if self.notify({'sampled_output': asdict(sample)}, maximum=32768):
            self.outputs_sent += 1
        else:
            self.outputs_dropped += 1

    def notify(self, values, *, maximum=4096):
        if self.channel is None or self.closed:
            return False
        try:
            data = orjson.dumps(values)
            if len(data) > maximum:
                raise ValueError("oversized observation notification")
            self.channel.send(data)
            return True
        except (BlockingIOError, OSError, ValueError, TypeError):
            self.dropped += 1
            return False

    async def _sample(self, run):
        assert self.process is not None
        exited = asyncio.create_task(self.process.wait())
        try:
            while True:
                deadline = perf_counter() + 1
                done, _ = await asyncio.wait({exited}, timeout=1)
                if done:
                    self.degraded = True
                    await asyncio.to_thread(self.registration.remove)
                    run.diagnostics.append({"observation": "process_exited", "code": exited.result()})
                    logger.warning("observation process exited; benchmark measurements remain active")
                    return
                self.notify({
                    "lag_seconds": max(0, perf_counter() - deadline), "dropped": self.dropped,
                    "output_messages": {"sent": self.outputs_sent, "dropped": self.outputs_dropped},
                    "writers": {name: asdict(value) if is_dataclass(value) else dict(value)
                                for name, source in self.sources.items() for value in (source(),)},
                })
        finally:
            exited.cancel()
            with suppress(asyncio.CancelledError):
                await exited

    async def close(self):
        if self.closed:
            return
        if self.task is not None:
            self.task.cancel()
            with suppress(asyncio.CancelledError):
                await self.task
        if self.process is not None and self.process.returncode is None:
            self.notify({"stop": True})
            try:
                async with asyncio.timeout(15):
                    await self.process.wait()
                if self.channel is not None:
                    with suppress(BlockingIOError, OSError, ValueError):
                        self.summary = orjson.loads(self.channel.recv(32768)).get("diagnostic_summary")
            except TimeoutError:
                self.degraded = True
                with suppress(ProcessLookupError):
                    self.process.kill()
                await self.process.wait()
                logger.warning("observation shutdown timed out; replay committed results with observe")
        if self.channel is not None:
            self.channel.close()
        if self.process is not None and self.process.returncode:
            self.degraded = True
            logger.warning("observation incomplete; replay committed results with observe (exit=%s)", self.process.returncode)
        await asyncio.to_thread(self.registration.remove)
        self.closed = True
