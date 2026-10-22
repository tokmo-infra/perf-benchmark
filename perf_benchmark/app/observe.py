"""Standalone/Host-spawned observation process; SQLite remains read-only."""

import argparse
import asyncio
from contextlib import suppress
import math
from pathlib import Path
from typing import cast
from perf_benchmark.pipeline.contracts.saver import SaverPlugin
import socket
import signal
import sys
from time import monotonic, time

import orjson
import yaml
from aiohttp import ClientError, ClientSession, ClientTimeout, web

from perf_benchmark.core.observation import ObservationSettings
from perf_benchmark.runtime.observation_discovery import DiscoveryRegistration


async def _gpu_samples(snapshot, run_id: str, base_url: str) -> None:
    names = {"utilization": "DCGM_FI_DEV_GPU_UTIL", "memory_mib": "DCGM_FI_DEV_FB_USED",
             "power_watts": "DCGM_FI_DEV_POWER_USAGE"}
    try:
        async with ClientSession(timeout=ClientTimeout(total=3)) as client:
            async def query(field, metric, original_time):
                selector = f'{metric}{{benchmark_run_id={orjson.dumps(run_id).decode()},observer_replay="0",job="gpu"}}'
                expression = f"timestamp({selector})" if original_time else selector
                async with client.get(base_url + "/api/v1/query", params={"query": expression}) as response:
                    response.raise_for_status()
                    data = await response.json()
                if data.get("status") != "success" or data.get("data", {}).get("resultType") != "vector":
                    raise ValueError("invalid Prometheus response")
                return field, original_time, data["data"]["result"]
            results = await asyncio.gather(*(query(field, name, original_time)
                                             for field, name in names.items() for original_time in (False, True)))
        devices = {}
        timestamps = {}
        for field, original_time, series in results:
            for item in series:
                labels, sample = item["metric"], item["value"]
                identity = labels.get("UUID") or labels.get("gpu") or labels.get("instance")
                value = float(sample[1])
                if not identity or not math.isfinite(value):
                    continue
                if original_time:
                    timestamps[identity, field] = value
                else:
                    devices.setdefault(identity, {"id": identity, "instance": labels.get("instance", "")})[field] = value
        valid = []
        for identity, device in devices.items():
            stamps = [timestamps.get((identity, field)) for field in names if field in device]
            stamps = [stamp for stamp in stamps if stamp is not None and math.isfinite(stamp)]
            device["sample_at"] = min(stamps) if len(stamps) == len([field for field in names if field in device]) and stamps else None
            device["state"] = "available" if device["sample_at"] is not None and time() - device["sample_at"] <= 15 else "stale"
            if device["sample_at"] is not None:
                valid.append(device["sample_at"])
        snapshot.gpu = {"state": "no_samples" if not devices else "available" if any(
            device["state"] == "available" for device in devices.values()) else "stale",
                        "devices": list(devices.values()), "sample_at": max(valid) if valid else None}
        snapshot.gpu_at = time()
    except (ClientError, OSError, TimeoutError, ValueError, KeyError, TypeError, IndexError):
        snapshot.gpu = {"state": "unavailable", "devices": []}
        snapshot.gpu_at = time()


async def _gpu_loop(snapshot, run_id: str, base_url: str) -> None:
    while True:
        await _gpu_samples(snapshot, run_id, base_url)
        await asyncio.sleep(5)


class ProcessSampler:
    """Sample the host and its workers externally; the observer is a separate resource group."""

    def __init__(self, parent_pid, display_pid=None):
        import psutil

        self.psutil = psutil
        self.observer = psutil.Process()
        self.parent = psutil.Process(parent_pid) if parent_pid else None
        self.display_pid = display_pid
        self.previous = {}

    def sample(self):
        now, samples, previous = monotonic(), {}, {}
        groups = {"observer": [self.observer]}
        alive = self.parent is None or self.parent.is_running()
        if self.parent is not None and alive:
            groups["benchmark"] = [self.parent]
            try:
                groups["workers"] = [p for p in self.parent.children(recursive=True)
                                     if p.pid not in {self.observer.pid, self.display_pid}]
            except self.psutil.Error:
                pass
        for name, processes in groups.items():
            rss, cpu, valid = 0, 0.0, True
            try:
                for process in processes:
                    with process.oneshot():
                        identity = process.pid, process.create_time()
                        times = process.cpu_times()
                        elapsed_cpu = times.user + times.system
                        rss += process.memory_info().rss
                    before = self.previous.get(identity)
                    valid &= before is not None
                    if before is not None:
                        cpu += max(0, elapsed_cpu - before[0]) / max(now - before[1], 1e-9) * 100
                    previous[identity] = elapsed_cpu, now
                samples[name, "rss_bytes"] = rss
                if valid:
                    samples[name, "cpu_percent"] = cpu
            except self.psutil.Error:
                continue
        for metric in ("cpu_percent", "rss_bytes"):
            if all((name, metric) in samples for name in ("benchmark", "workers")):
                samples["benchmark_total", metric] = sum(samples[name, metric] for name in ("benchmark", "workers"))
        self.previous = previous
        return samples, alive


def discovery_targets(settings, run_id, port, identity, *, replay):
    labels = {"benchmark_run_id": run_id, "observer_id": identity,
              "observer_replay": str(int(replay))}
    groups = {"benchmark": [{
        "targets": [f"{settings.advertise_host}:{port}"],
        "labels": {**labels, "service_name": "perf-benchmark"},
    }]}
    for kind in ("engines", "gpu", "profiles"):
        groups[kind] = []
        # Replay exposes saved benchmark results, not current remote service state.
        for target in (() if replay else getattr(settings.targets, kind)):
            target_labels = {**labels, "service_name": target.service_name,
                             "__scheme__": target.scheme}
            if kind != "profiles":
                target_labels["__metrics_path__"] = target.metrics_path
            for key in ("engine", "model", "version"):
                if value := getattr(target, key):
                    target_labels["benchmark_" + key] = value
            groups[kind].append({"targets": [target.address], "labels": target_labels})
    return groups


async def serve(args, settings):
    from prometheus_client import CollectorRegistry, start_http_server
    from perf_benchmark.plugins import create_registry
    from perf_benchmark.integrations.observation_export import ResultObserver, SpanQueue, Telemetry, sampled_output_span

    source = orjson.loads(args.saver_config) if args.saver_config else {
        "plugin": "sqlite", "settings": {"path": str(args.database)},
    }
    if not isinstance(source, dict) or not isinstance(source.get("plugin"), str) or not isinstance(source.get("settings"), dict):
        raise ValueError("saver config must contain plugin and settings")
    plugin = cast(SaverPlugin, create_registry().bind("saver", source["plugin"], source["settings"]).plugin)
    reader = await asyncio.to_thread(plugin.open_reader, str(args.database.resolve()))
    registry = CollectorRegistry()
    telemetry = Telemetry(registry)
    telemetry.replay.set(args.parent_pid is None)
    http, thread = start_http_server(settings.port, addr=settings.host, registry=registry)
    exporter = spans = None
    if settings.trace_mode != "benchmark":
        if settings.trace_export:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            exporter = OTLPSpanExporter(endpoint=settings.otlp_endpoint, timeout=2)
        spans = SpanQueue(exporter, telemetry, slow_rate=settings.slow_spans_per_second)
    observer = ResultObserver(telemetry, spans, settings, replay=args.parent_pid is None)
    snapshot = observer.snapshot
    snapshot.gpu = {"state": "waiting" if settings.targets.gpu else "not_configured", "devices": []}
    snapshot_server = None
    gpu_task = None
    if args.snapshot_socket:
        async def read_snapshot(_request):
            return web.Response(body=orjson.dumps(snapshot.view()), content_type="application/json",
                                headers={"Cache-Control": "no-store"})
        app = web.Application()
        app.router.add_get("/snapshot", read_snapshot)
        snapshot_server = web.AppRunner(app, access_log=None)
        await snapshot_server.setup()
        await web.UnixSite(snapshot_server, args.snapshot_socket).start()
    channel = socket.socket(fileno=args.control_fd) if args.control_fd is not None else None
    if channel:
        channel.setblocking(False)
    sender = asyncio.create_task(spans.run()) if spans is not None else None
    deadline = None
    last_sample = last_control = monotonic()
    sampler = ProcessSampler(args.parent_pid, args.display_pid)
    await asyncio.to_thread(sampler.sample)
    registration = DiscoveryRegistration(settings.discovery_dir, args.registration_id)
    run_id = None
    resource_labels = set()
    cursor = None
    next_read = 0.0
    caught_up = ready = False
    try:
        while True:
            if channel:
                while True:
                    try:
                        packet = orjson.loads(channel.recv(32768))
                    except BlockingIOError:
                        break
                    if 'commit_revision' in packet:
                        next_read = 0.0
                    if 'sampled_output' in packet:
                        if spans is None or settings.trace_mode != "debug":
                            telemetry.dropped.labels('output_mode_disabled').inc()
                        else:
                            try:
                                sample = packet['sampled_output']
                                if sample['run_id'] != snapshot.run_id:
                                    raise ValueError('output trace belongs to another Run')
                                spans.offer(sampled_output_span(sample, observer.builder.clock), False)
                            except (ValueError, TypeError, KeyError):
                                telemetry.dropped.labels('invalid_output_diagnostic').inc()
                    if packet.get("stop") and deadline is None:
                        deadline = monotonic() + 10
                    if "lag_seconds" in packet:
                        last_control = monotonic()
                        snapshot.diagnostic(packet, time())
                        for kind, count in packet.get("output_messages", {}).items():
                            telemetry.resources.labels("benchmark", "output_messages_" + kind).set(count)
                        telemetry.resources.labels("benchmark", "loop_lag_seconds").set(packet["lag_seconds"])
                        telemetry.resources.labels("benchmark", "notifications_dropped").set(packet["dropped"])
                        for name, values in packet["writers"].items():
                            if name == "load":
                                telemetry.update_load(values)
                                continue
                            for metric, value in values.items():
                                telemetry.resources.labels(name, metric).set(value)
            if monotonic() - last_sample >= 1:
                last_sample = monotonic()
                samples, parent_alive = await asyncio.to_thread(sampler.sample)
                snapshot.resource_sample(samples, time())
                for labels in resource_labels - samples.keys():
                    telemetry.resources.remove(*labels)
                for labels, value in samples.items():
                    telemetry.resources.labels(*labels).set(value)
                resource_labels = set(samples)
                if not parent_alive and deadline is None:
                    deadline = monotonic() + 10
                telemetry.resources.labels("benchmark", "state_age_seconds").set(monotonic() - last_control)
                if caught_up:
                    telemetry.retire_trials()
            if monotonic() < next_read:
                await asyncio.sleep(min(0.1, next_read - monotonic()))
                continue
            page = await asyncio.to_thread(reader.read_updates, cursor, settings.page_size)
            if run_id is None and page.updates:
                run_id = page.updates[0].run_id
            await observer.consume(page)
            cursor = page.cursor
            snapshot.revision = cursor.revision
            snapshot.pending_commits = page.commit_revision - cursor.revision
            snapshot.updated_at = time()
            caught_up = page.caught_up
            telemetry.ready.set(caught_up)
            telemetry.progress.set(cursor.revision)
            telemetry.backlog.set(page.commit_revision - cursor.revision)
            telemetry.resources.labels("observer", "trace_pending_spans").set(spans.count if spans else 0)
            telemetry.resources.labels("observer", "estimated_trace_bytes").set(spans.bytes if spans else 0)
            telemetry.updated.set(time())
            if caught_up and not ready:
                if run_id is None:
                    raise ValueError("Observer database has no Run identity")
                await asyncio.to_thread(registration.publish, discovery_targets(
                    settings, run_id, http.server_port, registration.identity, replay=args.parent_pid is None,
                ))
                print(orjson.dumps({"status": "ready", "run_id": run_id, "port": http.server_port}).decode(), flush=True)
                ready = True
                if settings.targets.gpu and args.snapshot_socket and args.parent_pid is not None:
                    gpu_task = asyncio.create_task(_gpu_loop(snapshot, run_id, args.prometheus_url.rstrip("/")))
            if deadline is not None and monotonic() >= deadline and caught_up:
                break
            if caught_up:
                next_read = monotonic() + settings.poll_seconds
            else:
                await asyncio.sleep(0)
    finally:
        if gpu_task is not None:
            gpu_task.cancel()
            with suppress(asyncio.CancelledError):
                await gpu_task
        if snapshot_server is not None:
            await snapshot_server.cleanup()
            Path(args.snapshot_socket).unlink(missing_ok=True)
        await asyncio.to_thread(registration.remove)
        if spans is not None and sender is not None:
            await spans.finish(sender)
        await asyncio.to_thread(http.shutdown)
        http.server_close()
        await asyncio.to_thread(thread.join, 1)
        if channel:
            summary = {'complete': caught_up and (spans is None or not spans.incomplete and spans.count == 0),
                       'commit_revision': cursor.revision if cursor else None}
            totals = {name: 0.0 for name in ('trace_dropped','trace_export_failures','trace_exported_spans',
                                           'trace_processed_spans','trace_processed_request_spans','trace_sampled_out','trace_unavailable')}
            for metric in registry.collect():
                for sample in metric.samples:
                    if sample.name in {'perf_benchmark_trace_dropped_total','perf_benchmark_trace_export_failures_total',
                                       'perf_benchmark_trace_exported_spans_total','perf_benchmark_trace_processed_spans_total',
                                       'perf_benchmark_trace_processed_request_spans_total',
                                       'perf_benchmark_trace_sampled_out_total','perf_benchmark_trace_unavailable_total'}:
                        key = sample.name + ':' + sample.labels.get('reason','')
                        summary[key] = sample.value
                        totals[sample.name.removeprefix('perf_benchmark_').removesuffix('_total')] += sample.value
            summary.update(totals)
            with suppress(BlockingIOError, OSError):
                channel.send(orjson.dumps({'diagnostic_summary': summary}))
            channel.close()
    return 3 if spans is not None and (spans.incomplete or spans.count) else 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="perf-benchmark observe")
    parser.add_argument("database", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--settings", default="{}", help=argparse.SUPPRESS)
    parser.add_argument("--saver-config", help=argparse.SUPPRESS)
    parser.add_argument("--control-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--registration-id", help=argparse.SUPPRESS)
    parser.add_argument("--parent-pid", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--snapshot-socket", help=argparse.SUPPRESS)
    parser.add_argument("--display-pid", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--prometheus-url", default="http://127.0.0.1:9090", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        raw = yaml.safe_load(args.config.read_text()) if args.config else orjson.loads(args.settings)
        settings = ObservationSettings.parse(raw, base_dir=args.config.resolve().parent if args.config else None)
        async def run():
            task = asyncio.current_task()
            assert task is not None
            asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
            try:
                return await serve(args, settings)
            except asyncio.CancelledError:
                return 0
        return asyncio.run(run())
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        # Startup errors go to the Host handshake without leaking endpoint credentials.
        print(f"observer failed: {type(exc).__name__}", flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
