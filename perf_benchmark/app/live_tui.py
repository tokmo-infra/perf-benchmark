"""Independent Rich display for the observer's bounded local snapshot."""

import argparse
import asyncio
from contextlib import suppress
import os
from pathlib import Path
import signal
import sys
import tempfile
import termios
import time
import tty
from urllib.parse import urlsplit

from aiohttp import ClientError, ClientSession, ClientTimeout, UnixConnector
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table


class LiveDisplay:
    def __init__(self, prometheus_url: str):
        parsed = urlsplit(prometheus_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("--prometheus-url requires an HTTP(S) base URL without credentials or query")
        self.prometheus_url = prometheus_url
        self.directory = tempfile.TemporaryDirectory(prefix="perf-tui-")
        self.socket_path = str(Path(self.directory.name) / "snapshot.sock")
        self.process = None
        self.closed = False

    async def start(self):
        ready_read, ready_write = os.pipe()
        try:
            self.process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "perf_benchmark.app.live_tui", "--child", self.socket_path,
                "--ready-fd", str(ready_write), pass_fds=(ready_write,),
            )
            os.close(ready_write)
            ready_write = -1
            async with asyncio.timeout(5):
                ready = await asyncio.to_thread(os.read, ready_read, 1)
            if ready != b"1":
                raise RuntimeError("TUI process failed to start")
        except BaseException:
            await self.stop()
            raise
        finally:
            os.close(ready_read)
            if ready_write != -1:
                os.close(ready_write)

    async def stop(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.process is not None and self.process.returncode is None:
                with suppress(ProcessLookupError):
                    self.process.terminate()
                try:
                    async with asyncio.timeout(3):
                        await self.process.wait()
                except TimeoutError:
                    with suppress(ProcessLookupError):
                        self.process.kill()
                    await self.process.wait()
        finally:
            self.directory.cleanup()


def _fresh(at, *, max_age=4):
    if at is None:
        return "未观测"
    age = max(0, time.time() - at)
    return f"{age:.0f}s 前" if age <= max_age else f"过期 {age:.0f}s"


def render(snapshot, *, show_gpu, show_details=False):
    if not snapshot:
        return Panel("等待 Observer 提交首批数据…\nq 关闭界面，g 切换 GPU 面板；Ctrl+C 取消 Run",
                     title="perf-benchmark · 运行中")
    run = snapshot.get("run_id") or "等待 Run"
    head = Table.grid(expand=True)
    head.add_column()
    head.add_column(justify="right")
    head.add_row(f"Run {run} · {snapshot['run_status']}",
                 f"SQLite {_fresh(snapshot.get('updated_at'))} · 提交修订号 {snapshot['revision']} · 修订号差 {snapshot['pending_commits']}")
    trials = Table(title="Trial 与已提交请求", expand=True)
    for name in ("Trial", "状态", "Turn 成功/失败/取消", "Attempt 成功/失败/取消"):
        trials.add_column(name)
    for trial_id, trial in sorted(snapshot["trials"].items(), key=lambda pair: pair[1]["ordinal"]):
        counts = snapshot["counts"].get(trial_id, {})
        def cells(scope):
            values = counts.get(scope, {})
            return "/".join(str(values.get(status, 0)) for status in ("succeeded", "failed", "cancelled"))
        trials.add_row(trial["name"], trial["status"], cells("turn"), cells("attempt"))
    if not snapshot["trials"]:
        trials.add_row("等待 Trial", "—", "—", "—")
    load = snapshot["load"] if snapshot.get("load_at") and time.time() - snapshot["load_at"] <= 4 else {}
    resources = snapshot["resources"] if snapshot.get("resources_at") and time.time() - snapshot["resources_at"] <= 4 else {}
    pressure = Table(title=f"执行与资源诊断 · {_fresh(snapshot.get('load_at'))}", expand=True)
    pressure.add_column("项")
    pressure.add_column("值", justify="right")
    for key in ("active_sessions", "in_flight", "request_pending", "session_pending"):
        pressure.add_row(key, str(load[key]) if key in load else "不可用")
    for writer_name, values in sorted(snapshot.get("writers", {}).items() if load else ()):
        for key in ("queued", "processing", "processed", "batch_seconds"):
            pressure.add_row(f"{writer_name}.{key}", str(values[key]) if key in values else "不可用")
    for key in ("benchmark.cpu_percent", "benchmark.rss_bytes", "workers.cpu_percent", "workers.rss_bytes",
                "observer.cpu_percent", "observer.rss_bytes"):
        value = resources.get(key)
        pressure.add_row(key, f"{value / 1048576:.1f} MiB" if value is not None and key.endswith("rss_bytes")
                         else f"{value:.1f}%" if value is not None else "不可用")
    metrics = Table(title="指标 · 运行中为已提交样本的近似预览；正式值以 SQLite 为准", expand=True)
    for name in ("指标", "样本数", "近似 P50", "近似 P99"):
        metrics.add_column(name)
    for key, item in sorted(snapshot["preview"].items()):
        metrics.add_row(key, str(item["n"]), str(item["p50"]) if item.get("p50") is not None else "超出桶范围",
                        str(item["p99"]) if item.get("p99") is not None else "超出桶范围")
    if not snapshot["preview"]:
        metrics.add_row("等待已提交样本", "—", "—", "—")
    sections = [head, trials, pressure, metrics]
    if show_details and snapshot["final_metrics"]:
        final = Table(title="SQLite 正式指标点 · 成功 Run 为最终结果，其余为部分结果", expand=True)
        for name in ("指标", "值", "单位", "可用性／原因"):
            final.add_column(name)
        for key, item in sorted(snapshot["final_metrics"].items()):
            final.add_row(key, str(item["value"]) if item["value"] is not None else "—", item["unit"],
                          item["availability"] + (" · " + item["reason"] if item["reason"] else ""))
        sections.append(final)
    if show_gpu and snapshot["gpu"]["state"] != "not_configured":
        gpu = snapshot["gpu"]
        table = Table(title=f"GPU · {gpu['state']} · {_fresh(gpu.get('sample_at'), max_age=15)}", expand=True)
        for name in ("设备", "利用率 %", "显存 MiB", "功率 W", "样本"):
            table.add_column(name)
        for device in gpu["devices"]:
            table.add_row(device["id"], *(str(device.get(key, "—")) for key in
                                            ("utilization", "memory_mib", "power_watts")),
                          f"{device['state']} · {_fresh(device.get('sample_at'), max_age=15)}")
        if not gpu["devices"]:
            table.add_row("无样本", "—", "—", "—", "—")
        sections.append(table)
    output_messages = snapshot.get("output_messages", {})
    pressure.add_row("输出诊断消息 发送 / 丢弃",
                     f"{output_messages.get('sent', '未知')} / {output_messages.get('dropped', '未知')}")
    sections.append(Panel(f"q 关闭界面 · d 正式指标详情 · g 切换 GPU · Ctrl+C 取消 Run · 通知丢弃 {snapshot['notifications_dropped']}",
                          border_style="dim"))
    return Group(*sections)


async def child(socket_path, ready_fd=None):
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, stop.set)
    previous = termios.tcgetattr(sys.stdin.fileno())
    tty.setcbreak(sys.stdin.fileno())
    show_gpu = False
    show_details = False
    last_snapshot = None
    try:
        async with ClientSession(connector=UnixConnector(path=socket_path), timeout=ClientTimeout(total=1)) as client:
            with Live(render(None, show_gpu=False), console=Console(), refresh_per_second=2,
                      transient=True) as live:
                def on_input():
                    nonlocal show_gpu, show_details
                    keys = os.read(sys.stdin.fileno(), 64).decode(errors="ignore")
                    if not keys or "q" in keys:
                        stop.set()
                        loop.remove_reader(sys.stdin.fileno())
                        return
                    for key in keys:
                        if key == "g" and last_snapshot and last_snapshot["gpu"]["state"] != "not_configured":
                            show_gpu = not show_gpu
                        if key == "d":
                            show_details = not show_details
                    live.update(render(last_snapshot, show_gpu=show_gpu, show_details=show_details))

                loop.add_reader(sys.stdin.fileno(), on_input)
                if ready_fd is not None:
                    os.write(ready_fd, b"1")
                    os.close(ready_fd)
                while not stop.is_set():
                    snapshot = None
                    try:
                        async with client.get("http://localhost/snapshot") as response:
                            response.raise_for_status()
                            snapshot = await response.json()
                    except (ClientError, TimeoutError, OSError):
                        pass
                    if snapshot is not None:
                        last_snapshot = snapshot
                    live.update(render(last_snapshot, show_gpu=show_gpu, show_details=show_details))
                    with suppress(TimeoutError):
                        async with asyncio.timeout(1):
                            await stop.wait()
    finally:
        loop.remove_reader(sys.stdin.fileno())
        for signum in (signal.SIGTERM, signal.SIGINT):
            loop.remove_signal_handler(signum)
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, previous)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", required=True)
    parser.add_argument("--ready-fd", type=int)
    args = parser.parse_args()
    with suppress(KeyboardInterrupt):
        asyncio.run(child(args.child, args.ready_fd))


if __name__ == "__main__":
    main()
