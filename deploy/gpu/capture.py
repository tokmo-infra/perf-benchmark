"""Bounded Nsight Systems diagnostic capture; never imported by the benchmark."""

import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid


REPORTS = ("cuda_gpu_kern_sum", "cuda_gpu_mem_time_sum", "cuda_api_sum")


def run(command, timeout):
    # Own only the diagnostic process group, including the inference process.
    with subprocess.Popen(command, start_new_session=True) as process:
        try:
            code = process.wait(timeout=timeout)
            if code:
                raise subprocess.CalledProcessError(code, command)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise


def capture(command):
    if not command:
        raise ValueError("Pass the inference executable and its arguments")
    run_id = os.environ["BENCHMARK_RUN_ID"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", run_id):
        raise ValueError("Invalid BENCHMARK_RUN_ID")
    duration = int(os.environ.get("CAPTURE_SECONDS", "30"))
    delay = int(os.environ.get("CAPTURE_DELAY_SECONDS", "60"))
    if not 1 <= duration <= 300 or not 0 <= delay <= 3600:
        raise ValueError("Capture requires 1..300 seconds and delay 0..3600 seconds")
    diagnostic_run_id = os.environ.get("DIAGNOSTIC_RUN_ID") or None
    if diagnostic_run_id is not None and not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", diagnostic_run_id):
        raise ValueError("Invalid DIAGNOSTIC_RUN_ID")
    diagnostic_id = str(uuid.uuid4())
    output = Path("/reports") / diagnostic_id
    output.mkdir(parents=True, exist_ok=False)
    version = subprocess.check_output(["nsys", "--version"], text=True, timeout=15).strip()
    gpus = subprocess.check_output([
        "nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader",
    ], text=True, timeout=15).splitlines()
    metadata = {
        "schema": "nsight-systems-summary-v2", "diagnostic_id": diagnostic_id,
        "benchmark_run_id": run_id, "diagnostic_run_id": diagnostic_run_id, "mode": "diagnostic", "nsys_version": version,
        "host": os.environ["DIAGNOSTIC_HOST"], "gpu_uuids": gpus,
        "association": "run-host-time-window", "started_unix_ns": time.time_ns(),
        "requested_delay_seconds": delay, "requested_duration_seconds": duration,
        "status": "capturing", "reports": {},
    }
    manifest = output / "capture.json"
    manifest.write_text(json.dumps(metadata, indent=2))
    try:
        run([
            "nsys", "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
            f"--delay={delay}", f"--duration={duration}", "--kill=sigterm",
            "--output", str(output / "report"), *command,
        ], timeout=delay + duration + 300)
        metadata["collection_command_finished_unix_ns"] = time.time_ns()
        for report in REPORTS:
            run([
                "nsys", "stats", "--report", report, "--format", "csv",
                "--output", str(output / "summary"), str(output / "report.nsys-rep"),
            ], timeout=120)
            metadata["reports"][report] = (
                "available" if (output / f"summary_{report}.csv").is_file()
                else "report_not_emitted"
            )
        metadata["status"] = "completed"
    except BaseException:
        metadata["status"] = "failed"
        raise
    finally:
        metadata["finished_unix_ns"] = time.time_ns()
        manifest.write_text(json.dumps(metadata, indent=2))
        print(json.dumps({"diagnostic_id": diagnostic_id, "status": metadata["status"]}))


if __name__ == "__main__":
    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    capture(sys.argv[1:])
