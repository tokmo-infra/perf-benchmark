"""Durable one-shot CPU stop notice; no service operations or network I/O."""
import json
import os
from pathlib import Path


def emit_stop_notice(run_id, trial_id, proof):
    emit_notice('native-stop.json', {"event": "native_underfill_cpu_cancel_requested",
                "run_id": run_id, "trial_id": trial_id, "proof": proof})


def emit_safety_notice(run_id, trial_id, request, occupancy):
    emit_notice('safety-stop.json', dict(event='ttft_watchdog_safety_abort',
        run_id=run_id, trial_id=trial_id, request=request, occupancy=dict(occupancy),
        requires_diagnosis_before_any_next_stage=True, reason='actual_sent_request_first_output_exceeded_limit'))


def emit_notice(filename, value):
    directory = os.environ.get("PERF_CONTROL_DIR")
    if not directory:
        return
    root = Path(directory)
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("PERF_CONTROL_DIR must be an existing absolute stage directory")
    path = root / filename
    if path.exists():
        return
    temporary = root / ('.' + filename + '.tmp')
    with temporary.open("x") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
