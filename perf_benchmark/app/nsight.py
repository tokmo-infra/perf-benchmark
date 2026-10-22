"""Offline Nsight Systems CSV summaries to Loki; no benchmark facts are written."""

import argparse
import csv
from decimal import Decimal, InvalidOperation
import heapq
import json
from pathlib import Path
from typing import NotRequired, TypedDict
import sys
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


REPORTS = {
    "cuda_gpu_kern_sum": ("Name", "Instances"),
    "cuda_gpu_mem_time_sum": ("Operation", "Count"),
    "cuda_api_sum": ("Name", "Num Calls"),
}


def number(value):
    try:
        result = Decimal(value)
    except (InvalidOperation, TypeError) as exc:
        raise ValueError("Invalid Nsight numeric field") from exc
    if not result.is_finite() or result < 0 or result > Decimal("1e30"):
        raise ValueError("Nsight numeric fields must be finite and nonnegative")
    return result


class SummaryEntry(TypedDict):
    name: str
    total_ns: str
    count: int
    share_percent: float | None


class Summary(TypedDict):
    availability: str
    reason: str
    total_ns: NotRequired[str | None]
    count: NotRequired[int | None]
    top: list[SummaryEntry]


def summarize(path, report) -> Summary:
    """Validate known CSV columns, retaining at most 20 entries by cumulative duration."""
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("Nsight summary exceeds 32 MiB")
    name_key, count_key = REPORTS[report]
    top = []
    total = 0
    count = 0
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {name_key, count_key, "Total Time (ns)"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Unsupported {report} columns; expected {sorted(required)}")
        for index, row in enumerate(reader):
            if index >= 100_000:
                raise ValueError("Nsight summary exceeds 100000 entries")
            name = row[name_key]
            if not name or len(name) > 4096 or None in row:
                raise ValueError("Invalid Nsight summary row")
            elapsed = number(row["Total Time (ns)"])
            calls = number(row[count_key])
            if calls != calls.to_integral_value() or elapsed != elapsed.to_integral_value():
                raise ValueError("Nsight counts and total nanoseconds must be integers")
            elapsed = int(elapsed)
            total += elapsed
            count += int(calls)
            item = (elapsed, index, name, int(calls))
            if len(top) < 20:
                heapq.heappush(top, item)
            elif item > top[0]:
                heapq.heapreplace(top, item)
    return {
        "availability": "available" if top else "unavailable",
        "reason": "" if top else "no_records",
        "total_ns": str(total) if top else None, "count": count if top else None,
        "top": [{"name": name, "total_ns": str(elapsed), "count": calls,
                 "share_percent": elapsed / total * 100 if total else None}
                for elapsed, _, name, calls in sorted(top, reverse=True)],
    }


def public_url(value):
    parsed = urlsplit(value)
    if (len(value) > 4096 or parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment):
        raise ValueError("Use an HTTP(S) URL without credentials, query or fragment")
    return value


def records(directory, expected_version, artifact_url):
    manifest = directory / "capture.json"
    if manifest.stat().st_size > 65536:
        raise ValueError("Capture metadata exceeds 64 KiB")
    meta = json.loads(manifest.read_text())
    if meta.get("schema") != "nsight-systems-summary-v2" or meta.get("status") != "completed":
        raise ValueError("Requires a completed nsight-systems-summary-v2 capture")
    if meta.get("nsys_version") != expected_version:
        raise ValueError("Nsight version does not match the explicitly selected version")
    if meta.get("mode") != "diagnostic" or meta.get("association") != "run-host-time-window":
        raise ValueError("Unsupported diagnostic association")
    for key in ("diagnostic_id", "benchmark_run_id", "host", "nsys_version"):
        if not isinstance(meta[key], str) or not 1 <= len(meta[key]) <= 256:
            raise ValueError(f"Invalid capture field: {key}")
    if (not isinstance(meta["gpu_uuids"], list) or not meta["gpu_uuids"]
            or len(meta["gpu_uuids"]) > 256
            or any(not isinstance(gpu, str) or not 1 <= len(gpu) <= 128
                   for gpu in meta["gpu_uuids"])):
        raise ValueError("Invalid GPU identity list")
    for key in ("started_unix_ns", "finished_unix_ns", "requested_delay_seconds",
                "requested_duration_seconds"):
        if type(meta[key]) is not int or meta[key] < 0:
            raise ValueError(f"Invalid capture time: {key}")
    if meta["finished_unix_ns"] < meta["started_unix_ns"]:
        raise ValueError("Capture end precedes start")
    diagnostic_run = meta["diagnostic_run_id"]
    if diagnostic_run is not None and (not isinstance(diagnostic_run, str) or not 1 <= len(diagnostic_run) <= 128):
        raise ValueError("Invalid diagnostic_run_id")
    common = {key: meta[key] for key in (
        "diagnostic_id", "benchmark_run_id", "diagnostic_run_id", "host", "gpu_uuids", "nsys_version",
        "mode", "association", "started_unix_ns", "finished_unix_ns",
        "requested_delay_seconds", "requested_duration_seconds",
    )}
    common["artifact_url"] = public_url(artifact_url)
    common["schema"] = meta["schema"]
    result = []
    for report in REPORTS:
        path = directory / f"summary_{report}.csv"
        availability = meta["reports"][report]
        if availability == "available":
            summary: Summary = summarize(path, report)
        elif availability == "report_not_emitted" and not path.exists():
            summary = {"availability": "unavailable", "reason": availability, "top": []}
        else:
            raise ValueError(f"Invalid report availability: {report}")
        result.append({**common, "report": report, "kind": "summary",
                       **{k: v for k, v in summary.items() if k != "top"}})
        for rank, item in enumerate(summary["top"], 1):
            result.append({**common, "report": report, "kind": "entry", "rank": rank, **item})
    return result


def push(endpoint, entries):
    endpoint = public_url(endpoint)
    imported = time.time_ns()
    # Import time is Loki event time; capture bounds remain explicit fields.
    # IDs/names stay in JSON, never in Loki's index labels.
    payload = json.dumps({"streams": [{
        "stream": {"service_name": "perf-nsight", "mode": "diagnostic"},
        "values": [[str(imported + i), json.dumps({**entry, "imported_unix_ns": imported})]
                   for i, entry in enumerate(entries)],
    }]}, allow_nan=False).encode()
    if len(payload) > 1024 * 1024:
        raise ValueError("Diagnostic payload exceeds 1 MiB")
    request = Request(endpoint, data=payload, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=15) as response:
        if response.status != 204:
            raise ValueError(f"Unexpected Loki response: {response.status}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--nsys-version", required=True, help="Exact nsys --version output")
    parser.add_argument("--artifact-url", required=True, help="Published original report URL")
    parser.add_argument("--loki-endpoint", default="http://127.0.0.1:3100/loki/api/v1/push")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        entries = records(args.directory, args.nsys_version, args.artifact_url)
        if args.dry_run:
            print(json.dumps(entries, indent=2, allow_nan=False))
        else:
            push(args.loki_endpoint, entries)
            print(json.dumps({"status": "imported", "records": len(entries),
                              "diagnostic_id": entries[0]["diagnostic_id"]}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as exc:
        print(f"nsight-import: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
