#!/usr/bin/env python3
"""Standalone executable script for static prefill and decode performance benchmarks.

Usage:
    python benchmarks/static_benchmark.py examples/static-benchmark.yaml
    python benchmarks/static_benchmark.py examples/static-benchmark.yaml --only-prefill
    python benchmarks/static_benchmark.py examples/static-benchmark.yaml --only-decode
    python benchmarks/static_benchmark.py examples/static-benchmark.yaml --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if sys.platform == "win32":
    from asyncio import run as run_async
else:
    from uvloop import run as run_async

from perf_benchmark.static_benchmark.cli import run_static_cli


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="static_benchmark",
        description="Static Prefill and Decode Performance Benchmark for LLM Serving Engines",
    )
    parser.add_argument("config", type=Path, help="Path to static benchmark YAML configuration file")
    parser.add_argument("--only-prefill", action="store_true", help="Run only prefill tests")
    parser.add_argument("--only-decode", action="store_true", help="Run only decode tests")
    parser.add_argument("--output", type=Path, help="Override output Excel path (.xlsx)")
    parser.add_argument("--base-url", help="Override service base URL")
    parser.add_argument("--model", help="Override model name")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and print execution plan without sending requests")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_async(run_static_cli(args))


if __name__ == "__main__":
    sys.exit(main())
