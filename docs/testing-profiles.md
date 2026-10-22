# Test profiles

Use the short smoke profile for routine local checks:

    scripts/test-smoke.sh

It selects a small set of high-value checks for SWE tool-call normalization and
replay, protocol end-to-end behavior, native request cleanup, partial/missing
usage billing, Excel partial-value/append behavior, and run resource cleanup.
The selected parametrizations currently collect **26 tests**. This is a fast
signal, not a replacement for full coverage.

Run the complete Python suite before a release or when changing a public
contract, scheduler/lifecycle semantics, metrics formulas, or result storage:

    uv run pytest

For a focused change, run the affected module in addition to the smoke profile,
for example `uv run pytest tests/normalization/` after normalization changes.
Executor/Rust, frontend, real-service, and performance acceptance checks remain
separate and should be run when their respective code or behavior changes.

This only reduces the default *developer regression check*. It does not change
any benchmark workload, selected sessions, concurrency, users, or sample count.
