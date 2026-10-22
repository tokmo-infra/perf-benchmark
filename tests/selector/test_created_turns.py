
from perf_benchmark.pipeline.context import SelectorContext
from datetime import datetime, timezone

import pyarrow as pa
import pytest

from perf_benchmark.core.models import Run
from perf_benchmark.pipeline.contracts.normalization import NORMALIZED_SCHEMA
from perf_benchmark.pipeline.contracts.selector import SELECTED_SCHEMA
from perf_benchmark.pipeline.preparation.streaming import SelectorHost
from perf_benchmark.plugins.selector.created_turns import (
    CreatedTurnsSelector,
    parse_settings,
)


def _row(order: int, session: str, key: int, created: int | None):
    return {
        "source_order": order,
        "session_id": session,
        "turn_id": f"{session}-{key}",
        "turn_order_key": key,
        "input_kind": "chat",
        "context_mode": "snapshot",
        "request_payload_offset": order,
        "request_payload_bytes": 1,
        "recorded_response_payload_offset": None,
        "recorded_response_payload_bytes": None,
        "estimated_input_tokens": 10,
        "reference_output_tokens": 2,
        "created_at": (
            datetime.fromtimestamp(created, timezone.utc)
            if created is not None else None
        ),
        "metadata_json": "{}",
    }


async def _source(rows):
    yield pa.RecordBatch.from_pylist(rows, schema=NORMALIZED_SCHEMA)


async def test_created_turns_selects_exact_global_limit_as_session_prefixes(
    run_context,
) -> None:
    rows = [
        _row(0, "a", 30, 30),
        _row(1, "b", 20, 20),
        _row(2, "a", 10, 10),
        _row(3, "b", 40, 40),
        _row(4, "a", 50, 50),
    ]
    run = Run(name="created")
    run_ctx = run_context("selector", {"limit": 4}, run)
    output = await SelectorHost(
        CreatedTurnsSelector(parse_settings({"limit": 4})), events=run_ctx.events
    ).select(_source(rows), SelectorContext(run_ctx.cancellation))
    selected = []
    for batch in output.batches:
        assert batch.schema.equals(SELECTED_SCHEMA)
        selected.extend(batch.to_pylist())

    assert [
        (row["session_id"], row["turn_order_key"], row["session_ordinal"], row["turn_ordinal"])
        for row in selected
    ] == [
        ("a", 10, 0, 0),
        ("a", 30, 0, 1),
        ("b", 20, 1, 0),
        ("b", 40, 1, 1),
    ]
    assert output.counts["selected"] == 2
    assert output.counts["selected_turns"] == 4


async def test_created_turns_rejects_ambiguous_or_missing_order(
    run_context,
) -> None:
    run = Run(name="created")
    run_ctx = run_context("selector", {"limit": 1}, run)
    host = SelectorHost(
        CreatedTurnsSelector(parse_settings({"limit": 1})), events=run_ctx.events
    )
    with pytest.raises(ValueError, match="ambiguous"):
        await host.select(
            _source([_row(0, "a", 10, 10), _row(1, "a", 10, 11)]),
            SelectorContext(run_ctx.cancellation),
        )

    with pytest.raises(ValueError, match="created_at"):
        await host.select(
            _source([_row(0, "a", 10, None)]),
            SelectorContext(run_ctx.cancellation),
        )


async def test_created_turns_fails_when_supply_is_short(run_context) -> None:
    run = Run(name="created")
    run_ctx = run_context("selector", {"limit": 2}, run)
    host = SelectorHost(
        CreatedTurnsSelector(parse_settings({"limit": 2})), events=run_ctx.events
    )
    with pytest.raises(ValueError, match="required 2"):
        await host.select(
            _source([_row(0, "a", 10, 10)]),
            SelectorContext(run_ctx.cancellation),
        )
