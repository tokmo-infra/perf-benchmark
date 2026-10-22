
from perf_benchmark.pipeline.context import SelectorContext
import pyarrow as pa
import pytest

from perf_benchmark.core.models import Run
from perf_benchmark.pipeline.contracts.normalization import NORMALIZED_SCHEMA
from perf_benchmark.pipeline.contracts.selector import SELECTED_SCHEMA
from perf_benchmark.pipeline.preparation.streaming import SelectorHost
from perf_benchmark.plugins.selector.token_length import (
    TokenLengthSelector,
    parse_settings,
)


def _row(order, session, key, counts):
    return {
        "source_order": order,
        "session_id": session,
        "turn_id": f"{session}-{key}",
        "turn_order_key": key,
        "input_kind": "chat",
        "context_mode": "recorded",
        "request_payload_offset": order,
        "request_payload_bytes": 1,
        "recorded_response_payload_offset": order + 10,
        "recorded_response_payload_bytes": 1,
        "estimated_input_tokens": counts[0],
        "reference_output_tokens": counts[1],
        "created_at": None,
        "metadata_json": "{}",
    }


async def _source(rows):
    yield pa.RecordBatch.from_pylist(rows, schema=NORMALIZED_SCHEMA)


async def test_selector_groups_interleaved_turns_and_keeps_complete_session(
    run_context,
) -> None:
    rows = [
        _row(0, "keep", 20, (22, 4)),
        _row(1, "drop", 10, (25, 3)),
        _row(2, "keep", 10, (21, 2)),
        _row(3, "drop", 20, (19, 4)),
    ]
    run = Run(name="selector")
    run_ctx = run_context("selector", {
        "input_tokens_gt": 20,
        "output_tokens_lt": 50,
        "limit": 1,
    }, run)
    output = await SelectorHost(
        TokenLengthSelector(parse_settings(dict(run_ctx.binding("selector").settings))),
        events=run_ctx.events,
    ).select(
        _source(rows),
        SelectorContext(run_ctx.cancellation),
    )
    selected = []
    for batch in output.batches:
        assert batch.schema.equals(SELECTED_SCHEMA)
        selected.extend(batch.to_pylist())

    assert [
        (row["session_id"], row["turn_order_key"], row["turn_ordinal"])
        for row in selected
    ] == [("keep", 10, 0), ("keep", 20, 1)]
    assert output.counts["selected"] == 1
    assert output.counts["selected_turns"] == 2


async def test_selector_uses_strict_boundaries_and_session_limit(run_context) -> None:
    rows = [
        _row(0, "s0", 0, (21, 2)),
        _row(1, "s1", 0, (22, 3)),
        _row(2, "s2", 0, (20, 2)),
        _row(3, "s3", 0, (25, 50)),
    ]
    run = Run(name="selector")
    run_ctx = run_context("selector", {
        "input_tokens_gt": 20,
        "output_tokens_lt": 50,
        "limit": 2,
    }, run)
    output = await SelectorHost(
        TokenLengthSelector(parse_settings(dict(run_ctx.binding("selector").settings))),
        events=run_ctx.events,
    ).select(
        _source(rows),
        SelectorContext(run_ctx.cancellation),
    )
    selected = [
        row
        for batch in output.batches
        for row in batch.to_pylist()
    ]
    assert [row["session_id"] for row in selected] == ["s0", "s1"]


async def test_selector_rejects_duplicate_turn_order_and_short_supply(
    run_context,
) -> None:
    run = Run(name="selector")
    run_ctx = run_context("selector", {
        "input_tokens_gt": 20,
        "output_tokens_lt": 50,
        "limit": 1,
    }, run)
    host = SelectorHost(
        TokenLengthSelector(parse_settings(dict(run_ctx.binding("selector").settings))),
        events=run_ctx.events,
    )
    ctx = SelectorContext(run_ctx.cancellation)
    with pytest.raises(ValueError, match="ambiguous"):
        await host.select(
            _source([_row(0, "s", 1, (21, 2)), _row(1, "s", 1, (22, 3))]),
            ctx,
        )

    with pytest.raises(ValueError, match="required 1"):
        await host.select(_source([_row(0, "s", 0, (20, 2))]), ctx)
