from __future__ import annotations

from perf_benchmark.pipeline.context import NormalizationContext, SelectorContext

import json

import pyarrow as pa
import pytest

from perf_benchmark.core.models import Run
from perf_benchmark.core.tokenization import ModelTokenizer
from perf_benchmark.pipeline.contracts.normalization import NORMALIZED_SCHEMA
from perf_benchmark.pipeline.preparation.datasets import SOURCE_ORDER_COLUMN
from perf_benchmark.pipeline.preparation.payloads import PayloadRef
from perf_benchmark.pipeline.preparation.streaming import NormalizationHost, SelectorHost
from perf_benchmark.plugins.normalization.openai_messages import (
    OpenAIMessagesNormalization,
    parse_settings,
)
from perf_benchmark.plugins.selector.token_length import (
    TokenLengthSelector,
    parse_settings as parse_selector_settings,
)


class _Source:
    def __init__(self, batches):
        self.values = batches

    @property
    def schema(self):
        return pa.Table.from_pylist(self.values[0]).schema.append(pa.field(SOURCE_ORDER_COLUMN, pa.int64()))

    async def close(self):
        pass

    async def batches(self):
        order = 0
        for records in self.values:
            table = pa.Table.from_pylist(records)
            orders = pa.array(range(order, order + len(records)), type=pa.int64())
            order += len(records)
            yield table.append_column(SOURCE_ORDER_COLUMN, orders).to_batches()[0]


def _record(identifier, session, created, content="hello"):
    return {
        "id": identifier,
        "token_name": session,
        "created": created,
        "messages": [{"role": "user", "content": content}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
    }


async def _normalize(records, run_context, settings=None, *, batches=False):
    run = Run(name="openai-messages")
    values = settings or {}
    ctx = run_context("normalization", values, run)
    plugin_ctx = NormalizationContext(ctx.base_dir, ctx.cancellation)
    source = _Source(records if batches else [records])
    rows = []
    host = NormalizationHost(
        OpenAIMessagesNormalization(parse_settings(values)),
        run_id=run.id,
        payloads=ctx.payloads,
        events=ctx.events,
    )
    async for batch in host.transform(source, plugin_ctx):
        rows.extend(batch.to_pylist())
    return rows, ctx


async def _payload(ctx, row):
    ctx.payloads.seal()
    ref = PayloadRef(row["request_payload_offset"], row["request_payload_bytes"])
    return json.loads(bytes(ctx.payloads.view(ref)))


async def test_single_turn_keeps_same_token_name_in_separate_sessions(
    run_context,
) -> None:
    first = _record("request-1", "source-session", 1_700_000_000)
    second = _record("request-2", "source-session", "2023-11-14T22:13:21Z")
    rows, _ = await _normalize([[first], [second]], run_context, batches=True)
    assert len({row["session_id"] for row in rows}) == 2
    assert [json.loads(row["metadata_json"])["source_session_id"] for row in rows] == [
        "source-session",
        "source-session",
    ]
    assert [row["estimated_input_tokens"] for row in rows] == [11, 11]
    assert [row["reference_output_tokens"] for row in rows] == [7, 7]
    assert all(row["created_at"].tzinfo is not None for row in rows)


async def test_snapshot_preserves_tools_tool_choice_and_tool_calls(run_context) -> None:
    record = _record("request", "session", 1_700_000_000)
    record["messages"] = [
        {"role": "system", "content": "system"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "result"},
    ]
    record["tools"] = [
        {
            "type": "function",
            "function": {"name": "lookup", "parameters": {"type": "object"}},
        }
    ]
    record["tool_choice"] = {"type": "function", "function": {"name": "lookup"}}
    rows, ctx = await _normalize([record], run_context)
    payload = await _payload(ctx, rows[0])
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["tool_calls"] == record["messages"][1]["tool_calls"]
    assert payload["messages"][2]["tool_call_id"] == "call-1"
    assert payload["tools"] == record["tools"]
    assert payload["tool_choice"] == record["tool_choice"]


async def test_declared_top_level_and_message_counts_skip_tokenizer(
    run_context,
    monkeypatch,
) -> None:
    record = _record("request", "session", 1_700_000_000)
    record.pop("usage")
    record["input_tokens"] = 23
    record["output_tokens"] = 5
    message_counted = _record("message-counted", "session", 1_700_000_001)
    message_counted.pop("usage")
    message_counted["messages"][0]["token_count"] = 13

    def unexpected_load(*args, **kwargs):
        raise AssertionError("tokenizer must not be loaded")

    monkeypatch.setattr(
        "perf_benchmark.pipeline.preparation.streaming.ModelTokenizer",
        unexpected_load,
    )
    rows, _ = await _normalize([record, message_counted], run_context)

    assert [row["estimated_input_tokens"] for row in rows] == [23, 13]
    assert [row["reference_output_tokens"] for row in rows] == [5, None]


async def test_multi_turn_emits_created_order_keys_without_sorting(run_context) -> None:
    records = [
        _record("a-later", "session-a", 30),
        _record("b-only", "session-b", 20),
        _record("a-earlier", "session-a", 10),
        _record("a-tied", "session-a", 10),
    ]
    rows, _ = await _normalize(records, run_context, {"multi_turn": True})
    source_ids = [json.loads(row["metadata_json"])["source_turn_id"] for row in rows]
    assert source_ids == ["a-later", "b-only", "a-earlier", "a-tied"]
    assert [row["turn_order_key"] for row in rows] == [30_000_000, 20_000_000, 10_000_000, 10_000_000]
    assert rows[0]["session_id"] == rows[2]["session_id"] == rows[3]["session_id"]
    assert rows[1]["session_id"] != rows[0]["session_id"]


async def test_live_continuation_is_explicitly_unsupported(run_context) -> None:
    with pytest.raises(ValueError, match="does not support live_continuation"):
        await _normalize(
            [_record("one", "session", 10)],
            run_context,
            {"multi_turn": True, "live_continuation": True},
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.pop("id"), "id"),
        (lambda row: row.pop("token_name"), "token_name"),
        (lambda row: row.pop("created"), "created"),
        (lambda row: row.update(messages="bad"), "messages"),
        (lambda row: row.update(created="2026-01-01"), "timezone"),
    ],
)
async def test_invalid_records_fail(mutation, message, run_context) -> None:
    record = _record("one", "session", 10)
    mutation(record)
    with pytest.raises((TypeError, ValueError), match=message):
        await _normalize([record], run_context)


@pytest.mark.parametrize("part", [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,PRIVATE_IMAGE"}},
    {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,PRIVATE_VIDEO"}},
    {"type": "input_audio", "input_audio": {"data": "PRIVATE_AUDIO", "format": "wav"}},
])
@pytest.mark.parametrize("declaration", ["input_tokens", "usage"])
async def test_media_requires_complete_declared_count_without_tokenizing_body(
    part, declaration, run_context, monkeypatch,
):
    record = _record("media", "session", 10)
    record["messages"][0]["content"] = [{"type": "text", "text": "describe"}, part]
    if declaration == "input_tokens":
        record.pop("usage")
        record["input_tokens"] = 128
    else:
        record["usage"]["prompt_tokens"] = 128
        record["input_tokens"] = 999

    def unexpected_estimation(*args, **kwargs):
        raise AssertionError("media body must not be serialized for text token estimation")

    monkeypatch.setattr(
        "perf_benchmark.plugins.normalization.openai_messages.compact_json_text",
        unexpected_estimation,
    )
    monkeypatch.setattr(
        "perf_benchmark.pipeline.preparation.streaming.ModelTokenizer",
        unexpected_estimation,
    )
    rows, ctx = await _normalize([record], run_context)
    assert rows[0]["estimated_input_tokens"] == 128
    content = (await _payload(ctx, rows[0]))["messages"][0]["content"]
    assert content[0]["text"] == "describe"
    assert all(content[1][key] == value for key, value in part.items())


@pytest.mark.parametrize("declaration", [
    {}, {"input_tokens": None}, {"usage": {"prompt_tokens": None}},
])
async def test_media_without_complete_declared_count_fails_preparation(declaration, run_context):
    record = _record("media", "session", 10)
    record.pop("usage")
    record.update(declaration)
    record["messages"][0].update(
        content=[{"type": "image_url", "image_url": {"url": "data:image/png;base64,PRIVATE"}}],
        token_count=128,
    )
    with pytest.raises(ValueError, match="non-text message parts require explicit"):
        await _normalize([record], run_context)


@pytest.mark.parametrize("name", ["input_tokens", "usage.prompt_tokens"])
@pytest.mark.parametrize("value", [-1, True, 1.5, "128"])
async def test_media_invalid_declared_count_fails_preparation(name, value, run_context):
    record = _record("media", "session", 10)
    record.pop("usage")
    record["messages"][0]["content"] = [{"type": "input_audio", "input_audio": {"data": "PRIVATE", "format": "wav"}}]
    if name == "input_tokens":
        record["input_tokens"] = value
    else:
        record["usage"] = {"prompt_tokens": value}
        record["input_tokens"] = 128
    with pytest.raises((TypeError, ValueError), match=name.replace(".", r"\.")):
        await _normalize([record], run_context)


async def test_media_selector_uses_declared_count_instead_of_encoded_body_size(run_context, monkeypatch):
    records = [_record("low", "low", 10), _record("high", "high", 11)]
    for record, count in zip(records, [10, 30], strict=True):
        record["messages"][0]["content"] = [{
            "type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 4096},
        }]
        record["usage"]["prompt_tokens"] = count

    def unexpected_tokenizer(*args, **kwargs):
        raise AssertionError("media must use the declared count")

    monkeypatch.setattr("perf_benchmark.pipeline.preparation.streaming.ModelTokenizer", unexpected_tokenizer)
    rows, ctx = await _normalize(records, run_context)

    async def source():
        yield pa.RecordBatch.from_pylist(rows, schema=NORMALIZED_SCHEMA)

    selected = await SelectorHost(
        TokenLengthSelector(parse_selector_settings({
            "input_tokens_gt": 20, "output_tokens_lt": 10, "limit": 1,
        })),
        events=ctx.events,
    ).select(source(), SelectorContext(ctx.cancellation))
    output = [row for batch in selected.batches for row in batch.to_pylist()]
    assert [row["estimated_input_tokens"] for row in output] == [30]
    assert [json.loads(row["metadata_json"])["source_turn_id"] for row in output] == ["high"]


async def test_text_snapshot_without_declared_count_keeps_tokenizer_estimate(run_context, tokenizer_path):
    record = _record("text", "session", 10)
    record.pop("usage")
    record["messages"][0]["content"] = [{"type": "text", "text": "hello world"}]
    rows, ctx = await _normalize([record], run_context, {"tokenizer_path": str(tokenizer_path)})
    expected = ModelTokenizer(tokenizer_path).count_batch([
        json.dumps({"messages": record["messages"]}, ensure_ascii=False, separators=(",", ":")),
    ])[0]
    assert expected > 0
    assert rows[0]["estimated_input_tokens"] == expected
    assert (await _payload(ctx, rows[0]))["messages"] == record["messages"]
