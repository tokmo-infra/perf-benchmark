from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from typing import Protocol

import pyarrow as pa

from perf_benchmark.core.lifecycle import LOADER_BATCH_LOADED
from perf_benchmark.pipeline.context import LoaderContext
from perf_benchmark.pipeline.contracts.loader import (
    LoaderPlugin,
    RawBatchSource,
    SOURCE_ORDER_COLUMN,
)


_END = object()


class _EventSink(Protocol):
    async def emit(self, name: str, *, stage: str, data: Mapping[str, object]) -> object: ...


class _HostedDatasetSource:
    def __init__(
        self,
        raw: RawBatchSource,
        ctx: LoaderContext,
        events: _EventSink,
    ) -> None:
        self._raw = raw
        self._ctx = ctx
        self._events = events
        self._raw_schema = raw.schema
        self._consumed = False
        self._closed = False
        self.rows_read = 0

    @property
    def schema(self) -> pa.Schema:
        return self._raw_schema.append(pa.field(SOURCE_ORDER_COLUMN, pa.int64(), False))

    async def batches(self) -> AsyncIterator[pa.RecordBatch]:
        if self._consumed:
            raise RuntimeError("dataset source can only be consumed once")
        self._consumed = True
        iterator = iter(self._raw.batches())
        order = 0
        batch_index = 0
        while not self._closed and not self._ctx.cancellation.is_cancelled():
            value = await asyncio.to_thread(next, iterator, _END)
            if value is _END:
                break
            if not isinstance(value, pa.RecordBatch):
                raise TypeError("loader source must produce Arrow RecordBatch values")
            if not value.schema.equals(self._raw_schema, check_metadata=False):
                raise TypeError("dataset batches do not have one Arrow schema")
            orders = pa.array(range(order, order + value.num_rows), type=pa.int64())
            order += value.num_rows
            self.rows_read = order
            output = value.append_column(SOURCE_ORDER_COLUMN, orders)
            await self._events.emit(
                LOADER_BATCH_LOADED,
                stage="loader",
                data={
                    "batch_index": batch_index,
                    "rows": output.num_rows,
                    "source_bytes": output.nbytes,
                },
            )
            batch_index += 1
            yield output

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._raw.close()


class LoaderHost:
    def __init__(self, plugin: LoaderPlugin, *, events: _EventSink) -> None:
        self._plugin = plugin
        self._events = events

    async def open(self, ctx: LoaderContext) -> _HostedDatasetSource:
        raw = await self._plugin.open(ctx)
        try:
            schema = raw.schema
            if not isinstance(schema, pa.Schema):
                raise TypeError("loader source schema must be an Arrow schema")
            if SOURCE_ORDER_COLUMN in schema.names:
                raise ValueError(
                    f"loader data contains reserved column {SOURCE_ORDER_COLUMN}"
                )
        except BaseException:
            await raw.close()
            raise
        return _HostedDatasetSource(raw, ctx, self._events)
