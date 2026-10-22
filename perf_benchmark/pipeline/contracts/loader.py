from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import Protocol

import pyarrow as pa

from perf_benchmark.pipeline.context import LoaderContext


ArrowBatchStream = AsyncIterator[pa.RecordBatch]
SOURCE_ORDER_COLUMN = "__perf_source_order"


class RawBatchSource(Protocol):
    """A loader-owned, synchronous Arrow source consumed by LoaderHost."""

    @property
    def schema(self) -> pa.Schema: ...

    def batches(self) -> Iterator[pa.RecordBatch]: ...

    async def close(self) -> None: ...


class DatasetSource(Protocol):
    """Canonical asynchronous source exposed to the normalization stage."""

    @property
    def schema(self) -> pa.Schema: ...

    def batches(self) -> ArrowBatchStream: ...

    async def close(self) -> None: ...


class LoaderPlugin(ABC):
    @abstractmethod
    async def open(self, ctx: LoaderContext) -> RawBatchSource:
        raise NotImplementedError
