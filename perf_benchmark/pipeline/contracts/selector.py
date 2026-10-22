from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MappingProxyType

import duckdb
import pyarrow as pa


from perf_benchmark.pipeline.contracts.normalization import NORMALIZED_SCHEMA

# Selection adds identities/order only; all normalized fields retain their types.
SELECTED_SCHEMA = NORMALIZED_SCHEMA.insert(1, pa.field("session_ordinal", pa.int32(), nullable=False))
SELECTED_SCHEMA = SELECTED_SCHEMA.insert(5, pa.field("turn_ordinal", pa.int32(), nullable=False))



@dataclass(frozen=True, slots=True)
class SelectionResult:
    """Final normalized row order selected by a Selector Plugin."""

    indices: pa.Array
    diagnostics: Mapping[str, int] = field(
        default_factory=lambda: MappingProxyType({})
    )


class SelectorPlugin(ABC):
    @abstractmethod
    def select(self, table: pa.Table) -> SelectionResult:
        raise NotImplementedError


@contextmanager
def metadata_query(table: pa.Table) -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """Expose canonical metadata plus a stable zero-based input ordinal."""

    with duckdb.connect(config={"threads": 1}) as db:
        db.register("raw_metadata", table)
        db.execute("""
            CREATE VIEW metadata AS
            SELECT *, row_number() OVER () - 1 AS input_ordinal FROM raw_metadata
        """)
        yield db


def selection_indices(
    db: duckdb.DuckDBPyConnection,
    query: str,
    parameters: Sequence[object] = (),
) -> pa.Array:
    rows = db.execute(query, parameters).fetchall()
    if any(len(row) != 1 for row in rows):
        raise TypeError("selector query must return only input_ordinal")
    return pa.array((row[0] for row in rows), type=pa.int64())
