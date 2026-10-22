from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from perf_benchmark.pipeline.contracts.selector import (
    metadata_query,
    SelectionResult,
    SelectorPlugin,
    selection_indices,
)
from perf_benchmark.plugins import register_plugin


@dataclass(frozen=True, slots=True)
class _Settings:
    limit: int


def parse_settings(settings: dict[str, Any]) -> _Settings:
    unknown = sorted(set(settings) - {"limit"})
    if unknown:
        raise ValueError(
            "unknown created-turns selector settings: " + ", ".join(unknown)
        )
    limit = settings.get("limit")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("created-turns limit must be positive")
    return _Settings(limit)


@register_plugin(
    stage="selector",
    name="created-turns",
    description="Select the earliest complete Session prefixes by Turn creation time",
    settings_parser=parse_settings,
)
class CreatedTurnsSelector(SelectorPlugin):
    def __init__(self, settings: object) -> None:
        if not isinstance(settings, _Settings):
            raise TypeError("created-turns selector requires parsed settings")
        self._settings = settings

    def select(self, table):
        limit = self._settings.limit
        if table.num_rows < limit:
            raise ValueError(f"only {table.num_rows} Turns are available; required {limit}")
        with metadata_query(table) as db:
            if table.column("created_at").null_count:
                raise ValueError("created-turns requires created_at")
            db.execute(
                """
                CREATE TEMP TABLE chosen AS
                SELECT *, row_number() OVER (
                    PARTITION BY session_id ORDER BY turn_order_key
                ) - 1 AS turn_ordinal,
                row_number() OVER (
                    ORDER BY created_at, source_order, input_ordinal
                ) AS selection_rank
                FROM metadata ORDER BY selection_rank LIMIT ?
            """,
                [limit],
            )
            indices = selection_indices(
                db,
                """
                SELECT input_ordinal FROM chosen
                ORDER BY min(selection_rank) OVER (PARTITION BY session_id), turn_order_key
            """,
            )
            return SelectionResult(indices)


__all__ = ["CreatedTurnsSelector", "parse_settings"]
