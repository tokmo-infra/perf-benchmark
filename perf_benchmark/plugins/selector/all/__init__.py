from collections.abc import Mapping
from typing import Any

from perf_benchmark.pipeline.contracts.selector import (
    metadata_query,
    SelectionResult,
    SelectorPlugin,
    selection_indices,
)
from perf_benchmark.plugins import register_plugin


def parse_settings(settings: dict[str, Any]) -> dict[str, Any]:
    if settings:
        raise ValueError("unknown all selector settings: " + ", ".join(sorted(settings)))
    return {}


@register_plugin(
    stage="selector",
    name="all",
    description="Keep every complete Session",
    settings_parser=parse_settings,
)
class AllSelector(SelectorPlugin):
    def __init__(self, settings: object) -> None:
        if not isinstance(settings, Mapping):
            raise TypeError("all selector requires parsed settings")

    def select(self, table):
        with metadata_query(table) as db:
            indices = selection_indices(
                db,
                """
                SELECT input_ordinal FROM metadata ORDER BY
                  min(source_order) OVER (PARTITION BY session_id),
                  min(input_ordinal) OVER (PARTITION BY session_id), turn_order_key
            """,
            )
            return SelectionResult(indices)


__all__ = ["AllSelector", "parse_settings"]
