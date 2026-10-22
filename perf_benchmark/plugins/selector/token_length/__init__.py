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
    input_tokens_gt: int
    output_tokens_lt: int
    limit: int


def parse_settings(settings: dict[str, Any]) -> _Settings:
    required = {"input_tokens_gt", "output_tokens_lt", "limit"}
    unknown = sorted(set(settings) - required)
    if unknown:
        raise ValueError(
            "unknown token-length selector settings: " + ", ".join(unknown)
        )
    missing = sorted(required - set(settings))
    if missing:
        raise ValueError(f"missing selector settings: {', '.join(missing)}")
    values = tuple(settings[name] for name in (
        "input_tokens_gt", "output_tokens_lt", "limit"
    ))
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("selector thresholds and limit must be integers")
    input_gt, output_lt, limit = values
    if input_gt < 0 or output_lt <= 0 or limit <= 0:
        raise ValueError("selector thresholds and limit are invalid")
    return _Settings(input_gt, output_lt, limit)


@register_plugin(
    stage="selector",
    name="token-length",
    description="Select complete Sessions whose every Turn meets token limits",
    settings_parser=parse_settings,
)
class TokenLengthSelector(SelectorPlugin):
    def __init__(self, settings: object) -> None:
        if not isinstance(settings, _Settings):
            raise TypeError("token-length selector requires parsed settings")
        self._settings = settings

    def select(self, table):
        input_gt = self._settings.input_tokens_gt
        output_lt = self._settings.output_tokens_lt
        limit = self._settings.limit
        with metadata_query(table) as db:
            db.execute(
                """
                CREATE TEMP TABLE sessions AS
                SELECT session_id, min(source_order) AS first_order,
                       min(input_ordinal) AS first_seen,
                       bool_and(estimated_input_tokens > ?)
                       AND count(reference_output_tokens) > 0
                       AND max(reference_output_tokens) < ? AS qualified
                FROM metadata GROUP BY session_id
            """,
                [input_gt, output_lt],
            )
            counts = db.execute(
                "SELECT count(*), count(*) FILTER (WHERE qualified) FROM sessions"
            ).fetchone()
            assert counts is not None
            candidates, qualified = counts
            if qualified < limit:
                raise ValueError(
                    f"only {qualified} of {candidates} candidates satisfy "
                    f"every Turn input_tokens>{input_gt} and "
                    f"output_tokens<{output_lt}; required {limit}"
                )
            indices = selection_indices(
                db,
                """
                SELECT metadata.input_ordinal FROM metadata JOIN (
                  SELECT session_id, first_order, first_seen FROM sessions
                  WHERE qualified ORDER BY first_order, first_seen LIMIT ?
                ) chosen USING(session_id)
                ORDER BY first_order, first_seen, turn_order_key
            """,
                [limit],
            )
            return SelectionResult(indices, {"qualified": qualified})


__all__ = ["TokenLengthSelector", "parse_settings"]
