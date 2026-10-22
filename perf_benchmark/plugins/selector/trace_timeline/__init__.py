from perf_benchmark.core.validation import positive_int
from perf_benchmark.pipeline.contracts.selector import SelectorPlugin, SelectionResult, metadata_query, selection_indices
from perf_benchmark.plugins import register_plugin


def parse_settings(settings):
    if set(settings) - {"limit"}:
        raise ValueError("trace-timeline selector only accepts limit")
    if "limit" in settings:
        positive_int(settings["limit"], "limit")
    return dict(settings)


@register_plugin(stage="selector", name="trace-timeline", settings_parser=parse_settings,
                 description="Chronologically select independent trace requests")
class TraceTimelineSelector(SelectorPlugin):
    def __init__(self, settings):
        self.settings = settings

    def select(self, table):
        if table.column("created_at").null_count:
            raise ValueError("trace-timeline requires timestamps for every request")
        with metadata_query(table) as db:
            duplicates = db.execute("SELECT count(*) FROM (SELECT session_id FROM metadata GROUP BY session_id HAVING count(*) != 1)").fetchone()
            assert duplicates is not None
            if duplicates[0]:
                raise ValueError("timeline requires one request per Session")
            limit = self.settings.get("limit", table.num_rows)
            if limit > table.num_rows:
                raise ValueError("trace limit exceeds available requests")
            return SelectionResult(selection_indices(db,
                "SELECT input_ordinal FROM metadata ORDER BY created_at, source_order, input_ordinal LIMIT ?", [limit]))
