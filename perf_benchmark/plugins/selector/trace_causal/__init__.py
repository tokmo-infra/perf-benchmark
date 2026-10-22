from perf_benchmark.core.causal import CausalNode, validate_graph
from perf_benchmark.core.validation import positive_int
from perf_benchmark.pipeline.contracts.selector import SelectorPlugin, SelectionResult, metadata_query, selection_indices
from perf_benchmark.plugins import register_plugin


def parse_settings(settings):
    if set(settings) - {"max_graphs"}:
        raise ValueError("trace-causal selector only accepts max_graphs")
    if "max_graphs" in settings:
        positive_int(settings["max_graphs"], "max_graphs")
    return dict(settings)


@register_plugin(stage="selector", name="trace-causal", settings_parser=parse_settings,
                 description="Select complete source graphs and preserve branch order")
class TraceCausalSelector(SelectorPlugin):
    def __init__(self, settings):
        self.settings = settings

    def select(self, table):
        if table.column("causal").null_count:
            raise ValueError("causal selector requires dependency facts for every node")
        validate_graph(CausalNode.decode(value) for value in table.column("causal").to_pylist())
        with metadata_query(table) as db:
            row = db.execute("SELECT count(DISTINCT causal.graph_id) FROM metadata").fetchone()
            assert row is not None
            count = row[0]
            limit = self.settings.get("max_graphs", count)
            if limit > count:
                raise ValueError("max_graphs exceeds available complete graphs")
            return SelectionResult(selection_indices(db, """
                SELECT input_ordinal FROM metadata
                WHERE causal.graph_id IN (
                  SELECT causal.graph_id FROM metadata GROUP BY causal.graph_id
                  ORDER BY min(causal.root_ns), min(input_ordinal) LIMIT ?)
                ORDER BY causal.root_ns, causal.graph_id, causal.branch_id, turn_order_key, input_ordinal
            """, [limit]))
