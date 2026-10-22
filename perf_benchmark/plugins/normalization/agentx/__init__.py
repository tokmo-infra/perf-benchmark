from perf_benchmark.pipeline.contracts.normalization import RecordNormalizationPlugin
from perf_benchmark.pipeline.preparation.prepared_trace import prepared_turn
from perf_benchmark.plugins import register_plugin


def parse_settings(value):
    if value:
        raise ValueError("agentx normalization has no settings")
    return {}


@register_plugin(stage="normalization", name="agentx", description="Prepared AgentX WEKA trace", settings_parser=parse_settings)
class AgentXNormalization(RecordNormalizationPlugin):
    def normalize_record(self, record, source_order):
        del source_order
        draft = prepared_turn(record, "agentx")
        if draft.metadata.get("source_block_size") != 64 or draft.metadata.get("hash_id_scope") != "local":
            raise ValueError("AgentX requires session-local 64-token block hashes")
        if draft.metadata.get("trace_mode") == "causal" and draft.causal is None:
            raise ValueError("AgentX causal row lacks dependency graph")
        yield draft
