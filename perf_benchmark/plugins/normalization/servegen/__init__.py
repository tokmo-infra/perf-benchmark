from perf_benchmark.pipeline.contracts.normalization import RecordNormalizationPlugin
from perf_benchmark.pipeline.preparation.prepared_trace import prepared_turn
from perf_benchmark.plugins import register_plugin


def parse_settings(value):
    if value:
        raise ValueError("servegen normalization has no settings")
    return {}


@register_plugin(stage="normalization", name="servegen", description="Prepared ServeGen workload", settings_parser=parse_settings)
class ServeGenNormalization(RecordNormalizationPlugin):
    def normalize_record(self, record, source_order):
        del source_order
        draft = prepared_turn(record, "servegen")
        if draft.metadata.get("content_kind") not in {"statistically_synthetic", "hashed_conversation"}:
            raise ValueError("ServeGen preparation kind is missing")
        if draft.metadata["content_kind"] == "hashed_conversation" and draft.causal is None:
            raise ValueError("ServeGen conversation row lacks turn dependency")
        yield draft
