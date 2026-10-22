from perf_benchmark.pipeline.contracts.normalization import RecordNormalizationPlugin
from perf_benchmark.pipeline.preparation.prepared_trace import prepared_turn
from perf_benchmark.plugins import register_plugin


def parse_settings(value):
    if value:
        raise ValueError("burstgpt normalization has no settings")
    return {}


@register_plugin(stage="normalization", name="burstgpt", description="Prepared BurstGPT trace", settings_parser=parse_settings)
class BurstGPTNormalization(RecordNormalizationPlugin):
    def normalize_record(self, record, source_order):
        del source_order
        draft = prepared_turn(record, "burstgpt")
        if draft.metadata.get("source_schema") not in {"legacy", "v2"}:
            raise ValueError("BurstGPT prepared schema must identify legacy or v2")
        yield draft
