from perf_benchmark.pipeline.contracts.normalization import RecordNormalizationPlugin
from perf_benchmark.pipeline.preparation.prepared_trace import prepared_turn
from perf_benchmark.plugins import register_plugin


def parse_settings(value):
    if value:
        raise ValueError("qwen-bailian normalization has no settings")
    return {}


@register_plugin(stage="normalization", name="qwen-bailian", description="Prepared Bailian trace", settings_parser=parse_settings)
class QwenBailianNormalization(RecordNormalizationPlugin):
    def normalize_record(self, record, source_order):
        del source_order
        draft = prepared_turn(record, "qwen-bailian")
        if draft.metadata.get("source_block_size") != 16 or draft.input_kind != "completion":
            raise ValueError("Bailian requires prepared 16-token text blocks")
        yield draft
