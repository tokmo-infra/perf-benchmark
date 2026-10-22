from perf_benchmark.pipeline.contracts.normalization import RecordNormalizationPlugin
from perf_benchmark.pipeline.preparation.prepared_trace import prepared_turn
from perf_benchmark.plugins import register_plugin


def parse_settings(value):
    if value:
        raise ValueError("mooncake normalization has no settings")
    return {}


@register_plugin(stage="normalization", name="mooncake", description="Prepared Mooncake trace", settings_parser=parse_settings)
class MooncakeNormalization(RecordNormalizationPlugin):
    def normalize_record(self, record, source_order):
        del source_order
        draft = prepared_turn(record, "mooncake")
        if draft.metadata.get("source_block_size") != 512 or draft.input_kind != "completion":
            raise ValueError("Mooncake requires prepared 512-token text blocks")
        yield draft
