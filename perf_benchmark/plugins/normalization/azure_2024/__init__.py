from perf_benchmark.pipeline.contracts.normalization import RecordNormalizationPlugin
from perf_benchmark.pipeline.preparation.prepared_trace import prepared_turn
from perf_benchmark.plugins import register_plugin


def parse_settings(value):
    if value:
        raise ValueError("azure-2024 normalization has no settings")
    return {}


@register_plugin(stage="normalization", name="azure-2024", description="Prepared Azure 2024 trace", settings_parser=parse_settings)
class Azure2024Normalization(RecordNormalizationPlugin):
    def normalize_record(self, record, source_order):
        del source_order
        draft = prepared_turn(record, "azure-2024")
        if draft.metadata.get("source_time_basis") != "utc":
            raise ValueError("Azure 2024 timestamps must be UTC")
        yield draft
