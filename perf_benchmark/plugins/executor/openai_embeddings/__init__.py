
from perf_benchmark.core.validation import positive_int
from perf_benchmark.plugins import register_plugin
from perf_benchmark.runtime.http.rust_executor import RustHttpExecutorRuntime
from perf_benchmark.runtime.http.request import HttpRequestPlugin, merge_extra_body, validate_extra_body
from perf_benchmark.runtime.http.snapshot_http import http_settings, snapshot_request, trial_settings


def parameters(value):
    validate_extra_body(value, owned={"input", "model", "encoding_format", "cache_salt", "cache_salt_enabled"})
    if "dimensions" in value:
        positive_int(value["dimensions"], "dimensions")


def validate(settings):
    parsed = http_settings(settings, owned_body={"input", "model", "encoding_format", "cache_salt", "cache_salt_enabled"})
    parameters(parsed["extra_body"])
    return parsed


@register_plugin(stage="executor", name="openai-embeddings", description="Float embedding vectors over Rust HTTP",
                 settings_parser=validate)
class EmbeddingsExecutor(HttpRequestPlugin):
    supports_start_notifications = True

    async def create_runtime(self, trial, ctx):
        return RustHttpExecutorRuntime(self, self.settings_for_trial(trial), ctx.cancellation, 'openai-embeddings')

    def settings_for_trial(self, trial):
        return trial_settings(self.settings, trial, parameters)

    def build_request(self, value, settings):
        count = value.turn.profile.input_items if value.turn.profile else None
        positive_int(count, "embedding input_items")
        assert isinstance(count, int)
        options = dict(model=settings["model"], encoding_format="float")
        if salt := self.cache_salt(value, settings):
            options["cache_salt"] = salt
        options = merge_extra_body(options, settings["extra_body"])
        return snapshot_request(value, settings, path="/v1/embeddings", parameters=options, input_kind="embedding",
                                headers=self.request_headers(settings, value),
                                expectations={"count": count, "dimensions": options.get("dimensions", 0)})
