
from perf_benchmark.core.validation import positive_int
from perf_benchmark.plugins import register_plugin
from perf_benchmark.runtime.http.rust_executor import RustHttpExecutorRuntime
from perf_benchmark.runtime.http.request import HttpRequestPlugin, merge_extra_body, validate_extra_body
from perf_benchmark.runtime.http.snapshot_http import http_settings, snapshot_request, trial_settings


def parameters(value):
    validate_extra_body(value, owned={"query", "documents", "model", "return_documents", "cache_salt", "cache_salt_enabled"})
    if "top_n" in value:
        positive_int(value["top_n"], "top_n")


def validate(settings):
    parsed = http_settings(settings, owned_body={"query", "documents", "model", "return_documents", "cache_salt", "cache_salt_enabled"})
    parameters(parsed["extra_body"])
    return parsed


@register_plugin(stage="executor", name="rerank", description="Validated /v1/rerank rankings over Rust HTTP",
                 settings_parser=validate)
class RerankExecutor(HttpRequestPlugin):
    supports_start_notifications = True

    async def create_runtime(self, trial, ctx):
        return RustHttpExecutorRuntime(self, self.settings_for_trial(trial), ctx.cancellation, 'rerank')

    def settings_for_trial(self, trial):
        return trial_settings(self.settings, trial, parameters)

    def build_request(self, value, settings):
        count = value.turn.profile.input_items if value.turn.profile else None
        positive_int(count, "rerank input_items")
        assert isinstance(count, int)
        options = dict(model=settings["model"], return_documents=False)
        if salt := self.cache_salt(value, settings):
            options["cache_salt"] = salt
        options = merge_extra_body(options, settings["extra_body"])
        top_n = options.get("top_n", count)
        if top_n > count:
            raise ValueError("top_n exceeds document count")
        return snapshot_request(value, settings, path="/v1/rerank", parameters=options, input_kind="rerank",
                                headers=self.request_headers(settings, value),
                                expectations={"count": count, "top_n": top_n})
