
from copy import deepcopy

from perf_benchmark.core.validation import positive_int, positive_number
from perf_benchmark.plugins import register_plugin
from perf_benchmark.runtime.http.rust_executor import RustHttpExecutorRuntime
from perf_benchmark.runtime.http.request import HttpRequestPlugin, merge_extra_body, validate_extra_body
from perf_benchmark.runtime.http.snapshot_http import http_settings, snapshot_request, trial_settings


def reasoning_probe(settings, effort):
    yield settings['base_url'].rstrip('/') + '/v1/responses', {
        **{key: deepcopy(settings.get('extra_body', {})[key])
           for key in ('thinking', 'enable_thinking', 'chat_template_kwargs')
           if key in settings.get('extra_body', {})},
        'model': settings['model'], 'input': 'Reply OK.', 'store': False,
        'stream': False, 'max_output_tokens': 16, 'reasoning': {'effort': effort},
    }


def parameters(value):
    validate_extra_body(value, owned={"input", "model", "stream", "store", "cache_salt", "cache_salt_enabled"})
    if "max_output_tokens" in value:
        positive_int(value["max_output_tokens"], "max_output_tokens")
    for key in ("temperature", "top_p"):
        if key in value:
            positive_number(value[key], key, allow_zero=True)
    if "parallel_tool_calls" in value and not isinstance(value["parallel_tool_calls"], bool):
        raise ValueError("parallel_tool_calls must be boolean")
    if value.get("tool_choice", "auto") not in ("auto", "none", "required"):
        raise ValueError("tool_choice must be auto, none or required")


def validate(settings):
    parsed = http_settings(settings, stream=True, owned_body={"input", "model", "stream", "store", "cache_salt", "cache_salt_enabled"})
    parameters(parsed["extra_body"])
    return parsed


@register_plugin(stage="executor", name="openai-responses", description="Stateless Responses text and tool events over Rust HTTP",
                 settings_parser=validate)
class ResponsesExecutor(HttpRequestPlugin):
    supports_start_notifications = True

    async def create_runtime(self, trial, ctx):
        return RustHttpExecutorRuntime(self, self.settings_for_trial(trial), ctx.cancellation, 'openai-responses')

    def settings_for_trial(self, trial):
        return trial_settings(self.settings, trial, parameters)

    def build_request(self, value, settings):
        options = dict(model=settings["model"], stream=settings["stream"], store=False)
        if salt := self.cache_salt(value, settings):
            options["cache_salt"] = salt
        options = merge_extra_body(options, settings["extra_body"])
        options.setdefault("temperature", 0.1)
        maximum = value.turn.requested_output_tokens
        positive_int(maximum, "requested_output_tokens", optional=True)
        if maximum is not None:
            options["max_output_tokens"] = maximum
        else:
            options.setdefault("max_output_tokens", 4096)
        return snapshot_request(value, settings, path="/v1/responses", parameters=options, input_kind="responses",
                                headers=self.request_headers(settings, value))
