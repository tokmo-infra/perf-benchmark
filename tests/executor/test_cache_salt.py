"""All HTTP protocols keep salt scope out of dataset payloads and persisted facts."""
import json

import pytest

from perf_benchmark.core.models import Session, Turn
from perf_benchmark.core.profiles import RequestProfile
from perf_benchmark.pipeline.contracts.executor import AttemptInput, ExecutorTrial
from perf_benchmark.pipeline.preparation.payloads import BlobStore
from perf_benchmark.plugins.executor import (
    openai_chat, openai_completions, openai_responses, openai_embeddings, rerank,
)


CASES = (
    (openai_chat, openai_chat.OpenAIChatExecutor, "chat", {"messages": [{"role": "user", "content": "q"}]}),
    (openai_completions, openai_completions.OpenAICompletionsExecutor, "completion", {"prompt": "q"}),
    (openai_responses, openai_responses.ResponsesExecutor, "responses", {"input": "q"}),
    (openai_embeddings, openai_embeddings.EmbeddingsExecutor, "embedding", {"input": ["q"]}),
    (rerank, rerank.RerankExecutor, "rerank", {"query": "q", "documents": ["d"]}),
)


@pytest.mark.parametrize("module,cls,kind,payload", CASES)
async def test_cache_salt_scopes_all_protocols(module, cls, kind, payload):
    options = {"base_url": "http://127.0.0.1:1", "model": "model", "cache_salt_enabled": True}
    if kind == "chat":
        options["max_tokens"] = 2
    plugin = cls(module.validate(options))
    payloads = BlobStore()
    ref = payloads.append(json.dumps(payload).encode())
    payloads.seal()
    profile = RequestProfile(input_items=1)

    async def request(trial_id, phase, cycle, session_id):
        turn = Turn(ref, input_kind=kind, requested_output_tokens=2,
                    profile=profile, id="turn-fixture-36")
        metadata = {"warmup": True} if phase == "warmup" else {"cycle": cycle}
        session = Session(id=session_id, turns=[turn], metadata=metadata)
        session.link()
        settings = plugin.settings_for_trial(ExecutorTrial(trial_id))
        value = AttemptInput.capture(session, turn, {}, attempt=1, payloads=payloads)
        spec = plugin.build_request(value, settings)
        body = await spec.build_body(spec)
        try:
            sent = json.loads(b"".join(body.segments))
        finally:
            body.release()
        assert spec.request_facts.metadata["cache_salt_applied"] is True
        assert spec.request_facts.profile.cache_isolation == "salt_sent"
        return sent

    try:
        warmup = await request("trial", "warmup", 0, "warmup")
        first = await request("trial", "formal", 0, "session-0")
        peer = await request("trial", "formal", 0, "session-1")
        replay = await request("trial", "formal", 1, "session-2")
        other = await request("trial-2", "formal", 0, "session-0")
        assert len({warmup["cache_salt"], first["cache_salt"], replay["cache_salt"], other["cache_salt"]}) == 4
        assert first["cache_salt"] == peer["cache_salt"]
        assert payload == json.loads(bytes(payloads.view(ref)))
    finally:
        payloads.close()


@pytest.mark.parametrize("module,cls,kind,payload", CASES)
@pytest.mark.parametrize("field", ["cache_salt", "cache_salt_enabled"])
def test_cache_salt_cannot_be_supplied_as_request_parameter(module, cls, kind, payload, field):
    settings = {"base_url": "http://127.0.0.1:1", "model": "model",
                "extra_body": {field: "manual"}}
    with pytest.raises((ValueError, TypeError)):
        module.validate(settings)
    clean = {"base_url": "http://127.0.0.1:1", "model": "model"}
    plugin = cls(module.validate(clean))
    with pytest.raises((ValueError, TypeError)):
        plugin.settings_for_trial(ExecutorTrial("trial", {field: "manual"}))
