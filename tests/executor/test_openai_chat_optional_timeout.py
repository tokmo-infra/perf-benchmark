"""Chat deadline validation and real Python-to-Rust execution."""

import asyncio

from aiohttp import web
from aiohttp.test_utils import TestServer
import pytest

from perf_benchmark.core.models import Session
from perf_benchmark.plugins.executor.openai_chat import validate
from tests.executor.test_openai_chat import _execute, _runtime, _turn

BASE = {"base_url": "http://localhost:8000", "model": "test"}


def test_default_deadline_is_unchanged():
    assert validate(BASE)["timeout_seconds"] == 300.0


@pytest.mark.parametrize("timeout", [0, -1, True, "none", float("nan"), float("inf")])
def test_invalid_deadline_is_rejected(timeout):
    with pytest.raises(ValueError, match="timeout_seconds"):
        validate({**BASE, "timeout_seconds": timeout})


@pytest.mark.parametrize("timeout,status", [(None, "succeeded"), (0.01, "failed")])
async def test_chat_deadline_reaches_native_runtime(trial_context, timeout, status):
    async def handle(request):
        await request.read()
        await asyncio.sleep(0.05)
        return web.json_response(
            {
                "choices": [
                    {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                ]
            }
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handle)
    ctx = trial_context("executor")
    turn = await _turn(ctx, "turn", "input")
    ctx.payloads.seal()
    async with TestServer(app) as server:
        runtime = _runtime(
            {
                **BASE,
                "base_url": str(server.make_url("")).rstrip("/"),
                "stream": False,
                "timeout_seconds": timeout,
            },
            ctx,
        )
        try:
            await runtime.prepare()
            result = await asyncio.wait_for(
                _execute(runtime, Session(id="s", turns=[turn]), turn, ctx), 2
            )
            assert result.status == status
            if status == "failed":
                assert result.error == "TimeoutError"
        finally:
            await runtime.close()
            ctx.payloads.close()
