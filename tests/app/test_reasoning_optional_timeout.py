"""Capability probes retain a bounded deadline with unlimited measured requests."""

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from perf_benchmark.app import reasoning


@pytest.mark.parametrize("request_timeout,probe_timeout", [(None, 30), (300, 30), (1.5, 1.5)])
async def test_optional_request_timeout_preserves_bounded_reasoning_probe(
    monkeypatch,
    request_timeout,
    probe_timeout,
):
    observed = []
    original = reasoning.aiohttp.ClientSession

    def client(*args, **kwargs):
        observed.append(kwargs["timeout"].total)
        return original(*args, **kwargs)

    monkeypatch.setattr(reasoning.aiohttp, "ClientSession", client)

    async def handle(request):
        body = await request.json()
        assert body["reasoning_effort"] == "max"
        return web.json_response({"choices": []})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handle)
    async with TestServer(app) as server:
        config = {
            "pipeline": {
                "executor": {
                    "plugin": "openai-chat",
                    "settings": {
                        "base_url": str(server.make_url("")).rstrip("/"),
                        "model": "mock",
                        "timeout_seconds": request_timeout,
                    },
                }
            }
        }
        resolved, evidence = await reasoning.resolve(config)
    assert observed == [probe_timeout]
    assert evidence["selected"] == "max"
    assert resolved["pipeline"]["executor"]["settings"]["timeout_seconds"] == request_timeout
