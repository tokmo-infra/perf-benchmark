from __future__ import annotations
from typing import Any

import asyncio
import json
import sqlite3
from threading import enumerate as active_threads
from types import SimpleNamespace

from aiohttp import web
from aiohttp.test_utils import TestServer
import pytest

from perf_benchmark.pipeline.coordinator import Coordinator
from perf_benchmark.app.main import execute_config
from perf_benchmark.app.preparation import PreparationFailed
from perf_benchmark.pipeline.hooks import HookRegistry, HookSpec
from perf_benchmark.plugins.saver.sqlite import SQLiteResultReader
from perf_benchmark.runtime.cancellation import CancellationToken


def _config(
    source,
    normalization,
    normalization_settings,
    base_url,
    database,
) -> dict[str, Any]:
    multi_turn = bool(normalization_settings.get("multi_turn"))
    scheduler = {
        "plugin": "multi-turn-closed-loop" if multi_turn else "single-turn-closed-loop",
        "settings": {
            **({"max_active_sessions": 1} if multi_turn else {}),
            "max_concurrency": 1,
            "stop": {"session_count": 1},
            "on_exhausted": "error",
        },
    }
    return {
        "case": f"End-to-end {normalization} normalization",
        "name": f"e2e-{normalization}",
        "logging": {"level": "WARNING", "stream": "stderr"},
        "pipeline": {
            "loader": {
                "plugin": "hf-datasets",
                "settings": {
                    "dataset": "json",
                    "data_files": [str(source)],
                    "split": "train",
                    "batch_rows": 2,
                },
            },
            "normalization": {
                "plugin": normalization,
                "settings": normalization_settings,
            },
            "selector": {
                "plugin": "token-length",
                "settings": {
                    "input_tokens_gt": 0,
                    "output_tokens_lt": 1000,
                    "limit": 1,
                },
            },
            "executor": {
                "plugin": "openai-chat",
                "settings": {
                    "base_url": base_url,
                    "model": "mock",
                    "max_tokens": 3, "extra_body": {"reasoning_effort": None},
                },
            },
            "scheduler": scheduler,
            "metrics": {
                "plugin": "standard",
                "settings": {"memory_limit": "64MB"},
            },
            "saver": {"plugin": "sqlite", "settings": {"path": str(database)}},
            "exporter": {},
        },
    }


async def _mock_server(requests):
    async def complete(request):
        requests.append(await request.json())
        answer = f"measured-{len(requests)}"
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(
            f'data: {{"choices":[{{"delta":{{"content":"{answer}"}}}}]}}\n\n'.encode()
        )
        await response.write(
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
            b'"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}\n\n'
        )
        await response.write(b"data: [DONE]\n\n")
        return response

    async def models(request):
        return web.json_response({"data": [{"id": "mock"}]})

    app = web.Application()
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/chat/completions", complete)
    server = TestServer(app)
    await server.start_server()
    return server


def _write(path, record):
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")


async def test_optional_prices_and_maxtpm_window_are_persisted_without_changing_requests(tmp_path):
    requests = []
    source, database = tmp_path / "prices.jsonl", tmp_path / "prices.db"
    _write(source, {
        "id": "request", "token_name": "session", "created": "2026-10-09T00:00:00Z",
        "messages": [{"role": "user", "content": "same prompt"}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 2},
    })
    server = await _mock_server(requests)
    try:
        config = _config(source, "openai-messages", {}, str(server.make_url("")), database)
        config["pipeline"]["scheduler"]["settings"].update(
            stop={"session_count": 3}, on_exhausted="cycle"
        )
        config["pipeline"]["metrics"]["settings"].update(
            maxtpm=True, pricing={"currency": "CNY", "per_tokens": 1000,
                                  "input_price": 2, "cached_input_price": 2, "output_price": 8}
        )
        run = await Coordinator().execute(config, base_dir=tmp_path)
    finally:
        await server.close()

    def read_results():
        reader = SQLiteResultReader(database)
        (trial,) = reader.trials(run.id)
        kwargs = dict(scope="trial", scope_id=trial["trial_id"])
        return (trial, reader.metric_points(run.id, window="overall", **kwargs),
                reader.metric_points(run.id, window="maxtpm", **kwargs))

    trial, costs, throughput = await asyncio.to_thread(read_results)
    assert run.status == "succeeded"
    assert trial["scheduling"]["dataset_session_count"] == 1
    assert trial["scheduling"]["dataset_turn_count"] == 1
    assert trial["session_count"] == len(requests) == 3
    assert costs["total_cost"]["statistics"]["value"] == pytest.approx(3 * (5 * 2 + 2 * 8) / 1000)
    assert costs["total_cost"]["unit"] == "CNY"
    assert costs["cost_observed_attempts"]["statistics"]["value"] == 3
    assert throughput["maxtpm"]["availability"] == "unavailable"
    assert throughput["maxtpm"]["unavailable_reason"] == "no_complete_minute_in_maxtpm_window"
    assert throughput["maxtpm_minutes"]["statistics"]["value"] == 0
    assert all("pricing" not in body and "maxtpm" not in body for body in requests)


async def test_cycle_replays_receive_distinct_tool_generated_cache_salts(tmp_path):
    requests = []
    source, database = tmp_path / "cycle.jsonl", tmp_path / "cycle.db"
    _write(
        source,
        {
            "id": "request",
            "token_name": "session",
            "created": "2026-09-04T00:00:00Z",
            "messages": [{"role": "user", "content": "same prompt"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2},
        },
    )
    server = await _mock_server(requests)
    try:
        config = _config(
            source,
            "openai-messages",
            {},
            str(server.make_url("")).rstrip("/"),
            database,
        )
        config["pipeline"]["scheduler"]["settings"].update(
            stop={"session_count": 3}, on_exhausted="cycle"
        )
        config["pipeline"]["executor"]["settings"]["cache_salt_enabled"] = True
        run = await Coordinator().execute(config, base_dir=tmp_path)
    finally:
        await server.close()

    assert run.status == "succeeded"
    assert len(requests) == 3
    replay_salts = [request["cache_salt"] for request in requests]
    assert len(set(replay_salts)) == 3
    assert all(
        request["messages"] == [{"role": "user", "content": "same prompt"}] for request in requests
    )
    assert all(salt.encode() not in database.read_bytes() for salt in replay_salts)

    def turn_metadata():
        with sqlite3.connect(database) as connection:
            return [
                json.loads(row[0])
                for row in connection.execute(
                    "SELECT metadata_json FROM turns ORDER BY rowid"
                ).fetchall()
            ]

    metadata = await asyncio.to_thread(turn_metadata)
    assert [item["cache_salt_applied"] for item in metadata] == [True, True, True]


async def test_selected_single_turn_warmup_is_separate_from_formal_cycles(tmp_path):
    requests = []
    source, database = tmp_path / "warmup.jsonl", tmp_path / "warmup.db"
    _write(source, {"id": "request", "token_name": "session", "created": "2026-09-04T00:00:00Z",
                    "messages": [{"role": "user", "content": "same prompt"}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 2}})
    server = await _mock_server(requests)
    try:
        config = _config(source, "openai-messages", {}, str(server.make_url("")).rstrip("/"), database)
        config["pipeline"]["executor"]["settings"]["cache_salt_enabled"] = True
        config["pipeline"]["scheduler"]["settings"].update(stop={"session_count": 2}, on_exhausted="cycle")
        config["application"] = {"preparation": {"adapter": "vllm", "exclusive": True, "warmup_requests": 1}}
        run = await execute_config(config, tmp_path)
    finally:
        await server.close()
    assert run.status == "succeeded"
    assert len(requests) == 3
    assert len({request["cache_salt"] for request in requests}) == 3
    assert all(request["cache_salt"].encode() not in database.read_bytes() for request in requests)
    reader = SQLiteResultReader(database)
    assert len(tuple(reader.iter_turn_records(run.id))) == 2
    fact = reader.trials(run.id)[0]["metadata"]["preparation"]
    assert (fact["planned_requests"], fact["succeeded_requests"]) == (1, 1)


async def test_warmup_fails_when_selected_turns_are_insufficient(tmp_path):
    requests = []
    source, database = tmp_path / "one.jsonl", tmp_path / "one.db"
    _write(source, {"id": "request", "token_name": "session", "created": "2026-09-04T00:00:00Z",
                    "messages": [{"role": "user", "content": "only one"}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 2}})
    server = await _mock_server(requests)
    try:
        config = _config(source, "openai-messages", {}, str(server.make_url("")).rstrip("/"), database)
        config["pipeline"]["executor"]["settings"]["cache_salt_enabled"] = True
        config["application"] = {"preparation": {"adapter": "vllm", "exclusive": True, "warmup_requests": 2}}
        with pytest.raises(PreparationFailed, match="fewer Turns"):
            await execute_config(config, tmp_path)
    finally:
        await server.close()
    assert requests == []
    reader = SQLiteResultReader(database)
    run_id, = reader.run_ids()
    assert tuple(reader.iter_turn_records(run_id)) == ()
    assert reader.trials(run_id)[0]["metadata"]["preparation"]["succeeded_requests"] == 0


async def test_warmup_rejects_service_that_rejects_cache_salt(tmp_path):
    source, database = tmp_path / "one.jsonl", tmp_path / "unsupported.db"
    _write(source, {"id": "request", "token_name": "session", "created": "2026-09-04T00:00:00Z",
                    "messages": [{"role": "user", "content": "prompt"}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 2}})
    async def models(request):
        return web.json_response({"data": [{"id": "mock"}]})
    async def unsupported(request):
        body = await request.json()
        assert "cache_salt" in body
        return web.json_response({"error": "unsupported cache_salt"}, status=400)
    app = web.Application()
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/chat/completions", unsupported)
    async with TestServer(app) as server:
        config = _config(source, "openai-messages", {}, str(server.make_url("")).rstrip("/"), database)
        config["pipeline"]["executor"]["settings"]["cache_salt_enabled"] = True
        config["application"] = {"preparation": {"adapter": "vllm", "exclusive": True, "warmup_requests": 1}}
        with pytest.raises(PreparationFailed, match="HTTP 400"):
            await execute_config(config, tmp_path)
    reader = SQLiteResultReader(database)
    run_id, = reader.run_ids()
    assert tuple(reader.iter_turn_records(run_id)) == ()
    assert reader.trials(run_id)[0]["metadata"]["preparation"]["cache_state"] == "salt_attempted_backend_unverified"


@pytest.mark.parametrize("mode", ["recorded", "live"])
async def test_multiturn_warmup_stops_mid_session_and_replays_formal(tmp_path, tokenizer_path, mode):
    requests = []
    source, database = tmp_path / "conversation.jsonl", tmp_path / "conversation.db"
    _write(source, {"id": "conversation", "conversations": [
        {"from": "human", "value": "first"}, {"from": "gpt", "value": "recorded-first"},
        {"from": "human", "value": "second"}, {"from": "gpt", "value": "recorded-second"},
        {"from": "human", "value": "third"}, {"from": "gpt", "value": "recorded-third"},
    ]})
    server = await _mock_server(requests)
    try:
        config = _config(source, "sharegpt", {"tokenizer_path": str(tokenizer_path),
                         "multi_turn": True, "live_continuation": mode == "live"},
                         str(server.make_url("")).rstrip("/"), database)
        config["pipeline"]["executor"]["settings"]["cache_salt_enabled"] = True
        config["application"] = {"preparation": {"adapter": "vllm", "exclusive": True, "warmup_requests": 2}}
        run = await execute_config(config, tmp_path)
    finally:
        await server.close()
    assert run.status == "succeeded"
    assert len(requests) == 5
    assert requests[0]["cache_salt"] == requests[1]["cache_salt"]
    assert len({request["cache_salt"] for request in requests[2:]}) == 1
    assert requests[0]["cache_salt"] != requests[2]["cache_salt"]
    assert requests[1]["messages"][1]["content"] == ("recorded-first" if mode == "recorded" else "measured-1")
    assert requests[3]["messages"][1]["content"] == ("recorded-first" if mode == "recorded" else "measured-3")
    reader = SQLiteResultReader(database)
    assert len(tuple(reader.iter_turn_records(run.id))) == 3


@pytest.mark.parametrize(
    "tail, expected",
    [
        (b"data: [DONE]\r\n\r\n", "succeeded"),
        (b"", "failed"),
        (b"data: [DONE]\r\n", "failed"),
        (b'data: {"private-response-body": bad}\r\n\r\n', "failed"),
        (b'data: {"error":{"message":"private-response-body"}}\r\n\r\n', "failed"),
    ],
)
async def test_stream_completion_and_safe_partial_results_are_persisted(
    tmp_path,
    caplog,
    capsys,
    tail,
    expected,
):
    release = asyncio.Event()

    async def complete(request):
        await request.read()
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(
            b'\xef\xbb\xbf: ping\r\nevent: message\r\ndata: {"choices":\r\n'
            b'data: [{"delta":{"content":"private-response-body"}}]}\r\n\r\n'
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
            b'"usage":{"prompt_tokens":1,"completion_tokens":2}}\r\n\r\n' + tail
        )
        if expected == "succeeded":
            # Completion must not wait for HTTP EOF after a full DONE event.
            await release.wait()
        return response

    source, database = tmp_path / "stream.jsonl", tmp_path / "stream.db"
    _write(
        source,
        {
            "id": "request",
            "token_name": "session",
            "created": "2026-09-03T00:00:00Z",
            "messages": [{"role": "user", "content": "private-request-body"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        },
    )
    app = web.Application()
    app.router.add_post("/v1/chat/completions", complete)
    async with TestServer(app) as server:
        config = _config(source, "openai-messages", {}, str(server.make_url("")), database)
        try:
            run = await asyncio.wait_for(Coordinator().execute(config, base_dir=tmp_path), 30)
        finally:
            release.set()

    def read_results():
        reader = SQLiteResultReader(database)
        summary = reader.summary(run.id)
        (trial,) = reader.trials(run.id)
        points = reader.metric_points(
            run.id,
            scope="trial",
            scope_id=trial["trial_id"],
            window="overall",
            names={"succeeded", "failed"},
        )
        return summary, trial, points

    summary, trial, points = await asyncio.to_thread(read_results)
    assert run.status == summary["status"] == trial["status"] == expected
    assert summary["turn_count"] == summary["session_count"] == 1
    assert points[expected]["statistics"]["value"] == 1
    assert run.event_counts["request.first_output"] == 1
    output = capsys.readouterr()
    for secret in ("private-request-body", "private-response-body"):
        assert secret.encode() not in database.read_bytes()
        assert secret not in str(run.diagnostics) + caplog.text + output.out + output.err


@pytest.mark.parametrize("stop", ["disconnect", "cancel"])
async def test_interrupted_http_stream_keeps_partial_results_and_closes_resources(
    tmp_path,
    monkeypatch,
    caplog,
    capsys,
    stop,
):
    finished = asyncio.Event()
    cancellation = CancellationToken()

    async def complete(request):
        await request.read()
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(
            b'data: {"choices":[{"delta":{"content":"private-partial-body"}}],'
            b'"usage":{"completion_tokens":2}}\n\n'
            b'data: {"test_barrier":true}\n\n'
        )
        # A real Rust reader consumes the prefix while this server keeps the body open.
        await asyncio.sleep(0.1)
        if stop == "cancel":
            cancellation.cancel("user cancellation")
        else:
            request.transport.close()
        await finished.wait()
        return response

    source, database = tmp_path / "interrupted.jsonl", tmp_path / "interrupted.db"
    _write(
        source,
        {
            "id": "request",
            "token_name": "session",
            "created": "2026-09-03T00:00:00Z",
            "messages": [{"role": "user", "content": "question"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        },
    )
    app = web.Application()
    app.router.add_post("/v1/chat/completions", complete)
    async with TestServer(app) as server:
        config = _config(source, "openai-messages", {}, str(server.make_url("")), database)
        try:
            execution = Coordinator().execute(
                config,
                base_dir=tmp_path,
                cancellation=cancellation,
            )
            run = await asyncio.wait_for(execution, 30)
            assert run.status == ("cancelled" if stop == "cancel" else "failed")
        finally:
            finished.set()

    def read_results():
        with sqlite3.connect(database) as connection:
            ((run_id,),) = connection.execute("SELECT run_id FROM runs")
        reader = SQLiteResultReader(database)
        (record,) = reader.iter_turn_records(run_id)
        row = record["status"], record["usage"], record["measurements"]
        (trial,) = reader.trials(run_id)
        return reader.summary(run_id), trial, row

    summary, trial, (status, usage, measurements) = await asyncio.to_thread(read_results)
    expected = "cancelled" if stop == "cancel" else "failed"
    assert summary["status"] == trial["status"] == status == expected
    assert summary["turn_count"] == summary["session_count"] == 1
    assert usage["completion_tokens"] == 2
    measured = measurements["request"]
    assert measured["output_event_count"] == 1
    assert measured["first_output_ns"] == measured["last_output_ns"]
    assert measured["first_output_ns"] is not None
    assert b"private-partial-body" not in database.read_bytes()
    output = capsys.readouterr()
    assert "private-partial-body" not in caplog.text + output.out + output.err
    assert not any(thread.name.startswith("perf-stream-decoder") for thread in active_threads())


async def test_gsm8k_mock_openai_e2e(tmp_path, tokenizer_path) -> None:
    source = tmp_path / "gsm8k.jsonl"
    _write(source, {"question": "What is two plus two?", "answer": "4"})
    requests = []
    server = await _mock_server(requests)
    try:
        run = await Coordinator().execute(
            _config(
                source,
                "gsm8k",
                {"tokenizer_path": str(tokenizer_path)},
                str(server.make_url("")).rstrip("/"),
                tmp_path / "gsm8k.db",
            ),
            base_dir=tmp_path,
        )
    finally:
        await server.close()
    assert run.status == "succeeded"
    assert "\\boxed{}" in requests[0]["messages"][0]["content"]


async def test_sharegpt_recorded_and_live_mock_openai_e2e(
    tmp_path,
    tokenizer_path,
) -> None:
    source = tmp_path / "sharegpt.jsonl"
    _write(
        source,
        {
            "id": "conversation",
            "conversations": [
                {"from": "human", "value": "first"},
                {"from": "gpt", "value": "recorded-first", "requested_output_tokens": 271},
                {"from": "human", "value": "second"},
                {"from": "gpt", "value": "recorded-second", "requested_output_tokens": 329},
            ],
        },
    )
    recorded_requests = []
    server = await _mock_server(recorded_requests)
    try:
        recorded = await Coordinator().execute(
            _config(
                source,
                "sharegpt",
                {"tokenizer_path": str(tokenizer_path), "multi_turn": True},
                str(server.make_url("")).rstrip("/"),
                tmp_path / "sharegpt-recorded.db",
            ),
            base_dir=tmp_path,
        )
    finally:
        await server.close()
    assert recorded.status == "succeeded"
    assert [r["max_tokens"] for r in recorded_requests] == [271, 329]
    assert recorded_requests[1]["messages"][1]["content"] == "recorded-first"

    live_requests = []
    server = await _mock_server(live_requests)
    try:
        live = await Coordinator().execute(
            _config(
                source,
                "sharegpt",
                {
                    "tokenizer_path": str(tokenizer_path),
                    "multi_turn": True,
                    "live_continuation": True,
                },
                str(server.make_url("")).rstrip("/"),
                tmp_path / "sharegpt-live.db",
            ),
            base_dir=tmp_path,
        )
    finally:
        await server.close()
    assert live.status == "succeeded"
    assert [r["max_tokens"] for r in live_requests] == [271, 329]
    assert live_requests[1]["messages"][1]["content"] == "measured-1"


async def test_openai_messages_snapshot_mock_openai_e2e(tmp_path) -> None:
    source = tmp_path / "messages.jsonl"
    _write(
        source,
        {
            "id": "request",
            "token_name": "source-session",
            "created": "2026-08-31T12:00:00Z",
            "messages": [
                {"role": "system", "content": "system"},
                {
                    "role": "user",
                    "content": "use the tool",
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "lookup", "parameters": {"type": "object"}},
                }
            ],
            "tool_choice": "auto",
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        },
    )
    requests = []
    server = await _mock_server(requests)
    try:
        run = await Coordinator().execute(
            _config(
                source,
                "openai-messages",
                {},
                str(server.make_url("")).rstrip("/"),
                tmp_path / "messages.db",
            ),
            base_dir=tmp_path,
        )
    finally:
        await server.close()
    assert run.status == "succeeded"
    assert requests[0]["messages"][0]["role"] == "system"
    assert requests[0]["tools"][0]["function"]["name"] == "lookup"
    assert requests[0]["tool_choice"] == "auto"


async def test_http_error_body_and_credentials_never_reach_results_or_hooks(
    tmp_path, monkeypatch
):
    secret = "HTTP_ERROR_BODY_MUST_NOT_ESCAPE"
    observed = []
    authorization = []

    class Observer:
        def __init__(self, settings):
            pass

        async def handle(self, event, payload, ctx):
            observed.append(str(payload))

    async def complete(request):
        authorization.append(request.headers.get("Authorization"))
        return web.Response(status=401, text=secret)

    source = tmp_path / "error.jsonl"
    _write(
        source,
        {
            "id": "request",
            "token_name": "session",
            "created": "2026-08-31T12:00:00Z",
            "messages": [{"role": "user", "content": "hello"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    )
    app = web.Application()
    app.router.add_post("/v1/chat/completions", complete)
    server = TestServer(app)
    await server.start_server()
    database = tmp_path / "error.db"
    config = _config(source, "openai-messages", {}, str(server.make_url("")).rstrip("/"), database)
    monkeypatch.setenv("PERF_TEST_API_KEY", "PRIVATE_AUTH_TOKEN")
    config["pipeline"]["executor"]["settings"]["api_keys"] = [{"env": "PERF_TEST_API_KEY"}]
    config["hooks"] = {"events": {"request.failed": [{"hook": "observer"}]}}
    registry = HookRegistry()
    registry.register(HookSpec("observer", Observer, "observe safe failure"))
    try:
        run = await Coordinator(hook_registry=registry).execute(config, base_dir=tmp_path)
    finally:
        await server.close()
    assert run.status == "failed"
    assert authorization == ["Bearer PRIVATE_AUTH_TOKEN"]
    assert secret not in str(observed)
    assert "PRIVATE_AUTH_TOKEN" not in str(observed)

    def errors():
        reader = SQLiteResultReader(database)
        row = next(reader.iter_turn_records(run.id))
        return row["error"], row["measurements"]

    error, measurements = await asyncio.to_thread(errors)
    assert "401" in error and secret not in error
    assert measurements["request"]["http_status"] == 401
    assert secret.encode() not in database.read_bytes()
    assert b"PRIVATE_AUTH_TOKEN" not in database.read_bytes()


@pytest.mark.parametrize("sampling", ["uniform", "weighted"])
async def test_api_key_sampling_reaches_real_http_requests_and_retries(
    tmp_path, monkeypatch, sampling,
):
    from perf_benchmark.runtime.http import request as http_request

    secrets = ("GATEWAY_PRIVATE_TOKEN_A", "GATEWAY_PRIVATE_TOKEN_B")
    for name, secret in zip(("PERF_GATEWAY_KEY_A", "PERF_GATEWAY_KEY_B"), secrets):
        monkeypatch.setenv(name, secret)
    selections = iter([0, 1, 0, 1])
    if sampling == "uniform":
        sampler = SimpleNamespace(choice=lambda keys: keys[next(selections)])
    else:
        sampler = SimpleNamespace(choices=lambda keys, **kwargs: [keys[next(selections)]])
    monkeypatch.setattr(http_request, "random", sampler)
    authorization, observed = [], []

    class Observer:
        def __init__(self, settings):
            pass

        async def handle(self, event, payload, ctx):
            observed.append((event.to_dict(), str(payload)))

    async def complete(request):
        await request.read()
        authorization.append(request.headers.get("Authorization"))
        if len(authorization) == 1:
            return web.json_response({"error": "temporary gateway failure"}, status=503)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b'data: {"choices":[{"delta":{"content":"answer"}}]}\n\n')
        await response.write(
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
            b'"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}\n\n'
        )
        await response.write(b"data: [DONE]\n\n")
        return response

    source, database = tmp_path / "gateway.jsonl", tmp_path / "gateway.db"
    _write(source, {"id": "request", "token_name": "session",
                    "created": "2026-10-14T00:00:00Z",
                    "messages": [{"role": "user", "content": "hello"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    app = web.Application()
    app.router.add_post("/v1/chat/completions", complete)
    registry = HookRegistry()
    registry.register(HookSpec("observer", Observer, "observe credential-free facts"))
    async with TestServer(app) as server:
        config = _config(source, "openai-messages", {}, str(server.make_url("")), database)
        executor = config["pipeline"]["executor"]["settings"]
        executor["api_key_sampling"] = sampling
        executor["api_keys"] = [{"env": "PERF_GATEWAY_KEY_A"}, {"env": "PERF_GATEWAY_KEY_B"}]
        if sampling == "weighted":
            executor["api_keys"][0]["weight"] = 1
            executor["api_keys"][1]["weight"] = 3
        config["pipeline"]["scheduler"]["settings"].update(
            stop={"session_count": 3}, on_exhausted="cycle",
            retry={"max_attempts": 2, "delay_s": 0},
        )
        config["hooks"] = {"events": {
            event: [{"hook": "observer"}]
            for event in ("request.started", "request.completed", "request.failed", "run.completed")
        }}
        run = await Coordinator(hook_registry=registry).execute(config, base_dir=tmp_path)

    assert run.status == "succeeded"
    assert authorization == [f"Bearer {secrets[index]}" for index in (0, 1, 0, 1)]
    assert observed

    def results():
        reader = SQLiteResultReader(database)
        with sqlite3.connect(database) as connection:
            attempts = connection.execute("SELECT status FROM attempts ORDER BY rowid").fetchall()
        return reader.summary(run.id), tuple(reader.iter_turn_records(run.id)), attempts

    summary, turns, attempts = await asyncio.to_thread(results)
    assert summary["turn_count"] == 3
    assert len(turns) == 3 and all(turn["status"] == "succeeded" for turn in turns)
    assert attempts == [("failed",), ("succeeded",), ("succeeded",), ("succeeded",)]
    for secret in secrets:
        assert secret not in str(observed)
        assert secret not in str((summary, turns, run))
        assert secret not in str(config)
        assert secret.encode() not in database.read_bytes()


async def test_dataset_headers_survive_retries_multiturn_and_cycle_without_leaking(
    tmp_path, monkeypatch,
):
    from perf_benchmark.runtime.http import request as http_request

    credentials = ("PRIVATE_SAMPLED_KEY_A", "PRIVATE_SAMPLED_KEY_B")
    for name, secret in zip(("PERF_ROW_KEY_A", "PERF_ROW_KEY_B"), credentials):
        monkeypatch.setenv(name, secret)
    selections = iter([0, 1, 0, 1, 0, 1, 0])
    monkeypatch.setattr(http_request, "random", SimpleNamespace(choice=lambda keys: keys[next(selections)]))
    requests, observed = [], []

    class Observer:
        def __init__(self, settings):
            pass

        async def handle(self, event, payload, ctx):
            observed.append((event.to_dict(), str(payload)))

    async def complete(request):
        requests.append((request.headers.copy(), await request.json()))
        if len(requests) == 1:
            return web.json_response({"error": "temporary failure"}, status=503)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b'data: {"choices":[{"delta":{"content":"answer"}}]}\n\n')
        await response.write(
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
            b'"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}\n\n'
        )
        await response.write(b"data: [DONE]\n\n")
        return response

    source, database = tmp_path / "row-headers.jsonl", tmp_path / "row-headers.db"
    records = [
        {"id": f"request-{index}", "token_name": "conversation", "created": index,
         "messages": [{"role": "user", "content": f"prompt-{index}"}],
         "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        for index in range(3)
    ]
    records[0]["headers"] = {"X-Tenant": "PRIVATE_TENANT_A", "X-Optional": "PRIVATE_FIRST_ONLY"}
    records[1]["headers"] = {"X-Tenant": "PRIVATE_TENANT_B", "X-Optional": None}
    source.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    app = web.Application()
    app.router.add_post("/v1/chat/completions", complete)
    registry = HookRegistry()
    registry.register(HookSpec("observer", Observer, "observe safe request facts"))
    async with TestServer(app) as server:
        config = _config(source, "openai-messages", {"multi_turn": True},
                         str(server.make_url("")), database)
        config["pipeline"]["executor"]["settings"].update(
            extra_headers={"X-Global": "global"},
            api_keys=[{"env": "PERF_ROW_KEY_A"}, {"env": "PERF_ROW_KEY_B"}],
        )
        config["pipeline"]["scheduler"]["settings"].update(
            stop={"session_count": 2}, on_exhausted="cycle",
            retry={"max_attempts": 2, "delay_s": 0},
        )
        config["hooks"] = {"events": {
            event: [{"hook": "observer"}]
            for event in ("request.started", "request.failed", "request.completed", "run.completed")
        }}
        run = await asyncio.wait_for(Coordinator(hook_registry=registry).execute(config, base_dir=tmp_path), 30)

    assert run.status == "succeeded"
    assert len(requests) == 7
    assert [headers.get("X-Tenant") for headers, _ in requests] == [
        "PRIVATE_TENANT_A", "PRIVATE_TENANT_A", "PRIVATE_TENANT_B", None,
        "PRIVATE_TENANT_A", "PRIVATE_TENANT_B", None,
    ]
    assert [headers.get("X-Optional") for headers, _ in requests] == [
        "PRIVATE_FIRST_ONLY", "PRIVATE_FIRST_ONLY", None, None, "PRIVATE_FIRST_ONLY", None, None,
    ]
    assert [headers.get("Authorization") for headers, _ in requests] == [
        f"Bearer {credentials[index]}" for index in (0, 1, 0, 1, 0, 1, 0)
    ]
    assert all(headers["X-Global"] == "global" for headers, _ in requests)
    assert all("headers" not in body and "extra_headers" not in body for _, body in requests)

    def results():
        reader = SQLiteResultReader(database)
        return reader.summary(run.id), tuple(reader.iter_turn_records(run.id)), reader.trials(run.id)

    summary, turns, trials = await asyncio.to_thread(results)
    assert summary["turn_count"] == len(turns) == 6
    assert all(turn["status"] == "succeeded" for turn in turns)
    assert observed
    for secret in (*credentials, "PRIVATE_TENANT_A", "PRIVATE_TENANT_B", "PRIVATE_FIRST_ONLY"):
        assert secret not in str(observed)
        assert secret not in str((summary, turns, trials, run))
        assert secret not in str(config)
        assert secret.encode() not in database.read_bytes()
        assert all(secret not in str(body) for _, body in requests)


@pytest.mark.parametrize("cancel", [False, True])
async def test_running_pipeline_commits_results_and_cancellation_keeps_only_admitted_work(
    tmp_path, cancel
):
    blocked, release = asyncio.Event(), asyncio.Event()
    requests = 0

    async def complete(request):
        nonlocal requests
        requests += 1
        if requests == 2:
            blocked.set()
            await release.wait()
        return web.json_response(
            {
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        )

    class SlowBrokenHook:
        def __init__(self, settings):
            pass

        async def handle(self, event, payload, ctx):
            await asyncio.sleep(0.02)
            raise RuntimeError("observer failed")

    source = tmp_path / "ongoing.jsonl"
    source.write_text(
        "".join(
            json.dumps(
                {
                    "id": f"request-{index}",
                    "token_name": f"session-{index}",
                    "created": "2026-08-31T12:00:00Z",
                    "messages": [{"role": "user", "content": "hello"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            )
            + "\n"
            for index in range(10)
        )
    )
    app = web.Application()
    app.router.add_post("/v1/chat/completions", complete)
    server = TestServer(app)
    await server.start_server()
    database = tmp_path / "ongoing.db"
    config = _config(source, "openai-messages", {}, str(server.make_url("")).rstrip("/"), database)
    config["pipeline"]["selector"]["settings"]["limit"] = 10
    config["pipeline"]["scheduler"]["settings"]["stop"] = {"session_count": 10}
    config["pipeline"]["executor"]["settings"]["stream"] = False
    config["hooks"] = {"queue_capacity": 1, "events": {"turn.completed": [{"hook": "broken"}]}}
    hooks = HookRegistry()
    hooks.register(HookSpec("broken", SlowBrokenHook, "slow observer"))
    cancellation = CancellationToken()

    def committed():
        reader = SQLiteResultReader(database)
        return tuple(reader.iter_turn_records(reader.run_ids()[0]))

    task = asyncio.create_task(
        Coordinator(hook_registry=hooks).execute(
            config,
            base_dir=tmp_path,
            cancellation=cancellation,
        )
    )
    try:
        await asyncio.wait_for(blocked.wait(), 5)

        async def wait_for_first_result():
            while not (rows := await asyncio.to_thread(committed)):
                await asyncio.sleep(0.01)
            return rows

        rows = await asyncio.wait_for(wait_for_first_result(), 5)
        assert not task.done()
        assert rows[0]["status"] == "succeeded"
        assert rows[0]["metrics"]["e2e"]["availability"] == "available"
        if cancel:
            cancellation.cancel("user cancellation")
        else:
            release.set()
        run = await asyncio.wait_for(task, 10)
        rows = await asyncio.to_thread(committed)
        assert run.status == ("cancelled" if cancel else "succeeded")
        assert 1 <= len(rows) < 10 if cancel else len(rows) == 10
        assert run.handler_failure_count > 0
        assert run.counts["turns"] == len(rows)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await server.close()


async def test_request_preparation_error_is_a_tool_failure_without_fake_attempt(tmp_path):
    from perf_benchmark.plugins import create_registry
    from perf_benchmark.plugins.executor.openai_chat import OpenAIChatExecutor

    class Broken(OpenAIChatExecutor):
        def build_request(self, value, settings):
            raise RuntimeError("request preparation bug")

    registry = create_registry()
    registry.register(
        stage="executor",
        name="broken",
        description="fail in preparation",
        settings_parser=registry.resolve("executor", "openai-chat").settings_parser,
    )(Broken)
    source, database = tmp_path / "preparation.jsonl", tmp_path / "preparation.db"
    _write(
        source,
        {
            "id": "request",
            "token_name": "session",
            "created": "2026-09-03T00:00:00Z",
            "messages": [{"role": "user", "content": "question"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        },
    )
    config = _config(source, "openai-messages", {}, "http://127.0.0.1:1", database)
    config["pipeline"]["executor"]["plugin"] = "broken"
    with pytest.raises(RuntimeError, match="request preparation bug"):
        await Coordinator(registry).execute(config, base_dir=tmp_path)
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT status FROM runs").fetchone() == ("failed",)
        assert connection.execute("SELECT count(*) FROM attempts").fetchone() == (0,)
        assert connection.execute("SELECT count(*) FROM turns").fetchone() == (0,)


async def test_native_execution_failure_fails_run_and_keeps_committed_turn(tmp_path):
    from perf_benchmark.plugins import create_registry
    from perf_benchmark.plugins.executor.openai_chat import OpenAIChatExecutor

    runtimes = []
    native_failure = asyncio.Event()

    class Captured(OpenAIChatExecutor):
        async def create_runtime(self, trial, ctx):
            runtime = await super().create_runtime(trial, ctx)
            execute = runtime.execute_attempt
            async def injected_failure(prepared, signal):
                result = await execute(prepared, signal)
                if native_failure.is_set():
                    ctx.cancellation.cancel("native task failure")
                    raise RuntimeError("native task failure")
                return result
            runtime.execute_attempt = injected_failure
            runtimes.append(runtime)
            return runtime

    registry = create_registry()
    registry.register(
        stage="executor", name="captured", description="capture native execution",
        settings_parser=registry.resolve("executor", "openai-chat").settings_parser,
    )(Captured)
    source, database = tmp_path / "native.jsonl", tmp_path / "native.db"
    source.write_text("".join(json.dumps({
        "id": f"request-{index}", "token_name": f"session-{index}",
        "created": "2026-09-03T00:00:00Z", "messages": [{"role": "user", "content": "question"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2},
    }) + "\n" for index in range(2)))
    calls = 0
    finished = asyncio.Event()

    def committed():
        with sqlite3.connect(database) as connection:
            return connection.execute("SELECT count(*) FROM turns WHERE status='succeeded'").fetchone()[0]

    async def complete(request):
        nonlocal calls
        await request.read()
        calls += 1
        if calls == 1:
            return web.Response(body=b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n',
                                content_type="text/event-stream")
        async with asyncio.timeout(5):
            while not await asyncio.to_thread(committed):
                await asyncio.sleep(.01)
        native_failure.set()
        await runtimes[0].native.close()
        await finished.wait()
        return web.Response()

    app = web.Application()
    app.router.add_post("/v1/chat/completions", complete)
    async with TestServer(app) as server:
        config = _config(source, "openai-messages", {}, str(server.make_url("")), database)
        config["pipeline"]["selector"]["settings"]["limit"] = 2
        config["pipeline"]["scheduler"]["settings"]["stop"] = {"session_count": 2}
        config["pipeline"]["executor"]["plugin"] = "captured"
        try:
            with pytest.raises(RuntimeError, match="native task failure"):
                await asyncio.wait_for(Coordinator(registry).execute(config, base_dir=tmp_path), 15)
        finally:
            finished.set()
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT status FROM runs").fetchone() == ("failed",)
        assert connection.execute("SELECT count(*) FROM turns WHERE status='succeeded'").fetchone() == (1,)
    assert calls == 2
    assert runtimes[0].native.active_count == 0
    assert runtimes[0].native.closed


async def test_request_analysis_html_profiles_and_offline_replay(tmp_path, tokenizer_path):
    from perf_benchmark.app.report import generate
    from perf_benchmark.core.tokenization import ModelTokenizer

    source, database, report = tmp_path / 'analysis.json', tmp_path / 'analysis.db', tmp_path / 'analysis.html'
    _write(source, {
        'id': 'profile', 'token_name': 'session', 'created': '2025-01-01T00:00:00Z',
        'input_tokens': 20, 'output_tokens': 4,
        'messages': [
            {'role': 'system', 'content': [{'type': 'text', 'text': 'a b'}]},
            {'role': 'developer', 'content': [{'type': 'text', 'text': 'c d'}]},
            {'role': 'user', 'content': [{'type': 'text', 'text': 'old'}]},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': 'yes'}], 'reasoning_content': 'PRIVATE_REASONING',
             'tool_calls': [{'id': 'call', 'type': 'function',
                             'function': {'name': 'f', 'arguments': '{"secret":"PRIVATE_ARGUMENT"}'}}]},
            {'role': 'user', 'content': [{'type': 'text', 'text': 'new'},
                                        {'type': 'image_url', 'image_url': {'url': 'PRIVATE_IMAGE'}}]},
        ],
        'tools': [{'type': 'function', 'function': {'name': 'f', 'parameters': {'type': 'object'}}}],
    })
    received = []

    async def complete(request):
        body = await request.read()
        received.append((body, json.loads(body)))
        return web.json_response({'choices': [{'message': {'role': 'assistant', 'content': 'answer'}}],
                                  'usage': {'prompt_tokens': 11, 'completion_tokens': 4,
                                            'prompt_tokens_details': {'cached_tokens': 0}}})

    app = web.Application()
    app.router.add_post('/v1/chat/completions', complete)
    async with TestServer(app) as server:
        config = _config(source, 'openai-messages', {'tokenizer_path': str(tokenizer_path)},
                         str(server.make_url('')), database)
        config['pipeline']['executor']['settings'].update(stream=False)
        config['pipeline']['executor']['settings']['extra_body'].update(temperature=0.2, top_p=0.8)
        config['pipeline']['exporter'] = {
            'plugins': [{'plugin': 'html', 'settings': {'output': str(report)}}],
        }
        run = await Coordinator().execute(config, base_dir=tmp_path)
    assert run.status == 'succeeded'
    assert run.artifacts['report'] == str(report)
    reader = SQLiteResultReader(database)
    turn = (await asyncio.to_thread(lambda: tuple(reader.iter_turn_records(run.id))))[0]
    profile = turn['measurements']['request']['profile']
    assert profile['body_bytes'] == len(received[0][0])
    assert profile['message_count'] == 5 and profile['history_message_count'] == 2
    assert profile['text_chars'] == 15 and profile['system_chars'] == profile['developer_chars'] == 3
    assert profile['tool_count'] == 1 and profile['model'] == 'mock' and profile['stream'] is False
    assert profile['temperature'] == 0.2 and profile['top_p'] == 0.8 and profile['max_tokens'] == 3
    tokenizer = await asyncio.to_thread(ModelTokenizer, str(tokenizer_path))
    assert profile['system_tokens_estimate'] == len(tokenizer.encode('a b'))
    assert profile['tokenizer_fingerprint'] == tokenizer.fingerprint
    assert turn['usage']['cached_tokens'] == 0
    trial = (await asyncio.to_thread(reader.trials, run.id))[0]
    groups = await asyncio.to_thread(reader.grouped_metrics, run.id, trial['trial_id'])
    assert groups and any(g.dimensions == (('cache', 'reported_miss'),) for g in groups)
    html = report.read_text()
    assert all(secret not in html for secret in ('PRIVATE_REASONING', 'PRIVATE_ARGUMENT', 'PRIVATE_IMAGE'))
    replay = await generate(database, tmp_path / 'replay.html')
    assert replay.outputs[0].format == 'html'
    # The archive is the permanent reader location; formal projection remains identical.
    assert (tmp_path / 'replay.html').read_text().replace(str(database) + '.archive', str(database)) == html


@pytest.mark.parametrize('mode', ['recorded', 'live'])
async def test_multiturn_request_profiles_match_wire(tmp_path, tokenizer_path, mode):
    source, database = tmp_path / 'conversation.jsonl', tmp_path / 'conversation.db'
    _write(source, {'id': 'conversation', 'conversations': [
        {'from': 'human', 'value': 'first'}, {'from': 'gpt', 'value': 'recorded-first'},
        {'from': 'human', 'value': 'second'}, {'from': 'gpt', 'value': 'recorded-second'},
    ]})
    requests = []
    server = await _mock_server(requests)
    try:
        config = _config(source, 'sharegpt', {'tokenizer_path': str(tokenizer_path),
                                            'multi_turn': True, 'live_continuation': mode == 'live'},
                         str(server.make_url('')), database)
        run = await Coordinator().execute(config, base_dir=tmp_path)
    finally:
        await server.close()
    assert run.status == 'succeeded'
    reader = SQLiteResultReader(database)
    rows = await asyncio.to_thread(lambda: tuple(reader.iter_turn_records(run.id)))
    for row, sent in zip(rows, requests, strict=True):
        profile = row['measurements']['request']['profile']
        assert profile['message_count'] == len(sent['messages'])
        assert profile['text_chars'] == sum(len(m['content']) for m in sent['messages'])
        assert profile['system_chars'] == profile['system_tokens_estimate'] == 0
        assert profile['context_mode'] == mode
    assert rows[1]['measurements']['request']['profile']['history_message_count'] == 2


async def test_controlled_request_analysis_changes_one_factor(tmp_path, tokenizer_path):
    # The cache cases deliberately use identical request bodies. Only the mock
    # service's reported cache usage differs; no cache speedup is assumed.
    variants = [(8, 16, 0), (32, 16, 0), (128, 16, 0), (8, 4, 0),
                (8, 64, 0), (8, 16, 4), (8, 16, None)]
    cases = [case for case in variants for _ in range(3)]
    source = tmp_path / "controlled.jsonl"
    source.write_text("".join(json.dumps({
        "id": str(index), "token_name": str(index), "created": 0,
        "messages": [{"role": "system", "content": "a b"},
                     {"role": "developer", "content": "b a"},
                     {"role": "user", "content": " ".join(["a"] * (input_tokens - 4))}],
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "requested_output_tokens": output_tokens,
    }) + "\n" for index, (input_tokens, output_tokens, _) in enumerate(cases)))
    received = []

    async def complete(request):
        body = await request.json()
        input_tokens, output_tokens, cached = cases[len(received)]
        received.append(body)
        assert sum(len(message["content"].split()) for message in body["messages"]) == input_tokens
        assert body["max_tokens"] == output_tokens
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await asyncio.sleep(0.003)
        for _ in range(output_tokens):
            await response.write(b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n')
            await asyncio.sleep(0.002)
        usage: dict[str, int | dict[str, int]] = {"prompt_tokens": input_tokens, "completion_tokens": output_tokens,
                 "total_tokens": input_tokens + output_tokens}
        if cached is not None:
            usage["prompt_tokens_details"] = {"cached_tokens": cached}
        final = {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": usage}
        await response.write(("data: " + json.dumps(final) + "\n\n").encode())
        await response.write(b"data: [DONE]\n\n")
        return response

    app = web.Application()
    app.router.add_post("/v1/chat/completions", complete)
    database, report = tmp_path / "controlled.db", tmp_path / "controlled.html"
    async with TestServer(app) as server:
        config = _config(source, "openai-messages", {"tokenizer_path": str(tokenizer_path)},
                         str(server.make_url("")), database)
        config["name"] = "MOCK · 单因素请求分析 / 不代表真实模型性能"
        config["pipeline"]["selector"]["settings"]["limit"] = len(cases)
        config["pipeline"]["scheduler"]["settings"]["stop"] = {"session_count": len(cases)}
        config["pipeline"]["exporter"] = {
            "plugins": [{"plugin": "html", "settings": {"output": str(report)}}],
        }
        run = await Coordinator().execute(config, base_dir=tmp_path)
    assert run.status == "succeeded" and len(received) == len(cases)
    assert received[0] == received[15] == received[18]
    reader = SQLiteResultReader(database)
    rows = await asyncio.to_thread(lambda: tuple(reader.iter_turn_records(run.id)))
    for row, (input_tokens, output_tokens, cached) in zip(rows, cases, strict=True):
        assert row["usage"]["prompt_tokens"] == input_tokens
        assert row["usage"]["completion_tokens"] == output_tokens
        assert row["usage"].get("cached_tokens") == cached
        assert row["measurements"]["request"]["profile"]["system_tokens_estimate"] == 2
    trial = (await asyncio.to_thread(reader.trials, run.id))[0]
    groups = await asyncio.to_thread(reader.grouped_metrics, run.id, trial["trial_id"])
    overall = [g for g in groups if g.window == "overall" and g.metric == "e2e"]
    outputs = {g.dimensions[0][1]: g for g in overall
               if len(g.dimensions) == 1 and g.dimensions[0][0] == "output_tokens"}
    assert set(outputs) == {"[4,8)", "[16,32)", "[64,128)"}
    assert outputs["[4,8)"].sample_count == outputs["[64,128)"].sample_count == 3
    high, low = outputs["[64,128)"].statistics["am"], outputs["[4,8)"].statistics["am"]
    assert high is not None and low is not None and high > low
    caches = {g.dimensions[0][1]: g.sample_count for g in overall
              if len(g.dimensions) == 1 and g.dimensions[0][0] == "cache"}
    assert caches == {"reported_miss": 15, "reported_hit": 3, "unknown": 3}
    assert report.is_file()
