"""Ownership and cancellation of the Python preparation side of Rust execution."""
import asyncio
import threading
from unittest.mock import AsyncMock

import pytest

from perf_benchmark.core.models import Session
from perf_benchmark.runtime.http.request import PreparedBody
from tests.executor.test_openai_chat import _attempt_input, _runtime, _turn


async def setup_request(trial_context):
    ctx = trial_context("executor")
    turn = await _turn(ctx, "turn", "hello")
    ctx.payloads.seal()
    runtime = _runtime({"base_url": "http://localhost", "model": "test"}, ctx)
    value = _attempt_input(Session(id="session", turns=[turn]), turn, ctx)
    return ctx, runtime, value


async def test_body_views_release_exactly_once_and_close_rejects_new_preparation(trial_context):
    ctx, runtime, value = await setup_request(trial_context)
    try:
        prepared = await runtime.prepare_attempt(value)
        views = [part for part in prepared.body.segments if isinstance(part, memoryview)]
        assert views
        await runtime.release(prepared)
        await runtime.release(prepared)
        for view in views:
            with pytest.raises(ValueError, match="released memoryview"):
                len(view)
        await runtime.close()
        await runtime.close()
        with pytest.raises(RuntimeError, match="closed"):
            await runtime.prepare_attempt(value)
        assert not runtime._prepared
    finally:
        await runtime.close()
        await ctx.events.close()


@pytest.mark.parametrize("failure", [False, True])
async def test_cancelled_preparation_is_joined_and_released_at_close(trial_context, failure):
    ctx, runtime, value = await setup_request(trial_context)
    started, release = threading.Event(), threading.Event()
    body = PreparedBody((memoryview(b"body"),), 4)
    spec = runtime.plugin.build_request(value, runtime.settings)

    def build(_):
        started.set()
        release.wait(5)
        if failure:
            raise ValueError("body construction failed")
        return body

    async def prepare_body(spec):
        return await asyncio.to_thread(build, spec)

    spec.build_body = prepare_body
    runtime.plugin.build_request = lambda *_: spec
    task = asyncio.create_task(runtime.prepare_attempt(value))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        closing = asyncio.create_task(runtime.close())
        await asyncio.sleep(0)
        assert not closing.done()
        release.set()
        if failure:
            with pytest.raises(ValueError, match="body construction failed"):
                await closing
            with pytest.raises(ValueError, match="body construction failed"):
                await runtime.close()
            body.release()
        else:
            await closing
            with pytest.raises(ValueError, match="released memoryview"):
                len(body.segments[0])
        assert not runtime._prepared and not runtime._preparations
    finally:
        release.set()
        await ctx.events.close()


@pytest.mark.parametrize("invalid", [False, True])
async def test_observed_preparation_error_does_not_poison_close(trial_context, invalid):
    ctx, runtime, value = await setup_request(trial_context)
    spec = runtime.plugin.build_request(value, runtime.settings)

    def build(_):
        if invalid:
            return b"unsegmented"
        raise ValueError("body construction failed")

    async def prepare_body(spec):
        return await asyncio.to_thread(build, spec)

    spec.build_body = prepare_body
    runtime.plugin.build_request = lambda *_: spec
    try:
        with pytest.raises((TypeError, ValueError)):
            await runtime.prepare_attempt(value)
        await runtime.close()
        assert not runtime._prepared
    finally:
        await ctx.events.close()


async def test_native_close_failure_still_releases_prepared_body(trial_context):
    ctx, runtime, value = await setup_request(trial_context)
    prepared = await runtime.prepare_attempt(value)
    from types import SimpleNamespace
    runtime.native = SimpleNamespace(close=AsyncMock(side_effect=RuntimeError("native close failed")))
    try:
        for _ in range(2):
            with pytest.raises(RuntimeError, match="native close failed"):
                await runtime.close()
        runtime.native.close.assert_awaited_once()
        assert not runtime._prepared
        for part in prepared.body.segments:
            if isinstance(part, memoryview):
                with pytest.raises(ValueError, match="released memoryview"):
                    len(part)
    finally:
        await ctx.events.close()
