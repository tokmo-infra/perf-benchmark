"""Native buffer ownership and lifecycle tests, without IPC compatibility fixtures."""
import asyncio
import gc
import mmap

from aiohttp import web
from aiohttp.test_utils import TestServer
import pytest

from perf_benchmark import _native
from perf_benchmark.pipeline.preparation.payloads import BlobStore
from tests.runtime.native_support import Events, command


@pytest.mark.parametrize('value', [bytearray(b'x'), memoryview(bytearray(b'x')).toreadonly(),
                                  memoryview(b'abc')[::2], memoryview(b'abcd').cast('B', (2, 2))])
def test_body_rejects_mutable_or_noncontiguous_storage(value):
    with pytest.raises((ValueError, BufferError)):
        _native.PreparedBody((value,), memoryview(value).nbytes)


def test_body_rejects_unregistered_readonly_mmap(tmp_path):
    path = tmp_path / 'body'
    path.write_bytes(b'abc')
    with path.open('rb') as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mapping:
        view = memoryview(mapping)
        with pytest.raises(ValueError, match='sealed BlobStore'):
            _native.PreparedBody((view,), 3)
        view.release()


def test_body_length_failure_releases_exports():
    view = memoryview(b'hello')
    with pytest.raises(ValueError, match='length mismatch'):
        _native.PreparedBody((view,), 4)
    view.release()


def test_prepared_body_pins_mapping_and_releases_idempotently():
    store = BlobStore()
    ref = store.append(b'hello')
    store.seal()
    view = store.view(ref)
    body = _native.PreparedBody((view,), 5)
    with pytest.raises(BufferError):
        view.release()
    with pytest.raises(BufferError):
        store.close()
    body.release()
    body.release()
    view.release()
    store.close()


async def test_segments_survive_python_gc_and_preserve_bytes():
    expected = b'prefix\x00' + b'large body ' * 10000 + b'\xffsuffix'
    received = []
    async def complete(request):
        received.append(await request.read())
        return web.json_response({'choices': [{'message': {'content': 'ok'}}]})
    app = web.Application()
    app.router.add_post('/', complete)
    async with TestServer(app) as server:
        store = BlobStore()
        ref = store.append(expected[7:-7])
        store.seal()
        body = _native.PreparedBody((expected[:7], store.view(ref), expected[-7:]), len(expected))
        gc.collect()
        trial = _native.create_trial('openai-chat')
        events = Events(trial)
        handle = trial.submit(body, command(url=str(server.make_url('/')), body_length=len(expected)))
        del body
        gc.collect()
        result = await events.result(handle)
        assert result.status == 'succeeded'
        assert received == [expected]
        await handle.settled()
        store.close()
        await trial.close()


async def test_cancelling_wait_does_not_discard_execution_result():
    entered, release = asyncio.Event(), asyncio.Event()
    async def complete(request):
        await request.read()
        entered.set()
        await release.wait()
        return web.json_response({'choices': [{'message': {'content': 'ok'}}]})
    app = web.Application()
    app.router.add_post('/', complete)
    async with TestServer(app) as server:
        trial = _native.create_trial('openai-chat')
        events = Events(trial)
        handle = trial.submit(_native.PreparedBody((), 0), command(url=str(server.make_url('/'))))
        waiting = asyncio.ensure_future(events.result(handle))
        await entered.wait()
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        handle.cancel()
        handle.cancel()
        result = await asyncio.wait_for(events.result(handle), 3)
        assert result.status == 'cancelled'
        release.set()
        await trial.close()
        assert trial.active_count == 0


async def test_pre_cancelled_attempt_never_emits_started():
    trial = _native.create_trial('openai-chat')
    events = Events(trial)
    handle = trial.submit(_native.PreparedBody((), 0), command(notify_start=True), cancelled=True)
    assert await events.started(handle) is None
    result = await events.result(handle)
    assert result.started_ns is None and result.status == 'cancelled'
    assert not result.samples
    await trial.close()


async def test_upload_backpressure_cancel_releases_mmap():
    entered, release = asyncio.Event(), asyncio.Event()
    async def hold(reader, writer):
        await reader.readuntil(b'\r\n\r\n')
        entered.set()
        await release.wait()
        writer.close()
        await writer.wait_closed()
    server = await asyncio.start_server(hold, '127.0.0.1', 0)
    trial = _native.create_trial('openai-chat')
    events = Events(trial)
    store = BlobStore()
    ref = store.append(b'x' * (32 * 1024 * 1024))
    store.seal()
    view = store.view(ref)
    body = _native.PreparedBody((view,), len(view))
    port = server.sockets[0].getsockname()[1]
    handle = trial.submit(body, command(url=f'http://127.0.0.1:{port}', body_length=len(view), timeout=5))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        with pytest.raises(BufferError):
            view.release()
        handle.cancel()
        result = await asyncio.wait_for(events.result(handle), 3)
        assert result.status == 'cancelled'
        await handle.settled()
        view.release()
        store.close()
        assert trial.active_count == 0
    finally:
        release.set()
        await trial.close()
        server.close()
        await server.wait_closed()


async def test_close_rejects_submit_and_releases_prepared_body():
    trial = _native.create_trial('openai-chat')
    await trial.close()
    await trial.close()
    body = _native.PreparedBody((b'x',), 1)
    with pytest.raises(RuntimeError, match='closed'):
        trial.submit(body, command(body_length=1))
    body.release()


async def test_unconsumed_result_does_not_block_other_attempt_cancellation():
    trial = _native.create_trial('openai-chat')
    events = Events(trial)
    handles = [trial.submit(_native.PreparedBody((), 0), command()) for _ in range(20)]
    cancelled = trial.submit(_native.PreparedBody((), 0), command(), cancelled=True)
    assert (await events.result(cancelled)).status == 'cancelled'
    await trial.close()
    for handle in handles:
        assert (await events.result(handle)).status in {'failed', 'cancelled'}


async def test_runtime_clock_check_and_prepare_once(trial_context, monkeypatch):
    from perf_benchmark.plugins.executor.openai_chat import OpenAIChatExecutor, validate
    from perf_benchmark.pipeline.contracts.executor import ExecutorTrial
    ctx = trial_context('executor')
    plugin = OpenAIChatExecutor(validate({'base_url': 'http://127.0.0.1:1', 'model': 'test'}))
    runtime = await plugin.create_runtime(ExecutorTrial('clock'), ctx)
    monkeypatch.setattr(_native, 'now', lambda: -1)
    try:
        with pytest.raises(RuntimeError, match='monotonic clock'):
            await runtime.prepare()
        assert runtime.native is None
        with pytest.raises(RuntimeError, match='exactly once'):
            await runtime.prepare()
        await runtime.close()
    finally:
        await ctx.events.close()


async def test_bad_native_body_releases_python_preparation(trial_context):
    from tests.runtime.test_request import setup_request
    from perf_benchmark.runtime.http.request import PreparedBody
    ctx, runtime, value = await setup_request(trial_context)
    original = runtime.plugin.build_request
    view = memoryview(bytearray(b'x')).toreadonly()
    async def body(_):
        return PreparedBody((view,), 1)
    def build(*args):
        spec = original(*args)
        spec.build_body = body
        return spec
    runtime.plugin.build_request = build
    try:
        with pytest.raises(ValueError, match='sealed BlobStore'):
            await runtime.prepare_attempt(value)
        assert not runtime._prepared
        with pytest.raises(ValueError, match='released memoryview'):
            len(view)
        await runtime.close()
    finally:
        await ctx.events.close()


async def test_started_callback_failure_settles_native_before_release(trial_context):
    from tests.runtime.test_request import setup_request
    from perf_benchmark.runtime.attempt import AttemptSignal
    from perf_benchmark.pipeline.contracts.executor import AttemptPlan
    ctx, runtime, value = await setup_request(trial_context)
    signal = AttemptSignal(AttemptPlan())
    def broken(_):
        raise RuntimeError('started callback failed')
    signal.started_callback = broken
    try:
        await runtime.prepare()
        prepared = await runtime.prepare_attempt(value)
        with pytest.raises(RuntimeError, match='started callback failed'):
            await runtime.execute_attempt(prepared, signal)
        assert ctx.cancellation.is_cancelled()
        assert not runtime._prepared and runtime.native.active_count == 0
        await runtime.close()
    finally:
        await ctx.events.close()


@pytest.mark.parametrize('code', [200, 400])
async def test_early_response_during_upload_releases_body(code):
    entered, release = asyncio.Event(), asyncio.Event()
    async def early(reader, writer):
        await reader.readuntil(b'\r\n\r\n')
        payload = b'{"choices":[{"message":{"content":"ok"}}]}'
        writer.write(f'HTTP/1.1 {code} Reply\r\nContent-Length: {len(payload)}\r\n\r\n'.encode() + payload)
        await writer.drain()
        entered.set()
        await release.wait()
        writer.close()
        await writer.wait_closed()
    server = await asyncio.start_server(early, '127.0.0.1', 0)
    trial = _native.create_trial('openai-chat')
    events = Events(trial)
    store = BlobStore()
    ref = store.append(b'x' * (32 * 1024 * 1024))
    store.seal()
    view = store.view(ref)
    port = server.sockets[0].getsockname()[1]
    handle = trial.submit(_native.PreparedBody((view,), len(view)), command(url=f'http://127.0.0.1:{port}', body_length=len(view)))
    try:
        await entered.wait()
        result = await asyncio.wait_for(events.result(handle), 3)
        assert result.status == ('succeeded' if code == 200 else 'failed')
        settled = asyncio.ensure_future(handle.settled())
        if code == 200:
            assert not settled.done()
            with pytest.raises(BufferError):
                view.release()
            app = web.Application()
            async def complete(request):
                await request.read()
                return web.json_response({'choices': [{'message': {'content': 'next'}}]})
            app.router.add_post('/', complete)
            async with TestServer(app) as quick:
                other = trial.submit(
                    _native.PreparedBody((), 0), command(url=str(quick.make_url('/')))
                )
                assert (await asyncio.wait_for(other.result(), 3)).status == 'succeeded'
                assert not settled.done()
                await other.settled()
        await settled
        view.release()
        store.close()
    finally:
        release.set()
        await trial.close()
        server.close()
        await server.wait_closed()


def test_trials_survive_other_trials_closing_across_event_loops():
    first, second = (_native.create_trial('openai-chat') for _ in range(2))

    async def close_first():
        await first.close()
        assert first.closed and not second.closed

    async def execute_second():
        events = Events(second)
        async def complete(request):
            return web.json_response({'choices': [{'message': {'content': 'ok'}}]})
        app = web.Application()
        app.router.add_post('/', complete)
        async with TestServer(app) as server:
            handle = second.submit(_native.PreparedBody((), 0), command(url=str(server.make_url('/'))))
            assert (await events.result(handle)).status == 'succeeded'
        await second.close()
        assert second.active_count == 0

    asyncio.run(close_first())
    asyncio.run(execute_second())


async def test_each_attempt_delivers_its_result_without_a_trial_consumer():
    trial = _native.create_trial('openai-chat')
    handles = [trial.submit(_native.PreparedBody((), 0), command(), cancelled=True) for _ in range(300)]
    results = await asyncio.gather(*(handle.result() for handle in handles))
    assert all(value.status == 'cancelled' for value in results)
    await asyncio.gather(*(handle.settled() for handle in handles))
    await trial.close()
    assert trial.finished


async def test_settled_releases_exports_independently_of_result():
    trial = _native.create_trial('openai-chat')
    store = BlobStore()
    ref = store.append(b'body')
    store.seal()
    view = store.view(ref)
    handle = trial.submit(_native.PreparedBody((view,), 4), command(body_length=4), cancelled=True)
    assert (await handle.result()).status == 'cancelled'
    await handle.settled()
    view.release()
    store.close()
    await trial.close()


async def test_abandoned_trial_releases_undelivered_exports():
    trial = _native.create_trial('openai-chat')
    store = BlobStore()
    ref = store.append(b'body')
    store.seal()
    view = store.view(ref)
    trial.submit(_native.PreparedBody((view,), 4), command(body_length=4), cancelled=True)
    del trial
    gc.collect()
    async with asyncio.timeout(3):
        while True:
            try:
                view.release()
                break
            except BufferError:
                await asyncio.sleep(.001)
    store.close()


async def test_native_start_notification_matches_independent_result():
    trial = _native.create_trial('openai-chat')
    handle = trial.submit(_native.PreparedBody((), 0), command(notify_start=True))
    started = await handle.started()
    result = await handle.result()
    assert started is not None and result.started_ns == started
    await handle.settled()
    await trial.close()


async def test_conversion_failure_settles_waiter_and_releases_body(trial_context, monkeypatch):
    from tests.runtime.test_request import setup_request
    from perf_benchmark.runtime.http.native_result import NativeResult
    from perf_benchmark.runtime.attempt import AttemptSignal
    from perf_benchmark.pipeline.contracts.executor import AttemptPlan
    ctx, runtime, value = await setup_request(trial_context)
    def fail(*args, **kwargs):
        raise RuntimeError('result conversion failed')
    monkeypatch.setattr(NativeResult, '__new__', staticmethod(fail))
    try:
        await runtime.prepare()
        prepared = await runtime.prepare_attempt(value)
        prepared.spec.url = 'http://127.0.0.1:1'
        with pytest.raises(RuntimeError, match='result conversion failed'):
            await asyncio.wait_for(runtime.execute_attempt(prepared, AttemptSignal(AttemptPlan())), 3)
        assert ctx.cancellation.is_cancelled()
        with pytest.raises(RuntimeError, match='result conversion failed'):
            await runtime.close()
        assert runtime.native.finished and not runtime._prepared
        for part in prepared.body.segments:
            if isinstance(part, memoryview):
                with pytest.raises(ValueError, match='released memoryview'):
                    len(part)
    finally:
        try:
            await runtime.close()
        except RuntimeError:
            pass
        await ctx.events.close()


async def test_attempt_cancellation_joins_its_body_before_return(trial_context):
    from tests.runtime.test_request import setup_request
    from perf_benchmark.runtime.attempt import AttemptSignal
    from perf_benchmark.pipeline.contracts.executor import AttemptPlan
    entered, release = asyncio.Event(), asyncio.Event()
    async def hold(request):
        await request.read()
        entered.set()
        await release.wait()
        return web.json_response({'choices': [{'message': {'content': 'ok'}}]})
    app = web.Application()
    app.router.add_post('/', hold)
    ctx, runtime, value = await setup_request(trial_context)
    async with TestServer(app) as server:
        try:
            await runtime.prepare()
            prepared = await runtime.prepare_attempt(value)
            prepared.spec.url = str(server.make_url('/'))
            task = asyncio.create_task(runtime.execute_attempt(prepared, AttemptSignal(AttemptPlan())))
            await asyncio.wait_for(entered.wait(), 3)
            task.cancel()
            result = await asyncio.wait_for(task, 3)
            assert result.status == 'cancelled'
            assert not runtime._prepared
        finally:
            release.set()
            await runtime.close()
            await ctx.events.close()


async def test_unconsumed_result_does_not_hold_exports_after_close():
    trial = _native.create_trial('openai-chat')
    view = memoryview(b'body')
    trial.submit(_native.PreparedBody((view,), 4), command(body_length=4), cancelled=True)
    await trial.close()
    assert trial.finished
    view.release()


async def test_independent_result_does_not_retain_last_result(trial_context, monkeypatch):
    import weakref
    from tests.runtime.test_request import setup_request
    from perf_benchmark.runtime.http.native_result import NativeResult
    from perf_benchmark.runtime.attempt import AttemptSignal
    from perf_benchmark.pipeline.contracts.executor import AttemptPlan
    class Usage(dict):
        pass
    retained = []
    original = NativeResult.__new__
    def track(cls, *fields):
        fields = list(fields)
        fields[8] = Usage(fields[8])
        retained.append(weakref.ref(fields[8]))
        return original(cls, *fields)
    monkeypatch.setattr(NativeResult, '__new__', staticmethod(track))
    ctx, runtime, value = await setup_request(trial_context)
    try:
        await runtime.prepare()
        prepared = await runtime.prepare_attempt(value)
        prepared.spec.url = 'http://127.0.0.1:1'
        result = await runtime.execute_attempt(prepared, AttemptSignal(AttemptPlan()))
        assert retained[0]() is result.usage
        del result
        await runtime.close_session(value.session.id)
        await runtime.close()
        gc.collect()
        assert retained[0]() is None
    finally:
        await runtime.close()
        await ctx.events.close()


async def test_close_releases_prepared_body_without_submission(trial_context):
    from tests.runtime.test_request import setup_request
    ctx, runtime, value = await setup_request(trial_context)
    try:
        prepared = await runtime.prepare_attempt(value)
        await runtime.prepare()
        await runtime.close()
        assert not runtime._prepared
        for part in prepared.body.segments:
            if isinstance(part, memoryview):
                with pytest.raises(ValueError, match='released memoryview'):
                    len(part)
    finally:
        await ctx.events.close()


def test_sample_buffer_is_readonly_and_retains_its_native_owner():
    import struct
    import sys
    raw = struct.pack('=6q', 1, 100, 8, 2, 200, 16)
    samples = _native.OutputSamples(raw)
    references = sys.getrefcount(samples)
    view = memoryview(samples)
    assert sys.getrefcount(samples) == references + 1
    assert view.readonly and view.contiguous and view.format == 'B'
    with pytest.raises(TypeError):
        view[0] = 0
    assert len(samples) == 2 and samples[-1] == (2, 200, 16)
    assert samples[::-1] == ((2, 200, 16), (1, 100, 8))
    view.release()
    assert sys.getrefcount(samples) == references
    retained = memoryview(samples)
    del samples
    assert retained.cast('q').tolist() == [1, 100, 8, 2, 200, 16]
    retained.release()


@pytest.mark.parametrize('size', [1, 23, 25])
def test_sample_buffer_rejects_partial_rows(size):
    with pytest.raises(ValueError, match='complete i64 triples'):
        _native.OutputSamples(b'\0' * size)


def test_empty_native_sample_buffer_has_no_rows():
    samples = _native.OutputSamples(b'')
    assert not samples and bytes(samples) == b''
    with pytest.raises(IndexError):
        samples[0]


@pytest.mark.parametrize('cancel_request', [False, True])
async def test_settlement_waiters_before_and_after_completion_survive_waiter_cancel(cancel_request):
    entered, release = asyncio.Event(), asyncio.Event()

    async def complete(request):
        await request.read()
        entered.set()
        await release.wait()
        return web.json_response({'choices': [{'message': {'content': 'ok'}}]})

    app = web.Application()
    app.router.add_post('/', complete)
    async with TestServer(app) as server:
        trial = _native.create_trial('openai-chat')
        handle = trial.submit(_native.PreparedBody((), 0), command(url=str(server.make_url('/'))))
        await entered.wait()
        first = handle.settled()
        assert not first.done() and not handle.settled_done
        assert handle.settled() is first

        async def wait():
            await asyncio.shield(first)

        waiter = asyncio.create_task(wait())
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert not first.cancelled()
        if cancel_request:
            handle.cancel()
        release.set()
        result = await asyncio.wait_for(handle.result(), 3)
        assert result.status == ('cancelled' if cancel_request else 'succeeded')
        await asyncio.wait_for(first, 3)
        assert handle.settled_done
        late = handle.settled()
        assert late.done() and late.result() is None
        assert late.get_loop() is asyncio.get_running_loop()
        await trial.close()
        assert trial.active_count == 0


async def test_unobserved_settlement_can_be_awaited_after_trial_close():
    trial = _native.create_trial('openai-chat')
    handle = trial.submit(_native.PreparedBody((), 0), command(), cancelled=True)
    assert (await handle.result()).status == 'cancelled'
    await trial.close()
    assert handle.settled_done
    assert handle.settled().done()
    assert await handle.settled() is None
