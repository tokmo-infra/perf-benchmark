"""Small request fixtures for the production native binding."""
from perf_benchmark import _native


def command(**overrides):
    value = dict(url="http://127.0.0.1:1", method="POST", headers={}, session_id="test",
                 connection_scope="trial", verify_ssl=False, body_length=0, stream=False,
                 retain=False, timeout=2., notify_start=False, expectations={},
                 plan=dict(after_outputs=None, after_first_output_ms=None, action="cancel"))
    value.update(overrides)
    value["plan"] = {"after_outputs": None, "after_first_output_ms": None, "action": "cancel", **value["plan"]}
    return value


class Events:
    """Retain each Attempt's awaitables across cancelled test waiters."""
    def __init__(self, native):
        self.native = native
        self.starts = {}
        self.results = {}

    async def started(self, handle):
        import asyncio
        if handle.id not in self.starts:
            self.starts[handle.id] = asyncio.ensure_future(handle.started())
        task = self.starts[handle.id]
        return await asyncio.shield(task)

    async def result(self, handle):
        import asyncio
        if handle.id not in self.results:
            self.results[handle.id] = asyncio.ensure_future(handle.result())
        task = self.results[handle.id]
        return await asyncio.shield(task)


async def request(native, value, body=None, *, on_started=None):
    segments = body.segments if body is not None else ()
    handle = native.submit(_native.PreparedBody(segments, value.get("body_length", 0)), command(**value))
    events = Events(native)
    if on_started is not None:
        at = await events.started(handle)
        if at is not None:
            on_started(at)
    return await events.result(handle)
