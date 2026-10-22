"""Register one close action, with optional early release and error preservation."""

import asyncio
from contextlib import AsyncExitStack
import logging

logger = logging.getLogger(__name__)


async def finish_task(task):
    """Join owned cleanup even if its waiter is cancelled more than once."""
    cancellation = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if task.cancelled():
                break
            cancellation = cancellation or exc
        except BaseException:
            break
    result = task.result()
    if cancellation is not None:
        raise cancellation
    return result


def own(scope: AsyncExitStack, close):
    task = None

    async def release():
        nonlocal task
        if task is None:
            task = asyncio.create_task(close())
        await finish_task(task)

    async def unwind(exc_type, error, traceback):
        try:
            await release()
        except BaseException as exc:
            if error is None:
                raise
            if exc is not error:
                error.add_note(f"resource close failed: {type(exc).__name__}: {exc}")
            logger.exception("resource close failed while handling %s", type(error).__name__)

    scope.push_async_exit(unwind)
    return release
