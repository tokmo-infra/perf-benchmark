from __future__ import annotations

import asyncio


class CancellationToken:
    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._reason: str | None = None

    @property
    def reason(self) -> str:
        return self._reason or "cancelled"

    def cancel(self, reason: str) -> None:
        if not self._event.is_set():
            self._reason = reason
            self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    async def wait(self) -> None:
        await self._event.wait()
