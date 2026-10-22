from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import logging
from time import perf_counter_ns
from typing import Any, Mapping

from perf_benchmark.core.lifecycle import (
    EVENT_NAMES,
    HANDLER_FAILED,
    INTERCEPTOR_EVENTS,
    INTERCEPTOR_FIELDS,
    LifecycleEvent,
)
from perf_benchmark.core.models import Run, freeze_snapshot, now
from perf_benchmark.pipeline.hooks import HookContext, HookRegistry, LifecycleHook

logger = logging.getLogger(__name__)
_STOP = object()
_HOT_EVENTS = frozenset({"scheduler.concurrency.changed"})
_PERFORMANCE_PREFIXES = (
    "request.",
    "scheduler.",
    "session.",
    "stream.",
    "turn.",
    "usage.",
)


@dataclass(slots=True)
class _Worker:
    name: str
    hook: LifecycleHook
    queue: asyncio.Queue
    task: asyncio.Task | None = None


class EventDispatcher:
    def __init__(self, run: Run, registry: HookRegistry, value: Any, context: HookContext) -> None:
        self._run = run
        self._context = context
        self._inline: dict[str, tuple[tuple[str, LifecycleHook], ...]] = {}
        self._background: dict[str, tuple[_Worker, ...]] = {}
        self._workers: list[_Worker] = []
        self._configure(registry, value)

    def _configure(self, registry: HookRegistry, value: Any) -> None:
        if value is None:
            return
        if not isinstance(value, dict):
            raise TypeError("hooks must be an object")
        events = value.get("events", {})
        if not isinstance(events, dict):
            raise TypeError("hooks.events must be an object")
        capacity = int(value.get("queue_capacity", 1024))
        if capacity <= 0:
            raise ValueError("hooks.queue_capacity must be positive")
        for event_name, items in events.items():
            if not isinstance(event_name, str) or not isinstance(items, list):
                raise TypeError("hooks.events values must be lists")
            if event_name not in EVENT_NAMES:
                raise ValueError(f"unknown hook event: {event_name}")
            inline: list[tuple[str, LifecycleHook]] = []
            background: list[_Worker] = []
            for index, item in enumerate(items):
                if not isinstance(item, dict) or not item.get("hook"):
                    raise ValueError(f"hooks.events.{event_name}[{index}].hook is required")
                name = str(item["hook"])
                mode = str(item.get("mode", "inline" if event_name in INTERCEPTOR_EVENTS else "background"))
                if event_name in INTERCEPTOR_EVENTS and mode != "inline":
                    raise ValueError(f"interceptor event {event_name} must be inline")
                if event_name not in INTERCEPTOR_EVENTS and mode != "background":
                    raise ValueError(f"fact event {event_name} must be background")
                settings = item.get("settings", {})
                hook = registry.create(name, settings)
                if mode == "inline":
                    inline.append((name, hook))
                else:
                    worker = _Worker(name, hook, asyncio.Queue(maxsize=capacity))
                    background.append(worker)
                    self._workers.append(worker)
            self._inline[event_name] = tuple(inline)
            self._background[event_name] = tuple(background)

    async def emit(
        self, name: str, payload: object | None = None, *, stage: str | None = None,
        trial_id: str | None = None, session_id: str | None = None,
        turn_id: str | None = None, attempt: int | None = None,
        data: Mapping[str, Any] | None = None, at_ns: int | None = None,
    ) -> object | None:
        inline = self._inline.get(name, ())
        background = self._background.get(name, ())
        if name in _HOT_EVENTS and not inline and not background:
            return payload
        if name in _HOT_EVENTS and not inline:
            ready = tuple(worker for worker in background if worker.queue.empty())
            self._run.dropped_hook_event_count += len(background) - len(ready)
            background = ready
            if not background:
                return payload
        if (
            not inline
            and not background
            and name.startswith(_PERFORMANCE_PREFIXES)
            and not name.endswith((".failed", ".cancelled"))
        ):
            self._run.record_event({"name": name})
            return payload
        event = LifecycleEvent(
            name=name, at=now(), at_ns=at_ns if at_ns is not None else perf_counter_ns(),
            run_id=self._run.id, stage=stage, trial_id=trial_id, session_id=session_id,
            turn_id=turn_id, attempt=attempt, data=freeze_snapshot(dict(data or {})),
        )
        self._run.record_event(event.to_dict())
        if (
            logger.isEnabledFor(logging.DEBUG)
            and not name.startswith(_PERFORMANCE_PREFIXES)
        ):
            logger.debug(
                "event=%s run_id=%s trial_id=%s session_id=%s "
                "turn_id=%s attempt=%s stage=%s data=%s",
                name,
                self._run.id,
                trial_id,
                session_id,
                turn_id,
                attempt,
                stage,
                dict(event.data),
            )
        original = None
        current = payload
        if name in INTERCEPTOR_EVENTS:
            if not isinstance(payload, Mapping):
                raise TypeError(f"{name} requires a mapping payload")
            original = dict(payload)
            current = dict(original)
        for hook_name, hook in inline:
            try:
                result = await hook.handle(event, current, self._context.for_stage(stage))
            except Exception as exc:
                self._record_failure(hook_name, event, exc)
                raise
            if result is not None:
                current = result
        if original is not None:
            if not isinstance(current, Mapping):
                raise TypeError(f"{name} must return a mapping")
            unknown = set(current) - set(INTERCEPTOR_FIELDS[name])
            if unknown:
                raise ValueError(f"{name} returned unsupported fields: {', '.join(sorted(unknown))}")
            return {**original, **dict(current)}
        if background:
            snapshot = freeze_snapshot(
                event.to_dict() if payload is None else payload
            )
            for worker in background:
                if worker.task is None:
                    worker.task = asyncio.create_task(self._work(worker))
                try:
                    worker.queue.put_nowait((event, snapshot))
                except asyncio.QueueFull:
                    self._run.dropped_hook_event_count += 1
        return current

    async def _work(self, worker: _Worker) -> None:
        while True:
            item = await worker.queue.get()
            try:
                if item is _STOP:
                    return
                event, payload = item
                try:
                    await worker.hook.handle(event, payload, self._context.for_stage(event.stage))
                except Exception as exc:
                    self._record_failure(worker.name, event, exc)
            finally:
                worker.queue.task_done()

    def _record_failure(self, hook_name: str, source: LifecycleEvent, error: Exception) -> None:
        self._run.handler_failure_count += 1
        failure = replace(
            source, name=HANDLER_FAILED, at=now(), at_ns=perf_counter_ns(), run_id=self._run.id,
            data=freeze_snapshot({"hook": hook_name, "event": source.name, "error": f"{type(error).__name__}: {error}"}),
        )
        self._run.record_event(failure.to_dict())
        logger.exception("hook failed hook=%s event=%s", hook_name, source.name)

    async def barrier(self) -> None:
        await asyncio.gather(*(worker.queue.join() for worker in self._workers))

    async def close(self) -> None:
        await self.barrier()
        for worker in self._workers:
            if worker.task is not None:
                await worker.queue.put(_STOP)
        await asyncio.gather(
            *(worker.task for worker in self._workers if worker.task is not None),
            return_exceptions=True,
        )
        if self._run.dropped_hook_event_count:
            logger.warning(
                "hook events dropped run_id=%s count=%d",
                self._run.id,
                self._run.dropped_hook_event_count,
            )
