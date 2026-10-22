from __future__ import annotations

import logging
from logging.handlers import QueueHandler, QueueListener
from queue import SimpleQueue
import sys
from typing import Any


class _LocalQueueHandler(QueueHandler):
    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        return record


_queue_handler: QueueHandler | None = None
_sink_handler: logging.Handler | None = None
_listener: QueueListener | None = None


def configure_logging(value: Any) -> None:
    shutdown_logging()
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise TypeError("logging must be an object")
    level_name = str(value.get("level", "INFO")).upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        raise ValueError(f"unknown logging level: {level_name}")
    stream_name = str(value.get("stream", "stderr"))
    if stream_name not in {"stdout", "stderr"}:
        raise ValueError("logging.stream must be stdout or stderr")
    root = logging.getLogger("perf_benchmark")
    root.setLevel(level)
    root.propagate = False
    sink = logging.StreamHandler(
        sys.stdout if stream_name == "stdout" else sys.stderr
    )
    sink.setLevel(level)
    sink.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    ))
    queue: SimpleQueue[logging.LogRecord] = SimpleQueue()
    handler = _LocalQueueHandler(queue)
    handler.setLevel(level)
    listener = QueueListener(queue, sink, respect_handler_level=True)
    listener.start()
    root.addHandler(handler)

    global _queue_handler, _sink_handler, _listener
    _queue_handler = handler
    _sink_handler = sink
    _listener = listener


def shutdown_logging() -> None:
    global _queue_handler, _sink_handler, _listener
    root = logging.getLogger("perf_benchmark")
    if _queue_handler is not None:
        root.removeHandler(_queue_handler)
    if _listener is not None:
        _listener.stop()
    if _sink_handler is not None:
        _sink_handler.close()
    _queue_handler = None
    _sink_handler = None
    _listener = None


__all__ = ["configure_logging", "shutdown_logging"]
