import io
import logging
import sys

from perf_benchmark.core.logging import configure_logging, shutdown_logging


def test_logging_flushes_background_queue_on_shutdown(monkeypatch) -> None:
    output = io.StringIO()
    monkeypatch.setattr(sys, "stderr", output)

    configure_logging({"level": "INFO", "stream": "stderr"})
    logging.getLogger("perf_benchmark.test").info(
        "stage=metrics event=trial.completed"
    )
    shutdown_logging()

    assert "stage=metrics event=trial.completed" in output.getvalue()
