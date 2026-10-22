import pytest

from perf_benchmark.core.models import Run, require_metadata


def test_run_preparation_is_separate_from_execution() -> None:
    run = Run(name="lifecycle")
    run.prepare()
    assert run.status == "preparing"
    assert run.preparing_at is not None
    assert run.started_at is None
    assert run.measurements.preparation.started_ns is not None

    run.start()
    assert run.status == "running"
    assert run.started_at is not None
    assert run.measurements.preparation.completed_ns is not None
    assert run.measurements.lifecycle.started_ns is not None


def test_run_cannot_start_before_preparation() -> None:
    with pytest.raises(RuntimeError, match="preparing"):
        Run(name="invalid").start()


def test_require_metadata_copies_nested_json_values() -> None:
    source = {"nested": {"items": [True, 1, 1.5, "text", None]}}

    metadata = require_metadata(source, event="run.before_start")

    assert metadata == source
    assert metadata is not source
    assert metadata["nested"] is not source["nested"]


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({b"key": "value"}, "keys must be strings"),
        ({"payload": b"body"}, "values must be JSON-compatible"),
        ({"nested": [{"payload": b"body"}]}, "values must be JSON-compatible"),
    ],
)
def test_require_metadata_rejects_non_json_values(value, message) -> None:
    with pytest.raises(TypeError, match=message):
        require_metadata(value, event="trial.before_start")


@pytest.mark.parametrize("kind", ["Trial", "Session", "Turn"])
def test_execution_entities_require_explicit_identity(kind):
    from perf_benchmark.core import models
    from perf_benchmark.pipeline.preparation.payloads import PayloadRef

    arguments = {"Trial": {}, "Session": {"turns": []},
                 "Turn": {"request_payload_ref": PayloadRef(0, 0)}}[kind]
    with pytest.raises(TypeError, match="id"):
        getattr(models, kind)(**arguments)
    assert getattr(models, kind)(id="explicit", **arguments).id == "explicit"
