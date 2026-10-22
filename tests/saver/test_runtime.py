
from perf_benchmark.pipeline.results.saver import SaverRuntime
from dataclasses import FrozenInstanceError
from threading import get_ident

import pytest

from perf_benchmark.core.models import Run, Session


class Backend:
    def __init__(self):
        self.calls = []
        self.records = []

    @property
    def artifact(self):
        from perf_benchmark.pipeline.contracts.saver import SaverArtifact

        return SaverArtifact("memory", "memory://results")

    def open(self):
        self.calls.append(("open", get_ident()))

    def write_batch(self, records):
        self.calls.append(("write_batch", get_ident()))
        assert isinstance(records, tuple)
        self.records.extend(records)

    def finalize(self, run):
        self.calls.append(("finalize", get_ident()))
        return object()

    def close(self):
        self.calls.append(("close", get_ident()))


class Plugin:
    def __init__(self, backend):
        self.backend = backend

    def create_backend(self, run, ctx):
        return self.backend

    def open_reader(self, location):
        raise NotImplementedError("this test exercises writes only")


async def test_saver_runtime_owns_queue_and_passes_only_immutable_records(stage_context):
    run = Run(id="run", name="runtime")
    backend = Backend()
    runtime = await SaverRuntime.create(Plugin(backend), run, stage_context("saver"))
    session = Session(id="session", trial_id="trial", turns=[], metadata={"labels": ["a"]})
    await runtime.write_session(session)
    labels = session.metadata["labels"]
    assert isinstance(labels, list)
    labels.append("changed")
    await runtime.flush()

    record = backend.records[0]
    assert record.metadata == {"labels": ("a",)}
    with pytest.raises(FrozenInstanceError):
        record.status = "failed"
    with pytest.raises(TypeError):
        record.metadata["labels"] = ()

    await runtime.finalize(run)
    await runtime.close()
    assert run.artifacts == {"memory": "memory://results"}
    assert [name for name, _ in backend.calls] == [
        "open",
        "write_batch",
        "write_batch",
        "finalize",
        "close",
    ]
    assert len({thread for _, thread in backend.calls}) == 1
