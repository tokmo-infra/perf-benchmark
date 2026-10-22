
from perf_benchmark.pipeline.context import LoaderContext
import json
import os
from threading import Event, Thread
import warnings

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from perf_benchmark.core.models import Run
from perf_benchmark.pipeline.preparation.datasets import LoaderHost, SOURCE_ORDER_COLUMN
from perf_benchmark.plugins.loader.hf_datasets import HFDatasetsLoader, parse_settings


async def _rows(settings, run_context):
    run = Run(name="loader")
    run_ctx = run_context("loader", settings, run)
    ctx = LoaderContext(run_ctx.base_dir, run_ctx.cancellation)
    source = await LoaderHost(
        HFDatasetsLoader(parse_settings(settings)), events=run_ctx.events
    ).open(ctx)
    try:
        return [
            row async for batch in source.batches() for row in batch.to_pylist()
        ], run
    finally:
        await source.close()


def test_hf_datasets_settings_reject_unknown_keys() -> None:
    with pytest.raises(ValueError, match="unknown hf-datasets settings"):
        parse_settings({"dataset": "json", "split": "train", "mystery": True})


@pytest.mark.parametrize("builder,suffix", [
    ("text", ".txt"), ("csv", ".csv"), ("json", ".json"),
    ("json", ".jsonl"), ("parquet", ".parquet"), ("arrow", ".arrow"),
])
async def test_hf_datasets_prepares_common_local_formats(
    builder, suffix, tmp_path, run_context, monkeypatch
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    path = tmp_path / f"data{suffix}"
    cache = tmp_path / "cache"
    table = pa.table({"question": ["first", "second"], "value": [1, 2]})
    if builder == "text":
        path.write_text("first\nsecond\n")
    elif builder == "csv":
        path.write_text("question,value\nfirst,1\nsecond,2\n")
    elif suffix == ".json":
        path.write_text(json.dumps(table.to_pylist()))
    elif builder == "json":
        path.write_text("".join(json.dumps(row) + "\n" for row in table.to_pylist()))
    elif builder == "parquet":
        pq.write_table(table, path)
    else:
        with path.open("wb") as stream, pa.ipc.new_stream(stream, table.schema) as writer:
            writer.write_table(table)
    rows, run = await _rows({
        "dataset": builder, "data_files": [str(path)], "split": "train",
        "cache_dir": str(cache),
    }, run_context)
    assert [row[SOURCE_ORDER_COLUMN] for row in rows] == [0, 1]
    assert run.event_counts["loader.batch.loaded"] >= 1
    assert any(cache.rglob("*.arrow"))


async def test_hf_cache_refreshes_after_local_source_change(tmp_path, run_context) -> None:
    path = tmp_path / "data.jsonl"
    settings = {
        "dataset": "json", "data_files": [str(path)], "split": "train",
        "cache_dir": str(tmp_path / "cache"),
    }
    path.write_text('{"question":"first"}\n')
    first, _ = await _rows(settings, run_context)
    path.write_text('{"question":"second"}\n')
    second, _ = await _rows(settings, run_context)
    assert first[0]["question"] == "first"
    assert second[0]["question"] == "second"


async def test_hf_datasets_caches_multiline_json_objects_with_mixed_content(
    tmp_path, run_context, monkeypatch
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    path = tmp_path / "record.json"
    record = {
        "messages": [
            {"role": "user", "content": "plain"},
            {"role": "user", "content": [{"type": "text", "text": "nested"}]},
        ]
    }
    path.write_text(json.dumps(record, indent=2))
    settings = {
        "dataset": "json", "data_files": [str(path)], "split": "train",
        "cache_dir": str(tmp_path / "cache"),
    }
    rows, _ = await _rows(settings, run_context)
    assert json.loads(rows[0]["messages"]) == record["messages"]
    assert any((tmp_path / "cache").rglob("*.arrow"))


async def test_hf_datasets_orders_local_directory(tmp_path, run_context) -> None:
    directory = tmp_path / "parts"
    directory.mkdir()
    (directory / "b.jsonl").write_text('{"question":"b"}\n')
    (directory / "a.jsonl").write_text('{"question":"a"}\n')
    (directory / "part-10.jsonl").write_text('{"question":"part-10"}\n')
    (directory / "part-2.jsonl").write_text('{"question":"part-2"}\n')
    rows, _ = await _rows({
        "dataset": "json", "data_files": str(directory), "split": "train",
        "cache_dir": str(tmp_path / "cache"),
    }, run_context)
    assert [row["question"] for row in rows] == ["a", "b", "part-2", "part-10"]


async def test_hf_datasets_does_not_fork_from_loader_thread(
    tmp_path,
    run_context,
    monkeypatch,
) -> None:
    paths = []
    for index in range(9):
        path = tmp_path / f"record-{index}.json"
        path.write_text(json.dumps({"question": f"question-{index}"}))
        paths.append(str(path))

    started = Event()
    stop = Event()
    thread = Thread(target=lambda: (started.set(), stop.wait()))
    thread.start()
    started.wait()
    forks = []
    real_fork = os.fork

    def tracked_fork():
        forks.append(True)
        return real_fork()

    monkeypatch.setattr(os, "fork", tracked_fork)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            rows, _ = await _rows(
                {
                    "dataset": "json",
                    "data_files": paths,
                    "split": "train",
                    "cache_dir": str(tmp_path / "cache"),
                },
                run_context,
            )
    finally:
        stop.set()
        thread.join()

    assert [row["question"] for row in rows] == [
        f"question-{index}" for index in range(9)
    ]
    assert forks == []
