
"""Reuse loader/normalization, owning their resources independently of Coordinator."""

import asyncio
from copy import deepcopy
from contextlib import AsyncExitStack
from itertools import groupby
import json
import logging
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import duckdb

from perf_benchmark.core.models import Run, Session, Turn
from perf_benchmark.pipeline.hooks import HookContext, HookRegistry
from perf_benchmark.runtime.events import EventDispatcher
from perf_benchmark.pipeline.context import LoaderContext, NormalizationContext, RunContext
from perf_benchmark.pipeline.contracts.loader import LoaderPlugin
from perf_benchmark.pipeline.contracts.normalization import (
    NORMALIZED_SCHEMA,
    NormalizationPlugin,
)
from perf_benchmark.pipeline.preparation.datasets import LoaderHost
from perf_benchmark.pipeline.preparation.payloads import BlobStore, PayloadRef
from perf_benchmark.pipeline.resources import finish_task, own
from perf_benchmark.pipeline.preparation.streaming import NormalizationHost
from perf_benchmark.plugins import create_registry
from perf_benchmark.plugins.executor.openai_chat.content import read_chat_request
from perf_benchmark.runtime.cancellation import CancellationToken

from .common import local_path


_METADATA_MEMORY_BYTES = 64 * 1024 * 1024


def validate_source(config, base):
    binding = config["loader"]
    if binding["plugin"] != "hf-datasets":
        raise ValueError("offline preparation supports the local hf-datasets loader")
    settings = binding["settings"]
    dataset = settings.get("dataset")
    files = settings.get("data_files")
    if dataset in {"json", "parquet", "arrow", "text", "csv"}:
        def check(value):
            if isinstance(value, dict):
                for item in value.values():
                    check(item)
            elif isinstance(value, list):
                for item in value:
                    check(item)
            else:
                local_path(base, value)
        if not files:
            raise ValueError("offline loader requires explicit local data_files")
        check(files)
    elif not local_path(base, dataset).is_dir():
        raise ValueError("offline loader requires local files or a saved dataset directory")
    if settings.get("load_kwargs"):
        raise ValueError("offline loader does not accept arbitrary load_kwargs")
    if config["normalization"].get("settings", {}).get("live_continuation"):
        raise ValueError("offline datasets cannot reconstruct live history")


async def analyze_source(config, base, consume):
    validate_source(config, base)

    async def work():
        run = Run(name="offline-dataset")
        payloads = BlobStore()
        events = EventDispatcher(run, HookRegistry(), {}, HookContext(run.id, run.name, base))
        registry = create_registry()
        async with AsyncExitStack() as scope:
            scope.callback(payloads.close)
            own(scope, events.close)
            settings = deepcopy(config)
            # An explicitly configured HF cache remains reusable; otherwise own the
            # offline reader cache, so cancelled/generated data leaves no extra copy.
            cache = scope.enter_context(tempfile.TemporaryDirectory(prefix="dataset-reader-"))
            settings["loader"]["settings"].setdefault("cache_dir", cache)
            bindings = {
                stage: registry.bind(
                    stage,
                    settings[stage]["plugin"],
                    settings[stage].get("settings", {}),
                )
                for stage in ("loader", "normalization")
            }
            ctx = RunContext(
                bindings,
                base,
                events,
                CancellationToken(),
                payloads,
            )
            loader_plugin = bindings["loader"].plugin
            normalization_plugin = bindings["normalization"].plugin
            if not isinstance(loader_plugin, LoaderPlugin):
                raise TypeError("bound loader does not implement LoaderPlugin")
            if not isinstance(normalization_plugin, NormalizationPlugin):
                raise TypeError(
                    "bound normalization does not implement NormalizationPlugin"
                )
            loader = LoaderHost(loader_plugin, events=events)
            normalizer = NormalizationHost(
                normalization_plugin,
                run_id=run.id,
                payloads=payloads,
                events=events,
            )
            # HF conversion errors may interpolate raw records. Surface a safe
            # diagnostic instead of allowing third-party error text into logs.
            reader_log = logging.getLogger("datasets")
            previous_level = reader_log.level
            reader_log.setLevel(logging.CRITICAL)
            scope.callback(reader_log.setLevel, previous_level)
            try:
                source = await loader.open(LoaderContext(ctx.base_dir, ctx.cancellation))
            except Exception as exc:
                raise ValueError(f"offline loader failed ({type(exc).__name__}); check local input format") from None
            own(scope, source.close)
            normalized_path = Path(cache) / "normalized.parquet"
            try:
                normalized = normalizer.transform(source, NormalizationContext(ctx.base_dir, ctx.cancellation))
                own(scope, normalized.aclose)
                batches = []
                metadata_bytes = 0
                writer = None
                try:
                    async for batch in normalized:
                        if writer is None and metadata_bytes + batch.nbytes <= _METADATA_MEMORY_BYTES:
                            batches.append(batch)
                            metadata_bytes += batch.nbytes
                            continue
                        if writer is None:
                            writer = pq.ParquetWriter(normalized_path, NORMALIZED_SCHEMA, compression=None)
                            for retained in batches:
                                await asyncio.to_thread(writer.write_batch, retained)
                            batches.clear()
                        await asyncio.to_thread(writer.write_batch, batch)
                finally:
                    if writer is not None:
                        await asyncio.to_thread(writer.close)
            except Exception as exc:
                raise ValueError(f"offline normalization failed ({type(exc).__name__}); check adapter fields and tokenizer") from None
            payloads.seal()
            # Spool only oversized metadata; payload text remains in the sealed BlobStore.
            metadata = normalized_path if writer is not None else pa.Table.from_batches(batches, schema=NORMALIZED_SCHEMA)
            return await asyncio.to_thread(consume, snapshots(metadata, payloads))

    # Do not close mmap/files under a still-running reader/tokenizer thread.
    return await finish_task(asyncio.create_task(work()))


def snapshots(metadata, payloads):
    database = str(metadata.parent / "metadata.duckdb") if isinstance(metadata, Path) else ":memory:"
    with duckdb.connect(database, config={"threads": 1}) as db:
        db.execute("SET memory_limit='512MB'")
        if isinstance(metadata, Path):
            db.read_parquet(str(metadata)).create_view("raw")
        else:
            db.register("raw", metadata)
        reader = db.sql("""WITH ordered AS (SELECT *,row_number() OVER () AS input_ordinal FROM raw)
            SELECT * EXCLUDE(input_ordinal) FROM ordered
            ORDER BY min(source_order) OVER (PARTITION BY session_id),
              min(input_ordinal) OVER (PARTITION BY session_id), turn_order_key""").to_arrow_reader(batch_size=128)
        rows = (row for batch in reader for row in batch.to_pylist())
        for identifier, group in groupby(rows, lambda row: row["session_id"]):
            turns = []
            previous_key = None
            for index, row in enumerate(group):
                if row["turn_order_key"] == previous_key:
                    raise ValueError("session has ambiguous turn order")
                previous_key = row["turn_order_key"]
                if row["input_kind"] != "chat" or row["context_mode"] == "live":
                    raise ValueError("offline analysis accepts only fixed text chat history")
                ref = None
                if row["recorded_response_payload_offset"] is not None:
                    ref = PayloadRef(row["recorded_response_payload_offset"], row["recorded_response_payload_bytes"])
                turns.append(Turn(
                    id=row["turn_id"], ordinal=index, context_mode=row["context_mode"],
                    request_payload_ref=PayloadRef(row["request_payload_offset"], row["request_payload_bytes"]),
                    recorded_response_payload_ref=ref,
                    estimated_input_tokens=row["estimated_input_tokens"],
                    reference_output_tokens=row["reference_output_tokens"],
                    requested_output_tokens=row["requested_output_tokens"],
                    metadata=json.loads(row["metadata_json"]),
                ))
            session = Session(id=identifier, turns=turns)
            yield turns[0].metadata["source_session_id"], ((turn, read_chat_request(session, turn, payloads)) for turn in turns)
