"""Read-only, bounded schema inspection before choosing a normalization mapping."""

import asyncio
from array import array
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import yaml

from perf_benchmark.pipeline.context import LoaderContext
from perf_benchmark.pipeline.context import NormalizationContext, SelectorContext
from perf_benchmark.pipeline.preparation.datasets import LoaderHost
from perf_benchmark.pipeline.preparation.streaming import NormalizationHost, SelectorHost
from perf_benchmark.pipeline.preparation.payloads import BlobStore
from perf_benchmark.plugins import create_registry
from perf_benchmark.runtime.cancellation import CancellationToken


_ROLES = {
    "session_id": ("session", "conversation", "dialog", "thread"),
    "message": ("message", "prompt", "question", "content", "text", "input"),
    "created_at": ("created", "timestamp", "time", "arrival"),
    "length": ("token", "length", "output_len", "max_new"),
}


class _NoEvents:
    async def emit(self, *_args, **_kwargs):
        return None


class _Distribution:
    def __init__(self):
        labels = ("<=128", "<=512", "<=2048", "<=8192", ">8192", "unknown")
        self.input = dict.fromkeys(labels, 0)
        self.output = dict.fromkeys(labels, 0)
        self.count = self.known = 0
        self.earliest = self.latest = None
        self.timestamps = array("q")

    def add(self, batch):
        self.count += batch.num_rows
        for field, target in (("estimated_input_tokens", self.input), ("requested_output_tokens", self.output)):
            values = batch.column(batch.schema.get_field_index(field)).to_pylist()
            for value in values:
                key = "unknown" if value is None else next((f"<={limit}" for limit in (128, 512, 2048, 8192) if value <= limit), ">8192")
                target[key] += 1
        for value in batch.column(batch.schema.get_field_index("created_at")).to_pylist():
            if value is not None:
                self.known += 1
                self.earliest = value if self.earliest is None else min(self.earliest, value)
                self.latest = value if self.latest is None else max(self.latest, value)
                self.timestamps.append(int(value.timestamp() * 1_000_000))

    def view(self):
        ordered = sorted(self.timestamps)
        def quartile(fraction):
            return datetime.fromtimestamp(ordered[int((len(ordered) - 1) * fraction)] / 1_000_000,
                                          tz=timezone.utc).isoformat() if ordered else None
        return {"turns": self.count, "input_tokens": self.input,
                "requested_output_tokens": self.output,
                "created_at": {"known": self.known,
                               "earliest": self.earliest.isoformat() if self.earliest else None,
                               "p25": quartile(.25), "p50": quartile(.5), "p75": quartile(.75),
                               "latest": self.latest.isoformat() if self.latest else None}}


async def _preparation_audit(source, base_dir: Path) -> dict[str, object]:
    registry = create_registry()
    loader = registry.bind("loader", source["loader"]["plugin"], source["loader"].get("settings", {})).plugin
    normalizer = registry.bind("normalization", source["normalization"]["plugin"], source["normalization"].get("settings", {})).plugin
    selector = registry.bind("selector", source["selector"]["plugin"], source["selector"].get("settings", {})).plugin
    cancellation, events = CancellationToken(), _NoEvents()
    payloads = BlobStore()
    async with AsyncExitStack() as scope:
        scope.callback(payloads.close)
        loaded = await LoaderHost(loader, events=events).open(LoaderContext(base_dir, cancellation))
        scope.push_async_callback(loaded.close)
        normalized = NormalizationHost(normalizer, run_id="dataset-inspect", payloads=payloads, events=events).transform(
            loaded, NormalizationContext(base_dir, cancellation))
        scope.push_async_callback(normalized.aclose)
        before = _Distribution()
        async def observed():
            async for batch in normalized:
                before.add(batch)
                yield batch
        selected = await SelectorHost(selector, events=events).select(observed(), SelectorContext(cancellation))
        after = _Distribution()
        for batch in selected.batches:
            after.add(batch)
        return {"source_rows": loaded.rows_read, "candidate_sessions": selected.counts["candidates"],
                "selected_sessions": selected.counts["selected"],
                "before": before.view(), "after": after.view()}


async def inspect(config_path: Path) -> dict[str, object]:
    path = config_path.expanduser().resolve(strict=True)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("inspection config must be an object")
    source = config.get("pipeline", config.get("source", config))
    if not isinstance(source, dict) or not isinstance(source.get("loader"), dict):
        raise ValueError("inspection requires a loader binding")
    binding = source["loader"]
    plugin = create_registry().bind("loader", binding["plugin"], binding.get("settings", {})).plugin
    try:
        dataset = await plugin.open(LoaderContext(path.parent, CancellationToken()))
    except Exception as exc:
        raise ValueError(f"inspection loader failed ({type(exc).__name__}); check source settings") from None
    try:
        schema = dataset.schema
        sentinel = object()
        batch = await asyncio.to_thread(next, iter(dataset.batches()), sentinel)
        if batch is sentinel:
            raise ValueError("dataset is empty")
        if not isinstance(batch, pa.RecordBatch):
            raise TypeError("loader did not yield an Arrow batch")
        if not batch.schema.equals(schema, check_metadata=False):
            raise TypeError("loader batch does not match its declared schema")
        sample = batch.slice(0, 32)
        fields = []
        candidates = {role: [] for role in _ROLES}
        ambiguous_units = []
        for field in schema:
            column = sample.column(sample.schema.get_field_index(field.name))
            fields.append({"name": field.name, "type": str(field.type),
                           "nullable": field.nullable, "sample_nulls": column.null_count})
            lowered = field.name.casefold()
            for role, hints in _ROLES.items():
                if any(hint in lowered for hint in hints):
                    candidates[role].append(field.name)
            if (any(hint in lowered for hint in _ROLES["created_at"])
                    and (pa.types.is_integer(field.type) or pa.types.is_floating(field.type))):
                ambiguous_units.append(field.name)
        result = {"sample_rows": sample.num_rows, "fields": fields,
                "candidate_columns": candidates,
                "time_unit_requires_confirmation": ambiguous_units,
                "mapping_status": "suggestions_only"}
    finally:
        await dataset.close()
    if "normalization" in source and "selector" in source:
        try:
            result["preparation_audit"] = await _preparation_audit(source, path.parent)
        except Exception as exc:
            raise ValueError(f"inspection preparation failed ({type(exc).__name__}); check mappings and tokenizer") from None
    return result
