"""Prepare source-specific production traces before a measured Run."""

from contextlib import ExitStack, closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

import pyarrow as pa
import pyarrow.parquet as pq

from perf_benchmark.core.tokenization import ModelTokenizer
from .common import RowsWriter, local_path, positive, publish_directory, write_yaml
from .trace_sources import source_requests
from .trace_text import plain_text


TRACE_SCHEMA = pa.schema([
    ("artifact_version", pa.int16()), ("trace_source", pa.string()), ("session_id", pa.string()),
    ("turn_id", pa.string()), ("turn_order_key", pa.int64()), ("created_at", pa.string()),
    ("input_kind", pa.string()), ("request_json", pa.string()), ("input_tokens", pa.int32()),
    ("output_tokens", pa.int32()), ("source_session_id", pa.string()),
    ("metadata_json", pa.string()), ("causal_json", pa.string()),
])

SOURCE_URLS = {
    "agentx": "https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k",
    "qwen-bailian": "https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon",
    "mooncake": "https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release/traces",
    "burstgpt": "https://github.com/HPMLL/BurstGPT",
    "azure-2024": "https://github.com/Azure/AzurePublicDataset",
}


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _prepared_row(source, entry, request, kind, metadata):
    return {
        "artifact_version": 1, "trace_source": source, "session_id": entry["session"],
        "turn_id": entry["id"], "turn_order_key": entry.get("turn_order_key", 0),
        "created_at": entry["created"], "input_kind": kind,
        "request_json": json.dumps(request, ensure_ascii=False, separators=(",", ":")),
        "input_tokens": entry["input"], "output_tokens": entry["output"],
        "source_session_id": entry.get("source_session"),
        "metadata_json": json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
        "causal_json": json.dumps(entry["causal"], separators=(",", ":")) if entry.get("causal") else None,
    }


def _binding(source, mode, destination):
    selection = ("trace-causal" if (source == "agentx" and mode == "causal")
                 or (source == "servegen" and mode == "hashed-conversations") else "trace-timeline")
    scheduling = ({"max_active_sessions": 4, "max_concurrency": 2} if selection == "trace-causal"
                  else {"max_concurrency": 16, "max_pending_requests": 128})
    return {
        "loader": {"plugin": "hf-datasets", "settings": {"dataset": "parquet", "split": "train",
                                                   "data_files": str(Path(destination).absolute() / "data.parquet")}},
        "normalization": {"plugin": source, "settings": {}},
        "selector": {"plugin": selection, "settings": {}},
        "scheduler": {"plugin": selection, "settings": scheduling},
    }


def prepare_trace(config, base, destination):
    if set(config) - {"source", "input", "source_revision", "model", "seed", "workers", "queue_capacity", "cache_max_bytes",
                       "root_interval_s", "mode", "servegen"}:
        raise ValueError("unknown production trace preparation setting")
    source = config.get("source")
    if source not in {"agentx", "qwen-bailian", "mooncake", "burstgpt", "azure-2024", "servegen"}:
        raise ValueError("source must name one of the six official production trace sources")
    model = config.get("model")
    if not isinstance(model, dict) or set(model) != {"path", "context_tokens"}:
        raise ValueError("model settings are required")
    mode = config.get("mode", "timeline")
    if source == "agentx":
        if mode not in {"timeline", "causal"}:
            raise ValueError("AgentX mode must be timeline or causal")
        if "servegen" in config:
            raise ValueError("AgentX does not use ServeGen generator settings")
    elif source == "servegen":
        if mode not in {"timeline", "hashed-conversations"}:
            raise ValueError("ServeGen mode must be timeline or hashed-conversations")
        if mode == "hashed-conversations" and "servegen" in config:
            raise ValueError("hashed conversations do not use ServeGen generator settings")
        if "root_interval_s" in config or "source_revision" in config:
            raise ValueError("ServeGen does not use AgentX root spacing or source_revision")
    elif mode != "timeline" or "root_interval_s" in config or "servegen" in config:
        raise ValueError("this source only supports timeline preparation")
    tokenizer = ModelTokenizer(local_path(base, model.get("path")))
    context = positive(model.get("context_tokens"), "context_tokens")
    seed = config.get("seed", 0)
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if source == "servegen":
        from .trace_servegen import servegen_requests
        entries, inputs, upstream = servegen_requests(config, base, tokenizer)
    else:
        input_path = local_path(base, config.get("input"))
        revision = config.get("source_revision")
        if not isinstance(revision, str) or not revision:
            raise ValueError("source_revision is required for official trace provenance")
        entries = source_requests(source, input_path, root_interval_s=config.get("root_interval_s"), mode=config.get("mode", "timeline"))
        inputs, upstream = [input_path], {"url": SOURCE_URLS[source], "declared_revision": revision}
    block_sizes = {"qwen-bailian": 16, "mooncake": 512, "agentx": 64}
    workers = positive(config.get("workers", 4), "workers")
    capacity = positive(config.get("queue_capacity", max(1, 1_048_576 // block_sizes[source])
                                   if source in block_sizes else 32), "queue_capacity")
    cache_max_bytes = positive(config.get("cache_max_bytes", 4 * 1024**3), "cache_max_bytes")
    with publish_directory(destination) as output, ExitStack() as scope:
        sampler = None
        if source in block_sizes:
            import tokenizers
            from perf_benchmark._native import TraceHashInflator
            if tokenizers.__version__ != "0.22.2":
                raise ValueError("production trace requires Python tokenizers 0.22.2")
            sampler = scope.enter_context(closing(TraceHashInflator(
                str(tokenizer.path / "tokenizer.json"), str(tokenizer.path / "tokenizer_config.json"),
                block_sizes[source], workers, capacity, cache_max_bytes)))
        scope_key = None
        count = 0
        rng = random.Random(seed)
        with RowsWriter(output / "data.parquet", TRACE_SCHEMA) as writer:
            for entry in entries:
                length, output_length = entry["input"], entry["output"]
                if type(length) is not int or length <= 0 or type(output_length) is not int or output_length <= 0:
                    raise ValueError("source input/output lengths must be positive integers")
                if length + output_length > context:
                    raise ValueError("source input plus output exceeds context_tokens")
                if "request" in entry:
                    request, kind = entry["request"], entry["kind"]
                else:
                    if entry.get("hashes") is not None:
                        key = entry.get("hash_scope", "dataset")
                        if key != scope_key:
                            if scope_key is not None and source != "agentx":
                                raise ValueError("non-AgentX hash scope changed")
                            if scope_key is not None:
                                sampler.reset_scope()
                            scope_key = key
                        assert sampler is not None
                        prompt = sampler.inflate(length, entry["hashes"])
                    else:
                        prompt = plain_text(tokenizer, length, rng)
                    # Each sampled block is validated and delimited by Qwen special
                    # tokens. Joining them preserves their counts; the upstream
                    # replayer only re-encodes a whole prompt in debug mode.
                    if entry.get("hashes") is None and len(tokenizer.encode(prompt)) != length:
                        raise ValueError(f"{entry['id']}: generated text does not meet source input length")
                    request, kind = {"prompt": prompt}, "completion"
                metadata = {"trace_source": source, "trace_mode": "causal" if entry.get("causal") else "timeline",
                            "dataset_output_policy": "fixed", "quality_status": "not_applicable",
                            "source_request_id": entry["id"], **entry["metadata"]}
                writer.append(_prepared_row(source, entry, request, kind, metadata))
                count += 1
        if not count:
            raise ValueError("official source produced no replayable requests")
        # Validate the published schema and row count before atomic rename.
        parquet = pq.ParquetFile(output / "data.parquet")
        if parquet.schema_arrow != TRACE_SCHEMA or parquet.metadata.num_rows != count:
            raise ValueError("prepared Parquet validation failed")
        hash_stats = None
        if sampler:
            hits, misses, partial_overwrites, peak_cache_bytes = sampler.stats()
            hash_stats = {"hits": hits, "misses": misses,
                          "partial_overwrites": partial_overwrites,
                          "peak_cache_bytes": peak_cache_bytes}
        manifest = {"version": 1, "source": source, "rows": count, "model_tokenizer": str(tokenizer.path),
                    "tokenizer_fingerprint": tokenizer.fingerprint, "context_tokens": context,
                    "seed": seed, "hash_block_sampling": "upstream_random" if sampler else None,
                    "hash_generator": ({"backend": "rust", "upstream_commit": "84c9193ac67ebbf0e08db161bc9b048d597c2f80",
                                        "tokenizers_version": "0.22.2", "cache_max_bytes": cache_max_bytes,
                                        "stats": hash_stats} if sampler else None),
                    "inputs": [{"path": str(path), "sha256": _sha256(path)} for path in inputs],
                    "upstream": upstream, "generated_at": datetime.now(timezone.utc).isoformat()}
        write_yaml(output / "manifest.yaml", manifest)
        write_yaml(output / "input.yaml", _binding(source, config.get("mode", "timeline"), destination))
        return manifest
