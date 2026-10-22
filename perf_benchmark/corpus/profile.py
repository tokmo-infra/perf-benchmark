"""Disk-backed, joint-session structure and first-order token statistics."""

from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
import hashlib

import duckdb
import pyarrow as pa

from .common import RowsWriter, canonical, fingerprint, write_yaml
from .requests import request_token_ids

STRUCTURES = pa.schema([("ordinal", pa.int64()), ("session_id", pa.string()), ("structure", pa.string())])
COUNTS = pa.schema([
    ("group", pa.string()), ("first", pa.int32()), ("second", pa.int32()), ("count", pa.int64()),
])
SEEDS = pa.schema([("ordinal", pa.int64()), ("group", pa.string()), ("tokens", pa.list_(pa.int32()))])
PREFIXES = pa.schema([("kind", pa.string()), ("depth", pa.int32()), ("digest", pa.string())])
LENGTHS = pa.schema([
    ("session", pa.int64()), ("turn", pa.int32()), ("input", pa.int64()),
    ("reference_output", pa.int64()), ("requested_output", pa.int64()), ("category", pa.string()),
])


def analyze(sessions, tokenizer, destination, *, keep_seeds=False):
    destination.mkdir(parents=True, exist_ok=True)
    count, turns, seed_count = 0, 0, 0
    with ExitStack() as scope:
        structures = scope.enter_context(RowsWriter(destination / "structures.parquet", STRUCTURES))
        tokens = scope.enter_context(RowsWriter(destination / "token-parts.parquet", COUNTS))
        prefixes = scope.enter_context(RowsWriter(destination / "prefix-parts.parquet", PREFIXES))
        lengths = scope.enter_context(RowsWriter(destination / "lengths.parquet", LENGTHS))
        seeds = scope.enter_context(RowsWriter(destination / "seeds.parquet", SEEDS)) if keep_seeds else None
        for ordinal, (identifier, requests) in enumerate(sessions):
            structure = []
            for index, (turn, request) in enumerate(requests):
                ids = request_token_ids(tokenizer, request)
                identity = turn.metadata.get("dataset_tokenizer_fingerprint")
                if identity is not None and (identity != tokenizer.fingerprint or len(ids) != turn.estimated_input_tokens):
                    raise ValueError("serialized dataset does not match its declared tokenizer/exact input length")
                category = str(turn.metadata.get("dataset_category", "default"))
                template = deepcopy(request)
                chain = []
                for message, skeleton in zip(request["messages"], template["messages"], strict=True):
                    chain.append(message)
                    prefix_id = fingerprint(chain)
                    prefixes.append({"kind": "message", "depth": len(chain), "digest": prefix_id})
                    content = message.get("content")
                    if not isinstance(content, str):
                        continue
                    role = message["role"]
                    group = category + ":" + role
                    encoded = tokenizer.encode(content)
                    for first, frequency in Counter(encoded).items():
                        tokens.append({"group": group, "first": first, "second": -1, "count": frequency})
                    for (first, second), frequency in Counter(zip(encoded, encoded[1:])).items():
                        tokens.append({"group": group, "first": first, "second": second, "count": frequency})
                    if seeds is not None and encoded:
                        seeds.append({"ordinal": seed_count, "group": group, "tokens": encoded})
                        seed_count += 1
                    # Fixed system/tool records remain sensitive structural material.
                    if role not in {"system", "tool"} and not message.get("tool_calls"):
                        skeleton["content"] = {"tokens": len(encoded), "group": group, "prefix_id": prefix_id}
                digest = hashlib.sha256()
                for depth, token in enumerate(ids, 1):
                    digest.update(int(token).to_bytes(8, "little"))
                    # Explicit sampled depths bound profile overhead; not a cache-hit estimate.
                    if depth in (1, 8, 32, 128, 256, 1024, 4096):
                        prefixes.append({"kind": "token", "depth": depth, "digest": digest.hexdigest()})
                lengths.append({
                    "session": ordinal, "turn": index, "input": len(ids),
                    "reference_output": turn.reference_output_tokens,
                    "requested_output": turn.requested_output_tokens, "category": category,
                })
                structure.append({
                    "input_tokens": len(ids), "reference_output_tokens": turn.reference_output_tokens,
                    "reference_source": turn.metadata.get("reference_output_source", "unavailable"),
                    "output_tokens": turn.requested_output_tokens, "category": category, "request": template,
                })
                turns += 1
            if not structure:
                raise ValueError("empty session in dataset")
            structures.append({"ordinal": ordinal, "session_id": identifier, "structure": canonical(structure)})
            count += 1
    if not count:
        raise ValueError("dataset has no chat sessions")
    with duckdb.connect() as db:
        db.execute("SET memory_limit='512MB'")
        db.read_parquet(str(destination / "token-parts.parquet")).create_view("parts")
        db.sql('SELECT "group",first,second,sum(count)::BIGINT AS count FROM parts GROUP BY ALL ORDER BY ALL').write_parquet(str(destination / "tokens.parquet"))
        db.read_parquet(str(destination / "prefix-parts.parquet")).create_view("prefixes")
        db.sql("SELECT kind,depth,digest,count(*) AS count FROM prefixes GROUP BY ALL ORDER BY ALL").write_parquet(str(destination / "prefixes.parquet"))
        db.read_parquet(str(destination / "lengths.parquet")).create_view("lengths")
        summary = db.sql("""SELECT min(input),avg(input),max(input),quantile_cont(input,0.5),
            quantile_cont(input,0.95),count(reference_output),count(requested_output) FROM lengths""").fetchone()
    assert summary is not None
    (destination / "token-parts.parquet").unlink()
    (destination / "prefix-parts.parquet").unlink()
    manifest = {
        "version": 1, "algorithm": "joint-session-unigram-bigram-v1",
        "tokenizer_fingerprint": tokenizer.fingerprint, "tokenizer": tokenizer.identity,
        "sessions": count, "turns": turns, "seed_segments": seed_count,
        "input_tokens": dict(zip(("min", "mean", "max", "p50", "p95", "reference_known", "requested_known"), summary, strict=True)),
        "token_prefix_depths": [1, 8, 32, 128, 256, 1024, 4096],
        "privacy": "Sensitive: fixed prompts/tools, token relationships and seeds are not anonymized.",
        "limits": "Statistical similarity does not establish semantic or production performance equivalence.",
    }
    write_yaml(destination / "profile.yaml", manifest)
    return manifest
