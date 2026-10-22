"""Offline random text inputs for embedding and reranking services."""

from contextlib import ExitStack
import random

import duckdb
import numpy as np
import pyarrow as pa

from perf_benchmark.core.validation import random_seed

from .common import RowsWriter, fingerprint, positive, write_yaml
from .synthetic import length_sampler


LENGTHS = pa.schema([
    ("session", pa.int64()), ("input_tokens", pa.int64()),
    ("input_items", pa.int64()), ("text_tokens", pa.list_(pa.int64())),
])


def generate_text_workload(recipe, tokenizer, directory):
    mode = recipe["mode"]
    allowed = {"model", "mode", "sessions", "seed", "input_items"}
    allowed |= {"input"} if mode == "embedding" else {"query_tokens", "document_tokens"}
    if set(recipe) - allowed:
        raise ValueError("unknown or conflicting text workload recipe fields")
    if "template_options" in recipe["model"]:
        raise ValueError("text workloads do not use chat template_options")
    count = positive(recipe.get("sessions"), "sessions")
    items = positive(recipe.get("input_items", 1), "input_items")
    context = positive(recipe["model"].get("context_tokens"), "context_tokens")
    seed = random_seed(recipe.get("seed", 0))
    lengths_rng = np.random.default_rng(int(fingerprint([seed, "lengths"]), 16))
    text_length = length_sampler(recipe.get("input" if mode == "embedding" else "document_tokens"), lengths_rng)
    query_length = length_sampler(recipe.get("query_tokens"), lengths_rng) if mode == "rerank" else None
    rng = random.Random(seed)
    schema = pa.schema([("id", pa.string()), ("input_tokens", pa.int64())] + (
        [("input", pa.list_(pa.string()))] if mode == "embedding" else
        [("query", pa.string()), ("documents", pa.list_(pa.string()))]
    ))
    profile = directory / "profile"
    profile.mkdir()

    def text(size):
        # Decode/encode need not preserve sampled ID count. Record the real local count.
        value = tokenizer.decode(rng.choices(tokenizer.ordinary, k=size))
        actual = len(tokenizer.encode(value))
        if not value or not actual:
            raise ValueError("sampled token IDs produced empty text; choose a different seed or length")
        return value, actual

    with ExitStack() as scope:
        writer = scope.enter_context(RowsWriter(directory / "data.parquet", schema))
        lengths = scope.enter_context(RowsWriter(profile / "lengths.parquet", LENGTHS))
        for index in range(count):
            query, query_count = text(query_length()) if query_length is not None else (None, 0)
            texts, actuals = [], []
            for _ in range(items):
                value, actual = text(text_length())
                if query_count + actual > context:
                    raise ValueError("local text input exceeds model context_tokens")
                texts.append(value)
                actuals.append(actual)
            total = query_count + sum(actuals)
            payload = {"input": texts} if mode == "embedding" else {"query": query, "documents": texts}
            writer.append({"id": f"session-{index:08d}", "input_tokens": total, **payload})
            lengths.append({"session": index, "input_tokens": total, "input_items": items,
                            "text_tokens": ([query_count] if mode == "rerank" else []) + actuals})
    with duckdb.connect() as db:
        db.read_parquet(str(profile / "lengths.parquet")).create_view("lengths")
        summary = db.sql("""SELECT min(input_tokens),avg(input_tokens),max(input_tokens),
            quantile_cont(input_tokens,0.5),quantile_cont(input_tokens,0.95) FROM lengths""").fetchone()
    result = {
        "version": 1, "algorithm": "random-text-inputs-v1", "mode": mode, "normalization": mode, "seed": seed,
        "sessions": count, "turns": count, "input_items": items,
        "recipe_fingerprint": fingerprint(recipe), "tokenizer_fingerprint": tokenizer.fingerprint,
        "tokenizer": tokenizer.identity, "context_tokens": context,
        "input_length_unit": "sampled_token_ids", "input_tokens_basis": "sum_of_encoded_texts",
        "context_limit_basis": "each_embedding_input" if mode == "embedding" else "query_plus_each_document",
        "input_tokens": dict(zip(("min", "mean", "max", "p50", "p95"), summary, strict=True)),
        "validation": {"exact_input_lengths": False, "actual_input_lengths_verified": True,
                       "local_text_context_limits": True, "service_input_tokens": "not_run"},
        "limits": "Local text counts exclude service special tokens and rerank query repetition; service usage remains authoritative.",
    }
    write_yaml(profile / "profile.yaml", result)
    write_yaml(directory / "manifest.yaml", result)
    return result
