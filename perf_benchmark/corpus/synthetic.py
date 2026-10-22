"""AISBench-style single-turn synthesis with explicit pre-template length units."""

from functools import lru_cache, partial
import math
import random

import numpy as np

from perf_benchmark.core.validation import random_seed

from .common import RowsWriter, fingerprint, positive, write_yaml
from .generation import DATASET_SCHEMA, dataset_row

# Bound batch work by rows and sampled tokens (the final row may exceed the token budget).
_BATCH_ROWS = 128
_BATCH_TOKENS = 65536


def _length(value, name):
    positive(value, name)
    if value > 2**20:
        raise ValueError(f"{name} must not exceed 1048576")
    return value


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def length_sampler(value, rng):
    if isinstance(value, int):
        fixed = _length(value, "length")
        return lambda: fixed
    if not isinstance(value, dict):
        raise ValueError("length requires an integer or a distribution object")
    method = value.get("method")
    fields = {"uniform": set(), "gaussian": {"mean", "variance"}, "zipf": {"alpha"},
              "lognormal": {"mean", "sigma"}}
    if method not in fields or set(value) != {"method", "min", "max"} | fields[method]:
        raise ValueError("distribution requires method, min, max and its exact parameters")
    low, high = _length(value["min"], "min"), _length(value["max"], "max")
    if low > high:
        raise ValueError("min must not exceed max")
    if method == "uniform":
        # AISBench samples a continuous uniform value then truncates to int.
        draw = partial(rng.uniform, low, high)
    elif method == "gaussian":
        mean = _number(value["mean"], "mean")
        variance = _number(value["variance"], "variance")
        if variance < 0:
            raise ValueError("variance must be nonnegative")
        draw = partial(rng.normal, mean, math.sqrt(variance))
    elif method == "lognormal":
        mean = _number(value["mean"], "mean")
        sigma = _number(value["sigma"], "sigma")
        if sigma < 0:
            raise ValueError("sigma must be nonnegative")
        draw = partial(rng.lognormal, mean, sigma)
    else:
        alpha = _number(value["alpha"], "alpha")
        if not 1 < alpha <= 10:
            raise ValueError("alpha must be greater than 1 and at most 10")
        draw = partial(rng.zipf, alpha)
    return lambda: int(np.clip(draw(), low, high))


def generate_synthetic(recipe, tokenizer, directory):
    mode = recipe["mode"]
    allowed = {"model", "mode", "sessions", "seed", "output_policy", "input", "output_tokens"}
    if mode == "tokenid":
        allowed.add("shared_prefix_tokens")
    if set(recipe) - allowed:
        raise ValueError("unknown or conflicting synthetic recipe fields")
    count = _length(recipe.get("sessions"), "sessions")
    context = positive(recipe["model"].get("context_tokens"), "context_tokens")
    seed = random_seed(recipe.get("seed", 0))
    policy = recipe.get("output_policy", "fixed")
    if policy not in {"fixed", "natural"}:
        raise ValueError("output_policy must be fixed or natural")
    # Hash signed integer seeds into NumPy's nonnegative seed domain.
    lengths_rng = np.random.default_rng(int(fingerprint([seed, "lengths"]), 16))
    output_length = length_sampler(recipe.get("output_tokens"), lengths_rng)
    rng = random.Random(seed)
    prefix_tokens, prefix = 0, ""
    if mode == "tokenid":
        size = _length(recipe.get("input"), "input")
        prefix_tokens = recipe.get("shared_prefix_tokens", 0)
        if type(prefix_tokens) is not int or not 0 <= prefix_tokens <= size:
            raise ValueError("shared_prefix_tokens must be an integer between 0 and input")
        prefix = tokenizer.decode(rng.choices(tokenizer.ordinary, k=prefix_tokens))
        input_length = length_sampler(size, lengths_rng)
    else:
        input_length = length_sampler(recipe.get("input"), lengths_rng)
    prefix_count = None
    if mode == "tokenid" and size == prefix_tokens:
        prefix_count = tokenizer.count_chat_batch([[{"role": "user", "content": prefix}]])[0]
    @lru_cache(maxsize=128)
    def string_count(size):
        # Retain only integer lengths/counts, never the generated request bodies.
        return tokenizer.count_chat_batch([[{"role": "user", "content": "A " * (size - 1) + "A"}]])[0]

    with RowsWriter(directory / "data.parquet", DATASET_SCHEMA) as writer:
        index = 0
        while index < count:
            lengths, outputs, sequences = [], [], []
            work = 0
            while index + len(lengths) < count and len(lengths) < _BATCH_ROWS and work < _BATCH_TOKENS:
                size, output = input_length(), output_length()
                lengths.append(size)
                outputs.append(output)
                work += size
                if mode == "tokenid":
                    sequences.append(rng.choices(tokenizer.ordinary, k=size - prefix_tokens))
            if mode == "string":
                texts = ["A " * (size - 1) + "A" for size in lengths]
                actuals = [string_count(size) for size in lengths]
            elif prefix_count is not None:
                texts, actuals = [prefix] * len(lengths), [prefix_count] * len(lengths)
            else:
                texts = [prefix + text for text in tokenizer.decode_batch(sequences)]
                actuals = tokenizer.count_chat_batch([[{"role": "user", "content": text}] for text in texts])
            for text, actual, output in zip(texts, actuals, outputs, strict=True):
                request = {"messages": [{"role": "user", "content": text}]}
                if actual + output > context:
                    raise ValueError("actual input plus output exceeds model context_tokens")
                writer.append(dataset_row(f"session-{index:08d}", 0, request, actual, output, mode, policy, tokenizer))
                index += 1
    result = {
        "version": 1, "mode": mode, "seed": seed, "sessions": count, "turns": count,
        "recipe_fingerprint": fingerprint(recipe), "tokenizer_fingerprint": tokenizer.fingerprint,
        "tokenizer": tokenizer.identity, "context_tokens": context, "output_policy": policy,
        "input_length_unit": "sampled_token_ids" if mode == "tokenid" else "A_repetitions",
        "shared_prefix_tokens": prefix_tokens,
        "validation": {"exact_input_lengths": False, "actual_input_lengths_verified": True,
                       "context_limits": True, "tool_history": True, "service_output_lengths": "not_run"},
    }
    write_yaml(directory / "manifest.yaml", result)
    return result
