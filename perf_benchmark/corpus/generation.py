"""Three content strategies, one snapshot constructor and exact-length gate."""

from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from itertools import accumulate
import json
import random
from perf_benchmark.core.validation import random_seed
import tempfile
from pathlib import Path

import duckdb
import pyarrow as pa

from .common import RowsWriter, canonical, fingerprint, positive, read_yaml, write_yaml
from .requests import fit_input, request_token_ids

DATASET_SCHEMA = pa.schema([
    ("id", pa.string()), ("token_name", pa.string()), ("created", pa.string()),
    ("messages", pa.json_()), ("tools", pa.json_()), ("tool_choice", pa.json_()),
    ("chat_template_kwargs", pa.json_()), ("add_generation_prompt", pa.bool_()),
    ("input_tokens", pa.int64()), ("requested_output_tokens", pa.int64()),
    ("dataset_category", pa.string()), ("dataset_output_policy", pa.string()),
    ("dataset_tokenizer_fingerprint", pa.string()),
])


def dataset_row(session_id, turn_index, request, actual, output, category, policy, tokenizer):
    def wire(value):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    created = datetime(2000, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=turn_index)
    return {
        "id": f"{session_id}-turn-{turn_index:05d}", "token_name": session_id,
        "created": created.isoformat(), "messages": wire(request["messages"]),
        "tools": wire(request["tools"]) if "tools" in request else None,
        "tool_choice": wire(request["tool_choice"]) if "tool_choice" in request else None,
        "chat_template_kwargs": canonical({key: value for key, value in tokenizer.options.items() if key != "add_generation_prompt"}),
        "add_generation_prompt": tokenizer.options["add_generation_prompt"],
        "input_tokens": actual, "requested_output_tokens": output,
        "dataset_category": category, "dataset_output_policy": policy,
        "dataset_tokenizer_fingerprint": tokenizer.fingerprint,
    }


def weighted(rng, entries):
    if not isinstance(entries, list) or not entries or not all(isinstance(entry, dict) for entry in entries):
        raise ValueError("weighted alternatives must be a non-empty list")
    weights = [entry.get("weight", 1) for entry in entries]
    if any(isinstance(w, bool) or not isinstance(w, (int, float)) or not 0 < w < float("inf") for w in weights):
        raise ValueError("weights must be finite and positive")
    return rng.choices(entries, weights=weights, k=1)[0]


class RandomContent:
    def __init__(self, tokenizer):
        self.ids = tokenizer.ordinary

    def sample(self, rng, count, group):
        return rng.choices(self.ids, k=count)


class SeedContent:
    def __init__(self, db, directory):
        self.db = db
        db.read_parquet(str(directory / "seeds.parquet")).create_view("seeds")
        self.count = db.sql("SELECT count(*) FROM seeds").fetchone()[0]
        if not self.count:
            raise ValueError("seed source has no text segments")

    def sample(self, rng, count, group):
        result = []
        while len(result) < count:
            ids = self.db.execute("SELECT tokens FROM seeds WHERE ordinal=?", [rng.randrange(self.count)]).fetchone()[0]
            start = rng.randrange(len(ids))
            result.extend(ids[start:start + count - len(result)])
        return result


class ProfileContent:
    def __init__(self, db, directory, smoothing):
        if isinstance(smoothing, bool) or not isinstance(smoothing, (int, float)) or not 0 < smoothing < 1:
            raise ValueError("unigram_mix must be strictly between zero and one")
        self.db, self.smoothing = db, smoothing
        db.read_parquet(str(directory / "tokens.parquet")).create_view("tokens")
        db.execute("CREATE TABLE prefix_texts (id VARCHAR PRIMARY KEY, content VARCHAR)")

    def prefix_text(self, identifier):
        row = self.db.execute("SELECT content FROM prefix_texts WHERE id=?", [identifier]).fetchone()
        return row[0] if row else None

    def remember_prefix(self, identifier, text):
        self.db.execute("INSERT INTO prefix_texts VALUES (?,?)", [identifier, text])

    @lru_cache(maxsize=64)
    def distribution(self, group, previous):
        if previous is None:
            rows = self.db.execute('SELECT first,count FROM tokens WHERE "group"=? AND second=-1 ORDER BY first', [group]).fetchall()
        else:
            rows = self.db.execute('SELECT second,count FROM tokens WHERE "group"=? AND first=? AND second>=0 ORDER BY second', [group, previous]).fetchall()
        return ([row[0] for row in rows], list(accumulate(row[1] for row in rows)))

    def sample(self, rng, count, group):
        unigram = self.distribution(group, None)
        if not unigram[0]:
            raise ValueError("profile has no token distribution for the requested role/category")
        result = []
        for _ in range(count):
            distribution = unigram
            if result and rng.random() >= self.smoothing:
                transition = self.distribution(group, result[-1])
                if transition[0]:
                    distribution = transition
            result.extend(rng.choices(distribution[0], cum_weights=distribution[1], k=1))
        return result


def _profile_request(spec, content, tokenizer, seed, scenario, variant):
    request = deepcopy(spec["request"])
    index, prefix, suffix = _compose(request, scenario, variant)
    current_key, current_cached = None, False
    # Persist equivalence on disk: later snapshots must reuse the *final fitted*
    # text, not the pre-correction decode of the same RNG stream.
    for position, message in enumerate(request["messages"]):
        segment = message.get("content")
        if isinstance(segment, dict):
            key = segment["prefix_id"]
            text = content.prefix_text(key)
            if position == index:
                current_key, current_cached = key, text is not None
            if text is None:
                rng = random.Random(fingerprint([seed, key]))
                text = tokenizer.decode(content.sample(rng, segment["tokens"], segment["group"]))
                if position != index:
                    content.remember_prefix(key, text)
            message["content"] = text
    return request, index, prefix, suffix, current_key, current_cached


def _compose(request, scenario, variant):
    system = scenario.get("system", [])
    if not isinstance(system, list) or not all(isinstance(text, str) for text in system):
        raise ValueError("scenario.system must be an ordered list of text")
    if system:
        text = "\n".join(system)
        if request["messages"][0]["role"] == "system":
            request["messages"][0]["content"] += "\n" + text
        else:
            request["messages"].insert(0, {"role": "system", "content": text})
    indices = [index for index, message in enumerate(request["messages"]) if message["role"] == "user"]
    if not indices:
        raise ValueError("generation requires a current user message")
    index = scenario.get("user_index", indices[-1])
    if isinstance(index, bool) or not isinstance(index, int) or index not in indices:
        raise ValueError("user_index must select an existing user message after system composition")
    prefix, suffix = variant.get("prefix", ""), variant.get("suffix", "")
    if not isinstance(prefix, str) or not isinstance(suffix, str):
        raise ValueError("user prefix/suffix must be text")
    return index, prefix, suffix


def generate(recipe, tokenizer, directory, source_profile=None):
    if recipe.get("mode") in {"embedding", "rerank"}:
        from .text_workloads import generate_text_workload
        return generate_text_workload(recipe, tokenizer, directory)
    if recipe.get("mode") in {"string", "tokenid"}:
        from .synthetic import generate_synthetic
        return generate_synthetic(recipe, tokenizer, directory)
    context_tokens = positive(recipe["model"].get("context_tokens"), "context_tokens")
    mode = recipe.get("mode")
    if mode not in {"random", "seed", "profile"}:
        raise ValueError("mode must be random, seed or profile")
    unknown = set(recipe) - {"model", "mode", "sessions", "seed", "output_policy", "lengths", "scenarios", "source", "profile", "unigram_mix", "output_tokens"}
    if unknown:
        raise ValueError("unknown recipe fields or conflicting request parameters")
    count = positive(recipe.get("sessions"), "sessions")
    seed = random_seed(recipe.get("seed", 0))
    policy = recipe.get("output_policy", "fixed")
    if policy not in {"fixed", "natural"}:
        raise ValueError("output_policy must be fixed or natural")
    rng = random.Random(seed)
    with ExitStack() as scope:
        state = scope.enter_context(tempfile.TemporaryDirectory(prefix="dataset-content-"))
        db = scope.enter_context(duckdb.connect(str(Path(state) / "content.duckdb")))
        db.execute("SET memory_limit='512MB'")
        if mode == "random":
            content = RandomContent(tokenizer)
        else:
            if source_profile is None:
                raise ValueError("seed/profile generation requires a source")
            manifest = read_yaml(source_profile / "profile.yaml")
            if manifest.get("algorithm") != "joint-session-unigram-bigram-v1" or manifest.get("version") != 1 or manifest["tokenizer_fingerprint"] != tokenizer.fingerprint:
                raise ValueError("source profile algorithm/tokenizer/version does not match the recipe")
            content = SeedContent(db, source_profile) if mode == "seed" else ProfileContent(db, source_profile, recipe.get("unigram_mix", 0.05))
        def shared_prefix(category, prefix_group, size):
            shared_rng = random.Random(fingerprint([seed, category, prefix_group]))
            return tokenizer.decode(content.sample(shared_rng, size, category + ":user")) + "\n"

        cached_prefix = lru_cache(maxsize=32)(shared_prefix)
        structure_count = 0
        if mode == "profile":
            assert source_profile is not None
            db.read_parquet(str(source_profile / "structures.parquet")).create_view("structures")
            structure_row = db.sql("SELECT count(*) FROM structures").fetchone()
            assert structure_row is not None
            structure_count = structure_row[0]
            if recipe.get("lengths"):
                raise ValueError("profile mode samples complete empirical sessions; lengths cannot replace their joint structure")
        writer = scope.enter_context(RowsWriter(directory / "data.parquet", DATASET_SCHEMA))
        turn_count = 0
        for session_index in range(count):
            scenario = weighted(rng, recipe.get("scenarios", [{"name": "default"}]))
            if mode == "profile":
                if recipe.get("scenarios"):
                    category = scenario.get("name", "default")
                    query = "SELECT structure FROM structures WHERE json_extract_string(structure,'$[0].category')=?"
                    eligible_row = db.execute("SELECT count(*) FROM (" + query + ")", [category]).fetchone()
                    assert eligible_row is not None
                    eligible = eligible_row[0]
                    if not eligible:
                        raise ValueError("profile scenario name has no matching source sessions")
                    row = db.execute(query + " ORDER BY ordinal LIMIT 1 OFFSET ?", [category, rng.randrange(eligible)]).fetchone()
                else:
                    row = db.execute("SELECT structure FROM structures WHERE ordinal=?", [rng.randrange(structure_count)]).fetchone()
                assert row is not None
                specs = json.loads(row[0])
            else:
                specs = weighted(rng, recipe.get("lengths"))["turns"]
            if not isinstance(specs, list) or not specs:
                raise ValueError("each length alternative requires non-empty turns")
            history = deepcopy(scenario.get("history", []))
            session_id = f"session-{session_index:08d}"
            for turn_index, spec in enumerate(specs):
                category = str(spec.get("category", scenario.get("name", "default")))
                group = category + ":user"
                variant = weighted(rng, scenario.get("user", [{}]))
                prefix_key, prefix_cached = None, False
                if mode == "profile":
                    request, index, prefix, suffix, prefix_key, prefix_cached = _profile_request(spec, content, tokenizer, seed, scenario, variant)
                else:
                    request = {"messages": deepcopy(history) + [{"role": "user", "content": ""}]}
                    if scenario.get("tools"):
                        request["tools"] = deepcopy(scenario["tools"])
                    if scenario.get("tool_choice") is not None:
                        request["tool_choice"] = scenario["tool_choice"]
                    composition = scenario if turn_index == 0 else {**scenario, "system": []}
                    index, prefix, suffix = _compose(request, composition, variant)
                shared_tokens = scenario.get("shared_prefix_tokens", 0)
                if shared_tokens:
                    positive(shared_tokens, "shared_prefix_tokens")
                    groups = positive(scenario.get("prefix_groups", 1), "prefix_groups")
                    # Bound retained work to 32 prefixes of at most 4096 sampled IDs.
                    make_prefix = cached_prefix if shared_tokens <= 4096 else shared_prefix
                    prefix += make_prefix(category, session_index % groups, shared_tokens)
                output = spec.get("output_tokens")
                if recipe.get("output_tokens") is not None:
                    output = recipe["output_tokens"]
                positive(output, "output_tokens (required when historical request targets are missing)")
                # A current profile message uses the same deterministic stream as its prefix group.
                current_rng = rng
                if prefix_key is not None:
                    current_rng = random.Random(fingerprint([seed, prefix_key]))
                if prefix_cached:
                    actual = len(request_token_ids(tokenizer, request))
                    text = request["messages"][index]["content"]
                    if actual != spec["input_tokens"] or not text.startswith(prefix) or not text.endswith(suffix):
                        raise ValueError("shared profile message conflicts with exact length or prompt composition")
                    if actual + output > context_tokens:
                        raise ValueError("input plus output exceeds model context_tokens")
                else:
                    actual = fit_input(tokenizer, request, index, prefix, suffix, spec["input_tokens"], output,
                        lambda n: content.sample(current_rng, n, group), context_tokens=context_tokens)
                    if prefix_key is not None:
                        assert isinstance(content, ProfileContent)
                        content.remember_prefix(prefix_key, request["messages"][index]["content"])
                writer.append(dataset_row(session_id, turn_index, request, actual, output, category, policy, tokenizer))
                turn_count += 1
                if mode != "profile" and turn_index + 1 < len(specs):
                    # Offline recorded history, never the response produced by a future benchmark.
                    history = deepcopy(request["messages"])
                    history.append({"role": "assistant", "content": tokenizer.decode(content.sample(rng, output, group))})
    result = {
        "version": 1, "mode": mode, "seed": seed, "sessions": count, "turns": turn_count,
        "recipe_fingerprint": fingerprint(recipe), "tokenizer_fingerprint": tokenizer.fingerprint,
        "tokenizer": tokenizer.identity, "context_tokens": context_tokens, "output_policy": policy,
        "validation": {"exact_input_lengths": True, "context_limits": True, "tool_history": True,
                       "service_output_lengths": "not_run"},
        "privacy": "Sensitive generated data; semantic validity and anonymity are not guaranteed.",
    }
    write_yaml(directory / "manifest.yaml", result)
    return result
