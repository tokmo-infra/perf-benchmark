import asyncio
from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
from filelock import AsyncFileLock
import pyarrow as pa

from perf_benchmark.pipeline.resources import finish_task
from perf_benchmark.core.tokenization import ModelTokenizer
from perf_benchmark.core.validation import random_seed

from .common import RowsWriter, local_path, positive, publish_directory, read_yaml, write_yaml
from .cache import cache_directory, generation_key, mark, materialize, valid
from .compare import compare
from .generation import DATASET_SCHEMA, dataset_row, generate
from .profile import analyze
from .source import analyze_source
from .requests import request_token_ids


async def threaded(function, *args):
    return await finish_task(asyncio.create_task(asyncio.to_thread(function, *args)))


def prepared_source(path, *, normalization="openai-messages"):
    return {
        "loader": {"plugin": "hf-datasets", "settings": {"dataset": "parquet", "split": "train", "data_files": str(path / "data.parquet")}},
        "normalization": {"plugin": normalization, "settings": {"multi_turn": True} if normalization == "openai-messages" else {}},
    }


def source_config(value, model, base):
    config = deepcopy(value)
    config["normalization"].setdefault("settings", {}).setdefault("tokenizer_path", str(local_path(base, model["path"])))
    return config


def stats(sessions, tokenizer, destination, baseline, context_tokens):
    if baseline is None:
        return analyze(sessions, tokenizer, destination)
    policy = baseline.get("output_policy")
    if policy not in {"fixed", "natural"}:
        raise ValueError("baseline export requires explicit output_policy: fixed or natural")
    path = destination / "baseline"
    path.mkdir()
    with RowsWriter(path / "data.parquet", DATASET_SCHEMA.append(pa.field("output_tokens", pa.int64()))) as writer:
        def counted():
            for session_index, (_, turns) in enumerate(sessions):
                identifier = f"session-{session_index:08d}"
                def requests():
                    for index, (turn, request) in enumerate(turns):
                        actual = len(request_token_ids(tokenizer, request))
                        output = baseline.get("output_tokens", turn.requested_output_tokens)
                        positive(output, "baseline output_tokens")
                        if actual + output > context_tokens:
                            raise ValueError("baseline input plus output exceeds context_tokens")
                        turn.requested_output_tokens = output
                        row = dataset_row(identifier, index, request, actual, output, str(turn.metadata.get("dataset_category", "default")), policy, tokenizer)
                        writer.append({**row, "output_tokens": turn.reference_output_tokens})
                        yield turn, request
                yield identifier, requests()
        result = analyze(counted(), tokenizer, destination)
    write_yaml(path / "manifest.yaml", {"version": 1, "mode": "original-snapshots", "tokenizer_fingerprint": tokenizer.fingerprint,
                                         "output_policy": policy, "service_output_lengths": "not_run"})
    return result


async def execute(command, config_path, output=None):
    if command not in {"stats", "generate", "compare", "prepare-trace"}:
        raise ValueError("unknown dataset command")
    config_path = Path(config_path).resolve()
    config, base = read_yaml(config_path), config_path.parent
    if command == "prepare-trace":
        from .trace_prepare import prepare_trace
        return await threaded(prepare_trace, config, base, output)
    if command == "compare":
        return await threaded(compare, config, base)
    if command == "generate":
        config["seed"] = random_seed(config.get("seed", 0))
    model = config["model"]
    context_tokens = positive(model.get("context_tokens"), "context_tokens")
    tokenizer = await threaded(lambda: ModelTokenizer(
        model.get("path"), base_dir=base, template_options=model.get("template_options"),
    ))
    if command == "generate":
        key = await threaded(generation_key, config, tokenizer, base)
        cached = cache_directory(key)
        cached.parent.mkdir(parents=True, exist_ok=True)
        async with AsyncFileLock(str(cached) + ".lock"):
            cache_hit = await threaded(valid, cached, key)
            if not cache_hit:
                if cached.is_symlink():
                    raise ValueError("generation cache path must not be a symlink")
                if cached.exists():
                    await threaded(shutil.rmtree, cached)
                with publish_directory(cached) as destination:
                    with tempfile.TemporaryDirectory(prefix="dataset-seeds-") as temporary:
                        profile = None
                        if config.get("mode") == "seed":
                            profile = Path(temporary)
                            source = source_config(config["source"], config["model"], base)
                            await analyze_source(source, base, lambda sessions: analyze(sessions, tokenizer, profile, keep_seeds=True))
                        elif config.get("mode") == "profile":
                            profile = local_path(base, config["profile"])
                        await threaded(generate, config, tokenizer, destination, profile)
                        normalization = config["mode"] if config["mode"] in {"embedding", "rerank"} else "openai-messages"
                        if normalization == "openai-messages":
                            await analyze_source(prepared_source(destination), base, lambda sessions: analyze(sessions, tokenizer, destination / "profile"))
                        binding = prepared_source(Path("."), normalization=normalization)
                        if normalization != "openai-messages":
                            binding["normalization"]["settings"]["tokenizer_path"] = str(tokenizer.path)
                        binding["selector"] = {"plugin": "all", "settings": {}}
                        write_yaml(destination / "input.yaml", binding)
                        if await threaded(generation_key, config, tokenizer, base) != key:
                            raise ValueError("generation source changed while preparing the corpus")
                        await threaded(mark, destination, key)
        await threaded(materialize, cached, output, key)
        return {**read_yaml(cached / "manifest.yaml"), "cache_hit": cache_hit}
    with publish_directory(output) as destination:
        source = source_config(config["source"], config["model"], base)
        return await analyze_source(source, base, lambda sessions: stats(sessions, tokenizer, destination, config.get("baseline"), context_tokens))
