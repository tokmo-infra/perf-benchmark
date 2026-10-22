from pathlib import Path
from types import MappingProxyType

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from perf_benchmark.core.models import Run, Trial
from perf_benchmark.pipeline.hooks import HookContext, HookRegistry
from perf_benchmark.runtime.events import EventDispatcher
from perf_benchmark.pipeline.context import RunContext
from perf_benchmark.pipeline.registry import PluginSpec, StageBinding
from perf_benchmark.pipeline.preparation.payloads import BlobStore
from perf_benchmark.runtime.cancellation import CancellationToken


@pytest.fixture
def tokenizer_path(tmp_path: Path) -> Path:
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    path = tmp_path / "tokenizer"
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]").save_pretrained(path)
    return path


@pytest.fixture
def run_context(tmp_path: Path):
    stores = []

    def create(stage: str, settings: dict | None = None, run: Run | None = None) -> RunContext:
        domain_run = run or Run(name="contract")
        dispatcher = EventDispatcher(
            domain_run,
            HookRegistry(),
            {},
            HookContext(domain_run.id, domain_run.name, tmp_path),
        )
        parsed = MappingProxyType(dict(settings or {}))
        spec = PluginSpec(
            "fixture",
            stage,
            lambda value: object(),
            "fixture",
            lambda value: value,
        )
        bindings = MappingProxyType({stage: StageBinding(spec, parsed, object())})
        context = RunContext(
            bindings,
            tmp_path,
            dispatcher,
            CancellationToken(),
            BlobStore(),
        )
        stores.append(context.payloads)
        return context

    yield create
    for store in stores:
        store.close()


@pytest.fixture
def stage_context(run_context):

    def create(stage: str, settings: dict | None = None, run: Run | None = None):
        context = run_context(stage, settings, run)
        from perf_benchmark.pipeline import context as contexts
        cls = getattr(contexts, stage.title() + "Context")
        from dataclasses import fields
        return cls(**{field.name: getattr(context, field.name) for field in fields(cls) if hasattr(context, field.name)})

    return create


@pytest.fixture
def trial_context(run_context):
    def create(stage, settings=None, run=None):
        domain_run = run or Run(name="contract")
        trial = Trial(id="trial")
        domain_run.register_trial(trial)
        return run_context(stage, settings, domain_run).for_trial(domain_run, trial)

    return create
