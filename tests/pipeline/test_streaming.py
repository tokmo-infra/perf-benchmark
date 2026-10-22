import json
from types import MappingProxyType

import pyarrow as pa
import pytest

from perf_benchmark.core.models import Run, Trial
from perf_benchmark.pipeline.contracts.loader import LoaderPlugin
from perf_benchmark.pipeline.contracts.normalization import NormalizationPlugin
from perf_benchmark.pipeline.contracts.selector import SelectorPlugin, SelectionResult
from perf_benchmark.pipeline.preparation.payloads import BlobStore, PayloadRef
from perf_benchmark.pipeline.registry import PluginSpec, StageBinding
from perf_benchmark.pipeline.preparation.streaming import (
    DatasetExhausted,
    PreparedCorpus,
    SELECTED_SCHEMA,
    _decode_session_plans,
    _select_table,
)


def _row(
    session_id,
    turn_id,
    session_ordinal,
    turn_ordinal,
    ref,
    *,
    metadata=None,
    extra_headers_ref=None,
):
    return {
        "source_order": turn_ordinal,
        "session_ordinal": session_ordinal,
        "session_id": session_id,
        "turn_id": turn_id,
        "turn_order_key": turn_ordinal,
        "turn_ordinal": turn_ordinal,
        "input_kind": "chat",
        "context_mode": "snapshot",
        "request_payload_offset": ref.offset,
        "request_payload_bytes": ref.byte_length,
        "extra_headers_payload_offset": extra_headers_ref.offset if extra_headers_ref else None,
        "extra_headers_payload_bytes": extra_headers_ref.byte_length if extra_headers_ref else None,
        "recorded_response_payload_offset": None,
        "recorded_response_payload_bytes": None,
        "estimated_input_tokens": 1,
        "reference_output_tokens": 1,
        "created_at": None,
        "metadata_json": json.dumps(metadata or {}),
    }


async def test_prepared_corpus_replays_selected_sessions_and_owns_payloads() -> None:
    payloads = BlobStore()
    first = payloads.append(b"{}")
    second = payloads.append(b"[]")
    payloads.seal()
    selected = pa.RecordBatch.from_pylist(
        [
            _row("source-session", "turn-0", 0, 0, first),
            _row("source-session", "turn-1", 0, 1, second),
        ],
        schema=SELECTED_SCHEMA,
    )
    corpus = PreparedCorpus(payloads, _decode_session_plans((selected,)))

    first_pass = corpus.materialize_sessions(Trial(id="trial-a"))
    second_pass = corpus.materialize_sessions(Trial(id="trial-b"))

    assert corpus.session_count == 1
    assert [turn.ordinal for turn in first_pass[0].turns] == [0, 1]
    assert first_pass[0].turns[0].request_payload_ref == PayloadRef(0, 2)
    assert second_pass[0].id == "trial-b:session-0"
    assert bytes(corpus.payloads.view(first)) == b"{}"

    await corpus.close()
    with pytest.raises(RuntimeError, match="closed"):
        corpus.payloads.view(first)


async def test_session_stream_cycles_plans_without_copying_payloads_and_error_preflights() -> None:
    payloads = BlobStore()
    ref = payloads.append(b"payload")
    payloads.seal()
    selected = pa.RecordBatch.from_pylist(
        [_row("source", "turn", 0, 0, ref)], schema=SELECTED_SCHEMA
    )
    corpus = PreparedCorpus(payloads, _decode_session_plans((selected,)))
    trial = Trial(id="trial")

    sessions = [
        session
        async for session in corpus.session_stream(
            trial, session_count=3, on_exhausted="cycle"
        )
    ]
    assert [session.id for session in sessions] == [
        "trial:session-0", "trial:session-1", "trial:session-2"
    ]
    assert [session.metadata["cycle"] for session in sessions] == [0, 1, 2]
    assert len({turn.id for session in sessions for turn in session.turns}) == 3
    assert all(session.turns[0].request_payload_ref == ref for session in sessions)

    with pytest.raises(DatasetExhausted, match="dataset has 1 Sessions"):
        corpus.session_stream(trial, session_count=2, on_exhausted="error")
    await corpus.close()


async def test_selected_header_refs_survive_multiturn_trial_and_cycle_replays():
    from perf_benchmark.pipeline.contracts.executor import AttemptInput

    payloads = BlobStore()
    body = payloads.append(b"{}")
    refs = [payloads.append(json.dumps({"X-Turn": value}).encode())
            for value in ("PRIVATE_HEADER_A", "PRIVATE_HEADER_B")]
    payloads.seal()
    selected = pa.RecordBatch.from_pylist([
        _row("source", "turn-0", 0, 0, body, extra_headers_ref=refs[0]),
        _row("source", "turn-1", 0, 1, body, extra_headers_ref=refs[1]),
        _row("source", "turn-2", 0, 2, body),
    ], schema=SELECTED_SCHEMA)
    corpus = PreparedCorpus(payloads, _decode_session_plans((selected,)))
    try:
        replays = list(corpus.materialize_sessions(Trial(id="trial-first")))
        replays += [session async for session in corpus.session_stream(
            Trial(id="trial-second"), session_count=2, on_exhausted="cycle")]
        assert len(replays) == 3
        for session in replays:
            assert [turn.extra_headers_payload_ref for turn in session.turns] == [*refs, None]
            for turn in session.turns:
                attempt = AttemptInput.capture(session, turn, {}, attempt=1, payloads=payloads)
                assert attempt.turn.extra_headers_payload_ref == turn.extra_headers_payload_ref
            assert all(secret not in str(session)
                       for secret in ("PRIVATE_HEADER_A", "PRIVATE_HEADER_B"))
    finally:
        await corpus.close()


@pytest.mark.parametrize("session_count", [None, 64])
async def test_finite_session_pool_stops_without_replaying_prefixes(session_count) -> None:
    payloads = BlobStore()
    ref = payloads.append(b"payload")
    payloads.seal()
    selected = pa.RecordBatch.from_pylist(
        [_row(f"source-{i}", f"turn-{i}", i, 0, ref) for i in range(32)],
        schema=SELECTED_SCHEMA,
    )
    corpus = PreparedCorpus(payloads, _decode_session_plans((selected,)))
    try:
        sessions = [session async for session in corpus.session_stream(
            Trial(id="trial"), session_count=session_count, on_exhausted="stop")]
        assert [session.ordinal for session in sessions] == list(range(32))
        assert len({session.id for session in sessions}) == 32
        assert all("cycle" not in session.metadata for session in sessions)
        assert all(session.turns[0].request_payload_ref == ref for session in sessions)
    finally:
        await corpus.close()


async def test_selector_host_rejects_non_contiguous_selection() -> None:
    payloads = BlobStore()
    ref = payloads.append(b"x")
    payloads.seal()
    selected = pa.RecordBatch.from_pylist(
        [
            _row("a", "a-0", 0, 0, ref),
            _row("b", "b-0", 1, 0, ref),
            _row("a", "a-1", 2, 1, ref),
        ],
        schema=SELECTED_SCHEMA,
    )
    with pytest.raises(ValueError, match="not contiguous"):
        class Selector(SelectorPlugin):
            def select(self, table):
                return SelectionResult(pa.array([0, 1, 2]))
        plugin = Selector()
        _select_table(plugin, pa.Table.from_batches([selected]).drop(["session_ordinal", "turn_ordinal"]))
    payloads.close()


def test_materialized_trials_deep_copy_nested_metadata(monkeypatch) -> None:
    payloads = BlobStore()
    ref = payloads.append(b"{}")
    payloads.seal()
    selected = pa.RecordBatch.from_pylist(
        [
            _row(
                "source-session",
                "turn-0",
                0,
                0,
                ref,
                metadata={"nested": {"labels": ["original"]}},
            ),
        ],
        schema=SELECTED_SCHEMA,
    )
    plans = _decode_session_plans((selected,))
    corpus = PreparedCorpus(payloads, plans)

    def unexpected_parse(value):
        raise AssertionError(f"metadata reparsed during Trial execution: {value!r}")

    monkeypatch.setattr(
        "perf_benchmark.pipeline.preparation.streaming.json.loads",
        unexpected_parse,
    )
    first = corpus.materialize_sessions(Trial(id="trial-a"))
    second = corpus.materialize_sessions(Trial(id="trial-b"))

    first_metadata = first[0].turns[0].metadata
    second_metadata = second[0].turns[0].metadata
    nested = first_metadata["nested"]
    assert isinstance(nested, dict) and isinstance(nested["labels"], list)
    nested["labels"].append("mutated")
    assert second_metadata["nested"] == {"labels": ["original"]}
    assert first[0].id == "trial-a:session-0"
    assert second[0].id == "trial-b:session-0"
    assert first[0].turns[0] is not second[0].turns[0]
    payloads.close()


async def test_preparation_failure_closes_blob_store(run_context) -> None:
    run = Run(name="failed-preparation")
    stage = run_context("loader", run=run)

    class Lifecycle:
        async def start(self, stage):
            pass

        async def finish(self, *stages, **data):
            pass

    class Source:
        schema = pa.schema([pa.field("value", pa.string())])

        def batches(self):
            yield pa.record_batch([["bad"]], schema=self.schema)

        async def close(self):
            pass

    class Loader(LoaderPlugin):
        async def open(self, ctx):
            return Source()

    class Normalizer(NormalizationPlugin):
        def normalize_batch(self, batch):
            raise ValueError("bad record")

    class Selector(SelectorPlugin):
        def select(self, table):
            raise AssertionError("selector must not run")

    def binding(name, plugin):
        spec = PluginSpec(name, name, lambda value: plugin, "fixture", lambda value: value)
        return StageBinding(spec, MappingProxyType({}), plugin)

    stage.bindings = MappingProxyType(
        {
            "loader": binding("loader", Loader()),
            "normalization": binding("normalization", Normalizer()),
            "selector": binding("selector", Selector()),
        }
    )

    corpus = PreparedCorpus(stage.payloads)
    try:
        with pytest.raises(ValueError, match="bad record"):
            await corpus.prepare(
                run,
                stage,
                Lifecycle(),
            )
    finally:
        await corpus.close()
    with pytest.raises(RuntimeError, match="closed"):
        stage.payloads.append(b"late")


@pytest.mark.parametrize("chars,expected", [(None, None), (0, 0)])
def test_partial_request_profile_does_not_turn_unknown_into_zero(tmp_path, chars, expected):
    from perf_benchmark.core.profiles import RequestProfile
    from perf_benchmark.pipeline.contracts.normalization import TurnDraft
    from perf_benchmark.pipeline.preparation.streaming import _resolve_draft_tokens, _TokenizerState

    draft = TurnDraft("s", "t", 0, {}, 1, None, profile=RequestProfile(
        message_count=1, system_chars=chars, developer_chars=chars,
    ))
    resolved = next(_resolve_draft_tokens([draft], _TokenizerState(None, tmp_path)))
    assert resolved.profile is not None
    assert resolved.profile.system_tokens_estimate is expected
    assert resolved.profile.developer_tokens_estimate is expected
    assert resolved.profile.tokenizer_fingerprint is None
