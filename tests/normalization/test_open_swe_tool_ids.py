from copy import deepcopy

import pytest

from perf_benchmark.plugins.normalization.open_swe_traces import (
    OpenSWETracesNormalization, parse_settings,
)


def trajectory():
    return {
        "benchmark_session_id": "recorded",
        "benchmark_input_tokens": [10, 30, 50],
        "benchmark_output_tokens": [2, 3, 4],
        "messages": [
            {"role": "user", "content": "inspect"},
            {"role": "assistant", "reasoning_content": "two independent calls",
             "tool_calls": [
                 {"id": "call-a", "type": "function",
                  "function": {"name": "bash", "arguments": '{"command":"a"}'}},
                 {"id": "call-b", "type": "function",
                  "function": {"name": "bash", "arguments": '{"command":"b"}'}},
             ]},
            {"role": "tool", "content": "result-b"},
            {"role": "tool", "content": "result-a", "tool_call_id": None},
            {"role": "assistant", "content": "next"},
            {"role": "tool", "content": "explicit", "tool_call_id": "kept-id"},
            {"role": "assistant", "content": "done"},
        ],
    }


def normalize(record, **settings):
    return list(OpenSWETracesNormalization(parse_settings(settings)).normalize_record(record, 0))


def test_empty_policy_is_explicit_non_mutating_and_does_not_guess_parallel_call_ids():
    record = trajectory()
    original = deepcopy(record)
    drafts = normalize(record, missing_tool_call_id="empty")
    assert record == original
    messages = drafts[-1].request["messages"]
    assert [m["tool_call_id"] for m in messages if m["role"] == "tool"] == ["", "", "kept-id"]
    assert messages[1] == original["messages"][1]
    assert [m["content"] for m in messages if m["role"] == "tool"] == [
        "result-b", "result-a", "explicit",
    ]
    assert [d.metadata["tool_call_id_placeholders"] for d in drafts] == [0, 2, 2]
    assert all(d.metadata["tool_call_id_policy"] == "empty" for d in drafts)
    assert [d.estimated_input_tokens.value for d in drafts] == [10, 30, 50]
    assert [d.reference_output_tokens.value for d in drafts] == [2, 3, 4]


def test_default_preserves_missing_and_existing_ids():
    drafts = normalize(trajectory())
    tools = [m for m in drafts[-1].request["messages"] if m["role"] == "tool"]
    assert "tool_call_id" not in tools[0] and "tool_call_id" not in tools[1]
    assert tools[2]["tool_call_id"] == "kept-id"
    assert all(d.metadata["tool_call_id_placeholders"] == 0 for d in drafts)


@pytest.mark.parametrize("value", [None, True, 1, "guess", {}, []])
def test_invalid_tool_id_policy_rejected(value):
    with pytest.raises(ValueError, match="missing_tool_call_id"):
        parse_settings({"missing_tool_call_id": value})


def test_non_string_existing_tool_id_is_not_silently_rewritten():
    record = trajectory()
    record["messages"][2]["tool_call_id"] = 42
    with pytest.raises(TypeError, match="tool_call_id"):
        normalize(record, missing_tool_call_id="empty")


def test_no_tokenizer_or_unknown_settings_allowed():
    with pytest.raises(ValueError, match="dataset token counts"):
        parse_settings({"tokenizer_path": "unused"})
    with pytest.raises(ValueError, match="unknown"):
        parse_settings({"guess_tool_ids": True})


def single_trajectory():
    record = trajectory()
    record["messages"][1]["tool_calls"] = record["messages"][1]["tool_calls"][:1]
    record["messages"].pop(3)
    record["messages"][3]["tool_calls"] = [{
        "id": "call-c", "type": "function",
        "function": {"name": "bash", "arguments": "{}"},
    }]
    record["messages"][4].pop("tool_call_id")
    return record


def test_match_single_repairs_only_unambiguous_real_ids_and_preserves_source():
    record = single_trajectory()
    original = deepcopy(record)
    drafts = normalize(record, missing_tool_call_id="match-single")
    assert record == original
    tools = [m for m in drafts[-1].request["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tools] == ["call-a", "call-c"]
    assert [m["content"] for m in tools] == ["result-b", "explicit"]
    assert [d.metadata["tool_call_id_repairs"] for d in drafts] == [0, 1, 2]
    assert all(d.metadata["tool_call_id_placeholders"] == 0 for d in drafts)
    assert [d.estimated_input_tokens.value for d in drafts] == [10, 30, 50]
    assert [d.reference_output_tokens.value for d in drafts] == [2, 3, 4]
    assert drafts[-1].request["messages"][1] == original["messages"][1]


@pytest.mark.parametrize("gap", [None, ""] )
def test_match_single_handles_null_or_empty_result_id(gap):
    record = single_trajectory()
    record["messages"][2]["tool_call_id"] = gap
    assert normalize(record, missing_tool_call_id="match-single")[-1].request["messages"][2]["tool_call_id"] == "call-a"


def test_match_single_uses_explicit_parallel_ids_before_unique_remaining_id():
    record = trajectory()
    record["messages"] = record["messages"][:5]
    record["benchmark_input_tokens"] = [10, 30]
    record["benchmark_output_tokens"] = [2, 3]
    record["messages"][2]["tool_call_id"] = "call-b"
    draft = normalize(record, missing_tool_call_id="match-single")[-1]
    assert [m["tool_call_id"] for m in draft.request["messages"] if m["role"] == "tool"] == ["call-b", "call-a"]
    assert draft.metadata["tool_call_id_repairs"] == 1


def test_match_single_rejects_ambiguous_parallel_results():
    with pytest.raises(ValueError, match="exactly one pending"):
        normalize(trajectory(), missing_tool_call_id="match-single")


@pytest.mark.parametrize("identifier", [None, "", 1])
def test_match_single_requires_real_assistant_call_ids(identifier):
    record = single_trajectory()
    record["messages"][1]["tool_calls"][0]["id"] = identifier
    with pytest.raises(ValueError, match="nonempty assistant"):
        normalize(record, missing_tool_call_id="match-single")


@pytest.mark.parametrize("identifier", ["different", 42])
def test_match_single_preserves_explicit_result_identity_by_rejecting_invalid_values(identifier):
    record = single_trajectory()
    record["messages"][2]["tool_call_id"] = identifier
    with pytest.raises((ValueError, TypeError), match="tool_call_id"):
        normalize(record, missing_tool_call_id="match-single")


def test_match_single_rejects_duplicate_calls_and_interrupted_association():
    record = single_trajectory()
    call = record["messages"][1]["tool_calls"][0]
    record["messages"][1]["tool_calls"].append(deepcopy(call))
    with pytest.raises(ValueError, match="duplicate"):
        normalize(record, missing_tool_call_id="match-single")
    record = single_trajectory()
    record["messages"].pop(2)
    with pytest.raises(ValueError, match="before all"):
        normalize(record, missing_tool_call_id="match-single")
