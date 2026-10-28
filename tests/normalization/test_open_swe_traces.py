from copy import deepcopy
import json
import pytest

from perf_benchmark.plugins.normalization.open_swe_traces import (
    OpenSWETracesNormalization, normalize_tools, parse_settings,
)
from perf_benchmark.core.tokenization import ModelTokenizer


@pytest.fixture
def prepared_plugin(stage_context, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("recorded SWE replay must not initialize a tokenizer")
    monkeypatch.setattr(ModelTokenizer, "__init__", forbidden)
    def create():
        plugin = OpenSWETracesNormalization(parse_settings({}))
        plugin.prepare(stage_context("normalization"))
        return plugin
    return create


def test_preserves_recorded_tools_reasoning_and_all_turns(prepared_plugin):
    record = {
        "benchmark_session_id": "swe-7", "trajectory_id": "source-7",
        "messages": [
            {"role": "system", "content": "system", "tool_calls": None},
            {"role": "user", "content": "fix"},
            {"role": "assistant", "content": "", "reasoning_content": "think",
             "tool_calls": [{"function": {"name": "bash", "arguments": "{}"},
                             "id": "a", "type": "function"}]},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": "next"},
        ],
        "tools": ['{"type":"function","function":{"name":"bash"}}'],
        "benchmark_input_tokens": [30, 100], "benchmark_output_tokens": [10, 20],
    }
    original = deepcopy(record)
    plugin = prepared_plugin()
    turns = list(plugin.normalize_record(record, 0))
    assert record == original
    assert len(turns) == 2
    assert turns[1].request["messages"][2]["reasoning_content"] == "think"
    assert turns[1].request["messages"][3] == record["messages"][3]
    assert turns[1].request["tools"][0]["function"]["name"] == "bash"
    assert turns[1].metadata["dataset_input_tokens"] == 100
    assert turns[1].estimated_input_tokens.value == 100
    assert turns[1].metadata["input_token_count_source"] == "dataset_reference"
    assert turns[1].requested_output_tokens is None
    assert turns[1].reference_output_tokens.value == 20
    assert turns[1].request["messages"][2]["tool_calls"][0]["function"]["arguments"] == "{}"
    assert turns[1].metadata["dataset_output_policy"] == "natural"
    assert turns[0].request["messages"] == [
        {key: value for key, value in message.items() if value is not None}
        for message in original["messages"][:2]
    ]


@pytest.mark.parametrize("as_json", [False, True])
def test_anthropic_tool_schemas_use_openai_function_wrapper(prepared_plugin, as_json):
    schema = {"type": "object", "properties": {"command": {"type": "string"}},
              "required": ["command"], "additionalProperties": False}
    tool = {"name": "Bash", "description": "Execute a command", "input_schema": schema}
    record = {"benchmark_session_id": "anthropic-swe", "messages": [
        {"role": "user", "content": "fix"},
        {"role": "assistant", "content": "answer"}],
        "tools": [json.dumps(tool) if as_json else tool],
        "benchmark_input_tokens": [123], "benchmark_output_tokens": [45]}
    original = deepcopy(record)
    turn = next(iter(prepared_plugin().normalize_record(record, 0)))
    assert turn.request["tools"] == [{"type": "function", "function": {
        "name": "Bash", "description": "Execute a command", "parameters": schema}}]
    assert turn.estimated_input_tokens.value == 123
    assert turn.reference_output_tokens.value == 45
    assert record == original


def test_mixed_tool_formats_preserve_openai_fields_and_nested_schema():
    schema = {"type": "object", "properties": {"items": {
        "type": "array", "items": {"type": "object", "additionalProperties": False}}}}
    tools = [{"type": "function", "function": {
        "name": "existing", "strict": True, "parameters": schema}},
        {"name": "converted", "input_schema": schema}]
    original = deepcopy(tools)
    assert normalize_tools(tools) == [original[0], {"type": "function", "function": {
        "name": "converted", "parameters": schema}}]
    assert tools == original


@pytest.mark.parametrize("tool, error", [
    ([], "tools must contain objects"),
    ({"input_schema": {}}, "nonempty name"),
    ({"name": "", "input_schema": {}}, "nonempty name"),
    ({"name": "bash", "input_schema": []}, "input_schema must be an object"),
])
def test_invalid_tool_schemas_report_source_error(tool, error):
    with pytest.raises(ValueError, match=error):
        normalize_tools([tool])


def test_rejects_unknown_settings_and_mismatched_counts(prepared_plugin):
    with pytest.raises(ValueError):
        parse_settings({"context_length": 100, "live": True})
    record = {"benchmark_session_id": "s", "messages": [{"role": "user", "content": "x"},
              {"role": "assistant", "content": "y"}], "benchmark_input_tokens": [],
              "benchmark_output_tokens": []}
    with pytest.raises(ValueError, match="match every assistant"):
        list(prepared_plugin().normalize_record(record, 0))


def test_full_history_is_replayed_without_local_length_filtering(
    prepared_plugin, monkeypatch,
):
    dataset_tokens = 2000000
    def forbidden(*args, **kwargs):
        raise AssertionError("must not tokenize a request before sending")
    monkeypatch.setattr(ModelTokenizer, "encode_chat", forbidden)
    record = {"benchmark_session_id": "s", "messages": [
        {"role": "user", "content": "fix"},
        {"role": "assistant", "content": "", "reasoning_content": "think",
         "tool_calls": [{"function": {"name": "bash", "arguments": '{"x":1}'}}]},
        {"role": "tool", "content": "result"},
        {"role": "assistant", "content": "answer"}],
        "tools": [{"type": "function", "function": {"name": "bash"}}],
        "benchmark_input_tokens": [30, dataset_tokens], "benchmark_output_tokens": [10, 20]}
    original = deepcopy(record)
    turns = list(prepared_plugin().normalize_record(record, 0))
    assert turns[-1].requested_output_tokens is None
    assert turns[-1].estimated_input_tokens.value == dataset_tokens
    assert turns[-1].request["messages"] == record["messages"][:-1]
    assert turns[-1].request["tools"] == record["tools"]
    assert len(turns) == 2
    assert record == original


@pytest.mark.parametrize("settings", [
    {"context_length": 1048576},
    {"tokenizer_path": "model"},
])
def test_rejects_retired_tokenizer_and_context_settings(settings):
    with pytest.raises((ValueError, TypeError)):
        parse_settings(settings)


def test_explicit_output_limit_is_preserved_without_context_checks(prepared_plugin):
    record = {"benchmark_session_id": "s", "messages": [
        {"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
        "benchmark_input_tokens": [1], "benchmark_output_tokens": [2]}
    turn = next(iter(prepared_plugin().normalize_record({**record, "requested_output_tokens": 2000000}, 0)))
    assert turn.requested_output_tokens == 2000000
    assert next(iter(prepared_plugin().normalize_record(record, 0))).requested_output_tokens is None
    for invalid in (0, -1, True):
        with pytest.raises((TypeError, ValueError)):
            list(prepared_plugin().normalize_record({**record, "requested_output_tokens": invalid}, 0))
