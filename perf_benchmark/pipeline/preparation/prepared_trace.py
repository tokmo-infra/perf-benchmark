"""Decode an offline prepared trace row into a request plan."""

from datetime import datetime
import json

from perf_benchmark.core.causal import CausalNode, Dependency
from perf_benchmark.pipeline.contracts.normalization import FixedTokens, TurnDraft


def prepared_turn(record, source):
    if record.get("trace_source") != source or record.get("artifact_version") != 1:
        raise ValueError(f"{source} requires its version 1 prepared trace artifact")
    if not isinstance(record.get("request_json"), str):
        raise ValueError("prepared request_json is missing")
    request = json.loads(record["request_json"])
    metadata = json.loads(record["metadata_json"])
    if not isinstance(request, dict) or not isinstance(metadata, dict):
        raise ValueError("prepared request and metadata must be objects")
    input_tokens, output_tokens = record["input_tokens"], record["output_tokens"]
    if type(input_tokens) is not int or input_tokens <= 0 or type(output_tokens) is not int or output_tokens <= 0:
        raise ValueError("prepared token lengths must be positive integers")
    kind = record["input_kind"]
    if kind == "completion":
        if not isinstance(request.get("prompt"), str) or not request["prompt"]:
            raise ValueError("prepared completion prompt is missing")
    elif kind == "chat":
        if not isinstance(request.get("messages"), list) or not request["messages"]:
            raise ValueError("prepared chat messages are missing")
    elif kind == "responses":
        if not isinstance(request.get("input"), list) or not request["input"]:
            raise ValueError("prepared Responses input is missing")
    else:
        raise ValueError("unsupported prepared trace request kind")
    created = datetime.fromisoformat(record["created_at"].replace("Z", "+00:00"))
    if created.tzinfo is None:
        raise ValueError("prepared timestamp must include timezone")
    causal = None
    if record.get("causal_json"):
        node = json.loads(record["causal_json"])
        causal = CausalNode(node["graph_id"], node["node_id"], node["branch_id"],
                            node["root_ns"], node["not_before_ns"],
                            tuple(Dependency(edge["node_id"], edge["event"], edge.get("delay_ns"))
                                  for edge in node["dependencies"]))
    return TurnDraft(
        record["session_id"], record["turn_id"], record["turn_order_key"], request,
        FixedTokens(input_tokens), FixedTokens(output_tokens),
        input_kind=kind, created_at=created, metadata=metadata,
        requested_output_tokens=output_tokens,
        source_session_id=record.get("source_session_id"), causal=causal,
    )
