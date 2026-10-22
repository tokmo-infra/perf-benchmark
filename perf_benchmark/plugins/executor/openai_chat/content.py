"""Chat content assembly shared by transport and offline workload inspection."""

import json
from copy import deepcopy


def attempt_message_segments(value):
    messages = []
    try:
        for turn in (*value.previous_turns, value.turn):
            messages.append(value.payloads.view(turn.request_payload_ref))
            if turn is value.turn:
                return tuple(messages)
            if value.turn.context_mode == "recorded":
                ref = turn.recorded_response_payload_ref
                if ref is None:
                    raise RuntimeError(f"recorded Turn payload is unavailable: {turn.id}")
                messages.append(value.payloads.view(ref))
            elif value.turn.context_mode == "live":
                response = value.live_history.get(turn.id)
                if response is not None:
                    messages.append(response.payload)
                elif turn.status not in {"failed", "cancelled"}:
                    raise RuntimeError(f"previous Turn payload is unavailable: {turn.id}")
            else:
                raise ValueError(f"unsupported context mode: {value.turn.context_mode}")
        raise ValueError(f"Turn is not part of Session: {value.turn.id}")
    except BaseException:
        for value in messages:
            if isinstance(value, memoryview):
                value.release()
        raise


def read_chat_request(session, turn, payloads):
    if turn.context_mode == "live":
        raise ValueError("offline datasets cannot reconstruct live history")
    if turn.context_mode == "snapshot":
        with payloads.view(turn.request_payload_ref) as value:
            return json.loads(bytes(value))
    segments = _domain_message_segments(session, turn, payloads)
    try:
        return {"messages": [json.loads(bytes(value)) for value in segments]}
    finally:
        for value in segments:
            if isinstance(value, memoryview):
                value.release()


def _domain_message_segments(session, current, payloads):
    messages = []
    try:
        for turn in session.turns:
            messages.append(payloads.view(turn.request_payload_ref))
            if turn is current:
                return tuple(messages)
            if current.context_mode != "recorded":
                raise ValueError(f"unsupported context mode: {current.context_mode}")
            ref = turn.recorded_response_payload_ref
            if ref is None:
                raise RuntimeError(f"recorded Turn payload is unavailable: {turn.id}")
            messages.append(payloads.view(ref))
        raise ValueError(f"Turn is not part of Session: {current.id}")
    except BaseException:
        for value in messages:
            if isinstance(value, memoryview):
                value.release()
        raise


def template_messages(request):
    """OpenAI wire arguments are JSON strings; HF chat templates expect objects."""
    messages = deepcopy(request["messages"])
    for message in messages:
        for call in message.get("tool_calls") or []:
            call["function"]["arguments"] = json.loads(call["function"]["arguments"])
    return messages
