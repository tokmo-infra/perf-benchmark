#[path = "../src/decoder.rs"]
mod decoder;
use perf_rusthttp::Command;
#[test]
fn rejects_invalid_protocol_inputs() {
    let mut command = Command::default();
    command.expectations.insert("count".into(), 2);
    command.expectations.insert("top_n".into(), 2);
    for body in [
        r#"{"type":"response.failed","response":{"status":"completed"}}"#,
        r#"{"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":-1}}}"#,
        r#"{"type":"response.output_text.delta"}"#,
        r#"{"type":"response.reasoning_part.unknown"}"#,
        r#"{"type":"unknown"}"#,
        r#"null"#,
    ] {
        let mut decoder = decoder::create(&command);
        let error = decoder
            .feed(body.as_bytes(), true)
            .err()
            .or_else(|| decoder.finish("succeeded").1);
        assert!(error.is_some(), "accepted {body}");
    }
}
#[test]
fn partial_cancellation_does_not_invent_usage() {
    let mut decoder = decoder::create(&Command::default());
    let (content, error) = decoder.finish("cancelled");
    assert!(error.is_none());
    assert!(content.usage.is_empty());
    assert!(decoder.finish("succeeded").1.is_some());
}

#[test]
fn explicit_null_usage_does_not_create_zero_tokens() {
    let mut decoder = decoder::create(&Command::default());
    decoder
        .feed(
            br#"{"status":"completed","output":null,"usage":null}"#,
            false,
        )
        .unwrap();
    let (content, error) = decoder.finish("succeeded");
    assert!(error.is_none());
    assert!(content.usage.is_empty());
    assert_eq!(content.kind, "non_text");
}

#[test]
fn refusal_reasoning_and_unknown_output_types() {
    for (output, valid) in [
        (
            r#"[{"type":"message","content":[{"type":"refusal"}]},{"type":"reasoning"}]"#,
            true,
        ),
        (
            r#"[{"type":"message","content":[{"type":"unknown"}]}]"#,
            false,
        ),
        (r#"[{"type":"unknown"}]"#, false),
    ] {
        let mut decoder = decoder::create(&Command::default());
        let body = format!(r#"{{"status":"completed","output":{output}}}"#);
        assert_eq!(decoder.feed(body.as_bytes(), false).is_ok(), valid);
        if valid {
            let (content, error) = decoder.finish("succeeded");
            assert!(error.is_none());
            assert_eq!(content.kind, "non_text");
            assert_eq!(content.chars, 0);
        }
    }
}

#[test]
fn vllm_reasoning_stream_preserves_text_and_usage_facts() {
    // vLLM responses/streaming_events.py emits this sequence from
    // emit_simple_reasoning_open/delta/done (and emit_reasoning_delta/done_events).
    let reasoning_events = serde_json::json!([
        {
            "type": "response.output_item.added", "sequence_number": 1,
            "output_index": 0,
            "item": {"type": "reasoning", "id": "reasoning-1", "summary": [], "status": "in_progress"}
        },
        {
            "type": "response.reasoning_part.added", "sequence_number": 2,
            "output_index": 0, "item_id": "reasoning-1", "content_index": 0,
            "part": {"type": "reasoning_text", "text": ""}
        },
        {
            "type": "response.reasoning_text.delta", "sequence_number": 3,
            "output_index": 0, "item_id": "reasoning-1", "content_index": 0,
            "delta": "Let me think."
        },
        {
            "type": "response.reasoning_text.done", "sequence_number": 4,
            "output_index": 0, "item_id": "reasoning-1", "content_index": 0,
            "text": "Let me think."
        },
        {
            "type": "response.reasoning_part.done", "sequence_number": 5,
            "output_index": 0, "item_id": "reasoning-1", "content_index": 0,
            "part": {"type": "reasoning_text", "text": "Let me think."}
        },
        {
            "type": "response.output_item.done", "sequence_number": 6,
            "output_index": 0,
            "item": {
                "type": "reasoning", "id": "reasoning-1", "summary": [], "status": "completed",
                "content": [{"type": "reasoning_text", "text": "Let me think."}]
            }
        }
    ]);
    let mut decoder = decoder::create(&Command::default());
    for event in reasoning_events.as_array().unwrap() {
        assert_eq!(
            decoder.feed(event.to_string().as_bytes(), true).unwrap(),
            (0, false)
        );
    }
    assert_eq!(
        decoder
            .feed(
                br#"{"type":"response.output_text.delta","delta":"42"}"#,
                true,
            )
            .unwrap(),
        (2, false)
    );
    assert_eq!(
        decoder
            .feed(
                br#"{"type":"response.completed","response":{"status":"completed","output":[{"type":"reasoning","summary":[]},{"type":"message","content":[{"type":"output_text","text":"42"}]}],"usage":{"input_tokens":10,"output_tokens":7,"total_tokens":17}}}"#,
                true,
            )
            .unwrap(),
        (0, true)
    );
    let (content, error) = decoder.finish("succeeded");
    assert!(error.is_none());
    assert_eq!(content.kind, "text");
    assert_eq!(content.chars, 2);
    assert_eq!(content.finish, "completed");
    assert_eq!(content.usage["prompt_tokens"], 10);
    assert_eq!(content.usage["completion_tokens"], 7);
    assert_eq!(content.usage["total_tokens"], 17);
}
