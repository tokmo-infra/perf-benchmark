#[path = "../src/decoder.rs"]
mod decoder;
use perf_rusthttp::Command;
#[test]
fn rejects_invalid_protocol_inputs() {
    let mut command = Command::default();
    command.expectations.insert("count".into(), 2);
    command.expectations.insert("top_n".into(), 2);
    for body in [
        r#"{"error":{}}"#,
        r#"{"choices":[{"index":1,"text":"x"}]}"#,
        r#"{"usage":{"completion_tokens":-1}}"#,
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
fn null_usage_is_unknown_and_explicit_error_is_failure() {
    let mut decoder = decoder::create(&Command::default());
    decoder
        .feed(
            br#"{"choices":[{"index":null,"text":"x","finish_reason":"length"}],"usage":null}"#,
            false,
        )
        .unwrap();
    let (content, error) = decoder.finish("succeeded");
    assert!(error.is_none());
    assert!(content.usage.is_empty());
    assert!(decoder
        .feed(br#"{"choices":[{"text":"x"}],"error":null}"#, false)
        .is_err());
}

#[test]
fn cached_usage_and_fixed_length_are_validated() {
    for (tokens, cached, reason, valid) in [
        (2, 1, "length", true),
        (1, 1, "length", false),
        (2, -1, "length", false),
        (2, 1, "unknown", false),
    ] {
        let mut cmd = Command::default();
        cmd.expectations.insert("output_tokens".into(), 2);
        let mut decoder = decoder::create(&cmd);
        let body = serde_json::to_vec(&serde_json::json!({"choices":[{"text":"好", "finish_reason":reason}], "usage":{"completion_tokens":tokens, "prompt_tokens_details":{"cached_tokens":cached}}})).unwrap();
        let feed = decoder.feed(&body, false);
        let (content, error) = decoder.finish("succeeded");
        assert_eq!(feed.is_ok() && error.is_none(), valid);
        if valid {
            assert_eq!(content.usage["cached_tokens"], 1);
        }
    }
}
