#[path = "../src/decoder.rs"]
mod decoder;
use perf_rusthttp::Command;
#[test]
fn rejects_invalid_protocol_inputs() {
    let mut command = Command::default();
    command.expectations.insert("count".into(), 2);
    command.expectations.insert("top_n".into(), 2);
    for body in [
        r#"{"data":[{"index":0,"embedding":[0.1]},{"index":0,"embedding":[0.2]}]}"#,
        r#"{"data":[{"index":0,"embedding":[0.1]},{"index":1,"embedding":[0.2,0.3]}]}"#,
        r#"{"data":[{"index":0,"embedding":[null]},{"index":1,"embedding":[0.2]}]}"#,
        r#"{"data":[{"index":0,"embedding":[1e400]},{"index":1,"embedding":[0.2]}]}"#,
        r#"{"data":[]}"#,
        r#"{"error":{}}"#,
        r#"null"#,
    ] {
        let mut decoder = decoder::create(&command);
        let error = decoder
            .feed(body.as_bytes(), false)
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
