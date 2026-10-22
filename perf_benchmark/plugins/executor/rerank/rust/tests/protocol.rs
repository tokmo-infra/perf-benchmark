#[path = "../src/decoder.rs"]
mod decoder;
use perf_rusthttp::Command;
#[test]
fn rejects_invalid_protocol_inputs() {
    let mut command = Command::default();
    command.expectations.insert("count".into(), 2);
    command.expectations.insert("top_n".into(), 2);
    for body in [
        r#"{"results":[{"index":0,"relevance_score":0.8},{"index":0,"relevance_score":0.2}]}"#,
        r#"{"results":[{"index":0,"relevance_score":0.1},{"index":1,"relevance_score":0.2}]}"#,
        r#"{"results":[{"index":0,"relevance_score":null},{"index":1,"relevance_score":0.2}]}"#,
        r#"{"results":[{"index":2,"relevance_score":0.8},{"index":1,"relevance_score":0.2}]}"#,
        r#"{"results":[]}"#,
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

#[test]
fn preserves_only_explicit_non_negative_rerank_usage() {
    let mut command = Command::default();
    command.expectations.insert("count".into(), 1);
    command.expectations.insert("top_n".into(), 1);
    for (usage, expected) in [
        ("", vec![]),
        (r#", "usage":null"#, vec![]),
        (r#", "usage":{}"#, vec![]),
        (
            r#", "usage":{"prompt_tokens":null,"total_tokens":null}"#,
            vec![],
        ),
        (
            r#", "usage":{"prompt_tokens":5,"total_tokens":5,"completion_tokens":2}"#,
            vec![("prompt_tokens", 5), ("total_tokens", 5)],
        ),
        (
            r#", "usage":{"prompt_tokens":0}"#,
            vec![("prompt_tokens", 0)],
        ),
        (r#", "usage":{"total_tokens":7}"#, vec![("total_tokens", 7)]),
    ] {
        let mut decoder = decoder::create(&command);
        let body = format!(r#"{{"results":[{{"index":0,"relevance_score":0.9}}]{usage}}}"#);
        assert_eq!(decoder.feed(body.as_bytes(), false).unwrap(), (0, true));
        let (content, error) = decoder.finish("succeeded");
        assert!(error.is_none());
        assert_eq!(content.kind, "non_text");
        assert_eq!(content.chars, 0);
        assert_eq!(
            content.usage,
            expected
                .into_iter()
                .map(|(key, n)| (key.into(), n))
                .collect()
        );
    }
}

#[test]
fn rejects_invalid_rerank_usage() {
    let mut command = Command::default();
    command.expectations.insert("count".into(), 1);
    command.expectations.insert("top_n".into(), 1);
    for usage in [
        r#"{"prompt_tokens":-1}"#,
        r#"{"total_tokens":-1}"#,
        r#"{"prompt_tokens":1.5}"#,
        r#"{"total_tokens":true}"#,
        r#"{"prompt_tokens":"3"}"#,
        r#"[]"#,
        r#""invalid""#,
    ] {
        let mut decoder = decoder::create(&command);
        let body =
            format!(r#"{{"results":[{{"index":0,"relevance_score":0.9}}],"usage":{usage}}}"#);
        assert!(
            decoder.feed(body.as_bytes(), false).is_err(),
            "accepted {usage}"
        );
    }
}
