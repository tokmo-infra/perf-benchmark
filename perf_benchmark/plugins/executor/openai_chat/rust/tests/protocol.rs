#[path = "../src/decoder.rs"]
mod decoder;
use perf_rusthttp::Command;
#[test]
fn tools_reasoning_usage_and_partial_continuation() {
    let cmd = Command {
        stream: true,
        retain: true,
        ..Default::default()
    };
    let mut decoder = decoder::create(&cmd);
    decoder.feed(br#"{"choices":[{"delta":{"tool_calls":[{"index":1,"id":"b","function":{"name":"foo","arguments":"{"}}]}}]}"#, true).unwrap();
    let (partial, error) = decoder.finish("cancelled");
    assert!(error.is_none());
    assert!(!partial.valid);
    assert!(partial.continuation.is_none());
    decoder.feed("{\"choices\":[{\"delta\":{\"content\":\"好\",\"reasoning_content\":\"想\",\"tool_calls\":[{\"index\":1,\"function\":{\"arguments\":\"}\"}}]},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"completion_tokens\":2,\"prompt_tokens_details\":{\"cached_tokens\":3}}}".as_bytes(), true).unwrap();
    decoder.feed(b"[DONE]", true).unwrap();
    let (content, error) = decoder.finish("succeeded");
    assert!(error.is_none());
    assert_eq!(content.chars, 7);
    assert_eq!(content.usage["cached_tokens"], 3);
    let message = content.continuation.unwrap();
    assert_eq!(message["reasoning_content"], "想");
    assert_eq!(message["tool_calls"][0]["function"]["arguments"], "{}");
}
#[test]
fn incomplete_and_invalid_streams() {
    for body in [
        b"null".as_slice(),
        b"{} garbage",
        b"{secret}",
        br#"{"error":{}}"#,
    ] {
        assert!(decoder::create(&Command::default())
            .feed(body, true)
            .is_err());
    }
    let mut decoder = decoder::create(&Command {
        stream: true,
        ..Default::default()
    });
    assert!(decoder.finish("succeeded").1.is_some());
    decoder.feed(b"[DONE]", true).unwrap();
    assert!(decoder.finish("succeeded").1.is_some());
}

#[test]
fn completed_empty_choice_preserves_terminal_and_usage() {
    for finish in [
        "stop",
        "length",
        "content_filter",
        "tool_calls",
        "function_call",
    ] {
        let mut decoder = decoder::create(&Command {
            stream: true,
            retain: true,
            ..Default::default()
        });
        decoder
            .feed(
                br#"{"choices":[{"delta":{"role":"assistant","content":""}}]}"#,
                true,
            )
            .unwrap();
        decoder
            .feed(
                &serde_json::to_vec(
                    &serde_json::json!({"choices":[{"delta":{},"finish_reason":finish}]}),
                )
                .unwrap(),
                true,
            )
            .unwrap();
        decoder.feed(br#"{"choices":[],"usage":{"prompt_tokens":133930,"completion_tokens":4096,"prompt_tokens_details":{"cached_tokens":126720}}}"#, true).unwrap();
        decoder.feed(b"[DONE]", true).unwrap();
        let (content, error) = decoder.finish("succeeded");
        assert!(error.is_none());
        assert_eq!(content.chars, 0);
        assert_eq!(content.finish, finish);
        assert_eq!(content.usage["completion_tokens"], 4096);
        assert_eq!(content.usage["cached_tokens"], 126720);
        assert_eq!(content.continuation.unwrap()["content"], "");
    }
}

#[test]
fn empty_stream_requires_structurally_valid_recognized_terminal_choice() {
    for event in [
        br#"{"choices":[{"delta":{"role":"assistant"}}]}"#.as_slice(),
        br#"{"choices":[],"usage":{"completion_tokens":1}}"#,
        br#"{"choices":[{"delta":null,"finish_reason":"length"}]}"#,
        br#"{"choices":[{"delta":[],"finish_reason":"stop"}]}"#,
        br#"{"choices":[{"finish_reason":"content_filter"}]}"#,
        br#"{"choices":[{"delta":{},"finish_reason":"unknown"}]}"#,
    ] {
        let mut decoder = decoder::create(&Command {
            stream: true,
            ..Default::default()
        });
        decoder.feed(event, true).unwrap();
        decoder.feed(b"[DONE]", true).unwrap();
        assert!(decoder.finish("succeeded").1.is_some());
    }
    let mut decoder = decoder::create(&Command {
        stream: true,
        ..Default::default()
    });
    decoder
        .feed(
            br#"{"choices":[{"delta":{},"finish_reason":"length"}]}"#,
            true,
        )
        .unwrap();
    assert!(decoder.finish("succeeded").1.is_some());
    assert!(decoder.feed(b"{malformed}", true).is_err());
    assert!(decoder
        .feed(br#"{"error":{"message":"upstream error"}}"#, true)
        .is_err());
}

#[test]
fn legacy_function_and_nonstream_tools_retain_only_live_continuation() {
    for streaming in [true, false] {
        for retain in [true, false] {
            let mut decoder = decoder::create(&Command {
                stream: streaming,
                retain,
                ..Default::default()
            });
            let message = serde_json::json!({"content":"好", "reasoning":"想", "tool_calls":[null, {"id":"a", "function":{"name":"foo","arguments":"{}"}}], "function_call":{"name":"bar","arguments":"{}"}});
            let mut choice = serde_json::json!({"finish_reason":"function_call"});
            choice[if streaming { "delta" } else { "message" }] = message;
            let body = serde_json::to_vec(&serde_json::json!({"choices":[choice]})).unwrap();
            decoder.feed(&body, streaming).unwrap();
            if streaming {
                decoder.feed(b"[DONE]", true).unwrap();
            }
            let (content, error) = decoder.finish("succeeded");
            assert!(error.is_none());
            assert_eq!(content.chars, 12);
            assert_eq!(content.continuation.is_some(), retain);
            if retain {
                let message = content.continuation.unwrap();
                assert_eq!(message["content"], "好");
                assert_eq!(message["reasoning_content"], "想");
                assert_eq!(message["function_call"]["arguments"], "{}");
                assert_eq!(message["tool_calls"][0]["function"]["name"], "foo");
            }
        }
    }
    for body in [
        br#"{"choices":[{"message":null}]}"#.as_slice(),
        br#"{"choices":[]}"#,
    ] {
        assert!(decoder::create(&Command::default())
            .feed(body, false)
            .is_err());
    }
}

#[test]
fn cached_usage_supports_flat_field_and_prefers_nested_details() {
    for streaming in [true, false] {
        for (usage, expected) in [
            (serde_json::json!({"cached_tokens": 4}), 4),
            (
                serde_json::json!({"cached_tokens": 4, "prompt_tokens_details": {"cached_tokens": 2}}),
                2,
            ),
            (
                serde_json::json!({"cached_tokens": 4, "prompt_tokens_details": {"cached_tokens": 0}}),
                0,
            ),
        ] {
            let mut decoder = decoder::create(&Command {
                stream: streaming,
                ..Default::default()
            });
            let mut choice = serde_json::json!({"finish_reason": "stop"});
            choice[if streaming { "delta" } else { "message" }] =
                serde_json::json!({"content": "ok"});
            let body =
                serde_json::to_vec(&serde_json::json!({"choices": [choice], "usage": usage}))
                    .unwrap();
            decoder.feed(&body, streaming).unwrap();
            if streaming {
                decoder.feed(b"[DONE]", true).unwrap();
            }
            let (content, error) = decoder.finish("succeeded");
            assert!(error.is_none());
            assert_eq!(content.usage["cached_tokens"], expected);
        }
    }
}
