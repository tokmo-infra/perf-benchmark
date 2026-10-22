use perf_rusthttp::{Command, Content, DecodeError, Decoder};
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;
pub fn create(cmd: &Command) -> Box<dyn Decoder> {
    Box::new(Completion {
        expected: *cmd.expectations.get("output_tokens").unwrap_or(&0),
        stream: cmd.stream,
        ..Default::default()
    })
}
#[derive(Default)]
struct Completion {
    expected: i64,
    stream: bool,
    done: bool,
    seen: bool,
    chars: i64,
    finish: String,
    usage: BTreeMap<String, i64>,
}
#[derive(Deserialize)]
struct Choice {
    #[serde(default, deserialize_with = "perf_rusthttp::null_default")]
    index: i64,
    text: Option<String>,
    finish_reason: Option<String>,
}
#[derive(Deserialize)]
struct Response {
    #[serde(default, deserialize_with = "perf_rusthttp::null_default")]
    choices: Vec<Choice>,
    #[serde(default, deserialize_with = "perf_rusthttp::null_default")]
    usage: BTreeMap<String, Value>,
    #[serde(default, deserialize_with = "perf_rusthttp::error_present")]
    error: bool,
}
impl Decoder for Completion {
    fn feed(&mut self, data: &[u8], streaming: bool) -> Result<(i64, bool), DecodeError> {
        if streaming && data == b"[DONE]" {
            self.done = true;
            return Ok((0, true));
        }
        let v: Response = serde_json::from_slice(data)?;
        if v.error || v.choices.len() > 1 || (!streaming && v.choices.len() != 1) {
            return Err(DecodeError::Protocol);
        }
        for key in ["prompt_tokens", "completion_tokens", "total_tokens"] {
            if let Some(value) = v.usage.get(key).filter(|v| !v.is_null()) {
                let n = value
                    .as_i64()
                    .filter(|n| *n >= 0)
                    .ok_or(DecodeError::Protocol)?;
                self.usage.insert(key.into(), n);
            }
        }
        if let Some(value) = v
            .usage
            .get("prompt_tokens_details")
            .and_then(|v| v.get("cached_tokens"))
            .filter(|v| !v.is_null())
        {
            let n = value
                .as_i64()
                .filter(|n| *n >= 0)
                .ok_or(DecodeError::Protocol)?;
            self.usage.insert("cached_tokens".into(), n);
        }
        let mut chars = 0;
        for choice in v.choices {
            if choice.index != 0 {
                return Err(DecodeError::Protocol);
            }
            chars += choice.text.ok_or(DecodeError::Protocol)?.chars().count() as i64;
            self.seen = true;
            if let Some(reason) = choice.finish_reason {
                if !matches!(reason.as_str(), "stop" | "length" | "content_filter") {
                    return Err(DecodeError::Protocol);
                }
                self.finish = reason;
            }
        }
        self.chars += chars;
        if !streaming {
            self.done = true;
            return Ok((0, true));
        }
        Ok((chars, false))
    }
    fn finish(&mut self, status: &str) -> (Content, Option<DecodeError>) {
        let error = if status == "succeeded"
            && (!self.done
                || !self.seen
                || self.finish.is_empty()
                || (self.stream && self.chars == 0))
        {
            Some(DecodeError::Incomplete)
        } else if status == "succeeded"
            && self.expected > 0
            && self.usage.get("completion_tokens") != Some(&self.expected)
        {
            Some(DecodeError::Protocol)
        } else {
            None
        };
        (
            Content {
                chars: self.chars,
                usage: std::mem::take(&mut self.usage),
                finish: self.finish.clone(),
                valid: true,
                ..Default::default()
            },
            error,
        )
    }
}
