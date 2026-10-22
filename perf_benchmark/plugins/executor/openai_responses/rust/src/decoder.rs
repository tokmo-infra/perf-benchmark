use perf_rusthttp::{Command, Content, DecodeError, Decoder};
use serde::Deserialize;
use std::collections::BTreeMap;
pub fn create(_: &Command) -> Box<dyn Decoder> {
    Box::new(Responses::default())
}
#[derive(Default)]
struct Responses {
    done: bool,
    text: bool,
    tools: bool,
    chars: i64,
    finish: String,
    usage: BTreeMap<String, i64>,
}
#[derive(Default, Deserialize)]
#[serde(default)]
struct Part {
    #[serde(rename = "type")]
    #[serde(deserialize_with = "perf_rusthttp::null_default")]
    kind: String,
    #[serde(deserialize_with = "perf_rusthttp::null_default")]
    text: String,
}
#[derive(Default, Deserialize)]
#[serde(default)]
struct Item {
    #[serde(rename = "type")]
    #[serde(deserialize_with = "perf_rusthttp::null_default")]
    kind: String,
    #[serde(deserialize_with = "perf_rusthttp::null_default")]
    content: Vec<Part>,
}
#[derive(Default, Deserialize)]
#[serde(default)]
struct Details {
    #[serde(deserialize_with = "perf_rusthttp::null_default")]
    cached_tokens: Option<i64>,
}
#[derive(Default, Deserialize)]
#[serde(default)]
struct Usage {
    input_tokens: Option<i64>,
    output_tokens: Option<i64>,
    total_tokens: Option<i64>,
    #[serde(deserialize_with = "perf_rusthttp::null_default")]
    input_tokens_details: Details,
}
#[derive(Default, Deserialize)]
#[serde(default)]
struct Response {
    #[serde(deserialize_with = "perf_rusthttp::null_default")]
    status: String,
    #[serde(deserialize_with = "perf_rusthttp::null_default")]
    output: Vec<Item>,
    #[serde(deserialize_with = "perf_rusthttp::null_default")]
    usage: Usage,
}
#[derive(Deserialize)]
struct Event {
    #[serde(rename = "type")]
    #[serde(deserialize_with = "perf_rusthttp::null_default")]
    kind: String,
    delta: Option<String>,
    #[serde(default)]
    #[serde(deserialize_with = "perf_rusthttp::null_default")]
    response: Response,
}
impl Responses {
    fn response(&mut self, v: Response, stream: bool) -> Result<(), DecodeError> {
        self.finish = v.status;
        for (key, value) in [
            ("prompt_tokens", v.usage.input_tokens),
            ("completion_tokens", v.usage.output_tokens),
            ("total_tokens", v.usage.total_tokens),
            ("cached_tokens", v.usage.input_tokens_details.cached_tokens),
        ] {
            if let Some(n) = value {
                if n < 0 {
                    return Err(DecodeError::Protocol);
                }
                self.usage.insert(key.into(), n);
            }
        }
        for item in v.output {
            match item.kind.as_str() {
                "message" => {
                    for part in item.content {
                        match part.kind.as_str() {
                            "output_text" => {
                                self.text |= !part.text.is_empty();
                                if !stream {
                                    self.chars += part.text.chars().count() as i64;
                                }
                            }
                            "refusal" => {}
                            _ => return Err(DecodeError::Protocol),
                        }
                    }
                }
                "function_call" => self.tools = true,
                "reasoning" => {}
                _ => return Err(DecodeError::Protocol),
            }
        }
        self.done = true;
        if self.finish != "completed" {
            return Err(DecodeError::Incomplete);
        }
        Ok(())
    }
}
impl Decoder for Responses {
    fn feed(&mut self, data: &[u8], stream: bool) -> Result<(i64, bool), DecodeError> {
        if !stream {
            let v = serde_json::from_slice(data)?;
            self.response(v, false)?;
            return Ok((0, true));
        }
        let v: Event = serde_json::from_slice(data)?;
        match v.kind.as_str() {
            "response.output_text.delta" => {
                let delta = v.delta.ok_or(DecodeError::Protocol)?;
                let chars = delta.chars().count() as i64;
                self.chars += chars;
                self.text |= chars > 0;
                Ok((chars, false))
            }
            "response.completed" | "response.failed" | "response.incomplete" => {
                if v.kind != format!("response.{}", v.response.status) {
                    return Err(DecodeError::Protocol);
                }
                self.response(v.response, true)?;
                Ok((0, true))
            }
            "response.function_call_arguments.delta" => {
                v.delta.ok_or(DecodeError::Protocol)?;
                self.tools = true;
                Ok((0, false))
            }
            "response.created"
            | "response.queued"
            | "response.in_progress"
            | "response.output_item.added"
            | "response.output_item.done"
            | "response.content_part.added"
            | "response.content_part.done"
            | "response.output_text.done"
            | "response.function_call_arguments.done"
            | "response.refusal.delta"
            | "response.refusal.done"
            | "response.reasoning_part.added"
            | "response.reasoning_part.done"
            | "response.reasoning_summary_part.added"
            | "response.reasoning_summary_part.done"
            | "response.reasoning_summary_text.delta"
            | "response.reasoning_summary_text.done"
            | "response.reasoning_text.delta"
            | "response.reasoning_text.done" => Ok((0, false)),
            _ => Err(DecodeError::Protocol),
        }
    }
    fn finish(&mut self, status: &str) -> (Content, Option<DecodeError>) {
        (
            Content {
                kind: if self.text && !self.tools {
                    "text"
                } else {
                    "non_text"
                },
                chars: self.chars,
                usage: std::mem::take(&mut self.usage),
                finish: self.finish.clone(),
                valid: true,
                ..Default::default()
            },
            if status == "succeeded" && !self.done {
                Some(DecodeError::Incomplete)
            } else {
                None
            },
        )
    }
}
