use perf_rusthttp::{Command, Content, DecodeError, Decoder};
use serde_json::{json, Value};
use std::collections::BTreeMap;
pub fn create(cmd: &Command) -> Box<dyn Decoder> {
    Box::new(Chat {
        retain: cmd.retain,
        stream: cmd.stream,
        ..Default::default()
    })
}
#[derive(Default)]
struct Chat {
    retain: bool,
    stream: bool,
    done: bool,
    terminal_choice: bool,
    chars: i64,
    usage: BTreeMap<String, i64>,
    finish: String,
    text: String,
    reasoning: String,
    tools: BTreeMap<i64, Value>,
    function: BTreeMap<String, String>,
}
fn text(v: &Value) -> &str {
    v.as_str().unwrap_or("")
}
fn count(v: &Value) -> i64 {
    text(v).chars().count() as i64
}
fn reason(v: &Value) -> String {
    match text(v) {
        "stop" | "length" | "tool_calls" | "function_call" | "content_filter" => text(v).into(),
        _ => String::new(),
    }
}
impl Chat {
    fn tools(&mut self, value: &Value, streaming: bool) -> i64 {
        let mut chars = 0;
        if let Some(tools) = value["tool_calls"].as_array() {
            for (position, raw) in tools.iter().enumerate() {
                if !raw.is_object() {
                    continue;
                }
                let function = &raw["function"];
                chars += count(&function["name"]) + count(&function["arguments"]);
                if !self.retain {
                    continue;
                }
                if !streaming {
                    self.tools.insert(position as i64, raw.clone());
                    continue;
                }
                let index = raw["index"].as_i64().unwrap_or(0);
                let current = self.tools.entry(index).or_insert_with(
                    || json!({"id":"", "type":"function", "function":{"name":"", "arguments":""}}),
                );
                for key in ["id", "type"] {
                    if !text(&raw[key]).is_empty() {
                        current[key] = raw[key].clone();
                    }
                }
                for key in ["name", "arguments"] {
                    current["function"][key] = Value::String(format!(
                        "{}{}",
                        text(&current["function"][key]),
                        text(&function[key])
                    ));
                }
            }
        }
        if let Some(function) = value["function_call"].as_object() {
            for key in ["name", "arguments"] {
                let value = function.get(key).map(text).unwrap_or("");
                chars += value.chars().count() as i64;
                if self.retain {
                    if streaming {
                        self.function.entry(key.into()).or_default().push_str(value);
                    } else {
                        self.function.insert(key.into(), value.into());
                    }
                }
            }
        }
        chars
    }
}
impl Decoder for Chat {
    fn feed(&mut self, data: &[u8], streaming: bool) -> Result<(i64, bool), DecodeError> {
        if streaming && data == b"[DONE]" {
            self.done = true;
            return Ok((0, true));
        }
        let v: Value = serde_json::from_slice(data)?;
        if !v.is_object() || v.get("error").is_some() {
            return Err(DecodeError::Protocol);
        }
        for key in ["prompt_tokens", "completion_tokens", "total_tokens"] {
            if let Some(n) = v["usage"][key].as_i64().filter(|n| *n >= 0) {
                self.usage.insert(key.into(), n);
            }
        }
        if let Some(n) = v["usage"]["prompt_tokens_details"]["cached_tokens"]
            .as_i64()
            .or_else(|| v["usage"]["cached_tokens"].as_i64())
        {
            self.usage.insert("cached_tokens".into(), n);
        }
        let choices = v["choices"].as_array();
        if !streaming {
            let choice = choices
                .and_then(|v| v.first())
                .ok_or(DecodeError::Protocol)?;
            let message = &choice["message"];
            if !message.is_object() {
                return Err(DecodeError::Protocol);
            }
            self.finish = reason(&choice["finish_reason"]);
            let thought = message
                .get("reasoning_content")
                .unwrap_or(&message["reasoning"]);
            self.chars += count(&message["content"]) + count(thought) + self.tools(message, false);
            if self.retain {
                self.text.push_str(text(&message["content"]));
                self.reasoning.push_str(text(thought));
            }
            self.done = true;
            return Ok((0, true));
        }
        let mut chars = 0;
        for choice in choices.into_iter().flatten() {
            let delta = &choice["delta"];
            if choice["finish_reason"].is_string() {
                self.finish = reason(&choice["finish_reason"]);
                self.terminal_choice |= delta.is_object() && !self.finish.is_empty();
            }
            let thought = if text(&delta["reasoning_content"]).is_empty() {
                &delta["reasoning"]
            } else {
                &delta["reasoning_content"]
            };
            chars += count(&delta["content"]) + count(thought) + self.tools(delta, true);
            if self.retain {
                self.text.push_str(text(&delta["content"]));
                self.reasoning.push_str(text(thought));
            }
        }
        self.chars += chars;
        Ok((chars, false))
    }
    fn finish(&mut self, status: &str) -> (Content, Option<DecodeError>) {
        let error = if status == "succeeded" && !self.done {
            Some(DecodeError::Incomplete)
        } else if status == "succeeded" && self.stream && self.chars == 0 && !self.terminal_choice {
            Some(DecodeError::Protocol)
        } else {
            None
        };
        let status = if error.is_some() { "failed" } else { status };
        let has_tool = !self.tools.is_empty() || self.function.values().any(|v| !v.is_empty());
        let valid = !(status != "succeeded"
            && has_tool
            && self.finish != "tool_calls"
            && self.finish != "function_call");
        let continuation = if self.retain && status != "failed" && valid {
            let mut message = json!({"role":"assistant", "content":std::mem::take(&mut self.text)});
            if !self.reasoning.is_empty() {
                message["reasoning_content"] = Value::String(std::mem::take(&mut self.reasoning));
            }
            if !self.tools.is_empty() {
                message["tool_calls"] =
                    Value::Array(std::mem::take(&mut self.tools).into_values().collect());
            }
            if self.function.values().any(|v| !v.is_empty()) {
                message["function_call"] = json!(self.function);
            }
            Some(message)
        } else {
            None
        };
        (
            Content {
                chars: self.chars,
                usage: std::mem::take(&mut self.usage),
                finish: self.finish.clone(),
                continuation,
                valid,
                ..Default::default()
            },
            error,
        )
    }
}
