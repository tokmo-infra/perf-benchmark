use perf_rusthttp::{Command, Content, DecodeError, Decoder};
use serde::Deserialize;
use std::collections::BTreeMap;
pub fn create(cmd: &Command) -> Box<dyn Decoder> {
    Box::new(Rerank {
        count: *cmd.expectations.get("count").unwrap_or(&0),
        top_n: *cmd.expectations.get("top_n").unwrap_or(&0),
        done: false,
        usage: BTreeMap::new(),
    })
}
struct Rerank {
    count: i64,
    top_n: i64,
    done: bool,
    usage: BTreeMap<String, i64>,
}
#[derive(Deserialize)]
struct Row {
    index: Option<usize>,
    relevance_score: Option<f64>,
}
#[derive(Deserialize)]
struct Response {
    results: Vec<Row>,
    #[serde(default)]
    usage: Option<Usage>,
    #[serde(default, deserialize_with = "perf_rusthttp::error_present")]
    error: bool,
}
#[derive(Default, Deserialize)]
struct Usage {
    prompt_tokens: Option<i64>,
    total_tokens: Option<i64>,
}
impl Decoder for Rerank {
    fn feed(&mut self, data: &[u8], stream: bool) -> Result<(i64, bool), DecodeError> {
        if stream {
            return Err(DecodeError::Protocol);
        }
        let v: Response = serde_json::from_slice(data)?;
        if v.error || self.count <= 0 || self.top_n <= 0 || v.results.len() != self.top_n as usize {
            return Err(DecodeError::Protocol);
        }
        let mut seen = vec![false; self.count as usize];
        let mut previous = f64::INFINITY;
        for row in v.results {
            let index = row
                .index
                .filter(|n| *n < seen.len())
                .ok_or(DecodeError::Protocol)?;
            let score = row
                .relevance_score
                .filter(|n| n.is_finite() && *n <= previous)
                .ok_or(DecodeError::Protocol)?;
            if seen[index] {
                return Err(DecodeError::Protocol);
            }
            seen[index] = true;
            previous = score;
        }
        let usage = v.usage.unwrap_or_default();
        for (key, value) in [
            ("prompt_tokens", usage.prompt_tokens),
            ("total_tokens", usage.total_tokens),
        ] {
            if let Some(n) = value {
                if n < 0 {
                    return Err(DecodeError::Protocol);
                }
                self.usage.insert(key.into(), n);
            }
        }
        self.done = true;
        Ok((0, true))
    }
    fn finish(&mut self, status: &str) -> (Content, Option<DecodeError>) {
        (
            Content {
                kind: "non_text",
                usage: std::mem::take(&mut self.usage),
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
