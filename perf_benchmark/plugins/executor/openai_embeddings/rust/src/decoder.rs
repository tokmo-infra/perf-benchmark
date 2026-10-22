use perf_rusthttp::{Command, Content, DecodeError, Decoder};
use serde::Deserialize;
use std::collections::BTreeMap;
pub fn create(cmd: &Command) -> Box<dyn Decoder> {
    Box::new(Embedding {
        count: *cmd.expectations.get("count").unwrap_or(&0),
        dimensions: *cmd.expectations.get("dimensions").unwrap_or(&0),
        ..Default::default()
    })
}
#[derive(Default)]
struct Embedding {
    count: i64,
    dimensions: i64,
    done: bool,
    usage: BTreeMap<String, i64>,
}
#[derive(Deserialize)]
struct Row {
    index: Option<usize>,
    embedding: Vec<Option<f64>>,
}
#[derive(Deserialize)]
struct Response {
    data: Vec<Row>,
    #[serde(default, deserialize_with = "perf_rusthttp::null_default")]
    usage: BTreeMap<String, Option<i64>>,
    #[serde(default, deserialize_with = "perf_rusthttp::error_present")]
    error: bool,
}
impl Decoder for Embedding {
    fn feed(&mut self, data: &[u8], stream: bool) -> Result<(i64, bool), DecodeError> {
        if stream {
            return Err(DecodeError::Protocol);
        }
        let v: Response = serde_json::from_slice(data)?;
        if v.error || self.count <= 0 || v.data.len() != self.count as usize {
            return Err(DecodeError::Protocol);
        }
        let mut seen = vec![false; self.count as usize];
        let mut dimensions = self.dimensions as usize;
        for row in v.data {
            let index = row
                .index
                .filter(|n| *n < seen.len())
                .ok_or(DecodeError::Protocol)?;
            if seen[index] {
                return Err(DecodeError::Protocol);
            }
            seen[index] = true;
            if dimensions == 0 {
                dimensions = row.embedding.len()
            }
            if dimensions == 0
                || row.embedding.len() != dimensions
                || row.embedding.iter().any(|v| !v.is_some_and(f64::is_finite))
            {
                return Err(DecodeError::Protocol);
            }
        }
        for key in ["prompt_tokens", "total_tokens"] {
            if let Some(Some(n)) = v.usage.get(key) {
                if *n < 0 {
                    return Err(DecodeError::Protocol);
                }
                self.usage.insert(key.into(), *n);
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
