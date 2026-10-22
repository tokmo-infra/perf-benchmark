use std::collections::BTreeMap;
#[derive(Default, pyo3::FromPyObject)]
#[pyo3(from_item_all)]
pub struct Command {
    pub expectations: BTreeMap<String, i64>,
    pub notify_start: bool,
    pub session_id: String,
    pub connection_scope: String,
    pub verify_ssl: bool,
    pub body_length: i64,
    pub url: String,
    pub method: String,
    pub headers: BTreeMap<String, String>,
    pub stream: bool,
    pub retain: bool,
    pub timeout: f64,
    pub plan: Plan,
}
#[derive(Default, pyo3::FromPyObject)]
#[pyo3(from_item_all)]
pub struct Plan {
    pub after_outputs: Option<i64>,
    pub after_first_output_ms: Option<f64>,
    pub action: String,
}
#[derive(Default)]
pub struct Outcome {
    pub started_ns: Option<i64>,
    pub completed_ns: Option<i64>,
    pub http_status: Option<u16>,
    pub status: &'static str,
    pub error: Option<String>,
    pub error_origin: Option<&'static str>,
    pub retryable: bool,
    pub samples: Vec<[i64; 3]>,
    pub output_chars: i64,
    pub usage: BTreeMap<String, i64>,
    pub finish_reason: Option<String>,
    pub continuation: Option<serde_json::Value>,
    pub continuation_valid: bool,
    pub output_kind: &'static str,
    pub trigger_ns: Option<i64>,
    pub decision_ns: Option<i64>,
    pub applied_ns: Option<i64>,
}
