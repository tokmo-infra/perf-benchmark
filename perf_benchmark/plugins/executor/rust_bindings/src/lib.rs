use perf_rusthttp::binding::{Attempt, NativeTrial, OutputSamples, PreparedBody};
use perf_trace_inflate::TraceHashInflator;
use pyo3::prelude::*;
#[pyfunction]
fn create_trial(protocol: &str) -> PyResult<NativeTrial> {
    let factory = match protocol {
        "openai-chat" => perf_openai_chat::create,
        "openai-completions" => perf_openai_completions::create,
        "openai-responses" => perf_openai_responses::create,
        "openai-embeddings" => perf_openai_embeddings::create,
        "rerank" => perf_rerank::create,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "unknown executor protocol",
            ))
        }
    };
    Ok(NativeTrial::new(factory))
}
#[pyfunction]
fn now() -> i64 {
    perf_rusthttp::now()
}
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(create_trial, m)?)?;
    m.add_function(wrap_pyfunction!(now, m)?)?;
    m.add_class::<NativeTrial>()?;
    m.add_class::<PreparedBody>()?;
    m.add_class::<Attempt>()?;
    m.add_class::<OutputSamples>()?;
    m.add_class::<TraceHashInflator>()?;
    m.add(
        "NativeResult",
        m.py()
            .import("perf_benchmark.runtime.http.native_result")?
            .getattr("NativeResult")?,
    )?;
    Ok(())
}
