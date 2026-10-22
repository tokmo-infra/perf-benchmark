use crate::{
    body::Body,
    http::Http,
    types::{Command, Outcome},
    Decoder,
};
use pyo3::{
    exceptions::{PyBufferError, PyIndexError, PyRuntimeError, PyValueError},
    ffi,
    prelude::*,
    sync::PyOnceLock,
    types::{PyBytes, PySlice, PyTuple},
    IntoPyObjectExt,
};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};
use tokio::sync::{oneshot, Notify};
use tokio_util::sync::CancellationToken;

// HTTP polling must never share workers with callbacks that acquire the GIL.
static HTTP_RUNTIME: OnceLock<Result<tokio::runtime::Runtime, std::io::Error>> = OnceLock::new();

type Factory = fn(&Command) -> Box<dyn Decoder>;
#[pyclass(module = "perf_benchmark._native")]
pub struct PreparedBody {
    body: Option<Body>,
}
#[pymethods]
impl PreparedBody {
    #[new]
    fn new(py: Python<'_>, segments: Vec<Bound<'_, PyAny>>, length: usize) -> PyResult<Self> {
        Ok(Self {
            body: Some(Body::acquire(py, segments, length)?),
        })
    }
    fn release(&mut self) {
        self.body.take();
    }
}
struct State {
    closed: bool,
    error: Option<String>,
    sequence: u64,
    tasks: HashMap<u64, (String, CancellationToken)>,
}
struct Trial {
    http: Http,
    state: Mutex<State>,
    changed: Notify,
    factory: Factory,
}
#[pyclass(module = "perf_benchmark._native")]
pub struct NativeTrial {
    inner: Arc<Trial>,
}
impl NativeTrial {
    pub fn new(factory: Factory) -> Self {
        Self {
            inner: Arc::new(Trial {
                http: Http::default(),
                state: Mutex::new(State {
                    closed: false,
                    error: None,
                    sequence: 0,
                    tasks: HashMap::new(),
                }),
                changed: Notify::new(),
                factory,
            }),
        }
    }
}
impl Drop for NativeTrial {
    fn drop(&mut self) {
        let inner = self.inner.clone();
        {
            let mut state = inner.state.lock().unwrap();
            state.closed = true;
            for (_, token) in state.tasks.values() {
                token.cancel();
            }
        }
        // Abandoned trials still join HTTP tasks before releasing buffer exports.
        pyo3_async_runtimes::tokio::get_runtime().spawn(async move {
            loop {
                let change = inner.changed.notified();
                tokio::pin!(change);
                change.as_mut().enable();
                if inner.state.lock().unwrap().tasks.is_empty() {
                    break;
                }
                change.await;
            }
            inner.http.close().await;
        });
    }
}
#[pymethods]
impl NativeTrial {
    #[pyo3(signature = (body, command, *, cancelled=false))]
    fn submit(
        &self,
        py: Python<'_>,
        mut body: PyRefMut<'_, PreparedBody>,
        command: Command,
        cancelled: bool,
    ) -> PyResult<Attempt> {
        let mut state = self.inner.state.lock().unwrap();
        if let Some(error) = &state.error {
            return Err(PyRuntimeError::new_err(error.clone()));
        }
        if state.closed {
            return Err(PyRuntimeError::new_err("native trial is closed"));
        }
        let network = HTTP_RUNTIME
            .get_or_init(|| {
                tokio::runtime::Builder::new_multi_thread()
                    .thread_name("perf-http")
                    .enable_all()
                    .build()
            })
            .as_ref()
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        let mut body = body
            .body
            .take()
            .ok_or_else(|| PyValueError::new_err("body already consumed"))?;
        if command.body_length < 0 || body.length != command.body_length as usize {
            return Err(PyValueError::new_err("body length mismatch"));
        }
        state.sequence += 1;
        let id = state.sequence;
        let cancel = CancellationToken::new();
        if cancelled {
            cancel.cancel();
        }
        state
            .tasks
            .insert(id, (command.session_id.clone(), cancel.clone()));
        drop(state);
        let (result_tx, result_rx) = oneshot::channel();
        let (start_tx, start_rx) = oneshot::channel();
        let loop_ref = py
            .import("asyncio")?
            .call_method0("get_running_loop")?
            .unbind();
        let result_future = loop_ref.bind(py).call_method0("create_future")?.unbind();
        let result_notify = result_future.clone_ref(py);
        let settled = Arc::new(Mutex::new(Settlement::default()));
        let task_settled = settled.clone();
        let inner = self.inner.clone();
        let token = cancel.clone();
        let started = if command.notify_start {
            Some(start_tx)
        } else {
            drop(start_tx);
            None
        };
        let request_segments = std::mem::take(&mut body.segments);
        let executing = inner.clone();
        let execution = network.spawn(async move {
            let decoder = (executing.factory)(&command);
            executing
                .http
                .execute(
                    &command,
                    request_segments,
                    decoder,
                    token,
                    started,
                    result_tx,
                )
                .await
        });
        pyo3_async_runtimes::tokio::get_runtime().spawn(async move {
            tokio::pin!(execution);
            let mut result_rx = result_rx;
            let (outcome, delivered) = tokio::select! {
                biased;
                value = &mut result_rx => {
                    let delivered = deliver_result(&loop_ref, &result_notify, value);
                    (execution.await, delivered)
                }
                outcome = &mut execution => {
                    let delivered = deliver_result(&loop_ref, &result_notify, result_rx.await);
                    (outcome, delivered)
                }
            };
            // The result was sent before HTTP cleanup. The body remains pinned
            // until every transport owner drops its Bytes reference.
            body.segments.clear();
            body.wait_released().await;
            Python::attach(|_| drop(body));
            {
                let mut state = inner.state.lock().unwrap();
                if outcome.is_err() || delivered.is_err() {
                    state.error = Some(if outcome.is_err() {
                        "native execution task panicked".to_owned()
                    } else {
                        delivered.unwrap_err()
                    });
                    for (_, token) in state.tasks.values() {
                        token.cancel();
                    }
                }
                state.tasks.remove(&id);
            }
            let notification = {
                let mut state = task_settled.lock().unwrap();
                state.complete = true;
                state.future.take()
            };
            // Most callers observe completion directly. Only registered waiters
            // require a Python callback; never acquire the GIL with this lock held.
            if let Some(future) = notification {
                if let Err(error) = deliver_settled(&loop_ref, &future) {
                    inner.state.lock().unwrap().error = Some(error);
                }
            }
            inner.changed.notify_waiters();
        });
        Ok(Attempt {
            cancel,
            id,
            started: Mutex::new(Some(start_rx)),
            result: result_future,
            settled,
        })
    }
    #[getter]
    fn active_count(&self) -> usize {
        self.inner.state.lock().unwrap().tasks.len()
    }
    #[getter]
    fn closed(&self) -> bool {
        self.inner.state.lock().unwrap().closed
    }
    #[getter]
    fn finished(&self) -> bool {
        let state = self.inner.state.lock().unwrap();
        state.closed && state.tasks.is_empty()
    }
    fn close_session<'py>(
        &self,
        py: Python<'py>,
        session_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            loop {
                let change = inner.changed.notified();
                tokio::pin!(change);
                change.as_mut().enable();
                let active = {
                    let state = inner.state.lock().unwrap();
                    let mut found = false;
                    for (session, token) in state.tasks.values() {
                        if session == &session_id {
                            found = true;
                            token.cancel();
                        }
                    }
                    found
                };
                if !active {
                    break;
                }
                change.await;
            }
            inner.http.close_session(&session_id).await;
            Ok(())
        })
    }
    fn close<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        {
            let mut state = inner.state.lock().unwrap();
            state.closed = true;
            for (_, token) in state.tasks.values() {
                token.cancel();
            }
        }
        inner.changed.notify_waiters();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            loop {
                let change = inner.changed.notified();
                tokio::pin!(change);
                change.as_mut().enable();
                if inner.state.lock().unwrap().tasks.is_empty() {
                    break;
                }
                change.await;
            }
            inner.http.close().await;
            let error = inner.state.lock().unwrap().error.clone();
            if let Some(error) = error {
                return Err(PyRuntimeError::new_err(error));
            }
            Ok(())
        })
    }
}
#[derive(Default)]
struct Settlement {
    complete: bool,
    future: Option<Py<PyAny>>,
}

#[pyclass(module = "perf_benchmark._native")]
pub struct Attempt {
    cancel: CancellationToken,
    #[pyo3(get)]
    id: u64,
    started: Mutex<Option<oneshot::Receiver<i64>>>,
    result: Py<PyAny>,
    settled: Arc<Mutex<Settlement>>,
}
#[pymethods]
impl Attempt {
    fn cancel(&self) {
        self.cancel.cancel();
    }
    fn started<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let rx = self
            .started
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("start notification already awaited"))?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move { Ok(rx.await.ok()) })
    }
    fn result(&self, py: Python<'_>) -> Py<PyAny> {
        self.result.clone_ref(py)
    }
    fn settled(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let mut state = self.settled.lock().unwrap();
        if let Some(future) = &state.future {
            return Ok(future.clone_ref(py));
        }
        let future = self
            .result
            .bind(py)
            .call_method0("get_loop")?
            .call_method0("create_future")?;
        if state.complete {
            future.call_method1("set_result", (py.None(),))?;
        }
        state.future = Some(future.clone().unbind());
        Ok(future.unbind())
    }
    #[getter]
    fn settled_done(&self) -> bool {
        self.settled.lock().unwrap().complete
    }
}
fn deliver_result(
    loop_ref: &Py<PyAny>,
    future: &Py<PyAny>,
    received: Result<Outcome, oneshot::error::RecvError>,
) -> Result<(), String> {
    Python::attach(|py| {
        let converted = received
            .map_err(|_| "native execution task stopped".to_owned())
            .and_then(|value| result_into_py(py, value).map_err(|error| error.to_string()));
        let failure = converted.as_ref().err().cloned();
        let (method, argument) = match converted {
            Ok(value) => ("set_result", value),
            Err(error) => (
                "set_exception",
                py.get_type::<PyRuntimeError>()
                    .call1((error,))
                    .map_err(|error| error.to_string())?
                    .unbind(),
            ),
        };
        let callback = future
            .bind(py)
            .getattr(method)
            .map_err(|error| error.to_string())?;
        loop_ref
            .bind(py)
            .call_method1("call_soon_threadsafe", (callback, argument))
            .map_err(|error| error.to_string())?;
        failure.map_or(Ok(()), Err)
    })
}

fn deliver_settled(loop_ref: &Py<PyAny>, future: &Py<PyAny>) -> Result<(), String> {
    Python::attach(|py| {
        let callback = future
            .bind(py)
            .getattr("set_result")
            .map_err(|error| error.to_string())?;
        loop_ref
            .bind(py)
            .call_method1("call_soon_threadsafe", (callback, py.None()))
            .map_err(|error| error.to_string())?;
        Ok(())
    })
}
// Own the contiguous native samples until the last Python reference is dropped.
// Indexing materializes only the requested row, never a second whole trace.
#[pyclass(frozen, sequence, module = "perf_benchmark._native")]
pub struct OutputSamples {
    rows: Vec<[i64; 3]>,
}
#[pymethods]
impl OutputSamples {
    #[new]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        if data.len() % 24 != 0 {
            return Err(PyValueError::new_err(
                "sample buffer must contain complete i64 triples",
            ));
        }
        let rows = data
            .chunks_exact(24)
            .map(|row| {
                std::array::from_fn(|index| {
                    i64::from_ne_bytes(row[index * 8..(index + 1) * 8].try_into().unwrap())
                })
            })
            .collect();
        Ok(Self { rows })
    }
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: std::ffi::c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(PyBufferError::new_err("sample buffer view is null"));
        }
        let samples = slf.borrow();
        // SAFETY: frozen rows never move or resize. FillInfo retains slf for the
        // exported view's lifetime, reports read-only bytes and rejects writes.
        // Vec allocation bounds guarantee the byte length fits Py_ssize_t.
        let status = unsafe {
            ffi::PyBuffer_FillInfo(
                view,
                slf.as_ptr(),
                samples.rows.as_ptr().cast_mut().cast(),
                (samples.rows.len() * 24) as isize,
                1,
                flags,
            )
        };
        if status == -1 {
            Err(PyErr::fetch(slf.py()))
        } else {
            Ok(())
        }
    }
    fn __len__(&self) -> usize {
        self.rows.len()
    }
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(slice) = key.cast::<PySlice>() {
            let indices = slice.indices(self.rows.len() as isize)?;
            let rows = (0..indices.slicelength).map(|offset| {
                let index = indices.start + offset as isize * indices.step;
                let [a, b, c] = self.rows[index as usize];
                (a, b, c)
            });
            return Ok(PyTuple::new(py, rows)?.into_any().unbind());
        }
        let mut index: isize = key.extract()?;
        if index < 0 {
            index += self.rows.len() as isize;
        }
        let [a, b, c] = *self
            .rows
            .get(index as usize)
            .ok_or_else(|| PyIndexError::new_err("output sample index out of range"))?;
        (a, b, c).into_py_any(py)
    }
}

// Convert each terminal result once; no other Attempt needs to finish first.
fn result_into_py(py: Python<'_>, value: Outcome) -> PyResult<Py<PyAny>> {
    static TYPE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    let class = TYPE.get_or_try_init(py, || {
        py.import("perf_benchmark.runtime.http.native_result")?
            .getattr("NativeResult")
            .map(|class| class.unbind())
    })?;
    let continuation = value
        .continuation
        .map(|value| {
            serde_json::to_vec(&value)
                .map(|data| PyBytes::new(py, &data))
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))
        })
        .transpose()?;
    let fields = vec![
        value.status.into_py_any(py)?,
        value.error.into_py_any(py)?,
        value.error_origin.into_py_any(py)?,
        value.retryable.into_py_any(py)?,
        value.started_ns.into_py_any(py)?,
        value.completed_ns.into_py_any(py)?,
        value.http_status.into_py_any(py)?,
        Py::new(
            py,
            OutputSamples {
                rows: value.samples,
            },
        )?
        .into_any(),
        value.usage.into_py_any(py)?,
        value.output_chars.into_py_any(py)?,
        value.finish_reason.into_py_any(py)?,
        continuation.into_py_any(py)?,
        value.continuation_valid.into_py_any(py)?,
        value.output_kind.into_py_any(py)?,
        value.trigger_ns.into_py_any(py)?,
        value.decision_ns.into_py_any(py)?,
        value.applied_ns.into_py_any(py)?,
    ];
    class.call1(py, PyTuple::new(py, fields)?)
}
