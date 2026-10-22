mod support;
use perf_rusthttp::{
    binding::{NativeTrial, PreparedBody},
    Command, Content, DecodeError, Decoder,
};
use pyo3::{prelude::*, types::PyDict};
use std::{
    ffi::CString,
    io::{Read, Write},
    net::TcpListener,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

static RELEASE: AtomicBool = AtomicBool::new(false);
static RECEIVED: AtomicBool = AtomicBool::new(false);
struct Probe;
impl Decoder for Probe {
    fn feed(&mut self, _: &[u8], _: bool) -> Result<(i64, bool), DecodeError> {
        RECEIVED.store(true, Ordering::Release);
        Ok((0, false))
    }
    fn finish(&mut self, _: &str) -> (Content, Option<DecodeError>) {
        (
            Content {
                valid: true,
                ..Content::default()
            },
            None,
        )
    }
}
fn decoder(_: &Command) -> Box<dyn Decoder> {
    Box::new(Probe)
}

// Occupy every bridge worker, modelling synchronous Python callback blocking.
// An explicit gate keeps them blocked across Python bytecode GIL handoffs.
#[pyfunction]
fn block_bridge() {
    let runtime = pyo3_async_runtimes::tokio::get_runtime();
    let (tx, rx) = std::sync::mpsc::channel();
    let count = runtime.metrics().num_workers();
    for _ in 0..count {
        let tx = tx.clone();
        runtime.spawn(async move {
            tx.send(()).unwrap();
            let deadline = Instant::now() + Duration::from_secs(5);
            while !RELEASE.load(Ordering::Acquire) && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(1));
            }
        });
    }
    for _ in 0..count {
        rx.recv_timeout(Duration::from_secs(2)).unwrap();
    }
}

#[pyfunction]
fn release_bridge() {
    RELEASE.store(true, Ordering::Release);
}

#[pyfunction]
fn received_while_holding_gil() -> bool {
    let deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline {
        if RECEIVED.load(Ordering::Acquire) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    false
}

#[test]
fn http_reads_continue_when_python_bridge_workers_are_blocked() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/", listener.local_addr().unwrap());
    let server = std::thread::spawn(move || {
        let (mut socket, _) = listener.accept().unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut request = Vec::new();
        while !request.ends_with(b"\r\n\r\n") {
            let mut byte = [0];
            socket.read_exact(&mut byte).unwrap();
            request.push(byte[0]);
        }
        socket
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}")
            .unwrap();
    });
    support::initialize_python();
    Python::attach(|py| {
        let scope = PyDict::new(py);
        scope
            .set_item(
                "release_bridge",
                wrap_pyfunction!(release_bridge, py).unwrap(),
            )
            .unwrap();
        scope
            .set_item("result_source", include_str!("../../native_result.py"))
            .unwrap();
        scope
            .set_item("trial", Py::new(py, NativeTrial::new(decoder)).unwrap())
            .unwrap();
        scope
            .set_item("Body", py.get_type::<PreparedBody>())
            .unwrap();
        scope.set_item("url", url).unwrap();
        scope
            .set_item("block_bridge", wrap_pyfunction!(block_bridge, py).unwrap())
            .unwrap();
        scope
            .set_item(
                "received",
                wrap_pyfunction!(received_while_holding_gil, py).unwrap(),
            )
            .unwrap();
        let code = CString::new(
            r#"
import asyncio, sys, types
for name in ("perf_benchmark", "perf_benchmark.runtime", "perf_benchmark.runtime.http"):
    sys.modules[name] = types.ModuleType(name)
result_module = types.ModuleType("perf_benchmark.runtime.http.native_result")
sys.modules[result_module.__name__] = result_module
exec(result_source, result_module.__dict__)
async def check():
    command = dict(url=url, method='POST', headers={}, session_id='test',
                   connection_scope='trial', verify_ssl=False, body_length=0, stream=False,
                   retain=False, timeout=5., notify_start=False, expectations={},
                   plan=dict(after_outputs=None, after_first_output_ms=None, action='cancel'))
    block_bridge()
    handle = trial.submit(Body((), 0), command)
    independent = received()
    release_bridge()
    await asyncio.wait_for(handle.result(), 5)
    await handle.settled()
    await trial.close()
    assert independent, 'HTTP reads stalled behind blocked Python bridge workers'
    assert trial.active_count == 0
asyncio.run(check())
"#,
        )
        .unwrap();
        py.run(&code, Some(&scope), None).unwrap();
    });
    server.join().unwrap();
}
