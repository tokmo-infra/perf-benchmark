mod support;
use perf_rusthttp::{
    binding::{NativeTrial, PreparedBody},
    Command, Decoder,
};
use pyo3::{prelude::*, types::PyDict};
use std::ffi::CString;

fn panics(_: &Command) -> Box<dyn Decoder> {
    panic!("intentional test panic")
}

#[test]
fn task_panic_settles_result_and_poisoned_trial() {
    support::initialize_python();
    Python::attach(|py| {
        let scope = PyDict::new(py);
        scope
            .set_item("trial", Py::new(py, NativeTrial::new(panics)).unwrap())
            .unwrap();
        scope
            .set_item("Body", py.get_type::<PreparedBody>())
            .unwrap();
        let code = CString::new(
            r#"
import asyncio
async def check():
    command = dict(url='http://127.0.0.1:1', method='POST', headers={}, session_id='test',
                   connection_scope='trial', verify_ssl=False, body_length=0, stream=False,
                   retain=False, timeout=1., notify_start=True, expectations={},
                   plan=dict(after_outputs=None, after_first_output_ms=None, action='cancel'))
    handle = trial.submit(Body((), 0), command)
    try:
        await asyncio.wait_for(handle.result(), 2)
        assert False, 'panicked task must fail its own result'
    except RuntimeError as error:
        assert 'stopped' in str(error)
    await handle.settled()
    assert trial.active_count == 0
    for operation in (lambda: trial.submit(Body((), 0), command),):
        try:
            operation()
            assert False, 'poisoned trial must reject work'
        except RuntimeError:
            pass
    try:
        await trial.close()
        assert False, 'close must preserve tool failure'
    except RuntimeError:
        pass
asyncio.run(check())
"#,
        )
        .unwrap();
        py.run(&code, Some(&scope), None).unwrap();
    });
}
