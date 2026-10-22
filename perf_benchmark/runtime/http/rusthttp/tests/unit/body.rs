#[path = "../support/mod.rs"]
mod support;
use super::*;
use pyo3::types::PyBytes;

#[test]
fn bytes_are_shared_and_python_export_is_pinned() {
    support::initialize_python();
    Python::attach(|py| {
        let source = PyBytes::new(py, b"not copied");
        let expected = source.as_bytes().as_ptr();
        let mut body = Body::acquire(py, vec![source.into_any()], 10).unwrap();
        assert_eq!(body.segments[0].as_ptr(), expected);
        let bytes = body.segments.pop().unwrap();
        let clone = bytes.clone();
        drop(bytes);
        assert_eq!(clone.as_ptr(), expected);
        assert_eq!(&clone[..], b"not copied");
        drop(clone);
        assert!(body
            .leases
            .iter()
            .all(|lease| Arc::strong_count(lease) == 1));
    });
}

#[test]
fn waiting_for_python_release_does_not_block_http_polling() {
    support::initialize_python();
    let thread = Python::attach(|py| {
        let source = PyBytes::new(py, b"owned");
        let mut body = Body::acquire(py, vec![source.into_any()], 5).unwrap();
        body.segments.clear();
        let (tx, rx) = std::sync::mpsc::channel();
        let thread = std::thread::spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(async move {
                    let release = tokio::spawn(async move {
                        body.wait_released().await;
                        body
                    });
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                    tx.send(()).unwrap();
                    // Return the body to the GIL-owning thread for release.
                    release.await.unwrap()
                })
        });
        // Keep the GIL unavailable until the I/O runtime has polled its timer.
        rx.recv_timeout(std::time::Duration::from_secs(2)).unwrap();
        thread
    });
    Python::attach(|_| drop(thread.join().unwrap()));
}
