//! A buffer export pins immutable Python storage until the last HTTP Bytes drops.
use bytes::Bytes;
use pyo3::{
    buffer::PyBuffer,
    exceptions::PyValueError,
    prelude::*,
    types::{PyBytes, PyMemoryView},
};
use std::sync::Arc;
use tokio::sync::Notify;

struct Lease {
    buffer: PyBuffer<u8>,
    released: Arc<Notify>,
}
struct Owner(Option<Arc<Lease>>);
impl AsRef<[u8]> for Owner {
    fn as_ref(&self) -> &[u8] {
        if self.0.as_ref().unwrap().buffer.len_bytes() == 0 {
            return &[];
        }
        // SAFETY: acquisition accepts only immutable bytes or a registered sealed,
        // read-only mmap. PyBuffer pins its exporter, including the underlying map.
        // The export is contiguous u8 storage and lives for every Bytes reference.
        unsafe {
            std::slice::from_raw_parts(
                self.0.as_ref().unwrap().buffer.buf_ptr().cast(),
                self.0.as_ref().unwrap().buffer.len_bytes(),
            )
        }
    }
}
impl Drop for Owner {
    fn drop(&mut self) {
        let lease = self.0.take().unwrap();
        let notify = lease.released.clone();
        drop(lease);
        notify.notify_one();
    }
}
pub struct Body {
    pub segments: Vec<Bytes>,
    leases: Vec<Arc<Lease>>,
    released: Arc<Notify>,
    pub length: usize,
}
impl Body {
    pub fn acquire(
        py: Python<'_>,
        segments: Vec<Bound<'_, PyAny>>,
        length: usize,
    ) -> PyResult<Self> {
        let released = Arc::new(Notify::new());
        let mut result = Self {
            segments: Vec::new(),
            leases: Vec::new(),
            released,
            length,
        };
        let mut actual = 0usize;
        for segment in segments {
            let buffer = PyBuffer::<u8>::get(&segment)?;
            if !buffer.readonly() || !buffer.is_c_contiguous() || buffer.dimensions() != 1 {
                return Err(PyValueError::new_err(
                    "body must be contiguous, one-dimensional, immutable bytes",
                ));
            }
            let root = if segment.is_instance_of::<PyMemoryView>() {
                segment.getattr("obj")?
            } else {
                segment.clone()
            };
            if !root.is_exact_instance_of::<PyBytes>() {
                let trusted: bool = py
                    .import("perf_benchmark.pipeline.preparation.payloads")?
                    .getattr("_is_sealed_mapping")?
                    .call1((&root,))?
                    .extract()?;
                if !trusted || !PyBuffer::<u8>::get(&root)?.readonly() {
                    return Err(PyValueError::new_err(
                        "body mmap must belong to a sealed BlobStore",
                    ));
                }
            }
            actual = actual
                .checked_add(buffer.len_bytes())
                .ok_or_else(|| PyValueError::new_err("body too large"))?;
            let lease = Arc::new(Lease {
                buffer,
                released: result.released.clone(),
            });
            result
                .segments
                .push(Bytes::from_owner(Owner(Some(lease.clone()))));
            result.leases.push(lease);
        }
        if actual != length {
            return Err(PyValueError::new_err("body length mismatch"));
        }
        Ok(result)
    }
    pub async fn wait_released(&self) {
        // Notifications are hints. Register before checking owners to avoid a lost wakeup.
        loop {
            let notified = self.released.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self
                .leases
                .iter()
                .all(|lease| Arc::strong_count(lease) == 1)
            {
                break;
            }
            notified.await;
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/body.rs"]
mod tests;
