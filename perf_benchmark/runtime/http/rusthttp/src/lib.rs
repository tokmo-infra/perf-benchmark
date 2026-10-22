pub mod binding;
mod body;
pub mod http;
pub mod sse;
mod transport;
pub mod types;
use std::collections::BTreeMap;
pub use types::Command;

#[derive(Default)]
pub struct Content {
    pub kind: &'static str,
    pub chars: i64,
    pub usage: BTreeMap<String, i64>,
    pub finish: String,
    pub continuation: Option<serde_json::Value>,
    pub valid: bool,
}
#[derive(Debug, Clone, Copy)]
pub enum DecodeError {
    Protocol,
    Incomplete,
}
impl From<serde_json::Error> for DecodeError {
    fn from(_: serde_json::Error) -> Self {
        Self::Protocol
    }
}
pub trait Decoder: Send {
    fn feed(&mut self, data: &[u8], streaming: bool) -> Result<(i64, bool), DecodeError>;
    fn finish(&mut self, status: &str) -> (Content, Option<DecodeError>);
}
#[cfg(unix)]
pub fn now() -> i64 {
    use nix::time::{clock_gettime, ClockId};
    #[cfg(target_os = "macos")]
    let clock = ClockId::from_raw(nix::libc::CLOCK_UPTIME_RAW);
    #[cfg(not(target_os = "macos"))]
    let clock = ClockId::CLOCK_MONOTONIC;
    let value = clock_gettime(clock).expect("monotonic clock unavailable");
    value.tv_sec() * 1_000_000_000 + value.tv_nsec()
}

#[cfg(windows)]
pub fn now() -> i64 {
    use std::sync::LazyLock;
    use windows_sys::Win32::System::Performance::{
        QueryPerformanceCounter, QueryPerformanceFrequency,
    };
    static FREQUENCY: LazyLock<i64> = LazyLock::new(|| {
        let mut frequency = 0;
        // SAFETY: the API writes one i64 to this live local variable.
        assert_ne!(unsafe { QueryPerformanceFrequency(&mut frequency) }, 0);
        assert!(frequency > 0);
        frequency
    });
    let frequency = *FREQUENCY;
    let mut ticks = 0;
    // SAFETY: the API writes one i64 to this live local variable.
    assert_ne!(unsafe { QueryPerformanceCounter(&mut ticks) }, 0);
    // CPython uses this same counter and integer conversion, without an IPC offset.
    i64::try_from(i128::from(ticks) * 1_000_000_000 / i128::from(frequency))
        .expect("performance clock overflow")
}
// Optional JSON containers treat null as absent, without making missing usage
// look like a measured zero.
pub fn null_default<'de, T, D>(deserializer: D) -> Result<T, D::Error>
where
    T: serde::Deserialize<'de> + Default,
    D: serde::Deserializer<'de>,
{
    Ok(<Option<T> as serde::Deserialize>::deserialize(deserializer)?.unwrap_or_default())
}
pub fn error_present<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<bool, D::Error> {
    let _ = <serde::de::IgnoredAny as serde::Deserialize>::deserialize(deserializer)?;
    Ok(true)
}
