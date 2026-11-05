//! First-output safety clock starts at dispatch, not scheduler/pool wait.
use std::sync::atomic::{AtomicI64, Ordering};
use tokio::sync::Notify;
use tokio::time::{sleep, Duration};

#[derive(Default)]
pub struct TtftWatchdog {
    sent: AtomicI64,
    first: AtomicI64,
    changed: Notify,
}
impl TtftWatchdog {
    pub fn sent(&self, at: i64) {
        self.sent.store(at, Ordering::Release);
        self.changed.notify_one();
    }
    pub fn first(&self, at: i64) {
        let _ = self
            .first
            .compare_exchange(0, at, Ordering::AcqRel, Ordering::Acquire);
    }
    pub fn exceeded(&self, now: i64, limit: i64) -> bool {
        let sent = self.sent.load(Ordering::Acquire);
        let first = self.first.load(Ordering::Acquire);
        sent > 0 && (if first > 0 { first } else { now }) - sent > limit
    }
    pub async fn wait(&self, limit: Option<i64>) {
        let Some(limit) = limit else {
            std::future::pending::<()>().await;
            return;
        };
        self.changed.notified().await;
        let sent = self.sent.load(Ordering::Acquire);
        sleep(Duration::from_nanos(
            (sent + limit + 1 - crate::now()).max(0) as u64,
        ))
        .await;
        if !self.exceeded(crate::now(), limit) {
            std::future::pending::<()>().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unsent_queue_never_exceeds_ttft() {
        assert!(!TtftWatchdog::default().exceeded(1_000_000_000_000, 60));
    }
    #[test]
    fn pending_first_output_and_strict_boundary() {
        let w = TtftWatchdog::default();
        w.sent(100);
        assert!(!w.exceeded(160, 60));
        assert!(w.exceeded(161, 60));
    }
    #[test]
    fn early_first_output_disarms_but_late_does_not() {
        let early = TtftWatchdog::default();
        early.sent(100);
        early.first(150);
        early.first(999);
        assert!(!early.exceeded(10_000, 60));
        let late = TtftWatchdog::default();
        late.sent(100);
        late.first(161);
        assert!(late.exceeded(200, 60));
    }
    #[tokio::test]
    async fn pending_native_request_fires_without_terminal_result() {
        let w = TtftWatchdog::default();
        w.sent(crate::now());
        assert!(
            tokio::time::timeout(Duration::from_millis(100), w.wait(Some(1_000_000)))
                .await
                .is_ok()
        );
    }
    #[tokio::test]
    async fn received_first_output_disarms_timer() {
        let w = TtftWatchdog::default();
        w.sent(crate::now());
        w.first(crate::now());
        assert!(
            tokio::time::timeout(Duration::from_millis(10), w.wait(Some(1_000_000)))
                .await
                .is_err()
        );
    }
}
