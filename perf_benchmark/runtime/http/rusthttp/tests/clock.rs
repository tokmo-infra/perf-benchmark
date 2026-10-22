use std::time::{Duration, Instant};

#[test]
fn native_clock_is_monotonic_and_uses_nanoseconds() {
    let wall = Instant::now();
    let before = perf_rusthttp::now();
    std::thread::sleep(Duration::from_millis(20));
    let after = perf_rusthttp::now();
    let elapsed = wall.elapsed().as_nanos();
    assert!(before > 0);
    assert!(after - before >= 10_000_000);
    assert!((after - before) as u128 <= elapsed);
}
