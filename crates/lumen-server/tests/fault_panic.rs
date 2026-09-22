//! `LUMEN_FAULT_PANIC_AT=decode:2`: the first job is answered with the
//! supervisor's error after exactly two tokens, and the next job is served.
#![cfg(feature = "fault-injection")]

mod fault_common;

use std::time::{Duration, Instant};

use fault_common::{boot_engine, drain, job};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_panic_at_the_third_token_is_answered_and_the_next_job_runs() {
    std::env::set_var("LUMEN_FAULT_PANIC_AT", "decode:2");
    let handle = boot_engine();

    let (ids, end) = drain(&handle, job(8)).await;
    assert_eq!(ids.len(), 2, "two tokens precede the injected panic");
    let err = end.expect_err("the panicking job ends in an error");
    assert!(err.starts_with("engine recovered from panic"), "{err}");

    let started = Instant::now();
    let (ids, end) = drain(&handle, job(8)).await;
    assert!(end.is_ok(), "the next job completes: {end:?}");
    assert_eq!(ids.len(), 8);
    assert!(started.elapsed() < Duration::from_secs(5));
}
