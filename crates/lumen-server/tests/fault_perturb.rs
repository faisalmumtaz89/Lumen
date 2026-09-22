//! `LUMEN_FAULT_PERTURB_US=100`: random CPU-side delays at the worker's
//! synchronisation points leave greedy output identical to an unperturbed
//! run. The fault configuration is read once per process, so the unperturbed
//! baseline comes from this same test binary run as a child without it.
#![cfg(feature = "fault-injection")]

mod fault_common;

use fault_common::{boot_engine, drain, job};

const TEST: &str = "perturbed_timing_leaves_greedy_output_identical";
const CHILD: &str = "LUMEN_FAULT_TEST_BASELINE";

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn perturbed_timing_leaves_greedy_output_identical() {
    if std::env::var_os(CHILD).is_some() {
        let (ids, end) = drain(&boot_engine(), job(16)).await;
        assert!(end.is_ok(), "{end:?}");
        println!("BASELINE {ids:?}");
        return;
    }

    // libtest prints its `test <name> ...` prefix on the same line.
    let child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", TEST, "--nocapture", "--test-threads=1"])
        .env(CHILD, "1")
        .env_remove("LUMEN_FAULT_PERTURB_US")
        .env_remove("LUMEN_FAULT_PANIC_AT")
        .output()
        .expect("run the unperturbed baseline");
    assert!(child.status.success(), "baseline child failed");
    let stdout = String::from_utf8_lossy(&child.stdout);
    let baseline = stdout
        .lines()
        .find_map(|l| l.split_once("BASELINE ").map(|(_, ids)| ids))
        .expect("the child prints its token ids")
        .to_string();

    std::env::set_var("LUMEN_FAULT_PERTURB_US", "100");
    let handle = boot_engine();
    let (a, end_a) = drain(&handle, job(16)).await;
    let (b, end_b) = drain(&handle, job(16)).await;
    assert!(end_a.is_ok() && end_b.is_ok(), "{end_a:?} {end_b:?}");
    assert_eq!(a.len(), 16);
    assert_eq!(
        format!("{a:?}"),
        baseline,
        "perturbed output differs from the unperturbed run"
    );
    assert_eq!(a, b);
    // 34 draws from 0..=100 us (a prefill and 16 steps per job): all zero
    // would mean the delays never happened.
    assert!(
        lumen_server::fault::slept_us() > 0,
        "no perturbation was applied"
    );
}
