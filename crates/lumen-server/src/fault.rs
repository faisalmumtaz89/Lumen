//! Fault injection for the engine worker, compiled in by the
//! `fault-injection` feature only; a build without it carries none of this.
//!
//! Two environment variables, read once per process and checked by
//! [`validate`] before the server starts:
//!
//! - `LUMEN_FAULT_PANIC_AT` = `prefill` or `decode:<n>`: the worker panics at
//!   that point of the first job that reaches it (`decode:<n>` is before the
//!   token after `n` generated ones), once per process. The supervisor
//!   answers that client with an error and the next job runs.
//! - `LUMEN_FAULT_PERTURB_US` = `<max>`: every job sleeps a uniformly random
//!   `0..=max` microseconds at the same two points, before the prefill and
//!   before every decode step, to perturb the CPU-side timing around the
//!   device synchronisations.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Point {
    Prefill,
    Decode(usize),
}

struct Config {
    panic_at: Option<Point>,
    perturb_us: u64,
}

fn config() -> &'static Result<Config, String> {
    static CONFIG: OnceLock<Result<Config, String>> = OnceLock::new();
    CONFIG.get_or_init(|| {
        let panic_at = match std::env::var("LUMEN_FAULT_PANIC_AT") {
            Ok(v) => Some(parse_point(&v)?),
            Err(_) => None,
        };
        let perturb_us = match std::env::var("LUMEN_FAULT_PERTURB_US") {
            Ok(v) => v
                .trim()
                .parse::<u64>()
                .ok()
                .filter(|&us| us < u64::MAX)
                .ok_or_else(|| {
                    format!("LUMEN_FAULT_PERTURB_US={v:?} is not a microsecond count")
                })?,
            Err(_) => 0,
        };
        Ok(Config {
            panic_at,
            perturb_us,
        })
    })
}

fn parse_point(value: &str) -> Result<Point, String> {
    let value = value.trim();
    if value == "prefill" {
        return Ok(Point::Prefill);
    }
    value
        .strip_prefix("decode:")
        .and_then(|n| n.parse().ok())
        .map(Point::Decode)
        .ok_or_else(|| format!("LUMEN_FAULT_PANIC_AT={value:?} is not `prefill` or `decode:<n>`"))
}

/// The environment's fault configuration is well-formed. Called once before
/// the server starts, so a bad value is a startup error rather than a panic
/// on every job.
pub fn validate() -> Result<(), String> {
    config().as_ref().map(|_| ()).map_err(Clone::clone)
}

/// Microseconds the hooks have slept in total (a test's proof that the
/// perturbation happened).
pub fn slept_us() -> u64 {
    SLEPT_US.load(Ordering::Relaxed)
}

static SLEPT_US: AtomicU64 = AtomicU64::new(0);

/// Called by the worker before it runs the prefill of a job.
pub fn before_prefill() {
    at(Point::Prefill);
}

/// Called by the worker before it decodes a token of a job, with the number
/// of tokens it has generated for that job so far.
pub fn before_decode_step(generated: usize) {
    at(Point::Decode(generated));
}

fn at(point: Point) {
    let Ok(cfg) = config() else {
        return;
    };
    if cfg.perturb_us > 0 {
        let us = random_below(cfg.perturb_us + 1);
        std::thread::sleep(Duration::from_micros(us));
        SLEPT_US.fetch_add(us, Ordering::Relaxed);
    }
    if cfg.panic_at == Some(point) {
        static FIRED: AtomicBool = AtomicBool::new(false);
        if !FIRED.swap(true, Ordering::Relaxed) {
            panic!("fault injection: {point:?}");
        }
    }
}

/// A xorshift draw in `0..bound`, seeded from the clock: the point is to be
/// unpredictable across runs, not to be a good generator.
fn random_below(bound: u64) -> u64 {
    static STATE: AtomicU64 = AtomicU64::new(0);
    let mut x = STATE.load(Ordering::Relaxed);
    if x == 0 {
        x = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E37_79B9_7F4A_7C15)
            | 1;
    }
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    STATE.store(x, Ordering::Relaxed);
    x % bound
}
