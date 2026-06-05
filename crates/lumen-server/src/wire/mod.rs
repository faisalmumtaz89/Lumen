//! Wire-format encoders.
//!
//! Each submodule owns the request DTO, the SSE state machine, and the
//! non-streaming response shape for one external API.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

pub mod anthropic;
pub mod openai;

/// Per-request random seed for sampling when the client does not supply one.
///
/// An OpenAI-/Anthropic-compatible endpoint returns *varied* output by default
/// (reproducibility is opt-in via an explicit `seed`), so an omitted seed must
/// resolve to a fresh value per request rather than a fixed constant.
///
/// A monotonic counter guarantees every request in this process gets a distinct
/// seed — even under concurrent same-nanosecond bursts, which the wall clock
/// alone cannot — and a one-time wall-clock offset makes the sequence differ
/// across process restarts. No bit-mixing is done here: the seed is avalanched
/// downstream by `Xorshift64::new` (`lumen_runtime::sampling`), so distinct
/// inputs are sufficient for distinct, well-separated RNG streams.
pub(crate) fn next_random_seed() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    static START: OnceLock<u64> = OnceLock::new();
    let start = *START.get_or_init(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0)
    });
    start.wrapping_add(COUNTER.fetch_add(1, Ordering::Relaxed))
}

/// Monotonic per-process sequence used to keep response `id`s unique even when
/// several requests share the same `created`/clock value (sub-second burst).
pub(crate) fn next_response_seq() -> u64 {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    SEQ.fetch_add(1, Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn next_random_seed_unique_across_threads() {
        // The counter's raison d'être: concurrent callers must never share a
        // seed. Exercise the atomic RMW under real contention — 8 threads x 20k.
        use std::thread;
        let (threads, per) = (8usize, 20_000usize);
        let handles: Vec<_> = (0..threads)
            .map(|_| thread::spawn(move || (0..per).map(|_| next_random_seed()).collect::<Vec<u64>>()))
            .collect();
        let mut seen = HashSet::with_capacity(threads * per);
        for h in handles {
            for s in h.join().unwrap() {
                assert!(seen.insert(s), "duplicate seed across threads");
            }
        }
        assert_eq!(seen.len(), threads * per);
    }

    #[test]
    fn next_response_seq_is_strictly_unique() {
        // Response ids must never collide even under sub-second concurrent burst.
        let n = 50_000;
        let mut seen = HashSet::with_capacity(n);
        for _ in 0..n {
            assert!(seen.insert(next_response_seq()), "duplicate response seq");
        }
        assert_eq!(seen.len(), n);
    }
}
