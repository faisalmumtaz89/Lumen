//! Q6_K HEAD DECISION GATE — a measurement harness, not a feature.
//!
//! # What this decides
//!
//! Whether candidate C3 (native Q6_K `output.weight`) is worth wiring at all.
//! It is NOT wired to the head dispatch, and must not be until this clears.
//!
//! The incumbent Q8_0 dp4a head runs ~613 us moving ~1080 MiB, i.e. ~1762 GB/s
//! or ~86% MBU on an A100-80GB. Native Q6_K moves 795.7 MiB, so:
//!
//! | outcome        | required effective rate |
//! |----------------|-------------------------|
//! | break-even     | 1361 GB/s  (67% MBU)    |
//! | +1% end-to-end | 1543 GB/s  (76% MBU)    |
//! | continue-build | **1500 GB/s** (74% MBU) |
//!
//! Below the continue-build threshold the head STAYS on requant-Q8_0. Fewer
//! bytes does not imply faster here: with F32 activations the kernel is
//! compute-bound (one `fma` per element, no dp4a fold), which is exactly why a
//! byte-count argument cannot settle this and a measurement must.
//!
//! # Why a microbench rather than an end-to-end A/B
//!
//! An end-to-end run cannot attribute a 1-2% token delta to one kernel, and the
//! head is only ~9% of a token. This measures the SAME kernel body at the SAME
//! shape in isolation, so the number is directly comparable to the 613 us
//! incumbent. Per the busy-time rule an isolated kernel µs is NOT a token-level
//! claim — it is a GO/NO-GO screen for whether the token-level lever can exist.
//!
//! # Running it (A100 container)
//!
//! ```text
//! cargo test --release -p lumen-runtime --features cuda \
//!     --test cuda_q6k_head_microbench -- --ignored --nocapture
//! ```
//!
//! `#[ignore]` because it allocates ~800 MB of VRAM and takes seconds; it must
//! never run as part of the ordinary suite. It self-skips (rather than failing)
//! on a host with no CUDA device, so the file is safe to keep in-tree.
//!
//! Env knobs: `LUMEN_Q6K_BENCH_ITERS` (default 50),
//! `LUMEN_Q6K_BENCH_WARMUP` (default 10).

#![cfg(feature = "cuda")]

use lumen_runtime::cuda::ffi::CudaDevice;
use lumen_runtime::cuda::shaders;
use lumen_runtime::q6k_ref::{Q6K_BLOCK_BYTE, Q6K_BLOCK_ELEM};
use std::time::Instant;

/// The real 9B head shape.
const VOCAB: usize = 248_320;
const HIDDEN: usize = 4096;

/// codex-sol's r3 thresholds, in GB/s of Q6_K weight bytes moved.
const BREAK_EVEN_GBPS: f64 = 1361.0;
const PLUS_ONE_PCT_GBPS: f64 = 1543.0;
const CONTINUE_BUILD_GBPS: f64 = 1500.0;
/// Incumbent Q8_0 dp4a head, for the direct µs comparison.
const Q8_INCUMBENT_US: f64 = 613.0;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Synthetic Q6_K row bytes with non-degenerate content: distinct nibbles per
/// byte and non-zero signed scales, so the kernel cannot be accidentally fast on
/// trivial data (e.g. an all-zero `d` short-circuiting nothing, but zero scales
/// would still be a misleading input).
fn synth_q6k_rows(n_rows: usize, nb: usize) -> Vec<u8> {
    let mut out = vec![0u8; n_rows * nb * Q6K_BLOCK_BYTE];
    // Build ONE row and replicate: the kernel's cost is per-byte-streamed, and
    // building 834 MB of distinct pseudorandom bytes on the host would dominate
    // the test's own runtime without changing what is measured.
    let mut row = Vec::with_capacity(nb * Q6K_BLOCK_BYTE);
    for b in 0..nb {
        for i in 0..128 {
            row.push((((i + b) % 15) as u8) | ((((i + 7 + b) % 15) as u8) << 4));
        }
        for i in 0..64 {
            row.push(((i * 7 + b) % 256) as u8);
        }
        for i in 0..16 {
            let mag = 1 + ((i + b) % 100) as i8;
            row.push((if i % 2 == 0 { mag } else { -mag }) as u8);
        }
        row.extend_from_slice(&0x1C00u16.to_le_bytes()); // f16 ~0.00098
    }
    assert_eq!(row.len(), nb * Q6K_BLOCK_BYTE);
    for r in 0..n_rows {
        let off = r * nb * Q6K_BLOCK_BYTE;
        out[off..off + row.len()].copy_from_slice(&row);
    }
    out
}

struct Measurement {
    label: &'static str,
    us: f64,
    gbps: f64,
}

#[test]
#[ignore = "allocates ~800 MB VRAM; run explicitly on an A100 with --ignored --nocapture"]
fn q6k_head_rate_decision_gate() {
    // CACHE ISOLATION -- must run BEFORE any compile touches the PTX cache.
    //
    // This test previously shared production's cache keys and poisoned them. When
    // it compiled a shader that genuinely could not build, the driver rejected the
    // PTX and a driver-reject marker landed in the PERSISTENT /cache/ptx volume;
    // every later production launch then skipped `matvec_q6_k_f32` and
    // `dequant_q6_k_to_f32` as "doomed" until the volume was purged by hand.
    //
    // Note the cache key did NOT differ from production's -- see the report. This
    // test used `compile_and_load`, the exact call the production `load_fn` uses
    // (arch `None` -> "default", fast_math false) on the exact same source, so the
    // digest was production's by construction. Widening the key would not have
    // helped; only isolating the DIRECTORY does.
    //
    // A per-run tmpdir means a bad compile here can never be visible to a
    // production process, whatever the key.
    let cache_dir =
        std::env::temp_dir().join(format!("lumen_q6k_microbench_ptx_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&cache_dir);
    std::env::set_var("LUMEN_CACHE_DIR", &cache_dir);
    // Belt and braces: even inside the tmpdir, never trust a marker this run.
    std::env::set_var("LUMEN_CUDA_PTX_REJECT_TTL_SECS", "0");
    println!("PTX cache isolated to {}", cache_dir.display());

    let device = match std::panic::catch_unwind(|| CudaDevice::new(0)) {
        Ok(Ok(d)) => d,
        Ok(Err(e)) => {
            eprintln!("skipping: CUDA init failed ({e}) -- no device on this host");
            return;
        }
        Err(_) => {
            eprintln!("skipping: CUDA driver not loadable on this host");
            return;
        }
    };

    let iters = env_usize("LUMEN_Q6K_BENCH_ITERS", 50);
    let warmup = env_usize("LUMEN_Q6K_BENCH_WARMUP", 10);
    let nb = HIDDEN / Q6K_BLOCK_ELEM;
    let weight_bytes = VOCAB * nb * Q6K_BLOCK_BYTE;

    println!("\n===== Q6_K HEAD DECISION GATE =====");
    println!("shape        [out={VOCAB} x in={HIDDEN}]  nb={nb}");
    println!(
        "weight bytes {weight_bytes} ({:.1} MiB, {:.4} B/weight)",
        weight_bytes as f64 / 1048576.0,
        weight_bytes as f64 / (VOCAB * HIDDEN) as f64
    );
    println!("iters {iters} (warmup {warmup})");
    println!(
        "thresholds   break-even {BREAK_EVEN_GBPS} GB/s | +1% {PLUS_ONE_PCT_GBPS} GB/s | \
         CONTINUE-BUILD {CONTINUE_BUILD_GBPS} GB/s"
    );
    println!("incumbent    Q8_0 dp4a head {Q8_INCUMBENT_US} us\n");

    // Plain `load_fn` equivalent: these two kernels use only baseline PTX.
    let module = match device.compile_and_load(shaders::MATVEC_Q6_K_F32_KERNEL_SOURCE) {
        Ok(m) => m,
        Err(e) => panic!("NVRTC failed to compile matvec_q6_k_f32 shader: {e}"),
    };

    let host_w = synth_q6k_rows(VOCAB, nb);
    assert_eq!(host_w.len(), weight_bytes);
    let d_w = device.htod_copy(&host_w).expect("upload Q6_K head");
    drop(host_w);
    let x: Vec<f32> = (0..HIDDEN).map(|i| ((i % 17) as f32 - 8.0) / 8.0).collect();
    let d_x = device.htod_copy(&x).expect("upload activations");
    let mut d_out = device.alloc_zeros::<f32>(VOCAB).expect("alloc logits");

    // Q8_1-encoded activations for the dp4a variant. The head ALREADY runs int8
    // activations today (it has no Q4ProjectionFamily entry, so the F32 plan
    // never covered it), which is why the dp4a kernel is a legitimate head
    // candidate and not a policy change -- and why measuring only the F32
    // variant would confirm a NO-GO I had already predicted while leaving the
    // real question unanswered.
    let mut q8_1 = vec![0u8; (HIDDEN / 32) * 36];
    for b in 0..(HIDDEN / 32) {
        let blk = &x[b * 32..(b + 1) * 32];
        let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = if amax > 0.0 { amax / 127.0 } else { 0.0 };
        let base = b * 36;
        let d_bits = lumen_runtime::q6k_ref::f32_to_f16_bits_rne(d);
        q8_1[base..base + 2].copy_from_slice(&d_bits.to_le_bytes());
        let mut sum = 0.0f32;
        for (k, &v) in blk.iter().enumerate() {
            let q = if d > 0.0 {
                (v / d).round().clamp(-127.0, 127.0) as i8
            } else {
                0
            };
            q8_1[base + 4 + k] = q as u8;
            sum += (q as f32) * d;
        }
        let s_bits = lumen_runtime::q6k_ref::f32_to_f16_bits_rne(sum);
        q8_1[base + 2..base + 4].copy_from_slice(&s_bits.to_le_bytes());
    }
    let d_q8_1 = device.htod_copy(&q8_1).expect("upload Q8_1 activations");

    // The dp4a kernel needs the sm80 fast-math pipeline; plain compile would
    // reject `dp4a.s32.s32`. Compiled separately so a failure here cannot mask
    // the F32 numbers.
    let module_dp4a = device.compile_and_load_with_arch_fast_math(
        shaders::MATVEC_Q6_K_Q8_1_KERNEL_SOURCE,
        "compute_80",
    );

    let mut results: Vec<Measurement> = Vec::new();

    for (label, sym, nr) in [
        ("matvec_q6_k_f32     (NR=1)", "matvec_q6_k_f32", 1u32),
        ("matvec_q6_k_f32_nr4 (NR=4)", "matvec_q6_k_f32_nr4", 4u32),
    ] {
        let f = match module.load_function(sym) {
            Ok(f) => f,
            Err(e) => {
                println!("{label}: SKIP (symbol {sym} not loadable: {e})");
                continue;
            }
        };
        let grid = (VOCAB as u32).div_ceil(nr);
        let out_u32 = VOCAB as u32;
        let in_u32 = HIDDEN as u32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };

        let launch = |d_out: &mut cudarc::driver::CudaSlice<f32>| {
            use cudarc::driver::PushKernelArg;
            unsafe {
                device
                    .stream
                    .launch_builder(&f)
                    .arg(&d_w)
                    .arg(&d_x)
                    .arg(d_out)
                    .arg(&out_u32)
                    .arg(&in_u32)
                    .launch(cfg)
            }
            .expect("kernel launch");
        };

        for _ in 0..warmup {
            launch(&mut d_out);
        }
        device.synchronize().expect("warmup sync");

        let t0 = Instant::now();
        for _ in 0..iters {
            launch(&mut d_out);
        }
        device.synchronize().expect("timed sync");
        let elapsed = t0.elapsed();

        let us = elapsed.as_secs_f64() * 1e6 / iters as f64;
        let gbps = weight_bytes as f64 / (us * 1e-6) / 1e9;
        println!(
            "{label}  grid={grid:>7}  {us:8.1} us  {gbps:7.1} GB/s  \
             ({:.0}% MBU @2039)  vs Q8_0 {:+.1} us",
            gbps / 2039.0 * 100.0,
            us - Q8_INCUMBENT_US
        );
        results.push(Measurement { label, us, gbps });
    }

    // dp4a variant (int8 activations). Same weights, same shape.
    match module_dp4a
        .as_ref()
        .map(|m| m.load_function("matvec_q6_k_q8_1"))
    {
        Ok(Ok(f)) => {
            use cudarc::driver::PushKernelArg;
            let out_u32 = VOCAB as u32;
            let in_u32 = HIDDEN as u32;
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (VOCAB as u32, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            let launch = |d_out: &mut cudarc::driver::CudaSlice<f32>| {
                unsafe {
                    device
                        .stream
                        .launch_builder(&f)
                        .arg(&d_w)
                        .arg(&d_q8_1)
                        .arg(d_out)
                        .arg(&out_u32)
                        .arg(&in_u32)
                        .launch(cfg)
                }
                .expect("dp4a kernel launch");
            };
            for _ in 0..warmup {
                launch(&mut d_out);
            }
            device.synchronize().expect("dp4a warmup sync");
            let t0 = Instant::now();
            for _ in 0..iters {
                launch(&mut d_out);
            }
            device.synchronize().expect("dp4a timed sync");
            let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            let gbps = weight_bytes as f64 / (us * 1e-6) / 1e9;
            let label = "matvec_q6_k_q8_1    (dp4a)";
            println!(
                "{label}  grid={:>7}  {us:8.1} us  {gbps:7.1} GB/s                   ({:.0}% MBU @2039)  vs Q8_0 {:+.1} us",
                VOCAB,
                gbps / 2039.0 * 100.0,
                us - Q8_INCUMBENT_US
            );
            results.push(Measurement { label, us, gbps });
        }
        Ok(Err(e)) => println!("matvec_q6_k_q8_1    (dp4a)  SKIP (symbol: {e})"),
        Err(e) => println!("matvec_q6_k_q8_1    (dp4a)  SKIP (sm80 fast-math compile: {e})"),
    }

    assert!(!results.is_empty(), "no Q6_K head kernel could be loaded");

    // Sanity: the output must not be all zeros, or we timed a kernel that did
    // nothing (a launch-config or arg-order error reads as "very fast").
    let logits = device.dtoh_copy(&d_out).expect("readback logits");
    let nonzero = logits.iter().filter(|v| **v != 0.0).count();
    assert!(
        nonzero > VOCAB / 2,
        "only {nonzero}/{VOCAB} logits are non-zero -- the timed kernel did not \
         write most rows, so the rate above is meaningless"
    );
    let finite = logits.iter().all(|v| v.is_finite());
    assert!(finite, "logits contain NaN/Inf -- kernel produced garbage");

    let best = results
        .iter()
        .max_by(|a, b| a.gbps.partial_cmp(&b.gbps).unwrap())
        .unwrap();

    println!("\n----- VERDICT -----");
    println!(
        "best: {} at {:.1} GB/s ({:.1} us)",
        best.label, best.gbps, best.us
    );
    let verdict = if best.gbps >= CONTINUE_BUILD_GBPS {
        "CONTINUE-BUILD: clears 1500 GB/s. Wiring the head dispatch is justified."
    } else if best.gbps >= PLUS_ONE_PCT_GBPS {
        "MARGINAL: clears +1% (1543) but not the 1500 continue-build bar -- \
         re-read, thresholds overlap; treat as CONTINUE with caution."
    } else if best.gbps >= BREAK_EVEN_GBPS {
        "NO-GO (marginal): clears break-even (1361) but not +1% (1543). \
         Head STAYS requant-Q8_0; the win is inside the noise floor."
    } else {
        "NO-GO: below break-even (1361 GB/s). Head STAYS requant-Q8_0. \
         Native Q6_K is SLOWER than the incumbent despite moving fewer bytes."
    };
    println!("{verdict}");
    println!(
        "NOTE: isolated-kernel rate is a GO/NO-GO screen, NOT a token-level \
         throughput claim (busy-time rule). The head is ~9% of a token."
    );
    println!("===================\n");

    // The gate REPORTS; it must not fail the suite on a NO-GO, because a NO-GO is
    // a valid and useful answer. It fails only if the measurement itself is
    // unsound, which the assertions above cover.
}
