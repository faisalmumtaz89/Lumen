//! Isolated microbench comparing NR=1 / NR=2 / NR=4 variants of the
//! BF16 fused gate+up+SwiGLU kernel.
//!
//! Gate criterion: per-kernel improvement >= +5% at Qwen3.5-9B
//! FFN gate+up shape (M=131, K=4096, N=12288) before integration. End-to-end
//! gate (separate): >= +1.5% BF16 prefill paired AND multi-prompt coherent.
//!
//! This test:
//!   1. Compiles all three kernel variants (NR=1, NR=2 baseline, NR=4) as
//!      both BC (boundary-checked) and aligned pipelines via function
//!      constants.
//!   2. Allocates synthetic BF16 gate+up weight buffers and an F32 input
//!      activation buffer at the Qwen3.5-9B FFN shape.
//!   3. Warms up each pipeline, then times each via wall-clock over many
//!      iterations.
//!   4. Reports mean+median+min times for each NR variant and the speedup
//!      ratio relative to NR=2 baseline.
//!   5. Asserts NR=1 and NR=4 outputs match NR=2 within a tight tolerance
//!      (BF16 ULP-level; all variants compute the same algorithm with the
//!      same F32 accumulators and float epilogue).
//!
//! Run with:
//!   cargo test --release --lib -p lumen-runtime \
//!     metal::tests::microbench -- --ignored --nocapture
//!
//! M=131 chosen to match the production prompt length used by the
//! reference bench (131 = "hello "*120 token count). K=4096,
//! N=12288 match Qwen3.5-9B FFN gate+up dimensions.

use crate::metal::MetalF32Backend;
use crate::metal::ffi::{MTLSize, MetalFunctionConstantValues};
use crate::metal::shaders::METAL_SHADER_SOURCE;

const TILE_N: u64 = 32; // unchanged across all NR variants

/// Convert a float to a BF16 bit pattern (round-to-nearest-even via the
/// host's f32->bf16 lossy cast helper). Matches the LBC repack used in
/// `convert::dequant`.
fn f32_to_bf16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    // Round-to-nearest-even: add 0x7FFF + lsb_of_truncated_high16
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7FFF + lsb);
    (rounded >> 16) as u16
}

/// Build a synthetic BF16 weight buffer of shape [N, K] with small magnitude
/// values to avoid accumulator overflow at large K.
fn make_synthetic_bf16(n_rows: usize, k_cols: usize, seed: u32) -> Vec<u16> {
    let mut s: u64 = seed as u64;
    let mut out = Vec::with_capacity(n_rows * k_cols);
    for _ in 0..(n_rows * k_cols) {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (s >> 32) as u32;
        // Map to [-0.1, 0.1]: keeps |gate|, |up| products small at large K.
        let f = ((u as f32) / (u32::MAX as f32) * 0.2 - 0.1) as f32;
        out.push(f32_to_bf16_bits(f));
    }
    out
}

fn make_random_f32_small(n: usize, seed: u32) -> Vec<f32> {
    let mut s: u64 = seed as u64;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (s >> 32) as u32;
        // [-0.5, 0.5]
        let f = (u as f32) / (u32::MAX as f32) - 0.5;
        v.push(f);
    }
    v
}

#[ignore = " microbench — run with --ignored for isolated kernel timing"]
#[test]
fn microbench_bf16_gate_up_nr_sweep() {
    const M: usize = 131;
    const K: usize = 4096;   // hidden_dim
    const N: usize = 12288;  // inter_dim
    const WARMUP: usize = 8;
    const ITERS: usize = 64;

    let backend = MetalF32Backend::new().expect("Metal backend create");
    let lib = backend.device
        .new_library_with_source(METAL_SHADER_SOURCE)
        .expect("Metal lib compile");

    // BC variant (FC_BC_M=FC_BC_N=FC_BC_K=true) — matches production for M=131
    // since M is not a multiple of 16, 32 or 64.
    let fcv = MetalFunctionConstantValues::new();
    fcv.set_bool(true, 10);
    fcv.set_bool(true, 11);
    fcv.set_bool(true, 12);

    let f_nr1 = lib.get_function_with_constants(
        "bf16_matmul_gate_up_swiglu_fused_nr1", &fcv
    ).expect("NR=1 kernel function");
    let pso_nr1 = backend.device.new_compute_pipeline_state(&f_nr1)
        .expect("NR=1 PSO");

    let f_nr2 = lib.get_function_with_constants(
        "bf16_matmul_gate_up_swiglu_fused", &fcv
    ).expect("NR=2 baseline kernel function");
    let pso_nr2 = backend.device.new_compute_pipeline_state(&f_nr2)
        .expect("NR=2 baseline PSO");

    let f_nr4 = lib.get_function_with_constants(
        "bf16_matmul_gate_up_swiglu_fused_nr4", &fcv
    ).expect("NR=4 kernel function");
    let pso_nr4 = backend.device.new_compute_pipeline_state(&f_nr4)
        .expect("NR=4 PSO");

    // Synthetic weights and activations. Seeds chosen so gate and up differ.
    let w_gate = make_synthetic_bf16(N, K, 1);
    let w_up   = make_synthetic_bf16(N, K, 2);
    let x      = make_random_f32_small(M * K, 3);

    let w_gate_buf = backend.device.new_buffer_with_bytes(unsafe {
        std::slice::from_raw_parts(w_gate.as_ptr() as *const u8, w_gate.len() * 2)
    }).expect("w_gate buf");
    let w_up_buf = backend.device.new_buffer_with_bytes(unsafe {
        std::slice::from_raw_parts(w_up.as_ptr() as *const u8, w_up.len() * 2)
    }).expect("w_up buf");
    let x_buf = backend.device.new_buffer_with_bytes(unsafe {
        std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4)
    }).expect("x buf");

    let y_nr1_buf = backend.device.new_buffer(M * N * 4).expect("y nr1 buf");
    let y_nr2_buf = backend.device.new_buffer(M * N * 4).expect("y nr2 buf");
    let y_nr4_buf = backend.device.new_buffer(M * N * 4).expect("y nr4 buf");

    let m_u32 = M as u32;
    let n_u32 = N as u32;
    let k_u32 = K as u32;

    // Dispatch closure parameterised on (PSO, output buffer, tile_m, shmem).
    let dispatch = |pso: &crate::metal::ffi::MetalPipelineState,
                    y_buf: &crate::metal::ffi::MetalBuffer,
                    tile_m: u64,
                    shmem_bytes: u64| -> u128 {
        let cmd = backend.queue.new_command_buffer().expect("cmd buf");
        let enc = cmd.new_compute_encoder().expect("enc");
        enc.set_pipeline_state(pso);
        enc.set_threadgroup_memory_length(shmem_bytes, 0);
        enc.set_buffer(&w_gate_buf, 0, 0);
        enc.set_buffer(&x_buf, 0, 1);
        enc.set_buffer(y_buf, 0, 2);
        enc.set_bytes(&m_u32.to_le_bytes(), 3);
        enc.set_bytes(&n_u32.to_le_bytes(), 4);
        enc.set_bytes(&k_u32.to_le_bytes(), 5);
        enc.set_buffer(&w_up_buf, 0, 6);
        enc.dispatch_threadgroups(
            MTLSize::new((N as u64).div_ceil(TILE_N), (M as u64).div_ceil(tile_m), 1),
            MTLSize::new(128, 1, 1),
        );
        enc.end_encoding();
        let t0 = std::time::Instant::now();
        cmd.commit_and_wait();
        t0.elapsed().as_nanos()
    };

    // Per-variant TILE_M and shmem in bytes (mirror dispatch wiring in
    // prefill_encode.rs).
    let (tile_m_1, shmem_1) = (16u64, 10240u64);
    let (tile_m_2, shmem_2) = (32u64, 12288u64);
    let (tile_m_4, shmem_4) = (64u64, 16384u64);

    // Warm-up: prime caches and force kernel JIT through the driver before
    // we sample timings. Interleaved order prevents one variant from being
    // systematically penalised by cold caches.
    for _ in 0..WARMUP {
        let _ = dispatch(&pso_nr1, &y_nr1_buf, tile_m_1, shmem_1);
        let _ = dispatch(&pso_nr2, &y_nr2_buf, tile_m_2, shmem_2);
        let _ = dispatch(&pso_nr4, &y_nr4_buf, tile_m_4, shmem_4);
    }

    // Correctness: NR=1 and NR=4 must match NR=2 within a tight tolerance.
    // The algorithm is identical (same float accumulators, same SwiGLU
    // computation, same input data), but BF16 weight load + MMA pipeline
    // ordering MAY differ across NR variants, so we allow a small per-element
    // BF16-ULP-level tolerance.
    let mut y_nr1_out = vec![0.0f32; M * N];
    let mut y_nr2_out = vec![0.0f32; M * N];
    let mut y_nr4_out = vec![0.0f32; M * N];
    y_nr1_buf.read_f32(&mut y_nr1_out);
    y_nr2_buf.read_f32(&mut y_nr2_out);
    y_nr4_buf.read_f32(&mut y_nr4_out);

    let check_diff = |a: &[f32], b: &[f32], label: &str| -> (f32, f64) {
        let mut max_abs_diff = 0f32;
        let mut sum_sq = 0f64;
        for (x, y) in a.iter().zip(b.iter()) {
            let d = (x - y).abs();
            if d > max_abs_diff { max_abs_diff = d; }
            sum_sq += (d as f64) * (d as f64);
        }
        let rmse = (sum_sq / (a.len() as f64)).sqrt();
        eprintln!("[microbench] correctness {label}: max_abs_diff={:.6e}, rmse={:.6e}",
            max_abs_diff, rmse);
        (max_abs_diff, rmse)
    };

    let (max_diff_1, _) = check_diff(&y_nr1_out, &y_nr2_out, "NR=1 vs NR=2");
    let (max_diff_4, _) = check_diff(&y_nr4_out, &y_nr2_out, "NR=4 vs NR=2");
    // Tolerance: per-element diff <= 1e-2. Each kernel does ~K=4096 BF16 MMA
    // accumulations into an F32 accumulator. BF16 has ~7 bits of mantissa so
    // each per-element absolute error is on the order of 1e-3 * |product|
    // and 4096 such products with random small values gives an RMSE of ~1e-3.
    // Anything beyond 1e-2 indicates a genuine algorithmic divergence, not
    // float-pipeline-ordering noise.
    assert!(max_diff_1 < 1e-2,
        "NR=1 diverges from NR=2 beyond BF16 tolerance: max_abs_diff={:.6e}",
        max_diff_1);
    assert!(max_diff_4 < 1e-2,
        "NR=4 diverges from NR=2 beyond BF16 tolerance: max_abs_diff={:.6e}",
        max_diff_4);

    // Bench. Use interleaved ordering (1-2-4, 1-2-4, ...) so thermal drift
    // affects all three variants symmetrically across the run.
    let mut t_nr1: Vec<u128> = Vec::with_capacity(ITERS);
    let mut t_nr2: Vec<u128> = Vec::with_capacity(ITERS);
    let mut t_nr4: Vec<u128> = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        t_nr1.push(dispatch(&pso_nr1, &y_nr1_buf, tile_m_1, shmem_1));
        t_nr2.push(dispatch(&pso_nr2, &y_nr2_buf, tile_m_2, shmem_2));
        t_nr4.push(dispatch(&pso_nr4, &y_nr4_buf, tile_m_4, shmem_4));
    }
    t_nr1.sort();
    t_nr2.sort();
    t_nr4.sort();
    let med_1 = t_nr1[ITERS / 2];
    let med_2 = t_nr2[ITERS / 2];
    let med_4 = t_nr4[ITERS / 2];
    let mean = |v: &[u128]| v.iter().sum::<u128>() / (v.len() as u128);
    let mean_1 = mean(&t_nr1);
    let mean_2 = mean(&t_nr2);
    let mean_4 = mean(&t_nr4);
    let min_1 = *t_nr1.first().unwrap();
    let min_2 = *t_nr2.first().unwrap();
    let min_4 = *t_nr4.first().unwrap();

    let speedup = |baseline: u128, treatment: u128| -> f64 {
        (baseline as f64) / (treatment as f64)
    };
    let su_med_1 = speedup(med_2, med_1);
    let su_med_4 = speedup(med_2, med_4);
    let su_mean_1 = speedup(mean_2, mean_1);
    let su_mean_4 = speedup(mean_2, mean_4);
    let su_min_1 = speedup(min_2, min_1);
    let su_min_4 = speedup(min_2, min_4);

    eprintln!("[microbench] Qwen3.5-9B FFN gate+up shape M={} K={} N={}:", M, K, N);
    eprintln!("[microbench] NR=1: median={} ns, mean={} ns, min={} ns",
        med_1, mean_1, min_1);
    eprintln!("[microbench] NR=2: median={} ns, mean={} ns, min={} ns (BASELINE)",
        med_2, mean_2, min_2);
    eprintln!("[microbench] NR=4: median={} ns, mean={} ns, min={} ns",
        med_4, mean_4, min_4);
    eprintln!("[microbench] NR=1 / NR=2: median={:+.3}%, mean={:+.3}%, min={:+.3}%",
        (su_med_1 - 1.0) * 100.0,
        (su_mean_1 - 1.0) * 100.0,
        (su_min_1 - 1.0) * 100.0);
    eprintln!("[microbench] NR=4 / NR=2: median={:+.3}%, mean={:+.3}%, min={:+.3}%",
        (su_med_4 - 1.0) * 100.0,
        (su_mean_4 - 1.0) * 100.0,
        (su_min_4 - 1.0) * 100.0);

    // Gate evaluation: >= +5% per-kernel improvement required
    // for integration.
    eprintln!("[microbench] GATE: per-kernel improvement >= +5% required");
    let pct_med_1 = (su_med_1 - 1.0) * 100.0;
    let pct_med_4 = (su_med_4 - 1.0) * 100.0;
    if pct_med_1 >= 5.0 {
        eprintln!("[microbench] NR=1 PASS: +{:.2}% median vs NR=2", pct_med_1);
    } else {
        eprintln!("[microbench] NR=1 FAIL gate: +{:.2}% median vs NR=2 (< +5%)", pct_med_1);
    }
    if pct_med_4 >= 5.0 {
        eprintln!("[microbench] NR=4 PASS: +{:.2}% median vs NR=2", pct_med_4);
    } else {
        eprintln!("[microbench] NR=4 FAIL gate: +{:.2}% median vs NR=2 (< +5%)", pct_med_4);
    }
}
