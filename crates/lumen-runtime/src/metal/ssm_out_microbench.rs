//! Microbenchmark helper for the ssm_out_gemm hot-spot in the GDN megakernel.
//!
//! The Metal deep-profile (`LUMEN_METAL_PROFILE_GDN=1`) reported
//! `gdn/ssm_out_gemm` at ~9.8 ms/layer (82% of the GDN megakernel total),
//! while the IDENTICAL-shape `gdn/attn_gate_gemm` ran at ~0.43 ms/layer with
//! the plain k64-aligned kernel.
//!
//! Both dispatches process [M=128, N=4096, K=4096] Q8_0 weights through the
//! same TILE_K=64 family. The only structural differences are:
//! - ssm_out uses the `_residual_batched` variant (fused R add at writeback).
//! - ssm_out is the LAST GEMM in the GDN chain (its weight tensor is cold).
//! - ssm_out has a true data dependency on the immediately-prior phase2b kernel.
//!
//! This helper runs both kernel variants over the same shape/buffer pattern
//! OUTSIDE the GDN megakernel chain. The verdict:
//!
//! - If both kernels finish in ~0.5 ms each (both modes), the 9.8 ms in
//!   production is **NOT** the kernel — it's an artifact of the deep-profile
//!   split (per-CB sync overhead, RAW hazard via the captured x_buf dependency,
//!   or cold-cache cost attributed to the last dispatch).
//! - If the residual kernel is genuinely ~20x slower, the residual variant has
//!   a structural bug at this shape.

use crate::error::RuntimeError;
use crate::metal::ffi::{MTLSize, MetalBuffer};
use crate::metal::types::MetalPipelines;
use crate::metal::MetalF32Backend;
use lumen_format::quantization::QuantScheme;
use std::time::Instant;

/// Median + raw timings for one microbench scenario.
#[derive(Debug, Clone)]
pub struct ScenarioResult {
    /// Short label for the scenario (e.g. "plain_aligned", "residual_aligned", "residual_zeros").
    pub label: String,
    /// Number of iterations measured (excluding warmup).
    pub iterations: usize,
    /// Total wall-time elapsed for `iterations` invocations in ONE command buffer (ms).
    pub single_cb_total_ms: f64,
    /// `single_cb_total_ms / iterations` (ms).
    pub single_cb_per_call_ms: f64,
    /// Per-trial wall-time of `iterations` invocations submitted as INDIVIDUAL command buffers (ms).
    /// One element per iteration; mimics the deep-profile per-section split.
    pub per_cb_trials_ms: Vec<f64>,
    /// Median of `per_cb_trials_ms` (ms).
    pub per_cb_median_ms: f64,
}

/// Full microbench report.
#[derive(Debug, Clone)]
pub struct MicrobenchReport {
    /// Apple GPU device name (for the reproducibility log).
    pub device_name: String,
    /// M / N / K used (matches Qwen3.5-9B ssm_out shape: 128, 4096, 4096).
    pub m: usize,
    pub n: usize,
    pub k: usize,
    /// Q8_0 weight buffer size in megabytes.
    pub weight_mb: f64,
    /// One entry per scenario.
    pub scenarios: Vec<ScenarioResult>,
}

impl MetalF32Backend {
    /// Run the ssm_out microbench for `dequant_tiled_matmul_q8_0_k64_aligned`
    /// vs. its `_residual_batched_aligned` sibling, on the exact shape of
    /// `gdn/ssm_out_gemm` in production (M=128, N=K=4096).
    ///
    /// `warmup_iters`: invocations to run-and-discard before timing
    /// (primes pipeline-state bindings, L2/SLC caches, and GPU clocks).
    ///
    /// `timed_iters`: invocations to time, both in single-CB and per-iteration-CB modes.
    ///
    /// This entry point is gated under `#[doc(hidden)]` because it is solely
    /// for the standalone test harness; production code never calls it.
    #[doc(hidden)]
    pub fn run_ssm_out_microbench(
        &mut self,
        warmup_iters: usize,
        timed_iters: usize,
    ) -> Result<MicrobenchReport, RuntimeError> {
        // Production Qwen3.5-MoE-35B-A3B ssm_out_gemm shape (GPU-time census).
        // M = batch_size (the prefill prompt = 1239 tokens, UNALIGNED:
        // 1239 % 32 = 23, so production dispatches the NON-aligned residual
        // kernel with FC boundary checks ON). N = hidden_dim = 2048,
        // K = q_dim = value_dim = 4096. Overridable via env for sweeps:
        //   LUMEN_SSMBENCH_M / _N / _K.
        let env_usize = |k: &str, d: usize| {
            std::env::var(k).ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(d)
        };
        // GEMM-convention upper-case dim names (M/N/K) kept for readability.
        #[allow(non_snake_case)]
        let (M, N, K) = (
            env_usize("LUMEN_SSMBENCH_M", 1239),
            env_usize("LUMEN_SSMBENCH_N", 2048),
            env_usize("LUMEN_SSMBENCH_K", 4096),
        );

        // Compile pipelines on demand (the production init() compiles them
        // through ComputeBackend::init, which the microbench harness does
        // NOT call — we don't want to allocate the full prefill scratch
        // buffers for a small kernel test).
        if self.pipelines.is_none() {
            let pipelines = self.compile_pipelines()?;
            self.pipelines = Some(pipelines);
        }
        let pipelines = self.pipelines.as_ref().ok_or_else(|| {
            RuntimeError::Compute("ssm_out_microbench: pipelines missing".into())
        })?;

        // -------- buffers --------
        let weight_bytes = build_q8_weight_bytes(N, K);
        let weight_mb = weight_bytes.len() as f64 / 1.0e6;
        let w_buf = self.device.new_buffer_with_bytes(&weight_bytes).ok_or_else(|| {
            RuntimeError::Compute("ssm_out_microbench: failed to allocate W buffer".into())
        })?;

        let x_bytes = build_random_f32_bytes(M * K, 0xC0FFEEu64);
        let x_buf = self.device.new_buffer_with_bytes(&x_bytes).ok_or_else(|| {
            RuntimeError::Compute("ssm_out_microbench: failed to allocate X buffer".into())
        })?;

        let y_buf = self.device.new_buffer(M * N * 4).ok_or_else(|| {
            RuntimeError::Compute("ssm_out_microbench: failed to allocate Y buffer".into())
        })?;

        // Random residual: mimics the production case where R = x_buf is
        // dense, non-zero data left by the prior layer's FFN-down kernel.
        let r_bytes = build_random_f32_bytes(M * N, 0xBADBEEFu64);
        let r_random_buf = self
            .device
            .new_buffer_with_bytes(&r_bytes)
            .ok_or_else(|| {
                RuntimeError::Compute("ssm_out_microbench: failed to allocate R (random) buffer".into())
            })?;

        // Zero residual: cold-cache control — same shape but never touched
        // before this dispatch. If the slowness is residual-buffer cache
        // state (H2), zeros vs random should differ.
        let zeros = vec![0u8; M * N * 4];
        let r_zeros_buf = self.device.new_buffer_with_bytes(&zeros).ok_or_else(|| {
            RuntimeError::Compute("ssm_out_microbench: failed to allocate R (zeros) buffer".into())
        })?;

        let device_name = self.device.name();

        // -------- scenarios --------
        // Variant 1: plain non-residual aligned k64 (the kernel that runs
        //            `gdn/attn_gate_gemm` at ~0.43 ms / call in production).
        // Variant 2: residual aligned k64 + random R (the kernel that runs
        //            `gdn/ssm_out_gemm` at ~9.8 ms / call in production).
        // Variant 3: residual aligned k64 + zero R (cold-cache control).

        // Production selects the `_aligned` k64 variant only when
        // M%32==0 && N%32==0 && K%32==0 (and K%64==0); otherwise the
        // NON-aligned variant (FC boundary checks ON). The default
        // (M=1239) is UNALIGNED so the non-aligned kernel is exercised —
        // exactly what production dispatches for ssm_out at T=1239.
        let aligned = M % 32 == 0 && N % 32 == 0 && K % 32 == 0 && K % 64 == 0;
        eprintln!(
            "[ssm-bench] shape M={M} N={N} K={K} aligned={aligned} \
             (production ssm_out at T=1239 is UNALIGNED)"
        );

        let plain = run_scenario_plain(
            "plain_aligned",
            self.queue_for_microbench(),
            pipelines,
            &w_buf,
            &x_buf,
            &y_buf,
            M as u32,
            N as u32,
            K as u32,
            warmup_iters,
            timed_iters,
        )?;

        let residual_random = run_scenario_residual(
            "residual_aligned_random_R",
            self.queue_for_microbench(),
            pipelines,
            &w_buf,
            &x_buf,
            &y_buf,
            &r_random_buf,
            M as u32,
            N as u32,
            K as u32,
            warmup_iters,
            timed_iters,
        )?;

        let residual_zeros = run_scenario_residual(
            "residual_aligned_zeros_R",
            self.queue_for_microbench(),
            pipelines,
            &w_buf,
            &x_buf,
            &y_buf,
            &r_zeros_buf,
            M as u32,
            N as u32,
            K as u32,
            warmup_iters,
            timed_iters,
        )?;

        Ok(MicrobenchReport {
            device_name,
            m: M,
            n: N,
            k: K,
            weight_mb,
            scenarios: vec![plain, residual_random, residual_zeros],
        })
    }

    /// Borrow the Metal command queue for microbench scenarios. Public to the
    /// metal module's submodules only.
    #[inline]
    fn queue_for_microbench(&self) -> &crate::metal::ffi::MetalCommandQueue {
        &self.queue
    }
}

// === scenario implementations =================================================

/// Encode the plain (non-residual) ssm_out_gemm shape into one encoder and
/// dispatch one threadgroup grid. Used both inside the single-CB and per-CB
/// timing loops.
#[allow(clippy::too_many_arguments)]
fn encode_plain(
    enc: &crate::metal::ffi::MetalComputeEncoder,
    pipelines: &MetalPipelines,
    w_buf: &MetalBuffer,
    x_buf: &MetalBuffer,
    y_buf: &MetalBuffer,
    m: u32,
    n: u32,
    k: u32,
) {
    let aligned = m % 32 == 0 && n % 32 == 0 && k % 32 == 0 && k % 64 == 0;
    if aligned {
        enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_k64_aligned);
    } else {
        enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_k64);
    }
    enc.set_threadgroup_memory_length(8192, 0);
    enc.set_buffer(w_buf, 0, 0);
    enc.set_buffer(x_buf, 0, 1);
    enc.set_buffer(y_buf, 0, 2);
    enc.set_bytes(&m.to_le_bytes(), 3);
    enc.set_bytes(&n.to_le_bytes(), 4);
    enc.set_bytes(&k.to_le_bytes(), 5);
    // Same dispatch geometry as production gdn::encode_batched_gdn_prefill:
    //   ((N / 32), (M / 32), 1) threadgroups × 128 threads (= 4 simdgroups).
    enc.dispatch_threadgroups(
        MTLSize::new((n as u64).div_ceil(32), (m as u64).div_ceil(32), 1),
        MTLSize::new(128, 1, 1),
    );
}

#[allow(clippy::too_many_arguments)]
fn encode_residual(
    enc: &crate::metal::ffi::MetalComputeEncoder,
    pipelines: &MetalPipelines,
    w_buf: &MetalBuffer,
    x_buf: &MetalBuffer,
    y_buf: &MetalBuffer,
    r_buf: &MetalBuffer,
    m: u32,
    n: u32,
    k: u32,
) {
    let aligned = m % 32 == 0 && n % 32 == 0 && k % 32 == 0 && k % 64 == 0;
    if aligned {
        enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_k64_residual_batched_aligned);
    } else {
        enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_k64_residual_batched);
    }
    enc.set_threadgroup_memory_length(8192, 0);
    enc.set_buffer(w_buf, 0, 0);
    enc.set_buffer(x_buf, 0, 1);
    enc.set_buffer(y_buf, 0, 2);
    enc.set_bytes(&m.to_le_bytes(), 3);
    enc.set_bytes(&n.to_le_bytes(), 4);
    enc.set_bytes(&k.to_le_bytes(), 5);
    enc.set_buffer(r_buf, 0, 6);
    enc.dispatch_threadgroups(
        MTLSize::new((n as u64).div_ceil(32), (m as u64).div_ceil(32), 1),
        MTLSize::new(128, 1, 1),
    );
}

#[allow(clippy::too_many_arguments)]
fn run_scenario_plain(
    label: &str,
    queue: &crate::metal::ffi::MetalCommandQueue,
    pipelines: &MetalPipelines,
    w_buf: &MetalBuffer,
    x_buf: &MetalBuffer,
    y_buf: &MetalBuffer,
    m: u32,
    n: u32,
    k: u32,
    warmup_iters: usize,
    timed_iters: usize,
) -> Result<ScenarioResult, RuntimeError> {
    // --- warmup (single CB, results discarded) ---
    {
        let cb = queue.new_command_buffer().ok_or_else(|| {
            RuntimeError::Compute("ssm_out_microbench: warmup CB alloc failed".into())
        })?;
        for _ in 0..warmup_iters {
            let enc = cb.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("ssm_out_microbench: warmup encoder failed".into())
            })?;
            encode_plain(&enc, pipelines, w_buf, x_buf, y_buf, m, n, k);
            enc.end_encoding();
        }
        cb.commit_and_wait();
    }

    // --- single CB timing ---
    let single_cb_total_ms = {
        let cb = queue.new_command_buffer().ok_or_else(|| {
            RuntimeError::Compute("ssm_out_microbench: timed CB alloc failed".into())
        })?;
        for _ in 0..timed_iters {
            let enc = cb.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("ssm_out_microbench: timed encoder failed".into())
            })?;
            encode_plain(&enc, pipelines, w_buf, x_buf, y_buf, m, n, k);
            enc.end_encoding();
        }
        let t0 = Instant::now();
        cb.commit_and_wait();
        t0.elapsed().as_secs_f64() * 1000.0
    };

    // --- per-CB timing (one commit_and_wait per iteration, like deep-profile) ---
    let mut per_cb_trials_ms = Vec::with_capacity(timed_iters);
    for _ in 0..timed_iters {
        let cb = queue.new_command_buffer().ok_or_else(|| {
            RuntimeError::Compute("ssm_out_microbench: per-CB alloc failed".into())
        })?;
        let enc = cb.new_compute_encoder().ok_or_else(|| {
            RuntimeError::Compute("ssm_out_microbench: per-CB encoder failed".into())
        })?;
        encode_plain(&enc, pipelines, w_buf, x_buf, y_buf, m, n, k);
        enc.end_encoding();
        let t0 = Instant::now();
        cb.commit_and_wait();
        per_cb_trials_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    let per_cb_median_ms = median(&per_cb_trials_ms);
    Ok(ScenarioResult {
        label: label.to_string(),
        iterations: timed_iters,
        single_cb_total_ms,
        single_cb_per_call_ms: single_cb_total_ms / timed_iters as f64,
        per_cb_trials_ms,
        per_cb_median_ms,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_scenario_residual(
    label: &str,
    queue: &crate::metal::ffi::MetalCommandQueue,
    pipelines: &MetalPipelines,
    w_buf: &MetalBuffer,
    x_buf: &MetalBuffer,
    y_buf: &MetalBuffer,
    r_buf: &MetalBuffer,
    m: u32,
    n: u32,
    k: u32,
    warmup_iters: usize,
    timed_iters: usize,
) -> Result<ScenarioResult, RuntimeError> {
    // warmup
    {
        let cb = queue.new_command_buffer().ok_or_else(|| {
            RuntimeError::Compute("ssm_out_microbench: residual warmup CB alloc failed".into())
        })?;
        for _ in 0..warmup_iters {
            let enc = cb.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("ssm_out_microbench: residual warmup encoder failed".into())
            })?;
            encode_residual(&enc, pipelines, w_buf, x_buf, y_buf, r_buf, m, n, k);
            enc.end_encoding();
        }
        cb.commit_and_wait();
    }

    // single CB
    let single_cb_total_ms = {
        let cb = queue.new_command_buffer().ok_or_else(|| {
            RuntimeError::Compute("ssm_out_microbench: residual timed CB alloc failed".into())
        })?;
        for _ in 0..timed_iters {
            let enc = cb.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("ssm_out_microbench: residual timed encoder failed".into())
            })?;
            encode_residual(&enc, pipelines, w_buf, x_buf, y_buf, r_buf, m, n, k);
            enc.end_encoding();
        }
        let t0 = Instant::now();
        cb.commit_and_wait();
        t0.elapsed().as_secs_f64() * 1000.0
    };

    // per-CB
    let mut per_cb_trials_ms = Vec::with_capacity(timed_iters);
    for _ in 0..timed_iters {
        let cb = queue.new_command_buffer().ok_or_else(|| {
            RuntimeError::Compute("ssm_out_microbench: residual per-CB alloc failed".into())
        })?;
        let enc = cb.new_compute_encoder().ok_or_else(|| {
            RuntimeError::Compute("ssm_out_microbench: residual per-CB encoder failed".into())
        })?;
        encode_residual(&enc, pipelines, w_buf, x_buf, y_buf, r_buf, m, n, k);
        enc.end_encoding();
        let t0 = Instant::now();
        cb.commit_and_wait();
        per_cb_trials_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    let per_cb_median_ms = median(&per_cb_trials_ms);
    Ok(ScenarioResult {
        label: label.to_string(),
        iterations: timed_iters,
        single_cb_total_ms,
        single_cb_per_call_ms: single_cb_total_ms / timed_iters as f64,
        per_cb_trials_ms,
        per_cb_median_ms,
    })
}

// === helpers =================================================================

fn median(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = s.len() / 2;
    if s.len() % 2 == 0 {
        0.5 * (s[mid - 1] + s[mid])
    } else {
        s[mid]
    }
}

/// Build a synthetic Q8_0 weight blob for an [N, K] tensor, matching the
/// row-major byte layout used by `dequant_tiled_matmul_q8_0_*` kernels.
///
/// Each row contains `K/32` Q8_0 blocks; each block is 34 bytes
/// (`half scale | int8 qs[32]`). The byte layout is byte-identical to the
/// `qwen35.gdn.X.ssm_out.weight` tensor in the Qwen3.5-9B Q8_0 LBC: the
/// kernel only cares about row stride and the access pattern, not values.
///
/// Scales are set to a small constant (f16 ~0.0078) so the dequant pass
/// produces well-conditioned outputs and the kernel's compute graph
/// exercises exactly the same simdgroup operations as production.
fn build_q8_weight_bytes(n: usize, k: usize) -> Vec<u8> {
    const Q8_GROUP: usize = 32;
    const Q8_BLOCK_BYTES: usize = 34;
    assert!(k % Q8_GROUP == 0, "K must be a multiple of 32 for Q8_0");
    let blocks_per_row = k / Q8_GROUP;
    let row_bytes = blocks_per_row * Q8_BLOCK_BYTES;
    let total = n * row_bytes;
    let mut data = vec![0u8; total];

    // Pick a small but representative scale: f16 1.0 / 128.0 ≈ 0.0078.
    // Bit-encoded as f16 0x1F00 ((15 - 7) = 8 exponent, mantissa 0 -> 2^-7).
    let scale_bits: u16 = f32_to_f16_bits(1.0 / 128.0);

    // Fill quant data with a deterministic LCG so the entire weight tensor
    // has nonzero bytes (mimicking real Q8_0 entropy and exercising the
    // memory subsystem the same way real weights do).
    let mut rng: u64 = 0x9E3779B97F4A7C15u64;
    for row in 0..n {
        for blk in 0..blocks_per_row {
            let off = row * row_bytes + blk * Q8_BLOCK_BYTES;
            data[off] = (scale_bits & 0xFF) as u8;
            data[off + 1] = (scale_bits >> 8) as u8;
            for j in 0..Q8_GROUP {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                // Map low byte to int8 range [-64, 63] to keep dequant outputs sane.
                let signed = ((rng >> 56) as i8) >> 1;
                data[off + 2 + j] = signed as u8;
            }
        }
    }
    data
}

/// Build a synthetic F32 byte blob with deterministic LCG-derived values
/// in [-1.0, 1.0). Used for X (input batch) and the random-R residual.
fn build_random_f32_bytes(num_floats: usize, seed: u64) -> Vec<u8> {
    let mut out = vec![0u8; num_floats * 4];
    let mut rng = seed.max(1);
    for i in 0..num_floats {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let bits24 = (rng >> 40) as u32 & 0x00FF_FFFF;
        // Scale to [-1.0, 1.0).
        let v = (bits24 as f32 / (1u32 << 23) as f32) - 1.0;
        let b = v.to_le_bytes();
        out[i * 4] = b[0];
        out[i * 4 + 1] = b[1];
        out[i * 4 + 2] = b[2];
        out[i * 4 + 3] = b[3];
    }
    out
}

/// Encode an f32 value to IEEE 754 half-precision bits.
fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x7fffff;
    if exp == 0xff {
        let f16_frac = if frac != 0 { 0x200u32 } else { 0 };
        return ((sign << 15) | (0x1f << 10) | f16_frac) as u16;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return ((sign << 15) | (0x1f << 10)) as u16;
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return (sign << 15) as u16;
        }
        let m = frac | 0x800000;
        let shift = 1 - new_exp;
        let f16_frac = m >> (13 + shift);
        return ((sign << 15) | f16_frac) as u16;
    }
    let f16_frac = frac >> 13;
    ((sign << 15) | ((new_exp as u32) << 10) | f16_frac) as u16
}

/// Quant scheme used by the microbench (always Q8_0).
#[allow(dead_code)]
pub(crate) const MICROBENCH_QUANT: QuantScheme = QuantScheme::Q8_0;

#[allow(dead_code)]
pub(crate) const MICROBENCH_LABEL: &str = "metal_ssm_out_k64";
