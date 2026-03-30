//! CUDA benchmark integration tests.
//!
//! Measures decode throughput and per-kernel latency for the CUDA backend
//! using a synthetic test model. These tests are `#[ignore]`d because they
//! take significant time and require a CUDA-capable GPU.
//!
//! Run via Modal:
//!     modal run modal/bench_cuda.py          # decode benchmark
//!     modal run modal/bench_kernels.py       # per-kernel micro-benchmarks
//!
//! Run locally (requires NVIDIA GPU):
//!     cargo test -p lumen-runtime --features cuda --release \
//!         --test cuda_bench_test -- --ignored --nocapture

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use lumen_format::test_model::{generate_test_model, TestModelConfig};
use lumen_runtime::compute::ComputeBackend;
use lumen_runtime::cuda::CudaBackend;
use lumen_runtime::kv::{KvCache, KvCacheConfig, KvPrecision};
use lumen_runtime::weight::provider_sync::SyncWeightProvider;
use lumen_runtime::WeightProvider;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Read an environment variable as usize, falling back to the default.
fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Timing statistics computed from a series of measurements.
struct TimingStats {
    mean_us: f64,
    min_us: f64,
    max_us: f64,
    median_us: f64,
    p99_us: f64,
    count: usize,
}

impl TimingStats {
    /// Compute statistics from a slice of microsecond measurements.
    ///
    /// Panics if `times_us` is empty.
    fn from_us(times_us: &[f64]) -> Self {
        assert!(!times_us.is_empty(), "no measurements to compute stats from");
        let count = times_us.len();
        let mean_us = times_us.iter().sum::<f64>() / count as f64;
        let min_us = times_us.iter().copied().fold(f64::INFINITY, f64::min);
        let max_us = times_us.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let mut sorted = times_us.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_us = if count % 2 == 0 {
            (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
        } else {
            sorted[count / 2]
        };
        let p99_idx = ((count as f64) * 0.99).ceil() as usize;
        let p99_us = sorted[p99_idx.min(count) - 1];

        Self { mean_us, min_us, max_us, median_us, p99_us, count }
    }
}

/// Create a synthetic test model and return (provider, CUDA backend).
///
/// Uses the provided TestModelConfig. Writes the LBC to a temporary file,
/// opens a SyncWeightProvider, and initializes a CudaBackend with global
/// tensors and hyperparams.
fn setup_bench_backend(
    config: &TestModelConfig,
) -> Result<(SyncWeightProvider, CudaBackend), String> {
    let lbc_data = generate_test_model(config);

    let id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("lumen_cuda_bench_{id}"));
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir: {e}"))?;
    let path = dir.join("bench_model.lbc");
    {
        let mut f = std::fs::File::create(&path).map_err(|e| format!("create: {e}"))?;
        f.write_all(&lbc_data).map_err(|e| format!("write: {e}"))?;
    }

    let provider = SyncWeightProvider::open(&path).map_err(|e| format!("open: {e}"))?;
    let hp = provider.lbc().header.hyperparams;

    let mut cuda = CudaBackend::new(0).map_err(|e| format!("cuda: {e}"))?;
    cuda.set_global_tensors(
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
    );
    cuda.init(&hp).map_err(|e| format!("init: {e}"))?;

    Ok((provider, cuda))
}

// ---------------------------------------------------------------------------
// BENCH 1: Decode throughput
// ---------------------------------------------------------------------------

/// Full decode throughput benchmark.
///
/// Creates a synthetic model, warms up, measures per-token decode time, and
/// reports tok/s with percentile statistics.
///
/// Environment variables:
///   LUMEN_BENCH_TOKENS   — number of tokens to measure (default 50)
///   LUMEN_BENCH_WARMUP   — number of warmup tokens (default 5)
///   LUMEN_BENCH_SCALE    — "tiny" (default) or "realistic"
///
/// The "realistic" scale uses TinyLlama-sized dimensions (hidden_dim=2048,
/// 22 layers, 32 heads, head_dim=64, inter_dim=5632, vocab=32000). This
/// takes significantly more GPU memory and time.
#[test]
#[ignore]
fn bench_cuda_decode() {
    let warmup_tokens = env_usize("LUMEN_BENCH_WARMUP", 5);
    let measure_tokens = env_usize("LUMEN_BENCH_TOKENS", 50);
    let scale = std::env::var("LUMEN_BENCH_SCALE").unwrap_or_else(|_| "tiny".into());

    let config = match scale.as_str() {
        "realistic" => TestModelConfig {
            num_layers: 22,
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 64,
            hidden_dim: 2048,
            intermediate_dim: 5632,
            vocab_size: 32000,
            max_seq_len: 2048,
            seed: 42,
        },
        _ => TestModelConfig::default(),
    };

    eprintln!("=== CUDA Decode Benchmark ===");
    eprintln!(
        "  Model: {} layers, hidden_dim={}, heads={}, head_dim={}, inter_dim={}, vocab={}",
        config.num_layers, config.hidden_dim, config.num_heads,
        config.head_dim, config.intermediate_dim, config.vocab_size,
    );
    eprintln!("  Scale: {scale}");
    eprintln!("  Warmup tokens: {warmup_tokens}");
    eprintln!("  Measure tokens: {measure_tokens}");
    eprintln!();

    let (provider, cuda) = match setup_bench_backend(&config) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Skipping: {e}");
            return;
        }
    };

    let hp = provider.lbc().header.hyperparams;
    let num_layers = hp.num_layers as usize;
    let total_tokens = warmup_tokens + measure_tokens;

    // Measure kernel compilation time separately.
    let t_compile = Instant::now();
    let mut cuda2 = CudaBackend::new(0).unwrap();
    cuda2.set_global_tensors(
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
    );
    cuda2.init(&hp).unwrap();
    let compile_time = t_compile.elapsed();
    eprintln!(
        "  Kernel compilation + init: {:.1} ms",
        compile_time.as_secs_f64() * 1000.0,
    );

    let kv_config = KvCacheConfig {
        max_seq_len: hp.max_seq_len as usize,
        num_layers,
        num_kv_heads: hp.num_kv_heads as usize,
        head_dim: hp.head_dim as usize,
        precision: KvPrecision::F32,
    };
    let mut kv = KvCache::new(kv_config).unwrap();

    // Run tokens (warmup + measured).
    let mut decode_times_us: Vec<f64> = Vec::with_capacity(measure_tokens);
    let mut token_id = 0u32;

    for tok_idx in 0..total_tokens {
        let t_tok_start = Instant::now();

        let mut x = cuda.embed_token(token_id).unwrap();

        for layer_idx in 0..num_layers {
            let layer_view = provider.get_layer_blocking(layer_idx).unwrap();
            let seq_pos = kv.seq_len();
            let mut kv_view = kv.view_mut(layer_idx).unwrap();
            cuda.compute_layer(layer_idx, &mut x, &layer_view, Some(&mut kv_view), seq_pos)
                .unwrap();
            kv.commit_view(kv_view).unwrap();
        }
        kv.advance_seq_len().unwrap();

        let logits = cuda.compute_final(&x).unwrap();
        token_id = logits.argmax() as u32;

        let tok_time = t_tok_start.elapsed();

        if tok_idx >= warmup_tokens {
            decode_times_us.push(tok_time.as_secs_f64() * 1_000_000.0);
        }
    }

    let stats = TimingStats::from_us(&decode_times_us);
    let tok_per_sec = 1_000_000.0 / stats.mean_us;
    let tok_per_sec_median = 1_000_000.0 / stats.median_us;

    eprintln!();
    eprintln!("=== Decode Results ===");
    eprintln!("  Tokens measured: {}", stats.count);
    eprintln!("  Mean:   {:.1} us/tok  ({tok_per_sec:.1} tok/s)", stats.mean_us);
    eprintln!("  Median: {:.1} us/tok  ({tok_per_sec_median:.1} tok/s)", stats.median_us);
    eprintln!("  Min:    {:.1} us/tok", stats.min_us);
    eprintln!("  Max:    {:.1} us/tok", stats.max_us);
    eprintln!("  P99:    {:.1} us/tok", stats.p99_us);
    eprintln!(
        "  Kernel compile+init: {:.1} ms",
        compile_time.as_secs_f64() * 1000.0,
    );
    eprintln!();

    // Sanity: even the tiny model on a slow GPU should decode in < 100ms/tok.
    assert!(
        stats.mean_us < 100_000.0,
        "Decode too slow: {:.0} us/tok (expected < 100ms)",
        stats.mean_us,
    );
}

// ---------------------------------------------------------------------------
// BENCH 2: Per-kernel micro-benchmarks
// ---------------------------------------------------------------------------

/// Helper: create CUDA context and stream for micro-benchmarks.
fn create_context() -> (
    std::sync::Arc<cudarc::driver::CudaContext>,
    std::sync::Arc<cudarc::driver::CudaStream>,
) {
    let ctx = CudaContext::new(0).expect("No CUDA GPU available");
    let stream = ctx.default_stream();
    (ctx, stream)
}

/// Run a kernel benchmark loop and report structured results.
///
/// Returns (mean_us, min_us, max_us, total_elapsed).
fn bench_kernel_loop<F>(
    stream: &cudarc::driver::CudaStream,
    warmup: usize,
    iterations: usize,
    mut launch: F,
) -> (f64, f64, f64, std::time::Duration)
where
    F: FnMut(),
{
    // Warmup
    for _ in 0..warmup {
        launch();
    }
    stream.synchronize().unwrap();

    // Measure
    let start = Instant::now();
    for _ in 0..iterations {
        launch();
    }
    stream.synchronize().unwrap();
    let elapsed = start.elapsed();

    let mean_us = elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64;

    // For min/max estimation from batch timing, we report the batch mean as
    // both min and max. Individual kernel launches are too fast for host-side
    // per-call timing (sub-microsecond). CUDA events would be needed for
    // per-call min/max, which adds complexity. The batch mean is the most
    // reliable number for bandwidth-bound kernels.
    (mean_us, mean_us, mean_us, elapsed)
}

#[test]
#[ignore]
fn bench_kernel_rmsnorm() {
    let iterations = env_usize("LUMEN_BENCH_ITERATIONS", 10000);
    let dim = 4096usize;

    let (ctx, stream) = create_context();
    let src = lumen_runtime::cuda::shaders::NORM_KERNEL_SOURCE;
    let ptx = compile_ptx(src).expect("NVRTC compile failed for norm.cu");
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("rmsnorm").unwrap();

    let x: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01).collect();
    let weight: Vec<f32> = vec![1.0f32; dim];
    let eps: f32 = 1e-5;

    let x_gpu = stream.clone_htod(&x).unwrap();
    let w_gpu = stream.clone_htod(&weight).unwrap();
    let mut out_gpu: CudaSlice<f32> = stream.alloc_zeros(dim).unwrap();

    let block_dim = 256u32;
    let num_warps = block_dim / 32;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: num_warps * 4,
    };

    let (mean_us, _min_us, _max_us, elapsed) = bench_kernel_loop(
        &stream, 100, iterations,
        || {
            unsafe {
                stream
                    .launch_builder(&func)
                    .arg(&x_gpu)
                    .arg(&w_gpu)
                    .arg(&mut out_gpu)
                    .arg(&eps)
                    .arg(&(dim as u32))
                    .launch(cfg)
            }
            .unwrap();
        },
    );

    // Memory traffic: read x (dim*4) + read weight (dim*4) + write out (dim*4)
    let bytes_per_op = 3 * dim * 4;
    let bandwidth_gbs =
        (bytes_per_op as f64 * iterations as f64) / elapsed.as_secs_f64() / 1e9;

    eprintln!();
    eprintln!("=== RMSNorm Benchmark (dim={dim}) ===");
    eprintln!("  Iterations: {iterations}");
    eprintln!("  Time/op:    {mean_us:.3} us");
    eprintln!("  Bandwidth:  {bandwidth_gbs:.1} GB/s");
    eprintln!("  Total:      {:.1} ms", elapsed.as_secs_f64() * 1000.0);
}

#[test]
#[ignore]
fn bench_kernel_matvec_f32() {
    let iterations = env_usize("LUMEN_BENCH_ITERATIONS", 10000);
    let out_dim = 4096usize;
    let in_dim = 4096usize;

    let (ctx, stream) = create_context();
    let src = lumen_runtime::cuda::shaders::MATVEC_F32_KERNEL_SOURCE;
    let ptx = compile_ptx(src).expect("NVRTC compile failed for matvec_f32.cu");
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("matvec_f32").unwrap();

    let weight: Vec<f32> =
        (0..out_dim * in_dim).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
    let x: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.001).collect();

    let w_gpu = stream.clone_htod(&weight).unwrap();
    let x_gpu = stream.clone_htod(&x).unwrap();
    let mut out_gpu: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();

    let block_dim = 256u32;
    let cfg = LaunchConfig {
        grid_dim: (out_dim as u32, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    };

    let (mean_us, _min_us, _max_us, elapsed) = bench_kernel_loop(
        &stream, 100, iterations,
        || {
            unsafe {
                stream
                    .launch_builder(&func)
                    .arg(&w_gpu)
                    .arg(&x_gpu)
                    .arg(&mut out_gpu)
                    .arg(&(out_dim as u32))
                    .arg(&(in_dim as u32))
                    .launch(cfg)
            }
            .unwrap();
        },
    );

    let flops_per_op = 2.0 * out_dim as f64 * in_dim as f64;
    let gflops = (flops_per_op * iterations as f64) / elapsed.as_secs_f64() / 1e9;
    let tflops = gflops / 1000.0;
    // Memory traffic: read weight + read x + write out
    let bytes_per_op = (out_dim * in_dim + in_dim + out_dim) * 4;
    let bandwidth_gbs =
        (bytes_per_op as f64 * iterations as f64) / elapsed.as_secs_f64() / 1e9;

    eprintln!();
    eprintln!("=== MatVec F32 Benchmark ({out_dim}x{in_dim}) ===");
    eprintln!("  Iterations: {iterations}");
    eprintln!("  Time/op:    {mean_us:.3} us");
    eprintln!("  GFLOPS:     {gflops:.1}  ({tflops:.3} TFLOPS)");
    eprintln!("  Bandwidth:  {bandwidth_gbs:.1} GB/s");
    eprintln!("  Total:      {:.1} ms", elapsed.as_secs_f64() * 1000.0);
}

#[test]
#[ignore]
fn bench_kernel_matvec_f32_large() {
    let iterations = env_usize("LUMEN_BENCH_ITERATIONS", 10000);
    // TinyLlama-scale: FFN gate projection 2048 -> 5632
    let out_dim = 5632usize;
    let in_dim = 2048usize;

    let (ctx, stream) = create_context();
    let src = lumen_runtime::cuda::shaders::MATVEC_F32_KERNEL_SOURCE;
    let ptx = compile_ptx(src).expect("NVRTC compile failed");
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("matvec_f32").unwrap();

    let weight: Vec<f32> =
        (0..out_dim * in_dim).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
    let x: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.001).collect();

    let w_gpu = stream.clone_htod(&weight).unwrap();
    let x_gpu = stream.clone_htod(&x).unwrap();
    let mut out_gpu: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();

    let block_dim = 256u32;
    let cfg = LaunchConfig {
        grid_dim: (out_dim as u32, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    };

    let (mean_us, _min_us, _max_us, elapsed) = bench_kernel_loop(
        &stream, 100, iterations,
        || {
            unsafe {
                stream
                    .launch_builder(&func)
                    .arg(&w_gpu)
                    .arg(&x_gpu)
                    .arg(&mut out_gpu)
                    .arg(&(out_dim as u32))
                    .arg(&(in_dim as u32))
                    .launch(cfg)
            }
            .unwrap();
        },
    );

    let flops_per_op = 2.0 * out_dim as f64 * in_dim as f64;
    let gflops = (flops_per_op * iterations as f64) / elapsed.as_secs_f64() / 1e9;
    let tflops = gflops / 1000.0;
    let bytes_per_op = (out_dim * in_dim + in_dim + out_dim) * 4;
    let bandwidth_gbs =
        (bytes_per_op as f64 * iterations as f64) / elapsed.as_secs_f64() / 1e9;

    eprintln!();
    eprintln!("=== MatVec F32 Benchmark ({out_dim}x{in_dim}, TinyLlama FFN scale) ===");
    eprintln!("  Iterations: {iterations}");
    eprintln!("  Time/op:    {mean_us:.3} us");
    eprintln!("  GFLOPS:     {gflops:.1}  ({tflops:.3} TFLOPS)");
    eprintln!("  Bandwidth:  {bandwidth_gbs:.1} GB/s");
    eprintln!("  Total:      {:.1} ms", elapsed.as_secs_f64() * 1000.0);
}

#[test]
#[ignore]
fn bench_kernel_matvec_q8_0() {
    let iterations = env_usize("LUMEN_BENCH_ITERATIONS", 10000);
    let out_dim = 4096usize;
    let in_dim = 4096usize;

    let (ctx, stream) = create_context();
    let src = lumen_runtime::cuda::shaders::MATVEC_Q8_0_KERNEL_SOURCE;
    let ptx = compile_ptx(src).expect("NVRTC compile failed for matvec_q8_0.cu");
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("matvec_q8_0").unwrap();

    // Generate deterministic pseudo-random Q8_0 data.
    // Q8_0 block: 2 bytes f16 scale + 32 bytes int8 quants = 34 bytes per block.
    let blocks_per_row = (in_dim + 31) / 32;
    let row_bytes = blocks_per_row * 34;
    let total_bytes = out_dim * row_bytes;

    // Fill with plausible Q8_0 data: scale=0.01 (f16), quants from LCG.
    let mut weight_bytes: Vec<u8> = Vec::with_capacity(total_bytes);
    let mut rng_state = 42u64;
    for _ in 0..out_dim {
        for _ in 0..blocks_per_row {
            // f16 scale ~ 0.01: f16 bits for 0.01 = 0x211E (approx)
            weight_bytes.push(0x1E);
            weight_bytes.push(0x21);
            for _ in 0..32 {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let q = ((rng_state >> 33) as i32 % 127) as i8;
                weight_bytes.push(q as u8);
            }
        }
    }

    let x: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.001).collect();

    // Upload as raw bytes (i8 for cudarc compatibility).
    let weight_i8: Vec<i8> = weight_bytes.iter().map(|&b| b as i8).collect();
    let w_gpu = stream.clone_htod(&weight_i8).unwrap();
    let x_gpu = stream.clone_htod(&x).unwrap();
    let mut out_gpu: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();

    let block_dim = 256u32;
    let cfg = LaunchConfig {
        grid_dim: (out_dim as u32, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    };

    let (mean_us, _min_us, _max_us, elapsed) = bench_kernel_loop(
        &stream, 100, iterations,
        || {
            unsafe {
                stream
                    .launch_builder(&func)
                    .arg(&w_gpu)
                    .arg(&x_gpu)
                    .arg(&mut out_gpu)
                    .arg(&(out_dim as u32))
                    .arg(&(in_dim as u32))
                    .launch(cfg)
            }
            .unwrap();
        },
    );

    // FLOPs: same as F32 matvec (2 * out * in), the dequant is pure overhead.
    let flops_per_op = 2.0 * out_dim as f64 * in_dim as f64;
    let gflops = (flops_per_op * iterations as f64) / elapsed.as_secs_f64() / 1e9;
    let tflops = gflops / 1000.0;
    // Memory traffic: read Q8_0 weight (total_bytes) + read x (in_dim*4) + write out (out_dim*4)
    let bytes_per_op = total_bytes + in_dim * 4 + out_dim * 4;
    let bandwidth_gbs =
        (bytes_per_op as f64 * iterations as f64) / elapsed.as_secs_f64() / 1e9;

    eprintln!();
    eprintln!("=== MatVec Q8_0 Benchmark ({out_dim}x{in_dim}) ===");
    eprintln!("  Iterations: {iterations}");
    eprintln!("  Time/op:    {mean_us:.3} us");
    eprintln!("  GFLOPS:     {gflops:.1}  ({tflops:.3} TFLOPS)");
    eprintln!("  Bandwidth:  {bandwidth_gbs:.1} GB/s");
    eprintln!(
        "  Q8_0 compression: {:.1}x vs F32",
        (out_dim * in_dim * 4) as f64 / total_bytes as f64,
    );
    eprintln!("  Total:      {:.1} ms", elapsed.as_secs_f64() * 1000.0);
}

#[test]
#[ignore]
fn bench_kernel_attention_decode() {
    let iterations = env_usize("LUMEN_BENCH_ITERATIONS", 10000);
    let num_heads = 32u32;
    let num_kv_heads = 32u32;
    let head_dim = 64u32;

    let (ctx, stream) = create_context();
    let src = lumen_runtime::cuda::shaders::ATTENTION_KERNEL_SOURCE;
    let ptx = compile_ptx(src).expect("NVRTC compile failed for attention.cu");
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("attention_decode").unwrap();

    // Test multiple sequence lengths to see scaling behavior.
    let seq_lengths: Vec<u32> = vec![32, 128, 512, 1024];
    let max_seq_len = *seq_lengths.iter().max().unwrap();

    let q_size = (num_heads * head_dim) as usize;
    let kv_cache_size =
        (num_kv_heads as usize) * (max_seq_len as usize) * (head_dim as usize);

    let q_data: Vec<f32> =
        (0..q_size).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
    let kv_data: Vec<f32> =
        (0..kv_cache_size).map(|i| ((i % 61) as f32 - 30.0) * 0.01).collect();

    let q_gpu = stream.clone_htod(&q_data).unwrap();
    let k_cache_gpu = stream.clone_htod(&kv_data).unwrap();
    let v_cache_gpu = stream.clone_htod(&kv_data).unwrap();
    let mut out_gpu: CudaSlice<f32> = stream.alloc_zeros(q_size).unwrap();

    let scale = 1.0f32 / (head_dim as f32).sqrt();

    eprintln!();
    eprintln!(
        "=== Attention Decode Benchmark (heads={num_heads}, head_dim={head_dim}) ===",
    );

    for &seq_len in &seq_lengths {
        let block_size = {
            let bs = (seq_len as usize).min(256);
            let bs = ((bs + 31) / 32) * 32;
            bs.max(32) as u32
        };
        let shared_bytes = (8 + seq_len) * 4;

        let cfg = LaunchConfig {
            grid_dim: (num_heads, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: shared_bytes,
        };

        // Fewer iterations for attention (more expensive per call).
        let attn_iters = iterations / 10;

        let (mean_us, _min_us, _max_us, _elapsed) = bench_kernel_loop(
            &stream, 50, attn_iters,
            || {
                unsafe {
                    stream
                        .launch_builder(&func)
                        .arg(&q_gpu)
                        .arg(&k_cache_gpu)
                        .arg(&v_cache_gpu)
                        .arg(&mut out_gpu)
                        .arg(&num_heads)
                        .arg(&num_kv_heads)
                        .arg(&head_dim)
                        .arg(&seq_len)
                        .arg(&max_seq_len)
                        .arg(&scale)
                        .launch(cfg)
                }
                .unwrap();
            },
        );

        eprintln!(
            "  seq_len={seq_len:4}: {mean_us:8.1} us/op  ({attn_iters} iterations)",
        );
    }
}

#[test]
#[ignore]
fn bench_kernel_swiglu() {
    let iterations = env_usize("LUMEN_BENCH_ITERATIONS", 10000);
    let dim = 5632usize; // TinyLlama intermediate_dim

    let (ctx, stream) = create_context();
    let src = lumen_runtime::cuda::shaders::ACTIVATIONS_KERNEL_SOURCE;
    let ptx = compile_ptx(src).expect("NVRTC compile failed for activations.cu");
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("swiglu_inplace").unwrap();

    let gate: Vec<f32> =
        (0..dim).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
    let up: Vec<f32> = (0..dim).map(|i| ((i % 61) as f32 - 30.0) * 0.01).collect();

    let mut gate_gpu = stream.clone_htod(&gate).unwrap();
    let up_gpu = stream.clone_htod(&up).unwrap();
    let n = dim as u32;

    let block_dim = 256u32;
    let grid_dim = n.div_ceil(block_dim);
    let cfg = LaunchConfig {
        grid_dim: (grid_dim, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    };

    let (mean_us, _min_us, _max_us, elapsed) = bench_kernel_loop(
        &stream, 100, iterations,
        || {
            unsafe {
                stream
                    .launch_builder(&func)
                    .arg(&mut gate_gpu)
                    .arg(&up_gpu)
                    .arg(&n)
                    .launch(cfg)
            }
            .unwrap();
        },
    );

    // Memory: read gate + read up + write gate = 3*dim*4 bytes
    let bytes_per_op = 3 * dim * 4;
    let bandwidth_gbs =
        (bytes_per_op as f64 * iterations as f64) / elapsed.as_secs_f64() / 1e9;

    eprintln!();
    eprintln!("=== SwiGLU Benchmark (dim={dim}) ===");
    eprintln!("  Iterations: {iterations}");
    eprintln!("  Time/op:    {mean_us:.3} us");
    eprintln!("  Bandwidth:  {bandwidth_gbs:.1} GB/s");
    eprintln!("  Total:      {:.1} ms", elapsed.as_secs_f64() * 1000.0);
}

#[test]
#[ignore]
fn bench_kernel_residual_add() {
    let iterations = env_usize("LUMEN_BENCH_ITERATIONS", 10000);
    let dim = 4096usize;

    let (ctx, stream) = create_context();
    let src = lumen_runtime::cuda::shaders::ACTIVATIONS_KERNEL_SOURCE;
    let ptx = compile_ptx(src).expect("NVRTC compile failed");
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("residual_add").unwrap();

    let x: Vec<f32> = (0..dim).map(|i| i as f32 * 0.01).collect();
    let residual: Vec<f32> = (0..dim).map(|i| i as f32 * 0.005).collect();

    let mut x_gpu = stream.clone_htod(&x).unwrap();
    let res_gpu = stream.clone_htod(&residual).unwrap();
    let n = dim as u32;

    let block_dim = 256u32;
    let grid_dim = n.div_ceil(block_dim);
    let cfg = LaunchConfig {
        grid_dim: (grid_dim, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    };

    let (mean_us, _min_us, _max_us, elapsed) = bench_kernel_loop(
        &stream, 100, iterations,
        || {
            unsafe {
                stream
                    .launch_builder(&func)
                    .arg(&mut x_gpu)
                    .arg(&res_gpu)
                    .arg(&n)
                    .launch(cfg)
            }
            .unwrap();
        },
    );

    // Memory: read x + read residual + write x = 3*dim*4
    let bytes_per_op = 3 * dim * 4;
    let bandwidth_gbs =
        (bytes_per_op as f64 * iterations as f64) / elapsed.as_secs_f64() / 1e9;

    eprintln!();
    eprintln!("=== Residual Add Benchmark (dim={dim}) ===");
    eprintln!("  Iterations: {iterations}");
    eprintln!("  Time/op:    {mean_us:.3} us");
    eprintln!("  Bandwidth:  {bandwidth_gbs:.1} GB/s");
    eprintln!("  Total:      {:.1} ms", elapsed.as_secs_f64() * 1000.0);
}

#[test]
#[ignore]
fn bench_kernel_rope() {
    let iterations = env_usize("LUMEN_BENCH_ITERATIONS", 10000);
    let num_q_heads = 32u32;
    let num_kv_heads = 32u32;
    let head_dim = 64u32;
    let theta = 10000.0f32;

    let (ctx, stream) = create_context();
    let src = lumen_runtime::cuda::shaders::ROPE_KERNEL_SOURCE;
    let ptx = compile_ptx(src).expect("NVRTC compile failed for rope.cu");
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("rope_apply").unwrap();

    let q_size = (num_q_heads * head_dim) as usize;
    let k_size = (num_kv_heads * head_dim) as usize;

    let q: Vec<f32> =
        (0..q_size).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
    let k: Vec<f32> =
        (0..k_size).map(|i| ((i % 61) as f32 - 30.0) * 0.01).collect();

    let mut q_gpu = stream.clone_htod(&q).unwrap();
    let mut k_gpu = stream.clone_htod(&k).unwrap();

    let half_dim = head_dim / 2;
    let total_q_pairs = num_q_heads * half_dim;
    let total_k_pairs = num_kv_heads * half_dim;
    let max_pairs = total_q_pairs.max(total_k_pairs);

    let block_dim = 256u32;
    let grid_dim = max_pairs.div_ceil(block_dim);
    let cfg = LaunchConfig {
        grid_dim: (grid_dim, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    };

    let pos = 42u32;

    let (mean_us, _min_us, _max_us, elapsed) = bench_kernel_loop(
        &stream, 100, iterations,
        || {
            unsafe {
                stream
                    .launch_builder(&func)
                    .arg(&mut q_gpu)
                    .arg(&mut k_gpu)
                    .arg(&pos)
                    .arg(&num_q_heads)
                    .arg(&num_kv_heads)
                    .arg(&head_dim)
                    .arg(&theta)
                    .launch(cfg)
            }
            .unwrap();
        },
    );

    // Memory: read+write q + read+write k
    let bytes_per_op = (q_size + k_size) * 4 * 2;
    let bandwidth_gbs =
        (bytes_per_op as f64 * iterations as f64) / elapsed.as_secs_f64() / 1e9;

    eprintln!();
    eprintln!(
        "=== RoPE Benchmark (heads={num_q_heads}, head_dim={head_dim}) ===",
    );
    eprintln!("  Iterations: {iterations}");
    eprintln!("  Time/op:    {mean_us:.3} us");
    eprintln!("  Bandwidth:  {bandwidth_gbs:.1} GB/s");
    eprintln!("  Total:      {:.1} ms", elapsed.as_secs_f64() * 1000.0);
}

// ---------------------------------------------------------------------------
// BENCH 3: Kernel compilation time
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn bench_kernel_compilation_time() {
    let (ctx, _stream) = create_context();

    eprintln!();
    eprintln!("=== Kernel Compilation Time ===");

    let kernel_sources = [
        ("norm.cu (rmsnorm)", lumen_runtime::cuda::shaders::NORM_KERNEL_SOURCE),
        ("matvec_f32.cu", lumen_runtime::cuda::shaders::MATVEC_F32_KERNEL_SOURCE),
        ("matvec_q8_0.cu", lumen_runtime::cuda::shaders::MATVEC_Q8_0_KERNEL_SOURCE),
        ("rope.cu", lumen_runtime::cuda::shaders::ROPE_KERNEL_SOURCE),
        ("activations.cu", lumen_runtime::cuda::shaders::ACTIVATIONS_KERNEL_SOURCE),
        ("attention.cu", lumen_runtime::cuda::shaders::ATTENTION_KERNEL_SOURCE),
        ("kv_cache.cu", lumen_runtime::cuda::shaders::KV_CACHE_KERNEL_SOURCE),
        ("embed.cu", lumen_runtime::cuda::shaders::EMBED_KERNEL_SOURCE),
    ];

    let mut total_ms = 0.0f64;

    for (name, source) in &kernel_sources {
        let start = Instant::now();
        let ptx = compile_ptx(source).unwrap();
        let compile_time = start.elapsed();

        let load_start = Instant::now();
        let _module = ctx.load_module(ptx).unwrap();
        let load_time = load_start.elapsed();

        let compile_ms = compile_time.as_secs_f64() * 1000.0;
        let load_ms = load_time.as_secs_f64() * 1000.0;
        total_ms += compile_ms + load_ms;

        eprintln!(
            "  {name:30}  compile: {compile_ms:7.1} ms  load: {load_ms:6.1} ms",
        );
    }

    eprintln!("  {:<30}  total:  {total_ms:7.1} ms", "ALL KERNELS");
}
