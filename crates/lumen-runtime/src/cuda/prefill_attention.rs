//! Sequential per-token causal attention for the CUDA prefill path.
//!
//! During prefill, each token t attends to positions 0..pos_start+t (causal
//! mask). Because each token sees a different number of keys, attention cannot
//! be trivially batched into a single GEMM. The pragmatic approach: after
//! batched QKV projection, process attention sequentially per token using the
//! existing `attention_decode` kernel.
//!
//! The cost is O(batch * seq_len * head_dim), but the expensive GEMM projections
//! (which dominate time) are already batched. This module isolates the attention
//! loop to keep `backend_impl` focused on orchestration.

use cudarc::driver::{CudaSlice, LaunchConfig as CudarcLaunchConfig, PushKernelArg};

use crate::error::RuntimeError;

use super::decode::{KernelSet, attention_block_size, attention_shared_bytes};
use super::ffi::CudaDevice;
use super::kv_cache::KvCacheGpu;
use super::prefill::{launch_extract_row, launch_scatter_row};

/// Run causal attention for all tokens in a prefill batch.
///
/// For each token t in 0..batch:
///   1. Extract row t from `q_batch` into `q_single`
///   2. Run `attention_decode` against the KV cache with seq_len = pos_start + t + 1
///   3. Scatter the result back into row t of `attn_out_batch`
///
/// This produces correct causal attention where each token only attends to
/// positions before and including itself. The KV cache must already contain
/// data for all positions 0..pos_start+batch-1 before calling this function.
///
/// # Arguments
///
/// * `q_batch` - Batched Q vectors, shape `[batch, q_dim]`
/// * `attn_out_batch` - Output buffer, shape `[batch, q_dim]`
/// * `q_single` - Scratch buffer for a single token's Q, shape `[q_dim]`
/// * `attn_out_single` - Scratch buffer for a single token's attention output, shape `[q_dim]`
/// * `kv_cache` - GPU KV cache with data for positions 0..pos_start+batch-1
/// * `batch` - Number of tokens in the prefill batch
/// * `num_heads` - Number of query attention heads
/// * `num_kv_heads` - Number of KV attention heads (for GQA)
/// * `head_dim` - Dimension per attention head
/// * `pos_start` - Position of the first token in the batch within the sequence
///
/// # Safety
///
/// * `q_batch` must have at least `batch * q_dim` elements
/// * `attn_out_batch` must have at least `batch * q_dim` elements
/// * `q_single` must have at least `q_dim` elements
/// * `attn_out_single` must have at least `q_dim` elements
/// * KV cache must have valid data for `pos_start + batch` positions
#[allow(dead_code)]
pub fn prefill_attention_sequential(
    device: &CudaDevice,
    kernels: &KernelSet,
    q_batch: &CudaSlice<f32>,
    kv_cache: &KvCacheGpu,
    attn_out_batch: &mut CudaSlice<f32>,
    batch: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    pos_start: usize,
    q_single: &mut CudaSlice<f32>,
    attn_out_single: &mut CudaSlice<f32>,
) -> Result<(), RuntimeError> {
    let q_dim = num_heads * head_dim;

    // Validate buffer sizes up front to catch mismatches before any GPU work.
    let q_batch_needed = batch * q_dim;
    if q_batch.len() < q_batch_needed {
        return Err(RuntimeError::Compute(format!(
            "prefill_attention: q_batch too small: have {} elements, \
             need {} (batch={batch}, q_dim={q_dim})",
            q_batch.len(),
            q_batch_needed,
        )));
    }
    if attn_out_batch.len() < q_batch_needed {
        return Err(RuntimeError::Compute(format!(
            "prefill_attention: attn_out_batch too small: have {} elements, \
             need {} (batch={batch}, q_dim={q_dim})",
            attn_out_batch.len(),
            q_batch_needed,
        )));
    }
    if q_single.len() < q_dim {
        return Err(RuntimeError::Compute(format!(
            "prefill_attention: q_single too small: have {} elements, need {q_dim}",
            q_single.len(),
        )));
    }
    if attn_out_single.len() < q_dim {
        return Err(RuntimeError::Compute(format!(
            "prefill_attention: attn_out_single too small: have {} elements, need {q_dim}",
            attn_out_single.len(),
        )));
    }

    let scale = 1.0f32 / (head_dim as f32).sqrt();

    for t in 0..batch {
        let seq_len = pos_start + t + 1;

        // 1. Extract this token's Q vector from the batch matrix.
        unsafe {
            launch_extract_row(device, kernels, q_batch, q_single, t, q_dim)?;
        }

        // 2. Run attention_decode for this single token against the KV cache.
        let block_size = attention_block_size(seq_len);
        let shared_bytes = attention_shared_bytes(seq_len as u32);
        let launch_cfg = CudarcLaunchConfig {
            grid_dim: (num_heads as u32, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: shared_bytes,
        };
        let nh = num_heads as u32;
        let nkvh = num_kv_heads as u32;
        let hd = head_dim as u32;
        let sl = seq_len as u32;
        let msl = kv_cache.max_seq_len as u32;

        unsafe {
            device
                .stream
                .launch_builder(&kernels.attention_decode)
                .arg(q_single as &CudaSlice<f32>)
                .arg(&kv_cache.k_cache)
                .arg(&kv_cache.v_cache)
                .arg(&mut *attn_out_single)
                .arg(&nh)
                .arg(&nkvh)
                .arg(&hd)
                .arg(&sl)
                .arg(&msl)
                .arg(&scale)
                .launch(launch_cfg)
                .map_err(|e| {
                    RuntimeError::Compute(format!(
                        "attention_decode prefill t={t} seq_len={seq_len}: {e}"
                    ))
                })?;
        }

        // 3. Scatter-write the attention output back into the batch matrix.
        unsafe {
            launch_scatter_row(
                device,
                kernels,
                attn_out_batch,
                &*attn_out_single,
                t,
                q_dim,
            )?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    //! Tests for prefill_attention_sequential.
    //!
    //! These tests require a CUDA GPU. They are gated behind `#[cfg(feature = "cuda")]`
    //! at the crate level, so they only compile/run when `--features cuda` is active.
    //!
    //! Test strategy: compute attention for a small batch using the sequential
    //! function, then verify each token's output matches what we get by running
    //! the attention_decode kernel individually for the same token.

    use super::*;

    /// Reference implementation of single-head attention on CPU for validation.
    ///
    /// Computes: softmax(Q * K^T / sqrt(head_dim)) * V
    /// where Q is [head_dim], K is [seq_len, head_dim], V is [seq_len, head_dim].
    ///
    /// Returns [head_dim] output for one head.
    fn cpu_attention_single_head(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Compute scaled dot-product scores.
        let mut scores = vec![0.0f32; seq_len];
        for t in 0..seq_len {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[d] * k[t * head_dim + d];
            }
            scores[t] = dot * scale;
        }

        // Numerically stable softmax: subtract max before exp.
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max_score).exp();
            sum += *s;
        }
        let inv_sum = 1.0 / sum;
        for s in scores.iter_mut() {
            *s *= inv_sum;
        }

        // Weighted V accumulation.
        let mut out = vec![0.0f32; head_dim];
        for t in 0..seq_len {
            for d in 0..head_dim {
                out[d] += scores[t] * v[t * head_dim + d];
            }
        }
        out
    }

    /// Verify that prefill_attention_sequential produces correct causal attention
    /// by comparing against the CPU reference for batch=4, num_heads=2, num_kv_heads=2,
    /// head_dim=4.
    ///
    /// Each token t should attend only to positions 0..t (causal), so token 0 attends
    /// to 1 key, token 1 to 2 keys, etc.
    #[test]
    fn test_prefill_attention_matches_cpu_reference() {
        // Skip if no CUDA device available.
        if super::super::ffi::device_count().unwrap_or(0) == 0 {
            eprintln!("Skipping test: no CUDA device");
            return;
        }

        let device = match super::super::ffi::CudaDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Skipping test: failed to init CUDA device: {e}");
                return;
            }
        };

        let kernels = match super::super::decode::compile_all_kernels(&device) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("Skipping test: failed to compile kernels: {e}");
                return;
            }
        };

        let batch = 4;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let pos_start = 0;
        let max_seq_len = 16;

        // Create deterministic test data.
        // Q: [batch, q_dim] -- each element is (token_idx * q_dim + elem) * 0.1
        let q_data: Vec<f32> = (0..batch * q_dim)
            .map(|i| (i as f32) * 0.1)
            .collect();

        // K and V: we need to fill the KV cache for positions 0..batch-1.
        // K/V data per token: [kv_dim]
        let k_data: Vec<f32> = (0..batch * kv_dim)
            .map(|i| ((i as f32) * 0.05 + 0.3).sin())
            .collect();
        let v_data: Vec<f32> = (0..batch * kv_dim)
            .map(|i| ((i as f32) * 0.07 + 0.5).cos())
            .collect();

        // Upload Q batch to GPU.
        let q_batch = device.htod_copy(&q_data).unwrap();
        let mut attn_out_batch = device.alloc_zeros::<f32>(batch * q_dim).unwrap();
        let mut q_single = device.alloc_zeros::<f32>(q_dim).unwrap();
        let mut attn_out_single = device.alloc_zeros::<f32>(q_dim).unwrap();

        // Create and populate KV cache.
        let mut kv_cache = KvCacheGpu::new(&device, num_kv_heads, max_seq_len, head_dim).unwrap();

        // Write each token's K and V to the cache one at a time.
        for t in 0..batch {
            let k_token: Vec<f32> = k_data[t * kv_dim..(t + 1) * kv_dim].to_vec();
            let v_token: Vec<f32> = v_data[t * kv_dim..(t + 1) * kv_dim].to_vec();
            let k_gpu = device.htod_copy(&k_token).unwrap();
            let v_gpu = device.htod_copy(&v_token).unwrap();
            kv_cache.append_kv(&device, &k_gpu, &v_gpu).unwrap();
        }

        // Run the function under test.
        prefill_attention_sequential(
            &device,
            &kernels,
            &q_batch,
            &kv_cache,
            &mut attn_out_batch,
            batch,
            num_heads,
            num_kv_heads,
            head_dim,
            pos_start,
            &mut q_single,
            &mut attn_out_single,
        )
        .unwrap();

        device.synchronize().unwrap();

        // Read back GPU results.
        let gpu_results = device.dtoh_copy(&attn_out_batch).unwrap();

        // Compute CPU reference for each token and compare.
        for t in 0..batch {
            let seq_len = pos_start + t + 1;

            // For each head, compute the expected attention output.
            for h in 0..num_heads {
                let kv_h = h / (num_heads / num_kv_heads);

                // Extract Q for this head from token t.
                let q_offset = t * q_dim + h * head_dim;
                let q_head = &q_data[q_offset..q_offset + head_dim];

                // Build K and V matrices for this head up to seq_len positions.
                // KV cache layout: [num_kv_heads, max_seq_len, head_dim]
                // But our k_data is [batch, kv_dim] row-major, so for position p,
                // head kv_h, the K vector is at k_data[p * kv_dim + kv_h * head_dim].
                let mut k_matrix = vec![0.0f32; seq_len * head_dim];
                let mut v_matrix = vec![0.0f32; seq_len * head_dim];
                for p in 0..seq_len {
                    for d in 0..head_dim {
                        k_matrix[p * head_dim + d] =
                            k_data[p * kv_dim + kv_h * head_dim + d];
                        v_matrix[p * head_dim + d] =
                            v_data[p * kv_dim + kv_h * head_dim + d];
                    }
                }

                let expected = cpu_attention_single_head(
                    q_head, &k_matrix, &v_matrix, seq_len, head_dim,
                );

                // Compare against GPU output.
                let out_offset = t * q_dim + h * head_dim;
                for d in 0..head_dim {
                    let gpu_val = gpu_results[out_offset + d];
                    let cpu_val = expected[d];
                    let diff = (gpu_val - cpu_val).abs();
                    assert!(
                        diff < 1e-4,
                        "Mismatch at token={t}, head={h}, dim={d}: \
                         gpu={gpu_val}, cpu={cpu_val}, diff={diff}"
                    );
                }
            }
        }
    }

    /// Verify batch=1 edge case works correctly.
    #[test]
    fn test_prefill_attention_batch_1() {
        if super::super::ffi::device_count().unwrap_or(0) == 0 {
            eprintln!("Skipping test: no CUDA device");
            return;
        }

        let device = match super::super::ffi::CudaDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Skipping test: failed to init CUDA device: {e}");
                return;
            }
        };

        let kernels = match super::super::decode::compile_all_kernels(&device) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("Skipping test: failed to compile kernels: {e}");
                return;
            }
        };

        let batch = 1;
        let num_heads = 2;
        let num_kv_heads = 1; // GQA: 2 Q heads share 1 KV head
        let head_dim = 8;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let pos_start = 0;
        let max_seq_len = 16;

        let q_data: Vec<f32> = (0..q_dim).map(|i| (i as f32) * 0.2).collect();
        let k_data: Vec<f32> = (0..kv_dim).map(|i| ((i as f32) * 0.1).sin()).collect();
        let v_data: Vec<f32> = (0..kv_dim).map(|i| ((i as f32) * 0.15).cos()).collect();

        let q_batch = device.htod_copy(&q_data).unwrap();
        let mut attn_out_batch = device.alloc_zeros::<f32>(q_dim).unwrap();
        let mut q_single = device.alloc_zeros::<f32>(q_dim).unwrap();
        let mut attn_out_single = device.alloc_zeros::<f32>(q_dim).unwrap();

        let mut kv_cache = KvCacheGpu::new(&device, num_kv_heads, max_seq_len, head_dim).unwrap();
        let k_gpu = device.htod_copy(&k_data).unwrap();
        let v_gpu = device.htod_copy(&v_data).unwrap();
        kv_cache.append_kv(&device, &k_gpu, &v_gpu).unwrap();

        prefill_attention_sequential(
            &device,
            &kernels,
            &q_batch,
            &kv_cache,
            &mut attn_out_batch,
            batch,
            num_heads,
            num_kv_heads,
            head_dim,
            pos_start,
            &mut q_single,
            &mut attn_out_single,
        )
        .unwrap();

        device.synchronize().unwrap();
        let gpu_results = device.dtoh_copy(&attn_out_batch).unwrap();

        // With batch=1, seq_len=1: softmax of a single score is always 1.0,
        // so attention output = V[0] for each head.
        for h in 0..num_heads {
            let kv_h = h / (num_heads / num_kv_heads);
            for d in 0..head_dim {
                let gpu_val = gpu_results[h * head_dim + d];
                let expected = v_data[kv_h * head_dim + d];
                let diff = (gpu_val - expected).abs();
                assert!(
                    diff < 1e-4,
                    "batch=1 mismatch at head={h}, dim={d}: \
                     gpu={gpu_val}, expected={expected}, diff={diff}"
                );
            }
        }
    }

    /// Verify that pos_start > 0 works correctly (continuation of an existing sequence).
    #[test]
    fn test_prefill_attention_with_pos_offset() {
        if super::super::ffi::device_count().unwrap_or(0) == 0 {
            eprintln!("Skipping test: no CUDA device");
            return;
        }

        let device = match super::super::ffi::CudaDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Skipping test: failed to init CUDA device: {e}");
                return;
            }
        };

        let kernels = match super::super::decode::compile_all_kernels(&device) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("Skipping test: failed to compile kernels: {e}");
                return;
            }
        };

        // Simulate: 3 tokens already in cache, then prefill 2 more tokens.
        let pre_existing = 3;
        let batch = 2;
        let num_heads = 1;
        let num_kv_heads = 1;
        let head_dim = 4;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let pos_start = pre_existing;
        let max_seq_len = 16;

        // Pre-fill the KV cache with 3 tokens.
        let mut kv_cache = KvCacheGpu::new(&device, num_kv_heads, max_seq_len, head_dim).unwrap();

        let all_k: Vec<f32> = (0..(pre_existing + batch) * kv_dim)
            .map(|i| ((i as f32) * 0.05 + 0.1).sin())
            .collect();
        let all_v: Vec<f32> = (0..(pre_existing + batch) * kv_dim)
            .map(|i| ((i as f32) * 0.07 + 0.2).cos())
            .collect();

        for t in 0..(pre_existing + batch) {
            let k_token = &all_k[t * kv_dim..(t + 1) * kv_dim];
            let v_token = &all_v[t * kv_dim..(t + 1) * kv_dim];
            let k_gpu = device.htod_copy(k_token).unwrap();
            let v_gpu = device.htod_copy(v_token).unwrap();
            kv_cache.append_kv(&device, &k_gpu, &v_gpu).unwrap();
        }

        // Q for the 2 new tokens.
        let q_data: Vec<f32> = (0..batch * q_dim)
            .map(|i| (i as f32) * 0.3)
            .collect();

        let q_batch = device.htod_copy(&q_data).unwrap();
        let mut attn_out_batch = device.alloc_zeros::<f32>(batch * q_dim).unwrap();
        let mut q_single = device.alloc_zeros::<f32>(q_dim).unwrap();
        let mut attn_out_single = device.alloc_zeros::<f32>(q_dim).unwrap();

        prefill_attention_sequential(
            &device,
            &kernels,
            &q_batch,
            &kv_cache,
            &mut attn_out_batch,
            batch,
            num_heads,
            num_kv_heads,
            head_dim,
            pos_start,
            &mut q_single,
            &mut attn_out_single,
        )
        .unwrap();

        device.synchronize().unwrap();
        let gpu_results = device.dtoh_copy(&attn_out_batch).unwrap();

        // Verify against CPU reference.
        for t in 0..batch {
            let seq_len = pos_start + t + 1;

            for h in 0..num_heads {
                let kv_h = h;
                let q_offset = t * q_dim + h * head_dim;
                let q_head = &q_data[q_offset..q_offset + head_dim];

                let mut k_matrix = vec![0.0f32; seq_len * head_dim];
                let mut v_matrix = vec![0.0f32; seq_len * head_dim];
                for p in 0..seq_len {
                    for d in 0..head_dim {
                        k_matrix[p * head_dim + d] =
                            all_k[p * kv_dim + kv_h * head_dim + d];
                        v_matrix[p * head_dim + d] =
                            all_v[p * kv_dim + kv_h * head_dim + d];
                    }
                }

                let expected = cpu_attention_single_head(
                    q_head, &k_matrix, &v_matrix, seq_len, head_dim,
                );

                let out_offset = t * q_dim + h * head_dim;
                for d in 0..head_dim {
                    let gpu_val = gpu_results[out_offset + d];
                    let cpu_val = expected[d];
                    let diff = (gpu_val - cpu_val).abs();
                    assert!(
                        diff < 1e-4,
                        "pos_offset mismatch at token={t}(pos={}), head={h}, dim={d}: \
                         gpu={gpu_val}, cpu={cpu_val}, diff={diff}",
                        pos_start + t,
                    );
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Flash Attention v2 tests
    // ------------------------------------------------------------------

    /// Verify flash_attention_causal_v2 (Br=1) matches CPU reference for batch=4.
    #[test]
    fn test_flash_attention_v2_matches_cpu() {
        if super::super::ffi::device_count().unwrap_or(0) == 0 {
            eprintln!("Skipping test: no CUDA device");
            return;
        }

        let device = match super::super::ffi::CudaDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Skipping test: failed to init CUDA device: {e}");
                return;
            }
        };

        let kernels = match super::super::decode::compile_all_kernels(&device) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("Skipping test: failed to compile kernels: {e}");
                return;
            }
        };

        let batch = 4;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let pos_start = 0;
        let max_seq_len = 16;

        let q_data: Vec<f32> = (0..batch * q_dim)
            .map(|i| (i as f32) * 0.1)
            .collect();
        let k_data: Vec<f32> = (0..batch * kv_dim)
            .map(|i| ((i as f32) * 0.05 + 0.3).sin())
            .collect();
        let v_data: Vec<f32> = (0..batch * kv_dim)
            .map(|i| ((i as f32) * 0.07 + 0.5).cos())
            .collect();

        let q_batch = device.htod_copy(&q_data).unwrap();
        let mut attn_out = device.alloc_zeros::<f32>(batch * q_dim).unwrap();

        let mut kv_cache = KvCacheGpu::new(
            &device, num_kv_heads, max_seq_len, head_dim,
        ).unwrap();
        for t in 0..batch {
            let k_gpu = device.htod_copy(&k_data[t * kv_dim..(t + 1) * kv_dim]).unwrap();
            let v_gpu = device.htod_copy(&v_data[t * kv_dim..(t + 1) * kv_dim]).unwrap();
            kv_cache.append_kv(&device, &k_gpu, &v_gpu).unwrap();
        }

        unsafe {
            super::super::prefill::launch_flash_attention_v2(
                &device, &kernels,
                &q_batch, &kv_cache, &mut attn_out,
                batch, num_heads, num_kv_heads, head_dim, pos_start,
            ).unwrap();
        }
        device.synchronize().unwrap();
        let gpu_results = device.dtoh_copy(&attn_out).unwrap();

        for t in 0..batch {
            let seq_len = pos_start + t + 1;
            for h in 0..num_heads {
                let kv_h = h / (num_heads / num_kv_heads);
                let q_offset = t * q_dim + h * head_dim;
                let q_head = &q_data[q_offset..q_offset + head_dim];

                let mut k_matrix = vec![0.0f32; seq_len * head_dim];
                let mut v_matrix = vec![0.0f32; seq_len * head_dim];
                for p in 0..seq_len {
                    for d in 0..head_dim {
                        k_matrix[p * head_dim + d] =
                            k_data[p * kv_dim + kv_h * head_dim + d];
                        v_matrix[p * head_dim + d] =
                            v_data[p * kv_dim + kv_h * head_dim + d];
                    }
                }

                let expected = cpu_attention_single_head(
                    q_head, &k_matrix, &v_matrix, seq_len, head_dim,
                );

                let out_offset = t * q_dim + h * head_dim;
                for d in 0..head_dim {
                    let gpu_val = gpu_results[out_offset + d];
                    let cpu_val = expected[d];
                    let diff = (gpu_val - cpu_val).abs();
                    assert!(
                        diff < 1e-3,
                        "flash_v2 mismatch at token={t}, head={h}, dim={d}: \
                         gpu={gpu_val}, cpu={cpu_val}, diff={diff}"
                    );
                }
            }
        }
    }

    /// Verify flash_attention_causal_br4 (Br=4) matches CPU reference.
    #[test]
    fn test_flash_attention_br4_matches_cpu() {
        if super::super::ffi::device_count().unwrap_or(0) == 0 {
            eprintln!("Skipping test: no CUDA device");
            return;
        }

        let device = match super::super::ffi::CudaDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Skipping test: failed to init CUDA device: {e}");
                return;
            }
        };

        let kernels = match super::super::decode::compile_all_kernels(&device) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("Skipping test: failed to compile kernels: {e}");
                return;
            }
        };

        let batch = 7; // Not a multiple of 4 -- tests tail handling
        let num_heads = 2;
        let num_kv_heads = 1; // GQA: 2 Q heads share 1 KV head
        let head_dim = 8;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let pos_start = 0;
        let max_seq_len = 16;

        let q_data: Vec<f32> = (0..batch * q_dim)
            .map(|i| ((i as f32) * 0.13 + 0.7).sin())
            .collect();
        let k_data: Vec<f32> = (0..batch * kv_dim)
            .map(|i| ((i as f32) * 0.05 + 0.3).sin())
            .collect();
        let v_data: Vec<f32> = (0..batch * kv_dim)
            .map(|i| ((i as f32) * 0.07 + 0.5).cos())
            .collect();

        let q_batch = device.htod_copy(&q_data).unwrap();
        let mut attn_out = device.alloc_zeros::<f32>(batch * q_dim).unwrap();

        let mut kv_cache = KvCacheGpu::new(
            &device, num_kv_heads, max_seq_len, head_dim,
        ).unwrap();
        for t in 0..batch {
            let k_gpu = device.htod_copy(&k_data[t * kv_dim..(t + 1) * kv_dim]).unwrap();
            let v_gpu = device.htod_copy(&v_data[t * kv_dim..(t + 1) * kv_dim]).unwrap();
            kv_cache.append_kv(&device, &k_gpu, &v_gpu).unwrap();
        }

        unsafe {
            super::super::prefill::launch_flash_attention_br4(
                &device, &kernels,
                &q_batch, &kv_cache, &mut attn_out,
                batch, num_heads, num_kv_heads, head_dim, pos_start,
            ).unwrap();
        }
        device.synchronize().unwrap();
        let gpu_results = device.dtoh_copy(&attn_out).unwrap();

        for t in 0..batch {
            let seq_len = pos_start + t + 1;
            for h in 0..num_heads {
                let kv_h = h / (num_heads / num_kv_heads);
                let q_offset = t * q_dim + h * head_dim;
                let q_head = &q_data[q_offset..q_offset + head_dim];

                let mut k_matrix = vec![0.0f32; seq_len * head_dim];
                let mut v_matrix = vec![0.0f32; seq_len * head_dim];
                for p in 0..seq_len {
                    for d in 0..head_dim {
                        k_matrix[p * head_dim + d] =
                            k_data[p * kv_dim + kv_h * head_dim + d];
                        v_matrix[p * head_dim + d] =
                            v_data[p * kv_dim + kv_h * head_dim + d];
                    }
                }

                let expected = cpu_attention_single_head(
                    q_head, &k_matrix, &v_matrix, seq_len, head_dim,
                );

                let out_offset = t * q_dim + h * head_dim;
                for d in 0..head_dim {
                    let gpu_val = gpu_results[out_offset + d];
                    let cpu_val = expected[d];
                    let diff = (gpu_val - cpu_val).abs();
                    assert!(
                        diff < 1e-3,
                        "flash_br4 mismatch at token={t}, head={h}, dim={d}: \
                         gpu={gpu_val}, cpu={cpu_val}, diff={diff}"
                    );
                }
            }
        }
    }

    /// Verify flash attention with pos_start > 0 (continuation).
    #[test]
    fn test_flash_attention_v2_with_pos_offset() {
        if super::super::ffi::device_count().unwrap_or(0) == 0 {
            eprintln!("Skipping test: no CUDA device");
            return;
        }

        let device = match super::super::ffi::CudaDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Skipping test: failed to init CUDA device: {e}");
                return;
            }
        };

        let kernels = match super::super::decode::compile_all_kernels(&device) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("Skipping test: failed to compile kernels: {e}");
                return;
            }
        };

        let pre_existing = 3;
        let batch = 2;
        let num_heads = 1;
        let num_kv_heads = 1;
        let head_dim = 4;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let pos_start = pre_existing;
        let max_seq_len = 16;

        let mut kv_cache = KvCacheGpu::new(
            &device, num_kv_heads, max_seq_len, head_dim,
        ).unwrap();

        let all_k: Vec<f32> = (0..(pre_existing + batch) * kv_dim)
            .map(|i| ((i as f32) * 0.05 + 0.1).sin())
            .collect();
        let all_v: Vec<f32> = (0..(pre_existing + batch) * kv_dim)
            .map(|i| ((i as f32) * 0.07 + 0.2).cos())
            .collect();

        for t in 0..(pre_existing + batch) {
            let k_gpu = device.htod_copy(&all_k[t * kv_dim..(t + 1) * kv_dim]).unwrap();
            let v_gpu = device.htod_copy(&all_v[t * kv_dim..(t + 1) * kv_dim]).unwrap();
            kv_cache.append_kv(&device, &k_gpu, &v_gpu).unwrap();
        }

        let q_data: Vec<f32> = (0..batch * q_dim)
            .map(|i| (i as f32) * 0.3)
            .collect();

        let q_batch = device.htod_copy(&q_data).unwrap();
        let mut attn_out = device.alloc_zeros::<f32>(batch * q_dim).unwrap();

        unsafe {
            super::super::prefill::launch_flash_attention_v2(
                &device, &kernels,
                &q_batch, &kv_cache, &mut attn_out,
                batch, num_heads, num_kv_heads, head_dim, pos_start,
            ).unwrap();
        }
        device.synchronize().unwrap();
        let gpu_results = device.dtoh_copy(&attn_out).unwrap();

        for t in 0..batch {
            let seq_len = pos_start + t + 1;
            for h in 0..num_heads {
                let kv_h = h;
                let q_offset = t * q_dim + h * head_dim;
                let q_head = &q_data[q_offset..q_offset + head_dim];

                let mut k_matrix = vec![0.0f32; seq_len * head_dim];
                let mut v_matrix = vec![0.0f32; seq_len * head_dim];
                for p in 0..seq_len {
                    for d in 0..head_dim {
                        k_matrix[p * head_dim + d] =
                            all_k[p * kv_dim + kv_h * head_dim + d];
                        v_matrix[p * head_dim + d] =
                            all_v[p * kv_dim + kv_h * head_dim + d];
                    }
                }

                let expected = cpu_attention_single_head(
                    q_head, &k_matrix, &v_matrix, seq_len, head_dim,
                );

                let out_offset = t * q_dim + h * head_dim;
                for d in 0..head_dim {
                    let gpu_val = gpu_results[out_offset + d];
                    let cpu_val = expected[d];
                    let diff = (gpu_val - cpu_val).abs();
                    assert!(
                        diff < 1e-3,
                        "flash_v2 pos_offset mismatch at token={t}(pos={}), head={h}, dim={d}: \
                         gpu={gpu_val}, cpu={cpu_val}, diff={diff}",
                        pos_start + t,
                    );
                }
            }
        }
    }

    /// Verify flash attention v2 matches sequential attention (cross-validate kernels).
    #[test]
    fn test_flash_attention_v2_matches_sequential() {
        if super::super::ffi::device_count().unwrap_or(0) == 0 {
            eprintln!("Skipping test: no CUDA device");
            return;
        }

        let device = match super::super::ffi::CudaDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Skipping test: failed to init CUDA device: {e}");
                return;
            }
        };

        let kernels = match super::super::decode::compile_all_kernels(&device) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("Skipping test: failed to compile kernels: {e}");
                return;
            }
        };

        let batch = 8;
        let num_heads = 4;
        let num_kv_heads = 2; // GQA ratio = 2
        let head_dim = 16;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let pos_start = 0;
        let max_seq_len = 32;

        let q_data: Vec<f32> = (0..batch * q_dim)
            .map(|i| ((i as f32) * 0.11 + 0.3).sin())
            .collect();
        let k_data: Vec<f32> = (0..batch * kv_dim)
            .map(|i| ((i as f32) * 0.05 + 0.7).cos())
            .collect();
        let v_data: Vec<f32> = (0..batch * kv_dim)
            .map(|i| ((i as f32) * 0.07 + 1.1).sin())
            .collect();

        // Setup KV cache
        let mut kv_cache = KvCacheGpu::new(
            &device, num_kv_heads, max_seq_len, head_dim,
        ).unwrap();
        for t in 0..batch {
            let k_gpu = device.htod_copy(&k_data[t * kv_dim..(t + 1) * kv_dim]).unwrap();
            let v_gpu = device.htod_copy(&v_data[t * kv_dim..(t + 1) * kv_dim]).unwrap();
            kv_cache.append_kv(&device, &k_gpu, &v_gpu).unwrap();
        }

        // Run sequential attention (reference)
        let q_batch = device.htod_copy(&q_data).unwrap();
        let mut seq_out = device.alloc_zeros::<f32>(batch * q_dim).unwrap();
        let mut q_single = device.alloc_zeros::<f32>(q_dim).unwrap();
        let mut attn_out_single = device.alloc_zeros::<f32>(q_dim).unwrap();

        prefill_attention_sequential(
            &device, &kernels,
            &q_batch, &kv_cache, &mut seq_out,
            batch, num_heads, num_kv_heads, head_dim, pos_start,
            &mut q_single, &mut attn_out_single,
        ).unwrap();
        device.synchronize().unwrap();
        let seq_results = device.dtoh_copy(&seq_out).unwrap();

        // Run flash attention v2
        let mut flash_out = device.alloc_zeros::<f32>(batch * q_dim).unwrap();
        unsafe {
            super::super::prefill::launch_flash_attention_v2(
                &device, &kernels,
                &q_batch, &kv_cache, &mut flash_out,
                batch, num_heads, num_kv_heads, head_dim, pos_start,
            ).unwrap();
        }
        device.synchronize().unwrap();
        let flash_results = device.dtoh_copy(&flash_out).unwrap();

        // Compare: flash vs sequential
        let mut max_diff = 0.0f32;
        for i in 0..seq_results.len() {
            let diff = (seq_results[i] - flash_results[i]).abs();
            max_diff = max_diff.max(diff);
            assert!(
                diff < 1e-3,
                "flash vs sequential mismatch at index {i}: \
                 seq={}, flash={}, diff={diff}",
                seq_results[i], flash_results[i],
            );
        }
        eprintln!("flash_v2 vs sequential: max diff = {max_diff:.6e}");
    }

    /// Verify buffer size validation catches undersized buffers.
    #[test]
    fn test_prefill_attention_buffer_validation() {
        if super::super::ffi::device_count().unwrap_or(0) == 0 {
            eprintln!("Skipping test: no CUDA device");
            return;
        }

        let device = match super::super::ffi::CudaDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Skipping test: failed to init CUDA device: {e}");
                return;
            }
        };

        let kernels = match super::super::decode::compile_all_kernels(&device) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("Skipping test: failed to compile kernels: {e}");
                return;
            }
        };

        let batch = 4;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let q_dim = num_heads * head_dim;

        let kv_cache = KvCacheGpu::new(&device, num_kv_heads, 16, head_dim).unwrap();

        // q_batch too small: allocate only 1 row instead of batch rows.
        let q_batch = device.alloc_zeros::<f32>(q_dim).unwrap(); // should be batch * q_dim
        let mut attn_out_batch = device.alloc_zeros::<f32>(batch * q_dim).unwrap();
        let mut q_single = device.alloc_zeros::<f32>(q_dim).unwrap();
        let mut attn_out_single = device.alloc_zeros::<f32>(q_dim).unwrap();

        let result = prefill_attention_sequential(
            &device,
            &kernels,
            &q_batch,
            &kv_cache,
            &mut attn_out_batch,
            batch,
            num_heads,
            num_kv_heads,
            head_dim,
            0,
            &mut q_single,
            &mut attn_out_single,
        );

        assert!(result.is_err(), "Expected error for undersized q_batch");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("q_batch too small"),
            "Error should mention q_batch: {err_msg}"
        );
    }
}
