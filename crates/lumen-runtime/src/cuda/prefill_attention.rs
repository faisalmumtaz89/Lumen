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

use cudarc::driver::CudaSlice;

use crate::error::RuntimeError;

use super::decode::KernelSet;
use super::ffi::CudaDevice;
#[cfg(test)]
use super::kv_cache::KvCacheGpu;
use super::kv_cache::{KvRef, KvView};
use super::prefill::{launch_attention_decode_gated, launch_extract_row, launch_scatter_row};

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
/// * `kv` - F32 view of the KV cache with data for positions 0..pos_start+batch-1
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
    kv: &KvView<'_>,
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

        // 2. Run decode-attention for this single token against the KV cache.
        // gate: routes to the tiled streaming-softmax kernel at long
        // context. Byte-identical to the prior single-block dispatch when
        // the gate selects SingleBlock (the default for typical prefill shapes
        // within the single-block ceiling).
        let nh = num_heads as u32;
        let nkvh = num_kv_heads as u32;
        let hd = head_dim as u32;
        let sl = seq_len as u32;
        let msl = kv.seq_stride as u32;

        unsafe {
            launch_attention_decode_gated(
                device,
                kernels,
                q_single as &CudaSlice<f32>,
                KvRef::F32 { k: kv.k, v: kv.v },
                None,
                &mut *attn_out_single,
                nh,
                nkvh,
                hd,
                sl,
                msl,
                scale,
            )
            .map_err(|e| {
                RuntimeError::Compute(format!(
                    "attention_decode prefill t={t} seq_len={seq_len}: {e}"
                ))
            })?;
        }

        // 3. Scatter-write the attention output back into the batch matrix.
        unsafe {
            launch_scatter_row(device, kernels, attn_out_batch, &*attn_out_single, t, q_dim)?;
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

        let kernels = match super::super::decode::compile_all_kernels(
            &device,
            crate::kv::KvPrecision::F32,
            None,
        ) {
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
        let q_data: Vec<f32> = (0..batch * q_dim).map(|i| (i as f32) * 0.1).collect();

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
            kv_cache.append_kv(&device, &k_gpu, &v_gpu, None).unwrap();
        }

        // Run the function under test.
        prefill_attention_sequential(
            &device,
            &kernels,
            &q_batch,
            &kv_cache.f32_view().unwrap(),
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
                        k_matrix[p * head_dim + d] = k_data[p * kv_dim + kv_h * head_dim + d];
                        v_matrix[p * head_dim + d] = v_data[p * kv_dim + kv_h * head_dim + d];
                    }
                }

                let expected =
                    cpu_attention_single_head(q_head, &k_matrix, &v_matrix, seq_len, head_dim);

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

        let kernels = match super::super::decode::compile_all_kernels(
            &device,
            crate::kv::KvPrecision::F32,
            None,
        ) {
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
        kv_cache.append_kv(&device, &k_gpu, &v_gpu, None).unwrap();

        prefill_attention_sequential(
            &device,
            &kernels,
            &q_batch,
            &kv_cache.f32_view().unwrap(),
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

        let kernels = match super::super::decode::compile_all_kernels(
            &device,
            crate::kv::KvPrecision::F32,
            None,
        ) {
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
            kv_cache.append_kv(&device, &k_gpu, &v_gpu, None).unwrap();
        }

        // Q for the 2 new tokens.
        let q_data: Vec<f32> = (0..batch * q_dim).map(|i| (i as f32) * 0.3).collect();

        let q_batch = device.htod_copy(&q_data).unwrap();
        let mut attn_out_batch = device.alloc_zeros::<f32>(batch * q_dim).unwrap();
        let mut q_single = device.alloc_zeros::<f32>(q_dim).unwrap();
        let mut attn_out_single = device.alloc_zeros::<f32>(q_dim).unwrap();

        prefill_attention_sequential(
            &device,
            &kernels,
            &q_batch,
            &kv_cache.f32_view().unwrap(),
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
                        k_matrix[p * head_dim + d] = all_k[p * kv_dim + kv_h * head_dim + d];
                        v_matrix[p * head_dim + d] = all_v[p * kv_dim + kv_h * head_dim + d];
                    }
                }

                let expected =
                    cpu_attention_single_head(q_head, &k_matrix, &v_matrix, seq_len, head_dim);

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

        let kernels = match super::super::decode::compile_all_kernels(
            &device,
            crate::kv::KvPrecision::F32,
            None,
        ) {
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

        let q_data: Vec<f32> = (0..batch * q_dim).map(|i| (i as f32) * 0.1).collect();
        let k_data: Vec<f32> = (0..batch * kv_dim)
            .map(|i| ((i as f32) * 0.05 + 0.3).sin())
            .collect();
        let v_data: Vec<f32> = (0..batch * kv_dim)
            .map(|i| ((i as f32) * 0.07 + 0.5).cos())
            .collect();

        let q_batch = device.htod_copy(&q_data).unwrap();
        let mut attn_out = device.alloc_zeros::<f32>(batch * q_dim).unwrap();

        let mut kv_cache = KvCacheGpu::new(&device, num_kv_heads, max_seq_len, head_dim).unwrap();
        for t in 0..batch {
            let k_gpu = device
                .htod_copy(&k_data[t * kv_dim..(t + 1) * kv_dim])
                .unwrap();
            let v_gpu = device
                .htod_copy(&v_data[t * kv_dim..(t + 1) * kv_dim])
                .unwrap();
            kv_cache.append_kv(&device, &k_gpu, &v_gpu, None).unwrap();
        }

        unsafe {
            super::super::prefill::launch_flash_attention_v2(
                &device,
                &kernels,
                &q_batch,
                &kv_cache.f32_view().unwrap(),
                &mut attn_out,
                batch,
                num_heads,
                num_kv_heads,
                head_dim,
                pos_start,
            )
            .unwrap();
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
                        k_matrix[p * head_dim + d] = k_data[p * kv_dim + kv_h * head_dim + d];
                        v_matrix[p * head_dim + d] = v_data[p * kv_dim + kv_h * head_dim + d];
                    }
                }

                let expected =
                    cpu_attention_single_head(q_head, &k_matrix, &v_matrix, seq_len, head_dim);

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

        let kernels = match super::super::decode::compile_all_kernels(
            &device,
            crate::kv::KvPrecision::F32,
            None,
        ) {
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

        let mut kv_cache = KvCacheGpu::new(&device, num_kv_heads, max_seq_len, head_dim).unwrap();
        for t in 0..batch {
            let k_gpu = device
                .htod_copy(&k_data[t * kv_dim..(t + 1) * kv_dim])
                .unwrap();
            let v_gpu = device
                .htod_copy(&v_data[t * kv_dim..(t + 1) * kv_dim])
                .unwrap();
            kv_cache.append_kv(&device, &k_gpu, &v_gpu, None).unwrap();
        }

        unsafe {
            super::super::prefill::launch_flash_attention_br4(
                &device,
                &kernels,
                &q_batch,
                &kv_cache.f32_view().unwrap(),
                &mut attn_out,
                batch,
                num_heads,
                num_kv_heads,
                head_dim,
                pos_start,
            )
            .unwrap();
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
                        k_matrix[p * head_dim + d] = k_data[p * kv_dim + kv_h * head_dim + d];
                        v_matrix[p * head_dim + d] = v_data[p * kv_dim + kv_h * head_dim + d];
                    }
                }

                let expected =
                    cpu_attention_single_head(q_head, &k_matrix, &v_matrix, seq_len, head_dim);

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

        let kernels = match super::super::decode::compile_all_kernels(
            &device,
            crate::kv::KvPrecision::F32,
            None,
        ) {
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

        let mut kv_cache = KvCacheGpu::new(&device, num_kv_heads, max_seq_len, head_dim).unwrap();

        let all_k: Vec<f32> = (0..(pre_existing + batch) * kv_dim)
            .map(|i| ((i as f32) * 0.05 + 0.1).sin())
            .collect();
        let all_v: Vec<f32> = (0..(pre_existing + batch) * kv_dim)
            .map(|i| ((i as f32) * 0.07 + 0.2).cos())
            .collect();

        for t in 0..(pre_existing + batch) {
            let k_gpu = device
                .htod_copy(&all_k[t * kv_dim..(t + 1) * kv_dim])
                .unwrap();
            let v_gpu = device
                .htod_copy(&all_v[t * kv_dim..(t + 1) * kv_dim])
                .unwrap();
            kv_cache.append_kv(&device, &k_gpu, &v_gpu, None).unwrap();
        }

        let q_data: Vec<f32> = (0..batch * q_dim).map(|i| (i as f32) * 0.3).collect();

        let q_batch = device.htod_copy(&q_data).unwrap();
        let mut attn_out = device.alloc_zeros::<f32>(batch * q_dim).unwrap();

        unsafe {
            super::super::prefill::launch_flash_attention_v2(
                &device,
                &kernels,
                &q_batch,
                &kv_cache.f32_view().unwrap(),
                &mut attn_out,
                batch,
                num_heads,
                num_kv_heads,
                head_dim,
                pos_start,
            )
            .unwrap();
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
                        k_matrix[p * head_dim + d] = all_k[p * kv_dim + kv_h * head_dim + d];
                        v_matrix[p * head_dim + d] = all_v[p * kv_dim + kv_h * head_dim + d];
                    }
                }

                let expected =
                    cpu_attention_single_head(q_head, &k_matrix, &v_matrix, seq_len, head_dim);

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

        let kernels = match super::super::decode::compile_all_kernels(
            &device,
            crate::kv::KvPrecision::F32,
            None,
        ) {
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
        let mut kv_cache = KvCacheGpu::new(&device, num_kv_heads, max_seq_len, head_dim).unwrap();
        for t in 0..batch {
            let k_gpu = device
                .htod_copy(&k_data[t * kv_dim..(t + 1) * kv_dim])
                .unwrap();
            let v_gpu = device
                .htod_copy(&v_data[t * kv_dim..(t + 1) * kv_dim])
                .unwrap();
            kv_cache.append_kv(&device, &k_gpu, &v_gpu, None).unwrap();
        }

        // Run sequential attention (reference)
        let q_batch = device.htod_copy(&q_data).unwrap();
        let mut seq_out = device.alloc_zeros::<f32>(batch * q_dim).unwrap();
        let mut q_single = device.alloc_zeros::<f32>(q_dim).unwrap();
        let mut attn_out_single = device.alloc_zeros::<f32>(q_dim).unwrap();

        prefill_attention_sequential(
            &device,
            &kernels,
            &q_batch,
            &kv_cache.f32_view().unwrap(),
            &mut seq_out,
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
        let seq_results = device.dtoh_copy(&seq_out).unwrap();

        // Run flash attention v2
        let mut flash_out = device.alloc_zeros::<f32>(batch * q_dim).unwrap();
        unsafe {
            super::super::prefill::launch_flash_attention_v2(
                &device,
                &kernels,
                &q_batch,
                &kv_cache.f32_view().unwrap(),
                &mut flash_out,
                batch,
                num_heads,
                num_kv_heads,
                head_dim,
                pos_start,
            )
            .unwrap();
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
                seq_results[i],
                flash_results[i],
            );
        }
        eprintln!("flash_v2 vs sequential: max diff = {max_diff:.6e}");
    }

    // ------------------------------------------------------------------
    // FA2 block-skip + Split-K tests
    // ------------------------------------------------------------------

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

        let kernels = match super::super::decode::compile_all_kernels(
            &device,
            crate::kv::KvPrecision::F32,
            None,
        ) {
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
            &kv_cache.f32_view().unwrap(),
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

    // ------------------------------------------------------------------
    // Tiled SGEMM prefill attention (launch_flash_attention_sgemm)
    // ------------------------------------------------------------------

    /// Absolute tolerance for the tiled SGEMM path against the CPU reference.
    ///
    /// Both sides are exact F32 with no F16 carrier, so the only difference is
    /// summation order: a BLAS SGEMM blocks each `head_dim`-long dot product
    /// and each `kv_len`-long P·V column its own way, and the block softmax
    /// folds four warp trees where the reference sums left to right. Replaying
    /// these four shapes in F32 on the host, sequential accumulation against
    /// blocked BLAS, moves the output by at most 2.6e-6 (outputs are bounded by
    /// max |V| = 1); 5e-4 leaves two orders of headroom over that and stays far
    /// below any real defect -- a transposed operand, a wrong stride or a
    /// missing mask moves the output by O(1).
    const SGEMM_CPU_TOL: f32 = 5e-4;

    /// Absolute tolerance between the tiled SGEMM path and `br4` on the same
    /// inputs. Looser than `SGEMM_CPU_TOL` because br4's online softmax
    /// rescales its running sum tile by tile, a third summation order with its
    /// own drift: the br4-vs-CPU test above allows 1e-3 at head_dim 8.
    const SGEMM_BR4_TOL: f32 = 2e-3;

    /// Run one shape through `launch_flash_attention_sgemm` and check it against
    /// both the CPU reference and `launch_flash_attention_br4` on the same
    /// inputs. Returns early (skips) when no CUDA device is present, like the
    /// other GPU tests in this module.
    fn check_sgemm_shape(
        case: &str,
        batch: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        pos_start: usize,
    ) {
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
        let kernels = match super::super::decode::compile_all_kernels(
            &device,
            crate::kv::KvPrecision::F32,
            None,
        ) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("Skipping test: failed to compile kernels: {e}");
                return;
            }
        };

        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let kv_total = pos_start + batch;
        let max_seq_len = kv_total + 3; // room past the last position

        // Deterministic inputs bounded by 1 in magnitude.
        let q_data: Vec<f32> = (0..batch * q_dim)
            .map(|i| ((i as f32) * 0.011 + 0.7).sin())
            .collect();
        let all_k: Vec<f32> = (0..kv_total * kv_dim)
            .map(|i| ((i as f32) * 0.013 + 0.3).sin())
            .collect();
        let all_v: Vec<f32> = (0..kv_total * kv_dim)
            .map(|i| ((i as f32) * 0.017 + 0.5).cos())
            .collect();

        // Fill the cache in its head-first [head][pos][dim] layout in one copy;
        // appending 1000+ tokens one kernel at a time would dominate the test.
        let mut kv_cache = KvCacheGpu::new(&device, num_kv_heads, max_seq_len, head_dim).unwrap();
        let mut k_host = vec![0.0f32; num_kv_heads * max_seq_len * head_dim];
        let mut v_host = vec![0.0f32; num_kv_heads * max_seq_len * head_dim];
        for kv_h in 0..num_kv_heads {
            for p in 0..kv_total {
                for d in 0..head_dim {
                    let dst = (kv_h * max_seq_len + p) * head_dim + d;
                    let src = p * kv_dim + kv_h * head_dim + d;
                    k_host[dst] = all_k[src];
                    v_host[dst] = all_v[src];
                }
            }
        }
        match &mut kv_cache.store {
            crate::cuda::kv_cache::KvStore::F32 { k, v } => {
                device.htod_copy_into(&k_host, k).unwrap();
                device.htod_copy_into(&v_host, v).unwrap();
            }
            crate::cuda::kv_cache::KvStore::F16 { .. } => {
                unreachable!("KvCacheGpu::new allocates F32")
            }
        }
        kv_cache.advance_seq_len_by(kv_total);

        let q_batch = device.htod_copy(&q_data).unwrap();
        let mut out_sgemm = device.alloc_zeros::<f32>(batch * q_dim).unwrap();
        let mut out_br4 = device.alloc_zeros::<f32>(batch * q_dim).unwrap();
        // The score block is sized up front, as the prefill scratch does it.
        let score_elems = super::super::prefill::attn_score_block_elems(
            batch,
            num_heads,
            num_kv_heads,
            pos_start,
        )
        .unwrap();
        let mut scores: Option<cudarc::driver::CudaSlice<f32>> =
            Some(device.alloc_zeros::<f32>(score_elems).unwrap());

        unsafe {
            super::super::prefill::launch_flash_attention_sgemm(
                &device,
                &kernels,
                &q_batch,
                &kv_cache.f32_view().unwrap(),
                &mut out_sgemm,
                &mut scores,
                batch,
                num_heads,
                num_kv_heads,
                head_dim,
                pos_start,
            )
            .unwrap();
            super::super::prefill::launch_flash_attention_br4(
                &device,
                &kernels,
                &q_batch,
                &kv_cache.f32_view().unwrap(),
                &mut out_br4,
                batch,
                num_heads,
                num_kv_heads,
                head_dim,
                pos_start,
            )
            .unwrap();
        }
        device.synchronize().unwrap();
        let got_sgemm = device.dtoh_copy(&out_sgemm).unwrap();
        let got_br4 = device.dtoh_copy(&out_br4).unwrap();

        let group = num_heads / num_kv_heads;
        for kv_h in 0..num_kv_heads {
            // Per-head K/V matrices, built once for the longest row.
            let mut k_mat = vec![0.0f32; kv_total * head_dim];
            let mut v_mat = vec![0.0f32; kv_total * head_dim];
            for p in 0..kv_total {
                for d in 0..head_dim {
                    k_mat[p * head_dim + d] = all_k[p * kv_dim + kv_h * head_dim + d];
                    v_mat[p * head_dim + d] = all_v[p * kv_dim + kv_h * head_dim + d];
                }
            }
            for h in kv_h * group..(kv_h + 1) * group {
                for t in 0..batch {
                    let seq_len = pos_start + t + 1;
                    let q_off = t * q_dim + h * head_dim;
                    let expected = cpu_attention_single_head(
                        &q_data[q_off..q_off + head_dim],
                        &k_mat[..seq_len * head_dim],
                        &v_mat[..seq_len * head_dim],
                        seq_len,
                        head_dim,
                    );
                    for d in 0..head_dim {
                        let got = got_sgemm[q_off + d];
                        let cpu = expected[d];
                        assert!(
                            (got - cpu).abs() < SGEMM_CPU_TOL,
                            "{case}: sgemm vs cpu at token={t}(pos={}), head={h}, dim={d}: \
                             gpu={got}, cpu={cpu}, diff={}",
                            pos_start + t,
                            (got - cpu).abs(),
                        );
                        let br4 = got_br4[q_off + d];
                        assert!(
                            (got - br4).abs() < SGEMM_BR4_TOL,
                            "{case}: sgemm vs br4 at token={t}(pos={}), head={h}, dim={d}: \
                             sgemm={got}, br4={br4}, diff={}",
                            pos_start + t,
                            (got - br4).abs(),
                        );
                    }
                }
            }
        }
    }

    /// Multi-block batch (600 rows = one full 512 block + an 88-row tail),
    /// GQA group 4, head_dim 128, from an empty cache.
    #[test]
    fn test_flash_attention_sgemm_multi_block_group4() {
        check_sgemm_shape("multi_block_group4", 600, 4, 1, 128, 0);
    }

    /// Multi-block batch with a 5-row tail (517 rows), no GQA (group 1),
    /// head_dim 256 -- the 9B's shape.
    #[test]
    fn test_flash_attention_sgemm_multi_block_group1_hd256() {
        check_sgemm_shape("multi_block_group1_hd256", 517, 2, 2, 256, 0);
    }

    /// Single block, batch a multiple of nothing, pos_start > 0: the causal
    /// bound must start at pos_start, not at 0.
    #[test]
    fn test_flash_attention_sgemm_pos_offset_group4() {
        check_sgemm_shape("pos_offset_group4", 37, 8, 2, 128, 11);
    }

    /// Long prefix (pos_start 1023) with a short continuation batch and
    /// head_dim 256: every row of the block is nearly fully unmasked.
    #[test]
    fn test_flash_attention_sgemm_long_prefix_hd256() {
        check_sgemm_shape("long_prefix_hd256", 13, 1, 1, 256, 1023);
    }

    /// Undersized `q` and `attn_out` are rejected before any GPU work.
    #[test]
    fn test_flash_attention_sgemm_buffer_validation() {
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
        let kernels = match super::super::decode::compile_all_kernels(
            &device,
            crate::kv::KvPrecision::F32,
            None,
        ) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("Skipping test: failed to compile kernels: {e}");
                return;
            }
        };

        let batch = 8;
        let num_heads = 2;
        let num_kv_heads = 1;
        let head_dim = 8;
        let q_dim = num_heads * head_dim;
        let kv_cache = KvCacheGpu::new(&device, num_kv_heads, 16, head_dim).unwrap();

        let small_q = device.alloc_zeros::<f32>(q_dim).unwrap();
        let full_q = device.alloc_zeros::<f32>(batch * q_dim).unwrap();
        let mut full_out = device.alloc_zeros::<f32>(batch * q_dim).unwrap();
        let mut small_out = device.alloc_zeros::<f32>(q_dim).unwrap();
        let mut scores: Option<cudarc::driver::CudaSlice<f32>> = None;

        let err = unsafe {
            super::super::prefill::launch_flash_attention_sgemm(
                &device,
                &kernels,
                &small_q,
                &kv_cache.f32_view().unwrap(),
                &mut full_out,
                &mut scores,
                batch,
                num_heads,
                num_kv_heads,
                head_dim,
                0,
            )
        }
        .unwrap_err()
        .to_string();
        assert!(err.contains("q too small"), "unexpected error: {err}");

        let err = unsafe {
            super::super::prefill::launch_flash_attention_sgemm(
                &device,
                &kernels,
                &full_q,
                &kv_cache.f32_view().unwrap(),
                &mut small_out,
                &mut scores,
                batch,
                num_heads,
                num_kv_heads,
                head_dim,
                0,
            )
        }
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("attn_out too small"),
            "unexpected error: {err}"
        );
        // Valid buffers but no score block: the route reports its contract
        // rather than allocating for itself.
        let mut scores_none: Option<cudarc::driver::CudaSlice<f32>> = None;
        let err = unsafe {
            super::super::prefill::launch_flash_attention_sgemm(
                &device,
                &kernels,
                &full_q,
                &kv_cache.f32_view().unwrap(),
                &mut full_out,
                &mut scores_none,
                batch,
                num_heads,
                num_kv_heads,
                head_dim,
                0,
            )
        }
        .unwrap_err()
        .to_string();
        assert!(err.contains("no score block"), "unexpected error: {err}");
        assert!(
            scores_none.is_none(),
            "the launcher must not allocate the block"
        );
    }

    // ------------------------------------------------------------------
    // Host-side arithmetic of the tiled SGEMM path (no GPU)
    // ------------------------------------------------------------------

    /// The block walk covers every query row exactly once, in blocks of at most
    /// `ATTN_PREFILL_SGEMM_ROWS`, and always advances.
    #[test]
    fn test_attn_sgemm_block_rows_partitions_the_batch() {
        use super::super::prefill::{attn_sgemm_block_rows, ATTN_PREFILL_SGEMM_ROWS};
        for batch in [1usize, 15, 511, 512, 513, 600, 1024, 1025, 4096] {
            let mut qb = 0usize;
            let mut blocks = 0usize;
            while qb < batch {
                let rows = attn_sgemm_block_rows(batch, qb);
                assert!(rows > 0, "batch={batch} qb={qb}: zero-row block would hang");
                assert!(rows <= ATTN_PREFILL_SGEMM_ROWS, "batch={batch} qb={qb}");
                qb += rows;
                blocks += 1;
            }
            assert_eq!(qb, batch, "batch={batch}: walk overshot or undershot");
            assert_eq!(
                blocks,
                batch.div_ceil(ATTN_PREFILL_SGEMM_ROWS),
                "batch={batch}"
            );
        }
        assert_eq!(attn_sgemm_block_rows(0, 0), 0);
    }

    /// The score scratch allocated once up front -- `group * min(batch, 512) *
    /// (pos_start + batch)` floats -- covers every block's `group * rows *
    /// kv_len` footprint, for every block of every shape.
    #[test]
    fn test_attn_sgemm_score_scratch_bounds_every_block() {
        use super::super::prefill::attn_sgemm_block_rows;
        for group in [1usize, 4, 8] {
            for batch in [1usize, 37, 512, 513, 600, 1300] {
                for pos_start in [0usize, 1, 1023] {
                    let allocated = super::super::prefill::attn_score_block_elems(
                        batch,
                        group * 2,
                        2,
                        pos_start,
                    )
                    .expect("a producible geometry has a score block size");
                    let mut qb = 0usize;
                    while qb < batch {
                        let rows = attn_sgemm_block_rows(batch, qb);
                        let kv_len = pos_start + qb + rows;
                        assert!(
                            group * rows * kv_len <= allocated,
                            "group={group} batch={batch} pos_start={pos_start} qb={qb}: \
                             block needs {} of {allocated} floats",
                            group * rows * kv_len,
                        );
                        qb += rows;
                    }
                }
            }
        }
    }

    /// The softmax kernel's four-warp fold and the launch's block size are one
    /// constant on two sides of NVRTC; the shader also carries a
    /// `static_assert` on the warp count.
    #[test]
    fn test_attn_softmax_causal_threads_match_the_shader() {
        use super::super::prefill::ATTN_SOFTMAX_CAUSAL_THREADS;
        let src = super::super::shaders::ATTN_SOFTMAX_CAUSAL_KERNEL_SOURCE;
        assert!(
            src.contains(&format!(
                "#define SMX_THREADS {ATTN_SOFTMAX_CAUSAL_THREADS}u"
            )),
            "attn_softmax_causal.cu must define SMX_THREADS as \
             {ATTN_SOFTMAX_CAUSAL_THREADS} to match the launch"
        );
        assert!(
            src.contains("static_assert(SMX_WARPS == 4u"),
            "the four-warp fold must stay guarded"
        );
        assert_eq!(ATTN_SOFTMAX_CAUSAL_THREADS % 32, 0);
    }

    fn host_f16_bits(x: f32) -> u16 {
        // Round to nearest even, normal range only (the inputs below are
        // small normals).
        let b = x.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = ((b >> 23) & 0xff) as i32;
        let mant = b & 0x7f_ffff;
        if x == 0.0 {
            return sign;
        }
        let e = exp - 127 + 15;
        assert!((1..0x1f).contains(&e), "test inputs must be half normals");
        let half_m = mant >> 13;
        let rem = mant & 0x1fff;
        let round_up = rem > 0x1000 || (rem == 0x1000 && (half_m & 1) == 1);
        sign | (((e as u32) << 10) | half_m) as u16 + u16::from(round_up)
    }

    fn host_f16_to_f32(h: u16) -> f32 {
        let sign = u32::from(h & 0x8000) << 16;
        let exp = u32::from((h >> 10) & 0x1f);
        let mant = u32::from(h & 0x3ff);
        assert!(exp != 0 && exp != 0x1f);
        f32::from_bits(sign | ((exp + 127 - 15) << 23) | (mant << 13))
    }

    /// The typed dispatch: the same Q over the same half-representable K/V
    /// held in an F32 store and in a half store produces bit-identical
    /// output, the half arm taking the half twin of whatever route the F32
    /// arm took (the GQA-shared pair where it is loaded and admitted, the
    /// tiled kernel otherwise).
    #[test]
    fn half_store_dispatch_reproduces_the_f32_store_bit_for_bit() {
        use super::super::decode::AttentionDecodeVariant as V;
        use crate::cuda::kv_cache::{compile_kv_module, KvCacheGpu, KvStore};
        use crate::kv::KvPrecision;
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
        // The split-K route and the GQA-shared pair are on by default only for
        // the model and device classes a bare kernel set does not declare:
        // force both on so the arm this test exists for is the one that runs.
        let _guard = crate::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        std::env::set_var("LUMEN_CUDA_ATTN_SPLITK", "1");
        std::env::set_var("LUMEN_CUDA_ATTN_SPLITK_GQA6", "1");
        std::env::remove_var("LUMEN_CUDA_ATTN_SPLITK_GQA6_MAX_CHUNKS");
        std::env::remove_var("LUMEN_CUDA_ATTN_SPLITK_GQA6_ONE_TILE");
        std::env::remove_var("LUMEN_CUDA_ATTN_SPLITK_GQA6_TARGET");
        struct Unset;
        impl Drop for Unset {
            fn drop(&mut self) {
                std::env::remove_var("LUMEN_CUDA_ATTN_SPLITK");
                std::env::remove_var("LUMEN_CUDA_ATTN_SPLITK_GQA6");
            }
        }
        let _unset = Unset;
        let kernels = match super::super::decode::compile_all_kernels(
            &device,
            KvPrecision::F16,
            Some(super::super::prefill::ATTN_SPLITK_GQA6_REVIEWED),
        ) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("Skipping test: failed to compile kernels: {e}");
                return;
            }
        };
        assert!(
            kernels.attention_decode_splitk_partial_gqa6.is_some() && kernels.kv_f16.is_some(),
            "the GQA-shared pair and the half twins must load for this test to mean anything"
        );
        let (num_heads, num_kv_heads, head_dim) = (24usize, 4usize, 256usize);
        let max_seq_len = 16_400usize;
        let cache = num_kv_heads * max_seq_len * head_dim;
        let q: Vec<f32> = (0..num_heads * head_dim)
            .map(|i| ((i as f32) * 0.011 + 0.7).sin() * 4.0)
            .collect();
        let k16: Vec<u16> = (0..cache)
            .map(|i| host_f16_bits(((i as f32) * 0.013 + 0.3).sin() * 0.5 + 0.75))
            .collect();
        let v16: Vec<u16> = (0..cache)
            .map(|i| host_f16_bits(((i as f32) * 0.017 + 0.5).cos() * 0.5 + 0.75))
            .collect();
        let k32: Vec<f32> = k16.iter().map(|&h| host_f16_to_f32(h)).collect();
        let v32: Vec<f32> = v16.iter().map(|&h| host_f16_to_f32(h)).collect();

        let m32 = compile_kv_module(&device, KvPrecision::F32).unwrap();
        let mut kv32 = KvCacheGpu::with_module_at(
            &device,
            num_kv_heads,
            max_seq_len,
            head_dim,
            &m32,
            KvPrecision::F32,
        )
        .unwrap();
        match &mut kv32.store {
            KvStore::F32 { k, v } => {
                device.htod_copy_into(&k32, k).unwrap();
                device.htod_copy_into(&v32, v).unwrap();
            }
            KvStore::F16 { .. } => unreachable!(),
        }
        let m16 = compile_kv_module(&device, KvPrecision::F16).unwrap();
        let mut kv16 = KvCacheGpu::with_module_at(
            &device,
            num_kv_heads,
            max_seq_len,
            head_dim,
            &m16,
            KvPrecision::F16,
        )
        .unwrap();
        match &mut kv16.store {
            KvStore::F16 { k, v } => {
                device.htod_copy_into(&k16, k).unwrap();
                device.htod_copy_into(&v16, v).unwrap();
            }
            KvStore::F32 { .. } => unreachable!(),
        }
        assert_eq!(kv16.bytes() * 2, kv32.bytes());

        let q_gpu = device.htod_copy(&q).unwrap();
        let s_max =
            super::super::prefill::attn_splitk_gqa6_scratch_chunks(max_seq_len as u32, true).max(
                super::super::prefill::attn_splitk_gqa6_scratch_chunks(max_seq_len as u32, false),
            ) as usize;
        let mut scratch = (
            device.alloc_zeros::<f32>(num_heads * s_max).unwrap(),
            device.alloc_zeros::<f32>(num_heads * s_max).unwrap(),
            device
                .alloc_zeros::<f32>(num_heads * s_max * head_dim)
                .unwrap(),
        );
        let mut out32 = device.alloc_zeros::<f32>(num_heads * head_dim).unwrap();
        let mut out16 = device.alloc_zeros::<f32>(num_heads * head_dim).unwrap();
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        // The two stores take the same split geometry at every context (one
        // policy), so the identity holds on both partitions: one-tile to 2,816
        // keys, whole-tile from 2,817.
        for seq_len in [
            1u32, 16, 129, 330, 1300, 2600, 2816, 2817, 3072, 3968, 4096, 4097, 6144, 16_384,
            16_385,
        ] {
            let a = unsafe {
                super::super::prefill::launch_attention_decode_gated(
                    &device,
                    &kernels,
                    &q_gpu,
                    kv32.as_ref(),
                    Some(&mut scratch),
                    &mut out32,
                    num_heads as u32,
                    num_kv_heads as u32,
                    head_dim as u32,
                    seq_len,
                    max_seq_len as u32,
                    scale,
                )
            }
            .unwrap();
            let b = unsafe {
                super::super::prefill::launch_attention_decode_gated(
                    &device,
                    &kernels,
                    &q_gpu,
                    kv16.as_ref(),
                    Some(&mut scratch),
                    &mut out16,
                    num_heads as u32,
                    num_kv_heads as u32,
                    head_dim as u32,
                    seq_len,
                    max_seq_len as u32,
                    scale,
                )
            }
            .unwrap();
            device.synchronize().unwrap();
            let got32: Vec<u32> = device
                .dtoh_copy(&out32)
                .unwrap()
                .iter()
                .map(|x| x.to_bits())
                .collect();
            let got16: Vec<u32> = device
                .dtoh_copy(&out16)
                .unwrap()
                .iter()
                .map(|x| x.to_bits())
                .collect();
            // A one-chunk context (128 keys or fewer) is the tiled kernel's; every
            // longer one must take the GQA-shared pair on both stores, on the
            // partition the shared policy picks — the arm this test guards.
            let want = if super::super::prefill::attn_splitk_chunks(seq_len) <= 1 {
                (V::Tiled, V::TiledF16)
            } else {
                (V::SplitKGqa6, V::SplitKGqa6F16)
            };
            assert_eq!(
                (a, b),
                want,
                "at seq_len {seq_len} the stores took ({a:?}, {b:?})"
            );
            assert_eq!(
                got32, got16,
                "outputs differ at seq_len {seq_len} ({a:?} / {b:?})"
            );
        }
    }
}
