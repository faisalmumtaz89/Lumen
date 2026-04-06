//! GPU-resident KV cache for the CUDA backend.
//!
//! Mirrors the CPU `KvCacheView` layout: head-first `[num_kv_heads][max_seq_len][head_dim]`
//! for both K and V caches. Data lives entirely on the GPU; the `kv_cache_write`
//! CUDA kernel handles scatter-writing new tokens at the correct position.
//!
//! This module manages per-layer GPU KV buffers. The engine maintains one
//! `KvCacheGpu` per layer, allocated at session start and reused across tokens.

use crate::error::RuntimeError;
use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, LaunchConfig as CudarcLaunchConfig, PushKernelArg};
use std::sync::Arc;

use super::ffi::CudaDevice;
use super::shaders::KV_CACHE_KERNEL_SOURCE;
use super::types::LaunchConfig;

/// GPU-resident KV cache for a single transformer layer.
///
/// Allocates K and V buffers of shape `[num_kv_heads, max_seq_len, head_dim]`
/// as contiguous f32 arrays on the CUDA device. Tracks the current sequence
/// position (`seq_len`) so new tokens are written at the correct offset.
pub struct KvCacheGpu {
    /// Key cache on GPU. Shape: `[num_kv_heads, max_seq_len, head_dim]`.
    pub k_cache: CudaSlice<f32>,
    /// Value cache on GPU. Shape: `[num_kv_heads, max_seq_len, head_dim]`.
    pub v_cache: CudaSlice<f32>,
    /// Current number of tokens with cached KV data.
    seq_len: usize,
    /// Maximum sequence length (allocated capacity).
    pub max_seq_len: usize,
    /// Number of KV attention heads.
    pub num_kv_heads: usize,
    /// Dimension per attention head.
    pub head_dim: usize,
    /// Compiled kv_cache_write kernel function.
    write_func: CudaFunction,
}

impl KvCacheGpu {
    /// Allocate a new GPU KV cache for one layer.
    ///
    /// Both K and V buffers are zeroed. The `kv_cache_write` kernel is compiled
    /// and cached for the lifetime of this struct.
    #[allow(dead_code)] // Used in #[cfg(test)] blocks in prefill_attention.rs.
    pub fn new(
        device: &CudaDevice,
        num_kv_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
    ) -> Result<Self, RuntimeError> {
        let total_elements = num_kv_heads * max_seq_len * head_dim;
        let k_cache = device.alloc_zeros::<f32>(total_elements)?;
        let v_cache = device.alloc_zeros::<f32>(total_elements)?;

        let module = device.compile_and_load(KV_CACHE_KERNEL_SOURCE)?;
        let write_func = module.load_function("kv_cache_write").map_err(|e| {
            RuntimeError::Compute(format!("Failed to load kv_cache_write: {e}"))
        })?;

        Ok(Self {
            k_cache,
            v_cache,
            seq_len: 0,
            max_seq_len,
            num_kv_heads,
            head_dim,
            write_func,
        })
    }

    /// Allocate a new GPU KV cache using a pre-compiled kernel module.
    ///
    /// Avoids redundant NVRTC compilation when creating multiple layers' caches
    /// from the same kernel source. The caller compiles the module once and
    /// passes it to each layer.
    pub fn with_module(
        device: &CudaDevice,
        num_kv_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
        module: &Arc<CudaModule>,
    ) -> Result<Self, RuntimeError> {
        let total_elements = num_kv_heads * max_seq_len * head_dim;
        let k_cache = device.alloc_zeros::<f32>(total_elements)?;
        let v_cache = device.alloc_zeros::<f32>(total_elements)?;

        let write_func = module.load_function("kv_cache_write").map_err(|e| {
            RuntimeError::Compute(format!("Failed to load kv_cache_write: {e}"))
        })?;

        Ok(Self {
            k_cache,
            v_cache,
            seq_len: 0,
            max_seq_len,
            num_kv_heads,
            head_dim,
            write_func,
        })
    }

    /// Current sequence length (number of tokens with cached KV data).
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Append one token's K and V data to the cache at the current position.
    ///
    /// `k_data` and `v_data` are GPU buffers of shape `[num_kv_heads * head_dim]`.
    /// The kernel scatter-writes each head's data to the correct position in the
    /// head-first `[head][pos][dim]` layout.
    ///
    /// Advances `seq_len` by 1 after writing.
    ///
    /// Returns an error if the cache is full (`seq_len >= max_seq_len`) or if
    /// the kernel launch fails.
    pub fn append_kv(
        &mut self,
        device: &CudaDevice,
        k_data: &CudaSlice<f32>,
        v_data: &CudaSlice<f32>,
    ) -> Result<(), RuntimeError> {
        if self.seq_len >= self.max_seq_len {
            return Err(RuntimeError::KvCache(format!(
                "KV cache full: seq_len={} >= max_seq_len={}",
                self.seq_len, self.max_seq_len,
            )));
        }

        let pos = self.seq_len as u32;
        let num_kv_heads = self.num_kv_heads as u32;
        let max_seq_len = self.max_seq_len as u32;
        let head_dim = self.head_dim as u32;
        let total_elements = self.num_kv_heads * self.head_dim;

        let config = LaunchConfig::for_elements(total_elements);
        let launch_cfg = CudarcLaunchConfig {
            grid_dim: (config.grid_dim, 1, 1),
            block_dim: (config.block_dim, 1, 1),
            shared_mem_bytes: 0,
        };

        // Write K data to k_cache.
        // SAFETY: k_data has num_kv_heads * head_dim elements (verified by caller).
        // k_cache has num_kv_heads * max_seq_len * head_dim elements (allocated
        // in new()). pos < max_seq_len (checked above). The kernel writes exactly
        // num_kv_heads * head_dim elements, each at a valid offset within k_cache.
        unsafe {
            device
                .stream
                .launch_builder(&self.write_func)
                .arg(&mut self.k_cache)
                .arg(k_data)
                .arg(&pos)
                .arg(&num_kv_heads)
                .arg(&max_seq_len)
                .arg(&head_dim)
                .launch(launch_cfg)
        }
        .map_err(|e| RuntimeError::Compute(format!("kv_cache_write K launch: {e}")))?;

        // Write V data to v_cache.
        // SAFETY: Same reasoning as K above, applied to v_cache and v_data.
        unsafe {
            device
                .stream
                .launch_builder(&self.write_func)
                .arg(&mut self.v_cache)
                .arg(v_data)
                .arg(&pos)
                .arg(&num_kv_heads)
                .arg(&max_seq_len)
                .arg(&head_dim)
                .launch(launch_cfg)
        }
        .map_err(|e| RuntimeError::Compute(format!("kv_cache_write V launch: {e}")))?;

        self.seq_len += 1;
        Ok(())
    }

    /// Advance the cache sequence length by `count` tokens at once.
    ///
    /// Used by the batched prefill path where all tokens' KV data is written
    /// in a single batch kernel launch. The kernel writes to positions
    /// `seq_len..seq_len+count-1` and then this method advances the counter.
    pub fn advance_seq_len_by(&mut self, count: usize) {
        self.seq_len += count;
    }

    /// Reset the cache to empty state (seq_len = 0).
    ///
    /// Does not deallocate GPU memory; the buffers are reused for the next
    /// inference session. Previously written data becomes stale but is
    /// overwritten naturally as new tokens are appended.
    pub fn reset(&mut self) {
        self.seq_len = 0;
    }
}
