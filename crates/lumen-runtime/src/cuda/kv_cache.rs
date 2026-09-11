//! GPU-resident KV cache for CUDA.
//!
//! One cache per full-attention layer, `[num_kv_heads, max_seq_len, head_dim]`
//! head-first, stored either as F32 or as IEEE half (`u16` bit patterns). The
//! storage type is a variant of [`KvStore`], not a flag beside an untyped
//! buffer: every reader and writer takes the variant it can consume, so no
//! program can hand half bits to a kernel that reads floats, or the reverse.
//!
//! Writers round to half on the device (round to nearest even) and count every
//! value that does not fit; the owner of the counter refuses to continue with
//! a poisoned cache (see `backend_impl.rs`).

use std::sync::Arc;

use cudarc::driver::{
    CudaFunction, CudaModule, CudaSlice, LaunchConfig as CudarcLaunchConfig, PushKernelArg,
};

use super::ffi::CudaDevice;
use super::shaders::{KV_CACHE_F16_KERNEL_SOURCE, KV_CACHE_KERNEL_SOURCE};
use super::types::LaunchConfig;
use crate::error::RuntimeError;
use crate::kv::KvPrecision;

/// The K and V buffers of one layer, typed by what they hold.
pub enum KvStore {
    /// F32, `[num_kv_heads, max_seq_len, head_dim]` each.
    F32 {
        k: CudaSlice<f32>,
        v: CudaSlice<f32>,
    },
    /// IEEE half bit patterns, same layout.
    F16 {
        k: CudaSlice<u16>,
        v: CudaSlice<u16>,
    },
}

impl KvStore {
    /// Bytes one element occupies.
    pub fn bytes_per_element(&self) -> usize {
        match self {
            KvStore::F32 { .. } => 4,
            KvStore::F16 { .. } => 2,
        }
    }

    /// Elements in the K buffer (the V buffer is the same size).
    pub fn elements(&self) -> usize {
        match self {
            KvStore::F32 { k, .. } => k.len(),
            KvStore::F16 { k, .. } => k.len(),
        }
    }
}

/// GPU-resident KV cache for one transformer layer.
pub struct KvCacheGpu {
    /// The K and V buffers, typed by storage.
    pub store: KvStore,
    /// Current number of tokens with cached KV data.
    seq_len: usize,
    /// Maximum sequence length (allocated capacity).
    pub max_seq_len: usize,
    /// Number of KV attention heads.
    pub num_kv_heads: usize,
    /// Dimension per attention head.
    pub head_dim: usize,
    /// The single-token write kernel for this store's type: `kv_cache_write`
    /// for F32, `kv_cache_write_f16` for half.
    write_func: CudaFunction,
}

/// The kernel module a cache's writer comes from, compiled once per precision.
pub fn compile_kv_module(
    device: &CudaDevice,
    precision: KvPrecision,
) -> Result<Arc<CudaModule>, RuntimeError> {
    let source = match precision {
        KvPrecision::F32 => KV_CACHE_KERNEL_SOURCE,
        KvPrecision::F16 => KV_CACHE_F16_KERNEL_SOURCE,
        other => {
            return Err(RuntimeError::Unsupported(format!(
                "CUDA KV cache precision {other:?} is not implemented"
            )))
        }
    };
    device.compile_and_load(source)
}

impl KvCacheGpu {
    /// Allocate an F32 cache for one layer, compiling its write kernel.
    /// Both buffers are zeroed.
    #[allow(dead_code)] // Used in #[cfg(test)] blocks in prefill_attention.rs.
    pub fn new(
        device: &CudaDevice,
        num_kv_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
    ) -> Result<Self, RuntimeError> {
        let module = compile_kv_module(device, KvPrecision::F32)?;
        Self::with_module(device, num_kv_heads, max_seq_len, head_dim, &module)
    }

    /// Allocate an F32 cache using a module compiled by [`compile_kv_module`]
    /// for F32.
    pub fn with_module(
        device: &CudaDevice,
        num_kv_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
        module: &Arc<CudaModule>,
    ) -> Result<Self, RuntimeError> {
        Self::with_module_at(
            device,
            num_kv_heads,
            max_seq_len,
            head_dim,
            module,
            KvPrecision::F32,
        )
    }

    /// Allocate a cache of the given storage using a module compiled by
    /// [`compile_kv_module`] for that same precision. Both buffers are zeroed.
    pub fn with_module_at(
        device: &CudaDevice,
        num_kv_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
        module: &Arc<CudaModule>,
        precision: KvPrecision,
    ) -> Result<Self, RuntimeError> {
        let total_elements = num_kv_heads * max_seq_len * head_dim;
        let (store, write_name) = match precision {
            KvPrecision::F32 => (
                KvStore::F32 {
                    k: device.alloc_zeros::<f32>(total_elements)?,
                    v: device.alloc_zeros::<f32>(total_elements)?,
                },
                "kv_cache_write",
            ),
            KvPrecision::F16 => (
                KvStore::F16 {
                    k: device.alloc_zeros::<u16>(total_elements)?,
                    v: device.alloc_zeros::<u16>(total_elements)?,
                },
                "kv_cache_write_f16",
            ),
            other => {
                return Err(RuntimeError::Unsupported(format!(
                    "CUDA KV cache precision {other:?} is not implemented"
                )))
            }
        };
        let write_func = module
            .load_function(write_name)
            .map_err(|e| RuntimeError::Compute(format!("Failed to load {write_name}: {e}")))?;

        Ok(Self {
            store,
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

    pub fn precision(&self) -> KvPrecision {
        match self.store {
            KvStore::F32 { .. } => KvPrecision::F32,
            KvStore::F16 { .. } => KvPrecision::F16,
        }
    }

    /// Bytes the two buffers occupy.
    pub fn bytes(&self) -> u64 {
        2 * self.store.elements() as u64 * self.store.bytes_per_element() as u64
    }

    /// Append one token's K and V data to the cache at the current position.
    ///
    /// `k_data` and `v_data` are GPU buffers of shape `[num_kv_heads * head_dim]`
    /// (F32 activations whatever the store is). For a half store the kernel
    /// rounds on the way in and counts every value that does not fit in
    /// `overflow`, which the caller must supply for that store.
    ///
    /// Advances `seq_len` by 1 after writing.
    pub fn append_kv(
        &mut self,
        device: &CudaDevice,
        k_data: &CudaSlice<f32>,
        v_data: &CudaSlice<f32>,
        overflow: Option<&mut CudaSlice<u32>>,
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

        match &mut self.store {
            KvStore::F32 { k, v } => {
                for (cache, data, which) in [(k, k_data, "K"), (v, v_data, "V")] {
                    unsafe {
                        device
                            .stream
                            .launch_builder(&self.write_func)
                            .arg(cache)
                            .arg(data)
                            .arg(&pos)
                            .arg(&num_kv_heads)
                            .arg(&max_seq_len)
                            .arg(&head_dim)
                            .launch(launch_cfg)
                    }
                    .map_err(|e| {
                        RuntimeError::Compute(format!("kv_cache_write {which} launch: {e}"))
                    })?;
                }
            }
            KvStore::F16 { k, v } => {
                let overflow = overflow.ok_or_else(|| {
                    RuntimeError::Compute(
                        "F16 KV cache write without an overflow counter".to_string(),
                    )
                })?;
                for (cache, data, which) in [(k, k_data, "K"), (v, v_data, "V")] {
                    unsafe {
                        device
                            .stream
                            .launch_builder(&self.write_func)
                            .arg(cache)
                            .arg(data)
                            .arg(&mut *overflow)
                            .arg(&pos)
                            .arg(&num_kv_heads)
                            .arg(&max_seq_len)
                            .arg(&head_dim)
                            .launch(launch_cfg)
                    }
                    .map_err(|e| {
                        RuntimeError::Compute(format!("kv_cache_write_f16 {which} launch: {e}"))
                    })?;
                }
            }
        }

        self.seq_len += 1;
        Ok(())
    }

    pub fn advance_seq_len_by(&mut self, count: usize) {
        self.seq_len += count;
    }

    pub fn reset(&mut self) {
        self.seq_len = 0;
    }
}

/// F32 K/V buffers a reader takes: the cache's own F32 store, or a widened
/// copy of a half store; `[num_kv_heads, seq_stride, head_dim]` either way.
pub struct KvView<'a> {
    pub k: &'a CudaSlice<f32>,
    pub v: &'a CudaSlice<f32>,
    /// The position stride of the layout (the cache's `max_seq_len`, or the
    /// widened copy's position count).
    pub seq_stride: usize,
}

/// A store borrowed by type, for a dispatch that must pick the reader the
/// bytes are for.
pub enum KvRef<'a> {
    F32 {
        k: &'a CudaSlice<f32>,
        v: &'a CudaSlice<f32>,
    },
    F16 {
        k: &'a CudaSlice<u16>,
        v: &'a CudaSlice<u16>,
    },
}

impl KvCacheGpu {
    pub fn as_ref(&self) -> KvRef<'_> {
        match &self.store {
            KvStore::F32 { k, v } => KvRef::F32 { k, v },
            KvStore::F16 { k, v } => KvRef::F16 { k, v },
        }
    }

    /// The store as an F32 view, when it is F32.
    pub fn f32_view(&self) -> Option<KvView<'_>> {
        match &self.store {
            KvStore::F32 { k, v } => Some(KvView {
                k,
                v,
                seq_stride: self.max_seq_len,
            }),
            KvStore::F16 { .. } => None,
        }
    }
}
