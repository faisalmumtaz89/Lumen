//! A test-only hook of the native prefill (feature `test-prefill-dump`): named tensors copied to the
//! host at layer boundaries, for the layer oracle (`tests/native_oracle/layer.rs`).
//!
//! A tensor is recorded by name for a layer, as the bytes of its first rows (the prompt's tokens; the
//! padding rows GEMMs run past them are not part of it). The names a layer records, for `t` tokens:
//!
//! | Name | Content |
//! |---|---|
//! | `x` | BF16 `[t][5120]`: the layer's input (the embedded rows at layer 0, the previous MLP output after) |
//! | `resid_in` | BF16 `[t][5120]`: the residual entering the layer (not at layer 0) |
//! | `ring_in`, `state_in`, `state_pos_in` | A GDN layer's conv ring `[3][10240]` F32, state `[48][128][128]` F32 and ring position (`u32`) before it |
//! | `resid_attn`, `normed`, `x8` | The input norm: residual BF16, normed rows BF16, FP8 codes `[t][5120]` |
//! | `qkv`, `z`, `ab` | GDN input projections: BF16 `[t][10240]`, BF16 `[t][6144]`, F32 `[t][96]` |
//! | `cv`, `ring`, `core`, `state` | GDN conv output BF16 `[t][10240]`, ring after, output BF16 `[t][6144]`, state after |
//! | `y8` | FP8 codes `[t][6144]` of the GDN gated norm |
//! | `qg`, `k`, `v` | Attention input projections: BF16 `[t][12288]`, `[t][1024]`, `[t][1024]` |
//! | `p0`, `cs`, `kv_in` | An attention layer's first position (`u32`), its RoPE table rows `[p0, p0 + t)` F32 `[t][64]`, and its KV cache `[2][4][max_seq][256]` (K, then V) before it, in the store's type (F32, or BF16 on a BF16 store) |
//! | `q`, `gate`, `kv` | The prep: normed and rotated queries and the gate, BF16 `[t][24][256]`, and the KV cache after it |
//! | `o`, `o8` | The attention output BF16 `[t][6144]` and the gated output's FP8 codes |
//! | `attn_out` | The output projection, BF16 `[t][5120]` |
//! | `resid_mlp`, `x4`, `x4sf` | The post-attention norm: residual BF16, NVFP4 codes `[t][2560]`, swizzled block scales |
//! | `gu`, `d4`, `d4sf`, `mlp_out` | Gate and up BF16 `[t][34816]`, SwiGLU NVFP4 codes and scales, down BF16 `[t][5120]` |
//! | `x_gpu` | F32 `[5120]`: the last token's `mlp_out + resid_mlp`, the row handed to decode |
//! | `state_pos` | A GDN layer's ring position after it (`u32`) |

use super::ffi::CudaDevice;
use crate::error::RuntimeError;
use std::collections::BTreeSet;

/// One recorded tensor.
pub struct DumpedTensor {
    pub layer: usize,
    pub name: &'static str,
    pub bytes: Vec<u8>,
}

/// The tensors recorded for the chosen layers, in recording order.
pub struct PrefillDump {
    layers: BTreeSet<usize>,
    tensors: Vec<DumpedTensor>,
}

impl PrefillDump {
    /// A dump that records the given layers.
    pub fn new(layers: impl IntoIterator<Item = usize>) -> Self {
        Self {
            layers: layers.into_iter().collect(),
            tensors: Vec::new(),
        }
    }

    /// Whether layer `layer` is recorded.
    pub fn wants(&self, layer: usize) -> bool {
        self.layers.contains(&layer)
    }

    /// Copy `bytes` bytes at device address `ptr` as `name` of `layer`, after the work queued on the
    /// device stream; a layer not chosen is skipped. A name recorded twice for a layer keeps the
    /// later copy.
    ///
    /// # Safety
    /// `ptr` addresses `bytes` bytes of live device memory.
    pub unsafe fn record(
        &mut self,
        device: &CudaDevice,
        layer: usize,
        name: &'static str,
        ptr: u64,
        bytes: usize,
    ) -> Result<(), RuntimeError> {
        if !self.wants(layer) {
            return Ok(());
        }
        let err = |e: cudarc::driver::DriverError| {
            RuntimeError::Compute(format!("dump layer {layer} {name}: {e}"))
        };
        device.ctx.bind_to_thread().map_err(err)?;
        device.stream.synchronize().map_err(err)?;
        let mut host = vec![0u8; bytes];
        cudarc::driver::result::memcpy_dtoh_sync(&mut host, ptr).map_err(err)?;
        self.tensors
            .retain(|t| !(t.layer == layer && t.name == name));
        self.tensors.push(DumpedTensor {
            layer,
            name,
            bytes: host,
        });
        Ok(())
    }

    /// Record host bytes as `name` of `layer` (host-side values such as a ring position).
    pub fn record_host(&mut self, layer: usize, name: &'static str, bytes: Vec<u8>) {
        if self.wants(layer) {
            self.tensors
                .retain(|t| !(t.layer == layer && t.name == name));
            self.tensors.push(DumpedTensor { layer, name, bytes });
        }
    }

    /// The bytes of `name` for `layer`, if recorded.
    pub fn get(&self, layer: usize, name: &str) -> Option<&[u8]> {
        self.tensors
            .iter()
            .find(|t| t.layer == layer && t.name == name)
            .map(|t| t.bytes.as_slice())
    }

    /// Every recorded tensor.
    pub fn tensors(&self) -> &[DumpedTensor] {
        &self.tensors
    }
}
