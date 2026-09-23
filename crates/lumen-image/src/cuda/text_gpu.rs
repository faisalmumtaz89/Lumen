//! The Qwen3-VL text tower, on the GPU, in the reference's arithmetic.
//!
//! The reference runs this tower in bf16: every tensor between two operations
//! is bf16, and each operation computes in f32 and rounds its result to bf16
//! once, nearest even. This forward does the same, operation for operation:
//!
//! - RMSNorm: `bf16(x * rsqrt(mean(x^2) + eps))`, then `bf16(weight * that)`
//! - projections: cuBLAS bf16 GEMMs with f32 accumulation, rounded to bf16
//!   once (the reference's cuBLAS may also round split-K partial sums to
//!   bf16 for some shapes; this path does not)
//! - per-head `q_norm`/`k_norm` after the head split, before the rotary
//! - rotary: `bf16(bf16(x * cos) + bf16(rotate_half(x) * sin))`, with the
//!   `cos`/`sin` tables themselves rounded to bf16
//! - causal attention over grouped heads: the key/value heads repeated to the
//!   query heads, then the flash kernel with the whole prompt as its causal
//!   text prefix (f32 softmax, bf16 in and out)
//! - residual adds and `down(bf16(silu(gate)) * up)`, each result rounded
//!
//! It returns the last layer's hidden state **without** the final RMS norm, the
//! state the pipeline consumes. [`crate::text_encoder`] is the f32 CPU form of
//! the same tower, kept as the reference checks' independent path; the two
//! differ by bf16 rounding, which is the reference's.
//!
//! # Weights
//!
//! Every matrix and the embedding table must be stored bf16, the checkpoint's
//! dtype and the only one the GEMMs take. The embedding table (151936 x 4096,
//! ~1.16 GiB) stays resident and the prompt's rows are gathered from it.
//! `model.language_model.norm.weight` is not loaded: the forward stops before
//! the final norm.
//!
//! # Memory
//!
//! 14.1 GiB of bf16 weights (the container also carries the vision tower, which
//! is not loaded) plus bf16 activations; the largest transient is a projection's
//! f32 output before rounding, `[seq, 12288]` f32 (48 KiB per token).

use std::collections::HashMap;
use std::path::Path;

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, LaunchConfig, PinnedHostSlice, PushKernelArg,
};
use lumen_format::QuantScheme;
use lumen_runtime::cuda::ffi::CudaDevice;
use lumen_runtime::error::RuntimeError;

use super::attention::{fused_block_causal_attention, FLASH_HEAD_DIM};
use super::blas::{convert_f32_to_bf16, gemm_bf16, Bf16Activation};
use super::launch::{self, DevVec};
use super::{ImageKernels, TEXT_OPS_SOURCE};
use crate::lbi::{LbiError, LbiFile};
use crate::tensor::{bf16_bits, Matrix};
use crate::text_encoder::{rope_tables, TextEncoderConfig, TextEncoderError};

/// Every text-tower tensor sits under this prefix.
///
/// A second spelling of `text_encoder.rs`'s private `PREFIX`: a drift between
/// the two shows up as a `MissingTensor` at [`TextGpu::load`], before a single
/// byte reaches the device.
const PREFIX: &str = "model.language_model.";

/// Threads per block for the elementwise and gather kernels, and the most the
/// rotary kernel takes.
const THREADS: u32 = 256;

/// `TEXT_NORM_THREADS` in `text_ops.cu`, which sizes its shared reduction.
const NORM_THREADS: u32 = 256;

/// What went wrong loading or running the text tower on the device.
#[derive(Debug)]
pub enum TextGpuError {
    /// Allocating, copying or launching on the device.
    Cuda(RuntimeError),
    /// A tensor is missing, the wrong shape, or the config is unusable.
    /// Reusing the reference's error keeps a shape failure reported the same
    /// way whichever path found it.
    Text(TextEncoderError),
    /// A weight is stored in a scheme this forward does not take.
    UnsupportedStorage { tensor: String, scheme: String },
}

impl std::fmt::Display for TextGpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cuda(e) => write!(f, "cuda: {e}"),
            Self::Text(e) => write!(f, "{e}"),
            Self::UnsupportedStorage { tensor, scheme } => {
                write!(
                    f,
                    "tensor {tensor} is stored as {scheme}; the GPU text tower takes bf16"
                )
            }
        }
    }
}

impl std::error::Error for TextGpuError {}

impl From<RuntimeError> for TextGpuError {
    fn from(e: RuntimeError) -> Self {
        Self::Cuda(e)
    }
}

impl From<TextEncoderError> for TextGpuError {
    fn from(e: TextEncoderError) -> Self {
        Self::Text(e)
    }
}

impl From<LbiError> for TextGpuError {
    fn from(e: LbiError) -> Self {
        Self::Text(TextEncoderError::Lbi(e))
    }
}

/// A `[n, k]` linear weight resident as bf16: `n` outputs over `k` inputs.
struct Weight {
    bits: CudaSlice<u16>,
    n: usize,
    k: usize,
}

/// One decoder layer, resident on the device. The norm weights are widened to
/// f32, which is exact for bf16.
struct GpuLayer {
    input_norm: DevVec,
    q_proj: Weight,
    k_proj: Weight,
    v_proj: Weight,
    o_proj: Weight,
    q_norm: DevVec,
    k_norm: DevVec,
    post_norm: DevVec,
    gate_proj: Weight,
    up_proj: Weight,
    down_proj: Weight,
}

/// The entry points `text_ops.cu` provides.
struct TextOps {
    embed_gather: CudaFunction,
    rms_norm: CudaFunction,
    rope: CudaFunction,
    repeat_kv: CudaFunction,
    silu_mul: CudaFunction,
    add: CudaFunction,
}

impl TextOps {
    fn load(dev: &CudaDevice) -> Result<Self, RuntimeError> {
        let module = dev.compile_and_load(TEXT_OPS_SOURCE)?;
        let get = |name: &str| -> Result<CudaFunction, RuntimeError> {
            module
                .load_function(name)
                .map_err(|e| RuntimeError::Compute(format!("load {name}: {e}")))
        };
        Ok(Self {
            embed_gather: get("embed_gather_bf16")?,
            rms_norm: get("rms_norm_bf16")?,
            rope: get("rope_bf16")?,
            repeat_kv: get("repeat_kv")?,
            silu_mul: get("silu_mul_bf16")?,
            add: get("add_bf16")?,
        })
    }
}

/// The text tower, with every weight resident on the device.
pub struct TextGpu {
    /// The tower drives one stream from host inputs end to end, so it holds its
    /// own device handle rather than borrowing the caller's. `CudaDevice::new`
    /// retains the same primary context, so this is a second stream on the same
    /// device, not a second device.
    dev: CudaDevice,
    /// `f32_to_bf16_bits` for the projections' outputs and the flash kernel.
    kernels: ImageKernels,
    text_ops: TextOps,
    config: TextEncoderConfig,
    embed: Weight,
    layers: Vec<GpuLayer>,
}

impl TextGpu {
    /// Load every text-tower weight from a converted `.lbi` onto the device,
    /// assuming the shipped Qwen-Image-2.1 architecture.
    pub fn load(lbi: &Path, dev: &CudaDevice) -> Result<Self, TextGpuError> {
        let file = LbiFile::open(lbi)?;
        let config = TextEncoderConfig::from_lbi_config(file.config())?;
        Self::load_with(&file, None, dev, config)
    }

    /// Load against an explicit architecture.
    ///
    /// The config's own `validate` has already run — `from_lbi_config` calls it,
    /// and a caller supplying one by hand is expected to have called it — so the
    /// dimensions here are bounded and the products below cannot overflow.
    ///
    /// A matrix with a copy in `pinned` is uploaded from that copy instead of
    /// the file's mapping.
    pub fn load_with(
        file: &LbiFile,
        pinned: Option<&PinnedMatrices>,
        dev: &CudaDevice,
        config: TextEncoderConfig,
    ) -> Result<Self, TextGpuError> {
        Self::check(file, &config)?;

        let own = CudaDevice::new(dev.ctx.ordinal())?;
        let kernels = ImageKernels::load(&own)?;
        let text_ops = TextOps::load(&own)?;

        let c = &config;
        let hidden = c.hidden_size;
        let q_width = c.num_attention_heads * c.head_dim;
        let kv_width = c.num_key_value_heads * c.head_dim;
        let embed = matrix(&own, file, pinned, "embed_tokens", c.vocab_size, hidden)?;

        let mut layers = Vec::with_capacity(c.num_layers);
        for i in 0..c.num_layers {
            let p = format!("layers.{i}");
            let w = |stem: &str, n, k| matrix(&own, file, pinned, &format!("{p}.{stem}"), n, k);
            let v = |stem: &str, len| vector(&own, file, &format!("{p}.{stem}"), len);
            layers.push(GpuLayer {
                input_norm: v("input_layernorm", hidden)?,
                q_proj: w("self_attn.q_proj", q_width, hidden)?,
                k_proj: w("self_attn.k_proj", kv_width, hidden)?,
                v_proj: w("self_attn.v_proj", kv_width, hidden)?,
                o_proj: w("self_attn.o_proj", hidden, q_width)?,
                q_norm: v("self_attn.q_norm", c.head_dim)?,
                k_norm: v("self_attn.k_norm", c.head_dim)?,
                post_norm: v("post_attention_layernorm", hidden)?,
                gate_proj: w("mlp.gate_proj", c.intermediate_size, hidden)?,
                up_proj: w("mlp.up_proj", c.intermediate_size, hidden)?,
                down_proj: w("mlp.down_proj", hidden, c.intermediate_size)?,
            });
        }

        Ok(Self {
            dev: own,
            kernels,
            text_ops,
            config,
            embed,
            layers,
        })
    }

    /// Refuse, from the container's index alone, what [`Self::load_with`]
    /// cannot run: a head width other than the one the attention kernel is
    /// tiled for, or any matrix missing, misshapen or not stored as bf16. No
    /// weight bytes are read, so a server can run this before it commits to
    /// the GPU path.
    pub fn check(file: &LbiFile, config: &TextEncoderConfig) -> Result<(), TextGpuError> {
        if config.head_dim != FLASH_HEAD_DIM {
            return Err(TextEncoderError::BadConfig(format!(
                "head_dim {} is not the {FLASH_HEAD_DIM} the attention kernel is tiled for",
                config.head_dim
            ))
            .into());
        }
        for (stem, n, k) in matrices(config) {
            bf16_entry(file, &stem, n, k)?;
        }
        Ok(())
    }

    pub fn config(&self) -> &TextEncoderConfig {
        &self.config
    }

    /// Encode a prompt to `[seq, hidden]`, **without** the final RMS norm.
    ///
    /// All three M-RoPE rows are the plain `0..seq-1`, because a text-to-image
    /// prompt has no grid and the model falls back to `arange` broadcast across
    /// T, H and W. The caller gets every row, template included, and drops the
    /// leading system-message rows itself.
    pub fn forward(&self, token_ids: &[u32]) -> Result<Matrix, TextGpuError> {
        if token_ids.is_empty() {
            return Err(TextEncoderError::EmptyInput.into());
        }
        let c = &self.config;
        let seq = token_ids.len();
        let hidden = c.hidden_size;

        // Range-checked here so the gather kernel can trust them: it reads at
        // `id * hidden`.
        for &id in token_ids {
            if id as usize >= self.embed.n {
                return Err(TextEncoderError::TokenOutOfRange {
                    id,
                    vocab: self.embed.n,
                }
                .into());
            }
        }

        // The rotary tables come from the reference's own schedule in f32 and
        // are rounded to bf16, as the reference casts them to the activations'
        // dtype. A text prompt puts the same `0..seq-1` on all three rows.
        let positions: Vec<f32> = (0..seq).map(|i| i as f32).collect();
        let (cos, sin) = rope_tables(c, [&positions, &positions, &positions])?;
        let to_bits = |m: &Matrix| m.data.iter().map(|&v| bf16_bits(v)).collect::<Vec<u16>>();
        let g_cos = self.dev.htod_copy(&to_bits(&cos))?;
        let g_sin = self.dev.htod_copy(&to_bits(&sin))?;
        let g_ids = self.dev.htod_copy(token_ids)?;

        let mut x = self.gather_embed(&g_ids, seq)?;
        for layer in &self.layers {
            x = self.decoder_layer(layer, &x, seq, &g_cos, &g_sin)?;
        }
        let bits = self.dev.dtoh_copy(&x.bits)?;
        self.dev.synchronize()?;
        let data = bits
            .iter()
            .map(|&b| f32::from_bits((b as u32) << 16))
            .collect();
        Ok(Matrix::new(seq, hidden, data))
    }

    // -- the pieces of the forward ------------------------------------------

    fn gather_embed(
        &self,
        ids: &CudaSlice<u32>,
        seq: usize,
    ) -> Result<Bf16Activation, TextGpuError> {
        let hidden = self.embed.k;
        let out = self.alloc_bits(seq * hidden)?;
        let (su, hu) = (seq as u32, hidden as u32);
        // Safety: `out` is `seq * hidden`; `forward` checked every id below
        // the vocabulary, so each gathered row lies inside the `[vocab, hidden]` table.
        unsafe {
            self.dev
                .stream
                .launch_builder(&self.text_ops.embed_gather)
                .arg(&self.embed.bits)
                .arg(ids)
                .arg(&out)
                .arg(&su)
                .arg(&hu)
                .launch(LaunchConfig {
                    grid_dim: (su, 1, 1),
                    block_dim: (THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| RuntimeError::Compute(format!("embed_gather_bf16: {e}")))?;
        }
        Ok(Bf16Activation::from_bits(out, seq, hidden)?)
    }

    /// One decoder layer: pre-norm attention and pre-norm MLP, each residual.
    fn decoder_layer(
        &self,
        layer: &GpuLayer,
        x: &Bf16Activation,
        seq: usize,
        cos: &CudaSlice<u16>,
        sin: &CudaSlice<u16>,
    ) -> Result<Bf16Activation, TextGpuError> {
        let hidden = self.config.hidden_size;
        let normed = self.rms_norm(&x.bits, &layer.input_norm, seq, hidden)?;
        let normed = Bf16Activation::from_bits(normed, seq, hidden)?;
        let attended = self.attention(layer, &normed, seq, cos, sin)?;
        let h = self.add(&x.bits, &attended)?;

        let normed = self.rms_norm(&h, &layer.post_norm, seq, hidden)?;
        let normed = Bf16Activation::from_bits(normed, seq, hidden)?;
        let gate = self.project(&normed, &layer.gate_proj)?;
        let up = self.project(&normed, &layer.up_proj)?;
        let act = self.silu_mul(&gate, &up)?;
        let act = Bf16Activation::from_bits(act, seq, self.config.intermediate_size)?;
        let down = self.project(&act, &layer.down_proj)?;
        Ok(Bf16Activation::from_bits(
            self.add(&h, &down)?,
            seq,
            hidden,
        )?)
    }

    /// One attention block: projections, per-head norms, rotary, causal
    /// attention over the repeated key/value heads, output projection.
    fn attention(
        &self,
        layer: &GpuLayer,
        x: &Bf16Activation,
        seq: usize,
        cos: &CudaSlice<u16>,
        sin: &CudaSlice<u16>,
    ) -> Result<CudaSlice<u16>, TextGpuError> {
        let c = &self.config;
        let (hd, nq, nkv) = (c.head_dim, c.num_attention_heads, c.num_key_value_heads);

        let q = self.project(x, &layer.q_proj)?;
        let mut q = self.rms_norm(&q, &layer.q_norm, seq * nq, hd)?;
        self.rope(&mut q, cos, sin, seq, nq)?;
        let k = self.project(x, &layer.k_proj)?;
        let mut k = self.rms_norm(&k, &layer.k_norm, seq * nkv, hd)?;
        self.rope(&mut k, cos, sin, seq, nkv)?;
        let v = self.project(x, &layer.v_proj)?;

        let k = self.repeat_kv(&k, seq)?;
        let v = self.repeat_kv(&v, seq)?;
        // Safety: q, k and v are `[seq, nq, 128]` bf16, the shape the kernel
        // reads; the whole prompt is the causal text prefix.
        let context = unsafe {
            fused_block_causal_attention(&self.dev, &self.kernels, &q, &k, &v, seq, seq, nq)
        }?;
        self.project(&context, &layer.o_proj)
    }

    // -- op wrappers ---------------------------------------------------------

    fn alloc_bits(&self, len: usize) -> Result<CudaSlice<u16>, TextGpuError> {
        // Safety: every caller's kernel writes all `len` elements before any read.
        Ok(unsafe { self.dev.alloc_uninit::<u16>(len) }?)
    }

    /// `x · Wᵀ`: a cuBLAS bf16 GEMM with f32 accumulation, rounded to bf16.
    fn project(&self, x: &Bf16Activation, w: &Weight) -> Result<CudaSlice<u16>, TextGpuError> {
        // Safety: `w.bits` is `n * k` bf16, checked at load, and every
        // activation fed here is `k` wide by the same config.
        let wide = unsafe { gemm_bf16(&self.dev, &w.bits, x, w.n) }?;
        let out = self.alloc_bits(x.m() * w.n)?;
        // Safety: `wide` holds `m * n` f32, the length of `out`.
        unsafe { convert_f32_to_bf16(&self.dev, &self.kernels.f32_to_bf16_bits, &wide, &out) }?;
        Ok(out)
    }

    /// RMSNorm over each `dim`-wide row of a `[rows, dim]` view.
    fn rms_norm(
        &self,
        x: &CudaSlice<u16>,
        weight: &DevVec,
        rows: usize,
        dim: usize,
    ) -> Result<CudaSlice<u16>, TextGpuError> {
        let out = self.alloc_bits(rows * dim)?;
        let (du, eps) = (dim as u32, self.config.rms_norm_eps);
        // Safety: `x` and `out` are `rows * dim`, `weight` is `dim`.
        unsafe {
            self.dev
                .stream
                .launch_builder(&self.text_ops.rms_norm)
                .arg(x)
                .arg(&weight.buf)
                .arg(&out)
                .arg(&du)
                .arg(&eps)
                .launch(LaunchConfig {
                    grid_dim: (rows as u32, 1, 1),
                    block_dim: (NORM_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| RuntimeError::Compute(format!("rms_norm_bf16: {e}")))?;
        }
        Ok(out)
    }

    /// The half-split rotary, in place on `[seq, heads, head_dim]`.
    fn rope(
        &self,
        x: &mut CudaSlice<u16>,
        cos: &CudaSlice<u16>,
        sin: &CudaSlice<u16>,
        seq: usize,
        heads: usize,
    ) -> Result<(), TextGpuError> {
        let hd = self.config.head_dim;
        let (su, hu, hdu) = (seq as u32, heads as u32, hd as u32);
        // Safety: `x` is `seq * heads * head_dim`; the tables are `seq * head_dim`.
        unsafe {
            self.dev
                .stream
                .launch_builder(&self.text_ops.rope)
                .arg(x)
                .arg(cos)
                .arg(sin)
                .arg(&su)
                .arg(&hu)
                .arg(&hdu)
                .launch(LaunchConfig {
                    grid_dim: (su, 1, 1),
                    block_dim: ((hd as u32 / 2).min(THREADS), 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| RuntimeError::Compute(format!("rope_bf16: {e}")))?;
        }
        Ok(())
    }

    /// `[seq, nkv, head_dim]` -> `[seq, nq, head_dim]`.
    fn repeat_kv(&self, x: &CudaSlice<u16>, seq: usize) -> Result<CudaSlice<u16>, TextGpuError> {
        let c = &self.config;
        let total = seq * c.num_attention_heads * c.head_dim;
        let out = self.alloc_bits(total)?;
        let (su, nkv, groups, hd) = (
            seq as u32,
            c.num_key_value_heads as u32,
            c.num_key_value_groups() as u32,
            c.head_dim as u32,
        );
        // Safety: `x` is `seq * nkv * head_dim`, `out` is `total`; the grid covers `total`.
        unsafe {
            self.dev
                .stream
                .launch_builder(&self.text_ops.repeat_kv)
                .arg(x)
                .arg(&out)
                .arg(&su)
                .arg(&nkv)
                .arg(&groups)
                .arg(&hd)
                .launch(flat_grid(total)?)
                .map_err(|e| RuntimeError::Compute(format!("repeat_kv: {e}")))?;
        }
        Ok(out)
    }

    /// `bf16(bf16(silu(gate)) * up)`.
    fn silu_mul(
        &self,
        gate: &CudaSlice<u16>,
        up: &CudaSlice<u16>,
    ) -> Result<CudaSlice<u16>, TextGpuError> {
        let out = self.alloc_bits(gate.len())?;
        let n = gate.len() as u32;
        // Safety: all three buffers are `n`, and the grid covers them.
        unsafe {
            self.dev
                .stream
                .launch_builder(&self.text_ops.silu_mul)
                .arg(gate)
                .arg(up)
                .arg(&out)
                .arg(&n)
                .launch(flat_grid(gate.len())?)
                .map_err(|e| RuntimeError::Compute(format!("silu_mul_bf16: {e}")))?;
        }
        Ok(out)
    }

    /// `bf16(a + b)`, into a fresh buffer.
    fn add(&self, a: &CudaSlice<u16>, b: &CudaSlice<u16>) -> Result<CudaSlice<u16>, TextGpuError> {
        let out = self.alloc_bits(a.len())?;
        let n = a.len() as u32;
        // Safety: all three buffers are `n`, and the grid covers them.
        unsafe {
            self.dev
                .stream
                .launch_builder(&self.text_ops.add)
                .arg(a)
                .arg(b)
                .arg(&out)
                .arg(&n)
                .launch(flat_grid(a.len())?)
                .map_err(|e| RuntimeError::Compute(format!("add_bf16: {e}")))?;
        }
        Ok(out)
    }
}

/// One thread per element, in blocks of [`THREADS`]; the kernels index with
/// `unsigned int`, so the element count must fit one.
fn flat_grid(total: usize) -> Result<LaunchConfig, TextGpuError> {
    let total = u32::try_from(total).map_err(|_| {
        RuntimeError::Compute(format!("{total} elements exceed the kernels' u32 index"))
    })?;
    Ok(LaunchConfig {
        grid_dim: (total.div_ceil(THREADS), 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    })
}

// ---------------------------------------------------------------------------
// Loading helpers
// ---------------------------------------------------------------------------

/// The tower tensor `stem`, with `.weight` appended.
fn tensor_name(stem: &str) -> String {
    format!("{PREFIX}{stem}.weight")
}

/// The tensor `name`, checked to have exactly `shape`.
fn looked_up<'a>(
    file: &'a LbiFile,
    name: &str,
    shape: &[u64],
) -> Result<&'a crate::lbi::TensorEntry, TextGpuError> {
    let entry = file
        .get(name)
        .ok_or_else(|| TextEncoderError::MissingTensor {
            name: name.to_string(),
        })?;
    if entry.shape != shape {
        return Err(TextEncoderError::ShapeMismatch {
            name: name.to_string(),
            expected: shape.to_vec(),
            found: entry.shape.clone(),
        }
        .into());
    }
    Ok(entry)
}

/// The `[n, k]` matrix `stem`, checked to be stored bf16; returns its full name.
fn bf16_entry(file: &LbiFile, stem: &str, n: usize, k: usize) -> Result<String, TextGpuError> {
    let name = tensor_name(stem);
    let entry = looked_up(file, &name, &[n as u64, k as u64])?;
    if entry.quant != QuantScheme::Bf16 {
        return Err(TextGpuError::UnsupportedStorage {
            tensor: name,
            scheme: format!("{:?}", entry.quant),
        });
    }
    Ok(name)
}

/// Every matrix the tower uploads, `(stem, n, k)`: the embedding table and
/// each layer's seven projections.
fn matrices(c: &TextEncoderConfig) -> Vec<(String, usize, usize)> {
    let q_width = c.num_attention_heads * c.head_dim;
    let kv_width = c.num_key_value_heads * c.head_dim;
    let mut all = vec![("embed_tokens".to_string(), c.vocab_size, c.hidden_size)];
    for i in 0..c.num_layers {
        for (stem, n, k) in [
            ("self_attn.q_proj", q_width, c.hidden_size),
            ("self_attn.k_proj", kv_width, c.hidden_size),
            ("self_attn.v_proj", kv_width, c.hidden_size),
            ("self_attn.o_proj", c.hidden_size, q_width),
            ("mlp.gate_proj", c.intermediate_size, c.hidden_size),
            ("mlp.up_proj", c.intermediate_size, c.hidden_size),
            ("mlp.down_proj", c.hidden_size, c.intermediate_size),
        ] {
            all.push((format!("layers.{i}.{stem}"), n, k));
        }
    }
    all
}

/// The tower's matrices copied once into page-locked host memory. The driver
/// transfers page-locked memory straight to the device, so a load from these
/// copies runs at the link's full rate, for the matrices' size in host memory
/// held while they live.
pub struct PinnedMatrices {
    copies: HashMap<String, PinnedHostSlice<u8>>,
}

impl PinnedMatrices {
    /// Copy every matrix [`TextGpu::load_with`] uploads.
    pub fn copy(
        file: &LbiFile,
        config: &TextEncoderConfig,
        ctx: &Arc<CudaContext>,
    ) -> Result<Self, TextGpuError> {
        let mut copies = HashMap::new();
        for (stem, n, k) in matrices(config) {
            let name = bf16_entry(file, &stem, n, k)?;
            let bytes = file.tensor_bytes(&name).expect("entry resolved above");
            let page_locked = |e: cudarc::driver::DriverError| {
                RuntimeError::Compute(format!("page-lock {} bytes for {name}: {e}", bytes.len()))
            };
            // Safety: every byte is written by the copy below before it is read.
            let mut copy = unsafe { ctx.alloc_pinned::<u8>(bytes.len()) }.map_err(page_locked)?;
            copy.as_mut_slice()
                .map_err(page_locked)?
                .copy_from_slice(bytes);
            copies.insert(name, copy);
        }
        Ok(Self { copies })
    }

    fn get(&self, name: &str) -> Option<&[u8]> {
        self.copies.get(name)?.as_slice().ok()
    }
}

/// A `[n, k]` matrix, which must be stored bf16, uploaded as stored — from its
/// page-locked copy when there is one.
fn matrix(
    dev: &CudaDevice,
    file: &LbiFile,
    pinned: Option<&PinnedMatrices>,
    stem: &str,
    n: usize,
    k: usize,
) -> Result<Weight, TextGpuError> {
    let name = bf16_entry(file, stem, n, k)?;
    let bytes = match pinned.and_then(|p| p.get(&name)) {
        Some(copy) => copy,
        None => file.tensor_bytes(&name).expect("entry resolved above"),
    };
    Ok(Weight {
        bits: super::dit_gpu::ops::upload_16bit(dev, bytes)?,
        n,
        k,
    })
}

/// A 1-D weight, widened to f32 (exact for the checkpoint's float dtypes).
fn vector(
    dev: &CudaDevice,
    file: &LbiFile,
    stem: &str,
    len: usize,
) -> Result<DevVec, TextGpuError> {
    let name = tensor_name(stem);
    looked_up(file, &name, &[len as u64])?;
    Ok(launch::upload(dev, &file.read_f32(&name)?)?)
}
