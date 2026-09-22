//! The Qwen3-VL text tower, on the GPU.
//!
//! This mirrors [`crate::text_encoder`] step for step: the same projections in
//! the same order, the same per-head `q_norm`/`k_norm` applied after the head
//! split and before rotary (`v` gets neither), the same interleaved mRoPE, the
//! same causal grouped-query softmax with the same operand order, and the same
//! return — the last layer's hidden state **without** the final RMS norm, which
//! the pipeline's forward hook suppresses. The CPU reference is the
//! specification, so a disagreement is a bug in one of the two rather than a
//! tolerance question.
//!
//! # What is reused, and what is not
//!
//! Reused unchanged: `gemm_16bit` and `gemm_f32_bias` for the projections,
//! `rmsnorm_per_head` from `norm.cu` in its shared-weight mode, `swiglu_inplace`
//! and `residual_add_copy` from `activations.cu`.
//!
//! Two kernels that look reusable are not, and both mistakes would produce a
//! correctly-shaped, finite, wrong tensor — the failure mode this driver is
//! written to avoid. `text_ops.cu` carries the full argument; in short:
//!
//! * `image_ops.cu`'s `mrope_interleaved` rotates adjacent complex pairs inside
//!   a head, while `apply_rope` pairs channel `j` with `j + head_dim/2`. The
//!   table is interleaved either way, so the name matches and the arithmetic
//!   does not.
//! * `image_ops.cu`'s `attn_block_causal` indexes keys and values by the *query*
//!   head index, so it cannot express this tower's 32-query/8-key grouping: it
//!   would read the wrong key heads. `attn_causal_gqa` carries the mapping.
//!
//! # Weights
//!
//! The tensor that needs care is the embedding table: 151936 x 4096 BF16 is
//! ~1.16 GiB and stays resident in its stored dtype, with the prompt's rows
//! gathered and widened by `embed_gather` — the same "only the prompt's rows"
//! economy [`crate::text_encoder::TextEncoder`]'s mmap path gets for free, and
//! the reason a forward does not add ~1.3 GiB by expanding the table to f32.
//!
//! `model.language_model.norm.weight` is deliberately **not** loaded. It is in
//! the CPU reference's manifest because that module states what a Qwen3-VL text
//! tower contains, but its bytes are never decoded there either: the forward
//! stops before the norm. Loading it here would reserve a device buffer for a
//! tensor no launch reads.
//!
//! # Memory
//!
//! 14.1 GiB of BF16 weights (the container also carries the vision tower,
//! which is not loaded) plus f32 activations (a `[seq, 12288]` MLP
//! intermediate is the largest, 1.2 MiB at the shipped 24-row prompt). The DiT
//! is not co-resident: the encoder's output feeds the transformer, so the two
//! run in sequence and the pipeline already holds one component at a time.

use std::path::Path;

use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
use lumen_format::QuantScheme;
use lumen_runtime::cuda::ffi::CudaDevice;
use lumen_runtime::error::RuntimeError;

use super::dit_gpu::ops as dit_ops;
use super::launch::{self, DevVec};
use super::{ImageKernels, TEXT_OPS_SOURCE};
use crate::lbi::{LbiError, LbiFile};
use crate::tensor::Matrix;
use crate::text_encoder::{rope_tables, TextEncoderConfig, TextEncoderError};

/// Every text-tower tensor sits under this prefix.
///
/// A second spelling of `text_encoder.rs`'s private `PREFIX`: this module cannot
/// see that one, and the task that produced this file is scoped to it. A drift
/// between the two shows up as a `MissingTensor` at [`TextGpu::load`], before a
/// single byte reaches the device — loud, not a silent numeric disagreement.
const PREFIX: &str = "model.language_model.";

/// Threads per block for the elementwise, gather and rotary kernels.
const THREADS: u32 = 256;

/// The largest block a CUDA launch may request.
///
/// A layer norm is 4096 wide and `rmsnorm_per_head` stages per-warp partials in
/// a shared array sized from `blockDim.x`, so `dim` is capped here rather than
/// passed through: a 4096-thread block is not a legal launch at all, and the
/// kernel's strided loops make the cap free.
const MAX_BLOCK: u32 = 1024;

/// The attention kernel launches one thread per head channel, so this is both
/// its block size and its ceiling on `head_dim`. `TextEncoderConfig`'s own
/// `MAX_HEAD_DIM` is 4096, so a config that passed there can still be wider than
/// a block may be; [`TextGpu::load`] refuses it rather than letting the launch
/// fail with a bare CUDA error. The shipped tower is 128.
const ATTN_THREADS: u32 = 1024;

/// What went wrong loading or running the text tower on the device.
#[derive(Debug)]
pub enum TextGpuError {
    /// Allocating, copying or launching on the device.
    Cuda(RuntimeError),
    /// A tensor is missing, the wrong shape, or the config is unusable.
    /// Reusing the reference's error keeps a shape failure reported the same
    /// way whichever path found it.
    Text(TextEncoderError),
    /// A weight is stored in a scheme with no dispatch.
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
                    "tensor {tensor} is stored as {scheme}, which has no dispatch"
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

/// The `.lbi` storage scheme of a weight, as the two GEMMs dispatch on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Storage {
    F32,
    F16,
    Bf16,
}

impl Storage {
    fn of(entry: &crate::lbi::TensorEntry, tensor: &str) -> Result<Self, TextGpuError> {
        match entry.quant {
            QuantScheme::F32 => Ok(Self::F32),
            QuantScheme::F16 => Ok(Self::F16),
            QuantScheme::Bf16 => Ok(Self::Bf16),
            other => Err(TextGpuError::UnsupportedStorage {
                tensor: tensor.to_string(),
                scheme: format!("{other:?}"),
            }),
        }
    }
}

/// A `[rows, cols]` linear weight resident on the device, in the dtype the
/// `.lbi` stores.
///
/// `rows` is the output width and `cols` the input width, kept alongside the
/// buffer rather than derived from its length: which factor is which is not
/// recoverable from an element count, and a transposed operand is a silent wrong
/// answer.
enum DevMatrix {
    F32 {
        buf: DevVec,
        rows: usize,
        cols: usize,
    },
    F16 {
        buf: CudaSlice<u16>,
        rows: usize,
        cols: usize,
    },
    Bf16 {
        buf: CudaSlice<u16>,
        rows: usize,
        cols: usize,
    },
}

impl DevMatrix {
    fn rows(&self) -> usize {
        match self {
            Self::F32 { rows, .. } | Self::F16 { rows, .. } | Self::Bf16 { rows, .. } => *rows,
        }
    }

    fn cols(&self) -> usize {
        match self {
            Self::F32 { cols, .. } | Self::F16 { cols, .. } | Self::Bf16 { cols, .. } => *cols,
        }
    }

    /// `out[M, N] = a[M, K] * W[N, K]^T` by weight dtype: `gemm_16bit` for a
    /// 16-bit weight, `gemm_f32_bias` without a bias for an f32 one.
    ///
    /// `m` is the row count; the weight supplies both widths. The width check
    /// runs here so a transposed operand is refused rather than read past.
    fn gemm(
        &self,
        dev: &CudaDevice,
        kernels: &ImageKernels,
        ops: &dit_ops::DitOps,
        a: &DevVec,
        m: usize,
    ) -> Result<DevVec, TextGpuError> {
        let (n, k) = (self.rows(), self.cols());
        if a.len != m * k {
            return Err(TextEncoderError::BadConfig(format!(
                "a projection input is {} values, but its weight wants {}",
                a.len,
                m * k
            ))
            .into());
        }
        Ok(match self {
            Self::F32 { buf, .. } => launch::linear(dev, kernels, a, buf, None, m, n, k)?,
            Self::F16 { buf, .. } => {
                dit_ops::gemm_16bit(dev, ops, buf, a, m, n, k, dit_ops::Gemm16::F16)?
            }
            Self::Bf16 { buf, .. } => {
                dit_ops::gemm_16bit(dev, ops, buf, a, m, n, k, dit_ops::Gemm16::Bf16)?
            }
        })
    }
}

/// One decoder layer, resident on the device.
struct GpuLayer {
    input_norm: DevVec,
    q_proj: DevMatrix,
    k_proj: DevMatrix,
    v_proj: DevMatrix,
    o_proj: DevMatrix,
    q_norm: DevVec,
    k_norm: DevVec,
    post_norm: DevVec,
    gate_proj: DevMatrix,
    up_proj: DevMatrix,
    down_proj: DevMatrix,
}

/// The text tower's embedding table, resident in its stored dtype.
struct DevEmbedding {
    bytes: CudaSlice<u8>,
    vocab: usize,
    hidden: usize,
    storage: Storage,
}

/// The entry points `text_ops.cu` provides.
struct TextOps {
    embed_gather: CudaFunction,
    rope_half: CudaFunction,
    attn_causal_gqa: CudaFunction,
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
            embed_gather: get("embed_gather")?,
            rope_half: get("rope_half_interleaved")?,
            attn_causal_gqa: get("attn_causal_gqa")?,
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
    /// `gemm_f32_bias`, for a tower whose weights are stored f32 (the CPU
    /// reference's own dtype, and the test fixtures').
    kernels: ImageKernels,
    /// `gemm_16bit`, for a BF16 or F16 checkpoint.
    ops: dit_ops::DitOps,
    text_ops: TextOps,
    /// `rmsnorm_per_head`, from `norm.cu`.
    per_head_norm: CudaFunction,
    /// `swiglu_inplace`, from `activations.cu`.
    swiglu: CudaFunction,
    /// `residual_add_copy`, from `activations.cu`.
    residual_add: CudaFunction,

    config: TextEncoderConfig,
    embed: DevEmbedding,
    layers: Vec<GpuLayer>,
}

impl TextGpu {
    /// Load every text-tower weight from a converted `.lbi` onto the device,
    /// assuming the shipped Qwen-Image-2.1 architecture.
    pub fn load(lbi: &Path, dev: &CudaDevice) -> Result<Self, TextGpuError> {
        let file = LbiFile::open(lbi)?;
        let config = TextEncoderConfig::from_lbi_config(file.config())?;
        Self::load_with(&file, dev, config)
    }

    /// Load against an explicit architecture.
    ///
    /// The config's own `validate` has already run — `from_lbi_config` calls it,
    /// and a caller supplying one by hand is expected to have called it — so the
    /// dimensions here are bounded and the products below cannot overflow.
    pub fn load_with(
        file: &LbiFile,
        dev: &CudaDevice,
        config: TextEncoderConfig,
    ) -> Result<Self, TextGpuError> {
        // `head_dim` is a launch block on two paths: the per-head norms index
        // `rows * head_dim` with one row per block, and `attn_causal_gqa` runs
        // one thread per head channel. A config wider than a legal block would
        // otherwise fail as a bare CUDA error mid-forward.
        if config.head_dim > ATTN_THREADS as usize {
            return Err(TextEncoderError::BadConfig(format!(
                "head_dim {} exceeds the {ATTN_THREADS}-thread block these kernels launch",
                config.head_dim
            ))
            .into());
        }

        let own = CudaDevice::new(dev.ctx.ordinal())?;
        let kernels = ImageKernels::load(&own)?;
        let ops = dit_ops::load(&own)?;
        let text_ops = TextOps::load(&own)?;
        let per_head_norm = load_from_source(
            &own,
            lumen_runtime::cuda::shaders::NORM_KERNEL_SOURCE,
            "rmsnorm_per_head",
        )?;
        let swiglu = load_from_source(
            &own,
            lumen_runtime::cuda::shaders::ACTIVATIONS_KERNEL_SOURCE,
            "swiglu_inplace",
        )?;
        let residual_add = load_from_source(
            &own,
            lumen_runtime::cuda::shaders::ACTIVATIONS_KERNEL_SOURCE,
            "residual_add_copy",
        )?;

        let c = &config;
        let hidden = c.hidden_size;
        let q_width = c.num_attention_heads * c.head_dim;
        let kv_width = c.num_key_value_heads * c.head_dim;

        let embed = load_embedding(&own, file, c)?;

        let mut layers = Vec::with_capacity(c.num_layers);
        for i in 0..c.num_layers {
            let p = format!("layers.{i}");
            layers.push(GpuLayer {
                input_norm: vector(&own, file, &format!("{p}.input_layernorm"), hidden)?,
                q_proj: matrix(
                    &own,
                    file,
                    &format!("{p}.self_attn.q_proj"),
                    q_width,
                    hidden,
                )?,
                k_proj: matrix(
                    &own,
                    file,
                    &format!("{p}.self_attn.k_proj"),
                    kv_width,
                    hidden,
                )?,
                v_proj: matrix(
                    &own,
                    file,
                    &format!("{p}.self_attn.v_proj"),
                    kv_width,
                    hidden,
                )?,
                o_proj: matrix(
                    &own,
                    file,
                    &format!("{p}.self_attn.o_proj"),
                    hidden,
                    q_width,
                )?,
                q_norm: vector(&own, file, &format!("{p}.self_attn.q_norm"), c.head_dim)?,
                k_norm: vector(&own, file, &format!("{p}.self_attn.k_norm"), c.head_dim)?,
                post_norm: vector(&own, file, &format!("{p}.post_attention_layernorm"), hidden)?,
                gate_proj: matrix(
                    &own,
                    file,
                    &format!("{p}.mlp.gate_proj"),
                    c.intermediate_size,
                    hidden,
                )?,
                up_proj: matrix(
                    &own,
                    file,
                    &format!("{p}.mlp.up_proj"),
                    c.intermediate_size,
                    hidden,
                )?,
                down_proj: matrix(
                    &own,
                    file,
                    &format!("{p}.mlp.down_proj"),
                    hidden,
                    c.intermediate_size,
                )?,
            });
        }

        Ok(Self {
            config,
            embed,
            layers,
            dev: own,
            kernels,
            ops,
            text_ops,
            per_head_norm,
            swiglu,
            residual_add,
        })
    }

    pub fn config(&self) -> &TextEncoderConfig {
        &self.config
    }

    /// Encode a prompt to `[seq, hidden]`, **without** the final RMS norm.
    ///
    /// Same contract as [`crate::text_encoder::TextEncoder::forward`]: all three
    /// M-RoPE rows are the plain `0..seq-1`, because a text-to-image prompt has
    /// no grid and the model falls back to `arange` broadcast across T, H and W.
    /// The caller gets every row, template included, and drops the leading
    /// system-message rows itself.
    pub fn forward(&self, token_ids: &[u32]) -> Result<Matrix, TextGpuError> {
        if token_ids.is_empty() {
            return Err(TextEncoderError::EmptyInput.into());
        }
        let c = &self.config;
        let seq = token_ids.len();
        let hidden = c.hidden_size;

        // The prompt's ids, range-checked here so the gather kernel can trust
        // them. `embed_gather` reads at `id * hidden`, so an id at or above the
        // vocabulary would read past the end of the table.
        for &id in token_ids {
            if id as usize >= self.embed.vocab {
                return Err(TextEncoderError::TokenOutOfRange {
                    id,
                    vocab: self.embed.vocab,
                }
                .into());
            }
        }

        // The rotary tables are host f32, built by the reference's own function
        // so the schedule is the specification's rather than a second
        // implementation of it. A text prompt puts the same `0..seq-1` on all
        // three rows, which is what makes the recomposition a no-op.
        let positions: Vec<f32> = (0..seq).map(|i| i as f32).collect();
        let (cos, sin) = rope_tables(c, [&positions, &positions, &positions])?;
        let g_cos = launch::upload(&self.dev, &cos.data)?;
        let g_sin = launch::upload(&self.dev, &sin.data)?;

        let g_ids = self.dev.htod_copy(token_ids)?;

        let mut x = self.gather_embed(&g_ids, seq)?;
        for layer in &self.layers {
            x = self.decoder_layer(layer, &x, seq, &g_cos, &g_sin)?;
        }
        let data = launch::download(&self.dev, &x)?;
        Ok(Matrix::new(seq, hidden, data))
    }

    // -- the pieces of the forward ------------------------------------------

    /// Gather the prompt's rows out of the resident embedding table.
    fn gather_embed(&self, ids: &CudaSlice<u32>, seq: usize) -> Result<DevVec, TextGpuError> {
        let out = launch::alloc(&self.dev, seq * self.embed.hidden)?;
        let (su, hu) = (seq as u32, self.embed.hidden as u32);
        let dt: u32 = match self.embed.storage {
            Storage::F16 => 0,
            Storage::Bf16 => 1,
            Storage::F32 => 2,
        };
        // Safety: `out` is `seq * hidden` floats; `ids` is `seq` entries, each
        // checked in range above; the table is `vocab * hidden * width` bytes,
        // which `looked_up` pinned against the declared shape.
        unsafe {
            self.dev
                .stream
                .launch_builder(&self.text_ops.embed_gather)
                .arg(&self.embed.bytes)
                .arg(ids)
                .arg(&out.buf)
                .arg(&su)
                .arg(&hu)
                .arg(&dt)
                .launch(LaunchConfig {
                    grid_dim: (seq as u32, 1, 1),
                    block_dim: (THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| RuntimeError::Compute(format!("embed_gather: {e}")))?;
        }
        Ok(out)
    }

    /// One decoder layer: pre-norm attention and pre-norm MLP, each residual.
    fn decoder_layer(
        &self,
        layer: &GpuLayer,
        x: &DevVec,
        seq: usize,
        cos: &DevVec,
        sin: &DevVec,
    ) -> Result<DevVec, TextGpuError> {
        let eps = self.config.rms_norm_eps;
        let width = self.config.hidden_size;

        let normed = self.rms_norm(x, &layer.input_norm, seq, width, eps)?;
        let attended = self.attention(layer, &normed, seq, cos, sin)?;
        let hidden = self.residual(x, &attended)?;

        let normed = self.rms_norm(&hidden, &layer.post_norm, seq, width, eps)?;
        let expanded = self.mlp(layer, &normed, seq)?;
        self.residual(&hidden, &expanded)
    }

    /// One attention block: per-head norms, rotary, causal GQA softmax, output
    /// projection.
    fn attention(
        &self,
        layer: &GpuLayer,
        x: &DevVec,
        seq: usize,
        cos: &DevVec,
        sin: &DevVec,
    ) -> Result<DevVec, TextGpuError> {
        let c = &self.config;
        let (head_dim, nq, nkv) = (c.head_dim, c.num_attention_heads, c.num_key_value_heads);

        // `q_norm`/`k_norm` are `[head_dim]` vectors applied to each head's
        // slice, which is the row view `[seq * heads, head_dim]`. `v` carries
        // neither a norm nor a rotation.
        let q = layer
            .q_proj
            .gemm(&self.dev, &self.kernels, &self.ops, x, seq)?;
        let q = self.rms_norm(&q, &layer.q_norm, seq * nq, head_dim, c.rms_norm_eps)?;
        let q = self.rope(q, cos, sin, seq, nq, head_dim)?;

        let k = layer
            .k_proj
            .gemm(&self.dev, &self.kernels, &self.ops, x, seq)?;
        let k = self.rms_norm(&k, &layer.k_norm, seq * nkv, head_dim, c.rms_norm_eps)?;
        let k = self.rope(k, cos, sin, seq, nkv, head_dim)?;

        let v = layer
            .v_proj
            .gemm(&self.dev, &self.kernels, &self.ops, x, seq)?;

        let context = self.causal_gqa(&q, &k, &v, seq, nq, nkv, head_dim)?;
        layer
            .o_proj
            .gemm(&self.dev, &self.kernels, &self.ops, &context, seq)
    }

    /// `down_proj(silu(gate_proj(x)) * up_proj(x))`.
    fn mlp(&self, layer: &GpuLayer, x: &DevVec, seq: usize) -> Result<DevVec, TextGpuError> {
        let mut gate = layer
            .gate_proj
            .gemm(&self.dev, &self.kernels, &self.ops, x, seq)?;
        let up = layer
            .up_proj
            .gemm(&self.dev, &self.kernels, &self.ops, x, seq)?;
        self.swiglu_inplace(&mut gate, &up)?;
        layer
            .down_proj
            .gemm(&self.dev, &self.kernels, &self.ops, &gate, seq)
    }

    // -- op wrappers ---------------------------------------------------------

    /// `rms_norm_rows` over a `[rows, dim]` view, one block per row.
    fn rms_norm(
        &self,
        x: &DevVec,
        weight: &DevVec,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<DevVec, TextGpuError> {
        let out = launch::alloc(&self.dev, x.len)?;
        let du = dim as u32;
        // "One shared `[dim]` weight" — what `rms_norm_rows(flat, weight, eps)`
        // does for every row, because `norm_per_head` views the matrix as
        // `[rows, dim]` and applies the same vector to each. Every norm in the
        // tower is a `[dim]` vector in the checkpoint, so a non-zero stride
        // would index the weight per row and read past its end.
        let stride = 0u32;
        // A layer norm is 4096 wide, past the 1024 a block may hold, so the
        // block is capped and the kernel strides the remainder. Its own
        // `for (i = tid; i < head_dim; i += block_size)` is what makes that
        // correct; the shared array it reduces through must be sized from the
        // block, not from `dim`.
        let block = (dim as u32).min(MAX_BLOCK);
        // Safety: `x` is `rows * dim` floats; one block per row covers it.
        unsafe {
            self.dev
                .stream
                .launch_builder(&self.per_head_norm)
                .arg(&x.buf)
                .arg(&weight.buf)
                .arg(&out.buf)
                .arg(&eps)
                .arg(&du)
                .arg(&stride)
                .launch(LaunchConfig {
                    grid_dim: (rows as u32, 1, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: (block / 32).max(1) * 4,
                })
                .map_err(|e| RuntimeError::Compute(format!("rmsnorm_per_head: {e}")))?;
        }
        Ok(out)
    }

    /// The interleaved mRoPE, in place on `x`, which is consumed.
    ///
    /// Taking ownership is what lets the rotation run in place: the projected
    /// `q`/`k` are fresh buffers the caller has no other use for, so no copy is
    /// needed and none is made.
    fn rope(
        &self,
        mut x: DevVec,
        cos: &DevVec,
        sin: &DevVec,
        seq: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<DevVec, TextGpuError> {
        let (su, hu, hdu) = (seq as u32, heads as u32, head_dim as u32);
        let slots = (head_dim / 2) as u32;
        // Safety: `x` is `seq * heads * head_dim` floats; `cos`/`sin` are
        // `seq * head_dim` each, the extent the kernel indexes.
        unsafe {
            self.dev
                .stream
                .launch_builder(&self.text_ops.rope_half)
                .arg(&mut x.buf)
                .arg(&cos.buf)
                .arg(&sin.buf)
                .arg(&su)
                .arg(&hu)
                .arg(&hdu)
                .launch(LaunchConfig {
                    grid_dim: (seq as u32, 1, 1),
                    block_dim: (slots.min(THREADS), 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| RuntimeError::Compute(format!("rope_half_interleaved: {e}")))?;
        }
        Ok(x)
    }

    /// Causal grouped-query attention, one block per (query position, head).
    #[allow(clippy::too_many_arguments)]
    fn causal_gqa(
        &self,
        q: &DevVec,
        k: &DevVec,
        v: &DevVec,
        seq: usize,
        nq: usize,
        nkv: usize,
        head_dim: usize,
    ) -> Result<DevVec, TextGpuError> {
        let out = launch::alloc(&self.dev, q.len)?;
        let (su, nqu, nkvu, hdu) = (seq as u32, nq as u32, nkv as u32, head_dim as u32);
        let scale = (head_dim as f32).powf(-0.5);
        let smem = (2 * head_dim * 4) as u32;
        // Safety: q/k/v/out hold `seq * heads * head_dim` floats for their
        // head counts, sized by the projections that produced them.
        unsafe {
            self.dev
                .stream
                .launch_builder(&self.text_ops.attn_causal_gqa)
                .arg(&q.buf)
                .arg(&k.buf)
                .arg(&v.buf)
                .arg(&out.buf)
                .arg(&su)
                .arg(&nqu)
                .arg(&nkvu)
                .arg(&hdu)
                .arg(&scale)
                .launch(LaunchConfig {
                    grid_dim: (seq as u32, nq as u32, 1),
                    block_dim: (head_dim as u32, 1, 1),
                    shared_mem_bytes: smem,
                })
                .map_err(|e| RuntimeError::Compute(format!("attn_causal_gqa: {e}")))?;
        }
        Ok(out)
    }

    /// `out[i] = silu(gate[i]) * up[i]`, in place on `gate`.
    fn swiglu_inplace(&self, gate: &mut DevVec, up: &DevVec) -> Result<(), TextGpuError> {
        let n = gate.len as u32;
        // Safety: both buffers are `n` floats, and the grid covers exactly that.
        unsafe {
            self.dev
                .stream
                .launch_builder(&self.swiglu)
                .arg(&mut gate.buf)
                .arg(&up.buf)
                .arg(&n)
                .launch(flat_grid(n))
                .map_err(|e| RuntimeError::Compute(format!("swiglu_inplace: {e}")))?;
        }
        Ok(())
    }

    /// `a + b` into a fresh buffer, so the pre-norm input survives the residual.
    fn residual(&self, a: &DevVec, b: &DevVec) -> Result<DevVec, TextGpuError> {
        let out = launch::alloc(&self.dev, a.len)?;
        let n = a.len as u32;
        // Safety: all three buffers are `n` floats, and the grid covers them.
        unsafe {
            self.dev
                .stream
                .launch_builder(&self.residual_add)
                .arg(&a.buf)
                .arg(&b.buf)
                .arg(&out.buf)
                .arg(&n)
                .launch(flat_grid(n))
                .map_err(|e| RuntimeError::Compute(format!("residual_add_copy: {e}")))?;
        }
        Ok(out)
    }
}

/// One block per 256 elements.
fn flat_grid(total: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (total.div_ceil(THREADS), 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
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

/// A `[rows, cols]` linear weight, uploaded in its stored dtype.
fn matrix(
    dev: &CudaDevice,
    file: &LbiFile,
    stem: &str,
    rows: usize,
    cols: usize,
) -> Result<DevMatrix, TextGpuError> {
    let name = tensor_name(stem);
    let entry = looked_up(file, &name, &[rows as u64, cols as u64])?;
    let bytes = file.tensor_bytes(&name).expect("entry resolved above");
    match Storage::of(entry, &name)? {
        Storage::F32 => Ok(DevMatrix::F32 {
            buf: launch::upload(dev, &file.read_f32(&name)?)?,
            rows,
            cols,
        }),
        Storage::F16 => Ok(DevMatrix::F16 {
            buf: dit_ops::upload_16bit(dev, bytes)?,
            rows,
            cols,
        }),
        Storage::Bf16 => Ok(DevMatrix::Bf16 {
            buf: dit_ops::upload_16bit(dev, bytes)?,
            rows,
            cols,
        }),
    }
}

/// A 1-D weight, widened to f32.
///
/// Every such weight is elementwise, so widening it on the host is exact for the
/// checkpoint's dtypes.
fn vector(
    dev: &CudaDevice,
    file: &LbiFile,
    stem: &str,
    len: usize,
) -> Result<DevVec, TextGpuError> {
    let name = tensor_name(stem);
    let entry = looked_up(file, &name, &[len as u64])?;
    Storage::of(entry, &name)?;
    Ok(launch::upload(dev, &file.read_f32(&name)?)?)
}

/// The embedding table, uploaded in its stored dtype.
///
/// The stored bytes move straight across — no re-encoding — so the gather
/// kernel's input format is the file's.
fn load_embedding(
    dev: &CudaDevice,
    file: &LbiFile,
    cfg: &TextEncoderConfig,
) -> Result<DevEmbedding, TextGpuError> {
    let name = tensor_name("embed_tokens");
    let entry = looked_up(
        file,
        &name,
        &[cfg.vocab_size as u64, cfg.hidden_size as u64],
    )?;
    let storage = Storage::of(entry, &name)?;
    let bytes = file.tensor_bytes(&name).expect("entry resolved above");
    Ok(DevEmbedding {
        bytes: dev.htod_copy(bytes)?,
        vocab: cfg.vocab_size,
        hidden: cfg.hidden_size,
        storage,
    })
}

/// Compile a source and resolve one entry point from it.
fn load_from_source(
    dev: &CudaDevice,
    source: &str,
    name: &str,
) -> Result<CudaFunction, RuntimeError> {
    let module = dev.compile_and_load(source)?;
    module
        .load_function(name)
        .map_err(|e| RuntimeError::Compute(format!("load {name}: {e}")))
}
