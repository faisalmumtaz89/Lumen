//! The Qwen-Image-2.1 VAE decoder, on the GPU.
//!
//! This mirrors [`crate::vae`] step for step: the same shapes and the same
//! operand order. The elementwise kernels keep the CPU's accumulation order
//! per output element. The convolutions run on cuBLAS as TF32 tensor-core
//! products with f32 accumulation, as the reference's cuDNN convolutions do
//! under torch's default `allow_tf32`; the attention products run in full f32,
//! the accuracy of the reference's f32 attention kernel. `vae-check-gpu`
//! checks the GPU decode against the reference's own output within 1.5e-4
//! relative error and against [`crate::vae`], which convolves in full f32,
//! within 1e-3.
//!
//! The four things [`crate::vae`]'s module doc calls out are the ones a
//! plausible "improvement" would silently break, and they carry over unchanged:
//!
//! 1. **Every convolution is 2-D.** `QwenImage21CausalConv3d` subclasses
//!    `nn.Conv2d` and squeezes the temporal axis away, so there is no temporal
//!    convolution anywhere in this model. `post_quant_conv` is 1x1, `conv_in`,
//!    `conv_out`, every residual block's convs and every upsampler's conv are
//!    3x3 with padding 1, and none is strided.
//! 2. **`QwenImage21RMS_norm` is `F.normalize`**, not an epsilon-RMS: the norm
//!    is clamped at 1e-12 and nothing is added under the square root. The two
//!    differ sharply for a small-magnitude channel vector, and
//!    `rms_norm_channels` in `vae_ops.cu` implements the former.
//! 3. **`is_residual` is true**, so each up block's output has the parameter-free
//!    `QwenImage21DupUp3D` shortcut added to it.
//! 4. **The feature cache is inert.** A decode processes exactly one frame, so
//!    the cache never influences a result and `upsampler.time_conv`'s weights
//!    are never read — they are the checkpoint's only VAE tensors this loader
//!    skips besides the encoder half.
//!
//! Activations stay on the device between launches. Every one is channel-major
//! `[batch, channels, h * w]`, which is the reference's row-major `Tensor4`
//! `[n, c, h, w]` flattened; the kernels split a flat index as
//! `(outer, row, column)` with the column innermost, which is exactly that
//! layout. There is one layout for the whole decoder and no transposes, so a
//! silent transposition has nowhere to hide — the one place the reference does
//! transpose (its attention block) is handled by reading the channel-major
//! activation directly.
//!
//! The two host round trips are the latents going up and the decoded f32 image
//! coming back. The clamp to `[-1, 1]` runs on the host after the download, as
//! the reference's `torch.clamp` runs on the finished tensor; it is the same two
//! `f32::min`/`f32::max` operations and costs nothing in precision.
//!
//! Dtype. Every weight is f32 on the device, widened on the host by the
//! container's own reader. That is not a lossy shortcut: the reference's loader
//! goes through `LbiFile::read_f32` for *every* tensor, so a checkpoint stored as
//! bf16 or f16 is already widened to f32 on the CPU side and the comparison is
//! one of summation order rather than of precision.

use std::path::Path;
use std::sync::Arc;

use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
use lumen_format::QuantScheme;
use lumen_runtime::cuda::ffi::CudaDevice;
use lumen_runtime::error::RuntimeError;

use super::launch::{self, DevVec};
use crate::lbi::{LbiError, LbiFile};
use crate::vae::{VaeConfig, VaeError};

/// The reference hard-codes `QwenImage21MidBlock(dims[0], dropout,
/// num_layers=1)` for both encoder and decoder. Mirrors `vae.rs`'s private
/// constant of the same name, which cannot be re-exported without editing that
/// file; a drift between the two would show up as a missing-tensor error,
/// because the attention it would add has its own weights.
const MID_BLOCK_NUM_LAYERS: usize = 1;

/// `F.normalize`'s default epsilon, passed to `rms_norm_channels`: the norm
/// is clamped up to this rather than an epsilon being added under the square
/// root. Mirrors `vae.rs`'s private `NORMALIZE_EPS`.
const NORMALIZE_EPS: f32 = 1e-12;

/// Threads per block for the elementwise kernels.
const THREADS: u32 = 256;

/// Threads per block for the attention softmax, one block per score row. Its
/// barriers require every thread of a block to arrive, which the kernel
/// guarantees by never returning before its last one.
const ATTN_THREADS: u32 = 256;

/// The most bytes of attention scores held at once: 1 GiB, which takes the
/// mid block's scores whole up to 2048x2048 (16,384 positions) and in chunks
/// of query rows beyond.
const ATTN_SCORE_BUDGET: usize = 1 << 30;

/// The grid-x limit CUDA allows (2^31 - 1). The decoder's largest activation —
/// 288 channels at 1024x1024 — needs 1.2 million blocks at 256 threads, four
/// orders of magnitude below this, so the clamp is a guard against a malformed
/// config rather than a limit the shipped architecture ever approaches.
const GRID_X_MAX: u64 = 0x7fff_ffff;

/// Output rows per convolution product. Fixed rather than derived from the
/// image height, so a decode in bands runs every row through products of the
/// same shape as the whole-image decode does.
const CONV_ROWS: usize = 16;

/// Bytes the convolution's column matrix may occupy per band.
///
/// 256 MiB keeps the largest layer's im2col to a few bands while leaving the
/// rest of the card for the activations, which reach 1.2 GB at 1024x1024.
const CONV_COL_BUDGET: usize = 256 * 1024 * 1024;

/// The kernel source this module compiles.
pub const VAE_OPS_SOURCE: &str = include_str!("vae_ops.cu");

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum VaeGpuError {
    /// Allocating, copying or launching on the device.
    Cuda(RuntimeError),
    /// The checkpoint, or the caller's arguments, disagree with the
    /// architecture. Reusing the reference's error keeps a shape failure
    /// reported the same way on both paths.
    Vae(VaeError),
    /// A weight is stored in a scheme with no dispatch.
    UnsupportedStorage { tensor: String, scheme: String },
    /// A launch a kernel cannot express. Every one of these is a programming
    /// error in this module rather than a property of a checkpoint.
    KernelLimit(String),
}

impl std::fmt::Display for VaeGpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cuda(e) => write!(f, "cuda: {e}"),
            Self::Vae(e) => write!(f, "{e}"),
            Self::UnsupportedStorage { tensor, scheme } => {
                write!(
                    f,
                    "tensor {tensor} is stored as {scheme}, which has no dispatch"
                )
            }
            Self::KernelLimit(m) => write!(f, "kernel limit: {m}"),
        }
    }
}

impl std::error::Error for VaeGpuError {}

impl From<RuntimeError> for VaeGpuError {
    fn from(e: RuntimeError) -> Self {
        Self::Cuda(e)
    }
}

impl From<VaeError> for VaeGpuError {
    fn from(e: VaeError) -> Self {
        Self::Vae(e)
    }
}

impl From<LbiError> for VaeGpuError {
    fn from(e: LbiError) -> Self {
        Self::Vae(e.into())
    }
}

fn mismatch(tensor: &str, expected: &[u64], actual: &[u64]) -> VaeGpuError {
    VaeGpuError::Vae(VaeError::ShapeMismatch {
        tensor: tensor.to_string(),
        expected: expected.to_vec(),
        actual: actual.to_vec(),
    })
}

/// Name the failed step in the error a caller sees. `cudarc` reports a launch
/// failure as a `DriverError` and a copy failure as a `RuntimeError`, so this
/// takes any `Display` rather than one of the two.
fn cuda<E: std::fmt::Display>(what: &str, e: E) -> VaeGpuError {
    VaeGpuError::Cuda(RuntimeError::Compute(format!("{what}: {e}")))
}

// ---------------------------------------------------------------------------
// Weights resident on the device
// ---------------------------------------------------------------------------

/// A 2-D convolution with stride 1 and symmetric zero padding, resident on the
/// device: `[out_c, in_c, k, k]` weights and an `[out_c]` bias, the `nn.Conv2d`
/// layout `vae.rs` also stores.
struct GpuConv {
    out_c: usize,
    in_c: usize,
    kh: usize,
    kw: usize,
    pad_h: usize,
    pad_w: usize,
    weight: DevVec,
    bias: DevVec,
}

impl GpuConv {
    /// Apply to `[batch, in_c, h, w]`, keeping the spatial extents.
    fn apply(
        &self,
        gpu: &VaeGpu,
        x: &DevVec,
        batch: usize,
        h: usize,
        w: usize,
    ) -> Result<DevVec, VaeGpuError> {
        gpu.conv(self, x, batch, h, w)
    }

    /// The tensor `prefix`.{weight,bias}, shape-checked, as `vae.rs`'s
    /// `load_conv` reads them.
    fn load(
        dev: &CudaDevice,
        file: &LbiFile,
        prefix: &str,
        out_c: usize,
        in_c: usize,
        k: usize,
        pad: usize,
    ) -> Result<Self, VaeGpuError> {
        let wname = format!("{prefix}.weight");
        let bname = format!("{prefix}.bias");
        let weight = weight_exact(dev, file, &wname, &[out_c, in_c, k, k])?;
        let bias = weight_exact(dev, file, &bname, &[out_c])?;
        Ok(Self {
            out_c,
            in_c,
            kh: k,
            kw: k,
            pad_h: pad,
            pad_w: pad,
            weight,
            bias,
        })
    }
}

/// `QwenImage21RMS_norm`'s `gamma`, resident on the device.
struct GpuNorm {
    gamma: DevVec,
    dim: usize,
}

impl GpuNorm {
    /// Apply to `[rows, hw]`, where `rows` is `batch * channels`.
    fn apply(
        &self,
        gpu: &VaeGpu,
        x: &DevVec,
        rows: usize,
        hw: usize,
    ) -> Result<DevVec, VaeGpuError> {
        gpu.rms_norm(self, x, rows, hw)
    }

    /// Load a `QwenImage21RMS_norm` `gamma`, accepting the same stored shapes
    /// `vae.rs`'s `load_norm` accepts: the reference creates the parameter as
    /// `torch.ones((dim, 1, 1))` or `torch.ones((dim, 1, 1, 1))` and the
    /// trailing axes exist only to broadcast, so any shape whose leading extent
    /// is `dim` and whose element count is `dim` carries the same values.
    fn load(
        dev: &CudaDevice,
        file: &LbiFile,
        name: &str,
        dim: usize,
        images: bool,
    ) -> Result<Self, VaeGpuError> {
        let entry = file
            .get(name)
            .ok_or_else(|| VaeGpuError::Vae(VaeError::MissingTensor(name.to_string())))?;
        let canonical: Vec<u64> = if images {
            vec![dim as u64, 1, 1]
        } else {
            vec![dim as u64, 1, 1, 1]
        };
        let count: u64 = entry.shape.iter().product();
        if entry.shape.first() != Some(&(dim as u64)) || count != dim as u64 {
            return Err(mismatch(name, &canonical, &entry.shape));
        }
        Ok(Self {
            gamma: read_f32_tensor(dev, file, name)?,
            dim,
        })
    }
}

/// `QwenImage21ResidualBlock`. Dropout is identity at inference.
struct GpuResnet {
    norm1: GpuNorm,
    conv1: GpuConv,
    norm2: GpuNorm,
    conv2: GpuConv,
    /// `conv_shortcut`, present only when the block changes channel count.
    shortcut: Option<GpuConv>,
}

impl GpuResnet {
    /// `vae.rs`'s `ResidualBlock::apply`. The channel counts come from the
    /// convolutions rather than from an argument, which is why they cannot
    /// disagree with the weights.
    fn apply(
        &self,
        gpu: &VaeGpu,
        x: &DevVec,
        batch: usize,
        h: usize,
        w: usize,
    ) -> Result<DevVec, VaeGpuError> {
        let hw = h * w;
        let residual = match &self.shortcut {
            Some(conv) => conv.apply(gpu, x, batch, h, w)?,
            None => gpu.copy(x)?,
        };
        // `h = rms_norm(x, norm1); silu(h); h = conv1(h); h = rms_norm(h,
        // norm2); silu(h); h = conv2(h); h += residual`.
        let mut cur = self.norm1.apply(gpu, x, batch * self.conv1.in_c, hw)?;
        gpu.silu_inplace(&mut cur)?;
        cur = self.conv1.apply(gpu, &cur, batch, h, w)?;
        let mut cur = self.norm2.apply(gpu, &cur, batch * self.conv1.out_c, hw)?;
        gpu.silu_inplace(&mut cur)?;
        let mut cur = self.conv2.apply(gpu, &cur, batch, h, w)?;
        gpu.add_inplace(&mut cur, &residual)?;
        Ok(cur)
    }

    /// `vae.rs`'s `load_residual_block`: `conv_shortcut` is `nn.Identity` when
    /// the channel count is unchanged, so the checkpoint has no tensor for it.
    fn load(
        dev: &CudaDevice,
        file: &LbiFile,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Self, VaeGpuError> {
        let norm1 = GpuNorm::load(dev, file, &format!("{prefix}.norm1.gamma"), in_dim, false)?;
        let conv1 = GpuConv::load(dev, file, &format!("{prefix}.conv1"), out_dim, in_dim, 3, 1)?;
        let norm2 = GpuNorm::load(dev, file, &format!("{prefix}.norm2.gamma"), out_dim, false)?;
        let conv2 = GpuConv::load(
            dev,
            file,
            &format!("{prefix}.conv2"),
            out_dim,
            out_dim,
            3,
            1,
        )?;
        let shortcut = if in_dim != out_dim {
            Some(GpuConv::load(
                dev,
                file,
                &format!("{prefix}.conv_shortcut"),
                out_dim,
                in_dim,
                1,
                0,
            )?)
        } else {
            None
        };
        Ok(Self {
            norm1,
            conv1,
            norm2,
            conv2,
            shortcut,
        })
    }
}

/// `QwenImage21AttentionBlock`. The projections stay 2-D convolutions here even
/// though they are pointwise: a 1x1 convolution is exact over the flattened
/// spatial plane, so the projections need no separate GEMM path and no
/// activation changes layout between them and the attention kernel.
struct GpuAttn {
    norm: GpuNorm,
    to_qkv: GpuConv,
    proj: GpuConv,
}

impl GpuAttn {
    fn apply(
        &self,
        gpu: &VaeGpu,
        x: &DevVec,
        batch: usize,
        c: usize,
        h: usize,
        w: usize,
    ) -> Result<DevVec, VaeGpuError> {
        let seq = h * w;
        // Note the absence of an activation after the norm: the reference's
        // attention block normalises and then goes straight into `to_qkv`.
        let normed = self.norm.apply(gpu, x, batch * c, seq)?;
        // `to_qkv` emits `[batch, 3C, h, w]`; because its kernel is 1x1 that is
        // already `[batch, 3C, seq]` with one plane per channel, which is what
        // the attention kernel splits into q, k and v.
        let qkv = self.to_qkv.apply(gpu, &normed, batch, h, w)?;
        let ctx = gpu.attention(&qkv, batch, seq, c)?;
        // `proj` and the residual add, the reference's `out.add_assign(x)`.
        let mut out = self.proj.apply(gpu, &ctx, batch, h, w)?;
        gpu.add_inplace(&mut out, x)?;
        Ok(out)
    }

    /// `vae.rs`'s `load_attention`.
    fn load(
        dev: &CudaDevice,
        file: &LbiFile,
        prefix: &str,
        dim: usize,
    ) -> Result<Self, VaeGpuError> {
        Ok(Self {
            norm: GpuNorm::load(dev, file, &format!("{prefix}.norm.gamma"), dim, true)?,
            to_qkv: GpuConv::load(dev, file, &format!("{prefix}.to_qkv"), dim * 3, dim, 1, 0)?,
            proj: GpuConv::load(dev, file, &format!("{prefix}.proj"), dim, dim, 1, 0)?,
        })
    }
}

/// `QwenImage21MidBlock`: `resnets[0]`, then each attention paired with the
/// next residual block.
struct GpuMidBlock {
    resnets: Vec<GpuResnet>,
    attentions: Vec<GpuAttn>,
}

impl GpuMidBlock {
    fn apply(
        &self,
        gpu: &VaeGpu,
        x: &DevVec,
        batch: usize,
        h: usize,
        w: usize,
    ) -> Result<DevVec, VaeGpuError> {
        let mut cur = self.resnets[0].apply(gpu, x, batch, h, w)?;
        for (attn, resnet) in self.attentions.iter().zip(&self.resnets[1..]) {
            let c = attn.to_qkv.in_c;
            cur = attn.apply(gpu, &cur, batch, c, h, w)?;
            cur = resnet.apply(gpu, &cur, batch, h, w)?;
        }
        Ok(cur)
    }
}

/// How a block's `avg_shortcut` (`QwenImage21DupUp3D`) is shaped.
#[derive(Debug, Clone, Copy)]
struct DupUp {
    out_c: usize,
    factor_t: usize,
    factor_s: usize,
    /// `out_channels * factor // in_channels`, the `repeat_interleave` count.
    repeats: usize,
}

impl DupUp {
    fn new(
        in_c: usize,
        out_c: usize,
        factor_t: usize,
        factor_s: usize,
    ) -> Result<Self, VaeGpuError> {
        let factor = factor_t * factor_s * factor_s;
        if in_c == 0 || factor == 0 || out_c * factor % in_c != 0 {
            return Err(VaeGpuError::Vae(VaeError::BadConfig(format!(
                "DupUp3D needs out_channels ({out_c}) * factor ({factor}) to be divisible by \
                 in_channels ({in_c})"
            ))));
        }
        Ok(Self {
            out_c,
            factor_t,
            factor_s,
            repeats: out_c * factor / in_c,
        })
    }
}

/// `QwenImage21ResidualUpBlock`: `num_res_blocks + 1` residual blocks, an
/// optional learned upsampler, and an optional parameter-free `DupUp3D`
/// shortcut added to the result.
struct GpuUpBlock {
    resnets: Vec<GpuResnet>,
    upsampler: Option<GpuConv>,
    shortcut: Option<DupUp>,
}

impl GpuUpBlock {
    /// `vae.rs`'s `ResidualUpBlock::apply`. The shortcut is built from the
    /// block's *input* (`dup_up_first_chunk(x, spec)`) and added to the block's
    /// output, which is why `x` is read again here after the resnets have
    /// finished with it.
    fn apply(
        &self,
        gpu: &VaeGpu,
        x: &DevVec,
        batch: usize,
        h: usize,
        w: usize,
    ) -> Result<DevVec, VaeGpuError> {
        let mut c = self.resnets[0].conv1.in_c;
        let mut cur = gpu.copy(x)?;
        for resnet in &self.resnets {
            cur = resnet.apply(gpu, &cur, batch, h, w)?;
            c = resnet.conv2.out_c;
        }
        if let Some(conv) = &self.upsampler {
            let up = gpu.nearest_2x(&cur, batch, c, h, w)?;
            cur = conv.apply(gpu, &up, batch, h * 2, w * 2)?;
        }
        if let Some(spec) = &self.shortcut {
            gpu.dup_up_add(&mut cur, x, batch, h, w, spec)?;
        }
        Ok(cur)
    }
}

/// The kernels in `vae_ops.cu`.
struct VaeOps {
    im2col: CudaFunction,
    bias_add: CudaFunction,
    nearest_2x: CudaFunction,
    dup_up: CudaFunction,
    rms_norm: CudaFunction,
    silu: CudaFunction,
    add: CudaFunction,
    softmax: CudaFunction,
    copy_rows: CudaFunction,
}

// ---------------------------------------------------------------------------
// The decoder
// ---------------------------------------------------------------------------

/// The Qwen-Image-2.1 VAE decoder, weights resident.
pub struct VaeGpu {
    /// The decoder drives one stream from host inputs end to end, so it holds
    /// its own device handle rather than borrowing the caller's.
    /// `CudaDevice::new` retains the same primary context, so this is a second
    /// stream on the same device, not a second device.
    dev: CudaDevice,
    ops: VaeOps,
    config: VaeConfig,
    post_quant_conv: GpuConv,
    conv_in: GpuConv,
    mid_block: GpuMidBlock,
    up_blocks: Vec<GpuUpBlock>,
    norm_out: GpuNorm,
    conv_out: GpuConv,
    /// The most bytes of attention scores held at once; see
    /// [`Self::set_attention_score_budget`].
    attention_score_budget: usize,
}

impl VaeGpu {
    /// Load the decoder half of a `vae` `.lbi` onto the device.
    ///
    /// The architecture is derived from the container's config (falling back to
    /// [`VaeConfig::qwen_image_2_1`] for anything the config omits), and every
    /// weight that architecture needs is looked up by name and shape-checked —
    /// as [`crate::vae::VaeDecoder::load`] does. Two groups of the checkpoint's
    /// tensors are deliberately not read: the `encoder.*`/`quant_conv.*` half,
    /// and `decoder.up_blocks.*.upsampler.time_conv.*`, whose `upsample3d`
    /// branch is never reached on the only frame a decode can process.
    pub fn load(lbi: &Path, dev: &CudaDevice) -> Result<Self, VaeGpuError> {
        Self::load_with(&LbiFile::open(lbi)?, dev, None)
    }

    /// Load from an open container, against an explicit architecture for a
    /// caller that does not want the container's own config overlaid on the
    /// shipped defaults.
    pub fn load_with(
        file: &LbiFile,
        dev: &CudaDevice,
        config: Option<VaeConfig>,
    ) -> Result<Self, VaeGpuError> {
        let own = CudaDevice::new(dev.ctx.ordinal())?;
        let ops = load_ops(&own)?;

        let config = match config {
            Some(c) => c,
            None => VaeConfig::from_lbi_config(file.config())?,
        };
        config.validate()?;

        let dims = config.decoder_dims();
        let temperal_upsample = config.temperal_upsample();
        let z = config.z_dim;

        let post_quant_conv = GpuConv::load(&own, file, "post_quant_conv", z, z, 1, 0)?;
        let conv_in = GpuConv::load(&own, file, "decoder.conv_in", dims[0], z, 3, 1)?;

        let mut resnets = Vec::with_capacity(MID_BLOCK_NUM_LAYERS + 1);
        let mut attentions = Vec::with_capacity(MID_BLOCK_NUM_LAYERS);
        for i in 0..=MID_BLOCK_NUM_LAYERS {
            resnets.push(GpuResnet::load(
                &own,
                file,
                &format!("decoder.mid_block.resnets.{i}"),
                dims[0],
                dims[0],
            )?);
        }
        for i in 0..MID_BLOCK_NUM_LAYERS {
            attentions.push(GpuAttn::load(
                &own,
                file,
                &format!("decoder.mid_block.attentions.{i}"),
                dims[0],
            )?);
        }
        let mid_block = GpuMidBlock {
            resnets,
            attentions,
        };

        let last_block = config.dim_mult.len() - 1;
        let mut up_blocks = Vec::with_capacity(dims.len() - 1);
        for i in 0..dims.len() - 1 {
            let (in_dim, out_dim) = (dims[i], dims[i + 1]);
            let prefix = format!("decoder.up_blocks.{i}");
            // `up_flag = i != len(dim_mult) - 1`.
            let up_flag = i != last_block;
            let temporal = up_flag && temperal_upsample[i];

            let mut blocks = Vec::with_capacity(config.num_res_blocks + 1);
            let mut current = in_dim;
            for j in 0..=config.num_res_blocks {
                blocks.push(GpuResnet::load(
                    &own,
                    file,
                    &format!("{prefix}.resnets.{j}"),
                    current,
                    out_dim,
                )?);
                current = out_dim;
            }

            let upsampler = if up_flag {
                // `QwenImage21Resample` wraps the interpolation and the
                // convolution in an `nn.Sequential`, so the convolution is
                // member `1`.
                Some(GpuConv::load(
                    &own,
                    file,
                    &format!("{prefix}.upsampler.resample.1"),
                    out_dim,
                    out_dim,
                    3,
                    1,
                )?)
            } else {
                None
            };
            let shortcut = if up_flag {
                Some(DupUp::new(
                    in_dim,
                    out_dim,
                    if temporal { 2 } else { 1 },
                    2,
                )?)
            } else {
                None
            };

            up_blocks.push(GpuUpBlock {
                resnets: blocks,
                upsampler,
                shortcut,
            });
        }

        let head_dim = *dims.last().expect("dims has dim_mult.len() + 1 entries");
        let norm_out = GpuNorm::load(&own, file, "decoder.norm_out.gamma", head_dim, false)?;
        let conv_out = GpuConv::load(
            &own,
            file,
            "decoder.conv_out",
            config.out_channels,
            head_dim,
            3,
            1,
        )?;

        Ok(Self {
            dev: own,
            ops,
            config,
            post_quant_conv,
            conv_in,
            mid_block,
            up_blocks,
            norm_out,
            conv_out,
            attention_score_budget: ATTN_SCORE_BUDGET,
        })
    }

    pub fn config(&self) -> &VaeConfig {
        &self.config
    }

    /// Hold at most `bytes` of attention scores at once (at least one row's):
    /// the mid block's queries run in chunks of rows whose scores fit. Each
    /// row's softmax and product with V stand alone, so any budget gives the
    /// same values up to the order the products' terms are summed in;
    /// `vae-check-gpu` sets a small one to exercise the chunking on latents
    /// whose scores would otherwise fit whole.
    pub fn set_attention_score_budget(&mut self, bytes: usize) {
        self.attention_score_budget = bytes;
    }

    /// Decode latents `[batch, z_dim, 1, h, w]` to `[batch, out_channels, 1,
    /// h * 16, w * 16]`, equal to [`crate::vae::VaeDecoder::decode`] on the same
    /// inputs, including the clamp to `[-1, 1]`.
    ///
    /// The temporal axis is folded away and stays folded, as in the reference:
    /// it is 1 at every point of a single-frame decode.
    pub fn decode(
        &self,
        latents: &[f32],
        batch: usize,
        h: usize,
        w: usize,
    ) -> Result<Vec<f32>, VaeGpuError> {
        if batch == 0 || h == 0 || w == 0 {
            return Err(mismatch(
                "latents",
                &[1, self.config.z_dim as u64, 1, 1, 1],
                &[
                    batch as u64,
                    self.config.z_dim as u64,
                    1,
                    h as u64,
                    w as u64,
                ],
            ));
        }
        let expected = batch * self.config.z_dim * h * w;
        if latents.len() != expected {
            return Err(mismatch(
                "latents",
                &[
                    batch as u64,
                    self.config.z_dim as u64,
                    1,
                    h as u64,
                    w as u64,
                ],
                &[latents.len() as u64],
            ));
        }

        self.decode_in_bands(latents, batch, h, w, 1)
    }

    /// [`Self::decode`] with the upsampling half run over `bands` horizontal
    /// bands of the latent rows instead of the whole image at once, so the
    /// largest activations are a band's size.
    ///
    /// Everything up to and including the mid block — the only operation that
    /// mixes distant positions, its attention — runs over the whole latent
    /// once. After it every operation is local: 3x3 and 1x1 convolutions with
    /// zero padding, the per-position channel norm, SiLU, nearest upsampling
    /// and the `DupUp3D` shortcut. So each band is decoded with
    /// [`Self::halo_rows`] extra latent rows on either side, which covers every
    /// convolution's reach, and only its own rows are kept: each kept value is
    /// computed from exactly the inputs the whole-image decode uses, with the
    /// image border where the image's is. `bands` is clamped to the latent's
    /// row count.
    pub fn decode_in_bands(
        &self,
        latents: &[f32],
        batch: usize,
        h: usize,
        w: usize,
        bands: usize,
    ) -> Result<Vec<f32>, VaeGpuError> {
        if batch == 0 || h == 0 || w == 0 {
            return Err(mismatch(
                "latents",
                &[1, self.config.z_dim as u64, 1, 1, 1],
                &[
                    batch as u64,
                    self.config.z_dim as u64,
                    1,
                    h as u64,
                    w as u64,
                ],
            ));
        }
        let expected = batch * self.config.z_dim * h * w;
        if latents.len() != expected {
            return Err(mismatch(
                "latents",
                &[
                    batch as u64,
                    self.config.z_dim as u64,
                    1,
                    h as u64,
                    w as u64,
                ],
                &[latents.len() as u64],
            ));
        }

        // `_decode` runs `post_quant_conv` over the whole latent, then feeds the
        // decoder one frame at a time with `first_chunk=True` on frame 0. There
        // is exactly one frame here, so that loop runs once.
        let z = launch::upload(&self.dev, latents)?;
        let x = self.post_quant_conv.apply(self, &z, batch, h, w)?;
        drop(z);
        let x = self.conv_in.apply(self, &x, batch, h, w)?;
        let x = self.mid_block.apply(self, &x, batch, h, w)?;

        // The elementwise kernels index an activation with 32 bits, so a band,
        // context rows included, never holds more than `i32::MAX` values in any
        // one activation.
        let row_limit = i32::MAX as usize / (batch * self.widest_row(w));
        let halo = self.halo_rows();
        let fewest = if h <= row_limit {
            1
        } else if row_limit > 2 * halo {
            h.div_ceil(row_limit - 2 * halo)
        } else {
            return Err(VaeGpuError::KernelLimit(format!(
                "a {w}-wide latent in batches of {batch} leaves {row_limit} rows per band, \
                 not above the {} of context a band carries",
                2 * halo
            )));
        };
        let bands = bands.max(fewest).clamp(1, h);
        let out = if bands == 1 {
            self.upsample_half(x, batch, h, w)?
        } else {
            let scale = self.scale();
            let c_mid = x.len / (batch * h * w);
            let c_out = self.conv_out.out_c;
            let mut out = launch::alloc(&self.dev, batch * c_out * h * scale * w * scale)?;
            let per_band = h.div_ceil(bands);
            for r0 in (0..h).step_by(per_band) {
                let r1 = (r0 + per_band).min(h);
                let (a, b) = (r0.saturating_sub(halo), (r1 + halo).min(h));
                let mut band = launch::alloc(&self.dev, batch * c_mid * (b - a) * w)?;
                self.copy_rows(
                    &x.buf.slice(..),
                    &mut band.buf.slice_mut(..),
                    batch * c_mid,
                    w,
                    (h, a),
                    (b - a, 0),
                    b - a,
                )?;
                let decoded = self.upsample_half(band, batch, b - a, w)?;
                self.copy_rows(
                    &decoded.buf.slice(..),
                    &mut out.buf.slice_mut(..),
                    batch * c_out,
                    w * scale,
                    ((b - a) * scale, (r0 - a) * scale),
                    (h * scale, r0 * scale),
                    (r1 - r0) * scale,
                )?;
            }
            out
        };

        // The clamp is the reference's `v.max(-1.0).min(1.0)` on the finished
        // tensor; `download` synchronizes first, so the copy is complete.
        let mut data = launch::download(&self.dev, &out)?;
        for v in data.iter_mut() {
            *v = v.max(-1.0).min(1.0);
        }
        Ok(data)
    }

    /// The decoder after the mid block, over `[batch, c, h, w]`: the up blocks,
    /// `norm_out`, SiLU and `conv_out`, to `[batch, 3, h * scale, w * scale]`.
    fn upsample_half(
        &self,
        mut x: DevVec,
        batch: usize,
        h: usize,
        w: usize,
    ) -> Result<DevVec, VaeGpuError> {
        // The spatial extent is carried through the stack because the
        // upsamplers double it. Each layer checks the activation it is handed
        // against the extent it is about to use, so a size that went out of
        // step with the data is a reported shape mismatch rather than a read
        // past the end of a plane.
        let (mut hh, mut ww) = (h, w);
        for block in &self.up_blocks {
            x = block.apply(self, &x, batch, hh, ww)?;
            if block.upsampler.is_some() {
                hh *= 2;
                ww *= 2;
            }
        }
        self.check_extent(&x, batch, self.norm_out.dim, hh, ww, "decoder.norm_out")?;
        let mut x = self
            .norm_out
            .apply(self, &x, batch * self.norm_out.dim, hh * ww)?;
        self.silu_inplace(&mut x)?;
        self.check_extent(&x, batch, self.norm_out.dim, hh, ww, "decoder.conv_out")?;
        self.conv_out.apply(self, &x, batch, hh, ww)
    }

    /// Values per latent row in the largest activation after the mid block, for
    /// a latent `w` wide: each activation's channels times the rows and columns
    /// one latent row becomes at its resolution.
    fn widest_row(&self, w: usize) -> usize {
        let mut scale = 1usize;
        let mut widest = 0usize;
        for block in &self.up_blocks {
            for resnet in &block.resnets {
                let channels = resnet.conv1.in_c.max(resnet.conv2.out_c);
                widest = widest.max(channels * scale * scale * w);
            }
            if let Some(conv) = &block.upsampler {
                scale *= 2;
                widest = widest.max(conv.in_c.max(conv.out_c) * scale * scale * w);
            }
        }
        widest.max(self.norm_out.dim * scale * scale * w)
    }

    /// How many times the up blocks double the latent's extent.
    fn scale(&self) -> usize {
        1 << self
            .up_blocks
            .iter()
            .filter(|b| b.upsampler.is_some())
            .count()
    }

    /// Latent rows of context a band needs on either side: the reach of every
    /// convolution after the mid block, each padding row counted at the
    /// resolution its convolution runs at and converted to latent rows.
    pub fn halo_rows(&self) -> usize {
        let scale = self.scale();
        // Reach in rows of the finished image, summed over the convolutions.
        let mut reach = 0usize;
        let mut level = 1usize;
        for block in &self.up_blocks {
            for resnet in &block.resnets {
                let convs = [
                    Some(&resnet.conv1),
                    Some(&resnet.conv2),
                    resnet.shortcut.as_ref(),
                ];
                reach += convs.iter().flatten().map(|c| c.pad_h).sum::<usize>() * (scale / level);
            }
            if let Some(conv) = &block.upsampler {
                level *= 2;
                reach += conv.pad_h * (scale / level);
            }
        }
        reach += self.conv_out.pad_h;
        reach.div_ceil(scale)
    }

    /// Assert that `x` holds `batch * c * h * w` values, naming the layer.
    ///
    /// The decoder's own op wrappers check every activation they touch against
    /// the extent they are given, so this is a restatement at the two points
    /// where the extent changes hands rather than a substitute for them.
    fn check_extent(
        &self,
        x: &DevVec,
        batch: usize,
        c: usize,
        h: usize,
        w: usize,
        what: &str,
    ) -> Result<(), VaeGpuError> {
        let want = (batch as u64) * (c as u64) * (h as u64) * (w as u64);
        if x.len as u64 != want {
            return Err(mismatch(what, &[want], &[x.len as u64]));
        }
        Ok(())
    }

    // -- op wrappers ---------------------------------------------------------

    /// One 2-D convolution, `[batch, in_c, h, w]` to `[batch, out_c, h, w]`,
    /// as im2col plus a GEMM.
    ///
    /// The output rows are processed in bands of a fixed row count so the
    /// column matrix stays bounded: at 1024x1024 with 144 channels and a 3x3
    /// kernel the full matrix would be 5.4 GB. Each band's GEMM writes a
    /// staging buffer `[out_c, band_rows * w]` (`ldc = band_rows * w`), and the
    /// band's rows are then copied into `out`, `[out_c, h * w]` per image.
    ///
    /// Column-major mapping for the row-major buffers: `col[K, P]` is
    /// `col_cm[P, K]`, `W[out_c, K]` is `W_cm[K, out_c]`, and the wanted
    /// `out[out_c, P] = W · col` is `out_cm[P, out_c] = col_cm · W_cm`, so both
    /// operands are un-transposed with `(m, n, k) = (P, out_c, K)`.
    fn conv(
        &self,
        w: &GpuConv,
        x: &DevVec,
        batch: usize,
        h: usize,
        wd: usize,
    ) -> Result<DevVec, VaeGpuError> {
        // Every convolution in this decoder preserves the spatial extent. If one
        // ever did not, the caller's (h, w) — handed unchanged to the next op —
        // would be silently wrong, so it is checked rather than assumed.
        // `checked_sub` makes a kernel larger than its padded input an error
        // rather than an underflow, which is what `vae.rs` asserts against.
        let oh = (h + 2 * w.pad_h + 1).checked_sub(w.kh);
        let ow = (wd + 2 * w.pad_w + 1).checked_sub(w.kw);
        if oh != Some(h) || ow != Some(wd) {
            return Err(VaeGpuError::KernelLimit(format!(
                "a {}x{} convolution with padding ({}, {}) maps a {h}x{wd} input to \
                 {oh:?}x{ow:?}, but the decoder's activations keep their spatial extent",
                w.kh, w.kw, w.pad_h, w.pad_w
            )));
        }
        if x.len != batch * w.in_c * h * wd {
            return Err(mismatch(
                "convolution input",
                &[(batch * w.in_c * h * wd) as u64],
                &[x.len as u64],
            ));
        }
        let hw = h * wd;
        let k = w.in_c * w.kh * w.kw;
        for (name, v) in [("out_c", w.out_c), ("hw", hw), ("k", k)] {
            if v > i32::MAX as usize {
                return Err(VaeGpuError::KernelLimit(format!(
                    "convolution {name}={v} exceeds the i32 cuBLAS limit"
                )));
            }
        }
        // Uninitialized: every element is written by a band's GEMM.
        let mut out = DevVec {
            buf: unsafe { self.dev.alloc_uninit::<f32>(batch * w.out_c * hw) }
                .map_err(|e| cuda("conv out alloc", e))?,
            len: batch * w.out_c * hw,
        };
        if batch * w.out_c * hw == 0 {
            return Ok(out);
        }
        // Rows per product: CONV_ROWS, or fewer when a row is so wide the
        // column matrix would pass its budget. The count depends only on the
        // width, and every product writes the same `staging` buffer with the
        // same stride before its rows are copied into place, so the same row
        // of an image goes through the same cuBLAS call — shape, operands'
        // alignment and strides — whether the image is decoded whole or in
        // bands, and is summed in the same order. The last band of rows is
        // padded to the full shape.
        let band_rows = (CONV_COL_BUDGET / (k * wd * 4)).clamp(1, CONV_ROWS);
        let col = DevVec {
            buf: unsafe { self.dev.alloc_uninit::<f32>(k * band_rows * wd) }
                .map_err(|e| cuda("conv col alloc", e))?,
            len: k * band_rows * wd,
        };
        let mut staging = launch::alloc(&self.dev, w.out_c * band_rows * wd)?;

        let (icu, ihu, iwu) = (w.in_c as u32, h as u32, wd as u32);
        let (khu, kwu, phu, pwu) = (w.kh as u32, w.kw as u32, w.pad_h as u32, w.pad_w as u32);
        for b in 0..batch {
            let x_img = x.buf.slice(b * w.in_c * hw..(b + 1) * w.in_c * hw);
            let mut row0 = 0usize;
            while row0 < h {
                let rows = band_rows.min(h - row0);
                // Always a full product; rows past the image are computed
                // from whatever the kernel window reads and dropped.
                let p = band_rows * wd;
                let total = (k * p) as u64;
                // Safety: `col` holds `k * band_rows * wd` floats and `x_img` is
                // one image's `[in_c, h, w]`; the grid covers exactly `k * p`.
                unsafe {
                    self.dev
                        .stream
                        .launch_builder(&self.ops.im2col)
                        .arg(&x_img)
                        .arg(&col.buf)
                        .arg(&icu)
                        .arg(&ihu)
                        .arg(&iwu)
                        .arg(&khu)
                        .arg(&kwu)
                        .arg(&phu)
                        .arg(&pwu)
                        .arg(&(row0 as u32))
                        .arg(&(band_rows as u32))
                        .launch(cfg_1d(total, THREADS))
                        .map_err(|e| cuda("im2col_band", e))?;
                }
                // Safety: `col` holds the band's `k * p` columns, the weight is
                // `[out_c, k]` and `staging` holds `out_c * p`.
                unsafe {
                    sgemm(
                        &self.dev,
                        Math::Tf32,
                        Op::N,
                        Op::N,
                        1.0,
                        &col.buf.slice(..),
                        p,
                        &w.weight.buf.slice(..),
                        k,
                        &mut staging.buf.slice_mut(..),
                        p,
                        p,
                        w.out_c,
                        k,
                    )
                    .map_err(VaeGpuError::Cuda)?;
                }
                let mut image = out.buf.slice_mut(b * w.out_c * hw..(b + 1) * w.out_c * hw);
                self.copy_rows(
                    &staging.buf.slice(..),
                    &mut image,
                    w.out_c,
                    wd,
                    (band_rows, 0),
                    (h, row0),
                    rows,
                )?;
                row0 += rows;
            }
            // Safety: the plane is `out_c * hw` floats and `bias` holds `out_c`.
            let out_img = out.buf.slice(b * w.out_c * hw..(b + 1) * w.out_c * hw);
            unsafe {
                self.dev
                    .stream
                    .launch_builder(&self.ops.bias_add)
                    .arg(&out_img)
                    .arg(&w.bias.buf)
                    .arg(&(w.out_c as u32))
                    .arg(&(hw as u32))
                    .launch(cfg_1d((w.out_c * hw) as u64, THREADS))
                    .map_err(|e| cuda("bias_add_channels", e))?;
            }
        }
        Ok(out)
    }

    /// `F.normalize` over the channel axis, times `sqrt(C)`, times `gamma`.
    ///
    /// `rows` is `batch * channels`, so the kernel can find the batch's channel
    /// zero from any element; `hw` is the spatial plane.
    fn rms_norm(
        &self,
        norm: &GpuNorm,
        x: &DevVec,
        rows: usize,
        hw: usize,
    ) -> Result<DevVec, VaeGpuError> {
        // The channel reduction reads `rows / C` batches of `C` channel rows, so
        // a row count that is not a multiple of the width would silently read
        // another batch's rows as this one's channels.
        if rows % norm.dim != 0 {
            return Err(mismatch(
                "rms_norm rows (must be a multiple of the channel width)",
                &[norm.dim as u64],
                &[rows as u64],
            ));
        }
        if x.len != rows * hw {
            return Err(mismatch(
                "rms_norm input",
                &[(rows * hw) as u64],
                &[x.len as u64],
            ));
        }
        let out = launch::alloc(&self.dev, x.len)?;
        let positions = (rows / norm.dim) as u64 * (hw as u64);
        if positions == 0 {
            return Ok(out);
        }
        let (cu, hwu, ru) = (norm.dim as u32, hw as u32, rows as u32);
        // Safety: the buffers are device allocations of the sizes the kernel
        // reads, and the grid covers exactly the spatial-position count.
        unsafe {
            self.dev
                .stream
                .launch_builder(&self.ops.rms_norm)
                .arg(&x.buf)
                .arg(&norm.gamma.buf)
                .arg(&out.buf)
                .arg(&cu)
                .arg(&hwu)
                .arg(&ru)
                .arg(&NORMALIZE_EPS)
                .launch(cfg_1d(positions, THREADS))
                .map_err(|e| cuda("rms_norm_channels", e))?;
        }
        Ok(out)
    }

    /// SiLU in place, the reference's `silu_in_place`.
    fn silu_inplace(&self, x: &mut DevVec) -> Result<(), VaeGpuError> {
        let n = x.len as u32;
        if n == 0 {
            return Ok(());
        }
        // Safety: the buffer is `n` floats and the grid covers exactly that.
        unsafe {
            self.dev
                .stream
                .launch_builder(&self.ops.silu)
                .arg(&mut x.buf)
                .arg(&n)
                .launch(cfg_1d(x.len as u64, THREADS))
                .map_err(|e| cuda("silu_inplace", e))?;
        }
        Ok(())
    }

    /// `dst += src`, the reference's `Tensor4::add_assign`.
    fn add_inplace(&self, dst: &mut DevVec, src: &DevVec) -> Result<(), VaeGpuError> {
        if dst.len != src.len {
            return Err(mismatch("add", &[dst.len as u64], &[src.len as u64]));
        }
        let n = dst.len as u32;
        if n == 0 {
            return Ok(());
        }
        // Safety: both buffers are `n` floats and the grid covers exactly that.
        unsafe {
            self.dev
                .stream
                .launch_builder(&self.ops.add)
                .arg(&mut dst.buf)
                .arg(&src.buf)
                .arg(&n)
                .launch(cfg_1d(dst.len as u64, THREADS))
                .map_err(|e| cuda("add_inplace", e))?;
        }
        Ok(())
    }

    /// `QwenImage21Upsample(scale_factor=(2, 2), mode="nearest-exact")`.
    fn nearest_2x(
        &self,
        x: &DevVec,
        batch: usize,
        c: usize,
        h: usize,
        w: usize,
    ) -> Result<DevVec, VaeGpuError> {
        if x.len != batch * c * h * w {
            return Err(mismatch(
                "upsample input",
                &[(batch * c * h * w) as u64],
                &[x.len as u64],
            ));
        }
        let out = launch::alloc(&self.dev, batch * c * h * 2 * w * 2)?;
        let total = (batch as u64) * (c as u64) * (2 * h as u64) * (2 * w as u64);
        if total == 0 {
            return Ok(out);
        }
        let (ncu, hu, wu) = ((batch * c) as u32, h as u32, w as u32);
        // Safety: the buffers are device allocations of the sizes the kernel
        // reads, and the grid covers exactly the output element count.
        unsafe {
            self.dev
                .stream
                .launch_builder(&self.ops.nearest_2x)
                .arg(&x.buf)
                .arg(&out.buf)
                .arg(&ncu)
                .arg(&hu)
                .arg(&wu)
                .launch(cfg_1d(total, THREADS))
                .map_err(|e| cuda("nearest_2x", e))?;
        }
        Ok(out)
    }

    /// The reference's `dup_up_first_chunk`, accumulated into `dst`.
    ///
    /// `dst` holds the block output at the *upsampled* size and `src` the
    /// block's input, the operand order the reference's
    /// `h.add_assign(&dup_up_first_chunk(x, spec))` uses. The kernel derives the
    /// source channel from `j / repeats`, so the channel count is checked rather
    /// than passed: `src.len * repeats == dst.len * factor_t` is the same
    /// statement as `dst.len = batch * out_c * (h * fs) * (w * fs)`, and a
    /// mismatch either way is a bug here rather than in a checkpoint.
    fn dup_up_add(
        &self,
        dst: &mut DevVec,
        src: &DevVec,
        batch: usize,
        h: usize,
        w: usize,
        spec: &DupUp,
    ) -> Result<(), VaeGpuError> {
        let out_h = h * spec.factor_s;
        let out_w = w * spec.factor_s;
        let plane = batch * h * w;
        if plane == 0 || src.len % plane != 0 {
            return Err(mismatch(
                "dup_up source",
                &[(batch * h * w) as u64],
                &[src.len as u64],
            ));
        }
        let in_c = src.len / plane;
        if dst.len != batch * spec.out_c * out_h * out_w {
            return Err(mismatch(
                "dup_up destination",
                &[(batch * spec.out_c * out_h * out_w) as u64],
                &[dst.len as u64],
            ));
        }
        if src.len * spec.repeats != dst.len * spec.factor_t {
            return Err(mismatch(
                "dup_up channels",
                &[(dst.len * spec.factor_t / spec.repeats.max(1)) as u64],
                &[src.len as u64],
            ));
        }
        let total = (batch as u64) * (spec.out_c as u64) * (out_h as u64) * (out_w as u64);
        if total == 0 {
            return Ok(());
        }
        let (nu, icu, hu, wu) = (batch as u32, in_c as u32, h as u32, w as u32);
        let (ocu, ftu, fsu) = (
            spec.out_c as u32,
            spec.factor_t as u32,
            spec.factor_s as u32,
        );
        let repu = spec.repeats as u32;
        // Safety: both buffers are device allocations of the sizes the kernel
        // reads, and the grid covers exactly the destination element count.
        unsafe {
            self.dev
                .stream
                .launch_builder(&self.ops.dup_up)
                .arg(&src.buf)
                .arg(&mut dst.buf)
                .arg(&nu)
                .arg(&icu)
                .arg(&hu)
                .arg(&wu)
                .arg(&ocu)
                .arg(&ftu)
                .arg(&fsu)
                .arg(&repu)
                .launch(cfg_1d(total, THREADS))
                .map_err(|e| cuda("dup_up_first_chunk", e))?;
        }
        Ok(())
    }

    /// `ctx = softmax(Q K^T / sqrt(C)) V` over the spatial positions.
    ///
    /// `qkv` is the projections' channel-major output `[batch, 3C, seq]`,
    /// `[q | k | v]` stacked on the channel axis, so each of the three is a
    /// `[C, seq]` row-major plane. Read as column-major each is `[seq, C]` with
    /// leading dimension `seq`, which is exactly the operand shape the two
    /// products want: the scores `S[t, j] = q[:, t] · k[:, j]` are, as the
    /// column-major `Sᵀ`, `k_cm · q_cmᵀ`; and the context `ctx[c, t] = Σ_j
    /// P[t, j] v[c, j]` is, as the column-major `ctxᵀ`, `P_cmᵀ · v_cm` with
    /// `P_cm` the row-major probability matrix read column-major. Both are one
    /// SGEMM per batch item, with the row softmax in between.
    fn attention(
        &self,
        qkv: &DevVec,
        batch: usize,
        seq: usize,
        c: usize,
    ) -> Result<DevVec, VaeGpuError> {
        if qkv.len != batch * 3 * c * seq {
            return Err(mismatch(
                "attention qkv",
                &[(batch * 3 * c * seq) as u64],
                &[qkv.len as u64],
            ));
        }
        if seq == 0 || c == 0 || batch == 0 {
            return Ok(launch::alloc(&self.dev, batch * c * seq)?);
        }
        if seq as u64 > GRID_X_MAX {
            return Err(VaeGpuError::KernelLimit(format!(
                "softmax_rows_f32 grid {seq} exceeds the {GRID_X_MAX} limit"
            )));
        }
        let mut ctx = launch::alloc(&self.dev, batch * c * seq)?;
        // The scores are `seq` rows of `seq`, one per query position, and each
        // row's softmax and product with V stand alone, so the queries run in
        // chunks of rows whose scores fit the budget, by default
        // [`ATTN_SCORE_BUDGET`]: the whole of
        // them up to 2048x2048 (64 MiB at 1024x1024's 4096 positions), and 1 GiB
        // at a time beyond, where all of them would take 16 GiB at 4096x4096.
        let chunk = (self.attention_score_budget / (seq * 4)).clamp(1, seq);
        let mut row = launch::alloc(&self.dev, chunk * seq)?;
        let scale = 1.0 / (c as f32).sqrt();
        let su = seq as u32;
        for (n, q0) in (0..batch).flat_map(|n| (0..seq).step_by(chunk).map(move |q0| (n, q0))) {
            let rows = chunk.min(seq - q0);
            let ru = rows as u32;
            let q = qkv.buf.slice(n * 3 * c * seq + q0..);
            let k = qkv.buf.slice((n * 3 + 1) * c * seq..);
            let v = qkv.buf.slice((n * 3 + 2) * c * seq..);
            let mut scores = row.buf.slice_mut(..rows * seq);
            // Safety: every operand is a device view of at least the size the
            // product reads or writes, checked against `qkv.len` above.
            unsafe {
                sgemm(
                    &self.dev,
                    Math::F32,
                    Op::N,
                    Op::T,
                    scale,
                    &k,
                    seq,
                    &q,
                    seq,
                    &mut scores,
                    seq,
                    seq,
                    rows,
                    c,
                )
                .map_err(VaeGpuError::Cuda)?;
            }
            // Safety: the grid is one block per row of this chunk's `[rows,
            // seq]` scores, and every block runs to the end.
            unsafe {
                self.dev
                    .stream
                    .launch_builder(&self.ops.softmax)
                    .arg(&mut scores)
                    .arg(&su)
                    .launch(LaunchConfig {
                        grid_dim: (ru, 1, 1),
                        block_dim: (ATTN_THREADS, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .map_err(|e| cuda("softmax_rows_f32", e))?;
            }
            let probs = row.buf.slice(..rows * seq);
            let mut out = ctx.buf.slice_mut(n * c * seq + q0..(n + 1) * c * seq);
            // Safety: as for the first product.
            unsafe {
                sgemm(
                    &self.dev,
                    Math::F32,
                    Op::T,
                    Op::N,
                    1.0,
                    &probs,
                    seq,
                    &v,
                    seq,
                    &mut out,
                    seq,
                    rows,
                    c,
                    seq,
                )
                .map_err(VaeGpuError::Cuda)?;
            }
        }
        Ok(ctx)
    }

    /// `rows` rows of every one of `planes` planes `w` wide, from `src`
    /// (`(height, first row)`) to `dst` (`(height, first row)`).
    #[allow(clippy::too_many_arguments)]
    fn copy_rows(
        &self,
        src: &cudarc::driver::CudaView<f32>,
        dst: &mut cudarc::driver::CudaViewMut<f32>,
        planes: usize,
        w: usize,
        (src_h, src_row0): (usize, usize),
        (dst_h, dst_row0): (usize, usize),
        rows: usize,
    ) -> Result<(), VaeGpuError> {
        if src_row0 + rows > src_h
            || dst_row0 + rows > dst_h
            || src.len() != planes * src_h * w
            || dst.len() != planes * dst_h * w
        {
            return Err(mismatch(
                "band rows",
                &[(planes * src_h * w) as u64, (planes * dst_h * w) as u64],
                &[src.len() as u64, dst.len() as u64],
            ));
        }
        let total = (planes as u64) * (rows as u64) * (w as u64);
        if total == 0 {
            return Ok(());
        }
        let args = [planes, w, src_h, src_row0, dst_h, dst_row0, rows].map(|v| v as u32);
        // Safety: both views hold `planes` planes of the heights checked
        // above, and the grid covers exactly the copied element count.
        unsafe {
            let mut launch = self.dev.stream.launch_builder(&self.ops.copy_rows);
            launch.arg(src).arg(dst);
            for v in &args {
                launch.arg(v);
            }
            launch
                .launch(cfg_1d(total, THREADS))
                .map_err(|e| cuda("copy_rows", e))?;
        }
        Ok(())
    }

    /// A device-to-device copy, for a residual path that keeps its operand.
    fn copy(&self, src: &DevVec) -> Result<DevVec, VaeGpuError> {
        let mut dst = launch::alloc(&self.dev, src.len)?;
        self.dev
            .stream
            .memcpy_dtod(&src.buf, &mut dst.buf)
            .map_err(|e| cuda("memcpy_dtod", e))?;
        Ok(dst)
    }
}

// ---------------------------------------------------------------------------
// Loading helpers
// ---------------------------------------------------------------------------

/// The tensor `name` at exactly `shape`, uploaded in f32.
///
/// The reference's default is f32: everything it reads goes through
/// `LbiFile::read_f32`, so a 16-bit weight is widened before use. Widening it on
/// the host and uploading f32 is the same value the reference computes with, and
/// it keeps every kernel here on one dtype — see this module's doc comment.
fn weight_exact(
    dev: &CudaDevice,
    file: &LbiFile,
    name: &str,
    shape: &[usize],
) -> Result<DevVec, VaeGpuError> {
    let want: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
    let entry = file
        .get(name)
        .ok_or_else(|| VaeGpuError::Vae(VaeError::MissingTensor(name.to_string())))?;
    if entry.shape != want {
        return Err(mismatch(name, &want, &entry.shape));
    }
    read_f32_tensor(dev, file, name)
}

/// A tensor decoded to f32 by the container's own reader, then uploaded.
fn read_f32_tensor(dev: &CudaDevice, file: &LbiFile, name: &str) -> Result<DevVec, VaeGpuError> {
    let entry = file
        .get(name)
        .ok_or_else(|| VaeGpuError::Vae(VaeError::MissingTensor(name.to_string())))?;
    match entry.quant {
        QuantScheme::F32 | QuantScheme::F16 | QuantScheme::Bf16 => {
            Ok(launch::upload(dev, &file.read_f32(name)?)?)
        }
        other => Err(VaeGpuError::UnsupportedStorage {
            tensor: name.to_string(),
            scheme: format!("{other:?}"),
        }),
    }
}

/// A cuBLAS operand transpose.
#[derive(Clone, Copy)]
enum Op {
    N,
    T,
}

impl Op {
    fn cublas(self) -> cudarc::cublas::sys::cublasOperation_t {
        match self {
            Op::N => cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            Op::T => cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
        }
    }
}

/// How an f32 GEMM multiplies, matching what the reference's operation does
/// in f32 under torch's defaults.
#[derive(Clone, Copy)]
enum Math {
    /// TF32 tensor-core products with f32 accumulation: a convolution, which
    /// cuDNN runs this way while `torch.backends.cudnn.allow_tf32` is true,
    /// its default.
    Tf32,
    /// Full f32 products: the attention, whose f32 kernel in the reference
    /// computes at full f32 accuracy.
    F32,
}

/// f32 `C_cm[m, n] = alpha · op(A_cm) · op(B_cm)`, `op(A)` being `[m, k]` and
/// `op(B)` `[k, n]`, with the products computed as `math` says.
///
/// # Safety
///
/// Each operand must hold, from its start, the column-major matrix its op,
/// leading dimension and role describe; `c` must hold `ldc * n` floats.
#[allow(clippy::too_many_arguments)]
unsafe fn sgemm(
    dev: &CudaDevice,
    math: Math,
    op_a: Op,
    op_b: Op,
    alpha: f32,
    a: &cudarc::driver::CudaView<f32>,
    lda: usize,
    b: &cudarc::driver::CudaView<f32>,
    ldb: usize,
    c: &mut cudarc::driver::CudaViewMut<f32>,
    ldc: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), RuntimeError> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    let beta: f32 = 0.0;
    // The guards live until the call has been issued on the stream.
    let (a_ptr, _a_guard) = a.device_ptr(&dev.stream);
    let (b_ptr, _b_guard) = b.device_ptr(&dev.stream);
    let (c_ptr, _c_guard) = c.device_ptr_mut(&dev.stream);
    let status = cudarc::cublas::sys::cublasGemmEx(
        *dev.blas.handle(),
        op_a.cublas(),
        op_b.cublas(),
        m as i32,
        n as i32,
        k as i32,
        &alpha as *const f32 as *const std::ffi::c_void,
        a_ptr as *const std::ffi::c_void,
        cudarc::cublas::sys::cudaDataType_t::CUDA_R_32F,
        lda as i32,
        b_ptr as *const std::ffi::c_void,
        cudarc::cublas::sys::cudaDataType_t::CUDA_R_32F,
        ldb as i32,
        &beta as *const f32 as *const std::ffi::c_void,
        c_ptr as *mut std::ffi::c_void,
        cudarc::cublas::sys::cudaDataType_t::CUDA_R_32F,
        ldc as i32,
        match math {
            Math::Tf32 => cudarc::cublas::sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
            Math::F32 => cudarc::cublas::sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
        },
        cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
    );
    if status != cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(RuntimeError::Compute(format!(
            "sgemm [{m}x{n}x{k}]: status={status:?}"
        )));
    }
    Ok(())
}

/// Compile `vae_ops.cu` and resolve its entry points.
fn load_ops(dev: &CudaDevice) -> Result<VaeOps, RuntimeError> {
    let module: Arc<_> = dev.compile_and_load(VAE_OPS_SOURCE)?;
    let get = |name: &str| -> Result<CudaFunction, RuntimeError> {
        module
            .load_function(name)
            .map_err(|e| RuntimeError::Compute(format!("load {name}: {e}")))
    };
    Ok(VaeOps {
        im2col: get("im2col_band")?,
        bias_add: get("bias_add_channels")?,
        nearest_2x: get("nearest_2x")?,
        dup_up: get("dup_up_first_chunk")?,
        rms_norm: get("rms_norm_channels")?,
        silu: get("silu_inplace")?,
        add: get("add_inplace")?,
        softmax: get("softmax_rows_f32")?,
        copy_rows: get("copy_rows")?,
    })
}

/// One block of `threads` threads per `threads` elements, never an empty grid
/// (the driver rejects a zero grid dimension).
///
/// The clamp is at the grid-x limit rather than at a smaller round number: a
/// clamp below it would silently leave the tail of a large activation
/// uncomputed, which is the worst of the possible failures.
fn cfg_1d(total: u64, threads: u32) -> LaunchConfig {
    let blocks = total.div_ceil(threads as u64).clamp(1, GRID_X_MAX);
    LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

// ---------------------------------------------------------------------------
// Tests
//
// These cover the arithmetic that has an answer without a device: the DupUp3D
// permute, the `F.normalize` scale, the nearest-exact index map, the launch
// geometry, and the config mirror. A GPU is not required to run them, which is
// the point — the parts of this module that can be checked without a CUDA
// device and the oracle are checked. The kernel-versus-reference comparison
// needs both and lives in `vae-check-gpu`.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// The permutation the kernel implements, evaluated on the host the way the
    /// kernel's index arithmetic does.
    fn dup_up_reference(
        x: &[f32],
        batch: usize,
        in_c: usize,
        h: usize,
        w: usize,
        spec: &DupUp,
    ) -> Vec<f32> {
        let fs = spec.factor_s;
        let ft = spec.factor_t - 1;
        let factor = spec.factor_t * fs * fs;
        let (oh, ow) = (h * fs, w * fs);
        let mut out = vec![0.0f32; batch * spec.out_c * oh * ow];
        for b in 0..batch {
            for o in 0..spec.out_c {
                for fs1 in 0..fs {
                    for fs2 in 0..fs {
                        let j = o * factor + ft * fs * fs + fs1 * fs + fs2;
                        let src_c = j / spec.repeats;
                        for y in 0..h {
                            for xi in 0..w {
                                let v = x[((b * in_c + src_c) * h + y) * w + xi];
                                let oy = y * fs + fs1;
                                let ox = xi * fs + fs2;
                                out[((b * spec.out_c + o) * oh + oy) * ow + ox] = v;
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// `vae.rs`'s `dup_up_replicates_when_the_channel_count_is_unchanged`: in ==
    /// out, factor_t = 2, factor_s = 2 -> factor 8 and repeats 8, so every one
    /// of the eight sub-positions reads the same input channel and the shortcut
    /// is a pure 2x2 nearest replication.
    #[test]
    fn dup_up_replicates_when_the_channel_count_is_unchanged() {
        let spec = DupUp::new(1, 1, 2, 2).unwrap();
        assert_eq!(spec.repeats, 8);
        let x = vec![1.0f32, 2.0];
        let out = dup_up_reference(&x, 1, 1, 1, 2, &spec);
        assert_eq!(out, vec![1., 1., 2., 2., 1., 1., 2., 2.]);
    }

    /// `vae.rs`'s `dup_up_groups_channels_across_the_spatial_copies`: in = 4,
    /// out = 2, factor_t = 1, factor_s = 2 -> repeats 2, so output channel `o`
    /// takes input channel `2o` in the top row of each 2x2 cell and `2o + 1` in
    /// the bottom row.
    #[test]
    fn dup_up_groups_channels_across_the_spatial_copies() {
        let spec = DupUp::new(4, 2, 1, 2).unwrap();
        assert_eq!(spec.repeats, 2);
        let x = vec![1.0f32, 2., 3., 4.];
        let out = dup_up_reference(&x, 1, 4, 1, 1, &spec);
        assert_eq!(out, vec![1., 1., 2., 2., 3., 3., 4., 4.]);
    }

    /// `vae.rs`'s `dup_up_first_chunk_keeps_the_last_temporal_copy`: with
    /// factor_t = 2 the slice keeps `ft = 1`, so channel 1 of the input is what
    /// survives.
    #[test]
    fn dup_up_first_chunk_keeps_the_last_temporal_copy() {
        let spec = DupUp::new(2, 1, 2, 1).unwrap();
        assert_eq!(spec.repeats, 1);
        let x = vec![1.0f32, 2., 3., 4., 5., 6., 7., 8.];
        let out = dup_up_reference(&x, 1, 2, 2, 2, &spec);
        assert_eq!(out, vec![5., 6., 7., 8.]);
    }

    /// The batch axis is a block offset in the kernel, not a stride folded into
    /// the channel count: a source channel read from the flat index instead of
    /// the element's own batch would put batch 0's values in batch 1's plane.
    #[test]
    fn dup_up_keeps_batches_apart() {
        let spec = DupUp::new(2, 1, 1, 2).unwrap();
        assert_eq!(spec.repeats, 2);
        let x = vec![1.0f32, 2., /* batch 1 */ 3., 4.];
        let out = dup_up_reference(&x, 2, 2, 1, 1, &spec);
        assert_eq!(out, vec![1., 1., 2., 2., 3., 3., 4., 4.]);
    }

    /// `vae.rs`'s `dup_up_rejects_an_indivisible_channel_count`: 3 * 4 is not
    /// divisible by 5, the reference's assertion.
    #[test]
    fn dup_up_rejects_an_indivisible_channel_count() {
        assert!(DupUp::new(5, 3, 1, 2).is_err());
    }

    /// `vae.rs`'s `rms_norm_is_normalize_times_sqrt_dim`, evaluated the way the
    /// kernel's expression is parenthesized: `(v / denom) * sqrt(C) * gamma`.
    #[test]
    fn rms_norm_mirrors_the_reference() {
        let x = [1.0f32, 2., 3., 4.];
        let denom = x
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt()
            .max(NORMALIZE_EPS);
        let scale = (x.len() as f32).sqrt();
        let got: Vec<f32> = x.iter().map(|v| (v / denom) * scale * 1.0).collect();
        let want = [0.36514837, 0.73029673, 1.0954452, 1.4605935];
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-6, "got {g}, want {w}");
        }
    }

    /// The clamp, not an epsilon, is what keeps an all-zero channel vector
    /// finite: `F.normalize` divides by `max(||x||, 1e-12)`, so the result is
    /// still zero. An epsilon-RMS with eps = 1e-6 would be the same zero here,
    /// which is why `vae.rs`'s discriminatory test uses a tiny non-zero vector —
    /// this one pins the clamp itself.
    #[test]
    fn rms_norm_clamps_a_zero_norm() {
        assert_eq!(0.0f32.max(NORMALIZE_EPS), NORMALIZE_EPS);
        assert_eq!((0.0f32 / NORMALIZE_EPS) * 2.0f32.sqrt(), 0.0);
        // A tiny non-zero vector is unaffected by the clamp, because the norm
        // exceeds it: this is the case an epsilon-RMS would scale down by ~1e-3.
        let x = [3e-6f32, 4e-6];
        let denom = x
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt()
            .max(NORMALIZE_EPS);
        assert!((denom - 5e-6).abs() < 1e-12);
        let got = (x[0] / denom) * 2.0f32.sqrt();
        assert!((got - 0.84852815).abs() < 1e-6, "got {got}");
    }

    /// The nearest-exact index map the kernel computes, including the batch
    /// fold into `NC` that the kernel relies on.
    #[test]
    fn nearest_2x_repeats_each_pixel_and_keeps_batches_apart() {
        let x = [1.0f32, 2., /* batch 1 */ 3., 4.];
        let (nc, h, w) = (2usize, 1usize, 2usize);
        let (oh, ow) = (2 * h, 2 * w);
        let mut out = vec![0.0f32; nc * oh * ow];
        for o in 0..nc {
            for y in 0..oh {
                for col in 0..ow {
                    out[(o * oh + y) * ow + col] = x[o * h * w + (y / 2) * w + col / 2];
                }
            }
        }
        assert_eq!(
            out,
            vec![1., 1., 2., 2., 1., 1., 2., 2., 3., 3., 4., 4., 3., 3., 4., 4.]
        );
    }

    /// The position split the `rms_norm_channels` kernel performs: thread `i`
    /// covers spatial position `i % HW` of batch item `i / HW`, whose channel
    /// column starts at row `n * C`, and the grid is one thread per position.
    #[test]
    fn rms_norm_covers_every_position_once() {
        let (batch, c, hw) = (2usize, 3usize, 2usize);
        let rows = batch * c;
        let positions = (rows / c) * hw;
        assert_eq!(positions, batch * hw);
        let mut seen = vec![0usize; rows * hw];
        for i in 0..positions {
            let p = i % hw;
            let n = i / hw;
            for cc in 0..c {
                seen[(n * c + cc) * hw + p] += 1;
            }
        }
        assert!(seen.iter().all(|&count| count == 1));
    }

    /// The shipped config's derived architecture, mirroring `vae.rs`'s
    /// `shipped_config_describes_the_expected_decoder`.
    #[test]
    fn shipped_config_derives_the_expected_dims() {
        let cfg = VaeConfig::qwen_image_2_1();
        cfg.validate()
            .expect("the shipped config must be supported");
        // dims = 144 * [8, 8, 8, 4, 2, 1].
        assert_eq!(cfg.decoder_dims(), vec![1152, 1152, 1152, 576, 288, 144]);
        // The decoder is handed temperal_downsample reversed.
        assert_eq!(cfg.temperal_upsample(), vec![true, true, true, false]);
        // Five of the six dims are reached, so four up blocks and four
        // upsamplers, and the spatial factor is 2^4 = 16.
        assert_eq!(cfg.decoder_dims().len() - 2, 4);
    }

    /// The launch geometry helper never emits a zero grid, which the driver
    /// rejects, and covers exactly `ceil(total / threads)` blocks — including
    /// the decoder's largest activation, where clamping below the grid-x limit
    /// would leave the tail uncomputed.
    #[test]
    fn grid_covers_the_work() {
        let c = cfg_1d(0, THREADS);
        assert_eq!(c.grid_dim, (1, 1, 1));
        assert_eq!(c.block_dim, (THREADS, 1, 1));
        assert_eq!(c.shared_mem_bytes, 0);
        let c = cfg_1d(THREADS as u64 + 1, THREADS);
        assert_eq!(c.grid_dim, (2, 1, 1));
        let c = cfg_1d((THREADS as u64) * 4096, THREADS);
        assert_eq!(c.grid_dim, (4096, 1, 1));
        // 288 channels at 1024x1024, the decoder's largest f32 activation.
        let total = 288u64 * 1024 * 1024;
        let c = cfg_1d(total, THREADS);
        assert_eq!(c.grid_dim, ((total / THREADS as u64) as u32, 1, 1));
        // The clamp is the grid-x limit, not a smaller number: one block per
        // 256 elements must cover all of it.
        assert!((c.grid_dim.0 as u64) * THREADS as u64 >= total);
    }
}
