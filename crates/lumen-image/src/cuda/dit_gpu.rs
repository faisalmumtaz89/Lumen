//! The Qwen-Image-2.1 diffusion transformer, on the GPU.
//!
//! This mirrors [`crate::dit`] step for step: the same operand order, the same
//! modulation row selection, the same RoPE table, the same activation
//! arguments. The CPU reference is the specification: the per-op check
//! (`cuda-ops-check`) pins the elementwise kernels this forward calls to the
//! CPU primitives within 1e-6 relative error and the attention within bf16
//! tolerance, and the generated image is compared against the reference decode.
//!
//! The AdaLN path has no counterpart in the text kernels.
//! `modulation` carries one row per timestep, and `causal_condition` makes the
//! target-image tokens read the `t = 0` row while every other token reads the
//! sampled-timestep row. `scale_one_plus` broadcasts a single row over every
//! token and cannot express that, so the row selection is folded into
//! `layernorm_scale_bf16` / `add_gated_gather` in `dit_ops.cu`.
//!
//! Block activations stay on the device between launches: the joint sequence
//! is gathered on the device from a per-row source table built on the host
//! (index arithmetic that has to match the reference exactly), and the result
//! comes back at the end. Only the timestep embedding, a few rows wide, goes
//! through the host between its two linears. Every projection runs as a bf16
//! tensor-core GEMM with f32 accumulation; inside a block the kernel that
//! produces a projection's input rounds it to bf16 itself, the residual
//! stream stays f32, and the projections outside the blocks convert their
//! f32 input in a separate pass.
//!
//! Weights keep the dtype the `.lbi` stores. The shipped Qwen-Image-2.1
//! checkpoint is BF16, so the transformer is 13.3 GiB of weights and the
//! activations fit beside it on a 32 GiB card; a forward never materialises an
//! f32 copy of a weight matrix. A bf16 weight is multiplied by a bf16
//! activation with f32 accumulation, the precision the upstream model runs
//! the transformer at; the SIMT `gemm_16bit` (an f32 widening of each weight)
//! serves F16 weights.

use std::path::Path;

use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
use lumen_format::QuantScheme;
use lumen_runtime::cuda::ffi::CudaDevice;
use lumen_runtime::error::RuntimeError;

use super::launch::{self, DevVec};
use super::ImageKernels;
use crate::dit::{DitConfig, DitError, DitForwardArgs};
use crate::lbi::{LbiError, LbiFile, TensorEntry};
use crate::tensor::{silu, Matrix};

/// Rows of the RoPE frequency table. The reference builds it as
/// `cat(rope_params(arange(8192)), rope_params(flip(arange(1024)) * -1 - 1))`,
/// so rows `[8192, 9216)` hold positions `[-1024, -1)` and a negative position
/// lands on its own row through Python's negative indexing. Reproducing the
/// table rather than evaluating a position directly also reproduces the
/// aliasing: a position below `-1024` wraps onto a positive row instead of
/// failing, exactly as it does in torch.
const ROPE_POS_ROWS: i64 = 8192;
const ROPE_NEG_ROWS: i64 = 1024;
const ROPE_ROWS: i64 = ROPE_POS_ROWS + ROPE_NEG_ROWS;

/// `QwenImage21TimestepProjEmbeddings` fixes the sinusoidal width at 256.
const TIMESTEP_DIM: usize = 256;
const TIMESTEP_MAX_PERIOD: f64 = 10000.0;
const TIMESTEP_TIME_FACTOR: f32 = 1000.0;

/// Each vision-language image slot stands for a 2x2 group of latent tokens.
const IMG_TOKENS_PER_SLOT: usize = 4;

/// Threads per block for the elementwise and row kernels.
const THREADS: u32 = 256;
// The row kernels in `dit_ops.cu` stage one partial per thread in a static
// shared array of 256 entries; the block size is that array's size.
const _: () = assert!(THREADS == 256);

/// `modulation.1` projects to `4 * hidden` and each chunk feeds a different
/// consumer. Named rather than inlined because the four are the same length:
/// swapping two of them keeps every buffer in bounds and every result finite,
/// and only changes the numbers. The reference's `add_gated`/`scale_rows` call
/// sites in `dit.rs` are the source of truth for this mapping.
const MOD_ATTN_SCALE: usize = 0;
const MOD_ATTN_GATE: usize = 1;
const MOD_MLP_SCALE: usize = 2;
const MOD_MLP_GATE: usize = 3;

/// `norm_out`'s scale is `[n_timesteps, hidden]`, not a chunk of the shared
/// modulation, so it is read from offset 0 with a row width of `hidden`.
const NORM_OUT_SCALE: usize = 0;

/// The kernel source this module compiles alongside `image_ops.cu`.
pub const DIT_OPS_SOURCE: &str = include_str!("dit_ops.cu");

#[derive(Debug)]
pub enum DitGpuError {
    /// Allocating, copying or launching on the device.
    Cuda(RuntimeError),
    /// The checkpoint does not carry a tensor, or an extent is wrong. Reusing
    /// the reference's error keeps a shape failure reported the same way.
    Dit(DitError),
    /// A weight is stored in a scheme with no dispatch.
    UnsupportedStorage { tensor: String, scheme: String },
    /// A sequence layout the attention mask cannot express.
    UnsupportedLayout(String),
}

impl std::fmt::Display for DitGpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cuda(e) => write!(f, "cuda: {e}"),
            Self::Dit(e) => write!(f, "{e}"),
            Self::UnsupportedStorage { tensor, scheme } => {
                write!(
                    f,
                    "tensor {tensor} is stored as {scheme}, which has no dispatch"
                )
            }
            Self::UnsupportedLayout(what) => write!(f, "{what}"),
        }
    }
}

impl std::error::Error for DitGpuError {}

impl From<RuntimeError> for DitGpuError {
    fn from(e: RuntimeError) -> Self {
        Self::Cuda(e)
    }
}

impl From<DitError> for DitGpuError {
    fn from(e: DitError) -> Self {
        Self::Dit(e)
    }
}

impl From<LbiError> for DitGpuError {
    fn from(e: LbiError) -> Self {
        Self::Dit(DitError::Lbi(e))
    }
}

/// A linear weight resident on the device, in the dtype the `.lbi` stores.
///
/// `rows` is the output width and `cols` the input width, kept alongside the
/// buffer rather than derived from its length: which factor is which is not
/// recoverable from an element count, and a transposed operand is a silent
/// wrong answer.
enum DevWeight {
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

impl DevWeight {
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

    fn scheme_name(&self) -> &'static str {
        match self {
            Self::F32 { .. } => "f32",
            Self::F16 { .. } => "f16",
            Self::Bf16 { .. } => "bf16",
        }
    }
}

/// One single-stream block, resident on the device.
struct GpuBlock {
    to_q: DevWeight,
    to_k: DevWeight,
    to_v: DevWeight,
    to_out: DevWeight,
    norm_q: DevVec,
    norm_k: DevVec,
    mlp_gate: DevWeight,
    mlp_proj: DevWeight,
    mlp_out: DevWeight,
}

/// One rotary axis: `ROPE_ROWS` positions of `half` complex frequencies.
struct RopeAxis {
    half: usize,
    data: Vec<(f32, f32)>,
}

impl RopeAxis {
    fn build(dim: usize, theta: f32) -> Self {
        // `range(0, dim, 2)`, so an odd `dim` rounds up — the reference derives
        // the width from the step range, not from `dim / 2`.
        let inv: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|j| 1.0 / theta.powf(j as f32 / dim as f32))
            .collect();
        let half = inv.len();
        let mut data = Vec::with_capacity(ROPE_ROWS as usize * half);
        for row in 0..ROPE_ROWS {
            let index = if row < ROPE_POS_ROWS {
                row
            } else {
                row - ROPE_ROWS
            };
            for &f in &inv {
                let angle = index as f32 * f;
                data.push((angle.cos(), angle.sin()));
            }
        }
        Self { half, data }
    }

    fn row(&self, index: i64) -> Result<&[(f32, f32)], DitGpuError> {
        let row = if index < 0 { ROPE_ROWS + index } else { index };
        if !(0..ROPE_ROWS).contains(&row) {
            return Err(DitError::ShapeMismatch {
                what: format!("rope position {index} is outside the frequency table"),
                expected: vec![ROPE_ROWS as u64],
                actual: vec![index.unsigned_abs()],
            }
            .into());
        }
        let start = row as usize * self.half;
        Ok(&self.data[start..start + self.half])
    }
}

/// Which 16-bit storage a [`ops::gemm_16bit`] call reads.
pub use ops::Gemm16;

/// The kernels in `dit_ops.cu`.
pub use ops::DitOps;

/// The kernels behind the ops that no existing module provides.
///
/// These live in their own module because two callers drive them: the forward
/// here, and `cuda-ops-check`, which pins each one to the CPU reference on
/// synthetic inputs. Routing both through one launcher is what makes that check
/// speak for the forward's own arithmetic rather than for a copy of it.
pub mod ops {
    use super::THREADS;
    use super::{launch, DevVec, DIT_OPS_SOURCE};
    use super::{CudaDevice, CudaFunction, CudaSlice, LaunchConfig, PushKernelArg, RuntimeError};

    /// Tiling of `gemm_16bit`, matching `image_ops.cu`'s `gemm_f32_bias`.
    const GEMM_TILE: u32 = 32;

    /// Which 16-bit float a weight buffer holds. The kernel widens either to
    /// f32 on the way into its tile, so the two differ only in the widening.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Gemm16 {
        F16,
        Bf16,
    }

    /// The kernels in `dit_ops.cu`.
    pub struct DitOps {
        gemm_16bit: CudaFunction,
        zero_center_rmsnorm: CudaFunction,
        gelu_tanh_inplace: CudaFunction,
        add_gated_gather: CudaFunction,
        pack_rows: CudaFunction,
        layernorm_scale_bf16: CudaFunction,
        swiglu_bf16: CudaFunction,
        head_norm_rope_bf16: CudaFunction,
    }

    /// Compile `dit_ops.cu` and resolve its entry points.
    pub fn load(dev: &CudaDevice) -> Result<DitOps, RuntimeError> {
        let module = dev.compile_and_load(DIT_OPS_SOURCE)?;
        let get = |name: &str| -> Result<CudaFunction, RuntimeError> {
            module
                .load_function(name)
                .map_err(|e| RuntimeError::Compute(format!("load {name}: {e}")))
        };
        Ok(DitOps {
            gemm_16bit: get("gemm_16bit")?,
            zero_center_rmsnorm: get("zero_center_rmsnorm")?,
            gelu_tanh_inplace: get("gelu_tanh_inplace")?,
            add_gated_gather: get("add_gated_gather")?,
            pack_rows: get("pack_rows")?,
            layernorm_scale_bf16: get("layernorm_scale_bf16")?,
            swiglu_bf16: get("swiglu_bf16")?,
            head_norm_rope_bf16: get("head_norm_rope_bf16")?,
        })
    }

    /// One block of [`THREADS`] threads per row.
    fn row_grid(rows: usize) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// The product of `factors` as the `u32` the kernels index with; a count
    /// that overflows either the product or `u32` is refused rather than
    /// wrapped.
    fn elements(name: &str, factors: &[usize]) -> Result<u32, RuntimeError> {
        factors
            .iter()
            .try_fold(1usize, |n, &f| n.checked_mul(f))
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| {
                RuntimeError::Compute(format!(
                    "{name}: {factors:?} elements exceed the u32 kernel index"
                ))
            })
    }

    pub fn flat_grid(total: u32) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (total.div_ceil(THREADS), 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// `out[M, N] = a[M, K] * W_16bit[N, K]^T`. Same tiling and geometry as
    /// `gemm_f32_bias`; the kernel widens each element into its tile.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_16bit(
        dev: &CudaDevice,
        k: &DitOps,
        w: &CudaSlice<u16>,
        a: &DevVec,
        m: usize,
        n: usize,
        kdim: usize,
        kind: Gemm16,
    ) -> Result<DevVec, RuntimeError> {
        let out = launch::alloc(dev, m * n)?;
        let (mu, nu, ku) = (m as u32, n as u32, kdim as u32);
        let flag: u32 = match kind {
            Gemm16::F16 => 0,
            Gemm16::Bf16 => 1,
        };
        // Safety: both buffers are device allocations of the sizes the kernel
        // reads, and the grid covers the output tile.
        unsafe {
            dev.stream
                .launch_builder(&k.gemm_16bit)
                .arg(&a.buf)
                .arg(w)
                .arg(&out.buf)
                .arg(&mu)
                .arg(&nu)
                .arg(&ku)
                .arg(&flag)
                .launch(LaunchConfig {
                    grid_dim: (
                        n.div_ceil(GEMM_TILE as usize) as u32,
                        m.div_ceil(GEMM_TILE as usize) as u32,
                        1,
                    ),
                    block_dim: (GEMM_TILE, GEMM_TILE, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| RuntimeError::Compute(format!("gemm_16bit: {e}")))?;
        }
        Ok(out)
    }

    /// `zero_center_rms_norm_rows`, one block per row.
    pub fn zero_center_rmsnorm(
        dev: &CudaDevice,
        k: &DitOps,
        x: &DevVec,
        weight: &DevVec,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<DevVec, RuntimeError> {
        let out = launch::alloc(dev, x.len)?;
        let du = dim as u32;
        // Safety: buffers match the kernel's declared shapes and geometry.
        unsafe {
            dev.stream
                .launch_builder(&k.zero_center_rmsnorm)
                .arg(&x.buf)
                .arg(&weight.buf)
                .arg(&out.buf)
                .arg(&du)
                .arg(&eps)
                .launch(row_grid(rows))
                .map_err(|e| RuntimeError::Compute(format!("zero_center_rmsnorm: {e}")))?;
        }
        Ok(out)
    }

    /// GELU in place, one thread per element.
    pub fn gelu_tanh_inplace(
        dev: &CudaDevice,
        k: &DitOps,
        x: &mut DevVec,
    ) -> Result<(), RuntimeError> {
        let n = x.len as u32;
        // Safety: the buffer is `n` floats and the grid covers exactly that.
        unsafe {
            dev.stream
                .launch_builder(&k.gelu_tanh_inplace)
                .arg(&mut x.buf)
                .arg(&n)
                .launch(flat_grid(n))
                .map_err(|e| RuntimeError::Compute(format!("gelu_tanh_inplace: {e}")))?;
        }
        Ok(())
    }

    /// The joint sequence: row `t` is `txt[source[t]]` when `source[t] >= 0`
    /// and `img[-source[t] - 1]` otherwise. Every source must name a row of
    /// the buffer it points into.
    pub fn pack_rows(
        dev: &CudaDevice,
        k: &DitOps,
        txt: &DevVec,
        img: &DevVec,
        source: &[i32],
        cols: usize,
    ) -> Result<DevVec, RuntimeError> {
        if cols == 0 || txt.len % cols != 0 || img.len % cols != 0 {
            return Err(RuntimeError::Compute(format!(
                "pack_rows: txt {} and img {} are not whole rows of {cols}",
                txt.len, img.len
            )));
        }
        let (text_rows, image_rows) = (txt.len / cols, img.len / cols);
        for &s in source {
            let (which, row, count) = if s >= 0 {
                ("text", s as usize, text_rows)
            } else {
                ("image", (-(s as i64) - 1) as usize, image_rows)
            };
            if row >= count {
                return Err(RuntimeError::Compute(format!(
                    "pack_rows: source {s} names {which} row {row} of {count}"
                )));
            }
        }
        let rows = source.len();
        let total = elements("pack_rows", &[rows, cols])?;
        let g_source: CudaSlice<i32> = dev.htod_copy(source)?;
        let out = launch::alloc(dev, rows * cols)?;
        let (rows_u, cols_u) = (rows as u32, cols as u32);
        // Safety: every source index was checked against the two row counts
        // above, and the grid covers exactly `rows * cols`.
        unsafe {
            dev.stream
                .launch_builder(&k.pack_rows)
                .arg(&txt.buf)
                .arg(&img.buf)
                .arg(&g_source)
                .arg(&out.buf)
                .arg(&rows_u)
                .arg(&cols_u)
                .launch(flat_grid(total))
                .map_err(|e| RuntimeError::Compute(format!("pack_rows: {e}")))?;
        }
        Ok(out)
    }

    /// Upload little-endian 16-bit float bits without changing them.
    ///
    /// The kernel's input format is the stored one, so the bytes move straight
    /// across from the mapped file: re-encoding through f32 would be an
    /// identity for representable values and a silent rounding for anything
    /// else, and a host-side copy into a `u16` buffer would touch every byte
    /// of the model once more than the transfer itself does.
    pub fn upload_16bit(dev: &CudaDevice, bytes: &[u8]) -> Result<CudaSlice<u16>, RuntimeError> {
        const {
            assert!(
                cfg!(target_endian = "little"),
                "stored bits are little-endian"
            )
        };
        if bytes.len() % 2 != 0 {
            return Err(RuntimeError::Compute(format!(
                "16-bit tensor has an odd byte length {}",
                bytes.len()
            )));
        }
        // Safety: every element is written by the copy below before the
        // buffer is read.
        let mut bits = unsafe { dev.alloc_uninit::<u16>(bytes.len() / 2)? };
        // Safety: the view spans exactly the buffer's bytes.
        let mut view = unsafe { bits.transmute_mut::<u8>(bytes.len()) }
            .ok_or_else(|| RuntimeError::Compute("16-bit upload view".into()))?;
        dev.stream
            .memcpy_htod(bytes, &mut view)
            .map_err(|e| RuntimeError::Compute(format!("16-bit upload: {e}")))?;
        Ok(bits)
    }

    /// The geometry the two AdaLN kernels share: `x` is whole rows of `cols`,
    /// `mod_row` names one modulation row per row of `x`, and the chunk
    /// `col_off..col_off + cols` lies inside a modulation row of `mod_stride`.
    /// Returns the row count. The values in `mod_row` are the caller's to
    /// keep inside `modulation`; the forward builds them from its own
    /// timestep count.
    fn adaln_rows(
        name: &str,
        x: &DevVec,
        modulation: &DevVec,
        mod_row: &CudaSlice<i32>,
        col_off: usize,
        cols: usize,
        mod_stride: usize,
    ) -> Result<usize, RuntimeError> {
        let whole_rows = cols != 0 && x.len % cols == 0 && mod_row.len() == x.len / cols;
        let chunk_inside = mod_stride != 0
            && col_off
                .checked_add(cols)
                .is_some_and(|end| end <= mod_stride)
            && modulation.len % mod_stride == 0;
        if !whole_rows || !chunk_inside {
            return Err(RuntimeError::Compute(format!(
                "{name}: x {} / mod_row {} / modulation {} do not fit cols {cols} at \
                 {col_off} of stride {mod_stride}",
                x.len,
                mod_row.len(),
                modulation.len
            )));
        }
        elements(name, &[x.len])?;
        Ok(x.len / cols)
    }

    /// `bf16(layernorm(x) * (1 + modulation[row]))`, the input a projection
    /// takes, in one pass over `x`.
    #[allow(clippy::too_many_arguments)]
    pub fn layernorm_scale_bf16(
        dev: &CudaDevice,
        k: &DitOps,
        x: &DevVec,
        modulation: &DevVec,
        mod_row: &CudaSlice<i32>,
        col_off: usize,
        cols: usize,
        mod_stride: usize,
        eps: f32,
    ) -> Result<CudaSlice<u16>, RuntimeError> {
        let rows = adaln_rows(
            "layernorm_scale_bf16",
            x,
            modulation,
            mod_row,
            col_off,
            cols,
            mod_stride,
        )?;
        // Safety: every element is written by the kernel.
        let mut out = unsafe { dev.alloc_uninit::<u16>(x.len)? };
        let (cols_u, off_u, stride_u) = (cols as u32, col_off as u32, mod_stride as u32);
        // Safety: buffers match the kernel's declared shapes, and the grid is
        // one block per row.
        unsafe {
            dev.stream
                .launch_builder(&k.layernorm_scale_bf16)
                .arg(&x.buf)
                .arg(&modulation.buf)
                .arg(mod_row)
                .arg(&mut out)
                .arg(&cols_u)
                .arg(&off_u)
                .arg(&stride_u)
                .arg(&eps)
                .launch(row_grid(rows))
                .map_err(|e| RuntimeError::Compute(format!("layernorm_scale_bf16: {e}")))?;
        }
        Ok(out)
    }

    /// `bf16(silu(gate) * up)`, the down projection's input.
    pub fn swiglu_bf16(
        dev: &CudaDevice,
        k: &DitOps,
        gate: &DevVec,
        up: &DevVec,
    ) -> Result<CudaSlice<u16>, RuntimeError> {
        if gate.len != up.len {
            return Err(RuntimeError::Compute(format!(
                "swiglu_bf16: gate has {} elements, up has {}",
                gate.len, up.len
            )));
        }
        let n = elements("swiglu_bf16", &[gate.len])?;
        // Safety: every element is written by the kernel.
        let mut out = unsafe { dev.alloc_uninit::<u16>(gate.len)? };
        // Safety: all three buffers are `n` elements and the grid covers them.
        unsafe {
            dev.stream
                .launch_builder(&k.swiglu_bf16)
                .arg(&gate.buf)
                .arg(&up.buf)
                .arg(&mut out)
                .arg(&n)
                .launch(flat_grid(n))
                .map_err(|e| RuntimeError::Compute(format!("swiglu_bf16: {e}")))?;
        }
        Ok(out)
    }

    /// Q or K from its projection to the attention operand: per-head RMSNorm,
    /// the rotation, and the truncating bf16 conversion, in one warp per
    /// (token, head) row.
    pub fn head_norm_rope_bf16(
        dev: &CudaDevice,
        k: &DitOps,
        x: &DevVec,
        weight: &DevVec,
        freqs: &DevVec,
        seq: usize,
        heads: usize,
        eps: f32,
    ) -> Result<CudaSlice<u16>, RuntimeError> {
        const HEAD_DIM: usize = crate::cuda::attention::FLASH_HEAD_DIM;
        let total = elements("head_norm_rope_bf16", &[seq, heads, HEAD_DIM])? as usize;
        if x.len != total || weight.len != HEAD_DIM || freqs.len != seq * HEAD_DIM {
            return Err(RuntimeError::Compute(format!(
                "head_norm_rope_bf16: x {} / weight {} / freqs {} do not match {seq}x{heads}x{HEAD_DIM}",
                x.len, weight.len, freqs.len
            )));
        }
        // One warp per row of 128: a quarter of the elements as threads.
        let threads = (total / 4) as u32;
        // Safety: every element is written by the kernel.
        let mut out = unsafe { dev.alloc_uninit::<u16>(x.len)? };
        let (seq_u, heads_u) = (seq as u32, heads as u32);
        // Safety: buffers match the kernel's declared shapes; the grid is one
        // warp per (token, head) row.
        unsafe {
            dev.stream
                .launch_builder(&k.head_norm_rope_bf16)
                .arg(&x.buf)
                .arg(&weight.buf)
                .arg(&freqs.buf)
                .arg(&mut out)
                .arg(&seq_u)
                .arg(&heads_u)
                .arg(&eps)
                .launch(flat_grid(threads))
                .map_err(|e| RuntimeError::Compute(format!("head_norm_rope_bf16: {e}")))?;
        }
        Ok(out)
    }

    /// `x + tanh(modulation[row]) * y`.
    #[allow(clippy::too_many_arguments)]
    pub fn add_gated(
        dev: &CudaDevice,
        k: &DitOps,
        x: &DevVec,
        y: &DevVec,
        modulation: &DevVec,
        mod_row: &CudaSlice<i32>,
        col_off: usize,
        cols: usize,
        mod_stride: usize,
    ) -> Result<DevVec, RuntimeError> {
        let rows = adaln_rows(
            "add_gated",
            x,
            modulation,
            mod_row,
            col_off,
            cols,
            mod_stride,
        )?;
        if y.len != x.len {
            return Err(RuntimeError::Compute(format!(
                "add_gated: y has {} elements, x has {}",
                y.len, x.len
            )));
        }
        let out = launch::alloc(dev, x.len)?;
        let (rows_u, cols_u) = (rows as u32, cols as u32);
        let (off_u, stride_u) = (col_off as u32, mod_stride as u32);
        let total = x.len as u32;
        // Safety: buffers match the kernel's declared shapes and geometry.
        unsafe {
            dev.stream
                .launch_builder(&k.add_gated_gather)
                .arg(&x.buf)
                .arg(&y.buf)
                .arg(&modulation.buf)
                .arg(mod_row)
                .arg(&out.buf)
                .arg(&rows_u)
                .arg(&cols_u)
                .arg(&off_u)
                .arg(&stride_u)
                .launch(flat_grid(total))
                .map_err(|e| RuntimeError::Compute(format!("add_gated_gather: {e}")))?;
        }
        Ok(out)
    }
}

/// The transformer, with every weight resident on the device.
pub struct DitGpu {
    /// The model drives one stream from host inputs end to end, so it holds its
    /// own device handle rather than borrowing the caller's. `CudaDevice::new`
    /// retains the same primary context, so this is a second stream on the same
    /// device, not a second device.
    dev: CudaDevice,
    kernels: ImageKernels,
    ops: DitOps,

    config: DitConfig,
    img_in: DevWeight,
    text_norm: DevVec,
    txt_in: DevWeight,
    txt_out: DevWeight,
    time_linear_1: DevWeight,
    time_linear_2: DevWeight,
    modulation: DevWeight,
    blocks: Vec<GpuBlock>,
    norm_out: DevWeight,
    proj_out: DevWeight,
    rope: [RopeAxis; 3],
}

impl DitGpu {
    /// Load every DiT weight from a converted `.lbi` onto the device, assuming
    /// the shipped Qwen-Image-2.1 architecture.
    pub fn load(lbi: &Path, dev: &CudaDevice) -> Result<Self, DitGpuError> {
        Self::load_with(lbi, dev, DitConfig::qwen_image_2_1())
    }

    /// Load against an explicit architecture.
    pub fn load_with(lbi: &Path, dev: &CudaDevice, config: DitConfig) -> Result<Self, DitGpuError> {
        let own = CudaDevice::new(dev.ctx.ordinal())?;
        let kernels = ImageKernels::load(&own)?;
        let ops = ops::load(&own)?;
        let file = LbiFile::open(lbi)?;
        let hidden = config.inner_dim();
        let mlp = config.mlp_hidden();
        // The attention kernels are tiled for the model's 128-wide heads
        // (`flash_attn.cu`, `head_norm_rope_bf16`); a narrower or wider head
        // is refused here rather than read past a tile.
        if config.attention_head_dim != crate::cuda::attention::FLASH_HEAD_DIM {
            return Err(DitError::ShapeMismatch {
                what: "attention head width".to_string(),
                expected: vec![crate::cuda::attention::FLASH_HEAD_DIM as u64],
                actual: vec![config.attention_head_dim as u64],
            }
            .into());
        }

        let rope = [
            RopeAxis::build(config.axes_dims_rope[0], config.rope_theta),
            RopeAxis::build(config.axes_dims_rope[1], config.rope_theta),
            RopeAxis::build(config.axes_dims_rope[2], config.rope_theta),
        ];
        // The axes must exactly cover half a head: that is what makes one
        // position's frequencies as wide as the complex half of a head.
        let rope_half: usize = rope.iter().map(|a| a.half).sum();
        if rope_half != config.attention_head_dim / 2 {
            return Err(DitError::ShapeMismatch {
                what: "rope axis dims do not cover a head".to_string(),
                expected: vec![(config.attention_head_dim / 2) as u64],
                actual: vec![rope_half as u64],
            }
            .into());
        }

        let mut blocks = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let p = format!("transformer_blocks.{i}");
            blocks.push(GpuBlock {
                to_q: projection(
                    &own,
                    &file,
                    &format!("{p}.attn.to_q.weight"),
                    hidden,
                    hidden,
                )?,
                to_k: projection(
                    &own,
                    &file,
                    &format!("{p}.attn.to_k.weight"),
                    hidden,
                    hidden,
                )?,
                to_v: projection(
                    &own,
                    &file,
                    &format!("{p}.attn.to_v.weight"),
                    hidden,
                    hidden,
                )?,
                to_out: projection(
                    &own,
                    &file,
                    &format!("{p}.attn.to_out.0.weight"),
                    hidden,
                    hidden,
                )?,
                norm_q: vector(
                    &own,
                    &file,
                    &format!("{p}.attn.norm_q.weight"),
                    config.attention_head_dim,
                )?,
                norm_k: vector(
                    &own,
                    &file,
                    &format!("{p}.attn.norm_k.weight"),
                    config.attention_head_dim,
                )?,
                mlp_gate: projection(
                    &own,
                    &file,
                    &format!("{p}.img_mlp.gate_layer.weight"),
                    mlp,
                    hidden,
                )?,
                mlp_proj: projection(
                    &own,
                    &file,
                    &format!("{p}.img_mlp.proj.weight"),
                    mlp,
                    hidden,
                )?,
                mlp_out: projection(&own, &file, &format!("{p}.img_mlp.out.weight"), hidden, mlp)?,
            });
        }

        Ok(Self {
            img_in: weight(&own, &file, "img_in.weight", hidden, config.in_channels)?,
            text_norm: vector(
                &own,
                &file,
                "txt_in.text_norm.weight",
                config.context_in_dim,
            )?,
            txt_in: weight(
                &own,
                &file,
                "txt_in.in_layer.weight",
                hidden,
                config.context_in_dim,
            )?,
            txt_out: weight(&own, &file, "txt_in.out_layer.weight", hidden, hidden)?,
            time_linear_1: weight(
                &own,
                &file,
                "time_text_embed.timestep_embedder.linear_1.weight",
                hidden,
                TIMESTEP_DIM,
            )?,
            time_linear_2: weight(
                &own,
                &file,
                "time_text_embed.timestep_embedder.linear_2.weight",
                hidden,
                hidden,
            )?,
            modulation: weight(&own, &file, "modulation.1.weight", 4 * hidden, hidden)?,
            norm_out: weight(&own, &file, "norm_out.linear.weight", hidden, hidden)?,
            proj_out: projection(&own, &file, "proj_out.weight", config.out_channels, hidden)?,
            blocks,
            rope,
            config,
            dev: own,
            kernels,
            ops,
        })
    }

    /// One forward pass over the whole joint sequence.
    ///
    /// Equal to [`crate::dit::Dit::forward`] on the same inputs. The result has
    /// one row per joint token, text included; the caller takes the trailing
    /// `target_tokens` rows, as the pipeline's
    /// `noise_pred[:, -latents.size(1):]` does.
    pub fn forward(&self, args: DitForwardArgs<'_>) -> Result<Matrix, DitGpuError> {
        let cfg = &self.config;
        let hidden = cfg.inner_dim();

        if args.hidden_states.cols != cfg.in_channels {
            return Err(mismatch(
                "hidden_states width",
                cfg.in_channels,
                args.hidden_states.cols,
            )
            .into());
        }
        if args.encoder_hidden_states.cols != cfg.context_in_dim {
            return Err(mismatch(
                "encoder_hidden_states width",
                cfg.context_in_dim,
                args.encoder_hidden_states.cols,
            )
            .into());
        }
        let Some(&(tf, th, tw)) = args.img_shapes.last() else {
            return Err(mismatch("img_shapes length", 1, 0).into());
        };
        let target_tokens = (tf * th * tw) as usize;
        if target_tokens % IMG_TOKENS_PER_SLOT != 0 {
            return Err(mismatch(
                "target image tokens per 2x2 slot",
                0,
                target_tokens % IMG_TOKENS_PER_SLOT,
            )
            .into());
        }
        let text_seq = args.encoder_hidden_states.rows;
        let want_slots = text_seq + target_tokens / IMG_TOKENS_PER_SLOT;
        if args.img_mask.len() != want_slots {
            return Err(mismatch("img_mask length", want_slots, args.img_mask.len()).into());
        }

        let img = self.linear(&self.img_in, args.hidden_states)?;
        let txt = self.text_projection(args.encoder_hidden_states)?;

        // Each image slot stands for four latent tokens, so expand the slot
        // sequence four-fold at the image slots. `source` names each joint
        // row's origin: a text slot's row, or (encoded negative) the next row
        // of the packed image projection.
        let mut image_pad_mask = Vec::with_capacity(args.img_mask.len());
        let mut source: Vec<i32> = Vec::with_capacity(args.img_mask.len());
        let mut next_image = 0i32;
        for (slot, &is_image) in args.img_mask.iter().enumerate() {
            let repeats = if is_image { IMG_TOKENS_PER_SLOT } else { 1 };
            for _ in 0..repeats {
                if is_image {
                    source.push(-next_image - 1);
                    next_image += 1;
                } else {
                    source.push(slot as i32);
                }
                image_pad_mask.push(is_image);
            }
        }
        let image_tokens = next_image as usize;
        if img.len != image_tokens * hidden {
            return Err(mismatch("packed latent rows", image_tokens, img.len / hidden).into());
        }
        let seq = source.len();

        let (frame_index, height_index, width_index) =
            rope_indices(args.img_shapes, &image_pad_mask)?;
        let g_freqs = launch::upload(
            &self.dev,
            &self.rope_freqs(&frame_index, &height_index, &width_index)?,
        )?;
        let (image_ids, target_token_mask) = token_metadata(&image_pad_mask, args.img_shapes)?;
        // The attention kernel takes the mask as a text-prefix length: a text
        // query at position q sees keys [0, q], an image query sees every key.
        // That is the reference's `(q >= kv) or same_image_block` rule only for
        // the text-then-one-image layout, so any other layout is refused here
        // rather than masked wrongly.
        let text_count = image_ids.iter().filter(|&&v| v < 0).count();
        let text_prefix = image_ids[..text_count].iter().all(|&v| v < 0)
            && image_ids[text_count..].iter().all(|&v| v == 0);
        if !text_prefix {
            return Err(DitGpuError::UnsupportedLayout(
                "attention mask: the sequence must be a text prefix followed by one image block"
                    .into(),
            ));
        }

        // With `causal_condition` the modulation carries an extra `t = 0` row,
        // which every token outside the target image reads.
        let timesteps: &[f32] = if cfg.causal_condition {
            &[args.timestep, 0.0]
        } else {
            std::slice::from_ref(&args.timestep)
        };
        let temb = self.timestep_embedding(timesteps)?;
        let g_modulation = self.linear(&self.modulation, &apply_silu(&temb))?;
        let mod_row: Vec<i32> = if cfg.causal_condition {
            target_token_mask
                .iter()
                .map(|&target| if target { 0 } else { 1 })
                .collect()
        } else {
            vec![0; seq]
        };
        let g_mod_row: CudaSlice<i32> = self.dev.htod_copy(&mod_row)?;

        // The shared modulation's row width, so a chunk offset is a multiple of
        // `hidden` rather than of the stride.
        let mod_stride = 4 * hidden;
        let heads = cfg.num_attention_heads;
        let mut x = ops::pack_rows(&self.dev, &self.ops, &txt, &img, &source, hidden)?;
        for block in &self.blocks {
            let normed = self.normed_input(
                &x,
                &g_modulation,
                &g_mod_row,
                MOD_ATTN_SCALE * hidden,
                hidden,
                mod_stride,
            )?;
            let attn = self.attention(block, &normed, seq, heads, &g_freqs, text_count)?;
            x = self.add_gated(
                &x,
                &attn,
                &g_modulation,
                &g_mod_row,
                MOD_ATTN_GATE * hidden,
                hidden,
                mod_stride,
            )?;

            let normed = self.normed_input(
                &x,
                &g_modulation,
                &g_mod_row,
                MOD_MLP_SCALE * hidden,
                hidden,
                mod_stride,
            )?;
            let mlp = self.feed_forward(block, &normed, seq)?;
            x = self.add_gated(
                &x,
                &mlp,
                &g_modulation,
                &g_mod_row,
                MOD_MLP_GATE * hidden,
                hidden,
                mod_stride,
            )?;
        }

        // `QwenImage21AdaLayerNormContinuous`: scale only, read from `temb`
        // rather than from the shared modulation.
        let scale = self.linear(&self.norm_out, &apply_silu(&temb))?;
        let normed = self.normed_input(&x, &scale, &g_mod_row, NORM_OUT_SCALE, hidden, hidden)?;
        let out = self.linear_bf16(&self.proj_out, &normed)?;
        let data = self.download(&out)?;
        Ok(Matrix::new(seq, cfg.out_channels, data))
    }

    // -- the pieces of the forward ------------------------------------------

    /// `QwenImage21TextProjection`: zero-centred RMSNorm, linear, GELU, linear.
    fn text_projection(&self, encoder_hidden_states: &Matrix) -> Result<DevVec, DitGpuError> {
        let rows = encoder_hidden_states.rows;
        let dim = encoder_hidden_states.cols;
        let gx = self.upload(encoder_hidden_states)?;
        let normed = self.zero_center_rmsnorm(&gx, &self.text_norm, rows, dim)?;
        let mut h = self.linear_buf(&self.txt_in, &normed, rows)?;
        self.gelu_tanh_inplace(&mut h)?;
        self.linear_buf(&self.txt_out, &h, rows)
    }

    /// The sinusoidal timestep embedding followed by `TimestepEmbedding`, whose
    /// `forward` puts the activation between the two linears.
    ///
    /// The activation is `silu(linear_1(proj))` — SiLU sits *after* `linear_1`
    /// and *before* `linear_2` in the reference. Applying it to the sinusoidal
    /// `proj` as well is wrong and leaves no symptom a shape check can catch:
    /// the result is still finite and still the right shape.
    ///
    /// The raw `temb` comes back to the host because `silu` of it is read twice
    /// more — by the shared modulation and by `norm_out` — and the reference
    /// evaluates both from the one tensor this returns.
    fn timestep_embedding(&self, timesteps: &[f32]) -> Result<Matrix, DitGpuError> {
        let mut proj = Matrix::zeros(timesteps.len(), TIMESTEP_DIM);
        for (r, &t) in timesteps.iter().enumerate() {
            temporal_timesteps(t, proj.row_mut(r));
        }
        let h = self.linear_host(&self.time_linear_1, &proj)?;
        self.linear_host(&self.time_linear_2, &apply_silu(&h))
    }

    /// One block's attention, up to and including `to_out`.
    fn attention(
        &self,
        block: &GpuBlock,
        act: &crate::cuda::blas::Bf16Activation,
        seq: usize,
        heads: usize,
        freqs: &DevVec,
        text_count: usize,
    ) -> Result<DevVec, DitGpuError> {
        // The three projections read the same normed input.
        let q = self.linear_bf16(&block.to_q, act)?;
        let k = self.linear_bf16(&block.to_k, act)?;
        let v = self.linear_bf16(&block.to_v, act)?;

        let eps = self.config.eps;
        let q = ops::head_norm_rope_bf16(
            &self.dev,
            &self.ops,
            &q,
            &block.norm_q,
            freqs,
            seq,
            heads,
            eps,
        )?;
        let k = ops::head_norm_rope_bf16(
            &self.dev,
            &self.ops,
            &k,
            &block.norm_k,
            freqs,
            seq,
            heads,
            eps,
        )?;

        // Safety: q and k are the `[seq, heads, 128]` bf16 operands the fused
        // norm just produced and v the projection's f32 output of the same
        // shape.
        let out = unsafe {
            crate::cuda::attention::fused_block_causal_attention(
                &self.dev,
                &self.kernels,
                &q,
                &k,
                &v,
                text_count,
                seq,
                heads,
            )
        }?;
        self.linear_bf16(&block.to_out, &out)
    }

    /// `QwenImage21SwiGLUFeedForward`: the gate branch goes through SiLU, the
    /// projection branch does not.
    fn feed_forward(
        &self,
        block: &GpuBlock,
        act: &crate::cuda::blas::Bf16Activation,
        seq: usize,
    ) -> Result<DevVec, DitGpuError> {
        let gate = self.linear_bf16(&block.mlp_gate, act)?;
        let proj = self.linear_bf16(&block.mlp_proj, act)?;
        // `silu(gate) * up`, the reference's `*g = silu(*g) * p` with the same
        // operands in the same order, rounded for the down projection.
        let hidden = ops::swiglu_bf16(&self.dev, &self.ops, &gate, &proj)?;
        let hidden =
            crate::cuda::blas::Bf16Activation::from_bits(hidden, seq, block.mlp_out.cols())?;
        self.linear_bf16(&block.mlp_out, &hidden)
    }

    /// `bf16(layernorm(x) * (1 + modulation[row]))`: a block's normed,
    /// modulated input, ready for its projections.
    fn normed_input(
        &self,
        x: &DevVec,
        modulation: &DevVec,
        mod_row: &CudaSlice<i32>,
        col_off: usize,
        cols: usize,
        mod_stride: usize,
    ) -> Result<crate::cuda::blas::Bf16Activation, DitGpuError> {
        let bits = ops::layernorm_scale_bf16(
            &self.dev,
            &self.ops,
            x,
            modulation,
            mod_row,
            col_off,
            cols,
            mod_stride,
            self.config.eps,
        )?;
        Ok(crate::cuda::blas::Bf16Activation::from_bits(
            bits,
            x.len / cols,
            cols,
        )?)
    }

    // -- op wrappers ---------------------------------------------------------

    fn upload(&self, m: &Matrix) -> Result<DevVec, DitGpuError> {
        Ok(launch::upload(&self.dev, &m.data)?)
    }

    fn download(&self, v: &DevVec) -> Result<Vec<f32>, DitGpuError> {
        Ok(launch::download(&self.dev, v)?)
    }

    /// `out[M, N] = a[M, K] * W[N, K]^T` on a host matrix, kept on the device.
    fn linear(&self, w: &DevWeight, a: &Matrix) -> Result<DevVec, DitGpuError> {
        let buf = self.upload(a)?;
        self.linear_buf(w, &buf, a.rows)
    }

    /// As [`linear`](Self::linear), with the result copied back.
    fn linear_host(&self, w: &DevWeight, a: &Matrix) -> Result<Matrix, DitGpuError> {
        let out = self.linear(w, a)?;
        let data = self.download(&out)?;
        Ok(Matrix::new(a.rows, w.rows(), data))
    }

    /// `out = a * w^T` against an activation already in bf16, so a caller
    /// projecting several weights from one input pays the conversion once.
    ///
    /// Only bf16 weights reach here: every weight the Qwen-Image-2.1 container
    /// stores is bf16.
    fn linear_bf16(
        &self,
        w: &DevWeight,
        a: &crate::cuda::blas::Bf16Activation,
    ) -> Result<DevVec, DitGpuError> {
        let m = a.m();
        let n = w.rows();
        let k = w.cols();
        if a.k() != k {
            return Err(DitError::ShapeMismatch {
                what: "linear bf16 input width".to_string(),
                expected: vec![(m * k) as u64],
                actual: vec![(m * a.k()) as u64],
            }
            .into());
        }
        let DevWeight::Bf16 { buf, .. } = w else {
            return Err(DitGpuError::UnsupportedStorage {
                tensor: "a bf16 projection".to_string(),
                scheme: w.scheme_name().to_string(),
            });
        };
        // Safety: `buf` holds the `n * k` bf16 bytes the container stored, and
        // `a` was converted from an `[m, k]` activation.
        let out = unsafe { crate::cuda::blas::gemm_bf16(&self.dev, buf, a, n) }?;
        Ok(DevVec {
            buf: out,
            len: m * n,
        })
    }

    fn linear_buf(&self, w: &DevWeight, a: &DevVec, m: usize) -> Result<DevVec, DitGpuError> {
        let (n, k) = (w.rows(), w.cols());
        if a.len != m * k {
            return Err(DitError::ShapeMismatch {
                what: "linear input width".to_string(),
                expected: vec![(m * k) as u64],
                actual: vec![a.len as u64],
            }
            .into());
        }
        let out = match w {
            DevWeight::F32 { buf, .. } => {
                launch::linear(&self.dev, &self.kernels, a, buf, None, m, n, k)?
            }
            // The linears outside the blocks (`img_in`, `txt_in`/`txt_out`,
            // the timestep and modulation linears, `norm_out`) arrive here
            // with an f32 input; a bf16 weight is converted once and run on
            // the tensor cores.
            DevWeight::Bf16 { buf, .. } => {
                // Safety: `buf` holds the `n * k` bf16 bytes the container
                // stored for this weight, and `a` holds `m * k` f32 elements
                // (checked inside `Bf16Activation::new`).
                let act = unsafe {
                    crate::cuda::blas::Bf16Activation::new(
                        &self.dev,
                        &self.kernels.f32_to_bf16_bits,
                        a,
                        m,
                        k,
                    )
                }?;
                let m16 = unsafe { crate::cuda::blas::gemm_bf16(&self.dev, buf, &act, n) }?;
                DevVec {
                    buf: m16,
                    len: m * n,
                }
            }
            DevWeight::F16 { buf, .. } => self.gemm_16bit(buf, a, m, n, k, Gemm16::F16)?,
        };
        Ok(out)
    }

    /// The GEMM for a weight stored as 16-bit floats.
    fn gemm_16bit(
        &self,
        w: &CudaSlice<u16>,
        a: &DevVec,
        m: usize,
        n: usize,
        k: usize,
        kind: Gemm16,
    ) -> Result<DevVec, RuntimeError> {
        ops::gemm_16bit(&self.dev, &self.ops, w, a, m, n, k, kind)
    }

    /// `zero_center_rms_norm_rows`, one block per row.
    fn zero_center_rmsnorm(
        &self,
        x: &DevVec,
        weight: &DevVec,
        rows: usize,
        dim: usize,
    ) -> Result<DevVec, RuntimeError> {
        ops::zero_center_rmsnorm(&self.dev, &self.ops, x, weight, rows, dim, self.config.eps)
    }

    /// GELU in place, one thread per element.
    fn gelu_tanh_inplace(&self, x: &mut DevVec) -> Result<(), RuntimeError> {
        ops::gelu_tanh_inplace(&self.dev, &self.ops, x)
    }

    /// `x + tanh(modulation[row]) * y`.
    fn add_gated(
        &self,
        x: &DevVec,
        y: &DevVec,
        modulation: &DevVec,
        mod_row: &CudaSlice<i32>,
        col_off: usize,
        cols: usize,
        mod_stride: usize,
    ) -> Result<DevVec, RuntimeError> {
        ops::add_gated(
            &self.dev, &self.ops, x, y, modulation, mod_row, col_off, cols, mod_stride,
        )
    }

    /// Every joint token's frame, height and width frequency row, laid out
    /// `[seq, head_dim]` as interleaved (cos, sin) pairs — the shape
    /// `rope_complex` reads, and the flattening of the reference's per-position
    /// `Vec<(f32, f32)>`.
    fn rope_freqs(
        &self,
        frame: &[i64],
        height: &[i64],
        width: &[i64],
    ) -> Result<Vec<f32>, DitGpuError> {
        let head_dim = self.config.attention_head_dim;
        let mut out = Vec::with_capacity(frame.len() * head_dim);
        for t in 0..frame.len() {
            for (axis, index) in [0, 1, 2].iter().zip([frame[t], height[t], width[t]]) {
                for &(cos, sin) in self.rope[*axis].row(index)? {
                    out.push(cos);
                    out.push(sin);
                }
            }
        }
        debug_assert_eq!(out.len(), frame.len() * head_dim);
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Loading helpers, mirroring `dit.rs`'s `matrix` and `vector`.
// ---------------------------------------------------------------------------

/// The tensor `name`, checked to have exactly `shape`.
fn looked_up<'a>(
    file: &'a LbiFile,
    name: &str,
    shape: &[u64],
) -> Result<&'a crate::lbi::TensorEntry, DitGpuError> {
    let entry = file
        .get(name)
        .ok_or_else(|| DitError::MissingTensor(name.to_string()))?;
    if entry.shape != shape {
        return Err(DitError::ShapeMismatch {
            what: name.to_string(),
            expected: shape.to_vec(),
            actual: entry.shape.clone(),
        }
        .into());
    }
    Ok(entry)
}

/// A projection weight for the bf16 tensor-core path: the blocks' q/k/v/out
/// and MLP projections and `proj_out` consume activations the fused kernels
/// write in bf16, so a weight stored any other way is refused here, at load,
/// rather than after a forward has run every block.
fn projection(
    dev: &CudaDevice,
    file: &LbiFile,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<DevWeight, DitGpuError> {
    let entry = looked_up(file, name, &[rows as u64, cols as u64])?;
    if entry.quant != QuantScheme::Bf16 {
        return Err(DitGpuError::UnsupportedStorage {
            tensor: name.to_string(),
            scheme: format!("{:?}", entry.quant),
        });
    }
    upload_weight(dev, file, entry, name, rows, cols)
}

/// A `[rows, cols]` linear weight, uploaded in its stored dtype.
fn weight(
    dev: &CudaDevice,
    file: &LbiFile,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<DevWeight, DitGpuError> {
    let entry = looked_up(file, name, &[rows as u64, cols as u64])?;
    upload_weight(dev, file, entry, name, rows, cols)
}

fn upload_weight(
    dev: &CudaDevice,
    file: &LbiFile,
    entry: &TensorEntry,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<DevWeight, DitGpuError> {
    let bytes = file.tensor_bytes(name).expect("entry resolved above");
    match entry.quant {
        QuantScheme::F32 => Ok(DevWeight::F32 {
            buf: launch::upload(dev, &file.read_f32(name)?)?,
            rows,
            cols,
        }),
        QuantScheme::F16 => Ok(DevWeight::F16 {
            buf: ops::upload_16bit(dev, bytes)?,
            rows,
            cols,
        }),
        QuantScheme::Bf16 => Ok(DevWeight::Bf16 {
            buf: ops::upload_16bit(dev, bytes)?,
            rows,
            cols,
        }),
        other => Err(DitGpuError::UnsupportedStorage {
            tensor: name.to_string(),
            scheme: format!("{other:?}"),
        }),
    }
}

/// A 1-D weight.
///
/// The per-head norm and the zero-centred norm are elementwise in the weight,
/// so widening it to f32 on the host is exact for the checkpoint's dtypes.
fn vector(dev: &CudaDevice, file: &LbiFile, name: &str, len: usize) -> Result<DevVec, DitGpuError> {
    let entry = looked_up(file, name, &[len as u64])?;
    match entry.quant {
        QuantScheme::F32 | QuantScheme::F16 | QuantScheme::Bf16 => {
            Ok(launch::upload(dev, &file.read_f32(name)?)?)
        }
        other => Err(DitGpuError::UnsupportedStorage {
            tensor: name.to_string(),
            scheme: format!("{other:?}"),
        }),
    }
}

// ---------------------------------------------------------------------------
// Policy, mirroring `dit.rs` because it is the specification, not a helper.
//
// The same functions live on the reference side as `crate::dit::policy`; the
// `dit-policy-check` binary compares these copies to those directly, so a drift
// between the two is caught without a GPU. The `crate::dit::policy` doc comment
// says why the duplication exists rather than a shared call.
// ---------------------------------------------------------------------------

pub mod policy {
    pub use super::{attends, rope_indices, temporal_timesteps, token_metadata};
}

fn apply_silu(m: &Matrix) -> Matrix {
    let mut out = m.clone();
    for v in out.data.iter_mut() {
        *v = silu(*v);
    }
    out
}

/// `QwenImage21TemporalTimesteps`: cos in the first half, sin in the second.
pub fn temporal_timesteps(timestep: f32, out: &mut [f32]) {
    let half = TIMESTEP_DIM / 2;
    let t = TIMESTEP_TIME_FACTOR * timestep;
    // `math.log(max_period)` is a Python float, which torch narrows against the
    // f32 arange, so the exponent is an f32 computation.
    let neg_log = -(TIMESTEP_MAX_PERIOD.ln() as f32);
    for i in 0..half {
        let arg = t * (neg_log * i as f32 / half as f32).exp();
        out[i] = arg.cos();
        out[half + i] = arg.sin();
    }
}

/// The block-causal rule `(q_idx >= kv_idx) or same_image_block`.
///
/// The forward does not call this: it admits only a text prefix followed by
/// one image block, for which the rule reduces to the text-prefix length the
/// fused attention kernel masks by. It is here so `dit-policy-check` can
/// compare the rule against the reference's.
pub fn attends(image_ids: &[i64], q: usize, kv: usize) -> bool {
    q >= kv || (image_ids[q] == image_ids[kv] && image_ids[q] >= 0)
}

/// Label every joint token with the image block it belongs to, and mark the
/// target image's tokens.
///
/// Boundaries come from `img_shapes`, not from runs of `true`: two adjacent
/// condition images form one run but must stay separate blocks.
pub fn token_metadata(
    image_pad_mask: &[bool],
    img_shapes: &[(u64, u64, u64)],
) -> Result<(Vec<i64>, Vec<bool>), DitGpuError> {
    let lengths: Vec<usize> = img_shapes
        .iter()
        .map(|&(f, h, w)| (f * h * w) as usize)
        .collect();
    let Some(&target_len) = lengths.last() else {
        return Err(mismatch("img_shapes length", 1, 0).into());
    };
    let positions: Vec<usize> = image_pad_mask
        .iter()
        .enumerate()
        .filter(|(_, &b)| b)
        .map(|(i, _)| i)
        .collect();
    let total: usize = lengths.iter().sum();
    if total != positions.len() {
        return Err(mismatch("image tokens", total, positions.len()).into());
    }

    let mut image_ids = vec![-1i64; image_pad_mask.len()];
    let mut next = 0;
    for (block, &len) in lengths.iter().enumerate() {
        for _ in 0..len {
            image_ids[positions[next]] = block as i64;
            next += 1;
        }
    }
    let mut target_token_mask = vec![false; image_pad_mask.len()];
    for &p in &positions[positions.len() - target_len..] {
        target_token_mask[p] = true;
    }
    Ok((image_ids, target_token_mask))
}

/// The frame, height and width position of every joint token, mirroring
/// `QwenImage21Rope.forward`.
///
/// Text advances one shared position on all three axes. An image block freezes
/// the frame axis at the position the preceding text reached and lays its tokens
/// on a height/width grid centred on zero, then advances the shared position by
/// `max(height, width)` — not by the token count.
pub fn rope_indices(
    img_shapes: &[(u64, u64, u64)],
    image_pad_mask: &[bool],
) -> Result<(Vec<i64>, Vec<i64>, Vec<i64>), DitGpuError> {
    let total_len = image_pad_mask.len();
    let mut frame = Vec::with_capacity(total_len);
    let mut grid_height = Vec::new();
    let mut grid_width = Vec::new();
    let mut cursor = 0usize;
    let mut position = 0i64;

    for (block, &(_, height, width)) in img_shapes.iter().enumerate() {
        let (h, w) = (height as usize, width as usize);
        let found = image_pad_mask[cursor..].iter().position(|&b| b);
        let Some(offset) = found else {
            return Err(DitError::ShapeMismatch {
                what: format!("no image token at or after position {cursor} for image block"),
                expected: vec![img_shapes.len() as u64],
                actual: vec![block as u64],
            }
            .into());
        };
        let block_start = cursor + offset;

        let text_len = block_start - cursor;
        for k in 0..text_len {
            frame.push(position + k as i64);
        }
        position += text_len as i64;

        cursor = block_start + h * w;
        for _ in 0..h * w {
            frame.push(position);
        }
        position += h.max(w) as i64;

        for row in -((h - h / 2) as i64)..(h / 2) as i64 {
            for _ in 0..w {
                grid_height.push(row);
            }
        }
        for _ in 0..h {
            for col in -((w - w / 2) as i64)..(w / 2) as i64 {
                grid_width.push(col);
            }
        }
    }
    if cursor < total_len {
        for k in 0..total_len - cursor {
            frame.push(position + k as i64);
        }
    }
    if frame.len() != total_len {
        return Err(mismatch("rope positions", total_len, frame.len()).into());
    }

    let mut height_index = frame.clone();
    let mut width_index = frame.clone();
    let mut next = 0;
    for (t, &is_image) in image_pad_mask.iter().enumerate() {
        if is_image {
            height_index[t] = grid_height[next];
            width_index[t] = grid_width[next];
            next += 1;
        }
    }
    Ok((frame, height_index, width_index))
}

fn mismatch(what: &str, expected: usize, actual: usize) -> DitError {
    DitError::ShapeMismatch {
        what: what.to_string(),
        expected: vec![expected as u64],
        actual: vec![actual as u64],
    }
}
