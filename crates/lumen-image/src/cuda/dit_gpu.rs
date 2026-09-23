//! The Qwen-Image-2.1 diffusion transformer, on the GPU, in the reference's
//! arithmetic.
//!
//! The reference runs the transformer in bf16: every tensor between two
//! operations is bf16, and each operation computes in f32 and rounds its result
//! to bf16 once, nearest even. This forward does the same. The projections are
//! cuBLAS bf16 GEMMs whose output is rounded as it is written, the kernels in
//! `dit_ops.cu` round where the reference's operations do, and the residual
//! stream is bf16. It follows [`crate::dit`], the f32 CPU form of the same
//! model, in operand order, modulation row selection, RoPE table and activation
//! arguments; the two differ by the bf16 rounding, which is the reference's.
//! `cuda-ops-check` pins each kernel to a CPU rendering of the same bf16
//! arithmetic.
//!
//! `modulation` carries one row per timestep, and `causal_condition` makes the
//! target-image tokens read the sampled-timestep row while every other token
//! reads the `t = 0` row, so the row selection lives in the AdaLN kernels
//! (`layernorm_scale`, `add_gated_layernorm_scale`). They receive the modulation as
//! `bf16(1 + scale)` and `bf16(tanh(gate))`, formed once per forward.
//!
//! Block activations stay on the device between launches: the joint sequence
//! is gathered on the device from a per-row source table built on the host
//! (index arithmetic that has to match the reference exactly), and the result
//! comes back at the end. The timestep embedding and the two modulations, a
//! few rows wide, go through the host between their linears.
//!
//! Every weight is bf16 on the device. The shipped checkpoint stores bf16 and
//! is uploaded as stored; a weight stored as f32 or f16 is rounded to bf16 at
//! load, as the reference casts its weights when it loads them, except the
//! block projections and `proj_out`, which must be stored bf16. The
//! transformer is 13.3 GiB of weights and its activations fit beside it on a
//! 32 GiB card.

use std::path::Path;

use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
use lumen_format::QuantScheme;
use lumen_runtime::cuda::ffi::CudaDevice;
use lumen_runtime::error::RuntimeError;

use super::blas::Bf16Activation;
use super::launch::{self, DevVec};
use super::ImageKernels;
use crate::dit::{DitConfig, DitError, DitForwardArgs};
use crate::lbi::{LbiError, LbiFile};
use crate::tensor::{bf16_bits, bf16_f32, bf16_round, silu, Matrix};

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

/// A linear weight resident on the device as bf16.
///
/// `rows` is the output width and `cols` the input width, kept alongside the
/// buffer rather than derived from its length: which factor is which is not
/// recoverable from an element count, and a transposed operand is a silent
/// wrong answer.
struct DevWeight {
    buf: CudaSlice<u16>,
    rows: usize,
    cols: usize,
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

/// The kernels in `dit_ops.cu`.
pub use ops::DitOps;

/// The kernels behind the ops that no existing module provides.
///
/// These live in their own module because two callers drive them: the forward
/// here, and `cuda-ops-check`, which pins each one to a CPU rendering of the
/// same bf16 arithmetic on synthetic inputs. Routing both through one launcher
/// is what makes that check speak for the forward's own arithmetic rather than
/// for a copy of it.
pub mod ops {
    use super::THREADS;
    use super::{CudaDevice, CudaFunction, CudaSlice, LaunchConfig, PushKernelArg, RuntimeError};
    use super::{DevVec, DIT_OPS_SOURCE};

    /// The kernels in `dit_ops.cu`.
    pub struct DitOps {
        zero_center_rmsnorm: CudaFunction,
        gelu_tanh: CudaFunction,
        pack_rows: CudaFunction,
        layernorm_scale: CudaFunction,
        add_gated_layernorm_scale: CudaFunction,
        swiglu: CudaFunction,
        head_norm_rope: CudaFunction,
    }

    /// Compile `dit_ops.cu` and resolve its entry points.
    pub fn load(dev: &CudaDevice) -> Result<DitOps, RuntimeError> {
        // The kernel sizes its shared row from its own define; a launcher
        // that allowed wider rows would overrun it.
        if !DIT_OPS_SOURCE.contains(&format!("#define ADD_NORM_MAX_DIM {ADD_NORM_MAX_DIM}\n")) {
            return Err(RuntimeError::Compute(format!(
                "dit_ops.cu does not define ADD_NORM_MAX_DIM as {ADD_NORM_MAX_DIM}"
            )));
        }
        let module = dev.compile_and_load(DIT_OPS_SOURCE)?;
        let get = |name: &str| -> Result<CudaFunction, RuntimeError> {
            module
                .load_function(name)
                .map_err(|e| RuntimeError::Compute(format!("load {name}: {e}")))
        };
        Ok(DitOps {
            zero_center_rmsnorm: get("zero_center_rmsnorm")?,
            gelu_tanh: get("gelu_tanh")?,
            pack_rows: get("pack_rows")?,
            layernorm_scale: get("layernorm_scale")?,
            add_gated_layernorm_scale: get("add_gated_layernorm_scale")?,
            swiglu: get("swiglu")?,
            head_norm_rope: get("head_norm_rope")?,
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

    /// A buffer every element of which the kernel about to run writes.
    fn alloc_bits(dev: &CudaDevice, len: usize) -> Result<CudaSlice<u16>, RuntimeError> {
        // Safety: each caller's kernel writes all `len` elements before any read.
        Ok(unsafe { dev.alloc_uninit::<u16>(len)? })
    }

    /// The zero-centred RMSNorm over each `dim`-wide row, one block per row.
    pub fn zero_center_rmsnorm(
        dev: &CudaDevice,
        k: &DitOps,
        x: &CudaSlice<u16>,
        weight: &DevVec,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<CudaSlice<u16>, RuntimeError> {
        if dim == 0 || x.len() != rows * dim || weight.len != dim {
            return Err(RuntimeError::Compute(format!(
                "zero_center_rmsnorm: x {} / weight {} do not match {rows}x{dim}",
                x.len(),
                weight.len
            )));
        }
        let out = alloc_bits(dev, x.len())?;
        let du = dim as u32;
        // Safety: `x` and `out` are `rows * dim`, `weight` is `dim`.
        unsafe {
            dev.stream
                .launch_builder(&k.zero_center_rmsnorm)
                .arg(x)
                .arg(&weight.buf)
                .arg(&out)
                .arg(&du)
                .arg(&eps)
                .launch(row_grid(rows))
                .map_err(|e| RuntimeError::Compute(format!("zero_center_rmsnorm: {e}")))?;
        }
        Ok(out)
    }

    /// GELU (tanh approximation), one thread per element.
    pub fn gelu_tanh(
        dev: &CudaDevice,
        k: &DitOps,
        x: &CudaSlice<u16>,
    ) -> Result<CudaSlice<u16>, RuntimeError> {
        let n = elements("gelu_tanh", &[x.len()])?;
        let out = alloc_bits(dev, x.len())?;
        // Safety: both buffers are `n` elements and the grid covers them.
        unsafe {
            dev.stream
                .launch_builder(&k.gelu_tanh)
                .arg(x)
                .arg(&out)
                .arg(&n)
                .launch(flat_grid(n))
                .map_err(|e| RuntimeError::Compute(format!("gelu_tanh: {e}")))?;
        }
        Ok(out)
    }

    /// The joint sequence: row `t` is `txt[source[t]]` when `source[t] >= 0`
    /// and `img[-source[t] - 1]` otherwise. Every source must name a row of
    /// the buffer it points into.
    pub fn pack_rows(
        dev: &CudaDevice,
        k: &DitOps,
        txt: &CudaSlice<u16>,
        img: &CudaSlice<u16>,
        source: &[i32],
        cols: usize,
    ) -> Result<CudaSlice<u16>, RuntimeError> {
        if cols == 0 || txt.len() % cols != 0 || img.len() % cols != 0 {
            return Err(RuntimeError::Compute(format!(
                "pack_rows: txt {} and img {} are not whole rows of {cols}",
                txt.len(),
                img.len()
            )));
        }
        let (text_rows, image_rows) = (txt.len() / cols, img.len() / cols);
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
        let out = alloc_bits(dev, rows * cols)?;
        let (rows_u, cols_u) = (rows as u32, cols as u32);
        // Safety: every source index was checked against the two row counts
        // above, and the grid covers exactly `rows * cols`.
        unsafe {
            dev.stream
                .launch_builder(&k.pack_rows)
                .arg(txt)
                .arg(img)
                .arg(&g_source)
                .arg(&out)
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

    /// The geometry the two AdaLN kernels share: `x_len` is whole rows of
    /// `cols`, `mod_row` names one modulation row per row of `x`, and the
    /// chunk `col_off..col_off + cols` lies inside a modulation row of
    /// `mod_stride`. Returns the row count. The values in `mod_row` are the
    /// caller's to keep inside `modulation`; the forward builds them from its
    /// own timestep count.
    fn adaln_rows(
        name: &str,
        x_len: usize,
        modulation: &CudaSlice<u16>,
        mod_row: &CudaSlice<i32>,
        col_off: usize,
        cols: usize,
        mod_stride: usize,
    ) -> Result<usize, RuntimeError> {
        let whole_rows = cols != 0 && x_len % cols == 0 && mod_row.len() == x_len / cols;
        let chunk_inside = mod_stride != 0
            && col_off
                .checked_add(cols)
                .is_some_and(|end| end <= mod_stride)
            && modulation.len() % mod_stride == 0;
        if !whole_rows || !chunk_inside {
            return Err(RuntimeError::Compute(format!(
                "{name}: x {x_len} / mod_row {} / modulation {} do not fit cols {cols} at \
                 {col_off} of stride {mod_stride}",
                mod_row.len(),
                modulation.len()
            )));
        }
        elements(name, &[x_len])?;
        Ok(x_len / cols)
    }

    /// `bf16(bf16(layernorm(x)) * one_plus[row])`, the input a projection
    /// takes; `one_plus` holds `bf16(1 + scale)`.
    #[allow(clippy::too_many_arguments)]
    pub fn layernorm_scale(
        dev: &CudaDevice,
        k: &DitOps,
        x: &CudaSlice<u16>,
        one_plus: &CudaSlice<u16>,
        mod_row: &CudaSlice<i32>,
        col_off: usize,
        cols: usize,
        mod_stride: usize,
        eps: f32,
    ) -> Result<CudaSlice<u16>, RuntimeError> {
        let rows = adaln_rows(
            "layernorm_scale",
            x.len(),
            one_plus,
            mod_row,
            col_off,
            cols,
            mod_stride,
        )?;
        let out = alloc_bits(dev, x.len())?;
        let (cols_u, off_u, stride_u) = (cols as u32, col_off as u32, mod_stride as u32);
        // Safety: buffers match the kernel's declared shapes, and the grid is
        // one block per row.
        unsafe {
            dev.stream
                .launch_builder(&k.layernorm_scale)
                .arg(x)
                .arg(one_plus)
                .arg(mod_row)
                .arg(&out)
                .arg(&cols_u)
                .arg(&off_u)
                .arg(&stride_u)
                .arg(&eps)
                .launch(row_grid(rows))
                .map_err(|e| RuntimeError::Compute(format!("layernorm_scale: {e}")))?;
        }
        Ok(out)
    }

    /// The widest row [`add_gated_layernorm_scale`] stages in shared memory:
    /// `ADD_NORM_MAX_DIM` in `dit_ops.cu`, which [`load`] checks.
    pub const ADD_NORM_MAX_DIM: usize = 4096;

    /// `x = bf16(x + bf16(tanh_gate[row] * y))` in place on the residual
    /// stream, then [`layernorm_scale`] of the updated `x` with `one_plus`:
    /// the next projection's input. `tanh_gate` holds `bf16(tanh(gate))`.
    #[allow(clippy::too_many_arguments)]
    pub fn add_gated_layernorm_scale(
        dev: &CudaDevice,
        k: &DitOps,
        x: &mut CudaSlice<u16>,
        y: &CudaSlice<u16>,
        tanh_gate: &CudaSlice<u16>,
        (gate_off, gate_stride): (usize, usize),
        one_plus: &CudaSlice<u16>,
        (scale_off, scale_stride): (usize, usize),
        mod_row: &CudaSlice<i32>,
        cols: usize,
        eps: f32,
    ) -> Result<CudaSlice<u16>, RuntimeError> {
        const NAME: &str = "add_gated_layernorm_scale";
        let rows = adaln_rows(
            NAME,
            x.len(),
            tanh_gate,
            mod_row,
            gate_off,
            cols,
            gate_stride,
        )?;
        adaln_rows(
            NAME,
            x.len(),
            one_plus,
            mod_row,
            scale_off,
            cols,
            scale_stride,
        )?;
        if y.len() != x.len() {
            return Err(RuntimeError::Compute(format!(
                "{NAME}: y has {} elements, x has {}",
                y.len(),
                x.len()
            )));
        }
        if cols > ADD_NORM_MAX_DIM {
            return Err(RuntimeError::Compute(format!(
                "{NAME}: rows of {cols} exceed the {ADD_NORM_MAX_DIM} the kernel stages"
            )));
        }
        // The kernel moves eight elements (16 bytes) at a time.
        if [cols, gate_off, gate_stride, scale_off, scale_stride]
            .iter()
            .any(|v| v % 8 != 0)
        {
            return Err(RuntimeError::Compute(format!(
                "{NAME}: width {cols}, offsets {gate_off}/{scale_off} and strides \
                 {gate_stride}/{scale_stride} must be multiples of 8"
            )));
        }
        let out = alloc_bits(dev, x.len())?;
        let cols_u = cols as u32;
        let (gate_off_u, gate_stride_u) = (gate_off as u32, gate_stride as u32);
        let (scale_off_u, scale_stride_u) = (scale_off as u32, scale_stride as u32);
        // Safety: buffers match the kernel's declared shapes, rows fit its
        // shared memory, every access is 16-byte aligned (the allocations are,
        // and every offset is a multiple of eight elements), and the grid is
        // one block per row.
        unsafe {
            dev.stream
                .launch_builder(&k.add_gated_layernorm_scale)
                .arg(x)
                .arg(y)
                .arg(tanh_gate)
                .arg(one_plus)
                .arg(mod_row)
                .arg(&out)
                .arg(&cols_u)
                .arg(&gate_off_u)
                .arg(&gate_stride_u)
                .arg(&scale_off_u)
                .arg(&scale_stride_u)
                .arg(&eps)
                .launch(row_grid(rows))
                .map_err(|e| RuntimeError::Compute(format!("{NAME}: {e}")))?;
        }
        Ok(out)
    }

    /// `bf16(bf16(silu(gate)) * up)`, the down projection's input.
    pub fn swiglu(
        dev: &CudaDevice,
        k: &DitOps,
        gate: &CudaSlice<u16>,
        up: &CudaSlice<u16>,
    ) -> Result<CudaSlice<u16>, RuntimeError> {
        if gate.len() != up.len() {
            return Err(RuntimeError::Compute(format!(
                "swiglu: gate has {} elements, up has {}",
                gate.len(),
                up.len()
            )));
        }
        let n = elements("swiglu", &[gate.len()])?;
        let out = alloc_bits(dev, gate.len())?;
        // Safety: all three buffers are `n` elements and the grid covers them,
        // eight per thread.
        unsafe {
            dev.stream
                .launch_builder(&k.swiglu)
                .arg(gate)
                .arg(up)
                .arg(&out)
                .arg(&n)
                .launch(flat_grid(n.div_ceil(8)))
                .map_err(|e| RuntimeError::Compute(format!("swiglu: {e}")))?;
        }
        Ok(out)
    }

    /// Q or K from its projection to the attention operand: the per-head
    /// RMSNorm and the rotation, in one warp per (token, head) row.
    #[allow(clippy::too_many_arguments)]
    pub fn head_norm_rope(
        dev: &CudaDevice,
        k: &DitOps,
        x: &CudaSlice<u16>,
        weight: &DevVec,
        freqs: &DevVec,
        seq: usize,
        heads: usize,
        eps: f32,
    ) -> Result<CudaSlice<u16>, RuntimeError> {
        const HEAD_DIM: usize = crate::cuda::attention::FLASH_HEAD_DIM;
        let total = elements("head_norm_rope", &[seq, heads, HEAD_DIM])? as usize;
        if x.len() != total || weight.len != HEAD_DIM || freqs.len != seq * HEAD_DIM {
            return Err(RuntimeError::Compute(format!(
                "head_norm_rope: x {} / weight {} / freqs {} do not match {seq}x{heads}x{HEAD_DIM}",
                x.len(),
                weight.len,
                freqs.len
            )));
        }
        // One warp per row of 128: a quarter of the elements as threads.
        let threads = (total / 4) as u32;
        let out = alloc_bits(dev, total)?;
        let (seq_u, heads_u) = (seq as u32, heads as u32);
        // Safety: buffers match the kernel's declared shapes; the grid is one
        // warp per (token, head) row.
        unsafe {
            dev.stream
                .launch_builder(&k.head_norm_rope)
                .arg(x)
                .arg(&weight.buf)
                .arg(&freqs.buf)
                .arg(&out)
                .arg(&seq_u)
                .arg(&heads_u)
                .arg(&eps)
                .launch(flat_grid(threads))
                .map_err(|e| RuntimeError::Compute(format!("head_norm_rope: {e}")))?;
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
        Self::load_with(&LbiFile::open(lbi)?, dev, DitConfig::qwen_image_2_1())
    }

    /// Load from an open container against an explicit architecture.
    pub fn load_with(
        file: &LbiFile,
        dev: &CudaDevice,
        config: DitConfig,
    ) -> Result<Self, DitGpuError> {
        let own = CudaDevice::new(dev.ctx.ordinal())?;
        let kernels = ImageKernels::load(&own)?;
        let ops = ops::load(&own)?;
        let hidden = config.inner_dim();
        let mlp = config.mlp_hidden();
        // The attention kernels are tiled for the model's 128-wide heads
        // (`flash_attn.cu`, `head_norm_rope`); a narrower or wider head
        // is refused here rather than read past a tile.
        if config.attention_head_dim != crate::cuda::attention::FLASH_HEAD_DIM {
            return Err(DitError::ShapeMismatch {
                what: "attention head width".to_string(),
                expected: vec![crate::cuda::attention::FLASH_HEAD_DIM as u64],
                actual: vec![config.attention_head_dim as u64],
            }
            .into());
        }
        // The fused residual update and norm stage one row of the hidden
        // width in shared memory.
        if hidden > ops::ADD_NORM_MAX_DIM {
            return Err(DitError::ShapeMismatch {
                what: format!("hidden width (at most {})", ops::ADD_NORM_MAX_DIM),
                expected: vec![ops::ADD_NORM_MAX_DIM as u64],
                actual: vec![hidden as u64],
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
                to_q: projection(&own, file, &format!("{p}.attn.to_q.weight"), hidden, hidden)?,
                to_k: projection(&own, file, &format!("{p}.attn.to_k.weight"), hidden, hidden)?,
                to_v: projection(&own, file, &format!("{p}.attn.to_v.weight"), hidden, hidden)?,
                to_out: projection(
                    &own,
                    file,
                    &format!("{p}.attn.to_out.0.weight"),
                    hidden,
                    hidden,
                )?,
                norm_q: vector(
                    &own,
                    file,
                    &format!("{p}.attn.norm_q.weight"),
                    config.attention_head_dim,
                )?,
                norm_k: vector(
                    &own,
                    file,
                    &format!("{p}.attn.norm_k.weight"),
                    config.attention_head_dim,
                )?,
                mlp_gate: projection(
                    &own,
                    file,
                    &format!("{p}.img_mlp.gate_layer.weight"),
                    mlp,
                    hidden,
                )?,
                mlp_proj: projection(&own, file, &format!("{p}.img_mlp.proj.weight"), mlp, hidden)?,
                mlp_out: projection(&own, file, &format!("{p}.img_mlp.out.weight"), hidden, mlp)?,
            });
        }

        Ok(Self {
            img_in: weight(&own, file, "img_in.weight", hidden, config.in_channels)?,
            text_norm: vector(&own, file, "txt_in.text_norm.weight", config.context_in_dim)?,
            txt_in: weight(
                &own,
                file,
                "txt_in.in_layer.weight",
                hidden,
                config.context_in_dim,
            )?,
            txt_out: weight(&own, file, "txt_in.out_layer.weight", hidden, hidden)?,
            time_linear_1: weight(
                &own,
                file,
                "time_text_embed.timestep_embedder.linear_1.weight",
                hidden,
                TIMESTEP_DIM,
            )?,
            time_linear_2: weight(
                &own,
                file,
                "time_text_embed.timestep_embedder.linear_2.weight",
                hidden,
                hidden,
            )?,
            modulation: weight(&own, file, "modulation.1.weight", 4 * hidden, hidden)?,
            norm_out: weight(&own, file, "norm_out.linear.weight", hidden, hidden)?,
            proj_out: projection(&own, file, "proj_out.weight", config.out_channels, hidden)?,
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
    /// The inputs and the timestep are rounded to bf16 on entry, as the
    /// reference casts them to its dtype. The result has one row per joint
    /// token, text included, as bf16 values widened to f32; the caller takes
    /// the trailing `target_tokens` rows, as the pipeline's
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

        let img = self.linear(&self.img_in, &self.upload(args.hidden_states)?)?;
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
        if img.m() != image_tokens {
            return Err(mismatch("packed latent rows", image_tokens, img.m()).into());
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

        // The model takes the timestep in its own dtype. With `causal_condition`
        // the modulation carries an extra `t = 0` row, which every token outside
        // the target image reads.
        let timestep = bf16_round(args.timestep);
        let timesteps: &[f32] = if cfg.causal_condition {
            &[timestep, 0.0]
        } else {
            std::slice::from_ref(&timestep)
        };
        let silu_temb = silu_bits(&self.timestep_embedding(timesteps)?);
        let modulation = self.linear_rows(&self.modulation, &silu_temb, timesteps.len())?;
        let modulation = self.dev.htod_copy(&adaln_factors(&modulation, hidden))?;
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
        // `QwenImage21AdaLayerNormContinuous`: scale only, read from `temb`
        // rather than from the shared modulation.
        let scale = self.linear_rows(&self.norm_out, &silu_temb, timesteps.len())?;
        let one_plus: Vec<u16> = scale
            .iter()
            .map(|&s| bf16_bits(1.0 + bf16_f32(s)))
            .collect();
        let norm_out_scale = self.dev.htod_copy(&one_plus)?;

        let mut x = ops::pack_rows(&self.dev, &self.ops, &txt.bits, &img.bits, &source, hidden)?;
        // The first norm feeds block 0's attention, or `norm_out` directly
        // when there are no blocks.
        let (first_scale, first_off, first_stride) = if self.blocks.is_empty() {
            (&norm_out_scale, NORM_OUT_SCALE, hidden)
        } else {
            (&modulation, MOD_ATTN_SCALE * hidden, mod_stride)
        };
        let mut normed =
            self.normed_input(&x, first_scale, &g_mod_row, first_off, hidden, first_stride)?;
        for (i, block) in self.blocks.iter().enumerate() {
            let attn = self.attention(block, &normed, seq, heads, &g_freqs, text_count)?;
            normed = self.add_gated_normed(
                &mut x,
                &attn,
                (&modulation, MOD_ATTN_GATE * hidden),
                (&modulation, MOD_MLP_SCALE * hidden, mod_stride),
                &g_mod_row,
                mod_stride,
            )?;
            let mlp = self.feed_forward(block, &normed, seq)?;
            // The next norm is the following block's attention input, or
            // `norm_out` after the last block.
            let next_scale = if i + 1 < self.blocks.len() {
                (&modulation, MOD_ATTN_SCALE * hidden, mod_stride)
            } else {
                (&norm_out_scale, NORM_OUT_SCALE, hidden)
            };
            normed = self.add_gated_normed(
                &mut x,
                &mlp,
                (&modulation, MOD_MLP_GATE * hidden),
                next_scale,
                &g_mod_row,
                mod_stride,
            )?;
        }

        let out = self.linear(&self.proj_out, &normed)?;
        let bits = self.dev.dtoh_copy(&out.bits)?;
        self.dev.synchronize()?;
        Ok(Matrix::new(
            seq,
            cfg.out_channels,
            bits.into_iter().map(bf16_f32).collect(),
        ))
    }

    // -- the pieces of the forward ------------------------------------------

    /// `QwenImage21TextProjection`: zero-centred RMSNorm, linear, GELU, linear.
    fn text_projection(
        &self,
        encoder_hidden_states: &Matrix,
    ) -> Result<Bf16Activation, DitGpuError> {
        let rows = encoder_hidden_states.rows;
        let dim = encoder_hidden_states.cols;
        let x = self.upload(encoder_hidden_states)?;
        let normed = ops::zero_center_rmsnorm(
            &self.dev,
            &self.ops,
            &x.bits,
            &self.text_norm,
            rows,
            dim,
            self.config.eps,
        )?;
        let h = self.linear(&self.txt_in, &Bf16Activation::from_bits(normed, rows, dim)?)?;
        let g = ops::gelu_tanh(&self.dev, &self.ops, &h.bits)?;
        self.linear(&self.txt_out, &Bf16Activation::from_bits(g, rows, h.k())?)
    }

    /// The sinusoidal timestep embedding followed by `TimestepEmbedding`,
    /// whose `forward` puts the activation between the two linears: `temb`,
    /// one bf16 row per timestep.
    ///
    /// The activation is `silu(linear_1(proj))` — SiLU sits *after* `linear_1`
    /// and *before* `linear_2` in the reference. Applying it to the sinusoidal
    /// `proj` as well is wrong and leaves no symptom a shape check can catch:
    /// the result is still finite and still the right shape.
    fn timestep_embedding(&self, timesteps: &[f32]) -> Result<Vec<u16>, DitGpuError> {
        let mut proj = vec![0.0f32; timesteps.len() * TIMESTEP_DIM];
        for (row, &t) in proj.chunks_mut(TIMESTEP_DIM).zip(timesteps) {
            temporal_timesteps(t, row);
        }
        let proj: Vec<u16> = proj.into_iter().map(bf16_bits).collect();
        let h = self.linear_rows(&self.time_linear_1, &proj, timesteps.len())?;
        self.linear_rows(&self.time_linear_2, &silu_bits(&h), timesteps.len())
    }

    /// One block's attention, up to and including `to_out`.
    fn attention(
        &self,
        block: &GpuBlock,
        act: &Bf16Activation,
        seq: usize,
        heads: usize,
        freqs: &DevVec,
        text_count: usize,
    ) -> Result<Bf16Activation, DitGpuError> {
        // The three projections read the same normed input.
        let q = self.linear(&block.to_q, act)?;
        let k = self.linear(&block.to_k, act)?;
        let v = self.linear(&block.to_v, act)?;

        let eps = self.config.eps;
        let q = ops::head_norm_rope(
            &self.dev,
            &self.ops,
            &q.bits,
            &block.norm_q,
            freqs,
            seq,
            heads,
            eps,
        )?;
        let k = ops::head_norm_rope(
            &self.dev,
            &self.ops,
            &k.bits,
            &block.norm_k,
            freqs,
            seq,
            heads,
            eps,
        )?;
        // Safety: q, k and v are the `[seq, heads, 128]` bf16 operands the
        // fused norm and the projection just produced.
        let out = unsafe {
            crate::cuda::attention::fused_block_causal_attention(
                &self.dev,
                &self.kernels,
                &q,
                &k,
                &v.bits,
                text_count,
                seq,
                heads,
            )
        }?;
        self.linear(&block.to_out, &out)
    }

    /// `QwenImage21SwiGLUFeedForward`: the gate branch goes through SiLU, the
    /// projection branch does not.
    fn feed_forward(
        &self,
        block: &GpuBlock,
        act: &Bf16Activation,
        seq: usize,
    ) -> Result<Bf16Activation, DitGpuError> {
        let gate = self.linear(&block.mlp_gate, act)?;
        let proj = self.linear(&block.mlp_proj, act)?;
        let hidden = ops::swiglu(&self.dev, &self.ops, &gate.bits, &proj.bits)?;
        let hidden = Bf16Activation::from_bits(hidden, seq, block.mlp_out.cols)?;
        self.linear(&block.mlp_out, &hidden)
    }

    /// The residual update `x += tanh_gate · y` followed by the next normed,
    /// modulated input: `gate` is the modulation and the offset of its gate
    /// chunk, `scale` the `1 + scale` factors, their offset and row width.
    fn add_gated_normed(
        &self,
        x: &mut CudaSlice<u16>,
        y: &Bf16Activation,
        (gate, gate_off): (&CudaSlice<u16>, usize),
        (scale, scale_off, scale_stride): (&CudaSlice<u16>, usize, usize),
        mod_row: &CudaSlice<i32>,
        mod_stride: usize,
    ) -> Result<Bf16Activation, DitGpuError> {
        let cols = self.config.inner_dim();
        let bits = ops::add_gated_layernorm_scale(
            &self.dev,
            &self.ops,
            x,
            &y.bits,
            gate,
            (gate_off, mod_stride),
            scale,
            (scale_off, scale_stride),
            mod_row,
            cols,
            self.config.eps,
        )?;
        Ok(Bf16Activation::from_bits(bits, x.len() / cols, cols)?)
    }

    /// `bf16(bf16(layernorm(x)) * one_plus[row])`: a block's normed, modulated
    /// input, ready for its projections.
    fn normed_input(
        &self,
        x: &CudaSlice<u16>,
        one_plus: &CudaSlice<u16>,
        mod_row: &CudaSlice<i32>,
        col_off: usize,
        cols: usize,
        mod_stride: usize,
    ) -> Result<Bf16Activation, DitGpuError> {
        let bits = ops::layernorm_scale(
            &self.dev,
            &self.ops,
            x,
            one_plus,
            mod_row,
            col_off,
            cols,
            mod_stride,
            self.config.eps,
        )?;
        Ok(Bf16Activation::from_bits(bits, x.len() / cols, cols)?)
    }

    // -- op wrappers ---------------------------------------------------------

    /// A host matrix as a bf16 activation, each value rounded to nearest even
    /// (exact for the bf16 values the pipeline passes).
    fn upload(&self, m: &Matrix) -> Result<Bf16Activation, DitGpuError> {
        let bits: Vec<u16> = m.data.iter().map(|&v| bf16_bits(v)).collect();
        Ok(Bf16Activation::from_bits(
            self.dev.htod_copy(&bits)?,
            m.rows,
            m.cols,
        )?)
    }

    /// `a * w^T` with the output rounded to bf16: a bf16 `nn.Linear`.
    fn linear(&self, w: &DevWeight, a: &Bf16Activation) -> Result<Bf16Activation, DitGpuError> {
        if a.k() != w.cols {
            return Err(DitError::ShapeMismatch {
                what: "linear input width".to_string(),
                expected: vec![w.cols as u64],
                actual: vec![a.k() as u64],
            }
            .into());
        }
        // Safety: `w.buf` holds the `rows * cols` bf16 weight, checked at load.
        Ok(unsafe { crate::cuda::blas::gemm_bf16_out(&self.dev, &w.buf, a, w.rows) }?)
    }

    /// [`linear`](Self::linear) on a few host rows of bf16 bits, the result
    /// copied back: the timestep embedding and the modulations.
    fn linear_rows(&self, w: &DevWeight, rows: &[u16], m: usize) -> Result<Vec<u16>, DitGpuError> {
        let a = Bf16Activation::from_bits(self.dev.htod_copy(rows)?, m, rows.len() / m)?;
        let out = self.linear(w, &a)?;
        let bits = self.dev.dtoh_copy(&out.bits)?;
        self.dev.synchronize()?;
        Ok(bits)
    }

    /// Every joint token's frame, height and width frequency row, laid out
    /// `[seq, head_dim]` as interleaved (cos, sin) pairs — the shape
    /// `head_norm_rope` reads, and the flattening of the reference's
    /// per-position `Vec<(f32, f32)>`.
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

/// `bf16(silu(x))` of bf16 values, as `nn.SiLU` on a bf16 tensor computes it.
fn silu_bits(x: &[u16]) -> Vec<u16> {
    x.iter().map(|&b| bf16_bits(silu(bf16_f32(b)))).collect()
}

/// The shared modulation `[rows, 4 * hidden]` in the form the AdaLN kernels
/// multiply by: `bf16(1 + scale)` for the two scale chunks and
/// `bf16(tanh(gate))` for the two gate chunks, each expression rounded as the
/// reference rounds it before multiplying.
fn adaln_factors(modulation: &[u16], hidden: usize) -> Vec<u16> {
    modulation
        .iter()
        .enumerate()
        .map(|(i, &b)| {
            let m = bf16_f32(b);
            match (i % (4 * hidden)) / hidden {
                MOD_ATTN_SCALE | MOD_MLP_SCALE => bf16_bits(1.0 + m),
                _ => bf16_bits(m.tanh()),
            }
        })
        .collect()
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

/// A projection weight inside the blocks, or `proj_out`, which must be stored
/// bf16: a weight stored any other way is refused here, at load, rather than
/// after a forward has run every block.
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
    weight(dev, file, name, rows, cols)
}

/// A `[rows, cols]` linear weight as bf16: uploaded as stored when it is bf16,
/// otherwise rounded to it, as the reference casts every weight to the
/// transformer's dtype when it loads.
fn weight(
    dev: &CudaDevice,
    file: &LbiFile,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<DevWeight, DitGpuError> {
    let entry = looked_up(file, name, &[rows as u64, cols as u64])?;
    let buf = match entry.quant {
        QuantScheme::Bf16 => {
            ops::upload_16bit(dev, file.tensor_bytes(name).expect("entry resolved above"))?
        }
        QuantScheme::F32 | QuantScheme::F16 => {
            let bits: Vec<u16> = file.read_f32(name)?.into_iter().map(bf16_bits).collect();
            dev.htod_copy(&bits)?
        }
        other => {
            return Err(DitGpuError::UnsupportedStorage {
                tensor: name.to_string(),
                scheme: format!("{other:?}"),
            })
        }
    };
    Ok(DevWeight { buf, rows, cols })
}

/// A 1-D weight, rounded to bf16 like every other weight and widened to f32
/// for the kernels (exact).
fn vector(dev: &CudaDevice, file: &LbiFile, name: &str, len: usize) -> Result<DevVec, DitGpuError> {
    let entry = looked_up(file, name, &[len as u64])?;
    match entry.quant {
        QuantScheme::F32 | QuantScheme::F16 | QuantScheme::Bf16 => {
            let values: Vec<f32> = file.read_f32(name)?.into_iter().map(bf16_round).collect();
            Ok(launch::upload(dev, &values)?)
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
