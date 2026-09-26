//! Measured cuBLASLt algorithm selection for the native prefill's plans, each choice verified before it
//! serves, and cached on disk per library identity.
//!
//! # Selection
//!
//! For each plan (a projection shape at one row bucket), up to [`CANDIDATES`] heuristic algorithms
//! that fit the workspace and reduce split-K partials only in the F32 compute type are timed: weights
//! rotating over the copies the caller supplies, the stream held by a spin kernel so every timed launch
//! is enqueued before the GPU reaches it, CUDA events around each launch, median of [`TIMED`] launches
//! after [`WARMUP`]. The fastest candidate that passes verification is chosen; a candidate whose
//! configuration reads back any other reduction scheme is never timed.
//!
//! # Verification
//!
//! The candidate's output, written into a NaN-filled buffer, is compared element by element with an
//! F32 cuBLAS SGEMM (pedantic F32 compute, no reduced-precision tensor math) of the operands decoded
//! by the F32 prefill's dequantization kernels, within the bound in `shaders/cublaslt_check.cu`.
//! One violation, or one element left unwritten, rejects the candidate.
//!
//! # Cache
//!
//! The chosen table is written to `<lumen cache>/cublaslt/<sha256 of identity>.txt`: a header naming
//! the identity (library path and version, device name and compute capability, driver version,
//! workspace cap, the descriptor configuration and the plan definition), then one line per plan
//! recording the raw algorithm, its nine configuration attributes, workspace size, waves, the four
//! layouts and the operand alignment,
//! its median time and its worst error-to-bound ratio, then a SHA-256 of all of it. A later load with
//! the same identity reads the table instead of selecting; a file whose identity, plan set, layouts or
//! checksum differ is ignored and selection runs again, so no algorithm serves on a library or device
//! it was not verified on.

use super::cublaslt::{
    LtAlgo, LtGemmShape, LtHandle, LtInput, LtLibrary, LtMatmul, LtOperands,
    ALGO_CONFIG_REDUCTION_SCHEME, COMPUTE_32F, OPERAND_ALIGNMENT, OP_N, OP_T,
    REDUCTION_COMPUTE_TYPE, REDUCTION_NONE, R_16BF, R_32F, R_4F_E2M1, R_8F_E4M3, SCALE_VEC16_UE4M3,
};
use super::ffi::CudaDevice;
use super::native_prefill_gemm::{bucket_rows, GemmShape, BUCKETS, MAX_ROWS, ROW_STEP, SHAPES};
use super::native_prefill_weights::swizzle_block_scales;
use super::shaders;
use crate::error::RuntimeError;
use cudarc::cublas::sys as cublas_sys;
use cudarc::driver::{CudaFunction, CudaSlice, DevicePtr, LaunchConfig, PushKernelArg};
use std::path::PathBuf;
use std::time::Instant;

/// Heuristic candidates timed per plan.
pub const CANDIDATES: usize = 8;
/// Untimed launches of each candidate before timing.
pub const WARMUP: usize = 3;
/// Timed launches of each candidate.
pub const TIMED: usize = 24;

/// One weight matrix as a plan's operand. `plane` is the device address of its planes (codes first,
/// then the linear block scales and the F32 global scale for NVFP4, or the F32 weight scale for
/// FP8), which the dequantization kernels decode for verification. `scale` is what the GEMM reads as
/// the weight's scale: the swizzled block scales (NVFP4) or the address of the weight scale inside
/// the planes (FP8). `unit_alpha` is the GEMM's alpha for an activation of unit scale: the global
/// scale for NVFP4 (cuBLASLt applies block scales only), 1 for FP8.
#[derive(Clone, Copy, Debug)]
pub struct WeightOperand {
    pub plane: u64,
    pub scale: u64,
    pub unit_alpha: f32,
}

/// A plan's chosen algorithm and what was measured about it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Selected {
    pub algo: LtAlgo,
    pub workspace: u64,
    pub waves: f32,
    /// The nine configuration attributes, in [`super::cublaslt::ALGO_CONFIG`] order.
    pub config: [u64; 9],
    pub median_us: f32,
    /// The largest error-to-bound ratio of its verification.
    pub worst: f32,
    /// Candidates the heuristic offered within the constraints.
    pub candidates: u32,
}

/// What happened to one candidate of a plan.
#[derive(Clone, Debug)]
pub struct CandidateOutcome {
    pub config: [u64; 9],
    /// `None` for a candidate rejected before timing.
    pub median_us: Option<f32>,
    pub verdict: Verdict,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Verdict {
    Selected,
    /// Verification found this many violations.
    Failed(u64),
    /// A faster candidate passed first.
    NotVerified,
    /// Its reduction scheme reads back as this value, outside {NONE, COMPUTE_TYPE}.
    ReductionRejected(u64),
}

/// A plan's selection with every candidate's outcome.
#[derive(Clone, Debug)]
pub struct PlanSelection {
    pub shape: &'static str,
    pub rows: usize,
    pub selected: Selected,
    pub outcomes: Vec<CandidateOutcome>,
}

/// A device activation: its planes (codes, then linear block scales and a global scale for NVFP4, or
/// the F32 scale for FP8), and for NVFP4 the swizzled block scales the GEMM reads.
pub struct Activation {
    pub input: LtInput,
    pub rows: usize,
    pub k: usize,
    pub plane: CudaSlice<u8>,
    pub swizzled: Option<CudaSlice<u8>>,
}

impl Activation {
    /// Upload `rows x k` codes (two E2M1 per byte for NVFP4) with `linear_scales` (`rows x k / 16`
    /// UE4M3, NVFP4 only) and `scale` (the NVFP4 global, or the FP8 per-tensor scale).
    pub fn upload(
        device: &CudaDevice,
        input: LtInput,
        rows: usize,
        k: usize,
        codes: &[u8],
        linear_scales: &[u8],
        scale: f32,
    ) -> Result<Self, RuntimeError> {
        let mut plane = codes.to_vec();
        plane.extend_from_slice(linear_scales);
        plane.extend_from_slice(&scale.to_le_bytes());
        let swizzled = match input {
            LtInput::Nvfp4 => {
                Some(device.htod_copy(&swizzle_block_scales(linear_scales, rows, k / 16))?)
            }
            LtInput::Fp8 => None,
        };
        Ok(Self {
            input,
            rows,
            k,
            plane: device.htod_copy(&plane)?,
            swizzled,
        })
    }

    /// Random codes over the whole format (no NaN codes), NVFP4 block scales between 2^-2 and 2^2
    /// under a unit global scale, and an FP8 scale of 2^-6.
    pub fn synthetic(
        device: &CudaDevice,
        input: LtInput,
        rows: usize,
        k: usize,
        seed: u64,
    ) -> Result<Self, RuntimeError> {
        let mut s = seed;
        let mut next = move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) as u8
        };
        match input {
            LtInput::Nvfp4 => {
                let codes: Vec<u8> = (0..rows * k / 2).map(|_| next()).collect();
                let scales: Vec<u8> = (0..rows * k / 16).map(|_| 0x28 + next() % 0x21).collect();
                Self::upload(device, input, rows, k, &codes, &scales, 1.0)
            }
            LtInput::Fp8 => {
                let codes: Vec<u8> = (0..rows * k)
                    .map(|_| match next() {
                        c if c & 0x7F == 0x7F => c - 1,
                        c => c,
                    })
                    .collect();
                Self::upload(device, input, rows, k, &codes, &[], 1.0 / 64.0)
            }
        }
    }

    pub fn x(&self, device: &CudaDevice) -> u64 {
        self.plane.device_ptr(&device.stream).0
    }

    /// The scale the GEMM reads for this activation.
    pub fn x_scale(&self, device: &CudaDevice) -> u64 {
        match &self.swizzled {
            Some(s) => s.device_ptr(&device.stream).0,
            None => self.x(device) + (self.rows * self.k) as u64,
        }
    }
}

/// The verification of one result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Check {
    pub violations: u64,
    pub worst: f32,
}

/// Checks GEMM outputs of one shape against F32 references of one weight and one activation.
pub struct Verifier {
    n: usize,
    k: usize,
    w: CudaSlice<f32>,
    w_abs: CudaSlice<f32>,
    x: CudaSlice<f32>,
    x_abs: CudaSlice<f32>,
    r: CudaSlice<f32>,
    mag: CudaSlice<f32>,
    rows_ready: usize,
    counts: CudaSlice<u32>,
    check_fn: CudaFunction,
}

impl Verifier {
    /// Decode the `n x k` weight at `w_plane` and the activation `x` (all its rows) to F32.
    ///
    /// # Safety
    /// `w_plane` must address the live device planes of an `n x k` weight of `x`'s encoding, and
    /// `x.k` must equal `k` with `x`'s planes holding its `x.rows x k` values.
    pub unsafe fn new(
        device: &CudaDevice,
        n: usize,
        k: usize,
        w_plane: u64,
        x: &Activation,
    ) -> Result<Self, RuntimeError> {
        let check = device.compile_and_load(shaders::CUBLASLT_CHECK_KERNEL_SOURCE)?;
        let func = |m: &std::sync::Arc<cudarc::driver::CudaModule>, name: &str| {
            m.load_function(name)
                .map_err(|e| RuntimeError::Compute(format!("load {name}: {e}")))
        };
        let abs_fn = func(&check, "lumen_lt_abs")?;
        let decode = |plane: u64, elements: usize| -> Result<CudaSlice<f32>, RuntimeError> {
            // SAFETY: every element is written by the decode below before it is read.
            let mut out = unsafe { device.alloc_uninit::<f32>(elements)? };
            let (source, name, per_thread) = match x.input {
                LtInput::Nvfp4 => (
                    shaders::DEQUANT_NVFP4_KERNEL_SOURCE,
                    "dequant_nvfp4_to_f32",
                    4,
                ),
                LtInput::Fp8 => (shaders::DEQUANT_FP8_KERNEL_SOURCE, "dequant_fp8_to_f32", 1),
            };
            let f = func(&device.compile_and_load(source)?, name)?;
            let n32 = elements as u32;
            let cfg = LaunchConfig {
                grid_dim: ((elements / per_thread).div_ceil(256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `plane` holds `elements` codes and their scales; `out` holds `elements`.
            unsafe {
                device
                    .stream
                    .launch_builder(&f)
                    .arg(&plane)
                    .arg(&mut out)
                    .arg(&n32)
                    .launch(cfg)
            }
            .map_err(|e| RuntimeError::Compute(format!("{name}: {e}")))?;
            Ok(out)
        };
        let absolute = |v: &CudaSlice<f32>| -> Result<CudaSlice<f32>, RuntimeError> {
            let mut a = device
                .stream
                .clone_dtod(v)
                .map_err(|e| RuntimeError::Compute(format!("copy: {e}")))?;
            let len = a.len() as u64;
            let cfg = LaunchConfig {
                grid_dim: (a.len().div_ceil(256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `a` holds `len` floats.
            unsafe {
                device
                    .stream
                    .launch_builder(&abs_fn)
                    .arg(&mut a)
                    .arg(&len)
                    .launch(cfg)
            }
            .map_err(|e| RuntimeError::Compute(format!("lumen_lt_abs: {e}")))?;
            Ok(a)
        };
        let w = decode(w_plane, n * k)?;
        let xd = decode(x.x(device), x.rows * k)?;
        Ok(Self {
            n,
            k,
            w_abs: absolute(&w)?,
            x_abs: absolute(&xd)?,
            w,
            x: xd,
            r: device.alloc_zeros::<f32>(x.rows * n)?,
            mag: device.alloc_zeros::<f32>(x.rows * n)?,
            rows_ready: 0,
            counts: device.alloc_zeros::<u32>(2)?,
            check_fn: func(&check, "lumen_lt_check")?,
        })
    }

    /// Check the BF16 output at `d` (row stride `ldd`) for the first `rows` rows of the activation.
    ///
    /// # Safety
    /// `rows` must not exceed the activation's rows, `ldd` must be at least `n`, and `d` must address
    /// `rows x ldd` live device BF16 values, written by work already enqueued on the device stream.
    pub unsafe fn check(
        &mut self,
        device: &CudaDevice,
        rows: usize,
        d: u64,
        ldd: usize,
    ) -> Result<Check, RuntimeError> {
        if rows != self.rows_ready {
            sgemm(device, self.n, rows, self.k, &self.w, &self.x, &mut self.r)?;
            sgemm(
                device,
                self.n,
                rows,
                self.k,
                &self.w_abs,
                &self.x_abs,
                &mut self.mag,
            )?;
            self.rows_ready = rows;
        }
        device
            .stream
            .memset_zeros(&mut self.counts)
            .map_err(|e| RuntimeError::Compute(format!("memset: {e}")))?;
        let total = rows * self.n;
        let (ldd32, rows32, n32) = (ldd as u32, rows as u32, self.n as u32);
        let cfg = LaunchConfig {
            grid_dim: (total.div_ceil(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (counts_ptr, _) = self.counts.device_ptr(&device.stream);
        let worst_ptr = counts_ptr + 4;
        // SAFETY: `d` holds `rows x ldd` BF16 values (the caller's contract); r and mag hold at least
        // `rows x n`; counts holds two words.
        unsafe {
            device
                .stream
                .launch_builder(&self.check_fn)
                .arg(&d)
                .arg(&ldd32)
                .arg(&self.r)
                .arg(&self.mag)
                .arg(&rows32)
                .arg(&n32)
                .arg(&counts_ptr)
                .arg(&worst_ptr)
                .launch(cfg)
        }
        .map_err(|e| RuntimeError::Compute(format!("lumen_lt_check: {e}")))?;
        let c = device.dtoh_copy(&self.counts)?;
        Ok(Check {
            violations: c[0] as u64,
            worst: f32::from_bits(c[1]),
        })
    }
}

/// `out[rows][n] = x[rows][k] * w[n][k]^T` in F32 with cuBLAS's pedantic F32 compute.
fn sgemm(
    device: &CudaDevice,
    n: usize,
    rows: usize,
    k: usize,
    w: &CudaSlice<f32>,
    x: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
) -> Result<(), RuntimeError> {
    let (one, zero) = (1.0f32, 0.0f32);
    let (w_ptr, _) = w.device_ptr(&device.stream);
    let (x_ptr, _) = x.device_ptr(&device.stream);
    let (o_ptr, _) = out.device_ptr(&device.stream);
    // SAFETY: w holds n x k, x at least rows x k and out at least rows x n F32 values.
    let status = unsafe {
        cublas_sys::cublasGemmEx(
            *device.blas.handle(),
            cublas_sys::cublasOperation_t::CUBLAS_OP_T,
            cublas_sys::cublasOperation_t::CUBLAS_OP_N,
            n as i32,
            rows as i32,
            k as i32,
            (&one as *const f32).cast(),
            w_ptr as *const std::ffi::c_void,
            cublas_sys::cudaDataType_t::CUDA_R_32F,
            k as i32,
            x_ptr as *const std::ffi::c_void,
            cublas_sys::cudaDataType_t::CUDA_R_32F,
            k as i32,
            (&zero as *const f32).cast(),
            o_ptr as *mut std::ffi::c_void,
            cublas_sys::cudaDataType_t::CUDA_R_32F,
            n as i32,
            cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_PEDANTIC,
            cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
    };
    if status != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(RuntimeError::Compute(format!(
            "reference SGEMM {n}x{rows}x{k}: status={status:?}"
        )));
    }
    Ok(())
}

/// Fill `bytes` bytes at `d` with 0xFF (BF16 NaN).
fn fill_nan(device: &CudaDevice, d: u64, bytes: usize) -> Result<(), RuntimeError> {
    device
        .ctx
        .bind_to_thread()
        .map_err(|e| RuntimeError::Compute(format!("CUDA driver error: {e}")))?;
    // SAFETY: the caller owns `bytes` bytes at `d`.
    unsafe { cudarc::driver::result::memset_d8_async(d, 0xFF, bytes, device.stream.cu_stream()) }
        .map_err(|e| RuntimeError::Compute(format!("memset: {e}")))
}

fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(f32::total_cmp);
    v[v.len() / 2]
}

/// Select and verify the algorithm of every bucket of `shape`. `plans` are the shape's plans in
/// bucket order; `weights` are copies the timing rotates over, the first also being the one
/// verified; `workspace` holds [`super::native_prefill_gemm::WORKSPACE_BYTES`].
///
/// # Safety
/// `shape.ldd` must be at least `shape.n`; `plans` must number at most [`BUCKETS`], plan `b` built
/// from `shape.lt(bucket_rows(b))`; every
/// address in `weights` must be a live device operand of `shape` as [`WeightOperand`] describes;
/// and `workspace` must address `workspace_bytes` live device bytes.
pub unsafe fn select_shape(
    device: &CudaDevice,
    handle: &LtHandle,
    shape: &GemmShape,
    plans: &mut [LtMatmul],
    weights: &[WeightOperand],
    workspace: u64,
    workspace_bytes: usize,
) -> Result<Vec<PlanSelection>, RuntimeError> {
    let lib = handle.library();
    let first = *weights.first().ok_or_else(|| {
        RuntimeError::Compute(format!("no weight to select the {} plans with", shape.name))
    })?;
    let x = Activation::synthetic(device, shape.input, MAX_ROWS, shape.k, shape.n as u64)?;
    let mut verifier = Verifier::new(device, shape.n, shape.k, first.plane, &x)?;
    let check = device.compile_and_load(shaders::CUBLASLT_CHECK_KERNEL_SOURCE)?;
    let spin = check
        .load_function("lumen_lt_spin")
        .map_err(|e| RuntimeError::Compute(format!("load lumen_lt_spin: {e}")))?;
    // SAFETY: every element the check reads is written by a GEMM first (it fills with NaN first).
    let d = unsafe { device.alloc_uninit::<u16>(MAX_ROWS * shape.ldd)? };
    let d_ptr = d.device_ptr(&device.stream).0;
    let (x_ptr, x_scale) = (x.x(device), x.x_scale(device));
    let mut out = Vec::with_capacity(plans.len());
    for (b, plan) in plans.iter_mut().enumerate() {
        let rows = bucket_rows(b);
        let found = plan.heuristics(
            handle,
            workspace_bytes as u64,
            CANDIDATES,
            first.scale,
            x_scale,
        )?;
        let mut outcomes = Vec::new();
        let mut timed = Vec::new();
        for h in &found {
            let config = lib.algo_config(&h.algo)?;
            let scheme = config[ALGO_CONFIG_REDUCTION_SCHEME];
            if scheme == REDUCTION_NONE as u64 || scheme == REDUCTION_COMPUTE_TYPE as u64 {
                timed.push((*h, config));
            } else {
                outcomes.push(CandidateOutcome {
                    config,
                    median_us: None,
                    verdict: Verdict::ReductionRejected(scheme),
                });
            }
        }
        if timed.is_empty() {
            return Err(RuntimeError::Compute(format!(
                "cuBLASLt offers no algorithm with an F32 reduction for {} at {rows} rows",
                shape.name
            )));
        }
        let ops = |w: &WeightOperand| LtOperands {
            w: w.plane,
            w_scale: w.scale,
            x: x_ptr,
            x_scale,
            d: d_ptr,
        };
        let algos: Vec<LtAlgo> = timed.iter().map(|(h, _)| h.algo).collect();
        let times = time_candidates(
            device,
            &spin,
            handle,
            plan,
            &algos,
            weights,
            &ops,
            workspace,
            workspace_bytes,
        )?;
        let mut order: Vec<usize> = (0..timed.len()).collect();
        order.sort_by(|&a, &b| times[a].total_cmp(&times[b]));
        let mut selected = None;
        let mut verdicts = vec![Verdict::NotVerified; timed.len()];
        for &i in &order {
            fill_nan(device, d_ptr, rows * shape.ldd * 2)?;
            // SAFETY: the operands are live allocations sized for MAX_ROWS rows.
            unsafe {
                plan.launch(
                    handle,
                    &device.stream,
                    &timed[i].0.algo,
                    first.unit_alpha,
                    &ops(&first),
                    workspace,
                    workspace_bytes,
                )?;
            }
            let c = verifier.check(device, rows, d_ptr, shape.ldd)?;
            if c.violations == 0 {
                verdicts[i] = Verdict::Selected;
                let (h, config) = timed[i];
                selected = Some(Selected {
                    algo: h.algo,
                    workspace: h.workspace_size as u64,
                    waves: h.waves_count,
                    config,
                    median_us: times[i],
                    worst: c.worst,
                    candidates: found.len() as u32,
                });
                break;
            }
            verdicts[i] = Verdict::Failed(c.violations);
        }
        outcomes.extend(timed.iter().zip(times.iter()).zip(verdicts).map(
            |(((_, config), &t), verdict)| CandidateOutcome {
                config: *config,
                median_us: Some(t),
                verdict,
            },
        ));
        let selected = selected.ok_or_else(|| {
            RuntimeError::Compute(format!(
                "no cuBLASLt algorithm for {} at {rows} rows passes verification: {outcomes:?}",
                shape.name
            ))
        })?;
        out.push(PlanSelection {
            shape: shape.name,
            rows,
            selected,
            outcomes,
        });
    }
    Ok(out)
}

/// Median time in microseconds of each algorithm in `algos`, timed as the module documentation
/// describes. The spin is lengthened until the host enqueues every timed launch before it ends.
fn time_candidates(
    device: &CudaDevice,
    spin: &CudaFunction,
    handle: &LtHandle,
    plan: &mut LtMatmul,
    algos: &[LtAlgo],
    weights: &[WeightOperand],
    ops: &dyn Fn(&WeightOperand) -> LtOperands,
    workspace: u64,
    workspace_bytes: usize,
) -> Result<Vec<f32>, RuntimeError> {
    let event = || {
        device
            .ctx
            .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| RuntimeError::Compute(format!("event: {e}")))
    };
    let events = (0..algos.len() * (TIMED + 1))
        .map(|_| event())
        .collect::<Result<Vec<_>, _>>()?;
    let mut launch = |algo: &LtAlgo, i: usize| {
        let w = &weights[i % weights.len()];
        // SAFETY: the operands are live allocations sized for the plan (select_shape's contract).
        unsafe {
            plan.launch(
                handle,
                &device.stream,
                algo,
                w.unit_alpha,
                &ops(w),
                workspace,
                workspace_bytes,
            )
        }
    };
    for algo in algos {
        for i in 0..WARMUP {
            launch(algo, i)?;
        }
    }
    device.synchronize()?;
    let mut spin_ns: u64 = 2_000_000;
    loop {
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the spin kernel takes one integer and touches no memory.
        unsafe { device.stream.launch_builder(spin).arg(&spin_ns).launch(cfg) }
            .map_err(|e| RuntimeError::Compute(format!("lumen_lt_spin: {e}")))?;
        let t0 = Instant::now();
        for (a, algo) in algos.iter().enumerate() {
            let ev = &events[a * (TIMED + 1)..(a + 1) * (TIMED + 1)];
            ev[0]
                .record(&device.stream)
                .map_err(|e| RuntimeError::Compute(format!("event: {e}")))?;
            for i in 0..TIMED {
                launch(algo, i)?;
                ev[i + 1]
                    .record(&device.stream)
                    .map_err(|e| RuntimeError::Compute(format!("event: {e}")))?;
            }
        }
        let host_ns = t0.elapsed().as_nanos() as u64;
        device.synchronize()?;
        if host_ns < spin_ns {
            break;
        }
        if spin_ns >= 1_000_000_000 {
            return Err(RuntimeError::Compute(format!(
                "cuBLASLt timing: the host took {host_ns} ns to enqueue what a {spin_ns} ns spin \
                 should cover"
            )));
        }
        spin_ns *= 4;
    }
    let mut medians = Vec::with_capacity(algos.len());
    for a in 0..algos.len() {
        let ev = &events[a * (TIMED + 1)..(a + 1) * (TIMED + 1)];
        let mut us = Vec::with_capacity(TIMED);
        for i in 0..TIMED {
            let ms = ev[i]
                .elapsed_ms(&ev[i + 1])
                .map_err(|e| RuntimeError::Compute(format!("event: {e}")))?;
            us.push(ms * 1000.0);
        }
        medians.push(median(us));
    }
    Ok(medians)
}

/// The library, device and plan definition a cached table is valid for, as the header of its file.
pub struct Identity {
    pub text: String,
}

impl Identity {
    pub fn new(
        device: &CudaDevice,
        lib: &LtLibrary,
        workspace_bytes: usize,
    ) -> Result<Self, RuntimeError> {
        let (major, minor) = device.compute_capability()?;
        let shapes: Vec<String> = SHAPES
            .iter()
            .map(|s| format!("{}:{:?}:{}x{}:ld{}", s.name, s.input, s.n, s.k, s.ldd))
            .collect();
        let text = format!(
            "lumen-cublaslt-plans 1\n\
             library {} {}\n\
             device {} cc {major}.{minor} driver {}\n\
             workspace {workspace_bytes}\n\
             descriptor compute {COMPUTE_32F} scale {R_32F} transa {OP_T} transb {OP_N} nvfp4 scale mode \
             {SCALE_VEC16_UE4M3} fp8 scale pointers, fast accumulation unset, beta 0, C = D, operand \
             alignment {OPERAND_ALIGNMENT}\n\
             plans {}; rows {ROW_STEP}..={MAX_ROWS} step {ROW_STEP}; reductions {REDUCTION_NONE},{REDUCTION_COMPUTE_TYPE}; \
             candidates {CANDIDATES} warmup {WARMUP} timed {TIMED}; bound 2^-8 rel + 2^-17 mag\n",
            lib.path,
            lib.version,
            device.name()?,
            super::ffi::driver_version()?,
            shapes.join(",")
        );
        Ok(Self { text })
    }

    fn path(&self) -> Option<PathBuf> {
        super::ptx_cache::lumen_cache_dir("cublaslt").map(|d| {
            d.join(format!(
                "{}.txt",
                super::ptx_cache::sha256_hex(self.text.as_bytes())
            ))
        })
    }
}

/// The four layouts (type, rows, cols, ld) of `shape` as recorded; C is D, which beta = 0 never reads.
fn layouts(s: &LtGemmShape) -> String {
    let code = match s.input {
        LtInput::Nvfp4 => R_4F_E2M1,
        LtInput::Fp8 => R_8F_E4M3,
    };
    format!(
        "a {code},{},{},{} b {code},{},{},{} c {R_16BF},{},{},{} d {R_16BF},{},{},{}",
        s.k, s.n, s.k, s.k, s.m, s.k, s.n, s.m, s.ldd, s.n, s.m, s.ldd
    )
}

/// Serialize a full table (every shape, every bucket, in order) under `identity`.
pub fn serialize(identity: &Identity, table: &[Selected]) -> String {
    let mut body = identity.text.clone();
    for (i, sel) in table.iter().enumerate() {
        let shape = &SHAPES[i / BUCKETS];
        let lt = shape.lt(bucket_rows(i % BUCKETS));
        let algo: Vec<String> = sel.algo.data.iter().map(|w| format!("{w:016x}")).collect();
        let config: Vec<String> = sel.config.iter().map(|c| c.to_string()).collect();
        body.push_str(&format!(
            "plan {} {} algo {} ws {} waves {} config {} {} align {OPERAND_ALIGNMENT} time_us {} \
             worst {} candidates {}\n",
            shape.name,
            lt.m,
            algo.join(","),
            sel.workspace,
            sel.waves,
            config.join(","),
            layouts(&lt),
            sel.median_us,
            sel.worst,
            sel.candidates
        ));
    }
    let sum = super::ptx_cache::sha256_hex(body.as_bytes());
    body.push_str(&format!("sha256 {sum}\n"));
    body
}

/// Parse a table written by [`serialize`] for `identity`; `Err` says why it cannot be used.
pub fn parse(identity: &Identity, text: &str) -> Result<Vec<Selected>, String> {
    let (body, sum_line) = text
        .trim_end_matches('\n')
        .rsplit_once('\n')
        .ok_or("the table is truncated")?;
    let body = format!("{body}\n");
    if sum_line != format!("sha256 {}", super::ptx_cache::sha256_hex(body.as_bytes())) {
        return Err("the checksum does not match".into());
    }
    let rest = body
        .strip_prefix(&identity.text)
        .ok_or("the identity differs")?;
    let lines: Vec<&str> = rest.lines().collect();
    if lines.len() != SHAPES.len() * BUCKETS {
        return Err(format!(
            "the table has {} plans; the file {}",
            SHAPES.len() * BUCKETS,
            lines.len()
        ));
    }
    let mut table = Vec::with_capacity(lines.len());
    for (i, line) in lines.iter().enumerate() {
        let shape = &SHAPES[i / BUCKETS];
        let lt = shape.lt(bucket_rows(i % BUCKETS));
        let bad = || format!("plan line {i} is malformed: {line}");
        let f: Vec<&str> = line.split(' ').collect();
        let layout = layouts(&lt);
        let expect_head = format!("plan {} {} algo", shape.name, lt.m);
        if f.len() != 27 || f[..4].join(" ") != expect_head || f[11..19].join(" ") != layout {
            return Err(bad());
        }
        let words: Vec<u64> = f[4]
            .split(',')
            .map(|w| u64::from_str_radix(w, 16))
            .collect::<Result<_, _>>()
            .map_err(|_| bad())?;
        let config: Vec<u64> = f[10]
            .split(',')
            .map(|c| c.parse())
            .collect::<Result<_, _>>()
            .map_err(|_| bad())?;
        let (Ok(data), Ok(config)) = (<[u64; 8]>::try_from(words), <[u64; 9]>::try_from(config))
        else {
            return Err(bad());
        };
        let labels = [
            (5, "ws"),
            (7, "waves"),
            (9, "config"),
            (19, "align"),
            (21, "time_us"),
            (23, "worst"),
            (25, "candidates"),
        ];
        if labels.iter().any(|&(at, l)| f[at] != l) || f[20] != OPERAND_ALIGNMENT.to_string() {
            return Err(bad());
        }
        let num = |at: usize| f[at].parse::<f32>().map_err(|_| bad());
        table.push(Selected {
            algo: LtAlgo { data },
            workspace: f[6].parse().map_err(|_| bad())?,
            waves: num(8)?,
            config,
            median_us: num(22)?,
            worst: num(24)?,
            candidates: f[26].parse().map_err(|_| bad())?,
        });
    }
    Ok(table)
}

/// The cached table for `identity` and its file. The outer `Err` is a library call that failed; the
/// inner one says why the cache cannot be used (absent, unreadable, for another identity, damaged, or
/// holding an algorithm the library now reads back with a different configuration).
pub fn load(
    identity: &Identity,
    lib: &LtLibrary,
) -> Result<Result<(Vec<Selected>, PathBuf), String>, RuntimeError> {
    let read = || -> Result<(Vec<Selected>, PathBuf), String> {
        let path = identity.path().ok_or("no cache directory")?;
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok((parse(identity, &text)?, path))
    };
    let (table, path) = match read() {
        Ok(found) => found,
        Err(reason) => return Ok(Err(reason)),
    };
    for (i, sel) in table.iter().enumerate() {
        let now = lib.algo_config(&sel.algo)?;
        if now != sel.config {
            return Ok(Err(format!(
                "plan {i}: the library reads the algorithm back as {now:?}, recorded {:?}",
                sel.config
            )));
        }
    }
    Ok(Ok((table, path)))
}

/// Write the table for `identity`; the path written, or `None` when it could not be.
pub fn store(identity: &Identity, table: &[Selected]) -> Option<PathBuf> {
    let path = identity.path()?;
    std::fs::create_dir_all(path.parent()?).ok()?;
    super::ptx_cache::write_atomically(&path, serialize(identity, table).as_bytes()).then_some(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> Identity {
        Identity {
            text: "lumen-cublaslt-plans 1\nlibrary /x/libcublasLt.so.13 130600\n".into(),
        }
    }

    fn table() -> Vec<Selected> {
        (0..SHAPES.len() * BUCKETS)
            .map(|i| Selected {
                algo: LtAlgo {
                    data: [i as u64, u64::MAX - i as u64, 3, 4, 5, 6, 7, 8],
                },
                workspace: (i * 1024) as u64,
                waves: 0.75 + i as f32,
                config: [i as u64, 2, 1, 2, 0, 0, 7, 3, 1],
                median_us: 12.5 + i as f32 / 3.0,
                worst: 0.125,
                candidates: 8,
            })
            .collect()
    }

    #[test]
    fn a_table_round_trips_exactly() {
        let t = table();
        let text = serialize(&identity(), &t);
        assert_eq!(parse(&identity(), &text).unwrap(), t);
        assert!(text.contains("plan gate_up 16 algo "), "{}", &text[..300]);
        assert!(
            text.contains(" a 33,5120,17408,5120 b 33,5120,16,5120 c 14,17408,16,34816 d 14,17408,16,34816 align 256 ")
        );
    }

    #[test]
    fn a_changed_table_or_identity_is_not_used() {
        let text = serialize(&identity(), &table());
        let other = Identity {
            text: "lumen-cublaslt-plans 1\nlibrary /x/libcublasLt.so.13 130700\n".into(),
        };
        assert_eq!(parse(&other, &text).unwrap_err(), "the identity differs");
        let tampered = text.replacen("ws 1024 ", "ws 2048 ", 1);
        assert_ne!(tampered, text);
        assert_eq!(
            parse(&identity(), &tampered).unwrap_err(),
            "the checksum does not match"
        );
        let truncated: String = text.lines().take(40).map(|l| format!("{l}\n")).collect();
        assert!(parse(&identity(), &truncated).is_err());
    }
}
