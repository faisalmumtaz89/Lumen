//! cuBLASLt, loaded at run time, for the native prefill's block-scaled NVFP4 and per-tensor FP8 GEMMs.
//!
//! cudarc's cuBLASLt binding cannot serve these GEMMs: under the pinned CUDA 12.2 API surface it has
//! no scale-mode attributes, no block-scale mode and no E2M1 data type, and its loader panics when the
//! library or one of its symbols is missing. This module binds the fourteen functions the route calls,
//! with every type, constant and payload width pinned against values generated from `cublasLt.h`
//! (`tests/fixtures/cublaslt_abi.txt`, written by `tests/fixtures/generate_cublaslt_abi.c` and
//! checked by the host test `abi_matches_the_header_fixture`).
//!
//! # Loading
//!
//! The candidates are tried in order through the normal loader search path (`LD_LIBRARY_PATH`):
//! `libcublasLt.so`, `libcublasLt.so.13`, `libcublasLt.so.12`. A candidate that loads but lacks a
//! symbol, or is older than [`MIN_VERSION`] (12.8, the first release with block-scaled FP4), is
//! recorded with its reason and closed, and the next one is tried, so an old unversioned library
//! cannot hide a newer versioned one. The accepted library is never closed: plans and enqueued work
//! reference its code for the life of the process. The loader is Linux-only; elsewhere it fails with
//! that reason.
//!
//! # Ownership
//!
//! [`LtHandle`] and [`LtMatmul`] own their cuBLASLt objects and destroy them on drop; a failure part
//! way through construction drops whatever was created. Every entry point binds the CUDA context the
//! handle was created in before it calls the library. Both types are `Send` and not `Sync`: they are
//! reached only through `&mut` under the backend's state lock, and a cuBLASLt handle may be used from
//! any host thread while its context is current, which every entry point makes it.

use crate::error::RuntimeError;
use cudarc::driver::{CudaContext, CudaStream};
use std::ffi::{c_int, c_void, CStr, CString};
use std::marker::PhantomData;
use std::sync::{Arc, OnceLock};

/// `cublasStatus_t`; zero is success.
pub type LtStatus = i32;

type RawHandle = *mut c_void;
type RawDesc = *mut c_void;
type RawLayout = *mut c_void;
type RawPref = *mut c_void;

/// The oldest cuBLASLt version (`cublasLtGetVersion`) the route accepts: 12.8.0.
pub const MIN_VERSION: usize = 120800;

/// `cublasLtMatmulAlgo_t`: an opaque algorithm, serializable for the same library version.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LtAlgo {
    pub data: [u64; 8],
}

/// `cublasLtMatmulHeuristicResult_t`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LtHeuristicResult {
    pub algo: LtAlgo,
    pub workspace_size: usize,
    pub state: LtStatus,
    pub waves_count: f32,
    pub reserved: [i32; 4],
}

impl LtHeuristicResult {
    const ZERO: Self = Self {
        algo: LtAlgo { data: [0; 8] },
        workspace_size: 0,
        state: 0,
        waves_count: 0.0,
        reserved: [0; 4],
    };
}

/// A descriptor or preference attribute and the type of the value it carries.
pub struct Attr<T> {
    pub id: u32,
    payload: PhantomData<T>,
}

impl<T> Attr<T> {
    const fn new(id: u32) -> Self {
        Self {
            id,
            payload: PhantomData,
        }
    }

    /// Bytes the attribute's value occupies.
    pub const fn payload_bytes(&self) -> usize {
        std::mem::size_of::<T>()
    }
}

/// `cublasComputeType_t::CUBLAS_COMPUTE_32F`.
pub const COMPUTE_32F: u32 = 68;
/// `cudaDataType_t` values.
pub const R_32F: u32 = 0;
pub const R_16BF: u32 = 14;
pub const R_8F_E4M3: u32 = 28;
pub const R_4F_E2M1: u32 = 33;
/// `cublasOperation_t` values.
pub const OP_N: u32 = 0;
pub const OP_T: u32 = 1;
/// `cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3`.
pub const SCALE_VEC16_UE4M3: u32 = 1;
/// `cublasLtReductionScheme_t` values the route admits.
pub const REDUCTION_NONE: u32 = 0;
pub const REDUCTION_COMPUTE_TYPE: u32 = 2;

/// `cublasLtMatmulDescAttributes_t` the route sets.
pub const DESC_TRANSA: Attr<u32> = Attr::new(3);
pub const DESC_TRANSB: Attr<u32> = Attr::new(4);
pub const DESC_A_SCALE_POINTER: Attr<u64> = Attr::new(17);
pub const DESC_B_SCALE_POINTER: Attr<u64> = Attr::new(18);
pub const DESC_A_SCALE_MODE: Attr<u32> = Attr::new(31);
pub const DESC_B_SCALE_MODE: Attr<u32> = Attr::new(32);
/// `cublasLtMatmulPreferenceAttributes_t` the route sets.
pub const PREF_MAX_WORKSPACE_BYTES: Attr<u64> = Attr::new(1);
pub const PREF_REDUCTION_SCHEME_MASK: Attr<u32> = Attr::new(3);

/// `cublasLtMatmulAlgoConfigAttributes_t`, all nine, with the bytes the library writes for each.
pub const ALGO_CONFIG: [(&str, u32, usize); 9] = [
    ("ID", 0, 4),
    ("TILE_ID", 1, 4),
    ("SPLITK_NUM", 2, 4),
    ("REDUCTION_SCHEME", 3, 4),
    ("CTA_SWIZZLING", 4, 4),
    ("CUSTOM_OPTION", 5, 4),
    ("STAGES_ID", 6, 4),
    ("INNER_SHAPE_ID", 7, 2),
    ("CLUSTER_SHAPE_ID", 8, 2),
];
/// Position of `REDUCTION_SCHEME` in [`ALGO_CONFIG`].
pub const ALGO_CONFIG_REDUCTION_SCHEME: usize = 3;

/// The operand pointers of a GEMM must be aligned to this many bytes: the heuristic's default
/// assumption (`CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_*_BYTES`), which the route does not relax, and
/// the alignment `cublasLtMatmul` requires of its workspace.
pub const OPERAND_ALIGNMENT: u64 = 256;

struct LtFns {
    create: unsafe extern "C" fn(*mut RawHandle) -> LtStatus,
    destroy: unsafe extern "C" fn(RawHandle) -> LtStatus,
    get_version: unsafe extern "C" fn() -> usize,
    desc_create: unsafe extern "C" fn(*mut RawDesc, u32, u32) -> LtStatus,
    desc_destroy: unsafe extern "C" fn(RawDesc) -> LtStatus,
    desc_set_attribute: unsafe extern "C" fn(RawDesc, u32, *const c_void, usize) -> LtStatus,
    layout_create: unsafe extern "C" fn(*mut RawLayout, u32, u64, u64, i64) -> LtStatus,
    layout_destroy: unsafe extern "C" fn(RawLayout) -> LtStatus,
    pref_create: unsafe extern "C" fn(*mut RawPref) -> LtStatus,
    pref_destroy: unsafe extern "C" fn(RawPref) -> LtStatus,
    pref_set_attribute: unsafe extern "C" fn(RawPref, u32, *const c_void, usize) -> LtStatus,
    algo_get_heuristic: unsafe extern "C" fn(
        RawHandle,
        RawDesc,
        RawLayout,
        RawLayout,
        RawLayout,
        RawLayout,
        RawPref,
        c_int,
        *mut LtHeuristicResult,
        *mut c_int,
    ) -> LtStatus,
    matmul: unsafe extern "C" fn(
        RawHandle,
        RawDesc,
        *const c_void,
        *const c_void,
        RawLayout,
        *const c_void,
        RawLayout,
        *const c_void,
        *const c_void,
        RawLayout,
        *mut c_void,
        RawLayout,
        *const LtAlgo,
        *mut c_void,
        usize,
        *mut c_void,
    ) -> LtStatus,
    algo_config_get_attribute:
        unsafe extern "C" fn(*const LtAlgo, u32, *mut c_void, usize, *mut usize) -> LtStatus,
}

impl LtFns {
    /// Resolve every symbol from `lib`, or name the first one it lacks.
    ///
    /// # Safety
    /// `lib` must be a live `dlopen` handle.
    unsafe fn resolve(lib: *mut c_void) -> Result<Self, &'static str> {
        unsafe fn sym<T: Copy>(lib: *mut c_void, name: &'static CStr) -> Result<T, &'static str> {
            let p = libc::dlsym(lib, name.as_ptr());
            if p.is_null() {
                return Err(name.to_str().unwrap_or("?"));
            }
            // SAFETY: T is the symbol's `extern "C"` function-pointer type, the size of a pointer.
            Ok(std::mem::transmute_copy::<*mut c_void, T>(&p))
        }
        Ok(Self {
            create: sym(lib, c"cublasLtCreate")?,
            destroy: sym(lib, c"cublasLtDestroy")?,
            get_version: sym(lib, c"cublasLtGetVersion")?,
            desc_create: sym(lib, c"cublasLtMatmulDescCreate")?,
            desc_destroy: sym(lib, c"cublasLtMatmulDescDestroy")?,
            desc_set_attribute: sym(lib, c"cublasLtMatmulDescSetAttribute")?,
            layout_create: sym(lib, c"cublasLtMatrixLayoutCreate")?,
            layout_destroy: sym(lib, c"cublasLtMatrixLayoutDestroy")?,
            pref_create: sym(lib, c"cublasLtMatmulPreferenceCreate")?,
            pref_destroy: sym(lib, c"cublasLtMatmulPreferenceDestroy")?,
            pref_set_attribute: sym(lib, c"cublasLtMatmulPreferenceSetAttribute")?,
            algo_get_heuristic: sym(lib, c"cublasLtMatmulAlgoGetHeuristic")?,
            matmul: sym(lib, c"cublasLtMatmul")?,
            algo_config_get_attribute: sym(lib, c"cublasLtMatmulAlgoConfigGetAttribute")?,
        })
    }
}

/// The loaded cuBLASLt: its entry points, where it was found, its version, and the candidates
/// rejected before it.
pub struct LtLibrary {
    fns: LtFns,
    /// Path of the shared object the functions resolved from.
    pub path: String,
    /// `cublasLtGetVersion()`, e.g. 130600.
    pub version: usize,
    /// Each candidate tried before this one, with the reason it was rejected.
    pub rejected: Vec<String>,
}

impl LtLibrary {
    /// One line naming the library, its version and every rejected candidate.
    pub fn describe(&self) -> String {
        let mut s = format!("cuBLASLt {} (version {})", self.path, self.version);
        if !self.rejected.is_empty() {
            s.push_str(&format!("; rejected {}", self.rejected.join("; ")));
        }
        s
    }

    /// The nine algorithm-configuration attributes of `algo`, in [`ALGO_CONFIG`] order, each checked
    /// to be written at its pinned width.
    pub fn algo_config(&self, algo: &LtAlgo) -> Result<[u64; 9], RuntimeError> {
        let mut out = [0u64; 9];
        for (i, &(name, id, width)) in ALGO_CONFIG.iter().enumerate() {
            let mut buf = [0u8; 8];
            let mut written = 0usize;
            construction_call("cublasLtMatmulAlgoConfigGetAttribute", || unsafe {
                (self.fns.algo_config_get_attribute)(
                    algo,
                    id,
                    buf.as_mut_ptr().cast(),
                    width,
                    &mut written,
                )
            })?;
            if written != width {
                return Err(RuntimeError::Compute(format!(
                    "cuBLASLt wrote {written} bytes for ALGO_CONFIG_{name}; the route pins {width}"
                )));
            }
            out[i] = u64::from_le_bytes(buf);
        }
        Ok(out)
    }
}

/// Whether a `cublasLtGetVersion` value is new enough for the route.
pub fn version_qualifies(version: usize) -> bool {
    version >= MIN_VERSION
}

#[cfg(target_os = "linux")]
const CANDIDATES: &[&str] = &["libcublasLt.so", "libcublasLt.so.13", "libcublasLt.so.12"];

static LIBRARY: OnceLock<Result<LtLibrary, String>> = OnceLock::new();

/// The process-wide cuBLASLt, loaded on first use; `Err` names why no candidate qualified.
pub fn library() -> Result<&'static LtLibrary, RuntimeError> {
    LIBRARY
        .get_or_init(|| {
            #[cfg(target_os = "linux")]
            {
                // SAFETY: every candidate is a cuBLASLt soname.
                unsafe { load(CANDIDATES) }
            }
            #[cfg(not(target_os = "linux"))]
            {
                Err("the cuBLASLt route is Linux-only".to_string())
            }
        })
        .as_ref()
        .map_err(|e| RuntimeError::Unsupported(e.clone()))
}

/// Try `candidates` in order and keep the first that resolves every symbol and qualifies.
///
/// # Safety
/// Loading a library runs its initializers, and a qualifying one is then called through the
/// cuBLASLt signatures: every candidate must name a genuine cuBLASLt or a library that lacks its
/// symbols.
pub unsafe fn load(candidates: &[&str]) -> Result<LtLibrary, String> {
    let mut rejected = Vec::new();
    for name in candidates {
        match open_candidate(name) {
            Ok(mut lib) => {
                lib.rejected = rejected;
                return Ok(lib);
            }
            Err(reason) => rejected.push(format!("{name}: {reason}")),
        }
    }
    Err(format!(
        "no cuBLASLt of version {MIN_VERSION} or newer: {}",
        rejected.join("; ")
    ))
}

fn open_candidate(name: &str) -> Result<LtLibrary, String> {
    let cname = CString::new(name).map_err(|_| "the name holds a NUL byte".to_string())?;
    // SAFETY: a NUL-terminated name; the handle is closed below on every rejection.
    let lib = unsafe { libc::dlopen(cname.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
    if lib.is_null() {
        return Err(dl_error());
    }
    let reject = |reason: String| {
        // SAFETY: nothing from this candidate was created or retained.
        unsafe { libc::dlclose(lib) };
        Err(reason)
    };
    // SAFETY: `lib` is live.
    let fns = match unsafe { LtFns::resolve(lib) } {
        Ok(fns) => fns,
        Err(symbol) => return reject(format!("lacks {symbol}")),
    };
    // SAFETY: resolved from a qualifying-by-name library; takes no arguments.
    let version = unsafe { (fns.get_version)() };
    if !version_qualifies(version) {
        return reject(format!("version {version} is older than {MIN_VERSION}"));
    }
    let path = object_path(fns.get_version as *const c_void).unwrap_or_else(|| name.to_string());
    Ok(LtLibrary {
        fns,
        path,
        version,
        rejected: Vec::new(),
    })
}

fn dl_error() -> String {
    // SAFETY: dlerror returns NULL or a NUL-terminated string owned by the loader.
    let e = unsafe { libc::dlerror() };
    if e.is_null() {
        "not found".to_string()
    } else {
        unsafe { CStr::from_ptr(e) }.to_string_lossy().into_owned()
    }
}

/// Path of the shared object that holds `addr`.
pub(crate) fn object_path(addr: *const c_void) -> Option<String> {
    let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
    // SAFETY: dladdr only reads the loader's tables; `info` outlives the call.
    if unsafe { libc::dladdr(addr, &mut info) } == 0 || info.dli_fname.is_null() {
        return None;
    }
    Some(
        unsafe { CStr::from_ptr(info.dli_fname) }
            .to_string_lossy()
            .into_owned(),
    )
}

/// One line naming the libraries the native prefill depends on: NVRTC's path and version, cuBLASLt's
/// path, version and rejected candidates (or why none qualified), and the driver version.
pub fn library_report() -> String {
    // SAFETY: cudarc's probe, which loads each NVRTC candidate and drops it again. cudarc's
    // `culib`, used below, panics when no candidate loads; the probe itself panics only on a library
    // that lacks one of cudarc's NVRTC symbols, on which the CUDA backend's own NVRTC load panics
    // the same way when it starts.
    let nvrtc = if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
        "NVRTC not found".to_string()
    } else {
        match super::ffi::nvrtc_version() {
            Ok((major, minor)) => format!(
                "NVRTC {} ({major}.{minor})",
                super::ffi::nvrtc_library_path().unwrap_or_else(|| "path unknown".into())
            ),
            Err(e) => format!("NVRTC unavailable ({e})"),
        }
    };
    let lt = match library() {
        Ok(lib) => lib.describe(),
        Err(e) => format!("cuBLASLt unavailable ({e})"),
    };
    let driver = match super::ffi::driver_version() {
        Ok(v) => format!("driver {v}"),
        Err(e) => format!("driver version unknown ({e})"),
    };
    format!("{nvrtc}; {lt}; {driver}")
}

fn status(step: &str, s: LtStatus) -> Result<(), RuntimeError> {
    if s == 0 {
        Ok(())
    } else {
        Err(RuntimeError::Compute(format!(
            "{step} failed: cuBLASLt status {s}"
        )))
    }
}

/// A call that creates or configures a cuBLASLt object. Under `test-fault-injection` each such call
/// is one numbered construction step that a test can make fail (see [`fault`]).
fn construction_call(step: &str, f: impl FnOnce() -> LtStatus) -> Result<(), RuntimeError> {
    construction_step(step)?;
    status(step, f())
}

/// Count one construction step, failing it when a test armed it. Also used for the device
/// allocations the plan builder makes.
pub(crate) fn construction_step(step: &str) -> Result<(), RuntimeError> {
    #[cfg(any(test, feature = "test-fault-injection"))]
    if fault::take(step) {
        return Err(RuntimeError::Compute(format!(
            "{step} failed: injected construction failure"
        )));
    }
    let _ = step;
    Ok(())
}

fn track(_created: i64) {
    #[cfg(any(test, feature = "test-fault-injection"))]
    fault::LIVE.fetch_add(_created, std::sync::atomic::Ordering::SeqCst);
}

/// Count a destroy call as one object fewer only when it succeeded, so a failed destroy shows as a
/// live object. Drop cannot report the failure otherwise.
fn destroyed(status: LtStatus) {
    if status == 0 {
        track(-1);
    }
}

/// Test-only construction-failure injection and a live-object count.
#[cfg(any(test, feature = "test-fault-injection"))]
pub mod fault {
    use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

    pub(super) static LIVE: AtomicI64 = AtomicI64::new(0);
    static STEPS: AtomicU64 = AtomicU64::new(0);
    static FAIL_AT: AtomicU64 = AtomicU64::new(u64::MAX);

    /// Restart the step count at zero and make step `step` (counted from zero) fail once;
    /// `u64::MAX` injects nothing.
    pub fn fail_step(step: u64) {
        FAIL_AT.store(step, Ordering::SeqCst);
        STEPS.store(0, Ordering::SeqCst);
    }

    /// Construction steps taken since the last [`fail_step`].
    pub fn steps() -> u64 {
        STEPS.load(Ordering::SeqCst)
    }

    /// cuBLASLt handles, descriptors, layouts and preferences created and not yet destroyed.
    pub fn live_objects() -> i64 {
        LIVE.load(Ordering::SeqCst)
    }

    pub(super) fn take(_step: &str) -> bool {
        let n = STEPS.fetch_add(1, Ordering::SeqCst);
        FAIL_AT
            .compare_exchange(n, u64::MAX, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
    }
}

/// A cuBLASLt handle, created in and bound to one CUDA context.
pub struct LtHandle {
    raw: RawHandle,
    lib: &'static LtLibrary,
    ctx: Arc<CudaContext>,
}

// SAFETY: see the module documentation: reached only through `&mut` under the backend's state lock,
// with its context bound on every entry.
unsafe impl Send for LtHandle {}

impl LtHandle {
    /// Load the library if needed and create a handle in `ctx`.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, RuntimeError> {
        let lib = library()?;
        bind(ctx)?;
        let mut raw: RawHandle = std::ptr::null_mut();
        construction_call("cublasLtCreate", || unsafe { (lib.fns.create)(&mut raw) })?;
        track(1);
        Ok(Self {
            raw,
            lib,
            ctx: ctx.clone(),
        })
    }

    pub fn library(&self) -> &'static LtLibrary {
        self.lib
    }
}

impl Drop for LtHandle {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        // SAFETY: created by cublasLtCreate; every plan made from it was dropped first (its owner
        // declares the plans before the handle).
        destroyed(unsafe { (self.lib.fns.destroy)(self.raw) });
    }
}

fn bind(ctx: &Arc<CudaContext>) -> Result<(), RuntimeError> {
    ctx.bind_to_thread()
        .map_err(|e| RuntimeError::Compute(format!("CUDA driver error: {e}")))
}

/// The two operand encodings the route multiplies; both operands of a GEMM share one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LtInput {
    /// E2M1 codes with a UE4M3 scale per 16 values in the swizzled 128x4 tile layout.
    Nvfp4,
    /// E4M3 codes with one F32 scale per operand.
    Fp8,
}

/// `D[m][n] = alpha * sum_k X[m][k] * W[n][k]`, accumulated in F32 and stored as BF16 with row
/// stride `ldd`. `W` (`[n][k]`, k contiguous) is cuBLASLt's transposed A; `X` (`[m][k]`) is B.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LtGemmShape {
    pub input: LtInput,
    pub n: u64,
    pub k: u64,
    pub m: u64,
    pub ldd: u64,
}

/// Device pointers of one GEMM call. `w_scale` / `x_scale` are the swizzled block scales (NVFP4) or
/// the F32 per-tensor scales (FP8).
#[derive(Clone, Copy, Debug)]
pub struct LtOperands {
    pub w: u64,
    pub w_scale: u64,
    pub x: u64,
    pub x_scale: u64,
    pub d: u64,
}

/// One GEMM's descriptor and layouts. The scale pointers are set on the descriptor at each launch,
/// which is why launching takes `&mut self`.
pub struct LtMatmul {
    desc: RawDesc,
    a: RawLayout,
    b: RawLayout,
    d: RawLayout,
    shape: LtGemmShape,
    lib: &'static LtLibrary,
    ctx: Arc<CudaContext>,
}

// SAFETY: as for LtHandle.
unsafe impl Send for LtMatmul {}

impl LtMatmul {
    /// Build the descriptor (F32 compute and scale, `W` transposed, block-scale modes for NVFP4) and
    /// the A, B and D layouts (D also serves as C, which beta = 0 never reads). The plan cache's
    /// identity (`cublaslt_algo_cache::Identity`) states this configuration; they change together.
    pub fn new(handle: &LtHandle, shape: LtGemmShape) -> Result<Self, RuntimeError> {
        bind(&handle.ctx)?;
        let lib = handle.lib;
        let mut m = Self {
            desc: std::ptr::null_mut(),
            a: std::ptr::null_mut(),
            b: std::ptr::null_mut(),
            d: std::ptr::null_mut(),
            shape,
            lib,
            ctx: handle.ctx.clone(),
        };
        construction_call("cublasLtMatmulDescCreate", || unsafe {
            (lib.fns.desc_create)(&mut m.desc, COMPUTE_32F, R_32F)
        })?;
        track(1);
        m.set_construction(&DESC_TRANSA, &OP_T)?;
        m.set_construction(&DESC_TRANSB, &OP_N)?;
        let code = match shape.input {
            LtInput::Nvfp4 => {
                m.set_construction(&DESC_A_SCALE_MODE, &SCALE_VEC16_UE4M3)?;
                m.set_construction(&DESC_B_SCALE_MODE, &SCALE_VEC16_UE4M3)?;
                R_4F_E2M1
            }
            LtInput::Fp8 => R_8F_E4M3,
        };
        let k = shape.k as i64;
        let ldd = shape.ldd as i64;
        m.a = layout(lib, code, shape.k, shape.n, k)?;
        m.b = layout(lib, code, shape.k, shape.m, k)?;
        m.d = layout(lib, R_16BF, shape.n, shape.m, ldd)?;
        Ok(m)
    }

    fn set_construction<T>(&mut self, attr: &Attr<T>, value: &T) -> Result<(), RuntimeError> {
        let (lib, desc) = (self.lib, self.desc);
        construction_call("cublasLtMatmulDescSetAttribute", || unsafe {
            (lib.fns.desc_set_attribute)(
                desc,
                attr.id,
                (value as *const T).cast(),
                attr.payload_bytes(),
            )
        })
    }

    /// Set the descriptor's A and B scale pointers.
    fn set_scale_pointers(&mut self, w_scale: u64, x_scale: u64) -> Result<(), RuntimeError> {
        for (attr, value) in [
            (&DESC_A_SCALE_POINTER, &w_scale),
            (&DESC_B_SCALE_POINTER, &x_scale),
        ] {
            // SAFETY: a live descriptor and a value of the attribute's pinned width.
            status("cublasLtMatmulDescSetAttribute", unsafe {
                (self.lib.fns.desc_set_attribute)(
                    self.desc,
                    attr.id,
                    (value as *const u64).cast(),
                    attr.payload_bytes(),
                )
            })?;
        }
        Ok(())
    }

    /// Refuse a `what` (handle or stream) of another CUDA context than the plan's.
    fn same_context(&self, ctx: &Arc<CudaContext>, what: &str) -> Result<(), RuntimeError> {
        if ctx.cu_ctx() == self.ctx.cu_ctx() {
            Ok(())
        } else {
            Err(RuntimeError::Compute(format!(
                "the cuBLASLt {what} belongs to another CUDA context than the plan"
            )))
        }
    }

    /// Up to `max` heuristic algorithms that fit in `workspace_cap` bytes and reduce split-K partials
    /// only in the F32 compute type. The query runs with the descriptor's scale pointers set to
    /// `w_scale` and `x_scale`, the operands the algorithms will be run on: for block-scaled NVFP4
    /// the library refuses the query (`CUBLAS_STATUS_INVALID_VALUE`) while they are unset. A `handle`
    /// of another CUDA context than the plan's is refused.
    ///
    /// # Safety
    /// `w_scale` and `x_scale` must address the live device scale operands of this shape, as for
    /// [`Self::launch`]: the library may read through them.
    pub unsafe fn heuristics(
        &mut self,
        handle: &LtHandle,
        workspace_cap: u64,
        max: usize,
        w_scale: u64,
        x_scale: u64,
    ) -> Result<Vec<LtHeuristicResult>, RuntimeError> {
        self.same_context(&handle.ctx, "handle")?;
        bind(&self.ctx)?;
        self.set_scale_pointers(w_scale, x_scale)?;
        let lib = self.lib;
        let pref = Preference::new(lib)?;
        pref.set(&PREF_MAX_WORKSPACE_BYTES, &workspace_cap)?;
        pref.set(&PREF_REDUCTION_SCHEME_MASK, &REDUCTION_COMPUTE_TYPE)?;
        let mut results = vec![LtHeuristicResult::ZERO; max];
        let mut returned: c_int = 0;
        construction_call("cublasLtMatmulAlgoGetHeuristic", || unsafe {
            (lib.fns.algo_get_heuristic)(
                handle.raw,
                self.desc,
                self.a,
                self.b,
                self.d,
                self.d,
                pref.raw,
                max as c_int,
                results.as_mut_ptr(),
                &mut returned,
            )
        })?;
        results.truncate(returned.max(0) as usize);
        results.retain(|r| r.state == 0 && r.workspace_size as u64 <= workspace_cap);
        Ok(results)
    }

    /// Enqueue `D = alpha * X * W^T` with `algo` on `stream`. A `handle` or `stream` of another CUDA
    /// context than the plan's is refused.
    ///
    /// # Safety
    /// Every pointer in `ops` and `workspace` must address live device memory of the sizes this
    /// shape reads and writes, owned by the plan's context, and stay live until the stream reaches
    /// this call.
    pub unsafe fn launch(
        &mut self,
        handle: &LtHandle,
        stream: &CudaStream,
        algo: &LtAlgo,
        alpha: f32,
        ops: &LtOperands,
        workspace: u64,
        workspace_size: usize,
    ) -> Result<(), RuntimeError> {
        self.same_context(&handle.ctx, "handle")?;
        self.same_context(stream.context(), "stream")?;
        bind(&self.ctx)?;
        let scale_alignment = match self.shape.input {
            LtInput::Nvfp4 => OPERAND_ALIGNMENT,
            LtInput::Fp8 => 4,
        };
        for (name, ptr, align) in [
            ("W", ops.w, OPERAND_ALIGNMENT),
            ("X", ops.x, OPERAND_ALIGNMENT),
            ("D", ops.d, OPERAND_ALIGNMENT),
            ("W scale", ops.w_scale, scale_alignment),
            ("X scale", ops.x_scale, scale_alignment),
            ("workspace", workspace, OPERAND_ALIGNMENT),
        ] {
            if ptr % align != 0 {
                return Err(RuntimeError::Compute(format!(
                    "cuBLASLt operand {name} at {ptr:#x} is not {align}-byte aligned"
                )));
            }
        }
        self.set_scale_pointers(ops.w_scale, ops.x_scale)?;
        let beta = 0.0f32;
        status(
            "cublasLtMatmul",
            (self.lib.fns.matmul)(
                handle.raw,
                self.desc,
                (&alpha as *const f32).cast(),
                ops.w as *const c_void,
                self.a,
                ops.x as *const c_void,
                self.b,
                (&beta as *const f32).cast(),
                ops.d as *const c_void,
                self.d,
                ops.d as *mut c_void,
                self.d,
                algo,
                workspace as *mut c_void,
                workspace_size,
                stream.cu_stream() as *mut c_void,
            ),
        )
    }
}

impl Drop for LtMatmul {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        for l in [self.a, self.b, self.d] {
            if !l.is_null() {
                // SAFETY: created by cublasLtMatrixLayoutCreate and not destroyed before.
                destroyed(unsafe { (self.lib.fns.layout_destroy)(l) });
            }
        }
        if !self.desc.is_null() {
            // SAFETY: created by cublasLtMatmulDescCreate.
            destroyed(unsafe { (self.lib.fns.desc_destroy)(self.desc) });
        }
    }
}

fn layout(
    lib: &'static LtLibrary,
    data_type: u32,
    rows: u64,
    cols: u64,
    ld: i64,
) -> Result<RawLayout, RuntimeError> {
    let mut raw: RawLayout = std::ptr::null_mut();
    construction_call("cublasLtMatrixLayoutCreate", || unsafe {
        (lib.fns.layout_create)(&mut raw, data_type, rows, cols, ld)
    })?;
    track(1);
    Ok(raw)
}

/// A heuristic-query preference, destroyed when the query is done.
struct Preference {
    raw: RawPref,
    lib: &'static LtLibrary,
}

impl Preference {
    fn new(lib: &'static LtLibrary) -> Result<Self, RuntimeError> {
        let mut raw: RawPref = std::ptr::null_mut();
        construction_call("cublasLtMatmulPreferenceCreate", || unsafe {
            (lib.fns.pref_create)(&mut raw)
        })?;
        track(1);
        Ok(Self { raw, lib })
    }

    fn set<T>(&self, attr: &Attr<T>, value: &T) -> Result<(), RuntimeError> {
        construction_call("cublasLtMatmulPreferenceSetAttribute", || unsafe {
            (self.lib.fns.pref_set_attribute)(
                self.raw,
                attr.id,
                (value as *const T).cast(),
                attr.payload_bytes(),
            )
        })
    }
}

impl Drop for Preference {
    fn drop(&mut self) {
        // SAFETY: created by cublasLtMatmulPreferenceCreate.
        destroyed(unsafe { (self.lib.fns.pref_destroy)(self.raw) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::mem::{align_of, offset_of, size_of};

    const FIXTURE: &str = include_str!("../../tests/fixtures/cublaslt_abi.txt");

    /// Every value this module relies on, by its fixture key.
    fn rust_abi() -> Vec<(String, u64)> {
        let mut v: Vec<(String, u64)> = vec![
            (
                "sizeof.cublasLtMatmulAlgo_t".into(),
                size_of::<LtAlgo>() as u64,
            ),
            (
                "alignof.cublasLtMatmulAlgo_t".into(),
                align_of::<LtAlgo>() as u64,
            ),
            (
                "sizeof.cublasLtMatmulHeuristicResult_t".into(),
                size_of::<LtHeuristicResult>() as u64,
            ),
            (
                "alignof.cublasLtMatmulHeuristicResult_t".into(),
                align_of::<LtHeuristicResult>() as u64,
            ),
            (
                "offsetof.cublasLtMatmulHeuristicResult_t.algo".into(),
                offset_of!(LtHeuristicResult, algo) as u64,
            ),
            (
                "offsetof.cublasLtMatmulHeuristicResult_t.workspaceSize".into(),
                offset_of!(LtHeuristicResult, workspace_size) as u64,
            ),
            (
                "offsetof.cublasLtMatmulHeuristicResult_t.state".into(),
                offset_of!(LtHeuristicResult, state) as u64,
            ),
            (
                "offsetof.cublasLtMatmulHeuristicResult_t.wavesCount".into(),
                offset_of!(LtHeuristicResult, waves_count) as u64,
            ),
            (
                "offsetof.cublasLtMatmulHeuristicResult_t.reserved".into(),
                offset_of!(LtHeuristicResult, reserved) as u64,
            ),
            ("sizeof.cublasStatus_t".into(), size_of::<LtStatus>() as u64),
            ("sizeof.cublasComputeType_t".into(), size_of::<u32>() as u64),
            ("sizeof.cudaDataType_t".into(), size_of::<u32>() as u64),
            ("sizeof.cublasOperation_t".into(), size_of::<u32>() as u64),
            ("const.CUBLAS_COMPUTE_32F".into(), COMPUTE_32F as u64),
            ("const.CUDA_R_32F".into(), R_32F as u64),
            ("const.CUDA_R_16BF".into(), R_16BF as u64),
            ("const.CUDA_R_8F_E4M3".into(), R_8F_E4M3 as u64),
            ("const.CUDA_R_4F_E2M1".into(), R_4F_E2M1 as u64),
            ("const.CUBLAS_OP_N".into(), OP_N as u64),
            ("const.CUBLAS_OP_T".into(), OP_T as u64),
            (
                "const.CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3".into(),
                SCALE_VEC16_UE4M3 as u64,
            ),
            (
                "const.CUBLASLT_REDUCTION_SCHEME_NONE".into(),
                REDUCTION_NONE as u64,
            ),
            (
                "const.CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE".into(),
                REDUCTION_COMPUTE_TYPE as u64,
            ),
        ];
        let desc: [(&str, u32, usize); 6] = [
            ("TRANSA", DESC_TRANSA.id, DESC_TRANSA.payload_bytes()),
            ("TRANSB", DESC_TRANSB.id, DESC_TRANSB.payload_bytes()),
            (
                "A_SCALE_POINTER",
                DESC_A_SCALE_POINTER.id,
                DESC_A_SCALE_POINTER.payload_bytes(),
            ),
            (
                "B_SCALE_POINTER",
                DESC_B_SCALE_POINTER.id,
                DESC_B_SCALE_POINTER.payload_bytes(),
            ),
            (
                "A_SCALE_MODE",
                DESC_A_SCALE_MODE.id,
                DESC_A_SCALE_MODE.payload_bytes(),
            ),
            (
                "B_SCALE_MODE",
                DESC_B_SCALE_MODE.id,
                DESC_B_SCALE_MODE.payload_bytes(),
            ),
        ];
        for (name, id, bytes) in desc {
            v.push((format!("const.CUBLASLT_MATMUL_DESC_{name}"), id as u64));
            v.push((format!("size.CUBLASLT_MATMUL_DESC_{name}"), bytes as u64));
        }
        let pref: [(&str, u32, usize); 2] = [
            (
                "MAX_WORKSPACE_BYTES",
                PREF_MAX_WORKSPACE_BYTES.id,
                PREF_MAX_WORKSPACE_BYTES.payload_bytes(),
            ),
            (
                "REDUCTION_SCHEME_MASK",
                PREF_REDUCTION_SCHEME_MASK.id,
                PREF_REDUCTION_SCHEME_MASK.payload_bytes(),
            ),
        ];
        for (name, id, bytes) in pref {
            v.push((format!("const.CUBLASLT_MATMUL_PREF_{name}"), id as u64));
            v.push((format!("size.CUBLASLT_MATMUL_PREF_{name}"), bytes as u64));
        }
        for (name, id, bytes) in ALGO_CONFIG {
            v.push((format!("const.CUBLASLT_ALGO_CONFIG_{name}"), id as u64));
            v.push((format!("size.CUBLASLT_ALGO_CONFIG_{name}"), bytes as u64));
        }
        v
    }

    /// Every key whose fixture value differs from the Rust side, or that the fixture lacks.
    fn abi_mismatches(fixture: &str) -> Vec<String> {
        let values: HashMap<&str, u64> = fixture
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
            .filter_map(|l| {
                let mut it = l.split_whitespace();
                Some((it.next()?, it.next()?.parse().ok()?))
            })
            .collect();
        rust_abi()
            .into_iter()
            .filter_map(|(key, rust)| match values.get(key.as_str()) {
                Some(&v) if v == rust => None,
                Some(&v) => Some(format!("{key}: header {v}, Rust {rust}")),
                None => Some(format!("{key}: missing from the fixture")),
            })
            .collect()
    }

    /// The header version a fixture records.
    fn fixture_header_version(fixture: &str) -> Option<usize> {
        fixture
            .lines()
            .find_map(|l| l.strip_prefix("header_version "))
            .and_then(|v| v.trim().parse().ok())
    }

    #[test]
    fn abi_matches_the_header_fixture() {
        let header = fixture_header_version(FIXTURE);
        assert!(
            header.is_some_and(version_qualifies),
            "the fixture must be generated from a cublasLt.h of version {MIN_VERSION} or newer, \
             got {header:?}"
        );
        let bad = abi_mismatches(FIXTURE);
        assert!(bad.is_empty(), "ABI mismatches: {bad:#?}");
    }

    #[test]
    fn a_fixture_with_one_offset_changed_fails() {
        let key = "offsetof.cublasLtMatmulHeuristicResult_t.wavesCount";
        let edit = |line: &str| match line.split_whitespace().next() {
            Some(k) if k == key => format!("{key} 80"),
            _ => line.to_string(),
        };
        let changed: Vec<String> = FIXTURE.lines().map(edit).collect();
        assert_eq!(
            abi_mismatches(&changed.join("\n")),
            vec![format!("{key}: header 80, Rust 76")]
        );
        let without: Vec<&str> = FIXTURE
            .lines()
            .filter(|l| !l.starts_with("const.CUDA_R_4F_E2M1 "))
            .collect();
        assert_eq!(
            abi_mismatches(&without.join("\n")),
            vec!["const.CUDA_R_4F_E2M1: missing from the fixture".to_string()]
        );
    }

    #[test]
    fn an_old_or_unversioned_header_fixture_is_refused() {
        assert_eq!(fixture_header_version("library_version 130600\n"), None);
        let old = "header_version 120205\n";
        assert!(!fixture_header_version(old).is_some_and(version_qualifies));
        assert!(fixture_header_version("header_version 130800\n").is_some_and(version_qualifies));
    }

    #[test]
    fn the_version_gate_is_12_8() {
        assert!(!version_qualifies(120799));
        assert!(version_qualifies(120800));
        assert!(version_qualifies(130600));
        assert!(!version_qualifies(0));
    }

    #[test]
    fn a_missing_library_is_an_error_naming_each_candidate() {
        // SAFETY: neither name exists, so nothing is loaded.
        let err = match unsafe { load(&["liblumen-absent-a.so", "liblumen-absent-b.so.13"]) } {
            Ok(_) => panic!("an absent library cannot load"),
            Err(e) => e,
        };
        assert!(err.contains("liblumen-absent-a.so:"), "{err}");
        assert!(err.contains("liblumen-absent-b.so.13:"), "{err}");
        assert!(err.contains("120800"), "{err}");
    }

    #[test]
    fn a_library_without_the_symbols_is_rejected_by_name() {
        // The C library loads everywhere and exports none of cuBLASLt's symbols.
        #[cfg(target_os = "linux")]
        let libc_name = "libc.so.6";
        #[cfg(target_os = "macos")]
        let libc_name = "/usr/lib/libSystem.B.dylib";
        // SAFETY: the C library is already loaded and lacks cuBLASLt's symbols.
        let err = match unsafe { load(&[libc_name]) } {
            Ok(_) => panic!("the C library is not cuBLASLt"),
            Err(e) => e,
        };
        assert!(
            err.contains(&format!("{libc_name}: lacks cublasLtCreate")),
            "{err}"
        );
    }
}
