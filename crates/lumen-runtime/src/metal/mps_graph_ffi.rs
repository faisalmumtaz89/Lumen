//! Apple MPSGraph BF16 GEMM bindings.
//!
//! `MPSMatrixMultiplication` (the legacy MPS BLAS-3 kernel) does **not**
//! support BF16 — it asserts at submission time on M3 Ultra. `MPSGraph`
//! (the inference-engine API introduced alongside Core ML) does support
//! BF16 GEMM and microbenches at ~7.5 TFLOPs on the GDN qkv-proj shape
//! (M=131, K=4096, N=8192) — measured 1.166 ms median.
//!
//! This module wires MPSGraph through Lumen's existing raw `objc_msgSend`
//! FFI pattern (no `metal-rs` / `objc2` dependency, per project policy in
//! `ffi.rs`). Public surface:
//!
//!   * [`MpsGraphContext`] — process-wide handle to MPSGraphDevice +
//!     compiled-graph cache (keyed by `(M, K, N)`) + dedicated GPU
//!     buffers for the BF16 weight slices that MPSGraph reads.
//!   * [`encode_bf16_matmul_into_cb`] — encode one BF16 matmul
//!     `Y[M,N] = X[M,K] @ W^T[K,N]` into the caller's existing
//!     MTLCommandBuffer. Apple's MPSGraph wrapper may internally call
//!     `commitAndContinue` for graphs that exceed undocumented
//!     resource limits, so the caller's `MetalCommandBuffer` is
//!     rebound to the new root CB on return.
//!   * [`encode_bf16_matmul_sync`] — alternative path that submits the
//!     matmul on a dedicated MPSGraph queue and blocks until complete.
//!     Useful when MPSGraph commit-and-continue would break the
//!     caller's CB-management invariants. Currently unused (the in-CB
//!     path proved equivalent or faster).
//!
//! Lifetime: compiled graphs live for the process lifetime (drop happens
//! at exit). The graph compile is ~3-5 ms; the encode path is the fast
//! part. Lumen's prefill shapes are static across all 2048-token contexts
//! so cache hit rate is 100% after the first prefill.
//!
//! # Math contract
//!
//! Lumen weights for BF16 are stored as **`[N, K]` row-major** (i.e.
//! transposed) with `bfloat` element type. The matmul computes
//! `Y[m, n] = sum_k X[m, k] * W[n, k]` — algebraically `Y = X * W^T`.
//!
//! Inside the graph the weight tensor is declared shape `[N, K]` and
//! explicitly transposed to `[K, N]` before `matrixMultiplication`. The
//! input X arrives as F32 `[M, K]` and is cast to BF16 inside the graph;
//! the BF16 result is cast back to F32 before the output buffer write.
//!
//! Both casts and the transpose are graph-time metadata operations and
//! fold into the runtime kernel (verified empirically — no extra dispatch
//! overhead beyond the matmul itself in the microbench at
//! ~1.17 ms vs Lumen's `tiled_matmul_bf16_k64` at ~1.34 ms).
//!
//! # Buffer-offset workaround
//!
//! MPSGraphTensorData has no public buffer-with-offset initialiser, and
//! `MTLDevice newBufferWithBytesNoCopy:` requires a page-aligned source
//! pointer. Lumen's per-tensor offsets inside the LBC blob are 32-byte
//! aligned but not 16 KB page-aligned. We work around this by allocating
//! a one-time dedicated `MTLBuffer` per BF16 weight slice and copying
//! the source bytes in — CPU memcpy for shared-storage source (mmap'd
//! LBC), GPU blit for private-storage source (GPU-resident weights).
//! The dedicated buffers are cached for the process lifetime; worst-case
//! extra VRAM is ~2.4 GB for Qwen3.5-9B BF16.

use std::collections::HashMap;
use std::ffi::{c_char, c_void, CString};
use std::ptr;
use std::sync::{Mutex, OnceLock};

use super::ffi::{MetalBuffer, MetalCommandBuffer, MetalCommandQueue, MetalDevice};

// ============================================================================
// Objective-C runtime FFI (minimal duplication from ffi.rs — local helpers
// avoid exposing the runtime to the rest of the crate)
// ============================================================================

type ObjcId = *mut c_void;
type ObjcClass = *mut c_void;
type ObjcSel = *mut c_void;

#[link(name = "objc", kind = "dylib")]
extern "C" {
    fn objc_getClass(name: *const c_char) -> ObjcClass;
    fn sel_registerName(name: *const c_char) -> ObjcSel;
    fn objc_msgSend(receiver: ObjcId, sel: ObjcSel, ...) -> ObjcId;
}

#[link(name = "MetalPerformanceShaders", kind = "framework")]
extern "C" {}

#[link(name = "MetalPerformanceShadersGraph", kind = "framework")]
extern "C" {}

#[inline]
fn cls(name: &str) -> ObjcClass {
    let cstr = CString::new(name).unwrap();
    unsafe { objc_getClass(cstr.as_ptr()) }
}

#[inline]
fn sel_cached(slot: &OnceLock<usize>, name: &'static str) -> ObjcSel {
    let val = *slot.get_or_init(|| {
        let cstr = CString::new(name).unwrap();
        unsafe { sel_registerName(cstr.as_ptr()) as usize }
    });
    val as ObjcSel
}

#[inline]
unsafe fn msg0(obj: ObjcId, s: ObjcSel) -> ObjcId {
    type F = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
    let f: F = std::mem::transmute(objc_msgSend as *const c_void);
    f(obj, s)
}

#[inline]
unsafe fn retain(obj: ObjcId) -> ObjcId {
    static SEL: OnceLock<usize> = OnceLock::new();
    msg0(obj, sel_cached(&SEL, "retain"))
}

#[inline]
unsafe fn release(obj: ObjcId) {
    static SEL: OnceLock<usize> = OnceLock::new();
    msg0(obj, sel_cached(&SEL, "release"));
}

// ============================================================================
// MPSDataType constants (from MPSCoreTypes.h)
// ============================================================================

/// `MPSDataTypeFloatBit | 32` — F32.
const MPS_DTYPE_F32: u32 = 0x1000_0020;
/// `MPSDataTypeAlternateEncodingBit | MPSDataTypeFloat16` — BF16.
const MPS_DTYPE_BF16: u32 = 0x9000_0010;

// ============================================================================
// NSNumber / NSArray helpers — used to build the shape arrays
// ============================================================================

/// Wrap a `usize` dimension as an `NSNumber*` (retained, caller releases).
unsafe fn nsnumber_with_uinteger(value: u64) -> ObjcId {
    static SEL_NUM: OnceLock<usize> = OnceLock::new();
    let ns_number_cls = cls("NSNumber");
    type Fn1 = unsafe extern "C" fn(ObjcId, ObjcSel, u64) -> ObjcId;
    let f: Fn1 = std::mem::transmute(objc_msgSend as *const c_void);
    let n = f(ns_number_cls as ObjcId,
        sel_cached(&SEL_NUM, "numberWithUnsignedInteger:"), value);
    // numberWithUnsignedInteger: returns autoreleased — retain to extend
    // lifetime beyond the autoreleasepool boundary.
    retain(n);
    n
}

/// Build an `NSArray<NSNumber*>*` from a sequence of dimensions.
///
/// Returned array is retained; caller must `release()`. The contained
/// NSNumbers are owned by the array (retained when added) and are also
/// released by us once placed in the array (NSArray retains its members).
unsafe fn nsarray_with_dims(dims: &[u64]) -> ObjcId {
    static SEL_ALLOC: OnceLock<usize> = OnceLock::new();
    static SEL_INIT_WITH_OBJECTS: OnceLock<usize> = OnceLock::new();

    let cls_array = cls("NSArray");
    let alloced = msg0(cls_array as ObjcId, sel_cached(&SEL_ALLOC, "alloc"));

    // Build a buffer of NSNumber pointers.
    let mut numbers: Vec<ObjcId> = Vec::with_capacity(dims.len());
    for &d in dims {
        numbers.push(nsnumber_with_uinteger(d));
    }

    // -[NSArray initWithObjects:count:] — non-variadic; expects a
    // const id _Nonnull *objects + NSUInteger.
    type Fn1 = unsafe extern "C" fn(ObjcId, ObjcSel, *const ObjcId, u64) -> ObjcId;
    let f: Fn1 = std::mem::transmute(objc_msgSend as *const c_void);
    let arr = f(
        alloced,
        sel_cached(&SEL_INIT_WITH_OBJECTS, "initWithObjects:count:"),
        numbers.as_ptr(),
        dims.len() as u64,
    );

    // NSArray retained each NSNumber; release our local refs.
    for n in numbers {
        release(n);
    }

    arr
}

// ============================================================================
// NSMutableDictionary helpers — used to build the feeds dict
// ============================================================================

/// Allocate an empty `NSMutableDictionary*` (retained, caller releases).
unsafe fn nsmutdict_new() -> ObjcId {
    static SEL_ALLOC: OnceLock<usize> = OnceLock::new();
    static SEL_INIT: OnceLock<usize> = OnceLock::new();
    let c = cls("NSMutableDictionary");
    let alloced = msg0(c as ObjcId, sel_cached(&SEL_ALLOC, "alloc"));
    msg0(alloced, sel_cached(&SEL_INIT, "init"))
}

/// `-[NSMutableDictionary setObject:forKey:]`.
unsafe fn nsmutdict_set(dict: ObjcId, value: ObjcId, key: ObjcId) {
    static SEL: OnceLock<usize> = OnceLock::new();
    type Fn1 = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, ObjcId);
    let f: Fn1 = std::mem::transmute(objc_msgSend as *const c_void);
    f(dict, sel_cached(&SEL, "setObject:forKey:"), value, key);
}

// ============================================================================
// MPSGraph types
// ============================================================================

/// One compiled graph (per (M, K, N) shape). Owns:
///   * the MPSGraph itself
///   * 2 placeholder tensors (X, W) — referenced by feeds
///   * the result tensor (Y) — referenced in targetTensors
///
/// All pointers are retained at construction; released in Drop.
struct CompiledGraph {
    graph: ObjcId,         // MPSGraph*
    x_placeholder: ObjcId, // MPSGraphTensor* (F32 [M, K])
    w_placeholder: ObjcId, // MPSGraphTensor* (BF16 [N, K])
    y_tensor: ObjcId,      // MPSGraphTensor* (F32 [M, N]) — final cast back
}

unsafe impl Send for CompiledGraph {}
unsafe impl Sync for CompiledGraph {}

impl Drop for CompiledGraph {
    fn drop(&mut self) {
        // Placeholders + intermediate tensors are owned by the MPSGraph;
        // we release them here to balance the retains we performed at
        // build time. MPSGraph holds its own internal references too,
        // so the placeholders survive until the graph itself drops.
        unsafe {
            release(self.x_placeholder);
            release(self.w_placeholder);
            release(self.y_tensor);
            release(self.graph);
        }
    }
}

/// Process-wide MPSGraph context. Owns the `MPSGraphDevice` (one per
/// `MetalDevice`), the dedicated `MTLCommandQueue` reserved for
/// MPSGraph submissions, and a `(M, K, N) -> CompiledGraph` cache.
///
/// Constructed lazily via [`get_or_init`]. The cache uses a `Mutex` for
/// the unlikely case of concurrent graph compilation; the hot encode
/// path takes the lock once per matmul which is a single fast lookup
/// (no work under the mutex other than `.get()`).
///
/// The `view_buf_cache` additionally caches per-tensor dedicated
/// `MTLBuffer`s keyed by `(base_ptr, offset, length)`. Lumen weight
/// tensors live for the process lifetime (mmap'd), so the
/// `(base_ptr, offset)` pair is a stable identity. The cache eliminates
/// allocation + copy cost on the hot path; each BF16 weight tensor is
/// duplicated at most once.
///
/// A dedicated `cmd_queue` separates MPSGraph submissions from Lumen's
/// own prefill / decode command queue, so an MPSGraph internal
/// `commitAndContinue` cannot interfere with the caller's outer
/// `MTLCommandBuffer`.
pub struct MpsGraphContext {
    /// MPSGraphDevice* — wraps the underlying MetalDevice.
    graph_device: ObjcId,
    /// Dedicated MTLCommandQueue used by [`encode_bf16_matmul_sync`]
    /// (alternative path) and by the blit-copy that stages
    /// private-storage source weights into the dedicated MPSGraph
    /// buffers. The default in-CB path encodes onto the caller's
    /// existing queue, so this is touched only on blit-copy fallback
    /// and the alternative sync path.
    #[allow(dead_code)]
    cmd_queue_wrapper: MetalCommandQueue,
    /// Raw MTLCommandQueue id from `cmd_queue_wrapper`, cached for the
    /// MPSGraph FFI calls (avoids re-borrowing the wrapper).
    cmd_queue_raw: ObjcId,
    cache: Mutex<HashMap<(u32, u32, u32), Box<CompiledGraph>>>,
    view_buf_cache: Mutex<HashMap<(usize, u64, u64), ObjcId>>,
}

unsafe impl Send for MpsGraphContext {}
unsafe impl Sync for MpsGraphContext {}

impl Drop for MpsGraphContext {
    fn drop(&mut self) {
        unsafe {
            if let Ok(mut views) = self.view_buf_cache.lock() {
                for (_, buf) in views.drain() {
                    if !buf.is_null() { release(buf) };
                }
            }
            // `cmd_queue_wrapper` drop releases the queue itself.
            release(self.graph_device);
        }
    }
}

static MPS_GRAPH_CONTEXT: OnceLock<Option<MpsGraphContext>> = OnceLock::new();

/// Get the process-wide MPSGraph context, initialising it on first call.
///
/// Returns `None` if the MPSGraph framework is unavailable on this
/// platform (e.g. macOS < 14 or no Metal device) — callers fall back to
/// the existing custom kernels.
///
/// Typically called once at backend init time. Subsequent dispatch-site
/// lookups use [`get`] which is no-arg.
pub fn get_or_init(device: &MetalDevice) -> Option<&'static MpsGraphContext> {
    MPS_GRAPH_CONTEXT
        .get_or_init(|| {
            // Verify MPSGraph + MPSGraphDevice classes are linked.
            let g_cls = cls("MPSGraph");
            let gd_cls = cls("MPSGraphDevice");
            if g_cls.is_null() || gd_cls.is_null() {
                return None;
            }
            unsafe {
                // +[MPSGraphDevice deviceWithMTLDevice:]
                static SEL_DEVICE: OnceLock<usize> = OnceLock::new();
                type Fn1 = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId) -> ObjcId;
                let f: Fn1 = std::mem::transmute(objc_msgSend as *const c_void);
                let gd = f(
                    gd_cls as ObjcId,
                    sel_cached(&SEL_DEVICE, "deviceWithMTLDevice:"),
                    device.raw(),
                );
                if gd.is_null() {
                    return None;
                }
                retain(gd); // autoreleased -> retain for process lifetime

                // Allocate a dedicated MetalCommandQueue for MPSGraph
                // submissions AND for re-allocating Lumen's main CB
                // after commit-and-wait drains prior work.
                let queue_wrapper = match device.new_command_queue() {
                    Some(q) => q,
                    None => {
                        release(gd);
                        return None;
                    }
                };
                let q_raw = queue_wrapper.raw();

                Some(MpsGraphContext {
                    graph_device: gd,
                    cmd_queue_wrapper: queue_wrapper,
                    cmd_queue_raw: q_raw,
                    cache: Mutex::new(HashMap::new()),
                    view_buf_cache: Mutex::new(HashMap::new()),
                })
            }
        })
        .as_ref()
}

/// Get the MPSGraph context after it has been initialised. Returns
/// `None` if [`get_or_init`] has not been called yet OR the framework
/// is unavailable. Dispatch-site lookups use this rather than threading
/// a `&MetalDevice` through every encode function.
#[inline]
pub fn get() -> Option<&'static MpsGraphContext> {
    MPS_GRAPH_CONTEXT.get().and_then(|opt| opt.as_ref())
}

// ============================================================================
// Graph construction
// ============================================================================

/// Build a compiled MPSGraph for a single BF16 matmul of the given
/// shape. The graph computes:
///
///   `Y[M, N] = (cast<F32>(  matmul( cast<BF16>(X), transpose(W) )  ))`
///
/// with X: F32 [M, K], W: BF16 [N, K] (Lumen weight layout — transposed
/// so we transpose in-graph back to [K, N] before the matmul).
fn build_graph(m: u32, k: u32, n: u32) -> Option<CompiledGraph> {
    static SEL_NEW: OnceLock<usize> = OnceLock::new();
    static SEL_PLACEHOLDER: OnceLock<usize> = OnceLock::new();
    static SEL_MATMUL: OnceLock<usize> = OnceLock::new();
    static SEL_TRANSPOSE: OnceLock<usize> = OnceLock::new();
    static SEL_CAST: OnceLock<usize> = OnceLock::new();

    let g_cls = cls("MPSGraph");
    if g_cls.is_null() {
        return None;
    }

    unsafe {
        // -[MPSGraph new]
        let graph = msg0(g_cls as ObjcId, sel_cached(&SEL_NEW, "new"));
        if graph.is_null() {
            return None;
        }

        // X placeholder: F32 [M, K]
        let shape_x = nsarray_with_dims(&[m as u64, k as u64]);
        type FnPh = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, u32, ObjcId) -> ObjcId;
        let f_ph: FnPh = std::mem::transmute(objc_msgSend as *const c_void);
        let x_ph = f_ph(
            graph,
            sel_cached(&SEL_PLACEHOLDER, "placeholderWithShape:dataType:name:"),
            shape_x,
            MPS_DTYPE_F32,
            ptr::null_mut(),
        );
        release(shape_x);
        if x_ph.is_null() {
            release(graph);
            return None;
        }
        retain(x_ph);

        // W placeholder: BF16 [N, K] (Lumen weight layout)
        let shape_w = nsarray_with_dims(&[n as u64, k as u64]);
        let w_ph = f_ph(
            graph,
            sel_cached(&SEL_PLACEHOLDER, "placeholderWithShape:dataType:name:"),
            shape_w,
            MPS_DTYPE_BF16,
            ptr::null_mut(),
        );
        release(shape_w);
        if w_ph.is_null() {
            release(x_ph);
            release(graph);
            return None;
        }
        retain(w_ph);

        // X_bf16 = cast(X, BF16)
        type FnCast = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, u32, ObjcId) -> ObjcId;
        let f_cast: FnCast = std::mem::transmute(objc_msgSend as *const c_void);
        let x_bf = f_cast(
            graph,
            sel_cached(&SEL_CAST, "castTensor:toType:name:"),
            x_ph,
            MPS_DTYPE_BF16,
            ptr::null_mut(),
        );
        if x_bf.is_null() {
            release(w_ph);
            release(x_ph);
            release(graph);
            return None;
        }

        // W_t = transpose(W, dim0=0, dim1=1) — [N,K] -> [K,N]
        type FnTrans = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, u64, u64, ObjcId) -> ObjcId;
        let f_trans: FnTrans = std::mem::transmute(objc_msgSend as *const c_void);
        let w_t = f_trans(
            graph,
            sel_cached(&SEL_TRANSPOSE, "transposeTensor:dimension:withDimension:name:"),
            w_ph,
            0,
            1,
            ptr::null_mut(),
        );
        if w_t.is_null() {
            release(w_ph);
            release(x_ph);
            release(graph);
            return None;
        }

        // Y_bf = matmul(X_bf, W_t)
        type FnMM = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, ObjcId, ObjcId) -> ObjcId;
        let f_mm: FnMM = std::mem::transmute(objc_msgSend as *const c_void);
        let y_bf = f_mm(
            graph,
            sel_cached(&SEL_MATMUL, "matrixMultiplicationWithPrimaryTensor:secondaryTensor:name:"),
            x_bf,
            w_t,
            ptr::null_mut(),
        );
        if y_bf.is_null() {
            release(w_ph);
            release(x_ph);
            release(graph);
            return None;
        }

        // Y = cast(Y_bf, F32)
        let y_f32 = f_cast(
            graph,
            sel_cached(&SEL_CAST, "castTensor:toType:name:"),
            y_bf,
            MPS_DTYPE_F32,
            ptr::null_mut(),
        );
        if y_f32.is_null() {
            release(w_ph);
            release(x_ph);
            release(graph);
            return None;
        }
        retain(y_f32);

        Some(CompiledGraph {
            graph,
            x_placeholder: x_ph,
            w_placeholder: w_ph,
            y_tensor: y_f32,
        })
    }
}

// ============================================================================
// MPSGraphTensorData binding
// ============================================================================

/// Wrap an `MTLBuffer` as `MPSGraphTensorData*` with the given shape +
/// dtype. Returned object is retained; caller must release.
unsafe fn make_tensor_data(buf: &MetalBuffer, shape: &[u64], dtype: u32) -> ObjcId {
    static SEL_ALLOC: OnceLock<usize> = OnceLock::new();
    static SEL_INIT: OnceLock<usize> = OnceLock::new();
    let c = cls("MPSGraphTensorData");
    let alloced = msg0(c as ObjcId, sel_cached(&SEL_ALLOC, "alloc"));
    let shape_arr = nsarray_with_dims(shape);

    // -[MPSGraphTensorData initWithMTLBuffer:shape:dataType:]
    type Fn1 = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, ObjcId, u32) -> ObjcId;
    let f: Fn1 = std::mem::transmute(objc_msgSend as *const c_void);
    let td = f(
        alloced,
        sel_cached(&SEL_INIT, "initWithMTLBuffer:shape:dataType:"),
        buf.raw(),
        shape_arr,
        dtype,
    );
    release(shape_arr);
    td
}

// ============================================================================
// Public entry point
// ============================================================================

impl MpsGraphContext {
    /// Look up (or build) the compiled graph for shape `(M, K, N)`. The
    /// returned reference is valid for the lifetime of the context (i.e.
    /// the entire process). Holds the cache mutex only across the
    /// `HashMap` access — graph build is done under the lock to avoid a
    /// race where two threads compile the same shape simultaneously.
    fn graph_for(&self, m: u32, k: u32, n: u32) -> Option<&CompiledGraph> {
        let mut cache = self.cache.lock().ok()?;
        if !cache.contains_key(&(m, k, n)) {
            let g = build_graph(m, k, n)?;
            cache.insert((m, k, n), Box::new(g));
        }
        // SAFETY: We never remove entries from the cache, and the
        // `Box<CompiledGraph>` keeps the inner allocation stable. The
        // returned reference is tied to the context's lifetime, not the
        // mutex guard.
        let ptr_box: *const CompiledGraph = &**cache.get(&(m, k, n))?;
        unsafe { Some(&*ptr_box) }
    }
}

/// Encode a BF16 matmul `Y[M,N] = X[M,K] @ W^T[K,N]` synchronously on
/// MPSGraph's dedicated command queue.
///
/// To preserve cross-CB ordering with the caller's `cmd`, the caller
/// MUST commit + wait on `cmd` BEFORE invoking this function (so prior
/// writes to `x_buf` are visible). After this function returns, the
/// matmul output in `y_buf` is fully visible to any subsequent
/// encoding on a fresh CB.
///
/// In practice the GDN dispatch sites do:
///   1. Encode all preceding ops onto an encoder on `cmd`.
///   2. `enc.end_encoding(); cmd.commit_and_swap_to_fresh(...)` —
///      drains `cmd` and rebinds it to a brand-new CB.
///   3. Call this function — submits + waits on MPSGraph queue.
///   4. Continue encoding on the fresh `cmd`.
///
/// Preconditions:
///   * `x_buf` contains F32 input at offset 0 (size `M*K*4` bytes).
///   * `w_buf_offset` points to BF16 weights inside `w_buf`
///     (`N*K*2` bytes valid from that offset).
///   * `y_buf` is the F32 output destination at offset 0
///     (size `M*N*4` bytes).
///
/// Returns `Ok(())` on success, `Err(reason)` on any FFI failure
/// (caller should fall back to the custom kernel path).
/// In-CB variant: encode the MPSGraph dispatch INTO the caller's
/// existing `MetalCommandBuffer` instead of running on a separate
/// queue + commit-and-wait. Eliminates the per-call sync overhead
/// (48 sync points per BF16 prefill in the GDN path), but may
/// trigger an internal `commitAndContinue` on Apple's wrapper for
/// graphs that exceed undocumented size limits. The wrapper's
/// `rootCommandBuffer` is read after encoding; if it differs from
/// the input CB, the caller's `MetalCommandBuffer` is rebound to
/// the new root so subsequent compute encoders land on an
/// uncommitted CB.
///
/// Preconditions identical to `encode_bf16_matmul_sync`, plus:
///   * `cmd`'s current encoder must have ended.
///   * `cmd` must not yet be committed.
///
/// Returns `Ok(())` on success, `Err(reason)` on FFI failure.
pub fn encode_bf16_matmul_into_cb(
    ctx: &MpsGraphContext,
    cmd: &MetalCommandBuffer,
    x_buf: &MetalBuffer,
    w_buf: &MetalBuffer,
    w_buf_offset: u64,
    y_buf: &MetalBuffer,
    m: u32,
    k: u32,
    n: u32,
) -> Result<(), &'static str> {
    static SEL_CB_WITH: OnceLock<usize> = OnceLock::new();
    static SEL_ENCODE: OnceLock<usize> = OnceLock::new();
    static SEL_ROOT: OnceLock<usize> = OnceLock::new();

    if m == 0 || k == 0 || n == 0 {
        return Err("zero dimension");
    }
    let cg = match ctx.graph_for(m, k, n) {
        Some(g) => g,
        None => return Err("graph build failed"),
    };

    let cb_raw = cmd.raw_command_buffer();
    if cb_raw.is_null() {
        return Err("command buffer null");
    }

    unsafe {
        // +[MPSCommandBuffer commandBufferWithCommandBuffer:] —
        // wraps the existing MTLCommandBuffer for MPSGraph encoding.
        let mcb_cls = cls("MPSCommandBuffer");
        if mcb_cls.is_null() {
            return Err("MPSCommandBuffer class not found");
        }
        type FnCBWith = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId) -> ObjcId;
        let f_cbwith: FnCBWith = std::mem::transmute(objc_msgSend as *const c_void);
        let mcb = f_cbwith(
            mcb_cls as ObjcId,
            sel_cached(&SEL_CB_WITH, "commandBufferWithCommandBuffer:"),
            cb_raw,
        );
        if mcb.is_null() {
            return Err("MPSCommandBuffer alloc failed");
        }
        retain(mcb);

        // Build feeds + results.
        let feeds = nsmutdict_new();
        if feeds.is_null() {
            release(mcb);
            return Err("feeds dict alloc failed");
        }
        let x_data = make_tensor_data(x_buf, &[m as u64, k as u64], MPS_DTYPE_F32);
        let w_data = ctx.make_w_tensor_data(w_buf, w_buf_offset, n, k);
        if x_data.is_null() || w_data.is_null() {
            if !x_data.is_null() { release(x_data); }
            if !w_data.is_null() { release(w_data); }
            release(feeds);
            release(mcb);
            return Err("tensor data alloc failed");
        }
        nsmutdict_set(feeds, x_data, cg.x_placeholder);
        nsmutdict_set(feeds, w_data, cg.w_placeholder);

        let results = nsmutdict_new();
        let y_data = make_tensor_data(y_buf, &[m as u64, n as u64], MPS_DTYPE_F32);
        if results.is_null() || y_data.is_null() {
            if !y_data.is_null() { release(y_data); }
            if !results.is_null() { release(results); }
            release(x_data);
            release(w_data);
            release(feeds);
            release(mcb);
            return Err("results dict alloc failed");
        }
        nsmutdict_set(results, y_data, cg.y_tensor);

        // -[MPSGraph encodeToCommandBuffer:feeds:targetOperations:resultsDictionary:executionDescriptor:]
        // Encodes the graph's GPU work into the wrapped command
        // buffer. MPSGraph MAY call commitAndContinue internally
        // for large graphs; we cope by reading rootCommandBuffer
        // afterwards.
        type FnEnc = unsafe extern "C" fn(
            ObjcId, ObjcSel, ObjcId, ObjcId, ObjcId, ObjcId, ObjcId,
        );
        let f_enc: FnEnc = std::mem::transmute(objc_msgSend as *const c_void);
        f_enc(
            cg.graph,
            sel_cached(&SEL_ENCODE,
                "encodeToCommandBuffer:feeds:targetOperations:resultsDictionary:executionDescriptor:"),
            mcb,
            feeds,
            ptr::null_mut(),
            results,
            ptr::null_mut(),
        );

        // If MPSGraph swapped the underlying MTLCB, the caller must
        // continue with the new one — otherwise subsequent encoders
        // assert "status < Committed".
        let new_cb = msg0(mcb, sel_cached(&SEL_ROOT, "rootCommandBuffer"));
        if !new_cb.is_null() && new_cb != cb_raw {
            cmd.replace_raw_command_buffer(new_cb);
        }

        release(y_data);
        release(results);
        release(x_data);
        release(w_data);
        release(feeds);
        release(mcb);
    }

    let _ = cg;
    Ok(())
}

#[allow(dead_code)]
pub fn encode_bf16_matmul_sync(
    ctx: &MpsGraphContext,
    x_buf: &MetalBuffer,
    w_buf: &MetalBuffer,
    w_buf_offset: u64,
    y_buf: &MetalBuffer,
    m: u32,
    k: u32,
    n: u32,
) -> Result<(), &'static str> {
    static SEL_RUN: OnceLock<usize> = OnceLock::new();

    if m == 0 || k == 0 || n == 0 {
        return Err("zero dimension");
    }
    let cg = match ctx.graph_for(m, k, n) {
        Some(g) => g,
        None => return Err("graph build failed"),
    };

    unsafe {
        // Build feeds dictionary { x_ph: x_data, w_ph: w_data }.
        let feeds = nsmutdict_new();
        if feeds.is_null() {
            return Err("feeds dict alloc failed");
        }

        let x_data = make_tensor_data(x_buf, &[m as u64, k as u64], MPS_DTYPE_F32);
        let w_data = ctx.make_w_tensor_data(w_buf, w_buf_offset, n, k);
        if x_data.is_null() || w_data.is_null() {
            let xr = x_data.is_null();
            let wr = w_data.is_null();
            if !x_data.is_null() { release(x_data); }
            if !w_data.is_null() { release(w_data); }
            release(feeds);
            return Err(if xr { "x_data alloc failed" } else if wr { "w_data alloc failed" } else { "tensor data alloc failed" });
        }
        nsmutdict_set(feeds, x_data, cg.x_placeholder);
        nsmutdict_set(feeds, w_data, cg.w_placeholder);

        // Results dictionary { y_tensor: y_data }.
        let results = nsmutdict_new();
        let y_data = make_tensor_data(y_buf, &[m as u64, n as u64], MPS_DTYPE_F32);
        if results.is_null() || y_data.is_null() {
            if !y_data.is_null() { release(y_data); }
            if !results.is_null() { release(results); }
            release(x_data);
            release(w_data);
            release(feeds);
            return Err("results dict alloc failed");
        }
        nsmutdict_set(results, y_data, cg.y_tensor);

        // -[MPSGraph runWithMTLCommandQueue:feeds:targetOperations:resultsDictionary:]
        // This variant submits its work to the queue and blocks until
        // the GPU has written into the resultsDictionary buffers.
        // Apple's MPSGraph.h marks this as the synchronous run path.
        type FnRun = unsafe extern "C" fn(
            ObjcId, ObjcSel, ObjcId, ObjcId, ObjcId, ObjcId,
        );
        let f: FnRun = std::mem::transmute(objc_msgSend as *const c_void);
        f(
            cg.graph,
            sel_cached(&SEL_RUN,
                "runWithMTLCommandQueue:feeds:targetOperations:resultsDictionary:"),
            ctx.cmd_queue_raw,
            feeds,
            ptr::null_mut(),
            results,
        );

        release(y_data);
        release(results);
        release(x_data);
        release(w_data);
        release(feeds);
    }

    let _ = cg;
    Ok(())
}

impl MpsGraphContext {
    /// Borrow the dedicated `MetalCommandQueue` reserved for MPSGraph
    /// submissions. Provided for the alternative
    /// [`encode_bf16_matmul_sync`] path that runs MPSGraph on a
    /// separate queue and may need to allocate a fresh CB rooted on
    /// it. Unused by the default in-CB path; marked `#[allow(dead_code)]`
    /// to preserve the utility for future revisit.
    #[allow(dead_code)]
    pub(crate) fn cmd_queue_ref(&self) -> &MetalCommandQueue {
        &self.cmd_queue_wrapper
    }

    /// Build (or look up) a dedicated MTLBuffer holding the BF16 weight
    /// slice `[byte_offset, byte_offset+elem_bytes)` of `w_buf`.
    ///
    /// MPSGraphTensorData has no public buffer-with-offset initialiser
    /// (only `initWithMTLBuffer:shape:dataType:` and the row-strided
    /// variant). `MTLDevice newBufferWithBytesNoCopy:` requires the
    /// pointer to be page-aligned, which Lumen's per-tensor offsets
    /// inside the LBC mmap blob almost never are. So we allocate a
    /// one-time dedicated `MTLBuffer` of exact `elem_bytes` length and
    /// copy the source bytes in.
    ///
    /// Two source-buffer paths:
    ///   * Shared-storage `w_buf` (mmap'd LBC) — `contents()` is the
    ///     CPU-visible mapping; we memcpy host-side.
    ///   * Private-storage `w_buf` (GPU-resident, see
    ///     `gpu_resident.rs`) — `contents()` is null, so we issue a
    ///     blit command via the MPSGraph queue to copy GPU-to-GPU,
    ///     then commit_and_wait so the copy is realised before the
    ///     first MPSGraph dispatch reads it.
    ///
    /// The buffer is cached by `(base_ptr_or_raw_id, offset, length)`
    /// so each weight tensor is duplicated at most once per process.
    /// Worst-case extra VRAM:
    ///   * 24 GDN layers × (qkv 67 MB + ssm_out 33 MB) ≈ 2.4 GB
    /// well below the 4.8 GB precedent of Q8 hot-weight repack.
    unsafe fn view_buffer(
        &self,
        w_buf: &MetalBuffer,
        byte_offset: u64,
        elem_bytes: u64,
    ) -> Option<ObjcId> {
        static SEL_DEV: OnceLock<usize> = OnceLock::new();
        static SEL_NEW_LEN: OnceLock<usize> = OnceLock::new();
        static SEL_CONTENTS: OnceLock<usize> = OnceLock::new();
        static SEL_CB: OnceLock<usize> = OnceLock::new();
        static SEL_BLIT: OnceLock<usize> = OnceLock::new();
        static SEL_COPY: OnceLock<usize> = OnceLock::new();
        static SEL_END_ENCODING: OnceLock<usize> = OnceLock::new();
        static SEL_COMMIT: OnceLock<usize> = OnceLock::new();
        static SEL_WAIT: OnceLock<usize> = OnceLock::new();

        // Identity key: prefer the CPU pointer when available (shared
        // mode), otherwise fall back to the source MTLBuffer raw id +
        // offset (private mode). Either form is stable for the
        // process lifetime.
        let src_contents = w_buf.contents() as usize;
        let key_base = if src_contents != 0 { src_contents } else { w_buf.raw() as usize };
        let key = (key_base, byte_offset, elem_bytes);

        let mut cache = self.view_buf_cache.lock().ok()?;
        if let Some(&buf) = cache.get(&key) {
            return Some(buf);
        }

        let device = msg0(w_buf.raw(), sel_cached(&SEL_DEV, "device"));
        if device.is_null() {
            return None;
        }

        // -[MTLDevice newBufferWithLength:options:] — shared mode = 0
        // so MPSGraphTensorData (which prefers shared/managed buffers
        // for input feeds) can bind it without an internal staging copy.
        type FnNewLen = unsafe extern "C" fn(ObjcId, ObjcSel, u64, u64) -> ObjcId;
        let f: FnNewLen = std::mem::transmute(objc_msgSend as *const c_void);
        let dst_buf = f(
            device,
            sel_cached(&SEL_NEW_LEN, "newBufferWithLength:options:"),
            elem_bytes,
            0u64, // MTLResourceStorageModeShared
        );
        if dst_buf.is_null() {
            return None;
        }

        if src_contents != 0 {
            // Shared-mode source: CPU memcpy through `contents()`.
            // Apple Silicon unified memory: zero driver copy.
            type FnContents = unsafe extern "C" fn(ObjcId, ObjcSel) -> *mut c_void;
            let g: FnContents = std::mem::transmute(objc_msgSend as *const c_void);
            let dst_ptr = g(dst_buf, sel_cached(&SEL_CONTENTS, "contents")) as *mut u8;
            let src_ptr = (src_contents as *mut u8).add(byte_offset as usize);
            if dst_ptr.is_null() {
                release(dst_buf);
                return None;
            }
            std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, elem_bytes as usize);
        } else {
            // Private-mode source: GPU blit copy. We use the MPSGraph
            // dedicated queue (`cmd_queue_raw`) so this stages cleanly
            // before any MPSGraph compute work submits there.
            type FnCB = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
            let f_cb: FnCB = std::mem::transmute(objc_msgSend as *const c_void);
            let cb = f_cb(self.cmd_queue_raw, sel_cached(&SEL_CB, "commandBuffer"));
            if cb.is_null() {
                release(dst_buf);
                return None;
            }
            retain(cb); // autoreleased -> retain for our local use
            let enc = f_cb(cb, sel_cached(&SEL_BLIT, "blitCommandEncoder"));
            if enc.is_null() {
                release(cb);
                release(dst_buf);
                return None;
            }
            retain(enc);
            // -[MTLBlitCommandEncoder copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:]
            type FnCopy = unsafe extern "C" fn(
                ObjcId, ObjcSel, ObjcId, u64, ObjcId, u64, u64,
            );
            let f_copy: FnCopy = std::mem::transmute(objc_msgSend as *const c_void);
            f_copy(
                enc,
                sel_cached(&SEL_COPY, "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:"),
                w_buf.raw(),
                byte_offset,
                dst_buf,
                0u64,
                elem_bytes,
            );
            msg0(enc, sel_cached(&SEL_END_ENCODING, "endEncoding"));
            release(enc);
            msg0(cb, sel_cached(&SEL_COMMIT, "commit"));
            msg0(cb, sel_cached(&SEL_WAIT, "waitUntilCompleted"));
            release(cb);
        }

        cache.insert(key, dst_buf);
        Some(dst_buf)
    }

    /// Build an `MPSGraphTensorData*` (retained, caller releases) over a
    /// BF16 weight tensor that starts at `byte_offset` into a larger
    /// `MTLBuffer`. Uses the cached view buffer for non-zero offsets.
    unsafe fn make_w_tensor_data(
        &self,
        w_buf: &MetalBuffer,
        byte_offset: u64,
        n: u32,
        k: u32,
    ) -> ObjcId {
        let elem_bytes: u64 = (n as u64) * (k as u64) * 2; // BF16 = 2 bytes/elem
        if byte_offset + elem_bytes > w_buf.length() {
            return ptr::null_mut();
        }

        // Resolve the underlying MTLBuffer: original at offset 0, view
        // buffer otherwise.
        let buf_to_bind: ObjcId = if byte_offset == 0 {
            w_buf.raw()
        } else {
            match self.view_buffer(w_buf, byte_offset, elem_bytes) {
                Some(b) => b,
                None => return ptr::null_mut(),
            }
        };

        static SEL_ALLOC_TD: OnceLock<usize> = OnceLock::new();
        static SEL_INIT_TD: OnceLock<usize> = OnceLock::new();
        let c = cls("MPSGraphTensorData");
        let alloced = msg0(c as ObjcId, sel_cached(&SEL_ALLOC_TD, "alloc"));
        let shape_arr = nsarray_with_dims(&[n as u64, k as u64]);
        type Fn2 = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, ObjcId, u32) -> ObjcId;
        let g: Fn2 = std::mem::transmute(objc_msgSend as *const c_void);
        let td = g(
            alloced,
            sel_cached(&SEL_INIT_TD, "initWithMTLBuffer:shape:dataType:"),
            buf_to_bind,
            shape_arr,
            MPS_DTYPE_BF16,
        );
        release(shape_arr);
        td
    }
}
