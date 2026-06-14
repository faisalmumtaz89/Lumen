//! Minimal Objective-C / Metal framework FFI bindings.
//!
//! Zero external crate dependencies. Uses raw `objc_msgSend` and `sel_registerName`
//! to call Metal APIs through the Objective-C runtime. All Metal objects are
//! reference-counted via `retain`/`release`.
//!
//! The guiding principle: wrap exactly the Metal API surface we need, nothing more.
//! Apple Silicon unified memory means `MTLResourceStorageModeShared` gives zero-copy
//! CPU/GPU access.

use std::ffi::{c_char, c_void, CStr, CString};
use std::fmt;
use std::ptr;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicPtr, Ordering as AOrdering};

// ============================================================================
// Objective-C runtime FFI
// ============================================================================

type ObjcId = *mut c_void;
type ObjcClass = *mut c_void;
type ObjcSel = *mut c_void;

#[link(name = "objc", kind = "dylib")]
extern "C" {
    fn objc_getClass(name: *const c_char) -> ObjcClass;
    fn sel_registerName(name: *const c_char) -> ObjcSel;
    fn objc_msgSend(receiver: ObjcId, sel: ObjcSel, ...) -> ObjcId;
    // Obj-C autorelease pool push/pop. Apple ABI, stable.
    // Required to drain autoreleased Metal objects (command buffers, encoders)
    // that accumulate in tokio worker threads which have no implicit Cocoa
    // autorelease pool. See `autoreleasepool` helper below.
    fn objc_autoreleasePoolPush() -> *mut c_void;
    fn objc_autoreleasePoolPop(pool: *mut c_void);
}

// ============================================================================
// Metal framework link
// ============================================================================

#[link(name = "Metal", kind = "framework")]
extern "C" {
    fn MTLCreateSystemDefaultDevice() -> ObjcId;
}

#[link(name = "Foundation", kind = "framework")]
extern "C" {}

// ============================================================================
// Helpers
// ============================================================================

/// Register an Objective-C selector from a Rust string (heap-allocating).
///
/// Retained for any future selectors not yet added to `cached_sel`.
/// All current call sites use cached selectors instead.
#[inline]
#[allow(dead_code)]
fn sel(name: &str) -> ObjcSel {
    let cstr = CString::new(name).unwrap();
    unsafe { sel_registerName(cstr.as_ptr()) }
}

/// Get an Objective-C class by name.
#[inline]
fn cls(name: &str) -> ObjcClass {
    let cstr = CString::new(name).unwrap();
    unsafe { objc_getClass(cstr.as_ptr()) }
}

// ============================================================================
// Obj-C autorelease pool wrapper
// ============================================================================

/// Run `f` inside an Obj-C autorelease pool.
///
/// Semantically equivalent to Apple's `@autoreleasepool { ... }` block.
/// Pushes a new pool before invoking `f`, pops it (draining every Obj-C
/// object that was autoreleased during the call) on normal return, on
/// `?`-style early return, and on panic.
///
/// **Why this exists**: every Metal selector that returns an autoreleased
/// object (`[queue commandBuffer]`, `[cmd computeCommandEncoder]`, etc.)
/// adds a +1 reference to the **outer** autorelease pool. Cocoa apps drain
/// that pool every run-loop iteration; non-Cocoa Rust threads (every tokio
/// worker) have no run-loop, so the outer pool effectively never drains.
/// On a server under sustained load this is a ~120 MB/h leak (per a prior
/// leak attribution).
///
/// **Safety contract**: Lumen's `MetalCommandBuffer`, `MetalComputeEncoder`,
/// and every other FFI wrapper constructor `retain`s the underlying Obj-C
/// object immediately. The Lumen-side +1 reference survives this pool's
/// drain; only the autorelease (+1) reference is released here. The Lumen
/// `Drop` impl releases the Lumen-side reference at the usual time. **Do
/// NOT** call `release()` on a Lumen wrapper from inside the closure and
/// then access the underlying pointer after the closure returns — the
/// autorelease drain would have freed it. (Existing call sites never do
/// this; they let `Drop` run after the closure ends.)
///
/// **Cost**: ~50 ns per push/pop pair on Apple Silicon. At decode TPOT of
/// ~15 ms/token this is <0.001% overhead. Negligible.
#[inline]
pub fn autoreleasepool<R>(f: impl FnOnce() -> R) -> R {
    // RAII guard: pop unconditionally on scope exit (normal return, `?`
    // early-return, panic) — exactly matches Apple's `@autoreleasepool { }`
    // semantics.
    struct PoolGuard(*mut c_void);
    impl Drop for PoolGuard {
        fn drop(&mut self) {
            unsafe { objc_autoreleasePoolPop(self.0) };
        }
    }
    let _guard = PoolGuard(unsafe { objc_autoreleasePoolPush() });
    f()
}

// ============================================================================
// Cached Selectors — eliminates ~1600 CString heap allocations per token
// ============================================================================

/// Helper macro: define a function that returns a cached ObjcSel.
///
/// Each selector is resolved once via `sel_registerName` and stored in a
/// `static OnceLock`. Subsequent calls return the cached pointer with zero
/// allocation overhead.
macro_rules! cached_sel {
    ($fn_name:ident, $sel_str:expr) => {
        #[inline(always)]
        pub(super) fn $fn_name() -> ObjcSel {
            static CACHED: OnceLock<usize> = OnceLock::new();
            // Store as usize because *mut c_void is not Sync.
            // sel_registerName returns a stable pointer for the process lifetime.
            let val = *CACHED.get_or_init(|| {
                let cstr = CString::new($sel_str).unwrap();
                unsafe { sel_registerName(cstr.as_ptr()) as usize }
            });
            val as ObjcSel
        }
    };
}

mod cached_sel {
    use super::*;

    // -- Command buffer / encoder lifecycle --
    cached_sel!(command_buffer, "commandBuffer");
    cached_sel!(
        command_buffer_with_unretained_references,
        "commandBufferWithUnretainedReferences"
    );
    cached_sel!(compute_command_encoder, "computeCommandEncoder");
    cached_sel!(blit_command_encoder, "blitCommandEncoder");
    cached_sel!(commit, "commit");
    cached_sel!(wait_until_completed, "waitUntilCompleted");
    cached_sel!(end_encoding, "endEncoding");

    // -- Compute encoder hot-path --
    cached_sel!(set_compute_pipeline_state, "setComputePipelineState:");
    cached_sel!(set_buffer_offset_at_index, "setBuffer:offset:atIndex:");
    cached_sel!(set_bytes_length_at_index, "setBytes:length:atIndex:");
    cached_sel!(dispatch_threadgroups, "dispatchThreadgroups:threadsPerThreadgroup:");
    cached_sel!(dispatch_threads, "dispatchThreads:threadsPerThreadgroup:");
    cached_sel!(set_threadgroup_memory_length, "setThreadgroupMemoryLength:atIndex:");
    cached_sel!(memory_barrier_with_scope, "memoryBarrierWithScope:");
    cached_sel!(memory_barrier_with_resources, "memoryBarrierWithResources:count:");

    // -- GPU timing (zero-overhead, read on a completed CB) --
    cached_sel!(gpu_start_time, "GPUStartTime");
    cached_sel!(gpu_end_time, "GPUEndTime");

    // -- Blit encoder --
    cached_sel!(copy_from_buffer, "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:");

    // -- Object lifecycle --
    cached_sel!(retain, "retain");
    cached_sel!(release, "release");

    // -- Buffer queries --
    cached_sel!(contents, "contents");
    cached_sel!(length, "length");

    // -- Pipeline state queries --
    cached_sel!(max_total_threads_per_threadgroup, "maxTotalThreadsPerThreadgroup");
    cached_sel!(thread_execution_width, "threadExecutionWidth");

    // -- Buffer creation --
    cached_sel!(new_buffer_with_length, "newBufferWithLength:options:");
    cached_sel!(new_buffer_with_bytes, "newBufferWithBytes:length:options:");
    cached_sel!(new_buffer_with_bytes_no_copy, "newBufferWithBytesNoCopy:length:options:deallocator:");

    // -- Library / function / pipeline creation --
    cached_sel!(new_library_with_source, "newLibraryWithSource:options:error:");
    cached_sel!(new_compute_pipeline_state, "newComputePipelineStateWithFunction:error:");
    cached_sel!(new_function_with_name, "newFunctionWithName:");
    cached_sel!(new_function_with_name_constant_values, "newFunctionWithName:constantValues:error:");
    // -- MTLCompileOptions language version (Metal 3.0+ for bfloat support) --
    cached_sel!(compile_options_set_language_version, "setLanguageVersion:");

    // -- Function constant values --
    cached_sel!(alloc, "alloc");
    cached_sel!(init, "init");
    cached_sel!(set_constant_value, "setConstantValue:type:atIndex:");

    // -- Device / queue --
    cached_sel!(new_command_queue, "newCommandQueue");
    cached_sel!(name, "name");

    // -- MTLSharedEvent (cross-queue GPU↔GPU synchronization) --
    //
    // Used by the dual-queue GDN branch-overlap path
    // (`LUMEN_METAL_GDN_DUAL_QUEUE=1`). `newSharedEvent` allocates a
    // device-shared event whose 64-bit `signaledValue` is monotonically
    // advanced by `encodeSignalEvent:value:` on a producer command buffer
    // and waited on by `encodeWaitForEvent:value:` on a consumer command
    // buffer (possibly on a DIFFERENT MTLCommandQueue). This is the only
    // Apple-supported mechanism to express a cross-queue dependency; CBs on
    // the SAME queue are FIFO and CBs on DIFFERENT queues are otherwise
    // unordered.
    cached_sel!(new_shared_event, "newSharedEvent");
    cached_sel!(encode_signal_event_value, "encodeSignalEvent:value:");
    cached_sel!(encode_wait_for_event_value, "encodeWaitForEvent:value:");
    cached_sel!(signaled_value, "signaledValue");

    // -- NSString --
    cached_sel!(string_with_utf8_string, "stringWithUTF8String:");
    cached_sel!(utf8_string, "UTF8String");
    cached_sel!(localized_description, "localizedDescription");

    // -- NSURL --
    cached_sel!(file_url_with_path, "fileURLWithPath:");

    // -- MTLIOCommandQueue (Metal 3, macOS 13+) --
    cached_sel!(new_io_command_queue, "newIOCommandQueueWithDescriptor:error:");
    cached_sel!(new_io_file_handle, "newIOFileHandleWithURL:error:");
    cached_sel!(io_command_buffer, "commandBuffer");  // same selector as MTLCommandQueue
    cached_sel!(load_buffer, "loadBuffer:offset:size:sourceHandle:sourceHandleOffset:");
    cached_sel!(set_type, "setType:");
    cached_sel!(status, "status");

    // -- in-process MTLDevice allocation accounting --
    //
    // `currentAllocatedSize` returns the total bytes the driver has
    // outstanding for THIS MTLDevice instance (all MTLBuffer / MTLTexture /
    // MTLHeap allocations the process group currently holds). On Apple
    // Silicon unified memory this is the authoritative GPU residency
    // counter and is the in-process equivalent of the
    // `mtl-mem-probe` external Swift probe — only this one sees the
    // lumen-server process's own allocations, not a separate Swift
    // process's allocations. Used by the server-side
    // `/debug/memory_breakdown` endpoint.
    cached_sel!(current_allocated_size, "currentAllocatedSize");
}

/// Send an Objective-C message with no arguments, returning an object pointer.
///
/// Uses a typed function pointer cast to avoid arm64 variadic ABI issues.
/// Even zero-argument message sends must go through a typed pointer because
/// the variadic calling convention may differ from the non-variadic one on
/// AArch64 (different register allocation for parameters).
///
/// Takes a pre-resolved ObjcSel to avoid per-call heap allocation.
#[inline]
unsafe fn msg_send_0_raw(obj: ObjcId, s: ObjcSel) -> ObjcId {
    type MsgSend0Fn = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
    let f: MsgSend0Fn = std::mem::transmute(objc_msgSend as *const c_void);
    f(obj, s)
}

/// Send a 0-arg message that returns a `double` (CFTimeInterval). Used for
/// `GPUStartTime` / `GPUEndTime` on a completed MTLCommandBuffer. On AArch64
/// the float return uses the FP registers, so a distinct fn signature is
/// required (cannot transmute the ObjcId-returning variant).
#[inline]
unsafe fn msg_send_0_f64(obj: ObjcId, s: ObjcSel) -> f64 {
    type MsgSend0F64Fn = unsafe extern "C" fn(ObjcId, ObjcSel) -> f64;
    let f: MsgSend0F64Fn = std::mem::transmute(objc_msgSend as *const c_void);
    f(obj, s)
}

/// Retain an Objective-C object.
#[inline]
unsafe fn retain(obj: ObjcId) -> ObjcId {
    msg_send_0_raw(obj, cached_sel::retain())
}

/// Release an Objective-C object.
#[inline]
unsafe fn release(obj: ObjcId) {
    msg_send_0_raw(obj, cached_sel::release());
}

/// Create an NSString from a Rust &str.
///
/// Uses typed function pointer to pass the `const char*` argument correctly
/// on arm64 (variadic ABI misroutes pointer arguments).
unsafe fn nsstring(s: &str) -> ObjcId {
    let cstr = CString::new(s).unwrap();
    let ns_class = cls("NSString");
    type StringWithUTF8Fn = unsafe extern "C" fn(ObjcId, ObjcSel, *const c_char) -> ObjcId;
    let f: StringWithUTF8Fn = std::mem::transmute(objc_msgSend as *const c_void);
    f(ns_class, cached_sel::string_with_utf8_string(), cstr.as_ptr())
}

/// Read an NSString into a Rust String.
///
/// Uses typed function pointer for the UTF8String message send.
unsafe fn nsstring_to_string(ns: ObjcId) -> String {
    if ns.is_null() {
        return String::new();
    }
    type UTF8StringFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> *const c_char;
    let f: UTF8StringFn = std::mem::transmute(objc_msgSend as *const c_void);
    let cstr: *const c_char = f(ns, cached_sel::utf8_string());
    if cstr.is_null() {
        return String::new();
    }
    CStr::from_ptr(cstr).to_string_lossy().into_owned()
}

/// Get the `localizedDescription` of an NSError (or any object).
unsafe fn error_description(err: ObjcId) -> String {
    if err.is_null() {
        return String::new();
    }
    let desc = msg_send_0_raw(err, cached_sel::localized_description());
    nsstring_to_string(desc)
}

// ============================================================================
// MTLResourceOptions
// ============================================================================

/// `MTLResourceStorageModeShared` = 0 on Apple Silicon unified memory.
/// CPU and GPU share the same physical memory. Zero-copy.
const MTL_RESOURCE_STORAGE_MODE_SHARED: u64 = 0;

/// `MTLResourceStorageModePrivate` = (2 << 4) = 32.
/// GPU-only memory. Not CPU-accessible. Allows the GPU memory controller
/// to apply hardware-level optimizations (lossless compression, optimal
/// caching). Data must be copied in via a blit command encoder.
const MTL_RESOURCE_STORAGE_MODE_PRIVATE: u64 = 2 << 4;

// ============================================================================
// MetalDevice — wraps MTLDevice
// ============================================================================

/// A Metal GPU device handle. Reference-counted.
pub struct MetalDevice {
    raw: ObjcId,
}

impl MetalDevice {
    /// Create the system default Metal device.
    /// Returns `None` if no Metal device is available.
    pub fn system_default() -> Option<Self> {
        let raw = unsafe { MTLCreateSystemDefaultDevice() };
        if raw.is_null() {
            None
        } else {
            // MTLCreateSystemDefaultDevice returns a retained object.
            Some(Self { raw })
        }
    }

    /// Get the device name as a String.
    pub fn name(&self) -> String {
        unsafe { nsstring_to_string(msg_send_0_raw(self.raw, cached_sel::name())) }
    }

    /// Total bytes the driver has currently allocated for this
    /// MTLDevice (in-process).
    ///
    /// Returns the sum of bytes outstanding for all MTLBuffer / MTLTexture /
    /// MTLHeap objects the process holds against this MTLDevice. On Apple
    /// Silicon unified memory this is the authoritative GPU residency
    /// figure and the only way to observe lumen-server's own GPU footprint
    /// (an external probe sees its own process's allocations, not the
    /// server's).
    ///
    /// Cost: one Objective-C selector dispatch (cached selector, no heap
    /// allocation). Cheap enough to call from a 30 s breakdown sampler.
    pub fn current_allocated_size(&self) -> u64 {
        unsafe {
            type GetSizeFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> u64;
            let f: GetSizeFn = std::mem::transmute(objc_msgSend as *const c_void);
            f(self.raw, cached_sel::current_allocated_size())
        }
    }

    /// Create a new command queue.
    pub fn new_command_queue(&self) -> Option<MetalCommandQueue> {
        let raw = unsafe { msg_send_0_raw(self.raw, cached_sel::new_command_queue()) };
        if raw.is_null() {
            None
        } else {
            // newCommandQueue returns a retained object.
            Some(MetalCommandQueue { raw })
        }
    }

    /// Create a new `MTLSharedEvent` for cross-queue GPU synchronization.
    ///
    /// Returns `None` if the driver could not allocate the event. The event
    /// is retained by the returned wrapper and released on `Drop`.
    pub fn new_shared_event(&self) -> Option<MetalSharedEvent> {
        let raw = unsafe { msg_send_0_raw(self.raw, cached_sel::new_shared_event()) };
        if raw.is_null() {
            None
        } else {
            // newSharedEvent returns a retained (+1) object per the
            // `new`-prefix Cocoa naming convention; do NOT retain again.
            Some(MetalSharedEvent { raw })
        }
    }

    /// Compile Metal shading language source into a library.
    ///
    /// Sets `MTLCompileOptions.languageVersion = MTLLanguageVersion3_1` so the
    /// shaders may use the `bfloat`, `bfloat4`, and `simdgroup_bfloat8x8`
    /// types. Without an explicit version the default chosen by Metal can be
    /// older (MSL 2.x) and reject these references, even on devices that
    /// physically support BF16 (M2+, Metal 3+).
    ///
    /// `MTLLanguageVersion3_1` is encoded as `(3 << 16) | 1` (see
    /// `MTLLanguageVersion` in `Metal/MTLLanguage.h`). Apple guarantees MSL
    /// 3.1 on macOS 14.x+, which is below the macOS 13 floor Lumen already
    /// enforces for `MTLIOCommandQueue` plus a minor delta — and the active
    /// targets (M2/M3) require BF16-vector support.
    pub fn new_library_with_source(&self, source: &str) -> Result<MetalLibrary, String> {
        unsafe {
            let ns_source = nsstring(source);
            let mut error: ObjcId = ptr::null_mut();

            // Build MTLCompileOptions with languageVersion = 3.1
            let opts_class = cls("MTLCompileOptions");
            let opts: ObjcId = msg_send_0_raw(opts_class as ObjcId, cached_sel::alloc());
            let opts: ObjcId = msg_send_0_raw(opts, cached_sel::init());

            // setLanguageVersion: takes a NSUInteger. MTLLanguageVersion3_1 = (3 << 16) | 1.
            // 3.1 is required for `bfloat4` / `simdgroup_bfloat8x8` (Apple
            // exposed BF16 vector and MMA types in this release; 3.0 only had
            // the scalar `bfloat` type).
            type SetLangVerFn = unsafe extern "C" fn(ObjcId, ObjcSel, usize);
            let set_lang_ver: SetLangVerFn = std::mem::transmute(objc_msgSend as *const c_void);
            const MTL_LANG_VERSION_3_1: usize = (3usize << 16) | 1;
            set_lang_ver(opts, cached_sel::compile_options_set_language_version(), MTL_LANG_VERSION_3_1);

            type NewLibFn = unsafe extern "C" fn(
                ObjcId, ObjcSel, ObjcId, ObjcId, *mut ObjcId,
            ) -> ObjcId;
            let f: NewLibFn = std::mem::transmute(objc_msgSend as *const c_void);
            let lib = f(
                self.raw,
                cached_sel::new_library_with_source(),
                ns_source,
                opts,
                &mut error as *mut ObjcId,
            );

            release(opts);

            if lib.is_null() || !error.is_null() {
                let desc = error_description(error);
                if !error.is_null() {
                    release(error);
                }
                return Err(format!("Metal shader compilation failed: {desc}"));
            }

            Ok(MetalLibrary { raw: lib })
        }
    }

    /// Create a compute pipeline state from a function.
    pub fn new_compute_pipeline_state(
        &self,
        function: &MetalFunction,
    ) -> Result<MetalPipelineState, String> {
        unsafe {
            let mut error: ObjcId = ptr::null_mut();
            type NewPSOFn = unsafe extern "C" fn(
                ObjcId, ObjcSel, ObjcId, *mut ObjcId,
            ) -> ObjcId;
            let f: NewPSOFn = std::mem::transmute(objc_msgSend as *const c_void);
            let pso = f(
                self.raw,
                cached_sel::new_compute_pipeline_state(),
                function.raw,
                &mut error as *mut ObjcId,
            );

            if pso.is_null() || !error.is_null() {
                let desc = error_description(error);
                if !error.is_null() {
                    release(error);
                }
                return Err(format!("Pipeline state creation failed: {desc}"));
            }

            Ok(MetalPipelineState { raw: pso })
        }
    }

    /// Create a new buffer of the given length with shared storage mode (zero-copy).
    ///
    /// Uses typed function pointer to avoid variadic ABI issues with u64 args on arm64.
    pub fn new_buffer(&self, length: usize) -> Option<MetalBuffer> {
        unsafe {
            type NewBufferFn = unsafe extern "C" fn(ObjcId, ObjcSel, u64, u64) -> ObjcId;
            let f: NewBufferFn = std::mem::transmute(objc_msgSend as *const c_void);
            let raw = f(
                self.raw,
                cached_sel::new_buffer_with_length(),
                length as u64,
                MTL_RESOURCE_STORAGE_MODE_SHARED,
            );
            if raw.is_null() {
                None
            } else {
                Some(MetalBuffer { raw })
            }
        }
    }

    /// Create a buffer wrapping existing bytes (copy into GPU-accessible memory).
    pub fn new_buffer_with_bytes(&self, bytes: &[u8]) -> Option<MetalBuffer> {
        if bytes.is_empty() {
            return self.new_buffer(4); // Metal requires non-zero length
        }
        unsafe {
            type NewBufferBytesFn = unsafe extern "C" fn(
                ObjcId, ObjcSel, *const c_void, u64, u64,
            ) -> ObjcId;
            let f: NewBufferBytesFn = std::mem::transmute(objc_msgSend as *const c_void);
            let raw = f(
                self.raw,
                cached_sel::new_buffer_with_bytes(),
                bytes.as_ptr() as *const c_void,
                bytes.len() as u64,
                MTL_RESOURCE_STORAGE_MODE_SHARED,
            );
            if raw.is_null() {
                None
            } else {
                Some(MetalBuffer { raw })
            }
        }
    }

    /// Create a buffer wrapping an existing pointer with NO copy.
    ///
    /// # Safety
    ///
    /// - `ptr` must remain valid for the lifetime of the returned `MetalBuffer`.
    /// - `length` must accurately reflect the size of the memory region.
    /// - The memory at `ptr` must be page-aligned (mmap memory is always page-aligned).
    /// - The deallocator is set to nil (we don't own the memory -- mmap does).
    pub unsafe fn new_buffer_no_copy(
        &self,
        ptr: *mut c_void,
        length: usize,
    ) -> Option<MetalBuffer> {
        if ptr.is_null() || length == 0 {
            return None;
        }

        // Block type for deallocator: void (^)(void* pointer, NSUInteger length)
        // We pass nil because the mmap region owns this memory.
        let deallocator: ObjcId = ptr::null_mut();

        type NewBufferNoCopyFn = unsafe extern "C" fn(
            ObjcId, ObjcSel, *mut c_void, u64, u64, ObjcId,
        ) -> ObjcId;
        let f: NewBufferNoCopyFn = std::mem::transmute(objc_msgSend as *const c_void);
        let raw = f(
            self.raw,
            cached_sel::new_buffer_with_bytes_no_copy(),
            ptr,
            length as u64,
            MTL_RESOURCE_STORAGE_MODE_SHARED,
            deallocator,
        );

        if raw.is_null() {
            None
        } else {
            Some(MetalBuffer { raw })
        }
    }

    /// Create a new buffer with private storage mode (GPU-only, not CPU-accessible).
    ///
    /// Private storage allows the GPU memory controller to apply hardware-level
    /// optimizations (lossless compression, optimal caching). Data must be loaded
    /// via a blit command encoder since CPU cannot access private buffers.
    pub fn new_buffer_private(&self, length: usize) -> Option<MetalBuffer> {
        unsafe {
            type NewBufferFn = unsafe extern "C" fn(ObjcId, ObjcSel, u64, u64) -> ObjcId;
            let f: NewBufferFn = std::mem::transmute(objc_msgSend as *const c_void);
            let raw = f(
                self.raw,
                cached_sel::new_buffer_with_length(),
                length as u64,
                MTL_RESOURCE_STORAGE_MODE_PRIVATE,
            );
            if raw.is_null() {
                None
            } else {
                Some(MetalBuffer { raw })
            }
        }
    }

    /// Raw Objective-C id for this device.
    pub fn raw(&self) -> ObjcId {
        self.raw
    }
}

impl Drop for MetalDevice {
    fn drop(&mut self) {
        unsafe { release(self.raw) }
    }
}

// SAFETY: MTLDevice is thread-safe (it's the GPU handle, used from any thread).
unsafe impl Send for MetalDevice {}
unsafe impl Sync for MetalDevice {}

impl fmt::Debug for MetalDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetalDevice")
            .field("name", &self.name())
            .finish()
    }
}

// ============================================================================
// MetalCommandQueue — wraps MTLCommandQueue
// ============================================================================

pub struct MetalCommandQueue {
    raw: ObjcId,
}

impl MetalCommandQueue {
    /// Create a new command buffer from this queue.
    ///
    /// The returned buffer does NOT carry a queue handle, so it cannot
    /// self-replace under profiling. For prefill (which auto-splits CBs
    /// at section boundaries when `LUMEN_METAL_PROFILE=1`), use
    /// `new_command_buffer_profilable()` instead.
    pub fn new_command_buffer(&self) -> Option<MetalCommandBuffer> {
        let raw = unsafe { msg_send_0_raw(self.raw, cached_sel::command_buffer()) };
        if raw.is_null() {
            None
        } else {
            // commandBuffer returns an autoreleased object; retain it.
            unsafe { retain(raw) };
            Some(MetalCommandBuffer {
                raw: AtomicPtr::new(raw),
                queue_raw: AtomicPtr::new(ptr::null_mut()),
            })
        }
    }

    /// Create a new command buffer that carries a back-reference to this
    /// queue, allowing it to self-replace when Metal profiling is enabled.
    ///
    /// Behaviour identical to `new_command_buffer` when
    /// `LUMEN_METAL_PROFILE` is OFF. When ON, calls to
    /// `new_compute_encoder()` / `new_concurrent_compute_encoder()` on the
    /// returned buffer will internally commit + wait on the prior CB and
    /// allocate a fresh CB from this queue, with CPU-side timing recorded
    /// into the thread-local profile accumulator.
    ///
    /// The lifetime relationship requires that this queue outlive the
    /// returned buffer. In Lumen the queue is owned by `MetalF32Backend`
    /// and the buffer lives entirely within a single `prefill()` call, so
    /// this is always satisfied.
    pub fn new_command_buffer_profilable(&self) -> Option<MetalCommandBuffer> {
        let raw = unsafe { msg_send_0_raw(self.raw, cached_sel::command_buffer()) };
        if raw.is_null() {
            None
        } else {
            unsafe { retain(raw) };
            // Retain queue for the lifetime of the buffer (released in Drop).
            unsafe { retain(self.raw) };
            Some(MetalCommandBuffer {
                raw: AtomicPtr::new(raw),
                queue_raw: AtomicPtr::new(self.raw),
            })
        }
    }

    /// Create a new command buffer using
    /// `commandBufferWithUnretainedReferences`, the variant that does NOT
    /// retain every Metal resource bound to it.
    ///
    /// Apple documents this on `MTLCommandQueue`: the default
    /// `[queue commandBuffer]` retains every `MTLBuffer`, `MTLTexture`,
    /// `MTLComputePipelineState`, etc. set on the buffer for the lifetime
    /// of the buffer; the unretained variant skips both halves of the
    /// `retain` + `release` pair. The unretained variant is the canonical
    /// Apple-documented choice when the caller can guarantee resource
    /// lifetime by other means (as Lumen does — see the safety contract
    /// below).
    ///
    /// ## Safety contract (caller responsibility)
    ///
    /// The caller MUST guarantee that every resource bound to any encoder
    /// created from this command buffer remains alive at least until the
    /// command buffer's GPU execution completes. Concretely, this means
    /// every `MTLBuffer` / `MTLTexture` / `MTLComputePipelineState` whose
    /// Rust wrapper is dropped before
    /// `cmd.commit_and_wait()` / `cmd.wait_until_completed()` returns is
    /// a use-after-free hazard at the Metal driver level.
    ///
    /// Lumen's prefill and decode hot paths satisfy this trivially: the
    /// `MetalF32Backend` owns `MetalBuffer`s and `MetalComputePipelineState`s
    /// for the entire model session, and the prefill/decode functions
    /// commit-and-wait on each command buffer before returning, so resource
    /// `Drop`s cannot run before GPU completion.
    pub fn new_command_buffer_unretained(&self) -> Option<MetalCommandBuffer> {
        let raw = unsafe {
            msg_send_0_raw(
                self.raw,
                cached_sel::command_buffer_with_unretained_references(),
            )
        };
        if raw.is_null() {
            None
        } else {
            // commandBufferWithUnretainedReferences also returns an
            // autoreleased object; retain it for our reference (only the
            // bound RESOURCES are unretained, not the CB itself).
            unsafe { retain(raw) };
            Some(MetalCommandBuffer {
                raw: AtomicPtr::new(raw),
                queue_raw: AtomicPtr::new(ptr::null_mut()),
            })
        }
    }

    /// Profilable unretained variant.
    ///
    /// Combines the `new_command_buffer_profilable` queue-back-reference
    /// machinery with the `commandBufferWithUnretainedReferences` resource
    /// retention semantics. The profile-mode auto-split path
    /// (`profile_split_if_needed`) issues a fresh CB through the same
    /// unretained selector so the savings persist across each split.
    pub fn new_command_buffer_profilable_unretained(&self) -> Option<MetalCommandBuffer> {
        let raw = unsafe {
            msg_send_0_raw(
                self.raw,
                cached_sel::command_buffer_with_unretained_references(),
            )
        };
        if raw.is_null() {
            None
        } else {
            unsafe { retain(raw) };
            unsafe { retain(self.raw) };
            Some(MetalCommandBuffer {
                raw: AtomicPtr::new(raw),
                queue_raw: AtomicPtr::new(self.raw),
            })
        }
    }

    /// Raw Objective-C id for this queue. Exposed for the MPSGraph
    /// path (`mps_graph_ffi.rs`) which needs the underlying
    /// `id<MTLCommandQueue>` to pass into MPSGraph's
    /// `runWithMTLCommandQueue:` API.
    #[inline]
    pub(crate) fn raw(&self) -> ObjcId {
        self.raw
    }
}

impl Drop for MetalCommandQueue {
    fn drop(&mut self) {
        unsafe { release(self.raw) }
    }
}

unsafe impl Send for MetalCommandQueue {}
unsafe impl Sync for MetalCommandQueue {}

impl fmt::Debug for MetalCommandQueue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetalCommandQueue").finish()
    }
}

// ============================================================================
// MetalSharedEvent — wraps MTLSharedEvent (cross-queue GPU↔GPU sync)
// ============================================================================

/// A device-shared event whose 64-bit `signaledValue` is advanced by
/// `MetalCommandBuffer::encode_signal_event` on a producer command buffer and
/// awaited by `MetalCommandBuffer::encode_wait_for_event` on a consumer command
/// buffer. The producer and consumer may be on DIFFERENT `MetalCommandQueue`s;
/// this is the only Apple-supported mechanism to express a cross-queue
/// ordering dependency. Used by the dual-queue GDN branch-overlap path.
pub struct MetalSharedEvent {
    raw: ObjcId,
}

impl MetalSharedEvent {
    /// Current signaled value (host read). Valid any time; mainly for
    /// debugging — the GPU advances it asynchronously.
    #[allow(dead_code)]
    pub fn signaled_value(&self) -> u64 {
        unsafe {
            type SignaledValueFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> u64;
            let f: SignaledValueFn = std::mem::transmute(objc_msgSend as *const c_void);
            f(self.raw, cached_sel::signaled_value())
        }
    }

    #[inline]
    fn raw(&self) -> ObjcId {
        self.raw
    }
}

impl Drop for MetalSharedEvent {
    fn drop(&mut self) {
        unsafe { release(self.raw) }
    }
}

unsafe impl Send for MetalSharedEvent {}
unsafe impl Sync for MetalSharedEvent {}

impl fmt::Debug for MetalSharedEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetalSharedEvent").finish()
    }
}

// ============================================================================
// MetalCommandBuffer — wraps MTLCommandBuffer
// ============================================================================

pub struct MetalCommandBuffer {
    /// Current MTLCommandBuffer raw handle. Wrapped in AtomicPtr so it
    /// can be atomically swapped during profile-mode auto-splitting
    /// (`profile_split_if_needed`). Loaded with `Relaxed` since access is
    /// single-threaded per buffer (use sites pass `&self`).
    raw: AtomicPtr<c_void>,
    /// Optional back-reference to the parent MTLCommandQueue. Non-null
    /// only when the buffer was created via
    /// `MetalCommandQueue::new_command_buffer_profilable()`. Required for
    /// profile-mode auto-splitting (we need a queue to allocate a fresh
    /// CB on). Retained on the parent queue for the lifetime of this CB
    /// (released in Drop).
    queue_raw: AtomicPtr<c_void>,
}

impl MetalCommandBuffer {
    /// Returns the current raw MTLCommandBuffer handle.
    #[inline]
    fn raw(&self) -> ObjcId {
        self.raw.load(AOrdering::Relaxed)
    }

    /// Profile-mode hook: if profiling is enabled AND this CB is
    /// profilable (has a queue back-ref) AND a CB is currently in flight,
    /// commit + wait the in-flight CB, record elapsed time under the
    /// current section label, then swap in a fresh CB from the queue.
    ///
    /// No-op in non-profile mode (single `Relaxed` atomic load). The
    /// hot-path cost when profiling is disabled is one branch on a
    /// global atomic, identical to the existing FFI dispatch overhead.
    ///
    /// Must be called BEFORE any `set_pipeline_state` / dispatch on the
    /// new encoder, since it issues `commit + wait_until_completed` on
    /// the prior buffer.
    #[inline]
    fn profile_split_if_needed(&self) {
        if !super::profile::is_enabled() {
            return;
        }
        let q = self.queue_raw.load(AOrdering::Relaxed);
        if q.is_null() {
            // Not profilable (created via plain new_command_buffer). The
            // call site picked the wrong constructor — skip silently to
            // avoid breaking non-prefill paths.
            return;
        }
        let cur = self.raw.load(AOrdering::Relaxed);
        if cur.is_null() {
            return;
        }
        // Close the in-flight CB. The encoder has already had end_encoding()
        // called by the prior call site.
        //
        // DEFER MODE (LUMEN_METAL_PROFILE_GDN_DEFER=1): commit
        // WITHOUT waiting. Metal executes CBs on one queue in FIFO commit
        // order, so the next CB's reads see this CB's writes (ordering/RAW
        // correctness preserved) — but no per-CB GPU-idle is injected. Retain
        // the CB and register its ptr for a single deferred GPU-time read at
        // prefill end. This removes the bogus drain/refill bubble that the
        // legacy per-CB wait charged to dependency-tail kernels (the ssm_out
        // 40ms artifact). `drain_deferred_cbs` (called at prefill end) waits
        // once and reads every CB's GPUEndTime-GPUStartTime.
        //
        // LEGACY MODE (LUMEN_METAL_PROFILE_GDN=1 without DEFER): commit + wait
        // each CB, read its GPU wall immediately. Kept for A/B of the method.
        if super::profile::is_defer_enabled() {
            // Commit (no wait). Transfer the wrapper's existing +1 reference to
            // the deferred registry (do NOT release here); `drain_deferred_cbs`
            // reads its GPU time and releases it after the single final wait.
            unsafe {
                msg_send_0_raw(cur, cached_sel::commit());
            }
            super::profile::register_deferred_cb(cur as usize);
            super::profile::record_section_end();
        } else {
            unsafe {
                msg_send_0_raw(cur, cached_sel::commit());
                msg_send_0_raw(cur, cached_sel::wait_until_completed());
                let start = msg_send_0_f64(cur, cached_sel::gpu_start_time());
                let end = msg_send_0_f64(cur, cached_sel::gpu_end_time());
                super::profile::record_section_gpu_time((end - start).max(0.0));
                release(cur);
            }
            super::profile::record_section_end();
        }

        // Allocate a fresh CB from the parent queue.
        let fresh = unsafe { msg_send_0_raw(q, cached_sel::command_buffer()) };
        if !fresh.is_null() {
            unsafe { retain(fresh) };
            self.raw.store(fresh, AOrdering::Relaxed);
        } else {
            // Allocation failed; mark raw as null so subsequent dispatch
            // sites fall through to the existing null-check error paths.
            self.raw.store(ptr::null_mut(), AOrdering::Relaxed);
        }
        super::profile::mark_section_start();
    }

    /// Create a compute command encoder.
    pub fn new_compute_encoder(&self) -> Option<MetalComputeEncoder> {
        self.profile_split_if_needed();
        let raw = unsafe { msg_send_0_raw(self.raw(), cached_sel::compute_command_encoder()) };
        if raw.is_null() {
            None
        } else {
            // computeCommandEncoder returns an autoreleased object; retain it.
            unsafe { retain(raw) };
            Some(MetalComputeEncoder { raw })
        }
    }

    /// Create a concurrent compute command encoder.
    ///
    /// Uses `computeCommandEncoderWithDispatchType:` with
    /// `MTLDispatchTypeConcurrent = 1`. Dispatches within a concurrent encoder
    /// MAY execute simultaneously on the GPU. Use `memory_barrier_with_scope`
    /// between concurrent writes and subsequent reads of the same buffer.
    pub fn new_concurrent_compute_encoder(&self) -> Option<MetalComputeEncoder> {
        self.profile_split_if_needed();
        unsafe {
            type EncoderWithDispatchTypeFn =
                unsafe extern "C" fn(ObjcId, ObjcSel, u64) -> ObjcId;
            let f: EncoderWithDispatchTypeFn =
                std::mem::transmute(objc_msgSend as *const c_void);
            let raw = f(
                self.raw(),
                sel("computeCommandEncoderWithDispatchType:"),
                1u64, // MTLDispatchTypeConcurrent
            );
            if raw.is_null() {
                None
            } else {
                // computeCommandEncoderWithDispatchType: returns an autoreleased object; retain it.
                retain(raw);
                Some(MetalComputeEncoder { raw })
            }
        }
    }

    /// Create a blit command encoder for buffer/texture copy operations.
    pub fn new_blit_encoder(&self) -> Option<MetalBlitEncoder> {
        let raw = unsafe { msg_send_0_raw(self.raw(), cached_sel::blit_command_encoder()) };
        if raw.is_null() {
            None
        } else {
            // blitCommandEncoder returns an autoreleased object; retain it.
            unsafe { retain(raw) };
            Some(MetalBlitEncoder { raw })
        }
    }

    /// Commit and wait for completion (synchronous).
    pub fn commit_and_wait(&self) {
        unsafe {
            msg_send_0_raw(self.raw(), cached_sel::commit());
            msg_send_0_raw(self.raw(), cached_sel::wait_until_completed());
        }
    }

    /// Commit (asynchronous).
    pub fn commit(&self) {
        unsafe {
            msg_send_0_raw(self.raw(), cached_sel::commit());
        }
    }

    /// Encode a GPU-side signal of `event` to `value` at this point in the
    /// command buffer's execution. After the GPU reaches this point (all
    /// previously-encoded work on THIS command buffer has retired), the
    /// event's `signaledValue` is set to `value`, unblocking any command
    /// buffer (on any queue) waiting for `>= value`.
    ///
    /// Must be called when NO compute/blit encoder is currently open on this
    /// command buffer — signal/wait operate at command-buffer granularity,
    /// between encoders. The caller is responsible for `end_encoding()` on
    /// any open encoder first.
    pub fn encode_signal_event(&self, event: &MetalSharedEvent, value: u64) {
        unsafe {
            type SignalFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, u64);
            let f: SignalFn = std::mem::transmute(objc_msgSend as *const c_void);
            f(self.raw(), cached_sel::encode_signal_event_value(), event.raw(), value);
        }
    }

    /// Encode a GPU-side wait: the command buffer will not begin executing
    /// work encoded AFTER this point until `event.signaledValue >= value`.
    /// Work encoded BEFORE this point on the same CB is unaffected.
    ///
    /// Must be called when no encoder is open (see `encode_signal_event`).
    pub fn encode_wait_for_event(&self, event: &MetalSharedEvent, value: u64) {
        unsafe {
            type WaitFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, u64);
            let f: WaitFn = std::mem::transmute(objc_msgSend as *const c_void);
            f(self.raw(), cached_sel::encode_wait_for_event_value(), event.raw(), value);
        }
    }

    /// True GPU execution wall time (seconds) for this command buffer.
    /// Valid only AFTER the CB has completed (`commit_and_wait`/
    /// `wait_until_completed`). Reads `GPUEndTime - GPUStartTime`; these are
    /// host-clock CFTimeIntervals captured by the GPU driver with zero added
    /// dispatch overhead (unlike split-CB profiling). Returns 0.0 if the
    /// driver did not populate the timestamps.
    pub fn gpu_elapsed_secs(&self) -> f64 {
        unsafe {
            let start = msg_send_0_f64(self.raw(), cached_sel::gpu_start_time());
            let end = msg_send_0_f64(self.raw(), cached_sel::gpu_end_time());
            (end - start).max(0.0)
        }
    }

    /// Wait until completed.
    pub fn wait_until_completed(&self) {
        unsafe {
            msg_send_0_raw(self.raw(), cached_sel::wait_until_completed());
        }
    }

    /// Returns the raw `MTLCommandBuffer` Objective-C handle.
    ///
    /// Exposed only for the optional MPSGraph BF16 GEMM path
    /// (`mps_graph_ffi.rs`). The returned pointer is **not** retained
    /// by the caller; it remains owned by this `MetalCommandBuffer`.
    #[inline]
    pub(crate) fn raw_command_buffer(&self) -> *mut c_void {
        self.raw.load(AOrdering::Relaxed)
    }

    /// Atomically replace the underlying `MTLCommandBuffer` pointer with
    /// a new one, releasing the previous handle and retaining the new
    /// one. Exposed only for the MPSGraph in-CB path — Apple's
    /// `MPSCommandBuffer.encodeToCommandBuffer:` may internally call
    /// `commitAndContinue`, committing the original CB and replacing it
    /// with a fresh one read via `rootCommandBuffer`. The caller must
    /// re-bind this wrapper to the new CB so subsequent compute
    /// encoders land on an open (uncommitted) CB.
    ///
    /// `new_raw` is treated as a `+0` (autoreleased) source and is
    /// retained inside; the previously held CB is released — its GPU
    /// work is already committed by MPSGraph by the time we get here,
    /// so dropping our Rust handle does not abort in-flight work.
    #[inline]
    pub(crate) fn replace_raw_command_buffer(&self, new_raw: *mut c_void) {
        unsafe {
            if new_raw.is_null() {
                return;
            }
            retain(new_raw);
            let prev = self.raw.swap(new_raw, AOrdering::Relaxed);
            if !prev.is_null() {
                release(prev);
            }
        }
    }

    /// Commit the in-flight CB, wait for GPU completion, then atomically
    /// swap in a fresh CB allocated from `queue`. Provided for the
    /// alternative MPSGraph BF16 GEMM path (`mps_graph_ffi.rs`) that
    /// drains Lumen's encoded work before MPSGraph reads from the same
    /// buffers on its own queue. The default in-CB path
    /// (`encode_bf16_matmul_into_cb`) does not need this drain;
    /// kept here as `#[allow(dead_code)]` because the alternative path
    /// is wired up env-OFF and may be revisited in a future revision.
    ///
    /// `unretained` should match how the original CB was created: if
    /// the caller wants the new CB to skip resource retain/release
    /// (matching `LUMEN_METAL_UNRETAINED_CMDBUFS=1`), pass `true`.
    #[allow(dead_code)]
    pub(crate) fn commit_and_swap_to_fresh(
        &self,
        queue: &MetalCommandQueue,
        unretained: bool,
    ) -> Result<(), String> {
        unsafe {
            let prev = self.raw.load(AOrdering::Relaxed);
            if prev.is_null() {
                return Err(
                    "MetalCommandBuffer::commit_and_swap_to_fresh: prior CB is null".into(),
                );
            }
            msg_send_0_raw(prev, cached_sel::commit());
            msg_send_0_raw(prev, cached_sel::wait_until_completed());

            let sel = if unretained {
                cached_sel::command_buffer_with_unretained_references()
            } else {
                cached_sel::command_buffer()
            };
            let fresh = msg_send_0_raw(queue.raw, sel);
            if fresh.is_null() {
                return Err(
                    "MetalCommandBuffer::commit_and_swap_to_fresh: fresh CB alloc returned null"
                        .into(),
                );
            }
            // `commandBuffer` / unretained variants return autoreleased
            // — retain for our wrapper's ownership.
            retain(fresh);
            self.raw.store(fresh, AOrdering::Relaxed);
            release(prev);
            Ok(())
        }
    }
}

impl Drop for MetalCommandBuffer {
    fn drop(&mut self) {
        let r = self.raw.load(AOrdering::Relaxed);
        if !r.is_null() {
            unsafe { release(r) }
        }
        let q = self.queue_raw.load(AOrdering::Relaxed);
        if !q.is_null() {
            unsafe { release(q) }
        }
    }
}

// SAFETY: MetalCommandBuffer wraps an MTLCommandBuffer pointer (single
// owner) and an optional MTLCommandQueue back-reference. The AtomicPtr
// fields provide interior mutability for profile-mode auto-splitting.
// Concurrent encoder creation on the same buffer is not supported by
// Metal regardless, so the existing Send/Sync contract (data lives behind
// &self; ordering is the caller's responsibility) is preserved.
unsafe impl Send for MetalCommandBuffer {}
unsafe impl Sync for MetalCommandBuffer {}

/// Drain the deferred per-CB GPU-time registry (commit-all-wait-once
/// census). Called once at prefill end in defer mode. Waits on the LAST
/// committed split CB (which, on a single FIFO queue, implies all prior CBs
/// completed), then reads each CB's `GPUEndTime - GPUStartTime`, attributes it
/// to its recorded section label, and releases the CB (the registry owns the
/// wrapper's original +1). No-op when the registry is empty / profiling off.
///
/// This is the contamination-free per-kernel census WITHOUT the per-CB-wait
/// idle confound: kernels run back-to-back on the GPU (no forced drain between
/// dispatches), so each CB's GPU wall is its true serial service time. The
/// sum is additive to a serial schedule; compare to production single-CB
/// gpu_busy for the overlap credit (two-metric model).
pub(crate) fn drain_deferred_cbs() {
    let cbs = super::profile::take_deferred_cbs();
    if cbs.is_empty() {
        return;
    }
    unsafe {
        // Wait on the last committed CB; FIFO single-queue ⇒ all prior done.
        let (last_ptr, _) = cbs[cbs.len() - 1];
        let last = last_ptr as ObjcId;
        msg_send_0_raw(last, cached_sel::wait_until_completed());
        // Read each CB's GPU wall, attribute to its label, release it.
        for (ptr, label) in cbs {
            let cb = ptr as ObjcId;
            let start = msg_send_0_f64(cb, cached_sel::gpu_start_time());
            let end = msg_send_0_f64(cb, cached_sel::gpu_end_time());
            super::profile::record_section_gpu_time_for(label, (end - start).max(0.0));
            release(cb);
        }
    }
}

// ============================================================================
// MetalBlitEncoder -- wraps MTLBlitCommandEncoder
// ============================================================================

/// A blit command encoder for buffer-to-buffer copy operations.
///
/// Used to copy data from shared (CPU-accessible) staging buffers into
/// private (GPU-only) buffers. Private storage allows GPU memory controller
/// optimizations that shared storage cannot achieve.
pub struct MetalBlitEncoder {
    raw: ObjcId,
}

impl MetalBlitEncoder {
    /// Copy bytes from one buffer to another.
    ///
    /// Encodes a GPU-side copy from `src` (starting at `src_offset`) to `dst`
    /// (starting at `dst_offset`), copying `size` bytes.
    pub fn copy_from_buffer(
        &self,
        src: &MetalBuffer,
        src_offset: u64,
        dst: &MetalBuffer,
        dst_offset: u64,
        size: u64,
    ) {
        unsafe {
            type CopyFn = unsafe extern "C" fn(
                ObjcId, ObjcSel, ObjcId, u64, ObjcId, u64, u64,
            );
            let f: CopyFn = std::mem::transmute(objc_msgSend as *const c_void);
            f(
                self.raw,
                cached_sel::copy_from_buffer(),
                src.raw,
                src_offset,
                dst.raw,
                dst_offset,
                size,
            );
        }
    }

    /// End encoding.
    pub fn end_encoding(&self) {
        unsafe {
            msg_send_0_raw(self.raw, cached_sel::end_encoding());
        }
    }
}

impl Drop for MetalBlitEncoder {
    fn drop(&mut self) {
        unsafe { release(self.raw) }
    }
}

unsafe impl Send for MetalBlitEncoder {}
unsafe impl Sync for MetalBlitEncoder {}

// ============================================================================
// MetalComputeEncoder — wraps MTLComputeCommandEncoder
// ============================================================================

pub struct MetalComputeEncoder {
    raw: ObjcId,
}

/// A 3D size for dispatching threadgroups.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct MTLSize {
    pub width: u64,
    pub height: u64,
    pub depth: u64,
}

impl MTLSize {
    pub fn new(width: u64, height: u64, depth: u64) -> Self {
        Self { width, height, depth }
    }
}

impl MetalComputeEncoder {
    /// Set the compute pipeline state.
    pub fn set_pipeline_state(&self, pso: &MetalPipelineState) {
        unsafe {
            type SetPSOFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId);
            let f: SetPSOFn = std::mem::transmute(objc_msgSend as *const c_void);
            f(self.raw, cached_sel::set_compute_pipeline_state(), pso.raw);
        }
    }

    /// Set a buffer at the given index.
    pub fn set_buffer(&self, buffer: &MetalBuffer, offset: u64, index: u64) {
        unsafe {
            type SetBufFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, u64, u64);
            let f: SetBufFn = std::mem::transmute(objc_msgSend as *const c_void);
            f(
                self.raw,
                cached_sel::set_buffer_offset_at_index(),
                buffer.raw,
                offset,
                index,
            );
        }
    }

    /// Set raw bytes at the given index (for small constant data).
    pub fn set_bytes(&self, bytes: &[u8], index: u64) {
        unsafe {
            type SetBytesFn = unsafe extern "C" fn(ObjcId, ObjcSel, *const c_void, u64, u64);
            let f: SetBytesFn = std::mem::transmute(objc_msgSend as *const c_void);
            f(
                self.raw,
                cached_sel::set_bytes_length_at_index(),
                bytes.as_ptr() as *const c_void,
                bytes.len() as u64,
                index,
            );
        }
    }

    /// Dispatch threadgroups.
    ///
    /// Uses a typed function pointer cast because passing MTLSize structs
    /// through variadic objc_msgSend does not work on arm64 (structs larger
    /// than registers get misrouted through the variadic ABI).
    pub fn dispatch_threadgroups(
        &self,
        threadgroups: MTLSize,
        threads_per_threadgroup: MTLSize,
    ) {
        unsafe {
            type DispatchFn = unsafe extern "C" fn(ObjcId, ObjcSel, MTLSize, MTLSize);
            let f: DispatchFn = std::mem::transmute(objc_msgSend as *const c_void);
            f(
                self.raw,
                cached_sel::dispatch_threadgroups(),
                threadgroups,
                threads_per_threadgroup,
            );
        }
    }

    /// Dispatch threads (non-uniform).
    ///
    /// Uses a typed function pointer cast (same reason as dispatch_threadgroups).
    pub fn dispatch_threads(
        &self,
        threads: MTLSize,
        threads_per_threadgroup: MTLSize,
    ) {
        unsafe {
            type DispatchFn = unsafe extern "C" fn(ObjcId, ObjcSel, MTLSize, MTLSize);
            let f: DispatchFn = std::mem::transmute(objc_msgSend as *const c_void);
            f(
                self.raw,
                cached_sel::dispatch_threads(),
                threads,
                threads_per_threadgroup,
            );
        }
    }

    /// Set threadgroup memory length at an index.
    ///
    /// Required for kernels that declare `threadgroup` arrays sized at dispatch
    /// time (e.g., tiled matmul shared memory tiles). The length is in bytes.
    pub fn set_threadgroup_memory_length(&self, length: u64, index: u64) {
        unsafe {
            type SetTGMemFn = unsafe extern "C" fn(ObjcId, ObjcSel, u64, u64);
            let f: SetTGMemFn = std::mem::transmute(objc_msgSend as *const c_void);
            f(
                self.raw,
                cached_sel::set_threadgroup_memory_length(),
                length,
                index,
            );
        }
    }

    /// Insert a memory barrier for the specified scope.
    ///
    /// Used between Split-K GEMM write and reduce read within the same encoder
    /// to ensure all threadgroups have finished writing partial results before
    /// the reduction kernel reads them.
    ///
    /// Scope values: 1 = MTLBarrierScope.buffers, 2 = MTLBarrierScope.textures,
    /// 4 = MTLBarrierScope.renderTargets.
    pub fn memory_barrier_with_scope(&self, scope: u64) {
        unsafe {
            type BarrierFn = unsafe extern "C" fn(ObjcId, ObjcSel, u64);
            let f: BarrierFn = std::mem::transmute(objc_msgSend as *const c_void);
            f(self.raw, cached_sel::memory_barrier_with_scope(), scope);
        }
    }

    /// Insert a resource-scoped memory barrier on the supplied buffers.
    ///
    /// Wraps `[id<MTLComputeCommandEncoder> memoryBarrierWithResources:count:]`.
    /// Use inside a concurrent compute encoder (`new_concurrent_compute_encoder`)
    /// to enforce that all prior dispatches finish writing to the listed
    /// buffers before any subsequent dispatch reads or writes them. This is
    /// finer-grained than `memory_barrier_with_scope(1)`, which serialises
    /// against the entire buffer namespace; per Apple docs, scoping the
    /// barrier to a specific resource list lets the GPU keep unrelated work
    /// in flight.
    ///
    /// Slices longer than 32 entries are silently truncated; no per-layer
    /// plan in Lumen exceeds that. A zero-length slice is a no-op.
    pub fn memory_barrier_with_resources(&self, buffers: &[&MetalBuffer]) {
        if buffers.is_empty() { return; }
        // Stack-allocated buffer (avoids pulling in a smallvec crate just
        // for the barrier path). 32 is well above any realistic per-layer
        // distinct-mutable-buffer count.
        let mut raws: [ObjcId; 32] = [std::ptr::null_mut(); 32];
        let n = buffers.len().min(raws.len());
        for i in 0..n { raws[i] = buffers[i].raw; }
        unsafe {
            type BarrierResFn =
                unsafe extern "C" fn(ObjcId, ObjcSel, *const ObjcId, u64);
            let f: BarrierResFn = std::mem::transmute(objc_msgSend as *const c_void);
            f(
                self.raw,
                cached_sel::memory_barrier_with_resources(),
                raws.as_ptr(),
                n as u64,
            );
        }
    }

    /// End encoding.
    pub fn end_encoding(&self) {
        unsafe {
            msg_send_0_raw(self.raw, cached_sel::end_encoding());
        }
    }
}

impl Drop for MetalComputeEncoder {
    fn drop(&mut self) {
        unsafe { release(self.raw) }
    }
}

unsafe impl Send for MetalComputeEncoder {}
unsafe impl Sync for MetalComputeEncoder {}

// ============================================================================
// MetalLibrary — wraps MTLLibrary
// ============================================================================

pub struct MetalLibrary {
    raw: ObjcId,
}

impl MetalLibrary {
    /// Get a function by name from the library.
    pub fn get_function(&self, name: &str) -> Option<MetalFunction> {
        unsafe {
            let ns_name = nsstring(name);
            type NewFuncFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId) -> ObjcId;
            let f: NewFuncFn = std::mem::transmute(objc_msgSend as *const c_void);
            let func = f(self.raw, cached_sel::new_function_with_name(), ns_name);
            if func.is_null() {
                None
            } else {
                Some(MetalFunction { raw: func })
            }
        }
    }

    /// Get a function by name, specialized with function constant values.
    ///
    /// Uses `newFunctionWithName:constantValues:error:` to create a function
    /// variant where `[[function_constant(N)]]` declarations are bound to the
    /// provided constant values. The Metal compiler uses these constants to
    /// dead-code-eliminate branches, producing a specialized kernel.
    pub fn get_function_with_constants(
        &self,
        name: &str,
        constants: &MetalFunctionConstantValues,
    ) -> Result<MetalFunction, String> {
        unsafe {
            let ns_name = nsstring(name);
            let mut error: ObjcId = ptr::null_mut();
            type NewFuncConstFn = unsafe extern "C" fn(
                ObjcId, ObjcSel, ObjcId, ObjcId, *mut ObjcId,
            ) -> ObjcId;
            let f: NewFuncConstFn = std::mem::transmute(objc_msgSend as *const c_void);
            let func = f(
                self.raw,
                cached_sel::new_function_with_name_constant_values(),
                ns_name,
                constants.raw,
                &mut error as *mut ObjcId,
            );

            if func.is_null() || !error.is_null() {
                let desc = error_description(error);
                if !error.is_null() {
                    release(error);
                }
                return Err(format!(
                    "Function '{}' with constants failed: {desc}",
                    name
                ));
            }

            Ok(MetalFunction { raw: func })
        }
    }
}

impl Drop for MetalLibrary {
    fn drop(&mut self) {
        unsafe { release(self.raw) }
    }
}

unsafe impl Send for MetalLibrary {}
unsafe impl Sync for MetalLibrary {}

// ============================================================================
// MetalFunctionConstantValues — wraps MTLFunctionConstantValues
// ============================================================================

/// A set of function constant values used to specialize Metal shader functions.
///
/// Function constants allow the Metal compiler to dead-code-eliminate branches
/// at pipeline creation time, producing specialized kernel variants without
/// runtime branching overhead.
pub struct MetalFunctionConstantValues {
    raw: ObjcId,
}

impl MetalFunctionConstantValues {
    /// Create a new empty function constant values set.
    pub fn new() -> Self {
        unsafe {
            let fcv_class = cls("MTLFunctionConstantValues");
            let alloc = msg_send_0_raw(fcv_class, cached_sel::alloc());
            let raw = msg_send_0_raw(alloc, cached_sel::init());
            Self { raw }
        }
    }

    /// Set a boolean constant at the given index.
    ///
    /// The index corresponds to `[[function_constant(index)]]` in the shader.
    pub fn set_bool(&self, value: bool, index: u64) {
        unsafe {
            // MTLDataType.bool = 53
            const MTL_DATA_TYPE_BOOL: u64 = 53;
            let val: u8 = if value { 1 } else { 0 };
            type SetConstFn =
                unsafe extern "C" fn(ObjcId, ObjcSel, *const c_void, u64, u64);
            let f: SetConstFn = std::mem::transmute(objc_msgSend as *const c_void);
            f(
                self.raw,
                cached_sel::set_constant_value(),
                &val as *const u8 as *const c_void,
                MTL_DATA_TYPE_BOOL,
                index,
            );
        }
    }

    /// Set a 16-bit signed integer constant at the given index.
    ///
    /// The index corresponds to `[[function_constant(index)]]` in the shader,
    /// where the declaration is `constant short FC_name [[function_constant(N)]]`.
    /// Wraps `MTLFunctionConstantValues setConstantValue:type:atIndex:` with
    /// `MTLDataType.short` (42) for `int16_t` function constants.
    pub fn set_int16(&self, value: i16, index: u64) {
        unsafe {
            // MTLDataType.short = 42
            const MTL_DATA_TYPE_SHORT: u64 = 42;
            type SetConstFn =
                unsafe extern "C" fn(ObjcId, ObjcSel, *const c_void, u64, u64);
            let f: SetConstFn = std::mem::transmute(objc_msgSend as *const c_void);
            f(
                self.raw,
                cached_sel::set_constant_value(),
                &value as *const i16 as *const c_void,
                MTL_DATA_TYPE_SHORT,
                index,
            );
        }
    }
}

impl Drop for MetalFunctionConstantValues {
    fn drop(&mut self) {
        unsafe { release(self.raw) }
    }
}

unsafe impl Send for MetalFunctionConstantValues {}
unsafe impl Sync for MetalFunctionConstantValues {}

// ============================================================================
// MetalFunction — wraps MTLFunction
// ============================================================================

pub struct MetalFunction {
    raw: ObjcId,
}

impl Drop for MetalFunction {
    fn drop(&mut self) {
        unsafe { release(self.raw) }
    }
}

unsafe impl Send for MetalFunction {}
unsafe impl Sync for MetalFunction {}

// ============================================================================
// MetalPipelineState — wraps MTLComputePipelineState
// ============================================================================

pub struct MetalPipelineState {
    raw: ObjcId,
}

impl MetalPipelineState {
    /// Maximum total threads per threadgroup.
    pub fn max_total_threads_per_threadgroup(&self) -> u64 {
        unsafe {
            // This returns NSUInteger (u64 on 64-bit). Use typed function pointer.
            type GetU64Fn = unsafe extern "C" fn(ObjcId, ObjcSel) -> u64;
            let f: GetU64Fn = std::mem::transmute(objc_msgSend as *const c_void);
            f(self.raw, cached_sel::max_total_threads_per_threadgroup())
        }
    }

    /// Thread execution width (SIMD group width, typically 32 on Apple GPU).
    pub fn thread_execution_width(&self) -> u64 {
        unsafe {
            type GetU64Fn = unsafe extern "C" fn(ObjcId, ObjcSel) -> u64;
            let f: GetU64Fn = std::mem::transmute(objc_msgSend as *const c_void);
            f(self.raw, cached_sel::thread_execution_width())
        }
    }
}

impl Drop for MetalPipelineState {
    fn drop(&mut self) {
        unsafe { release(self.raw) }
    }
}

unsafe impl Send for MetalPipelineState {}
unsafe impl Sync for MetalPipelineState {}

// ============================================================================
// MetalBuffer — wraps MTLBuffer
// ============================================================================

pub struct MetalBuffer {
    raw: ObjcId,
}

impl MetalBuffer {
    /// Get a raw pointer to the buffer contents.
    ///
    /// For `StorageModeShared` on Apple Silicon, this is CPU-accessible
    /// and points to the same physical memory the GPU uses. Zero-copy.
    pub fn contents(&self) -> *mut c_void {
        unsafe {
            type ContentsFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> *mut c_void;
            let f: ContentsFn = std::mem::transmute(objc_msgSend as *const c_void);
            f(self.raw, cached_sel::contents())
        }
    }

    /// Buffer length in bytes.
    pub fn length(&self) -> u64 {
        unsafe {
            type LengthFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> u64;
            let f: LengthFn = std::mem::transmute(objc_msgSend as *const c_void);
            f(self.raw, cached_sel::length())
        }
    }

    /// Write f32 data into the buffer.
    ///
    /// # Panics
    ///
    /// Panics if the data exceeds the buffer length.
    pub fn write_f32(&self, data: &[f32]) {
        let byte_len = data.len() * 4;
        assert!(
            byte_len as u64 <= self.length(),
            "MetalBuffer::write_f32: data size ({byte_len}) exceeds buffer length ({})",
            self.length()
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                self.contents() as *mut u8,
                byte_len,
            );
        }
    }

    /// Read f32 data from the buffer into a slice.
    ///
    /// # Panics
    ///
    /// Panics if the read would exceed the buffer length.
    pub fn read_f32(&self, out: &mut [f32]) {
        let byte_len = out.len() * 4;
        assert!(
            byte_len as u64 <= self.length(),
            "MetalBuffer::read_f32: read size ({byte_len}) exceeds buffer length ({})",
            self.length()
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.contents() as *const u8,
                out.as_mut_ptr() as *mut u8,
                byte_len,
            );
        }
    }

    /// Read u32 data from the buffer into a slice.
    ///
    /// # Panics
    ///
    /// Panics if the read would exceed the buffer length.
    pub fn read_u32(&self, out: &mut [u32]) {
        let byte_len = out.len() * 4;
        assert!(
            byte_len as u64 <= self.length(),
            "MetalBuffer::read_u32: read size ({byte_len}) exceeds buffer length ({})",
            self.length()
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.contents() as *const u8,
                out.as_mut_ptr() as *mut u8,
                byte_len,
            );
        }
    }

    /// Read u64 data out of the buffer (for DET-001 per-layer FNV hash readback).
    ///
    /// # Panics
    ///
    /// Panics if the requested size exceeds the buffer length.
    pub fn read_u64(&self, out: &mut [u64]) {
        let byte_len = out.len() * 8;
        assert!(
            byte_len as u64 <= self.length(),
            "MetalBuffer::read_u64: read size ({byte_len}) exceeds buffer length ({})",
            self.length()
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.contents() as *const u8,
                out.as_mut_ptr() as *mut u8,
                byte_len,
            );
        }
    }

    /// Write u16 data into the buffer (for f16 KV cache).
    ///
    /// # Panics
    ///
    /// Panics if the data exceeds the buffer length.
    pub fn write_u16(&self, data: &[u16]) {
        let byte_len = data.len() * 2;
        assert!(
            byte_len as u64 <= self.length(),
            "MetalBuffer::write_u16: data size ({byte_len}) exceeds buffer length ({})",
            self.length()
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                self.contents() as *mut u8,
                byte_len,
            );
        }
    }

    /// Read u16 data from the buffer into a slice (for f16 KV cache).
    ///
    /// # Panics
    ///
    /// Panics if the read would exceed the buffer length.
    pub fn read_u16(&self, out: &mut [u16]) {
        let byte_len = out.len() * 2;
        assert!(
            byte_len as u64 <= self.length(),
            "MetalBuffer::read_u16: read size ({byte_len}) exceeds buffer length ({})",
            self.length()
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.contents() as *const u8,
                out.as_mut_ptr() as *mut u8,
                byte_len,
            );
        }
    }

    /// Raw Objective-C id for this buffer.
    pub fn raw(&self) -> ObjcId {
        self.raw
    }
}

impl Drop for MetalBuffer {
    fn drop(&mut self) {
        unsafe { release(self.raw) }
    }
}

// SAFETY: MTLBuffer with StorageModeShared is thread-safe for read access.
// Writes require synchronization, which our backend handles via command buffer ordering.
unsafe impl Send for MetalBuffer {}
unsafe impl Sync for MetalBuffer {}

impl fmt::Debug for MetalBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetalBuffer")
            .field("length", &self.length())
            .finish()
    }
}

// ============================================================================
// NSURL helper
// ============================================================================

/// Create an NSURL from a file path string.
///
/// Uses `[NSURL fileURLWithPath:]` to create a file URL.
unsafe fn nsurl_from_path(path: &str) -> ObjcId {
    let ns_path = nsstring(path);
    let nsurl_class = cls("NSURL");
    type FileURLFn = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId) -> ObjcId;
    let f: FileURLFn = std::mem::transmute(objc_msgSend as *const c_void);
    f(nsurl_class, cached_sel::file_url_with_path(), ns_path)
}

// ============================================================================
// MetalIOCommandQueue — wraps MTLIOCommandQueue (Metal 3, macOS 13+)
// ============================================================================
//
// MTLIOCommandQueue enables direct NVMe-to-GPU DMA transfers, bypassing the
// CPU memory subsystem entirely. This is the optimal path for streaming
// large weight tensors from SSD into GPU-private buffers.
//
// Requirements: macOS 13+, Metal 3 (M2 or later).

/// MTLIOCommandQueue type for concurrent file-to-GPU DMA.
pub struct MetalIOCommandQueue {
    raw: ObjcId,
}

impl MetalIOCommandQueue {
    /// Create a new concurrent IO command queue on the given device.
    ///
    /// Returns `None` if the device does not support MTLIOCommandQueue
    /// (pre-Metal 3 hardware or pre-macOS 13).
    pub fn new(device: &MetalDevice) -> Option<Self> {
        unsafe {
            // Allocate and init MTLIOCommandQueueDescriptor.
            let desc_class = cls("MTLIOCommandQueueDescriptor");
            if desc_class.is_null() {
                // MTLIOCommandQueueDescriptor class not available (pre-macOS 13).
                return None;
            }
            let desc_alloc = msg_send_0_raw(desc_class, cached_sel::alloc());
            if desc_alloc.is_null() {
                return None;
            }
            let desc = msg_send_0_raw(desc_alloc, cached_sel::init());
            if desc.is_null() {
                return None;
            }

            // Set type to MTLIOCommandQueueTypeConcurrent (0).
            type SetTypeFn = unsafe extern "C" fn(ObjcId, ObjcSel, u64);
            let set_type_f: SetTypeFn = std::mem::transmute(objc_msgSend as *const c_void);
            set_type_f(desc, cached_sel::set_type(), 0u64); // Concurrent = 0

            // Create the IO command queue.
            let mut error: ObjcId = ptr::null_mut();
            type NewIOQueueFn = unsafe extern "C" fn(
                ObjcId, ObjcSel, ObjcId, *mut ObjcId,
            ) -> ObjcId;
            let f: NewIOQueueFn = std::mem::transmute(objc_msgSend as *const c_void);
            let io_queue = f(
                device.raw(),
                cached_sel::new_io_command_queue(),
                desc,
                &mut error as *mut ObjcId,
            );

            // Release the descriptor (we're done with it).
            release(desc);

            if io_queue.is_null() {
                if !error.is_null() {
                    release(error);
                }
                return None;
            }

            Some(Self { raw: io_queue })
        }
    }

    /// Open a file handle for IO operations.
    ///
    /// Returns a `MetalIOFileHandle` that can be used with `load_buffer`.
    pub fn open_file(device: &MetalDevice, path: &str) -> Result<MetalIOFileHandle, String> {
        unsafe {
            let url = nsurl_from_path(path);
            if url.is_null() {
                return Err(format!("Failed to create NSURL for path: {path}"));
            }

            let mut error: ObjcId = ptr::null_mut();
            type NewFileHandleFn = unsafe extern "C" fn(
                ObjcId, ObjcSel, ObjcId, *mut ObjcId,
            ) -> ObjcId;
            let f: NewFileHandleFn = std::mem::transmute(objc_msgSend as *const c_void);
            let handle = f(
                device.raw(),
                cached_sel::new_io_file_handle(),
                url,
                &mut error as *mut ObjcId,
            );

            if handle.is_null() {
                let desc = if !error.is_null() {
                    let d = error_description(error);
                    release(error);
                    d
                } else {
                    "unknown error".to_string()
                };
                return Err(format!("Failed to open IO file handle for {path}: {desc}"));
            }

            Ok(MetalIOFileHandle { raw: handle })
        }
    }

    /// Create a new IO command buffer.
    pub fn command_buffer(&self) -> Option<MetalIOCommandBuffer> {
        let raw = unsafe { msg_send_0_raw(self.raw, cached_sel::io_command_buffer()) };
        if raw.is_null() {
            None
        } else {
            // IO commandBuffer may be autoreleased; retain it.
            unsafe { retain(raw) };
            Some(MetalIOCommandBuffer { raw })
        }
    }
}

impl Drop for MetalIOCommandQueue {
    fn drop(&mut self) {
        unsafe { release(self.raw) }
    }
}

unsafe impl Send for MetalIOCommandQueue {}
unsafe impl Sync for MetalIOCommandQueue {}

// ============================================================================
// MetalIOFileHandle — wraps MTLIOFileHandle
// ============================================================================

/// A file handle for Metal IO operations. Created via
/// `MetalIOCommandQueue::open_file()`.
pub struct MetalIOFileHandle {
    raw: ObjcId,
}

impl Drop for MetalIOFileHandle {
    fn drop(&mut self) {
        unsafe { release(self.raw) }
    }
}

unsafe impl Send for MetalIOFileHandle {}
unsafe impl Sync for MetalIOFileHandle {}

// ============================================================================
// MetalIOCommandBuffer — wraps MTLIOCommandBuffer
// ============================================================================

/// A command buffer for Metal IO operations (file-to-buffer DMA).
pub struct MetalIOCommandBuffer {
    raw: ObjcId,
}

/// MTLIOStatus values for checking IO command buffer completion status.
#[repr(u64)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalIOStatus {
    Pending = 0,
    Cancelled = 1,
    Error = 2,
    Complete = 3,
}

impl MetalIOCommandBuffer {
    /// Enqueue a load from a file handle into a Metal buffer.
    ///
    /// Loads `byte_count` bytes from the file at `source_offset` into
    /// `dest_buffer` at `dest_offset`.
    pub fn load_buffer(
        &self,
        dest_buffer: &MetalBuffer,
        dest_offset: u64,
        byte_count: u64,
        source_handle: &MetalIOFileHandle,
        source_offset: u64,
    ) {
        unsafe {
            type LoadBufferFn = unsafe extern "C" fn(
                ObjcId, ObjcSel, ObjcId, u64, u64, ObjcId, u64,
            );
            let f: LoadBufferFn = std::mem::transmute(objc_msgSend as *const c_void);
            f(
                self.raw,
                cached_sel::load_buffer(),
                dest_buffer.raw(),
                dest_offset,
                byte_count,
                source_handle.raw,
                source_offset,
            );
        }
    }

    /// Commit the IO command buffer for execution.
    pub fn commit(&self) {
        unsafe {
            msg_send_0_raw(self.raw, cached_sel::commit());
        }
    }

    /// Wait until the IO command buffer has completed.
    pub fn wait_until_completed(&self) {
        unsafe {
            msg_send_0_raw(self.raw, cached_sel::wait_until_completed());
        }
    }

    /// Commit and wait for completion (synchronous).
    pub fn commit_and_wait(&self) {
        self.commit();
        self.wait_until_completed();
    }

    /// Query the status of the IO command buffer.
    pub fn status(&self) -> MetalIOStatus {
        unsafe {
            type StatusFn = unsafe extern "C" fn(ObjcId, ObjcSel) -> u64;
            let f: StatusFn = std::mem::transmute(objc_msgSend as *const c_void);
            let val = f(self.raw, cached_sel::status());
            match val {
                0 => MetalIOStatus::Pending,
                1 => MetalIOStatus::Cancelled,
                2 => MetalIOStatus::Error,
                3 => MetalIOStatus::Complete,
                _ => MetalIOStatus::Error, // Unknown status treated as error.
            }
        }
    }
}

impl Drop for MetalIOCommandBuffer {
    fn drop(&mut self) {
        unsafe { release(self.raw) }
    }
}

unsafe impl Send for MetalIOCommandBuffer {}
unsafe impl Sync for MetalIOCommandBuffer {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_device_available() {
        let device = MetalDevice::system_default();
        assert!(device.is_some(), "Metal device should be available on macOS");
        if let Some(dev) = &device {
            let name = dev.name();
            assert!(!name.is_empty(), "Device name should not be empty");
            eprintln!("Metal device: {name}");
        }
    }

    #[test]
    fn test_metal_command_queue() {
        let device = MetalDevice::system_default().unwrap();
        let queue = device.new_command_queue();
        assert!(queue.is_some(), "Should create command queue");
    }

    #[test]
    fn test_metal_buffer_create_and_rw() {
        let device = MetalDevice::system_default().unwrap();
        let buf = device.new_buffer(1024).unwrap();
        assert!(buf.length() >= 1024);

        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        buf.write_f32(&data);

        let mut out = vec![0.0f32; 4];
        buf.read_f32(&mut out);
        assert_eq!(data, out);
    }

    #[test]
    fn test_metal_buffer_with_bytes() {
        let device = MetalDevice::system_default().unwrap();
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        };
        let buf = device.new_buffer_with_bytes(bytes).unwrap();
        assert!(buf.length() >= 16);

        let mut out = vec![0.0f32; 4];
        buf.read_f32(&mut out);
        assert_eq!(&data[..], &out[..]);
    }

    #[test]
    fn test_shader_compilation() {
        let device = MetalDevice::system_default().unwrap();
        let lib = device.new_library_with_source(crate::metal::shaders::METAL_SHADER_SOURCE);
        assert!(lib.is_ok(), "Shader compilation failed: {:?}", lib.err());

        let lib = lib.unwrap();

        // Verify we can get all expected kernel functions
        let kernel_names = [
            "matmul_f32", "matmul_bytes_f32", "rmsnorm", "rmsnorm_bytes",
            "rope", "swiglu", "softmax", "attention_scores",
            "attention_output", "add_residual", "embed_token",
            "dequant_tiled_matmul_q8_0_splitk", "reduce_splitk",
        ];
        for name in &kernel_names {
            let func = lib.get_function(name);
            assert!(func.is_some(), "Kernel function '{name}' not found in compiled library");
        }
    }

    #[test]
    fn test_pipeline_state_creation() {
        let device = MetalDevice::system_default().unwrap();
        let lib = device.new_library_with_source(crate::metal::shaders::METAL_SHADER_SOURCE).unwrap();
        let func = lib.get_function("add_residual").unwrap();
        let pso = device.new_compute_pipeline_state(&func);
        assert!(pso.is_ok(), "Pipeline state creation failed: {:?}", pso.err());

        let pso = pso.unwrap();
        let max_threads = pso.max_total_threads_per_threadgroup();
        let simd_width = pso.thread_execution_width();
        eprintln!("add_residual pipeline: max_threads={max_threads}, simd_width={simd_width}");
        assert!(max_threads >= 32, "Expected at least 32 threads per threadgroup");
        assert!(simd_width >= 32, "Expected SIMD width >= 32 on Apple GPU");
    }

    #[test]
    fn test_simple_add_residual_gpu() {
        let device = MetalDevice::system_default().unwrap();
        let queue = device.new_command_queue().unwrap();
        let lib = device.new_library_with_source(crate::metal::shaders::METAL_SHADER_SOURCE).unwrap();
        let func = lib.get_function("add_residual").unwrap();
        let pso = device.new_compute_pipeline_state(&func).unwrap();

        // Create test data
        let dim = 8u32;
        let dst_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let src_data = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];

        let dst_buf = device.new_buffer(32).unwrap();
        let src_buf = device.new_buffer(32).unwrap();
        let dim_buf = device.new_buffer_with_bytes(&dim.to_le_bytes()).unwrap();

        dst_buf.write_f32(&dst_data);
        src_buf.write_f32(&src_data);

        // Encode and dispatch
        let cmd = queue.new_command_buffer().unwrap();
        let enc = cmd.new_compute_encoder().unwrap();
        enc.set_pipeline_state(&pso);
        enc.set_buffer(&dst_buf, 0, 0);
        enc.set_buffer(&src_buf, 0, 1);
        enc.set_buffer(&dim_buf, 0, 2);
        enc.dispatch_threads(
            MTLSize::new(dim as u64, 1, 1),
            MTLSize::new(dim as u64, 1, 1),
        );
        enc.end_encoding();
        cmd.commit_and_wait();

        // Verify
        let mut result = vec![0.0f32; 8];
        dst_buf.read_f32(&mut result);

        let expected: Vec<f32> = dst_data.iter().zip(src_data.iter()).map(|(a, b)| a + b).collect();
        assert_eq!(result, expected, "GPU add_residual should produce correct results");
    }

    #[test]
    fn test_concurrent_compute_encoder() {
        // Verify concurrent encoder produces correct results when dispatching
        // two independent add_residual operations to non-overlapping buffers.
        let device = MetalDevice::system_default().unwrap();
        let queue = device.new_command_queue().unwrap();
        let lib = device.new_library_with_source(crate::metal::shaders::METAL_SHADER_SOURCE).unwrap();
        let func = lib.get_function("add_residual").unwrap();
        let pso = device.new_compute_pipeline_state(&func).unwrap();

        let dim = 8u32;
        // First pair: dst_a += src_a
        let dst_a_data = vec![1.0f32; 8];
        let src_a_data = vec![10.0f32; 8];
        let dst_a_buf = device.new_buffer(32).unwrap();
        let src_a_buf = device.new_buffer(32).unwrap();
        dst_a_buf.write_f32(&dst_a_data);
        src_a_buf.write_f32(&src_a_data);

        // Second pair: dst_b += src_b
        let dst_b_data = vec![100.0f32; 8];
        let src_b_data = vec![200.0f32; 8];
        let dst_b_buf = device.new_buffer(32).unwrap();
        let src_b_buf = device.new_buffer(32).unwrap();
        dst_b_buf.write_f32(&dst_b_data);
        src_b_buf.write_f32(&src_b_data);

        let dim_buf = device.new_buffer_with_bytes(&dim.to_le_bytes()).unwrap();

        // Use concurrent encoder -- both dispatches can overlap
        let cmd = queue.new_command_buffer().unwrap();
        let enc = cmd.new_concurrent_compute_encoder().unwrap();

        // Dispatch A
        enc.set_pipeline_state(&pso);
        enc.set_buffer(&dst_a_buf, 0, 0);
        enc.set_buffer(&src_a_buf, 0, 1);
        enc.set_buffer(&dim_buf, 0, 2);
        enc.dispatch_threads(
            MTLSize::new(dim as u64, 1, 1),
            MTLSize::new(dim as u64, 1, 1),
        );

        // Dispatch B (writes to completely separate buffers)
        enc.set_buffer(&dst_b_buf, 0, 0);
        enc.set_buffer(&src_b_buf, 0, 1);
        enc.dispatch_threads(
            MTLSize::new(dim as u64, 1, 1),
            MTLSize::new(dim as u64, 1, 1),
        );

        enc.memory_barrier_with_scope(1); // MTLBarrierScope.buffers
        enc.end_encoding();
        cmd.commit_and_wait();

        // Verify both operations completed correctly
        let mut result_a = vec![0.0f32; 8];
        let mut result_b = vec![0.0f32; 8];
        dst_a_buf.read_f32(&mut result_a);
        dst_b_buf.read_f32(&mut result_b);

        assert_eq!(result_a, vec![11.0f32; 8], "Concurrent dispatch A should produce correct results");
        assert_eq!(result_b, vec![300.0f32; 8], "Concurrent dispatch B should produce correct results");
    }
}
