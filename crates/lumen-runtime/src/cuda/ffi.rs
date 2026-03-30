//! Thin wrapper around cudarc for CUDA device, stream, and buffer management.
//!
//! Provides error conversion from cudarc errors to `RuntimeError` and
//! encapsulates the cudarc types used by the CUDA backend.
//!
//! cudarc 0.19 API: CudaContext (device) -> CudaStream (ops) -> CudaModule (kernels).
//! cuBLAS handle is created alongside the stream for GEMV dispatch.

use crate::error::RuntimeError;
use cudarc::cublas::CudaBlas;
use cudarc::driver::{
    CudaContext, CudaModule, CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits,
};
use std::sync::Arc;

/// Wraps cudarc handles for a single CUDA device.
///
/// Owns the device context, default stream, compiled kernel modules,
/// and a cuBLAS handle bound to the default stream for GEMV operations.
pub struct CudaDevice {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub blas: CudaBlas,
}

/// Query the number of available CUDA devices.
///
/// Calls `cuInit(0)` before querying device count. The CUDA driver API
/// requires initialization before any other driver call; without it,
/// `cuDeviceGetCount` fails with `CUDA_ERROR_NOT_INITIALIZED` on fresh
/// containers (e.g. Modal). `cuInit` is idempotent -- safe to call
/// multiple times.
pub fn device_count() -> Result<usize, RuntimeError> {
    cudarc::driver::result::init().map_err(cuda_driver_err)?;
    let count = cudarc::driver::result::device::get_count().map_err(cuda_driver_err)?;
    Ok(count as usize)
}

impl CudaDevice {
    /// Create a new CUDA device context on the specified device ordinal.
    ///
    /// `device_id` selects which GPU (0-indexed). Fails with a descriptive
    /// message if the device ordinal is out of range or no CUDA driver is found.
    ///
    /// Initializes a cuBLAS handle bound to the device's default stream.
    pub fn new(device_id: usize) -> Result<Self, RuntimeError> {
        let count = device_count().map_err(|_| {
            RuntimeError::Compute(
                "No CUDA driver found -- is the NVIDIA driver installed?".into(),
            )
        })?;

        if count == 0 {
            return Err(RuntimeError::Compute(
                "No CUDA devices found -- is the NVIDIA driver installed?".into(),
            ));
        }

        if device_id >= count {
            return Err(RuntimeError::Compute(format!(
                "CUDA device {device_id} not found -- only {count} device(s) available",
            )));
        }

        let ctx = CudaContext::new(device_id).map_err(cuda_driver_err)?;

        // Disable cudarc's automatic event tracking BEFORE creating any streams
        // or allocating any buffers. We use a single stream for all operations,
        // so multi-stream synchronization via events is unnecessary. More
        // critically, event tracking breaks CUDA graph capture: during capture,
        // cudarc's PushKernelArg impl inserts cuStreamWaitEvent calls for every
        // CudaSlice argument, which creates cross-stream event dependencies that
        // cause CUDA_ERROR_STREAM_CAPTURE_ISOLATION. By disabling event tracking,
        // CudaSlice objects are created without read/write events (None), and
        // kernel launches skip the cuStreamWaitEvent calls entirely.
        //
        // SAFETY: We use exactly one stream for all GPU operations (kernels,
        // memcpy, cuBLAS). Single-stream execution is inherently ordered --
        // operations on the same stream execute in submission order. No
        // cross-stream synchronization is needed.
        unsafe { ctx.disable_event_tracking(); }

        // Use a NEW stream (not default_stream) -- the legacy default stream
        // (stream 0) does NOT support CUDA graph capture. A non-default stream
        // created via cuStreamCreate supports stream capture, graph recording,
        // and replay.
        let stream = ctx.new_stream().map_err(cuda_driver_err)?;
        let blas = CudaBlas::new(stream.clone()).map_err(cuda_cublas_err)?;
        Ok(Self { ctx, stream, blas })
    }

    /// Query the device name (e.g. "NVIDIA A10G").
    pub fn name(&self) -> Result<String, RuntimeError> {
        self.ctx.name().map_err(cuda_driver_err)
    }

    /// Query total device memory in bytes (VRAM).
    pub fn total_memory(&self) -> Result<usize, RuntimeError> {
        let (_free, total) =
            cudarc::driver::result::mem_get_info().map_err(cuda_driver_err)?;
        Ok(total)
    }

    /// Query free device memory in bytes.
    pub fn free_memory(&self) -> Result<usize, RuntimeError> {
        let (free, _total) =
            cudarc::driver::result::mem_get_info().map_err(cuda_driver_err)?;
        Ok(free)
    }

    /// Query compute capability as (major, minor), e.g. (8, 6) for SM 8.6.
    pub fn compute_capability(&self) -> Result<(i32, i32), RuntimeError> {
        self.ctx.compute_capability().map_err(cuda_driver_err)
    }

    /// Compile CUDA source to PTX via NVRTC and load as a module.
    pub fn compile_and_load(
        &self,
        cuda_source: &str,
    ) -> Result<Arc<CudaModule>, RuntimeError> {
        let ptx = cudarc::nvrtc::compile_ptx(cuda_source).map_err(cuda_nvrtc_err)?;
        self.ctx.load_module(ptx).map_err(cuda_driver_err)
    }

    /// Compile CUDA source targeting a specific SM architecture.
    ///
    /// Used for kernels requiring specific hardware features (e.g., tensor cores
    /// need SM 70+ for WMMA, SM 80+ for mma.sync.aligned.m16n8k16).
    ///
    /// `arch` is the SM target string, e.g. "compute_80" for A100.
    pub fn compile_and_load_with_arch(
        &self,
        cuda_source: &str,
        arch: &'static str,
    ) -> Result<Arc<CudaModule>, RuntimeError> {
        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some(arch),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(cuda_source, opts)
            .map_err(cuda_nvrtc_err)?;
        self.ctx.load_module(ptx).map_err(cuda_driver_err)
    }

    /// Compile CUDA source targeting a specific SM architecture with --use_fast_math.
    ///
    /// Enables --fmad=true --ftz=true --prec-div=false --prec-sqrt=false which
    /// accelerates scale multiplications in bandwidth-bound GEMV kernels.
    /// Used for dp4a Q8_1 kernels where scale * acc is the only FP math.
    pub fn compile_and_load_with_arch_fast_math(
        &self,
        cuda_source: &str,
        arch: &'static str,
    ) -> Result<Arc<CudaModule>, RuntimeError> {
        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some(arch),
            // Use raw --use_fast_math flag via options vec for full effect:
            // --fmad=true --ftz=true --prec-div=false --prec-sqrt=false
            // (cudarc's use_fast_math field only adds --fmad=true)
            options: vec!["--use_fast_math".to_string()],
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(cuda_source, opts)
            .map_err(cuda_nvrtc_err)?;
        self.ctx.load_module(ptx).map_err(cuda_driver_err)
    }

    /// Copy host data to a new device buffer.
    pub fn htod_copy<T: DeviceRepr>(
        &self,
        host_data: &[T],
    ) -> Result<CudaSlice<T>, RuntimeError> {
        self.stream.clone_htod(host_data).map_err(cuda_driver_err)
    }

    /// Copy host data into an existing device buffer (no allocation).
    ///
    /// The destination buffer must have capacity >= source length.
    /// Used in the hot path to reuse pre-allocated scratch buffers.
    pub fn htod_copy_into<T: DeviceRepr>(
        &self,
        host_data: &[T],
        dst: &mut CudaSlice<T>,
    ) -> Result<(), RuntimeError> {
        self.stream
            .memcpy_htod(host_data, dst)
            .map_err(cuda_driver_err)
    }

    /// Allocate a zeroed device buffer of `len` elements.
    pub fn alloc_zeros<T: DeviceRepr + ValidAsZeroBits>(
        &self,
        len: usize,
    ) -> Result<CudaSlice<T>, RuntimeError> {
        self.stream.alloc_zeros(len).map_err(cuda_driver_err)
    }

    /// Copy device buffer contents back to host.
    pub fn dtoh_copy<T: DeviceRepr>(
        &self,
        device_buf: &CudaSlice<T>,
    ) -> Result<Vec<T>, RuntimeError> {
        self.stream
            .clone_dtoh(device_buf)
            .map_err(cuda_driver_err)
    }

    /// Synchronize the default stream (wait for all pending GPU work).
    pub fn synchronize(&self) -> Result<(), RuntimeError> {
        self.stream.synchronize().map_err(cuda_driver_err)
    }

    /// Set a user-managed cuBLAS workspace buffer.
    ///
    /// Required for CUDA graph capture: cuBLAS must NOT allocate memory
    /// internally during graph capture (cudaMalloc is forbidden on a
    /// capturing stream). By providing a pre-allocated workspace, cuBLAS
    /// uses it instead of allocating on-the-fly.
    ///
    /// 4 MB is sufficient for GEMV (N=1) operations on A100. cuBLAS uses
    /// at most ~1 MB for GEMV workspace; 4 MB provides headroom.
    ///
    /// # Safety
    ///
    /// The workspace buffer must remain valid for the lifetime of the
    /// cuBLAS handle (i.e., the lifetime of this CudaDevice).
    /// Set the L2 cache persistence window for a device buffer.
    ///
    /// Configures the CUDA stream's L2 access policy to keep the specified
    /// memory region in the L2 persistent partition. On A100, the L2 cache
    /// is 40 MB with a configurable persistent partition.
    ///
    /// Used for the Q8_1 pre-quantized input buffer in dp4a matvec:
    /// - The Q8_1 buffer is 4-16 KB (tiny relative to 40 MB L2)
    /// - Every thread block reads the entire buffer during matvec
    /// - Without persistence, weight read traffic (64 MB+) evicts the input
    /// - With persistence, the input stays in L2 across all thread blocks
    ///
    /// Falls back silently if the API is unavailable (pre-Ampere GPUs).
    ///
    /// # Safety
    ///
    /// The buffer must remain allocated for the lifetime of the stream attribute.
    /// The attribute persists until explicitly reset or the stream is destroyed.
    pub fn set_l2_persistence(
        &self,
        buffer: &CudaSlice<u8>,
    ) -> Result<(), RuntimeError> {
        use cudarc::driver::sys as cuda_sys;
        use cudarc::driver::DevicePtr;

        let (ptr, _sync) = buffer.device_ptr(&self.stream);
        let num_bytes = buffer.len();

        if num_bytes == 0 {
            return Ok(());
        }

        // Construct CUaccessPolicyWindow with the buffer as persistent.
        let window = cuda_sys::CUaccessPolicyWindow {
            base_ptr: ptr as *mut std::ffi::c_void,
            num_bytes,
            hitRatio: 1.0,
            hitProp: cuda_sys::CUaccessProperty::CU_ACCESS_PROPERTY_PERSISTING,
            missProp: cuda_sys::CUaccessProperty::CU_ACCESS_PROPERTY_STREAMING,
        };

        // CUstreamAttrValue is a union whose first field is accessPolicyWindow.
        // We construct a zeroed value and write the window into it.
        let mut attr_value: cuda_sys::CUstreamAttrValue = unsafe { std::mem::zeroed() };
        attr_value.accessPolicyWindow = window;

        // CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1 in all CUDA versions.
        // Use the type alias which maps to the correct enum for the compiled CUDA version.
        let attr_id = unsafe {
            std::mem::transmute::<u32, cuda_sys::CUstreamAttrID>(1u32)
        };

        let result = unsafe {
            cuda_sys::cuStreamSetAttribute(
                self.stream.cu_stream(),
                attr_id,
                &attr_value as *const cuda_sys::CUstreamAttrValue,
            )
        };

        if result != cuda_sys::CUresult::CUDA_SUCCESS {
            // Non-fatal: L2 persistence is an optimization hint, not required.
            // May fail on pre-Ampere GPUs or if the driver doesn't support it.
            eprintln!(
                "[CUDA] L2 persistence hint failed (status={result:?}, {} bytes) -- non-fatal",
                num_bytes
            );
        } else {
            eprintln!(
                "[CUDA] L2 persistence: {} bytes pinned for Q8_1 input buffer",
                num_bytes
            );
        }

        Ok(())
    }

    pub fn set_cublas_workspace(
        &self,
        workspace: &CudaSlice<u8>,
    ) -> Result<(), RuntimeError> {
        use cudarc::cublas::sys as cublas_sys;
        use cudarc::driver::DevicePtr;
        let (ptr, _sync) = workspace.device_ptr(&self.stream);
        let size_bytes = workspace.len(); // CudaSlice<u8>.len() = number of bytes
        // SAFETY: workspace is a valid device allocation of `size_bytes` bytes.
        // cuBLAS stores the pointer and size internally for subsequent operations.
        // The workspace buffer must outlive the cuBLAS handle (guaranteed by
        // storing it in MutableState alongside the CudaDevice).
        let status = unsafe {
            cublas_sys::cublasSetWorkspace_v2(
                *self.blas.handle(),
                ptr as *mut std::ffi::c_void,
                size_bytes,
            )
        };
        if status != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(RuntimeError::Compute(format!(
                "cublasSetWorkspace failed: status={status:?}",
            )));
        }
        Ok(())
    }
}

/// Convert a cudarc driver error to RuntimeError.
fn cuda_driver_err(e: cudarc::driver::DriverError) -> RuntimeError {
    RuntimeError::Compute(format!("CUDA driver error: {e}"))
}

/// Convert a cudarc NVRTC error to RuntimeError.
fn cuda_nvrtc_err(e: cudarc::nvrtc::CompileError) -> RuntimeError {
    RuntimeError::Compute(format!("CUDA NVRTC compilation error: {e}"))
}

/// Convert a cudarc cuBLAS error to RuntimeError.
fn cuda_cublas_err(e: cudarc::cublas::result::CublasError) -> RuntimeError {
    RuntimeError::Compute(format!("cuBLAS error: {e}"))
}
