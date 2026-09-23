//! CUDA execution of the image pipeline's ops.
//!
//! The CPU reference in the parent crate stays the source of truth. The
//! elementwise ops compute the same thing in the same order per output element
//! and are held to it within 1e-6 relative error, the row norms and the
//! f32 attention within 1e-5 (`cuda-ops-check`);
//! the projections and the attention run on bf16 tensor cores, as the
//! reference model does, and are held at bf16 tolerance and end-to-end on
//! the generated image.
//!
//! The kernels live in the `.cu` files beside this module and are compiled at
//! runtime by NVRTC, the way `lumen-runtime` compiles its own.

use std::sync::Arc;

use lumen_runtime::cuda::ffi::CudaDevice;
use lumen_runtime::error::RuntimeError;

pub mod attention;
pub mod blas;
pub mod dit_gpu;
pub mod launch;
pub mod text_gpu;
pub mod vae_gpu;

/// The image-specific kernel source, compiled by NVRTC at load.
pub const IMAGE_OPS_SOURCE: &str = include_str!("image_ops.cu");

/// The text tower's kernel source, compiled by NVRTC at load.
///
/// The tower's bf16 elementwise, norm and rotary kernels, in a source
/// of their own because NVRTC compiles each source into its own module; its
/// projections and attention use the shared cuBLAS and flash paths.
pub const TEXT_OPS_SOURCE: &str = include_str!("text_ops.cu");

/// The fused attention kernel's source. Its own module because it targets
/// `compute_80`: the bf16 `mma.sync`, `ldmatrix` and `cp.async` it is built on
/// need it, and the other sources keep NVRTC's default target.
pub const FLASH_ATTN_SOURCE: &str = include_str!("flash_attn.cu");

/// Kernels this module launches.
pub struct ImageKernels {
    pub layernorm_noaffine: cudarc::driver::CudaFunction,
    pub scale_one_plus: cudarc::driver::CudaFunction,
    pub rope_complex: cudarc::driver::CudaFunction,
    pub mrope_interleaved: cudarc::driver::CudaFunction,
    /// `attn_block_causal`, the per-query reference `cuda-ops-check` runs.
    pub attn_block_causal: cudarc::driver::CudaFunction,
    /// `f32_to_bf16_bits`, nearest-even.
    pub f32_to_bf16_bits: cudarc::driver::CudaFunction,
    /// `mask_softmax_rows`, the masked softmax from f32 scores to bf16
    /// probabilities, for the unfused attention `cuda-ops-check` compares
    /// the fused kernel against.
    pub mask_softmax_rows: cudarc::driver::CudaFunction,
    /// `flash_attn_bf16`, the fused attention over one 128-wide head.
    pub flash_attn: cudarc::driver::CudaFunction,
}

impl ImageKernels {
    /// Compile `image_ops.cu` for the device and resolve each entry point.
    pub fn load(device: &CudaDevice) -> Result<Self, RuntimeError> {
        let module: Arc<_> = device.compile_and_load(IMAGE_OPS_SOURCE)?;
        let get = |name: &str| -> Result<cudarc::driver::CudaFunction, RuntimeError> {
            module
                .load_function(name)
                .map_err(|e| RuntimeError::Compute(format!("load {name}: {e}")))
        };
        Ok(Self {
            layernorm_noaffine: get("layernorm_noaffine")?,
            scale_one_plus: get("scale_one_plus")?,
            rope_complex: get("rope_complex")?,
            mrope_interleaved: get("mrope_interleaved")?,
            attn_block_causal: get("attn_block_causal")?,
            f32_to_bf16_bits: get("f32_to_bf16_bits")?,
            mask_softmax_rows: get("mask_softmax_rows")?,
            flash_attn: device
                .compile_and_load_with_arch(FLASH_ATTN_SOURCE, "compute_80")?
                .load_function("flash_attn_bf16")
                .map_err(|e| RuntimeError::Compute(format!("load flash_attn_bf16: {e}")))?,
        })
    }
}
