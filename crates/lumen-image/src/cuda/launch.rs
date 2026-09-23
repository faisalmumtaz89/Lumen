//! Launchers for the image ops, and the buffer helper they share.

use cudarc::driver::PushKernelArg;
use lumen_runtime::cuda::ffi::CudaDevice;
use lumen_runtime::error::RuntimeError;

use super::ImageKernels;

/// A device buffer plus its length, so a caller cannot confuse the two.
pub struct DevVec {
    pub buf: cudarc::driver::CudaSlice<f32>,
    pub len: usize,
}

/// Copy a host slice to the device.
pub fn upload(dev: &CudaDevice, data: &[f32]) -> Result<DevVec, RuntimeError> {
    Ok(DevVec {
        buf: dev.htod_copy(data)?,
        len: data.len(),
    })
}

/// Copy a device buffer back, after synchronising.
pub fn download(dev: &CudaDevice, v: &DevVec) -> Result<Vec<f32>, RuntimeError> {
    let out = dev.dtoh_copy(&v.buf)?;
    dev.synchronize()?;
    Ok(out)
}

/// Allocate a device buffer of `len` floats for a kernel output.
///
/// The contents are unspecified until the kernel writes them: every launcher
/// that takes one of these covers the whole buffer, so a memset first would be
/// a second full pass over memory that the kernel overwrites anyway. A kernel
/// that leaves an element unwritten leaves it unspecified rather than zero.
pub fn alloc(dev: &CudaDevice, len: usize) -> Result<DevVec, RuntimeError> {
    // Safety: any bit pattern is a valid f32, and the callers write every
    // element before reading the buffer.
    let buf = unsafe { dev.alloc_uninit::<f32>(len)? };
    Ok(DevVec { buf, len })
}

fn cfg(grid: (u32, u32, u32), block: (u32, u32, u32), smem: u32) -> cudarc::driver::LaunchConfig {
    cudarc::driver::LaunchConfig {
        grid_dim: grid,
        block_dim: block,
        shared_mem_bytes: smem,
    }
}

/// LayerNorm with no affine parameters over each row.
pub fn layernorm_noaffine(
    dev: &CudaDevice,
    k: &ImageKernels,
    x: &DevVec,
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<DevVec, RuntimeError> {
    let out = alloc(dev, x.len)?;
    let (ru, du) = (rows as u32, dim as u32);
    // Safety: buffers match the kernel's declared shapes and geometry.
    unsafe {
        dev.stream
            .launch_builder(&k.layernorm_noaffine)
            .arg(&x.buf)
            .arg(&out.buf)
            .arg(&ru)
            .arg(&du)
            .arg(&eps)
            .launch(cfg((rows as u32, 1, 1), (256, 1, 1), 0))
            .map_err(|e| RuntimeError::Compute(format!("layernorm_noaffine: {e}")))?;
    }
    Ok(out)
}

/// `out = x * (1 + scale)`, broadcasting the scale over rows.
pub fn scale_one_plus(
    dev: &CudaDevice,
    k: &ImageKernels,
    x: &DevVec,
    scale: &DevVec,
    dim: usize,
) -> Result<DevVec, RuntimeError> {
    let out = alloc(dev, x.len)?;
    let total = x.len as u32;
    let du = dim as u32;
    let threads = 256u32;
    // Safety: buffers match the kernel's declared shapes and geometry.
    unsafe {
        dev.stream
            .launch_builder(&k.scale_one_plus)
            .arg(&x.buf)
            .arg(&scale.buf)
            .arg(&out.buf)
            .arg(&total)
            .arg(&du)
            .launch(cfg((total.div_ceil(threads), 1, 1), (threads, 1, 1), 0))
            .map_err(|e| RuntimeError::Compute(format!("scale_one_plus: {e}")))?;
    }
    Ok(out)
}

/// The DiT's complex RoPE, in place on `q`, pinned to a host rotation by
/// `cuda-ops-check`; the forward uses the fused `head_norm_rope`.
pub fn rope_complex(
    dev: &CudaDevice,
    k: &ImageKernels,
    q: &mut DevVec,
    freqs: &DevVec,
    seq: usize,
    heads: usize,
    head_dim: usize,
) -> Result<(), RuntimeError> {
    let threads = 256u32;
    let pairs = (head_dim / 2) as u32;
    let total = seq as u32 * heads as u32 * pairs;
    let (su, hu, hdu) = (seq as u32, heads as u32, head_dim as u32);
    // Safety: buffers match the kernel's declared shapes and geometry.
    unsafe {
        dev.stream
            .launch_builder(&k.rope_complex)
            .arg(&q.buf)
            .arg(&freqs.buf)
            .arg(&su)
            .arg(&hu)
            .arg(&hdu)
            .launch(cfg((total.div_ceil(threads), 1, 1), (threads, 1, 1), 0))
            .map_err(|e| RuntimeError::Compute(format!("rope_complex: {e}")))?;
    }
    Ok(())
}

/// Block-causal attention. `image_id` is -1 at text positions; `key_valid` is 1
/// at attendable keys. Both are passed explicitly — the caller supplies an
/// all-ones `key_valid` when nothing is padded, which keeps one code path.
#[allow(clippy::too_many_arguments)]
pub fn attn_block_causal(
    dev: &CudaDevice,
    k: &ImageKernels,
    q: &DevVec,
    kk: &DevVec,
    v: &DevVec,
    image_id: &[i32],
    key_valid: &[i32],
    seq: usize,
    heads: usize,
    head_dim: usize,
) -> Result<DevVec, RuntimeError> {
    let out = alloc(dev, q.len)?;
    let ids = dev.htod_copy(image_id)?;
    let kv = dev.htod_copy(key_valid)?;
    let (su, hu, hdu) = (seq as u32, heads as u32, head_dim as u32);
    let smem = (2 * head_dim * 4) as u32;
    // Safety: buffers match the kernel's declared shapes and geometry.
    unsafe {
        dev.stream
            .launch_builder(&k.attn_block_causal)
            .arg(&q.buf)
            .arg(&kk.buf)
            .arg(&v.buf)
            .arg(&out.buf)
            .arg(&ids)
            .arg(&kv)
            .arg(&su)
            .arg(&hu)
            .arg(&hdu)
            .launch(cfg(
                (seq as u32, heads as u32, 1),
                (head_dim as u32, 1, 1),
                smem,
            ))
            .map_err(|e| RuntimeError::Compute(format!("attn_block_causal: {e}")))?;
    }
    Ok(out)
}
