//! Weight views the native prefill reads beside the decode planes, built at load for an admitted
//! model (`native_prefill::admit`).
//!
//! - **Swizzled MLP block scales.** cuBLASLt's block-scaled NVFP4 GEMM reads each UE4M3 block scale in
//!   tiles of 128 rows x 4 scale columns (512 bytes, row-major over tiles), while the planes and the
//!   decode kernels keep them linear (`[row][k / 16]`). Each MLP matrix gets a swizzled copy.
//! - **BF16 `in_proj_a` / `in_proj_b`.** One `[2 * 48][hidden]` BF16 copy per GDN layer, the `a` rows
//!   first. Exact, because admission (Q6) refused any value BF16 cannot hold.
//! - **Activation scales.** Projections that share an input share one activation scale, the maximum
//!   of their members' `input_scale`: GDN qkv and z, attention q, k and v, MLP gate and up. The FP8
//!   ones are also kept on the device, where the GEMMs read them by pointer.

use super::ffi::CudaDevice;
use super::native_prefill::{input_scale, is_attention_layer, SliceSource};
use crate::error::RuntimeError;
use cudarc::driver::CudaSlice;
use lumen_format::index::{SubtensorOffsets, TensorSlice};
use lumen_format::Nvfp4Planes;

/// Byte offset of block scale `(row, blk)` in the swizzled layout of a matrix with `blk_cols`
/// scale columns.
pub fn swizzled_offset(row: usize, blk: usize, blk_cols: usize) -> usize {
    let tiles_c = blk_cols.div_ceil(4);
    ((row / 128) * tiles_c + blk / 4) * 512 + (row % 32) * 16 + (row % 128 / 32) * 4 + blk % 4
}

/// Bytes of the swizzled layout: whole 128 x 4 tiles.
pub fn swizzled_len(rows: usize, blk_cols: usize) -> usize {
    rows.div_ceil(128) * blk_cols.div_ceil(4) * 512
}

/// The swizzled copy of `rows x blk_cols` linear block scales; the padding of partial tiles is zero.
pub fn swizzle_block_scales(linear: &[u8], rows: usize, blk_cols: usize) -> Vec<u8> {
    assert_eq!(
        linear.len(),
        rows * blk_cols,
        "linear block-scale plane size"
    );
    let mut out = vec![0u8; swizzled_len(rows, blk_cols)];
    for (row, scales) in linear.chunks_exact(blk_cols).enumerate() {
        for (blk, &s) in scales.iter().enumerate() {
            out[swizzled_offset(row, blk, blk_cols)] = s;
        }
    }
    out
}

/// The BF16 bits of F32 values BF16 holds exactly (the high halves).
pub fn exact_bf16(f32_bytes: &[u8]) -> Vec<u16> {
    f32_bytes
        .chunks_exact(4)
        .map(|c| u16::from_le_bytes([c[2], c[3]]))
        .collect()
}

/// The activation scales of one layer's GEMM inputs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerScales {
    /// FP8 input projections: max over GDN qkv and z, or attention q, k and v.
    pub proj_in: f32,
    /// FP8 output projection (GDN out_proj or attention o_proj).
    pub proj_out: f32,
    /// NVFP4 gate and up: the max over both.
    pub gate_up: f32,
    /// NVFP4 down.
    pub down: f32,
}

/// The views of an admitted model.
pub struct PrefillWeightViews {
    /// Per layer: the swizzled block scales of gate, up and down.
    pub mlp_scales: Vec<[CudaSlice<u8>; 3]>,
    /// Per layer: the BF16 `a` then `b` rows of a GDN layer; `None` on attention layers.
    pub ab: Vec<Option<CudaSlice<u16>>>,
    pub scales: Vec<LayerScales>,
    /// `[layer][proj_in, proj_out]`, read by the FP8 GEMMs as their activation-scale pointers.
    pub fp8_scales: CudaSlice<f32>,
}

impl PrefillWeightViews {
    /// Build the views of an admitted model of hidden size `hidden` and MLP size `inter`.
    pub fn build(
        device: &CudaDevice,
        hidden: usize,
        inter: usize,
        layers: &[SubtensorOffsets],
        src: &dyn SliceSource,
    ) -> Result<Self, RuntimeError> {
        let mut mlp_scales = Vec::with_capacity(layers.len());
        let mut ab = Vec::with_capacity(layers.len());
        let mut scales = Vec::with_capacity(layers.len());
        for (l, layer) in layers.iter().enumerate() {
            let swizzled = |slice: &TensorSlice, n: usize, k: usize| {
                let planes = Nvfp4Planes::for_shape(n as u64, k as u64)?;
                let linear = src.read(l, slice, planes.weight_bytes, planes.block_scale_bytes)?;
                device.htod_copy(&swizzle_block_scales(&linear, n, k / 16))
            };
            mlp_scales.push([
                swizzled(&layer.w_gate, inter, hidden)?,
                swizzled(&layer.w_up, inter, hidden)?,
                swizzled(&layer.w_down, hidden, inter)?,
            ]);
            let nvfp4 = |slice: &TensorSlice, n: usize, k: usize| {
                let planes = Nvfp4Planes::for_shape(n as u64, k as u64)?;
                input_scale(src, l, slice, planes.total_bytes())
            };
            let fp8 = |slice: &TensorSlice| {
                // An FP8 slice is `n * k` weight bytes, a weight scale and the activation scale.
                let planes_bytes = slice.length - lumen_format::PLANAR_INPUT_SCALE_BYTES;
                input_scale(src, l, slice, planes_bytes)
            };
            let gate_up =
                nvfp4(&layer.w_gate, inter, hidden)?.max(nvfp4(&layer.w_up, inter, hidden)?);
            let down = nvfp4(&layer.w_down, hidden, inter)?;
            let missing = |name: &str| {
                RuntimeError::Compute(format!(
                    "layer {l}: {name} is missing from an admitted model"
                ))
            };
            let (proj_in, proj_out) = if is_attention_layer(l) {
                (
                    fp8(&layer.wq)?.max(fp8(&layer.wk)?).max(fp8(&layer.wv)?),
                    fp8(&layer.wo)?,
                )
            } else {
                let z = layer
                    .attn_gate
                    .as_ref()
                    .ok_or_else(|| missing("attn_gate"))?;
                let out = layer.ssm_out.as_ref().ok_or_else(|| missing("ssm_out"))?;
                (fp8(&layer.wq)?.max(fp8(z)?), fp8(out)?)
            };
            scales.push(LayerScales {
                proj_in,
                proj_out,
                gate_up,
                down,
            });
            ab.push(match (&layer.ssm_alpha, &layer.ssm_beta) {
                (Some(a), Some(b)) => {
                    let mut bits = exact_bf16(&src.read(l, a, 0, a.length)?);
                    bits.extend(exact_bf16(&src.read(l, b, 0, b.length)?));
                    Some(device.htod_copy(&bits)?)
                }
                _ => None,
            });
        }
        let table: Vec<f32> = scales
            .iter()
            .flat_map(|s| [s.proj_in, s.proj_out])
            .collect();
        Ok(Self {
            mlp_scales,
            ab,
            scales,
            fp8_scales: device.htod_copy(&table)?,
        })
    }

    /// Device bytes the views hold.
    pub fn device_bytes(&self) -> usize {
        self.mlp_scales
            .iter()
            .flatten()
            .map(|s| s.len())
            .chain(self.ab.iter().flatten().map(|s| s.len() * 2))
            .sum::<usize>()
            + self.fp8_scales.len() * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The layout as cuBLASLt defines it for block-scaled operands: tiles of 128 rows x 4 scale
    /// columns (512 bytes), tiles row-major, and inside a tile the offset
    /// `(r % 32) * 16 + (r % 128 / 32) * 4 + c % 4`.
    fn reference_offset(row: usize, blk: usize, n_blk_cols: usize) -> usize {
        let tiles_c = (n_blk_cols + 3) / 4;
        ((row / 128) * tiles_c + blk / 4) * 512 + (row % 32) * 16 + (row % 128 / 32) * 4 + blk % 4
    }

    #[test]
    fn the_swizzle_is_the_reference_layout_and_a_bijection() {
        for (rows, blk_cols) in [(17408, 320), (5120, 1088), (2048, 320), (130, 6), (1, 1)] {
            let len = swizzled_len(rows, blk_cols);
            assert_eq!(len % 512, 0);
            let mut hits = vec![0u32; len];
            for row in 0..rows {
                for blk in 0..blk_cols {
                    let o = swizzled_offset(row, blk, blk_cols);
                    assert_eq!(o, reference_offset(row, blk, blk_cols), "({row}, {blk})");
                    hits[o] += 1;
                }
            }
            assert!(
                hits.iter().all(|&h| h <= 1),
                "{rows}x{blk_cols}: an offset used twice"
            );
            let used = hits.iter().filter(|&&h| h == 1).count();
            assert_eq!(used, rows * blk_cols, "{rows}x{blk_cols}");
            if rows % 128 == 0 && blk_cols % 4 == 0 {
                assert_eq!(used, len, "{rows}x{blk_cols} fills whole tiles");
            }
        }
    }

    #[test]
    fn swizzling_moves_every_scale_and_zeroes_the_padding() {
        let (rows, blk_cols) = (130, 6);
        let linear: Vec<u8> = (0..rows * blk_cols).map(|i| (i % 251 + 1) as u8).collect();
        let out = swizzle_block_scales(&linear, rows, blk_cols);
        for row in 0..rows {
            for blk in 0..blk_cols {
                assert_eq!(
                    out[reference_offset(row, blk, blk_cols)],
                    linear[row * blk_cols + blk]
                );
            }
        }
        assert_eq!(
            out.iter().filter(|&&b| b == 0).count(),
            out.len() - rows * blk_cols
        );
        // A linear copy is not the swizzled layout (the negative control of the GPU gate).
        assert_ne!(&out[..linear.len()], &linear[..]);
    }

    #[test]
    fn bf16_copies_keep_the_high_halves() {
        let values = [1.0f32, -2.5, 0.0, f32::from_bits(0x3F81_0000)];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let bits = exact_bf16(&bytes);
        for (v, b) in values.iter().zip(&bits) {
            assert_eq!(f32::from_bits((*b as u32) << 16), *v);
        }
    }
}
