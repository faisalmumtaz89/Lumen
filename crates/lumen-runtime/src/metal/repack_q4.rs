//! Runtime Q4_0 hot-weight repack into a Metal-friendly stripe layout.
//!
//! Port of the Q8_0 repack pattern (see `metal/repack_q8.rs`) to Q4_0.
//! Repacks Q4_0 weight tensors at load time from the default row-major block
//! layout into a stripe layout grouped by 32-row TILE_N tile. Pairs each
//! (row_group, k_block) of 32 rows × 32 K-elements as:
//!
//! ```text
//!   bytes [  0.. 64]: 32 × f16 scales (one per row in the row-group)
//!   bytes [ 64..576]: 32 × 16-byte qdata (row-major within the row-group)
//! ```
//!
//! Total bytes per (row_group, k_block) = 64 + 512 = 576 = 32 × 18 (same as
//! the AoS layout it replaces -- this is a pure transposition of bytes within
//! the per-row-group block, NOT an expansion of storage).
//!
//! ## Why this layout
//!
//! Just like the Q8 case, the current `dequant_tiled_matmul_q4_0_k64_residual_batched`
//! kernel in `gemm_q4.msl:1006` reads the B (weight) tile via 4 threads per
//! row-block. Each thread independently fetches the same 2-byte f16 scale --
//! 4× redundant scale loads per row-block. With SoA, the 32 scales for a
//! row-group sit on a contiguous 64-byte segment and the 32 16-byte qdata
//! rows sit on a contiguous 512-byte segment. Apple AGX coalesces same-line
//! loads regardless, but the more compact stripe layout makes the matrix-
//! multiply tile access pattern dramatically friendlier: a 32×32 row-group
//! is a contiguous 576-byte stripe vs. 32 scattered 18-byte rows.
//!
//! ## Pair-packed gate+up
//!
//! For the fused gate+up+SwiGLU kernel, both gate and up tensors share the
//! same K iteration but are stored in two separate buffers. By interleaving
//! them into a single buffer of `[gate_scales | up_scales | gate_qdata | up_qdata]`
//! per (row_group, k_block), the fused kernel pulls adjacent gate/up bytes
//! from the same cache line.
//!
//! Layout per paired (row_group, k_block):
//! ```text
//!   bytes [   0..  64]: gate scales (32 × f16)
//!   bytes [  64.. 128]: up scales   (32 × f16)
//!   bytes [ 128.. 640]: gate qdata  (32 × 16)
//!   bytes [ 640..1152]: up qdata    (32 × 16)
//! ```
//!
//! Total bytes per pair = 2 × 576 = 1152.
//!
//! ## Bit-identical algorithm
//!
//! The repacked kernels MUST produce mathematically identical output to their
//! AoS counterparts -- only the memory access pattern changes. Every byte
//! that was in the source Q4 tensor is preserved in the repacked buffer
//! (no precision loss, no requantization). The de-interleaved nibble layout
//! within each 16-byte qdata row is preserved verbatim (low nibbles for
//! elements 0..15 in the low 4 bits, high nibbles for elements 16..31 in the
//! high 4 bits — same as the on-disk Q4_0 spec).
//!
//! ## Env gating
//!
//! Repack itself is opt-in via `LUMEN_METAL_Q4_REPACKED=1`. When OFF, no
//! extra buffers are allocated -- the existing buffer path is unchanged.

use super::ffi::{MetalBuffer, MetalDevice};
use crate::error::RuntimeError;

/// Q4_0 block size: 32 elements per block.
pub(crate) const Q4_GROUP_SIZE: usize = 32;
/// Q4_0 block bytes: 2 bytes scale (f16) + 16 bytes qdata (32 nibbles).
pub(crate) const Q4_BLOCK_SIZE: usize = 18;
/// TILE_N: 32 output rows per tile (matches kernel constant).
pub(crate) const TILE_N: usize = 32;

/// Per-(row_group, k_block) byte count in the single-tensor stripe layout.
/// 64 (scales) + 512 (qdata) = 576 = 32 × 18 (bit-preserving transposition).
pub(crate) const STRIPE_BYTES_SINGLE: usize = 576;
/// Per-(row_group, k_block) byte count in the paired gate+up stripe layout.
/// 2 × 576 = 1152.
pub(crate) const STRIPE_BYTES_PAIR: usize = 1152;

/// Repack a Q4_0 tensor of shape `[N, K_bytes]` into the stripe SoA layout
/// described in the module docstring.
///
/// # Arguments
/// * `src` — source Q4_0 row-major bytes. Length = N × (K/32) × 18.
/// * `n_rows` — number of output rows (N). MUST be a multiple of `TILE_N` (32).
/// * `k_elems` — K dimension in elements. MUST be a multiple of `Q4_GROUP_SIZE` (32).
///
/// # Returns
/// A `Vec<u8>` of the same length as `src`, with bytes rearranged.
///
/// # Layout invariants
///
/// For each row_group `rg` in `0..n_rows/32`, for each k_block `kb` in
/// `0..k_elems/32`:
///   `dst[rg * stripe_stride + kb * 576]`:
///     bytes [0..64]   = 32 scales (1 per row in the row-group, 2 bytes each)
///     bytes [64..576] = 32 × 16 qdata bytes
///
/// where `stripe_stride = (k_elems / 32) * 576`.
///
/// # Errors
/// Returns `RuntimeError::Compute` if alignment requirements are not met or
/// the source byte length doesn't match `n_rows * (k_elems / 32) * 18`.
pub(crate) fn repack_q4_single(
    src: &[u8],
    n_rows: usize,
    k_elems: usize,
) -> Result<Vec<u8>, RuntimeError> {
    if n_rows % TILE_N != 0 {
        return Err(RuntimeError::Compute(format!(
            "repack_q4_single: n_rows ({}) must be a multiple of TILE_N ({})",
            n_rows, TILE_N
        )));
    }
    if k_elems % Q4_GROUP_SIZE != 0 {
        return Err(RuntimeError::Compute(format!(
            "repack_q4_single: k_elems ({}) must be a multiple of Q4_GROUP_SIZE ({})",
            k_elems, Q4_GROUP_SIZE
        )));
    }
    let num_blocks_per_row = k_elems / Q4_GROUP_SIZE;
    let row_bytes = num_blocks_per_row * Q4_BLOCK_SIZE;
    let expected_len = n_rows * row_bytes;
    if src.len() != expected_len {
        return Err(RuntimeError::Compute(format!(
            "repack_q4_single: src len {} != expected {} (n_rows={}, k_elems={})",
            src.len(),
            expected_len,
            n_rows,
            k_elems
        )));
    }

    let num_row_groups = n_rows / TILE_N;
    let stripe_stride: usize = num_blocks_per_row * STRIPE_BYTES_SINGLE;
    let total_out = num_row_groups * stripe_stride;
    debug_assert_eq!(
        total_out, expected_len,
        "Total bytes after repack must equal source length (pure transposition)"
    );

    let mut dst = vec![0u8; total_out];

    for rg in 0..num_row_groups {
        let dst_rg_base = rg * stripe_stride;
        for kb in 0..num_blocks_per_row {
            let dst_kb_base = dst_rg_base + kb * STRIPE_BYTES_SINGLE;
            // Scales region: 32 × f16 = 64 bytes
            // Qdata region: 32 × 16 = 512 bytes
            let dst_scales = dst_kb_base;
            let dst_qdata = dst_kb_base + 64;

            for r in 0..TILE_N {
                let row = rg * TILE_N + r;
                let src_block_off = row * row_bytes + kb * Q4_BLOCK_SIZE;
                // Scale: 2 bytes at the start of the block
                dst[dst_scales + r * 2] = src[src_block_off];
                dst[dst_scales + r * 2 + 1] = src[src_block_off + 1];
                // Qdata: 16 bytes after the scale (32 de-interleaved nibbles)
                let dst_row_qdata = dst_qdata + r * 16;
                dst[dst_row_qdata..dst_row_qdata + 16]
                    .copy_from_slice(&src[src_block_off + 2..src_block_off + 2 + 16]);
            }
        }
    }

    Ok(dst)
}

/// Repack two Q4_0 tensors (gate and up) into a single pair-packed buffer.
///
/// Both inputs MUST have the same shape `[N, K_bytes]`. The output layout
/// per (row_group, k_block):
/// ```text
///   bytes [   0..  64]: gate scales (32 × f16)
///   bytes [  64.. 128]: up scales   (32 × f16)
///   bytes [ 128.. 640]: gate qdata  (32 × 16)
///   bytes [ 640..1152]: up qdata    (32 × 16)
/// ```
///
/// # Returns
/// A `Vec<u8>` of length `2 * src_gate.len()` (= `2 * src_up.len()`).
pub(crate) fn repack_q4_pair_gate_up(
    src_gate: &[u8],
    src_up: &[u8],
    n_rows: usize,
    k_elems: usize,
) -> Result<Vec<u8>, RuntimeError> {
    if src_gate.len() != src_up.len() {
        return Err(RuntimeError::Compute(format!(
            "repack_q4_pair_gate_up: gate len {} != up len {}",
            src_gate.len(),
            src_up.len()
        )));
    }
    if n_rows % TILE_N != 0 {
        return Err(RuntimeError::Compute(format!(
            "repack_q4_pair_gate_up: n_rows ({}) must be a multiple of TILE_N ({})",
            n_rows, TILE_N
        )));
    }
    if k_elems % Q4_GROUP_SIZE != 0 {
        return Err(RuntimeError::Compute(format!(
            "repack_q4_pair_gate_up: k_elems ({}) must be a multiple of Q4_GROUP_SIZE ({})",
            k_elems, Q4_GROUP_SIZE
        )));
    }
    let num_blocks_per_row = k_elems / Q4_GROUP_SIZE;
    let row_bytes = num_blocks_per_row * Q4_BLOCK_SIZE;
    let expected_len = n_rows * row_bytes;
    if src_gate.len() != expected_len {
        return Err(RuntimeError::Compute(format!(
            "repack_q4_pair_gate_up: gate len {} != expected {} (n_rows={}, k_elems={})",
            src_gate.len(),
            expected_len,
            n_rows,
            k_elems
        )));
    }

    let num_row_groups = n_rows / TILE_N;
    let stripe_stride: usize = num_blocks_per_row * STRIPE_BYTES_PAIR;
    let total_out = num_row_groups * stripe_stride;
    debug_assert_eq!(
        total_out,
        2 * expected_len,
        "Pair-packed total bytes must equal 2× source length"
    );

    let mut dst = vec![0u8; total_out];

    for rg in 0..num_row_groups {
        let dst_rg_base = rg * stripe_stride;
        for kb in 0..num_blocks_per_row {
            let dst_kb_base = dst_rg_base + kb * STRIPE_BYTES_PAIR;
            let dst_gate_scales = dst_kb_base;
            let dst_up_scales = dst_kb_base + 64;
            let dst_gate_qdata = dst_kb_base + 128;
            let dst_up_qdata = dst_kb_base + 640;

            for r in 0..TILE_N {
                let row = rg * TILE_N + r;
                let src_block_off = row * row_bytes + kb * Q4_BLOCK_SIZE;
                // Gate scale + qdata
                dst[dst_gate_scales + r * 2] = src_gate[src_block_off];
                dst[dst_gate_scales + r * 2 + 1] = src_gate[src_block_off + 1];
                dst[dst_gate_qdata + r * 16..dst_gate_qdata + r * 16 + 16]
                    .copy_from_slice(&src_gate[src_block_off + 2..src_block_off + 2 + 16]);
                // Up scale + qdata
                dst[dst_up_scales + r * 2] = src_up[src_block_off];
                dst[dst_up_scales + r * 2 + 1] = src_up[src_block_off + 1];
                dst[dst_up_qdata + r * 16..dst_up_qdata + r * 16 + 16]
                    .copy_from_slice(&src_up[src_block_off + 2..src_block_off + 2 + 16]);
            }
        }
    }

    Ok(dst)
}

/// Build a Metal buffer from a single repacked Q4_0 tensor.
pub(crate) fn build_repacked_buffer_single(
    device: &MetalDevice,
    src: &[u8],
    n_rows: usize,
    k_elems: usize,
) -> Result<MetalBuffer, RuntimeError> {
    let dst = repack_q4_single(src, n_rows, k_elems)?;
    device.new_buffer_with_bytes(&dst).ok_or_else(|| {
        RuntimeError::Compute(format!(
            "Failed to allocate Q4 repacked single buffer ({} bytes, n_rows={}, k_elems={})",
            dst.len(),
            n_rows,
            k_elems
        ))
    })
}

/// Build a Metal buffer from a paired (gate, up) repacked Q4_0 tensor.
pub(crate) fn build_repacked_buffer_pair(
    device: &MetalDevice,
    src_gate: &[u8],
    src_up: &[u8],
    n_rows: usize,
    k_elems: usize,
) -> Result<MetalBuffer, RuntimeError> {
    let dst = repack_q4_pair_gate_up(src_gate, src_up, n_rows, k_elems)?;
    device.new_buffer_with_bytes(&dst).ok_or_else(|| {
        RuntimeError::Compute(format!(
            "Failed to allocate Q4 repacked pair buffer ({} bytes, n_rows={}, k_elems={})",
            dst.len(),
            n_rows,
            k_elems
        ))
    })
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)] // crate-internal repack helpers intentionally follow the tests
mod tests {
    use super::*;

    /// Build a deterministic Q4_0 byte buffer of shape [n_rows, k_elems/32 * 18].
    /// Each block has 2 scale-bytes set to (row, kb) and 16 qdata bytes set to
    /// `(row + kb * 32 + i) % 256` for i in 0..16. Nibble values are arbitrary
    /// for these tests; only byte-identity through the repack is checked.
    fn make_test_q4(n_rows: usize, k_elems: usize) -> Vec<u8> {
        let num_blocks_per_row = k_elems / Q4_GROUP_SIZE;
        let row_bytes = num_blocks_per_row * Q4_BLOCK_SIZE;
        let mut buf = vec![0u8; n_rows * row_bytes];
        for row in 0..n_rows {
            for kb in 0..num_blocks_per_row {
                let off = row * row_bytes + kb * Q4_BLOCK_SIZE;
                // Distinct scale per (row, kb): low byte = row, high byte = kb.
                buf[off] = (row % 256) as u8;
                buf[off + 1] = (kb % 256) as u8;
                for i in 0..16 {
                    buf[off + 2 + i] = ((row + kb * 32 + i) % 256) as u8;
                }
            }
        }
        buf
    }

    #[test]
    fn repack_single_roundtrip_bit_identical() {
        // Choose dimensions that map cleanly: n=64 (2 row-groups), k=64 (2 blocks)
        let n_rows = 64;
        let k_elems = 64;
        let src = make_test_q4(n_rows, k_elems);
        let repacked = repack_q4_single(&src, n_rows, k_elems).expect("repack_q4_single");

        // Verify total length
        assert_eq!(
            repacked.len(),
            src.len(),
            "repack must preserve total bytes"
        );

        // Verify: for each (rg, kb), the scales are 32 contiguous f16s
        // and qdata is 32 contiguous 16-byte rows.
        let num_blocks_per_row = k_elems / Q4_GROUP_SIZE;
        let row_bytes = num_blocks_per_row * Q4_BLOCK_SIZE;
        let num_row_groups = n_rows / TILE_N;
        let stripe_stride = num_blocks_per_row * STRIPE_BYTES_SINGLE;

        for rg in 0..num_row_groups {
            for kb in 0..num_blocks_per_row {
                let dst_kb_base = rg * stripe_stride + kb * STRIPE_BYTES_SINGLE;
                for r in 0..TILE_N {
                    let row = rg * TILE_N + r;
                    let src_block_off = row * row_bytes + kb * Q4_BLOCK_SIZE;
                    // Scale at dst_kb_base + r*2 must match src[src_block_off..src_block_off+2]
                    assert_eq!(
                        &repacked[dst_kb_base + r * 2..dst_kb_base + r * 2 + 2],
                        &src[src_block_off..src_block_off + 2],
                        "scale mismatch rg={} kb={} r={} (row={})",
                        rg,
                        kb,
                        r,
                        row
                    );
                    // qdata at dst_kb_base + 64 + r*16 must match src[src_block_off+2..src_block_off+18]
                    let dst_qd = dst_kb_base + 64 + r * 16;
                    assert_eq!(
                        &repacked[dst_qd..dst_qd + 16],
                        &src[src_block_off + 2..src_block_off + 18],
                        "qdata mismatch rg={} kb={} r={} (row={})",
                        rg,
                        kb,
                        r,
                        row
                    );
                }
            }
        }
    }

    #[test]
    fn repack_pair_roundtrip_bit_identical() {
        let n_rows = 64;
        let k_elems = 64;
        let src_gate = make_test_q4(n_rows, k_elems);
        let mut src_up = make_test_q4(n_rows, k_elems);
        // Mutate src_up so it differs from src_gate
        for b in src_up.iter_mut() {
            *b = b.wrapping_add(1);
        }

        let repacked = repack_q4_pair_gate_up(&src_gate, &src_up, n_rows, k_elems)
            .expect("repack_q4_pair_gate_up");
        assert_eq!(repacked.len(), 2 * src_gate.len());

        let num_blocks_per_row = k_elems / Q4_GROUP_SIZE;
        let row_bytes = num_blocks_per_row * Q4_BLOCK_SIZE;
        let num_row_groups = n_rows / TILE_N;
        let stripe_stride = num_blocks_per_row * STRIPE_BYTES_PAIR;

        for rg in 0..num_row_groups {
            for kb in 0..num_blocks_per_row {
                let dst_kb_base = rg * stripe_stride + kb * STRIPE_BYTES_PAIR;
                for r in 0..TILE_N {
                    let row = rg * TILE_N + r;
                    let src_block_off = row * row_bytes + kb * Q4_BLOCK_SIZE;

                    // Gate scale
                    assert_eq!(
                        &repacked[dst_kb_base + r * 2..dst_kb_base + r * 2 + 2],
                        &src_gate[src_block_off..src_block_off + 2],
                        "gate scale mismatch rg={} kb={} r={}",
                        rg,
                        kb,
                        r
                    );
                    // Up scale (offset 64)
                    assert_eq!(
                        &repacked[dst_kb_base + 64 + r * 2..dst_kb_base + 64 + r * 2 + 2],
                        &src_up[src_block_off..src_block_off + 2],
                        "up scale mismatch rg={} kb={} r={}",
                        rg,
                        kb,
                        r
                    );
                    // Gate qdata (offset 128)
                    let dst_gd = dst_kb_base + 128 + r * 16;
                    assert_eq!(
                        &repacked[dst_gd..dst_gd + 16],
                        &src_gate[src_block_off + 2..src_block_off + 18],
                        "gate qdata mismatch rg={} kb={} r={}",
                        rg,
                        kb,
                        r
                    );
                    // Up qdata (offset 640)
                    let dst_ud = dst_kb_base + 640 + r * 16;
                    assert_eq!(
                        &repacked[dst_ud..dst_ud + 16],
                        &src_up[src_block_off + 2..src_block_off + 18],
                        "up qdata mismatch rg={} kb={} r={}",
                        rg,
                        kb,
                        r
                    );
                }
            }
        }
    }

    #[test]
    fn repack_qwen35_ffn_down_shape_q4() {
        // FFN-down: N = 4096 (hidden_dim), K = 12288 (inter_dim).
        // Truncated test: 32 rows × 128 elements to keep test fast but exercise
        // the shape arithmetic (4 k-blocks per row).
        let n_rows = 32;
        let k_elems = 128;
        let src = make_test_q4(n_rows, k_elems);
        let repacked = repack_q4_single(&src, n_rows, k_elems).expect("repack_q4_single");
        assert_eq!(repacked.len(), src.len());
        // Sanity: scale of (row=0, kb=0) at repacked[0..2] equals src[0..2]
        assert_eq!(&repacked[0..2], &src[0..2]);
        // Scale of (row=31, kb=0) at repacked[62..64] equals src[31*row_bytes..+2]
        let num_blocks_per_row = k_elems / Q4_GROUP_SIZE;
        let row_bytes = num_blocks_per_row * Q4_BLOCK_SIZE;
        let src_off_row31_kb0 = 31 * row_bytes;
        assert_eq!(
            &repacked[62..64],
            &src[src_off_row31_kb0..src_off_row31_kb0 + 2]
        );
    }

    #[test]
    fn repack_q4_rejects_misaligned_n() {
        let src = vec![0u8; 100];
        let err = repack_q4_single(&src, 30, 32).expect_err("must reject n=30");
        assert!(format!("{:?}", err).contains("n_rows"));
    }

    #[test]
    fn repack_q4_rejects_misaligned_k() {
        let src = vec![0u8; 100];
        let err = repack_q4_single(&src, 32, 30).expect_err("must reject k=30");
        assert!(format!("{:?}", err).contains("k_elems"));
    }

    #[test]
    fn qmv_decode_repack_roundtrip() {
        // 1 row, 1 block (32 weights). GGUF de-interleaved: byte j -> val[j] lo, val[j+16] hi.
        let sf16 = super::f32_to_f16_bits(0.5);
        let mut src = vec![0u8; 18];
        src[0..2].copy_from_slice(&sf16.to_le_bytes());
        let vals: Vec<u8> = (0..32).map(|i| (i % 15) as u8).collect();
        for j in 0..16 {
            src[2 + j] = vals[j] | (vals[j + 16] << 4);
        }
        let (qw, sc) = super::repack_q4_qmv_decode(&src, 1, 32).unwrap();
        let s = f32::from_le_bytes([sc[0], sc[1], sc[2], sc[3]]);
        assert!((s - 0.5).abs() < 1e-3);
        // sequential packing: byte m holds val[2m] | val[2m+1]<<4
        for m in 0..16 {
            assert_eq!(qw[m] & 0x0F, vals[2 * m], "lo nibble byte {}", m);
            assert_eq!(qw[m] >> 4, vals[2 * m + 1], "hi nibble byte {}", m);
        }
    }

    /// Re-quantizing Q8_0 -> Q4_0 then dequantizing (kernel math: (nibble-8)*scale)
    /// must approximate the original values within Q4 granularity, and the output
    /// length must be exactly (n_elems/32)*18 GGUF Q4_0 bytes.
    #[test]
    fn requant_q8_to_q4_roundtrip_approx() {
        // Build two Q8_0 blocks (64 elements). Block 0: a ramp; block 1: a spike.
        let n_elems = 64usize;
        let s0 = super::f32_to_f16_bits(0.1); // q8 scale block 0
        let s1 = super::f32_to_f16_bits(0.5); // q8 scale block 1
        let mut q8 = Vec::new();
        // Block 0
        q8.extend_from_slice(&s0.to_le_bytes());
        let mut orig = [0.0f32; 64];
        for i in 0..32 {
            let qi = (i as i32 - 16) as i8; // [-16, 15]
            q8.push(qi as u8);
            orig[i] = (qi as f32) * super::f16_to_f32(s0);
        }
        // Block 1
        q8.extend_from_slice(&s1.to_le_bytes());
        for i in 0..32 {
            let qi: i8 = if i == 7 { 100 } else { (i as i8) - 4 };
            q8.push(qi as u8);
            orig[32 + i] = (qi as f32) * super::f16_to_f32(s1);
        }

        let q4 = super::requant_q8_0_to_q4_0(&q8, n_elems).expect("requant");
        assert_eq!(q4.len(), (n_elems / 32) * 18, "Q4_0 byte length");

        // Dequantize each Q4_0 block with the SAME math the kernel uses:
        // value = scale * (nibble - 8); de-interleaved nibble layout.
        for b in 0..(n_elems / 32) {
            let off = b * 18;
            let scale = super::f16_to_f32(u16::from_le_bytes([q4[off], q4[off + 1]]));
            // Reconstruct max abs of the original block to bound the tolerance.
            let mut amax = 0.0f32;
            for i in 0..32 {
                amax = amax.max(orig[b * 32 + i].abs());
            }
            // Q4 step = scale (== amax/7). Round-trip error <= step (0.5 step round
            // + the Q8->Q4 dequant is exact at Q8 granularity). Use 1.01*step slack.
            let tol = scale * 1.01 + 1e-6;
            for i in 0..16 {
                let byte = q4[off + 2 + i];
                let lo = (byte & 0x0F) as i32 - 8;
                let hi = (byte >> 4) as i32 - 8;
                let dq_lo = scale * lo as f32;
                let dq_hi = scale * hi as f32;
                assert!(
                    (dq_lo - orig[b * 32 + i]).abs() <= tol,
                    "block {} elem {} lo: dq {} vs orig {} tol {}",
                    b,
                    i,
                    dq_lo,
                    orig[b * 32 + i],
                    tol
                );
                assert!(
                    (dq_hi - orig[b * 32 + i + 16]).abs() <= tol,
                    "block {} elem {} hi: dq {} vs orig {} tol {}",
                    b,
                    i,
                    dq_hi,
                    orig[b * 32 + i + 16],
                    tol
                );
            }
        }
    }

    /// requant must reject element counts that are not a multiple of 32 and
    /// sources that are too short.
    #[test]
    fn requant_q8_to_q4_rejects_bad_input() {
        let err = super::requant_q8_0_to_q4_0(&[0u8; 34], 30).expect_err("non-mult-32");
        assert!(format!("{:?}", err).contains("multiple of 32"));
        // 64 elems needs 2*34 = 68 bytes; give 34.
        let err2 = super::requant_q8_0_to_q4_0(&[0u8; 34], 64).expect_err("too short");
        assert!(format!("{:?}", err2).contains("too small"));
    }
}

/// IEEE-754 half -> f32. Standard bit expansion (no `half` crate dependency).
#[inline]
pub(crate) fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 1;
    let exp = (bits >> 10) & 0x1f;
    let mant = bits & 0x3ff;
    let out: u32 = if exp == 0 {
        if mant == 0 {
            (sign as u32) << 31
        } else {
            let mut e: i32 = -14;
            let mut m = mant as u32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            ((sign as u32) << 31) | (((e + 127) as u32) << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        ((sign as u32) << 31) | (0xff << 23) | ((mant as u32) << 13)
    } else {
        ((sign as u32) << 31) | (((exp as i32 - 15 + 127) as u32) << 23) | ((mant as u32) << 13)
    };
    f32::from_bits(out)
}

/// f32 -> IEEE-754 half bits with round-to-nearest-even and proper subnormal
/// handling. Byte-for-byte matches the converter's `f32_to_f16_bits_convert`
/// (lumen-convert::dequant), so a Q4_0 block produced by `requant_q8_0_to_q4_0`
/// here is bit-identical to one produced by the offline converter. Used to write
/// the per-block f16 scale when re-quantizing the lm_head Q8_0 -> Q4_0.
#[inline]
pub(crate) fn f32_to_f16_bits_rne(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7F_FFFF;

    if exp == 255 {
        // Inf / NaN
        let f16_frac = if frac != 0 { 0x200 } else { 0 };
        return ((sign << 15) | (0x1F << 10) | f16_frac) as u16;
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        // Overflow -> Inf
        return ((sign << 15) | (0x1F << 10)) as u16;
    }
    if new_exp <= 0 {
        // f16 subnormal range. Below the smallest representable subnormal
        // (2^-24) is true underflow -> signed zero.
        if new_exp < -10 {
            return (sign << 15) as u16;
        }
        let mantissa = frac | 0x0080_0000; // restore implicit leading 1 (24-bit)
        let shift = (14 - new_exp) as u32; // new_exp in [-10, 0] -> shift in [14, 24]
        let f16_frac = mantissa >> shift; // truncation, consistent with normal path
        return ((sign << 15) | (f16_frac & 0x3FF)) as u16;
    }

    let f16_frac = frac >> 13;
    ((sign << 15) | ((new_exp as u32) << 10) | f16_frac) as u16
}

/// Re-quantize a Q8_0 weight tensor (row-major GGUF blocks, 34 bytes per
/// 32-element block: 2-byte f16 scale + 32 i8) into a Q8 lm_head -> standard GGUF
/// Q4_0 (row-major, 18 bytes per block: 2-byte f16 scale + 16 de-interleaved
/// nibbles). The Q8->Q4 path is a deliberate, validated precision tradeoff used
/// only for the optional Q4 lm_head (`LUMEN_METAL_Q4_QMV_LMHEAD`).
///
/// Each Q8_0 block is dequantized `value = f16_scale * i8` (exact GGUF Q8_0
/// semantics), then re-quantized with the canonical GGUF Q4_0 convention used
/// across Lumen (`scale = amax / 7`, clamped to the f16 max normal 65504;
/// `nibble = round(v / scale) + 8`, clamped to [0, 15]; de-interleaved packing
/// vals[0..16] -> low nibbles, vals[16..32] -> high nibbles). Producing canonical
/// GGUF Q4_0 lets the standard `repack_q4_qmv_decode` consume the result with the
/// same `s*(dot - 8*sumx)` zero-point fold the kernel already implements.
///
/// `total_elems` is the full element count (n_rows * k). Returns Q4_0 bytes of
/// length `(total_elems / 32) * 18`.
///
/// # Errors
/// `RuntimeError::Compute` if `total_elems % 32 != 0` or `src` is shorter than
/// `(total_elems / 32) * 34`.
pub(crate) fn requant_q8_0_to_q4_0(
    src: &[u8],
    total_elems: usize,
) -> Result<Vec<u8>, RuntimeError> {
    const Q8_BLOCK: usize = 34; // 2 (f16 scale) + 32 (i8)
    if total_elems % 32 != 0 {
        return Err(RuntimeError::Compute(format!(
            "requant_q8_0_to_q4_0: total_elems {} not a multiple of 32",
            total_elems
        )));
    }
    let num_blocks = total_elems / 32;
    let need = num_blocks * Q8_BLOCK;
    if src.len() < need {
        return Err(RuntimeError::Compute(format!(
            "requant_q8_0_to_q4_0: src too small {} < {} ({} blocks)",
            src.len(),
            need,
            num_blocks
        )));
    }

    let mut out = vec![0u8; num_blocks * Q4_BLOCK_SIZE];
    for b in 0..num_blocks {
        let blk = b * Q8_BLOCK;
        let q8_scale = f16_to_f32(u16::from_le_bytes([src[blk], src[blk + 1]]));
        // Dequantize the 32 i8 quants -> f32, tracking abs-max for the Q4 scale.
        let mut vals = [0.0f32; 32];
        let mut amax = 0.0f32;
        for i in 0..32 {
            let v = (src[blk + 2 + i] as i8 as f32) * q8_scale;
            vals[i] = v;
            let a = v.abs();
            if a > amax {
                amax = a;
            }
        }
        // Q4_0 scale: map [-amax, amax] -> [-8, 7]. amax==0 -> any nonzero scale
        // (all nibbles become 8 -> dequant 0). Clamp to f16 max normal so the
        // stored f16 scale never overflows to Inf.
        let scale = if amax == 0.0 {
            1.0
        } else {
            (amax / 7.0).min(65504.0)
        };
        let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };

        let out_off = b * Q4_BLOCK_SIZE;
        out[out_off..out_off + 2].copy_from_slice(&f32_to_f16_bits_rne(scale).to_le_bytes());
        // De-interleaved GGUF Q4_0 packing: byte m holds val[m] (lo) | val[m+16] (hi).
        for m in 0..16 {
            let q_lo = ((vals[m] * inv_scale).round() as i32 + 8).clamp(0, 15) as u8;
            let q_hi = ((vals[m + 16] * inv_scale).round() as i32 + 8).clamp(0, 15) as u8;
            out[out_off + 2 + m] = q_lo | (q_hi << 4);
        }
    }
    Ok(out)
}

/// Build a standalone **native GGUF Q4_0 block layout** weight buffer for a
/// per-layer projection (the GDN `ssm_out`) by RE-QUANTIZING its Q8_0 weights to
/// Q4_0. `q8_src` is the raw Q8_0 tensor bytes `[n_rows, k]` (row-major); the
/// result is `n_rows*(k/32)` 18-byte Q4_0 blocks (2-byte f16 scale + 16
/// de-interleaved nibble bytes) laid out exactly as the fused NR2 matvec kernels
/// (`dequant_matmul_q4_0_*_nr2`) expect at buffer(0). Distinct from
/// [`build_qmv_decode_buffers_from_q8`] (which produces the *qmv* sequential-nibble
/// layout for the `qmv_q4_0_*` kernels); this keeps the on-disk GGUF block layout
/// so the existing Q8 NR2 ssm_out dispatch can swap in the Q4 weight + Q4 kernel
/// with no other change. (`LUMEN_METAL_Q4_SSMOUT_NR2`.)
pub(crate) fn build_nr2_q4_buffer_from_q8(
    device: &MetalDevice,
    q8_src: &[u8],
    n_rows: usize,
    k: usize,
) -> Result<MetalBuffer, RuntimeError> {
    let q4 = requant_q8_0_to_q4_0(q8_src, n_rows * k)?;
    device.new_buffer_with_bytes(&q4).ok_or_else(|| {
        RuntimeError::Compute(format!(
            "Failed to allocate Q4 NR2 ssm_out buffer ({} bytes, n_rows={}, k={})",
            q4.len(),
            n_rows,
            k
        ))
    })
}

/// Build the decode-qmv (qweights, scales) buffers for the lm_head / output
/// projection by RE-QUANTIZING its Q8_0 weights to Q4_0 first. `q8_src` is the
/// raw Q8_0 output_proj bytes [vocab, hidden]; `n_rows = vocab`, `k = hidden`.
/// Equivalent to `build_qmv_decode_buffers(requant_q8_0_to_q4_0(...), n_rows, k)`.
pub(crate) fn build_qmv_lmhead_buffers_from_q8(
    device: &MetalDevice,
    q8_src: &[u8],
    n_rows: usize,
    k: usize,
) -> Result<(MetalBuffer, MetalBuffer), RuntimeError> {
    let q4 = requant_q8_0_to_q4_0(q8_src, n_rows * k)?;
    build_qmv_decode_buffers(device, &q4, n_rows, k)
}

/// Build the decode-qmv (qweights, scales) buffers for a per-layer projection
/// (e.g. the GDN `ssm_out`) by RE-QUANTIZING its Q8_0 weights to Q4_0 first.
/// `q8_src` is the raw Q8_0 tensor bytes `[n_rows, k]` (row-major), `n_rows`/`k`
/// the matvec out/in dims. Identical to [`build_qmv_lmhead_buffers_from_q8`] but
/// named for the per-layer projection use; the Q4 scales are emitted as f32.
pub(crate) fn build_qmv_decode_buffers_from_q8(
    device: &MetalDevice,
    q8_src: &[u8],
    n_rows: usize,
    k: usize,
) -> Result<(MetalBuffer, MetalBuffer), RuntimeError> {
    let q4 = requant_q8_0_to_q4_0(q8_src, n_rows * k)?;
    build_qmv_decode_buffers(device, &q4, n_rows, k)
}

/// F16-scale variant of [`build_qmv_decode_buffers_from_q8`]: re-quantizes the
/// Q8_0 tensor to Q4_0, then builds the decode-qmv buffers with the per-block
/// scales emitted as **f16** (2 B/block) instead of f32 (4 B). The Q4_0 scale
/// produced by requant is natively f16-representable (the f16sc repack copies the
/// on-disk f16 scale bytes verbatim), so the f16sc kernel is BYTE-IDENTICAL to the
/// f32-scale kernel on the same re-quantized weights. Shaves ~10% off this matvec's
/// weight stream on top of the Q8->Q4 halving.
#[allow(dead_code)] // dormant builder: f16sc Q8->Q4 path not yet wired into a default codepath
pub(crate) fn build_qmv_decode_buffers_from_q8_f16sc(
    device: &MetalDevice,
    q8_src: &[u8],
    n_rows: usize,
    k: usize,
) -> Result<(MetalBuffer, MetalBuffer), RuntimeError> {
    let q4 = requant_q8_0_to_q4_0(q8_src, n_rows * k)?;
    build_qmv_decode_buffers_f16sc(device, &q4, n_rows, k)
}

/// F16-SCALES variant of [`build_qmv_lmhead_buffers_from_q8`]: re-quantizes the
/// Q8_0 output_proj to Q4_0, then builds the decode-qmv buffers with the per-block
/// scales emitted as **f16** (2 B/block) instead of f32 (4 B). The Q4_0 scale
/// produced by requant is natively f16-representable (the f16sc repack copies the
/// on-disk f16 scale bytes verbatim), so the f16sc lm_head kernel
/// `qmv_q4_0_rmsnorm_f16sc` is BYTE-IDENTICAL to the f32-scale `qmv_q4_0_rmsnorm`
/// on the same re-quantized weights. Shaves ~10% off the lm_head weight stream.
pub(crate) fn build_qmv_lmhead_buffers_from_q8_f16sc(
    device: &MetalDevice,
    q8_src: &[u8],
    n_rows: usize,
    k: usize,
) -> Result<(MetalBuffer, MetalBuffer), RuntimeError> {
    let q4 = requant_q8_0_to_q4_0(q8_src, n_rows * k)?;
    build_qmv_decode_buffers_f16sc(device, &q4, n_rows, k)
}

/// f32 -> IEEE-754 half bits (test helper only).
#[cfg(test)]
pub(crate) fn f32_to_f16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let mant = (b >> 13) & 0x3ff;
    if exp <= 0 {
        sign
    } else if exp >= 0x1f {
        sign | 0x7c00
    } else {
        sign | ((exp as u16) << 10) | (mant as u16)
    }
}

/// Repack one Q4_0 tensor `[n_rows, k]` into the MLX-style decode-qmv layout:
/// two SEPARATE row-major buffers consumed by `qmv_q4_0_residual`:
///   - qweights: `[n_rows, k/2]` bytes. Nibbles RE-ORDERED from GGUF de-interleaved
///     (byte j -> value[j] lo, value[j+16] hi) to SEQUENTIAL (byte m -> value[2m] lo,
///     value[2m+1] hi), so a uint16 load yields 4 consecutive values for the kernel.
///   - scales:   `[n_rows, k/32]` f32 (converted from the GGUF per-block f16 scale).
/// The kernel folds the Q4_0 symmetric zero-point as `s*(dot - 8*sum_x)`, so no
/// bias buffer is needed. Bit-preserving (no requantization).
pub(crate) fn repack_q4_qmv_decode(
    src: &[u8],
    n_rows: usize,
    k: usize,
) -> Result<(Vec<u8>, Vec<u8>), RuntimeError> {
    if k % 32 != 0 {
        return Err(RuntimeError::Compute(format!(
            "qmv decode repack: k={} not a multiple of 32",
            k
        )));
    }
    let blocks_per_row = k / 32;
    let row_bytes = blocks_per_row * Q4_BLOCK_SIZE;
    let need = n_rows * row_bytes;
    if src.len() < need {
        return Err(RuntimeError::Compute(format!(
            "qmv decode repack: src too small {} < {} (n_rows={}, k={})",
            src.len(),
            need,
            n_rows,
            k
        )));
    }
    let mut qweights = vec![0u8; n_rows * (k / 2)];
    let mut scales = vec![0u8; n_rows * blocks_per_row * 4];
    for row in 0..n_rows {
        for b in 0..blocks_per_row {
            let blk = row * row_bytes + b * Q4_BLOCK_SIZE;
            let sf16 = u16::from_le_bytes([src[blk], src[blk + 1]]);
            let sf32 = f16_to_f32(sf16);
            let s_off = (row * blocks_per_row + b) * 4;
            scales[s_off..s_off + 4].copy_from_slice(&sf32.to_le_bytes());
            let q = &src[blk + 2..blk + 18]; // 16 qdata bytes (32 de-interleaved nibbles)
            let qw_off = (row * blocks_per_row + b) * 16;
            for m in 0..16 {
                let lo_i = 2 * m;
                let hi_i = 2 * m + 1;
                let lo = if lo_i < 16 {
                    q[lo_i] & 0x0F
                } else {
                    q[lo_i - 16] >> 4
                };
                let hi = if hi_i < 16 {
                    q[hi_i] & 0x0F
                } else {
                    q[hi_i - 16] >> 4
                };
                qweights[qw_off + m] = lo | (hi << 4);
            }
        }
    }
    Ok((qweights, scales))
}

/// Build the two decode-qmv Metal buffers (qweights, scales) for one Q4_0 tensor.
pub(crate) fn build_qmv_decode_buffers(
    device: &MetalDevice,
    src: &[u8],
    n_rows: usize,
    k: usize,
) -> Result<(MetalBuffer, MetalBuffer), RuntimeError> {
    let (qw, sc) = repack_q4_qmv_decode(src, n_rows, k)?;
    let qbuf = device
        .new_buffer_with_bytes(&qw)
        .ok_or_else(|| RuntimeError::Compute("qmv decode: alloc qweights buffer failed".into()))?;
    let sbuf = device
        .new_buffer_with_bytes(&sc)
        .ok_or_else(|| RuntimeError::Compute("qmv decode: alloc scales buffer failed".into()))?;
    Ok((qbuf, sbuf))
}

/// Repack one Q4_0 tensor `[n_rows, k]` into the decode-qmv layout BUT with the
/// per-32-block scales emitted as **f16** (2 bytes) instead of f32 (4 bytes).
///
/// Identical nibble re-ordering to [`repack_q4_qmv_decode`] (same sequential
/// `uint16`-friendly qweight bytes); the ONLY difference is the scale buffer:
///   - scales: `[n_rows, k/32]` f16 — the GGUF per-block scale bytes copied
///     VERBATIM (the on-disk Q4_0 scale is already f16, so this is bit-perfect
///     to the source — NO requant, NO f16->f32->f16 round-trip).
///
/// This shaves the qmv weight stream from 20 -> 18 bytes per 32-value block
/// (16 nibble bytes + 2 scale bytes), a ~10% bytes-moved reduction on the
/// bandwidth-bound decode matvec. Consumed by the `*_f16sc` qmv kernel variants
/// which read `device const half*` scales.
pub(crate) fn repack_q4_qmv_decode_f16sc(
    src: &[u8],
    n_rows: usize,
    k: usize,
) -> Result<(Vec<u8>, Vec<u8>), RuntimeError> {
    if k % 32 != 0 {
        return Err(RuntimeError::Compute(format!(
            "qmv decode f16sc repack: k={} not a multiple of 32",
            k
        )));
    }
    let blocks_per_row = k / 32;
    let row_bytes = blocks_per_row * Q4_BLOCK_SIZE;
    let need = n_rows * row_bytes;
    if src.len() < need {
        return Err(RuntimeError::Compute(format!(
            "qmv decode f16sc repack: src too small {} < {} (n_rows={}, k={})",
            src.len(),
            need,
            n_rows,
            k
        )));
    }
    let mut qweights = vec![0u8; n_rows * (k / 2)];
    let mut scales = vec![0u8; n_rows * blocks_per_row * 2]; // f16: 2 bytes/block
    for row in 0..n_rows {
        for b in 0..blocks_per_row {
            let blk = row * row_bytes + b * Q4_BLOCK_SIZE;
            // Copy the source f16 scale bytes VERBATIM (bit-perfect, no requant).
            let s_off = (row * blocks_per_row + b) * 2;
            scales[s_off] = src[blk];
            scales[s_off + 1] = src[blk + 1];
            let q = &src[blk + 2..blk + 18]; // 16 qdata bytes (32 de-interleaved nibbles)
            let qw_off = (row * blocks_per_row + b) * 16;
            for m in 0..16 {
                let lo_i = 2 * m;
                let hi_i = 2 * m + 1;
                let lo = if lo_i < 16 {
                    q[lo_i] & 0x0F
                } else {
                    q[lo_i - 16] >> 4
                };
                let hi = if hi_i < 16 {
                    q[hi_i] & 0x0F
                } else {
                    q[hi_i - 16] >> 4
                };
                qweights[qw_off + m] = lo | (hi << 4);
            }
        }
    }
    Ok((qweights, scales))
}

/// Build the two decode-qmv Metal buffers (qweights, **f16 scales**) for one
/// Q4_0 tensor. See [`repack_q4_qmv_decode_f16sc`].
pub(crate) fn build_qmv_decode_buffers_f16sc(
    device: &MetalDevice,
    src: &[u8],
    n_rows: usize,
    k: usize,
) -> Result<(MetalBuffer, MetalBuffer), RuntimeError> {
    let (qw, sc) = repack_q4_qmv_decode_f16sc(src, n_rows, k)?;
    let qbuf = device.new_buffer_with_bytes(&qw).ok_or_else(|| {
        RuntimeError::Compute("qmv decode f16sc: alloc qweights buffer failed".into())
    })?;
    let sbuf = device.new_buffer_with_bytes(&sc).ok_or_else(|| {
        RuntimeError::Compute("qmv decode f16sc: alloc scales buffer failed".into())
    })?;
    Ok((qbuf, sbuf))
}

/// INTERLEAVED gate+up decode-qmv repack (env LUMEN_METAL_Q4_GATEUP_IL).
///
/// Co-resides the gate and up weights of the dense FFN gate/up matvec into ONE
/// nibble buffer + ONE f16-scale buffer, so the matvec reads TWO contiguous
/// streams instead of FOUR (gate-nibbles, up-nibbles, gate-scale, up-scale). The
/// nibble re-ordering of EACH tensor is byte-for-byte the SAME as
/// [`repack_q4_qmv_decode_f16sc`]; only the placement interleaves gate and up at
/// the 256-byte super-iteration-stripe granularity (one full simdgroup's
/// 512-value block = 256 nibble bytes).
///
/// Layout (both gate `g` and up `u` are `[out_rows, k]` Q4_0; `SI = k / 512`
/// super-iterations per row, `k % 512 == 0` required so the qmv block_size=512
/// tiling is exact):
///   - qweights `[out_rows, k]` bytes (gate k/2 + up k/2 = k bytes/row):
///       per row, per super-iter j: `[256 B gate stripe j][256 B up stripe j]`
///   - scales `[out_rows, SI, 64]` bytes (16 gate f16 + 16 up f16 per super-iter):
///       per row, per super-iter j: `[16 gate f16 (32 B)][16 up f16 (32 B)]`
/// Consumed by `qmv_q4_0_gate_up_swiglu_il`, which strides gate at `j*512`, up at
/// `j*512+256`, gate-scale at `j*64`, up-scale at `j*64+32`. BYTE-IDENTICAL math.
#[allow(clippy::type_complexity)]
pub(crate) fn repack_q4_gate_up_interleaved(
    gate_src: &[u8],
    up_src: &[u8],
    n_rows: usize,
    k: usize,
) -> Result<(Vec<u8>, Vec<u8>), RuntimeError> {
    if k % 512 != 0 {
        return Err(RuntimeError::Compute(format!(
            "qmv gate/up interleaved repack: k={} not a multiple of 512",
            k
        )));
    }
    // Re-order each tensor's nibbles exactly as the f16sc path does, then weave.
    let (gate_qw, gate_sc) = repack_q4_qmv_decode_f16sc(gate_src, n_rows, k)?;
    let (up_qw, up_sc) = repack_q4_qmv_decode_f16sc(up_src, n_rows, k)?;
    let si = k / 512; // super-iterations per row (each = 512 values = 256 nibble bytes)
    let row_qw = k / 2; // bytes per row in the single-tensor f16sc qweight layout
    let row_sc = (k / 32) * 2; // f16 scale bytes per row (k/32 blocks x 2 B)

    let mut qweights = vec![0u8; n_rows * k]; // gate k/2 + up k/2 per row
    let mut scales = vec![0u8; n_rows * si * 64]; // 32 gate + 32 up f16 bytes per super-iter
    for row in 0..n_rows {
        for j in 0..si {
            // --- nibbles: 256 B gate stripe j, then 256 B up stripe j ---
            let dst = (row * si + j) * 512;
            let g0 = row * row_qw + j * 256;
            let u0 = row * row_qw + j * 256;
            qweights[dst..dst + 256].copy_from_slice(&gate_qw[g0..g0 + 256]);
            qweights[dst + 256..dst + 512].copy_from_slice(&up_qw[u0..u0 + 256]);
            // --- scales: 16 gate f16 (32 B), then 16 up f16 (32 B) ---
            // 512 values = 16 blocks of 32 -> 16 scales each, 2 B/scale.
            let sdst = (row * si + j) * 64;
            let gs0 = row * row_sc + j * 32; // 16 blocks x 2 B
            let us0 = row * row_sc + j * 32;
            scales[sdst..sdst + 32].copy_from_slice(&gate_sc[gs0..gs0 + 32]);
            scales[sdst + 32..sdst + 64].copy_from_slice(&up_sc[us0..us0 + 32]);
        }
    }
    Ok((qweights, scales))
}

/// Build the two INTERLEAVED gate+up decode-qmv Metal buffers (one packed nibble
/// buffer, one packed f16-scale buffer). See [`repack_q4_gate_up_interleaved`].
pub(crate) fn build_qmv_gate_up_interleaved_buffers(
    device: &MetalDevice,
    gate_src: &[u8],
    up_src: &[u8],
    n_rows: usize,
    k: usize,
) -> Result<(MetalBuffer, MetalBuffer), RuntimeError> {
    let (qw, sc) = repack_q4_gate_up_interleaved(gate_src, up_src, n_rows, k)?;
    let qbuf = device.new_buffer_with_bytes(&qw).ok_or_else(|| {
        RuntimeError::Compute("qmv gate/up IL: alloc qweights buffer failed".into())
    })?;
    let sbuf = device.new_buffer_with_bytes(&sc).ok_or_else(|| {
        RuntimeError::Compute("qmv gate/up IL: alloc scales buffer failed".into())
    })?;
    Ok((qbuf, sbuf))
}
