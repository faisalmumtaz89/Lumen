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
            src.len(), expected_len, n_rows, k_elems
        )));
    }

    let num_row_groups = n_rows / TILE_N;
    let stripe_stride: usize = num_blocks_per_row * STRIPE_BYTES_SINGLE;
    let total_out = num_row_groups * stripe_stride;
    debug_assert_eq!(total_out, expected_len,
        "Total bytes after repack must equal source length (pure transposition)");

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
            src_gate.len(), src_up.len()
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
            src_gate.len(), expected_len, n_rows, k_elems
        )));
    }

    let num_row_groups = n_rows / TILE_N;
    let stripe_stride: usize = num_blocks_per_row * STRIPE_BYTES_PAIR;
    let total_out = num_row_groups * stripe_stride;
    debug_assert_eq!(total_out, 2 * expected_len,
        "Pair-packed total bytes must equal 2× source length");

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
            dst.len(), n_rows, k_elems
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
            dst.len(), n_rows, k_elems
        ))
    })
}

#[cfg(test)]
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
        assert_eq!(repacked.len(), src.len(), "repack must preserve total bytes");

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
                        rg, kb, r, row
                    );
                    // qdata at dst_kb_base + 64 + r*16 must match src[src_block_off+2..src_block_off+18]
                    let dst_qd = dst_kb_base + 64 + r * 16;
                    assert_eq!(
                        &repacked[dst_qd..dst_qd + 16],
                        &src[src_block_off + 2..src_block_off + 18],
                        "qdata mismatch rg={} kb={} r={} (row={})",
                        rg, kb, r, row
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
                        "gate scale mismatch rg={} kb={} r={}", rg, kb, r
                    );
                    // Up scale (offset 64)
                    assert_eq!(
                        &repacked[dst_kb_base + 64 + r * 2..dst_kb_base + 64 + r * 2 + 2],
                        &src_up[src_block_off..src_block_off + 2],
                        "up scale mismatch rg={} kb={} r={}", rg, kb, r
                    );
                    // Gate qdata (offset 128)
                    let dst_gd = dst_kb_base + 128 + r * 16;
                    assert_eq!(
                        &repacked[dst_gd..dst_gd + 16],
                        &src_gate[src_block_off + 2..src_block_off + 18],
                        "gate qdata mismatch rg={} kb={} r={}", rg, kb, r
                    );
                    // Up qdata (offset 640)
                    let dst_ud = dst_kb_base + 640 + r * 16;
                    assert_eq!(
                        &repacked[dst_ud..dst_ud + 16],
                        &src_up[src_block_off + 2..src_block_off + 18],
                        "up qdata mismatch rg={} kb={} r={}", rg, kb, r
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
        assert_eq!(&repacked[62..64], &src[src_off_row31_kb0..src_off_row31_kb0 + 2]);
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
}
