//! Runtime Q8_0 hot-weight repack into a Metal-friendly stripe layout.
//!
//! Repacks Q8_0 weight tensors at load time from the default row-major block
//! layout into a stripe layout grouped by 32-row TILE_N tile. Pairs each
//! (row_group, k_block) of 32 rows × 32 K-elements as:
//!
//! ```text
//!   bytes [  0.. 64]: 32 × f16 scales (one per row in the row-group)
//!   bytes [ 64..1088]: 32 × 32-byte qdata (row-major within the row-group)
//! ```
//!
//! Total bytes per (row_group, k_block) = 64 + 1024 = 1088 = 32 × 34 (same as
//! the AoS layout it replaces -- this is a pure transposition of bytes within
//! the per-row-group block, NOT an expansion of storage).
//!
//! ## Why this layout
//!
//! The current `dequant_tiled_matmul_q8_0_k64_residual_batched` kernel in
//! `gemm_q8_0.msl:473` reads the B (weight) tile via 4 threads per row-block.
//! Each thread independently fetches the same 2-byte f16 scale -- 4× redundant
//! scale loads per row-block. With SoA, one thread can fetch the scale and the
//! others broadcast (or all 4 threads read from the same f16 in the same cache
//! line). The Apple AGX hardware coalesces same-line loads regardless, so the
//! win is more about memory layout efficiency for the matrix-multiply tile load
//! pattern: a 32×32 row-group is a contiguous 1088-byte stripe vs. 32 scattered
//! 34-byte rows.
//!
//! ## Pair-packed gate+up
//!
//! For the fused gate+up+SwiGLU kernel, both gate and up tensors share the same
//! K iteration but are stored in two separate buffers. By interleaving them
//! into a single buffer of `[gate_scales | up_scales | gate_qdata | up_qdata]`
//! per (row_group, k_block), the fused kernel pulls adjacent gate/up bytes
//! from the same cache line.
//!
//! Layout per paired (row_group, k_block):
//! ```text
//!   bytes [   0..  64]: gate scales (32 × f16)
//!   bytes [  64.. 128]: up scales   (32 × f16)
//!   bytes [ 128..1152]: gate qdata  (32 × 32)
//!   bytes [1152..2176]: up qdata    (32 × 32)
//! ```
//!
//! Total bytes per pair = 2 × 1088 = 2176.
//!
//! ## Bit-identical algorithm
//!
//! The repacked kernels MUST produce mathematically identical output to their
//! AoS counterparts -- only the memory access pattern changes. Every byte
//! that was in the source Q8 tensor is preserved in the repacked buffer
//! (no precision loss, no requantization).
//!
//! ## Env gating
//!
//! Repack itself is opt-in via `LUMEN_METAL_Q8_REPACKED=1`. When OFF, no
//! extra buffers are allocated -- the existing buffer path is unchanged.

use super::ffi::{MetalBuffer, MetalDevice};
use crate::error::RuntimeError;

/// Q8_0 block size: 32 elements per block.
pub(crate) const Q8B_GROUP_SIZE: usize = 32;
/// Q8_0 block bytes: 2 bytes scale (f16) + 32 bytes qdata.
pub(crate) const Q8B_BLOCK_SIZE: usize = 34;
/// TILE_N: 32 output rows per tile (matches kernel constant).
pub(crate) const TILE_N: usize = 32;

/// Repack a Q8_0 tensor of shape `[N, K_bytes]` into the stripe SoA layout
/// described in the module docstring.
///
/// # Arguments
/// * `src` — source Q8_0 row-major bytes. Length = N × (K/32) × 34.
/// * `n_rows` — number of output rows (N). MUST be a multiple of `TILE_N` (32).
/// * `k_elems` — K dimension in elements. MUST be a multiple of `Q8B_GROUP_SIZE` (32).
///
/// # Returns
/// A `Vec<u8>` of the same length as `src`, with bytes rearranged.
///
/// # Layout invariants
///
/// For each row_group `rg` in `0..n_rows/32`, for each k_block `kb` in
/// `0..k_elems/32`:
///   `dst[rg * stripe_stride + kb * 1088]`:
///     bytes [0..64]   = 32 scales (1 per row in the row-group)
///     bytes [64..1088] = 32 × 32 qdata bytes
///
/// where `stripe_stride = (k_elems / 32) * 1088`.
///
/// # Errors
/// Returns `RuntimeError::Compute` if alignment requirements are not met or
/// the source byte length doesn't match `n_rows * (k_elems / 32) * 34`.
pub(crate) fn repack_q8_single(
    src: &[u8],
    n_rows: usize,
    k_elems: usize,
) -> Result<Vec<u8>, RuntimeError> {
    if n_rows % TILE_N != 0 {
        return Err(RuntimeError::Compute(format!(
            "repack_q8_single: n_rows ({}) must be a multiple of TILE_N ({})",
            n_rows, TILE_N
        )));
    }
    if k_elems % Q8B_GROUP_SIZE != 0 {
        return Err(RuntimeError::Compute(format!(
            "repack_q8_single: k_elems ({}) must be a multiple of Q8B_GROUP_SIZE ({})",
            k_elems, Q8B_GROUP_SIZE
        )));
    }
    let num_blocks_per_row = k_elems / Q8B_GROUP_SIZE;
    let row_bytes = num_blocks_per_row * Q8B_BLOCK_SIZE;
    let expected_len = n_rows * row_bytes;
    if src.len() != expected_len {
        return Err(RuntimeError::Compute(format!(
            "repack_q8_single: src len {} != expected {} (n_rows={}, k_elems={})",
            src.len(), expected_len, n_rows, k_elems
        )));
    }

    let num_row_groups = n_rows / TILE_N;
    // Each (row_group, k_block) emits: 32 scales (64 bytes) + 32 × 32 qdata (1024 bytes) = 1088 bytes
    let stripe_bytes_per_kblock: usize = 64 + 1024; // 1088
    let stripe_stride: usize = num_blocks_per_row * stripe_bytes_per_kblock;
    let total_out = num_row_groups * stripe_stride;
    debug_assert_eq!(total_out, expected_len,
        "Total bytes after repack must equal source length (pure transposition)");

    let mut dst = vec![0u8; total_out];

    for rg in 0..num_row_groups {
        let dst_rg_base = rg * stripe_stride;
        for kb in 0..num_blocks_per_row {
            let dst_kb_base = dst_rg_base + kb * stripe_bytes_per_kblock;
            // Scales region: 32 × f16 = 64 bytes
            // qdata region: 32 × 32 = 1024 bytes
            let dst_scales = dst_kb_base;
            let dst_qdata = dst_kb_base + 64;

            for r in 0..TILE_N {
                let row = rg * TILE_N + r;
                let src_block_off = row * row_bytes + kb * Q8B_BLOCK_SIZE;
                // Scale: 2 bytes at the start of the block
                dst[dst_scales + r * 2] = src[src_block_off];
                dst[dst_scales + r * 2 + 1] = src[src_block_off + 1];
                // Qdata: 32 bytes after the scale
                let dst_row_qdata = dst_qdata + r * 32;
                dst[dst_row_qdata..dst_row_qdata + 32]
                    .copy_from_slice(&src[src_block_off + 2..src_block_off + 2 + 32]);
            }
        }
    }

    Ok(dst)
}

/// Repack two Q8_0 tensors (gate and up) into a single pair-packed buffer.
///
/// Both inputs MUST have the same shape `[N, K_bytes]`. The output layout
/// per (row_group, k_block):
/// ```text
///   bytes [   0..  64]: gate scales (32 × f16)
///   bytes [  64.. 128]: up scales   (32 × f16)
///   bytes [ 128..1152]: gate qdata  (32 × 32)
///   bytes [1152..2176]: up qdata    (32 × 32)
/// ```
///
/// # Returns
/// A `Vec<u8>` of length `2 * src_gate.len()` (= `2 * src_up.len()`).
pub(crate) fn repack_q8_pair_gate_up(
    src_gate: &[u8],
    src_up: &[u8],
    n_rows: usize,
    k_elems: usize,
) -> Result<Vec<u8>, RuntimeError> {
    if src_gate.len() != src_up.len() {
        return Err(RuntimeError::Compute(format!(
            "repack_q8_pair_gate_up: gate len {} != up len {}",
            src_gate.len(), src_up.len()
        )));
    }
    if n_rows % TILE_N != 0 {
        return Err(RuntimeError::Compute(format!(
            "repack_q8_pair_gate_up: n_rows ({}) must be a multiple of TILE_N ({})",
            n_rows, TILE_N
        )));
    }
    if k_elems % Q8B_GROUP_SIZE != 0 {
        return Err(RuntimeError::Compute(format!(
            "repack_q8_pair_gate_up: k_elems ({}) must be a multiple of Q8B_GROUP_SIZE ({})",
            k_elems, Q8B_GROUP_SIZE
        )));
    }
    let num_blocks_per_row = k_elems / Q8B_GROUP_SIZE;
    let row_bytes = num_blocks_per_row * Q8B_BLOCK_SIZE;
    let expected_len = n_rows * row_bytes;
    if src_gate.len() != expected_len {
        return Err(RuntimeError::Compute(format!(
            "repack_q8_pair_gate_up: gate len {} != expected {} (n_rows={}, k_elems={})",
            src_gate.len(), expected_len, n_rows, k_elems
        )));
    }

    let num_row_groups = n_rows / TILE_N;
    // Each (row_group, k_block) emits 2176 bytes:
    //  gate_scales(64) | up_scales(64) | gate_qdata(1024) | up_qdata(1024)
    let stripe_bytes_per_kblock: usize = 2 * (64 + 1024); // 2176
    let stripe_stride: usize = num_blocks_per_row * stripe_bytes_per_kblock;
    let total_out = num_row_groups * stripe_stride;
    debug_assert_eq!(total_out, 2 * expected_len,
        "Pair-packed total bytes must equal 2× source length");

    let mut dst = vec![0u8; total_out];

    for rg in 0..num_row_groups {
        let dst_rg_base = rg * stripe_stride;
        for kb in 0..num_blocks_per_row {
            let dst_kb_base = dst_rg_base + kb * stripe_bytes_per_kblock;
            let dst_gate_scales = dst_kb_base;
            let dst_up_scales = dst_kb_base + 64;
            let dst_gate_qdata = dst_kb_base + 128;
            let dst_up_qdata = dst_kb_base + 128 + 1024;

            for r in 0..TILE_N {
                let row = rg * TILE_N + r;
                let src_block_off = row * row_bytes + kb * Q8B_BLOCK_SIZE;
                // Gate scale + qdata
                dst[dst_gate_scales + r * 2] = src_gate[src_block_off];
                dst[dst_gate_scales + r * 2 + 1] = src_gate[src_block_off + 1];
                dst[dst_gate_qdata + r * 32..dst_gate_qdata + r * 32 + 32]
                    .copy_from_slice(&src_gate[src_block_off + 2..src_block_off + 2 + 32]);
                // Up scale + qdata
                dst[dst_up_scales + r * 2] = src_up[src_block_off];
                dst[dst_up_scales + r * 2 + 1] = src_up[src_block_off + 1];
                dst[dst_up_qdata + r * 32..dst_up_qdata + r * 32 + 32]
                    .copy_from_slice(&src_up[src_block_off + 2..src_block_off + 2 + 32]);
            }
        }
    }

    Ok(dst)
}

/// Build a Metal buffer from a single repacked Q8_0 tensor.
pub(crate) fn build_repacked_buffer_single(
    device: &MetalDevice,
    src: &[u8],
    n_rows: usize,
    k_elems: usize,
) -> Result<MetalBuffer, RuntimeError> {
    let dst = repack_q8_single(src, n_rows, k_elems)?;
    device.new_buffer_with_bytes(&dst).ok_or_else(|| {
        RuntimeError::Compute(format!(
            "Failed to allocate Q8 repacked single buffer ({} bytes, n_rows={}, k_elems={})",
            dst.len(), n_rows, k_elems
        ))
    })
}

/// Build a Metal buffer from a paired (gate, up) repacked Q8_0 tensor.
pub(crate) fn build_repacked_buffer_pair(
    device: &MetalDevice,
    src_gate: &[u8],
    src_up: &[u8],
    n_rows: usize,
    k_elems: usize,
) -> Result<MetalBuffer, RuntimeError> {
    let dst = repack_q8_pair_gate_up(src_gate, src_up, n_rows, k_elems)?;
    device.new_buffer_with_bytes(&dst).ok_or_else(|| {
        RuntimeError::Compute(format!(
            "Failed to allocate Q8 repacked pair buffer ({} bytes, n_rows={}, k_elems={})",
            dst.len(), n_rows, k_elems
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a deterministic Q8_0 byte buffer of shape [n_rows, k_elems/32 * 34].
    /// Each block has 2 scale-bytes set to (row, kb) and qdata bytes set to
    /// `(row + kb * 32 + i) % 256` for i in 0..32. Scale value is arbitrary
    /// for these tests; only byte-identity through the repack is checked.
    fn make_test_q8(n_rows: usize, k_elems: usize) -> Vec<u8> {
        let num_blocks_per_row = k_elems / Q8B_GROUP_SIZE;
        let row_bytes = num_blocks_per_row * Q8B_BLOCK_SIZE;
        let mut buf = vec![0u8; n_rows * row_bytes];
        for row in 0..n_rows {
            for kb in 0..num_blocks_per_row {
                let off = row * row_bytes + kb * Q8B_BLOCK_SIZE;
                // Distinct scale per (row, kb): low byte = row, high byte = kb.
                buf[off] = (row % 256) as u8;
                buf[off + 1] = (kb % 256) as u8;
                for i in 0..Q8B_GROUP_SIZE {
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
        let src = make_test_q8(n_rows, k_elems);
        let repacked = repack_q8_single(&src, n_rows, k_elems).expect("repack_q8_single");

        // Verify total length
        assert_eq!(repacked.len(), src.len(), "repack must preserve total bytes");

        // Verify: for each (rg, kb), the scales are 32 contiguous f16s
        // and qdata is 32 contiguous 32-byte rows.
        let num_blocks_per_row = k_elems / Q8B_GROUP_SIZE;
        let row_bytes = num_blocks_per_row * Q8B_BLOCK_SIZE;
        let num_row_groups = n_rows / TILE_N;
        let stripe_bytes_per_kblock = 64 + 1024;
        let stripe_stride = num_blocks_per_row * stripe_bytes_per_kblock;

        for rg in 0..num_row_groups {
            for kb in 0..num_blocks_per_row {
                let dst_kb_base = rg * stripe_stride + kb * stripe_bytes_per_kblock;
                for r in 0..TILE_N {
                    let row = rg * TILE_N + r;
                    let src_block_off = row * row_bytes + kb * Q8B_BLOCK_SIZE;
                    // Scale at dst_kb_base + r*2 must match src[src_block_off..src_block_off+2]
                    assert_eq!(
                        &repacked[dst_kb_base + r * 2..dst_kb_base + r * 2 + 2],
                        &src[src_block_off..src_block_off + 2],
                        "scale mismatch rg={} kb={} r={} (row={})",
                        rg, kb, r, row
                    );
                    // qdata at dst_kb_base + 64 + r*32 must match src[src_block_off+2..src_block_off+34]
                    let dst_qd = dst_kb_base + 64 + r * 32;
                    assert_eq!(
                        &repacked[dst_qd..dst_qd + 32],
                        &src[src_block_off + 2..src_block_off + 34],
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
        let src_gate = make_test_q8(n_rows, k_elems);
        let mut src_up = make_test_q8(n_rows, k_elems);
        // Mutate src_up so it differs from src_gate
        for b in src_up.iter_mut() {
            *b = b.wrapping_add(1);
        }

        let repacked = repack_q8_pair_gate_up(&src_gate, &src_up, n_rows, k_elems)
            .expect("repack_q8_pair_gate_up");
        assert_eq!(repacked.len(), 2 * src_gate.len());

        let num_blocks_per_row = k_elems / Q8B_GROUP_SIZE;
        let row_bytes = num_blocks_per_row * Q8B_BLOCK_SIZE;
        let num_row_groups = n_rows / TILE_N;
        let stripe_bytes_per_kblock = 2 * (64 + 1024);
        let stripe_stride = num_blocks_per_row * stripe_bytes_per_kblock;

        for rg in 0..num_row_groups {
            for kb in 0..num_blocks_per_row {
                let dst_kb_base = rg * stripe_stride + kb * stripe_bytes_per_kblock;
                for r in 0..TILE_N {
                    let row = rg * TILE_N + r;
                    let src_block_off = row * row_bytes + kb * Q8B_BLOCK_SIZE;

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
                    let dst_gd = dst_kb_base + 128 + r * 32;
                    assert_eq!(
                        &repacked[dst_gd..dst_gd + 32],
                        &src_gate[src_block_off + 2..src_block_off + 34],
                        "gate qdata mismatch rg={} kb={} r={}", rg, kb, r
                    );
                    // Up qdata (offset 1152)
                    let dst_ud = dst_kb_base + 1152 + r * 32;
                    assert_eq!(
                        &repacked[dst_ud..dst_ud + 32],
                        &src_up[src_block_off + 2..src_block_off + 34],
                        "up qdata mismatch rg={} kb={} r={}", rg, kb, r
                    );
                }
            }
        }
    }

    #[test]
    fn repack_qwen35_ffn_down_shape() {
        // FFN-down: N = 4096 (hidden_dim), K = 12288 (inter_dim).
        // Truncated test: 32 rows × 128 elements to keep test fast but exercise
        // the shape arithmetic (4 k-blocks per row).
        let n_rows = 32;
        let k_elems = 128;
        let src = make_test_q8(n_rows, k_elems);
        let repacked = repack_q8_single(&src, n_rows, k_elems).expect("repack_q8_single");
        assert_eq!(repacked.len(), src.len());
        // Sanity: scale of (row=0, kb=0) at repacked[0..2] equals src[0..2]
        assert_eq!(&repacked[0..2], &src[0..2]);
        // Scale of (row=31, kb=0) at repacked[62..64] equals src[31 * 17 * 4..]
        let num_blocks_per_row = k_elems / Q8B_GROUP_SIZE;
        let row_bytes = num_blocks_per_row * Q8B_BLOCK_SIZE;
        let src_off_row31_kb0 = 31 * row_bytes;
        assert_eq!(&repacked[62..64], &src[src_off_row31_kb0..src_off_row31_kb0 + 2]);
    }

    #[test]
    fn repack_rejects_misaligned_n() {
        let src = vec![0u8; 100];
        let err = repack_q8_single(&src, 30, 32).expect_err("must reject n=30");
        assert!(format!("{:?}", err).contains("n_rows"));
    }

    #[test]
    fn repack_rejects_misaligned_k() {
        let src = vec![0u8; 100];
        let err = repack_q8_single(&src, 32, 30).expect_err("must reject k=30");
        assert!(format!("{:?}", err).contains("k_elems"));
    }
}
