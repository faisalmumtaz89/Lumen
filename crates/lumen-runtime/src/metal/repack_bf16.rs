//! Runtime BF16 hot-weight repack into a Metal-friendly stripe layout
//! for the GDN qkv-proj + attn-gate-proj pair on linear-attention layers.
//!
//! Repacks two BF16 weight tensors `W_qkv [N_qkv=8192, K=4096]` and
//! `W_gate [N_gate=4096, K=4096]` (Qwen3.5-9B GDN per-layer shapes) into a
//! single packed buffer. Tensors are concatenated along the output (N)
//! dimension into a logical `[N_total=12288, K=4096]` matrix; that combined
//! matrix is then byte-permuted into the stripe layout described in the
//! reference, grouped by 32-row TILE_N tile and 64-element TILE_K_64
//! K-block:
//!
//! ```text
//!   bytes [0..4096] of each (row_group, k_block):
//!     32 × (64 × 2 bytes) row-major within the stripe
//!     = row 0's 64 BF16 elements, then row 1's, ... row 31's.
//! ```
//!
//! Total bytes per `(row_group, k_block)` = `32 * 64 * 2` = 4096. Per layer
//! (24 GDN layers in Qwen3.5-9B): `(12288 / 32) × (4096 / 64) × 4096`
//! = 96 MB. Total across 24 layers: ~2.3 GB (under the 4.8 GB Apple AGX TLB
//! threshold established vs).
//!
//! ## Why concat-then-stripe (vs interleave-by-output)
//!
//! The two projections share the SAME input `X = normed_buf [M, K=4096]` and
//! are dispatched back-to-back in Phase 1 of the GDN per-layer encoder
//! (`gdn.rs::run_prefill_layer`). A single tiled-GEMM dispatch that walks the
//! combined `N_total = 12288` output rows reads `X` only once across both
//! projections (each K-tile of `X` is loaded into shmem once and reused for
//! every row-group). The two sequential `tiled_matmul_bf16_k64` dispatches
//! reload `X` twice in the unpacked path.
//!
//! Concatenating along N gives the simplest kernel: identical MMA loop to
//! `tiled_matmul_bf16_k64`, only the writeback branches by `gn < qkv_dim`
//! into either `Y_qkv` or `Y_gate`.
//!
//! ## Bit-identical algorithm
//!
//! The repacked kernel MUST produce mathematically identical output to the
//! two separate `tiled_matmul_bf16_k64` dispatches it replaces — only the
//! memory access pattern changes. Every BF16 weight byte that was in the
//! source qkv and attn_gate tensors is preserved verbatim in the repacked
//! buffer (no precision loss, no requantization, no reordering of the K
//! axis).
//!
//! ## Env gating
//!
//! Repack itself is opt-in via `LUMEN_METAL_BF16_GDN_QKV_GATE_PAIRED=1`.
//! When OFF, no extra buffers are allocated — the existing two-dispatch
//! path preserves bit-exact behaviour when disabled.

use super::ffi::{MetalBuffer, MetalDevice};
use crate::error::RuntimeError;

/// TILE_N: 32 output rows per tile (matches kernel constant).
pub(crate) const TILE_N: usize = 32;
/// TILE_K_64: 64 K-elements per stripe block (matches kernel constant TILE_K_64).
pub(crate) const TILE_K_64: usize = 64;
/// BF16 element size in bytes.
pub(crate) const BF16_ELEM_BYTES: usize = 2;
/// Bytes per (row_group, k_block) stripe for a single tensor: 32 * 64 * 2.
pub(crate) const STRIPE_BYTES_SINGLE: usize = TILE_N * TILE_K_64 * BF16_ELEM_BYTES; // 4096

/// Repack a concatenated `[qkv | attn_gate]` BF16 tensor pair into the
/// single-tensor stripe layout. The output buffer is a single contiguous
/// `Vec<u8>` of length `(qkv_n + gate_n) * k_elems * 2` bytes representing the
/// virtual matrix `W_combined[N_total, K]` where `N_total = qkv_n + gate_n`.
///
/// # Arguments
/// * `src_qkv` — source qkv BF16 row-major bytes. Length = `qkv_n * k_elems * 2`.
/// * `src_gate` — source attn_gate BF16 row-major bytes. Length = `gate_n * k_elems * 2`.
/// * `qkv_n` — number of qkv output rows. MUST be a multiple of `TILE_N` (32).
/// * `gate_n` — number of attn_gate output rows. MUST be a multiple of `TILE_N` (32).
/// * `k_elems` — shared K dimension in elements. MUST be a multiple of `TILE_K_64` (64).
///
/// # Returns
/// A `Vec<u8>` of length `(qkv_n + gate_n) * k_elems * 2`.
///
/// # Layout invariants
///
/// Treating the combined matrix as `W[N_total, K]` row-major where
/// `W[0..qkv_n] = src_qkv` and `W[qkv_n..N_total] = src_gate`:
///
/// For each `row_group rg in 0..N_total/32`, for each `k_block kb in 0..k_elems/64`:
///   `dst[rg * stripe_stride + kb * 4096 .. + 4096]`:
///     contains 32 contiguous row-tiles, each `64 * 2 = 128` bytes.
///     Row `r` (0..32) lives at `[r * 128 .. (r+1) * 128]` within the stripe.
///
/// where `stripe_stride = (k_elems / 64) * 4096`.
///
/// # Errors
/// Returns `RuntimeError::Compute` if alignment requirements are not met or
/// any source byte length doesn't match the declared shape.
pub(crate) fn repack_bf16_qkv_gate_concat(
    src_qkv: &[u8],
    src_gate: &[u8],
    qkv_n: usize,
    gate_n: usize,
    k_elems: usize,
) -> Result<Vec<u8>, RuntimeError> {
    if qkv_n % TILE_N != 0 {
        return Err(RuntimeError::Compute(format!(
            "repack_bf16_qkv_gate_concat: qkv_n ({}) must be a multiple of TILE_N ({})",
            qkv_n, TILE_N
        )));
    }
    if gate_n % TILE_N != 0 {
        return Err(RuntimeError::Compute(format!(
            "repack_bf16_qkv_gate_concat: gate_n ({}) must be a multiple of TILE_N ({})",
            gate_n, TILE_N
        )));
    }
    if k_elems % TILE_K_64 != 0 {
        return Err(RuntimeError::Compute(format!(
            "repack_bf16_qkv_gate_concat: k_elems ({}) must be a multiple of TILE_K_64 ({})",
            k_elems, TILE_K_64
        )));
    }
    let row_bytes = k_elems * BF16_ELEM_BYTES;
    let expected_qkv = qkv_n * row_bytes;
    let expected_gate = gate_n * row_bytes;
    if src_qkv.len() != expected_qkv {
        return Err(RuntimeError::Compute(format!(
            "repack_bf16_qkv_gate_concat: src_qkv len {} != expected {} (qkv_n={}, k_elems={})",
            src_qkv.len(), expected_qkv, qkv_n, k_elems
        )));
    }
    if src_gate.len() != expected_gate {
        return Err(RuntimeError::Compute(format!(
            "repack_bf16_qkv_gate_concat: src_gate len {} != expected {} (gate_n={}, k_elems={})",
            src_gate.len(), expected_gate, gate_n, k_elems
        )));
    }

    let n_total = qkv_n + gate_n;
    let num_row_groups = n_total / TILE_N;
    let num_k_blocks = k_elems / TILE_K_64;
    let stripe_bytes_per_kblock = STRIPE_BYTES_SINGLE; // 4096
    let stripe_stride = num_k_blocks * stripe_bytes_per_kblock;
    let total_out = num_row_groups * stripe_stride;
    debug_assert_eq!(
        total_out,
        expected_qkv + expected_gate,
        "Total bytes after concat-repack must equal qkv_bytes + gate_bytes (pure transposition)"
    );

    let mut dst = vec![0u8; total_out];

    // Per-row K-tile size in bytes.
    let row_tile_bytes = TILE_K_64 * BF16_ELEM_BYTES; // 128

    for rg in 0..num_row_groups {
        let dst_rg_base = rg * stripe_stride;
        for kb in 0..num_k_blocks {
            let dst_kb_base = dst_rg_base + kb * stripe_bytes_per_kblock;
            for r in 0..TILE_N {
                let row = rg * TILE_N + r;
                // Route to qkv vs gate source by row index.
                let (src, src_row) = if row < qkv_n {
                    (src_qkv, row)
                } else {
                    (src_gate, row - qkv_n)
                };
                let src_off = src_row * row_bytes + kb * row_tile_bytes;
                let dst_off = dst_kb_base + r * row_tile_bytes;
                dst[dst_off..dst_off + row_tile_bytes]
                    .copy_from_slice(&src[src_off..src_off + row_tile_bytes]);
            }
        }
    }

    Ok(dst)
}

/// Build a Metal buffer holding the qkv + attn_gate concat-then-stripe
/// repacked BF16 weights.
pub(crate) fn build_repacked_buffer_qkv_gate(
    device: &MetalDevice,
    src_qkv: &[u8],
    src_gate: &[u8],
    qkv_n: usize,
    gate_n: usize,
    k_elems: usize,
) -> Result<MetalBuffer, RuntimeError> {
    let dst = repack_bf16_qkv_gate_concat(src_qkv, src_gate, qkv_n, gate_n, k_elems)?;
    device.new_buffer_with_bytes(&dst).ok_or_else(|| {
        RuntimeError::Compute(format!(
            "Failed to allocate BF16 QKV+gate repacked buffer ({} bytes, qkv_n={}, gate_n={}, k_elems={})",
            dst.len(), qkv_n, gate_n, k_elems
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a deterministic BF16 byte buffer of shape `[n_rows, k_elems]`.
    /// Each row contains `k_elems` BF16 values (2 bytes each). The byte at
    /// offset `(row * k_elems + i) * 2 + b` is set to a value derived from
    /// `(tag, row, i, b)` so every byte is uniquely identifiable.
    fn make_test_bf16(n_rows: usize, k_elems: usize, tag: u8) -> Vec<u8> {
        let row_bytes = k_elems * BF16_ELEM_BYTES;
        let mut buf = vec![0u8; n_rows * row_bytes];
        for row in 0..n_rows {
            for i in 0..k_elems {
                let off = row * row_bytes + i * BF16_ELEM_BYTES;
                buf[off] = ((row * 7 + i * 3 + tag as usize) % 256) as u8;
                buf[off + 1] = ((row * 11 + i * 5 + tag as usize * 2) % 256) as u8;
            }
        }
        buf
    }

    /// Verify a single source row appears verbatim in the stripe layout at
    /// the expected position.
    fn assert_row_in_stripe(
        repacked: &[u8],
        src_row: &[u8],
        stripe_row_index: usize,
        k_elems: usize,
    ) {
        let row_tile_bytes = TILE_K_64 * BF16_ELEM_BYTES;
        let num_k_blocks = k_elems / TILE_K_64;
        let stripe_stride = num_k_blocks * STRIPE_BYTES_SINGLE;
        let rg = stripe_row_index / TILE_N;
        let r = stripe_row_index % TILE_N;
        for kb in 0..num_k_blocks {
            let dst_kb_base = rg * stripe_stride + kb * STRIPE_BYTES_SINGLE;
            let dst_off = dst_kb_base + r * row_tile_bytes;
            let src_off = kb * row_tile_bytes;
            assert_eq!(
                &repacked[dst_off..dst_off + row_tile_bytes],
                &src_row[src_off..src_off + row_tile_bytes],
                "row-tile mismatch at stripe row {} kb {}", stripe_row_index, kb
            );
        }
    }

    #[test]
    fn concat_byte_identity_small() {
        // Smallest meaningful shape: qkv 32 rows + gate 32 rows = 64 total,
        // 2 row-groups × 2 k-blocks.
        let qkv_n = 32;
        let gate_n = 32;
        let k_elems = 128;
        let src_q = make_test_bf16(qkv_n, k_elems, 0x11);
        let src_g = make_test_bf16(gate_n, k_elems, 0x22);
        let repacked = repack_bf16_qkv_gate_concat(&src_q, &src_g, qkv_n, gate_n, k_elems)
            .expect("repack_bf16_qkv_gate_concat");
        let expected_len = (qkv_n + gate_n) * k_elems * BF16_ELEM_BYTES;
        assert_eq!(repacked.len(), expected_len, "concat repack must preserve total bytes");

        let row_bytes = k_elems * BF16_ELEM_BYTES;
        // First TILE_N rows come from qkv.
        for r in 0..qkv_n {
            let src_row = &src_q[r * row_bytes..(r + 1) * row_bytes];
            assert_row_in_stripe(&repacked, src_row, r, k_elems);
        }
        // Remaining rows come from gate.
        for r in 0..gate_n {
            let src_row = &src_g[r * row_bytes..(r + 1) * row_bytes];
            assert_row_in_stripe(&repacked, src_row, qkv_n + r, k_elems);
        }
    }

    #[test]
    fn concat_qwen35_gdn_shape() {
        // Qwen3.5-9B GDN per-layer shape, but with k_elems reduced to 256
        // to keep the test fast. The layout invariant is independent of K.
        let qkv_n = 8192;
        let gate_n = 4096;
        let k_elems = 256; // reduced from 4096 for test speed; layout is identical
        let src_q = make_test_bf16(qkv_n, k_elems, 0xAA);
        let src_g = make_test_bf16(gate_n, k_elems, 0xBB);
        let repacked = repack_bf16_qkv_gate_concat(&src_q, &src_g, qkv_n, gate_n, k_elems)
            .expect("repack qwen35 gdn shape");
        let expected_len = (qkv_n + gate_n) * k_elems * BF16_ELEM_BYTES;
        assert_eq!(repacked.len(), expected_len);

        let row_bytes = k_elems * BF16_ELEM_BYTES;
        // Spot-check three boundary regions: first qkv row, last qkv row, first gate row.
        let last_q = qkv_n - 1;
        let first_g_stripe_idx = qkv_n;
        let last_g = gate_n - 1;
        for (stripe_idx, src, src_row) in [
            (0usize, &src_q, 0usize),
            (last_q, &src_q, last_q),
            (first_g_stripe_idx, &src_g, 0usize),
            (qkv_n + last_g, &src_g, last_g),
        ] {
            let src_slice = &src[src_row * row_bytes..(src_row + 1) * row_bytes];
            assert_row_in_stripe(&repacked, src_slice, stripe_idx, k_elems);
        }
    }

    #[test]
    fn concat_rejects_misaligned_qkv_n() {
        let src_q = vec![0u8; 31 * 64 * 2];
        let src_g = vec![0u8; 32 * 64 * 2];
        let err = repack_bf16_qkv_gate_concat(&src_q, &src_g, 31, 32, 64);
        assert!(err.is_err());
    }

    #[test]
    fn concat_rejects_misaligned_gate_n() {
        let src_q = vec![0u8; 32 * 64 * 2];
        let src_g = vec![0u8; 17 * 64 * 2];
        let err = repack_bf16_qkv_gate_concat(&src_q, &src_g, 32, 17, 64);
        assert!(err.is_err());
    }

    #[test]
    fn concat_rejects_misaligned_k() {
        let src_q = vec![0u8; 32 * 63 * 2];
        let src_g = vec![0u8; 32 * 63 * 2];
        let err = repack_bf16_qkv_gate_concat(&src_q, &src_g, 32, 32, 63);
        assert!(err.is_err());
    }

    #[test]
    fn concat_rejects_length_mismatch() {
        // gate_n claims 32 but src_gate buffer is too short.
        let src_q = vec![0u8; 32 * 64 * 2];
        let src_g = vec![0u8; 32 * 64]; // half the bytes required
        let err = repack_bf16_qkv_gate_concat(&src_q, &src_g, 32, 32, 64);
        assert!(err.is_err());
    }
}
