//! Lossless plane transforms for [`QuantScheme::CtInt4G32`] tensors.
//!
//! The pack-quantized planes (see the scheme doc for the layout) sometimes
//! need reindexing during import — e.g. the GDN v-head permutation that the
//! rest of the pipeline's tensor convention requires. Every transform here
//! is a pure permutation of the stored 4-bit/BF16 values: unpack → reindex →
//! repack, with no arithmetic on the values themselves.
//!
//! [`QuantScheme::CtInt4G32`]: lumen_format::QuantScheme::CtInt4G32

/// Unpack little-nibble-first 4-bit values from i32-word bytes.
/// `n_values` trims a possibly padded final word. Test-support: the naive
/// reference for verifying the packed-plane transforms value-by-value.
#[cfg(test)]
pub(crate) fn unpack_nibbles(words: &[u8], n_values: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n_values);
    for chunk in words.chunks_exact(4) {
        let w = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        for j in 0..8 {
            if out.len() == n_values {
                return out;
            }
            out.push(((w >> (4 * j)) & 0xF) as u8);
        }
    }
    out
}

/// Pack 4-bit values into little-nibble-first i32-word bytes, zero-padding
/// the final word. Test-support counterpart of [`unpack_nibbles`].
#[cfg(test)]
pub(crate) fn pack_nibbles(values: &[u8]) -> Vec<u8> {
    let n_words = values.len().div_ceil(8);
    let mut out = Vec::with_capacity(n_words * 4);
    for chunk in values.chunks(8) {
        let mut w: u32 = 0;
        for (j, &v) in chunk.iter().enumerate() {
            w |= u32::from(v & 0xF) << (4 * j);
        }
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

/// Reorder rows of a row-major plane with `row_bytes` bytes per row:
/// output row `i` = input row `perm[i]`.
pub(crate) fn permute_rows(data: &[u8], row_bytes: usize, perm: &[usize]) -> Vec<u8> {
    debug_assert_eq!(data.len(), row_bytes * perm.len());
    let mut out = Vec::with_capacity(data.len());
    for &src in perm {
        out.extend_from_slice(&data[src * row_bytes..(src + 1) * row_bytes]);
    }
    out
}

/// Reorder a `weight_zero_point` plane (4-bit zero-points packed 8 rows per
/// i32 word, `groups` columns) under a row permutation of the LOGICAL rows:
/// output logical row `i` = input logical row `perm[i]`. `n` is the logical
/// row count (`perm.len()`); the word rows are `ceil(n / 8)`.
pub(crate) fn permute_zero_point_rows(
    data: &[u8],
    n: usize,
    groups: usize,
    perm: &[usize],
) -> Vec<u8> {
    debug_assert_eq!(perm.len(), n);
    let word_rows = n.div_ceil(8);
    debug_assert_eq!(data.len(), word_rows * groups * 4);
    // Unpack to [n][groups] logical values (column g of word row r holds
    // rows r*8..r*8+8 in its 8 nibbles).
    let mut logical = vec![0u8; n * groups];
    for wr in 0..word_rows {
        for g in 0..groups {
            let off = (wr * groups + g) * 4;
            let w = u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
            for j in 0..8 {
                let row = wr * 8 + j;
                if row < n {
                    logical[row * groups + g] = ((w >> (4 * j)) & 0xF) as u8;
                }
            }
        }
    }
    let mut out = vec![0u8; data.len()];
    for wr in 0..word_rows {
        for g in 0..groups {
            let mut w: u32 = 0;
            for j in 0..8 {
                let row = wr * 8 + j;
                if row < n {
                    let src = perm[row];
                    w |= u32::from(logical[src * groups + g]) << (4 * j);
                }
            }
            let off = (wr * groups + g) * 4;
            out[off..off + 4].copy_from_slice(&w.to_le_bytes());
        }
    }
    out
}

/// Reorder K-dimension blocks of a CtInt4G32 tensor's planes: the K axis is
/// split into `perm.len()` equal blocks of `block_cols` columns each
/// (`block_cols % 32 == 0`, so scale/zero-point groups move whole), and
/// output block `i` = input block `perm[i]` within every row.
///
/// Returns the three transformed planes `(qweight, scale, zero_point)`.
pub(crate) fn permute_k_blocks(
    qweight: &[u8],
    scale: &[u8],
    zero_point: &[u8],
    n: usize,
    k: usize,
    block_cols: usize,
    perm: &[usize],
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    debug_assert_eq!(block_cols % 32, 0);
    debug_assert_eq!(k, block_cols * perm.len());
    let groups = k / 32;
    let word_rows = n.div_ceil(8);
    debug_assert_eq!(qweight.len(), n * k / 2);
    debug_assert_eq!(scale.len(), n * groups * 2);
    debug_assert_eq!(zero_point.len(), word_rows * groups * 4);

    // qweight: block = block_cols/8 words = block_cols/2 bytes, per row.
    let qrow = k / 2;
    let qblock = block_cols / 2;
    let mut q_out = vec![0u8; qweight.len()];
    for row in 0..n {
        let base = row * qrow;
        for (i, &src) in perm.iter().enumerate() {
            q_out[base + i * qblock..base + (i + 1) * qblock]
                .copy_from_slice(&qweight[base + src * qblock..base + (src + 1) * qblock]);
        }
    }
    // scale: block = block_cols/32 groups × 2 bytes, per row.
    let srow = groups * 2;
    let sblock = (block_cols / 32) * 2;
    let mut s_out = vec![0u8; scale.len()];
    for row in 0..n {
        let base = row * srow;
        for (i, &src) in perm.iter().enumerate() {
            s_out[base + i * sblock..base + (i + 1) * sblock]
                .copy_from_slice(&scale[base + src * sblock..base + (src + 1) * sblock]);
        }
    }
    // zero_point: same column-group blocks, per WORD row (columns are K
    // groups; the 8-row packing along N is untouched by a K permutation).
    let zrow = groups * 4;
    let zblock = (block_cols / 32) * 4;
    let mut z_out = vec![0u8; zero_point.len()];
    for wr in 0..word_rows {
        let base = wr * zrow;
        for (i, &src) in perm.iter().enumerate() {
            z_out[base + i * zblock..base + (i + 1) * zblock]
                .copy_from_slice(&zero_point[base + src * zblock..base + (src + 1) * zblock]);
        }
    }
    (q_out, s_out, z_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng_bytes(len: usize, seed: u64) -> Vec<u8> {
        let mut s = seed;
        (0..len)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 33) as u8
            })
            .collect()
    }

    #[test]
    fn nibble_pack_roundtrip() {
        let vals: Vec<u8> = rng_bytes(100, 1).iter().map(|b| b & 0xF).collect();
        assert_eq!(unpack_nibbles(&pack_nibbles(&vals), 100), vals);
        // trims final-word padding
        let vals: Vec<u8> = rng_bytes(9, 2).iter().map(|b| b & 0xF).collect();
        let packed = pack_nibbles(&vals);
        assert_eq!(packed.len(), 8);
        assert_eq!(unpack_nibbles(&packed, 9), vals);
    }

    #[test]
    fn nibble_order_is_little_first() {
        // word 0x87654321 → values [1,2,3,4,5,6,7,8]
        let packed = 0x8765_4321u32.to_le_bytes().to_vec();
        assert_eq!(unpack_nibbles(&packed, 8), vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn row_permutation_moves_whole_rows() {
        let data: Vec<u8> = (0..12).collect();
        let out = permute_rows(&data, 4, &[2, 0, 1]);
        assert_eq!(out, vec![8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn zero_point_row_permutation_roundtrip_and_values() {
        let n = 12; // crosses a word-row boundary (8)
        let groups = 3;
        let vals: Vec<u8> = rng_bytes(n * groups, 3).iter().map(|b| b & 0xF).collect();
        // pack [n][groups] logically: word row wr, col g holds rows wr*8+j
        let word_rows = n.div_ceil(8);
        let mut plane = vec![0u8; word_rows * groups * 4];
        for wr in 0..word_rows {
            for g in 0..groups {
                let mut w = 0u32;
                for j in 0..8 {
                    let row = wr * 8 + j;
                    if row < n {
                        w |= u32::from(vals[row * groups + g]) << (4 * j);
                    }
                }
                plane[(wr * groups + g) * 4..(wr * groups + g) * 4 + 4]
                    .copy_from_slice(&w.to_le_bytes());
            }
        }
        let perm: Vec<usize> = (0..n).rev().collect();
        let permuted = permute_zero_point_rows(&plane, n, groups, &perm);
        // identity permutation restores the original
        let inverse: Vec<usize> = (0..n).rev().collect();
        assert_eq!(
            permute_zero_point_rows(&permuted, n, groups, &inverse),
            plane
        );
        // spot-check values landed where expected
        let back = permute_zero_point_rows(&permuted, n, groups, &(0..n).collect::<Vec<_>>());
        for row in 0..n {
            for g in 0..groups {
                let off = ((row / 8) * groups + g) * 4;
                let w =
                    u32::from_le_bytes([back[off], back[off + 1], back[off + 2], back[off + 3]]);
                let got = ((w >> (4 * (row % 8))) & 0xF) as u8;
                assert_eq!(got, vals[perm[row] * groups + g], "row {row} group {g}");
            }
        }
    }

    #[test]
    fn k_block_permutation_roundtrip() {
        let (n, k, block) = (5, 256, 64); // 4 blocks of 64 cols (2 groups each)
        let groups = k / 32;
        let q = rng_bytes(n * k / 2, 4);
        let s = rng_bytes(n * groups * 2, 5);
        let z = rng_bytes(n.div_ceil(8) * groups * 4, 6);
        let perm = vec![3usize, 1, 0, 2];
        let inv = {
            let mut inv = vec![0usize; 4];
            for (i, &p) in perm.iter().enumerate() {
                inv[p] = i;
            }
            inv
        };
        let (q1, s1, z1) = permute_k_blocks(&q, &s, &z, n, k, block, &perm);
        let (q2, s2, z2) = permute_k_blocks(&q1, &s1, &z1, n, k, block, &inv);
        assert_eq!(q2, q);
        assert_eq!(s2, s);
        assert_eq!(z2, z);
        // and the forward move is real: block 0 of output row 0 == block 3 of input
        assert_eq!(&q1[..block / 2], &q[3 * (block / 2)..4 * (block / 2)]);
    }
}
