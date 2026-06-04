// Repack Q4Raw (18-byte AoS blocks) into per-row tile-grouped layout.
//
// Source per-row layout (existing Q4Raw / standard Q4_0):
//   For each of `nb` blocks (in_dim / 32):
//     bytes [0..1]:  f16 scale
//     bytes [2..17]: 16 bytes of de-interleaved nibbles
//   Row stride: 18 * nb bytes.
//
// Destination per-row layout (tile-grouped):
//   For each of `num_tiles` tiles (nb / 8):
//     bytes [0..15]:  8 x f16 scales (16 bytes)
//     bytes [16..143]: 8 x 16 nibble bytes (128 bytes)
//   Tile stride: 144 bytes
//   Row stride: num_tiles * 144 = (nb/8) * 144 bytes
//
// Per-row byte size shrinks from 18*nb to (nb/8)*144 = 18*nb -- identical
// density (no compression). The win is layout: each tile colocates its 8
// scales and 8 nibble blocks, so the matvec kernel touches 144 contiguous
// bytes per outer-loop strided iteration instead of two distant ranges in
// the SoA split layout.
//
// Tile alignment within the destination row:
//   * tile_base = row_base + tile_idx * 144
//   * Scale region [0, 16) within tile: 8 x f16 = 16 bytes contiguous
//   * Nibble region [16, 144): 8 x 16 nibble bytes contiguous
//   * Per-block scale offset within tile: slot * 2 (2-byte aligned)
//   * Per-block nibble offset within tile: 16 + slot * 16 (4-byte aligned)
//
// nb must be a multiple of 8 (one tile = 8 blocks). All Qwen3.5-9B model
// dims satisfy this constraint (hidden=4096 -> nb=128, inter=12288 -> nb=384,
// kv_dim=1024 -> nb=32, head_dim=256 -> nb=8).
//
// Runs ONCE per layer during preload_weights -- not on the decode hot path.
//
// Grid:  (ceil(num_blocks_total / 256), 1, 1)
// Block: (256, 1, 1)
//
// NVRTC-compatible: no system includes, extern "C" linkage.

extern "C" __global__ void repack_q4_raw_to_tile(
    const char* __restrict__ src,   // [num_rows * nb * 18] Q4Raw AoS
    char* __restrict__ dst,         // [num_rows * (nb/8) * 144] tile-grouped
    unsigned int nb,                // blocks per row (must be multiple of 8)
    unsigned int num_rows)
{
    unsigned int global_block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total_blocks = (unsigned long long)nb * (unsigned long long)num_rows;
    if ((unsigned long long)global_block_idx >= total_blocks) return;

    unsigned int row = global_block_idx / nb;
    unsigned int block_in_row = global_block_idx - row * nb;

    // Source block: row_base_src + block_in_row * 18.
    const char* s = src
        + (unsigned long long)row * (unsigned long long)nb * 18ULL
        + (unsigned long long)block_in_row * 18ULL;

    // Tile-local indexing within destination row.
    unsigned int tile_idx = block_in_row >> 3;   // / 8
    unsigned int slot     = block_in_row & 0x7;  // % 8
    unsigned int num_tiles = nb >> 3;            // / 8

    unsigned long long row_bytes = (unsigned long long)num_tiles * 144ULL;
    char* row_dst = dst + (unsigned long long)row * row_bytes;
    char* tile_dst = row_dst + (unsigned long long)tile_idx * 144ULL;

    // Scale destination: tile_dst[0..15], slot offset is slot * 2.
    char* scale_dst  = tile_dst + (unsigned long long)slot * 2ULL;
    // Nibble destination: tile_dst[16..143], slot offset 16 + slot*16.
    char* nibble_dst = tile_dst + 16ULL + (unsigned long long)slot * 16ULL;

    // Copy 2-byte f16 scale (source offset +0..1).
    scale_dst[0] = s[0];
    scale_dst[1] = s[1];

    // Copy 16 nibble bytes (source offset +2..17).
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        nibble_dst[i] = s[2 + i];
    }
}
