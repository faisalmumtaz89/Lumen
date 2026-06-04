// Repack Q4Raw (18-byte AoS blocks) into per-row split layout.
//
// Source per-row layout (existing Q4Raw / standard Q4_0):
//   For each of `nb` blocks (in_dim / 32):
//     bytes [0..1]:  f16 scale
//     bytes [2..17]: 16 bytes of de-interleaved nibbles
//   Row stride: 18 * nb bytes.
//
// Destination per-row layout (split / SoA):
//   bytes [0 .. 2*nb):     nb x f16 scales (contiguous)
//   bytes [2*nb .. 18*nb): nb x 16 = 16*nb nibble bytes (contiguous)
//   Row stride: 18 * nb bytes.
//
// Per-row byte size: identical (18*nb). The source interleaves scales with
// nibble data per block; the destination separates them so the matvec kernel
// can issue native int* loads on the contiguous nibble stream (4 int* loads
// per block vs 16 individual byte loads in the AoS Q4Raw path).
//
// Nibble base offset within the destination row is `2*nb`. Every model dim
// shipped today yields an even `nb` (hidden=4096->nb=128, kv_dim=1024->nb=32,
// inter=12288->nb=384, head_dim=256->nb=8), so `2*nb` is at minimum 4-byte
// aligned; the consumer kernel's native `int*` loads on the nibble stream
// remain legal.
//
// Runs ONCE per layer during preload_weights -- not on the decode hot path.
//
// Grid:  (ceil(num_blocks_total / 256), 1, 1)
// Block: (256, 1, 1)
//
// NVRTC-compatible: no system includes, extern "C" linkage.

extern "C" __global__ void repack_q4_raw_to_split(
    const char* __restrict__ src,   // [num_rows * nb * 18] Q4Raw AoS
    char* __restrict__ dst,         // [num_rows * nb * 18] split layout
    unsigned int nb,                // blocks per row
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

    // Destination row base, scales region at [0, 2*nb), nibbles region at [2*nb, 18*nb).
    unsigned long long row_bytes = (unsigned long long)nb * 18ULL;
    char* row_dst = dst + (unsigned long long)row * row_bytes;
    char* scale_dst  = row_dst + (unsigned long long)block_in_row * 2ULL;
    char* nibble_dst = row_dst + (unsigned long long)nb * 2ULL
                                + (unsigned long long)block_in_row * 16ULL;

    // Copy 2-byte f16 scale (source offset +0..1).
    scale_dst[0] = s[0];
    scale_dst[1] = s[1];

    // Copy 16 nibble bytes (source offset +2..17) into the dense nibble stream.
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        nibble_dst[i] = s[2 + i];
    }
}
