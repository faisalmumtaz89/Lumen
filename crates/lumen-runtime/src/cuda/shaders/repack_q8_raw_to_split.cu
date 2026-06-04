// Repack Q8Raw (34-byte AoS blocks) into per-row split layout.
//
// Source per-row layout (existing Q8Raw / standard Q8_0):
//   For each of `nb` blocks (in_dim / 32):
//     bytes [0..1]:  f16 scale
//     bytes [2..33]: 32 x int8 quants
//   Row stride: 34 * nb bytes.
//
// Destination per-row layout (split / SoA):
//   bytes [0 .. 2*nb):     nb x f16 scales (contiguous)
//   bytes [2*nb .. 34*nb): nb x 32 = 32*nb x int8 quants (contiguous)
//   Row stride: 34 * nb bytes.
//
// Per-row byte size: identical (34*nb). The source has scales interleaved
// with quants per block; the destination separates them so the matvec
// kernel can issue native int* loads on the quants stream.
//
// Quants base offset within the destination row is `2*nb`, which is a
// multiple of 4 when `nb` is even. Every model dim shipped today satisfies
// this (hidden=4096, kv_dim=1024, inter=12288, head_dim=256 all yield even
// nb), so native int* loads remain legal in the consumer kernel.
//
// Runs ONCE per layer during preload_weights -- not on the decode hot path.
//
// Grid:  (ceil(num_blocks_total / 256), 1, 1)
// Block: (256, 1, 1)
//
// NVRTC-compatible: no system includes, extern "C" linkage.

extern "C" __global__ void repack_q8_raw_to_split(
    const char* __restrict__ src,   // [num_rows * nb * 34] Q8Raw AoS
    char* __restrict__ dst,         // [num_rows * nb * 34] split layout
    unsigned int nb,                // blocks per row
    unsigned int num_rows)
{
    unsigned int global_block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total_blocks = (unsigned long long)nb * (unsigned long long)num_rows;
    if ((unsigned long long)global_block_idx >= total_blocks) return;

    unsigned int row = global_block_idx / nb;
    unsigned int block_in_row = global_block_idx - row * nb;

    // Source block: row_base_src + block_in_row * 34.
    const char* s = src
        + (unsigned long long)row * (unsigned long long)nb * 34ULL
        + (unsigned long long)block_in_row * 34ULL;

    // Destination row base, scales region at [0, 2*nb), quants region at [2*nb, 34*nb).
    unsigned long long row_bytes = (unsigned long long)nb * 34ULL;
    char* row_dst = dst + (unsigned long long)row * row_bytes;
    char* scale_dst = row_dst + (unsigned long long)block_in_row * 2ULL;
    char* quant_dst = row_dst + (unsigned long long)nb * 2ULL
                              + (unsigned long long)block_in_row * 32ULL;

    // Copy 2-byte f16 scale (source offset +0..1).
    scale_dst[0] = s[0];
    scale_dst[1] = s[1];

    // Copy 32 int8 quants (source offset +2..33) into the dense quants stream.
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        quant_dst[i] = s[2 + i];
    }
}
