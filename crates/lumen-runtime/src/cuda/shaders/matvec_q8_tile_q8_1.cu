// ==========================================================================
// Q8 tile-grouped matvec against pre-quantized Q8_1 input (dp4a, NR=2).
//
// Tile layout (272 bytes per tile of 8 blocks = 256 elements):
//   bytes  [0..15]:  8 x f16 scales (16 bytes)
//   bytes [16..271]: 8 x 32 int8 quants (256 bytes)
//
// Per row, tiles stride linearly:
//   row_bytes = num_tiles * 272 = (nb / 8) * 272
//
// Hypothesis: vs raw Q8_0 (matvec_q8_0_q8_1.cu):
//   * Quant data at offset 16 IS 4-byte aligned -> native int* loads
//     (8 instructions per block vs 8 uint16 loads + 8 shift/OR = 16 ops).
//   * Same byte density as raw Q8_0 (272 bytes / 256 elements = 1.0625 B/elem)
//     vs Q8Aligned at 36 B/block = 1.125 B/elem (6% extra storage).
//   * Scales contiguous at tile head -> one LDG.128 fetches all 8 scales.
//
// vs Q8Aligned 36-byte (matvec_q8_aligned_q8_1.cu):
//   * 4.4% fewer bytes per tile (272 vs 288 per 8 blocks).
//   * Same alignment benefits (4-byte aligned quants).
//
// vs Q8Split (matvec_q8_split_q8_1.cu, row-level scales+quants streams):
//   * Same byte density (272 = 16 scales + 256 quants per tile).
//   * Tile-local layout: one CTA can prefetch a tile's worth (272B) without
//     traversing two distant ranges (scales-base vs quants-base far apart).
//   * Better L1/sector locality on outer-loop strides.
//
// Each CTA processes NR=2 output rows. With THREADS_PER_BLOCK=128:
//   8 threads collaborate on 1 tile (each thread handles 1 of 8 blocks).
//   128 threads -> 16 tiles processed per outer iteration per row.
//
// Two kernels:
//   1. matvec_q8_tile_q8_1:          W*x -> out
//   2. matvec_q8_tile_q8_1_residual: W*x + residual -> out
//
// Alignment contract enforced by the host:
//   * nb (= in_dim / 32) MUST be a multiple of 8. Every model dim shipped
//     today (hidden=4096->nb=128, kv_dim=1024->nb=32, inter=12288->nb=384,
//     head_dim=256->nb=8) satisfies this.
//   * cudaMalloc returns 256-byte aligned base pointers, so row_base inherits
//     16-byte alignment (tile header is 16B-aligned for LDG.E.128).
//
// NVRTC-compatible: no system includes, extern "C" linkage.
// Requires compute capability >= 6.1 for dp4a_s32() (Pascal+).
// ==========================================================================

#define NR       2     // rows per thread block
#define NW       32    // warp size
#define THREADS_PER_BLOCK 128  // 4 warps
#define NWARPS   (THREADS_PER_BLOCK / NW)  // 4
#define Q8_BLOCK_SIZE     32   // elements per Q8 block
#define Q8_TILE_BLOCKS    8    // blocks per tile
#define Q8_TILE_ELEMENTS  256  // elements per tile (32 * 8)
#define Q8_TILE_BYTES     272  // 16 (scales) + 256 (quants) per tile
#define Q8_TILE_QUANT_OFF 16   // offset of quants stream within a tile
#define Q8_1_BYTES        36   // 2B f16 scale + 2B f16 sum + 32B int8 data

// Hardware f16->f32 conversion via PTX (single instruction on SM 53+).
__device__ __forceinline__ float f16_bits_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

// inline-PTX dp4a wrapper. The `__dp4a` intrinsic NVRTC-fails in
// this build env (driver 580.126.20 / CUDA 12.2 / sm_80). The inline
// `dp4a.s32.s32` opcode loads cleanly on compute_80.
__device__ __forceinline__ int dp4a_s32(int a, int b, int c) {
    int d;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// Warp-level reduction: sum all lanes in a warp using butterfly shuffle.
__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

// ==========================================================================
// Kernel 1: tile-grouped Q8 weight x Q8_1 input -> F32 output (dp4a, NR=2).
//
// Grid:  (ceil(out_dim / NR), 1, 1)
// Block: (128, 1, 1)
//
// Each thread iterates over Q8 blocks ib = tid, tid+128, tid+256, ...
// For block ib, the enclosing tile index is `tile = ib >> 3` and the slot
// within the tile is `slot = ib & 7`. Scales for the tile sit at tile*272
// (16 contiguous bytes); quants at tile*272 + 16 + slot*32 (4-byte aligned).
// ==========================================================================
extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 2) void matvec_q8_tile_q8_1(
    const char* __restrict__ weight_q8_tile,     // [out_dim * num_tiles * 272]
    const char* __restrict__ input_q8_1,         // [nb * 36] Q8_1 pre-quantized input
    float* __restrict__ out,                     // [out_dim] F32 output
    unsigned int out_dim,
    unsigned int in_dim)
{
    unsigned int r0 = blockIdx.x * NR;

    unsigned int warp_id = threadIdx.x / NW;
    unsigned int lane    = threadIdx.x % NW;

    unsigned int nb = in_dim >> 5;  // in_dim / 32
    // nb is required to be a multiple of 8 (one tile = 8 blocks).
    unsigned int num_tiles = nb >> 3;
    unsigned long long row_bytes = (unsigned long long)num_tiles * (unsigned long long)Q8_TILE_BYTES;

    float sumf[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) sumf[r] = 0.0f;

    for (unsigned int ib = threadIdx.x; ib < nb; ib += THREADS_PER_BLOCK) {

        // --- Load Q8_1 input block (shared across NR rows) ---
        const char* x_block = input_q8_1 + (unsigned long long)ib * Q8_1_BYTES;

        unsigned short x_scale_bits = (unsigned short)(unsigned char)x_block[0]
                                    | ((unsigned short)(unsigned char)x_block[1] << 8);
        float x_scale = f16_bits_to_f32(x_scale_bits);

        // Q8_1 quants at +4 are 4-byte aligned (native int* loads).
        const int* x_packed = (const int*)(x_block + 4);
        int xv[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) xv[k] = x_packed[k];

        // Tile-local indexing for this block.
        unsigned int tile_idx = ib >> 3;        // ib / 8
        unsigned int slot     = ib & 0x7;       // ib % 8

        // --- Process NR output rows ---
        #pragma unroll
        for (int row = 0; row < NR; row++) {
            if (r0 + row >= out_dim) break;

            const char* row_base = weight_q8_tile
                + (unsigned long long)(r0 + row) * row_bytes;
            const char* tile_base = row_base
                + (unsigned long long)tile_idx * Q8_TILE_BYTES;

            // Scale lookup: 8 f16 scales at tile_base[0..15]. Block slot's scale
            // is at offset slot * 2 (2 bytes per scale).
            const unsigned short* scales_u16 = (const unsigned short*)tile_base;
            float w_scale = f16_bits_to_f32(scales_u16[slot]);

            // Quant lookup: 8 x 32 int8 quants start at tile_base + 16.
            // Block slot's quants at offset slot * 32. 4-byte aligned because
            // tile_base is 16-byte aligned and 16 + slot*32 is multiple of 4.
            const int* w_packed = (const int*)(tile_base
                + Q8_TILE_QUANT_OFF
                + (unsigned long long)slot * 32ULL);

            int acc = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                acc = dp4a_s32(w_packed[k], xv[k], acc);
            }

            sumf[row] += w_scale * x_scale * (float)acc;
        }
    }

    // --- Cross-warp reduction via simple shmem ---
    #pragma unroll
    for (int r = 0; r < NR; r++) {
        sumf[r] = warp_reduce_sum(sumf[r]);
    }

    __shared__ float shmem[(NWARPS - 1) * NR];

    if (warp_id > 0 && lane == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            shmem[(warp_id - 1) * NR + r] = sumf[r];
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            float total = sumf[r];
            #pragma unroll
            for (int w = 0; w < NWARPS - 1; w++) {
                total += shmem[w * NR + r];
            }
            if (r0 + r < out_dim) {
                out[r0 + r] = total;
            }
        }
    }
}

// ==========================================================================
// Kernel 2: tile-grouped Q8 weight x Q8_1 input + residual -> F32 output.
// ==========================================================================
extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 2) void matvec_q8_tile_q8_1_residual(
    const char* __restrict__ weight_q8_tile,
    const char* __restrict__ input_q8_1,
    const float* __restrict__ residual,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim)
{
    unsigned int r0 = blockIdx.x * NR;

    unsigned int warp_id = threadIdx.x / NW;
    unsigned int lane    = threadIdx.x % NW;

    unsigned int nb = in_dim >> 5;
    unsigned int num_tiles = nb >> 3;
    unsigned long long row_bytes = (unsigned long long)num_tiles * (unsigned long long)Q8_TILE_BYTES;

    float sumf[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) sumf[r] = 0.0f;

    for (unsigned int ib = threadIdx.x; ib < nb; ib += THREADS_PER_BLOCK) {

        const char* x_block = input_q8_1 + (unsigned long long)ib * Q8_1_BYTES;

        unsigned short x_scale_bits = (unsigned short)(unsigned char)x_block[0]
                                    | ((unsigned short)(unsigned char)x_block[1] << 8);
        float x_scale = f16_bits_to_f32(x_scale_bits);

        const int* x_packed = (const int*)(x_block + 4);
        int xv[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) xv[k] = x_packed[k];

        unsigned int tile_idx = ib >> 3;
        unsigned int slot     = ib & 0x7;

        #pragma unroll
        for (int row = 0; row < NR; row++) {
            if (r0 + row >= out_dim) break;

            const char* row_base = weight_q8_tile
                + (unsigned long long)(r0 + row) * row_bytes;
            const char* tile_base = row_base
                + (unsigned long long)tile_idx * Q8_TILE_BYTES;

            const unsigned short* scales_u16 = (const unsigned short*)tile_base;
            float w_scale = f16_bits_to_f32(scales_u16[slot]);

            const int* w_packed = (const int*)(tile_base
                + Q8_TILE_QUANT_OFF
                + (unsigned long long)slot * 32ULL);

            int acc = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                acc = dp4a_s32(w_packed[k], xv[k], acc);
            }

            sumf[row] += w_scale * x_scale * (float)acc;
        }
    }

    #pragma unroll
    for (int r = 0; r < NR; r++) {
        sumf[r] = warp_reduce_sum(sumf[r]);
    }

    __shared__ float shmem[(NWARPS - 1) * NR];

    if (warp_id > 0 && lane == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            shmem[(warp_id - 1) * NR + r] = sumf[r];
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            float total = sumf[r];
            #pragma unroll
            for (int w = 0; w < NWARPS - 1; w++) {
                total += shmem[w * NR + r];
            }
            if (r0 + r < out_dim) {
                out[r0 + r] = total + residual[r0 + r];
            }
        }
    }
}
