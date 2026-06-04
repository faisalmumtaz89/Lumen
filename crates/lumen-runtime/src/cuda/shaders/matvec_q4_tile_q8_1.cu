// ==========================================================================
// Q4 tile-grouped matvec against pre-quantized Q8_1 input (dp4a, NR=4).
//
// Adapts the Q8 tile-grouped pattern (matvec_q8_tile_q8_1) to Q4_0. The Q8
// tile variant places 8 f16 scales and 8 x 32 int8 quants contiguously per
// tile, so a single CTA's outer-loop iteration touches one tile worth of
// bytes (272 B for Q8) before striding. For Q4, the byte budget is roughly
// half:
//
//   Per Q4 tile (8 blocks = 256 elements):
//     bytes  [ 0..15]: 8 x f16 scales (16 bytes, one LDG.128 worth)
//     bytes [16..143]: 8 x 16 nibble bytes (128 bytes)
//   Tile bytes: 144   (Q8 tile = 272 B; Q4 tile = 144 B)
//   Density:   144 / 256 = 0.5625 B/element  (identical to Q4_SPLIT)
//
// Per row, tiles stride linearly:
//   row_bytes = num_tiles * 144 = (nb / 8) * 144
//
// Hypothesis vs matvec_q4_split_q8_1 (the current SoA Q4 split layout):
//   * SoA Q4_SPLIT has the scales stream at row offset [0, 2*nb) and the
//     nibble stream at row offset [2*nb, 18*nb). For an FFN matvec with
//     in_dim=4096 (nb=128) the two streams are separated by 256 B; for
//     ffn_down with in_dim=12288 (nb=384) they sit 768 B apart -- well
//     beyond an L1 sector (32 B) and a TLB line. Each block's CTA pair of
//     loads (scale + nibbles) therefore burns two distinct sectors.
//   * Tile-grouped colocates scale + nibbles for 8 consecutive blocks
//     within 144 contiguous bytes. With THREADS_PER_BLOCK=256 and tile
//     reuse across the block, the L1 sector footprint per matvec is
//     materially smaller and prefetch streams are unified.
//   * Same nibble alignment guarantees: nibble stream offset within a tile
//     is 16, which is 4-byte aligned, so native int* loads survive.
//   * Identical byte density (0.5625 B/element) -- the optimization is L1
//     locality + reduced TLB pressure, not byte count.
//
// vs matvec_q4_aligned_q8_1 (20-byte AoS blocks):
//   * 10% fewer bytes per tile (144 vs 160 per 8 blocks).
//   * Same alignment benefits for nibble loads.
//
// Each CTA processes NR=4 output rows. With THREADS_PER_BLOCK=256:
//   - 256 threads stride through `nb` blocks linearly.
//   - For ffn_up/gate (in_dim=4096 -> nb=128), each thread handles 0-1 blocks.
//   - For ffn_down (in_dim=12288 -> nb=384), each thread handles 1-2 blocks.
//
// Two kernels:
//   1. matvec_q4_tile_q8_1:          W*x -> out
//   2. matvec_q4_tile_q8_1_residual: W*x + residual -> out
//
// Alignment contract enforced by the host:
//   * `nb` (= in_dim / 32) MUST be a multiple of 8 (one tile = 8 blocks).
//     Every Qwen3.5-9B model dim today satisfies this:
//       hidden=4096   -> nb=128 (16 tiles)
//       inter=12288   -> nb=384 (48 tiles)
//       head_dim=256  -> nb=8   (1  tile)
//       kv_dim=1024   -> nb=32  (4  tiles)
//   * `cudaMalloc` returns 256-byte aligned base pointers; per-row base
//     inherits 16-byte alignment because 144 % 16 = 0 only when num_tiles
//     is even, but row_bytes = num_tiles * 144 is at minimum 144 B aligned
//     within the global allocation, and the per-row base is 16-byte aligned
//     because the allocation base is 256-byte aligned and we lay rows out
//     contiguously. The scale lookup uses 2-byte halfword loads (no SIMD
//     load width assumption), and the nibble lookup uses 4-byte aligned
//     int* loads (offset 16 + slot*16, always multiple of 4).
//
// NVRTC-compatible: no system includes, extern "C" linkage.
// Requires compute capability >= 6.1 for dp4a_s32() (Pascal+).
// ==========================================================================

#define NR       4     // rows per thread block (same as matvec_q4_split_q8_1)
#define NW       32    // warp size
#define THREADS_PER_BLOCK 256  // 8 warps
#define NWARPS   (THREADS_PER_BLOCK / NW)  // 8
#define Q4_BLOCK_SIZE     32   // elements per Q4_0 block
#define Q4_TILE_BLOCKS    8    // blocks per tile
#define Q4_TILE_ELEMENTS  256  // elements per tile (32 * 8)
#define Q4_TILE_BYTES     144  // 16 (scales) + 128 (nibbles) per tile
#define Q4_TILE_NIBBLE_OFF 16  // offset of nibble stream within a tile
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

// Unpack 4 nibble bytes (packed in an int32) for GGML de-interleaved Q4_0.
// Output: 2 int32 words for dp4a (unsigned nibbles 0-15; zero-point handled
// in the accumulation formula via x_sum).
__device__ __forceinline__ void unpack_nibbles_4bytes_deinterleaved(
    unsigned int packed, int &out_lo, int &out_hi)
{
    unsigned int lo = packed & 0x0F0F0F0Fu;
    unsigned int hi = (packed >> 4) & 0x0F0F0F0Fu;
    out_lo = (int)lo;
    out_hi = (int)hi;
}

// ==========================================================================
// Kernel 1: tile-grouped Q4 weight x Q8_1 input -> F32 output (dp4a, NR=4).
//
// Grid:  (ceil(out_dim / NR), 1, 1)
// Block: (256, 1, 1)
//
// Each thread iterates over Q4 blocks ib = tid, tid+256, tid+512, ...
// For block ib, the enclosing tile index is `tile = ib >> 3` and the slot
// within the tile is `slot = ib & 7`. Scales for the tile sit at tile*144
// (16 contiguous bytes); nibbles at tile*144 + 16 + slot*16 (4-byte aligned).
// ==========================================================================
extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 1) void matvec_q4_tile_q8_1(
    const char* __restrict__ weight_q4_tile,     // [out_dim * num_tiles * 144]
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
    unsigned long long row_bytes = (unsigned long long)num_tiles
                                 * (unsigned long long)Q4_TILE_BYTES;

    float sumf[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) sumf[r] = 0.0f;

    for (unsigned int ib = threadIdx.x; ib < nb; ib += THREADS_PER_BLOCK) {

        // --- Load Q8_1 input block (shared across NR rows) ---
        const char* x_block = input_q8_1 + (unsigned long long)ib * Q8_1_BYTES;

        unsigned short x_scale_bits = *(const unsigned short*)x_block;
        float x_scale = f16_bits_to_f32(x_scale_bits);

        unsigned short x_sum_bits = *(const unsigned short*)(x_block + 2);
        float x_sum = f16_bits_to_f32(x_sum_bits);

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

            const char* row_base = weight_q4_tile
                + (unsigned long long)(r0 + row) * row_bytes;
            const char* tile_base = row_base
                + (unsigned long long)tile_idx * Q4_TILE_BYTES;

            // Scale lookup: 8 f16 scales at tile_base[0..15]. Block slot's
            // scale is at offset slot * 2 (2 bytes per scale).
            const unsigned short* scales_u16 = (const unsigned short*)tile_base;
            float w_scale = f16_bits_to_f32(scales_u16[slot]);

            // Nibble lookup: 8 x 16 nibble bytes start at tile_base + 16.
            // Block slot's nibbles at offset slot * 16. 4-byte aligned
            // because tile_base is 16-byte aligned (cudaMalloc + 144-byte
            // tile stride keeps row_base 16B-aligned only when num_tiles is
            // even; for odd num_tiles the per-row base alignment falls to
            // 8 B, still sufficient for 4-byte int* loads at the +16+slot*16
            // offset).
            const unsigned int* w_nibbles = (const unsigned int*)(tile_base
                + Q4_TILE_NIBBLE_OFF
                + (unsigned long long)slot * 16ULL);

            int acc = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                unsigned int packed = w_nibbles[k];
                int w_lo, w_hi;
                unpack_nibbles_4bytes_deinterleaved(packed, w_lo, w_hi);
                acc = dp4a_s32(w_lo, xv[k],     acc);
                acc = dp4a_s32(w_hi, xv[k + 4], acc);
            }

            // Combined result with zero-point correction:
            //   dot(w, x) = w_scale * x_scale * dp4a_sum - 8 * w_scale * x_sum
            sumf[row] += w_scale * (x_scale * (float)acc - 8.0f * x_sum);
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
// Kernel 2: tile-grouped Q4 weight x Q8_1 input + residual -> F32 output.
//
// Same structure as matvec_q4_tile_q8_1 with a fused residual add at the
// final write step. Used for Wo (attention output) and down projection.
// ==========================================================================
extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 1) void matvec_q4_tile_q8_1_residual(
    const char* __restrict__ weight_q4_tile,
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
    unsigned long long row_bytes = (unsigned long long)num_tiles
                                 * (unsigned long long)Q4_TILE_BYTES;

    float sumf[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) sumf[r] = 0.0f;

    for (unsigned int ib = threadIdx.x; ib < nb; ib += THREADS_PER_BLOCK) {

        const char* x_block = input_q8_1 + (unsigned long long)ib * Q8_1_BYTES;

        unsigned short x_scale_bits = *(const unsigned short*)x_block;
        float x_scale = f16_bits_to_f32(x_scale_bits);

        unsigned short x_sum_bits = *(const unsigned short*)(x_block + 2);
        float x_sum = f16_bits_to_f32(x_sum_bits);

        const int* x_packed = (const int*)(x_block + 4);
        int xv[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) xv[k] = x_packed[k];

        unsigned int tile_idx = ib >> 3;
        unsigned int slot     = ib & 0x7;

        #pragma unroll
        for (int row = 0; row < NR; row++) {
            if (r0 + row >= out_dim) break;

            const char* row_base = weight_q4_tile
                + (unsigned long long)(r0 + row) * row_bytes;
            const char* tile_base = row_base
                + (unsigned long long)tile_idx * Q4_TILE_BYTES;

            const unsigned short* scales_u16 = (const unsigned short*)tile_base;
            float w_scale = f16_bits_to_f32(scales_u16[slot]);

            const unsigned int* w_nibbles = (const unsigned int*)(tile_base
                + Q4_TILE_NIBBLE_OFF
                + (unsigned long long)slot * 16ULL);

            int acc = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                unsigned int packed = w_nibbles[k];
                int w_lo, w_hi;
                unpack_nibbles_4bytes_deinterleaved(packed, w_lo, w_hi);
                acc = dp4a_s32(w_lo, xv[k],     acc);
                acc = dp4a_s32(w_hi, xv[k + 4], acc);
            }

            sumf[row] += w_scale * (x_scale * (float)acc - 8.0f * x_sum);
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
