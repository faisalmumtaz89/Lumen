// ==========================================================================
// Q4 split layout matvec against pre-quantized Q8_1 input (dp4a, NR=4).
//
// Sibling of matvec_q4_aligned_q8_1 that consumes the split (SoA) per-row
// layout produced by repack_q4_raw_to_split:
//
//   Per row (in `nb`-block units):
//     [f16 scale * nb][nibble[16] * nb]
//   Row stride: 2*nb + 16*nb = 18*nb bytes
//   (same density as Q4Raw, vs 20*nb in the Q4Aligned AoS layout).
//
// Hypothesis: the Q4Aligned AoS layout transports 2 padding bytes per block
// (1 byte per 16 elements / block = 12.5% overhead relative to the 16-byte
// nibble payload). The split layout removes the padding while preserving the
// 4-byte alignment of the nibble stream (which lives at row offset 2*nb,
// guaranteed aligned because `nb` is even on every shipped model dim).
//
// On Q4_0 the absolute byte savings are smaller in tok/s than the Q8 split
// (~10% fewer payload bytes vs Q4Aligned), but the saving stacks with the
// existing dp4a NR=4 micro-architecture. Expected: +3-5% end-to-end decode.
//
// Two kernels:
//   1. matvec_q4_split_q8_1:          W*x -> out
//   2. matvec_q4_split_q8_1_residual: W*x + residual -> out
//
// Structure (delta from matvec_q4_aligned_q8_1):
//   * Same NR=4 rows per CTA, 256 threads, 8 warps, deferred shmem reduction.
//   * Same Q8_1 input layout (36-byte blocks, quants at offset +4).
//   * Weight scale read from a contiguous scales stream: w_scale_bits =
//     row_scales[ib] (2 bytes, halfword load).
//   * Nibble data read as 4 native int* loads from a contiguous nibble stream:
//     row_nibbles + ib*16 (always 4-byte aligned when nb is even).
//   * Same de-interleaved nibble unpack and zero-point correction.
//
// Alignment contract enforced by the host:
//   * `nb` MUST be even so each row stride 18*nb is a multiple of 2, and the
//     nibble stream offset 2*nb is at least 4-byte aligned.
//   * `cudaMalloc` returns 256-byte aligned base pointers, so the nibble
//     stream inherits >=4-byte alignment from cudaMalloc + even nb.
//
// Requires compute capability >= 6.1 for __dp4a() (Pascal+).
//
// NVRTC-compatible: no system includes, extern "C" linkage.
// ==========================================================================

#define NR       4     // rows per thread block
#define NW       32    // warp size
#define THREADS_PER_BLOCK 256  // 8 warps
#define NWARPS   (THREADS_PER_BLOCK / NW)  // 8
#define Q4_BLOCK_SIZE     32   // elements per Q4_0 block
#define Q8_1_BYTES        36   // 2B f16 scale + 2B f16 sum + 32B int8 data

// Hardware f16->f32 conversion via PTX (single instruction on SM 53+).
__device__ __forceinline__ float f16_bits_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

// inline-PTX dp4a wrapper. See matvec_q8_split_q8_1.cu for the
// rationale (the `__dp4a` intrinsic NVRTC-fails in this build env).
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

// Unpack 4 nibble bytes (packed in an int32) for GGML de-interleaved Q4_0 layout.
// Output: 2 int32 words for dp4a (unsigned nibbles 0-15, zero-point handled in
// the accumulation formula via x_sum).
__device__ __forceinline__ void unpack_nibbles_4bytes_deinterleaved(
    unsigned int packed, int &out_lo, int &out_hi)
{
    unsigned int lo = packed & 0x0F0F0F0Fu;
    unsigned int hi = (packed >> 4) & 0x0F0F0F0Fu;
    out_lo = (int)lo;
    out_hi = (int)hi;
}

// ==========================================================================
// Kernel 1: split Q4 weight x Q8_1 input -> F32 output (dp4a, NR=4).
//
// Grid:  (ceil(out_dim / NR), 1, 1)
// Block: (256, 1, 1)
// ==========================================================================
extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 1) void matvec_q4_split_q8_1(
    const char* __restrict__ weight_q4_split,    // [out_dim * nb * 18] split bytes
    const char* __restrict__ input_q8_1,         // [nb * 36] Q8_1 pre-quantized input
    float* __restrict__ out,                     // [out_dim] F32 output
    unsigned int out_dim,
    unsigned int in_dim)
{
    unsigned int r0 = blockIdx.x * NR;

    unsigned int warp_id = threadIdx.x / NW;
    unsigned int lane    = threadIdx.x % NW;

    unsigned int nb = in_dim >> 5;  // in_dim / 32
    unsigned long long row_bytes = (unsigned long long)nb * 18ULL;
    unsigned long long scales_bytes_per_row = (unsigned long long)nb * 2ULL;

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

        const int* x_packed = (const int*)(x_block + 4);
        int xv[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) xv[k] = x_packed[k];

        // --- Process NR output rows ---
        #pragma unroll
        for (int row = 0; row < NR; row++) {
            if (r0 + row >= out_dim) break;

            const char* row_base = weight_q4_split
                + (unsigned long long)(r0 + row) * row_bytes;

            // Scale stream: nb x f16 starting at offset 0 in the row.
            const char* scale_byte = row_base + (unsigned long long)ib * 2ULL;
            unsigned short w_scale_bits = *(const unsigned short*)scale_byte;
            float w_scale = f16_bits_to_f32(w_scale_bits);

            // Nibble stream: nb x 16B starting at offset 2*nb in the row.
            // Native int* loads -- 4-byte aligned because 2*nb % 4 == 0 when
            // nb is even (guaranteed by all shipped model dims).
            const unsigned int* w_nibbles = (const unsigned int*)(row_base
                + scales_bytes_per_row
                + (unsigned long long)ib * 16ULL);

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
// Kernel 2: split Q4 weight x Q8_1 input + residual -> F32 output.
//
// Same structure as matvec_q4_split_q8_1 with a fused residual add at the
// final write step. Used for Wo (attention output) and down projection.
// ==========================================================================
extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 1) void matvec_q4_split_q8_1_residual(
    const char* __restrict__ weight_q4_split,    // [out_dim * nb * 18] split bytes
    const char* __restrict__ input_q8_1,         // [nb * 36] Q8_1 pre-quantized input
    const float* __restrict__ residual,          // [out_dim] F32 residual
    float* __restrict__ out,                     // [out_dim] F32 output
    unsigned int out_dim,
    unsigned int in_dim)
{
    unsigned int r0 = blockIdx.x * NR;

    unsigned int warp_id = threadIdx.x / NW;
    unsigned int lane    = threadIdx.x % NW;

    unsigned int nb = in_dim >> 5;
    unsigned long long row_bytes = (unsigned long long)nb * 18ULL;
    unsigned long long scales_bytes_per_row = (unsigned long long)nb * 2ULL;

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

        #pragma unroll
        for (int row = 0; row < NR; row++) {
            if (r0 + row >= out_dim) break;

            const char* row_base = weight_q4_split
                + (unsigned long long)(r0 + row) * row_bytes;

            const char* scale_byte = row_base + (unsigned long long)ib * 2ULL;
            unsigned short w_scale_bits = *(const unsigned short*)scale_byte;
            float w_scale = f16_bits_to_f32(w_scale_bits);

            const unsigned int* w_nibbles = (const unsigned int*)(row_base
                + scales_bytes_per_row
                + (unsigned long long)ib * 16ULL);

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
