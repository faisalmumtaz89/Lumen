// ==========================================================================
// Q8 split layout matvec against pre-quantized Q8_1 input (dp4a, NR=2).
//
// Sibling of matvec_q8_aligned_q8_1 that consumes the split (SoA) per-row
// layout produced by repack_q8_aligned_to_split:
//
//   Per row (in `nb`-block units):
//     [f16 scale * nb][int8 quant[32] * nb]
//   Row stride: 2*nb + 32*nb = 34*nb bytes (vs 36*nb in the AoS layout).
//
// Hypothesis: the AoS layout transports 2 padding bytes per block (1 byte per
// element / 32 = 6.25% overhead). Removing those bytes drops per-row weight
// traffic from 36*nb to 34*nb (~5.56% fewer bytes per row). Q8_0 decode is
// bandwidth-bound at the memory-system ceiling, so the byte-for-byte saving
// is expected to translate into ~5-8% end-to-end speedup.
//
// Two kernels:
//   1. matvec_q8_split_q8_1:          W*x -> out
//   2. matvec_q8_split_q8_1_residual: W*x + residual -> out
//
// Structure (delta from matvec_q8_aligned_q8_1):
//   * Same NR=2 rows per CTA, 128 threads, 4 warps, deferred shmem reduction.
//   * Same Q8_1 input layout (36-byte blocks, quants at offset +4).
//   * Weight scale is read from a contiguous scales stream: w_scale_bits =
//     row_scales[ib] (2 bytes, 2-byte aligned via the per-block array).
//   * Weight quants are read as 8 native int* loads from a contiguous quants
//     stream: row_quants + ib*32 (always 4-byte aligned when nb is even).
//
// Alignment contract enforced by the host:
//   * `nb` MUST be even so each row stride 34*nb is a multiple of 4. Every
//     model dim shipped today (hidden=4096, kv_dim=1024, inter=12288, etc.)
//     satisfies this — nb is a power of two for those shapes.
//   * `cudaMalloc` returns 256-byte aligned base pointers, so row_scales and
//     row_quants inherit at least 4-byte alignment.
//
// Requires compute capability >= 6.1 for __dp4a() (Pascal+).
//
// NVRTC-compatible: no system includes, extern "C" linkage.
// ==========================================================================

#define NR       2     // rows per thread block
#define NW       32    // warp size
#define THREADS_PER_BLOCK 128  // 4 warps
#define NWARPS   (THREADS_PER_BLOCK / NW)  // 4
#define Q8_BLOCK_SIZE     32   // elements per Q8 block
#define Q8_1_BYTES        36   // 2B f16 scale + 2B f16 sum + 32B int8 data

// Hardware f16->f32 conversion via PTX (single instruction on SM 53+).
__device__ __forceinline__ float f16_bits_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

// inline-PTX dp4a wrapper. The `__dp4a` intrinsic in this build
// environment (driver 580.126.20 / CUDA 12.2 / sm_80) fails NVRTC PTX-load
// with `Unresolved extern function '_Z6__dp4aiii'` causing the kernel to be
// dropped at compile_and_load. The inline `dp4a.s32.s32` opcode is the
// header-independent equivalent and loads cleanly on compute_80.
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
// Kernel 1: split Q8 weight x Q8_1 input -> F32 output (dp4a, NR=2).
//
// Grid:  (ceil(out_dim / NR), 1, 1)
// Block: (128, 1, 1)
// ==========================================================================
extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 2) void matvec_q8_split_q8_1(
    const char* __restrict__ weight_q8_split,    // [out_dim * nb * 34] split bytes
    const char* __restrict__ input_q8_1,         // [nb * 36] Q8_1 pre-quantized input
    float* __restrict__ out,                     // [out_dim] F32 output
    unsigned int out_dim,
    unsigned int in_dim)
{
    unsigned int r0 = blockIdx.x * NR;

    unsigned int warp_id = threadIdx.x / NW;
    unsigned int lane    = threadIdx.x % NW;

    unsigned int nb = in_dim >> 5;  // in_dim / 32
    unsigned long long row_bytes = (unsigned long long)nb * 34ULL;
    unsigned long long scales_bytes_per_row = (unsigned long long)nb * 2ULL;

    float sumf[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) sumf[r] = 0.0f;

    for (unsigned int ib = threadIdx.x; ib < nb; ib += THREADS_PER_BLOCK) {

        // --- Load Q8_1 input block (shared across NR rows) ---
        const char* x_block = input_q8_1 + (unsigned long long)ib * Q8_1_BYTES;

        unsigned short x_scale_bits = (unsigned short)(unsigned char)x_block[0]
                                    | ((unsigned short)(unsigned char)x_block[1] << 8);
        float x_scale = f16_bits_to_f32(x_scale_bits);

        const int* x_packed = (const int*)(x_block + 4);
        int xv[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) xv[k] = x_packed[k];

        // --- Process NR output rows ---
        #pragma unroll
        for (int row = 0; row < NR; row++) {
            if (r0 + row >= out_dim) break;

            const char* row_base = weight_q8_split
                + (unsigned long long)(r0 + row) * row_bytes;

            // Scale stream: nb x f16 starting at offset 0 in the row.
            // Read 2 bytes for block `ib`.
            const char* scale_byte = row_base + (unsigned long long)ib * 2ULL;
            unsigned short w_scale_bits = (unsigned short)(unsigned char)scale_byte[0]
                                        | ((unsigned short)(unsigned char)scale_byte[1] << 8);
            float w_scale = f16_bits_to_f32(w_scale_bits);

            // Quant stream: nb x 32B starting at offset 2*nb in the row.
            // Native int* loads — 4-byte aligned because (row_bytes % 4 == 0) when
            // nb is even and (2*nb % 4 == 0) when nb is even.
            const int* w_packed = (const int*)(row_base
                + scales_bytes_per_row
                + (unsigned long long)ib * 32ULL);

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
// Kernel 2: split Q8 weight x Q8_1 input + residual -> F32 output.
//
// Same structure as matvec_q8_split_q8_1 with a fused residual add at the
// final write step. Used for Wo (attention output) and down projection.
// ==========================================================================
extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 2) void matvec_q8_split_q8_1_residual(
    const char* __restrict__ weight_q8_split,    // [out_dim * nb * 34] split bytes
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
    unsigned long long row_bytes = (unsigned long long)nb * 34ULL;
    unsigned long long scales_bytes_per_row = (unsigned long long)nb * 2ULL;

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

        #pragma unroll
        for (int row = 0; row < NR; row++) {
            if (r0 + row >= out_dim) break;

            const char* row_base = weight_q8_split
                + (unsigned long long)(r0 + row) * row_bytes;

            const char* scale_byte = row_base + (unsigned long long)ib * 2ULL;
            unsigned short w_scale_bits = (unsigned short)(unsigned char)scale_byte[0]
                                        | ((unsigned short)(unsigned char)scale_byte[1] << 8);
            float w_scale = f16_bits_to_f32(w_scale_bits);

            const int* w_packed = (const int*)(row_base
                + scales_bytes_per_row
                + (unsigned long long)ib * 32ULL);

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
