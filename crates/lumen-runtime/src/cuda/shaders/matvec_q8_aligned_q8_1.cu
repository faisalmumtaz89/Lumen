// ==========================================================================
// Optimal Q8_0 Decode Kernel: Aligned Weights + Pre-Quantized Q8_1 Input
//
// Combines ALL proven optimizations:
//   1. Q8Aligned weights (36-byte blocks): native int* loads (8 instructions)
//      instead of byte-level manual packing (56 instructions per block)
//   2. Pre-quantized Q8_1 input: native int* loads, zero per-call quantization
//   3. dp4a INT8 dot product: 4 multiply-accumulates per instruction
//   4. NR=2: 2 output rows per block, 2x x-vector bandwidth amortization
//
// This is the product of matvec_q8_0_aligned.cu (aligned weights, NR=2,
// on-the-fly x quantization) and matvec_dp4a_q8_1.cu (pre-quantized Q8_1
// input, dp4a, unaligned weights). By combining both, we eliminate:
//   - 56 byte-load+shift instructions per weight block (aligned int* loads)
//   - On-the-fly x quantization (32 float loads + absmax + 32 rounding ops)
//   - Redundant x-vector bandwidth (NR=2 shares quantized x across 2 rows)
//
// Two kernels:
//   1. matvec_q8_aligned_q8_1:          W*x -> out
//   2. matvec_q8_aligned_q8_1_residual: W*x + residual -> out
//
// The quantize_f32_to_q8_1 kernel from matvec_dp4a_q8_1.cu is reused as-is.
// It runs once per token; the Q8_1 buffer is shared across all matvec calls.
//
// Block layout (both weight and input):
//   Q8Aligned (36 bytes): [f16 scale][2B pad][32 x int8]  quants at +4
//   Q8_1      (36 bytes): [f16 scale][f16 sum][32 x int8] quants at +4
//   Both have quant data at offset 4 (4-byte aligned), both are 36 bytes.
//
// dp4a dot product (Q8_0 has zero-point = 0):
//   For each aligned block pair:
//     int_dot = dp4a(w_int[0..7], x_int[0..7], 0)  -- 8 dp4a calls
//     partial = w_scale * x_scale * (float)int_dot
//   The Q8_1 sum field (bytes 2-3) is unused because Q8_0 has zero_point=0.
//
// Architecture: NR=2 rows per block, 128 threads (4 warps).
// Cross-warp reduction: simple shmem (3 warps write, warp 0 sums).
// in_dim must be a multiple of 32 (Q8_0 block size).
// Requires compute capability >= 6.1 for __dp4a() (Pascal+).
//
// NVRTC-compatible: no system includes, extern "C" linkage.
// ==========================================================================

#define NR       2     // rows per thread block
#define NW       32    // warp size
#define THREADS_PER_BLOCK 128  // 4 warps
#define NWARPS   (THREADS_PER_BLOCK / NW)  // 4
#define Q8_BLOCK_SIZE     32   // elements per Q8 block
#define Q8_ALIGNED_BYTES  36   // 2B f16 scale + 2B pad + 32B int8 data
#define Q8_1_BYTES        36   // 2B f16 scale + 2B f16 sum + 32B int8 data

// Hardware f16->f32 conversion via PTX (single instruction on SM 53+).
__device__ __forceinline__ float f16_bits_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
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
// Kernel 1: Q8Aligned weight x Q8_1 input -> F32 output (dp4a, NR=2).
//
// Grid:  (ceil(out_dim / NR), 1, 1)  -- one block per NR rows
// Block: (128, 1, 1)                 -- 4 warps x 32 threads
//
// Both weight and input have quant data at offset +4 (4-byte aligned),
// enabling native int* loads on BOTH sides. Zero byte-packing overhead.
// ==========================================================================
extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 2) void matvec_q8_aligned_q8_1(
    const char* __restrict__ weight_q8_aligned,  // [out_dim * nb * 36] Q8Aligned bytes
    const char* __restrict__ input_q8_1,         // [nb * 36] Q8_1 pre-quantized input
    float* __restrict__ out,                     // [out_dim] F32 output
    unsigned int out_dim,
    unsigned int in_dim)
{
    unsigned int r0 = blockIdx.x * NR;  // first output row for this block

    unsigned int warp_id = threadIdx.x / NW;
    unsigned int lane    = threadIdx.x % NW;

    unsigned int nb = in_dim >> 5;  // number of Q8 blocks per row (in_dim / 32)
    unsigned long long row_bytes = (unsigned long long)nb * Q8_ALIGNED_BYTES;

    // Per-row accumulators.
    float sumf[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) sumf[r] = 0.0f;

    // Main loop: each thread handles 1 Q8 block pair per iteration,
    // striding by THREADS_PER_BLOCK (128) blocks.
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

        // --- Process NR output rows with weight data from global memory ---
        #pragma unroll
        for (int row = 0; row < NR; row++) {
            if (r0 + row >= out_dim) break;

            const char* w_block = weight_q8_aligned
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * Q8_ALIGNED_BYTES;

            // Read f16 weight scale (bytes 0-1).
            unsigned short w_scale_bits = (unsigned short)(unsigned char)w_block[0]
                                        | ((unsigned short)(unsigned char)w_block[1] << 8);
            float w_scale = f16_bits_to_f32(w_scale_bits);

            // Native int* loads for weight quant data (4-byte aligned at +4).
            const int* w_packed = (const int*)(w_block + 4);

            // dp4a dot product: 8 x int32 pairs = 32 x int8 multiply-accumulates.
            int acc = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                acc = __dp4a(w_packed[k], xv[k], acc);
            }

            // Combined scale: w_scale * x_scale * int_dot_product.
            sumf[row] += w_scale * x_scale * (float)acc;
        }
    }

    // --- Cross-warp reduction via simple shmem (3 warps write, warp 0 sums) ---
    #pragma unroll
    for (int r = 0; r < NR; r++) {
        sumf[r] = warp_reduce_sum(sumf[r]);
    }

    // shmem for cross-warp reduction: (NWARPS-1) * NR floats.
    // Warps 1-3 write their partial sums; warp 0 reads and accumulates.
    __shared__ float shmem[(NWARPS - 1) * NR];

    if (warp_id > 0 && lane == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            shmem[(warp_id - 1) * NR + r] = sumf[r];
        }
    }
    __syncthreads();

    // Warp 0, lane 0: accumulate partials from warps 1-3 and write output.
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
// Kernel 2: Q8Aligned weight x Q8_1 input + residual -> F32 output (dp4a, NR=2).
//
// Same as matvec_q8_aligned_q8_1 but with fused residual addition at final write.
// Used for Wo (attention output) and down projection.
//
// Grid:  (ceil(out_dim / NR), 1, 1)  -- one block per NR rows
// Block: (128, 1, 1)                 -- 4 warps x 32 threads
// ==========================================================================
extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 2) void matvec_q8_aligned_q8_1_residual(
    const char* __restrict__ weight_q8_aligned,  // [out_dim * nb * 36] Q8Aligned bytes
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
    unsigned long long row_bytes = (unsigned long long)nb * Q8_ALIGNED_BYTES;

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

            const char* w_block = weight_q8_aligned
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * Q8_ALIGNED_BYTES;

            unsigned short w_scale_bits = (unsigned short)(unsigned char)w_block[0]
                                        | ((unsigned short)(unsigned char)w_block[1] << 8);
            float w_scale = f16_bits_to_f32(w_scale_bits);

            const int* w_packed = (const int*)(w_block + 4);

            int acc = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                acc = __dp4a(w_packed[k], xv[k], acc);
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
                out[r0 + r] = total + residual[r0 + r];
            }
        }
    }
}
