// ==========================================================================
// Q4_0 dp4a Decode Kernel: Native Q4_0 Weights + Pre-Quantized Q8_1 Input
//
// Combines:
//   1. Native Q4_0 weights (18-byte blocks): no repacking needed
//   2. Pre-quantized Q8_1 input: native int* loads, zero per-call quantization
//   3. dp4a INT8 dot product: 4 multiply-accumulates per instruction
//   4. NR=2: 2 output rows per block, 2x x-vector bandwidth amortization
//
// Q4_0 block layout (18 bytes per 32 elements):
//   bytes [0..1]: f16 scale (d)
//   bytes [2..17]: 16 bytes of packed nibble pairs
//     byte[i] (i=0..15): lo_nibble = element[2*i], hi_nibble = element[2*i+1]
//   Dequantized value: scale * (nibble - 8)
//
// Q8_1 block layout (36 bytes per 32 elements):
//   bytes [0..1]: f16 scale (d)
//   bytes [2..3]: f16 weighted sum (s = d * sum(quants))
//   bytes [4..35]: 32 x int8 quantized values (sequential order)
//
// dp4a dot product for Q4_0 x Q8_1:
//   For each Q4_0 block and corresponding Q8_1 block:
//     1. Unpack 16 nibble bytes into 8 int32 words (4 sequential int8 per word)
//     2. Load 8 int32 words from Q8_1 quant data
//     3. dp4a: 8 calls = 32 multiply-accumulates
//     4. result += w_scale * x_scale * dp4a_sum - 8 * w_scale * x_sum
//        where x_sum = d * sum(quants) from Q8_1 block (bytes 2-3)
//
// The zero-point correction (-8):
//   dot(w, x) = scale * sum(nibble_i * x_quant_i) - 8 * scale * sum(x_quant_i)
//             = scale * sum(nibble_i * x_quant_i) - 8 * x_sum / x_scale * w_scale * x_scale
//   Simplification: partial = w_scale * x_scale * dp4a_sum
//                   correction = w_scale * 8.0f * x_sum
//                   (x_sum = f16_to_f32(Q8_1 bytes [2..3]) = d * sum(quants))
//   result += partial - correction
//
// Architecture: NR=2 rows per block, 128 threads (4 warps).
// Requires compute capability >= 6.1 for __dp4a() (Pascal+).
// in_dim must be a multiple of 32 (Q4_0 block size).
//
// Two kernels:
//   1. matvec_q4_0_dp4a:          W*x -> out
//   2. matvec_q4_0_dp4a_residual: W*x + residual -> out
//
// The quantize_f32_to_q8_1 kernel from matvec_dp4a_q8_1.cu is reused as-is.
// It runs once per token; the Q8_1 buffer is shared across all matvec calls.
//
// NVRTC-compatible: no system includes, extern "C" linkage.
// ==========================================================================

#define NR       2     // rows per thread block
#define NW       32    // warp size
#define THREADS_PER_BLOCK 128  // 4 warps
#define NWARPS   (THREADS_PER_BLOCK / NW)  // 4
#define Q4_BLOCK_SIZE     32   // elements per Q4_0 block
#define Q4_BLOCK_BYTES    18   // 2B f16 scale + 16B nibble data
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

// Unpack 2 nibble bytes into 4 sequential signed int8 values packed in an int32.
//
// Input: 2 bytes from Q4_0 nibble data (byte_lo, byte_hi).
//   byte_lo: lo_nibble = element[2*i],     hi_nibble = element[2*i + 1]
//   byte_hi: lo_nibble = element[2*i + 2], hi_nibble = element[2*i + 3]
//
// Output: int32 = {elem[2*i]-8, elem[2*i+1]-8, elem[2*i+2]-8, elem[2*i+3]-8}
//         as 4 packed signed int8 in little-endian order.
//
// This matches Q8_1's sequential int8 layout for dp4a.
// Uses vectorized mask+shift+byte_perm for ~10 ops vs ~20 scalar ops.
__device__ __forceinline__ int unpack_nibbles_2bytes(unsigned char b0, unsigned char b1) {
    // Combine 2 bytes into a single uint, then use vectorized nibble extraction.
    unsigned int packed = (unsigned int)b0 | ((unsigned int)b1 << 8);
    unsigned int lo = packed & 0x0F0Fu;           // lo nibbles of both bytes
    unsigned int hi = (packed >> 4) & 0x0F0Fu;    // hi nibbles of both bytes
    // Interleave: {b0_lo, b0_hi, b1_lo, b1_hi} using byte_perm.
    // lo=bytes 0-3 (only 0,1 valid), hi=bytes 4-7 (only 4,5 valid).
    unsigned int interleaved = __byte_perm(lo, hi, 0x5140);
    // Subtract zero-point (8) from all 4 bytes simultaneously.
    return (int)(interleaved - 0x08080808u);
}

// ==========================================================================
// Kernel 1: Q4_0 weight x Q8_1 input -> F32 output (dp4a, NR=2).
//
// Grid:  (ceil(out_dim / NR), 1, 1)  -- one block per NR rows
// Block: (128, 1, 1)                 -- 4 warps x 32 threads
// ==========================================================================
extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 2) void matvec_q4_0_dp4a(
    const char* __restrict__ weight_q4,    // [out_dim * nb * 18] raw Q4_0 bytes
    const char* __restrict__ input_q8_1,   // [nb * 36] Q8_1 pre-quantized input
    float* __restrict__ out,               // [out_dim] F32 output
    unsigned int out_dim,
    unsigned int in_dim)
{
    unsigned int r0 = blockIdx.x * NR;  // first output row for this block

    unsigned int warp_id = threadIdx.x / NW;
    unsigned int lane    = threadIdx.x % NW;

    unsigned int nb = in_dim >> 5;  // number of blocks per row (in_dim / 32)
    unsigned long long row_bytes = (unsigned long long)nb * Q4_BLOCK_BYTES;

    // Per-row accumulators
    float sumf[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) sumf[r] = 0.0f;

    // Main loop: each thread handles one Q4_0/Q8_1 block pair per iteration,
    // striding by THREADS_PER_BLOCK (128) blocks.
    for (unsigned int ib = threadIdx.x; ib < nb; ib += THREADS_PER_BLOCK) {
        // --- Load Q8_1 input block (36 bytes, quant data at +4) ---
        const char* x_block = input_q8_1 + (unsigned long long)ib * Q8_1_BYTES;

        // Read f16 input scale (bytes 0-1, little-endian).
        unsigned short x_scale_bits = (unsigned short)(unsigned char)x_block[0]
                                    | ((unsigned short)(unsigned char)x_block[1] << 8);
        float x_scale = f16_bits_to_f32(x_scale_bits);

        // Read f16 weighted sum (bytes 2-3): s = d * sum(quants).
        // Used for the -8 zero-point correction.
        unsigned short x_sum_bits = (unsigned short)(unsigned char)x_block[2]
                                  | ((unsigned short)(unsigned char)x_block[3] << 8);
        float x_sum = f16_bits_to_f32(x_sum_bits);

        // Native int* load for input quant data (4-byte aligned at +4).
        const int* x_packed = (const int*)(x_block + 4);

        // Preload 8 packed int32 words (32 int8 values) from Q8_1 input.
        int xv[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) xv[k] = x_packed[k];

        // --- Process NR output rows with the same x-values ---
        #pragma unroll
        for (int row = 0; row < NR; row++) {
            if (r0 + row >= out_dim) break;

            // Pointer to this row's Q4_0 weight block.
            const char* w_block = weight_q4
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * Q4_BLOCK_BYTES;

            // Read f16 weight scale (bytes 0-1).
            unsigned short w_scale_bits = (unsigned short)(unsigned char)w_block[0]
                                        | ((unsigned short)(unsigned char)w_block[1] << 8);
            float w_scale = f16_bits_to_f32(w_scale_bits);

            // Nibble data starts at w_block + 2 (16 bytes = 32 nibbles).
            const unsigned char* qs = (const unsigned char*)(w_block + 2);

            // Unpack 16 nibble bytes into 8 int32 words (4 sequential elements each)
            // and dp4a against the corresponding Q8_1 int32 words.
            int acc = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                int w_packed = unpack_nibbles_2bytes(qs[2 * k], qs[2 * k + 1]);
                acc = __dp4a(w_packed, xv[k], acc);
            }

            // Combined result with zero-point correction:
            //   dot(w, x) = w_scale * x_scale * dp4a_sum - 8 * w_scale * x_sum
            // where x_sum = d * sum(quants) already includes x_scale.
            sumf[row] += w_scale * (x_scale * (float)acc - 8.0f * x_sum);
        }
    }

    // --- Cross-warp reduction via simple shmem (3 warps write, warp 0 sums) ---
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
// Kernel 2: Q4_0 weight x Q8_1 input + residual -> F32 output (dp4a, NR=2).
//
// Same as matvec_q4_0_dp4a but with fused residual addition at final write.
// Used for Wo (attention output) and down projection.
//
// Grid:  (ceil(out_dim / NR), 1, 1)  -- one block per NR rows
// Block: (128, 1, 1)                 -- 4 warps x 32 threads
// ==========================================================================
extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 2) void matvec_q4_0_dp4a_residual(
    const char* __restrict__ weight_q4,    // [out_dim * nb * 18] raw Q4_0 bytes
    const char* __restrict__ input_q8_1,   // [nb * 36] Q8_1 pre-quantized input
    const float* __restrict__ residual,    // [out_dim] F32 residual
    float* __restrict__ out,               // [out_dim] F32 output
    unsigned int out_dim,
    unsigned int in_dim)
{
    unsigned int r0 = blockIdx.x * NR;

    unsigned int warp_id = threadIdx.x / NW;
    unsigned int lane    = threadIdx.x % NW;

    unsigned int nb = in_dim >> 5;
    unsigned long long row_bytes = (unsigned long long)nb * Q4_BLOCK_BYTES;

    float sumf[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) sumf[r] = 0.0f;

    for (unsigned int ib = threadIdx.x; ib < nb; ib += THREADS_PER_BLOCK) {
        // Load Q8_1 input block.
        const char* x_block = input_q8_1 + (unsigned long long)ib * Q8_1_BYTES;

        unsigned short x_scale_bits = (unsigned short)(unsigned char)x_block[0]
                                    | ((unsigned short)(unsigned char)x_block[1] << 8);
        float x_scale = f16_bits_to_f32(x_scale_bits);

        unsigned short x_sum_bits = (unsigned short)(unsigned char)x_block[2]
                                  | ((unsigned short)(unsigned char)x_block[3] << 8);
        float x_sum = f16_bits_to_f32(x_sum_bits);

        const int* x_packed = (const int*)(x_block + 4);

        int xv[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) xv[k] = x_packed[k];

        // Process NR output rows.
        #pragma unroll
        for (int row = 0; row < NR; row++) {
            if (r0 + row >= out_dim) break;

            const char* w_block = weight_q4
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * Q4_BLOCK_BYTES;

            unsigned short w_scale_bits = (unsigned short)(unsigned char)w_block[0]
                                        | ((unsigned short)(unsigned char)w_block[1] << 8);
            float w_scale = f16_bits_to_f32(w_scale_bits);

            const unsigned char* qs = (const unsigned char*)(w_block + 2);

            int acc = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                int w_packed = unpack_nibbles_2bytes(qs[2 * k], qs[2 * k + 1]);
                acc = __dp4a(w_packed, xv[k], acc);
            }

            sumf[row] += w_scale * (x_scale * (float)acc - 8.0f * x_sum);
        }
    }

    // Cross-warp reduction via simple shmem.
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
