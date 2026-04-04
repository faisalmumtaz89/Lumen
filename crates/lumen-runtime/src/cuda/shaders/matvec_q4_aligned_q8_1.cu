// ==========================================================================
// Q4_0 Aligned Decode Kernel: Aligned Q4_0 Weights + Pre-Quantized Q8_1 Input
//
// Combines ALL proven optimizations for Q4_0:
//   1. Q4Aligned weights (20-byte blocks): nibble data at offset +4 is
//      4-byte aligned, enabling int* loads (4 instructions loading 16 nibble
//      bytes) instead of 16 individual byte loads.
//   2. Pre-quantized Q8_1 input: native int* loads, zero per-call quantization
//   3. dp4a INT8 dot product: 4 multiply-accumulates per instruction
//   4. NR=4: 4 output rows per block, 4x x-vector bandwidth amortization
//
// Q4Aligned block layout (20 bytes per 32 elements):
//   bytes [0..1]:   f16 scale (d)
//   bytes [2..3]:   padding (alignment)
//   bytes [4..19]:  16 bytes of packed nibble pairs
//     byte[i] (i=0..15): lo_nibble = element[2*i], hi_nibble = element[2*i+1]
//   Dequantized value: scale * (nibble - 8)
//
// Q8_1 block layout (36 bytes per 32 elements):
//   bytes [0..1]: f16 scale (d)
//   bytes [2..3]: f16 weighted sum (s = d * sum(quants))
//   bytes [4..35]: 32 x int8 quantized values (sequential order)
//
// dp4a dot product for Q4_0 x Q8_1:
//   For each Q4Aligned block and corresponding Q8_1 block:
//     1. Load nibble data as 4 aligned int32 words (4 bytes each = 16 nibble bytes)
//     2. Unpack each int32 word into 2 dp4a-compatible int32 words (8 sequential int8)
//     3. Load 8 int32 words from Q8_1 quant data
//     4. dp4a: 8 calls = 32 multiply-accumulates
//     5. result += w_scale * x_scale * dp4a_sum - 8 * w_scale * x_sum
//
// Key improvement over matvec_q4_0_dp4a.cu:
//   - Aligned int* loads: 4 loads vs 16 byte loads for nibble data
//   - Nibble unpacking from int registers (no byte-gather from memory)
//   - Same dp4a core as the unaligned version
//
// Architecture: NR=4 rows per block, 256 threads (8 warps).
// Requires compute capability >= 6.1 for __dp4a() (Pascal+).
// in_dim must be a multiple of 32 (Q4_0 block size).
//
// Two kernels:
//   1. matvec_q4_aligned_q8_1:          W*x -> out
//   2. matvec_q4_aligned_q8_1_residual: W*x + residual -> out
//
// The quantize_f32_to_q8_1 kernel from matvec_dp4a_q8_1.cu is reused as-is.
// It runs once per token; the Q8_1 buffer is shared across all matvec calls.
//
// NVRTC-compatible: no system includes, extern "C" linkage.
// ==========================================================================

#define NR       4     // rows per thread block
#define NW       32    // warp size
#define THREADS_PER_BLOCK 256  // 8 warps
#define NWARPS   (THREADS_PER_BLOCK / NW)  // 8
#define Q4_BLOCK_SIZE     32   // elements per Q4_0 block
#define Q4_ALIGNED_BYTES  20   // 2B f16 scale + 2B pad + 16B nibble data
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

// Unpack 4 nibble bytes (packed in an int32) into 2 dp4a-compatible int32 words.
//
// Input: int32 containing 4 nibble bytes (b0 at bits 0-7, b1 at 8-15, b2 at 16-23, b3 at 24-31).
// Each byte has 2 nibbles: lo = element[2*i], hi = element[2*i+1].
//
// Output: 2 int32 words, each containing 4 sequential signed int8 (nibble - 8) values.
//   out0 = {b0.lo-8, b0.hi-8, b1.lo-8, b1.hi-8}
//   out1 = {b2.lo-8, b2.hi-8, b3.lo-8, b3.hi-8}
//
// Uses __byte_perm (PRMT instruction, SM 35+) for register-level byte rearrangement.
// 7 ops vs ~43 ops in the scalar version (6x fewer instructions).
__device__ __forceinline__ void unpack_nibbles_4bytes(unsigned int packed, int &out0, int &out1) {
    // Split each byte into lo and hi nibbles using vectorized masks.
    unsigned int lo = packed & 0x0F0F0F0Fu;           // lo nibbles in bytes 0-3
    unsigned int hi = (packed >> 4) & 0x0F0F0F0Fu;    // hi nibbles in bytes 0-3

    // Interleave lo/hi nibbles into sequential order using byte permute:
    //   __byte_perm(a, b, sel): a=bytes 0-3, b=bytes 4-7, sel picks 4 output bytes.
    //   0x5140: byte0=lo[0], byte1=hi[0](=4), byte2=lo[1](=1), byte3=hi[1](=5)
    //   0x7362: byte0=lo[2], byte1=hi[2](=6), byte2=lo[3](=3), byte3=hi[3](=7)
    unsigned int interleaved0 = __byte_perm(lo, hi, 0x5140);
    unsigned int interleaved1 = __byte_perm(lo, hi, 0x7362);

    // Subtract zero-point (8) from all 4 bytes simultaneously.
    // Each byte holds 0-15; after subtract, range is -8..+7 (fits signed int8).
    out0 = (int)(interleaved0 - 0x08080808u);
    out1 = (int)(interleaved1 - 0x08080808u);
}

// ==========================================================================
// Kernel 1: Q4Aligned weight x Q8_1 input -> F32 output (dp4a, NR=4).
//
// Grid:  (ceil(out_dim / NR), 1, 1)  -- one block per NR rows
// Block: (256, 1, 1)                 -- 8 warps x 32 threads
// ==========================================================================
extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 1) void matvec_q4_aligned_q8_1(
    const char* __restrict__ weight_q4a,   // [out_dim * nb * 20] Q4Aligned bytes
    const char* __restrict__ input_q8_1,   // [nb * 36] Q8_1 pre-quantized input
    float* __restrict__ out,               // [out_dim] F32 output
    unsigned int out_dim,
    unsigned int in_dim)
{
    unsigned int r0 = blockIdx.x * NR;  // first output row for this block

    unsigned int warp_id = threadIdx.x / NW;
    unsigned int lane    = threadIdx.x % NW;

    unsigned int nb = in_dim >> 5;  // number of blocks per row (in_dim / 32)
    unsigned long long row_bytes = (unsigned long long)nb * Q4_ALIGNED_BYTES;

    // Per-row accumulators
    float sumf[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) sumf[r] = 0.0f;

    // Main loop: each thread handles one Q4/Q8_1 block pair per iteration,
    // striding by THREADS_PER_BLOCK (256) blocks.
    for (unsigned int ib = threadIdx.x; ib < nb; ib += THREADS_PER_BLOCK) {
        // --- Load Q8_1 input block (36 bytes, quant data at +4) ---
        const char* x_block = input_q8_1 + (unsigned long long)ib * Q8_1_BYTES;

        // Read f16 input scale (bytes 0-1, native halfword load).
        unsigned short x_scale_bits = *(const unsigned short*)x_block;
        float x_scale = f16_bits_to_f32(x_scale_bits);

        // Read f16 weighted sum (bytes 2-3): s = d * sum(quants).
        unsigned short x_sum_bits = *(const unsigned short*)(x_block + 2);
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

            // Pointer to this row's Q4Aligned weight block.
            const char* w_block = weight_q4a
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * Q4_ALIGNED_BYTES;

            // Read f16 weight scale (bytes 0-1, native halfword load).
            unsigned short w_scale_bits = *(const unsigned short*)w_block;
            float w_scale = f16_bits_to_f32(w_scale_bits);

            // KEY OPTIMIZATION: aligned int* loads for nibble data.
            // Nibble data at w_block+4 is 4-byte aligned (20-byte blocks with 2B pad).
            // Load 16 nibble bytes as 4 int32 words (vs 16 byte loads in unaligned kernel).
            const unsigned int* w_nibbles = (const unsigned int*)(w_block + 4);

            // Unpack 4 int32 words (16 nibble bytes) into 8 dp4a-compatible int32 words
            // and compute dp4a dot product against Q8_1 input.
            int acc = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                unsigned int packed = w_nibbles[k];
                int w0, w1;
                unpack_nibbles_4bytes(packed, w0, w1);
                acc = __dp4a(w0, xv[2 * k],     acc);
                acc = __dp4a(w1, xv[2 * k + 1], acc);
            }

            // Combined result with zero-point correction:
            //   dot(w, x) = w_scale * x_scale * dp4a_sum - 8 * w_scale * x_sum
            sumf[row] += w_scale * (x_scale * (float)acc - 8.0f * x_sum);
        }
    }

    // --- Cross-warp reduction via simple shmem (7 warps write, warp 0 sums) ---
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
// Kernel 2: Q4Aligned weight x Q8_1 input + residual -> F32 output (dp4a, NR=4).
//
// Same as matvec_q4_aligned_q8_1 but with fused residual addition at final write.
// Used for Wo (attention output) and down projection.
//
// Grid:  (ceil(out_dim / NR), 1, 1)  -- one block per NR rows
// Block: (256, 1, 1)                 -- 8 warps x 32 threads
// ==========================================================================
extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 1) void matvec_q4_aligned_q8_1_residual(
    const char* __restrict__ weight_q4a,   // [out_dim * nb * 20] Q4Aligned bytes
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
    unsigned long long row_bytes = (unsigned long long)nb * Q4_ALIGNED_BYTES;

    float sumf[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) sumf[r] = 0.0f;

    for (unsigned int ib = threadIdx.x; ib < nb; ib += THREADS_PER_BLOCK) {
        // Load Q8_1 input block.
        const char* x_block = input_q8_1 + (unsigned long long)ib * Q8_1_BYTES;

        unsigned short x_scale_bits = *(const unsigned short*)x_block;
        float x_scale = f16_bits_to_f32(x_scale_bits);

        unsigned short x_sum_bits = *(const unsigned short*)(x_block + 2);
        float x_sum = f16_bits_to_f32(x_sum_bits);

        const int* x_packed = (const int*)(x_block + 4);

        int xv[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) xv[k] = x_packed[k];

        // Process NR output rows.
        #pragma unroll
        for (int row = 0; row < NR; row++) {
            if (r0 + row >= out_dim) break;

            const char* w_block = weight_q4a
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * Q4_ALIGNED_BYTES;

            unsigned short w_scale_bits = *(const unsigned short*)w_block;
            float w_scale = f16_bits_to_f32(w_scale_bits);

            // Aligned int* loads for nibble data (4-byte aligned at +4).
            const unsigned int* w_nibbles = (const unsigned int*)(w_block + 4);

            int acc = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                unsigned int packed = w_nibbles[k];
                int w0, w1;
                unpack_nibbles_4bytes(packed, w0, w1);
                acc = __dp4a(w0, xv[2 * k],     acc);
                acc = __dp4a(w1, xv[2 * k + 1], acc);
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
