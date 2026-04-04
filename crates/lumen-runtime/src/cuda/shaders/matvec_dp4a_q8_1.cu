// ==========================================================================
// dp4a GEMV with pre-quantized Q8_1 input vector.
//
// Three kernels:
//   1. quantize_f32_to_q8_1: Quantize F32 input vector to Q8_1 format.
//      Run ONCE per token, result reused across all matvec calls in a layer.
//   2. matvec_q8_0_q8_1: Q8_0 weight x Q8_1 input -> F32 output.
//      dp4a INT8 dot products on both sides -- no on-the-fly quantization.
//   3. matvec_q8_0_q8_1_residual: Same + fused residual addition.
//
// Optimized dp4a approach for quantized decode on A100:
//   - Pre-quantize F32 x to Q8_1 blocks (32-element blocks, f16 scale + f16 sum + 32 int8)
//   - Weight reads: 1.0625 B/elem (native Q8_0, 34 bytes per 32 elements)
//   - Input reads: 1.125 B/elem (Q8_1, 36 bytes per 32 elements)
//   - Total: 2.1875 B/elem per output row -- but dp4a throughput compensates
//   - NO shared memory for input -- L2 cache handles x reuse across blocks
//
// Why this beats the current smem/hgemv kernels:
//   - smem kernel: scalar FMA (1 multiply per cycle), reads F32 x (4 B/elem)
//   - hgemv kernel: scalar FMA, reads F16 x (2 B/elem) -- lossy conversion
//   - This kernel: dp4a (4 multiply-accumulates per instruction), INT8 x (1.125 B/elem)
//   - On A100, dp4a has 2x throughput of FMA32 for GEMV workloads
//
// Q8_1 block layout (36 bytes per 32 elements):
//   bytes [0..1]: f16 scale (d)
//   bytes [2..3]: f16 weighted sum (s = d * sum(quants))  [unused for Q8_0 weights]
//   bytes [4..35]: 32 x int8 quantized values
//   Quant data at byte 4 is 4-byte aligned -> native int* loads (8 instructions)
//
// Q8_0 block layout (34 bytes per 32 elements):
//   bytes [0..1]: f16 scale (d)
//   bytes [2..33]: 32 x int8 quantized values
//   Quant data at byte 2 is 2-byte aligned -> uint16 pair loads + shift/OR packing
//
// Architecture:
//   - 128 threads per block (4 warps)
//   - NR=2 output rows per block (2x x-vector amortization)
//   - Each thread processes 1 Q8_0 block per iteration, striding by 128
//   - dp4a(weight_int32, input_int32, accumulator) for the core dot product
//   - Warp shuffle + shmem for cross-warp reduction
//   - NO shared memory for input vector (rely on L2 cache for reuse)
//
// For Q8_0 x Q8_1 (Q8_0 has zero-point = 0):
//   dot = sum_k( w_quant[k] * x_quant[k] )  // dp4a handles 4 at a time
//   result = w_scale * x_scale * dot
//   (The Q8_1 sum field is unused since Q8_0 zero-point = 0)
//
// Requires compute capability >= 6.1 for __dp4a() (Pascal+).
// in_dim must be a multiple of 32 (Q8_0 block size).
//
// NVRTC-compatible: no system includes, extern "C" linkage.
// ==========================================================================

#define Q8_0_BLOCK_SIZE  32
#define Q8_0_BYTES       34   // 2B f16 scale + 32B int8 quants
#define Q8_1_BYTES       36   // 2B f16 scale + 2B f16 sum + 32B int8 quants
#define WARP_SIZE        32

// ---- Matvec kernel constants ----
#define MV_THREADS  128   // 4 warps per block
#define MV_NR       2     // 2 output rows per block (2x x-vector amortization)
#define MV_NWARPS   (MV_THREADS / WARP_SIZE)  // 4

// Hardware f16->f32 conversion via PTX (single cycle on SM 53+).
__device__ __forceinline__ float f16_bits_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

// Hardware f32->f16 conversion via PTX.
__device__ __forceinline__ unsigned short f32_to_f16_bits(float val) {
    unsigned short result;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(result) : "f"(val));
    return result;
}

// Warp-level reduction: sum all lanes via butterfly shuffle.
__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

// ==========================================================================
// Kernel 1: Quantize F32 input vector to Q8_1 format.
//
// Grid:  (num_blocks, 1, 1)  where num_blocks = dim / 32
// Block: (32, 1, 1)          one warp per Q8_1 block
//
// Each block of 32 threads quantizes 32 consecutive F32 values into one
// Q8_1 block. Uses warp shuffle for the max reduction (no shared memory).
//
// Output format per block (36 bytes):
//   [f16 scale] [f16 sum] [32 x int8 quants]
//   where scale = max(|x|) / 127.0
//         sum = scale * sum(quants)  (precomputed for Q8_1 consumers)
//         quants[i] = round(x[i] / scale)
// ==========================================================================
extern "C" __global__ __launch_bounds__(32, 1) void quantize_f32_to_q8_1(
    const float* __restrict__ input,   // [dim] F32 input vector
    char* __restrict__ output,         // [dim/32 * 36] Q8_1 output
    unsigned int dim)
{
    unsigned int block_idx = blockIdx.x;
    unsigned int lane = threadIdx.x;  // 0..31
    unsigned int base = block_idx * Q8_0_BLOCK_SIZE;

    if (base + lane >= dim) return;

    // Each thread loads one element.
    float val = input[base + lane];

    // Warp-wide absolute max reduction via butterfly shuffle.
    // Each step halves the distance, broadcasting the max to all 32 lanes.
    float amax = val < 0.0f ? -val : val;
    float tmp;
    tmp = __shfl_xor_sync(0xffffffff, amax, 16);
    amax = tmp > amax ? tmp : amax;
    tmp = __shfl_xor_sync(0xffffffff, amax, 8);
    amax = tmp > amax ? tmp : amax;
    tmp = __shfl_xor_sync(0xffffffff, amax, 4);
    amax = tmp > amax ? tmp : amax;
    tmp = __shfl_xor_sync(0xffffffff, amax, 2);
    amax = tmp > amax ? tmp : amax;
    tmp = __shfl_xor_sync(0xffffffff, amax, 1);
    amax = tmp > amax ? tmp : amax;

    // amax is now broadcast to all 32 lanes.
    float scale = amax / 127.0f;
    float scale_inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;

    // Quantize this thread's value.
    int qi = __float2int_rn(val * scale_inv);
    // Clamp to [-127, 127] (should already be in range, but be safe).
    qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);

    // Compute partial sum for the Q8_1 sum field: sum(quants).
    // Use warp shuffle to sum all 32 quantized values.
    float qi_f = (float)qi;
    float qsum = qi_f;
    qsum += __shfl_xor_sync(0xffffffff, qsum, 16);
    qsum += __shfl_xor_sync(0xffffffff, qsum, 8);
    qsum += __shfl_xor_sync(0xffffffff, qsum, 4);
    qsum += __shfl_xor_sync(0xffffffff, qsum, 2);
    qsum += __shfl_xor_sync(0xffffffff, qsum, 1);
    // qsum is now sum(all 32 quants), broadcast to all lanes.

    // Weighted sum: s = d * sum(quants).
    float weighted_sum = scale * qsum;

    // Write the Q8_1 block. Lane 0 writes the header; all lanes write their quant byte.
    char* block_out = output + (unsigned long long)block_idx * Q8_1_BYTES;

    if (lane == 0) {
        // Write f16 scale (bytes 0-1).
        unsigned short d_f16 = f32_to_f16_bits(scale);
        block_out[0] = (char)(d_f16 & 0xFF);
        block_out[1] = (char)((d_f16 >> 8) & 0xFF);

        // Write f16 weighted sum (bytes 2-3).
        unsigned short s_f16 = f32_to_f16_bits(weighted_sum);
        block_out[2] = (char)(s_f16 & 0xFF);
        block_out[3] = (char)((s_f16 >> 8) & 0xFF);
    }

    // All lanes write their quantized byte at byte offset 4 + lane.
    block_out[4 + lane] = (char)(qi & 0xFF);
}

// ==========================================================================
// Kernel 2: Q8_0 weight x Q8_1 input -> F32 output (dp4a, NR=2).
//
// Grid:  (ceil(out_dim / MV_NR), 1, 1)  -- one block per MV_NR rows
// Block: (MV_THREADS, 1, 1)             -- 128 threads (4 warps)
//
// Each thread processes multiple Q8_0/Q8_1 block pairs, striding by 128.
// NR=2: each block processes 2 output rows, sharing the x-vector loads
// across all rows to 2x amortize input bandwidth.
//
// Memory access pattern:
//   Weight: sequential Q8_0 blocks (34 bytes each), coalesced across threads
//   Input:  Q8_1 blocks (36 bytes each), reused across NR rows + via L2
//
// Q8_0 x Q8_1 dot product (Q8_0 has zero-point = 0):
//   For each block pair:
//     int_dot = dp4a(w_packed[0..7], x_packed[0..7], 0)  // 8 dp4a calls
//     partial = w_scale * x_scale * (float)int_dot
//   output = sum(all partials)
//
// The Q8_1 sum field (bytes 2-3) is NOT used because Q8_0 weights have
// symmetric quantization (zero_point = 0). The field exists only for
// format compatibility with Q4_1 x Q8_1 where zero_point != 0.
// ==========================================================================
extern "C" __global__ __launch_bounds__(MV_THREADS, 1) void matvec_q8_0_q8_1(
    const char* __restrict__ weight_q8,  // [out_dim * nb * 34] Q8_0 weight bytes
    const char* __restrict__ input_q8_1, // [nb * 36] Q8_1 pre-quantized input
    float* __restrict__ out,             // [out_dim] F32 output
    unsigned int out_dim,
    unsigned int in_dim)
{
    const unsigned int r0 = blockIdx.x * MV_NR;  // first output row for this block

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane = tid % WARP_SIZE;

    const unsigned int nb = in_dim >> 5;  // Number of Q8_0 blocks per row
    const unsigned long long row_bytes = (unsigned long long)nb * Q8_0_BYTES;

    // Per-row accumulators
    float sumf[MV_NR];
    #pragma unroll
    for (int r = 0; r < MV_NR; r++) sumf[r] = 0.0f;

    // Main loop: each thread handles 1 Q8_0/Q8_1 block pair per iteration,
    // striding by MV_THREADS (128) blocks.
    for (unsigned int ib = tid; ib < nb; ib += MV_THREADS) {

        // --- Load Q8_1 input block (shared across NR rows) ---
        const char* x_block = input_q8_1 + (unsigned long long)ib * Q8_1_BYTES;

        // Read f16 input scale (bytes 0-1, little-endian).
        unsigned short x_scale_bits = (unsigned short)(unsigned char)x_block[0]
                                    | ((unsigned short)(unsigned char)x_block[1] << 8);
        float x_scale = f16_bits_to_f32(x_scale_bits);

        // Load 32 int8 input values from x_block+4 (4-byte aligned).
        const int* xq = (const int*)(x_block + 4);

        // Preload input packed words.
        int xv[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) xv[k] = xq[k];

        // --- Process NR output rows with same x-values ---
        #pragma unroll
        for (int row = 0; row < MV_NR; row++) {
            if (r0 + row >= out_dim) break;

            const char* w_block = weight_q8
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * Q8_0_BYTES;

            // Read f16 weight scale (bytes 0-1, little-endian).
            unsigned short w_scale_bits = (unsigned short)(unsigned char)w_block[0]
                                        | ((unsigned short)(unsigned char)w_block[1] << 8);
            float w_scale = f16_bits_to_f32(w_scale_bits);

            // Load 32 int8 weight values from w_block+2 as uint16 pairs.
            // Q8_0 quant data at +2 is 2-byte aligned (34-byte blocks, even offset).
            // uint16 loads: 2 loads + 1 shift + 1 OR = 4 ops per int32 word,
            // vs byte loads: 4 loads + 3 shifts + 3 ORs = 10 ops per int32 word.
            // NOT 4-byte aligned, so int* cast is unsafe (XID 13 on A100).
            const unsigned short* w16 = (const unsigned short*)(w_block + 2);

            int acc = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                int w_word = (int)w16[k * 2] | ((int)w16[k * 2 + 1] << 16);
                acc = __dp4a(w_word, xv[k], acc);
            }

            // Combined scale: w_scale * x_scale * int_dot_product.
            // Q8_0 has zero-point=0, so no correction term needed.
            sumf[row] += w_scale * x_scale * (float)acc;
        }
    }

    // --- Cross-warp reduction via simple shmem (3 warps write, warp 0 sums) ---
    #pragma unroll
    for (int r = 0; r < MV_NR; r++) {
        sumf[r] = warp_reduce_sum(sumf[r]);
    }

    __shared__ float shmem[(MV_NWARPS - 1) * MV_NR];

    if (warp_id > 0 && lane == 0) {
        #pragma unroll
        for (int r = 0; r < MV_NR; r++) {
            shmem[(warp_id - 1) * MV_NR + r] = sumf[r];
        }
    }
    __syncthreads();

    if (tid == 0) {
        #pragma unroll
        for (int r = 0; r < MV_NR; r++) {
            float total = sumf[r];
            #pragma unroll
            for (int w = 0; w < MV_NWARPS - 1; w++) {
                total += shmem[w * MV_NR + r];
            }
            if (r0 + r < out_dim) {
                out[r0 + r] = total;
            }
        }
    }
}

// ==========================================================================
// Kernel 3: Q8_0 weight x Q8_1 input + residual -> F32 output (dp4a, NR=2).
//
// Same as matvec_q8_0_q8_1 but with fused residual addition at final write.
// ==========================================================================
extern "C" __global__ __launch_bounds__(MV_THREADS, 1) void matvec_q8_0_q8_1_residual(
    const char* __restrict__ weight_q8,  // [out_dim * nb * 34] Q8_0 weight bytes
    const char* __restrict__ input_q8_1, // [nb * 36] Q8_1 pre-quantized input
    const float* __restrict__ residual,  // [out_dim] F32 residual
    float* __restrict__ out,             // [out_dim] F32 output
    unsigned int out_dim,
    unsigned int in_dim)
{
    const unsigned int r0 = blockIdx.x * MV_NR;

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane = tid % WARP_SIZE;

    const unsigned int nb = in_dim >> 5;
    const unsigned long long row_bytes = (unsigned long long)nb * Q8_0_BYTES;

    float sumf[MV_NR];
    #pragma unroll
    for (int r = 0; r < MV_NR; r++) sumf[r] = 0.0f;

    // Main loop: 1 Q8 block per thread per iteration, stride MV_THREADS.
    for (unsigned int ib = tid; ib < nb; ib += MV_THREADS) {

        // Load Q8_1 input block (shared across NR rows).
        const char* x_block = input_q8_1 + (unsigned long long)ib * Q8_1_BYTES;

        unsigned short x_scale_bits = (unsigned short)(unsigned char)x_block[0]
                                    | ((unsigned short)(unsigned char)x_block[1] << 8);
        float x_scale = f16_bits_to_f32(x_scale_bits);

        const int* xq = (const int*)(x_block + 4);

        int xv[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) xv[k] = xq[k];

        // Process NR output rows.
        #pragma unroll
        for (int row = 0; row < MV_NR; row++) {
            if (r0 + row >= out_dim) break;

            const char* w_block = weight_q8
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * Q8_0_BYTES;

            unsigned short w_scale_bits = (unsigned short)(unsigned char)w_block[0]
                                        | ((unsigned short)(unsigned char)w_block[1] << 8);
            float w_scale = f16_bits_to_f32(w_scale_bits);

            // uint16 loads: 2-byte aligned at w_block+2, avoids XID 13 from int* cast.
            const unsigned short* w16 = (const unsigned short*)(w_block + 2);

            int acc = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                int w_word = (int)w16[k * 2] | ((int)w16[k * 2 + 1] << 16);
                acc = __dp4a(w_word, xv[k], acc);
            }

            sumf[row] += w_scale * x_scale * (float)acc;
        }
    }

    // Cross-warp reduction via simple shmem.
    #pragma unroll
    for (int r = 0; r < MV_NR; r++) {
        sumf[r] = warp_reduce_sum(sumf[r]);
    }

    __shared__ float shmem[(MV_NWARPS - 1) * MV_NR];

    if (warp_id > 0 && lane == 0) {
        #pragma unroll
        for (int r = 0; r < MV_NR; r++) {
            shmem[(warp_id - 1) * MV_NR + r] = sumf[r];
        }
    }
    __syncthreads();

    if (tid == 0) {
        #pragma unroll
        for (int r = 0; r < MV_NR; r++) {
            float total = sumf[r];
            #pragma unroll
            for (int w = 0; w < MV_NWARPS - 1; w++) {
                total += shmem[w * MV_NR + r];
            }
            if (r0 + r < out_dim) {
                out[r0 + r] = total + residual[r0 + r];
            }
        }
    }
}
