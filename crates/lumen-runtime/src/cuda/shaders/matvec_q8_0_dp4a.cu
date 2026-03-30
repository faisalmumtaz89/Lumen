// Q8_0 matrix-vector multiply (GEMV) dp4a: INT8 dot product via __dp4a().
//
// Key optimization: instead of dequantizing int8 weights to float and doing
// float multiply-accumulate, we quantize x on-the-fly to int8 (Q8_1 style)
// and use __dp4a() for native INT8 dot products. This is the same approach
// used by llama.cpp's CUDA Q8_0 x Q8_1 kernel.
//
// Per Q8_0 block (32 elements):
//   1. Load f16 weight scale -> f32
//   2. Load 32 int8 weight values as 8 x int32 (packed 4 bytes each)
//   3. Quantize 32 x-values to int8 on-the-fly:
//      x_amax = max(|x[block_start..+32]|)
//      x_scale = x_amax / 127.0
//      x_q[i] = round(x[i] / x_scale)   (clamped to [-127, 127])
//   4. Pack 4 x_q values into int32
//   5. Use __dp4a(w_packed, x_packed, acc) for 4 products at once
//      (8 dp4a calls per block = 32 products)
//   6. Final: sum += w_scale * x_scale * (float)acc
//
// Thread mapping: each thread processes one full Q8_0 block per iteration
// (32 elements), unlike v1-v3 where 4 threads share a block. This makes
// on-the-fly x quantization entirely thread-local (no warp cooperation
// needed for the max reduction).
//
// Architecture: NR=2 rows per block, 128 threads (4 warps).
// Each thread strides over blocks: thread t handles blocks t, t+128, t+256, ...
//
// Q8_0 block layout (GGML): 34 bytes per block of 32 elements.
//   bytes [0..1]: f16 scale (IEEE 754 half-precision, little-endian)
//   bytes [2..33]: 32 x int8 quantized values
//   Dequant: float_val = scale * (float)(int8_t)quant[j]
//
// Operation: out[i] = sum_j(dequant(weight_q8[i, j]) * x[j])
// Weight matrix: [out_dim, in_dim] stored as Q8_0 blocks, row-major
// Input vector:  [in_dim] f32
// Output vector: [out_dim] f32
//
// Requires compute capability >= 6.1 for __dp4a() (Pascal+).
//
// in_dim must be a multiple of Q8_0_BLOCK_SIZE (32).
//
// NVRTC-compatible: no system includes, extern "C" linkage.

#define NR       2     // rows per thread block
#define NW       32    // warp size
#define THREADS_PER_BLOCK 128  // 4 warps
#define Q8_0_BLOCK_SIZE   32
#define Q8_0_BYTES        34   // 2 bytes f16 scale + 32 bytes int8 data

// Hardware f16->f32 conversion via PTX (single instruction on SM 53+).
// Replaces ~15 ALU software bit-manipulation with the native CVT instruction.
// NVRTC-compatible: inline PTX requires no headers or include paths.
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

// Pack 4 signed bytes into one int32 for __dp4a().
// dp4a interprets each byte of the int32 as a signed 8-bit integer.
__device__ __forceinline__ int pack_i8x4(int a, int b, int c, int d) {
    return (a & 0xFF) | ((b & 0xFF) << 8) | ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
}

// Q8_0 matrix-vector multiply with dp4a INT8 dot products.
//
// Grid:  (ceil(out_dim / NR), 1, 1)  -- one block per NR rows
// Block: (128, 1, 1)                 -- 4 warps x 32 threads
//
// in_dim must be a multiple of Q8_0_BLOCK_SIZE (32).
extern "C" __global__ void matvec_q8_0_dp4a(
    const char* __restrict__ weight_q8,  // [out_dim * num_blocks_per_row * 34] raw Q8_0 bytes
    const float* __restrict__ x,         // [in_dim]
    float* __restrict__ out,             // [out_dim]
    unsigned int out_dim,
    unsigned int in_dim)
{
    unsigned int r0 = blockIdx.x * NR;  // first output row for this block

    unsigned int warp_id = threadIdx.x / NW;
    unsigned int lane    = threadIdx.x % NW;

    unsigned int nb = in_dim >> 5;  // number of Q8_0 blocks per row
    unsigned long long row_bytes = (unsigned long long)nb * Q8_0_BYTES;

    // Per-row accumulators
    float sumf[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) sumf[r] = 0.0f;

    // Main loop: each thread handles one full Q8_0 block per iteration,
    // striding by THREADS_PER_BLOCK (128) blocks.
    for (unsigned int ib = threadIdx.x; ib < nb; ib += THREADS_PER_BLOCK) {
        // ---------------------------------------------------------------
        // On-the-fly x quantization to int8 (Q8_1 style)
        // ---------------------------------------------------------------
        unsigned int x_base = ib * Q8_0_BLOCK_SIZE;

        // Load 32 x-values and find absolute max.
        // Use float4 vectorized loads (8 x float4 = 32 floats).
        // Alignment: x is CUDA-allocated (>= 256-byte aligned). x_base = ib * 32,
        // so byte offset from x is ib * 128, which is always 16-byte aligned (128
        // is divisible by 16). The float4 cast is safe.
        float xv[32];
        float amax = 0.0f;

        const float4* x4 = (const float4*)(x + x_base);
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            float4 v = x4[k];
            xv[k * 4 + 0] = v.x;
            xv[k * 4 + 1] = v.y;
            xv[k * 4 + 2] = v.z;
            xv[k * 4 + 3] = v.w;
            // Track absolute max
            float a0 = v.x < 0.0f ? -v.x : v.x;
            float a1 = v.y < 0.0f ? -v.y : v.y;
            float a2 = v.z < 0.0f ? -v.z : v.z;
            float a3 = v.w < 0.0f ? -v.w : v.w;
            if (a0 > amax) amax = a0;
            if (a1 > amax) amax = a1;
            if (a2 > amax) amax = a2;
            if (a3 > amax) amax = a3;
        }

        // Compute x_scale and its reciprocal for quantization.
        // If amax == 0, all x-values are zero and the block contributes nothing.
        float x_scale = amax / 127.0f;
        float x_scale_inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;

        // Quantize x-values to int8 and pack into int32 (4 per int).
        int x_packed[8];  // 8 x int32 = 32 x int8
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int q0 = (int)__float2int_rn(xv[k * 4 + 0] * x_scale_inv);
            int q1 = (int)__float2int_rn(xv[k * 4 + 1] * x_scale_inv);
            int q2 = (int)__float2int_rn(xv[k * 4 + 2] * x_scale_inv);
            int q3 = (int)__float2int_rn(xv[k * 4 + 3] * x_scale_inv);
            x_packed[k] = pack_i8x4(q0, q1, q2, q3);
        }

        // ---------------------------------------------------------------
        // Process both output rows with the same quantized x-values
        // ---------------------------------------------------------------
        #pragma unroll
        for (int row = 0; row < NR; row++) {
            if (r0 + row >= out_dim) break;

            const char* bp = weight_q8
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * Q8_0_BYTES;

            // Read f16 weight scale
            unsigned short scale_bits = (unsigned short)(unsigned char)bp[0]
                                      | ((unsigned short)(unsigned char)bp[1] << 8);
            float w_scale = f16_bits_to_f32(scale_bits);

            // Load 32 int8 weight values as 8 x int32 (packed 4 bytes each).
            // Q8_0 quant data at bp+2 is NOT 4-byte aligned (34-byte blocks).
            // CONFIRMED: int* cast causes XID 13 Misaligned Address on A100 PCIe
            // when compiled with compute_80. Use byte loads + manual packing.
            const unsigned char* wq = (const unsigned char*)(bp + 2);

            int acc = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                int w_word = (int)(signed char)wq[k * 4 + 0]
                           | ((int)(signed char)wq[k * 4 + 1] << 8)
                           | ((int)(signed char)wq[k * 4 + 2] << 16)
                           | ((int)(signed char)wq[k * 4 + 3] << 24);
                acc = __dp4a(w_word, x_packed[k], acc);
            }

            // Combined scale: w_scale * x_scale * int_dot_product
            sumf[row] += w_scale * x_scale * (float)acc;
        }
    }

    // Cross-warp reduction via shared memory.
    // Layout: NR rows x NW slots.
    __shared__ float shmem[NR * NW];

    // Initialize shmem (only warp 0 needs to zero it)
    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            shmem[r * NW + lane] = 0.0f;
        }
    }

    // Intra-warp reduction
    #pragma unroll
    for (int r = 0; r < NR; r++) {
        sumf[r] = warp_reduce_sum(sumf[r]);
    }

    __syncthreads();

    // Lane 0 of each warp writes its partial sum
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            shmem[r * NW + warp_id] = sumf[r];
        }
    }

    __syncthreads();

    // Warp 0 does the final reduction across warps
    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            if (r0 + r < out_dim) {
                float val = (lane < (THREADS_PER_BLOCK / NW)) ? shmem[r * NW + lane] : 0.0f;
                val = warp_reduce_sum(val);
                if (lane == 0) {
                    out[r0 + r] = val;
                }
            }
        }
    }
}

// Q8_0 matrix-vector multiply with dp4a and fused residual addition:
// out = dequant(weight_q8) * x + residual
//
// Same dp4a INT8 dot product pattern as matvec_q8_0_dp4a,
// with fused residual addition at the final write.
//
// Grid:  (ceil(out_dim / NR), 1, 1)  -- one block per NR rows
// Block: (128, 1, 1)                 -- 4 warps x 32 threads
extern "C" __global__ void matvec_q8_0_dp4a_residual(
    const char* __restrict__ weight_q8,    // [out_dim * num_blocks_per_row * 34] raw Q8_0 bytes
    const float* __restrict__ x,           // [in_dim]
    const float* __restrict__ residual,    // [out_dim], added to output
    float* __restrict__ out,               // [out_dim]
    unsigned int out_dim,
    unsigned int in_dim)
{
    unsigned int r0 = blockIdx.x * NR;

    unsigned int warp_id = threadIdx.x / NW;
    unsigned int lane    = threadIdx.x % NW;

    unsigned int nb = in_dim >> 5;
    unsigned long long row_bytes = (unsigned long long)nb * Q8_0_BYTES;

    float sumf[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) sumf[r] = 0.0f;

    for (unsigned int ib = threadIdx.x; ib < nb; ib += THREADS_PER_BLOCK) {
        unsigned int x_base = ib * Q8_0_BLOCK_SIZE;

        // Load 32 x-values and find absolute max.
        // float4 alignment guaranteed: x is >= 256-byte aligned, x_base * 4 is
        // always a multiple of 16 (see comment in matvec_q8_0_dp4a above).
        float xv[32];
        float amax = 0.0f;

        const float4* x4 = (const float4*)(x + x_base);
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            float4 v = x4[k];
            xv[k * 4 + 0] = v.x;
            xv[k * 4 + 1] = v.y;
            xv[k * 4 + 2] = v.z;
            xv[k * 4 + 3] = v.w;
            float a0 = v.x < 0.0f ? -v.x : v.x;
            float a1 = v.y < 0.0f ? -v.y : v.y;
            float a2 = v.z < 0.0f ? -v.z : v.z;
            float a3 = v.w < 0.0f ? -v.w : v.w;
            if (a0 > amax) amax = a0;
            if (a1 > amax) amax = a1;
            if (a2 > amax) amax = a2;
            if (a3 > amax) amax = a3;
        }

        float x_scale = amax / 127.0f;
        float x_scale_inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;

        // Quantize x to int8 and pack into int32
        int x_packed[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int q0 = (int)__float2int_rn(xv[k * 4 + 0] * x_scale_inv);
            int q1 = (int)__float2int_rn(xv[k * 4 + 1] * x_scale_inv);
            int q2 = (int)__float2int_rn(xv[k * 4 + 2] * x_scale_inv);
            int q3 = (int)__float2int_rn(xv[k * 4 + 3] * x_scale_inv);
            x_packed[k] = pack_i8x4(q0, q1, q2, q3);
        }

        #pragma unroll
        for (int row = 0; row < NR; row++) {
            if (r0 + row >= out_dim) break;

            const char* bp = weight_q8
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * Q8_0_BYTES;

            unsigned short scale_bits = (unsigned short)(unsigned char)bp[0]
                                      | ((unsigned short)(unsigned char)bp[1] << 8);
            float w_scale = f16_bits_to_f32(scale_bits);

            // Byte-level loads: int* cast at +2 causes XID 13 on A100 (confirmed).
            const unsigned char* wq = (const unsigned char*)(bp + 2);

            int acc = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                int w_word = (int)(signed char)wq[k * 4 + 0]
                           | ((int)(signed char)wq[k * 4 + 1] << 8)
                           | ((int)(signed char)wq[k * 4 + 2] << 16)
                           | ((int)(signed char)wq[k * 4 + 3] << 24);
                acc = __dp4a(w_word, x_packed[k], acc);
            }

            sumf[row] += w_scale * x_scale * (float)acc;
        }
    }

    __shared__ float shmem[NR * NW];

    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            shmem[r * NW + lane] = 0.0f;
        }
    }

    #pragma unroll
    for (int r = 0; r < NR; r++) {
        sumf[r] = warp_reduce_sum(sumf[r]);
    }

    __syncthreads();

    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            shmem[r * NW + warp_id] = sumf[r];
        }
    }

    __syncthreads();

    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            if (r0 + r < out_dim) {
                float val = (lane < (THREADS_PER_BLOCK / NW)) ? shmem[r * NW + lane] : 0.0f;
                val = warp_reduce_sum(val);
                if (lane == 0) {
                    out[r0 + r] = val + residual[r0 + r];
                }
            }
        }
    }
}
