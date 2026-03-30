// Q8_0 Aligned matrix-vector multiply (GEMV) dp4a: INT8 dot product via __dp4a()
// with 36-byte aligned blocks for native int* loads.
//
// This is the aligned variant of matvec_q8_0_dp4a.cu. The key difference:
// Q8_0 blocks are repacked from 34 bytes to 36 bytes during preload_weights,
// inserting 2 bytes of padding after the f16 scale so that the 32-byte int8
// quant data starts at offset +4 (4-byte aligned). This allows the inner loop
// to use `const int*` loads (8 instructions) instead of byte-level manual
// packing (56 instructions per block).
//
// Aligned block layout (36 bytes per block of 32 elements):
//   bytes [0..1]:  f16 scale (IEEE 754 half-precision, little-endian)
//   bytes [2..3]:  padding (zeroed)
//   bytes [4..35]: 32 x int8 quantized values  <-- NOW 4-BYTE ALIGNED
//
// Original Q8_0 layout (34 bytes):
//   bytes [0..1]:  f16 scale
//   bytes [2..33]: 32 x int8 quantized values  <-- NOT aligned (34-byte stride)
//
// The aligned layout trades 5.9% more memory (36/34 = 1.059x) for dramatically
// fewer ALU instructions in the inner loop: 8 int* loads vs 32 byte loads +
// 24 shift-or instructions = 56 ops.
//
// On-the-fly x quantization is identical to matvec_q8_0_dp4a.cu.
//
// Architecture: NR=2 rows per block, 128 threads (4 warps).
// Requires compute capability >= 6.1 for __dp4a() (Pascal+).
// in_dim must be a multiple of 32 (Q8_0 block size).
//
// NVRTC-compatible: no system includes, extern "C" linkage.

#define NR       2     // rows per thread block
#define NW       32    // warp size
#define THREADS_PER_BLOCK 128  // 4 warps
#define Q8_0_BLOCK_SIZE   32
#define Q8_ALIGNED_BYTES  36   // 2 bytes f16 scale + 2 bytes pad + 32 bytes int8 data

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

// Pack 4 signed bytes into one int32 for __dp4a().
__device__ __forceinline__ int pack_i8x4(int a, int b, int c, int d) {
    return (a & 0xFF) | ((b & 0xFF) << 8) | ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
}

// Q8_0 Aligned matrix-vector multiply with dp4a INT8 dot products.
//
// Grid:  (ceil(out_dim / NR), 1, 1)  -- one block per NR rows
// Block: (128, 1, 1)                 -- 4 warps x 32 threads
//
// weight_q8_aligned: 36-byte aligned blocks (repacked from 34-byte Q8_0).
extern "C" __global__ void matvec_q8_0_aligned(
    const char* __restrict__ weight_q8_aligned,  // [out_dim * num_blocks_per_row * 36]
    const float* __restrict__ x,                 // [in_dim]
    float* __restrict__ out,                     // [out_dim]
    unsigned int out_dim,
    unsigned int in_dim)
{
    unsigned int r0 = blockIdx.x * NR;  // first output row for this block

    unsigned int warp_id = threadIdx.x / NW;
    unsigned int lane    = threadIdx.x % NW;

    unsigned int nb = in_dim >> 5;  // number of Q8_0 blocks per row
    unsigned long long row_bytes = (unsigned long long)nb * Q8_ALIGNED_BYTES;

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

        // Load 32 x-values and find absolute max (vectorized float4 loads).
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

        // Quantize x-values to int8 and pack into int32 (4 per int).
        int x_packed[8];
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

            const char* bp = weight_q8_aligned
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * Q8_ALIGNED_BYTES;

            // Read f16 weight scale (bytes 0-1)
            unsigned short scale_bits = (unsigned short)(unsigned char)bp[0]
                                      | ((unsigned short)(unsigned char)bp[1] << 8);
            float w_scale = f16_bits_to_f32(scale_bits);

            // KEY OPTIMIZATION: quant data at bp+4 is 4-byte aligned (36-byte blocks).
            // Safe to use int* load — 8 instructions instead of 56 byte loads + shifts.
            const int* w_int = (const int*)(bp + 4);

            int acc = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                acc = __dp4a(w_int[k], x_packed[k], acc);
            }

            // Combined scale: w_scale * x_scale * int_dot_product
            sumf[row] += w_scale * x_scale * (float)acc;
        }
    }

    // Cross-warp reduction via shared memory.
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
                    out[r0 + r] = val;
                }
            }
        }
    }
}

// Q8_0 Aligned matrix-vector multiply with dp4a and fused residual addition:
// out = dequant(weight_q8_aligned) * x + residual
//
// Same aligned dp4a pattern, with fused residual addition at the final write.
//
// Grid:  (ceil(out_dim / NR), 1, 1)  -- one block per NR rows
// Block: (128, 1, 1)                 -- 4 warps x 32 threads
extern "C" __global__ void matvec_q8_0_aligned_residual(
    const char* __restrict__ weight_q8_aligned,  // [out_dim * num_blocks_per_row * 36]
    const float* __restrict__ x,                 // [in_dim]
    const float* __restrict__ residual,          // [out_dim], added to output
    float* __restrict__ out,                     // [out_dim]
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
        unsigned int x_base = ib * Q8_0_BLOCK_SIZE;

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

            const char* bp = weight_q8_aligned
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * Q8_ALIGNED_BYTES;

            unsigned short scale_bits = (unsigned short)(unsigned char)bp[0]
                                      | ((unsigned short)(unsigned char)bp[1] << 8);
            float w_scale = f16_bits_to_f32(scale_bits);

            // Aligned int* loads (bp+4 is 4-byte aligned in 36-byte blocks).
            const int* w_int = (const int*)(bp + 4);

            int acc = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                acc = __dp4a(w_int[k], x_packed[k], acc);
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
