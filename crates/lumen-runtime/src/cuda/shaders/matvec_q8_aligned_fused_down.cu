// ==========================================================================
// Fused Down Projection Kernels: inline F32->Q8_1 quantization + dp4a matvec.
//
// Eliminates the separate quantize_f32_to_q8_1 dispatch by quantizing the
// input vector to Q8_1 on-the-fly within each thread's block iteration.
//
// Four kernels:
//   1. matvec_q8_aligned_f32:          W * quantize(x_f32) -> out
//   2. matvec_q8_aligned_f32_residual: W * quantize(x_f32) + residual -> out
//   3. matvec_q8_aligned_f32_swiglu:   W * quantize(silu(gate)*up) -> out
//   4. matvec_q8_aligned_f32_swiglu_residual: same + residual
//
// Kernels 1-2: Replace quantize_f32_to_q8_1 + matvec_q8_aligned_q8_1 (2->1).
//   Used after fused_glu_gemv when SwiGLU is already computed.
//
// Kernels 3-4: Replace swiglu_inplace + quantize_f32_to_q8_1 +
//   matvec_q8_aligned_q8_1 (3->1).
//   Used when gate and up are separate F32 buffers.
//
// Inline quantization per thread (no warp-level reduction needed):
//   Each thread processes one block of 32 F32 values per iteration.
//   - Load 32 floats, find per-thread absmax, compute scale
//   - Quantize to int8, pack into 8 int32 words
//   - dp4a dot product against Q8Aligned weight block
//   This is equivalent to the separate quantize_f32_to_q8_1 kernel but
//   done per-thread instead of per-warp, avoiding a global memory round-trip.
//
// Architecture: NR=2 rows per block, 128 threads (4 warps).
// Requires compute capability >= 6.1 for __dp4a() (Pascal+).
// in_dim must be a multiple of 32 (Q8_0 block size).
//
// NVRTC-compatible: no system includes, extern "C" linkage.
// ==========================================================================

#define NR       2     // rows per thread block
#define NW       32    // warp size
#define THREADS_PER_BLOCK 128  // 4 warps
#define NWARPS   (THREADS_PER_BLOCK / NW)  // 4
#define Q8_BLOCK_SIZE     32   // elements per Q8 block
#define Q8_ALIGNED_BYTES  36   // 2B f16 scale + 2B pad + 32B int8 data

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
// Kernel 1: Q8Aligned weight x F32 input -> F32 output (inline quantize + dp4a).
//
// Reads F32 input, quantizes to Q8_1 in registers per-block, then dp4a
// against Q8Aligned weights. Eliminates the separate quantize kernel.
//
// Grid:  (ceil(out_dim / NR), 1, 1)
// Block: (128, 1, 1)
// ==========================================================================
extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 1) void matvec_q8_aligned_f32(
    const char* __restrict__ weight_q8_aligned,  // [out_dim * nb * 36] Q8Aligned bytes
    const float* __restrict__ input_f32,         // [in_dim] F32 input vector
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
        unsigned int base = ib * Q8_BLOCK_SIZE;

        // --- Inline F32 -> Q8_1 quantization for this block of 32 elements ---
        // Load 32 F32 values.
        float xf[32];
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            xf[j] = input_f32[base + j];
        }

        // Find per-thread absmax (all 32 values belong to this thread).
        float amax = 0.0f;
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            float a = xf[j] < 0.0f ? -xf[j] : xf[j];
            amax = a > amax ? a : amax;
        }

        // Compute scale and inverse scale.
        float x_scale = amax / 127.0f;
        float scale_inv = amax > 0.0f ? 127.0f / amax : 0.0f;

        // Quantize to int8 and pack into 8 int32 words for dp4a.
        int xv[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int b0 = __float2int_rn(xf[k * 4 + 0] * scale_inv);
            int b1 = __float2int_rn(xf[k * 4 + 1] * scale_inv);
            int b2 = __float2int_rn(xf[k * 4 + 2] * scale_inv);
            int b3 = __float2int_rn(xf[k * 4 + 3] * scale_inv);
            // Clamp to [-127, 127].
            b0 = b0 < -127 ? -127 : (b0 > 127 ? 127 : b0);
            b1 = b1 < -127 ? -127 : (b1 > 127 ? 127 : b1);
            b2 = b2 < -127 ? -127 : (b2 > 127 ? 127 : b2);
            b3 = b3 < -127 ? -127 : (b3 > 127 ? 127 : b3);
            // Pack 4 int8 into one int32 (little-endian byte order for dp4a).
            xv[k] = (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24);
        }

        // --- dp4a dot product against NR weight rows ---
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
                out[r0 + r] = total;
            }
        }
    }
}

// ==========================================================================
// Kernel 2: Q8Aligned weight x F32 input + residual -> F32 output.
//
// Same as matvec_q8_aligned_f32 but with fused residual addition.
// ==========================================================================
extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 1) void matvec_q8_aligned_f32_residual(
    const char* __restrict__ weight_q8_aligned,
    const float* __restrict__ input_f32,
    const float* __restrict__ residual,
    float* __restrict__ out,
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
        unsigned int base = ib * Q8_BLOCK_SIZE;

        float xf[32];
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            xf[j] = input_f32[base + j];
        }

        float amax = 0.0f;
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            float a = xf[j] < 0.0f ? -xf[j] : xf[j];
            amax = a > amax ? a : amax;
        }

        float x_scale = amax / 127.0f;
        float scale_inv = amax > 0.0f ? 127.0f / amax : 0.0f;

        int xv[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int b0 = __float2int_rn(xf[k * 4 + 0] * scale_inv);
            int b1 = __float2int_rn(xf[k * 4 + 1] * scale_inv);
            int b2 = __float2int_rn(xf[k * 4 + 2] * scale_inv);
            int b3 = __float2int_rn(xf[k * 4 + 3] * scale_inv);
            b0 = b0 < -127 ? -127 : (b0 > 127 ? 127 : b0);
            b1 = b1 < -127 ? -127 : (b1 > 127 ? 127 : b1);
            b2 = b2 < -127 ? -127 : (b2 > 127 ? 127 : b2);
            b3 = b3 < -127 ? -127 : (b3 > 127 ? 127 : b3);
            xv[k] = (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24);
        }

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

// ==========================================================================
// Kernel 3: Q8Aligned weight x SwiGLU(gate, up) -> F32 output.
//
// Fuses: SwiGLU activation + F32->Q8_1 quantization + dp4a matvec.
// Replaces 3 separate dispatches (swiglu + quantize + matvec) with 1.
//
// Input: separate F32 gate[] and up[] buffers (from gate/up projections).
// Each thread computes silu(gate[j]) * up[j] for its 32 elements,
// quantizes inline, and does dp4a against weight rows.
//
// Grid:  (ceil(out_dim / NR), 1, 1)
// Block: (128, 1, 1)
// ==========================================================================
extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 1) void matvec_q8_aligned_f32_swiglu(
    const char* __restrict__ weight_q8_aligned,  // [out_dim * nb * 36] Q8Aligned bytes
    const float* __restrict__ gate,              // [in_dim] F32 gate projection
    const float* __restrict__ up,                // [in_dim] F32 up projection
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
        unsigned int base = ib * Q8_BLOCK_SIZE;

        // --- Compute SwiGLU: silu(gate[j]) * up[j] for 32 elements ---
        float xf[32];
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            float g = gate[base + j];
            float silu_g = g / (1.0f + expf(-g));
            xf[j] = silu_g * up[base + j];
        }

        // --- Inline quantize to Q8_1 ---
        float amax = 0.0f;
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            float a = xf[j] < 0.0f ? -xf[j] : xf[j];
            amax = a > amax ? a : amax;
        }

        float x_scale = amax / 127.0f;
        float scale_inv = amax > 0.0f ? 127.0f / amax : 0.0f;

        int xv[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int b0 = __float2int_rn(xf[k * 4 + 0] * scale_inv);
            int b1 = __float2int_rn(xf[k * 4 + 1] * scale_inv);
            int b2 = __float2int_rn(xf[k * 4 + 2] * scale_inv);
            int b3 = __float2int_rn(xf[k * 4 + 3] * scale_inv);
            b0 = b0 < -127 ? -127 : (b0 > 127 ? 127 : b0);
            b1 = b1 < -127 ? -127 : (b1 > 127 ? 127 : b1);
            b2 = b2 < -127 ? -127 : (b2 > 127 ? 127 : b2);
            b3 = b3 < -127 ? -127 : (b3 > 127 ? 127 : b3);
            xv[k] = (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24);
        }

        // --- dp4a dot product against NR weight rows ---
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
                out[r0 + r] = total;
            }
        }
    }
}

// ==========================================================================
// Kernel 4: Q8Aligned weight x SwiGLU(gate, up) + residual -> F32 output.
//
// Same as matvec_q8_aligned_f32_swiglu but with fused residual addition.
// ==========================================================================
extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 1) void matvec_q8_aligned_f32_swiglu_residual(
    const char* __restrict__ weight_q8_aligned,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const float* __restrict__ residual,
    float* __restrict__ out,
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
        unsigned int base = ib * Q8_BLOCK_SIZE;

        float xf[32];
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            float g = gate[base + j];
            float silu_g = g / (1.0f + expf(-g));
            xf[j] = silu_g * up[base + j];
        }

        float amax = 0.0f;
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            float a = xf[j] < 0.0f ? -xf[j] : xf[j];
            amax = a > amax ? a : amax;
        }

        float x_scale = amax / 127.0f;
        float scale_inv = amax > 0.0f ? 127.0f / amax : 0.0f;

        int xv[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int b0 = __float2int_rn(xf[k * 4 + 0] * scale_inv);
            int b1 = __float2int_rn(xf[k * 4 + 1] * scale_inv);
            int b2 = __float2int_rn(xf[k * 4 + 2] * scale_inv);
            int b3 = __float2int_rn(xf[k * 4 + 3] * scale_inv);
            b0 = b0 < -127 ? -127 : (b0 > 127 ? 127 : b0);
            b1 = b1 < -127 ? -127 : (b1 > 127 ? 127 : b1);
            b2 = b2 < -127 ? -127 : (b2 > 127 ? 127 : b2);
            b3 = b3 < -127 ? -127 : (b3 > 127 ? 127 : b3);
            xv[k] = (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24);
        }

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
