// ============================================================================
// Q8_0 dequant-in-register HGEMV: native Q8_0 reads with F16 x-vector shmem.
//
// Problem: the smem kernel stores x as F32 in shared memory (4 B/elem),
// limiting coverage to in_dim <= 12288 (48KB shmem / 4). For FFN down
// projections (in_dim=14336 on Llama 8B), the smem kernel can't fire and
// the fallback is HGEMV via pre-dequanted F16 cache (2 B/elem reads).
//
// Solution: store x as F16 in shmem (2 B/elem), halving shmem usage.
// Covers up to in_dim = 24576 (49152 / 2). Dequant Q8_0 in registers,
// convert x back to F32 for the FMA. Net effect: reads 1.0625 B/elem
// from HBM (native Q8_0) instead of 2 B/elem (HGEMV F16 cache).
//
// Architecture: NR=4 output rows per block, 256 threads (8 warps).
//   - x-vector converted to F16 and stored in shmem ONCE (in_dim * 2 bytes)
//   - Each thread processes one Q8_0 block (32 elements) per iteration
//   - 4 output rows per block amortize the x-vector shmem load 4x
//   - F32 accumulation for numerical stability
//   - Warp shuffle + shmem for cross-warp reduction
//
// Q8_0 block: 34 bytes = 2B f16 scale + 32B int8 quants (32 elements)
// Bandwidth: 1.0625 B/elem (vs 2 B/elem for HGEMV F16 cache) = 1.88x savings
//
// Grid:  (ceil(out_dim / NR), 1, 1)
// Block: (BLOCK_DIM, 1, 1)
// Shared memory: in_dim * 2 bytes (F16 x-vector)
//
// NVRTC-compatible: no system includes, extern "C" linkage.
// ============================================================================

#define NR              4       // rows per thread block (4x amortization)
#define WARP_SIZE       32
#define BLOCK_DIM       256     // 8 warps
#define Q8_BLOCK_SIZE   32      // elements per Q8_0 block
#define Q8_BLOCK_BYTES  34      // bytes per Q8_0 block

// Hardware f16->f32 conversion via PTX (single cycle on SM 53+).
__device__ __forceinline__ float f16_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

// Hardware f32->f16 conversion via PTX (single cycle on SM 53+).
__device__ __forceinline__ unsigned short f32_to_f16(float val) {
    unsigned short result;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(result) : "f"(val));
    return result;
}

// Warp-level reduction: sum all 32 lanes via butterfly shuffle.
__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

// Q8_0 HGEMV: dequant-in-register with F16 x-vector in shared memory.
//
// shmem layout: unsigned short x_f16[in_dim]  (in_dim * 2 bytes)
// After main loop, reuse shmem for cross-warp reduction.
extern "C" __global__ void hgemv_q8_0(
    const char* __restrict__ weight_q8,  // [out_dim * num_blocks * 34] raw Q8_0 bytes
    const float* __restrict__ x,         // [in_dim] F32 input vector
    float* __restrict__ out,             // [out_dim] F32 output vector
    unsigned int out_dim,
    unsigned int in_dim)
{
    extern __shared__ unsigned short x_f16[];  // [in_dim] — x-vector as F16

    const unsigned int r0 = blockIdx.x * NR;
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane    = threadIdx.x % WARP_SIZE;
    const unsigned int num_blocks = in_dim / Q8_BLOCK_SIZE;
    const unsigned long long row_bytes = (unsigned long long)num_blocks * Q8_BLOCK_BYTES;

    // Cooperatively convert x to F16 and store in shmem.
    // 256 threads, each converts multiple elements.
    for (unsigned int i = threadIdx.x; i < in_dim; i += BLOCK_DIM) {
        x_f16[i] = f32_to_f16(x[i]);
    }
    __syncthreads();

    // Per-row F32 accumulators (NR=4 rows).
    float sumf[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) sumf[r] = 0.0f;

    // Main loop: each thread processes one Q8_0 block (32 elements) per iteration.
    // Stride by BLOCK_DIM blocks across the in_dim.
    for (unsigned int ib = threadIdx.x; ib < num_blocks; ib += BLOCK_DIM) {
        const unsigned int x_base = ib * Q8_BLOCK_SIZE;

        // Load 32 x-values from shmem (F16) and convert to F32 in registers.
        // Each Q8_0 block has 32 elements. Load as 16 x uint (32-bit) for
        // bandwidth: each uint holds 2 consecutive F16 values.
        float xv[32];
        const unsigned int* x_u32 = (const unsigned int*)(x_f16 + x_base);
        #pragma unroll
        for (int k = 0; k < 16; k++) {
            unsigned int packed = x_u32[k];
            xv[2 * k + 0] = f16_to_f32((unsigned short)(packed & 0xFFFFu));
            xv[2 * k + 1] = f16_to_f32((unsigned short)(packed >> 16));
        }

        // Process all NR=4 output rows with the cached x-values.
        #pragma unroll
        for (int row = 0; row < NR; row++) {
            if (r0 + row >= out_dim) break;

            const char* bp = weight_q8
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * Q8_BLOCK_BYTES;

            // Read f16 scale (2 bytes, little-endian).
            unsigned short scale_bits = (unsigned short)(unsigned char)bp[0]
                                      | ((unsigned short)(unsigned char)bp[1] << 8);
            float scale = f16_to_f32(scale_bits);

            // Dot product: scale * sum(int8_quant[j] * x[j]) for j in [0..32)
            const signed char* qs = (const signed char*)(bp + 2);
            float block_sum = 0.0f;

            #pragma unroll
            for (int j = 0; j < 32; j++) {
                block_sum += (float)qs[j] * xv[j];
            }

            sumf[row] += scale * block_sum;
        }
    }

    // === Cross-warp reduction ===
    // Intra-warp reduction via shuffle (5 steps).
    const unsigned int num_warps = BLOCK_DIM / WARP_SIZE;

    #pragma unroll
    for (int r = 0; r < NR; r++) {
        sumf[r] = warp_reduce_sum(sumf[r]);
    }

    __syncthreads();  // Ensure shmem x-values are no longer needed

    // Reuse shmem for cross-warp reduction (need NR * num_warps floats).
    float* reduce_smem = (float*)x_f16;

    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            reduce_smem[r * num_warps + warp_id] = sumf[r];
        }
    }

    __syncthreads();

    // Warp 0 performs final reduction across all warps.
    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            if (r0 + r < out_dim) {
                float val = (lane < num_warps) ? reduce_smem[r * num_warps + lane] : 0.0f;
                val = warp_reduce_sum(val);
                if (lane == 0) {
                    out[r0 + r] = val;
                }
            }
        }
    }
}

// Q8_0 HGEMV with fused residual addition:
// out[i] = dot(dequant(weight_q8[i]), x) + residual[i]
extern "C" __global__ void hgemv_q8_0_residual(
    const char* __restrict__ weight_q8,  // [out_dim * num_blocks * 34]
    const float* __restrict__ x,         // [in_dim]
    const float* __restrict__ residual,  // [out_dim]
    float* __restrict__ out,             // [out_dim]
    unsigned int out_dim,
    unsigned int in_dim)
{
    extern __shared__ unsigned short x_f16[];

    const unsigned int r0 = blockIdx.x * NR;
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane    = threadIdx.x % WARP_SIZE;
    const unsigned int num_blocks = in_dim / Q8_BLOCK_SIZE;
    const unsigned long long row_bytes = (unsigned long long)num_blocks * Q8_BLOCK_BYTES;

    // Convert x to F16 and store in shmem.
    for (unsigned int i = threadIdx.x; i < in_dim; i += BLOCK_DIM) {
        x_f16[i] = f32_to_f16(x[i]);
    }
    __syncthreads();

    float sumf[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) sumf[r] = 0.0f;

    for (unsigned int ib = threadIdx.x; ib < num_blocks; ib += BLOCK_DIM) {
        const unsigned int x_base = ib * Q8_BLOCK_SIZE;

        float xv[32];
        const unsigned int* x_u32 = (const unsigned int*)(x_f16 + x_base);
        #pragma unroll
        for (int k = 0; k < 16; k++) {
            unsigned int packed = x_u32[k];
            xv[2 * k + 0] = f16_to_f32((unsigned short)(packed & 0xFFFFu));
            xv[2 * k + 1] = f16_to_f32((unsigned short)(packed >> 16));
        }

        #pragma unroll
        for (int row = 0; row < NR; row++) {
            if (r0 + row >= out_dim) break;

            const char* bp = weight_q8
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * Q8_BLOCK_BYTES;

            unsigned short scale_bits = (unsigned short)(unsigned char)bp[0]
                                      | ((unsigned short)(unsigned char)bp[1] << 8);
            float scale = f16_to_f32(scale_bits);

            const signed char* qs = (const signed char*)(bp + 2);
            float block_sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < 32; j++) {
                block_sum += (float)qs[j] * xv[j];
            }

            sumf[row] += scale * block_sum;
        }
    }

    const unsigned int num_warps = BLOCK_DIM / WARP_SIZE;

    #pragma unroll
    for (int r = 0; r < NR; r++) {
        sumf[r] = warp_reduce_sum(sumf[r]);
    }

    __syncthreads();

    float* reduce_smem = (float*)x_f16;

    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            reduce_smem[r * num_warps + warp_id] = sumf[r];
        }
    }

    __syncthreads();

    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            if (r0 + r < out_dim) {
                float val = (lane < num_warps) ? reduce_smem[r * num_warps + lane] : 0.0f;
                val = warp_reduce_sum(val);
                if (lane == 0) {
                    out[r0 + r] = val + residual[r0 + r];
                }
            }
        }
    }
}
