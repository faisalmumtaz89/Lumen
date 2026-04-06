// ============================================================================
// Q4_0 dequant-in-register HGEMV: native Q4_0 reads with F16 x-vector shmem.
//
// Same architecture as hgemv_q8_0.cu but for 4-bit quantization.
// Reads 0.5625 B/elem (18 bytes / 32 elements) vs HGEMV's 2 B/elem = 3.55x savings.
//
// Q4_0 block: 18 bytes = 2B f16 scale + 16B nibble pairs (32 elements)
//   De-interleaved layout: elements 0-15 from lo nibbles, elements 16-31 from hi nibbles
//   dequant: float_value = scale * ((float)(nibble) - 8.0f)
//
// Architecture: NR=4 output rows per block, 256 threads (8 warps).
//   - x-vector stored as F16 in shmem (in_dim * 2 bytes, covers up to 24576)
//   - F32 accumulation for numerical stability
//   - 4 output rows per block amortize x-vector load
//
// Grid:  (ceil(out_dim / NR), 1, 1)
// Block: (BLOCK_DIM, 1, 1)
// Shared memory: in_dim * 2 bytes (F16 x-vector)
//
// NVRTC-compatible: no system includes, extern "C" linkage.
// ============================================================================

#define NR              4       // rows per thread block
#define WARP_SIZE       32
#define BLOCK_DIM       256     // 8 warps
#define Q4_BLOCK_ELEMS  32      // elements per Q4_0 block
#define Q4_BLOCK_BYTES  18      // bytes per Q4_0 block

// Hardware f16->f32 conversion via PTX.
__device__ __forceinline__ float f16_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

// Hardware f32->f16 conversion via PTX.
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

// Q4_0 HGEMV: dequant-in-register with F16 x-vector in shared memory.
extern "C" __global__ void hgemv_q4_0(
    const char* __restrict__ weight_q4,  // [out_dim * num_blocks * 18] raw Q4_0 bytes
    const float* __restrict__ x,         // [in_dim] F32 input vector
    float* __restrict__ out,             // [out_dim] F32 output vector
    unsigned int out_dim,
    unsigned int in_dim)
{
    extern __shared__ unsigned short x_f16[];  // [in_dim] — x-vector as F16

    const unsigned int r0 = blockIdx.x * NR;
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane    = threadIdx.x % WARP_SIZE;
    const unsigned int num_blocks = in_dim / Q4_BLOCK_ELEMS;
    const unsigned long long row_bytes = (unsigned long long)num_blocks * Q4_BLOCK_BYTES;

    // Cooperatively convert x to F16 and store in shmem.
    for (unsigned int i = threadIdx.x; i < in_dim; i += BLOCK_DIM) {
        x_f16[i] = f32_to_f16(x[i]);
    }
    __syncthreads();

    // Per-row F32 accumulators.
    float sumf[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) sumf[r] = 0.0f;

    // Main loop: each thread handles one Q4_0 block per iteration.
    for (unsigned int ib = threadIdx.x; ib < num_blocks; ib += BLOCK_DIM) {
        const unsigned int x_base = ib * Q4_BLOCK_ELEMS;

        // Load 32 x-values from shmem (F16) and convert to F32.
        // Load as 16 x uint (32-bit), each holding 2 packed F16 values.
        float xv[32];
        const unsigned int* x_u32 = (const unsigned int*)(x_f16 + x_base);
        #pragma unroll
        for (int k = 0; k < 16; k++) {
            unsigned int packed = x_u32[k];
            xv[2 * k + 0] = f16_to_f32((unsigned short)(packed & 0xFFFFu));
            xv[2 * k + 1] = f16_to_f32((unsigned short)(packed >> 16));
        }

        // Process all NR=4 output rows.
        #pragma unroll
        for (int row = 0; row < NR; row++) {
            if (r0 + row >= out_dim) break;

            const char* bp = weight_q4
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * Q4_BLOCK_BYTES;

            // Read f16 scale.
            unsigned short scale_bits = (unsigned short)(unsigned char)bp[0]
                                      | ((unsigned short)(unsigned char)bp[1] << 8);
            float scale = f16_to_f32(scale_bits);

            // Unpack 16 bytes of nibble pairs into 32 dequantized values.
            const unsigned char* qs = (const unsigned char*)(bp + 2);
            float block_sum = 0.0f;

            #pragma unroll
            for (int i = 0; i < 16; i++) {
                unsigned int byte_val = qs[i];
                float dq_lo = (float)(byte_val & 0x0Fu) - 8.0f;
                float dq_hi = (float)((byte_val >> 4) & 0x0Fu) - 8.0f;
                block_sum += dq_lo * xv[i] + dq_hi * xv[i + 16];
            }

            sumf[row] += scale * block_sum;
        }
    }

    // === Cross-warp reduction ===
    const unsigned int num_warps = BLOCK_DIM / WARP_SIZE;

    #pragma unroll
    for (int r = 0; r < NR; r++) {
        sumf[r] = warp_reduce_sum(sumf[r]);
    }

    __syncthreads();

    // Reuse shmem for reduction (need NR * num_warps floats).
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
                    out[r0 + r] = val;
                }
            }
        }
    }
}

// Q4_0 HGEMV with fused residual addition:
// out[i] = dot(dequant(weight_q4[i]), x) + residual[i]
extern "C" __global__ void hgemv_q4_0_residual(
    const char* __restrict__ weight_q4,  // [out_dim * num_blocks * 18]
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
    const unsigned int num_blocks = in_dim / Q4_BLOCK_ELEMS;
    const unsigned long long row_bytes = (unsigned long long)num_blocks * Q4_BLOCK_BYTES;

    for (unsigned int i = threadIdx.x; i < in_dim; i += BLOCK_DIM) {
        x_f16[i] = f32_to_f16(x[i]);
    }
    __syncthreads();

    float sumf[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) sumf[r] = 0.0f;

    for (unsigned int ib = threadIdx.x; ib < num_blocks; ib += BLOCK_DIM) {
        const unsigned int x_base = ib * Q4_BLOCK_ELEMS;

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

            const char* bp = weight_q4
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * Q4_BLOCK_BYTES;

            unsigned short scale_bits = (unsigned short)(unsigned char)bp[0]
                                      | ((unsigned short)(unsigned char)bp[1] << 8);
            float scale = f16_to_f32(scale_bits);

            const unsigned char* qs = (const unsigned char*)(bp + 2);
            float block_sum = 0.0f;

            #pragma unroll
            for (int i = 0; i < 16; i++) {
                unsigned int byte_val = qs[i];
                float dq_lo = (float)(byte_val & 0x0Fu) - 8.0f;
                float dq_hi = (float)((byte_val >> 4) & 0x0Fu) - 8.0f;
                block_sum += dq_lo * xv[i] + dq_hi * xv[i + 16];
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
