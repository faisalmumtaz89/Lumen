// ============================================================================
// Fused Gate+Up+SwiGLU GEMV kernel for single-token decode.
//
// Reads the input vector ONCE and computes BOTH gate and up projections
// simultaneously, applying SwiGLU inline:
//   output[row] = silu(dot(w_gate[row], normed_x)) * dot(w_up[row], normed_x)
//
// RMSNorm is fused: the rms_scale is pre-computed by compute_rms_scale(),
// and normalization is applied inline during the dot product (same two-pass
// approach as fused_rmsnorm_matvec.cu).
//
// Savings vs separate dispatch:
//   - Eliminates 2-4 kernel launches per layer:
//     (rmsnorm + f32_to_f16 + gate HGEMV + up HGEMV + swiglu)
//     becomes: (compute_rms_scale + fused_glu_gemv)
//   - Input vector bandwidth halved (read once, not twice)
//   - No intermediate gate[] and up[] buffers needed
//   - SwiGLU applied in-register (no global memory round-trip)
//
// Architecture: NR=2 output rows per block, 256 threads (8 warps).
//   - Each block computes NR output rows of the fused gate+up+SwiGLU
//   - x-vector cached in shared memory (F32: in_dim*4, F16: in_dim*2 bytes)
//   - For each row: accumulate gate_dot and up_dot simultaneously
//   - After reduction: output = silu(gate_dot) * up_dot
//   - Norm weights applied inline: normed_x[j] = x[j] * rms_scale * norm_w[j]
//
// Variants:
//   - fused_glu_gemv_q8_0:  Q8_0 gate+up weights, F32 x in shmem
//   - fused_glu_gemv_q4_0:  Q4_0 gate+up weights, F32 x in shmem
//   - fused_glu_gemv_f16:   F16 gate+up weights, F32 x in shmem
//   - fused_glu_gemv_q8_0_hg: Q8_0 gate+up weights, F16 x in shmem (large dims)
//   - fused_glu_gemv_q4_0_hg: Q4_0 gate+up weights, F16 x in shmem (large dims)
//
// Grid:  (ceil(inter_dim / NR), 1, 1)
// Block: (BLOCK_DIM, 1, 1)
// Shared memory: varies by variant (see below)
//
// NVRTC-compatible: no system includes, extern "C" linkage.
// ============================================================================

#define NR              2       // rows per thread block
#define WARP_SIZE       32
#define BLOCK_DIM       256     // 8 warps
#define Q8_BLOCK_SIZE   32      // elements per Q8_0 block
#define Q8_BLOCK_BYTES  34      // bytes per Q8_0 block (2B scale + 32B int8)
#define Q8_ALIGNED_BYTES 36     // bytes per aligned Q8_0 block (2B scale + 2B pad + 32B int8)
#define Q4_BLOCK_ELEMS  32      // elements per Q4_0 block
#define Q4_BLOCK_BYTES  18      // bytes per Q4_0 block (2B scale + 16B nibbles)

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


// ============================================================================
// Q8_0 fused gate+up+SwiGLU GEMV with inline RMSNorm.
//
// Shared memory: in_dim * 4 bytes (F32 x-vector) + in_dim * 4 bytes (F32 normed x).
// Wait -- we normalize inline, so we only need the raw x + norm_weight in shmem.
// Actually, to avoid redundant multiply per row, we pre-compute normed_x in shmem:
//   normed_x[j] = x[j] * rms_scale * norm_weight[j]
// Then each row just does: dot(w[row], normed_x)
//
// Shared memory: in_dim * 4 bytes (normed x-vector, F32).
// Covers up to in_dim = 12288 (48KB / 4 = 12288 floats).
//
// Grid:  (ceil(inter_dim / NR), 1, 1)
// Block: (BLOCK_DIM, 1, 1)
// ============================================================================
extern "C" __global__ void fused_glu_gemv_q8_0(
    const char*  __restrict__ w_gate,       // [inter_dim * num_blocks * 34] Q8_0
    const char*  __restrict__ w_up,         // [inter_dim * num_blocks * 34] Q8_0
    const float* __restrict__ x,            // [hidden_dim] input activation
    const float* __restrict__ norm_weight,  // [hidden_dim] RMSNorm gamma
    const float* __restrict__ rms_scale,    // [1] precomputed 1/sqrt(mean(x^2)+eps)
    float*       __restrict__ output,       // [inter_dim] silu(gate) * up
    unsigned int inter_dim,
    unsigned int hidden_dim)
{
    extern __shared__ float nx_smem[];  // [hidden_dim] normed x-vector

    const unsigned int r0 = blockIdx.x * NR;
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane    = threadIdx.x % WARP_SIZE;
    const unsigned int num_blocks = hidden_dim / Q8_BLOCK_SIZE;
    const unsigned long long row_bytes = (unsigned long long)num_blocks * Q8_BLOCK_BYTES;

    const float scale = rms_scale[0];

    // Cooperatively compute normed x-vector and store in shmem.
    // normed_x[i] = x[i] * rms_scale * norm_weight[i]
    for (unsigned int i = threadIdx.x; i < hidden_dim; i += BLOCK_DIM) {
        nx_smem[i] = x[i] * scale * norm_weight[i];
    }
    __syncthreads();

    // Per-row accumulators for gate and up dot products.
    float gate_sum[NR];
    float up_sum[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) {
        gate_sum[r] = 0.0f;
        up_sum[r]   = 0.0f;
    }

    // Main loop: each thread processes one Q8_0 block (32 elements) per iteration.
    for (unsigned int ib = threadIdx.x; ib < num_blocks; ib += BLOCK_DIM) {
        const unsigned int x_base = ib * Q8_BLOCK_SIZE;

        // Load normed x-values from shmem into registers.
        float xv[32];
        const float4* x4 = (const float4*)(nx_smem + x_base);
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            float4 v = x4[k];
            xv[k * 4 + 0] = v.x;
            xv[k * 4 + 1] = v.y;
            xv[k * 4 + 2] = v.z;
            xv[k * 4 + 3] = v.w;
        }

        // Process all NR output rows with the same cached normed x-values.
        #pragma unroll
        for (int row = 0; row < NR; row++) {
            if (r0 + row >= inter_dim) break;

            // Gate weight block.
            const char* gp = w_gate
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * Q8_BLOCK_BYTES;
            unsigned short g_scale_bits = (unsigned short)(unsigned char)gp[0]
                                        | ((unsigned short)(unsigned char)gp[1] << 8);
            float g_scale = f16_to_f32(g_scale_bits);
            const signed char* gq = (const signed char*)(gp + 2);

            // Up weight block.
            const char* up_ = w_up
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * Q8_BLOCK_BYTES;
            unsigned short u_scale_bits = (unsigned short)(unsigned char)up_[0]
                                        | ((unsigned short)(unsigned char)up_[1] << 8);
            float u_scale = f16_to_f32(u_scale_bits);
            const signed char* uq = (const signed char*)(up_ + 2);

            float g_block_sum = 0.0f;
            float u_block_sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < 32; j++) {
                g_block_sum += (float)gq[j] * xv[j];
                u_block_sum += (float)uq[j] * xv[j];
            }

            gate_sum[row] += g_scale * g_block_sum;
            up_sum[row]   += u_scale * u_block_sum;
        }
    }

    // Cross-warp reduction via shared memory.
    // Reuse nx_smem for reduction (normed x is no longer needed).
    const unsigned int num_warps = BLOCK_DIM / WARP_SIZE;

    // Intra-warp reduction for gate sums.
    #pragma unroll
    for (int r = 0; r < NR; r++) {
        gate_sum[r] = warp_reduce_sum(gate_sum[r]);
    }

    __syncthreads();

    // Warp 0 lane 0 of each warp writes gate partial sums.
    float* reduce_smem = nx_smem;
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            reduce_smem[r * num_warps + warp_id] = gate_sum[r];
        }
    }
    __syncthreads();

    // Warp 0 reduces gate sums.
    float final_gate[NR];
    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            float val = (lane < num_warps) ? reduce_smem[r * num_warps + lane] : 0.0f;
            val = warp_reduce_sum(val);
            final_gate[r] = val;
        }
    }
    __syncthreads();

    // Now reduce up sums.
    #pragma unroll
    for (int r = 0; r < NR; r++) {
        up_sum[r] = warp_reduce_sum(up_sum[r]);
    }

    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            reduce_smem[r * num_warps + warp_id] = up_sum[r];
        }
    }
    __syncthreads();

    // Warp 0 reduces up sums and writes final output with SwiGLU.
    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            if (r0 + r < inter_dim) {
                float val = (lane < num_warps) ? reduce_smem[r * num_warps + lane] : 0.0f;
                val = warp_reduce_sum(val);
                if (lane == 0) {
                    // SwiGLU: silu(gate) * up
                    float g = final_gate[r];
                    float silu_g = g / (1.0f + expf(-g));
                    output[r0 + r] = silu_g * val;
                }
            }
        }
    }
}


// ============================================================================
// Q4_0 fused gate+up+SwiGLU GEMV with inline RMSNorm.
//
// Same approach as Q8_0 variant but for 4-bit quantization.
// Shared memory: hidden_dim * 4 bytes (normed F32 x-vector).
// Covers up to hidden_dim = 12288 (48KB shmem limit).
// ============================================================================
extern "C" __global__ void fused_glu_gemv_q4_0(
    const char*  __restrict__ w_gate,       // [inter_dim * num_blocks * 18] Q4_0
    const char*  __restrict__ w_up,         // [inter_dim * num_blocks * 18] Q4_0
    const float* __restrict__ x,            // [hidden_dim] input activation
    const float* __restrict__ norm_weight,  // [hidden_dim] RMSNorm gamma
    const float* __restrict__ rms_scale,    // [1] precomputed rms_scale
    float*       __restrict__ output,       // [inter_dim] silu(gate) * up
    unsigned int inter_dim,
    unsigned int hidden_dim)
{
    extern __shared__ float nx_smem[];

    const unsigned int r0 = blockIdx.x * NR;
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane    = threadIdx.x % WARP_SIZE;
    const unsigned int num_blocks = hidden_dim / Q4_BLOCK_ELEMS;
    const unsigned long long row_bytes = (unsigned long long)num_blocks * Q4_BLOCK_BYTES;

    const float scale = rms_scale[0];

    // Pre-compute normed x-vector in shmem.
    for (unsigned int i = threadIdx.x; i < hidden_dim; i += BLOCK_DIM) {
        nx_smem[i] = x[i] * scale * norm_weight[i];
    }
    __syncthreads();

    float gate_sum[NR];
    float up_sum[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) {
        gate_sum[r] = 0.0f;
        up_sum[r]   = 0.0f;
    }

    for (unsigned int ib = threadIdx.x; ib < num_blocks; ib += BLOCK_DIM) {
        const unsigned int x_base = ib * Q4_BLOCK_ELEMS;

        // Load normed x-values from shmem into registers.
        float xv[32];
        const float4* x4 = (const float4*)(nx_smem + x_base);
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            float4 v = x4[k];
            xv[k * 4 + 0] = v.x;
            xv[k * 4 + 1] = v.y;
            xv[k * 4 + 2] = v.z;
            xv[k * 4 + 3] = v.w;
        }

        #pragma unroll
        for (int row = 0; row < NR; row++) {
            if (r0 + row >= inter_dim) break;

            // Gate weight block (Q4_0).
            const char* gp = w_gate
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * Q4_BLOCK_BYTES;
            unsigned short g_scale_bits = (unsigned short)(unsigned char)gp[0]
                                        | ((unsigned short)(unsigned char)gp[1] << 8);
            float g_scale = f16_to_f32(g_scale_bits);
            const unsigned char* gq = (const unsigned char*)(gp + 2);

            // Up weight block (Q4_0).
            const char* up_ = w_up
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * Q4_BLOCK_BYTES;
            unsigned short u_scale_bits = (unsigned short)(unsigned char)up_[0]
                                        | ((unsigned short)(unsigned char)up_[1] << 8);
            float u_scale = f16_to_f32(u_scale_bits);
            const unsigned char* uq = (const unsigned char*)(up_ + 2);

            float g_block_sum = 0.0f;
            float u_block_sum = 0.0f;

            // Unpack nibbles (GGML de-interleaved layout):
            // Lo nibble of byte b = element b, hi nibble = element b+16.
            // Dequant: scale * ((float)nibble - 8.0f)
            #pragma unroll
            for (int b = 0; b < 16; b++) {
                unsigned char gb = gq[b];
                unsigned char ub = uq[b];

                float gq_lo = (float)(gb & 0x0F) - 8.0f;
                float gq_hi = (float)(gb >> 4)    - 8.0f;
                float uq_lo = (float)(ub & 0x0F) - 8.0f;
                float uq_hi = (float)(ub >> 4)    - 8.0f;

                g_block_sum += gq_lo * xv[b]     + gq_hi * xv[b + 16];
                u_block_sum += uq_lo * xv[b]     + uq_hi * xv[b + 16];
            }

            gate_sum[row] += g_scale * g_block_sum;
            up_sum[row]   += u_scale * u_block_sum;
        }
    }

    // Cross-warp reduction + SwiGLU (same as Q8_0 variant).
    const unsigned int num_warps = BLOCK_DIM / WARP_SIZE;

    #pragma unroll
    for (int r = 0; r < NR; r++) {
        gate_sum[r] = warp_reduce_sum(gate_sum[r]);
    }
    __syncthreads();

    float* reduce_smem = nx_smem;
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            reduce_smem[r * num_warps + warp_id] = gate_sum[r];
        }
    }
    __syncthreads();

    float final_gate[NR];
    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            float val = (lane < num_warps) ? reduce_smem[r * num_warps + lane] : 0.0f;
            val = warp_reduce_sum(val);
            final_gate[r] = val;
        }
    }
    __syncthreads();

    #pragma unroll
    for (int r = 0; r < NR; r++) {
        up_sum[r] = warp_reduce_sum(up_sum[r]);
    }

    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            reduce_smem[r * num_warps + warp_id] = up_sum[r];
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            if (r0 + r < inter_dim) {
                float val = (lane < num_warps) ? reduce_smem[r * num_warps + lane] : 0.0f;
                val = warp_reduce_sum(val);
                if (lane == 0) {
                    float g = final_gate[r];
                    float silu_g = g / (1.0f + expf(-g));
                    output[r0 + r] = silu_g * val;
                }
            }
        }
    }
}


// ============================================================================
// F16 fused gate+up+SwiGLU GEMV with inline RMSNorm.
//
// Gate and up weights are IEEE F16 (2 bytes/element, row-major).
// Input x is F32, normed inline via precomputed rms_scale.
// Shared memory: hidden_dim * 4 bytes (normed F32 x-vector).
//
// Grid:  (ceil(inter_dim / NR), 1, 1)
// Block: (BLOCK_DIM, 1, 1)
// ============================================================================
extern "C" __global__ void fused_glu_gemv_f16(
    const unsigned short* __restrict__ w_gate,  // [inter_dim * hidden_dim] F16
    const unsigned short* __restrict__ w_up,    // [inter_dim * hidden_dim] F16
    const float* __restrict__ x,                // [hidden_dim] input activation
    const float* __restrict__ norm_weight,      // [hidden_dim] RMSNorm gamma
    const float* __restrict__ rms_scale,        // [1] precomputed rms_scale
    float*       __restrict__ output,           // [inter_dim] silu(gate) * up
    unsigned int inter_dim,
    unsigned int hidden_dim)
{
    extern __shared__ float nx_smem[];

    const unsigned int r0 = blockIdx.x * NR;
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane    = threadIdx.x % WARP_SIZE;

    const float scale = rms_scale[0];

    // Pre-compute normed x-vector in shmem.
    for (unsigned int i = threadIdx.x; i < hidden_dim; i += BLOCK_DIM) {
        nx_smem[i] = x[i] * scale * norm_weight[i];
    }
    __syncthreads();

    float gate_sum[NR];
    float up_sum[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) {
        gate_sum[r] = 0.0f;
        up_sum[r]   = 0.0f;
    }

    // Main loop: iterate over hidden_dim elements.
    // Use float4 vectorization on normed x when hidden_dim is 4-aligned.
    #pragma unroll
    for (int row = 0; row < NR; row++) {
        if (r0 + row >= inter_dim) continue;

        const unsigned short* g_row = w_gate + (unsigned long long)(r0 + row) * hidden_dim;
        const unsigned short* u_row = w_up   + (unsigned long long)(r0 + row) * hidden_dim;

        float g_acc = 0.0f;
        float u_acc = 0.0f;

        for (unsigned int j = threadIdx.x; j < hidden_dim; j += BLOCK_DIM) {
            float nx_val = nx_smem[j];
            g_acc += f16_to_f32(g_row[j]) * nx_val;
            u_acc += f16_to_f32(u_row[j]) * nx_val;
        }

        gate_sum[row] = g_acc;
        up_sum[row]   = u_acc;
    }

    // Cross-warp reduction + SwiGLU.
    const unsigned int num_warps = BLOCK_DIM / WARP_SIZE;

    #pragma unroll
    for (int r = 0; r < NR; r++) {
        gate_sum[r] = warp_reduce_sum(gate_sum[r]);
    }
    __syncthreads();

    float* reduce_smem = nx_smem;
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            reduce_smem[r * num_warps + warp_id] = gate_sum[r];
        }
    }
    __syncthreads();

    float final_gate[NR];
    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            float val = (lane < num_warps) ? reduce_smem[r * num_warps + lane] : 0.0f;
            val = warp_reduce_sum(val);
            final_gate[r] = val;
        }
    }
    __syncthreads();

    #pragma unroll
    for (int r = 0; r < NR; r++) {
        up_sum[r] = warp_reduce_sum(up_sum[r]);
    }

    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            reduce_smem[r * num_warps + warp_id] = up_sum[r];
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            if (r0 + r < inter_dim) {
                float val = (lane < num_warps) ? reduce_smem[r * num_warps + lane] : 0.0f;
                val = warp_reduce_sum(val);
                if (lane == 0) {
                    float g = final_gate[r];
                    float silu_g = g / (1.0f + expf(-g));
                    output[r0 + r] = silu_g * val;
                }
            }
        }
    }
}


// ============================================================================
// Q8_0 fused gate+up+SwiGLU GEMV with F16 x-vector in shmem (large dims).
//
// For hidden_dim > 12288 where F32 shmem is insufficient (> 48KB).
// Stores normed x as F16 in shmem (hidden_dim * 2 bytes).
// Covers up to hidden_dim = 24576 (49152 / 2).
//
// Grid:  (ceil(inter_dim / NR), 1, 1)
// Block: (BLOCK_DIM, 1, 1)
// Shared memory: hidden_dim * 2 bytes (F16 normed x-vector).
// ============================================================================
extern "C" __global__ void fused_glu_gemv_q8_0_hg(
    const char*  __restrict__ w_gate,
    const char*  __restrict__ w_up,
    const float* __restrict__ x,
    const float* __restrict__ norm_weight,
    const float* __restrict__ rms_scale,
    float*       __restrict__ output,
    unsigned int inter_dim,
    unsigned int hidden_dim)
{
    extern __shared__ unsigned short nx_f16[];  // [hidden_dim] normed x as F16

    const unsigned int r0 = blockIdx.x * NR;
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane    = threadIdx.x % WARP_SIZE;
    const unsigned int num_blocks = hidden_dim / Q8_BLOCK_SIZE;
    const unsigned long long row_bytes = (unsigned long long)num_blocks * Q8_BLOCK_BYTES;

    const float scale = rms_scale[0];

    // Pre-compute normed x-vector, convert to F16, store in shmem.
    for (unsigned int i = threadIdx.x; i < hidden_dim; i += BLOCK_DIM) {
        nx_f16[i] = f32_to_f16(x[i] * scale * norm_weight[i]);
    }
    __syncthreads();

    float gate_sum[NR];
    float up_sum[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) {
        gate_sum[r] = 0.0f;
        up_sum[r]   = 0.0f;
    }

    for (unsigned int ib = threadIdx.x; ib < num_blocks; ib += BLOCK_DIM) {
        const unsigned int x_base = ib * Q8_BLOCK_SIZE;

        // Load normed x from shmem (F16) and convert to F32 in registers.
        float xv[32];
        const unsigned int* x_u32 = (const unsigned int*)(nx_f16 + x_base);
        #pragma unroll
        for (int k = 0; k < 16; k++) {
            unsigned int packed = x_u32[k];
            xv[k * 2 + 0] = f16_to_f32((unsigned short)(packed & 0xFFFF));
            xv[k * 2 + 1] = f16_to_f32((unsigned short)(packed >> 16));
        }

        #pragma unroll
        for (int row = 0; row < NR; row++) {
            if (r0 + row >= inter_dim) break;

            const char* gp = w_gate
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * Q8_BLOCK_BYTES;
            unsigned short g_scale_bits = (unsigned short)(unsigned char)gp[0]
                                        | ((unsigned short)(unsigned char)gp[1] << 8);
            float g_scale = f16_to_f32(g_scale_bits);
            const signed char* gq = (const signed char*)(gp + 2);

            const char* up_ = w_up
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * Q8_BLOCK_BYTES;
            unsigned short u_scale_bits = (unsigned short)(unsigned char)up_[0]
                                        | ((unsigned short)(unsigned char)up_[1] << 8);
            float u_scale = f16_to_f32(u_scale_bits);
            const signed char* uq = (const signed char*)(up_ + 2);

            float g_block_sum = 0.0f;
            float u_block_sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < 32; j++) {
                g_block_sum += (float)gq[j] * xv[j];
                u_block_sum += (float)uq[j] * xv[j];
            }

            gate_sum[row] += g_scale * g_block_sum;
            up_sum[row]   += u_scale * u_block_sum;
        }
    }

    // Cross-warp reduction + SwiGLU. Reuse shmem as float*.
    const unsigned int num_warps = BLOCK_DIM / WARP_SIZE;

    #pragma unroll
    for (int r = 0; r < NR; r++) {
        gate_sum[r] = warp_reduce_sum(gate_sum[r]);
    }
    __syncthreads();

    float* reduce_smem = (float*)nx_f16;
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            reduce_smem[r * num_warps + warp_id] = gate_sum[r];
        }
    }
    __syncthreads();

    float final_gate[NR];
    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            float val = (lane < num_warps) ? reduce_smem[r * num_warps + lane] : 0.0f;
            val = warp_reduce_sum(val);
            final_gate[r] = val;
        }
    }
    __syncthreads();

    #pragma unroll
    for (int r = 0; r < NR; r++) {
        up_sum[r] = warp_reduce_sum(up_sum[r]);
    }

    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            reduce_smem[r * num_warps + warp_id] = up_sum[r];
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            if (r0 + r < inter_dim) {
                float val = (lane < num_warps) ? reduce_smem[r * num_warps + lane] : 0.0f;
                val = warp_reduce_sum(val);
                if (lane == 0) {
                    float g = final_gate[r];
                    float silu_g = g / (1.0f + expf(-g));
                    output[r0 + r] = silu_g * val;
                }
            }
        }
    }
}


// ============================================================================
// Q4_0 fused gate+up+SwiGLU GEMV with F16 x-vector in shmem (large dims).
//
// For hidden_dim > 12288 where F32 shmem is insufficient.
// Stores normed x as F16 in shmem (hidden_dim * 2 bytes).
// Covers up to hidden_dim = 24576.
// ============================================================================
extern "C" __global__ void fused_glu_gemv_q4_0_hg(
    const char*  __restrict__ w_gate,
    const char*  __restrict__ w_up,
    const float* __restrict__ x,
    const float* __restrict__ norm_weight,
    const float* __restrict__ rms_scale,
    float*       __restrict__ output,
    unsigned int inter_dim,
    unsigned int hidden_dim)
{
    extern __shared__ unsigned short nx_f16[];

    const unsigned int r0 = blockIdx.x * NR;
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane    = threadIdx.x % WARP_SIZE;
    const unsigned int num_blocks = hidden_dim / Q4_BLOCK_ELEMS;
    const unsigned long long row_bytes = (unsigned long long)num_blocks * Q4_BLOCK_BYTES;

    const float scale = rms_scale[0];

    // Pre-compute normed x as F16 in shmem.
    for (unsigned int i = threadIdx.x; i < hidden_dim; i += BLOCK_DIM) {
        nx_f16[i] = f32_to_f16(x[i] * scale * norm_weight[i]);
    }
    __syncthreads();

    float gate_sum[NR];
    float up_sum[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) {
        gate_sum[r] = 0.0f;
        up_sum[r]   = 0.0f;
    }

    for (unsigned int ib = threadIdx.x; ib < num_blocks; ib += BLOCK_DIM) {
        const unsigned int x_base = ib * Q4_BLOCK_ELEMS;

        // Load normed x from shmem (F16) -> F32 in registers.
        float xv[32];
        const unsigned int* x_u32 = (const unsigned int*)(nx_f16 + x_base);
        #pragma unroll
        for (int k = 0; k < 16; k++) {
            unsigned int packed = x_u32[k];
            xv[k * 2 + 0] = f16_to_f32((unsigned short)(packed & 0xFFFF));
            xv[k * 2 + 1] = f16_to_f32((unsigned short)(packed >> 16));
        }

        #pragma unroll
        for (int row = 0; row < NR; row++) {
            if (r0 + row >= inter_dim) break;

            const char* gp = w_gate
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * Q4_BLOCK_BYTES;
            unsigned short g_scale_bits = (unsigned short)(unsigned char)gp[0]
                                        | ((unsigned short)(unsigned char)gp[1] << 8);
            float g_scale = f16_to_f32(g_scale_bits);
            const unsigned char* gq = (const unsigned char*)(gp + 2);

            const char* up_ = w_up
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * Q4_BLOCK_BYTES;
            unsigned short u_scale_bits = (unsigned short)(unsigned char)up_[0]
                                        | ((unsigned short)(unsigned char)up_[1] << 8);
            float u_scale = f16_to_f32(u_scale_bits);
            const unsigned char* uq = (const unsigned char*)(up_ + 2);

            float g_block_sum = 0.0f;
            float u_block_sum = 0.0f;

            #pragma unroll
            for (int b = 0; b < 16; b++) {
                unsigned char gb = gq[b];
                unsigned char ub = uq[b];

                float gq_lo = (float)(gb & 0x0F) - 8.0f;
                float gq_hi = (float)(gb >> 4)    - 8.0f;
                float uq_lo = (float)(ub & 0x0F) - 8.0f;
                float uq_hi = (float)(ub >> 4)    - 8.0f;

                g_block_sum += gq_lo * xv[b]     + gq_hi * xv[b + 16];
                u_block_sum += uq_lo * xv[b]     + uq_hi * xv[b + 16];
            }

            gate_sum[row] += g_scale * g_block_sum;
            up_sum[row]   += u_scale * u_block_sum;
        }
    }

    // Cross-warp reduction + SwiGLU.
    const unsigned int num_warps = BLOCK_DIM / WARP_SIZE;

    #pragma unroll
    for (int r = 0; r < NR; r++) {
        gate_sum[r] = warp_reduce_sum(gate_sum[r]);
    }
    __syncthreads();

    float* reduce_smem = (float*)nx_f16;
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            reduce_smem[r * num_warps + warp_id] = gate_sum[r];
        }
    }
    __syncthreads();

    float final_gate[NR];
    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            float val = (lane < num_warps) ? reduce_smem[r * num_warps + lane] : 0.0f;
            val = warp_reduce_sum(val);
            final_gate[r] = val;
        }
    }
    __syncthreads();

    #pragma unroll
    for (int r = 0; r < NR; r++) {
        up_sum[r] = warp_reduce_sum(up_sum[r]);
    }

    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            reduce_smem[r * num_warps + warp_id] = up_sum[r];
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            if (r0 + r < inter_dim) {
                float val = (lane < num_warps) ? reduce_smem[r * num_warps + lane] : 0.0f;
                val = warp_reduce_sum(val);
                if (lane == 0) {
                    float g = final_gate[r];
                    float silu_g = g / (1.0f + expf(-g));
                    output[r0 + r] = silu_g * val;
                }
            }
        }
    }
}


// ============================================================================
// Q8_0 Aligned fused gate+up+SwiGLU GEMV with inline RMSNorm.
//
// Same as fused_glu_gemv_q8_0 but for 36-byte aligned Q8_0 blocks:
//   bytes [0..1]:  f16 scale
//   bytes [2..3]:  padding (zeroed)
//   bytes [4..35]: 32 x int8 quantized values  <-- 4-byte aligned
//
// The aligned layout enables native int* loads for the quant data.
// Standard models (Qwen2.5, Llama) have Q8_0 weights repacked to Q8Aligned
// during preload_weights() for better matvec throughput.
//
// Shared memory: hidden_dim * 4 bytes (normed F32 x-vector).
// Covers up to hidden_dim = 12288 (48KB / 4 = 12288 floats).
//
// Grid:  (ceil(inter_dim / NR), 1, 1)
// Block: (BLOCK_DIM, 1, 1)
// ============================================================================
extern "C" __global__ void fused_glu_gemv_q8_aligned(
    const char*  __restrict__ w_gate,       // [inter_dim * num_blocks * 36] Q8Aligned
    const char*  __restrict__ w_up,         // [inter_dim * num_blocks * 36] Q8Aligned
    const float* __restrict__ x,            // [hidden_dim] input activation
    const float* __restrict__ norm_weight,  // [hidden_dim] RMSNorm gamma
    const float* __restrict__ rms_scale,    // [1] precomputed 1/sqrt(mean(x^2)+eps)
    float*       __restrict__ output,       // [inter_dim] silu(gate) * up
    unsigned int inter_dim,
    unsigned int hidden_dim)
{
    extern __shared__ float nx_smem[];  // [hidden_dim] normed x-vector

    const unsigned int r0 = blockIdx.x * NR;
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane    = threadIdx.x % WARP_SIZE;
    const unsigned int num_blocks = hidden_dim / Q8_BLOCK_SIZE;
    const unsigned long long row_bytes = (unsigned long long)num_blocks * Q8_ALIGNED_BYTES;

    const float scale = rms_scale[0];

    // Cooperatively compute normed x-vector and store in shmem.
    for (unsigned int i = threadIdx.x; i < hidden_dim; i += BLOCK_DIM) {
        nx_smem[i] = x[i] * scale * norm_weight[i];
    }
    __syncthreads();

    float gate_sum[NR];
    float up_sum[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) {
        gate_sum[r] = 0.0f;
        up_sum[r]   = 0.0f;
    }

    // Main loop: each thread processes one Q8_0 aligned block (32 elements) per iteration.
    for (unsigned int ib = threadIdx.x; ib < num_blocks; ib += BLOCK_DIM) {
        const unsigned int x_base = ib * Q8_BLOCK_SIZE;

        // Load normed x-values from shmem into registers.
        float xv[32];
        const float4* x4 = (const float4*)(nx_smem + x_base);
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            float4 v = x4[k];
            xv[k * 4 + 0] = v.x;
            xv[k * 4 + 1] = v.y;
            xv[k * 4 + 2] = v.z;
            xv[k * 4 + 3] = v.w;
        }

        #pragma unroll
        for (int row = 0; row < NR; row++) {
            if (r0 + row >= inter_dim) break;

            // Gate weight block (Q8Aligned: scale at +0, quants at +4).
            const char* gp = w_gate
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * Q8_ALIGNED_BYTES;
            unsigned short g_scale_bits = (unsigned short)(unsigned char)gp[0]
                                        | ((unsigned short)(unsigned char)gp[1] << 8);
            float g_scale = f16_to_f32(g_scale_bits);
            const signed char* gq = (const signed char*)(gp + 4);  // +4 for aligned

            // Up weight block (Q8Aligned: scale at +0, quants at +4).
            const char* up_ = w_up
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * Q8_ALIGNED_BYTES;
            unsigned short u_scale_bits = (unsigned short)(unsigned char)up_[0]
                                        | ((unsigned short)(unsigned char)up_[1] << 8);
            float u_scale = f16_to_f32(u_scale_bits);
            const signed char* uq = (const signed char*)(up_ + 4);  // +4 for aligned

            float g_block_sum = 0.0f;
            float u_block_sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < 32; j++) {
                g_block_sum += (float)gq[j] * xv[j];
                u_block_sum += (float)uq[j] * xv[j];
            }

            gate_sum[row] += g_scale * g_block_sum;
            up_sum[row]   += u_scale * u_block_sum;
        }
    }

    // Cross-warp reduction via shared memory.
    const unsigned int num_warps = BLOCK_DIM / WARP_SIZE;

    #pragma unroll
    for (int r = 0; r < NR; r++) {
        gate_sum[r] = warp_reduce_sum(gate_sum[r]);
    }

    __syncthreads();

    float* reduce_smem = nx_smem;
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            reduce_smem[r * num_warps + warp_id] = gate_sum[r];
        }
    }
    __syncthreads();

    float final_gate[NR];
    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            float val = (lane < num_warps) ? reduce_smem[r * num_warps + lane] : 0.0f;
            val = warp_reduce_sum(val);
            final_gate[r] = val;
        }
    }
    __syncthreads();

    #pragma unroll
    for (int r = 0; r < NR; r++) {
        up_sum[r] = warp_reduce_sum(up_sum[r]);
    }

    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            reduce_smem[r * num_warps + warp_id] = up_sum[r];
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            if (r0 + r < inter_dim) {
                float val = (lane < num_warps) ? reduce_smem[r * num_warps + lane] : 0.0f;
                val = warp_reduce_sum(val);
                if (lane == 0) {
                    float g = final_gate[r];
                    float silu_g = g / (1.0f + expf(-g));
                    output[r0 + r] = silu_g * val;
                }
            }
        }
    }
}


// ============================================================================
// Q8_0 Aligned fused gate+up+SwiGLU GEMV with F16 x-vector in shmem (large dims).
//
// For hidden_dim > 12288 where F32 shmem is insufficient (> 48KB).
// Stores normed x as F16 in shmem (hidden_dim * 2 bytes).
// Covers up to hidden_dim = 24576 (49152 / 2).
//
// Grid:  (ceil(inter_dim / NR), 1, 1)
// Block: (BLOCK_DIM, 1, 1)
// Shared memory: hidden_dim * 2 bytes (F16 normed x-vector).
// ============================================================================
extern "C" __global__ void fused_glu_gemv_q8_aligned_hg(
    const char*  __restrict__ w_gate,
    const char*  __restrict__ w_up,
    const float* __restrict__ x,
    const float* __restrict__ norm_weight,
    const float* __restrict__ rms_scale,
    float*       __restrict__ output,
    unsigned int inter_dim,
    unsigned int hidden_dim)
{
    extern __shared__ unsigned short nx_f16[];  // [hidden_dim] normed x as F16

    const unsigned int r0 = blockIdx.x * NR;
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane    = threadIdx.x % WARP_SIZE;
    const unsigned int num_blocks = hidden_dim / Q8_BLOCK_SIZE;
    const unsigned long long row_bytes = (unsigned long long)num_blocks * Q8_ALIGNED_BYTES;

    const float scale = rms_scale[0];

    // Pre-compute normed x-vector, convert to F16, store in shmem.
    for (unsigned int i = threadIdx.x; i < hidden_dim; i += BLOCK_DIM) {
        nx_f16[i] = f32_to_f16(x[i] * scale * norm_weight[i]);
    }
    __syncthreads();

    float gate_sum[NR];
    float up_sum[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) {
        gate_sum[r] = 0.0f;
        up_sum[r]   = 0.0f;
    }

    for (unsigned int ib = threadIdx.x; ib < num_blocks; ib += BLOCK_DIM) {
        const unsigned int x_base = ib * Q8_BLOCK_SIZE;

        // Load normed x from shmem (F16) and convert to F32 in registers.
        float xv[32];
        const unsigned int* x_u32 = (const unsigned int*)(nx_f16 + x_base);
        #pragma unroll
        for (int k = 0; k < 16; k++) {
            unsigned int packed = x_u32[k];
            xv[k * 2 + 0] = f16_to_f32((unsigned short)(packed & 0xFFFF));
            xv[k * 2 + 1] = f16_to_f32((unsigned short)(packed >> 16));
        }

        #pragma unroll
        for (int row = 0; row < NR; row++) {
            if (r0 + row >= inter_dim) break;

            // Gate weight block (Q8Aligned: scale at +0, quants at +4).
            const char* gp = w_gate
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * Q8_ALIGNED_BYTES;
            unsigned short g_scale_bits = (unsigned short)(unsigned char)gp[0]
                                        | ((unsigned short)(unsigned char)gp[1] << 8);
            float g_scale = f16_to_f32(g_scale_bits);
            const signed char* gq = (const signed char*)(gp + 4);  // +4 for aligned

            // Up weight block (Q8Aligned: scale at +0, quants at +4).
            const char* up_ = w_up
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * Q8_ALIGNED_BYTES;
            unsigned short u_scale_bits = (unsigned short)(unsigned char)up_[0]
                                        | ((unsigned short)(unsigned char)up_[1] << 8);
            float u_scale = f16_to_f32(u_scale_bits);
            const signed char* uq = (const signed char*)(up_ + 4);  // +4 for aligned

            float g_block_sum = 0.0f;
            float u_block_sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < 32; j++) {
                g_block_sum += (float)gq[j] * xv[j];
                u_block_sum += (float)uq[j] * xv[j];
            }

            gate_sum[row] += g_scale * g_block_sum;
            up_sum[row]   += u_scale * u_block_sum;
        }
    }

    // Cross-warp reduction + SwiGLU. Reuse shmem as float*.
    const unsigned int num_warps = BLOCK_DIM / WARP_SIZE;

    #pragma unroll
    for (int r = 0; r < NR; r++) {
        gate_sum[r] = warp_reduce_sum(gate_sum[r]);
    }
    __syncthreads();

    float* reduce_smem = (float*)nx_f16;
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            reduce_smem[r * num_warps + warp_id] = gate_sum[r];
        }
    }
    __syncthreads();

    float final_gate[NR];
    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            float val = (lane < num_warps) ? reduce_smem[r * num_warps + lane] : 0.0f;
            val = warp_reduce_sum(val);
            final_gate[r] = val;
        }
    }
    __syncthreads();

    #pragma unroll
    for (int r = 0; r < NR; r++) {
        up_sum[r] = warp_reduce_sum(up_sum[r]);
    }

    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            reduce_smem[r * num_warps + warp_id] = up_sum[r];
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            if (r0 + r < inter_dim) {
                float val = (lane < num_warps) ? reduce_smem[r * num_warps + lane] : 0.0f;
                val = warp_reduce_sum(val);
                if (lane == 0) {
                    float g = final_gate[r];
                    float silu_g = g / (1.0f + expf(-g));
                    output[r0 + r] = silu_g * val;
                }
            }
        }
    }
}
