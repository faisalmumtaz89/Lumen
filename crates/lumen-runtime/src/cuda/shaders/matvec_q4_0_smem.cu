// Q4_0 matrix-vector multiply with shared-memory x-vector caching.
//
// Same architectural approach as matvec_q8_0_smem.cu but for 4-bit quantization.
// Reads 0.5625 bytes/element (18 bytes / 32 elements) — ~3.5x less than F16.
//
// Q4_0 block layout (GGML): 18 bytes per block of 32 elements.
//   bytes [0..1]: f16 scale (IEEE 754 half-precision, little-endian)
//   bytes [2..17]: 16 bytes = 32 x 4-bit unsigned values packed as nibble pairs
//     De-interleaved layout: elements 0-15 from lo nibbles of bytes 0-15,
//     elements 16-31 from hi nibbles of bytes 0-15
//   Dequantize: float_value = scale * ((float)(nibble) - 8.0f)
//
// Architecture: NR=2 rows per block, 256 threads (8 warps).
//   - x-vector loaded into shared memory ONCE, reused for all NR rows
//   - Each thread processes one full Q4_0 block (32 elements) per iteration
//   - Nibble unpacking: 2 elements per byte, 16 bytes per block
//   - No x-quantization overhead
//
// Memory traffic per output element:
//   Weight: 18 bytes / 32 elems = 0.5625 B/elem (vs 2 B/elem for F16 HGEMV)
//   This should yield ~3.5x bandwidth improvement over the HGEMV path.
//
// Grid:  (ceil(out_dim / NR), 1, 1)
// Block: (BLOCK_DIM, 1, 1)
// Shared memory: in_dim * sizeof(float) bytes for x-vector cache.
//
// NVRTC-compatible: no system includes, extern "C" linkage.

#define NR              2       // rows per thread block
#define WARP_SIZE       32
#define BLOCK_DIM       256     // 8 warps
#define Q4_BLOCK_ELEMS  32      // elements per Q4_0 block
#define Q4_BLOCK_BYTES  18      // bytes per Q4_0 block

// Hardware f16->f32 conversion via PTX (single instruction on SM 53+).
__device__ __forceinline__ float f16_bits_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
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

// Q4_0 matrix-vector multiply with shared-memory x-vector caching.
//
// Grid:  (ceil(out_dim / NR), 1, 1)
// Block: (BLOCK_DIM, 1, 1)
// Shared memory: in_dim * sizeof(float) bytes.
extern "C" __global__ void matvec_q4_0_smem(
    const char* __restrict__ weight_q4,  // [out_dim * num_blocks * 18] raw Q4_0 bytes
    const float* __restrict__ x,         // [in_dim]
    float* __restrict__ out,             // [out_dim]
    unsigned int out_dim,
    unsigned int in_dim)
{
    extern __shared__ float x_smem[];  // [in_dim]

    const unsigned int r0 = blockIdx.x * NR;
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane    = threadIdx.x % WARP_SIZE;
    const unsigned int num_blocks = in_dim / Q4_BLOCK_ELEMS;
    const unsigned long long row_bytes = (unsigned long long)num_blocks * Q4_BLOCK_BYTES;

    // Cooperatively load x-vector into shared memory.
    for (unsigned int i = threadIdx.x; i < in_dim; i += BLOCK_DIM) {
        x_smem[i] = x[i];
    }
    __syncthreads();

    // Per-row accumulators
    float sumf[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) sumf[r] = 0.0f;

    // Main loop: each thread handles one full Q4_0 block per iteration.
    for (unsigned int ib = threadIdx.x; ib < num_blocks; ib += BLOCK_DIM) {
        const unsigned int x_base = ib * Q4_BLOCK_ELEMS;

        // Load x-values from shared memory into registers.
        // Use float4 loads: 8 x float4 = 32 floats.
        float xv[32];
        const float4* x4 = (const float4*)(x_smem + x_base);
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            float4 v = x4[k];
            xv[k * 4 + 0] = v.x;
            xv[k * 4 + 1] = v.y;
            xv[k * 4 + 2] = v.z;
            xv[k * 4 + 3] = v.w;
        }

        // Process all NR output rows.
        #pragma unroll
        for (int row = 0; row < NR; row++) {
            if (r0 + row >= out_dim) break;

            const char* bp = weight_q4
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * Q4_BLOCK_BYTES;

            // Read f16 scale.
            unsigned short scale_bits = (unsigned short)(unsigned char)bp[0]
                                      | ((unsigned short)(unsigned char)bp[1] << 8);
            float scale = f16_bits_to_f32(scale_bits);

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

    // Cross-warp reduction.
    const unsigned int num_warps = BLOCK_DIM / WARP_SIZE;

    #pragma unroll
    for (int r = 0; r < NR; r++) {
        sumf[r] = warp_reduce_sum(sumf[r]);
    }

    __syncthreads();

    float* reduce_smem = x_smem;
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

// Q4_0 shared-memory matvec with fused residual addition:
// out[i] = dot(dequant(weight_q4[i]), x) + residual[i]
//
// Grid:  (ceil(out_dim / NR), 1, 1)
// Block: (BLOCK_DIM, 1, 1)
// Shared memory: in_dim * sizeof(float) bytes.
extern "C" __global__ void matvec_q4_0_smem_residual(
    const char* __restrict__ weight_q4,  // [out_dim * num_blocks * 18]
    const float* __restrict__ x,         // [in_dim]
    const float* __restrict__ residual,  // [out_dim]
    float* __restrict__ out,             // [out_dim]
    unsigned int out_dim,
    unsigned int in_dim)
{
    extern __shared__ float x_smem[];

    const unsigned int r0 = blockIdx.x * NR;
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane    = threadIdx.x % WARP_SIZE;
    const unsigned int num_blocks = in_dim / Q4_BLOCK_ELEMS;
    const unsigned long long row_bytes = (unsigned long long)num_blocks * Q4_BLOCK_BYTES;

    for (unsigned int i = threadIdx.x; i < in_dim; i += BLOCK_DIM) {
        x_smem[i] = x[i];
    }
    __syncthreads();

    float sumf[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) sumf[r] = 0.0f;

    for (unsigned int ib = threadIdx.x; ib < num_blocks; ib += BLOCK_DIM) {
        const unsigned int x_base = ib * Q4_BLOCK_ELEMS;

        float xv[32];
        const float4* x4 = (const float4*)(x_smem + x_base);
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
            if (r0 + row >= out_dim) break;

            const char* bp = weight_q4
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * Q4_BLOCK_BYTES;

            unsigned short scale_bits = (unsigned short)(unsigned char)bp[0]
                                      | ((unsigned short)(unsigned char)bp[1] << 8);
            float scale = f16_bits_to_f32(scale_bits);

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

    float* reduce_smem = x_smem;
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
