// Q8_0 matrix-vector multiply with shared-memory x-vector caching.
//
// Key insight: matvec is bandwidth-bound on A100. The optimal kernel reads
// the minimum bytes per element (1.0625 B/elem for Q8_0) and amortizes the
// x-vector loads across multiple output rows via shared memory.
//
// This kernel is designed to match llama.cpp's native Q8_0 matvec throughput
// by eliminating the two bottlenecks of the existing kernels:
//   1. dp4a kernel: x-quantization overhead (~50 ALU ops per block)
//   2. HGEMV path: 2 bytes/elem (F16) instead of 1.0625 (Q8_0)
//
// Architecture: NR=2 rows per block, 256 threads (8 warps).
//   - Each thread processes one full Q8_0 block (32 elements) per iteration
//   - x-vector loaded into shared memory ONCE, reused for all NR rows
//   - Simple scalar dequant: scale * (float)(int8_t)q * x_smem[j]
//   - No x-quantization, no dp4a — pure FMA on dequantized values
//   - Warp shuffle + shared memory for cross-warp reduction
//
// Q8_0 block layout (GGML): 34 bytes per block of 32 elements.
//   bytes [0..1]: f16 scale (IEEE 754 half-precision, little-endian)
//   bytes [2..33]: 32 x int8 quantized values
//   Dequant: float_val = scale * (float)(int8_t)quant[j]
//
// Memory traffic per output element:
//   Weight: 34 bytes / 32 elems = 1.0625 B/elem
//   x-vector: amortized across NR rows via shmem
//   Total per row: in_dim * 1.0625 bytes + reduction overhead
//
// Grid:  (ceil(out_dim / NR), 1, 1)
// Block: (BLOCK_DIM, 1, 1)
//
// NVRTC-compatible: no system includes, extern "C" linkage.

#define NR              2       // rows per thread block
#define WARP_SIZE       32
#define BLOCK_DIM       256     // 8 warps — fills A100 SM better than 128
#define Q8_BLOCK_SIZE   32      // elements per Q8_0 block
#define Q8_BLOCK_BYTES  34      // bytes per Q8_0 block

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

// Q8_0 matrix-vector multiply with shared-memory x-vector caching.
//
// Grid:  (ceil(out_dim / NR), 1, 1)
// Block: (BLOCK_DIM, 1, 1)
//
// Shared memory: in_dim * sizeof(float) bytes for x-vector cache.
// The caller must set shared_mem_bytes = in_dim * 4.
extern "C" __global__ void matvec_q8_0_smem(
    const char* __restrict__ weight_q8,  // [out_dim * num_blocks * 34] raw Q8_0 bytes
    const float* __restrict__ x,         // [in_dim]
    float* __restrict__ out,             // [out_dim]
    unsigned int out_dim,
    unsigned int in_dim)
{
    extern __shared__ float x_smem[];  // [in_dim] — x-vector cached in shmem

    const unsigned int r0 = blockIdx.x * NR;
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane    = threadIdx.x % WARP_SIZE;
    const unsigned int num_blocks = in_dim / Q8_BLOCK_SIZE;
    const unsigned long long row_bytes = (unsigned long long)num_blocks * Q8_BLOCK_BYTES;

    // Cooperatively load x-vector into shared memory.
    // 256 threads can load 256 floats per iteration.
    for (unsigned int i = threadIdx.x; i < in_dim; i += BLOCK_DIM) {
        x_smem[i] = x[i];
    }
    __syncthreads();

    // Per-row accumulators
    float sumf[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) sumf[r] = 0.0f;

    // Main loop: each thread handles one full Q8_0 block (32 elements) per iteration,
    // striding by BLOCK_DIM blocks. Each thread accumulates across all its assigned blocks
    // before any cross-thread reduction.
    for (unsigned int ib = threadIdx.x; ib < num_blocks; ib += BLOCK_DIM) {
        const unsigned int x_base = ib * Q8_BLOCK_SIZE;

        // Load x-values from shared memory into registers for this block.
        // Use float4 loads for bandwidth: 8 x float4 = 32 floats.
        // x_smem is float-aligned (from cooperative load above).
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

        // Process all NR output rows with the same cached x-values.
        #pragma unroll
        for (int row = 0; row < NR; row++) {
            if (r0 + row >= out_dim) break;

            const char* bp = weight_q8
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * Q8_BLOCK_BYTES;

            // Read f16 scale (2 bytes, little-endian).
            unsigned short scale_bits = (unsigned short)(unsigned char)bp[0]
                                      | ((unsigned short)(unsigned char)bp[1] << 8);
            float scale = f16_bits_to_f32(scale_bits);

            // Dot product: scale * sum(int8_quant[j] * x[j]) for j in [0..32)
            // Read int8 quant data at bp+2. NOT 4-byte aligned (34-byte blocks),
            // so we use byte loads (safe on A100, unlike int* which causes XID 13).
            const signed char* qs = (const signed char*)(bp + 2);

            float block_sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < 32; j++) {
                block_sum += (float)qs[j] * xv[j];
            }

            sumf[row] += scale * block_sum;
        }
    }

    // Cross-warp reduction via shared memory.
    // Reuse x_smem for reduction (x is no longer needed after the main loop).
    // Layout: NR rows x (BLOCK_DIM / WARP_SIZE) slots.
    const unsigned int num_warps = BLOCK_DIM / WARP_SIZE;

    // Intra-warp reduction first (5 shuffles).
    #pragma unroll
    for (int r = 0; r < NR; r++) {
        sumf[r] = warp_reduce_sum(sumf[r]);
    }

    __syncthreads();  // Ensure x_smem loads are complete before reuse

    // Lane 0 of each warp writes its partial sum to shared memory.
    // We reuse x_smem as reduction scratch (we need NR * num_warps floats).
    float* reduce_smem = x_smem;  // Safe: we only need NR * num_warps << in_dim

    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            reduce_smem[r * num_warps + warp_id] = sumf[r];
        }
    }

    __syncthreads();

    // Warp 0 does the final reduction across all warps.
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

// Q8_0 shared-memory matvec with fused residual addition:
// out[i] = dot(dequant(weight_q8[i]), x) + residual[i]
//
// Grid:  (ceil(out_dim / NR), 1, 1)
// Block: (BLOCK_DIM, 1, 1)
// Shared memory: in_dim * sizeof(float) bytes.
extern "C" __global__ void matvec_q8_0_smem_residual(
    const char* __restrict__ weight_q8,  // [out_dim * num_blocks * 34]
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
    const unsigned int num_blocks = in_dim / Q8_BLOCK_SIZE;
    const unsigned long long row_bytes = (unsigned long long)num_blocks * Q8_BLOCK_BYTES;

    // Cooperatively load x into shared memory.
    for (unsigned int i = threadIdx.x; i < in_dim; i += BLOCK_DIM) {
        x_smem[i] = x[i];
    }
    __syncthreads();

    float sumf[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) sumf[r] = 0.0f;

    for (unsigned int ib = threadIdx.x; ib < num_blocks; ib += BLOCK_DIM) {
        const unsigned int x_base = ib * Q8_BLOCK_SIZE;

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

            const char* bp = weight_q8
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * Q8_BLOCK_BYTES;

            unsigned short scale_bits = (unsigned short)(unsigned char)bp[0]
                                      | ((unsigned short)(unsigned char)bp[1] << 8);
            float scale = f16_bits_to_f32(scale_bits);

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
