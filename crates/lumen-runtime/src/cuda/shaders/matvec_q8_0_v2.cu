// Q8_0 matrix-vector multiply (GEMV) v2: shared-memory x vector caching.
//
// Improvement over v1: the x vector tile is cooperatively loaded into shared
// memory once per outer iteration, so all 4 warps (processing NR=2 rows)
// read from fast shared memory (~30 TB/s) instead of relying on L2 cache
// (~2 TB/s on A10G). For a 4096-element x vector, v1 has each of the 2048
// thread blocks loading x from global/L2; v2 loads into shared memory once
// per block, guaranteeing zero L2 misses on x.
//
// Architecture: NR=2 rows per block, 128 threads (4 warps), same as v1.
// Thread mapping identical to v1 (deferred_nr2 pattern).
//
// Q8_0 block layout (GGML): 34 bytes per block of 32 elements.
//   bytes [0..1]: f16 scale (IEEE 754 half-precision, little-endian)
//   bytes [2..33]: 32 x int8 quantized values
//   Dequant: float_val = scale * (float)(int8_t)quant[j]
//
// Operation: out[i] = sum_j(dequant(weight_q8[i, j]) * x[j])
//
// NVRTC-compatible: no system includes, extern "C" linkage.

#define NR       2     // rows per thread block
#define NW       32    // warp size
#define NSG      4     // warps per block
#define NQ       8     // elements per thread per iteration
#define THREADS_PER_BLOCK (NSG * NW)  // 128
#define Q8_0_BLOCK_SIZE   32
#define Q8_0_BYTES        34

// Shared memory tile: x values for NSG*NQ=32 Q8_0 blocks = 1024 floats = 4 KB.
#define X_TILE_SIZE (NSG * NQ * Q8_0_BLOCK_SIZE)  // 1024

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

// Q8_0 matrix-vector multiply v2: shared-memory x caching.
//
// Grid:  (ceil(out_dim / NR), 1, 1)  -- one block per NR rows
// Block: (128, 1, 1)                 -- 4 warps x 32 threads
//
// in_dim must be a multiple of Q8_0_BLOCK_SIZE (32).
extern "C" __global__ void matvec_q8_0_v2(
    const char* __restrict__ weight_q8,
    const float* __restrict__ x,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim)
{
    unsigned int r0 = blockIdx.x * NR;

    unsigned int warp_id = threadIdx.x / NW;
    unsigned int lane    = threadIdx.x % NW;
    unsigned int ix      = lane / (NW / NQ);
    unsigned int il      = lane % (NW / NQ);

    unsigned int nb = in_dim >> 5;
    unsigned long long row_bytes = (unsigned long long)nb * Q8_0_BYTES;

    unsigned int ib0 = warp_id * NQ + ix;

    float sumf[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) sumf[r] = 0.0f;

    // Shared memory: x vector tile (4 KB) + reduction scratch (256 B).
    __shared__ float x_tile[X_TILE_SIZE];

    for (unsigned int ib_base = 0; ib_base < nb; ib_base += NSG * NQ) {
        // Cooperative x tile load: 128 threads load 1024 floats (8 each).
        {
            unsigned int x_global_base = ib_base * Q8_0_BLOCK_SIZE;
            unsigned int x_tile_elems = X_TILE_SIZE;
            unsigned int x_remaining = in_dim - x_global_base;
            if (x_remaining < x_tile_elems) x_tile_elems = x_remaining;

            #pragma unroll
            for (unsigned int t = threadIdx.x; t < X_TILE_SIZE; t += THREADS_PER_BLOCK) {
                x_tile[t] = (t < x_tile_elems) ? x[x_global_base + t] : 0.0f;
            }
        }
        __syncthreads();

        unsigned int ib = ib_base + ib0;
        if (ib < nb) {
            // This thread's x values in shared memory
            unsigned int tile_base = ib0 * Q8_0_BLOCK_SIZE + il * NQ;

            #pragma unroll
            for (int row = 0; row < NR; row++) {
                if (r0 + row >= out_dim) break;

                const char* bp = weight_q8
                    + (unsigned long long)(r0 + row) * row_bytes
                    + (unsigned long long)ib * Q8_0_BYTES;

                unsigned short scale_bits = (unsigned short)(unsigned char)bp[0]
                                          | ((unsigned short)(unsigned char)bp[1] << 8);
                float scale = f16_bits_to_f32(scale_bits);

                const char* qs = bp + 2 + il * NQ;
                float sumq = 0.0f;
                #pragma unroll
                for (int i = 0; i < NQ; i++) {
                    sumq += (float)(signed char)qs[i] * x_tile[tile_base + i];
                }
                sumf[row] += sumq * scale;
            }
        }

        __syncthreads();
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
                float val = (lane < NSG) ? shmem[r * NW + lane] : 0.0f;
                val = warp_reduce_sum(val);
                if (lane == 0) {
                    out[r0 + r] = val;
                }
            }
        }
    }
}

// Q8_0 matrix-vector multiply v2 with fused residual addition:
// out = dequant(weight_q8) * x + residual
//
// Same shared-memory x caching pattern as matvec_q8_0_v2.
//
// Grid:  (ceil(out_dim / NR), 1, 1)
// Block: (128, 1, 1)
extern "C" __global__ void matvec_q8_0_v2_residual(
    const char* __restrict__ weight_q8,
    const float* __restrict__ x,
    const float* __restrict__ residual,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim)
{
    unsigned int r0 = blockIdx.x * NR;

    unsigned int warp_id = threadIdx.x / NW;
    unsigned int lane    = threadIdx.x % NW;
    unsigned int ix      = lane / (NW / NQ);
    unsigned int il      = lane % (NW / NQ);

    unsigned int nb = in_dim >> 5;
    unsigned long long row_bytes = (unsigned long long)nb * Q8_0_BYTES;

    unsigned int ib0 = warp_id * NQ + ix;

    float sumf[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) sumf[r] = 0.0f;

    __shared__ float x_tile[X_TILE_SIZE];

    for (unsigned int ib_base = 0; ib_base < nb; ib_base += NSG * NQ) {
        {
            unsigned int x_global_base = ib_base * Q8_0_BLOCK_SIZE;
            unsigned int x_tile_elems = X_TILE_SIZE;
            unsigned int x_remaining = in_dim - x_global_base;
            if (x_remaining < x_tile_elems) x_tile_elems = x_remaining;

            #pragma unroll
            for (unsigned int t = threadIdx.x; t < X_TILE_SIZE; t += THREADS_PER_BLOCK) {
                x_tile[t] = (t < x_tile_elems) ? x[x_global_base + t] : 0.0f;
            }
        }
        __syncthreads();

        unsigned int ib = ib_base + ib0;
        if (ib < nb) {
            unsigned int tile_base = ib0 * Q8_0_BLOCK_SIZE + il * NQ;

            #pragma unroll
            for (int row = 0; row < NR; row++) {
                if (r0 + row >= out_dim) break;

                const char* bp = weight_q8
                    + (unsigned long long)(r0 + row) * row_bytes
                    + (unsigned long long)ib * Q8_0_BYTES;

                unsigned short scale_bits = (unsigned short)(unsigned char)bp[0]
                                          | ((unsigned short)(unsigned char)bp[1] << 8);
                float scale = f16_bits_to_f32(scale_bits);

                const char* qs = bp + 2 + il * NQ;
                float sumq = 0.0f;
                #pragma unroll
                for (int i = 0; i < NQ; i++) {
                    sumq += (float)(signed char)qs[i] * x_tile[tile_base + i];
                }
                sumf[row] += sumq * scale;
            }
        }

        __syncthreads();
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
                float val = (lane < NSG) ? shmem[r * NW + lane] : 0.0f;
                val = warp_reduce_sum(val);
                if (lane == 0) {
                    out[r0 + r] = val + residual[r0 + r];
                }
            }
        }
    }
}
