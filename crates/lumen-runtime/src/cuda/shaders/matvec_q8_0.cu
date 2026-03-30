// Q8_0 matrix-vector multiply (GEMV) kernels for decode.
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
// Architecture: Multi-row deferred-reduction (NR=2, 4 warps, 128 threads).
//
// Each thread block processes NR=2 output rows. 4 warps (128 threads) cooperate
// to process all Q8_0 blocks for both rows. Each thread handles NQ=8 elements
// from one Q8_0 block per iteration, loads x once into registers, and reuses
// across both rows. Final reduction: warp shuffle + shared memory cross-warp.
//
// This mirrors the Metal backend's deferred_nr2 pattern, which was proven to be
// the best-performing Q8_0 matvec on Apple Silicon (more TGs = better occupancy).
//
// NVRTC-compatible: no system includes, extern "C" linkage.

#define NR       2     // rows per thread block
#define NW       32    // warp size
#define NSG      4     // warps per block
#define NQ       8     // elements per thread per iteration (sub-chunk of Q8_0 block)
#define THREADS_PER_BLOCK (NSG * NW)  // 128
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

// Q8_0 matrix-vector multiply: multi-row deferred-reduction.
//
// Grid:  (ceil(out_dim / NR), 1, 1)  -- one block per NR rows
// Block: (128, 1, 1)                 -- 4 warps x 32 threads
//
// Thread mapping (identical to Metal deferred_nr2):
//   warp_id = threadIdx.x / 32       -> 0..3 (which warp)
//   lane    = threadIdx.x % 32       -> 0..31 (lane within warp)
//   ix      = lane / 4               -> 0..7 (block index in stride)
//   il      = lane % 4               -> 0..3 (sub-chunk of 8 within block)
//
// 4 threads collectively process one 32-element Q8_0 block (4 x 8 = 32).
// Stride = NSG * NQ = 4 * 8 = 32 blocks per outer iteration.
//
// in_dim must be a multiple of Q8_0_BLOCK_SIZE (32).
extern "C" __global__ void matvec_q8_0(
    const char* __restrict__ weight_q8,  // [out_dim * num_blocks_per_row * 34] raw Q8_0 bytes
    const float* __restrict__ x,         // [in_dim]
    float* __restrict__ out,             // [out_dim]
    unsigned int out_dim,
    unsigned int in_dim)
{
    unsigned int r0 = blockIdx.x * NR;  // first output row for this block

    unsigned int warp_id = threadIdx.x / NW;
    unsigned int lane    = threadIdx.x % NW;
    unsigned int ix      = lane / (NW / NQ);   // lane / 4 -> 0..7
    unsigned int il      = lane % (NW / NQ);   // lane % 4 -> 0..3

    unsigned int nb = in_dim >> 5;  // number of Q8_0 blocks per row
    unsigned long long row_bytes = (unsigned long long)nb * Q8_0_BYTES;

    unsigned int ib0 = warp_id * NQ + ix;  // starting block for this thread

    // Per-row accumulators
    float sumf[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) sumf[r] = 0.0f;

    // Register-cached x-values (loaded once, reused for both rows)
    float yl[NQ];

    // Pointer into x for this thread's starting position
    const float* yb = x + (unsigned long long)ib0 * Q8_0_BLOCK_SIZE + il * NQ;

    // Main loop: 4 warps cooperatively process all blocks with stride NSG*NQ=32
    for (unsigned int ib = ib0; ib < nb; ib += NSG * NQ) {
        // Load NQ=8 x-values into registers
        #pragma unroll
        for (int i = 0; i < NQ; i++) {
            yl[i] = yb[i];
        }

        // Process both rows with the same cached x-values
        #pragma unroll
        for (int row = 0; row < NR; row++) {
            if (r0 + row >= out_dim) break;

            const char* bp = weight_q8
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * Q8_0_BYTES;

            // Read f16 scale
            unsigned short scale_bits = (unsigned short)(unsigned char)bp[0]
                                      | ((unsigned short)(unsigned char)bp[1] << 8);
            float scale = f16_bits_to_f32(scale_bits);

            // Dot product of this thread's NQ=8 sub-chunk
            const char* qs = bp + 2 + il * NQ;
            float sumq = 0.0f;
            #pragma unroll
            for (int i = 0; i < NQ; i++) {
                sumq += (float)(signed char)qs[i] * yl[i];
            }
            sumf[row] += sumq * scale;
        }

        yb += NSG * NQ * Q8_0_BLOCK_SIZE;
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
                float val = (lane < NSG) ? shmem[r * NW + lane] : 0.0f;
                val = warp_reduce_sum(val);
                if (lane == 0) {
                    out[r0 + r] = val;
                }
            }
        }
    }
}

// Q8_0 matrix-vector multiply with fused residual addition:
// out = dequant(weight_q8) * x + residual
//
// Same multi-row deferred-reduction pattern as matvec_q8_0,
// with fused residual addition at the final write.
//
// Grid:  (ceil(out_dim / NR), 1, 1)  -- one block per NR rows
// Block: (128, 1, 1)                 -- 4 warps x 32 threads
extern "C" __global__ void matvec_q8_0_residual(
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
    unsigned int ix      = lane / (NW / NQ);
    unsigned int il      = lane % (NW / NQ);

    unsigned int nb = in_dim >> 5;
    unsigned long long row_bytes = (unsigned long long)nb * Q8_0_BYTES;

    unsigned int ib0 = warp_id * NQ + ix;

    float sumf[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) sumf[r] = 0.0f;

    float yl[NQ];
    const float* yb = x + (unsigned long long)ib0 * Q8_0_BLOCK_SIZE + il * NQ;

    for (unsigned int ib = ib0; ib < nb; ib += NSG * NQ) {
        #pragma unroll
        for (int i = 0; i < NQ; i++) {
            yl[i] = yb[i];
        }

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
                sumq += (float)(signed char)qs[i] * yl[i];
            }
            sumf[row] += sumq * scale;
        }

        yb += NSG * NQ * Q8_0_BLOCK_SIZE;
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
