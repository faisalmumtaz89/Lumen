// ==========================================================================
// Q8Aligned x Q8_1 dp4a matvec, scale-halfword variant.
//
// Structural difference from `matvec_q8_aligned_q8_1`:
//   * Scale loads use a single 16-bit halfword cast
//       `*(const unsigned short*)w_block`
//     instead of the byte-OR-shift pattern
//       `(unsigned short)(unsigned char)w_block[0]
//        | ((unsigned short)(unsigned char)w_block[1] << 8)`.
//
// Rationale:
//   * Q8Aligned blocks are 36 bytes, so `w_block` is always 4-byte aligned.
//   * Q8_1 blocks are 36 bytes, so `x_block` is always 4-byte aligned.
//   * The byte-OR-shift pattern is defensive against unaligned `char*`
//     pointers. NVCC cannot statically prove the alignment from
//     `const char*`, so it likely emits two `LD.U8 + SHL + OR` per scale.
//     The halfword cast lets the compiler emit a single `LD.U16`.
//   * Lumen's `matvec_q4_0_dp4a.cu` already uses the halfword cast for the
//     scale read at offset 0; this variant simply mirrors that proven idiom
//     for the Q8Aligned + Q8_1 path. The unaligned `matvec_dp4a_q8_1`
//     kernel intentionally keeps the byte-OR-shift form because Q8_0 raw
//     blocks (34 bytes) are only 2-byte aligned; that path is unchanged.
//
// Estimated ALU saving (back-of-envelope, must be confirmed on A100):
//   * NR=2 rows -> 3 scale loads per block iter (1 input + 2 weight).
//   * Byte-OR-shift: ~6 instr/scale; halfword cast: 1 instr/scale.
//   * Savings: ~15 instr/iter on top of ~16 dp4a-related instr/iter.
//
// Safety:
//   * `w_block` = base + ib*36, where `base` is a CUDA-allocated buffer
//     (CUDA returns at least 256-byte alignment per `cudaMalloc`). 36 is a
//     multiple of 4, so `w_block` is 4-byte aligned -> halfword cast is
//     undefined-behavior-free.
//   * `x_block` = base + ib*36 is similarly 4-byte aligned.
//
// What this kernel does NOT change:
//   * NR=2 rows/block, 128 threads (4 warps).
//   * `__launch_bounds__(128, 2)` (matches production).
//   * dp4a inner loop, cross-warp reduction, and final write logic.
//   * Memory access pattern (1 Q8 block per thread per iteration, stride
//     by THREADS_PER_BLOCK).
//
// NVRTC-compatible: no system includes, extern "C" linkage.
// ==========================================================================

#define NR       2     // rows per thread block
#define NW       32    // warp size
#define THREADS_PER_BLOCK 128  // 4 warps
#define NWARPS   (THREADS_PER_BLOCK / NW)  // 4
#define Q8_BLOCK_SIZE     32   // elements per Q8 block
#define Q8_ALIGNED_BYTES  36   // 2B f16 scale + 2B pad + 32B int8 data
#define Q8_1_BYTES        36   // 2B f16 scale + 2B f16 sum + 32B int8 data

// Hardware f16->f32 conversion via PTX (single instruction on SM 53+).
__device__ __forceinline__ float f16_bits_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

// inline-PTX dp4a wrapper. See matvec_q8_split_q8_1.cu for the
// rationale (the `__dp4a` intrinsic NVRTC-fails in this build env).
__device__ __forceinline__ int dp4a_s32(int a, int b, int c) {
    int d;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
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
// Kernel 1: Q8Aligned weight x Q8_1 input -> F32 output (dp4a, NR=2).
// ==========================================================================
extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 2) void matvec_q8_aligned_q8_1_hw(
    const char* __restrict__ weight_q8_aligned,  // [out_dim * nb * 36] Q8Aligned bytes
    const char* __restrict__ input_q8_1,         // [nb * 36] Q8_1 pre-quantized input
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

        const char* x_block = input_q8_1 + (unsigned long long)ib * Q8_1_BYTES;

        // Halfword cast: x_block is 4-byte aligned (36-byte stride).
        unsigned short x_scale_bits = *(const unsigned short*)x_block;
        float x_scale = f16_bits_to_f32(x_scale_bits);

        const int* x_packed = (const int*)(x_block + 4);
        int xv[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) xv[k] = x_packed[k];

        #pragma unroll
        for (int row = 0; row < NR; row++) {
            if (r0 + row >= out_dim) break;

            const char* w_block = weight_q8_aligned
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * Q8_ALIGNED_BYTES;

            // Halfword cast: w_block is 4-byte aligned (36-byte stride from
            // CUDA-allocated buffer, which is at least 256-byte aligned).
            unsigned short w_scale_bits = *(const unsigned short*)w_block;
            float w_scale = f16_bits_to_f32(w_scale_bits);

            const int* w_packed = (const int*)(w_block + 4);

            // Dual-accumulator chain split: split the 8-deep RAW
            // chain on `acc` into two independent 4-deep chains. Bit-identical
            // (integer addition associative). +1 int register, no occupancy
            // impact. See matvec_q8_aligned_q8_1.cu primary kernel for the
            // full rationale (IDP.4A latency hiding on SM 80).
            int acc_lo = 0, acc_hi = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                acc_lo = dp4a_s32(w_packed[k],     xv[k],     acc_lo);
                acc_hi = dp4a_s32(w_packed[k + 4], xv[k + 4], acc_hi);
            }
            int acc = acc_lo + acc_hi;

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
// Kernel 2: Q8Aligned weight x Q8_1 input + residual -> F32 output.
// ==========================================================================
extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 2) void matvec_q8_aligned_q8_1_hw_residual(
    const char* __restrict__ weight_q8_aligned,
    const char* __restrict__ input_q8_1,
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

        const char* x_block = input_q8_1 + (unsigned long long)ib * Q8_1_BYTES;

        unsigned short x_scale_bits = *(const unsigned short*)x_block;
        float x_scale = f16_bits_to_f32(x_scale_bits);

        const int* x_packed = (const int*)(x_block + 4);
        int xv[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) xv[k] = x_packed[k];

        #pragma unroll
        for (int row = 0; row < NR; row++) {
            if (r0 + row >= out_dim) break;

            const char* w_block = weight_q8_aligned
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * Q8_ALIGNED_BYTES;

            unsigned short w_scale_bits = *(const unsigned short*)w_block;
            float w_scale = f16_bits_to_f32(w_scale_bits);

            const int* w_packed = (const int*)(w_block + 4);

            // Dual-accumulator chain split — see primary HW kernel
            // comment for rationale. Bit-identical to single-chain version.
            int acc_lo = 0, acc_hi = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                acc_lo = dp4a_s32(w_packed[k],     xv[k],     acc_lo);
                acc_hi = dp4a_s32(w_packed[k + 4], xv[k + 4], acc_hi);
            }
            int acc = acc_lo + acc_hi;

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
