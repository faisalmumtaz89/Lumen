// ==========================================================================
// Fused Down Projection Kernels for Q4Aligned: inline F32->Q8_1 quantization
// + __byte_perm nibble unpack + dp4a matvec against Q4Aligned weights.
//
// Eliminates the separate quantize_f32_to_q8_1 dispatch by quantizing the
// input vector to Q8_1 on-the-fly within each thread's block iteration.
//
// Four kernels:
//   1. matvec_q4_aligned_f32:          W * quantize(x_f32) -> out
//   2. matvec_q4_aligned_f32_residual: W * quantize(x_f32) + residual -> out
//   3. matvec_q4_aligned_f32_swiglu:   W * quantize(silu(gate)*up) -> out
//   4. matvec_q4_aligned_f32_swiglu_residual: same + residual
//
// Kernels 1-2: Replace quantize_f32_to_q8_1 + matvec_q4_aligned_q8_1 (2->1).
//   Used after fused_glu_gemv when SwiGLU is already computed.
//
// Kernels 3-4: Replace swiglu_inplace + quantize_f32_to_q8_1 +
//   matvec_q4_aligned_q8_1 (3->1).
//   Used when gate and up are separate F32 buffers.
//
// Inline quantization per thread (no warp-level reduction needed):
//   Each thread processes one block of 32 F32 values per iteration.
//   - Load 32 floats, find per-thread absmax, compute scale
//   - Quantize to int8, pack into 8 int32 words for dp4a
//   - Unpack Q4Aligned nibbles via __byte_perm (7 ops per 4 nibble bytes)
//   - dp4a dot product + zero-point correction
//
// Q4Aligned block layout (20 bytes per 32 elements):
//   bytes [0..1]:   f16 scale (d)
//   bytes [2..3]:   padding (alignment)
//   bytes [4..19]:  16 bytes of packed nibble pairs
//     byte[i] (i=0..15): lo_nibble = element[2*i], hi_nibble = element[2*i+1]
//   Dequantized value: scale * (nibble - 8)
//
// dp4a dot product for Q4Aligned x inline-Q8_1:
//   For each block:
//     1. Quantize 32 F32 values to int8, compute scale and weighted sum
//     2. Load nibble data as 4 aligned int32 words via int* loads
//     3. Unpack via __byte_perm into dp4a-compatible int32 words
//     4. dp4a: 8 calls = 32 multiply-accumulates
//     5. result += w_scale * x_scale * dp4a_sum - 8 * w_scale * x_sum
//
// Architecture: NR=4 rows per block, 256 threads (8 warps).
// __launch_bounds__(256, 1) for occupancy parity with matvec_q4_aligned_q8_1.
// Requires compute capability >= 6.1 for __dp4a() (Pascal+).
// in_dim must be a multiple of 32 (Q4_0 block size).
//
// NVRTC-compatible: no system includes, extern "C" linkage.
// ==========================================================================

#define NR       4     // rows per thread block
#define NW       32    // warp size
#define THREADS_PER_BLOCK 256  // 8 warps
#define NWARPS   (THREADS_PER_BLOCK / NW)  // 8
#define Q4_BLOCK_SIZE     32   // elements per Q4_0 block
#define Q4_ALIGNED_BYTES  20   // 2B f16 scale + 2B pad + 16B nibble data

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

// Unpack 4 nibble bytes (packed in an int32) into 2 dp4a-compatible int32 words.
// Uses __byte_perm (PRMT instruction) for register-level byte rearrangement.
// 7 ops vs ~43 ops in the scalar version.
__device__ __forceinline__ void unpack_nibbles_4bytes(unsigned int packed, int &out0, int &out1) {
    unsigned int lo = packed & 0x0F0F0F0Fu;
    unsigned int hi = (packed >> 4) & 0x0F0F0F0Fu;
    unsigned int interleaved0 = __byte_perm(lo, hi, 0x5140);
    unsigned int interleaved1 = __byte_perm(lo, hi, 0x7362);
    out0 = (int)(interleaved0 - 0x08080808u);
    out1 = (int)(interleaved1 - 0x08080808u);
}

// Inline F32->Q8_1 quantization + dp4a against Q4Aligned weight rows.
// Shared inner loop used by all 4 kernel variants.
//
// Parameters:
//   xf[32]: input F32 values (already loaded, possibly with SwiGLU applied)
//   sumf[NR]: per-row accumulators (accumulated across blocks)
//   weight_q4a: Q4Aligned weight base pointer
//   row_bytes: bytes per weight row
//   r0: first output row for this block
//   ib: current block index
//   out_dim: total output rows
__device__ __forceinline__ void fused_q4a_dp4a_block(
    const float* xf,
    float sumf[NR],
    const char* __restrict__ weight_q4a,
    unsigned long long row_bytes,
    unsigned int r0,
    unsigned int ib,
    unsigned int out_dim)
{
    // --- Inline F32 -> Q8_1 quantization ---
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

    // Quantize to int8, pack into 8 int32 words for dp4a, and compute
    // weighted sum (x_sum = x_scale * sum(quants)) for zero-point correction.
    int xv[8];
    int raw_sum = 0;
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
        raw_sum += b0 + b1 + b2 + b3;
        // Pack 4 int8 into one int32 (little-endian byte order for dp4a).
        xv[k] = (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24);
    }
    float x_sum = x_scale * (float)raw_sum;

    // --- dp4a dot product against NR weight rows ---
    #pragma unroll
    for (int row = 0; row < NR; row++) {
        if (r0 + row >= out_dim) break;

        const char* w_block = weight_q4a
            + (unsigned long long)(r0 + row) * row_bytes
            + (unsigned long long)ib * Q4_ALIGNED_BYTES;

        // Read f16 weight scale (bytes 0-1, native halfword load).
        unsigned short w_scale_bits = *(const unsigned short*)w_block;
        float w_scale = f16_bits_to_f32(w_scale_bits);

        // Aligned int* loads for nibble data (4-byte aligned at +4).
        const unsigned int* w_nibbles = (const unsigned int*)(w_block + 4);

        // Unpack + dp4a.
        int acc = 0;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            unsigned int packed = w_nibbles[k];
            int w0, w1;
            unpack_nibbles_4bytes(packed, w0, w1);
            acc = __dp4a(w0, xv[2 * k],     acc);
            acc = __dp4a(w1, xv[2 * k + 1], acc);
        }

        // Combined result with zero-point correction:
        //   dot(w, x) = w_scale * x_scale * dp4a_sum - 8 * w_scale * x_sum
        sumf[row] += w_scale * (x_scale * (float)acc - 8.0f * x_sum);
    }
}

// Cross-warp reduction: warp reduce + shmem partial sums -> thread 0 final sum.
// Writes results to output (without residual).
__device__ __forceinline__ void cross_warp_reduce_and_write(
    float sumf[NR],
    float* __restrict__ out,
    unsigned int r0,
    unsigned int out_dim)
{
    unsigned int warp_id = threadIdx.x / NW;
    unsigned int lane    = threadIdx.x % NW;

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

// Cross-warp reduction with fused residual addition.
__device__ __forceinline__ void cross_warp_reduce_and_write_residual(
    float sumf[NR],
    const float* __restrict__ residual,
    float* __restrict__ out,
    unsigned int r0,
    unsigned int out_dim)
{
    unsigned int warp_id = threadIdx.x / NW;
    unsigned int lane    = threadIdx.x % NW;

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
// Kernel 1: Q4Aligned weight x F32 input -> F32 output (inline quantize + dp4a).
//
// Reads F32 input, quantizes to Q8_1 in registers per-block, then dp4a
// against Q4Aligned weights. Eliminates the separate quantize kernel.
//
// Grid:  (ceil(out_dim / NR), 1, 1)
// Block: (256, 1, 1)
// ==========================================================================
extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 1) void matvec_q4_aligned_f32(
    const char* __restrict__ weight_q4a,   // [out_dim * nb * 20] Q4Aligned bytes
    const float* __restrict__ input_f32,   // [in_dim] F32 input vector
    float* __restrict__ out,               // [out_dim] F32 output
    unsigned int out_dim,
    unsigned int in_dim)
{
    unsigned int r0 = blockIdx.x * NR;
    unsigned int nb = in_dim >> 5;
    unsigned long long row_bytes = (unsigned long long)nb * Q4_ALIGNED_BYTES;

    float sumf[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) sumf[r] = 0.0f;

    for (unsigned int ib = threadIdx.x; ib < nb; ib += THREADS_PER_BLOCK) {
        unsigned int base = ib * Q4_BLOCK_SIZE;

        // Load 32 F32 values.
        float xf[32];
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            xf[j] = input_f32[base + j];
        }

        fused_q4a_dp4a_block(xf, sumf, weight_q4a, row_bytes, r0, ib, out_dim);
    }

    cross_warp_reduce_and_write(sumf, out, r0, out_dim);
}

// ==========================================================================
// Kernel 2: Q4Aligned weight x F32 input + residual -> F32 output.
// ==========================================================================
extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 1) void matvec_q4_aligned_f32_residual(
    const char* __restrict__ weight_q4a,
    const float* __restrict__ input_f32,
    const float* __restrict__ residual,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim)
{
    unsigned int r0 = blockIdx.x * NR;
    unsigned int nb = in_dim >> 5;
    unsigned long long row_bytes = (unsigned long long)nb * Q4_ALIGNED_BYTES;

    float sumf[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) sumf[r] = 0.0f;

    for (unsigned int ib = threadIdx.x; ib < nb; ib += THREADS_PER_BLOCK) {
        unsigned int base = ib * Q4_BLOCK_SIZE;

        float xf[32];
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            xf[j] = input_f32[base + j];
        }

        fused_q4a_dp4a_block(xf, sumf, weight_q4a, row_bytes, r0, ib, out_dim);
    }

    cross_warp_reduce_and_write_residual(sumf, residual, out, r0, out_dim);
}

// ==========================================================================
// Kernel 3: Q4Aligned weight x SwiGLU(gate, up) -> F32 output.
//
// Fuses: SwiGLU activation + F32->Q8_1 quantization + dp4a matvec.
// Replaces 3 separate dispatches (swiglu + quantize + matvec) with 1.
//
// Input: separate F32 gate[] and up[] buffers (from gate/up projections).
// Each thread computes silu(gate[j]) * up[j] for its 32 elements,
// quantizes inline, and does dp4a against Q4Aligned weight rows.
//
// Grid:  (ceil(out_dim / NR), 1, 1)
// Block: (128, 1, 1)
// ==========================================================================
extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 1) void matvec_q4_aligned_f32_swiglu(
    const char* __restrict__ weight_q4a,   // [out_dim * nb * 20] Q4Aligned bytes
    const float* __restrict__ gate,        // [in_dim] F32 gate projection
    const float* __restrict__ up,          // [in_dim] F32 up projection
    float* __restrict__ out,               // [out_dim] F32 output
    unsigned int out_dim,
    unsigned int in_dim)
{
    unsigned int r0 = blockIdx.x * NR;
    unsigned int nb = in_dim >> 5;
    unsigned long long row_bytes = (unsigned long long)nb * Q4_ALIGNED_BYTES;

    float sumf[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) sumf[r] = 0.0f;

    for (unsigned int ib = threadIdx.x; ib < nb; ib += THREADS_PER_BLOCK) {
        unsigned int base = ib * Q4_BLOCK_SIZE;

        // --- Compute SwiGLU: silu(gate[j]) * up[j] for 32 elements ---
        float xf[32];
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            float g = gate[base + j];
            float silu_g = g / (1.0f + expf(-g));
            xf[j] = silu_g * up[base + j];
        }

        fused_q4a_dp4a_block(xf, sumf, weight_q4a, row_bytes, r0, ib, out_dim);
    }

    cross_warp_reduce_and_write(sumf, out, r0, out_dim);
}

// ==========================================================================
// Kernel 4: Q4Aligned weight x SwiGLU(gate, up) + residual -> F32 output.
// ==========================================================================
extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 1) void matvec_q4_aligned_f32_swiglu_residual(
    const char* __restrict__ weight_q4a,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const float* __restrict__ residual,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim)
{
    unsigned int r0 = blockIdx.x * NR;
    unsigned int nb = in_dim >> 5;
    unsigned long long row_bytes = (unsigned long long)nb * Q4_ALIGNED_BYTES;

    float sumf[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) sumf[r] = 0.0f;

    for (unsigned int ib = threadIdx.x; ib < nb; ib += THREADS_PER_BLOCK) {
        unsigned int base = ib * Q4_BLOCK_SIZE;

        float xf[32];
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            float g = gate[base + j];
            float silu_g = g / (1.0f + expf(-g));
            xf[j] = silu_g * up[base + j];
        }

        fused_q4a_dp4a_block(xf, sumf, weight_q4a, row_bytes, r0, ib, out_dim);
    }

    cross_warp_reduce_and_write_residual(sumf, residual, out, r0, out_dim);
}
