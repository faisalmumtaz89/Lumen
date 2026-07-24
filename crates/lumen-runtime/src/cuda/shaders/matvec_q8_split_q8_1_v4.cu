// ==========================================================================
// Q8 split-layout matvec against pre-quantized Q8_1 input, 128-bit WEIGHT LOADS.
//
// Bit-for-bit-identical sibling of matvec_q8_split_q8_1 (NR=2). The ONLY delta
// is the weight-quant stream load width:
//
//   matvec_q8_split_q8_1 (scalar):  8 x `int`  loads per block  (8 x LDG.E.32)
//   matvec_q8_split_q8_1_v4 (this): 2 x `int4` loads per block  (2 x LDG.E.128)
//
// The dp4a integer dot consumes the SAME 8 int32 words in the SAME order, so
// the accumulator, the per-block F32 epilogue, the warp all-reduce, and the
// cross-warp fold are UNCHANGED. Output is byte-identical to the scalar kernel
// (verified by construction: same op sequence, wider load only). This makes the
// kernel safe for the greedy-decode byte-identical gate and the MoE router.
//
// RATIONALE (why wider is faster on A100 decode)
// ----------------------------------------------
// Each thread owns one Q8 block (ib = threadIdx.x; ib += 128). Consecutive
// lanes therefore address the weight stream with a 32-BYTE stride, so a single
// `int` load spreads the warp across 32 distinct 32-byte sectors (128 bytes
// requested, 1 KB of sectors touched) -- fully consumed across the k-loop but at
// a high L1/LSU request rate. Coalescing the 8 words into 2 x int4 cuts the
// weight load-instruction count 4x (8 -> 2) and the sector-request rate ~4x, so
// the same DRAM bytes are moved with far less LSU/issue pressure. On the Q8_0
// decode matvec (measured ~58% of the A100 SXM4 roofline, i.e. NOT bandwidth-
// saturated) that request-rate is a real secondary limiter; wider loads target
// it directly. Default-OFF (LUMEN_CUDA_Q8_MATVEC_FAST) pending on-A100 A/B.
//
// ALIGNMENT CONTRACT (enforced by the host dispatch)
// --------------------------------------------------
//   * The int4 loads require the quant stream base
//       row_base + 2*nb + ib*32
//     to be 16-byte aligned. `ib*32` is always 16-aligned; `2*nb` and the
//     per-row stride `34*nb` are 16-aligned IFF nb % 8 == 0, i.e. in_dim % 256
//     == 0. The host only routes here when `in_dim % 256 == 0` (every shipped
//     Q8 matvec dim: 4096, 12288, ... satisfies this). `cudaMalloc` returns
//     256-byte aligned bases, so the row-0 pointer is 16-aligned.
//   * The Q8_1 INPUT stream keeps scalar `int` loads: x_block = input + ib*36,
//     quants at +4 -> NOT 16-aligned, and it is L2-resident (not the DRAM
//     bottleneck), so widening it is neither safe nor necessary.
//
// Requires compute capability >= 6.1 for __dp4a() (Pascal+).
// NVRTC-compatible: no system includes, extern "C" linkage. `int4` is a CUDA
// builtin vector type (no header needed).
// ==========================================================================

#define NR       2     // rows per thread block
#define NW       32    // warp size
#define THREADS_PER_BLOCK 128  // 4 warps
#define NWARPS   (THREADS_PER_BLOCK / NW)  // 4
#define Q8_BLOCK_SIZE     32   // elements per Q8 block
#define Q8_1_BYTES        36   // 2B f16 scale + 2B f16 sum + 32B int8 data

// Hardware f16->f32 conversion via PTX (single instruction on SM 53+).
__device__ __forceinline__ float f16_bits_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

// inline-PTX dp4a wrapper (matches matvec_q8_split_q8_1.cu: the `__dp4a`
// intrinsic NVRTC-fails PTX-load on this sm_80 build env).
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
// Kernel 1: split Q8 weight x Q8_1 input -> F32 output (dp4a, NR=2, int4 wload).
// Grid:  (ceil(out_dim / NR), 1, 1)   Block: (128, 1, 1)
// ==========================================================================
extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 2) void matvec_q8_split_q8_1_v4(
    const char* __restrict__ weight_q8_split,    // [out_dim * nb * 34] split bytes
    const char* __restrict__ input_q8_1,         // [nb * 36] Q8_1 pre-quantized input
    float* __restrict__ out,                     // [out_dim] F32 output
    unsigned int out_dim,
    unsigned int in_dim)
{
    unsigned int r0 = blockIdx.x * NR;

    unsigned int warp_id = threadIdx.x / NW;
    unsigned int lane    = threadIdx.x % NW;

    unsigned int nb = in_dim >> 5;  // in_dim / 32
    unsigned long long row_bytes = (unsigned long long)nb * 34ULL;
    unsigned long long scales_bytes_per_row = (unsigned long long)nb * 2ULL;

    float sumf[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) sumf[r] = 0.0f;

    for (unsigned int ib = threadIdx.x; ib < nb; ib += THREADS_PER_BLOCK) {

        // --- Load Q8_1 input block (scalar int; shared across NR rows) ---
        const char* x_block = input_q8_1 + (unsigned long long)ib * Q8_1_BYTES;

        unsigned short x_scale_bits = (unsigned short)(unsigned char)x_block[0]
                                    | ((unsigned short)(unsigned char)x_block[1] << 8);
        float x_scale = f16_bits_to_f32(x_scale_bits);

        const int* x_packed = (const int*)(x_block + 4);
        int xv[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) xv[k] = x_packed[k];

        // --- Process NR output rows ---
        #pragma unroll
        for (int row = 0; row < NR; row++) {
            if (r0 + row >= out_dim) break;

            const char* row_base = weight_q8_split
                + (unsigned long long)(r0 + row) * row_bytes;

            // Scale stream: nb x f16 at offset 0. Read 2 bytes for block `ib`.
            const char* scale_byte = row_base + (unsigned long long)ib * 2ULL;
            unsigned short w_scale_bits = (unsigned short)(unsigned char)scale_byte[0]
                                        | ((unsigned short)(unsigned char)scale_byte[1] << 8);
            float w_scale = f16_bits_to_f32(w_scale_bits);

            // Quant stream: 32 int8 = 8 int32 = 2 int4 at offset 2*nb + ib*32.
            // 16-byte aligned (host guards in_dim % 256 == 0). Two 128-bit loads
            // replace eight 32-bit loads; the eight int32 words feed dp4a in the
            // SAME order as the scalar kernel -> identical accumulator.
            const int4* w_packed4 = (const int4*)(row_base
                + scales_bytes_per_row
                + (unsigned long long)ib * 32ULL);
            int4 w0 = w_packed4[0];   // words 0..3
            int4 w1 = w_packed4[1];   // words 4..7

            int acc = 0;
            acc = dp4a_s32(w0.x, xv[0], acc);
            acc = dp4a_s32(w0.y, xv[1], acc);
            acc = dp4a_s32(w0.z, xv[2], acc);
            acc = dp4a_s32(w0.w, xv[3], acc);
            acc = dp4a_s32(w1.x, xv[4], acc);
            acc = dp4a_s32(w1.y, xv[5], acc);
            acc = dp4a_s32(w1.z, xv[6], acc);
            acc = dp4a_s32(w1.w, xv[7], acc);

            sumf[row] += w_scale * x_scale * (float)acc;
        }
    }

    // --- Cross-warp reduction via simple shmem (identical to scalar kernel) ---
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
// Kernel 2: split Q8 weight x Q8_1 input + residual -> F32 output (int4 wload).
// Fused residual add at the final write. Used for FFN down / attention out.
// ==========================================================================
extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 2) void matvec_q8_split_q8_1_v4_residual(
    const char* __restrict__ weight_q8_split,    // [out_dim * nb * 34] split bytes
    const char* __restrict__ input_q8_1,         // [nb * 36] Q8_1 pre-quantized input
    const float* __restrict__ residual,          // [out_dim] F32 residual
    float* __restrict__ out,                     // [out_dim] F32 output
    unsigned int out_dim,
    unsigned int in_dim)
{
    unsigned int r0 = blockIdx.x * NR;

    unsigned int warp_id = threadIdx.x / NW;
    unsigned int lane    = threadIdx.x % NW;

    unsigned int nb = in_dim >> 5;
    unsigned long long row_bytes = (unsigned long long)nb * 34ULL;
    unsigned long long scales_bytes_per_row = (unsigned long long)nb * 2ULL;

    float sumf[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) sumf[r] = 0.0f;

    for (unsigned int ib = threadIdx.x; ib < nb; ib += THREADS_PER_BLOCK) {

        const char* x_block = input_q8_1 + (unsigned long long)ib * Q8_1_BYTES;

        unsigned short x_scale_bits = (unsigned short)(unsigned char)x_block[0]
                                    | ((unsigned short)(unsigned char)x_block[1] << 8);
        float x_scale = f16_bits_to_f32(x_scale_bits);

        const int* x_packed = (const int*)(x_block + 4);
        int xv[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) xv[k] = x_packed[k];

        #pragma unroll
        for (int row = 0; row < NR; row++) {
            if (r0 + row >= out_dim) break;

            const char* row_base = weight_q8_split
                + (unsigned long long)(r0 + row) * row_bytes;

            const char* scale_byte = row_base + (unsigned long long)ib * 2ULL;
            unsigned short w_scale_bits = (unsigned short)(unsigned char)scale_byte[0]
                                        | ((unsigned short)(unsigned char)scale_byte[1] << 8);
            float w_scale = f16_bits_to_f32(w_scale_bits);

            const int4* w_packed4 = (const int4*)(row_base
                + scales_bytes_per_row
                + (unsigned long long)ib * 32ULL);
            int4 w0 = w_packed4[0];
            int4 w1 = w_packed4[1];

            int acc = 0;
            acc = dp4a_s32(w0.x, xv[0], acc);
            acc = dp4a_s32(w0.y, xv[1], acc);
            acc = dp4a_s32(w0.z, xv[2], acc);
            acc = dp4a_s32(w0.w, xv[3], acc);
            acc = dp4a_s32(w1.x, xv[4], acc);
            acc = dp4a_s32(w1.y, xv[5], acc);
            acc = dp4a_s32(w1.z, xv[6], acc);
            acc = dp4a_s32(w1.w, xv[7], acc);

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
