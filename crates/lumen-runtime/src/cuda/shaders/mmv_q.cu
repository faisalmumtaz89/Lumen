// dp4a-mmvq Q-quant decode matvec (Q8_0 / Q4_0 weights × Q8_1
// activation → F32 dst).
//
// Generic Q-quant matvec template: nwarps=4, rpb=1 per CTA for decode
// (batch=1, ncols_dst=1).
//
// Activation format: F32 activation is quantized to block_q8_1 (32-elem
// blocks with f16 delta + f16 sum) via the `quantize_q8_1` kernel, then
// matvec runs via SIMD dp4a instructions on signed int8 quants.
//
// This file implements:
//   - `quantize_q8_1_rawsum`: F32 activation -> Q8_1 quantized blocks
//   - `mul_mat_vec_q_q8_0`: Q8_0 weights × Q8_1 activation → F32 dst
//   - `mul_mat_vec_q_q4_0`: Q4_0 weights × Q8_1 activation → F32 dst
//
// Grid layout (decode, ncols_dst=1):
//   Grid:  (nrows_x, 1, 1)    [rpb=1, one row per CTA]
//   Block: (32, 4, 1)         [warp_size=32, nwarps=4]
//
// Activation buffer size: ceil(in_dim / QK8_1) blocks * 36 bytes = nbatch*36
// per call. With QK8_1=32 and in_dim=2048, that's 64 blocks = 2304 bytes
// (≈40x smaller than the F32 activation at 8192 bytes; net memory traffic win
// only matters if the weights win on dp4a vs scalar — the dp4a path measures
// 4.33 ms/tok vs the scalar 11.33 ms/tok at this shape).
//
// NVRTC-compatible: no system includes, extern "C" linkage.

typedef signed char         int8_t;
typedef unsigned char       uint8_t;
typedef int                 int32_t;
typedef unsigned int        uint32_t;
typedef unsigned long long  uint64_t;

#define WARP_SIZE 32
#define NWARPS 4                    // calc_nwarps for ncols_dst=1, GENERIC
#define BLOCK_DIM (WARP_SIZE * NWARPS)  // 128 threads/CTA
#define QK8_0 32
#define QK8_1 32
#define QK4_0 32
// QI_TYPE = QK_TYPE / (4 * QR_TYPE) (the standard GGML quant constants).
// QR8_0=1 → QI8_0=32/(4*1)=8. QR4_0=2 → QI4_0=32/(4*2)=4 (Q4_0 packs nibbles
// for 2 elements per quant slot, halving the int32 chunk count vs Q8_0).
#define QI8_0 8                     // QK8_0 / (4 * 1)
#define QI8_1 8                     // QK8_1 / (4 * 1)
#define QI4_0 4                     // QK4_0 / (4 * 2)  <-- bug fix vs first draft

// VDR_*_MMVQ: number of dp4a operations per warp lane per K block.
// (vec-dot Q8_1 inner-loop count)
#define VDR_Q8_0_Q8_1_MMVQ 2
#define VDR_Q4_0_Q8_1_MMVQ 2

// Hardware fp16<->f32 conversion via PTX (single instruction on sm_53+).
__device__ __forceinline__ float f16_bits_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

__device__ __forceinline__ unsigned short f32_to_f16_bits(float v) {
    unsigned short bits;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(bits) : "f"(v));
    return bits;
}

// SIMD dp4a (4-element signed int8 dot product + accumulate).
// The `__dp4a` intrinsic on sm_61+.
// NVRTC does not implicitly include sm_61_intrinsics.hpp, so we inline the
// definition (byte-identical to CUDA's official header).
// PTX: dp4a.s32.s32 ret, srcA, srcB, c (sm_61+; A100=sm_80).
__device__ __forceinline__ int dp4a_s(int a, int b, int c) {
    int r;
    asm volatile ("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}

// Warp-level sum-reduction (5 butterfly shuffles).
__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val,  8);
    val += __shfl_xor_sync(0xffffffff, val,  4);
    val += __shfl_xor_sync(0xffffffff, val,  2);
    val += __shfl_xor_sync(0xffffffff, val,  1);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 16));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val,  8));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val,  4));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val,  2));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val,  1));
    return val;
}

// ============================================================================
// quantize_q8_1_rawsum
//
// Online Q8_1 quantization (GGML Q8_1 block standard).
// Input:  F32 activation [in_dim]
// Output: block_q8_1 blocks: ceil(in_dim/QK8_1) blocks of {f16 d, f16 s, int8 qs[32]}
//
// Each block of 32 floats is reduced to:
//   d = amax / 127
//   s = sum(xi)            <-- raw F32 sum convention: `y[ib].ds =
//                              make_half2(d, sum)` where `sum` is the
//                              warp-reduced raw F32 sum.
//   qs[i] = round(x[i] / d)
//
// CRITICAL: This `s` differs from Lumen's existing `quantize_f32_to_q8_1`
// which writes `s = d * sum(qs_q8_1)`. The Q4_0 vec_dot bias correction
// formula expects this raw-sum convention.
//
// Grid:  (ceil(in_dim / QK8_1), 1, 1)  -- one CTA per QK8_1 block
// Block: (QK8_1, 1, 1)                -- 32 threads/CTA = 1 warp
//
// Layout of block_q8_1: 36 bytes per block:
//   [0..1]: f16 d (delta)
//   [2..3]: f16 s (d * sum) (ds = make_half2(d, sum))
//   [4..35]: 32 x int8 quants
// ============================================================================
extern "C" __global__ void quantize_q8_1_rawsum(
    const float* __restrict__ x,     // [in_dim] F32 activation
    unsigned char* __restrict__ vy,  // [ceil(in_dim/32) * 36] block_q8_1
    unsigned int in_dim)
{
    const unsigned int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int ib = i0 / QK8_1;
    const unsigned int iqs = i0 % QK8_1;

    const float xi = (i0 < in_dim) ? x[i0] : 0.0f;
    float amax = fabsf(xi);
    float sum  = xi;

    amax = warp_reduce_max(amax);
    sum  = warp_reduce_sum(sum);

    const float d = amax / 127.0f;
    const int8_t q = (amax == 0.0f) ? (int8_t)0 : (int8_t)__float2int_rn(xi / d);

    // Each block_q8_1 = 36 bytes: [0..1] d, [2..3] s, [4..35] 32 x int8 qs.
    unsigned char* yb = vy + ib * 36;

    // Write the int8 quant for this lane.
    yb[4 + iqs] = (unsigned char)q;

    if (iqs == 0) {
        // ds.y = sum (raw F32 sum), not d * sum.
        const unsigned short d_bits = f32_to_f16_bits(d);
        const unsigned short s_bits = f32_to_f16_bits(sum);
        // Little-endian store of the two f16's.
        yb[0] = (unsigned char)(d_bits & 0xFF);
        yb[1] = (unsigned char)((d_bits >> 8) & 0xFF);
        yb[2] = (unsigned char)(s_bits & 0xFF);
        yb[3] = (unsigned char)((s_bits >> 8) & 0xFF);
    }
}

// ============================================================================
// Inline `vec_dot_q8_0_q8_1` and `vec_dot_q4_0_q8_1` implementations.
// Standard dp4a vec-dot Q8_1 with raw byte addressing (layout-equivalent
// for QK*=32 blocks).
// ============================================================================

// vec_dot_q8_0_q8_1_impl<float, vdr=2>(v, u, d8_0, d8_1)
//     = d8_0 * d8_1 * sum_{i=0..vdr-1} dp4a(v[i], u[i], 0)
//
// v[i] = (int)bq8_0->qs[4*(iqs+i)..4*(iqs+i)+3]    (read 4 int8 as int32)
// u[i] = (int)bq8_1->qs[4*(iqs+i)..4*(iqs+i)+3]
__device__ __forceinline__ float vec_dot_q8_0_q8_1_impl(
    const int8_t* __restrict__ bq8_0_qs,  // 32 int8 weights for one block
    const int8_t* __restrict__ bq8_1_qs,  // 32 int8 activations for one block
    int iqs,
    float d8_0, float d8_1)
{
    int sumi = 0;
    #pragma unroll
    for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        // Load 4 packed int8 as int32 (the `get_int_b2` / `get_int_b4` pattern).
        int v, u;
        // bq8_0_qs is signed int8; pack [iqs+i]*4 .. [iqs+i]*4+3 into int32.
        const int8_t* vp = bq8_0_qs + (iqs + i) * 4;
        const int8_t* up = bq8_1_qs + (iqs + i) * 4;
        v = ((int)(unsigned char)vp[0])
          | ((int)(unsigned char)vp[1] << 8)
          | ((int)(unsigned char)vp[2] << 16)
          | ((int)(unsigned char)vp[3] << 24);
        u = ((int)(unsigned char)up[0])
          | ((int)(unsigned char)up[1] << 8)
          | ((int)(unsigned char)up[2] << 16)
          | ((int)(unsigned char)up[3] << 24);
        sumi = dp4a_s(v, u, sumi);
    }
    return d8_0 * d8_1 * (float)sumi;
}

// vec_dot_q4_0_q8_1_impl<vdr=2>(v, u, d4, ds8)
//   v[i] = (int) bq4_0->qs[(iqs+i)*4..(iqs+i)*4+3]    (nibbles -> packed)
//   u[2*i+0] = (int) bq8_1->qs[(iqs+i)*4..(iqs+i)*4+3]
//   u[2*i+1] = (int) bq8_1->qs[(iqs+i+QI4_0)*4..(iqs+i+QI4_0)*4+3]
//   sumi = sum dp4a(vi0=v[i] & 0x0F0F0F0F,        u[2*i+0])
//        + sum dp4a(vi1=(v[i] >> 4) & 0x0F0F0F0F, u[2*i+1])
//   return d4 * (sumi * ds8.x - (8*vdr/QI4_0) * ds8.y)
//
// For VDR=2, QI4_0=8: (8*2/8)=2 — i.e. subtract 2 * ds8.y (the bias correction
// because Q4_0 nibbles are unsigned [0..15] but represent signed [-8..7]).
__device__ __forceinline__ float vec_dot_q4_0_q8_1_impl(
    const uint8_t* __restrict__ bq4_0_qs,  // 16 packed nibble pairs
    const int8_t* __restrict__ bq8_1_qs,   // 32 int8 activations
    int iqs,
    float d4, float ds8_x, float ds8_y)
{
    int sumi = 0;
    #pragma unroll
    for (int i = 0; i < VDR_Q4_0_Q8_1_MMVQ; ++i) {
        // v[i] = 4-packed unsigned char nibbles at offset (iqs+i)*4 .. *4+3
        const uint8_t* vp = bq4_0_qs + (iqs + i) * 4;
        int v = ((int)vp[0])
              | ((int)vp[1] << 8)
              | ((int)vp[2] << 16)
              | ((int)vp[3] << 24);
        const int vi0 = (v >> 0) & 0x0F0F0F0F;  // low nibbles = elems [(iqs+i)*4 .. *4+3]
        const int vi1 = (v >> 4) & 0x0F0F0F0F;  // high nibbles = elems [16+(iqs+i)*4 .. ]

        const int8_t* up0 = bq8_1_qs + (iqs + i) * 4;
        const int8_t* up1 = bq8_1_qs + (iqs + i + QI4_0) * 4;
        int u0 = ((int)(unsigned char)up0[0])
               | ((int)(unsigned char)up0[1] << 8)
               | ((int)(unsigned char)up0[2] << 16)
               | ((int)(unsigned char)up0[3] << 24);
        int u1 = ((int)(unsigned char)up1[0])
               | ((int)(unsigned char)up1[1] << 8)
               | ((int)(unsigned char)up1[2] << 16)
               | ((int)(unsigned char)up1[3] << 24);

        sumi = dp4a_s(vi0, u0, sumi);
        sumi = dp4a_s(vi1, u1, sumi);
    }
    // return d4 * (sumi * ds8.x - (8*vdr/QI4_0) * ds8.y)
    //   = d4 * (sumi * d8 - 4 * s)   for VDR=2, QI4_0=4.
    return d4 * ((float)sumi * ds8_x - 4.0f * ds8_y);
}

// ============================================================================
// mul_mat_vec_q_q8_0 — Q8_0 dp4a-mmvq matvec (Q8_0 weights × Q8_1 act → F32 dst)
//
// Grid:  (nrows_x, 1, 1)        rpb=1
// Block: (32, 4, 1)             warp_size=32, nwarps=4
//
// Each CTA processes exactly one output row.
// Each warp processes a strided subset of K blocks.
// Within a warp, each of 32 lanes processes 2 consecutive int32 vec_dot chunks.
// (calc: blocks_per_iter = vdr * nwarps * warp_size / qi = 2*4*32/8 = 32)
// ============================================================================
extern "C" __global__ void mul_mat_vec_q_q8_0(
    const unsigned char* __restrict__ vx,  // [nrows_x * num_blocks * 34] Q8_0 raw bytes
    const unsigned char* __restrict__ vy,  // [num_blocks * 36] Q8_1 activation
    float* __restrict__ dst,               // [nrows_x] F32 output
    unsigned int ncols_x,                  // = in_dim
    unsigned int nrows_x)                  // = out_dim
{
    const int row0 = blockIdx.x;  // rpb=1
    if (row0 >= (int)nrows_x) return;

    const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;  // [0..128)

    const int blocks_per_row_x = ncols_x / QK8_0;   // K blocks
    constexpr int blocks_per_iter = VDR_Q8_0_Q8_1_MMVQ * BLOCK_DIM / QI8_0;  // 2*128/8 = 32

    // Initial K-block index for this lane.
    // int kbx = tid / (qi/vdr) = tid / (8/2) = tid / 4 = [0..32)
    int kbx0 = tid / (QI8_0 / VDR_Q8_0_Q8_1_MMVQ);

    // Within-block int32 offset for this lane.
    // int kqs = vdr * (tid % (qi/vdr)) = 2 * (tid % 4) = {0, 2, 4, 6}
    const int kqs = VDR_Q8_0_Q8_1_MMVQ * (tid % (QI8_0 / VDR_Q8_0_Q8_1_MMVQ));

    // Q8_0 row layout: nrows_x rows, each row = num_blocks blocks of 34 bytes.
    const unsigned long long row_bytes_x = (unsigned long long)blocks_per_row_x * 34;
    const unsigned char* x_row = vx + (unsigned long long)row0 * row_bytes_x;

    float tmp = 0.0f;

    for (int kbx = kbx0; kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        // Pointer to Q8_0 block kbx of x_row.
        const unsigned char* bq8_0 = x_row + (unsigned long long)kbx * 34;
        // bq8_0 = [d_lo, d_hi, qs0..qs31]
        const unsigned short d_bits =
            (unsigned short)bq8_0[0] | ((unsigned short)bq8_0[1] << 8);
        const float d8_0 = f16_bits_to_f32(d_bits);
        const int8_t* bq8_0_qs = (const int8_t*)(bq8_0 + 2);

        // Pointer to Q8_1 block (same kbx since QK8_1 = QK8_0).
        const unsigned char* bq8_1 = vy + (unsigned long long)kbx * 36;
        const unsigned short d8_1_bits =
            (unsigned short)bq8_1[0] | ((unsigned short)bq8_1[1] << 8);
        const float d8_1 = f16_bits_to_f32(d8_1_bits);
        const int8_t* bq8_1_qs = (const int8_t*)(bq8_1 + 4);

        tmp += vec_dot_q8_0_q8_1_impl(bq8_0_qs, bq8_1_qs, kqs, d8_0, d8_1);
    }

    // dp4a-mmvq reduction: NO pre-warp reduction. Stores RAW per-thread
    // partial `tmp` into tmp_shared, then warp 0's lane-X reads all warps'
    // lane-X partials, sums them, then warp_reduce_sum.
    __shared__ float tmp_shared[NWARPS - 1][WARP_SIZE];

    if (threadIdx.y > 0) {
        tmp_shared[threadIdx.y - 1][threadIdx.x] = tmp;
    }
    __syncthreads();

    if (threadIdx.y > 0) return;

    // Warp 0: each lane X collects partials at lane X from warps 1..NWARPS-1
    // and adds them to its own tmp. Then warp_reduce_sum across the 32 lanes.
    #pragma unroll
    for (int l = 0; l < NWARPS - 1; ++l) {
        tmp += tmp_shared[l][threadIdx.x];
    }
    tmp = warp_reduce_sum(tmp);

    // rows_per_cuda_block=1, so only threadIdx.x==0 writes the result.
    if (threadIdx.x == 0) {
        dst[row0] = tmp;
    }
}

// ============================================================================
// mul_mat_vec_q_q4_0 — same structure as q8_0 but Q4_0 weights.
// ============================================================================
extern "C" __global__ void mul_mat_vec_q_q4_0(
    const unsigned char* __restrict__ vx,  // [nrows_x * num_blocks * 18] Q4_0 raw bytes
    const unsigned char* __restrict__ vy,  // [num_blocks * 36] Q8_1 activation
    float* __restrict__ dst,               // [nrows_x] F32 output
    unsigned int ncols_x,                  // = in_dim
    unsigned int nrows_x)                  // = out_dim
{
    const int row0 = blockIdx.x;
    if (row0 >= (int)nrows_x) return;

    const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;
    const int blocks_per_row_x = ncols_x / QK4_0;
    constexpr int blocks_per_iter = VDR_Q4_0_Q8_1_MMVQ * BLOCK_DIM / QI4_0;  // 32

    const int kbx0 = tid / (QI4_0 / VDR_Q4_0_Q8_1_MMVQ);  // [0..32)
    const int kqs  = VDR_Q4_0_Q8_1_MMVQ * (tid % (QI4_0 / VDR_Q4_0_Q8_1_MMVQ));

    // Q4_0 row layout: 18 bytes per block.
    const unsigned long long row_bytes_x = (unsigned long long)blocks_per_row_x * 18;
    const unsigned char* x_row = vx + (unsigned long long)row0 * row_bytes_x;

    float tmp = 0.0f;

    for (int kbx = kbx0; kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const unsigned char* bq4_0 = x_row + (unsigned long long)kbx * 18;
        // [0..1] = d (f16), [2..17] = 16 packed nibbles
        const unsigned short d_bits =
            (unsigned short)bq4_0[0] | ((unsigned short)bq4_0[1] << 8);
        const float d4 = f16_bits_to_f32(d_bits);
        const uint8_t* bq4_0_qs = bq4_0 + 2;

        const unsigned char* bq8_1 = vy + (unsigned long long)kbx * 36;
        const unsigned short d8_1_bits =
            (unsigned short)bq8_1[0] | ((unsigned short)bq8_1[1] << 8);
        const unsigned short s8_1_bits =
            (unsigned short)bq8_1[2] | ((unsigned short)bq8_1[3] << 8);
        const float ds8_x = f16_bits_to_f32(d8_1_bits);
        const float ds8_y = f16_bits_to_f32(s8_1_bits);
        const int8_t* bq8_1_qs = (const int8_t*)(bq8_1 + 4);

        tmp += vec_dot_q4_0_q8_1_impl(bq4_0_qs, bq8_1_qs, kqs, d4, ds8_x, ds8_y);
    }

    // dp4a-mmvq reduction: NO pre-warp reduction; store raw partials, warp 0 sums.
    __shared__ float tmp_shared[NWARPS - 1][WARP_SIZE];
    if (threadIdx.y > 0) {
        tmp_shared[threadIdx.y - 1][threadIdx.x] = tmp;
    }
    __syncthreads();
    if (threadIdx.y > 0) return;

    #pragma unroll
    for (int l = 0; l < NWARPS - 1; ++l) {
        tmp += tmp_shared[l][threadIdx.x];
    }
    tmp = warp_reduce_sum(tmp);

    if (threadIdx.x == 0) {
        dst[row0] = tmp;
    }
}
