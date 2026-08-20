// ============================================================================
// BF16 (brain floating-point) matrix-vector multiply for CUDA.
//
// IEEE-style bfloat16 weights: each weight is 2 bytes (unsigned short),
// no block structure, no scale factors. Plain contiguous bf16 values.
// Weight matrix [out_dim, in_dim] stored row-major as out_dim * in_dim * 2 bytes.
//
// BF16 layout: 1 sign | 8 exponent | 7 mantissa. Same dynamic range as F32
// (both have 8-bit exponent); precision is 7 fractional bits (~2.4 decimal).
//
// Strategy: one block per output row, 256 threads per block, warp shuffle
// reduction. Each thread strides across the input dimension, accumulating
// in f32. Uses packed u32 loads for 2 BF16 values at once.
//
// NVRTC-compatible: hardware cvt.f32.bf16 PTX instruction available on
// SM_80+ (PTX ISA 7.0). The runtime loads this file with compute_80; older
// GPUs do not support the conversion intrinsic and will not select this
// dispatch path.
// ============================================================================

// ---------------------------------------------------------------------------
// BF16 -> F32 conversion (NVRTC-safe, no cuda_bf16.h required)
//
// BF16 is the top 16 bits of an IEEE 754 binary32 value, so conversion to
// F32 is a left-shift by 16. Use this fallback when cvt.f32.bf16 is not
// available; on SM_80+ the compiler typically promotes the shift to the
// dedicated CVT instruction anyway, but the explicit bit-cast is provably
// equivalent and avoids any compute-capability dependency at compile time.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float bf16_bits_to_f32(unsigned short b) {
    unsigned int x = ((unsigned int)b) << 16;
    return __int_as_float((int)x);
}

// ---------------------------------------------------------------------------
// Warp-level reduction (sum)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum_bf16(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffffu, val, offset);
    }
    return val;
}

// ---------------------------------------------------------------------------
// Block-level reduction (sum) using shared memory + warp shuffle
// ---------------------------------------------------------------------------
__device__ float block_reduce_sum_bf16(float val) {
    __shared__ float shared[8]; // max 256 threads = 8 warps

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    val = warp_reduce_sum_bf16(val);

    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    // First warp reduces across all warps
    int num_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < (unsigned int)num_warps) ? shared[threadIdx.x] : 0.0f;
    if (warp_id == 0) {
        val = warp_reduce_sum_bf16(val);
    }
    return val;
}

// ============================================================================
// matvec_bf16: out[row] = dot(W_bf16[row, :], x[:])
//
// Dispatch: grid = (out_dim,), block = (256,)
// One block per output row.
// ============================================================================
extern "C" __global__ void matvec_bf16(
    const unsigned short* __restrict__ weight_bf16, // [out_dim * in_dim] bf16 bits
    const float*          __restrict__ x,           // [in_dim] f32
    float*                __restrict__ out,          // [out_dim] f32
    unsigned int out_dim,
    unsigned int in_dim)
{
    unsigned int row = blockIdx.x;
    if (row >= out_dim) return;

    const unsigned short* row_ptr =
        weight_bf16 + (unsigned long long)row * in_dim;

    float sum = 0.0f;

    // Stride across in_dim, accumulating in f32.
    // Load 2 BF16 values at once via a single 32-bit load when aligned.
    unsigned int aligned_in = in_dim & ~1u; // floor to even
    for (unsigned int j = threadIdx.x * 2u; j < aligned_in; j += blockDim.x * 2u) {
        unsigned int packed = *(const unsigned int*)(row_ptr + j);
        float w0 = bf16_bits_to_f32((unsigned short)(packed & 0xffffu));
        float w1 = bf16_bits_to_f32((unsigned short)(packed >> 16));
        sum += w0 * x[j] + w1 * x[j + 1u];
    }

    // Handle odd trailing element
    if (in_dim & 1u) {
        unsigned int j = aligned_in + threadIdx.x;
        if (j < in_dim) {
            sum += bf16_bits_to_f32(row_ptr[j]) * x[j];
        }
    }

    // Block-level reduction
    sum = block_reduce_sum_bf16(sum);

    if (threadIdx.x == 0) {
        out[row] = sum;
    }
}

// ============================================================================
// matvec_bf16_residual: out[row] = dot(W_bf16[row, :], x[:]) + residual[row]
//
// Fused residual add: saves one kernel launch and one global memory pass.
// Dispatch: grid = (out_dim,), block = (256,)
// ============================================================================
extern "C" __global__ void matvec_bf16_residual(
    const unsigned short* __restrict__ weight_bf16, // [out_dim * in_dim] bf16 bits
    const float*          __restrict__ x,           // [in_dim] f32
    float*                __restrict__ out,          // [out_dim] f32
    const float*          __restrict__ residual,     // [out_dim] f32
    unsigned int out_dim,
    unsigned int in_dim)
{
    unsigned int row = blockIdx.x;
    if (row >= out_dim) return;

    const unsigned short* row_ptr =
        weight_bf16 + (unsigned long long)row * in_dim;

    float sum = 0.0f;

    unsigned int aligned_in = in_dim & ~1u;
    for (unsigned int j = threadIdx.x * 2u; j < aligned_in; j += blockDim.x * 2u) {
        unsigned int packed = *(const unsigned int*)(row_ptr + j);
        float w0 = bf16_bits_to_f32((unsigned short)(packed & 0xffffu));
        float w1 = bf16_bits_to_f32((unsigned short)(packed >> 16));
        sum += w0 * x[j] + w1 * x[j + 1u];
    }

    if (in_dim & 1u) {
        unsigned int j = aligned_in + threadIdx.x;
        if (j < in_dim) {
            sum += bf16_bits_to_f32(row_ptr[j]) * x[j];
        }
    }

    sum = block_reduce_sum_bf16(sum);

    if (threadIdx.x == 0) {
        out[row] = sum + residual[row];
    }
}

// ============================================================================
// matvec_bf16_v4: bandwidth-optimal BF16 GEMV via uint4 (16-byte) vectorized
// weight loads. out[row] = dot(W_bf16[row, :], x[:]), F32 accumulate.
//
// Motivation: the legacy `matvec_bf16` above loads only 2 BF16 per 32-bit
// instruction (4 bytes/thread/iter). At batch-1 decode every BF16 projection
// matvec is HBM-bandwidth bound, so this variant issues one uint4 load
// (16 bytes = 8 BF16) per row per iteration to maximise coalesced transaction
// width and halve the load-instruction count, and processes NR_BF16=2 output
// rows per block so the (small) F32 x reload is amortised. Warp+block
// reduction identical in shape to matvec_q8_0_dp4a.
//
// Numerics: BF16->F32 is a lossless top-16-bits bit-cast (bf16v4_to_f32),
// multiplied by the F32 x and accumulated in F32 — i.e. the SAME exact-F32
// math as the legacy matvec_bf16 fallback, and strictly more precise than the
// cuBLAS GemmEx F16-tensor-core downcast. Only the F32 summation ORDER differs
// (chunks of 8, NR=2 blocking) which is within F32 tolerance.
//
// REQUIRES in_dim % 8 == 0 so that (a) every 8-element uint4 load is 16-byte
// aligned (row start weight_bf16 + row*in_dim is 16B-aligned when in_dim%8==0,
// and the per-thread offset is a multiple of 8) and (b) there is no scalar
// tail. The Rust caller gates on `in_dim % 8 == 0`; other shapes fall through
// to the cuBLAS GemmEx path. FFN/attention/GDN in-dims are multiples of 8.
//
// Dispatch: grid = (ceil(out_dim / NR_BF16), 1, 1), block = (128, 1, 1).
// ============================================================================
#define NR_BF16  2    // output rows per block (amortise x reload)
#define NW_BF16  32   // warp size
#define TPB_BF16 128  // threads per block = 4 warps

__device__ __forceinline__ float warp_reduce_sum_bf16v4(float val) {
    val += __shfl_xor_sync(0xffffffffu, val, 16);
    val += __shfl_xor_sync(0xffffffffu, val, 8);
    val += __shfl_xor_sync(0xffffffffu, val, 4);
    val += __shfl_xor_sync(0xffffffffu, val, 2);
    val += __shfl_xor_sync(0xffffffffu, val, 1);
    return val;
}

extern "C" __global__ void matvec_bf16_v4(
    const unsigned short* __restrict__ weight_bf16, // [out_dim * in_dim] bf16 bits
    const float*          __restrict__ x,           // [in_dim] f32
    float*                __restrict__ out,          // [out_dim] f32
    unsigned int out_dim,
    unsigned int in_dim)
{
    unsigned int r0 = blockIdx.x * NR_BF16;  // first output row for this block
    unsigned int warp_id = threadIdx.x / NW_BF16;
    unsigned int lane    = threadIdx.x % NW_BF16;

    unsigned int nvec = in_dim >> 3;  // # of 8-element uint4 chunks (in_dim%8==0)

    float sumf[NR_BF16];
    #pragma unroll
    for (int r = 0; r < NR_BF16; r++) sumf[r] = 0.0f;

    // Each thread strides over 8-element chunks; stride = TPB_BF16 chunks.
    for (unsigned int c = threadIdx.x; c < nvec; c += TPB_BF16) {
        unsigned int base = c << 3;  // element index (c * 8)

        // 8 F32 x-values via 2 x float4 (base*4 is 32-byte aligned).
        const float4* x4 = (const float4*)(x + base);
        float4 xa = x4[0];
        float4 xb = x4[1];

        #pragma unroll
        for (int row = 0; row < NR_BF16; row++) {
            if (r0 + row >= out_dim) break;
            const unsigned short* row_ptr =
                weight_bf16 + (unsigned long long)(r0 + row) * in_dim;
            // uint4 = 16 bytes = 8 BF16. 16-byte aligned: row start is 16B
            // aligned when in_dim%8==0, and base is a multiple of 8.
            uint4 w = *(const uint4*)(row_ptr + base);
            // Unpack 8 bf16 from 4 u32 words (little-endian: low half first),
            // upcast each to F32 losslessly (top-16-bits bit-cast).
            float w0 = bf16_bits_to_f32((unsigned short)(w.x & 0xffffu));
            float w1 = bf16_bits_to_f32((unsigned short)(w.x >> 16));
            float w2 = bf16_bits_to_f32((unsigned short)(w.y & 0xffffu));
            float w3 = bf16_bits_to_f32((unsigned short)(w.y >> 16));
            float w4 = bf16_bits_to_f32((unsigned short)(w.z & 0xffffu));
            float w5 = bf16_bits_to_f32((unsigned short)(w.z >> 16));
            float w6 = bf16_bits_to_f32((unsigned short)(w.w & 0xffffu));
            float w7 = bf16_bits_to_f32((unsigned short)(w.w >> 16));
            sumf[row] += w0 * xa.x + w1 * xa.y + w2 * xa.z + w3 * xa.w
                       + w4 * xb.x + w5 * xb.y + w6 * xb.z + w7 * xb.w;
        }
    }

    // Cross-warp reduction via shared memory (NR rows x NW slots).
    __shared__ float shmem[NR_BF16 * NW_BF16];
    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < NR_BF16; r++) shmem[r * NW_BF16 + lane] = 0.0f;
    }
    #pragma unroll
    for (int r = 0; r < NR_BF16; r++) sumf[r] = warp_reduce_sum_bf16v4(sumf[r]);
    __syncthreads();
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < NR_BF16; r++) shmem[r * NW_BF16 + warp_id] = sumf[r];
    }
    __syncthreads();
    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < NR_BF16; r++) {
            if (r0 + r < out_dim) {
                float val = (lane < (TPB_BF16 / NW_BF16)) ? shmem[r * NW_BF16 + lane] : 0.0f;
                val = warp_reduce_sum_bf16v4(val);
                if (lane == 0) out[r0 + r] = val;
            }
        }
    }
}

// ============================================================================
// matvec_bf16_v4_nr1: ONE output row per block specialization of
// matvec_bf16_v4 (same uint4 weight loads, same F32 unpack/accumulate, same
// two-level reduction shape). NR=2 amortises each x load across two rows
// but halves the launched grid size; NR=1's extra row-level parallelism
// measured faster on H100 decode shapes.
// Selected via `LUMEN_CUDA_BF16_NR1`.
// Same in_dim % 8 == 0 contract as matvec_bf16_v4. Grid = (out_dim, 1, 1).
// ============================================================================
extern "C" __global__ void matvec_bf16_v4_nr1(
    const unsigned short* __restrict__ weight_bf16, // [out_dim * in_dim] bf16 bits
    const float*          __restrict__ x,           // [in_dim] f32
    float*                __restrict__ out,         // [out_dim] f32
    unsigned int out_dim,
    unsigned int in_dim)
{
    unsigned int row = blockIdx.x;
    if (row >= out_dim) return;
    unsigned int warp_id = threadIdx.x / NW_BF16;
    unsigned int lane    = threadIdx.x % NW_BF16;
    unsigned int nvec = in_dim >> 3;

    const unsigned short* row_ptr = weight_bf16 + (unsigned long long)row * in_dim;
    float sum = 0.0f;
    for (unsigned int c = threadIdx.x; c < nvec; c += TPB_BF16) {
        unsigned int base = c << 3;
        const float4* x4 = (const float4*)(x + base);
        float4 xa = x4[0];
        float4 xb = x4[1];
        uint4 w = *(const uint4*)(row_ptr + base);
        float w0 = bf16_bits_to_f32((unsigned short)(w.x & 0xffffu));
        float w1 = bf16_bits_to_f32((unsigned short)(w.x >> 16));
        float w2 = bf16_bits_to_f32((unsigned short)(w.y & 0xffffu));
        float w3 = bf16_bits_to_f32((unsigned short)(w.y >> 16));
        float w4 = bf16_bits_to_f32((unsigned short)(w.z & 0xffffu));
        float w5 = bf16_bits_to_f32((unsigned short)(w.z >> 16));
        float w6 = bf16_bits_to_f32((unsigned short)(w.w & 0xffffu));
        float w7 = bf16_bits_to_f32((unsigned short)(w.w >> 16));
        sum += w0 * xa.x + w1 * xa.y + w2 * xa.z + w3 * xa.w
             + w4 * xb.x + w5 * xb.y + w6 * xb.z + w7 * xb.w;
    }

    __shared__ float shmem[NW_BF16];
    if (warp_id == 0) shmem[lane] = 0.0f;
    sum = warp_reduce_sum_bf16v4(sum);
    __syncthreads();
    if (lane == 0) shmem[warp_id] = sum;
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane < (TPB_BF16 / NW_BF16)) ? shmem[lane] : 0.0f;
        val = warp_reduce_sum_bf16v4(val);
        if (lane == 0) out[row] = val;
    }
}

// ============================================================================
// fused_glu_gemv_bf16_nr1: fused gate+up+SwiGLU for BF16 dense FFN decode.
// One CTA computes row r of BOTH W_gate and W_up off the shared F32 normed
// activation (each x chunk read once feeds both dot products), then applies
// SwiGLU in-register: out[r] = silu(gate_dot) * up_dot. Replaces the
// separate gate matvec + up matvec + swiglu_inplace sub-sequence.
//
// Per-stream arithmetic is matvec_bf16_v4_nr1's exactly (same chunk order,
// eight-term expression, warp/block reduction); the SiLU is swiglu_inplace's
// exact formula (g / (1 + expf(-g))). The separate path's F32 round-trip of
// gate_dot / up_dot through scratch is lossless, so this kernel's output is
// byte-identical to the separate sub-sequence.
// Same in_dim % 8 == 0 contract. Grid = (inter_dim, 1, 1), block = 128.
// ============================================================================
extern "C" __global__ void fused_glu_gemv_bf16_nr1(
    const unsigned short* __restrict__ w_gate,  // [inter_dim * in_dim] bf16 bits
    const unsigned short* __restrict__ w_up,    // [inter_dim * in_dim] bf16 bits
    const float*          __restrict__ x,       // [in_dim] f32 (normed)
    float*                __restrict__ out,     // [inter_dim] f32 = silu(g)*u
    unsigned int inter_dim,
    unsigned int in_dim)
{
    unsigned int row = blockIdx.x;
    if (row >= inter_dim) return;
    unsigned int warp_id = threadIdx.x / NW_BF16;
    unsigned int lane    = threadIdx.x % NW_BF16;
    unsigned int nvec = in_dim >> 3;

    const unsigned short* gp = w_gate + (unsigned long long)row * in_dim;
    const unsigned short* up = w_up + (unsigned long long)row * in_dim;
    float sg = 0.0f;
    float su = 0.0f;
    for (unsigned int c = threadIdx.x; c < nvec; c += TPB_BF16) {
        unsigned int base = c << 3;
        const float4* x4 = (const float4*)(x + base);
        float4 xa = x4[0];
        float4 xb = x4[1];
        uint4 wg = *(const uint4*)(gp + base);
        float g0 = bf16_bits_to_f32((unsigned short)(wg.x & 0xffffu));
        float g1 = bf16_bits_to_f32((unsigned short)(wg.x >> 16));
        float g2 = bf16_bits_to_f32((unsigned short)(wg.y & 0xffffu));
        float g3 = bf16_bits_to_f32((unsigned short)(wg.y >> 16));
        float g4 = bf16_bits_to_f32((unsigned short)(wg.z & 0xffffu));
        float g5 = bf16_bits_to_f32((unsigned short)(wg.z >> 16));
        float g6 = bf16_bits_to_f32((unsigned short)(wg.w & 0xffffu));
        float g7 = bf16_bits_to_f32((unsigned short)(wg.w >> 16));
        sg += g0 * xa.x + g1 * xa.y + g2 * xa.z + g3 * xa.w
            + g4 * xb.x + g5 * xb.y + g6 * xb.z + g7 * xb.w;
        uint4 wu = *(const uint4*)(up + base);
        float u0 = bf16_bits_to_f32((unsigned short)(wu.x & 0xffffu));
        float u1 = bf16_bits_to_f32((unsigned short)(wu.x >> 16));
        float u2 = bf16_bits_to_f32((unsigned short)(wu.y & 0xffffu));
        float u3 = bf16_bits_to_f32((unsigned short)(wu.y >> 16));
        float u4 = bf16_bits_to_f32((unsigned short)(wu.z & 0xffffu));
        float u5 = bf16_bits_to_f32((unsigned short)(wu.z >> 16));
        float u6 = bf16_bits_to_f32((unsigned short)(wu.w & 0xffffu));
        float u7 = bf16_bits_to_f32((unsigned short)(wu.w >> 16));
        su += u0 * xa.x + u1 * xa.y + u2 * xa.z + u3 * xa.w
            + u4 * xb.x + u5 * xb.y + u6 * xb.z + u7 * xb.w;
    }

    __shared__ float shmem[2 * NW_BF16];
    if (warp_id == 0) {
        shmem[lane] = 0.0f;
        shmem[NW_BF16 + lane] = 0.0f;
    }
    sg = warp_reduce_sum_bf16v4(sg);
    su = warp_reduce_sum_bf16v4(su);
    __syncthreads();
    if (lane == 0) {
        shmem[warp_id] = sg;
        shmem[NW_BF16 + warp_id] = su;
    }
    __syncthreads();
    if (warp_id == 0) {
        float g = (lane < (TPB_BF16 / NW_BF16)) ? shmem[lane] : 0.0f;
        float u = (lane < (TPB_BF16 / NW_BF16)) ? shmem[NW_BF16 + lane] : 0.0f;
        g = warp_reduce_sum_bf16v4(g);
        u = warp_reduce_sum_bf16v4(u);
        if (lane == 0) {
            float silu_g = g / (1.0f + expf(-g));
            out[row] = silu_g * u;
        }
    }
}

// ============================================================================
// matvec_bf16_v4_nr1_residual: residual variant of matvec_bf16_v4_nr1.
// out[row] = residual[row] + dot(W_bf16[row, :], x). Serves the full-attention
// `wo` decode projection (`LUMEN_CUDA_BF16_WO_NR1`), replacing the cuBLAS
// chain (dtod residual copy + F32->BF16 activation conversion + GemmEx
// beta=1) with one launch reading the F32 activation directly. Numerics:
// lossless BF16->F32 weight upcast with F32 activation and accumulation,
// avoiding the GemmEx route's BF16 activation-input downcast; the differing
// reduction order means output is not guaranteed byte-identical to it.
// Same in_dim % 8 == 0 contract. Grid = (out_dim, 1, 1).
// ============================================================================
extern "C" __global__ void matvec_bf16_v4_nr1_residual(
    const unsigned short* __restrict__ weight_bf16, // [out_dim * in_dim] bf16 bits
    const float*          __restrict__ x,           // [in_dim] f32
    const float*          __restrict__ residual,    // [out_dim] f32
    float*                __restrict__ out,         // [out_dim] f32
    unsigned int out_dim,
    unsigned int in_dim)
{
    unsigned int row = blockIdx.x;
    if (row >= out_dim) return;
    unsigned int warp_id = threadIdx.x / NW_BF16;
    unsigned int lane    = threadIdx.x % NW_BF16;
    unsigned int nvec = in_dim >> 3;

    const unsigned short* row_ptr = weight_bf16 + (unsigned long long)row * in_dim;
    float sum = 0.0f;
    for (unsigned int c = threadIdx.x; c < nvec; c += TPB_BF16) {
        unsigned int base = c << 3;
        const float4* x4 = (const float4*)(x + base);
        float4 xa = x4[0];
        float4 xb = x4[1];
        uint4 w = *(const uint4*)(row_ptr + base);
        float w0 = bf16_bits_to_f32((unsigned short)(w.x & 0xffffu));
        float w1 = bf16_bits_to_f32((unsigned short)(w.x >> 16));
        float w2 = bf16_bits_to_f32((unsigned short)(w.y & 0xffffu));
        float w3 = bf16_bits_to_f32((unsigned short)(w.y >> 16));
        float w4 = bf16_bits_to_f32((unsigned short)(w.z & 0xffffu));
        float w5 = bf16_bits_to_f32((unsigned short)(w.z >> 16));
        float w6 = bf16_bits_to_f32((unsigned short)(w.w & 0xffffu));
        float w7 = bf16_bits_to_f32((unsigned short)(w.w >> 16));
        sum += w0 * xa.x + w1 * xa.y + w2 * xa.z + w3 * xa.w
             + w4 * xb.x + w5 * xb.y + w6 * xb.z + w7 * xb.w;
    }

    __shared__ float shmem[NW_BF16];
    if (warp_id == 0) shmem[lane] = 0.0f;
    sum = warp_reduce_sum_bf16v4(sum);
    __syncthreads();
    if (lane == 0) shmem[warp_id] = sum;
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane < (TPB_BF16 / NW_BF16)) ? shmem[lane] : 0.0f;
        val = warp_reduce_sum_bf16v4(val);
        if (lane == 0) out[row] = residual[row] + val;
    }
}
