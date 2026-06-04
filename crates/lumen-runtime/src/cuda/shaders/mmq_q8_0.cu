// MMQ Q8_0 prefill kernel (Q8_0 weights × INT8-quantized activation → F32
// partial sums × F32 scale).
//
// Standard MMQ-style INT8 x INT8 -> INT32 -> F32-scale math: `dp4a(int8_w,
// int8_a) * (w_scale * a_scale)`. Reused from Lumen's existing
// `matvec_q8_0_dp4a` kernel for a single vector, generalized to batch
// T tokens.
//
// Per-token Q8_1 activation quantization:
//   x_amax = max(|x[t, block_start..+32]|)
//   x_scale = x_amax / 127.0
//   x_q[t, j] = round(x[t, j] / x_scale)   (clamped to [-127, 127])
//
// Per (token, out_row) accumulation:
//   for each Q8_0 K-block of 32 elements:
//     w_word = pack4(weight_int8[r, k..k+4])
//     x_word = pack4(x_q[t, k..k+4])
//     acc += dp4a(w_word, x_word)    (INT32 exact)
//   out[t, r] = sum over k-blocks of (w_scale[r,kb] * x_scale[t,kb] * acc[kb])
//
// MMQ INT32-sum-then-F32-scale differs from HGEMM-F16's dequant-then-F32-GEMM:
//   - HGEMM-F16: F32 -> F16 cast loses precision in activation, then F16*F16
//     multiplies with F32 accumulator. Cross-K rounding accumulates differently.
//   - MMQ:       F32 -> INT8 cast per 32-elem block. INT8 dot products are
//     EXACT in INT32. Per-block scale is applied only ONCE at sum-time.
//   - Net: MMQ keeps INT32-exact intra-block sums; HGEMM has F32-of-F16-product
//     rounding at each accumulate.
//
// This matches's element-precision evidence: qkv_pre_conv drift is
// 5.85e-2 max-abs / 24% rel of elements when Lumen uses HGEMM-F16 vs MMQ.
// F32-fallback (PREFILL_F32=1 Phase 1) did NOT close the drift
// because the F32 path still does sum-then-multiply-then-sum F32-accumulator
// (the dequant-then-F32-GEMM order, not the INT32-sum then F32-scale order).
//
// Architecture: one CUDA block per (token, NR=2 output rows). 128 threads / 4
// warps stride over K-blocks. Each thread per K-block:
//   1. Load 32 x[t, k..+32] and compute amax (warp reduction not needed; thread
//      handles whole block).
//   2. Quantize x to int8 with per-block scale.
//   3. For each output row in NR:
//      - Load weight scale, 32 weight int8s as 8 packed int32 words.
//      - 8 dp4a calls -> INT32 acc.
//      - sumf[row] += w_scale * x_scale * (float)acc.
//   4. Warp-reduce sumf[NR] within block.
//   5. Write out[t, r0..r0+NR].
//
// Grid:  (ceil(out_dim / NR), batch, 1)  -- one block per (NR rows, token)
// Block: (128, 1, 1)
//
// in_dim must be a multiple of Q8_0_BLOCK_SIZE (32).
//
// NVRTC-compatible: no system includes, extern "C" linkage.

#define MMQ_NR              2     // rows per CUDA block
#define MMQ_WARP_SIZE       32
#define MMQ_THREADS_PER_BLOCK 128 // 4 warps
#define MMQ_Q8_0_BLOCK_SIZE 32
#define MMQ_Q8_0_BYTES      34    // 2 bytes f16 scale + 32 bytes int8 data

// Hardware f16->f32 conversion via PTX (single instruction on SM 53+).
__device__ __forceinline__ float mmq_f16_bits_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

// Warp-level reduction: sum all lanes in a warp using butterfly shuffle.
__device__ __forceinline__ float mmq_warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

// Pack 4 signed bytes into one int32 for dp4a.
__device__ __forceinline__ int mmq_pack_i8x4(int a, int b, int c, int d) {
    return (a & 0xFF) | ((b & 0xFF) << 8) | ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
}

// dp4a inline-PTX wrapper (matches __dp4a semantics).
//
// `dp4a.s32.s32 d, a, b, c` interprets `a` and `b` as four signed 8-bit
// integers each, computes the dot product (sum of products), and adds `c`
// (signed 32-bit accumulator). This avoids the header-dependent `__dp4a`
// intrinsic that NVRTC fails to resolve in this build environment.
//
// Requires Pascal (SM 6.1+); A100 is SM 8.0.
__device__ __forceinline__ int mmq_dp4a_s32(int a, int b, int c) {
    int d;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// MMQ-style Q8_0 batched matmul: out[t, r] = sum_k(dequant(w[r,k]) * x[t,k]).
//
// Activation x is quantized to Q8_1 per 32-element block PER TOKEN on the fly.
//
// in_dim must be a multiple of 32.
extern "C" __global__ void mmq_q8_0_batched(
    const char* __restrict__ weight_q8, // [out_dim * nb * 34] raw Q8_0 bytes
    const float* __restrict__ x,        // [batch, in_dim]
    float* __restrict__ out,            // [batch, out_dim]
    unsigned int out_dim,
    unsigned int in_dim,
    unsigned int batch)
{
    unsigned int r0 = blockIdx.x * MMQ_NR;
    unsigned int tok = blockIdx.y;
    if (tok >= batch) return;

    unsigned int warp_id = threadIdx.x / MMQ_WARP_SIZE;
    unsigned int lane    = threadIdx.x % MMQ_WARP_SIZE;

    unsigned int nb = in_dim >> 5; // number of Q8_0 blocks per row
    unsigned long long row_bytes = (unsigned long long)nb * MMQ_Q8_0_BYTES;

    const float* x_row = x + (unsigned long long)tok * in_dim;

    float sumf[MMQ_NR];
    #pragma unroll
    for (int r = 0; r < MMQ_NR; r++) sumf[r] = 0.0f;

    // Main loop: each thread handles one K-block per iteration, striding by
    // THREADS_PER_BLOCK over the K dimension.
    for (unsigned int ib = threadIdx.x; ib < nb; ib += MMQ_THREADS_PER_BLOCK) {
        unsigned int x_base = ib * MMQ_Q8_0_BLOCK_SIZE;

        // Load 32 x-values, compute amax via float4 vectorized loads.
        float xv[32];
        float amax = 0.0f;

        const float4* x4 = (const float4*)(x_row + x_base);
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            float4 v = x4[k];
            xv[k * 4 + 0] = v.x;
            xv[k * 4 + 1] = v.y;
            xv[k * 4 + 2] = v.z;
            xv[k * 4 + 3] = v.w;
            float a0 = v.x < 0.0f ? -v.x : v.x;
            float a1 = v.y < 0.0f ? -v.y : v.y;
            float a2 = v.z < 0.0f ? -v.z : v.z;
            float a3 = v.w < 0.0f ? -v.w : v.w;
            if (a0 > amax) amax = a0;
            if (a1 > amax) amax = a1;
            if (a2 > amax) amax = a2;
            if (a3 > amax) amax = a3;
        }

        // x_scale: amax/127, x_scale_inv: 127/amax (for quantization).
        float x_scale = amax / 127.0f;
        float x_scale_inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;

        // Quantize x to int8 and pack into int32 (4 per int).
        int x_packed[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int q0 = (int)__float2int_rn(xv[k * 4 + 0] * x_scale_inv);
            int q1 = (int)__float2int_rn(xv[k * 4 + 1] * x_scale_inv);
            int q2 = (int)__float2int_rn(xv[k * 4 + 2] * x_scale_inv);
            int q3 = (int)__float2int_rn(xv[k * 4 + 3] * x_scale_inv);
            x_packed[k] = mmq_pack_i8x4(q0, q1, q2, q3);
        }

        // Process NR output rows with the same quantized x-values.
        #pragma unroll
        for (int row = 0; row < MMQ_NR; row++) {
            if (r0 + row >= out_dim) break;

            const char* bp = weight_q8
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * MMQ_Q8_0_BYTES;

            // Read f16 weight scale.
            unsigned short scale_bits = (unsigned short)(unsigned char)bp[0]
                                      | ((unsigned short)(unsigned char)bp[1] << 8);
            float w_scale = mmq_f16_bits_to_f32(scale_bits);

            // Load 32 int8 weight values as 8 packed int32 words via 16-bit aligned loads.
            const unsigned short* w16 = (const unsigned short*)(bp + 2);

            int acc = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                int w_word = (int)w16[k * 2] | ((int)w16[k * 2 + 1] << 16);
                acc = mmq_dp4a_s32(w_word, x_packed[k], acc);
            }

            // Combined scale: w_scale * x_scale * int_dot_product.
            sumf[row] += w_scale * x_scale * (float)acc;
        }
    }

    // Cross-warp reduction via shared memory (NR rows x WARP_SIZE slots).
    __shared__ float shmem[MMQ_NR * MMQ_WARP_SIZE];

    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < MMQ_NR; r++) {
            shmem[r * MMQ_WARP_SIZE + lane] = 0.0f;
        }
    }

    // Intra-warp reduction.
    #pragma unroll
    for (int r = 0; r < MMQ_NR; r++) {
        sumf[r] = mmq_warp_reduce_sum(sumf[r]);
    }

    __syncthreads();

    // Lane 0 of each warp writes its partial sum.
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < MMQ_NR; r++) {
            shmem[r * MMQ_WARP_SIZE + warp_id] = sumf[r];
        }
    }

    __syncthreads();

    // Warp 0 does the final reduction across warps.
    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < MMQ_NR; r++) {
            if (r0 + r < out_dim) {
                float val = (lane < (MMQ_THREADS_PER_BLOCK / MMQ_WARP_SIZE))
                    ? shmem[r * MMQ_WARP_SIZE + lane]
                    : 0.0f;
                val = mmq_warp_reduce_sum(val);
                if (lane == 0) {
                    out[(unsigned long long)tok * out_dim + r0 + r] = val;
                }
            }
        }
    }
}

// MMQ-style Q8_0 batched matmul WITH RESIDUAL ADD.
//
// Computes out[t, r] = residual[t, r] + sum_k(dequant(w[r,k]) * x[t,k]).
// Identical math to `mmq_q8_0_batched` except the per-(token, row) write
// path adds the residual element from the input `residual` buffer instead of
// overwriting `out`. This matches `launch_gemm_residual`'s HGEMM beta=1.0
// semantics (output = residual + W @ x) and lets the MMQ inner-loop
// math close the `linear_attn_out` drift at the GDN-block exit projection.
//
// Note: `out` and `residual` MAY alias; this kernel reads `residual[t, r]`
// only at the single store site (lane==0 in warp 0 of NR), AFTER the warp
// reductions are complete and __syncthreads has run. Aliasing is safe because
// the kernel never re-reads `out` once it's overwritten. The Lumen caller
// always passes `output = attn_proj` and `residual = x` which are distinct
// buffers, so aliasing is never exercised in practice.
extern "C" __global__ void mmq_q8_0_batched_residual(
    const char* __restrict__ weight_q8,    // [out_dim * nb * 34] raw Q8_0 bytes
    const float* __restrict__ x,           // [batch, in_dim]
    const float* __restrict__ residual,    // [batch, out_dim] additive residual
    float* __restrict__ out,               // [batch, out_dim] = residual + W @ x
    unsigned int out_dim,
    unsigned int in_dim,
    unsigned int batch)
{
    unsigned int r0 = blockIdx.x * MMQ_NR;
    unsigned int tok = blockIdx.y;
    if (tok >= batch) return;

    unsigned int warp_id = threadIdx.x / MMQ_WARP_SIZE;
    unsigned int lane    = threadIdx.x % MMQ_WARP_SIZE;

    unsigned int nb = in_dim >> 5; // number of Q8_0 blocks per row
    unsigned long long row_bytes = (unsigned long long)nb * MMQ_Q8_0_BYTES;

    const float* x_row = x + (unsigned long long)tok * in_dim;

    float sumf[MMQ_NR];
    #pragma unroll
    for (int r = 0; r < MMQ_NR; r++) sumf[r] = 0.0f;

    // Main loop: each thread handles one K-block per iteration, striding by
    // THREADS_PER_BLOCK over the K dimension. Identical to mmq_q8_0_batched.
    for (unsigned int ib = threadIdx.x; ib < nb; ib += MMQ_THREADS_PER_BLOCK) {
        unsigned int x_base = ib * MMQ_Q8_0_BLOCK_SIZE;

        // Load 32 x-values, compute amax via float4 vectorized loads.
        float xv[32];
        float amax = 0.0f;

        const float4* x4 = (const float4*)(x_row + x_base);
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            float4 v = x4[k];
            xv[k * 4 + 0] = v.x;
            xv[k * 4 + 1] = v.y;
            xv[k * 4 + 2] = v.z;
            xv[k * 4 + 3] = v.w;
            float a0 = v.x < 0.0f ? -v.x : v.x;
            float a1 = v.y < 0.0f ? -v.y : v.y;
            float a2 = v.z < 0.0f ? -v.z : v.z;
            float a3 = v.w < 0.0f ? -v.w : v.w;
            if (a0 > amax) amax = a0;
            if (a1 > amax) amax = a1;
            if (a2 > amax) amax = a2;
            if (a3 > amax) amax = a3;
        }

        float x_scale = amax / 127.0f;
        float x_scale_inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;

        int x_packed[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int q0 = (int)__float2int_rn(xv[k * 4 + 0] * x_scale_inv);
            int q1 = (int)__float2int_rn(xv[k * 4 + 1] * x_scale_inv);
            int q2 = (int)__float2int_rn(xv[k * 4 + 2] * x_scale_inv);
            int q3 = (int)__float2int_rn(xv[k * 4 + 3] * x_scale_inv);
            x_packed[k] = mmq_pack_i8x4(q0, q1, q2, q3);
        }

        #pragma unroll
        for (int row = 0; row < MMQ_NR; row++) {
            if (r0 + row >= out_dim) break;

            const char* bp = weight_q8
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * MMQ_Q8_0_BYTES;

            unsigned short scale_bits = (unsigned short)(unsigned char)bp[0]
                                      | ((unsigned short)(unsigned char)bp[1] << 8);
            float w_scale = mmq_f16_bits_to_f32(scale_bits);

            const unsigned short* w16 = (const unsigned short*)(bp + 2);

            int acc = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                int w_word = (int)w16[k * 2] | ((int)w16[k * 2 + 1] << 16);
                acc = mmq_dp4a_s32(w_word, x_packed[k], acc);
            }

            sumf[row] += w_scale * x_scale * (float)acc;
        }
    }

    // Cross-warp reduction via shared memory (NR rows x WARP_SIZE slots).
    __shared__ float shmem[MMQ_NR * MMQ_WARP_SIZE];

    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < MMQ_NR; r++) {
            shmem[r * MMQ_WARP_SIZE + lane] = 0.0f;
        }
    }

    #pragma unroll
    for (int r = 0; r < MMQ_NR; r++) {
        sumf[r] = mmq_warp_reduce_sum(sumf[r]);
    }

    __syncthreads();

    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < MMQ_NR; r++) {
            shmem[r * MMQ_WARP_SIZE + warp_id] = sumf[r];
        }
    }

    __syncthreads();

    // Warp 0 does the final reduction across warps.
    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < MMQ_NR; r++) {
            if (r0 + r < out_dim) {
                float val = (lane < (MMQ_THREADS_PER_BLOCK / MMQ_WARP_SIZE))
                    ? shmem[r * MMQ_WARP_SIZE + lane]
                    : 0.0f;
                val = mmq_warp_reduce_sum(val);
                if (lane == 0) {
                    unsigned long long idx = (unsigned long long)tok * out_dim + r0 + r;
                    // fused residual add. Matches HGEMM beta=1.0 semantics.
                    out[idx] = residual[idx] + val;
                }
            }
        }
    }
}
