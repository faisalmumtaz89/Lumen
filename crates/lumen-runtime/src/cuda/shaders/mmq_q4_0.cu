// MMQ Q4_0 prefill kernel (Q4_0 weights × INT8-quantized activation → F32
// partial sums × F32 scale).
//
// q4-specific twin of `mmq_q8_0_batched` (see mmq_q8_0.cu). The Q4_0 default
// prefill path is dequant->F16->cuBLAS HGEMM, which is F16-grade: the F32->F16
// activation cast plus F16*F16 products with cross-K F32 accumulation drift
// from llama.cpp's `mul_mat_q` INT4 numerics. On Qwen3.5-MoE-35B-A3B that drift
// is harmless for dense, but the 256-expert top-K router AMPLIFIES it into
// flipped expert selection and garbled arithmetic ("17 x 23 = 491 x 23 = 391").
//
// This kernel matches llama's MMQ INT4 math instead of raising precision
// (raising precision via generic PREFILL_F32 made q4 WORSE -- rep30):
//   - Activation x is quantized to Q8_1 per 32-element block PER TOKEN on the
//     fly (amax -> int8, exactly as mmq_q8_0_batched does).
//   - Q4_0 weight nibbles are de-interleaved and dp4a'd against the int8
//     activation: INT32-EXACT intra-block dot products.
//   - Per-block scale (w_scale * x_scale) is applied only ONCE at sum-time,
//     with the GGML Q4_0 zero-point -8 correction folded in:
//       dot = w_scale * (x_scale * dp4a_sum - 8 * x_scale * sum(x_quant))
//     where sum(x_quant) is recomputed per block (the activation is quantized
//     in-kernel, so there is no precomputed Q8_1 `s` field as the decode
//     matvec_q4_0_dp4a kernel has -- we form it from the int8 lane sums).
//
// MMQ INT32-sum-then-F32-scale vs HGEMM-F16's dequant-then-F32-GEMM:
//   - HGEMM-F16: F32 -> F16 cast loses activation precision, then F16*F16
//     products with F32 accumulate. Cross-K rounding accumulates.
//   - MMQ:       F32 -> INT8 cast per 32-elem block. INT8 dot products are
//     EXACT in INT32. Per-block scale applied once at sum-time.
//   - Net: MMQ keeps INT32-exact intra-block sums; matches llama mul_mat_q.
//
// Q4_0 block layout (18 bytes per 32 elements):
//   bytes [0..1]: f16 scale (d)
//   bytes [2..17]: 16 bytes of de-interleaved nibbles
//     Elements 0-15:  lo nibbles of bytes 0-15
//     Elements 16-31: hi nibbles of bytes 0-15
//   Dequantized value: d * (nibble - 8)
//
// Architecture: one CUDA block per (token, NR=2 output rows). 128 threads / 4
// warps stride over K-blocks. Each thread per K-block:
//   1. Load 32 x[t, k..+32] and compute amax (thread handles whole block).
//   2. Quantize x to int8 with per-block scale; accumulate x_quant_sum.
//   3. For each output row in NR:
//      - Load weight scale, 16 nibble bytes -> 8 de-interleaved int32 words.
//      - 8 dp4a calls -> INT32 acc (unsigned nibbles * signed int8).
//      - sumf[row] += w_scale * (x_scale * acc - 8 * x_scale * x_quant_sum).
//   4. Cross-warp reduce sumf[NR] within block.
//   5. Write out[t, r0..r0+NR].
//
// Grid:  (ceil(out_dim / NR), batch, 1)  -- one block per (NR rows, token)
// Block: (128, 1, 1)
//
// in_dim must be a multiple of Q4_0_BLOCK_SIZE (32).
//
// NVRTC-compatible: no system includes, extern "C" linkage.

#define MMQ4_NR              2     // rows per CUDA block (match Q8 MMQ)
#define MMQ4_WARP_SIZE       32
#define MMQ4_THREADS_PER_BLOCK 128 // 4 warps
#define MMQ4_Q4_0_BLOCK_SIZE 32
#define MMQ4_Q4_0_BYTES      18    // 2 bytes f16 scale + 16 bytes nibble data

// Hardware f16->f32 conversion via PTX (single instruction on SM 53+).
__device__ __forceinline__ float mmq4_f16_bits_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

// Warp-level reduction: sum all lanes in a warp using butterfly shuffle.
__device__ __forceinline__ float mmq4_warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

// Pack 4 signed bytes into one int32 for dp4a.
__device__ __forceinline__ int mmq4_pack_i8x4(int a, int b, int c, int d) {
    return (a & 0xFF) | ((b & 0xFF) << 8) | ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
}

// dp4a inline-PTX wrapper (matches __dp4a semantics).
//
// `dp4a.s32.s32 d, a, b, c` interprets `a` and `b` as four signed 8-bit
// integers each, computes the dot product, and adds `c`. The `__dp4a`
// intrinsic NVRTC-fails in this build env; the inline opcode loads cleanly on
// compute_80. Requires Pascal (SM 6.1+); A100 is SM 8.0.
__device__ __forceinline__ int mmq4_dp4a_s32(int a, int b, int c) {
    int d;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// Pack 4 consecutive de-interleaved Q4_0 elements into a dp4a-compatible int32.
//
// GGML Q4_0 de-interleaved layout (16 nibble bytes per 32 elements):
//   Elements 0-15:  lo nibbles of bytes 0-15
//   Elements 16-31: hi nibbles of bytes 0-15
// k = 0..7 selects which group of 4 consecutive elements:
//   k<4: lo nibbles of bytes k*4..k*4+3       (elements k*4..k*4+3)
//   k>=4: hi nibbles of bytes (k-4)*4..(k-4)*4+3 (elements (k-4)*4+16..+19)
// Returns UNSIGNED nibbles (0-15) packed as 4 bytes. The -8 zero-point is
// corrected in the accumulation formula.
__device__ __forceinline__ int mmq4_pack_deinterleaved(const unsigned char* qs, int k) {
    if (k < 4) {
        unsigned int b = k * 4;
        unsigned int packed = ((unsigned int)(qs[b]   & 0x0Fu))
                            | ((unsigned int)(qs[b+1] & 0x0Fu) << 8)
                            | ((unsigned int)(qs[b+2] & 0x0Fu) << 16)
                            | ((unsigned int)(qs[b+3] & 0x0Fu) << 24);
        return (int)packed;
    } else {
        unsigned int b = (k - 4) * 4;
        unsigned int packed = ((unsigned int)((qs[b]   >> 4) & 0x0Fu))
                            | ((unsigned int)((qs[b+1] >> 4) & 0x0Fu) << 8)
                            | ((unsigned int)((qs[b+2] >> 4) & 0x0Fu) << 16)
                            | ((unsigned int)((qs[b+3] >> 4) & 0x0Fu) << 24);
        return (int)packed;
    }
}

// Sum the four signed int8 lanes of a packed int32 (for the -8 correction).
__device__ __forceinline__ int mmq4_sum_i8x4(int packed) {
    // dp4a against the all-ones vector (0x01010101) sums the four signed bytes.
    return mmq4_dp4a_s32(packed, 0x01010101, 0);
}

// MMQ-style Q4_0 batched matmul: out[t, r] = sum_k(dequant(w[r,k]) * x[t,k]).
//
// Activation x is quantized to Q8_1 per 32-element block PER TOKEN on the fly.
// in_dim must be a multiple of 32.
extern "C" __global__ void mmq_q4_0_batched(
    const char* __restrict__ weight_q4, // [out_dim * nb * 18] raw Q4_0 bytes
    const float* __restrict__ x,        // [batch, in_dim]
    float* __restrict__ out,            // [batch, out_dim]
    unsigned int out_dim,
    unsigned int in_dim,
    unsigned int batch)
{
    unsigned int r0 = blockIdx.x * MMQ4_NR;
    unsigned int tok = blockIdx.y;
    if (tok >= batch) return;

    unsigned int warp_id = threadIdx.x / MMQ4_WARP_SIZE;
    unsigned int lane    = threadIdx.x % MMQ4_WARP_SIZE;

    unsigned int nb = in_dim >> 5; // number of Q4_0 blocks per row
    unsigned long long row_bytes = (unsigned long long)nb * MMQ4_Q4_0_BYTES;

    const float* x_row = x + (unsigned long long)tok * in_dim;

    float sumf[MMQ4_NR];
    #pragma unroll
    for (int r = 0; r < MMQ4_NR; r++) sumf[r] = 0.0f;

    // Main loop: each thread handles one K-block per iteration, striding by
    // THREADS_PER_BLOCK over the K dimension.
    for (unsigned int ib = threadIdx.x; ib < nb; ib += MMQ4_THREADS_PER_BLOCK) {
        unsigned int x_base = ib * MMQ4_Q4_0_BLOCK_SIZE;

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

        // Quantize x to int8 and pack into int32 (4 per int). Track the int8
        // lane sum for the Q4_0 -8 zero-point correction.
        int x_packed[8];
        int x_quant_sum = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int q0 = (int)__float2int_rn(xv[k * 4 + 0] * x_scale_inv);
            int q1 = (int)__float2int_rn(xv[k * 4 + 1] * x_scale_inv);
            int q2 = (int)__float2int_rn(xv[k * 4 + 2] * x_scale_inv);
            int q3 = (int)__float2int_rn(xv[k * 4 + 3] * x_scale_inv);
            x_packed[k] = mmq4_pack_i8x4(q0, q1, q2, q3);
            x_quant_sum += q0 + q1 + q2 + q3;
        }

        // Process NR output rows with the same quantized x-values.
        #pragma unroll
        for (int row = 0; row < MMQ4_NR; row++) {
            if (r0 + row >= out_dim) break;

            const char* bp = weight_q4
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * MMQ4_Q4_0_BYTES;

            // Read f16 weight scale (bytes 0-1).
            unsigned short scale_bits = (unsigned short)(unsigned char)bp[0]
                                      | ((unsigned short)(unsigned char)bp[1] << 8);
            float w_scale = mmq4_f16_bits_to_f32(scale_bits);

            // 16 nibble bytes start at bp + 2.
            const unsigned char* qs = (const unsigned char*)(bp + 2);

            int acc = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                int w_word = mmq4_pack_deinterleaved(qs, k);
                acc = mmq4_dp4a_s32(w_word, x_packed[k], acc);
            }

            // Q4_0 dequant with -8 zero-point:
            //   sum((nibble-8) * x_quant) * w_scale * x_scale
            //   = w_scale * (x_scale * dp4a_sum - 8 * x_scale * sum(x_quant))
            sumf[row] += w_scale * x_scale * ((float)acc - 8.0f * (float)x_quant_sum);
        }
    }

    // Cross-warp reduction via shared memory (NR rows x WARP_SIZE slots).
    __shared__ float shmem[MMQ4_NR * MMQ4_WARP_SIZE];

    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < MMQ4_NR; r++) {
            shmem[r * MMQ4_WARP_SIZE + lane] = 0.0f;
        }
    }

    // Intra-warp reduction.
    #pragma unroll
    for (int r = 0; r < MMQ4_NR; r++) {
        sumf[r] = mmq4_warp_reduce_sum(sumf[r]);
    }

    __syncthreads();

    // Lane 0 of each warp writes its partial sum.
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < MMQ4_NR; r++) {
            shmem[r * MMQ4_WARP_SIZE + warp_id] = sumf[r];
        }
    }

    __syncthreads();

    // Warp 0 does the final reduction across warps.
    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < MMQ4_NR; r++) {
            if (r0 + r < out_dim) {
                float val = (lane < (MMQ4_THREADS_PER_BLOCK / MMQ4_WARP_SIZE))
                    ? shmem[r * MMQ4_WARP_SIZE + lane]
                    : 0.0f;
                val = mmq4_warp_reduce_sum(val);
                if (lane == 0) {
                    out[(unsigned long long)tok * out_dim + r0 + r] = val;
                }
            }
        }
    }
}

// MMQ-style Q4_0 batched matmul WITH RESIDUAL ADD.
//
// out[t, r] = residual[t, r] + sum_k(dequant(w[r,k]) * x[t,k]). Identical math
// to `mmq_q4_0_batched` except the per-(token, row) write adds residual.
// Matches `launch_gemm_residual`'s HGEMM beta=1.0 semantics. `out` and
// `residual` MAY alias safely (residual read only at the single store site
// after reductions). Used at the GDN-block exit projection (linear_attn_out)
// and FFN-down, exactly mirroring mmq_q8_0_batched_residual.
extern "C" __global__ void mmq_q4_0_batched_residual(
    const char* __restrict__ weight_q4,    // [out_dim * nb * 18] raw Q4_0 bytes
    const float* __restrict__ x,           // [batch, in_dim]
    const float* __restrict__ residual,    // [batch, out_dim] additive residual
    float* __restrict__ out,               // [batch, out_dim] = residual + W @ x
    unsigned int out_dim,
    unsigned int in_dim,
    unsigned int batch)
{
    unsigned int r0 = blockIdx.x * MMQ4_NR;
    unsigned int tok = blockIdx.y;
    if (tok >= batch) return;

    unsigned int warp_id = threadIdx.x / MMQ4_WARP_SIZE;
    unsigned int lane    = threadIdx.x % MMQ4_WARP_SIZE;

    unsigned int nb = in_dim >> 5;
    unsigned long long row_bytes = (unsigned long long)nb * MMQ4_Q4_0_BYTES;

    const float* x_row = x + (unsigned long long)tok * in_dim;

    float sumf[MMQ4_NR];
    #pragma unroll
    for (int r = 0; r < MMQ4_NR; r++) sumf[r] = 0.0f;

    for (unsigned int ib = threadIdx.x; ib < nb; ib += MMQ4_THREADS_PER_BLOCK) {
        unsigned int x_base = ib * MMQ4_Q4_0_BLOCK_SIZE;

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
        int x_quant_sum = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int q0 = (int)__float2int_rn(xv[k * 4 + 0] * x_scale_inv);
            int q1 = (int)__float2int_rn(xv[k * 4 + 1] * x_scale_inv);
            int q2 = (int)__float2int_rn(xv[k * 4 + 2] * x_scale_inv);
            int q3 = (int)__float2int_rn(xv[k * 4 + 3] * x_scale_inv);
            x_packed[k] = mmq4_pack_i8x4(q0, q1, q2, q3);
            x_quant_sum += q0 + q1 + q2 + q3;
        }

        #pragma unroll
        for (int row = 0; row < MMQ4_NR; row++) {
            if (r0 + row >= out_dim) break;

            const char* bp = weight_q4
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * MMQ4_Q4_0_BYTES;

            unsigned short scale_bits = (unsigned short)(unsigned char)bp[0]
                                      | ((unsigned short)(unsigned char)bp[1] << 8);
            float w_scale = mmq4_f16_bits_to_f32(scale_bits);

            const unsigned char* qs = (const unsigned char*)(bp + 2);

            int acc = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                int w_word = mmq4_pack_deinterleaved(qs, k);
                acc = mmq4_dp4a_s32(w_word, x_packed[k], acc);
            }

            sumf[row] += w_scale * x_scale * ((float)acc - 8.0f * (float)x_quant_sum);
        }
    }

    __shared__ float shmem[MMQ4_NR * MMQ4_WARP_SIZE];

    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < MMQ4_NR; r++) {
            shmem[r * MMQ4_WARP_SIZE + lane] = 0.0f;
        }
    }

    #pragma unroll
    for (int r = 0; r < MMQ4_NR; r++) {
        sumf[r] = mmq4_warp_reduce_sum(sumf[r]);
    }

    __syncthreads();

    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < MMQ4_NR; r++) {
            shmem[r * MMQ4_WARP_SIZE + warp_id] = sumf[r];
        }
    }

    __syncthreads();

    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < MMQ4_NR; r++) {
            if (r0 + r < out_dim) {
                float val = (lane < (MMQ4_THREADS_PER_BLOCK / MMQ4_WARP_SIZE))
                    ? shmem[r * MMQ4_WARP_SIZE + lane]
                    : 0.0f;
                val = mmq4_warp_reduce_sum(val);
                if (lane == 0) {
                    unsigned long long idx = (unsigned long long)tok * out_dim + r0 + r;
                    out[idx] = residual[idx] + val;
                }
            }
        }
    }
}
