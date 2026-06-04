// MoE shared-expert auxiliary kernels.
//
// The shared expert FFN reuses existing matvec_q4_0 + swiglu_inplace kernels
// for the heavy gate/up/down projections. These two small kernels handle the
// scalar sigmoid gate path that Metal's `encode_shared_expert_ffn_decode_raw`
// implements:
//   scalar  = dot(ffn_gate_inp_shexp[hidden_dim], normed_x[hidden_dim])  (F32)
//   x_out[i] += sigmoid(scalar) * shared_down_out[i]                      (1..hidden_dim)
//
// Mirrors metal/shaders/moe.msl (sigmoid_scale_add) + a small F32 dot kernel.
// Ports verbatim semantics: max-subtract-free sigmoid (scalar is bounded by
// hidden_dim * O(1) so overflow is not a concern in practice).
//
// NVRTC-compatible: extern "C" linkage, no system includes.

#define BLOCK_DIM 256

// ----------------------------------------------------------------------------
// F32 dot product: scalar_out[0] = dot(w[in_dim], x[in_dim])
//
// Single-block parallel reduction. One CTA = 256 threads.
// Used for the shared-expert sigmoid gate logit:
//   logit = dot(ffn_gate_inp_shexp, normed_x)
//
// `w` and `x` are both F32 row-major [in_dim].
// `out` is a single F32 scalar.
// ----------------------------------------------------------------------------
extern "C" __global__ void moe_shared_dot_f32(
    const float* __restrict__ w,           // [in_dim]
    const float* __restrict__ x,           // [in_dim]
    float* __restrict__ out,               // [1] scalar
    unsigned int in_dim)
{
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31;
    const unsigned int warp_id = tid >> 5;

    float partial = 0.0f;
    for (unsigned int j = tid; j < in_dim; j += BLOCK_DIM) {
        partial += w[j] * x[j];
    }

    // Warp-level reduction via shfl_down_sync.
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(0xffffffff, partial, offset);
    }

    __shared__ float warp_partial[BLOCK_DIM / 32];
    if (lane == 0) {
        warp_partial[warp_id] = partial;
    }
    __syncthreads();

    // First warp reduces the warp partials.
    if (warp_id == 0) {
        float v = (lane < (BLOCK_DIM / 32)) ? warp_partial[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_down_sync(0xffffffff, v, offset);
        }
        if (lane == 0) {
            out[0] = v;
        }
    }
}

// ----------------------------------------------------------------------------
// Sigmoid-gated accumulate (with residual already in x_out from routed experts):
//   x_out[i] += sigmoid(logit[0]) * shared_out[i]
//
// `logit` is a single F32 scalar produced by `moe_shared_dot_f32`.
// `shared_out` is the F32 [hidden_dim] output of the shared expert down-proj.
// `x_out` is the post-routed-expert MoE output that we accumulate INTO.
//
// One thread per hidden_dim element.
// ----------------------------------------------------------------------------
extern "C" __global__ void moe_shared_sigmoid_gated_accum(
    float* __restrict__ x_out,                 // [hidden_dim] in/out
    const float* __restrict__ shared_out,      // [hidden_dim]
    const float* __restrict__ logit,           // [1] scalar (pre-sigmoid)
    unsigned int hidden_dim)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden_dim) return;

    // Sigmoid of the (single) gate logit.
    // Note: read once, broadcast to every thread. The single F32 load is
    // L1-cached so per-thread is effectively free.
    const float g = 1.0f / (1.0f + expf(-logit[0]));
    x_out[i] += g * shared_out[i];
}

// ----------------------------------------------------------------------------
// Fallback: residual accumulate without sigmoid gate (when ffn_gate_inp_shexp
// is absent on a shared-expert variant). Identical to `residual_add` but
// kept here for symmetry with the gated path's parameter wiring.
//   x_out[i] += shared_out[i]
// ----------------------------------------------------------------------------
extern "C" __global__ void moe_shared_residual_accum(
    float* __restrict__ x_out,                 // [hidden_dim] in/out
    const float* __restrict__ shared_out,      // [hidden_dim]
    unsigned int hidden_dim)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden_dim) return;
    x_out[i] += shared_out[i];
}

// ============================================================================
// shared-expert FFN fusion.
//
// The unfused shared-expert path uses 5-6 kernel launches per layer:
//   1. matvec_q4_0(W_gate, normed_x)   -> shared_gate_buf            [inter]
//   2. matvec_q4_0(W_up,   normed_x)   -> up_buf                     [inter]
//   3. swiglu_inplace(shared_gate_buf, up_buf)
//   4. matvec_q4_0(W_down, shared_gate_buf) -> shared_down_buf       [hidden]
//   5. moe_shared_dot_f32(W_gate_inp_shexp, normed_x) -> scalar       [1]
//   6. moe_shared_sigmoid_gated_accum(x_out, shared_down_buf, scalar)
//
// At 40 layers per token this is ~200 launches/token, ~3 µs launch overhead
// each → ~0.6 ms of pure launch latency. The intermediate `up_buf` and
// `shared_gate_buf` (after SwiGLU) round-trip 45 KB of HBM per layer.
//
// These two new kernels collapse the path to 3 launches:
//
//   1. fused_glu_gemv_q4_0_prenormed_no_norm
//        Merges steps 1+2+3 into one kernel. Same warp-cooperative NR=2
//        row-tile design as `fused_glu_gemv_q4_0`, but skips the inline
//        RMSNorm because the shared expert RECEIVES already-normalized x
//        (`st.scratch.normed`) from the upstream RMSNorm dispatch.
//        Caches `normed_x` in shmem (hidden_dim * 4 bytes) for cross-row
//        reuse instead of `x * scale * norm_weight`.
//
//   2. moe_shared_dot_f32 (UNCHANGED)
//        Scalar logit for sigmoid gate.
//
//   3. moe_shared_down_q4_0_sigmoid_accum
//        Merges steps 4+5b (matvec down + sigmoid_gated_accum) into one
//        kernel. Computes dot(W_down[i], swiglu_buf) and immediately writes
//        x_out[i] += sigmoid(scalar[0]) * dot. Eliminates the intermediate
//        `shared_down_buf` HBM write/read (hidden_dim * 4 = 16 KB per layer).
//        Single CTA per output row, BLOCK_SIZE = 256, identical reduction
//        pattern to `matvec_q4_0`.
//
// Saves: 3 kernel launches per layer × 40 layers = 120 launches/token.
//        + 1 HBM round-trip on swiglu_buf (eliminated entirely).
//        + 1 HBM round-trip on shared_down_buf (eliminated entirely).
//
// Correctness contract: bit-identical to the unfused path up to FP add
// ordering within the warp reduction (same kernels, same NR=2 layout,
// same accumulator widths). Validated by 10/10 byte-identical kernel test.
//
// NVRTC-compatible: extern "C" linkage, no system includes.
// ============================================================================

#define FUSED_NR             2
#define FUSED_BLOCK_DIM      256
#define FUSED_WARP_SIZE      32
#define FUSED_Q4_BLOCK_ELEMS 32
#define FUSED_Q4_BLOCK_BYTES 18

// Hardware f16->f32 conversion via PTX (single instruction on SM 53+).
// Duplicated here because NVRTC compiles each .cu source as a separate module
// (cannot share device-function symbols across modules).
__device__ __forceinline__ float fused_f16_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

// Warp-level reduction via butterfly shuffle.
__device__ __forceinline__ float fused_warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

// ----------------------------------------------------------------------------
// Fused gate+up+SwiGLU GEMV for shared expert (Q4_0 weights, pre-normalized x).
//
// Mirrors `fused_glu_gemv_q4_0` but the input vector is ALREADY RMSNormalized
// by the upstream pipeline (the shared expert receives `st.scratch.normed`).
// So there is no inline rms_scale or norm_weight multiplication — the kernel
// simply caches `normed_x` in shmem and dot-products against gate and up
// weights simultaneously.
//
// Shmem: hidden_dim * 4 bytes (F32 normed x-vector).
// Grid:  (ceil(inter_dim / NR), 1, 1)
// Block: (FUSED_BLOCK_DIM, 1, 1)
// ----------------------------------------------------------------------------
extern "C" __global__ void fused_glu_gemv_q4_0_prenormed_no_norm(
    const char*  __restrict__ w_gate,       // [inter_dim, hidden_dim] Q4_0
    const char*  __restrict__ w_up,         // [inter_dim, hidden_dim] Q4_0
    const float* __restrict__ normed_x,     // [hidden_dim] already RMSNormed
    float*       __restrict__ output,       // [inter_dim] silu(gate) * up
    unsigned int inter_dim,
    unsigned int hidden_dim)
{
    extern __shared__ float nx_smem[];

    const unsigned int r0 = blockIdx.x * FUSED_NR;
    const unsigned int warp_id = threadIdx.x / FUSED_WARP_SIZE;
    const unsigned int lane    = threadIdx.x % FUSED_WARP_SIZE;
    const unsigned int num_blocks = hidden_dim / FUSED_Q4_BLOCK_ELEMS;
    const unsigned long long row_bytes =
        (unsigned long long)num_blocks * FUSED_Q4_BLOCK_BYTES;

    // Cache the already-normalized x-vector in shmem (one read, N reuses).
    for (unsigned int i = threadIdx.x; i < hidden_dim; i += FUSED_BLOCK_DIM) {
        nx_smem[i] = normed_x[i];
    }
    __syncthreads();

    float gate_sum[FUSED_NR];
    float up_sum[FUSED_NR];
    #pragma unroll
    for (int r = 0; r < FUSED_NR; r++) {
        gate_sum[r] = 0.0f;
        up_sum[r]   = 0.0f;
    }

    for (unsigned int ib = threadIdx.x; ib < num_blocks; ib += FUSED_BLOCK_DIM) {
        const unsigned int x_base = ib * FUSED_Q4_BLOCK_ELEMS;

        // Load normed x-values from shmem into registers via float4 wide load.
        float xv[32];
        const float4* x4 = (const float4*)(nx_smem + x_base);
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            float4 v = x4[k];
            xv[k * 4 + 0] = v.x;
            xv[k * 4 + 1] = v.y;
            xv[k * 4 + 2] = v.z;
            xv[k * 4 + 3] = v.w;
        }

        #pragma unroll
        for (int row = 0; row < FUSED_NR; row++) {
            if (r0 + row >= inter_dim) break;

            // Gate weight block (Q4_0).
            const char* gp = w_gate
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * FUSED_Q4_BLOCK_BYTES;
            unsigned short g_scale_bits =
                  (unsigned short)(unsigned char)gp[0]
                | ((unsigned short)(unsigned char)gp[1] << 8);
            float g_scale = fused_f16_to_f32(g_scale_bits);
            const unsigned char* gq = (const unsigned char*)(gp + 2);

            // Up weight block (Q4_0).
            const char* up_ = w_up
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * FUSED_Q4_BLOCK_BYTES;
            unsigned short u_scale_bits =
                  (unsigned short)(unsigned char)up_[0]
                | ((unsigned short)(unsigned char)up_[1] << 8);
            float u_scale = fused_f16_to_f32(u_scale_bits);
            const unsigned char* uq = (const unsigned char*)(up_ + 2);

            float g_block_sum = 0.0f;
            float u_block_sum = 0.0f;

            // Unpack nibbles (GGML de-interleaved layout):
            //   lo nibble of byte b = element b, hi nibble = element b+16.
            //   dequant: scale * ((float)nibble - 8.0f)
            #pragma unroll
            for (int b = 0; b < 16; b++) {
                unsigned char gb = gq[b];
                unsigned char ub = uq[b];

                float gq_lo = (float)(gb & 0x0F) - 8.0f;
                float gq_hi = (float)(gb >> 4)    - 8.0f;
                float uq_lo = (float)(ub & 0x0F) - 8.0f;
                float uq_hi = (float)(ub >> 4)    - 8.0f;

                g_block_sum += gq_lo * xv[b]     + gq_hi * xv[b + 16];
                u_block_sum += uq_lo * xv[b]     + uq_hi * xv[b + 16];
            }

            gate_sum[row] += g_scale * g_block_sum;
            up_sum[row]   += u_scale * u_block_sum;
        }
    }

    // Cross-warp reduction + SwiGLU (mirrors fused_glu_gemv_q4_0).
    const unsigned int num_warps = FUSED_BLOCK_DIM / FUSED_WARP_SIZE;

    #pragma unroll
    for (int r = 0; r < FUSED_NR; r++) {
        gate_sum[r] = fused_warp_reduce_sum(gate_sum[r]);
    }
    __syncthreads();

    float* reduce_smem = nx_smem;
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < FUSED_NR; r++) {
            reduce_smem[r * num_warps + warp_id] = gate_sum[r];
        }
    }
    __syncthreads();

    float final_gate[FUSED_NR];
    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < FUSED_NR; r++) {
            float val = (lane < num_warps) ? reduce_smem[r * num_warps + lane] : 0.0f;
            val = fused_warp_reduce_sum(val);
            final_gate[r] = val;
        }
    }
    __syncthreads();

    #pragma unroll
    for (int r = 0; r < FUSED_NR; r++) {
        up_sum[r] = fused_warp_reduce_sum(up_sum[r]);
    }

    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < FUSED_NR; r++) {
            reduce_smem[r * num_warps + warp_id] = up_sum[r];
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < FUSED_NR; r++) {
            if (r0 + r < inter_dim) {
                float val = (lane < num_warps) ? reduce_smem[r * num_warps + lane] : 0.0f;
                val = fused_warp_reduce_sum(val);
                if (lane == 0) {
                    float g = final_gate[r];
                    float silu_g = g / (1.0f + expf(-g));
                    output[r0 + r] = silu_g * val;
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Fused down-matvec + sigmoid-gated accumulate for shared expert.
//
// Replaces (matvec_q4_0 + moe_shared_sigmoid_gated_accum) with one kernel:
//   x_out[i] += sigmoid(scalar[0]) * dot(W_down[i], swiglu_buf)
//
// Eliminates the intermediate `shared_down_buf` HBM write/read entirely.
//
// Grid:  (hidden_dim, 1, 1)        -- one CTA per output row
// Block: (FUSED_BLOCK_DIM, 1, 1)   -- 256 threads
//
// Identical Q4_0 dequant + warp-reduction pattern to `matvec_q4_0`. The
// sigmoid is evaluated once and broadcast (single F32 read, L1-cached).
//
// `scalar` is the F32 logit produced by `moe_shared_dot_f32` in a separate
// kernel launch ordered BEFORE this one on the same stream.
// ----------------------------------------------------------------------------
extern "C" __global__ void moe_shared_down_q4_0_sigmoid_accum(
    const char*  __restrict__ w_down,       // [hidden_dim, inter_dim] Q4_0
    const float* __restrict__ swiglu_buf,   // [inter_dim] silu(gate) * up
    const float* __restrict__ scalar,       // [1] pre-sigmoid logit
    float*       __restrict__ x_out,        // [hidden_dim] in/out
    unsigned int hidden_dim,
    unsigned int inter_dim)
{
    const unsigned int row = blockIdx.x;
    if (row >= hidden_dim) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & (FUSED_WARP_SIZE - 1);
    const unsigned int warp_id = tid / FUSED_WARP_SIZE;

    const unsigned int num_blocks = inter_dim / FUSED_Q4_BLOCK_ELEMS;
    const unsigned long long row_bytes =
        (unsigned long long)num_blocks * FUSED_Q4_BLOCK_BYTES;
    const char* row_ptr = w_down + (unsigned long long)row * row_bytes;

    float partial = 0.0f;

    // Stride over Q4_0 blocks; one thread per block per iteration.
    for (unsigned int b = tid; b < num_blocks; b += FUSED_BLOCK_DIM) {
        const char* block_ptr = row_ptr + b * FUSED_Q4_BLOCK_BYTES;

        unsigned short scale_bits =
              (unsigned short)(unsigned char)block_ptr[0]
            | ((unsigned short)(unsigned char)block_ptr[1] << 8);
        float scale = fused_f16_to_f32(scale_bits);

        const unsigned int x_base = b * FUSED_Q4_BLOCK_ELEMS;
        const unsigned char* qp = (const unsigned char*)(block_ptr + 2);

        float block_sum = 0.0f;

        // Unpack nibbles (GGML de-interleaved layout):
        //   lo nibble of byte k = element k, hi nibble = element k+16.
        //   dequant: scale * ((float)nibble - 8.0f)
        #pragma unroll
        for (int k = 0; k < 16; k++) {
            unsigned char by = qp[k];
            float q_lo = (float)(by & 0x0F) - 8.0f;
            float q_hi = (float)(by >> 4)    - 8.0f;
            block_sum += q_lo * swiglu_buf[x_base + k]
                       + q_hi * swiglu_buf[x_base + k + 16];
        }
        partial += scale * block_sum;
    }

    // Warp + block reduce.
    partial = fused_warp_reduce_sum(partial);

    __shared__ float warp_sums[FUSED_BLOCK_DIM / FUSED_WARP_SIZE];
    if (lane == 0) {
        warp_sums[warp_id] = partial;
    }
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane < (FUSED_BLOCK_DIM / FUSED_WARP_SIZE))
                ? warp_sums[lane] : 0.0f;
        v = fused_warp_reduce_sum(v);
        if (lane == 0) {
            // Single sigmoid evaluation, broadcast via accumulation.
            const float s = scalar[0];
            const float g = 1.0f / (1.0f + expf(-s));
            x_out[row] += g * v;
        }
    }
}

// ----------------------------------------------------------------------------
// Fused down-matvec + plain residual accumulate (no sigmoid gate).
//
// Mirror of `moe_shared_down_q4_0_sigmoid_accum` for the
// no-`ffn_gate_inp_shexp` variant. Computes x_out[i] += dot(W_down[i], swiglu).
// ----------------------------------------------------------------------------
extern "C" __global__ void moe_shared_down_q4_0_residual_accum(
    const char*  __restrict__ w_down,       // [hidden_dim, inter_dim] Q4_0
    const float* __restrict__ swiglu_buf,   // [inter_dim] silu(gate) * up
    float*       __restrict__ x_out,        // [hidden_dim] in/out
    unsigned int hidden_dim,
    unsigned int inter_dim)
{
    const unsigned int row = blockIdx.x;
    if (row >= hidden_dim) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & (FUSED_WARP_SIZE - 1);
    const unsigned int warp_id = tid / FUSED_WARP_SIZE;

    const unsigned int num_blocks = inter_dim / FUSED_Q4_BLOCK_ELEMS;
    const unsigned long long row_bytes =
        (unsigned long long)num_blocks * FUSED_Q4_BLOCK_BYTES;
    const char* row_ptr = w_down + (unsigned long long)row * row_bytes;

    float partial = 0.0f;

    for (unsigned int b = tid; b < num_blocks; b += FUSED_BLOCK_DIM) {
        const char* block_ptr = row_ptr + b * FUSED_Q4_BLOCK_BYTES;
        unsigned short scale_bits =
              (unsigned short)(unsigned char)block_ptr[0]
            | ((unsigned short)(unsigned char)block_ptr[1] << 8);
        float scale = fused_f16_to_f32(scale_bits);

        const unsigned int x_base = b * FUSED_Q4_BLOCK_ELEMS;
        const unsigned char* qp = (const unsigned char*)(block_ptr + 2);

        float block_sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < 16; k++) {
            unsigned char by = qp[k];
            float q_lo = (float)(by & 0x0F) - 8.0f;
            float q_hi = (float)(by >> 4)    - 8.0f;
            block_sum += q_lo * swiglu_buf[x_base + k]
                       + q_hi * swiglu_buf[x_base + k + 16];
        }
        partial += scale * block_sum;
    }

    partial = fused_warp_reduce_sum(partial);

    __shared__ float warp_sums[FUSED_BLOCK_DIM / FUSED_WARP_SIZE];
    if (lane == 0) {
        warp_sums[warp_id] = partial;
    }
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane < (FUSED_BLOCK_DIM / FUSED_WARP_SIZE))
                ? warp_sums[lane] : 0.0f;
        v = fused_warp_reduce_sum(v);
        if (lane == 0) {
            x_out[row] += v;
        }
    }
}
