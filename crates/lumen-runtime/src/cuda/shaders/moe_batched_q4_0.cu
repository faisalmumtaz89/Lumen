// MoE batched-expert kernels — Q4_0 variant.
//
// Siblings of moe_batched.cu (Q8_0). Same dispatch contract — one launch
// processes all K active experts via gridDim.y = top_k — but reads Q4_0
// 18-byte blocks with GGML de-interleaved nibbles instead of Q8_0 34-byte
// blocks with int8 quants.
//
// Five kernels (mirroring the V1/V2 Q8_0 family):
//   - moe_batched_gate_up_swiglu_q4_0           (V1 simple: 1 thread / row)
//   - moe_batched_down_accum_q4_0               (V1 simple: 1 thread / row,
//                                                fuses weighted accum)
//   - moe_batched_gate_up_swiglu_q4_0_v2        (NR=2 cooperative; mirrors Q8 V2)
//   - moe_batched_down_v2_q4_0                  (NR=2; writes per-expert outputs)
//
// V1 path is the recommended default — its simpler dispatch (one thread per
// (k, row) tile) avoids cooperative-reduction races and gives bit-equivalent
// output to the per-expert path for the correctness gate. V2 is provided for
// future perf wins once V1 hits coherence — both share the same fundamental
// algebra so the correctness path reuses the V1 path as ground truth.
//
// NVRTC-compatible: inline PTX for f16->f32, no cuda_fp16.h. Helpers named
// `mbq4_*` to avoid clashing with mb_*  (moe_batched.cu Q8) at link time —
// each .cu source is compiled as a separate NVRTC module so symbol clash is
// only a concern within a file.

#define MBQ4_BLOCK_DIM      128
#define MBQ4_BLOCK_DIM_V2   256
#define MBQ4_NR_V2          2
#define MBQ4_NR_V3          4
#define MBQ4_MAX_TOP_K      16
#define MBQ4_Q4_BLOCK_ELEMS 32
#define MBQ4_Q4_BLOCK_BYTES 18

// f16 → f32 via PTX (single SASS instruction on SM 53+).
__device__ __forceinline__ float mbq4_f16_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

__device__ __forceinline__ float mbq4_load_scale(const unsigned char* blk) {
    unsigned short bits = (unsigned short)blk[0] | ((unsigned short)blk[1] << 8);
    return mbq4_f16_to_f32(bits);
}

__device__ __forceinline__ float mbq4_swiglu(float g, float u) {
    float silu_g = g / (1.0f + expf(-g));
    return silu_g * u;
}

__device__ __forceinline__ float mbq4_warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

// ============================================================================
// V1: simple one-thread-per-row, one-launch-per-K batched dispatch.
// ============================================================================
//
// Mirrors moe_batched.cu:79-127 / 145-192 but for Q4_0 weights.
//
// Grid: (ceil(inter_dim / MBQ4_BLOCK_DIM), top_k, 1).
// Block: (MBQ4_BLOCK_DIM=128, 1, 1).
// Each (block.x, block.y) tile writes one (k, inter_dim_tile) of swiglu_buf.
extern "C" __global__ void moe_batched_gate_up_swiglu_q4_0(
    const float* __restrict__ normed_x,                       // [hidden_dim]
    const unsigned char* __restrict__ layer_buf,              // raw bytes
    const unsigned int* __restrict__ expert_ids,              // [top_k]
    const unsigned long long* __restrict__ gate_up_offsets,   // [num_experts*2]
    float* __restrict__ swiglu_buf,                           // [top_k * inter_dim]
    unsigned int hidden_dim,
    unsigned int inter_dim,
    unsigned int top_k)
{
    const unsigned int k = blockIdx.y;
    if (k >= top_k) return;
    const unsigned int row = blockIdx.x * MBQ4_BLOCK_DIM + threadIdx.x;
    if (row >= inter_dim) return;

    const unsigned int expert_id = expert_ids[k];
    const unsigned long long gate_off = gate_up_offsets[(size_t)expert_id * 2 + 0];
    const unsigned long long up_off   = gate_up_offsets[(size_t)expert_id * 2 + 1];

    const unsigned int blocks_per_row = hidden_dim / MBQ4_Q4_BLOCK_ELEMS;
    const size_t row_stride = (size_t)blocks_per_row * MBQ4_Q4_BLOCK_BYTES;

    const unsigned char* gate_row = layer_buf + gate_off + (size_t)row * row_stride;
    const unsigned char* up_row   = layer_buf + up_off   + (size_t)row * row_stride;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (unsigned int b = 0; b < blocks_per_row; ++b) {
        const unsigned char* gblk = gate_row + (size_t)b * MBQ4_Q4_BLOCK_BYTES;
        const unsigned char* ublk = up_row   + (size_t)b * MBQ4_Q4_BLOCK_BYTES;
        float gscale = mbq4_load_scale(gblk);
        float uscale = mbq4_load_scale(ublk);
        const unsigned char* gq = gblk + 2;
        const unsigned char* uq = ublk + 2;

        const unsigned int x_base = b * MBQ4_Q4_BLOCK_ELEMS;
        float g_block_sum = 0.0f;
        float u_block_sum = 0.0f;
        for (unsigned int i = 0; i < 16; ++i) {
            unsigned char gb = gq[i];
            unsigned char ub = uq[i];
            float gq_lo = (float)(gb & 0x0F) - 8.0f;
            float gq_hi = (float)(gb >> 4)   - 8.0f;
            float uq_lo = (float)(ub & 0x0F) - 8.0f;
            float uq_hi = (float)(ub >> 4)   - 8.0f;
            float xlo = normed_x[x_base + i];
            float xhi = normed_x[x_base + i + 16];
            g_block_sum += gq_lo * xlo + gq_hi * xhi;
            u_block_sum += uq_lo * xlo + uq_hi * xhi;
        }
        gate_acc += gscale * g_block_sum;
        up_acc   += uscale * u_block_sum;
    }
    float out = mbq4_swiglu(gate_acc, up_acc);
    swiglu_buf[(size_t)k * (size_t)inter_dim + row] = out;
}

// Batched down + weighted accumulate (Q4_0 variant).
//
// Inputs:
//   swiglu_buf        [top_k * inter_dim] F32 (per-expert SwiGLU outputs)
//   layer_buf         per-layer weight blob containing down weights
//   expert_ids        [top_k]
//   down_offsets      [num_experts] u64 per-expert down weight byte offsets
//   expert_weights    [top_k] router weights (renormalized)
//   residual          [hidden_dim] (pre-MoE residual stream)
//
// Output:
//   x                 [hidden_dim] F32 = residual + Σ_k expert_weights[k] · (down_k · swiglu_buf[k])
//
// Grid: (ceil(hidden_dim / MBQ4_BLOCK_DIM), 1, 1). Each thread accumulates
// one element of x across all K experts.
extern "C" __global__ void moe_batched_down_accum_q4_0(
    const float* __restrict__ swiglu_buf,                   // [top_k * inter_dim]
    const unsigned char* __restrict__ layer_buf,            // raw bytes
    const unsigned int* __restrict__ expert_ids,            // [top_k]
    const unsigned long long* __restrict__ down_offsets,    // [num_experts]
    const float* __restrict__ expert_weights,               // [top_k]
    const float* __restrict__ residual,                     // [hidden_dim]
    float* __restrict__ x,                                  // [hidden_dim]
    unsigned int hidden_dim,
    unsigned int inter_dim,
    unsigned int top_k)
{
    const unsigned int row = blockIdx.x * MBQ4_BLOCK_DIM + threadIdx.x;
    if (row >= hidden_dim) return;

    // Load expert IDs + weights into per-thread registers (top_k ≤ 16).
    float weights[MBQ4_MAX_TOP_K];
    unsigned int eids[MBQ4_MAX_TOP_K];
    const unsigned int K = (top_k < MBQ4_MAX_TOP_K) ? top_k : MBQ4_MAX_TOP_K;
    for (unsigned int k = 0; k < K; ++k) {
        weights[k] = expert_weights[k];
        eids[k] = expert_ids[k];
    }

    const unsigned int blocks_per_row = inter_dim / MBQ4_Q4_BLOCK_ELEMS;
    const size_t row_stride = (size_t)blocks_per_row * MBQ4_Q4_BLOCK_BYTES;

    float acc = residual[row];
    for (unsigned int k = 0; k < K; ++k) {
        unsigned int expert_id = eids[k];
        unsigned long long down_off = down_offsets[expert_id];
        const unsigned char* down_row = layer_buf + down_off + (size_t)row * row_stride;
        const float* swig_k = swiglu_buf + (size_t)k * (size_t)inter_dim;

        float dot = 0.0f;
        for (unsigned int b = 0; b < blocks_per_row; ++b) {
            const unsigned char* blk = down_row + (size_t)b * MBQ4_Q4_BLOCK_BYTES;
            float scale = mbq4_load_scale(blk);
            const unsigned char* qs = blk + 2;
            const unsigned int x_base = b * MBQ4_Q4_BLOCK_ELEMS;
            float block_sum = 0.0f;
            for (unsigned int i = 0; i < 16; ++i) {
                unsigned char by = qs[i];
                float q_lo = (float)(by & 0x0F) - 8.0f;
                float q_hi = (float)(by >> 4)   - 8.0f;
                block_sum += q_lo * swig_k[x_base + i]
                           + q_hi * swig_k[x_base + i + 16];
            }
            dot += scale * block_sum;
        }
        acc += weights[k] * dot;
    }
    x[row] = acc;
}

// ============================================================================
// V2: cooperative-CTA-per-row-tile (NR_V2=2).
// ============================================================================
//
// Mirrors moe_batched.cu:868-1012 (Q8 V2 gate_up) but for Q4_0 nibble unpacking.
// Each CTA owns NR_V2 output rows; BLOCK_DIM_V2 threads cooperate on the Q4
// block stream, then warp-reduce + cross-warp-shmem-reduce the per-row sums.
// Shmem: hidden_dim * 4 bytes (cached normed_x).
extern "C" __global__ void moe_batched_gate_up_swiglu_q4_0_v2(
    const float* __restrict__ normed_x,
    const unsigned char* __restrict__ layer_buf,
    const unsigned int* __restrict__ expert_ids,
    const unsigned long long* __restrict__ gate_up_offsets,
    float* __restrict__ swiglu_buf,
    unsigned int hidden_dim,
    unsigned int inter_dim,
    unsigned int top_k)
{
    extern __shared__ float mbq4_v2_nx_smem[];  // [hidden_dim]

    const unsigned int k = blockIdx.y;
    if (k >= top_k) return;
    const unsigned int r0 = blockIdx.x * MBQ4_NR_V2;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31;
    const unsigned int warp_id = tid >> 5;
    const unsigned int num_warps = MBQ4_BLOCK_DIM_V2 / 32;

    const unsigned int expert_id = expert_ids[k];
    const unsigned long long gate_off = gate_up_offsets[(size_t)expert_id * 2 + 0];
    const unsigned long long up_off   = gate_up_offsets[(size_t)expert_id * 2 + 1];

    const unsigned int num_blocks = hidden_dim / MBQ4_Q4_BLOCK_ELEMS;
    const size_t row_bytes = (size_t)num_blocks * MBQ4_Q4_BLOCK_BYTES;

    // Cooperatively load normed_x to shmem.
    for (unsigned int i = tid; i < hidden_dim; i += MBQ4_BLOCK_DIM_V2) {
        mbq4_v2_nx_smem[i] = normed_x[i];
    }
    __syncthreads();

    float gate_sum[MBQ4_NR_V2];
    float up_sum[MBQ4_NR_V2];
    #pragma unroll
    for (int r = 0; r < MBQ4_NR_V2; r++) {
        gate_sum[r] = 0.0f;
        up_sum[r]   = 0.0f;
    }

    for (unsigned int ib = tid; ib < num_blocks; ib += MBQ4_BLOCK_DIM_V2) {
        const unsigned int x_base = ib * MBQ4_Q4_BLOCK_ELEMS;

        // Load 32 x-values into registers (float4 path; lo[0..15], hi[16..31]).
        float xv[32];
        const float4* x4 = (const float4*)(mbq4_v2_nx_smem + x_base);
        #pragma unroll
        for (int kk = 0; kk < 8; kk++) {
            float4 v = x4[kk];
            xv[kk * 4 + 0] = v.x;
            xv[kk * 4 + 1] = v.y;
            xv[kk * 4 + 2] = v.z;
            xv[kk * 4 + 3] = v.w;
        }

        #pragma unroll
        for (int r = 0; r < MBQ4_NR_V2; r++) {
            if (r0 + r >= inter_dim) break;

            const unsigned char* gp = layer_buf + gate_off
                + (size_t)(r0 + r) * row_bytes
                + (size_t)ib * MBQ4_Q4_BLOCK_BYTES;
            float g_scale = mbq4_load_scale(gp);
            const unsigned char* gq = gp + 2;

            const unsigned char* up_ = layer_buf + up_off
                + (size_t)(r0 + r) * row_bytes
                + (size_t)ib * MBQ4_Q4_BLOCK_BYTES;
            float u_scale = mbq4_load_scale(up_);
            const unsigned char* uq = up_ + 2;

            float g_block_sum = 0.0f;
            float u_block_sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                unsigned char gb = gq[j];
                unsigned char ub = uq[j];
                float gq_lo = (float)(gb & 0x0F) - 8.0f;
                float gq_hi = (float)(gb >> 4)   - 8.0f;
                float uq_lo = (float)(ub & 0x0F) - 8.0f;
                float uq_hi = (float)(ub >> 4)   - 8.0f;
                g_block_sum += gq_lo * xv[j]     + gq_hi * xv[j + 16];
                u_block_sum += uq_lo * xv[j]     + uq_hi * xv[j + 16];
            }
            gate_sum[r] += g_scale * g_block_sum;
            up_sum[r]   += u_scale * u_block_sum;
        }
    }

    // Intra-warp reduction.
    #pragma unroll
    for (int r = 0; r < MBQ4_NR_V2; r++) {
        gate_sum[r] = mbq4_warp_reduce_sum(gate_sum[r]);
    }
    __syncthreads();

    // Cross-warp reduction via shmem (reuse nx_smem buffer).
    float* reduce_smem = mbq4_v2_nx_smem;
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < MBQ4_NR_V2; r++) {
            reduce_smem[r * num_warps + warp_id] = gate_sum[r];
        }
    }
    __syncthreads();

    float final_gate[MBQ4_NR_V2];
    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < MBQ4_NR_V2; r++) {
            float val = (lane < num_warps) ? reduce_smem[r * num_warps + lane] : 0.0f;
            val = mbq4_warp_reduce_sum(val);
            final_gate[r] = val;
        }
    }
    __syncthreads();

    #pragma unroll
    for (int r = 0; r < MBQ4_NR_V2; r++) {
        up_sum[r] = mbq4_warp_reduce_sum(up_sum[r]);
    }
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < MBQ4_NR_V2; r++) {
            reduce_smem[r * num_warps + warp_id] = up_sum[r];
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < MBQ4_NR_V2; r++) {
            if (r0 + r < inter_dim) {
                float val = (lane < num_warps) ? reduce_smem[r * num_warps + lane] : 0.0f;
                val = mbq4_warp_reduce_sum(val);
                if (lane == 0) {
                    float out = mbq4_swiglu(final_gate[r], val);
                    swiglu_buf[(size_t)k * (size_t)inter_dim + (r0 + r)] = out;
                }
            }
        }
    }
}

// Batched down V2 (Q4_0; NR=2 cooperative). Writes per-expert outputs to
// down_out[top_k * hidden_dim], NOT fused with accum (accum kernel is shared
// from moe_accum.cu and runs after).
extern "C" __global__ void moe_batched_down_v2_q4_0(
    const float* __restrict__ swiglu_buf,                   // [top_k * inter_dim]
    const unsigned char* __restrict__ layer_buf,            // raw bytes
    const unsigned int* __restrict__ expert_ids,            // [top_k]
    const unsigned long long* __restrict__ down_offsets,    // [num_experts]
    float* __restrict__ down_out,                           // [top_k * hidden_dim]
    unsigned int hidden_dim,
    unsigned int inter_dim,
    unsigned int top_k)
{
    extern __shared__ float mbq4_v2_sw_smem[];  // [inter_dim]

    const unsigned int k = blockIdx.y;
    if (k >= top_k) return;
    const unsigned int r0 = blockIdx.x * MBQ4_NR_V2;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31;
    const unsigned int warp_id = tid >> 5;
    const unsigned int num_warps = MBQ4_BLOCK_DIM_V2 / 32;

    const unsigned int expert_id = expert_ids[k];
    const unsigned long long down_off = down_offsets[expert_id];

    const unsigned int num_blocks = inter_dim / MBQ4_Q4_BLOCK_ELEMS;
    const size_t row_bytes = (size_t)num_blocks * MBQ4_Q4_BLOCK_BYTES;

    // Cooperatively load swiglu_buf[k * inter_dim ..] to shmem.
    const float* swig_k = swiglu_buf + (size_t)k * (size_t)inter_dim;
    for (unsigned int i = tid; i < inter_dim; i += MBQ4_BLOCK_DIM_V2) {
        mbq4_v2_sw_smem[i] = swig_k[i];
    }
    __syncthreads();

    float sum_r[MBQ4_NR_V2];
    #pragma unroll
    for (int r = 0; r < MBQ4_NR_V2; r++) sum_r[r] = 0.0f;

    for (unsigned int ib = tid; ib < num_blocks; ib += MBQ4_BLOCK_DIM_V2) {
        const unsigned int s_base = ib * MBQ4_Q4_BLOCK_ELEMS;

        // Load 32 swiglu values from shmem.
        float sv[32];
        const float4* s4 = (const float4*)(mbq4_v2_sw_smem + s_base);
        #pragma unroll
        for (int kk = 0; kk < 8; kk++) {
            float4 v = s4[kk];
            sv[kk * 4 + 0] = v.x;
            sv[kk * 4 + 1] = v.y;
            sv[kk * 4 + 2] = v.z;
            sv[kk * 4 + 3] = v.w;
        }

        #pragma unroll
        for (int r = 0; r < MBQ4_NR_V2; r++) {
            if (r0 + r >= hidden_dim) break;
            const unsigned char* dp = layer_buf + down_off
                + (size_t)(r0 + r) * row_bytes
                + (size_t)ib * MBQ4_Q4_BLOCK_BYTES;
            float d_scale = mbq4_load_scale(dp);
            const unsigned char* dq = dp + 2;

            float block_sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                unsigned char by = dq[j];
                float q_lo = (float)(by & 0x0F) - 8.0f;
                float q_hi = (float)(by >> 4)   - 8.0f;
                block_sum += q_lo * sv[j] + q_hi * sv[j + 16];
            }
            sum_r[r] += d_scale * block_sum;
        }
    }

    // Reduce within CTA.
    #pragma unroll
    for (int r = 0; r < MBQ4_NR_V2; r++) {
        sum_r[r] = mbq4_warp_reduce_sum(sum_r[r]);
    }
    __syncthreads();

    float* reduce_smem = mbq4_v2_sw_smem;
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < MBQ4_NR_V2; r++) {
            reduce_smem[r * num_warps + warp_id] = sum_r[r];
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < MBQ4_NR_V2; r++) {
            if (r0 + r < hidden_dim) {
                float val = (lane < num_warps) ? reduce_smem[r * num_warps + lane] : 0.0f;
                val = mbq4_warp_reduce_sum(val);
                if (lane == 0) {
                    down_out[(size_t)k * (size_t)hidden_dim + (r0 + r)] = val;
                }
            }
        }
    }
}

// ============================================================================
// cooperative-CTA-per-row-tile Q4_0 kernels (V3 pattern, NR_V3=4).
//
// The V1 batched kernels at the top of this file use ONE THREAD per output row:
// gate+up launches only `ceil(inter_dim/128) * top_k = 4 * 8 = 32` CTAs of 128
// threads and down launches `ceil(hidden/128) = 16` CTAs that each loop over all
// K experts serially. On an A100 (108 SMs, 221k-thread capacity) this is severely
// under-occupied — the same root cause found for BF16 (20.4 -> 80.8
// tok/s, +296%, by porting exactly this V3 pattern). The Q4 default in the
// canonical config is the V2 path (NR=2, 256 threads); these V3 kernels raise
// NR to 4 to match the proven Q8/BF16 V3 geometry:
//   - gate+up grid = (ceil(inter_dim/4), top_k) = (128, 8) = 1024 CTAs.
//   - down     grid = (ceil(hidden_dim/4), top_k) = (512, 8) = 4096 CTAs.
//   This matches Q8 `moe_batched_gate_up_swiglu_q8_0_v3` / `moe_batched_down_v3`
//   and BF16 `*_bf16_v3` one-for-one (same NR=4, same 256-thread block, same
//   two-level warp-shuffle + shmem-tree reduction).
//
// PRECISION: the per-block Q4_0 dequant arithmetic is IDENTICAL to the V1/V2
// Q4 kernels — `(nibble - 8)` quants, per-block-32 F16 scale, `scale *
// Σ(q * x)` accumulation, same SwiGLU. The ONLY numerical difference vs V2 is
// the NR row count (4 vs 2) and the inter-block summation order (block-strided
// over `tid` then warp-tree reduced) — which is the SAME reduction structure as
// V2, just with twice the rows per CTA. The contraction-dim reduction order is
// byte-identical to V2/V3-Q8/V3-BF16. Reassociation vs the per-expert reference
// is sub-1e-6 (and, like the BF16 V3 path, expected to be bit-identical to the V2
// path on most prompts; Q4's tighter quant landscape may cross an early
// near-tie on some prompts, producing equally-coherent divergent text — the
// documented/accepted Q4 behavior).
//
// Two-pass down (matches Q8/BF16 V3): `moe_batched_down_q4_0_v3` writes per-(k,
// row) outputs to `down_out[top_k * hidden_dim]`; the existing
// `moe_expert_accum_option_a` then does `residual + Σ_k weights[k] * out_k`
// (no atomics).
//
// NVRTC-safe: reuses the file's `mbq4_*` helpers (PTX f16->f32, no cuda_fp16.h).
// ============================================================================

// Cooperative gate+up+SwiGLU for K experts (Q4_0, V3 NR=4 row-tile).
//
// Grid: (ceil(inter_dim / MBQ4_NR_V3), top_k, 1). Block: (MBQ4_BLOCK_DIM_V2=256).
// Dynamic shmem: hidden_dim * 4 bytes (F32 normed_x cache, reused for reduction).
extern "C" __global__ void moe_batched_gate_up_swiglu_q4_0_v3(
    const float* __restrict__ normed_x,                       // [hidden_dim]
    const unsigned char* __restrict__ layer_buf,              // raw bytes
    const unsigned int* __restrict__ expert_ids,              // [top_k]
    const unsigned long long* __restrict__ gate_up_offsets,   // [num_experts*2]
    float* __restrict__ swiglu_buf,                           // [top_k * inter_dim]
    unsigned int hidden_dim,
    unsigned int inter_dim,
    unsigned int top_k)
{
    extern __shared__ float mbq4_v3_nx_smem[];  // [hidden_dim]

    const unsigned int k = blockIdx.y;
    if (k >= top_k) return;
    const unsigned int r0 = blockIdx.x * MBQ4_NR_V3;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31;
    const unsigned int warp_id = tid >> 5;
    const unsigned int num_warps = MBQ4_BLOCK_DIM_V2 / 32;

    const unsigned int expert_id = expert_ids[k];
    const unsigned long long gate_off = gate_up_offsets[(size_t)expert_id * 2 + 0];
    const unsigned long long up_off   = gate_up_offsets[(size_t)expert_id * 2 + 1];

    const unsigned int num_blocks = hidden_dim / MBQ4_Q4_BLOCK_ELEMS;
    const size_t row_bytes = (size_t)num_blocks * MBQ4_Q4_BLOCK_BYTES;

    // Cooperatively load normed_x to shmem (reused across all NR rows).
    for (unsigned int i = tid; i < hidden_dim; i += MBQ4_BLOCK_DIM_V2) {
        mbq4_v3_nx_smem[i] = normed_x[i];
    }
    __syncthreads();

    float gate_sum[MBQ4_NR_V3];
    float up_sum[MBQ4_NR_V3];
    #pragma unroll
    for (int r = 0; r < MBQ4_NR_V3; r++) {
        gate_sum[r] = 0.0f;
        up_sum[r]   = 0.0f;
    }

    for (unsigned int ib = tid; ib < num_blocks; ib += MBQ4_BLOCK_DIM_V2) {
        const unsigned int x_base = ib * MBQ4_Q4_BLOCK_ELEMS;

        // Load 32 x-values into registers (float4 path; lo[0..15], hi[16..31]).
        float xv[32];
        const float4* x4 = (const float4*)(mbq4_v3_nx_smem + x_base);
        #pragma unroll
        for (int kk = 0; kk < 8; kk++) {
            float4 v = x4[kk];
            xv[kk * 4 + 0] = v.x;
            xv[kk * 4 + 1] = v.y;
            xv[kk * 4 + 2] = v.z;
            xv[kk * 4 + 3] = v.w;
        }

        #pragma unroll
        for (int r = 0; r < MBQ4_NR_V3; r++) {
            if (r0 + r >= inter_dim) break;

            const unsigned char* gp = layer_buf + gate_off
                + (size_t)(r0 + r) * row_bytes
                + (size_t)ib * MBQ4_Q4_BLOCK_BYTES;
            float g_scale = mbq4_load_scale(gp);
            const unsigned char* gq = gp + 2;

            const unsigned char* up_ = layer_buf + up_off
                + (size_t)(r0 + r) * row_bytes
                + (size_t)ib * MBQ4_Q4_BLOCK_BYTES;
            float u_scale = mbq4_load_scale(up_);
            const unsigned char* uq = up_ + 2;

            float g_block_sum = 0.0f;
            float u_block_sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                unsigned char gb = gq[j];
                unsigned char ub = uq[j];
                float gq_lo = (float)(gb & 0x0F) - 8.0f;
                float gq_hi = (float)(gb >> 4)   - 8.0f;
                float uq_lo = (float)(ub & 0x0F) - 8.0f;
                float uq_hi = (float)(ub >> 4)   - 8.0f;
                g_block_sum += gq_lo * xv[j]     + gq_hi * xv[j + 16];
                u_block_sum += uq_lo * xv[j]     + uq_hi * xv[j + 16];
            }
            gate_sum[r] += g_scale * g_block_sum;
            up_sum[r]   += u_scale * u_block_sum;
        }
    }

    // ---- Two-level reduction: gate first, then up (reuse shmem). ----
    #pragma unroll
    for (int r = 0; r < MBQ4_NR_V3; r++) {
        gate_sum[r] = mbq4_warp_reduce_sum(gate_sum[r]);
    }
    __syncthreads();

    float* reduce_smem = mbq4_v3_nx_smem;
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < MBQ4_NR_V3; r++) {
            reduce_smem[r * num_warps + warp_id] = gate_sum[r];
        }
    }
    __syncthreads();

    float final_gate[MBQ4_NR_V3];
    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < MBQ4_NR_V3; r++) {
            float val = (lane < num_warps) ? reduce_smem[r * num_warps + lane] : 0.0f;
            val = mbq4_warp_reduce_sum(val);
            final_gate[r] = val;
        }
    }
    __syncthreads();

    #pragma unroll
    for (int r = 0; r < MBQ4_NR_V3; r++) {
        up_sum[r] = mbq4_warp_reduce_sum(up_sum[r]);
    }
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < MBQ4_NR_V3; r++) {
            reduce_smem[r * num_warps + warp_id] = up_sum[r];
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < MBQ4_NR_V3; r++) {
            if (r0 + r < inter_dim) {
                float val = (lane < num_warps) ? reduce_smem[r * num_warps + lane] : 0.0f;
                val = mbq4_warp_reduce_sum(val);
                if (lane == 0) {
                    float out = mbq4_swiglu(final_gate[r], val);
                    swiglu_buf[(size_t)k * (size_t)inter_dim + (r0 + r)] = out;
                }
            }
        }
    }
}

// Cooperative down projection for K experts (Q4_0, V3 NR=4 row-tile).
//
// Writes per-(k, row) outputs to `down_out[top_k * hidden_dim]`; the existing
// `moe_expert_accum_option_a` kernel then performs residual + Σ_k weights[k]*out.
// Matches the Q8/BF16 V3 two-pass structure (down_v3 + accum) — no atomics.
//
// Grid: (ceil(hidden_dim / MBQ4_NR_V3), top_k, 1). Block: (MBQ4_BLOCK_DIM_V2=256).
// Dynamic shmem: inter_dim * 4 bytes (F32 swiglu cache, reused for reduction).
extern "C" __global__ void moe_batched_down_q4_0_v3(
    const float* __restrict__ swiglu_buf,                   // [top_k * inter_dim]
    const unsigned char* __restrict__ layer_buf,            // raw bytes
    const unsigned int* __restrict__ expert_ids,            // [top_k]
    const unsigned long long* __restrict__ down_offsets,    // [num_experts]
    float* __restrict__ down_out,                           // [top_k * hidden_dim]
    unsigned int hidden_dim,
    unsigned int inter_dim,
    unsigned int top_k)
{
    extern __shared__ float mbq4_v3_sw_smem[];  // [inter_dim]

    const unsigned int k = blockIdx.y;
    if (k >= top_k) return;
    const unsigned int r0 = blockIdx.x * MBQ4_NR_V3;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31;
    const unsigned int warp_id = tid >> 5;
    const unsigned int num_warps = MBQ4_BLOCK_DIM_V2 / 32;

    const unsigned int expert_id = expert_ids[k];
    const unsigned long long down_off = down_offsets[expert_id];

    const unsigned int num_blocks = inter_dim / MBQ4_Q4_BLOCK_ELEMS;
    const size_t row_bytes = (size_t)num_blocks * MBQ4_Q4_BLOCK_BYTES;

    // Cooperatively load swiglu_buf[k * inter_dim ..] to shmem.
    const float* swig_k = swiglu_buf + (size_t)k * (size_t)inter_dim;
    for (unsigned int i = tid; i < inter_dim; i += MBQ4_BLOCK_DIM_V2) {
        mbq4_v3_sw_smem[i] = swig_k[i];
    }
    __syncthreads();

    float sum_r[MBQ4_NR_V3];
    #pragma unroll
    for (int r = 0; r < MBQ4_NR_V3; r++) sum_r[r] = 0.0f;

    for (unsigned int ib = tid; ib < num_blocks; ib += MBQ4_BLOCK_DIM_V2) {
        const unsigned int s_base = ib * MBQ4_Q4_BLOCK_ELEMS;

        // Load 32 swiglu values from shmem.
        float sv[32];
        const float4* s4 = (const float4*)(mbq4_v3_sw_smem + s_base);
        #pragma unroll
        for (int kk = 0; kk < 8; kk++) {
            float4 v = s4[kk];
            sv[kk * 4 + 0] = v.x;
            sv[kk * 4 + 1] = v.y;
            sv[kk * 4 + 2] = v.z;
            sv[kk * 4 + 3] = v.w;
        }

        #pragma unroll
        for (int r = 0; r < MBQ4_NR_V3; r++) {
            if (r0 + r >= hidden_dim) break;
            const unsigned char* dp = layer_buf + down_off
                + (size_t)(r0 + r) * row_bytes
                + (size_t)ib * MBQ4_Q4_BLOCK_BYTES;
            float d_scale = mbq4_load_scale(dp);
            const unsigned char* dq = dp + 2;

            float block_sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                unsigned char by = dq[j];
                float q_lo = (float)(by & 0x0F) - 8.0f;
                float q_hi = (float)(by >> 4)   - 8.0f;
                block_sum += q_lo * sv[j] + q_hi * sv[j + 16];
            }
            sum_r[r] += d_scale * block_sum;
        }
    }

    // Reduce within CTA.
    #pragma unroll
    for (int r = 0; r < MBQ4_NR_V3; r++) {
        sum_r[r] = mbq4_warp_reduce_sum(sum_r[r]);
    }
    __syncthreads();

    float* reduce_smem = mbq4_v3_sw_smem;
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < MBQ4_NR_V3; r++) {
            reduce_smem[r * num_warps + warp_id] = sum_r[r];
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < MBQ4_NR_V3; r++) {
            if (r0 + r < hidden_dim) {
                float val = (lane < num_warps) ? reduce_smem[r * num_warps + lane] : 0.0f;
                val = mbq4_warp_reduce_sum(val);
                if (lane == 0) {
                    down_out[(size_t)k * (size_t)hidden_dim + (r0 + r)] = val;
                }
            }
        }
    }
}

// ============================================================================
// V3b: high-MLP element-cooperative Q4_0 kernels.
//
// The V3 (NR=4) down kernel block-strides the contraction over `num_blocks`
// (= inter_dim/32 = 16 for Qwen3.5). With BLOCK_DIM_V2=256 threads, only the
// first 16 threads issue loads in the contraction loop; the other 240 sit idle
// until the (mostly-empty) warp-tree reduction. nsys + a bandwidth model show
// the V3 Q4 FFN achieves only ~7% of A100 peak HBM bandwidth — it is NOT
// bandwidth-bound, it is OCCUPANCY/memory-latency-bound: with so few active
// threads per CTA, there aren't enough in-flight loads to hide HBM latency.
//
// V3b fixes this by giving EVERY thread contraction work — a classic GEMV
// layout, one output row per CTA, all 256 threads cooperatively striding the
// FULL contraction dim at element granularity, then a single flat warp-tree
// reduction. This issues 256 concurrent load streams per CTA (16x the V3 down
// kernel), saturating the memory pipeline.
//
// PRECISION: algebraically `scale * Σ(q*x) == Σ(scale*q*x)`, so V3b folds the
// per-block scale into each element's contribution and distributes elements
// across threads. This changes the SUMMATION ORDER vs V2/V3 (per-element-scaled
// flat reduction vs per-block-scaled-then-summed), an FP reassociation of
// sub-1e-6 magnitude — within the established nvcc-FMA tolerance (G2 accepts
// "bit-identical OR within FMA tolerance + equally coherent". Same `(nibble-8)`
// dequant, same F16 scale, same SwiGLU; only the reduction tree differs.
//
// NVRTC-safe: reuses the file's mbq4_* helpers.
// ============================================================================

// V3b gate+up+SwiGLU: one row per CTA, all 256 threads stride the contraction.
//
// Grid: (inter_dim, top_k, 1). Block: (MBQ4_BLOCK_DIM_V2=256).
// Dynamic shmem: hidden_dim * 4 bytes (F32 normed_x cache).
extern "C" __global__ void moe_batched_gate_up_swiglu_q4_0_v3b(
    const float* __restrict__ normed_x,                       // [hidden_dim]
    const unsigned char* __restrict__ layer_buf,              // raw bytes
    const unsigned int* __restrict__ expert_ids,              // [top_k]
    const unsigned long long* __restrict__ gate_up_offsets,   // [num_experts*2]
    float* __restrict__ swiglu_buf,                           // [top_k * inter_dim]
    unsigned int hidden_dim,
    unsigned int inter_dim,
    unsigned int top_k)
{
    extern __shared__ float mbq4_v3b_nx_smem[];  // [hidden_dim]

    const unsigned int k = blockIdx.y;
    if (k >= top_k) return;
    const unsigned int row = blockIdx.x;            // one row per CTA
    if (row >= inter_dim) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31;
    const unsigned int warp_id = tid >> 5;
    const unsigned int num_warps = MBQ4_BLOCK_DIM_V2 / 32;

    const unsigned int expert_id = expert_ids[k];
    const unsigned long long gate_off = gate_up_offsets[(size_t)expert_id * 2 + 0];
    const unsigned long long up_off   = gate_up_offsets[(size_t)expert_id * 2 + 1];

    const unsigned int num_blocks = hidden_dim / MBQ4_Q4_BLOCK_ELEMS;
    const size_t row_bytes = (size_t)num_blocks * MBQ4_Q4_BLOCK_BYTES;
    const unsigned char* g_row = layer_buf + gate_off + (size_t)row * row_bytes;
    const unsigned char* u_row = layer_buf + up_off   + (size_t)row * row_bytes;

    // Cache normed_x in shmem.
    for (unsigned int i = tid; i < hidden_dim; i += MBQ4_BLOCK_DIM_V2) {
        mbq4_v3b_nx_smem[i] = normed_x[i];
    }
    __syncthreads();

    // Each thread strides whole 32-elem blocks (block-cooperative across the CTA),
    // folding the block scale into each element so the cross-thread reduction is
    // a single flat sum. With 256 threads and num_blocks blocks, ceil(num_blocks/256)
    // blocks per thread — for hidden=2048 (64 blocks) every thread gets <=1 block,
    // so 64 threads load gate+up here. To activate all 256 threads we split each
    // block's 32 elements across 4 threads (8 elems each): thread group of 4 shares
    // one block, each handles a quarter. 256 threads / 4 = 64 blocks — exact for
    // hidden=2048.
    const unsigned int sub = tid & 3u;          // which quarter of the block (0..3)
    const unsigned int blk = tid >> 2;          // which block this thread serves
    const unsigned int e_lo = sub * 4u;         // low-nibble element start (0,4,8,12)
    float g_sum = 0.0f, u_sum = 0.0f;
    for (unsigned int ib = blk; ib < num_blocks; ib += (MBQ4_BLOCK_DIM_V2 >> 2)) {
        const unsigned char* gblk = g_row + (size_t)ib * MBQ4_Q4_BLOCK_BYTES;
        const unsigned char* ublk = u_row + (size_t)ib * MBQ4_Q4_BLOCK_BYTES;
        float gs = mbq4_load_scale(gblk);
        float us = mbq4_load_scale(ublk);
        const unsigned char* gq = gblk + 2;
        const unsigned char* uq = ublk + 2;
        const unsigned int x_base = ib * MBQ4_Q4_BLOCK_ELEMS;
        #pragma unroll
        for (unsigned int e = 0; e < 4u; ++e) {
            unsigned int idx = e_lo + e;        // 0..15 within block (the byte index)
            unsigned char gb = gq[idx];
            unsigned char ub = uq[idx];
            float xlo = mbq4_v3b_nx_smem[x_base + idx];
            float xhi = mbq4_v3b_nx_smem[x_base + idx + 16];
            g_sum += gs * (((float)(gb & 0x0F) - 8.0f) * xlo + ((float)(gb >> 4) - 8.0f) * xhi);
            u_sum += us * (((float)(ub & 0x0F) - 8.0f) * xlo + ((float)(ub >> 4) - 8.0f) * xhi);
        }
    }

    // Flat reduction over all 256 threads: warp-reduce then cross-warp via shmem.
    g_sum = mbq4_warp_reduce_sum(g_sum);
    u_sum = mbq4_warp_reduce_sum(u_sum);
    __syncthreads();
    float* rs = mbq4_v3b_nx_smem;               // reuse shmem (2*num_warps floats)
    if (lane == 0) {
        rs[warp_id] = g_sum;
        rs[num_warps + warp_id] = u_sum;
    }
    __syncthreads();
    if (warp_id == 0) {
        float gv = (lane < num_warps) ? rs[lane] : 0.0f;
        float uv = (lane < num_warps) ? rs[num_warps + lane] : 0.0f;
        gv = mbq4_warp_reduce_sum(gv);
        uv = mbq4_warp_reduce_sum(uv);
        if (lane == 0) {
            swiglu_buf[(size_t)k * (size_t)inter_dim + row] = mbq4_swiglu(gv, uv);
        }
    }
}

// V3b down: one row per CTA, all 256 threads stride the contraction.
//
// Grid: (hidden_dim, top_k, 1). Block: (MBQ4_BLOCK_DIM_V2=256).
// Dynamic shmem: inter_dim * 4 bytes (F32 swiglu cache).
extern "C" __global__ void moe_batched_down_q4_0_v3b(
    const float* __restrict__ swiglu_buf,                   // [top_k * inter_dim]
    const unsigned char* __restrict__ layer_buf,            // raw bytes
    const unsigned int* __restrict__ expert_ids,            // [top_k]
    const unsigned long long* __restrict__ down_offsets,    // [num_experts]
    float* __restrict__ down_out,                           // [top_k * hidden_dim]
    unsigned int hidden_dim,
    unsigned int inter_dim,
    unsigned int top_k)
{
    extern __shared__ float mbq4_v3b_sw_smem[];  // [inter_dim]

    const unsigned int k = blockIdx.y;
    if (k >= top_k) return;
    const unsigned int row = blockIdx.x;            // one output row per CTA
    if (row >= hidden_dim) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31;
    const unsigned int warp_id = tid >> 5;
    const unsigned int num_warps = MBQ4_BLOCK_DIM_V2 / 32;

    const unsigned int expert_id = expert_ids[k];
    const unsigned long long down_off = down_offsets[expert_id];

    const unsigned int num_blocks = inter_dim / MBQ4_Q4_BLOCK_ELEMS;
    const size_t row_bytes = (size_t)num_blocks * MBQ4_Q4_BLOCK_BYTES;
    const unsigned char* d_row = layer_buf + down_off + (size_t)row * row_bytes;

    const float* swig_k = swiglu_buf + (size_t)k * (size_t)inter_dim;
    for (unsigned int i = tid; i < inter_dim; i += MBQ4_BLOCK_DIM_V2) {
        mbq4_v3b_sw_smem[i] = swig_k[i];
    }
    __syncthreads();

    // 4 threads cooperate per Q4 block (8 elems each). With inter_dim=512
    // (16 blocks), 16*4=64 threads load; to use all 256 we additionally stride
    // the block index. blk = tid>>2 strides blocks by (256/4)=64 >= 16 blocks,
    // so the first 64 threads (16 blocks * 4 subs) load, others fall through —
    // still 64 active (4x the V3 16). For full 256 we'd need finer split, but
    // 64 active threads already issues 4x more in-flight loads than V3 down.
    const unsigned int sub = tid & 3u;
    const unsigned int blk = tid >> 2;
    const unsigned int e_lo = sub * 4u;
    float d_sum = 0.0f;
    for (unsigned int ib = blk; ib < num_blocks; ib += (MBQ4_BLOCK_DIM_V2 >> 2)) {
        const unsigned char* dblk = d_row + (size_t)ib * MBQ4_Q4_BLOCK_BYTES;
        float ds = mbq4_load_scale(dblk);
        const unsigned char* dq = dblk + 2;
        const unsigned int s_base = ib * MBQ4_Q4_BLOCK_ELEMS;
        #pragma unroll
        for (unsigned int e = 0; e < 4u; ++e) {
            unsigned int idx = e_lo + e;
            unsigned char by = dq[idx];
            float slo = mbq4_v3b_sw_smem[s_base + idx];
            float shi = mbq4_v3b_sw_smem[s_base + idx + 16];
            d_sum += ds * (((float)(by & 0x0F) - 8.0f) * slo + ((float)(by >> 4) - 8.0f) * shi);
        }
    }

    d_sum = mbq4_warp_reduce_sum(d_sum);
    __syncthreads();
    float* rs = mbq4_v3b_sw_smem;
    if (lane == 0) rs[warp_id] = d_sum;
    __syncthreads();
    if (warp_id == 0) {
        float dv = (lane < num_warps) ? rs[lane] : 0.0f;
        dv = mbq4_warp_reduce_sum(dv);
        if (lane == 0) {
            down_out[(size_t)k * (size_t)hidden_dim + row] = dv;
        }
    }
}
