// MoE BF16 expert kernels.
//
// Port of `moe_batched.cu` (Q8_0) to plain BF16 row-major weights. BF16 is
// 2 bytes per element with no block structure, no scale factors; the
// dispatch contract matches the Q8_0 batched path one-for-one:
//
//   `moe_batched_gate_up_swiglu_bf16` : K-expert gate+up+SwiGLU one launch.
//   `moe_batched_down_accum_bf16`     : K-expert down + weighted accum one launch.
//   `moe_expert_gate_up_swiglu_bf16`  : per-expert gate+up+SwiGLU (reference path).
//   `moe_expert_down_bf16`            : per-expert down (reference path).
//
// BF16 row layout: each row of an [out_dim, in_dim] weight matrix is
// `in_dim * 2` bytes starting at `weight_base + row * in_dim * 2`. No
// blocks, no scales, no metadata; bf16-bits are loaded as `unsigned short`
// and bit-shifted into F32 (`bits << 16` reinterpreted as float).
//
// Algebraic equivalence: per-expert and batched paths produce byte-identical
// F32 outputs given identical inputs and weights (same dot-product order,
// same SwiGLU formulation, same weighted accumulation).
//
// NVRTC-compatible: no cuda_bf16.h. BF16 conversion uses the explicit
// bit-cast `((u32)b) << 16 -> __int_as_float`, identical to the proven
// `matvec_bf16.cu` pattern.

#define MOE_BLOCK_DIM 128
#define MOE_MAX_TOP_K 16

// BF16 -> F32: top 16 bits of an IEEE 754 binary32. Provably equivalent to
// PTX `cvt.f32.bf16` on SM_80+; works on SM_70 as well via the manual
// bit-cast.
__device__ __forceinline__ float mbf16_to_f32(unsigned short bits) {
    unsigned int x = ((unsigned int)bits) << 16;
    return __int_as_float((int)x);
}

// SwiGLU: silu(g) * u = (g * sigmoid(g)) * u.
__device__ __forceinline__ float mbf16_swiglu(float g, float u) {
    float silu_g = g / (1.0f + expf(-g));
    return silu_g * u;
}

// ============================================================================
// Per-expert kernels (reference path; one launch per (expert, token) pair).
// ============================================================================

// Per-expert gate + up + SwiGLU (BF16).
//
// Reads `gate` and `up` weight rows from `layer_buf` at the runtime-computed
// `gate_off` / `up_off` offsets. Both are row-major [inter_dim, hidden_dim] BF16.
//
// Output: `swiglu_out[row] = silu(gate_row · normed_x) * (up_row · normed_x)`.
extern "C" __global__ void moe_expert_gate_up_swiglu_bf16(
    const float* __restrict__ normed_x,             // [hidden_dim] F32
    const unsigned char* __restrict__ layer_buf,    // raw bytes
    unsigned long long gate_off,
    unsigned long long up_off,
    float* __restrict__ swiglu_out,                 // [inter_dim] F32
    unsigned int hidden_dim,
    unsigned int inter_dim)
{
    const unsigned int row = blockIdx.x * MOE_BLOCK_DIM + threadIdx.x;
    if (row >= inter_dim) return;

    // Each weight row is `hidden_dim` BF16 = `hidden_dim * 2` bytes.
    const size_t row_stride = (size_t)hidden_dim * 2u;

    const unsigned short* gate_row =
        reinterpret_cast<const unsigned short*>(layer_buf + gate_off + (size_t)row * row_stride);
    const unsigned short* up_row =
        reinterpret_cast<const unsigned short*>(layer_buf + up_off + (size_t)row * row_stride);

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (unsigned int j = 0; j < hidden_dim; ++j) {
        float xv = normed_x[j];
        gate_acc += mbf16_to_f32(gate_row[j]) * xv;
        up_acc   += mbf16_to_f32(up_row[j])   * xv;
    }
    swiglu_out[row] = mbf16_swiglu(gate_acc, up_acc);
}

// Per-expert down projection (BF16).
//
// Reads `down` weight rows from `layer_buf` at `down_off`. Row-major
// [hidden_dim, inter_dim] BF16. Output: `expert_out[row] = down_row · swiglu_in`.
extern "C" __global__ void moe_expert_down_bf16(
    const float* __restrict__ swiglu_in,            // [inter_dim] F32
    const unsigned char* __restrict__ layer_buf,    // raw bytes
    unsigned long long down_off,
    float* __restrict__ expert_out,                 // [hidden_dim] F32
    unsigned int hidden_dim,
    unsigned int inter_dim)
{
    const unsigned int row = blockIdx.x * MOE_BLOCK_DIM + threadIdx.x;
    if (row >= hidden_dim) return;

    const size_t row_stride = (size_t)inter_dim * 2u;
    const unsigned short* down_row =
        reinterpret_cast<const unsigned short*>(layer_buf + down_off + (size_t)row * row_stride);

    float dot = 0.0f;
    for (unsigned int j = 0; j < inter_dim; ++j) {
        dot += mbf16_to_f32(down_row[j]) * swiglu_in[j];
    }
    expert_out[row] = dot;
}

// ============================================================================
// Batched kernels (V1 pattern: one launch processes all K experts).
//
// These are the bandwidth-bound default for BF16 dispatch (mirrors the
// Q8_0 V1 path at moe_batched.cu lines 79-192). Same NR=1 / BLOCK_DIM=128
// row-per-thread pattern; a future revision can port the V2 cooperative-CTA
// pattern when perf headroom warrants.
// ============================================================================

// Batched gate+up+SwiGLU for K selected experts on one token (BF16).
//
// Grid: gridDim.x = ceil(inter_dim / BLOCK_DIM), gridDim.y = top_k.
// Each block computes one (k, inter_dim_tile) output tile for one expert.
extern "C" __global__ void moe_batched_gate_up_swiglu_bf16(
    const float* __restrict__ normed_x,             // [hidden_dim]
    const unsigned char* __restrict__ layer_buf,    // raw byte blob
    const unsigned int* __restrict__ expert_ids,    // [top_k]
    const unsigned long long* __restrict__ gate_up_offsets, // [num_experts * 2]
    float* __restrict__ swiglu_buf,                 // [top_k * inter_dim] F32
    unsigned int hidden_dim,
    unsigned int inter_dim,
    unsigned int top_k)
{
    const unsigned int k = blockIdx.y;
    if (k >= top_k) return;
    const unsigned int row = blockIdx.x * MOE_BLOCK_DIM + threadIdx.x;
    if (row >= inter_dim) return;

    const unsigned int expert_id = expert_ids[k];
    const unsigned long long gate_off = gate_up_offsets[(size_t)expert_id * 2 + 0];
    const unsigned long long up_off   = gate_up_offsets[(size_t)expert_id * 2 + 1];

    const size_t row_stride = (size_t)hidden_dim * 2u;
    const unsigned short* gate_row =
        reinterpret_cast<const unsigned short*>(layer_buf + gate_off + (size_t)row * row_stride);
    const unsigned short* up_row =
        reinterpret_cast<const unsigned short*>(layer_buf + up_off + (size_t)row * row_stride);

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (unsigned int j = 0; j < hidden_dim; ++j) {
        float xv = normed_x[j];
        gate_acc += mbf16_to_f32(gate_row[j]) * xv;
        up_acc   += mbf16_to_f32(up_row[j])   * xv;
    }

    swiglu_buf[(size_t)k * (size_t)inter_dim + row] = mbf16_swiglu(gate_acc, up_acc);
}

// Batched down + weighted accumulation for K selected experts on one token (BF16).
//
// Grid: gridDim.x = ceil(hidden_dim / BLOCK_DIM). One thread accumulates one
// output element across all K experts (replaces per-expert down loop +
// post-accum kernel).
extern "C" __global__ void moe_batched_down_accum_bf16(
    const float* __restrict__ swiglu_buf,           // [top_k * inter_dim]
    const unsigned char* __restrict__ layer_buf,    // raw bytes
    const unsigned int* __restrict__ expert_ids,    // [top_k]
    const unsigned long long* __restrict__ down_offsets, // [num_experts]
    const float* __restrict__ expert_weights,       // [top_k]
    const float* __restrict__ residual,             // [hidden_dim]
    float* __restrict__ x,                          // [hidden_dim]
    unsigned int hidden_dim,
    unsigned int inter_dim,
    unsigned int top_k)
{
    const unsigned int row = blockIdx.x * MOE_BLOCK_DIM + threadIdx.x;
    if (row >= hidden_dim) return;

    float weights[MOE_MAX_TOP_K];
    unsigned int eids[MOE_MAX_TOP_K];
    const unsigned int K = (top_k < MOE_MAX_TOP_K) ? top_k : MOE_MAX_TOP_K;
    for (unsigned int k = 0; k < K; ++k) {
        weights[k] = expert_weights[k];
        eids[k] = expert_ids[k];
    }

    const size_t row_stride = (size_t)inter_dim * 2u;

    float acc = residual[row];
    for (unsigned int k = 0; k < K; ++k) {
        unsigned int expert_id = eids[k];
        unsigned long long down_off = down_offsets[expert_id];
        const unsigned short* down_row = reinterpret_cast<const unsigned short*>(
            layer_buf + down_off + (size_t)row * row_stride);
        const float* swig_k = swiglu_buf + (size_t)k * (size_t)inter_dim;

        float dot = 0.0f;
        for (unsigned int j = 0; j < inter_dim; ++j) {
            dot += mbf16_to_f32(down_row[j]) * swig_k[j];
        }
        acc += weights[k] * dot;
    }
    x[row] = acc;
}

// ============================================================================
// cooperative-CTA-per-row-tile BF16 kernels (V3 pattern).
//
// The V1 batched kernels above use ONE THREAD per output row: gate+up launches
// only `ceil(inter_dim/128) * top_k = 4 * 8 = 32` CTAs of 128 threads (4096
// threads) and down launches `ceil(hidden/128) = 16` CTAs that each loop over
// all K experts serially. On A100 (108 SMs, 221k-thread capacity) this is
// severely under-occupied — the dominant reason BF16 decode is 20.4 tok/s while
// the bandwidth-comparable Q8 path (which has the V2/V3 cooperative kernels at
// `moe_batched.cu`) reaches 71.8.
//
// These V3-BF16 kernels are a direct port of `moe_batched_gate_up_swiglu_q8_0_v3`
// / `moe_batched_down_v3` (moe_batched.cu) to BF16 weights:
//   - Each CTA computes NR_BF16_V3 output rows COOPERATIVELY across
//     BLOCK_DIM_BF16_V3 threads (block-strided over the contraction dim with a
//     warp-shuffle + shared-memory tree reduction).
//   - gate+up grid = (ceil(inter_dim/NR), top_k) = (128, 8) = 1024 CTAs.
//   - down     grid = (ceil(hidden_dim/NR), top_k) = (512, 8) = 4096 CTAs.
//   This is ~32x more parallelism than V1, saturating the SMs.
//
// PRECISION: identical activation handling to V1 — the activation stays F32
// throughout (cached in shmem as F32, never rounded to BF16), only the BF16
// weight is bit-cast to F32 via `mbf16_to_f32`. This preserves the F32-precise
// dot product that keeps P3 coherent (the whole point of
// `LUMEN_CUDA_BF16_GEMMEX=0`). The ONLY numerical difference vs V1 is the
// summation order (warp-tree vs linear), a sub-1e-6 reassociation — validated
// against the V1 reference text at gen=128.
//
// NVRTC-safe: no cuda_bf16.h, manual bit-cast (matches matvec_bf16.cu).
// ============================================================================

#define BLOCK_DIM_BF16_V3 256   // 8 warps per CTA
#define NR_BF16_V3        4      // output rows per CTA

__device__ __forceinline__ float mb_bf16_v3_warp_reduce(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffffu, val, offset);
    }
    return val;
}

// Cooperative gate+up+SwiGLU for K experts (BF16, V3 NR-tiled).
//
// Grid: (ceil(inter_dim / NR_BF16_V3), top_k, 1). Block: (BLOCK_DIM_BF16_V3).
// Dynamic shmem: hidden_dim * 4 bytes (F32 normed_x cache, reused for reduction).
extern "C" __global__ void moe_batched_gate_up_swiglu_bf16_v3(
    const float* __restrict__ normed_x,             // [hidden_dim] F32
    const unsigned char* __restrict__ layer_buf,    // raw byte blob
    const unsigned int* __restrict__ expert_ids,    // [top_k]
    const unsigned long long* __restrict__ gate_up_offsets, // [num_experts * 2]
    float* __restrict__ swiglu_buf,                 // [top_k * inter_dim] F32
    unsigned int hidden_dim,
    unsigned int inter_dim,
    unsigned int top_k)
{
    extern __shared__ float nx_smem_bf16_v3[];  // [hidden_dim] F32

    const unsigned int k = blockIdx.y;
    if (k >= top_k) return;
    const unsigned int r0 = blockIdx.x * NR_BF16_V3;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31;
    const unsigned int warp_id = tid >> 5;
    const unsigned int num_warps = BLOCK_DIM_BF16_V3 / 32;

    const unsigned int expert_id = expert_ids[k];
    const unsigned long long gate_off = gate_up_offsets[(size_t)expert_id * 2 + 0];
    const unsigned long long up_off   = gate_up_offsets[(size_t)expert_id * 2 + 1];

    // Each BF16 weight row is `hidden_dim` BF16 = `hidden_dim * 2` bytes.
    const size_t row_stride = (size_t)hidden_dim * 2u;

    // Cooperatively cache normed_x in shmem (reused across all NR rows).
    for (unsigned int i = tid; i < hidden_dim; i += BLOCK_DIM_BF16_V3) {
        nx_smem_bf16_v3[i] = normed_x[i];
    }
    __syncthreads();

    float gate_sum[NR_BF16_V3];
    float up_sum[NR_BF16_V3];
    #pragma unroll
    for (int r = 0; r < NR_BF16_V3; r++) { gate_sum[r] = 0.0f; up_sum[r] = 0.0f; }

    // Block-strided over the contraction dim in pairs (2 BF16 per 32-bit load).
    const unsigned int aligned_hidden = hidden_dim & ~1u;
    for (unsigned int j = tid * 2u; j < aligned_hidden; j += BLOCK_DIM_BF16_V3 * 2u) {
        float x0 = nx_smem_bf16_v3[j];
        float x1 = nx_smem_bf16_v3[j + 1u];
        #pragma unroll
        for (int r = 0; r < NR_BF16_V3; r++) {
            if (r0 + r >= inter_dim) break;
            const unsigned short* grow = reinterpret_cast<const unsigned short*>(
                layer_buf + gate_off + (size_t)(r0 + r) * row_stride);
            const unsigned short* urow = reinterpret_cast<const unsigned short*>(
                layer_buf + up_off + (size_t)(r0 + r) * row_stride);
            unsigned int gp = *(const unsigned int*)(grow + j);
            unsigned int upp = *(const unsigned int*)(urow + j);
            gate_sum[r] += mbf16_to_f32((unsigned short)(gp & 0xffffu)) * x0
                         + mbf16_to_f32((unsigned short)(gp >> 16)) * x1;
            up_sum[r]   += mbf16_to_f32((unsigned short)(upp & 0xffffu)) * x0
                         + mbf16_to_f32((unsigned short)(upp >> 16)) * x1;
        }
    }
    // Odd trailing element (hidden_dim is even for Qwen3.5, but stay correct).
    if (hidden_dim & 1u) {
        unsigned int j = aligned_hidden + tid;
        if (j < hidden_dim) {
            float xv = nx_smem_bf16_v3[j];
            #pragma unroll
            for (int r = 0; r < NR_BF16_V3; r++) {
                if (r0 + r >= inter_dim) break;
                const unsigned short* grow = reinterpret_cast<const unsigned short*>(
                    layer_buf + gate_off + (size_t)(r0 + r) * row_stride);
                const unsigned short* urow = reinterpret_cast<const unsigned short*>(
                    layer_buf + up_off + (size_t)(r0 + r) * row_stride);
                gate_sum[r] += mbf16_to_f32(grow[j]) * xv;
                up_sum[r]   += mbf16_to_f32(urow[j]) * xv;
            }
        }
    }

    // ---- Two-level reduction: gate first, then up (reuse shmem). ----
    #pragma unroll
    for (int r = 0; r < NR_BF16_V3; r++) gate_sum[r] = mb_bf16_v3_warp_reduce(gate_sum[r]);
    __syncthreads();

    float* reduce_smem = nx_smem_bf16_v3;
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < NR_BF16_V3; r++)
            reduce_smem[r * num_warps + warp_id] = gate_sum[r];
    }
    __syncthreads();

    float final_gate[NR_BF16_V3];
    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < NR_BF16_V3; r++) {
            float val = (lane < num_warps) ? reduce_smem[r * num_warps + lane] : 0.0f;
            val = mb_bf16_v3_warp_reduce(val);
            final_gate[r] = val;
        }
    }
    __syncthreads();

    #pragma unroll
    for (int r = 0; r < NR_BF16_V3; r++) up_sum[r] = mb_bf16_v3_warp_reduce(up_sum[r]);
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < NR_BF16_V3; r++)
            reduce_smem[r * num_warps + warp_id] = up_sum[r];
    }
    __syncthreads();

    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < NR_BF16_V3; r++) {
            if (r0 + r < inter_dim) {
                float val = (lane < num_warps) ? reduce_smem[r * num_warps + lane] : 0.0f;
                val = mb_bf16_v3_warp_reduce(val);
                if (lane == 0) {
                    swiglu_buf[(size_t)k * (size_t)inter_dim + (r0 + r)] =
                        mbf16_swiglu(final_gate[r], val);
                }
            }
        }
    }
}

// Cooperative down projection for K experts (BF16, V3 NR-tiled).
//
// Writes per-(k, row) outputs to `down_out[top_k * hidden_dim]`; the existing
// `moe_expert_accum_option_a` kernel then performs residual + Σ_k weights[k]*out.
// This matches the Q8 V3 two-pass structure (down_v3 + accum) — no atomics.
//
// Grid: (ceil(hidden_dim / NR_BF16_V3), top_k, 1). Block: (BLOCK_DIM_BF16_V3).
// Dynamic shmem: inter_dim * 4 bytes (F32 swiglu cache, reused for reduction).
extern "C" __global__ void moe_batched_down_bf16_v3(
    const float* __restrict__ swiglu_buf,           // [top_k * inter_dim] F32
    const unsigned char* __restrict__ layer_buf,    // raw byte blob
    const unsigned int* __restrict__ expert_ids,    // [top_k]
    const unsigned long long* __restrict__ down_offsets, // [num_experts]
    float* __restrict__ down_out,                   // [top_k * hidden_dim] F32
    unsigned int hidden_dim,
    unsigned int inter_dim,
    unsigned int top_k)
{
    extern __shared__ float sw_smem_bf16_v3[];  // [inter_dim] F32

    const unsigned int k = blockIdx.y;
    if (k >= top_k) return;
    const unsigned int r0 = blockIdx.x * NR_BF16_V3;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31;
    const unsigned int warp_id = tid >> 5;
    const unsigned int num_warps = BLOCK_DIM_BF16_V3 / 32;

    const unsigned int expert_id = expert_ids[k];
    const unsigned long long down_off = down_offsets[expert_id];

    const size_t row_stride = (size_t)inter_dim * 2u;

    const float* swig_k = swiglu_buf + (size_t)k * (size_t)inter_dim;
    for (unsigned int i = tid; i < inter_dim; i += BLOCK_DIM_BF16_V3) {
        sw_smem_bf16_v3[i] = swig_k[i];
    }
    __syncthreads();

    float sum_r[NR_BF16_V3];
    #pragma unroll
    for (int r = 0; r < NR_BF16_V3; r++) sum_r[r] = 0.0f;

    const unsigned int aligned_inter = inter_dim & ~1u;
    for (unsigned int j = tid * 2u; j < aligned_inter; j += BLOCK_DIM_BF16_V3 * 2u) {
        float s0 = sw_smem_bf16_v3[j];
        float s1 = sw_smem_bf16_v3[j + 1u];
        #pragma unroll
        for (int r = 0; r < NR_BF16_V3; r++) {
            if (r0 + r >= hidden_dim) break;
            const unsigned short* drow = reinterpret_cast<const unsigned short*>(
                layer_buf + down_off + (size_t)(r0 + r) * row_stride);
            unsigned int dp = *(const unsigned int*)(drow + j);
            sum_r[r] += mbf16_to_f32((unsigned short)(dp & 0xffffu)) * s0
                      + mbf16_to_f32((unsigned short)(dp >> 16)) * s1;
        }
    }
    if (inter_dim & 1u) {
        unsigned int j = aligned_inter + tid;
        if (j < inter_dim) {
            float sv = sw_smem_bf16_v3[j];
            #pragma unroll
            for (int r = 0; r < NR_BF16_V3; r++) {
                if (r0 + r >= hidden_dim) break;
                const unsigned short* drow = reinterpret_cast<const unsigned short*>(
                    layer_buf + down_off + (size_t)(r0 + r) * row_stride);
                sum_r[r] += mbf16_to_f32(drow[j]) * sv;
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < NR_BF16_V3; r++) sum_r[r] = mb_bf16_v3_warp_reduce(sum_r[r]);
    __syncthreads();

    float* reduce_smem = sw_smem_bf16_v3;
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < NR_BF16_V3; r++)
            reduce_smem[r * num_warps + warp_id] = sum_r[r];
    }
    __syncthreads();

    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < NR_BF16_V3; r++) {
            if (r0 + r < hidden_dim) {
                float val = (lane < num_warps) ? reduce_smem[r * num_warps + lane] : 0.0f;
                val = mb_bf16_v3_warp_reduce(val);
                if (lane == 0) {
                    down_out[(size_t)k * (size_t)hidden_dim + (r0 + r)] = val;
                }
            }
        }
    }
}
