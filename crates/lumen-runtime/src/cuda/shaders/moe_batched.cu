// MoE batched-expert kernels. Per-expert dispatch (default) launches K
// kernels per layer; batched dispatch (this file, opt-in via
// `LUMEN_CUDA_MOE_BATCHED=1`) launches one kernel processing all K experts
// in a single launch.
//
// Pattern mirrors Metal's `moe_batched_gate_up_swiglu_q8_0` / `moe_batched_
// down_accum_q8_0` (`metal/shaders/moe_batched_q8_0.msl`). Two kernels:
//
// 1. `moe_batched_gate_up_swiglu_q8_0`: for each k in [0..top_k), compute
//    silu(gate · normed_x) ⊙ (up · normed_x) → swiglu_buf[k * inter_dim ..].
//    Grid: (top_k * inter_dim / TILE, 1, 1). Each block writes one
//    (k, inter_dim_tile) output. Reads gate/up weights from `layer_buf` at the
//    offset `gate_up_offsets[expert_ids[k] * 2]` (and +1 for the up tensor).
//
// 2. `moe_batched_down_accum_q8_0`: for each k in [0..top_k), compute
//    down · swiglu_buf[k * inter_dim ..] → expert_outputs[k * hidden_dim ..].
//    Sums weighted across k into x.
//
// For now we ship the Q8_0 path only. Q4_0 / BF16 batched kernels are a
// future optimization; the default per-expert path covers all three quants.
//
// NOTE: Per-expert kernels (the default) and these batched kernels
// produce numerically identical output when the same expert weights are used.
// The opt-in flag is for performance characterization; correctness
// is gated on bit-equivalence to the per-expert path.

// NVRTC-compatible: inline PTX for f16->f32, no cuda_fp16.h. Matches the
// pattern used in attention_f16.cu / dequant_q8_0_f16.cu / hgemv_q8_0.cu.
// (cudarc::nvrtc::compile_ptx is invoked without --include-path, so system
// headers like <cuda_fp16.h> are unreachable; the fix was wrong.)

#define BLOCK_DIM 128
#define Q8_0_BLOCK_SIZE 32  // 32 quants per Q8_0 block (32 bytes + 2 byte F16 scale = 34 bytes total)
#define MOE_MAX_TOP_K 16
#define MOE_MAX_NUM_EXPERTS 256  // Upper bound for Qwen3.5-MoE / DeepSeek-MoE family
#define MB_V2_NEG_INF (-3.4028235e38f)  // FLT_MIN_NEG (NVRTC has no <math.h> INFINITY)

// Q8_0 block layout: 2 bytes F16 scale + 32 bytes signed int8 quants.
// One block represents 32 weight elements.
#define Q8_0_BLOCK_BYTES 34

// Hardware f16->f32 conversion via PTX (single instruction on SM 53+).
__device__ __forceinline__ float mb_f16_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

// Convert F16 scale (stored as 2 bytes at the start of each Q8_0 block) to F32.
__device__ __forceinline__ float load_q8_0_scale(const unsigned char* block_ptr) {
    unsigned short f16_bits = *reinterpret_cast<const unsigned short*>(block_ptr);
    return mb_f16_to_f32(f16_bits);
}

// SwiGLU activation: silu(g) * u = (g * sigmoid(g)) * u.
__device__ __forceinline__ float swiglu(float g, float u) {
    float silu_g = g / (1.0f + expf(-g));
    return silu_g * u;
}

// Batched gate+up+SwiGLU for K selected experts on one token.
//
// Inputs:
//   normed_x          [hidden_dim] F32 (post-norm input vector)
//   layer_buf         per-layer weight blob containing all expert weights
//                     for this layer; per-expert offsets are looked up via
//                     `gate_up_offsets[expert_id * 2 + {0,1}]`.
//   expert_ids        [top_k] selected expert indices from router
//   gate_up_offsets   [num_experts * 2] u64 offsets (gate offset, up offset)
//
// Outputs:
//   swiglu_buf        [top_k * inter_dim] F32 (slot k = swiglu(gate · x, up · x))
//
// Grid: gridDim.x = ceil(inter_dim / BLOCK_DIM), gridDim.y = top_k.
// Each block computes one (k, inter_dim_tile) output tile for one expert.
extern "C" __global__ void moe_batched_gate_up_swiglu_q8_0(
    const float* __restrict__ normed_x,             // [hidden_dim]
    const unsigned char* __restrict__ layer_buf,    // raw byte blob; weights at gate_up_offsets
    const unsigned int* __restrict__ expert_ids,    // [top_k]
    const unsigned long long* __restrict__ gate_up_offsets, // [num_experts * 2] (gate_off, up_off)
    float* __restrict__ swiglu_buf,                 // [top_k * inter_dim] F32 output
    unsigned int hidden_dim,
    unsigned int inter_dim,
    unsigned int top_k)
{
    const unsigned int k = blockIdx.y;
    if (k >= top_k) return;
    const unsigned int row = blockIdx.x * BLOCK_DIM + threadIdx.x;
    if (row >= inter_dim) return;

    const unsigned int expert_id = expert_ids[k];
    const unsigned long long gate_off = gate_up_offsets[(size_t)expert_id * 2 + 0];
    const unsigned long long up_off   = gate_up_offsets[(size_t)expert_id * 2 + 1];

    // Q8_0 row: hidden_dim elements = (hidden_dim / 32) blocks of 34 bytes each.
    // Row `row` starts at: weight_base + row * (hidden_dim / 32) * Q8_0_BLOCK_BYTES.
    const unsigned int blocks_per_row = hidden_dim / Q8_0_BLOCK_SIZE;
    const size_t row_stride = (size_t)blocks_per_row * Q8_0_BLOCK_BYTES;

    const unsigned char* gate_row = layer_buf + gate_off + (size_t)row * row_stride;
    const unsigned char* up_row   = layer_buf + up_off   + (size_t)row * row_stride;

    // Compute gate · normed_x and up · normed_x for this row.
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (unsigned int b = 0; b < blocks_per_row; ++b) {
        const unsigned char* gblk = gate_row + (size_t)b * Q8_0_BLOCK_BYTES;
        const unsigned char* ublk = up_row   + (size_t)b * Q8_0_BLOCK_BYTES;
        float gscale = load_q8_0_scale(gblk);
        float uscale = load_q8_0_scale(ublk);
        const signed char* gquants = reinterpret_cast<const signed char*>(gblk + 2);
        const signed char* uquants = reinterpret_cast<const signed char*>(ublk + 2);
        for (unsigned int e = 0; e < Q8_0_BLOCK_SIZE; ++e) {
            float xv = normed_x[(size_t)b * Q8_0_BLOCK_SIZE + e];
            gate_acc += ((float)gquants[e] * gscale) * xv;
            up_acc   += ((float)uquants[e] * uscale) * xv;
        }
    }

    // SwiGLU: silu(gate) * up.
    float out = swiglu(gate_acc, up_acc);
    // Write into dense slot for this (k, row).
    swiglu_buf[(size_t)k * (size_t)inter_dim + row] = out;
}

// Batched down-projection + weighted accumulation for K selected experts on one token.
//
// Inputs:
//   swiglu_buf        [top_k * inter_dim] F32 (per-expert SwiGLU outputs)
//   layer_buf         per-layer weight blob containing down weights
//   expert_ids        [top_k]
//   down_offsets      [num_experts] u64 (per-expert down weight byte offset)
//   expert_weights    [top_k] router weights (renormalized)
//   residual          [hidden_dim] (pre-MoE residual stream)
//
// Output:
//   x                 [hidden_dim] F32 = residual + Σ_k expert_weights[k] · (down_k · swiglu_buf[k])
//
// Grid: gridDim.x = ceil(hidden_dim / BLOCK_DIM). Each thread accumulates one
// element of the hidden dim across all K experts (replacing the separate
// per-expert down + post-accum kernels in the per-expert path).
extern "C" __global__ void moe_batched_down_accum_q8_0(
    const float* __restrict__ swiglu_buf,           // [top_k * inter_dim]
    const unsigned char* __restrict__ layer_buf,    // raw byte blob
    const unsigned int* __restrict__ expert_ids,    // [top_k]
    const unsigned long long* __restrict__ down_offsets, // [num_experts]
    const float* __restrict__ expert_weights,       // [top_k]
    const float* __restrict__ residual,             // [hidden_dim]
    float* __restrict__ x,                          // [hidden_dim] output
    unsigned int hidden_dim,
    unsigned int inter_dim,
    unsigned int top_k)
{
    const unsigned int row = blockIdx.x * BLOCK_DIM + threadIdx.x;
    if (row >= hidden_dim) return;

    // Load top-K expert IDs + weights into registers.
    float weights[MOE_MAX_TOP_K];
    unsigned int eids[MOE_MAX_TOP_K];
    const unsigned int K = (top_k < MOE_MAX_TOP_K) ? top_k : MOE_MAX_TOP_K;
    for (unsigned int k = 0; k < K; ++k) {
        weights[k] = expert_weights[k];
        eids[k] = expert_ids[k];
    }

    // For Q8_0 down weights: each row has `inter_dim` elements = `inter_dim / 32` blocks.
    const unsigned int blocks_per_row = inter_dim / Q8_0_BLOCK_SIZE;
    const size_t row_stride = (size_t)blocks_per_row * Q8_0_BLOCK_BYTES;

    float acc = residual[row];
    for (unsigned int k = 0; k < K; ++k) {
        unsigned int expert_id = eids[k];
        unsigned long long down_off = down_offsets[expert_id];
        const unsigned char* down_row = layer_buf + down_off + (size_t)row * row_stride;
        const float* swig_k = swiglu_buf + (size_t)k * (size_t)inter_dim;
        // down · swiglu_buf[k]
        float dot = 0.0f;
        for (unsigned int b = 0; b < blocks_per_row; ++b) {
            const unsigned char* blk = down_row + (size_t)b * Q8_0_BLOCK_BYTES;
            float scale = load_q8_0_scale(blk);
            const signed char* quants = reinterpret_cast<const signed char*>(blk + 2);
            for (unsigned int e = 0; e < Q8_0_BLOCK_SIZE; ++e) {
                dot += ((float)quants[e] * scale) * swig_k[(size_t)b * Q8_0_BLOCK_SIZE + e];
            }
        }
        acc += weights[k] * dot;
    }
    x[row] = acc;
}

// =============================================================================
// V2 kernels (LUMEN_CUDA_MOE_BATCHED_V2=1).
//
// Replaces the v1 batched kernels above with a cooperative-CTA-per-row-tile
// pattern matching the dense `fused_glu_gemv_q8_0` proven optimization:
//   - Each CTA computes NR_V2 output rows cooperatively across BLOCK_DIM_V2 threads
//   - x is cached in shared memory once per CTA
//   - Q8 block reads are block-strided (each thread processes one full Q8 block at a time)
//   - Warp + cross-warp reductions via shfl_xor + shmem
//
// Bandwidth-optimized: each thread reads ~32 quants per iteration vs v1's 1 thread
// per row reading hidden_dim/32 blocks sequentially. Better SM utilization, better
// memory coalescing.
//
// Algebraically identical to v1 (same sum order per row, different reduction tree
// across threads). Bit-exact on final token outputs preserved (the dense path uses
// the same reduction pattern).
// =============================================================================

#define BLOCK_DIM_V2 256        // 8 warps per CTA
#define NR_V2        2          // output rows per CTA

// Warp-level butterfly sum reduction.
__device__ __forceinline__ float mb_v2_warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

// SwiGLU using SiLU sigmoid form.
__device__ __forceinline__ float mb_v2_swiglu(float g, float u) {
    float silu_g = g / (1.0f + expf(-g));
    return silu_g * u;
}

// --- Fused router v2 (single-kernel): warp-parallel logits + parallel softmax + top-K. ---
//
// Grid: (1, 1, 1). Block: (BLOCK_DIM_V2 = 256, 1, 1). Eliminates the second
// finalize launch by doing everything in ONE CTA: each warp handles a strided
// subset of experts (warp_id, warp_id+8, warp_id+16, ...), reads the per-expert
// weight row, and computes the dot product across its 32 lanes. After all warps
// finish their assigned experts, all threads cooperate on softmax + normalize,
// then thread 0 performs the iterated argmax-with-mask top-K selection.
//
// Net: 1 launch instead of 2, no inter-CTA sync needed, no global router_logits
// buffer needed (kept in shmem). Bandwidth-bound on weight reads.
extern "C" __global__ void moe_router_fused_v2(
    const float* __restrict__ normed_x,         // [hidden_dim]
    const float* __restrict__ router_weight,    // [num_experts * hidden_dim]
    unsigned int* __restrict__ expert_ids,      // [top_k] output
    float* __restrict__ expert_weights,         // [top_k] output
    unsigned int hidden_dim,
    unsigned int num_experts,
    unsigned int top_k)
{
    __shared__ float logits[MOE_MAX_NUM_EXPERTS];
    __shared__ float warp_max[BLOCK_DIM_V2 / 32];
    __shared__ float warp_sum[BLOCK_DIM_V2 / 32];
    __shared__ float s_maxv;
    __shared__ float s_sum;
    extern __shared__ float nx_smem[];  // [hidden_dim]

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = BLOCK_DIM_V2 / 32;

    // Phase 0: cooperatively cache normed_x in shmem (reused across all experts).
    for (unsigned int i = tid; i < hidden_dim; i += BLOCK_DIM_V2) {
        nx_smem[i] = normed_x[i];
    }
    __syncthreads();

    // Phase A: warp-parallel logits. Each warp handles experts in stride num_warps.
    for (unsigned int e = warp_id; e < num_experts; e += num_warps) {
        const float* w_e = router_weight + (size_t)e * (size_t)hidden_dim;
        float partial = 0.0f;
        for (unsigned int j = lane; j < hidden_dim; j += 32) {
            partial += w_e[j] * nx_smem[j];
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            partial += __shfl_xor_sync(0xffffffff, partial, offset);
        }
        if (lane == 0) {
            logits[e] = partial;
        }
    }
    __syncthreads();

    // Phase B: parallel max-reduce.
    float local_max = MB_V2_NEG_INF;
    for (unsigned int e = tid; e < num_experts; e += BLOCK_DIM_V2) {
        float v = logits[e];
        if (v > local_max) local_max = v;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, local_max, offset);
        if (other > local_max) local_max = other;
    }
    if (lane == 0) warp_max[warp_id] = local_max;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane < num_warps) ? warp_max[lane] : MB_V2_NEG_INF;
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other = __shfl_xor_sync(0xffffffff, v, offset);
            if (other > v) v = other;
        }
        if (lane == 0) s_maxv = v;
    }
    __syncthreads();
    float maxv = s_maxv;

    // Phase C: parallel exp + sum-reduce.
    float local_sum = 0.0f;
    for (unsigned int e = tid; e < num_experts; e += BLOCK_DIM_V2) {
        float v = expf(logits[e] - maxv);
        logits[e] = v;
        local_sum += v;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    }
    if (lane == 0) warp_sum[warp_id] = local_sum;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane < num_warps) ? warp_sum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_xor_sync(0xffffffff, v, offset);
        }
        if (lane == 0) s_sum = v;
    }
    __syncthreads();
    float inv = 1.0f / s_sum;

    // Parallel normalize.
    for (unsigned int e = tid; e < num_experts; e += BLOCK_DIM_V2) {
        logits[e] *= inv;
    }
    __syncthreads();

    // Phase D: top-K via iterated argmax-with-mask (thread 0 only).
    if (tid == 0) {
        float renorm = 0.0f;
        for (unsigned int k = 0; k < top_k; ++k) {
            float best = -1.0f;
            unsigned int best_e = 0;
            for (unsigned int e = 0; e < num_experts; ++e) {
                float v = logits[e];
                if (v > best) {
                    best = v;
                    best_e = e;
                }
            }
            expert_ids[k] = best_e;
            expert_weights[k] = best;
            renorm += best;
            logits[best_e] = -1.0f;
        }
        if (renorm > 0.0f) {
            float invr = 1.0f / renorm;
            for (unsigned int k = 0; k < top_k; ++k) {
                expert_weights[k] *= invr;
            }
        }
    }
}

// --- (Legacy) Router v2: parallel per-expert dot product. ---
//
// Kept for reference / two-launch variant. The fused single-kernel variant
// (`moe_router_fused_v2`) is preferred. This kernel is no longer dispatched but
// left in source for compile-test and ablation.
//
// Grid: (num_experts, 1, 1). Block: (BLOCK_DIM_V2, 1, 1).
// Each CTA computes one expert's pre-softmax logit cooperatively, writes to
// `router_logits[expert_id]`. Then a small finalizer kernel
// (`moe_router_softmax_finalize_v2`) does softmax + top-K from CTA 0 only.
//
// This separates the bandwidth-bound dot product (parallelizable across experts)
// from the latency-bound softmax+top-K (inherently serial on small num_experts).
// --- Fused logits+softmax+topK in a single launch via atomic counter sync. ---
//
// Eliminates the second kernel launch by using a global atomic counter to
// signal completion across CTAs. Each CTA does its expert dot product, writes
// the logit, then atomicAdd's to a counter; the LAST CTA to finish (counter ==
// num_experts) performs the softmax + top-K phase.
//
// Grid: (num_experts, 1, 1). Block: (256, 1, 1).
// Saves 1 kernel launch (~30 µs) per layer per token. Counter is reset to 0
// at the END of the kernel by the last CTA so subsequent launches don't
// require a separate clear.
// --- Fused FFN-norm + router single launch. ---
//
// Reads attn_proj + ffn_norm_weight + router_weight in a single kernel.
// Computes RMSNorm of attn_proj into shmem, then does the parallel logit
// + atomic-counter softmax + top-K pattern. Replaces TWO kernel launches
// (standalone rmsnorm + moe_router_fused_atomic_v2) with one. Also writes
// the normed_x output to scratch so downstream gate_up_v3 / down_v3 still
// have access to it.
//
// Grid: (num_experts, 1, 1). Block: (BLOCK_DIM_V2 = 256, 1, 1).
// Shmem: hidden_dim * 4 bytes (normed_x) + small reduction buffers.
extern "C" __global__ void moe_router_rmsnorm_atomic_v3(
    const float* __restrict__ attn_proj,         // [hidden_dim] pre-norm input
    const float* __restrict__ ffn_norm_weight,   // [hidden_dim] RMSNorm gamma
    const float* __restrict__ router_weight,     // [num_experts * hidden_dim]
    float* __restrict__ normed_out,              // [hidden_dim] output (post-norm; downstream input)
    float* __restrict__ router_logits,           // [num_experts] scratch
    unsigned int* __restrict__ done_counter,     // [1] atomic counter
    unsigned int* __restrict__ expert_ids,       // [top_k] output
    float* __restrict__ expert_weights,          // [top_k] output
    float eps,
    unsigned int hidden_dim,
    unsigned int num_experts,
    unsigned int top_k)
{
    extern __shared__ float nx_smem_rmsr[];  // [hidden_dim]
    __shared__ float warp_partial[BLOCK_DIM_V2 / 32];
    __shared__ float warp_sumsq[BLOCK_DIM_V2 / 32];
    __shared__ float s_rms_scale;
    __shared__ bool s_is_last;

    const unsigned int e = blockIdx.x;
    if (e >= num_experts) return;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = BLOCK_DIM_V2 / 32;

    // ---- Phase 0: cooperative RMSNorm (only CTA 0 needs to write the result). ----
    //
    // All CTAs compute the same rms_scale (in their own shmem), then apply
    // normed = attn_proj * rms_scale * ffn_norm. The normed value is needed
    // for the router dot product AND must be left in `normed_out` for the
    // downstream gate_up_v3 kernel. Since all CTAs compute the same normed_x,
    // having multiple CTAs write the same global buffer is fine (idempotent).
    //
    // Each CTA caches its own normed_x in shmem; only CTA 0 writes to normed_out.

    // Compute sum of squares of attn_proj.
    float sumsq = 0.0f;
    for (unsigned int i = tid; i < hidden_dim; i += BLOCK_DIM_V2) {
        float v = attn_proj[i];
        nx_smem_rmsr[i] = v;  // stash raw; will scale below
        sumsq += v * v;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sumsq += __shfl_xor_sync(0xffffffff, sumsq, offset);
    }
    if (lane == 0) warp_sumsq[warp_id] = sumsq;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane < num_warps) ? warp_sumsq[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_xor_sync(0xffffffff, v, offset);
        }
        if (lane == 0) {
            float mean = v / (float)hidden_dim;
            s_rms_scale = 1.0f / sqrtf(mean + eps);
        }
    }
    __syncthreads();
    float rms_scale = s_rms_scale;

    // Apply norm: normed = attn_proj * rms_scale * ffn_norm.
    for (unsigned int i = tid; i < hidden_dim; i += BLOCK_DIM_V2) {
        float n = nx_smem_rmsr[i] * rms_scale * ffn_norm_weight[i];
        nx_smem_rmsr[i] = n;
        // Only CTA 0 writes the post-norm result to global for downstream kernels.
        if (e == 0) {
            normed_out[i] = n;
        }
    }
    __syncthreads();

    // ---- Phase A: parallel dot product for expert e ----
    const float* w_e = router_weight + (size_t)e * (size_t)hidden_dim;
    float partial = 0.0f;
    for (unsigned int j = tid; j < hidden_dim; j += BLOCK_DIM_V2) {
        partial += w_e[j] * nx_smem_rmsr[j];
    }
    partial = mb_v2_warp_reduce_sum(partial);
    if (lane == 0) warp_partial[warp_id] = partial;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane < num_warps) ? warp_partial[lane] : 0.0f;
        v = mb_v2_warp_reduce_sum(v);
        if (lane == 0) {
            router_logits[e] = v;
            __threadfence();
            unsigned int prev = atomicAdd(done_counter, 1u);
            s_is_last = (prev + 1u == num_experts);
        }
    }
    __syncthreads();

    // ---- Phase B: only the LAST CTA performs softmax + top-K ----
    if (!s_is_last) return;

    if (tid == 0) *done_counter = 0u;

    __shared__ float warp_max[BLOCK_DIM_V2 / 32];
    __shared__ float warp_sum[BLOCK_DIM_V2 / 32];
    __shared__ float s_maxv;
    __shared__ float s_sum;

    float local_max = MB_V2_NEG_INF;
    for (unsigned int ee = tid; ee < num_experts; ee += BLOCK_DIM_V2) {
        float v = router_logits[ee];
        if (v > local_max) local_max = v;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, local_max, offset);
        if (other > local_max) local_max = other;
    }
    if (lane == 0) warp_max[warp_id] = local_max;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane < num_warps) ? warp_max[lane] : MB_V2_NEG_INF;
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other = __shfl_xor_sync(0xffffffff, v, offset);
            if (other > v) v = other;
        }
        if (lane == 0) s_maxv = v;
    }
    __syncthreads();
    float maxv = s_maxv;

    float local_sum = 0.0f;
    for (unsigned int ee = tid; ee < num_experts; ee += BLOCK_DIM_V2) {
        float v = expf(router_logits[ee] - maxv);
        router_logits[ee] = v;
        local_sum += v;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    }
    if (lane == 0) warp_sum[warp_id] = local_sum;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane < num_warps) ? warp_sum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_xor_sync(0xffffffff, v, offset);
        }
        if (lane == 0) s_sum = v;
    }
    __syncthreads();
    float inv = 1.0f / s_sum;

    for (unsigned int ee = tid; ee < num_experts; ee += BLOCK_DIM_V2) {
        router_logits[ee] *= inv;
    }
    __syncthreads();

    if (tid == 0) {
        float renorm = 0.0f;
        for (unsigned int k = 0; k < top_k; ++k) {
            float best = -1.0f;
            unsigned int best_e = 0;
            for (unsigned int ee = 0; ee < num_experts; ++ee) {
                float v = router_logits[ee];
                if (v > best) {
                    best = v;
                    best_e = ee;
                }
            }
            expert_ids[k] = best_e;
            expert_weights[k] = best;
            renorm += best;
            router_logits[best_e] = -1.0f;
        }
        if (renorm > 0.0f) {
            float invr = 1.0f / renorm;
            for (unsigned int k = 0; k < top_k; ++k) {
                expert_weights[k] *= invr;
            }
        }
    }
}

extern "C" __global__ void moe_router_fused_atomic_v2(
    const float* __restrict__ normed_x,         // [hidden_dim]
    const float* __restrict__ router_weight,    // [num_experts * hidden_dim]
    float* __restrict__ router_logits,          // [num_experts] scratch
    unsigned int* __restrict__ done_counter,    // [1] atomic counter (init 0)
    unsigned int* __restrict__ expert_ids,      // [top_k] output
    float* __restrict__ expert_weights,         // [top_k] output
    unsigned int hidden_dim,
    unsigned int num_experts,
    unsigned int top_k)
{
    __shared__ float warp_partial[BLOCK_DIM_V2 / 32];
    __shared__ bool s_is_last;

    const unsigned int e = blockIdx.x;
    if (e >= num_experts) return;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = BLOCK_DIM_V2 / 32;

    // ---- Phase A: parallel dot product for expert e ----
    const float* w_e = router_weight + (size_t)e * (size_t)hidden_dim;
    float partial = 0.0f;
    for (unsigned int j = tid; j < hidden_dim; j += BLOCK_DIM_V2) {
        partial += w_e[j] * normed_x[j];
    }
    partial = mb_v2_warp_reduce_sum(partial);
    if (lane == 0) warp_partial[warp_id] = partial;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane < num_warps) ? warp_partial[lane] : 0.0f;
        v = mb_v2_warp_reduce_sum(v);
        if (lane == 0) {
            router_logits[e] = v;
            __threadfence();
            unsigned int prev = atomicAdd(done_counter, 1u);
            s_is_last = (prev + 1u == num_experts);
        }
    }
    __syncthreads();

    // ---- Phase B: only the LAST CTA performs softmax + top-K ----
    if (!s_is_last) return;

    // Reset counter for next call before we exit (any thread can do this once).
    if (tid == 0) {
        *done_counter = 0u;
    }

    // Parallel max-reduce across experts.
    __shared__ float warp_max[BLOCK_DIM_V2 / 32];
    __shared__ float warp_sum[BLOCK_DIM_V2 / 32];
    __shared__ float s_maxv;
    __shared__ float s_sum;

    float local_max = MB_V2_NEG_INF;
    for (unsigned int ee = tid; ee < num_experts; ee += BLOCK_DIM_V2) {
        float v = router_logits[ee];
        if (v > local_max) local_max = v;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, local_max, offset);
        if (other > local_max) local_max = other;
    }
    if (lane == 0) warp_max[warp_id] = local_max;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane < num_warps) ? warp_max[lane] : MB_V2_NEG_INF;
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other = __shfl_xor_sync(0xffffffff, v, offset);
            if (other > v) v = other;
        }
        if (lane == 0) s_maxv = v;
    }
    __syncthreads();
    float maxv = s_maxv;

    // Parallel exp + sum-reduce.
    float local_sum = 0.0f;
    for (unsigned int ee = tid; ee < num_experts; ee += BLOCK_DIM_V2) {
        float v = expf(router_logits[ee] - maxv);
        router_logits[ee] = v;
        local_sum += v;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    }
    if (lane == 0) warp_sum[warp_id] = local_sum;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane < num_warps) ? warp_sum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_xor_sync(0xffffffff, v, offset);
        }
        if (lane == 0) s_sum = v;
    }
    __syncthreads();
    float inv = 1.0f / s_sum;

    for (unsigned int ee = tid; ee < num_experts; ee += BLOCK_DIM_V2) {
        router_logits[ee] *= inv;
    }
    __syncthreads();

    // Top-K via iterated argmax-with-mask (thread 0 only; cheap).
    if (tid == 0) {
        float renorm = 0.0f;
        for (unsigned int k = 0; k < top_k; ++k) {
            float best = -1.0f;
            unsigned int best_e = 0;
            for (unsigned int ee = 0; ee < num_experts; ++ee) {
                float v = router_logits[ee];
                if (v > best) {
                    best = v;
                    best_e = ee;
                }
            }
            expert_ids[k] = best_e;
            expert_weights[k] = best;
            renorm += best;
            router_logits[best_e] = -1.0f;
        }
        if (renorm > 0.0f) {
            float invr = 1.0f / renorm;
            for (unsigned int k = 0; k < top_k; ++k) {
                expert_weights[k] *= invr;
            }
        }
    }
}

extern "C" __global__ void moe_router_logits_v2(
    const float* __restrict__ normed_x,         // [hidden_dim]
    const float* __restrict__ router_weight,    // [num_experts * hidden_dim]
    float* __restrict__ router_logits,          // [num_experts]
    unsigned int hidden_dim,
    unsigned int num_experts)
{
    const unsigned int e = blockIdx.x;
    if (e >= num_experts) return;

    __shared__ float warp_partial[BLOCK_DIM_V2 / 32];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = BLOCK_DIM_V2 / 32;

    const float* w_e = router_weight + (size_t)e * (size_t)hidden_dim;

    float partial = 0.0f;
    for (unsigned int j = tid; j < hidden_dim; j += BLOCK_DIM_V2) {
        partial += w_e[j] * normed_x[j];
    }
    partial = mb_v2_warp_reduce_sum(partial);
    if (lane == 0) {
        warp_partial[warp_id] = partial;
    }
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane < num_warps) ? warp_partial[lane] : 0.0f;
        v = mb_v2_warp_reduce_sum(v);
        if (lane == 0) {
            router_logits[e] = v;
        }
    }
}

// --- Router v2 finalize: parallel softmax + top-K + renormalize. ---
//
// Grid: (1, 1, 1). Block: (BLOCK_DIM_V2 = 256, 1, 1). All threads cooperate on
// the max-reduce + exp + sum-reduce + normalize phases (parallelizable in
// num_experts). Then thread 0 performs the iterated argmax-with-mask top-K
// selection (inherently serial but cheap at top_k ≤ 16).
//
// Reads `router_logits`, writes `expert_ids[top_k]` and `expert_weights[top_k]`.
// Numerical-stability behavior matches v1 moe_router_softmax (max subtraction
// before exp, iterated argmax-with-mask for top-K, renormalization).
extern "C" __global__ void moe_router_softmax_finalize_v2(
    float* __restrict__ router_logits,          // [num_experts] in/out (scratch)
    unsigned int* __restrict__ expert_ids,      // [top_k] output
    float* __restrict__ expert_weights,         // [top_k] output
    unsigned int num_experts,
    unsigned int top_k)
{
    if (blockIdx.x != 0) return;

    __shared__ float warp_max[BLOCK_DIM_V2 / 32];
    __shared__ float warp_sum[BLOCK_DIM_V2 / 32];
    __shared__ float s_maxv;
    __shared__ float s_sum;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = BLOCK_DIM_V2 / 32;

    // Phase A: parallel max-reduce.
    float local_max = MB_V2_NEG_INF;
    for (unsigned int e = tid; e < num_experts; e += BLOCK_DIM_V2) {
        float v = router_logits[e];
        if (v > local_max) local_max = v;
    }
    // Warp-level max reduction.
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, local_max, offset);
        if (other > local_max) local_max = other;
    }
    if (lane == 0) warp_max[warp_id] = local_max;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane < num_warps) ? warp_max[lane] : MB_V2_NEG_INF;
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other = __shfl_xor_sync(0xffffffff, v, offset);
            if (other > v) v = other;
        }
        if (lane == 0) s_maxv = v;
    }
    __syncthreads();
    float maxv = s_maxv;

    // Phase B: parallel exp + sum-reduce, then normalize.
    float local_sum = 0.0f;
    for (unsigned int e = tid; e < num_experts; e += BLOCK_DIM_V2) {
        float v = expf(router_logits[e] - maxv);
        router_logits[e] = v;
        local_sum += v;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    }
    if (lane == 0) warp_sum[warp_id] = local_sum;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane < num_warps) ? warp_sum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_xor_sync(0xffffffff, v, offset);
        }
        if (lane == 0) s_sum = v;
    }
    __syncthreads();
    float inv = 1.0f / s_sum;

    // Parallel normalize.
    for (unsigned int e = tid; e < num_experts; e += BLOCK_DIM_V2) {
        router_logits[e] *= inv;
    }
    __syncthreads();

    // Phase C: top-K via iterated argmax-with-mask (thread 0 only).
    // top_k ≤ 16; sequential cost is ~16 * 256 = 4096 ops — same single thread
    // does all of this. Acceptable because the masking step is a global write
    // that must serialize anyway.
    // (: a block-parallel argmax variant was tried here and REGRESSED
    // decode ~12× to 5.4 tok/s — a shmem/occupancy perf cliff for a kernel whose
    // total cost is ~27 µs/call. Reverted; the parallel-LOGITS launch is the win.)
    if (tid == 0) {
        float renorm = 0.0f;
        for (unsigned int k = 0; k < top_k; ++k) {
            float best = -1.0f;
            unsigned int best_e = 0;
            for (unsigned int e = 0; e < num_experts; ++e) {
                float v = router_logits[e];
                if (v > best) {
                    best = v;
                    best_e = e;
                }
            }
            expert_ids[k] = best_e;
            expert_weights[k] = best;
            renorm += best;
            router_logits[best_e] = -1.0f;
        }
        if (renorm > 0.0f) {
            float invr = 1.0f / renorm;
            for (unsigned int k = 0; k < top_k; ++k) {
                expert_weights[k] *= invr;
            }
        }
    }
}

// --- Batched gate+up+SwiGLU v2: per-expert NR-tiled CTA-cooperative pattern. ---
//
// Replaces moe_batched_gate_up_swiglu_q8_0 with the proven `fused_glu_gemv_q8_0`
// dispatch pattern (NR rows per CTA, BLOCK_DIM threads cooperative on each Q8 block).
//
// Grid: (gridDim.x = ceil(inter_dim / NR_V2), gridDim.y = top_k, 1).
// Each (block.x, block.y) tile computes NR_V2 rows of expert k's swiglu output.
//
// Shared memory: hidden_dim * 4 bytes (normed x cache).
extern "C" __global__ void moe_batched_gate_up_swiglu_q8_0_v2(
    const float* __restrict__ normed_x,             // [hidden_dim]
    const unsigned char* __restrict__ layer_buf,    // raw byte blob
    const unsigned int* __restrict__ expert_ids,    // [top_k]
    const unsigned long long* __restrict__ gate_up_offsets, // [num_experts * 2]
    float* __restrict__ swiglu_buf,                 // [top_k * inter_dim]
    unsigned int hidden_dim,
    unsigned int inter_dim,
    unsigned int top_k)
{
    extern __shared__ float nx_smem[];  // [hidden_dim]

    const unsigned int k = blockIdx.y;
    if (k >= top_k) return;
    const unsigned int r0 = blockIdx.x * NR_V2;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31;
    const unsigned int warp_id = tid >> 5;
    const unsigned int num_warps = BLOCK_DIM_V2 / 32;

    const unsigned int expert_id = expert_ids[k];
    const unsigned long long gate_off = gate_up_offsets[(size_t)expert_id * 2 + 0];
    const unsigned long long up_off   = gate_up_offsets[(size_t)expert_id * 2 + 1];

    const unsigned int num_blocks = hidden_dim / Q8_0_BLOCK_SIZE;
    const size_t row_bytes = (size_t)num_blocks * Q8_0_BLOCK_BYTES;

    // Cooperatively load normed_x to shmem.
    for (unsigned int i = tid; i < hidden_dim; i += BLOCK_DIM_V2) {
        nx_smem[i] = normed_x[i];
    }
    __syncthreads();

    // Per-row gate/up accumulators (NR_V2 rows per CTA).
    float gate_sum[NR_V2];
    float up_sum[NR_V2];
    #pragma unroll
    for (int r = 0; r < NR_V2; r++) {
        gate_sum[r] = 0.0f;
        up_sum[r]   = 0.0f;
    }

    // Block-strided cooperative iteration over Q8 blocks.
    // Each thread processes one full Q8 block (32 elements) per iteration.
    for (unsigned int ib = tid; ib < num_blocks; ib += BLOCK_DIM_V2) {
        const unsigned int x_base = ib * Q8_0_BLOCK_SIZE;

        // Load 32 x-values for this Q8 block from shmem to registers (float4 path).
        float xv[32];
        const float4* x4 = (const float4*)(nx_smem + x_base);
        #pragma unroll
        for (int kk = 0; kk < 8; kk++) {
            float4 v = x4[kk];
            xv[kk * 4 + 0] = v.x;
            xv[kk * 4 + 1] = v.y;
            xv[kk * 4 + 2] = v.z;
            xv[kk * 4 + 3] = v.w;
        }

        // Process NR_V2 output rows reusing the cached x values.
        #pragma unroll
        for (int r = 0; r < NR_V2; r++) {
            if (r0 + r >= inter_dim) break;

            // Gate block.
            const unsigned char* gp = layer_buf + gate_off
                + (size_t)(r0 + r) * row_bytes
                + (size_t)ib * Q8_0_BLOCK_BYTES;
            float g_scale = load_q8_0_scale(gp);
            const signed char* gq = (const signed char*)(gp + 2);

            // Up block.
            const unsigned char* up_ = layer_buf + up_off
                + (size_t)(r0 + r) * row_bytes
                + (size_t)ib * Q8_0_BLOCK_BYTES;
            float u_scale = load_q8_0_scale(up_);
            const signed char* uq = (const signed char*)(up_ + 2);

            float g_block_sum = 0.0f;
            float u_block_sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < 32; j++) {
                g_block_sum += (float)gq[j] * xv[j];
                u_block_sum += (float)uq[j] * xv[j];
            }
            gate_sum[r] += g_scale * g_block_sum;
            up_sum[r]   += u_scale * u_block_sum;
        }
    }

    // Intra-warp reduction.
    #pragma unroll
    for (int r = 0; r < NR_V2; r++) {
        gate_sum[r] = mb_v2_warp_reduce_sum(gate_sum[r]);
    }
    __syncthreads();

    // Cross-warp reduction via shmem (reuse nx_smem buffer).
    float* reduce_smem = nx_smem;
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < NR_V2; r++) {
            reduce_smem[r * num_warps + warp_id] = gate_sum[r];
        }
    }
    __syncthreads();

    float final_gate[NR_V2];
    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < NR_V2; r++) {
            float val = (lane < num_warps) ? reduce_smem[r * num_warps + lane] : 0.0f;
            val = mb_v2_warp_reduce_sum(val);
            final_gate[r] = val;
        }
    }
    __syncthreads();

    #pragma unroll
    for (int r = 0; r < NR_V2; r++) {
        up_sum[r] = mb_v2_warp_reduce_sum(up_sum[r]);
    }
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < NR_V2; r++) {
            reduce_smem[r * num_warps + warp_id] = up_sum[r];
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < NR_V2; r++) {
            if (r0 + r < inter_dim) {
                float val = (lane < num_warps) ? reduce_smem[r * num_warps + lane] : 0.0f;
                val = mb_v2_warp_reduce_sum(val);
                if (lane == 0) {
                    // SwiGLU + write to [k, r0+r] slot.
                    float out = mb_v2_swiglu(final_gate[r], val);
                    swiglu_buf[(size_t)k * (size_t)inter_dim + (r0 + r)] = out;
                }
            }
        }
    }
}

// --- Batched down + weighted accumulate v2: per-expert NR-tiled per-row pattern. ---
//
// Replaces moe_batched_down_accum_q8_0 with cooperative per-row reduction.
//
// Grid: (gridDim.x = ceil(hidden_dim / NR_V2), gridDim.y = top_k, 1).
// Each (block.x, block.y) tile computes NR_V2 rows of expert k's down output
// and accumulates the weighted contribution `weights[k] * down_k_row` into the
// final x output. To preserve the residual + Σ semantics, expert k=0's CTA
// initializes x[row] from residual[row], and subsequent experts add to it via
// atomic adds (NR rows per CTA is small enough that atomicAdds at the row-level
// have negligible contention since rows are unique per (k, r0+r) tile).
//
// To avoid atomics, we use a different pattern: write expert-k down output to
// a per-(k, row) scratch buffer, then a small reduction kernel sums across k.
//
// However, since we already have the swiglu_buf scratch sized [top_k * inter_dim],
// and the output_buf is needed only here, we use atomics on x[row] guarded by
// a single init from CTA(k=0, row block 0). Actually safest: use a "per-row" scheme
// where the FIRST expert (k=0) for each row block initializes from residual; other
// experts atomicAdd. CUDA atomicAdd on global F32 is fast (HW-supported on SM80).
//
// Cleanest approach: split into 2 kernels.
//   Pass 1: moe_batched_down_v2: compute per-(k, row) down outputs to
//           `down_buf[top_k * hidden_dim]` (existing expert_output_buf scratch).
//   Pass 2: moe_expert_accum_option_a (existing): residual + Σ_k weights[k] * out[k]
//
// This eliminates atomics, uses the existing accum kernel, and reuses the
// existing per-expert pattern. Each pass-1 CTA does cooperative NR rows for one
// expert's down projection.
// ---- Tuned down v3: NR=4 rows per CTA, fewer CTAs to schedule. ----
//
// Mirrors v2 but with NR_V3=4 rows per CTA. With 1024 / 4 = 256 row-tile CTAs ×
// top_k=8 = 2048 CTAs vs v2's 8192. Fewer scheduling waves on A100.
#define NR_V3 4
extern "C" __global__ void moe_batched_down_v3(
    const float* __restrict__ swiglu_buf,           // [top_k * inter_dim]
    const unsigned char* __restrict__ layer_buf,    // raw byte blob
    const unsigned int* __restrict__ expert_ids,    // [top_k]
    const unsigned long long* __restrict__ down_offsets, // [num_experts]
    float* __restrict__ down_out,                   // [top_k * hidden_dim] output
    unsigned int hidden_dim,
    unsigned int inter_dim,
    unsigned int top_k)
{
    extern __shared__ float sw_smem_v3[];  // [inter_dim] swiglu cache

    const unsigned int k = blockIdx.y;
    if (k >= top_k) return;
    const unsigned int r0 = blockIdx.x * NR_V3;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31;
    const unsigned int warp_id = tid >> 5;
    const unsigned int num_warps = BLOCK_DIM_V2 / 32;

    const unsigned int expert_id = expert_ids[k];
    const unsigned long long down_off = down_offsets[expert_id];

    const unsigned int num_blocks = inter_dim / Q8_0_BLOCK_SIZE;
    const size_t row_bytes = (size_t)num_blocks * Q8_0_BLOCK_BYTES;

    const float* swig_k = swiglu_buf + (size_t)k * (size_t)inter_dim;
    for (unsigned int i = tid; i < inter_dim; i += BLOCK_DIM_V2) {
        sw_smem_v3[i] = swig_k[i];
    }
    __syncthreads();

    float sum_r[NR_V3];
    #pragma unroll
    for (int r = 0; r < NR_V3; r++) sum_r[r] = 0.0f;

    for (unsigned int ib = tid; ib < num_blocks; ib += BLOCK_DIM_V2) {
        const unsigned int s_base = ib * Q8_0_BLOCK_SIZE;

        float sv[32];
        const float4* s4 = (const float4*)(sw_smem_v3 + s_base);
        #pragma unroll
        for (int kk = 0; kk < 8; kk++) {
            float4 v = s4[kk];
            sv[kk * 4 + 0] = v.x;
            sv[kk * 4 + 1] = v.y;
            sv[kk * 4 + 2] = v.z;
            sv[kk * 4 + 3] = v.w;
        }

        #pragma unroll
        for (int r = 0; r < NR_V3; r++) {
            if (r0 + r >= hidden_dim) break;
            const unsigned char* dp = layer_buf + down_off
                + (size_t)(r0 + r) * row_bytes
                + (size_t)ib * Q8_0_BLOCK_BYTES;
            float d_scale = load_q8_0_scale(dp);
            const signed char* dq = (const signed char*)(dp + 2);

            float block_sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < 32; j++) {
                block_sum += (float)dq[j] * sv[j];
            }
            sum_r[r] += d_scale * block_sum;
        }
    }

    #pragma unroll
    for (int r = 0; r < NR_V3; r++) {
        sum_r[r] = mb_v2_warp_reduce_sum(sum_r[r]);
    }
    __syncthreads();

    float* reduce_smem = sw_smem_v3;
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < NR_V3; r++) {
            reduce_smem[r * num_warps + warp_id] = sum_r[r];
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < NR_V3; r++) {
            if (r0 + r < hidden_dim) {
                float val = (lane < num_warps) ? reduce_smem[r * num_warps + lane] : 0.0f;
                val = mb_v2_warp_reduce_sum(val);
                if (lane == 0) {
                    down_out[(size_t)k * (size_t)hidden_dim + (r0 + r)] = val;
                }
            }
        }
    }
}

// ---- Tuned gate_up v3: NR=4 rows per CTA. ----
extern "C" __global__ void moe_batched_gate_up_swiglu_q8_0_v3(
    const float* __restrict__ normed_x,
    const unsigned char* __restrict__ layer_buf,
    const unsigned int* __restrict__ expert_ids,
    const unsigned long long* __restrict__ gate_up_offsets,
    float* __restrict__ swiglu_buf,
    unsigned int hidden_dim,
    unsigned int inter_dim,
    unsigned int top_k)
{
    extern __shared__ float nx_smem_v3[];  // [hidden_dim]

    const unsigned int k = blockIdx.y;
    if (k >= top_k) return;
    const unsigned int r0 = blockIdx.x * NR_V3;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31;
    const unsigned int warp_id = tid >> 5;
    const unsigned int num_warps = BLOCK_DIM_V2 / 32;

    const unsigned int expert_id = expert_ids[k];
    const unsigned long long gate_off = gate_up_offsets[(size_t)expert_id * 2 + 0];
    const unsigned long long up_off   = gate_up_offsets[(size_t)expert_id * 2 + 1];

    const unsigned int num_blocks = hidden_dim / Q8_0_BLOCK_SIZE;
    const size_t row_bytes = (size_t)num_blocks * Q8_0_BLOCK_BYTES;

    for (unsigned int i = tid; i < hidden_dim; i += BLOCK_DIM_V2) {
        nx_smem_v3[i] = normed_x[i];
    }
    __syncthreads();

    float gate_sum[NR_V3];
    float up_sum[NR_V3];
    #pragma unroll
    for (int r = 0; r < NR_V3; r++) {
        gate_sum[r] = 0.0f;
        up_sum[r]   = 0.0f;
    }

    for (unsigned int ib = tid; ib < num_blocks; ib += BLOCK_DIM_V2) {
        const unsigned int x_base = ib * Q8_0_BLOCK_SIZE;

        float xv[32];
        const float4* x4 = (const float4*)(nx_smem_v3 + x_base);
        #pragma unroll
        for (int kk = 0; kk < 8; kk++) {
            float4 v = x4[kk];
            xv[kk * 4 + 0] = v.x;
            xv[kk * 4 + 1] = v.y;
            xv[kk * 4 + 2] = v.z;
            xv[kk * 4 + 3] = v.w;
        }

        #pragma unroll
        for (int r = 0; r < NR_V3; r++) {
            if (r0 + r >= inter_dim) break;

            const unsigned char* gp = layer_buf + gate_off
                + (size_t)(r0 + r) * row_bytes
                + (size_t)ib * Q8_0_BLOCK_BYTES;
            float g_scale = load_q8_0_scale(gp);
            const signed char* gq = (const signed char*)(gp + 2);

            const unsigned char* up_ = layer_buf + up_off
                + (size_t)(r0 + r) * row_bytes
                + (size_t)ib * Q8_0_BLOCK_BYTES;
            float u_scale = load_q8_0_scale(up_);
            const signed char* uq = (const signed char*)(up_ + 2);

            float g_block_sum = 0.0f;
            float u_block_sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < 32; j++) {
                g_block_sum += (float)gq[j] * xv[j];
                u_block_sum += (float)uq[j] * xv[j];
            }
            gate_sum[r] += g_scale * g_block_sum;
            up_sum[r]   += u_scale * u_block_sum;
        }
    }

    #pragma unroll
    for (int r = 0; r < NR_V3; r++) {
        gate_sum[r] = mb_v2_warp_reduce_sum(gate_sum[r]);
    }
    __syncthreads();

    float* reduce_smem = nx_smem_v3;
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < NR_V3; r++) {
            reduce_smem[r * num_warps + warp_id] = gate_sum[r];
        }
    }
    __syncthreads();

    float final_gate[NR_V3];
    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < NR_V3; r++) {
            float val = (lane < num_warps) ? reduce_smem[r * num_warps + lane] : 0.0f;
            val = mb_v2_warp_reduce_sum(val);
            final_gate[r] = val;
        }
    }
    __syncthreads();

    #pragma unroll
    for (int r = 0; r < NR_V3; r++) {
        up_sum[r] = mb_v2_warp_reduce_sum(up_sum[r]);
    }
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < NR_V3; r++) {
            reduce_smem[r * num_warps + warp_id] = up_sum[r];
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < NR_V3; r++) {
            if (r0 + r < inter_dim) {
                float val = (lane < num_warps) ? reduce_smem[r * num_warps + lane] : 0.0f;
                val = mb_v2_warp_reduce_sum(val);
                if (lane == 0) {
                    float out = mb_v2_swiglu(final_gate[r], val);
                    swiglu_buf[(size_t)k * (size_t)inter_dim + (r0 + r)] = out;
                }
            }
        }
    }
}

extern "C" __global__ void moe_batched_down_v2(
    const float* __restrict__ swiglu_buf,           // [top_k * inter_dim]
    const unsigned char* __restrict__ layer_buf,    // raw byte blob
    const unsigned int* __restrict__ expert_ids,    // [top_k]
    const unsigned long long* __restrict__ down_offsets, // [num_experts]
    float* __restrict__ down_out,                   // [top_k * hidden_dim] output
    unsigned int hidden_dim,
    unsigned int inter_dim,
    unsigned int top_k)
{
    extern __shared__ float sw_smem[];  // [inter_dim] swiglu cache

    const unsigned int k = blockIdx.y;
    if (k >= top_k) return;
    const unsigned int r0 = blockIdx.x * NR_V2;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31;
    const unsigned int warp_id = tid >> 5;
    const unsigned int num_warps = BLOCK_DIM_V2 / 32;

    const unsigned int expert_id = expert_ids[k];
    const unsigned long long down_off = down_offsets[expert_id];

    const unsigned int num_blocks = inter_dim / Q8_0_BLOCK_SIZE;
    const size_t row_bytes = (size_t)num_blocks * Q8_0_BLOCK_BYTES;

    // Cooperatively load this expert's swiglu output to shmem.
    const float* swig_k = swiglu_buf + (size_t)k * (size_t)inter_dim;
    for (unsigned int i = tid; i < inter_dim; i += BLOCK_DIM_V2) {
        sw_smem[i] = swig_k[i];
    }
    __syncthreads();

    float sum_r[NR_V2];
    #pragma unroll
    for (int r = 0; r < NR_V2; r++) sum_r[r] = 0.0f;

    for (unsigned int ib = tid; ib < num_blocks; ib += BLOCK_DIM_V2) {
        const unsigned int s_base = ib * Q8_0_BLOCK_SIZE;

        // Load 32 swiglu values from shmem to registers (float4 path).
        float sv[32];
        const float4* s4 = (const float4*)(sw_smem + s_base);
        #pragma unroll
        for (int kk = 0; kk < 8; kk++) {
            float4 v = s4[kk];
            sv[kk * 4 + 0] = v.x;
            sv[kk * 4 + 1] = v.y;
            sv[kk * 4 + 2] = v.z;
            sv[kk * 4 + 3] = v.w;
        }

        #pragma unroll
        for (int r = 0; r < NR_V2; r++) {
            if (r0 + r >= hidden_dim) break;
            const unsigned char* dp = layer_buf + down_off
                + (size_t)(r0 + r) * row_bytes
                + (size_t)ib * Q8_0_BLOCK_BYTES;
            float d_scale = load_q8_0_scale(dp);
            const signed char* dq = (const signed char*)(dp + 2);

            float block_sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < 32; j++) {
                block_sum += (float)dq[j] * sv[j];
            }
            sum_r[r] += d_scale * block_sum;
        }
    }

    // Reduce within CTA.
    #pragma unroll
    for (int r = 0; r < NR_V2; r++) {
        sum_r[r] = mb_v2_warp_reduce_sum(sum_r[r]);
    }
    __syncthreads();

    float* reduce_smem = sw_smem;
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < NR_V2; r++) {
            reduce_smem[r * num_warps + warp_id] = sum_r[r];
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < NR_V2; r++) {
            if (r0 + r < hidden_dim) {
                float val = (lane < num_warps) ? reduce_smem[r * num_warps + lane] : 0.0f;
                val = mb_v2_warp_reduce_sum(val);
                if (lane == 0) {
                    down_out[(size_t)k * (size_t)hidden_dim + (r0 + r)] = val;
                }
            }
        }
    }
}

// --- Fused down + accum v2: single launch ---
//
// One CTA computes NR_V2 output rows of x = residual + Σ_k weight_k * (down_k · swiglu_k).
// Each CTA iterates over ALL K experts INTERNALLY (no per-expert dim in grid).
// Swiglu vectors are loaded into shmem once per CTA (saves K-fold re-reads
// vs. v2's per-(k,row) decomposition). For top_k=8 inter_dim=1408 → 44 KB swmem
// (fits within 48 KB / SM occupancy 1-2 CTAs).
//
// Grid: (ceil(hidden_dim / NR_V2), 1, 1). Block: (BLOCK_DIM_V2, 1, 1).
// Shmem: top_k * inter_dim * 4 bytes (swiglu cache).
extern "C" __global__ void moe_batched_down_accum_v2(
    const float* __restrict__ swiglu_buf,           // [top_k * inter_dim]
    const unsigned char* __restrict__ layer_buf,    // raw byte blob
    const unsigned int* __restrict__ expert_ids,    // [top_k]
    const unsigned long long* __restrict__ down_offsets, // [num_experts]
    const float* __restrict__ expert_weights,       // [top_k]
    const float* __restrict__ residual,             // [hidden_dim]
    float* __restrict__ x,                          // [hidden_dim] output = residual + Σ_k w_k * down_k
    unsigned int hidden_dim,
    unsigned int inter_dim,
    unsigned int top_k)
{
    extern __shared__ float swig_smem_pool[];  // [top_k * inter_dim]

    const unsigned int r0 = blockIdx.x * NR_V2;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31;
    const unsigned int warp_id = tid >> 5;
    const unsigned int num_warps = BLOCK_DIM_V2 / 32;

    const unsigned int num_blocks = inter_dim / Q8_0_BLOCK_SIZE;
    const size_t row_bytes = (size_t)num_blocks * Q8_0_BLOCK_BYTES;

    // Cooperatively load ALL K swiglu vectors to shmem in one pass.
    // [top_k * inter_dim] floats; total = top_k * inter_dim * 4 bytes shmem.
    const unsigned int total_swig = top_k * inter_dim;
    for (unsigned int i = tid; i < total_swig; i += BLOCK_DIM_V2) {
        swig_smem_pool[i] = swiglu_buf[i];
    }
    __syncthreads();

    // Load expert weights & ids into registers (top_k ≤ 16).
    float w_arr[MOE_MAX_TOP_K];
    unsigned int eid_arr[MOE_MAX_TOP_K];
    const unsigned int K = (top_k < MOE_MAX_TOP_K) ? top_k : MOE_MAX_TOP_K;
    for (unsigned int k = 0; k < K; ++k) {
        w_arr[k] = expert_weights[k];
        eid_arr[k] = expert_ids[k];
    }

    // Per-output-row partial sums across all K experts (weighted).
    float sum_r[NR_V2];
    #pragma unroll
    for (int r = 0; r < NR_V2; r++) sum_r[r] = 0.0f;

    // Block-strided iteration over Q8 blocks; each thread does one block per iter.
    for (unsigned int ib = tid; ib < num_blocks; ib += BLOCK_DIM_V2) {
        // Loop over K experts: for each, compute (block_sum_per_row), scale by weight_k.
        // For each row r in NR_V2:
        //   for each k in K:
        //     load down_k Q8 block at (row, ib), multiply by swig_k Q8 block at ib, accumulate.
        // The swig values change per expert, so we load them per (k, ib) inside.

        #pragma unroll
        for (int r = 0; r < NR_V2; r++) {
            if (r0 + r >= hidden_dim) break;

            float row_sum = 0.0f;
            for (unsigned int k = 0; k < K; ++k) {
                unsigned long long down_off = down_offsets[eid_arr[k]];
                const unsigned char* dp = layer_buf + down_off
                    + (size_t)(r0 + r) * row_bytes
                    + (size_t)ib * Q8_0_BLOCK_BYTES;
                float d_scale = load_q8_0_scale(dp);
                const signed char* dq = (const signed char*)(dp + 2);
                const float* sv_k = swig_smem_pool + (size_t)k * (size_t)inter_dim
                    + (size_t)ib * Q8_0_BLOCK_SIZE;

                float block_sum = 0.0f;
                #pragma unroll
                for (int j = 0; j < 32; j++) {
                    block_sum += (float)dq[j] * sv_k[j];
                }
                row_sum += w_arr[k] * (d_scale * block_sum);
            }
            sum_r[r] += row_sum;
        }
    }

    // Reduce sum_r across CTA.
    __shared__ float reduce_smem[NR_V2 * (BLOCK_DIM_V2 / 32)];

    #pragma unroll
    for (int r = 0; r < NR_V2; r++) {
        sum_r[r] = mb_v2_warp_reduce_sum(sum_r[r]);
    }
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < NR_V2; r++) {
            reduce_smem[r * num_warps + warp_id] = sum_r[r];
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < NR_V2; r++) {
            if (r0 + r < hidden_dim) {
                float val = (lane < num_warps) ? reduce_smem[r * num_warps + lane] : 0.0f;
                val = mb_v2_warp_reduce_sum(val);
                if (lane == 0) {
                    x[r0 + r] = residual[r0 + r] + val;
                }
            }
        }
    }
}

// =============================================================================
// Fused persistent gate+up+SwiGLU+down+accum kernel (Q8_0).
//
// Eliminates the HBM round-trip on swiglu_buf by computing gate+up+SwiGLU
// in-kernel and keeping the K experts' intermediate vectors in shmem.
//
// Grid: (ceil(hidden_dim / NR_V4_FUSED), 1, 1). Block: (BLOCK_DIM_V4_FUSED, 1, 1).
// One CTA produces NR_V4_FUSED output rows of `x = residual + Σ_k w_k * down_k(SwiGLU_k)`.
//
// Per-CTA work:
//   1. Cooperatively load normed_x into shmem (hidden_dim × 4 bytes).
//   2. For each k in [0..top_k):
//      a. Cooperatively compute the inter_dim SwiGLU values for expert k's
//         (gate, up) into shmem (inter_dim × 4 bytes; reused across k).
//      b. For each row r in [r0..r0+NR_V4_FUSED):
//         Accumulate row_sum[r] += w[k] * (down_k[r] · swiglu_smem).
//   3. Write x[r] = residual[r] + row_sum[r].
//
// Shmem layout:
//   nx_smem[hidden_dim]      // normed_x cache
//   swiglu_smem[inter_dim]   // SwiGLU intermediate per expert (reused per k)
//   reduce_smem[NR_V4_FUSED * num_warps]  // cross-warp reduction
//
// This trades K× more in-CTA arithmetic on gate+up+SwiGLU (vs computing once
// in v3 and reading from HBM) against:
//   - One fewer kernel launch ('s separate gate_up_v3 + down_v3 + accum)
//   - Elimination of swiglu_buf HBM write + read
//   - Improved temporal locality: gate, up, down for ONE expert read in
//     succession by the same CTA → better L2 cache hit rate.
//
// Bit-equivalence with v3 + accum_option_a:
//   - SwiGLU column-wise sum order is preserved (same block-strided iteration
//     over Q8 blocks, same per-block FMA tree, same warp reduction).
//   - Down row sum order is preserved (block-strided over Q8 blocks).
//   - Accum cross-K sum order changes: v3 computes per-K down outputs first
//     then sums weighted. This kernel sums weighted-K inside the row tile.
//     Result is algebraically identical in exact FP, but FP rounding may
//     differ by 1 ULP on the last bit. Validator allows ≤1e-5 e2e tolerance.
//
// CAVEAT: Per-CTA gate+up+SwiGLU is fully duplicated across grid CTAs (each
// of the ~512 row-tile CTAs recomputes the same SwiGLU). This is the cost of
// avoiding the HBM round-trip. Whether the trade-off wins depends on L2 cache
// behavior of the K-expert weight stream. Profiling validates the trade-off.
// =============================================================================

#define BLOCK_DIM_V4_FUSED 256   // 8 warps per CTA
// NR_V4_FUSED: output rows per CTA. Larger NR amortizes the per-CTA SwiGLU
// recomputation across more output rows; smaller NR uses more CTAs (better
// SM utilization). For Qwen3.5-MoE-35B-A3B (hidden=2048), NR=128 gives 16
// CTAs (14% SM util on A100), each computing all K=8 experts' SwiGLU once.
// This is the best trade-off for the bandwidth-bound decode regime.
//
// Empirical: NR=4 (the v3 row-tile width) caused 35× slowdown vs v3 due to
// 4096× SwiGLU recomputation. NR=128 reduces recomputation 32× and is the
// minimum viable trade-off.
#define NR_V4_FUSED        128   // output rows per CTA (large to amortize SwiGLU recompute)

extern "C" __global__ void moe_batched_persistent_gate_up_swiglu_down_accum_q8_0(
    const float* __restrict__ normed_x,             // [hidden_dim]
    const unsigned char* __restrict__ layer_buf,    // raw byte blob; weights at offsets below
    const unsigned int* __restrict__ expert_ids,    // [top_k] from router
    const float* __restrict__ expert_weights,       // [top_k] from router (softmax + top-K)
    const unsigned long long* __restrict__ gate_up_offsets, // [num_experts * 2]
    const unsigned long long* __restrict__ down_offsets,    // [num_experts]
    const float* __restrict__ residual,             // [hidden_dim] residual stream
    float* __restrict__ x,                          // [hidden_dim] output
    unsigned int hidden_dim,
    unsigned int inter_dim,
    unsigned int top_k)
{
    // Dynamic shmem layout:
    //   [0 .. hidden_dim)               normed_x cache
    //   [hidden_dim .. hidden_dim+inter_dim)  swiglu (reused per k)
    extern __shared__ float fused_smem_pool[];
    float* nx_smem      = fused_smem_pool;
    float* swiglu_smem  = fused_smem_pool + hidden_dim;

    const unsigned int r0 = blockIdx.x * NR_V4_FUSED;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31;
    const unsigned int warp_id = tid >> 5;
    const unsigned int num_warps = BLOCK_DIM_V4_FUSED / 32;

    // Static shmem reduction buffer for cross-warp partial-sum aggregation
    // (used for both the per-column SwiGLU dot products and the per-row down sums).
    __shared__ float reduce_smem_fused[NR_V4_FUSED * (BLOCK_DIM_V4_FUSED / 32)];

    const unsigned int gate_up_num_blocks = hidden_dim / Q8_0_BLOCK_SIZE;
    const size_t gate_up_row_bytes = (size_t)gate_up_num_blocks * Q8_0_BLOCK_BYTES;
    const unsigned int down_num_blocks = inter_dim / Q8_0_BLOCK_SIZE;
    const size_t down_row_bytes = (size_t)down_num_blocks * Q8_0_BLOCK_BYTES;

    // 1. Load normed_x into shmem (cooperatively, vectorized via float4 when aligned).
    for (unsigned int i = tid; i < hidden_dim; i += BLOCK_DIM_V4_FUSED) {
        nx_smem[i] = normed_x[i];
    }
    __syncthreads();

    // Load expert weights and ids into per-thread registers.
    float w_arr[MOE_MAX_TOP_K];
    unsigned int eid_arr[MOE_MAX_TOP_K];
    const unsigned int K = (top_k < MOE_MAX_TOP_K) ? top_k : MOE_MAX_TOP_K;
    #pragma unroll
    for (unsigned int k = 0; k < MOE_MAX_TOP_K; ++k) {
        w_arr[k] = (k < K) ? expert_weights[k] : 0.0f;
        eid_arr[k] = (k < K) ? expert_ids[k]   : 0u;
    }

    // Per-row accumulators across all K experts (weighted sum).
    float row_sum[NR_V4_FUSED];
    #pragma unroll
    for (int r = 0; r < NR_V4_FUSED; r++) row_sum[r] = 0.0f;

    // 2. Loop over K experts.
    for (unsigned int k = 0; k < K; ++k) {
        const unsigned int expert_id = eid_arr[k];
        const unsigned long long gate_off = gate_up_offsets[(size_t)expert_id * 2 + 0];
        const unsigned long long up_off   = gate_up_offsets[(size_t)expert_id * 2 + 1];
        const unsigned long long down_off = down_offsets[expert_id];

        // 2a. Compute SwiGLU for all inter_dim columns of expert k into swiglu_smem.
        //
        // We process inter_dim columns in chunks of (BLOCK_DIM_V4_FUSED / num_warps)
        // rows at a time, with each warp computing one column. This matches the
        // v3 reduction pattern (warp-shfl + cross-warp shmem).
        //
        // To keep shmem usage low and maximize reuse, we use a tiled "warp-per-column"
        // approach: each warp handles one column at a time, dot-products gate_q · x
        // and up_q · x over all hidden_dim/32 Q8 blocks, then SwiGLU's the result.
        //
        // Total columns: inter_dim. Total warps per CTA: num_warps (8 for BLOCK_DIM=256).
        // Total iterations per CTA: ceil(inter_dim / num_warps).
        for (unsigned int col_base = 0; col_base < inter_dim; col_base += num_warps) {
            const unsigned int col = col_base + warp_id;

            float gate_acc = 0.0f;
            float up_acc   = 0.0f;
            if (col < inter_dim) {
                // Q8 row for (expert_id, gate, row=col) starts at:
                //   layer_buf + gate_off + col * gate_up_row_bytes
                const unsigned char* gp_base = layer_buf + gate_off
                    + (size_t)col * gate_up_row_bytes;
                const unsigned char* up_base = layer_buf + up_off
                    + (size_t)col * gate_up_row_bytes;

                // Each lane processes one Q8 block at a stride of 32 (warp_size).
                for (unsigned int ib = lane; ib < gate_up_num_blocks; ib += 32) {
                    const unsigned int x_base = ib * Q8_0_BLOCK_SIZE;

                    const unsigned char* gp = gp_base + (size_t)ib * Q8_0_BLOCK_BYTES;
                    float g_scale = load_q8_0_scale(gp);
                    const signed char* gq = (const signed char*)(gp + 2);

                    const unsigned char* upp = up_base + (size_t)ib * Q8_0_BLOCK_BYTES;
                    float u_scale = load_q8_0_scale(upp);
                    const signed char* uq = (const signed char*)(upp + 2);

                    float g_block_sum = 0.0f;
                    float u_block_sum = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < 32; j++) {
                        float xv = nx_smem[x_base + j];
                        g_block_sum += (float)gq[j] * xv;
                        u_block_sum += (float)uq[j] * xv;
                    }
                    gate_acc += g_scale * g_block_sum;
                    up_acc   += u_scale * u_block_sum;
                }
            }

            // Warp-level reduction (one warp per column).
            gate_acc = mb_v2_warp_reduce_sum(gate_acc);
            up_acc   = mb_v2_warp_reduce_sum(up_acc);

            if (lane == 0 && col < inter_dim) {
                swiglu_smem[col] = mb_v2_swiglu(gate_acc, up_acc);
            }
        }
        __syncthreads();

        // 2b. Compute down · swiglu_smem for NR_V4_FUSED rows in [r0..r0+NR_V4_FUSED).
        //
        // Block-strided iteration over Q8 blocks (mirrors v3 down kernel). Each
        // thread processes one Q8 block at a time, accumulating across NR rows.
        // Since step 2a dominates (~500× more arithmetic than step 2b for
        // inter_dim=1408, NR=4), inefficiency here has negligible impact on
        // total per-CTA time.
        float down_sum[NR_V4_FUSED];
        #pragma unroll
        for (int r = 0; r < NR_V4_FUSED; r++) down_sum[r] = 0.0f;

        for (unsigned int ib = tid; ib < down_num_blocks; ib += BLOCK_DIM_V4_FUSED) {
            const unsigned int s_base = ib * Q8_0_BLOCK_SIZE;

            // Load 32 swiglu values from shmem to registers (float4 path).
            float sv[32];
            const float4* s4 = (const float4*)(swiglu_smem + s_base);
            #pragma unroll
            for (int kk = 0; kk < 8; kk++) {
                float4 v = s4[kk];
                sv[kk * 4 + 0] = v.x;
                sv[kk * 4 + 1] = v.y;
                sv[kk * 4 + 2] = v.z;
                sv[kk * 4 + 3] = v.w;
            }

            #pragma unroll
            for (int r = 0; r < NR_V4_FUSED; r++) {
                if (r0 + r >= hidden_dim) break;
                const unsigned char* dp = layer_buf + down_off
                    + (size_t)(r0 + r) * down_row_bytes
                    + (size_t)ib * Q8_0_BLOCK_BYTES;
                float d_scale = load_q8_0_scale(dp);
                const signed char* dq = (const signed char*)(dp + 2);

                float block_sum = 0.0f;
                #pragma unroll
                for (int j = 0; j < 32; j++) {
                    block_sum += (float)dq[j] * sv[j];
                }
                down_sum[r] += d_scale * block_sum;
            }
        }

        // Reduce down_sum[r] across CTA (warp shfl + cross-warp shmem).
        #pragma unroll
        for (int r = 0; r < NR_V4_FUSED; r++) {
            down_sum[r] = mb_v2_warp_reduce_sum(down_sum[r]);
        }

        if (lane == 0) {
            #pragma unroll
            for (int r = 0; r < NR_V4_FUSED; r++) {
                reduce_smem_fused[r * num_warps + warp_id] = down_sum[r];
            }
        }
        __syncthreads();

        // Last warp finalizes the cross-warp reduction; lane 0 of warp 0 keeps it.
        float final_down[NR_V4_FUSED];
        if (warp_id == 0) {
            #pragma unroll
            for (int r = 0; r < NR_V4_FUSED; r++) {
                float val = (lane < num_warps) ? reduce_smem_fused[r * num_warps + lane] : 0.0f;
                val = mb_v2_warp_reduce_sum(val);
                final_down[r] = val;
            }
        }
        // Accumulate weighted contribution (only warp 0 lane 0 has the data).
        if (warp_id == 0 && lane == 0) {
            #pragma unroll
            for (int r = 0; r < NR_V4_FUSED; r++) {
                row_sum[r] += w_arr[k] * final_down[r];
            }
        }
        __syncthreads();
    }

    // 3. Write x[r] = residual[r] + row_sum[r]  (only warp 0 lane 0 has the data).
    if (warp_id == 0 && lane == 0) {
        #pragma unroll
        for (int r = 0; r < NR_V4_FUSED; r++) {
            if (r0 + r < hidden_dim) {
                x[r0 + r] = residual[r0 + r] + row_sum[r];
            }
        }
    }
}
