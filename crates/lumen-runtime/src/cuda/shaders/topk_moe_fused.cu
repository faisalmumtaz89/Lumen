// Fused MoE top-K router (single kernel: logits + activation +
// top-K + renorm) replacing the prior two-launch sequence
// (`moe_router_logits_v2` + `moe_router_softmax_finalize_v2`).
//
//   In:  normed_x [hidden_dim]         (F32)
//        router_weight [E, hidden_dim] (F32, row-major)
//   Out: expert_ids [top_k]            (u32; int32_t in our path)
//        expert_weights [top_k]        (F32, renormalized so Σ_k = 1)
//
// Profile mapping:
//   Lumen V2 router @ Qwen3.5-MoE-35B-A3B, Q8 decode, 64 tok:
//     `moe_router_logits_v2`            : 0.21 ms/tok ( 1.5% TPOT)
//     `moe_router_softmax_finalize_v2`  : 1.32 ms/tok ( 8.7% TPOT)
//                          subtotal     : 1.53 ms/tok (10.2% TPOT)
//   Target TPOT after fusion             : ~2.7% (was 8.7%)
//   Predicted Δ                          : +1.3 ms/tok recovered ≈ +14 tok/s
//
// Algorithm:
//   1. Each CTA processes `rows_per_block = 4` tokens; one warp per row.
//   2. Per warp: load logits into `experts_per_thread = n_experts/WARP_SIZE`
//      registers (n_experts must be power-of-two ≤ 512, or 576).
//   3. Apply sigmoid (Qwen3.5-MoE) or softmax (per `use_sigmoid` arg).
//   4. NaN-sanitize to -FLT_MAX (matches the upstream NaN-sanitize fix for the
//      cuBLAS path).
//   5. Iterated argmax-with-mask over top_K, using `__shfl_xor_sync` warp
//      reductions to find the per-warp argmax, then masking the winner to
//      -3.402823466e+38f before the next round.
//   6. Optionally renormalize the selected weights so Σ_k = 1 (Qwen3.5-MoE
//      sets with_norm=true).
//   7. Apply scale_val (=1.0 for Qwen3.5-MoE).
//   8. Write expert_ids[k], expert_weights[k] for k ∈ [0, top_k).
//
// Block layout:
//   blockDim = (WARP_SIZE=32, rows_per_block=4, 1)  → 128 threads
//   gridDim  = (ceil(n_rows / rows_per_block), 1, 1)
//
// For DECODE the caller passes n_rows=1, so gridDim=(1,1,1) and only warp 0
// of the single CTA is active. For PREFILL n_rows=batch_size, so each CTA
// rounds up 4 tokens at a time.
//
// NO bias variant is exposed: Qwen3.5-MoE does not use expert-gate bias. The
// has_bias=true path is omitted to keep the NVRTC module compact (the
// downstream kernel string is ~1500 LoC; adding has_bias doubles all reduce
// loops).
//
// Validation gates:
//   - Per-expert routing probabilities at L0-L39: within F32 ULP.
//   - Expert IDs at L0-L39: must match baseline (L0 6/6, L1-39
//     cascade); the algorithm is structurally deterministic.
//   - 8-prompt PURE-greedy: ≥7/8.
//   - 24-prompt rigorous: ≥20/24.

#define WARP_SIZE 32

__device__ __forceinline__ float topk_moe_fused_warp_reduce_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 16));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v,  8));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v,  4));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v,  2));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v,  1));
    return v;
}

__device__ __forceinline__ float topk_moe_fused_warp_reduce_sum(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 16);
    v += __shfl_xor_sync(0xffffffff, v,  8);
    v += __shfl_xor_sync(0xffffffff, v,  4);
    v += __shfl_xor_sync(0xffffffff, v,  2);
    v += __shfl_xor_sync(0xffffffff, v,  1);
    return v;
}

// Warp-local softmax used either pre-top-K or post-top-K
// (delayed_softmax path; not exercised by Qwen3.5-MoE).
template <int experts_per_thread, bool use_limit>
__device__ void topk_moe_fused_softmax_warp_inplace(
    float (&vals)[experts_per_thread],
    const int limit,
    const int lane)
{
    float max_val = -3.402823466e+38f;
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            max_val = fmaxf(max_val, vals[i]);
        }
    }
    max_val = topk_moe_fused_warp_reduce_max(max_val);

    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            const float val = __expf(vals[i] - max_val);
            vals[i]         = val;
            sum += val;
        } else {
            vals[i] = 0.f;
        }
    }
    sum = topk_moe_fused_warp_reduce_sum(sum);

    const float inv_sum = 1.0f / sum;
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            vals[i] *= inv_sum;
        }
    }
}

template <int experts_per_thread, bool use_limit>
__device__ void topk_moe_fused_sigmoid_warp_inplace(
    float (&vals)[experts_per_thread],
    const int limit,
    const int lane)
{
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        vals[i] = active ? 1.f / (1.f + __expf(-vals[i])) : -3.402823466e+38f;
    }
}

// Compute logits via per-warp dot-product of `normed_x` and one row of
// `router_weight`. Each warp processes `rows_per_block`-th token's logits in
// turn — but only ONE row index per call because the caller passes the logits
// already (or we compute them here when fused=true).
//
// For the port we keep the logits OUTSIDE this kernel and pass them in,
// keeping the separation of concerns: the logits come from a prior mul_mat_vec
// kernel. Lumen's existing `moe_router_logits_v2` produces the identical F32
// logits buffer, so the kernel is a drop-in replacement for ONLY the
// second launch (`moe_router_softmax_finalize_v2`) — the most expensive of
// the 5 router kernels (8.7% TPOT).
//
// PHASE 1 (this kernel): logits already in `router_logits` global — apply
// sigmoid/softmax + top-K + renorm + scale, write out `expert_ids` and
// `expert_weights`.

// Kernel: topk_moe_fused_no_bias
//
// has_bias = false (Qwen3.5-MoE has no expert-gate bias).
//
// Grid:  ((n_rows + 3) / 4, 1, 1) with rows_per_block = 4.
// Block: (32, 4, 1) — 128 threads, 4 warps, one warp per row.
//
// Args:
//   logits         : [n_rows, n_experts]   F32 IN
//   weights        : [n_rows, n_expert_used] F32 OUT  (renormalized, scaled)
//   ids            : [n_rows, n_experts]   u32 OUT    (only first n_expert_used
//                                                      slots are written)
//   n_rows         : token count (decode = 1; prefill = batch_size)
//   n_expert_used  : top_k (≤ 16 for Qwen3.5-MoE)
//   clamp_val      : -3.402823466e+38f when no renorm; otherwise the min Σ for renorm
//   scale_val      : multiplier on output_weights (=1.0 for Qwen3.5-MoE)
//   use_sigmoid    : 1 → sigmoid + iterate (Qwen3.5-MoE), 0 → softmax
//   with_norm      : 1 → renormalize selected weights so Σ_k = 1
//   delayed_softmax: 1 → softmax AFTER top-K (mutually exclusive with with_norm)
//
// NB: The kernel reads ids as int32_t. Lumen's `scratch.expert_ids` is CudaSlice<u32>.
// The expert IDs are non-negative in [0, n_experts), so signed/unsigned reinterp
// is a no-op at this bit pattern. We bind the kernel arg as `unsigned int*`.
template <int n_experts>
__device__ void topk_moe_fused_no_bias_impl(
    const float* __restrict__ logits,
    float*       __restrict__ weights,
    unsigned int* __restrict__ ids,
    const int           n_rows,
    const int           n_expert_used,
    const float         clamp_val,
    const float         scale_val,
    const unsigned int  use_sigmoid,
    const unsigned int  with_norm,
    const unsigned int  delayed_softmax)
{
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= n_rows) {
        return;
    }

    logits  += (size_t)n_experts * (size_t)row;
    weights += (size_t)n_expert_used * (size_t)row;
    ids     += (size_t)n_experts * (size_t)row;

    constexpr int experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

    float wt[experts_per_thread];

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        wt[i] = -3.402823466e+38f;
    }

#pragma unroll
    for (int i = 0; i < n_experts; i += WARP_SIZE) {
        const int expert = i + (int)threadIdx.x;
        wt[i / WARP_SIZE] = (n_experts % WARP_SIZE == 0 || expert < n_experts)
                            ? logits[expert] : -3.402823466e+38f;
    }

    if (!delayed_softmax) {
        if (use_sigmoid) {
            topk_moe_fused_sigmoid_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        } else {
            topk_moe_fused_softmax_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        }
    }

    // NaN sanitize to -FLT_MAX (matches the upstream NaN-sanitize fix).
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        if (__isnanf(wt[i])) {
            wt[i] = -3.402823466e+38f;
        }
    }

    float wt_sum = 0.f;
    float output_weights[experts_per_thread];

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.f;
    }

    // Iterated argmax-with-mask over n_expert_used.
    for (int k = 0; k < n_expert_used; k++) {
        float max_val    = wt[0];
        int   max_expert = (int)threadIdx.x;

#pragma unroll
        for (int i = 1; i < experts_per_thread; i++) {
            const int expert = (int)threadIdx.x + i * WARP_SIZE;
            if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && wt[i] > max_val) {
                max_val    = wt[i];
                max_expert = expert;
            }
        }

#pragma unroll
        for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
            const float val    = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
            const int   expert = __shfl_xor_sync(0xFFFFFFFF, max_expert, mask, WARP_SIZE);
            if (val > max_val || (val == max_val && expert < max_expert)) {
                max_val    = val;
                max_expert = expert;
            }
        }

        if ((max_expert & (WARP_SIZE - 1)) == (int)threadIdx.x) {
            wt[max_expert / WARP_SIZE] = -3.402823466e+38f;
        }

        if ((k & (WARP_SIZE - 1)) == (int)threadIdx.x) {
            output_weights[k / WARP_SIZE] = max_val;
        }

        if ((max_expert & (WARP_SIZE - 1)) == (int)threadIdx.x) {
            ids[k] = (unsigned int)max_expert;
            if (with_norm) {
                wt_sum += max_val;
            }
        }
    }

    if (with_norm) {
        wt_sum              = topk_moe_fused_warp_reduce_sum(wt_sum);
        wt_sum              = fmaxf(wt_sum, clamp_val);
        const float inv_sum = 1.0f / wt_sum;
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            output_weights[i] *= inv_sum;
        }
    }

    if (delayed_softmax) {
        topk_moe_fused_softmax_warp_inplace<experts_per_thread, true>(output_weights, n_expert_used, threadIdx.x);
    }

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int idx = i * WARP_SIZE + (int)threadIdx.x;
        if (idx < n_expert_used) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

// NVRTC-compatible C-linkage entry points. We ship two instantiations
// (n_experts=128 for Qwen3.5-MoE-30B/35B-A3B; n_experts=256 forward-compatible
// for larger variants). The Lumen dispatch will pick the matching variant at
// runtime based on `meta.num_experts`. Power-of-two ≤ 256 is sufficient for
// all known production Qwen3.5-MoE configs.
//
// __launch_bounds__(128, 1) = 4 * WARP_SIZE = 128 threads/block.

extern "C" __launch_bounds__(128, 1) __global__ void topk_moe_fused_128_no_bias(
    const float* __restrict__ logits,
    float*       __restrict__ weights,
    unsigned int* __restrict__ ids,
    const int           n_rows,
    const int           n_expert_used,
    const float         clamp_val,
    const float         scale_val,
    const unsigned int  use_sigmoid,
    const unsigned int  with_norm,
    const unsigned int  delayed_softmax)
{
    topk_moe_fused_no_bias_impl<128>(
        logits, weights, ids,
        n_rows, n_expert_used, clamp_val, scale_val,
        use_sigmoid, with_norm, delayed_softmax);
}

extern "C" __launch_bounds__(128, 1) __global__ void topk_moe_fused_256_no_bias(
    const float* __restrict__ logits,
    float*       __restrict__ weights,
    unsigned int* __restrict__ ids,
    const int           n_rows,
    const int           n_expert_used,
    const float         clamp_val,
    const float         scale_val,
    const unsigned int  use_sigmoid,
    const unsigned int  with_norm,
    const unsigned int  delayed_softmax)
{
    topk_moe_fused_no_bias_impl<256>(
        logits, weights, ids,
        n_rows, n_expert_used, clamp_val, scale_val,
        use_sigmoid, with_norm, delayed_softmax);
}

// 64-expert variant covers small-MoE test models and the eventual 30B-A3B-like
// quant variants that might ship with fewer routed experts. Kept as a safety
// net; not directly referenced by the production dispatch unless num_experts==64.
extern "C" __launch_bounds__(128, 1) __global__ void topk_moe_fused_64_no_bias(
    const float* __restrict__ logits,
    float*       __restrict__ weights,
    unsigned int* __restrict__ ids,
    const int           n_rows,
    const int           n_expert_used,
    const float         clamp_val,
    const float         scale_val,
    const unsigned int  use_sigmoid,
    const unsigned int  with_norm,
    const unsigned int  delayed_softmax)
{
    topk_moe_fused_no_bias_impl<64>(
        logits, weights, ids,
        n_rows, n_expert_used, clamp_val, scale_val,
        use_sigmoid, with_norm, delayed_softmax);
}
