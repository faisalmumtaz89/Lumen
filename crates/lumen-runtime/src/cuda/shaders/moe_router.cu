// MoE top-K router kernel.
//
// Computes: router_logits = router_weight * normed_x       [num_experts]
//           softmax + top_K + renormalize                 [top_k]
//
// Ported verbatim from metal/shaders/moe.msl:1-100.
//
// Numerical-stability properties (preserve):
//   - max subtraction before exp (logits overflow guard)
//   - top-K via repeated argmax-with-mask (no full sort)
//   - renormalize selected weights so Σ_k weight[k] = 1
//
// Grid: one CTA per token (decode = 1 CTA total; prefill = batch_size CTAs).
// Block: BLOCK_DIM threads; warps participate in the per-expert dot product;
//        thread 0 of CTA 0 performs the (small) softmax + top-K phase since
//        num_experts ≤ 256 and top_k ≤ 8 in practice.
//
// Inputs:
//   normed_x         [hidden_dim] float
//   router_weight    [num_experts, hidden_dim] float (row-major; row e is the
//                                                     dot-product partner for x)
//   hidden_dim       u32 scalar
//   num_experts      u32 scalar (≤ MOE_MAX_NUM_EXPERTS)
//   top_k            u32 scalar (≤ MOE_MAX_TOP_K)
//
// Outputs:
//   expert_ids       [top_k] u32
//   expert_weights   [top_k] float (renormalized, Σ = 1)

// Compile-time bounds: matches Metal's `MOE_MAX_NUM_EXPERTS` / `MOE_MAX_TOP_K`.
// 256 experts is the upper bound for Qwen3.5-MoE (235B-A22B = 128 experts);
// 16 top-K covers all known Qwen3.5-MoE variants (top-K is typically 6 or 8).
#define MOE_MAX_NUM_EXPERTS 256
#define MOE_MAX_TOP_K 16
#define BLOCK_DIM 256

extern "C" __global__ void moe_router_softmax(
    const float* __restrict__ normed_x,         // [hidden_dim]
    const float* __restrict__ router_weight,    // [num_experts * hidden_dim]
    unsigned int* __restrict__ expert_ids,      // [top_k] output
    float* __restrict__ expert_weights,         // [top_k] output
    unsigned int hidden_dim,
    unsigned int num_experts,
    unsigned int top_k)
{
    __shared__ float logits[MOE_MAX_NUM_EXPERTS];
    __shared__ float warp_partial[BLOCK_DIM / 32];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = BLOCK_DIM / 32;

    // Phase 1: per-expert dot product (parallel reduction across CTA threads).
    // Each expert is processed sequentially; within an expert all threads share
    // the work in stride BLOCK_DIM.
    for (unsigned int e = 0; e < num_experts; ++e) {
        float partial = 0.0f;
        const float* w_e = router_weight + (size_t)e * (size_t)hidden_dim;
        for (unsigned int j = tid; j < hidden_dim; j += BLOCK_DIM) {
            partial += w_e[j] * normed_x[j];
        }
        // Warp-level reduction via shfl_down_sync.
        for (int offset = 16; offset > 0; offset >>= 1) {
            partial += __shfl_down_sync(0xffffffff, partial, offset);
        }
        if (lane == 0) {
            warp_partial[warp_id] = partial;
        }
        __syncthreads();
        // First warp reduces the warp partials.
        if (warp_id == 0) {
            float v = (lane < num_warps) ? warp_partial[lane] : 0.0f;
            for (int offset = 16; offset > 0; offset >>= 1) {
                v += __shfl_down_sync(0xffffffff, v, offset);
            }
            if (lane == 0) {
                logits[e] = v;
            }
        }
        __syncthreads();
    }

    // Phase 2: softmax + top-K (single thread; num_experts is small).
    if (tid == 0) {
        // Max subtraction.
        float maxv = logits[0];
        for (unsigned int e = 1; e < num_experts; ++e) {
            float v = logits[e];
            if (v > maxv) maxv = v;
        }
        // Compute exp + sum.
        float sum = 0.0f;
        for (unsigned int e = 0; e < num_experts; ++e) {
            float v = expf(logits[e] - maxv);
            logits[e] = v;
            sum += v;
        }
        float inv = 1.0f / sum;
        for (unsigned int e = 0; e < num_experts; ++e) {
            logits[e] *= inv;
        }

        // Top-K via repeated argmax-with-mask.
        // Masked entries get -1.0 (probabilities are non-negative).
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
            logits[best_e] = -1.0f;  // mask out for next iteration
        }
        // Renormalize selected weights so Σ_k weight[k] = 1.
        if (renorm > 0.0f) {
            float invr = 1.0f / renorm;
            for (unsigned int k = 0; k < top_k; ++k) {
                expert_weights[k] *= invr;
            }
        }
    }
}
