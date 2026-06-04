// MoE weighted accumulation kernels.
//
// Two accumulator variants:
//
// (A) `moe_expert_accum_option_a` — dense top-K layout (default for).
//     Expert outputs are stored at slot k in a `[top_k * hidden_dim]` buffer;
//     slot k corresponds to expert_ids[k] (the selected expert at rank k).
//     Computes: x[i] = residual[i] + Σ_{k=0..K-1} expert_weights[k] * expert_outputs[k * hidden_dim + i]
//
// (B) `moe_expert_accum_batched_b` — sparse `num_experts` layout.
//     Expert outputs are stored at slot expert_ids[k] in a
//     `[num_experts * hidden_dim]` buffer; reserved for the batched
//     kernel path (currently unused but kept for parity with Metal).
//     Computes: x[i] = residual[i] + Σ_{k=0..K-1} expert_weights[k] * expert_outputs[expert_ids[k] * hidden_dim + i]
//
// Each thread accumulates one element of the hidden dimension across all K
// experts; grid covers `hidden_dim / BLOCK_DIM` CTAs.

#define BLOCK_DIM 128
#define MOE_MAX_TOP_K 16

extern "C" __global__ void moe_expert_accum_option_a(
    float* __restrict__ x,                          // [hidden_dim] in/out (residual + Σ weighted outputs)
    const float* __restrict__ residual,             // [hidden_dim]
    const float* __restrict__ expert_outputs,       // [top_k * hidden_dim] (dense slot layout)
    const float* __restrict__ expert_weights,       // [top_k]
    unsigned int hidden_dim,
    unsigned int top_k)
{
    const unsigned int i = blockIdx.x * BLOCK_DIM + threadIdx.x;
    if (i >= hidden_dim) return;

    // Load expert weights into per-thread registers (top_k is small).
    float w[MOE_MAX_TOP_K];
    const unsigned int K = (top_k < MOE_MAX_TOP_K) ? top_k : MOE_MAX_TOP_K;
    for (unsigned int k = 0; k < K; ++k) {
        w[k] = expert_weights[k];
    }

    // Accumulate.
    float acc = residual[i];
    for (unsigned int k = 0; k < K; ++k) {
        acc += w[k] * expert_outputs[(size_t)k * (size_t)hidden_dim + i];
    }
    x[i] = acc;
}

// Sparse-layout variant (batched path).
//
// `expert_outputs` is laid out as `[num_experts * hidden_dim]`; the kernel
// indexes by `expert_ids[k]` so unused experts can sit zero-initialized.
// This is the layout used by Metal's `moe_expert_accum` (sparse) when
// the batched kernels write directly to per-expert slots.
extern "C" __global__ void moe_expert_accum_batched_b(
    float* __restrict__ x,                          // [hidden_dim] in/out
    const float* __restrict__ residual,             // [hidden_dim]
    const float* __restrict__ expert_outputs,       // [num_experts * hidden_dim] (sparse)
    const float* __restrict__ expert_weights,       // [top_k]
    const unsigned int* __restrict__ expert_ids,    // [top_k]
    unsigned int hidden_dim,
    unsigned int top_k)
{
    const unsigned int i = blockIdx.x * BLOCK_DIM + threadIdx.x;
    if (i >= hidden_dim) return;

    // Load top-K weights and IDs into per-thread registers.
    float w[MOE_MAX_TOP_K];
    unsigned int eids[MOE_MAX_TOP_K];
    const unsigned int K = (top_k < MOE_MAX_TOP_K) ? top_k : MOE_MAX_TOP_K;
    for (unsigned int k = 0; k < K; ++k) {
        w[k] = expert_weights[k];
        eids[k] = expert_ids[k];
    }

    float acc = residual[i];
    for (unsigned int k = 0; k < K; ++k) {
        unsigned int eid = eids[k];
        acc += w[k] * expert_outputs[(size_t)eid * (size_t)hidden_dim + i];
    }
    x[i] = acc;
}
