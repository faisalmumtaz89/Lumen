// Batched Rotary Position Embedding (RoPE) kernel for CUDA prefill.
//
// Applies rotary embeddings to a [batch, dim] Q and K matrix where each
// token t in the batch has position `pos_start + t`. This replaces batch
// sequential calls to the single-token rope_apply kernel.
//
// Grid: 1D, one thread per (token, head, pair) in Q space.
// Threads where pair_idx < num_kv_heads * half_dim also process K.
//
// Layout:
//   Q: [batch, num_q_heads * head_dim]  -- interleaved (x0, x1) pairs
//   K: [batch, num_kv_heads * head_dim] -- interleaved (x0, x1) pairs
//
// For each pair at dimension index d within a head, the rotation is:
//   angle = (pos_start + token_idx) * theta_base^(-2d / head_dim)
//   x0' = x0 * cos(angle) - x1 * sin(angle)
//   x1' = x0 * sin(angle) + x1 * cos(angle)
//
// NVRTC-compatible: no system includes, extern "C" linkage.

extern "C" __global__ void rope_apply_batched(
    float* __restrict__ q,         // [batch, num_q_heads * head_dim]
    float* __restrict__ k,         // [batch, num_kv_heads * head_dim]
    unsigned int pos_start,        // position of first token in batch
    unsigned int batch,
    unsigned int num_q_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    float theta_base)
{
    unsigned int half_dim = head_dim >> 1;
    unsigned int total_q_pairs = num_q_heads * half_dim;
    unsigned int total_k_pairs = num_kv_heads * half_dim;
    unsigned int q_dim = num_q_heads * head_dim;
    unsigned int kv_dim = num_kv_heads * head_dim;

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_work = batch * total_q_pairs;
    if (idx >= total_work) return;

    // Decompose linear index -> (token, pair_within_q_space).
    unsigned int token = idx / total_q_pairs;
    unsigned int pair_idx = idx % total_q_pairs;
    unsigned int pos = pos_start + token;

    // Decompose pair_idx -> (head, dimension_pair).
    unsigned int d = pair_idx % half_dim;
    unsigned int head_in_q = pair_idx / half_dim;
    unsigned int head_offset_q = head_in_q * head_dim;

    // Compute rotation angle. Subtract max not needed here (no exp/softmax).
    float freq = 1.0f / powf(theta_base, (float)(2 * d) / (float)head_dim);
    float angle = (float)pos * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    // Apply rotation to Q.
    {
        unsigned int base = token * q_dim + head_offset_q + 2 * d;
        float x0 = q[base];
        float x1 = q[base + 1];
        q[base]     = x0 * cos_a - x1 * sin_a;
        q[base + 1] = x0 * sin_a + x1 * cos_a;
    }

    // Apply rotation to K (only if this thread maps to a valid K head).
    // In GQA, num_kv_heads <= num_q_heads, so some Q threads have no K work.
    if (pair_idx < total_k_pairs) {
        unsigned int head_in_k = pair_idx / half_dim;
        unsigned int head_offset_k = head_in_k * head_dim;
        unsigned int base = token * kv_dim + head_offset_k + 2 * d;
        float x0 = k[base];
        float x1 = k[base + 1];
        k[base]     = x0 * cos_a - x1 * sin_a;
        k[base + 1] = x0 * sin_a + x1 * cos_a;
    }
}
