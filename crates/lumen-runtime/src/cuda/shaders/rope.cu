// Rotary Position Embedding (RoPE) kernel for CUDA.
//
// Applies rotary embeddings to interleaved (x0, x1) pairs.
// Each pair at dimension index d is rotated by angle = pos * theta^(-2d/dim):
//   x0' = x0 * cos(angle) - x1 * sin(angle)
//   x1' = x0 * sin(angle) + x1 * cos(angle)
//
// Grid: 1D, one thread per pair (total = num_heads * half_dim).

// `rotary_dim`: number of dimensions to rotate per head (0 = full head_dim).
// For Qwen3.5: rotary_dim=64 out of head_dim=256 — only first 32 pairs rotated.
// Freq base uses rotary_dim (not head_dim) for correct frequency spacing.
extern "C" __global__ void rope_apply(
    float* __restrict__ q,
    float* __restrict__ k,
    unsigned int pos,
    unsigned int num_q_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    float theta_base,
    unsigned int rotary_dim)
{
    // Actual rotary dimension: 0 means full head_dim (backward compatible)
    unsigned int actual_rot = (rotary_dim > 0 && rotary_dim < head_dim) ? rotary_dim : head_dim;
    unsigned int half_rot = actual_rot >> 1;
    unsigned int total_q_pairs = num_q_heads * half_rot;
    unsigned int total_k_pairs = num_kv_heads * half_rot;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Process Q heads.
    if (idx < total_q_pairs) {
        unsigned int d = idx % half_rot;
        unsigned int head_offset = (idx / half_rot) * head_dim;

        // Freq base uses rotary_dim for correct frequency spacing
        float freq = 1.0f / powf(theta_base, (float)(2 * d) / (float)actual_rot);
        float angle = (float)pos * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);

        unsigned int i0 = head_offset + 2 * d;
        unsigned int i1 = i0 + 1;
        float x0 = q[i0];
        float x1 = q[i1];
        q[i0] = x0 * cos_a - x1 * sin_a;
        q[i1] = x0 * sin_a + x1 * cos_a;
    }

    // Process K heads (reusing the same thread grid).
    if (idx < total_k_pairs) {
        unsigned int d = idx % half_rot;
        unsigned int head_offset = (idx / half_rot) * head_dim;

        float freq = 1.0f / powf(theta_base, (float)(2 * d) / (float)actual_rot);
        float angle = (float)pos * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);

        unsigned int i0 = head_offset + 2 * d;
        unsigned int i1 = i0 + 1;
        float x0 = k[i0];
        float x1 = k[i1];
        k[i0] = x0 * cos_a - x1 * sin_a;
        k[i1] = x0 * sin_a + x1 * cos_a;
    }
}

// NeoX-style RoPE: pairs at (d, d + half_rot) instead of interleaved (2d, 2d+1).
// Used by Qwen2, Qwen3.5 family models. Same frequency calculation as standard RoPE.
extern "C" __global__ void rope_apply_neox(
    float* __restrict__ q,
    float* __restrict__ k,
    unsigned int pos,
    unsigned int num_q_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    float theta_base,
    unsigned int rotary_dim)
{
    unsigned int actual_rot = (rotary_dim > 0 && rotary_dim < head_dim) ? rotary_dim : head_dim;
    unsigned int half_rot = actual_rot >> 1;
    unsigned int total_q_pairs = num_q_heads * half_rot;
    unsigned int total_k_pairs = num_kv_heads * half_rot;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Process Q heads (NeoX pairing: d and d + half_rot).
    if (idx < total_q_pairs) {
        unsigned int d = idx % half_rot;
        unsigned int head_offset = (idx / half_rot) * head_dim;

        float freq = 1.0f / powf(theta_base, (float)(2 * d) / (float)actual_rot);
        float angle = (float)pos * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);

        unsigned int i0 = head_offset + d;
        unsigned int i1 = head_offset + d + half_rot;
        float x0 = q[i0];
        float x1 = q[i1];
        q[i0] = x0 * cos_a - x1 * sin_a;
        q[i1] = x0 * sin_a + x1 * cos_a;
    }

    // Process K heads (NeoX pairing).
    if (idx < total_k_pairs) {
        unsigned int d = idx % half_rot;
        unsigned int head_offset = (idx / half_rot) * head_dim;

        float freq = 1.0f / powf(theta_base, (float)(2 * d) / (float)actual_rot);
        float angle = (float)pos * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);

        unsigned int i0 = head_offset + d;
        unsigned int i1 = head_offset + d + half_rot;
        float x0 = k[i0];
        float x1 = k[i1];
        k[i0] = x0 * cos_a - x1 * sin_a;
        k[i1] = x0 * sin_a + x1 * cos_a;
    }
}
