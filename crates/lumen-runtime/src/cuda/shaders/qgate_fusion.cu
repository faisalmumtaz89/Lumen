// Q+gate fusion kernels for Qwen3.5 full-attention layers.
//
// Qwen3.5 full-attention layers have fused Q+gate in attn_q.weight:
//   wq output: [Q_h0(head_dim), gate_h0(head_dim), Q_h1(head_dim), gate_h1(head_dim), ...]
//   Total output: q_dim * 2 = num_heads * head_dim * 2
//
// After deinterleaving, the gate is applied after attention:
//   attn_out_gated = sigmoid(gate) * attn_out
//
// Kernels:
//   deinterleave_qgate: Split interleaved Q+gate into separate Q and gate buffers
//   sigmoid_mul: sigmoid(gate) * x in-place
//   rmsnorm_per_head_inplace: Per-head RMSNorm (shared weight across heads)
//
// NVRTC-compatible: no system includes, extern "C" linkage.

// ============================================================================
// deinterleave_qgate: Split interleaved [Q_h0, gate_h0, Q_h1, gate_h1, ...]
// into separate Q [Q_h0, Q_h1, ...] and gate [gate_h0, gate_h1, ...] buffers.
//
// Input:  qgate[num_heads * head_dim * 2] -- interleaved Q+gate
// Output: q[num_heads * head_dim]         -- Q vectors only
//         gate[num_heads * head_dim]      -- gate vectors only
//
// Grid: ceil(q_dim / 256) blocks of 256 threads, where q_dim = num_heads * head_dim.
// ============================================================================
extern "C" __global__ void deinterleave_qgate(
    const float* __restrict__ qgate,   // [num_heads * head_dim * 2]
    float* __restrict__ q,             // [num_heads * head_dim]
    float* __restrict__ gate,          // [num_heads * head_dim]
    unsigned int head_dim,
    unsigned int num_heads)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int q_dim = num_heads * head_dim;
    if (idx >= q_dim) return;

    // Element idx belongs to head h, position p within head
    unsigned int h = idx / head_dim;
    unsigned int p = idx % head_dim;

    // In the interleaved layout, head h's Q is at offset h * 2 * head_dim
    // and gate is at h * 2 * head_dim + head_dim
    unsigned int qgate_q_offset = h * 2 * head_dim + p;
    unsigned int qgate_g_offset = h * 2 * head_dim + head_dim + p;

    q[idx] = qgate[qgate_q_offset];
    gate[idx] = qgate[qgate_g_offset];
}

// ============================================================================
// sigmoid_mul: Compute sigmoid(gate) * x, writing result to out.
//
// For each element i:
//   out[i] = sigmoid(gate[i]) * x[i]
//         = x[i] / (1 + exp(-gate[i]))
//
// Grid: ceil(n / 256) blocks of 256 threads.
// ============================================================================
extern "C" __global__ void sigmoid_mul(
    const float* __restrict__ gate,    // [n]
    const float* __restrict__ x,       // [n]
    float* __restrict__ out,           // [n]
    unsigned int n)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = gate[idx];
    float sig = 1.0f / (1.0f + expf(-g));
    out[idx] = sig * x[idx];
}

// ============================================================================
// rmsnorm_per_head_inplace: Per-head RMSNorm with shared weight across heads.
//
// For each head h in [0, num_heads):
//   rms = sqrt(mean(x[h*head_dim .. (h+1)*head_dim]^2) + eps)
//   x[h*head_dim + i] = x[h*head_dim + i] / rms * weight[i]
//
// weight is [head_dim], shared across all heads (not [num_heads * head_dim]).
//
// Grid: (num_heads, 1, 1) -- one block per head
// Block: (block_dim, 1, 1) -- threads cooperate within a head
// Shared memory: (block_size / 32) * sizeof(float)
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum_qgate(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

extern "C" __global__ void rmsnorm_per_head_inplace(
    float* __restrict__ x,             // [num_heads * head_dim] modified in-place
    const float* __restrict__ weight,  // [head_dim] shared across heads
    unsigned int num_heads,
    unsigned int head_dim,
    float eps)
{
    extern __shared__ float shared[];

    unsigned int h = blockIdx.x;
    if (h >= num_heads) return;

    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;
    unsigned int warp_id = tid >> 5;
    unsigned int lane_id = tid & 31u;
    unsigned int num_warps = block_size >> 5;

    float* head = x + h * head_dim;

    // Phase 1: Sum of squares
    float ss = 0.0f;
    for (unsigned int i = tid; i < head_dim; i += block_size) {
        float v = head[i];
        ss += v * v;
    }

    ss = warp_reduce_sum_qgate(ss);
    if (lane_id == 0) shared[warp_id] = ss;
    __syncthreads();

    float total_ss = 0.0f;
    if (warp_id == 0) {
        total_ss = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        total_ss = warp_reduce_sum_qgate(total_ss);
    }
    if (tid == 0) shared[0] = total_ss;
    __syncthreads();
    total_ss = shared[0];

    // RMSNorm: x[i] = x[i] / rms * weight[i]
    float rms = sqrtf(total_ss / (float)head_dim + eps);
    float inv_rms = 1.0f / rms;

    // Phase 2: Normalize in-place with shared weight
    for (unsigned int i = tid; i < head_dim; i += block_size) {
        head[i] = head[i] * inv_rms * weight[i];
    }
}
