// GatedDeltaNet (GDN) CUDA kernels for decode and prefill.
//
// GDN is a recurrent attention mechanism used by Qwen3.5 models. Instead of
// softmax attention, each GDN layer maintains a recurrent state matrix
// h_state[num_heads, val_dim, key_dim] that is updated every token via the
// delta rule.
//
// Decode kernels (single-token):
//   ssm_conv1d_decode     -- 1D causal convolution (one token, circular buffer)
//   gdn_compute_gates     -- GDN gating: alpha (decay), beta (mixing)
//   l2_normalize_heads    -- Per-head L2 normalization
//   gdn_state_update      -- Delta rule state update + output query
//
// Prefill kernels (batched across T tokens):
//   ssm_conv1d_silu_prefill   -- Batched conv1d + SiLU for T tokens (sequential state update)
//   gdn_compute_gates_batched -- Batched gate computation for T * num_heads entries
//   l2_normalize_qk_strided   -- Batched L2 norm for Q and K across T tokens with stride
//   gdn_prefill_fused_v3      -- Warp-parallel fused state update (4x unrolled, 4096 blocks)
//   gdn_prefill_norm_gate     -- Batched RMSNorm + SiLU gate on raw GDN output
//
// NVRTC-compatible: no system includes, extern "C" linkage.

// Warp-level sum reduction using butterfly shuffle.
__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

// ============================================================================
// ssm_conv1d_decode: 1D causal convolution for decode (process one token)
//
// Implements the same circular-buffer convolution as the Metal kernel:
//   for tap in 0..kernel_size-1:
//       sum += weight[gid * kernel_size + tap] * conv_state[(state_pos + tap) % buf_slots * dim + gid]
//   sum += weight[gid * kernel_size + (kernel_size-1)] * input[gid]  // current token
//   output[gid] = sum
//   conv_state[state_pos * dim + gid] = input[gid]  // update circular buffer
//
// conv_state layout: [buf_slots, conv_dim] where buf_slots = kernel_size - 1
// weight layout:     [conv_dim, kernel_size]
//
// Grid: 1D, ceil(conv_dim / 256) blocks of 256 threads
// ============================================================================
extern "C" __global__ void ssm_conv1d_decode(
    float* __restrict__ conv_state,   // [buf_slots, conv_dim] circular buffer R/W
    const float* __restrict__ input,  // [conv_dim] new token values
    const float* __restrict__ weight, // [conv_dim, kernel_size] convolution weights
    float* __restrict__ output,       // [conv_dim] convolved output
    unsigned int conv_dim,
    unsigned int kernel_size,
    unsigned int state_pos)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= conv_dim) return;

    float sum = 0.0f;
    unsigned int buf_slots = kernel_size - 1;

    // Taps 0..kernel_size-2: read from circular buffer (oldest to newest)
    for (unsigned int tap = 0; tap < buf_slots; tap++) {
        unsigned int slot = (state_pos + tap) % buf_slots;
        sum += weight[gid * kernel_size + tap] * conv_state[slot * conv_dim + gid];
    }

    // Tap kernel_size-1: current input (newest)
    sum += weight[gid * kernel_size + buf_slots] * input[gid];

    output[gid] = sum;

    // Update circular buffer: overwrite oldest entry (at state_pos) with current input
    conv_state[state_pos * conv_dim + gid] = input[gid];
}

// ============================================================================
// gdn_compute_gates: Compute GDN decay (alpha) and mixing (beta) gates
//
// Per-head computation (matches Metal gdn_compute_gates and NVLabs reference):
//   softplus(x) = x > 20 ? x : log(1 + exp(x))   (numerically stable)
//   gate = ssm_a[h] * softplus(alpha_proj[h] + dt_bias[h])
//   alpha[h] = exp(gate)        // decay factor in (0, 1)
//   beta[h]  = sigmoid(beta_proj[h])  // mixing rate in (0, 1)
//
// ssm_a stores -exp(A_log) (pre-negated and exponentiated in GGUF).
// Typical values: ssm_a ~ -0.036 (slow decay) to -72 (fast decay).
// gate is negative, so alpha = exp(gate) is in (0, 1).
//
// Grid: 1D, ceil(num_heads / 256) blocks of 256 threads
// ============================================================================
extern "C" __global__ void gdn_compute_gates(
    const float* __restrict__ dt_bias,     // [num_heads] per-head dt bias
    const float* __restrict__ ssm_a,       // [num_heads] -exp(A_log)
    const float* __restrict__ beta_proj,   // [num_heads] pre-sigmoid mixing input
    const float* __restrict__ alpha_proj,  // [num_heads] gk_proj output (from matvec)
    float* __restrict__ alpha_out,         // [num_heads] OUTPUT: decay factors
    float* __restrict__ beta_out,          // [num_heads] OUTPUT: mixing rates
    unsigned int num_heads)
{
    unsigned int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= num_heads) return;

    // Numerically stable softplus: softplus(x) = log(1 + exp(x))
    // For x > 20, softplus(x) ~= x (avoids exp overflow)
    float sp_input = alpha_proj[h] + dt_bias[h];
    float sp;
    if (sp_input > 20.0f) {
        sp = sp_input;
    } else {
        sp = logf(1.0f + expf(sp_input));
    }

    // gate = ssm_a * softplus(gk_proj + dt_bias)
    // ssm_a is negative => gate is negative => alpha in (0, 1)
    float gate = ssm_a[h] * sp;
    alpha_out[h] = expf(gate);

    // beta = sigmoid(beta_proj)
    beta_out[h] = 1.0f / (1.0f + expf(-beta_proj[h]));
}

// ============================================================================
// l2_normalize_heads: Per-head L2 normalization
//
// For each head h: x[h*head_dim .. (h+1)*head_dim] /= max(||x_head||, eps)
//
// Uses shared memory for cross-warp reduction of sum-of-squares.
//
// Grid: (num_heads, 1, 1) -- one block per head
// Block: (min(head_dim, 1024), 1, 1) -- threads cooperate within a head
// Shared memory: (block_size / 32) * 4 bytes
// ============================================================================
extern "C" __global__ void l2_normalize_heads(
    float* __restrict__ x,        // [num_heads * head_dim] modified in-place
    unsigned int num_heads,
    unsigned int head_dim,
    float eps)
{
    extern __shared__ float shared[];

    unsigned int head_idx = blockIdx.x;
    if (head_idx >= num_heads) return;

    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;
    unsigned int warp_id = tid >> 5;
    unsigned int lane_id = tid & 31u;
    unsigned int num_warps = block_size >> 5;

    float* head = x + head_idx * head_dim;

    // Phase 1: Accumulate sum of squares
    float ss = 0.0f;
    for (unsigned int i = tid; i < head_dim; i += block_size) {
        float v = head[i];
        ss += v * v;
    }

    // Warp-level reduction
    ss = warp_reduce_sum(ss);

    // Cross-warp reduction via shared memory
    if (lane_id == 0) {
        shared[warp_id] = ss;
    }
    __syncthreads();

    float total_ss = 0.0f;
    if (warp_id == 0) {
        total_ss = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        total_ss = warp_reduce_sum(total_ss);
    }
    if (tid == 0) {
        shared[0] = total_ss;
    }
    __syncthreads();
    total_ss = shared[0];

    // Compute scale: 1/max(norm, eps)
    float norm = sqrtf(total_ss);
    float scale = (norm > eps) ? (1.0f / norm) : (1.0f / eps);

    // Phase 2: Normalize in-place
    for (unsigned int i = tid; i < head_dim; i += block_size) {
        head[i] *= scale;
    }
}

// ============================================================================
// gdn_state_update: Delta rule recurrent state update + output query
//
// Implements the GDN recurrence for each head (matches Metal gdn_state_output_norm):
//
//   Reference implementation:
//     1. s_decayed = alpha * s_old                    (decay FIRST)
//     2. retrieval = s_decayed^T @ k                  (retrieve from DECAYED state)
//     3. delta = beta * (v - retrieval)
//     4. s_new = s_decayed + outer(k, delta)
//     5. output = s_new @ (q * scale)                 (scale = 1/sqrt(key_dim))
//
// h_state layout: [num_heads, val_dim, key_dim] (transposed for coalesced access)
//   h_state[h, vj, ki] = h_state[h * val_dim * key_dim + vj * key_dim + ki]
//
// Q and K use GQA: kv_head = h % num_kv_heads
// V uses direct head index h (num_heads V-heads, no GQA)
//
// Grid: (num_heads, 1, 1) -- one block per head
// Block: (val_dim, 1, 1) -- one thread per val_dim column
// ============================================================================
extern "C" __global__ void gdn_state_update(
    float* __restrict__ h_state,       // [num_heads, val_dim, key_dim] R/W
    const float* __restrict__ k_norm,  // [num_kv_heads * key_dim]
    const float* __restrict__ v,       // [num_heads * val_dim]
    const float* __restrict__ alpha,   // [num_heads] decay factors
    const float* __restrict__ beta,    // [num_heads] mixing rates
    const float* __restrict__ q_norm,  // [num_kv_heads * key_dim]
    float* __restrict__ output,        // [num_heads * val_dim]
    unsigned int num_heads,
    unsigned int val_dim,
    unsigned int key_dim,
    unsigned int num_kv_heads)
{
    unsigned int h = blockIdx.x;
    if (h >= num_heads) return;

    unsigned int vj = threadIdx.x;
    if (vj >= val_dim) return;

    unsigned int kv_head = h / (num_heads / num_kv_heads);
    float a = alpha[h];
    float b = beta[h];
    float q_scale = rsqrtf((float)key_dim);

    // V has num_heads heads (no GQA) -- use h directly
    float v_val = v[h * val_dim + vj];

    // State row pointer: h_state[h, vj, :] is contiguous in ki
    float* h_row = h_state + h * val_dim * key_dim + vj * key_dim;

    // Q and K both use kv_head for GQA
    const float* q_head = q_norm + kv_head * key_dim;
    const float* k_head = k_norm + kv_head * key_dim;

    // Phase 1: Decay state, then retrieve from DECAYED state
    float retrieval = 0.0f;
    for (unsigned int ki = 0; ki < key_dim; ki++) {
        float h_decayed = a * h_row[ki];
        h_row[ki] = h_decayed;
        retrieval += h_decayed * k_head[ki];
    }

    // Phase 2: Delta rule update + output
    float v_delta = b * (v_val - retrieval);
    float my_out = 0.0f;
    for (unsigned int ki = 0; ki < key_dim; ki++) {
        float h_updated = h_row[ki] + k_head[ki] * v_delta;
        h_row[ki] = h_updated;
        my_out += h_updated * q_head[ki] * q_scale;
    }

    output[h * val_dim + vj] = my_out;
}

// ============================================================================
// PREFILL KERNELS: Batched operations across T tokens
// ============================================================================

// ============================================================================
// ssm_conv1d_silu_prefill: Batched Conv1D + SiLU for T tokens
//
// Processes T tokens sequentially through the conv1d circular buffer state,
// then applies SiLU activation. Sequential in T because each token's conv
// state depends on the previous token's write. Parallel across conv_dim.
//
// Grid: (ceil(conv_dim / 256), 1, 1) blocks of 256 threads
// Each thread handles one channel across all T tokens.
// ============================================================================
extern "C" __global__ void ssm_conv1d_silu_prefill(
    const float* __restrict__ input,    // [T * conv_dim] batched QKV input
    float* __restrict__ conv_state,     // [buf_slots, conv_dim] circular buffer R/W
    const float* __restrict__ weight,   // [conv_dim, kernel_size] convolution weights
    float* __restrict__ output,         // [T * conv_dim] SiLU-activated conv output
    unsigned int conv_dim,
    unsigned int kernel_size,
    unsigned int state_pos,             // initial circular buffer position
    unsigned int T)                     // number of tokens
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= conv_dim) return;

    unsigned int buf_slots = kernel_size - 1;
    unsigned int cur_pos = state_pos;

    for (unsigned int t = 0; t < T; t++) {
        unsigned int t_off = t * conv_dim;
        float inp = input[t_off + gid];

        // Convolution: dot product of circular buffer history with weights
        float sum = 0.0f;
        for (unsigned int tap = 0; tap < buf_slots; tap++) {
            unsigned int slot = (cur_pos + tap) % buf_slots;
            sum += weight[gid * kernel_size + tap] * conv_state[slot * conv_dim + gid];
        }
        // Current token tap (newest)
        sum += weight[gid * kernel_size + buf_slots] * inp;

        // Update circular buffer
        conv_state[cur_pos * conv_dim + gid] = inp;
        cur_pos = (cur_pos + 1) % buf_slots;

        // SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))
        float activated = sum / (1.0f + expf(-sum));
        output[t_off + gid] = activated;
    }
}

// ============================================================================
// gdn_compute_gates_batched: Batched gate computation for T * num_heads entries
//
// Same formula as gdn_compute_gates but processes T tokens at once.
// alpha_proj and beta_proj are [T, num_heads] layout.
// dt_bias and ssm_a are per-head (broadcast across T).
//
// Grid: 1D, ceil(T * num_heads / 256) blocks of 256 threads
// ============================================================================
extern "C" __global__ void gdn_compute_gates_batched(
    const float* __restrict__ dt_bias,     // [num_heads] per-head dt bias
    const float* __restrict__ ssm_a,       // [num_heads] -exp(A_log)
    const float* __restrict__ beta_proj,   // [T * num_heads] pre-sigmoid mixing input
    const float* __restrict__ alpha_proj,  // [T * num_heads] gk_proj output
    float* __restrict__ alpha_out,         // [T * num_heads] OUTPUT: decay factors (overwrite in-place OK)
    float* __restrict__ beta_out,          // [T * num_heads] OUTPUT: mixing rates (overwrite in-place OK)
    unsigned int num_heads,
    unsigned int T)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = T * num_heads;
    if (idx >= total) return;

    unsigned int h = idx % num_heads;

    // Numerically stable softplus
    float sp_input = alpha_proj[idx] + dt_bias[h];
    float sp;
    if (sp_input > 20.0f) {
        sp = sp_input;
    } else {
        sp = logf(1.0f + expf(sp_input));
    }

    float gate = ssm_a[h] * sp;
    alpha_out[idx] = expf(gate);
    beta_out[idx] = 1.0f / (1.0f + expf(-beta_proj[idx]));
}

// ============================================================================
// l2_normalize_qk_strided: Batched L2 normalization for Q and K across T tokens
//
// Normalizes both Q and K vectors for all T tokens in a single launch.
// Input layout: conv_out[T, qkv_dim] where Q is at offset q_offset and K at k_offset.
// Each block handles one (token, kv_head) pair, normalizing both Q and K.
//
// Grid: (num_kv_heads * T, 1, 1) -- one block per (token, kv_head) pair
// Block: (head_dim, 1, 1) -- threads cooperate within a head
// Shared memory: (block_size / 32 + 1) * 4 bytes
// ============================================================================
extern "C" __global__ void l2_normalize_qk_strided(
    float* __restrict__ data,          // [T * qkv_dim] conv output (modified in-place)
    unsigned int num_kv_heads,
    unsigned int head_dim,
    unsigned int T,
    unsigned int stride,               // qkv_dim (distance between tokens)
    unsigned int q_offset,             // offset of Q within each token's data
    unsigned int k_offset)             // offset of K within each token's data
{
    extern __shared__ float shared[];

    unsigned int block_id = blockIdx.x;
    if (block_id >= num_kv_heads * T) return;

    unsigned int t = block_id / num_kv_heads;
    unsigned int kv_head = block_id % num_kv_heads;

    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;
    unsigned int warp_id = tid >> 5;
    unsigned int lane_id = tid & 31u;
    unsigned int num_warps = (block_size + 31) >> 5;
    float eps = 1e-12f;

    // Normalize Q head
    {
        float* head = data + t * stride + q_offset + kv_head * head_dim;
        float ss = 0.0f;
        for (unsigned int i = tid; i < head_dim; i += block_size) {
            float v = head[i];
            ss += v * v;
        }
        ss = warp_reduce_sum(ss);
        if (lane_id == 0) shared[warp_id] = ss;
        __syncthreads();
        float total_ss = 0.0f;
        if (warp_id == 0) {
            total_ss = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
            total_ss = warp_reduce_sum(total_ss);
        }
        if (tid == 0) shared[0] = total_ss;
        __syncthreads();
        total_ss = shared[0];
        float norm = sqrtf(total_ss);
        float scale = (norm > eps) ? (1.0f / norm) : (1.0f / eps);
        for (unsigned int i = tid; i < head_dim; i += block_size) {
            head[i] *= scale;
        }
    }
    __syncthreads();

    // Normalize K head
    {
        float* head = data + t * stride + k_offset + kv_head * head_dim;
        float ss = 0.0f;
        for (unsigned int i = tid; i < head_dim; i += block_size) {
            float v = head[i];
            ss += v * v;
        }
        ss = warp_reduce_sum(ss);
        if (lane_id == 0) shared[warp_id] = ss;
        __syncthreads();
        float total_ss = 0.0f;
        if (warp_id == 0) {
            total_ss = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
            total_ss = warp_reduce_sum(total_ss);
        }
        if (tid == 0) shared[0] = total_ss;
        __syncthreads();
        total_ss = shared[0];
        float norm = sqrtf(total_ss);
        float scale = (norm > eps) ? (1.0f / norm) : (1.0f / eps);
        for (unsigned int i = tid; i < head_dim; i += block_size) {
            head[i] *= scale;
        }
    }
}

// ============================================================================
// gdn_prefill_fused_v3: Warp-parallel GDN prefill state update (4x unrolled)
//
// Direct port of Metal's gdn_prefill_fused_v3_chunked kernel.
// Processes ALL T tokens' state updates in a single kernel launch.
//
// Algorithm: Each block handles one (head, val_dim_column) pair.
// 32 threads (1 warp) cooperate on key_dim=128 (4 elements per thread).
// State is register-resident (4 floats per thread).
// Warp reductions via __shfl_xor_sync (no shared memory needed).
// 4-token loop unrolling hides memory latency.
//
// Grid: (val_dim, num_heads, 1) = (128, 32, 1) = 4096 blocks
// Block: (32, 1, 1) = 1 warp
//
// Buffers:
//   h_state:      [num_heads, val_dim, key_dim] R/W (transposed layout)
//   conv_out_all: [T, qkv_dim] post-conv1d+SiLU+L2-normalized QKV
//   alpha_all:    [T, num_heads] pre-computed decay gates
//   beta_all:     [T, num_heads] pre-computed mixing gates
//   raw_out:      [T, num_heads, val_dim] OUTPUT (raw state query)
// ============================================================================
extern "C" __global__ void gdn_prefill_fused_v3(
    float* __restrict__ h_state,           // [num_heads, val_dim, key_dim] R/W
    const float* __restrict__ conv_out_all, // [T, qkv_dim]
    const float* __restrict__ alpha_all,    // [T, num_heads]
    const float* __restrict__ beta_all,     // [T, num_heads]
    float* __restrict__ raw_out,            // [T, num_heads, val_dim]
    unsigned int n_heads,
    unsigned int key_dim,
    unsigned int val_dim,
    unsigned int n_kv_heads,
    unsigned int T,
    unsigned int qk_dim,
    unsigned int qkv_dim)
{
    unsigned int vj = blockIdx.x;
    unsigned int h  = blockIdx.y;
    if (h >= n_heads || vj >= val_dim) return;

    unsigned int lane = threadIdx.x;  // 0..31
    unsigned int kv_head = h % n_kv_heads;
    float q_scale = rsqrtf((float)key_dim);

    // Each thread handles 4 key_dim elements: lane*4 .. lane*4+3
    unsigned int k_base = lane * 4;

    // Load 4 state elements into registers (transposed layout, contiguous access)
    float* h_row = h_state + h * val_dim * key_dim + vj * key_dim;
    float s0 = h_row[k_base + 0];
    float s1 = h_row[k_base + 1];
    float s2 = h_row[k_base + 2];
    float s3 = h_row[k_base + 3];

    // Precomputed offsets for Q/K/V indexing within conv_out_all
    unsigned int q_head_off = kv_head * key_dim + k_base;
    unsigned int k_head_off = qk_dim + kv_head * key_dim + k_base;
    unsigned int v_head_off = qk_dim + qk_dim + h * val_dim + vj;
    unsigned int out_stride = n_heads * val_dim;
    unsigned int out_base = h * val_dim + vj;

    // Process T tokens in chunks of 4
    unsigned int t = 0;
    unsigned int T_aligned = T & ~3u;  // T rounded down to multiple of 4

    for (; t < T_aligned; t += 4) {
        // --- Prefetch all data for 4 tokens into registers ---
        const float* c0 = conv_out_all + t * qkv_dim;
        const float* c1 = c0 + qkv_dim;
        const float* c2 = c1 + qkv_dim;
        const float* c3 = c2 + qkv_dim;

        float a_0 = alpha_all[t * n_heads + h];
        float a_1 = alpha_all[(t+1) * n_heads + h];
        float a_2 = alpha_all[(t+2) * n_heads + h];
        float a_3 = alpha_all[(t+3) * n_heads + h];

        float b_0 = beta_all[t * n_heads + h];
        float b_1 = beta_all[(t+1) * n_heads + h];
        float b_2 = beta_all[(t+2) * n_heads + h];
        float b_3 = beta_all[(t+3) * n_heads + h];

        // Q values for 4 tokens (4 key elements each)
        float q0_0 = c0[q_head_off];     float q0_1 = c0[q_head_off+1];
        float q0_2 = c0[q_head_off+2];   float q0_3 = c0[q_head_off+3];
        float q1_0 = c1[q_head_off];     float q1_1 = c1[q_head_off+1];
        float q1_2 = c1[q_head_off+2];   float q1_3 = c1[q_head_off+3];
        float q2_0 = c2[q_head_off];     float q2_1 = c2[q_head_off+1];
        float q2_2 = c2[q_head_off+2];   float q2_3 = c2[q_head_off+3];
        float q3_0 = c3[q_head_off];     float q3_1 = c3[q_head_off+1];
        float q3_2 = c3[q_head_off+2];   float q3_3 = c3[q_head_off+3];

        // K values for 4 tokens
        float k0_0 = c0[k_head_off];     float k0_1 = c0[k_head_off+1];
        float k0_2 = c0[k_head_off+2];   float k0_3 = c0[k_head_off+3];
        float k1_0 = c1[k_head_off];     float k1_1 = c1[k_head_off+1];
        float k1_2 = c1[k_head_off+2];   float k1_3 = c1[k_head_off+3];
        float k2_0 = c2[k_head_off];     float k2_1 = c2[k_head_off+1];
        float k2_2 = c2[k_head_off+2];   float k2_3 = c2[k_head_off+3];
        float k3_0 = c3[k_head_off];     float k3_1 = c3[k_head_off+1];
        float k3_2 = c3[k_head_off+2];   float k3_3 = c3[k_head_off+3];

        // V values for 4 tokens
        float v0 = c0[v_head_off];
        float v1 = c1[v_head_off];
        float v2 = c2[v_head_off];
        float v3_v = c3[v_head_off];

        // --- Token 0: recurrence step ---
        {
            float d0 = a_0 * s0;  float d1 = a_0 * s1;
            float d2 = a_0 * s2;  float d3 = a_0 * s3;
            float retrieval = warp_reduce_sum(d0*k0_0 + d1*k0_1 + d2*k0_2 + d3*k0_3);
            float v_delta = b_0 * (v0 - retrieval);
            s0 = d0 + k0_0 * v_delta;  s1 = d1 + k0_1 * v_delta;
            s2 = d2 + k0_2 * v_delta;  s3 = d3 + k0_3 * v_delta;
            float my_out = warp_reduce_sum(s0*q0_0 + s1*q0_1 + s2*q0_2 + s3*q0_3) * q_scale;
            if (lane == 0) raw_out[t * out_stride + out_base] = my_out;
        }

        // --- Token 1: recurrence step ---
        {
            float d0 = a_1 * s0;  float d1 = a_1 * s1;
            float d2 = a_1 * s2;  float d3 = a_1 * s3;
            float retrieval = warp_reduce_sum(d0*k1_0 + d1*k1_1 + d2*k1_2 + d3*k1_3);
            float v_delta = b_1 * (v1 - retrieval);
            s0 = d0 + k1_0 * v_delta;  s1 = d1 + k1_1 * v_delta;
            s2 = d2 + k1_2 * v_delta;  s3 = d3 + k1_3 * v_delta;
            float my_out = warp_reduce_sum(s0*q1_0 + s1*q1_1 + s2*q1_2 + s3*q1_3) * q_scale;
            if (lane == 0) raw_out[(t+1) * out_stride + out_base] = my_out;
        }

        // --- Token 2: recurrence step ---
        {
            float d0 = a_2 * s0;  float d1 = a_2 * s1;
            float d2 = a_2 * s2;  float d3 = a_2 * s3;
            float retrieval = warp_reduce_sum(d0*k2_0 + d1*k2_1 + d2*k2_2 + d3*k2_3);
            float v_delta = b_2 * (v2 - retrieval);
            s0 = d0 + k2_0 * v_delta;  s1 = d1 + k2_1 * v_delta;
            s2 = d2 + k2_2 * v_delta;  s3 = d3 + k2_3 * v_delta;
            float my_out = warp_reduce_sum(s0*q2_0 + s1*q2_1 + s2*q2_2 + s3*q2_3) * q_scale;
            if (lane == 0) raw_out[(t+2) * out_stride + out_base] = my_out;
        }

        // --- Token 3: recurrence step ---
        {
            float d0 = a_3 * s0;  float d1 = a_3 * s1;
            float d2 = a_3 * s2;  float d3 = a_3 * s3;
            float retrieval = warp_reduce_sum(d0*k3_0 + d1*k3_1 + d2*k3_2 + d3*k3_3);
            float v_delta = b_3 * (v3_v - retrieval);
            s0 = d0 + k3_0 * v_delta;  s1 = d1 + k3_1 * v_delta;
            s2 = d2 + k3_2 * v_delta;  s3 = d3 + k3_3 * v_delta;
            float my_out = warp_reduce_sum(s0*q3_0 + s1*q3_1 + s2*q3_2 + s3*q3_3) * q_scale;
            if (lane == 0) raw_out[(t+3) * out_stride + out_base] = my_out;
        }
    }

    // --- Tail: process remaining 0-3 tokens one at a time ---
    for (; t < T; t++) {
        float a = alpha_all[t * n_heads + h];
        float b = beta_all[t * n_heads + h];

        const float* conv_t = conv_out_all + t * qkv_dim;

        float qn0 = conv_t[q_head_off];     float qn1 = conv_t[q_head_off+1];
        float qn2 = conv_t[q_head_off+2];   float qn3 = conv_t[q_head_off+3];
        float kn0 = conv_t[k_head_off];     float kn1 = conv_t[k_head_off+1];
        float kn2 = conv_t[k_head_off+2];   float kn3 = conv_t[k_head_off+3];
        float v_val = conv_t[v_head_off];

        float d0 = a * s0;  float d1 = a * s1;
        float d2 = a * s2;  float d3 = a * s3;
        float retrieval = warp_reduce_sum(d0*kn0 + d1*kn1 + d2*kn2 + d3*kn3);
        float v_delta = b * (v_val - retrieval);
        s0 = d0 + kn0 * v_delta;  s1 = d1 + kn1 * v_delta;
        s2 = d2 + kn2 * v_delta;  s3 = d3 + kn3 * v_delta;
        float my_out = warp_reduce_sum(s0*qn0 + s1*qn1 + s2*qn2 + s3*qn3) * q_scale;
        if (lane == 0) raw_out[t * out_stride + out_base] = my_out;
    }

    // Write state back to device memory (transposed layout, contiguous)
    h_row[k_base + 0] = s0;
    h_row[k_base + 1] = s1;
    h_row[k_base + 2] = s2;
    h_row[k_base + 3] = s3;
}

// ============================================================================
// gdn_prefill_norm_gate: Batched RMSNorm + SiLU gate on raw GDN output
//
// Post-processing for gdn_prefill_fused_v3. Applies per-head RMSNorm + learned
// scale + SiLU output gating for all T tokens.
//
// This kernel CANNOT be fused into gdn_prefill_fused_v3 because:
// - The state kernel is grid (val_dim, num_heads) -- each block owns one vj
// - RMSNorm needs sum-of-squares across all val_dim values for a given (token, head)
// - Those val_dim values live in val_dim SEPARATE blocks
// - CUDA has no cross-block synchronization within a single launch
//
// raw_out:    [T, num_heads, val_dim] -- raw output from state kernel
// gate_all:   [T, num_heads * val_dim] -- SiLU gate (from batched GEMM)
// norm_scale: [scale_n_heads, val_dim] -- learned RMSNorm scale
// ssm_out:    [T, num_heads * val_dim] -- OUTPUT (normed + gated)
//
// Grid: (num_heads, T, 1) -- one block per (head, token) pair
// Block: (val_dim, 1, 1) -- threads cooperate across val_dim
// Shared memory: (block_size / 32 + 1) * 4 bytes for cross-warp reduction
// ============================================================================
extern "C" __global__ void gdn_prefill_norm_gate(
    const float* __restrict__ raw_out,    // [T, num_heads, val_dim]
    const float* __restrict__ gate_all,   // [T, num_heads * val_dim]
    const float* __restrict__ norm_scale, // [scale_n_heads, val_dim]
    float* __restrict__ ssm_out,          // [T, num_heads * val_dim]
    unsigned int num_heads,
    unsigned int val_dim,
    float eps,
    unsigned int scale_n_heads,
    unsigned int T)
{
    extern __shared__ float shared[];

    unsigned int h = blockIdx.x;
    unsigned int t = blockIdx.y;
    if (h >= num_heads || t >= T) return;

    unsigned int vj = threadIdx.x;
    unsigned int block_size = blockDim.x;
    unsigned int warp_id = vj >> 5;
    unsigned int lane_id = vj & 31u;
    unsigned int num_warps = (block_size + 31) >> 5;

    unsigned int idx = t * num_heads * val_dim + h * val_dim + vj;

    // Load raw output value
    float val = (vj < val_dim) ? raw_out[idx] : 0.0f;

    // RMSNorm: compute sum of squares across val_dim
    float ss = val * val;
    ss = warp_reduce_sum(ss);

    // Cross-warp reduction via shared memory
    if (lane_id == 0) {
        shared[warp_id] = ss;
    }
    __syncthreads();

    float total_ss = 0.0f;
    if (warp_id == 0) {
        total_ss = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        total_ss = warp_reduce_sum(total_ss);
    }
    if (vj == 0) {
        shared[0] = total_ss;
    }
    __syncthreads();
    total_ss = shared[0];

    if (vj >= val_dim) return;

    // RMSNorm: normalize
    float rms = sqrtf(total_ss / (float)val_dim + eps);
    float inv_rms = 1.0f / rms;

    // Apply learned scale (broadcast if scale_n_heads == 1)
    unsigned int scale_h = (scale_n_heads == 1) ? 0 : h;
    float normed = val * inv_rms * norm_scale[scale_h * val_dim + vj];

    // SiLU gate: silu(gate) * normed
    unsigned int gate_idx = t * num_heads * val_dim + h * val_dim + vj;
    float g = gate_all[gate_idx];
    float silu_g = g / (1.0f + expf(-g));

    ssm_out[gate_idx] = silu_g * normed;
}
