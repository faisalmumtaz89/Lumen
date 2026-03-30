// GatedDeltaNet (GDN) CUDA kernels for single-token decode.
//
// GDN is a recurrent attention mechanism used by Qwen3.5 models. Instead of
// softmax attention, each GDN layer maintains a recurrent state matrix
// h_state[num_heads, val_dim, key_dim] that is updated every token via the
// delta rule.
//
// Kernels:
//   ssm_conv1d_decode     -- 1D causal convolution (one token, circular buffer)
//   gdn_compute_gates     -- GDN gating: alpha (decay), beta (mixing)
//   l2_normalize_heads    -- Per-head L2 normalization
//   gdn_state_update      -- Delta rule state update + output query
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
//   Reference (llama.cpp build_delta_net_autoregressive):
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
