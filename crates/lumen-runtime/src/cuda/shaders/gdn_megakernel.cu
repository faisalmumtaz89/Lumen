// GDN Decode Megakernel: fuses 8 per-token kernel launches into 2.
//
// Kernel 1 (gdn_decode_megakernel): fuses 6 kernels into 1
//   Phase 1: Conv1D + SiLU on this head's Q/K/V elements + conv_state update
//   Phase 2: Compute gates (alpha/beta) from SSM parameters
//   Phase 3: L2-normalize Q and K per head
//   Phase 4: Delta-rule state update + output query
//
// Kernel 2 (gdn_rmsnorm_silu_gate): fuses 2 kernels into 1
//   Phase 1: Global RMSNorm on state output
//   Phase 2: SiLU(gate) * normed_output
//
// Net result: 8 launches -> 2 per GDN layer, 192 -> 48 per token.
//
// Kernel 1 grid: (num_heads, 1, 1), block: (block_dim, 1, 1)
//   block_dim should be >= head_dim (128 for Qwen3.5)
//   Shared memory: (32 + 2*head_dim) * sizeof(float)
//
// Kernel 2 grid: (1, 1, 1), block: (block_size, 1, 1)
//   Shared memory: (block_size / 32) * sizeof(float)
//
// NVRTC-compatible: no system includes, extern "C" linkage.

__device__ __forceinline__ float warp_reduce_sum_mega(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

// Block-level sum reduction. Returns result broadcast to all threads.
__device__ __forceinline__ float block_reduce_sum_mega(
    float val, float* warp_scratch,
    unsigned int tid, unsigned int block_size)
{
    unsigned int warp_id = tid >> 5;
    unsigned int lane_id = tid & 31u;
    unsigned int num_warps = block_size >> 5;

    val = warp_reduce_sum_mega(val);
    if (lane_id == 0) warp_scratch[warp_id] = val;
    __syncthreads();

    float total = 0.0f;
    if (warp_id == 0) {
        total = (lane_id < num_warps) ? warp_scratch[lane_id] : 0.0f;
        total = warp_reduce_sum_mega(total);
    }
    if (tid == 0) warp_scratch[0] = total;
    __syncthreads();
    return warp_scratch[0];
}

// Conv1D + SiLU for a single element, with conv_state update.
// Reads conv_state history BEFORE writing current input to avoid RAW hazard.
__device__ __forceinline__ float conv1d_silu_update(
    unsigned int idx,
    const float* __restrict__ qkv_buf,
    const float* __restrict__ conv_weight,
    float* __restrict__ conv_state,
    unsigned int qkv_dim,
    unsigned int kernel_size,
    unsigned int buf_slots,
    unsigned int state_pos)
{
    float sum = 0.0f;
    float input_val = qkv_buf[idx];

    // Read history taps from circular buffer (oldest to newest)
    for (unsigned int tap = 0; tap < buf_slots; tap++) {
        unsigned int slot = (state_pos + tap) % buf_slots;
        sum += conv_weight[idx * kernel_size + tap] * conv_state[slot * qkv_dim + idx];
    }

    // Current input (newest tap)
    sum += conv_weight[idx * kernel_size + buf_slots] * input_val;

    // Update circular buffer: write current input at state_pos
    conv_state[state_pos * qkv_dim + idx] = input_val;

    // SiLU activation: x / (1 + exp(-x))
    return sum / (1.0f + expf(-sum));
}


// ============================================================================
// gdn_decode_megakernel
//
// Grid: (num_heads) blocks. Each block owns one head.
// Block: block_dim threads (>= head_dim, typically 128 or 256).
//
// Phase 1: Conv1D + SiLU for this head's Q, K, V elements + conv_state update.
//   Each element's conv_state is updated immediately after reading, so there is
//   no cross-block read-write hazard. Q/K elements shared across GQA blocks
//   get the same write value (qkv_buf[idx]) -- benign race.
//
// Phase 2: Compute gates (1 thread per block, broadcast via shmem).
// Phase 3: L2-normalize Q, K via block reduction.
// Phase 4: Delta-rule state update (1 thread per val_dim element).
// ============================================================================
extern "C" __global__ void gdn_decode_megakernel(
    // Mutable persistent state
    float* __restrict__ conv_state,    // [(kernel_size-1) * qkv_dim] circular buffer
    float* __restrict__ h_state,       // [num_heads * head_dim * head_dim] recurrent state

    // Inputs from prior matvec steps
    const float* __restrict__ qkv_buf,     // [qkv_dim] from QKV matvec
    const float* __restrict__ alpha_raw,   // [num_heads] from alpha matvec
    const float* __restrict__ beta_raw,    // [num_heads] from beta matvec

    // Layer weights
    const float* __restrict__ conv_weight, // [qkv_dim, kernel_size] row-major
    const float* __restrict__ dt_bias,     // [num_heads]
    const float* __restrict__ ssm_a,       // [num_heads] stores -exp(A_log)

    // Output
    float* __restrict__ output,            // [value_dim] raw state query output

    // Dimensions
    unsigned int num_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,       // val_dim per head = key_dim per head
    unsigned int qkv_dim,        // qk_dim + qk_dim + value_dim
    unsigned int qk_dim,         // num_kv_heads * head_dim
    unsigned int value_dim,      // num_heads * head_dim
    unsigned int kernel_size,    // conv1d kernel size (4)
    unsigned int state_pos)      // current circular buffer position [0..kernel_size-2]
{
    extern __shared__ float shmem[];

    unsigned int h = blockIdx.x;
    if (h >= num_heads) return;

    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;
    unsigned int buf_slots = kernel_size - 1;
    unsigned int kv_head = h % num_kv_heads;

    // Shared memory layout:
    //   warp_scratch[32]   -- reduction temporaries
    //   q_shmem[head_dim]  -- Q for this head's kv_head (after conv+silu+norm)
    //   k_shmem[head_dim]  -- K for this head's kv_head (after conv+silu+norm)
    float* warp_scratch = shmem;
    float* q_shmem = shmem + 32;
    float* k_shmem = shmem + 32 + head_dim;

    // ====================================================================
    // Phase 1: Conv1D + SiLU for this head's Q, K, V elements.
    //
    // Q range: [q_base .. q_base + head_dim) within [0..qk_dim)
    // K range: [k_base .. k_base + head_dim) within [qk_dim..2*qk_dim)
    // V range: [v_base .. v_base + head_dim) within [2*qk_dim..qkv_dim)
    //
    // Conv_state is updated per-element inside conv1d_silu_update().
    // GQA blocks sharing the same kv_head redundantly process Q/K elements
    // but write the same value -- benign write-write race.
    // All qkv_dim elements are covered: Q(2048) + K(2048) + V(4096) = 8192.
    // ====================================================================
    unsigned int q_base = kv_head * head_dim;
    unsigned int k_base = qk_dim + kv_head * head_dim;
    unsigned int v_base = 2 * qk_dim + h * head_dim;

    for (unsigned int i = tid; i < head_dim; i += block_size) {
        q_shmem[i] = conv1d_silu_update(
            q_base + i, qkv_buf, conv_weight, conv_state,
            qkv_dim, kernel_size, buf_slots, state_pos);
        k_shmem[i] = conv1d_silu_update(
            k_base + i, qkv_buf, conv_weight, conv_state,
            qkv_dim, kernel_size, buf_slots, state_pos);
        output[h * head_dim + i] = conv1d_silu_update(
            v_base + i, qkv_buf, conv_weight, conv_state,
            qkv_dim, kernel_size, buf_slots, state_pos);
    }

    __syncthreads();

    // ====================================================================
    // Phase 2: Compute gates (alpha decay, beta mixing) for this head.
    // Single thread computes, broadcasts via shared memory.
    // ====================================================================
    float alpha_val, beta_val;
    if (tid == 0) {
        float sp_input = alpha_raw[h] + dt_bias[h];
        float sp = (sp_input > 20.0f) ? sp_input : logf(1.0f + expf(sp_input));
        warp_scratch[0] = expf(ssm_a[h] * sp);                  // alpha
        warp_scratch[1] = 1.0f / (1.0f + expf(-beta_raw[h]));   // beta
    }
    __syncthreads();
    alpha_val = warp_scratch[0];
    beta_val = warp_scratch[1];

    // ====================================================================
    // Phase 3: L2-normalize Q and K per head.
    // ====================================================================

    // L2-normalize Q
    {
        float ss = 0.0f;
        for (unsigned int i = tid; i < head_dim; i += block_size) {
            float v = q_shmem[i];
            ss += v * v;
        }
        float total = block_reduce_sum_mega(ss, warp_scratch, tid, block_size);
        float norm = sqrtf(total);
        float inv = (norm > 1e-12f) ? (1.0f / norm) : (1.0f / 1e-12f);
        for (unsigned int i = tid; i < head_dim; i += block_size) {
            q_shmem[i] *= inv;
        }
    }
    __syncthreads();

    // L2-normalize K
    {
        float ss = 0.0f;
        for (unsigned int i = tid; i < head_dim; i += block_size) {
            float v = k_shmem[i];
            ss += v * v;
        }
        float total = block_reduce_sum_mega(ss, warp_scratch, tid, block_size);
        float norm = sqrtf(total);
        float inv = (norm > 1e-12f) ? (1.0f / norm) : (1.0f / 1e-12f);
        for (unsigned int i = tid; i < head_dim; i += block_size) {
            k_shmem[i] *= inv;
        }
    }
    __syncthreads();

    // ====================================================================
    // Phase 4: Delta-rule state update + output query.
    //
    // h_state[h, vj, ki] = h_state[h*head_dim*head_dim + vj*head_dim + ki]
    //
    //   s_decayed = alpha * s_old
    //   retrieval = s_decayed^T @ k
    //   delta = beta * (v - retrieval)
    //   s_new = s_decayed + outer(k, delta)
    //   output = s_new @ (q * rsqrt(key_dim))
    //
    // One thread per val_dim element. Threads >= head_dim are idle.
    // ====================================================================
    if (tid < head_dim) {
        unsigned int vj = tid;
        float inv_sqrt_key = rsqrtf((float)head_dim);
        float v_val = output[h * head_dim + vj];  // V from phase 1 temp storage

        float* h_row = h_state + (unsigned long long)h * head_dim * head_dim
                                + (unsigned long long)vj * head_dim;

        // Decay + retrieve from decayed state
        float retrieval = 0.0f;
        for (unsigned int ki = 0; ki < head_dim; ki++) {
            float h_decayed = alpha_val * h_row[ki];
            h_row[ki] = h_decayed;
            retrieval += h_decayed * k_shmem[ki];
        }

        // Delta update + output
        float v_delta = beta_val * (v_val - retrieval);
        float my_out = 0.0f;
        for (unsigned int ki = 0; ki < head_dim; ki++) {
            float h_updated = h_row[ki] + k_shmem[ki] * v_delta;
            h_row[ki] = h_updated;
            my_out += h_updated * q_shmem[ki] * inv_sqrt_key;
        }

        output[h * head_dim + vj] = my_out;
    }
}


// ============================================================================
// gdn_rmsnorm_silu_gate: Fused RMSNorm + SiLU(gate) * normed_output.
//
// Combines steps 8 and 10 of the GDN decode pipeline:
//   normed[i] = raw_output[i] * rsqrt(mean(raw_output^2) + eps) * ssm_norm[i]
//   out[i] = silu(gate[i]) * normed[i]
//
// Single-block kernel. Grid: (1), Block: (block_size).
// Shared memory: (block_size / 32) * sizeof(float).
// ============================================================================
extern "C" __global__ void gdn_rmsnorm_silu_gate(
    const float* __restrict__ raw_output,  // [dim] raw state query output
    const float* __restrict__ ssm_norm,    // [dim] RMSNorm scale weights
    const float* __restrict__ gate,        // [dim] gate matvec output
    float* __restrict__ out,               // [dim] final fused output
    float eps,
    unsigned int dim)
{
    extern __shared__ float shared[];

    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;
    unsigned int warp_id = tid >> 5;
    unsigned int lane_id = tid & 31u;
    unsigned int num_warps = block_size >> 5;

    // Phase 1: Sum of squares for RMSNorm.
    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < dim; i += block_size) {
        float val = raw_output[i];
        sum_sq += val * val;
    }

    sum_sq = warp_reduce_sum_mega(sum_sq);
    if (lane_id == 0) shared[warp_id] = sum_sq;
    __syncthreads();

    float total = 0.0f;
    if (warp_id == 0) {
        total = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        total = warp_reduce_sum_mega(total);
    }
    if (tid == 0) shared[0] = 1.0f / sqrtf(total / (float)dim + eps);
    __syncthreads();
    float rms = shared[0];

    // Phase 2: Fused RMSNorm + SiLU(gate) * normed.
    for (unsigned int i = tid; i < dim; i += block_size) {
        float normed = raw_output[i] * rms * ssm_norm[i];
        float g = gate[i];
        float silu_g = g / (1.0f + expf(-g));
        out[i] = silu_g * normed;
    }
}
