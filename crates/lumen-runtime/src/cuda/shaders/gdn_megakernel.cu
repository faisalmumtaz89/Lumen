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
    //
    // Vectorized with float4 loads/stores: 32 iterations instead of 128.
    // h_state is 256-byte aligned (CUDA alloc_zeros), head_dim divisible by 4.
    // k_shmem/q_shmem are loaded into register groups of 4 for reuse.
    // ====================================================================
    if (tid < head_dim) {
        unsigned int vj = tid;
        float inv_sqrt_key = rsqrtf((float)head_dim);
        float v_val = output[h * head_dim + vj];  // V from phase 1 temp storage

        float* h_row = h_state + (unsigned long long)h * head_dim * head_dim
                                + (unsigned long long)vj * head_dim;
        unsigned int vec_len = head_dim >> 2;  // head_dim / 4
        float4* h_row4 = (float4*)h_row;

        // Pass 1: Decay + retrieve from decayed state (float4 vectorized)
        float retrieval = 0.0f;
        #pragma unroll 8
        for (unsigned int ki4 = 0; ki4 < vec_len; ki4++) {
            float4 h4 = h_row4[ki4];
            unsigned int base = ki4 << 2;
            float k0 = k_shmem[base];
            float k1 = k_shmem[base + 1];
            float k2 = k_shmem[base + 2];
            float k3 = k_shmem[base + 3];
            h4.x *= alpha_val;
            h4.y *= alpha_val;
            h4.z *= alpha_val;
            h4.w *= alpha_val;
            h_row4[ki4] = h4;
            retrieval += h4.x * k0 + h4.y * k1 + h4.z * k2 + h4.w * k3;
        }

        // Pass 2: Delta update + output query (float4 vectorized)
        float v_delta = beta_val * (v_val - retrieval);
        float my_out = 0.0f;
        #pragma unroll 8
        for (unsigned int ki4 = 0; ki4 < vec_len; ki4++) {
            float4 h4 = h_row4[ki4];
            unsigned int base = ki4 << 2;
            float k0 = k_shmem[base];
            float k1 = k_shmem[base + 1];
            float k2 = k_shmem[base + 2];
            float k3 = k_shmem[base + 3];
            h4.x += k0 * v_delta;
            h4.y += k1 * v_delta;
            h4.z += k2 * v_delta;
            h4.w += k3 * v_delta;
            h_row4[ki4] = h4;
            float q0 = q_shmem[base];
            float q1 = q_shmem[base + 1];
            float q2 = q_shmem[base + 2];
            float q3 = q_shmem[base + 3];
            my_out += h4.x * q0 + h4.y * q1 + h4.z * q2 + h4.w * q3;
        }

        output[h * head_dim + vj] = my_out * inv_sqrt_key;
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


// ============================================================================
// gdn_decode_megakernel_graph -- CUDA graph-compatible variant
//
// Identical to gdn_decode_megakernel except state_pos is read from a device
// pointer instead of a scalar argument. The device pointer is baked into the
// captured graph; only its value (updated via htod memcpy before replay) changes
// between tokens.
//
// Grid: (num_heads) blocks. Each block owns one head.
// Block: block_dim threads (>= head_dim, typically 128 or 256).
// ============================================================================
extern "C" __global__ void gdn_decode_megakernel_graph(
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
    const unsigned int* __restrict__ p_state_pos)  // device pointer to state_pos
{
    unsigned int state_pos = *p_state_pos;

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
    // ====================================================================
    if (tid < head_dim) {
        unsigned int vj = tid;
        float inv_sqrt_key = rsqrtf((float)head_dim);
        float v_val = output[h * head_dim + vj];

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
// F64-internal-accumulator twins of the decode megakernel.
//
// PURPOSE: restore DECODE/PREFILL PRECISION PARITY on the LIVE MoE decode path.
//
// The F32 `gdn_decode_megakernel` above is the active default MoE decode kernel.
// The batched MoE PREFILL scan, by contrast, runs F64-accum
// (`gdn_prefill_fused_v3_f64accum` + `l2_normalize_qk_strided_f64accum`,
// selected by `gdn_f64_accum_enabled() = model_is_moe()`). Same decay-first
// FORM, different PRECISION (decode F32 vs prefill F64). The per-step F32-ULP
// rounding makes the decode-built recurrent `h_state` drift away from the
// prefill-built state; the 256-expert MoE router amplifies the drift into
// expert-selection flips → cascaded garble.
//
// These twins keep the EXACT same structure / dispatch geometry / shmem layout
// as the F32 megakernels but accumulate the L2-norm sum-of-squares and the
// delta-rule recurrence (decay / retrieval / v_delta / state) in F64 in
// registers, writing F32 back to `h_state` — mirroring the prefill F64 kernels
// bit-for-bit in arithmetic. Conv1D+SiLU stays F32 (the prefill conv is F32 in
// `ssm_conv1d_silu_prefill`); the alpha/beta gates stay F32 (matching the F32
// `gdn_compute_gates_batched` the prefill uses). Only the two F64-sensitive
// stages — L2-norm reduction and the recurrent state update — are promoted, so
// the decode state tracks the prefill state to F64 rounding.
//
// Gated MoE-only (env `LUMEN_CUDA_GDN_DECODE_MEGAKERNEL_F64`); OFF is
// byte-identical to the F32 path (these kernels are simply not dispatched).
//
// NVRTC-compatible: no system includes, extern "C" linkage.
// ============================================================================

// Block-level F64 sum reduction. Same butterfly-XOR ordering pattern as
// `block_reduce_sum_mega` but accumulates in double. `warp_scratch_d` is a
// double scratch region in shmem (the megakernel's shmem is laid out as float;
// we reinterpret the first 32 floats = 16 doubles, enough for <=16 warps).
__device__ __forceinline__ double warp_reduce_sum_mega_f64(double val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

__device__ __forceinline__ double block_reduce_sum_mega_f64(
    double val, double* warp_scratch_d,
    unsigned int tid, unsigned int block_size)
{
    unsigned int warp_id = tid >> 5;
    unsigned int lane_id = tid & 31u;
    unsigned int num_warps = block_size >> 5;

    val = warp_reduce_sum_mega_f64(val);
    if (lane_id == 0) warp_scratch_d[warp_id] = val;
    __syncthreads();

    double total = 0.0;
    if (warp_id == 0) {
        total = (lane_id < num_warps) ? warp_scratch_d[lane_id] : 0.0;
        total = warp_reduce_sum_mega_f64(total);
    }
    if (tid == 0) warp_scratch_d[0] = total;
    __syncthreads();
    return warp_scratch_d[0];
}

// Shared body for both the eager and graph F64 megakernel variants. The only
// difference between the two public entry points is how `state_pos` is sourced
// (scalar arg vs device pointer), exactly as for the F32 pair.
__device__ __forceinline__ void gdn_decode_megakernel_f64accum_body(
    float* __restrict__ conv_state,
    float* __restrict__ h_state,
    const float* __restrict__ qkv_buf,
    const float* __restrict__ alpha_raw,
    const float* __restrict__ beta_raw,
    const float* __restrict__ conv_weight,
    const float* __restrict__ dt_bias,
    const float* __restrict__ ssm_a,
    float* __restrict__ output,
    unsigned int num_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    unsigned int qkv_dim,
    unsigned int qk_dim,
    unsigned int value_dim,
    unsigned int kernel_size,
    unsigned int state_pos)
{
    extern __shared__ float shmem[];

    unsigned int h = blockIdx.x;
    if (h >= num_heads) return;

    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;
    unsigned int buf_slots = kernel_size - 1;
    unsigned int kv_head = h % num_kv_heads;

    // Shared memory layout (identical to F32 megakernel):
    //   warp_scratch[32]   -- reduction temporaries (reinterpreted as 16 doubles)
    //   q_shmem[head_dim]  -- Q for this head's kv_head (after conv+silu+norm)
    //   k_shmem[head_dim]  -- K for this head's kv_head (after conv+silu+norm)
    float* warp_scratch = shmem;
    double* warp_scratch_d = (double*)shmem;  // first 32 floats = 16 doubles
    float* q_shmem = shmem + 32;
    float* k_shmem = shmem + 32 + head_dim;

    // ====================================================================
    // Phase 1: Conv1D + SiLU (F32, matches prefill ssm_conv1d_silu_prefill).
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
    // Phase 2: Gates (F32, matches prefill gdn_compute_gates_batched).
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
    __syncthreads();  // protect warp_scratch reuse as double scratch below

    // ====================================================================
    // Phase 3: L2-normalize Q and K per head — F64 reduction
    // (mirrors l2_normalize_qk_strided_f64accum: double ss, sqrt, 1/norm,
    //  eps = 1e-12, F32 write-back).
    // ====================================================================
    {
        double ss = 0.0;
        for (unsigned int i = tid; i < head_dim; i += block_size) {
            double v = (double)q_shmem[i];
            ss += v * v;
        }
        double total = block_reduce_sum_mega_f64(ss, warp_scratch_d, tid, block_size);
        double norm = sqrt(total);
        double inv = (norm > 1e-12) ? (1.0 / norm) : (1.0 / 1e-12);
        for (unsigned int i = tid; i < head_dim; i += block_size) {
            q_shmem[i] = (float)((double)q_shmem[i] * inv);
        }
    }
    __syncthreads();
    {
        double ss = 0.0;
        for (unsigned int i = tid; i < head_dim; i += block_size) {
            double v = (double)k_shmem[i];
            ss += v * v;
        }
        double total = block_reduce_sum_mega_f64(ss, warp_scratch_d, tid, block_size);
        double norm = sqrt(total);
        double inv = (norm > 1e-12) ? (1.0 / norm) : (1.0 / 1e-12);
        for (unsigned int i = tid; i < head_dim; i += block_size) {
            k_shmem[i] = (float)((double)k_shmem[i] * inv);
        }
    }
    __syncthreads();

    // ====================================================================
    // Phase 4: Delta-rule state update + output query — F64 recurrence
    // (mirrors gdn_prefill_fused_v3_f64accum: double decay/retrieval/v_delta/
    //  state, q_scale = 1/sqrt((double)head_dim), F32 state write-back).
    //
    // One thread per val_dim element (scalar loop form, identical to the
    // graph F32 variant; mathematically equal to the float4 eager form).
    // ====================================================================
    if (tid < head_dim) {
        unsigned int vj = tid;
        double q_scale = 1.0 / sqrt((double)head_dim);
        double v_val = (double)output[h * head_dim + vj];  // V from phase 1 temp

        float* h_row = h_state + (unsigned long long)h * head_dim * head_dim
                                + (unsigned long long)vj * head_dim;

        // Decay + retrieve from decayed state (F64).
        double a = (double)alpha_val;
        double retrieval = 0.0;
        for (unsigned int ki = 0; ki < head_dim; ki++) {
            double h_decayed = a * (double)h_row[ki];
            h_row[ki] = (float)h_decayed;
            retrieval += h_decayed * (double)k_shmem[ki];
        }

        // Delta update + output query (F64).
        double v_delta = (double)beta_val * (v_val - retrieval);
        double my_out = 0.0;
        for (unsigned int ki = 0; ki < head_dim; ki++) {
            double h_updated = (double)h_row[ki] + (double)k_shmem[ki] * v_delta;
            h_row[ki] = (float)h_updated;
            my_out += h_updated * (double)q_shmem[ki];
        }

        output[h * head_dim + vj] = (float)(my_out * q_scale);
    }
}

// Eager F64 megakernel: scalar state_pos (byte-identical-OFF twin of
// gdn_decode_megakernel). Same signature/argument order as the F32 eager kernel.
extern "C" __global__ void gdn_decode_megakernel_f64accum(
    float* __restrict__ conv_state,
    float* __restrict__ h_state,
    const float* __restrict__ qkv_buf,
    const float* __restrict__ alpha_raw,
    const float* __restrict__ beta_raw,
    const float* __restrict__ conv_weight,
    const float* __restrict__ dt_bias,
    const float* __restrict__ ssm_a,
    float* __restrict__ output,
    unsigned int num_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    unsigned int qkv_dim,
    unsigned int qk_dim,
    unsigned int value_dim,
    unsigned int kernel_size,
    unsigned int state_pos)
{
    gdn_decode_megakernel_f64accum_body(
        conv_state, h_state, qkv_buf, alpha_raw, beta_raw,
        conv_weight, dt_bias, ssm_a, output,
        num_heads, num_kv_heads, head_dim, qkv_dim, qk_dim,
        value_dim, kernel_size, state_pos);
}

// Graph F64 megakernel: device-pointer state_pos (byte-identical-OFF twin of
// gdn_decode_megakernel_graph). Same signature/argument order as the F32 graph
// kernel.
extern "C" __global__ void gdn_decode_megakernel_graph_f64accum(
    float* __restrict__ conv_state,
    float* __restrict__ h_state,
    const float* __restrict__ qkv_buf,
    const float* __restrict__ alpha_raw,
    const float* __restrict__ beta_raw,
    const float* __restrict__ conv_weight,
    const float* __restrict__ dt_bias,
    const float* __restrict__ ssm_a,
    float* __restrict__ output,
    unsigned int num_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    unsigned int qkv_dim,
    unsigned int qk_dim,
    unsigned int value_dim,
    unsigned int kernel_size,
    const unsigned int* __restrict__ p_state_pos)
{
    gdn_decode_megakernel_f64accum_body(
        conv_state, h_state, qkv_buf, alpha_raw, beta_raw,
        conv_weight, dt_bias, ssm_a, output,
        num_heads, num_kv_heads, head_dim, qkv_dim, qk_dim,
        value_dim, kernel_size, *p_state_pos);
}
