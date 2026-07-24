// GDN decode register-resident state-update kernel pair. Hand-tuned for
// Qwen3.5-9B's GDN block (S_v=128, qkv_dim=8192, conv_kernel_size=4).
//
// Background
// ----------
// A reference implementation of the delta-rule recurrence (Lumen's Phase 4)
// exists as a chain of 5 separate kernels (`ssm_conv` + `silu` + 2x `l2_norm`
// + `gated_delta_net`). Lumen's existing `gdn_decode_megakernel` already
// fuses all five of these phases into a single launch -- a structural
// ADVANTAGE for dispatch overhead. The Phase 4 implementation, however, is
// materially different and is the locus of the per-token wall-clock gap.
//
// Differences in Phase 4 (delta-rule state update):
//
// Lumen (`gdn_decode_megakernel`):
// - Grid: (num_heads) = 32 blocks
// - Block: 256 threads, only first 128 (= head_dim) participate in Phase 4
// - State traffic: 2 reads + 2 writes per h_state element
// (Pass 1: decay+retrieve reads h, writes decayed, computes retrieval)
// (Pass 2: update+output reads decayed h, writes updated, computes out)
// - Per-thread h_row pointer iterates ki = 0..128 SERIALLY (low ILP)
//
// Register-resident state-update form:
// - Grid: (H, n_seqs, ceil(S_v / 4)) = (32, 1, 32) = 1024 blocks
// - Block: (warp_size, 4 warps, 1) = 128 threads
// - Each warp owns ONE column of the state
// - Each lane in a warp owns 4 rows (rows_per_lane = S_v / warp_size = 4)
// - h_state lives in REGISTERS as `float s_shard[4]` -- ZERO global R/W
// between decay and update steps (1 read at start + 1 write at end)
// - Inner reductions via __shfl_xor_sync (warp_size = 32 lanes per column)
//
// Predicted memory-traffic savings on Qwen3.5-9B Q8 decode (head_dim=128,
// num_heads=32, 24 GDN layers):
// Lumen current: 24 * 32 * 128 * 128 floats * 16 B (2R + 2W) = 6.0 MB
// Register-resident: 24 * 32 * 128 * 128 floats * 8 B (1R + 1W) = 3.0 MB
// At 1.55 TB/s peak HBM: 3.0 MB savings ~ 2.0 us per token (lower bound).
// Realistic at 70% HBM peak: ~3 us per token reduction.
//
// Caveat: the realized speedup also depends on warp-launch overhead at the
// 1024-block grid vs the 32-block grid. The 32x increase in block
// count amortizes better on A100 (108 SMs * 2 blocks/SM = 216 concurrent
// blocks), but each block is only 4 warps vs the megakernel's 8. The
// microbench resolves whether the register-resident win exceeds the
// increased block count cost.
//
// Design
// ------
// Two NEW kernels (env-gated; do NOT replace existing `gdn_decode_megakernel`):
//
// 1. `gdn_phase123_register_resident` -- Conv1D + SiLU + Gates + L2 Norm
// Grid: (num_heads) = 32 blocks
// Block: 256 threads
// Writes Q_norm, K_norm, V, alpha, beta to device buffers
// (Same as existing Phases 1-3 of `gdn_decode_megakernel`, but stops
// before Phase 4 and materializes the intermediates instead of holding
// them in shared memory.)
//
// 2. `gdn_phase4_register_resident` -- register-resident delta-rule
// Grid: (num_heads, 1, ceil(head_dim/4)) = (32, 1, 32) = 1024 blocks
// Block: (32, 4, 1) = 128 threads
// Each warp owns one column, each lane owns 4 rows in s_shard[4].
//
// Hardcoded for Qwen3.5-9B GDN (head_dim=128, num_heads=32, num_kv_heads=16,
// qkv_dim=8192, conv_kernel_size=4). S_v=128 head_dim specialization.
//
// NVRTC-compatible: no system includes, extern "C" linkage.

// ============================================================================
// Shared utilities
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum_rr(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

__device__ __forceinline__ float block_reduce_sum_rr(
    float val, float* warp_scratch,
    unsigned int tid, unsigned int block_size)
{
    unsigned int warp_id = tid >> 5;
    unsigned int lane_id = tid & 31u;
    unsigned int num_warps = block_size >> 5;

    val = warp_reduce_sum_rr(val);
    if (lane_id == 0) warp_scratch[warp_id] = val;
    __syncthreads();

    float total = 0.0f;
    if (warp_id == 0) {
        total = (lane_id < num_warps) ? warp_scratch[lane_id] : 0.0f;
        total = warp_reduce_sum_rr(total);
    }
    if (tid == 0) warp_scratch[0] = total;
    __syncthreads();
    return warp_scratch[0];
}

__device__ __forceinline__ float conv1d_silu_rr(
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

    for (unsigned int tap = 0; tap < buf_slots; tap++) {
        unsigned int slot = (state_pos + tap) % buf_slots;
        sum += conv_weight[idx * kernel_size + tap] * conv_state[slot * qkv_dim + idx];
    }
    sum += conv_weight[idx * kernel_size + buf_slots] * input_val;

    // Update circular buffer
    conv_state[state_pos * qkv_dim + idx] = input_val;

    // SiLU activation
    return sum / (1.0f + expf(-sum));
}


// ============================================================================
// gdn_phase123_register_resident
//
// Performs the pre-Phase-4 work of the existing megakernel:
// - Conv1D + SiLU on Q, K, V channels (with conv_state update)
// - Gate computation (alpha decay, beta mixing) per head
// - Per-head L2 normalization of Q and K
//
// Writes the post-norm Q, K, V vectors AND the computed alpha/beta gates to
// device buffers. The companion `gdn_phase4_register_resident` kernel consumes these
// to perform the register-resident state update.
//
// Grid: (num_heads) = 32 blocks (matches existing megakernel)
// Block: block_dim threads (>= head_dim, typically 256)
// Shared memory: (32 + 2*head_dim) * sizeof(float)
// Same shmem usage as existing megakernel Phases 1-3.
//
// Outputs:
// q_norm_buf[num_kv_heads * head_dim]: post-conv1d, post-SiLU, post-L2-norm Q
// k_norm_buf[num_kv_heads * head_dim]: post-conv1d, post-SiLU, post-L2-norm K
// v_buf [num_heads * head_dim]: post-conv1d, post-SiLU V (no norm on V)
// alpha_buf[num_heads]: exp(ssm_a * softplus(alpha_raw + dt_bias)) per head
// beta_buf [num_heads]: sigmoid(beta_raw) per head
//
// Race-condition note: when multiple blocks share the same kv_head (GQA), they
// both write the same Q and K values (a benign write-write race -- same output
// for the same input). The conv_state update is also benign in the same sense.
// ============================================================================
extern "C" __global__ void gdn_phase123_register_resident(
    // Mutable persistent state
    float* __restrict__ conv_state,        // [(kernel_size-1) * qkv_dim] R/W

    // Inputs from prior matvec steps
    const float* __restrict__ qkv_buf,     // [qkv_dim]
    const float* __restrict__ alpha_raw,   // [num_heads]
    const float* __restrict__ beta_raw,    // [num_heads]

    // Layer weights
    const float* __restrict__ conv_weight, // [qkv_dim, kernel_size] row-major
    const float* __restrict__ dt_bias,     // [num_heads]
    const float* __restrict__ ssm_a,       // [num_heads] = -exp(A_log)

    // Outputs to device (consumed by gdn_phase4_register_resident)
    float* __restrict__ q_norm_buf,        // [num_kv_heads * head_dim]
    float* __restrict__ k_norm_buf,        // [num_kv_heads * head_dim]
    float* __restrict__ v_buf,             // [num_heads * head_dim]
    float* __restrict__ alpha_buf,         // [num_heads]
    float* __restrict__ beta_buf,          // [num_heads]

    // Dimensions
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

    float* warp_scratch = shmem;
    float* q_shmem = shmem + 32;
    float* k_shmem = shmem + 32 + head_dim;

    unsigned int q_base = kv_head * head_dim;
    unsigned int k_base = qk_dim + kv_head * head_dim;
    unsigned int v_base = 2 * qk_dim + h * head_dim;

    // Phase 1: Conv1D + SiLU on Q, K, V elements + conv_state update.
    for (unsigned int i = tid; i < head_dim; i += block_size) {
        q_shmem[i] = conv1d_silu_rr(
            q_base + i, qkv_buf, conv_weight, conv_state,
            qkv_dim, kernel_size, buf_slots, state_pos);
        k_shmem[i] = conv1d_silu_rr(
            k_base + i, qkv_buf, conv_weight, conv_state,
            qkv_dim, kernel_size, buf_slots, state_pos);
        v_buf[h * head_dim + i] = conv1d_silu_rr(
            v_base + i, qkv_buf, conv_weight, conv_state,
            qkv_dim, kernel_size, buf_slots, state_pos);
    }
    __syncthreads();

    // Phase 2: Compute alpha, beta gates per head (one thread per block).
    if (tid == 0) {
        float sp_input = alpha_raw[h] + dt_bias[h];
        float sp = (sp_input > 20.0f) ? sp_input : logf(1.0f + expf(sp_input));
        alpha_buf[h] = expf(ssm_a[h] * sp);
        beta_buf[h]  = 1.0f / (1.0f + expf(-beta_raw[h]));
    }

    // Phase 3: L2-normalize Q and K per head, write to device buffers.
    // L2-normalize Q
    {
        float ss = 0.0f;
        for (unsigned int i = tid; i < head_dim; i += block_size) {
            float v = q_shmem[i];
            ss += v * v;
        }
        float total = block_reduce_sum_rr(ss, warp_scratch, tid, block_size);
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
        float total = block_reduce_sum_rr(ss, warp_scratch, tid, block_size);
        float norm = sqrtf(total);
        float inv = (norm > 1e-12f) ? (1.0f / norm) : (1.0f / 1e-12f);
        for (unsigned int i = tid; i < head_dim; i += block_size) {
            k_shmem[i] *= inv;
        }
    }
    __syncthreads();

    // Write Q_norm, K_norm to device.
    // GQA: heads are mapped to kv_heads via `kv_head = h % num_kv_heads`, so
    // for num_heads=32, num_kv_heads=16:
    // kv_head=0: h in {0, 16}, kv_head=1: h in {1, 17}, ...
    // kv_head=k: h in {k, k+16}
    // Both h=k and h=k+16 produce the same q_shmem/k_shmem (identical input).
    // We elect ONE block per kv_head to write. The first num_kv_heads blocks
    // (h < num_kv_heads) cover every kv_head exactly once, since for h in
    // [0, num_kv_heads) we have kv_head=h. (An earlier version used the
    // guard `h % (num_heads/num_kv_heads) == 0` which selects {0, 2, 4, ...}
    // and covers ONLY the even-indexed kv_heads -- a correctness bug that
    // left half the q_norm_buf/k_norm_buf entries unwritten.)
    if (h < num_kv_heads) {
        unsigned int kv_base = kv_head * head_dim;
        for (unsigned int i = tid; i < head_dim; i += block_size) {
            q_norm_buf[kv_base + i] = q_shmem[i];
            k_norm_buf[kv_base + i] = k_shmem[i];
        }
    }
}




// ============================================================================
// gdn_phase4_register_resident
//
// Register-resident delta-rule state update for GDN decode. Hardcoded
// S_v=128 (Qwen3.5 head_dim=128) specialization for n_tokens=1 (decode).
//
// Algorithm (per (head, column)):
// 1. Load h_state[h, :, col] into register shard s_shard[4]
// (4 rows per lane, 32 lanes per warp = 128 = head_dim rows)
// 2. Compute kv_col = warp_reduce(sum_i s_shard[i] * k[i])
// 3. delta_col = (v[col] - alpha * kv_col) * beta
// 4. Update shard: s_shard[i] = alpha * s_shard[i] + k[i] * delta_col
// 5. Compute attn_col = warp_reduce(sum_i s_shard[i] * q[i]) * scale
// 6. Write attn_col to output[col] (only lane 0)
// 7. Write shard back to h_state at end of kernel
//
// Hardcoded for S_v = 128 (Qwen3.5 head_dim = 128, num_heads = 32).
//
// Grid: (num_heads, 1, ceil(head_dim / num_warps)) = (32, 1, 32) = 1024 blocks
// Block: (warp_size=32, num_warps=4, 1) = 128 threads
//
// h_state layout: [num_heads, head_dim (val=row), head_dim (key=col)] f32
// h_state[h, vi, ki] = h_state[h*head_dim*head_dim + vi*head_dim + ki]
// An alternative store layout is transposed: M[col][row] is contiguous in row.
//
// We translate between the two conventions: Lumen's h_state stores row vi
// contiguously in ki. The transposed form stores col ki contiguously in vi
// (rows). Since the delta-rule is symmetric in s[i][col] = alpha*s[i][col] +
// k[i]*delta[col], we can choose either layout. We KEEP Lumen's existing
// layout (vi major, ki minor) so the h_state buffer stays unchanged. Each
// warp in this kernel owns one (h, col=vi) and the lanes cover ki rows.
//
// Concretely: warp owns column vi, lane l owns rows ki = l*4 .. l*4+3
// (rows_per_lane = head_dim / warp_size = 128 / 32 = 4)
//
// s_shard[r=0..3] = h_state[h, vi, l*4 + r]
//
// At decode (n_tokens=1) the inner k-loop has one iteration.
// ============================================================================
extern "C" __global__ void gdn_phase4_register_resident(
    // Mutable persistent state
    float* __restrict__ h_state,            // [num_heads, head_dim, head_dim] R/W

    // Inputs from gdn_phase123_register_resident
    const float* __restrict__ q_norm,       // [num_kv_heads * head_dim]
    const float* __restrict__ k_norm,       // [num_kv_heads * head_dim]
    const float* __restrict__ v,            // [num_heads * head_dim]
    const float* __restrict__ alpha,        // [num_heads]
    const float* __restrict__ beta,         // [num_heads]

    // Output
    float* __restrict__ output,             // [num_heads * head_dim]

    // Dimensions
    unsigned int num_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim)
{
    // Hardcode S_v = head_dim = 128 for Qwen3.5-9B (head_dim=128 specialization)
    const unsigned int S_v = 128;
    const unsigned int warp_size = 32;
    const unsigned int num_warps = 4;
    const unsigned int rows_per_lane = S_v / warp_size; // = 4

    const unsigned int h    = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    const unsigned int warp = threadIdx.y;
    // Block z dim sweeps groups of `num_warps` columns:
    // col = blockIdx.z * num_warps + warp
    const unsigned int vi = blockIdx.z * num_warps + warp;
    if (h >= num_heads || vi >= head_dim) return;

    const unsigned int kv_head = h % num_kv_heads;
    const float a_val  = alpha[h];
    const float b_val  = beta[h];
    const float v_val  = v[h * head_dim + vi];
    const float q_scale = rsqrtf((float)head_dim);

    // Pointers to Q, K for this head's kv_head
    const float* q_head = q_norm + kv_head * head_dim;
    const float* k_head = k_norm + kv_head * head_dim;

    // Pointer to this (h, vi)'s state row in Lumen layout
    float* h_row = h_state + (unsigned long long)h * head_dim * head_dim
                            + (unsigned long long)vi * head_dim;

    // Lane l owns rows ki = l*4 + r for r in 0..4. Load into registers.
    const unsigned int k_base = lane * rows_per_lane;
    float s_shard[4];
    #pragma unroll
    for (unsigned int r = 0; r < rows_per_lane; r++) {
        s_shard[r] = h_row[k_base + r];
    }

    // Cache k, q for this lane in registers (4 elements each)
    float k_reg[4];
    float q_reg[4];
    #pragma unroll
    for (unsigned int r = 0; r < rows_per_lane; r++) {
        k_reg[r] = k_head[k_base + r];
        q_reg[r] = q_head[k_base + r];
    }

    // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
    // In Lumen layout: S[i=row][col=vi] is h_state[h, vi, i] -- contiguous in i.
    // s_shard[r] = S[k_base + r][vi] -- exactly the rows this lane owns.
    float kv_shard = 0.0f;
    #pragma unroll
    for (unsigned int r = 0; r < rows_per_lane; r++) {
        kv_shard += s_shard[r] * k_reg[r];
    }
    float kv_col = warp_reduce_sum_rr(kv_shard);

    // delta[col] = (v[col] - alpha * kv[col]) * beta
    // (An alternative store form is g = log(alpha) exponentiated inside the
    // kernel; Lumen pre-computes alpha = exp(...). Both give the same delta.)
    float delta_col = (v_val - a_val * kv_col) * b_val;

    // fused update + retrieve:
    // S[i][col] = alpha * S[i][col] + k[i] * delta[col]
    // attn[col] = sum_i S[i][col] * q[i]
    float attn_partial = 0.0f;
    #pragma unroll
    for (unsigned int r = 0; r < rows_per_lane; r++) {
        s_shard[r] = a_val * s_shard[r] + k_reg[r] * delta_col;
        attn_partial += s_shard[r] * q_reg[r];
    }
    float attn_col = warp_reduce_sum_rr(attn_partial);

    if (lane == 0) {
        output[h * head_dim + vi] = attn_col * q_scale;
    }

    // Write shard back to h_state.
    #pragma unroll
    for (unsigned int r = 0; r < rows_per_lane; r++) {
        h_row[k_base + r] = s_shard[r];
    }
}


