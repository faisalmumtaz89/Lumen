// GDN F64-internal-accumulator variants.
//
// Goal: Make GDN reductions robust to F32-non-associativity / reduction-order
// differences across implementations by accumulating in F64 in registers.
// Storage stays F32. Convert to F64 on load, accumulate / reduce in F64,
// convert back to F32 on output.
//
// Per bisection, the L0 drift in `linear_attn_out` (3.93%) is
// cumulative F32 ULP noise from different reduction orders. No single
// component dominates; sub-1% sub-component noise compounds via the 4096-K
// `ssm_out` GEMM. F64 internal accumulators eliminate the non-associativity
// at source.
//
// Env-gated default OFF via `LUMEN_CUDA_GDN_F64_ACCUM=1` at backend_impl.rs.
// When gate is off, original `gdn_prefill_fused_v3` / `gdn_prefill_norm_gate`
// / `l2_normalize_qk_strided` / `gdn_phase4_register_resident[_coal]` run unchanged.
//
// Kernels here are ADD-only: they do not replace the F32 originals.
//
// NVRTC-compatible: no system includes, extern "C" linkage.

// ============================================================================
// F64-internal warp reduce.
//
// Identical ordering pattern to `warp_reduce_sum` (butterfly XOR) but operates
// in double precision. Since F64 addition is closer to associative for the
// magnitudes encountered here (no catastrophic cancellation), this kills the
// F32-ULP non-associativity that drove's per-element drift.
// ============================================================================
__device__ __forceinline__ double warp_reduce_sum_f64(double val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

// ============================================================================
// l2_normalize_qk_strided_f64accum
//
// Same algorithm as `l2_normalize_qk_strided` (gdn.cu) but the sum-of-squares
// accumulation, warp+cross-warp reduction, sqrt, and 1/norm are computed in
// F64. The final scale multiplies F32 head values back into F32 storage.
// ============================================================================
extern "C" __global__ void l2_normalize_qk_strided_f64accum(
    float* __restrict__ data,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    unsigned int T,
    unsigned int stride,
    unsigned int q_offset,
    unsigned int k_offset)
{
    extern __shared__ double sharedD[];

    unsigned int block_id = blockIdx.x;
    if (block_id >= num_kv_heads * T) return;

    unsigned int t = block_id / num_kv_heads;
    unsigned int kv_head = block_id % num_kv_heads;

    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;
    unsigned int warp_id = tid >> 5;
    unsigned int lane_id = tid & 31u;
    unsigned int num_warps = (block_size + 31) >> 5;
    double eps = 1e-12;

    // Normalize Q head
    {
        float* head = data + t * stride + q_offset + kv_head * head_dim;
        double ss = 0.0;
        for (unsigned int i = tid; i < head_dim; i += block_size) {
            double v = (double)head[i];
            ss += v * v;
        }
        ss = warp_reduce_sum_f64(ss);
        if (lane_id == 0) sharedD[warp_id] = ss;
        __syncthreads();
        double total_ss = 0.0;
        if (warp_id == 0) {
            total_ss = (lane_id < num_warps) ? sharedD[lane_id] : 0.0;
            total_ss = warp_reduce_sum_f64(total_ss);
        }
        if (tid == 0) sharedD[0] = total_ss;
        __syncthreads();
        total_ss = sharedD[0];
        double norm = sqrt(total_ss);
        double scale = (norm > eps) ? (1.0 / norm) : (1.0 / eps);
        for (unsigned int i = tid; i < head_dim; i += block_size) {
            head[i] = (float)((double)head[i] * scale);
        }
    }
    __syncthreads();

    // Normalize K head
    {
        float* head = data + t * stride + k_offset + kv_head * head_dim;
        double ss = 0.0;
        for (unsigned int i = tid; i < head_dim; i += block_size) {
            double v = (double)head[i];
            ss += v * v;
        }
        ss = warp_reduce_sum_f64(ss);
        if (lane_id == 0) sharedD[warp_id] = ss;
        __syncthreads();
        double total_ss = 0.0;
        if (warp_id == 0) {
            total_ss = (lane_id < num_warps) ? sharedD[lane_id] : 0.0;
            total_ss = warp_reduce_sum_f64(total_ss);
        }
        if (tid == 0) sharedD[0] = total_ss;
        __syncthreads();
        total_ss = sharedD[0];
        double norm = sqrt(total_ss);
        double scale = (norm > eps) ? (1.0 / norm) : (1.0 / eps);
        for (unsigned int i = tid; i < head_dim; i += block_size) {
            head[i] = (float)((double)head[i] * scale);
        }
    }
}

// ============================================================================
// gdn_prefill_fused_v3_f64accum
//
// Same algorithm as `gdn_prefill_fused_v3` (gdn.cu) but with the delta-rule
// state, all `retrieval`/`my_out` reductions, and `v_delta` arithmetic kept
// in F64 in registers. Loads from `conv_out_all`/`alpha_all`/`beta_all` are
// promoted to F64; the final `raw_out[]` write and `h_state[]` write back
// converts back to F32 for storage compatibility.
//
// Performance cost: F64 ops on A100 run at 1/2 throughput of F32. This kernel
// is reduction-and-arithmetic-bound; expected ~2x slowdown on the kernel
// itself. Whether end-to-end Q8/Q4 MoE decode breaches a perf gate is an
// empirical question.
// ============================================================================
extern "C" __global__ void gdn_prefill_fused_v3_f64accum(
    float* __restrict__ h_state,
    const float* __restrict__ conv_out_all,
    const float* __restrict__ alpha_all,
    const float* __restrict__ beta_all,
    float* __restrict__ raw_out,
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

    unsigned int lane = threadIdx.x;
    unsigned int kv_head = h % n_kv_heads;
    double q_scale = 1.0 / sqrt((double)key_dim);

    unsigned int k_base = lane * 4;

    // Load 4 state elements as F64 in registers.
    float* h_row = h_state + h * val_dim * key_dim + vj * key_dim;
    double s0 = (double)h_row[k_base + 0];
    double s1 = (double)h_row[k_base + 1];
    double s2 = (double)h_row[k_base + 2];
    double s3 = (double)h_row[k_base + 3];

    unsigned int q_head_off = kv_head * key_dim + k_base;
    unsigned int k_head_off = qk_dim + kv_head * key_dim + k_base;
    unsigned int v_head_off = qk_dim + qk_dim + h * val_dim + vj;
    unsigned int out_stride = n_heads * val_dim;
    unsigned int out_base = h * val_dim + vj;

    // Tail-only sequential loop (no 4x unroll) — keeps the F64 variant simple
    // and avoids 4x state of registers that would crush occupancy. The 4-token
    // unroll in the F32 variant is a perf optimization for HBM latency hiding;
    // for correctness validation we accept the per-token loop here.
    for (unsigned int t = 0; t < T; t++) {
        double a = (double)alpha_all[t * n_heads + h];
        double b = (double)beta_all[t * n_heads + h];

        const float* conv_t = conv_out_all + t * qkv_dim;

        double qn0 = (double)conv_t[q_head_off];
        double qn1 = (double)conv_t[q_head_off + 1];
        double qn2 = (double)conv_t[q_head_off + 2];
        double qn3 = (double)conv_t[q_head_off + 3];
        double kn0 = (double)conv_t[k_head_off];
        double kn1 = (double)conv_t[k_head_off + 1];
        double kn2 = (double)conv_t[k_head_off + 2];
        double kn3 = (double)conv_t[k_head_off + 3];
        double v_val = (double)conv_t[v_head_off];

        // s_decayed = a * s
        double d0 = a * s0;
        double d1 = a * s1;
        double d2 = a * s2;
        double d3 = a * s3;
        // retrieval = warp_sum(d . k)
        double retrieval = warp_reduce_sum_f64(d0*kn0 + d1*kn1 + d2*kn2 + d3*kn3);
        double v_delta = b * (v_val - retrieval);
        // s_new = s_decayed + k * v_delta
        s0 = d0 + kn0 * v_delta;
        s1 = d1 + kn1 * v_delta;
        s2 = d2 + kn2 * v_delta;
        s3 = d3 + kn3 * v_delta;
        // my_out = warp_sum(s . q) * q_scale
        double my_out = warp_reduce_sum_f64(s0*qn0 + s1*qn1 + s2*qn2 + s3*qn3) * q_scale;
        if (lane == 0) raw_out[t * out_stride + out_base] = (float)my_out;
    }

    // Write F64 state back to F32 storage.
    h_row[k_base + 0] = (float)s0;
    h_row[k_base + 1] = (float)s1;
    h_row[k_base + 2] = (float)s2;
    h_row[k_base + 3] = (float)s3;
}

// ============================================================================
// gdn_prefill_norm_gate_f64accum
//
// F64-accumulator variant of `gdn_prefill_norm_gate`. The RMSNorm
// sum-of-squares + cross-warp reduction + scale computation are in F64.
// The SiLU sigmoid and output write are F32 (the multiplication by `normed`
// is in F32 — F64 isn't needed there; the perf gain is also marginal).
// ============================================================================
extern "C" __global__ void gdn_prefill_norm_gate_f64accum(
    const float* __restrict__ raw_out,
    const float* __restrict__ gate_all,
    const float* __restrict__ norm_scale,
    float* __restrict__ ssm_out,
    unsigned int num_heads,
    unsigned int val_dim,
    float eps,
    unsigned int scale_n_heads,
    unsigned int T)
{
    extern __shared__ double sharedD2[];

    unsigned int h = blockIdx.x;
    unsigned int t = blockIdx.y;
    if (h >= num_heads || t >= T) return;

    unsigned int vj = threadIdx.x;
    unsigned int block_size = blockDim.x;
    unsigned int warp_id = vj >> 5;
    unsigned int lane_id = vj & 31u;
    unsigned int num_warps = (block_size + 31) >> 5;

    unsigned int idx = t * num_heads * val_dim + h * val_dim + vj;

    float val_f = (vj < val_dim) ? raw_out[idx] : 0.0f;
    double val = (double)val_f;

    // RMSNorm: F64 SS reduction.
    double ss = val * val;
    ss = warp_reduce_sum_f64(ss);

    if (lane_id == 0) sharedD2[warp_id] = ss;
    __syncthreads();

    double total_ss = 0.0;
    if (warp_id == 0) {
        total_ss = (lane_id < num_warps) ? sharedD2[lane_id] : 0.0;
        total_ss = warp_reduce_sum_f64(total_ss);
    }
    if (vj == 0) sharedD2[0] = total_ss;
    __syncthreads();
    total_ss = sharedD2[0];

    if (vj >= val_dim) return;

    // RMS in F64, then cast to F32 for final muls.
    double rms_d = sqrt(total_ss / (double)val_dim + (double)eps);
    float inv_rms = (float)(1.0 / rms_d);

    unsigned int scale_h = (scale_n_heads == 1) ? 0 : h;
    float normed = val_f * inv_rms * norm_scale[scale_h * val_dim + vj];

    unsigned int gate_idx = t * num_heads * val_dim + h * val_dim + vj;
    float g = gate_all[gate_idx];
    float silu_g = g / (1.0f + expf(-g));

    ssm_out[gate_idx] = silu_g * normed;
}

// ============================================================================
// gdn_phase4_register_resident_f64accum
//
// F64-accumulator variant of `gdn_phase4_register_resident` (gdn_register_resident.cu) for
// the decode path. Keeps the lane-ownership pattern (contiguous 4-row block per
// lane) and shfl-xor warp reduce, but reductions and state in F64.
//
// This is the kernel used when `LUMEN_CUDA_GDN_REGISTER_RESIDENT=1` (the canonical
// MoE Q8 config) does decode. F64 here matters for both the per-decode-step
// expert-ID determinism and the multi-step rep-coherence.
// ============================================================================
extern "C" __global__ void gdn_phase4_register_resident_f64accum(
    float* __restrict__ h_state,
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ v,
    const float* __restrict__ alpha,
    const float* __restrict__ beta,
    float* __restrict__ output,
    unsigned int num_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim)
{
    const unsigned int S_v = 128;
    const unsigned int warp_size = 32;
    const unsigned int num_warps = 4;
    const unsigned int rows_per_lane = S_v / warp_size;

    const unsigned int h    = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    const unsigned int warp = threadIdx.y;
    const unsigned int vi = blockIdx.z * num_warps + warp;
    if (h >= num_heads || vi >= head_dim) return;

    const unsigned int kv_head = h % num_kv_heads;
    const double a_val  = (double)alpha[h];
    const double b_val  = (double)beta[h];
    const double v_val  = (double)v[h * head_dim + vi];
    const double q_scale = 1.0 / sqrt((double)head_dim);

    const float* q_head = q_norm + kv_head * head_dim;
    const float* k_head = k_norm + kv_head * head_dim;

    float* h_row = h_state + (unsigned long long)h * head_dim * head_dim
                            + (unsigned long long)vi * head_dim;

    const unsigned int k_base = lane * rows_per_lane;
    double s_shard[4];
    #pragma unroll
    for (unsigned int r = 0; r < rows_per_lane; r++) {
        s_shard[r] = (double)h_row[k_base + r];
    }
    double k_reg[4];
    double q_reg[4];
    #pragma unroll
    for (unsigned int r = 0; r < rows_per_lane; r++) {
        k_reg[r] = (double)k_head[k_base + r];
        q_reg[r] = (double)q_head[k_base + r];
    }

    double kv_shard = 0.0;
    #pragma unroll
    for (unsigned int r = 0; r < rows_per_lane; r++) {
        kv_shard += s_shard[r] * k_reg[r];
    }
    double kv_col = warp_reduce_sum_f64(kv_shard);

    double delta_col = (v_val - a_val * kv_col) * b_val;

    double attn_partial = 0.0;
    #pragma unroll
    for (unsigned int r = 0; r < rows_per_lane; r++) {
        s_shard[r] = a_val * s_shard[r] + k_reg[r] * delta_col;
        attn_partial += s_shard[r] * q_reg[r];
    }
    double attn_col = warp_reduce_sum_f64(attn_partial);

    if (lane == 0) {
        output[h * head_dim + vi] = (float)(attn_col * q_scale);
    }

    // Write F64 state back to F32 storage.
    #pragma unroll
    for (unsigned int r = 0; r < rows_per_lane; r++) {
        h_row[k_base + r] = (float)s_shard[r];
    }
}

// ============================================================================
// gdn_rmsnorm_silu_gate_f64accum
//
// Used by the decode-side path that runs after `gdn_phase4_register_resident[_f64accum]`.
// F64 reduction for the RMSNorm sum-of-squares; rest of pipeline F32.
// ============================================================================
extern "C" __global__ void gdn_rmsnorm_silu_gate_f64accum(
    const float* __restrict__ raw_output,
    const float* __restrict__ ssm_norm,
    const float* __restrict__ gate,
    float* __restrict__ out,
    float eps,
    unsigned int dim)
{
    extern __shared__ double sharedD3[];

    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;
    unsigned int warp_id = tid >> 5;
    unsigned int lane_id = tid & 31u;
    unsigned int num_warps = block_size >> 5;

    double sum_sq = 0.0;
    for (unsigned int i = tid; i < dim; i += block_size) {
        double val = (double)raw_output[i];
        sum_sq += val * val;
    }
    sum_sq = warp_reduce_sum_f64(sum_sq);
    if (lane_id == 0) sharedD3[warp_id] = sum_sq;
    __syncthreads();

    double total = 0.0;
    if (warp_id == 0) {
        total = (lane_id < num_warps) ? sharedD3[lane_id] : 0.0;
        total = warp_reduce_sum_f64(total);
    }
    if (tid == 0) sharedD3[0] = 1.0 / sqrt(total / (double)dim + (double)eps);
    __syncthreads();
    float rms = (float)sharedD3[0];

    for (unsigned int i = tid; i < dim; i += block_size) {
        float normed = raw_output[i] * rms * ssm_norm[i];
        float g = gate[i];
        float silu_g = g / (1.0f + expf(-g));
        out[i] = silu_g * normed;
    }
}
