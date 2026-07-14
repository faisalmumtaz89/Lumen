// Grouped (expert-sorted) MoE prefill FFN kernels.
//
// Replaces the per-token Rust loop in `backend_impl.rs::prefill_moe_ffn_layer`
// (which calls the single-token decode MoE dispatch `batch` times) with a
// batched, expert-grouped dispatch. Mirrors llama.cpp `mul_mat_id` and the
// Metal-validated `moe_prefill_grouped` design (P0-B, byte-identical, +308%).
//
// Routing/gather is host-prepared (one DtoH sync per layer, sort, HtoD of the
// gather/scatter tables) so the kernels themselves are pure GEMM + activation.
//
// Compact-column layout (built on host from `expert_ids[batch, top_k]`):
//   total = batch * top_k columns, SORTED BY ASSIGNED EXPERT.
//   col_expert[c]  : the expert id that owns compact column c.
//   col_src_tok[c] : the source token index whose `normed[tok, :]` feeds col c.
//   col_dst[c]     : destination slot = src_tok * top_k + slot, used to scatter
//                    the column's down-output and pick the matching router weight.
// Because the column already carries (expert, src_tok), the GEMM kernels do NOT
// need `expert_bounds` — each CTA reads `col_expert[c]` to find its weight base
// and `col_src_tok[c]` to find its activation row. Weight HBM reads are still
// amortized: columns of the same expert are contiguous, so an expert's weight
// rows stream into L2 once and are reused across its sibling columns.
//
// MATH IS BIT-IDENTICAL to the per-token oracle (`moe_batched_gate_up_swiglu_
// q8_0_v2` / `moe_batched_down_v3`): same F32 per-block-scale accumulation
// (gate_acc = Σ_b gscale[b] * Σ_e gquant[b,e]*x[e]), same SwiGLU SiLU form, same
// warp+cross-warp reduction tree (NR rows per CTA, BLOCK_DIM=256, 8 warps). The
// ONLY change vs the oracle is the outer iteration shape (compact columns vs
// per-token loop). For a fixed (token, expert) pair the produced value is the
// SAME float, so the scatter-accumulated result matches the per-token loop
// modulo nothing (identical op order).
//
// NVRTC-compatible: inline PTX for f16->f32, no system includes, extern "C".

#define MG_BLOCK_DIM       256   // 8 warps per CTA (matches moe_batched v2/v3)
#define MG_Q8_0_BLOCK_SIZE 32
#define MG_Q8_0_BLOCK_BYTES 34   // 2-byte f16 scale + 32 int8 quants
#define MG_NR_GU           4     // rows per CTA for gate_up (matches v3)
#define MG_NR_DOWN         4     // rows per CTA for down   (matches v3)

__device__ __forceinline__ float mg_f16_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

__device__ __forceinline__ float mg_load_scale(const unsigned char* block_ptr) {
    unsigned short f16_bits = *reinterpret_cast<const unsigned short*>(block_ptr);
    return mg_f16_to_f32(f16_bits);
}

__device__ __forceinline__ float mg_warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

// SwiGLU: silu(g) * u = (g * sigmoid(g)) * u. IDENTICAL to mb_v2_swiglu.
__device__ __forceinline__ float mg_swiglu(float g, float u) {
    float silu_g = g / (1.0f + expf(-g));
    return silu_g * u;
}

// ===========================================================================
// Stage 0: batched router logits over ALL tokens.
//
// logits[tok, e] = Σ_j router_weight[e, j] * normed[tok, j].
// Identical per-(tok,e) math to `moe_router_logits_v2` (one expert per CTA,
// BLOCK_DIM threads cooperative, warp + cross-warp reduction) but with a token
// dimension on blockIdx.y. The downstream top-K finalize is the SAME
// `topk_moe_fused_*_no_bias` kernel at n_rows=batch — so router selection is
// bit-identical to the per-token path by construction.
//
// Grid:  (num_experts, batch, 1)
// Block: (MG_BLOCK_DIM, 1, 1)
// ===========================================================================
extern "C" __global__ void moe_router_logits_batched(
    const float* __restrict__ normed,         // [batch, hidden_dim]
    const float* __restrict__ router_weight,  // [num_experts * hidden_dim]
    float* __restrict__ router_logits,        // [batch, num_experts]
    unsigned int hidden_dim,
    unsigned int num_experts,
    unsigned int batch)
{
    const unsigned int e = blockIdx.x;
    const unsigned int tok = blockIdx.y;
    if (e >= num_experts || tok >= batch) return;

    __shared__ float warp_partial[MG_BLOCK_DIM / 32];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = MG_BLOCK_DIM / 32;

    const float* w_e = router_weight + (size_t)e * (size_t)hidden_dim;
    const float* x_row = normed + (size_t)tok * (size_t)hidden_dim;

    float partial = 0.0f;
    for (unsigned int j = tid; j < hidden_dim; j += MG_BLOCK_DIM) {
        partial += w_e[j] * x_row[j];
    }
    partial = mg_warp_reduce_sum(partial);
    if (lane == 0) warp_partial[warp_id] = partial;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane < num_warps) ? warp_partial[lane] : 0.0f;
        v = mg_warp_reduce_sum(v);
        if (lane == 0) {
            router_logits[(size_t)tok * (size_t)num_experts + e] = v;
        }
    }
}

// ===========================================================================
// Stage 1: grouped gate+up+SwiGLU.
//
// Grid:  (ceil(inter_dim / MG_NR_GU), total_cols, 1)
// Block: (MG_BLOCK_DIM, 1, 1)
// Shmem: hidden_dim * 4 bytes (normed-x row cache for the column's source token)
//
// Each (blockIdx.y = compact col c, blockIdx.x = NR-row-tile) computes MG_NR_GU
// rows of the SwiGLU output for column c. It looks up:
//   expert  = col_expert[c]      -> gate_off/up_off via gate_up_offsets[expert*2]
//   src_tok = col_src_tok[c]     -> activation row normed[src_tok, :]
// and writes to swiglu_compact[c * inter_dim + row].
// ===========================================================================
// ===========================================================================
// M-TILED grouped gate+up+SwiGLU. Weight read ONCE per expert, reused
// across MT compact columns (tokens) of that expert.
//
// Grid:  (ceil(inter_dim / MG_NR_GU), ceil(max_cols_per_expert / MG_MT), num_experts)
// Block: (MG_BLOCK_DIM, 1, 1)
//
// blockIdx.z = expert e. The CTA processes compact columns
//   [expert_bounds[e] + blockIdx.y*MG_MT, ... up to expert_bounds[e+1]).
// For each K-block, each thread loads the NR gate+up weight blocks ONCE and
// accumulates against up to MG_MT token columns' x-blocks (read from global;
// x is small + L2-cached, the weight is the HBM bottleneck). Per (col,row) the
// K-accumulation order + warp reduction is IDENTICAL to the per-column kernel,
// so output is BIT-IDENTICAL — only the weight LOAD is shared across columns.
//
// No shmem (MT token rows would exceed 48 KB); x read from global per column.
// ===========================================================================
#define MG_MT 4   // compact columns (tokens) per CTA
extern "C" __global__ void moe_grouped_gate_up_swiglu_q8_0_mtiled(
    const float* __restrict__ normed,                       // [batch, hidden_dim]
    const unsigned char* __restrict__ layer_buf,            // raw weight blob
    const int* __restrict__ col_src_tok,                    // [total_cols]
    const int* __restrict__ expert_bounds,                  // [num_experts+1]
    const unsigned long long* __restrict__ gate_up_offsets, // [num_experts * 2]
    float* __restrict__ swiglu_compact,                     // [total_cols * inter_dim]
    unsigned int hidden_dim,
    unsigned int inter_dim)
{
    const unsigned int e = blockIdx.z;
    const int c_lo = expert_bounds[e] + (int)(blockIdx.y * MG_MT);
    const int c_hi = expert_bounds[e + 1];
    if (c_lo >= c_hi) return; // this column-tile is empty for expert e
    const int mt = (c_hi - c_lo) < (int)MG_MT ? (c_hi - c_lo) : (int)MG_MT;

    const unsigned int r0 = blockIdx.x * MG_NR_GU;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31;
    const unsigned int warp_id = tid >> 5;
    const unsigned int num_warps = MG_BLOCK_DIM / 32;

    const unsigned long long gate_off = gate_up_offsets[(size_t)e * 2 + 0];
    const unsigned long long up_off   = gate_up_offsets[(size_t)e * 2 + 1];
    const unsigned int num_blocks = hidden_dim / MG_Q8_0_BLOCK_SIZE;
    const size_t row_bytes = (size_t)num_blocks * MG_Q8_0_BLOCK_BYTES;

    // Per (row r, col m) accumulators for gate and up.
    float gate_acc[MG_NR_GU][MG_MT];
    float up_acc[MG_NR_GU][MG_MT];
    #pragma unroll
    for (int r = 0; r < MG_NR_GU; r++)
        #pragma unroll
        for (int m = 0; m < MG_MT; m++) { gate_acc[r][m] = 0.0f; up_acc[r][m] = 0.0f; }

    // Source token row pointers for the mt columns.
    const float* x_rows[MG_MT];
    #pragma unroll
    for (int m = 0; m < MG_MT; m++) {
        if (m < mt) {
            int src = col_src_tok[c_lo + m];
            x_rows[m] = normed + (size_t)src * (size_t)hidden_dim;
        } else {
            x_rows[m] = normed; // unused
        }
    }

    for (unsigned int ib = tid; ib < num_blocks; ib += MG_BLOCK_DIM) {
        const unsigned int x_base = ib * MG_Q8_0_BLOCK_SIZE;
        #pragma unroll
        for (int r = 0; r < MG_NR_GU; r++) {
            if (r0 + r >= inter_dim) break;
            // Load gate+up weight blocks for this (row, K-block) ONCE into regs.
            const unsigned char* gp = layer_buf + gate_off
                + (size_t)(r0 + r) * row_bytes + (size_t)ib * MG_Q8_0_BLOCK_BYTES;
            float g_scale = mg_load_scale(gp);
            const signed char* gq = (const signed char*)(gp + 2);
            const unsigned char* upp = layer_buf + up_off
                + (size_t)(r0 + r) * row_bytes + (size_t)ib * MG_Q8_0_BLOCK_BYTES;
            float u_scale = mg_load_scale(upp);
            const signed char* uq = (const signed char*)(upp + 2);
            float gqf[32], uqf[32];
            #pragma unroll
            for (int j = 0; j < 32; j++) { gqf[j] = (float)gq[j]; uqf[j] = (float)uq[j]; }
            // Reuse the weight block across all mt columns (weight read once).
            // STRICT sequential j-accumulation to match the per-column kernel's
            // FP association exactly (g_block += gqf[j]*x[j], j=0..31) ⇒ bit-identical.
            #pragma unroll
            for (int m = 0; m < MG_MT; m++) {
                if (m >= mt) break;
                float xb[32];
                const float4* x4 = (const float4*)(x_rows[m] + x_base);
                #pragma unroll
                for (int k = 0; k < 8; k++) {
                    float4 v = x4[k];
                    xb[k*4+0]=v.x; xb[k*4+1]=v.y; xb[k*4+2]=v.z; xb[k*4+3]=v.w;
                }
                float g_block = 0.0f, u_block = 0.0f;
                #pragma unroll
                for (int j = 0; j < 32; j++) {
                    g_block += gqf[j] * xb[j];
                    u_block += uqf[j] * xb[j];
                }
                gate_acc[r][m] += g_scale * g_block;
                up_acc[r][m]   += u_scale * u_block;
            }
        }
    }

    // Cross-thread reduction per (row, col) via shmem. Reduce one (r,m) at a time
    // using the same warp+cross-warp tree as the per-column kernel ⇒ bit-identical.
    __shared__ float red[MG_BLOCK_DIM / 32];
    #pragma unroll
    for (int r = 0; r < MG_NR_GU; r++) {
        if (r0 + r >= inter_dim) break;
        #pragma unroll
        for (int m = 0; m < MG_MT; m++) {
            if (m >= mt) break;
            float gv = mg_warp_reduce_sum(gate_acc[r][m]);
            if (lane == 0) red[warp_id] = gv;
            __syncthreads();
            float gfin = 0.0f;
            if (warp_id == 0) {
                float v = (lane < num_warps) ? red[lane] : 0.0f;
                gfin = mg_warp_reduce_sum(v);
            }
            __syncthreads();
            float uv = mg_warp_reduce_sum(up_acc[r][m]);
            if (lane == 0) red[warp_id] = uv;
            __syncthreads();
            if (warp_id == 0) {
                float v = (lane < num_warps) ? red[lane] : 0.0f;
                float ufin = mg_warp_reduce_sum(v);
                if (lane == 0) {
                    int c = c_lo + m;
                    float out = mg_swiglu(gfin, ufin);
                    swiglu_compact[(size_t)c * (size_t)inter_dim + (r0 + r)] = out;
                }
            }
            __syncthreads();
        }
    }
}

extern "C" __global__ void moe_grouped_gate_up_swiglu_q8_0(
    const float* __restrict__ normed,                       // [batch, hidden_dim]
    const unsigned char* __restrict__ layer_buf,            // raw weight blob
    const int* __restrict__ col_expert,                     // [total_cols]
    const int* __restrict__ col_src_tok,                    // [total_cols]
    const unsigned long long* __restrict__ gate_up_offsets, // [num_experts * 2]
    float* __restrict__ swiglu_compact,                     // [total_cols * inter_dim]
    unsigned int hidden_dim,
    unsigned int inter_dim,
    unsigned int total_cols)
{
    extern __shared__ float nx_smem[]; // [hidden_dim]

    const unsigned int c = blockIdx.y;
    if (c >= total_cols) return;
    const unsigned int r0 = blockIdx.x * MG_NR_GU;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31;
    const unsigned int warp_id = tid >> 5;
    const unsigned int num_warps = MG_BLOCK_DIM / 32;

    const int expert_id = col_expert[c];
    const int src_tok = col_src_tok[c];
    const unsigned long long gate_off = gate_up_offsets[(size_t)expert_id * 2 + 0];
    const unsigned long long up_off   = gate_up_offsets[(size_t)expert_id * 2 + 1];

    const unsigned int num_blocks = hidden_dim / MG_Q8_0_BLOCK_SIZE;
    const size_t row_bytes = (size_t)num_blocks * MG_Q8_0_BLOCK_BYTES;

    // Cache the source token's normed row in shmem (reused across NR rows).
    const float* x_row = normed + (size_t)src_tok * (size_t)hidden_dim;
    for (unsigned int i = tid; i < hidden_dim; i += MG_BLOCK_DIM) {
        nx_smem[i] = x_row[i];
    }
    __syncthreads();

    float gate_sum[MG_NR_GU];
    float up_sum[MG_NR_GU];
    #pragma unroll
    for (int r = 0; r < MG_NR_GU; r++) { gate_sum[r] = 0.0f; up_sum[r] = 0.0f; }

    for (unsigned int ib = tid; ib < num_blocks; ib += MG_BLOCK_DIM) {
        const unsigned int x_base = ib * MG_Q8_0_BLOCK_SIZE;
        float xv[32];
        const float4* x4 = (const float4*)(nx_smem + x_base);
        #pragma unroll
        for (int kk = 0; kk < 8; kk++) {
            float4 v = x4[kk];
            xv[kk * 4 + 0] = v.x; xv[kk * 4 + 1] = v.y;
            xv[kk * 4 + 2] = v.z; xv[kk * 4 + 3] = v.w;
        }
        #pragma unroll
        for (int r = 0; r < MG_NR_GU; r++) {
            if (r0 + r >= inter_dim) break;
            const unsigned char* gp = layer_buf + gate_off
                + (size_t)(r0 + r) * row_bytes + (size_t)ib * MG_Q8_0_BLOCK_BYTES;
            float g_scale = mg_load_scale(gp);
            const signed char* gq = (const signed char*)(gp + 2);
            const unsigned char* upp = layer_buf + up_off
                + (size_t)(r0 + r) * row_bytes + (size_t)ib * MG_Q8_0_BLOCK_BYTES;
            float u_scale = mg_load_scale(upp);
            const signed char* uq = (const signed char*)(upp + 2);
            float g_block = 0.0f, u_block = 0.0f;
            #pragma unroll
            for (int j = 0; j < 32; j++) {
                g_block += (float)gq[j] * xv[j];
                u_block += (float)uq[j] * xv[j];
            }
            gate_sum[r] += g_scale * g_block;
            up_sum[r]   += u_scale * u_block;
        }
    }

    // Intra-warp reductions, then cross-warp via shmem (reuse nx_smem).
    #pragma unroll
    for (int r = 0; r < MG_NR_GU; r++) gate_sum[r] = mg_warp_reduce_sum(gate_sum[r]);
    __syncthreads();
    float* reduce_smem = nx_smem;
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < MG_NR_GU; r++) reduce_smem[r * num_warps + warp_id] = gate_sum[r];
    }
    __syncthreads();
    float final_gate[MG_NR_GU];
    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < MG_NR_GU; r++) {
            float val = (lane < num_warps) ? reduce_smem[r * num_warps + lane] : 0.0f;
            final_gate[r] = mg_warp_reduce_sum(val);
        }
    }
    __syncthreads();
    #pragma unroll
    for (int r = 0; r < MG_NR_GU; r++) up_sum[r] = mg_warp_reduce_sum(up_sum[r]);
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < MG_NR_GU; r++) reduce_smem[r * num_warps + warp_id] = up_sum[r];
    }
    __syncthreads();
    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < MG_NR_GU; r++) {
            if (r0 + r < inter_dim) {
                float val = (lane < num_warps) ? reduce_smem[r * num_warps + lane] : 0.0f;
                val = mg_warp_reduce_sum(val);
                if (lane == 0) {
                    float out = mg_swiglu(final_gate[r], val);
                    swiglu_compact[(size_t)c * (size_t)inter_dim + (r0 + r)] = out;
                }
            }
        }
    }
}

// ===========================================================================
// Stage 2: grouped down projection (NO accumulate; writes compact down output).
//
// Grid:  (ceil(hidden_dim / MG_NR_DOWN), total_cols, 1)
// Block: (MG_BLOCK_DIM, 1, 1)
// Shmem: inter_dim * 4 bytes (swiglu row cache for the column)
//
// Writes down_compact[c * hidden_dim + row]. Scatter-accumulate is a separate
// kernel (stage 3) so this stays a pure GEMM with the v3 reduction tree.
// ===========================================================================
extern "C" __global__ void moe_grouped_down_q8_0(
    const float* __restrict__ swiglu_compact,         // [total_cols * inter_dim]
    const unsigned char* __restrict__ layer_buf,      // raw weight blob
    const int* __restrict__ col_expert,               // [total_cols]
    const unsigned long long* __restrict__ down_offsets, // [num_experts]
    float* __restrict__ down_compact,                 // [total_cols * hidden_dim]
    unsigned int hidden_dim,
    unsigned int inter_dim,
    unsigned int total_cols)
{
    extern __shared__ float sw_smem[]; // [inter_dim]

    const unsigned int c = blockIdx.y;
    if (c >= total_cols) return;
    const unsigned int r0 = blockIdx.x * MG_NR_DOWN;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31;
    const unsigned int warp_id = tid >> 5;
    const unsigned int num_warps = MG_BLOCK_DIM / 32;

    const int expert_id = col_expert[c];
    const unsigned long long down_off = down_offsets[(size_t)expert_id];

    const unsigned int num_blocks = inter_dim / MG_Q8_0_BLOCK_SIZE;
    const size_t row_bytes = (size_t)num_blocks * MG_Q8_0_BLOCK_BYTES;

    const float* swig_c = swiglu_compact + (size_t)c * (size_t)inter_dim;
    for (unsigned int i = tid; i < inter_dim; i += MG_BLOCK_DIM) {
        sw_smem[i] = swig_c[i];
    }
    __syncthreads();

    float sum_r[MG_NR_DOWN];
    #pragma unroll
    for (int r = 0; r < MG_NR_DOWN; r++) sum_r[r] = 0.0f;

    for (unsigned int ib = tid; ib < num_blocks; ib += MG_BLOCK_DIM) {
        const unsigned int s_base = ib * MG_Q8_0_BLOCK_SIZE;
        float sv[32];
        const float4* s4 = (const float4*)(sw_smem + s_base);
        #pragma unroll
        for (int kk = 0; kk < 8; kk++) {
            float4 v = s4[kk];
            sv[kk * 4 + 0] = v.x; sv[kk * 4 + 1] = v.y;
            sv[kk * 4 + 2] = v.z; sv[kk * 4 + 3] = v.w;
        }
        #pragma unroll
        for (int r = 0; r < MG_NR_DOWN; r++) {
            if (r0 + r >= hidden_dim) break;
            const unsigned char* dp = layer_buf + down_off
                + (size_t)(r0 + r) * row_bytes + (size_t)ib * MG_Q8_0_BLOCK_BYTES;
            float d_scale = mg_load_scale(dp);
            const signed char* dq = (const signed char*)(dp + 2);
            float block_sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < 32; j++) block_sum += (float)dq[j] * sv[j];
            sum_r[r] += d_scale * block_sum;
        }
    }

    #pragma unroll
    for (int r = 0; r < MG_NR_DOWN; r++) sum_r[r] = mg_warp_reduce_sum(sum_r[r]);
    __syncthreads();
    float* reduce_smem = sw_smem;
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < MG_NR_DOWN; r++) reduce_smem[r * num_warps + warp_id] = sum_r[r];
    }
    __syncthreads();
    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < MG_NR_DOWN; r++) {
            if (r0 + r < hidden_dim) {
                float val = (lane < num_warps) ? reduce_smem[r * num_warps + lane] : 0.0f;
                val = mg_warp_reduce_sum(val);
                if (lane == 0) {
                    down_compact[(size_t)c * (size_t)hidden_dim + (r0 + r)] = val;
                }
            }
        }
    }
}

// ===========================================================================
// Stage 3: scatter-accumulate into the output stream.
//
// out[tok, i] = residual[tok, i]
//             + Σ_{slot} expert_weights[tok, slot] * down_compact[col(tok,slot), i]
//
// where col(tok, slot) is the compact column whose col_dst == tok*top_k+slot.
// To avoid a reverse lookup, we pre-build `dst_to_col[tok*top_k + slot] = c`
// on the host (inverse permutation of col_dst). Each thread owns one (tok, i)
// output element and walks the token's top_k slots.
//
// Grid:  (ceil(hidden_dim / MG_BLOCK_DIM), batch, 1)
// Block: (MG_BLOCK_DIM, 1, 1)
// ===========================================================================
#define MG_MAX_TOP_K 16
extern "C" __global__ void moe_grouped_scatter_accum_q8_0(
    const float* __restrict__ down_compact,       // [total_cols * hidden_dim]
    const float* __restrict__ residual,           // [batch, hidden_dim]
    const float* __restrict__ expert_weights,     // [batch, top_k] (router weights)
    const int* __restrict__ dst_to_col,           // [batch * top_k] -> compact col, or -1
    float* __restrict__ out,                       // [batch, hidden_dim]
    unsigned int hidden_dim,
    unsigned int top_k)
{
    const unsigned int tok = blockIdx.y;
    const unsigned int i = blockIdx.x * MG_BLOCK_DIM + threadIdx.x;
    if (i >= hidden_dim) return;

    const unsigned int K = (top_k < MG_MAX_TOP_K) ? top_k : MG_MAX_TOP_K;
    const size_t base = (size_t)tok * (size_t)top_k;
    float acc = residual[(size_t)tok * (size_t)hidden_dim + i];
    #pragma unroll 1
    for (unsigned int slot = 0; slot < K; ++slot) {
        int c = dst_to_col[base + slot];
        if (c < 0) continue; // defensive; every slot should map to a column
        float w = expert_weights[base + slot];
        acc += w * down_compact[(size_t)c * (size_t)hidden_dim + i];
    }
    out[(size_t)tok * (size_t)hidden_dim + i] = acc;
}

// ===========================================================================
// TILED shmem-staged grouped gate+up+SwiGLU.
//
// Replaces the per-column matvec (`moe_grouped_gate_up_swiglu_q8_0`) — the
// dominant prefill cost (routed FFN = 73% of prefill, runtime-measured). Uses a
// shmem-staged tiled design (per-thread accumulators, NO cross-thread reduction)
// adapted for the expert-grouped case via a host-built flattened
// column-tile list.
//
// CTA tile = TGU_BM=16 compact columns (same expert) x TGU_BN=64 rows x TGU_BK=8
// q-blocks. 256 threads (8 warps). Each thread owns 1 column x 4 rows -> 8 F32
// accumulators (gate[4]+up[4]), NO reduction. ~42.4 KB static shmem (<48 KB).
//
// Grid: (num_tiles, inter_dim/TGU_BN, 1). blockIdx.x indexes a ColTile16
// {expert, col_start, col_count}; blockIdx.y the 64-row tile.
//
// NUMERICS (load-bearing, router-fidelity): per 32-block exact int32 dp4a dot ->
// f32(w_scale*x_scale) -> f32 accumulate, single-thread sequential across blocks
// (the allowed regrouping vs the warp-tree per-column kernel). Same __float2int_rn
// activation quant (amax/127). No fast math.
//
// SHAPE REQUIREMENTS (host-guarded): hidden_dim % 256 == 0 (K-blocks multiple of
// TGU_BK=8) and inter_dim % TGU_BN == 0 (no row tail). Qwen3.5-MoE: H=2048 (64
// q-blocks, 8 stages), I=1408 (22 row-tiles) — both exact.
// ===========================================================================
#define TGU_BM       16
#define TGU_BN       64
#define TGU_BK       8
#define TGU_COL_PAD  (TGU_BM + 1)   // 17
#define TGU_ROW_PAD  (TGU_BN + 1)   // 65

// dp4a signed 8-bit dot-accumulate.
__device__ __forceinline__ int mg_dp4a_s8(int a, int b, int c) {
    int d;
    asm volatile("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}
// Pack 4 signed bytes into one int32 (little-endian lanes) for dp4a.
__device__ __forceinline__ int mg_pack_i8x4(int q0, int q1, int q2, int q3) {
    unsigned int u0 = (unsigned int)((unsigned char)q0);
    unsigned int u1 = (unsigned int)((unsigned char)q1);
    unsigned int u2 = (unsigned int)((unsigned char)q2);
    unsigned int u3 = (unsigned int)((unsigned char)q3);
    return (int)(u0 | (u1 << 8) | (u2 << 16) | (u3 << 24));
}

extern "C" __global__ __launch_bounds__(256, 2)
void moe_grouped_gate_up_swiglu_q8_0_tiled(
    const float* __restrict__ normed,                       // [batch, hidden_dim]
    const unsigned char* __restrict__ layer_buf,            // raw weight blob
    const int* __restrict__ col_src_tok,                    // [total_cols]
    const int* __restrict__ tiles,                          // [num_tiles*4] {expert,col_start,col_count,pad}
    unsigned int num_tiles,
    const unsigned long long* __restrict__ gate_up_offsets, // [num_experts*2]
    float* __restrict__ swiglu_compact,                     // [total_cols * inter_dim]
    unsigned int hidden_dim,
    unsigned int inter_dim)
{
    const unsigned int tile_id  = blockIdx.x;
    const unsigned int row_tile = blockIdx.y;
    if (tile_id >= num_tiles) return;

    const unsigned int tid      = threadIdx.x;
    const unsigned int lane8    = tid & 7;
    const unsigned int group8   = tid >> 3;     // 0..31
    const unsigned int col      = tid & 15;     // 0..15
    const unsigned int row_quad = tid >> 4;     // 0..15
    const unsigned int r0       = row_quad << 2; // local row base 0,4,..,60

    const unsigned int row_base = row_tile * TGU_BN;
    if (row_base >= inter_dim) return;

    const unsigned int num_blocks = hidden_dim / MG_Q8_0_BLOCK_SIZE; // K-blocks
    const size_t row_bytes = (size_t)num_blocks * MG_Q8_0_BLOCK_BYTES;

    __shared__ int   s_expert;
    __shared__ int   s_col_start;
    __shared__ int   s_col_count;
    __shared__ unsigned long long s_gate_off;
    __shared__ unsigned long long s_up_off;
    __shared__ int   s_src_tok[TGU_BM];
    __shared__ int   s_xq[TGU_BK][8][TGU_COL_PAD];
    __shared__ float s_xs[TGU_BK][TGU_COL_PAD];
    __shared__ int   s_gwq[TGU_BK][8][TGU_ROW_PAD];
    __shared__ float s_gws[TGU_BK][TGU_ROW_PAD];
    __shared__ int   s_uwq[TGU_BK][8][TGU_ROW_PAD];
    __shared__ float s_uws[TGU_BK][TGU_ROW_PAD];

    if (tid == 0) {
        s_expert    = tiles[tile_id * 4 + 0];
        s_col_start = tiles[tile_id * 4 + 1];
        s_col_count = tiles[tile_id * 4 + 2];
        s_gate_off  = gate_up_offsets[(size_t)s_expert * 2 + 0];
        s_up_off    = gate_up_offsets[(size_t)s_expert * 2 + 1];
    }
    __syncthreads();

    const int active_cols = s_col_count;
    const int col_start   = s_col_start;
    if ((int)tid < TGU_BM) {
        s_src_tok[tid] = ((int)tid < active_cols) ? col_src_tok[col_start + (int)tid] : 0;
    }
    __syncthreads();

    const unsigned char* gate_base = layer_buf + s_gate_off;
    const unsigned char* up_base   = layer_buf + s_up_off;

    float g0=0.f, g1=0.f, g2=0.f, g3=0.f;
    float u0=0.f, u1=0.f, u2=0.f, u3=0.f;

    for (unsigned int k0 = 0; k0 < num_blocks; k0 += TGU_BK) {
        // ---- Stage 1: gather + quantize x into shared (8-thread subgroups). ----
        {
            const int m       = group8 & 15;        // column 0..15
            const int kk_base = group8 >> 4;        // 0 or 1
            const bool active = (m < active_cols);
            const int tok     = s_src_tok[m];
            const float* xrow = active
                ? (normed + (size_t)tok * (size_t)hidden_dim)
                : normed;
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int kk = kk_base + (t << 1); // 0,2,4,6 or 1,3,5,7
                float v0=0.f, v1=0.f, v2=0.f, v3=0.f;
                if (active) {
                    const unsigned int k_elem = (k0 + (unsigned int)kk) << 5; // *32
                    const unsigned int offs   = lane8 << 2;
                    const float* xp = xrow + k_elem + offs;
                    v0=xp[0]; v1=xp[1]; v2=xp[2]; v3=xp[3];
                }
                float vmax = fabsf(v0);
                float a = fabsf(v1); if (a>vmax) vmax=a;
                a = fabsf(v2); if (a>vmax) vmax=a;
                a = fabsf(v3); if (a>vmax) vmax=a;
                float other;
                other=__shfl_xor_sync(0xffffffffu,vmax,4,8); if(other>vmax)vmax=other;
                other=__shfl_xor_sync(0xffffffffu,vmax,2,8); if(other>vmax)vmax=other;
                other=__shfl_xor_sync(0xffffffffu,vmax,1,8); if(other>vmax)vmax=other;
                const float amax = __shfl_sync(0xffffffffu, vmax, 0, 8);
                const float scale = (amax>0.f) ? (amax * (1.0f/127.0f)) : 0.f;
                const float inv   = (amax>0.f) ? (127.0f/amax) : 0.f;
                int q0=0,q1=0,q2=0,q3=0;
                if (inv>0.f) {
                    q0=__float2int_rn(v0*inv); q1=__float2int_rn(v1*inv);
                    q2=__float2int_rn(v2*inv); q3=__float2int_rn(v3*inv);
                }
                s_xq[kk][lane8][m] = mg_pack_i8x4(q0,q1,q2,q3);
                if (lane8==0) s_xs[kk][m] = scale;
            }
        }
        // ---- Stage 2: stage gate+up weight tiles into shared. ----
        {
            const int row0_local = group8;  // 0..31
            const int kk         = lane8;   // 0..7
            #pragma unroll
            for (int pass = 0; pass < 2; ++pass) {
                const int row_local  = row0_local + (pass << 5); // +32
                const unsigned int row_global = row_base + (unsigned int)row_local;
                int gp[8]; int up[8]; float gsc=0.f, usc=0.f;
                if (row_global < inter_dim) {
                    const unsigned char* gblk = gate_base
                        + (size_t)row_global * row_bytes
                        + (size_t)(k0 + (unsigned int)kk) * MG_Q8_0_BLOCK_BYTES;
                    const unsigned char* ublk = up_base
                        + (size_t)row_global * row_bytes
                        + (size_t)(k0 + (unsigned int)kk) * MG_Q8_0_BLOCK_BYTES;
                    gsc = mg_load_scale(gblk);
                    usc = mg_load_scale(ublk);
                    const signed char* gq = (const signed char*)(gblk + 2);
                    const signed char* uq = (const signed char*)(ublk + 2);
                    #pragma unroll
                    for (int p=0; p<8; ++p) {
                        gp[p]=mg_pack_i8x4(gq[p*4+0],gq[p*4+1],gq[p*4+2],gq[p*4+3]);
                        up[p]=mg_pack_i8x4(uq[p*4+0],uq[p*4+1],uq[p*4+2],uq[p*4+3]);
                    }
                } else {
                    #pragma unroll
                    for (int p=0;p<8;++p){gp[p]=0;up[p]=0;}
                }
                #pragma unroll
                for (int p=0;p<8;++p){ s_gwq[kk][p][row_local]=gp[p]; s_uwq[kk][p][row_local]=up[p]; }
                s_gws[kk][row_local]=gsc; s_uws[kk][row_local]=usc;
            }
        }
        __syncthreads();
        // ---- Stage 3: compute, no reductions. ----
        #pragma unroll
        for (int kk=0; kk<TGU_BK; ++kk) {
            const float xs = s_xs[kk][col];
            int dg0=0,dg1=0,dg2=0,dg3=0,du0=0,du1=0,du2=0,du3=0;
            #pragma unroll
            for (int p=0;p<8;++p) {
                const int xp = s_xq[kk][p][col];
                dg0=mg_dp4a_s8(xp, s_gwq[kk][p][r0+0], dg0);
                dg1=mg_dp4a_s8(xp, s_gwq[kk][p][r0+1], dg1);
                dg2=mg_dp4a_s8(xp, s_gwq[kk][p][r0+2], dg2);
                dg3=mg_dp4a_s8(xp, s_gwq[kk][p][r0+3], dg3);
                du0=mg_dp4a_s8(xp, s_uwq[kk][p][r0+0], du0);
                du1=mg_dp4a_s8(xp, s_uwq[kk][p][r0+1], du1);
                du2=mg_dp4a_s8(xp, s_uwq[kk][p][r0+2], du2);
                du3=mg_dp4a_s8(xp, s_uwq[kk][p][r0+3], du3);
            }
            g0 += xs * s_gws[kk][r0+0] * (float)dg0;
            g1 += xs * s_gws[kk][r0+1] * (float)dg1;
            g2 += xs * s_gws[kk][r0+2] * (float)dg2;
            g3 += xs * s_gws[kk][r0+3] * (float)dg3;
            u0 += xs * s_uws[kk][r0+0] * (float)du0;
            u1 += xs * s_uws[kk][r0+1] * (float)du1;
            u2 += xs * s_uws[kk][r0+2] * (float)du2;
            u3 += xs * s_uws[kk][r0+3] * (float)du3;
        }
        __syncthreads();
    }
    // ---- SwiGLU + store (compact-col major: [col*inter_dim + row]). ----
    if (col < (unsigned int)active_cols) {
        const size_t out_col = (size_t)(col_start + (int)col);
        const size_t base = out_col * (size_t)inter_dim + (size_t)(row_base + r0);
        if (row_base + r0 + 0 < inter_dim) swiglu_compact[base+0] = mg_swiglu(g0, u0);
        if (row_base + r0 + 1 < inter_dim) swiglu_compact[base+1] = mg_swiglu(g1, u1);
        if (row_base + r0 + 2 < inter_dim) swiglu_compact[base+2] = mg_swiglu(g2, u2);
        if (row_base + r0 + 3 < inter_dim) swiglu_compact[base+3] = mg_swiglu(g3, u3);
    }
}

// ===========================================================================
// TILED shmem-staged grouped DOWN projection.
//
// Replaces the per-column matvec down (`moe_grouped_down_q8_0`) — the remaining
// ~1/3 of the routed FFN cost after the tiled gate+up. Same shmem-staged,
// per-thread-accumulator, NO-cross-thread-reduction design as the gate+up tiled
// kernel, reusing the SAME host-built `tiles16` column-tile list.
//
// CTA tile = TD_BM=16 compact columns (same expert) x TD_BN=128 rows x TD_BK=8
// q-blocks. 256 threads (8 warps). Each thread owns 1 column x 8 rows -> 8 F32
// accumulators, NO reduction. ~42.1 KB static shmem (<48 KB). One projection
// only (down) -> one weight buffer in shared (vs gate+up's two).
//
// Grid: (num_tiles, hidden_dim/TD_BN, 1). blockIdx.x indexes a ColTile16
// {expert, col_start, col_count}; blockIdx.y the 128-row tile.
//
// INPUT activation = swiglu_compact[col, inter_dim] (F32, already compact-col
// major from the gate+up stage). K = inter_dim. Unlike gate+up (K=hidden_dim
// =2048=64 q-blocks, exact 8 stages), down's K = inter_dim = 1408 = 44 q-blocks
// = 5 full BK=8 stages + 1 partial stage of 4 -> the loop MASKS k-blocks
// >= num_blocks in BOTH staging and compute (zero-padded), so the K-tail
// contributes nothing. Output N = hidden_dim = 2048 = 16 row-tiles of 128, exact.
//
// NUMERICS (load-bearing, router-fidelity): per 32-block exact int32 dp4a dot ->
// f32(w_scale*x_scale) -> f32 accumulate, single-thread sequential across blocks
// (the allowed regrouping vs the warp-tree per-column kernel). Same
// __float2int_rn activation quant (amax/127). No fast math.
//
// SHAPE REQUIREMENTS (host-guarded): inter_dim % 32 == 0 (whole q-blocks) and
// hidden_dim % TD_BN == 0 (no row tail). Qwen3.5-MoE: I=1408 (44 q-blocks),
// H=2048 (16 row-tiles) — both exact.
// ===========================================================================
#define TD_BM       16
#define TD_BN       128
#define TD_BK       8
#define TD_COL_PAD  (TD_BM + 1)   // 17
#define TD_ROW_PAD  (TD_BN + 1)   // 129

extern "C" __global__ __launch_bounds__(256, 2)
void moe_grouped_down_q8_0_tiled(
    const float* __restrict__ swiglu_compact,               // [total_cols * inter_dim]
    const unsigned char* __restrict__ layer_buf,            // raw weight blob
    const int* __restrict__ tiles,                          // [num_tiles*4] {expert,col_start,col_count,pad}
    unsigned int num_tiles,
    const unsigned long long* __restrict__ down_offsets,    // [num_experts]
    float* __restrict__ down_compact,                       // [total_cols * hidden_dim]
    unsigned int hidden_dim,
    unsigned int inter_dim)
{
    const unsigned int tile_id  = blockIdx.x;
    const unsigned int row_tile = blockIdx.y;
    if (tile_id >= num_tiles) return;

    const unsigned int tid      = threadIdx.x;
    const unsigned int lane8    = tid & 7;
    const unsigned int group8   = tid >> 3;     // 0..31
    const unsigned int col      = tid & 15;     // 0..15
    const unsigned int row_oct  = tid >> 4;     // 0..15
    const unsigned int r0       = row_oct << 3; // local row base 0,8,..,120

    const unsigned int row_base = row_tile * TD_BN;
    if (row_base >= hidden_dim) return;

    const unsigned int num_blocks = inter_dim / MG_Q8_0_BLOCK_SIZE; // K-blocks (e.g. 44)
    const size_t row_bytes = (size_t)num_blocks * MG_Q8_0_BLOCK_BYTES;

    __shared__ int   s_expert;
    __shared__ int   s_col_start;
    __shared__ int   s_col_count;
    __shared__ unsigned long long s_down_off;
    __shared__ int   s_src_col[TD_BM];
    __shared__ int   s_xq[TD_BK][8][TD_COL_PAD];
    __shared__ float s_xs[TD_BK][TD_COL_PAD];
    __shared__ int   s_wq[TD_BK][8][TD_ROW_PAD];
    __shared__ float s_ws[TD_BK][TD_ROW_PAD];

    if (tid == 0) {
        s_expert    = tiles[tile_id * 4 + 0];
        s_col_start = tiles[tile_id * 4 + 1];
        s_col_count = tiles[tile_id * 4 + 2];
        s_down_off  = down_offsets[(size_t)s_expert];
    }
    __syncthreads();

    const int active_cols = s_col_count;
    const int col_start   = s_col_start;
    // Compact-column index in [0, total_cols): the column's swiglu row is
    // swiglu_compact[(col_start+m) * inter_dim], its output row is
    // down_compact[(col_start+m) * hidden_dim]. (Down has no token gather — input
    // is already compact-col major.)
    if ((int)tid < TD_BM) {
        s_src_col[tid] = ((int)tid < active_cols) ? (col_start + (int)tid) : 0;
    }
    __syncthreads();

    const unsigned char* down_base = layer_buf + s_down_off;

    float a0=0.f,a1=0.f,a2=0.f,a3=0.f,a4=0.f,a5=0.f,a6=0.f,a7=0.f;

    for (unsigned int k0 = 0; k0 < num_blocks; k0 += TD_BK) {
        // ---- Stage 1: gather + quantize swiglu activation x into shared. ----
        // 8-thread subgroups, group8 0..31 -> (m=group8&15, kk parity=group8>>4).
        // x source = swiglu_compact (compact-col major, contiguous per column).
        {
            const int m       = group8 & 15;        // column 0..15
            const int kk_base = group8 >> 4;        // 0 or 1
            const bool active = (m < active_cols);
            const int ccol    = s_src_col[m];
            const float* xrow = active
                ? (swiglu_compact + (size_t)ccol * (size_t)inter_dim)
                : swiglu_compact;
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int kk = kk_base + (t << 1); // 0,2,4,6 or 1,3,5,7
                const unsigned int kblk = k0 + (unsigned int)kk;
                const bool kvalid = (kblk < num_blocks);
                float v0=0.f, v1=0.f, v2=0.f, v3=0.f;
                if (active && kvalid) {
                    const unsigned int k_elem = kblk << 5; // *32
                    const unsigned int offs   = lane8 << 2;
                    const float* xp = xrow + k_elem + offs;
                    v0=xp[0]; v1=xp[1]; v2=xp[2]; v3=xp[3];
                }
                float vmax = fabsf(v0);
                float a = fabsf(v1); if (a>vmax) vmax=a;
                a = fabsf(v2); if (a>vmax) vmax=a;
                a = fabsf(v3); if (a>vmax) vmax=a;
                float other;
                other=__shfl_xor_sync(0xffffffffu,vmax,4,8); if(other>vmax)vmax=other;
                other=__shfl_xor_sync(0xffffffffu,vmax,2,8); if(other>vmax)vmax=other;
                other=__shfl_xor_sync(0xffffffffu,vmax,1,8); if(other>vmax)vmax=other;
                const float amax = __shfl_sync(0xffffffffu, vmax, 0, 8);
                const float scale = (amax>0.f) ? (amax * (1.0f/127.0f)) : 0.f;
                const float inv   = (amax>0.f) ? (127.0f/amax) : 0.f;
                int q0=0,q1=0,q2=0,q3=0;
                if (inv>0.f) {
                    q0=__float2int_rn(v0*inv); q1=__float2int_rn(v1*inv);
                    q2=__float2int_rn(v2*inv); q3=__float2int_rn(v3*inv);
                }
                s_xq[kk][lane8][m] = mg_pack_i8x4(q0,q1,q2,q3);
                if (lane8==0) s_xs[kk][m] = scale;
            }
        }
        // ---- Stage 2: stage down weight tile into shared (one projection). ----
        // 128 rows x 8 kblocks. group8 0..31 = row-in-pass, 4 passes (+32 each),
        // lane8 = kk 0..7. Each thread loads one 34-byte q-block per pass.
        {
            const int row0_local = group8;  // 0..31
            const int kk         = lane8;   // 0..7
            const unsigned int kblk = k0 + (unsigned int)kk;
            const bool kvalid = (kblk < num_blocks);
            #pragma unroll
            for (int pass = 0; pass < 4; ++pass) {
                const int row_local  = row0_local + (pass << 5); // +32
                const unsigned int row_global = row_base + (unsigned int)row_local;
                int wp[8]; float wsc=0.f;
                if (row_global < hidden_dim && kvalid) {
                    const unsigned char* wblk = down_base
                        + (size_t)row_global * row_bytes
                        + (size_t)kblk * MG_Q8_0_BLOCK_BYTES;
                    wsc = mg_load_scale(wblk);
                    const signed char* wq = (const signed char*)(wblk + 2);
                    #pragma unroll
                    for (int p=0; p<8; ++p) {
                        wp[p]=mg_pack_i8x4(wq[p*4+0],wq[p*4+1],wq[p*4+2],wq[p*4+3]);
                    }
                } else {
                    #pragma unroll
                    for (int p=0;p<8;++p) wp[p]=0;
                }
                #pragma unroll
                for (int p=0;p<8;++p) s_wq[kk][p][row_local]=wp[p];
                s_ws[kk][row_local]=wsc;
            }
        }
        __syncthreads();
        // ---- Stage 3: compute, no reductions. 8 rows/thread. ----
        #pragma unroll
        for (int kk=0; kk<TD_BK; ++kk) {
            const float xs = s_xs[kk][col];
            int d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0,d7=0;
            #pragma unroll
            for (int p=0;p<8;++p) {
                const int xp = s_xq[kk][p][col];
                d0=mg_dp4a_s8(xp, s_wq[kk][p][r0+0], d0);
                d1=mg_dp4a_s8(xp, s_wq[kk][p][r0+1], d1);
                d2=mg_dp4a_s8(xp, s_wq[kk][p][r0+2], d2);
                d3=mg_dp4a_s8(xp, s_wq[kk][p][r0+3], d3);
                d4=mg_dp4a_s8(xp, s_wq[kk][p][r0+4], d4);
                d5=mg_dp4a_s8(xp, s_wq[kk][p][r0+5], d5);
                d6=mg_dp4a_s8(xp, s_wq[kk][p][r0+6], d6);
                d7=mg_dp4a_s8(xp, s_wq[kk][p][r0+7], d7);
            }
            a0 += xs * s_ws[kk][r0+0] * (float)d0;
            a1 += xs * s_ws[kk][r0+1] * (float)d1;
            a2 += xs * s_ws[kk][r0+2] * (float)d2;
            a3 += xs * s_ws[kk][r0+3] * (float)d3;
            a4 += xs * s_ws[kk][r0+4] * (float)d4;
            a5 += xs * s_ws[kk][r0+5] * (float)d5;
            a6 += xs * s_ws[kk][r0+6] * (float)d6;
            a7 += xs * s_ws[kk][r0+7] * (float)d7;
        }
        __syncthreads();
    }
    // ---- Store (compact-col major: [col*hidden_dim + row]). NO activation. ----
    if (col < (unsigned int)active_cols) {
        const size_t out_col = (size_t)(col_start + (int)col);
        const size_t base = out_col * (size_t)hidden_dim + (size_t)(row_base + r0);
        if (row_base + r0 + 0 < hidden_dim) down_compact[base+0] = a0;
        if (row_base + r0 + 1 < hidden_dim) down_compact[base+1] = a1;
        if (row_base + r0 + 2 < hidden_dim) down_compact[base+2] = a2;
        if (row_base + r0 + 3 < hidden_dim) down_compact[base+3] = a3;
        if (row_base + r0 + 4 < hidden_dim) down_compact[base+4] = a4;
        if (row_base + r0 + 5 < hidden_dim) down_compact[base+5] = a5;
        if (row_base + r0 + 6 < hidden_dim) down_compact[base+6] = a6;
        if (row_base + r0 + 7 < hidden_dim) down_compact[base+7] = a7;
    }
}

// ===========================================================================
// TILED shmem-staged grouped DOWN projection with EXACT
// F32 ACTIVATION (the down-tiled quality rescue).
//
// Identical shmem-staged, per-thread-accumulator, NO-cross-thread-reduction
// structure as `moe_grouped_down_q8_0_tiled` — same tile shape, same
// host `tiles16` list — but it does NOT quantize the F32 swiglu activation to
// int8. Instead it stages the RAW F32 activation into shared and computes the
// per-32-block dot in F32 against the dequantized int8 weight, EXACTLY matching
// the per-column reference's per-block numerics:
//     block_sum = Σ_{j<32} (float)wq[j] * xf[j];   acc += w_scale * block_sum;
//
// Why: the per-column PRISTINE reference (`moe_grouped_down_q8_0`) uses raw F32
// activation. Wave-6's int8 activation quant (amax/127) is the ONLY numeric
// divergence from that reference and is the suspected cause of the GQ-004
// vlong-story DD-SPAM regression. This kernel removes that quant entirely. The
// only remaining reorder vs the reference is per-thread-sequential block
// accumulation (each thread owns its full K reduction) vs the reference's
// warp-tree reduction — the SAME allowed regrouping Wave-5 gate+up used and
// passed PRISTINE. Expected oracle envelope: far tighter than Wave-6's 5.37%.
//
// SHMEM BUDGET: with TD2_BN=64 (vs Wave-6's 128) the weight buffer halves so the
// raw-F32 activation buffer (4× the int8 buffer) fits under 48 KB static:
//   s_xf : TD2_BK(8) * 32 * TD2_COL_PAD(17) * 4B = 17,408 B
//   s_wq : TD2_BK(8) * 8  * TD2_ROW_PAD(65)  * 4B = 16,640 B  (int8 packed int32)
//   s_ws : TD2_BK(8) * TD2_ROW_PAD(65)       * 4B =  2,080 B
//   + small scalars  ≈ 36.2 KB < 48 KB  → no dynamic-shmem opt-in needed.
// TD2_BN=64 ⇒ hidden_dim/64 = 32 row-tiles (2× Wave-6's 16) — more CTAs, more
// occupancy headroom; the F32 inner loop is heavier than dp4a but the per-block
// F32 dot is what guarantees reference-exact numerics.
//
// Each thread owns 1 column × 4 rows → 4 F32 accumulators (TD2_BN=64 / 16 row
// groups = 4 rows/thread). 256 threads = 16 cols × 16 row-octets... here:
//   col   = tid & 15  (0..15)
//   row_q = tid >> 4  (0..15) → r0 = row_q * 4  (0,4,..,60)
//
// SHAPE (host-guarded): inter_dim % 32 == 0, hidden_dim % TD2_BN(64) == 0.
// Qwen3.5-MoE I=1408 (44 q-blocks), H=2048 (32 row-tiles) — both exact.
// ===========================================================================
#define TD2_BM       16
#define TD2_BN       64
#define TD2_BK       8
#define TD2_COL_PAD  (TD2_BM + 1)   // 17
#define TD2_ROW_PAD  (TD2_BN + 1)   // 65

extern "C" __global__ __launch_bounds__(256, 2)
void moe_grouped_down_q8_0_tiled_f32act(
    const float* __restrict__ swiglu_compact,               // [total_cols * inter_dim]
    const unsigned char* __restrict__ layer_buf,            // raw weight blob
    const int* __restrict__ tiles,                          // [num_tiles*4] {expert,col_start,col_count,pad}
    unsigned int num_tiles,
    const unsigned long long* __restrict__ down_offsets,    // [num_experts]
    float* __restrict__ down_compact,                       // [total_cols * hidden_dim]
    unsigned int hidden_dim,
    unsigned int inter_dim)
{
    const unsigned int tile_id  = blockIdx.x;
    const unsigned int row_tile = blockIdx.y;
    if (tile_id >= num_tiles) return;

    const unsigned int tid      = threadIdx.x;
    const unsigned int lane8    = tid & 7;
    const unsigned int group8   = tid >> 3;     // 0..31
    const unsigned int col      = tid & 15;     // 0..15
    const unsigned int row_q    = tid >> 4;     // 0..15
    const unsigned int r0       = row_q << 2;   // local row base 0,4,..,60

    const unsigned int row_base = row_tile * TD2_BN;
    if (row_base >= hidden_dim) return;

    const unsigned int num_blocks = inter_dim / MG_Q8_0_BLOCK_SIZE; // K-blocks (e.g. 44)
    const size_t row_bytes = (size_t)num_blocks * MG_Q8_0_BLOCK_BYTES;

    __shared__ int   s_expert;
    __shared__ int   s_col_start;
    __shared__ int   s_col_count;
    __shared__ unsigned long long s_down_off;
    __shared__ int   s_src_col[TD2_BM];
    __shared__ float s_xf[TD2_BK][32][TD2_COL_PAD];   // raw F32 activation (no quant)
    __shared__ signed char s_wb[TD2_BK][32][TD2_ROW_PAD]; // int8 weight bytes
    __shared__ float s_ws[TD2_BK][TD2_ROW_PAD];       // weight per-block f32 scale

    if (tid == 0) {
        s_expert    = tiles[tile_id * 4 + 0];
        s_col_start = tiles[tile_id * 4 + 1];
        s_col_count = tiles[tile_id * 4 + 2];
        s_down_off  = down_offsets[(size_t)s_expert];
    }
    __syncthreads();

    const int active_cols = s_col_count;
    const int col_start   = s_col_start;
    if ((int)tid < TD2_BM) {
        s_src_col[tid] = ((int)tid < active_cols) ? (col_start + (int)tid) : 0;
    }
    __syncthreads();

    const unsigned char* down_base = layer_buf + s_down_off;

    float a0=0.f,a1=0.f,a2=0.f,a3=0.f;

    for (unsigned int k0 = 0; k0 < num_blocks; k0 += TD2_BK) {
        // ---- Stage 1: gather RAW F32 swiglu activation into shared (no quant). ----
        // 8-thread subgroups: group8 0..31 -> (m=group8&15, kk parity=group8>>4).
        {
            const int m       = group8 & 15;        // column 0..15
            const int kk_base = group8 >> 4;        // 0 or 1
            const bool active = (m < active_cols);
            const int ccol    = s_src_col[m];
            const float* xrow = active
                ? (swiglu_compact + (size_t)ccol * (size_t)inter_dim)
                : swiglu_compact;
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int kk = kk_base + (t << 1); // 0,2,4,6 or 1,3,5,7
                const unsigned int kblk = k0 + (unsigned int)kk;
                const bool kvalid = (kblk < num_blocks);
                const unsigned int offs = lane8 << 2; // 4 elems per lane8
                float v0=0.f,v1=0.f,v2=0.f,v3=0.f;
                if (active && kvalid) {
                    const unsigned int k_elem = kblk << 5; // *32
                    const float* xp = xrow + k_elem + offs;
                    v0=xp[0]; v1=xp[1]; v2=xp[2]; v3=xp[3];
                }
                s_xf[kk][offs+0][m] = v0;
                s_xf[kk][offs+1][m] = v1;
                s_xf[kk][offs+2][m] = v2;
                s_xf[kk][offs+3][m] = v3;
            }
        }
        // ---- Stage 2: stage down weight tile (int8 bytes + f32 scale). ----
        // 64 rows x 8 kblocks. group8 0..31 = row-in-pass, 2 passes (+32 each),
        // lane8 = kk 0..7. Each thread loads one 34-byte q-block per pass.
        {
            const int row0_local = group8;  // 0..31
            const int kk         = lane8;   // 0..7
            const unsigned int kblk = k0 + (unsigned int)kk;
            const bool kvalid = (kblk < num_blocks);
            #pragma unroll
            for (int pass = 0; pass < 2; ++pass) {
                const int row_local  = row0_local + (pass << 5); // +32
                const unsigned int row_global = row_base + (unsigned int)row_local;
                signed char wb[32]; float wsc=0.f;
                if (row_global < hidden_dim && kvalid) {
                    const unsigned char* wblk = down_base
                        + (size_t)row_global * row_bytes
                        + (size_t)kblk * MG_Q8_0_BLOCK_BYTES;
                    wsc = mg_load_scale(wblk);
                    const signed char* wq = (const signed char*)(wblk + 2);
                    #pragma unroll
                    for (int p=0; p<32; ++p) wb[p]=wq[p];
                } else {
                    #pragma unroll
                    for (int p=0;p<32;++p) wb[p]=0;
                }
                #pragma unroll
                for (int p=0;p<32;++p) s_wb[kk][p][row_local]=wb[p];
                s_ws[kk][row_local]=wsc;
            }
        }
        __syncthreads();
        // ---- Stage 3: compute, no reductions. 4 rows/thread, F32 dot. ----
        // EXACT per-column reference math: per 32-block, block_sum = Σ_j
        // (float)wq[j]*xf[j], then acc += w_scale * block_sum (per-block scale,
        // sequential across blocks). Raw F32 activation — no int8 quant.
        #pragma unroll
        for (int kk=0; kk<TD2_BK; ++kk) {
            float bs0=0.f,bs1=0.f,bs2=0.f,bs3=0.f;
            #pragma unroll
            for (int j=0;j<32;++j) {
                const float xv = s_xf[kk][j][col];
                bs0 += (float)s_wb[kk][j][r0+0] * xv;
                bs1 += (float)s_wb[kk][j][r0+1] * xv;
                bs2 += (float)s_wb[kk][j][r0+2] * xv;
                bs3 += (float)s_wb[kk][j][r0+3] * xv;
            }
            a0 += s_ws[kk][r0+0] * bs0;
            a1 += s_ws[kk][r0+1] * bs1;
            a2 += s_ws[kk][r0+2] * bs2;
            a3 += s_ws[kk][r0+3] * bs3;
        }
        __syncthreads();
    }
    // ---- Store (compact-col major: [col*hidden_dim + row]). NO activation. ----
    if (col < (unsigned int)active_cols) {
        const size_t out_col = (size_t)(col_start + (int)col);
        const size_t base = out_col * (size_t)hidden_dim + (size_t)(row_base + r0);
        if (row_base + r0 + 0 < hidden_dim) down_compact[base+0] = a0;
        if (row_base + r0 + 1 < hidden_dim) down_compact[base+1] = a1;
        if (row_base + r0 + 2 < hidden_dim) down_compact[base+2] = a2;
        if (row_base + r0 + 3 < hidden_dim) down_compact[base+3] = a3;
    }
}

// ===========================================================================
// Port: Q4_0 grouped TILED FFN — f32-activation design.
//
// Ports the Wave-5/Wave-7 shmem-staged, per-thread-accumulator, no-cross-thread-
// reduction tiled grouped GEMM to Q4_0 expert weights. The ONLY structural delta
// vs the Q8 f32act kernels is the WEIGHT decode: Q4_0 blocks are 18 bytes
// (2-byte f16 scale + 16 nibble bytes, DE-INTERLEAVED). Byte idx in [0,15] holds
// K-element `idx` in the LOW nibble ((b&0xF)-8) and K-element `idx+16` in the HIGH
// nibble ((b>>4)-8). After decode-to-int8 the per-32-block F32 dot + per-block-scale
// accumulation is IDENTICAL to the per-token Q4_0 reference (moe_batched_q4_0.cu),
// so these kernels stay in the PRISTINE near-tie class (only the cross-block F32
// accumulation grouping changes: per-thread sequential vs the per-column warp-tree).
//
// RAW F32 activation (no int8 quant) — matches the per-token Q4_0 reference exactly
// AND avoids the int8-activation-quant near-tie that tips the verylong gate.
// Shape guards (host): hidden_dim % TQ4_BN == 0 (gate_up rows), hidden_dim % 32 == 0
// (gate_up K-blocks), inter_dim % 32 == 0 (down K-blocks), hidden_dim % TQ4D_BN == 0
// (down rows). Qwen3.5-MoE: H=2048, I=1408 — all exact.
// ===========================================================================
#define MG_Q4_0_BLOCK_SIZE  32
#define MG_Q4_0_BLOCK_BYTES 18   // 2-byte f16 scale + 16 nibble bytes (32 quants)

// Decode a Q4_0 block's 32 quants into 32 signed int8 (K-order), de-interleaved:
// dst[idx] = (qs[idx] & 0xF) - 8  (K-elements 0..15)
// dst[idx+16] = (qs[idx] >> 4) - 8 (K-elements 16..31)
__device__ __forceinline__ void mg_q4_0_decode32(const unsigned char* qs, signed char* dst) {
    #pragma unroll
    for (int idx = 0; idx < 16; ++idx) {
        const unsigned char b = qs[idx];
        dst[idx]      = (signed char)((int)(b & 0x0F) - 8);
        dst[idx + 16] = (signed char)((int)(b >> 4)   - 8);
    }
}

// ---- gate+up+SwiGLU, f32-activation tiled (Q4_0). ----
// CTA = TQ4_BM cols x TQ4_BN rows x TQ4_BK k-blocks, 256 threads, 4 F32 acc/thread
// (gate[?]+up[?]). Mirrors moe_grouped_down_q8_0_tiled_f32act layout: BN=64 row tile,
// 1 col x 4 rows per thread. Two weight buffers (gate+up) -> BN=64 fits 48 KB.
// SHMEM: s_xf 8*32*17*4=17408 + (s_gwb+s_uwb) 2*8*32*65=33280 + (s_gws+s_uws)
//   2*8*65*4=4160 = ~54.8 KB > 48 KB static. Use BN=32 to fit: weight bufs halve.
#define TQ4_BM       16
#define TQ4_BN       32
#define TQ4_BK       8
#define TQ4_COL_PAD  (TQ4_BM + 1)   // 17
#define TQ4_ROW_PAD  (TQ4_BN + 1)   // 33

extern "C" __global__ __launch_bounds__(256, 2)
void moe_grouped_gate_up_swiglu_q4_0_tiled_f32act(
    const float* __restrict__ normed,                       // [batch, hidden_dim]
    const unsigned char* __restrict__ layer_buf,            // raw weight blob
    const int* __restrict__ col_src_tok,                    // [total_cols]
    const int* __restrict__ tiles,                          // [num_tiles*4] {expert,col_start,col_count,pad}
    unsigned int num_tiles,
    const unsigned long long* __restrict__ gate_up_offsets, // [num_experts*2]
    float* __restrict__ swiglu_compact,                     // [total_cols * inter_dim]
    unsigned int hidden_dim,
    unsigned int inter_dim)
{
    const unsigned int tile_id  = blockIdx.x;
    const unsigned int row_tile = blockIdx.y;
    if (tile_id >= num_tiles) return;

    const unsigned int tid      = threadIdx.x;
    const unsigned int lane8    = tid & 7;
    const unsigned int group8   = tid >> 3;     // 0..31
    const unsigned int col      = tid & 15;     // 0..15
    const unsigned int row_q    = tid >> 4;     // 0..15
    const unsigned int r0       = row_q << 1;   // local row base 0,2,..,30 (TQ4_BN/16=2)

    const unsigned int row_base = row_tile * TQ4_BN;
    if (row_base >= inter_dim) return;

    const unsigned int num_blocks = hidden_dim / MG_Q4_0_BLOCK_SIZE; // K-blocks
    const size_t row_bytes = (size_t)num_blocks * MG_Q4_0_BLOCK_BYTES;

    __shared__ int   s_expert;
    __shared__ int   s_col_start;
    __shared__ int   s_col_count;
    __shared__ unsigned long long s_gate_off;
    __shared__ unsigned long long s_up_off;
    __shared__ int   s_src_tok[TQ4_BM];
    __shared__ float s_xf[TQ4_BK][32][TQ4_COL_PAD];        // raw F32 activation
    __shared__ signed char s_gwb[TQ4_BK][32][TQ4_ROW_PAD]; // gate int8 weight bytes
    __shared__ signed char s_uwb[TQ4_BK][32][TQ4_ROW_PAD]; // up   int8 weight bytes
    __shared__ float s_gws[TQ4_BK][TQ4_ROW_PAD];           // gate per-block scale
    __shared__ float s_uws[TQ4_BK][TQ4_ROW_PAD];           // up   per-block scale

    if (tid == 0) {
        s_expert    = tiles[tile_id * 4 + 0];
        s_col_start = tiles[tile_id * 4 + 1];
        s_col_count = tiles[tile_id * 4 + 2];
        s_gate_off  = gate_up_offsets[(size_t)s_expert * 2 + 0];
        s_up_off    = gate_up_offsets[(size_t)s_expert * 2 + 1];
    }
    __syncthreads();

    const int active_cols = s_col_count;
    const int col_start   = s_col_start;
    if ((int)tid < TQ4_BM) {
        s_src_tok[tid] = ((int)tid < active_cols) ? col_src_tok[col_start + (int)tid] : 0;
    }
    __syncthreads();

    const unsigned char* gate_base = layer_buf + s_gate_off;
    const unsigned char* up_base   = layer_buf + s_up_off;

    float g0=0.f, g1=0.f, u0=0.f, u1=0.f;

    for (unsigned int k0 = 0; k0 < num_blocks; k0 += TQ4_BK) {
        // ---- Stage 1: gather RAW F32 activation into shared (no quant). ----
        {
            const int m       = group8 & 15;        // column 0..15
            const int kk_base = group8 >> 4;        // 0 or 1
            const bool active = (m < active_cols);
            const int tok     = s_src_tok[m];
            const float* xrow = active
                ? (normed + (size_t)tok * (size_t)hidden_dim)
                : normed;
            const unsigned int offs = lane8 << 2; // 4 elems per lane8
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int kk = kk_base + (t << 1); // 0,2,4,6 or 1,3,5,7
                const unsigned int kblk = k0 + (unsigned int)kk;
                const bool kvalid = (kblk < num_blocks);
                float v0=0.f,v1=0.f,v2=0.f,v3=0.f;
                if (active && kvalid) {
                    const unsigned int k_elem = kblk << 5; // *32
                    const float* xp = xrow + k_elem + offs;
                    v0=xp[0]; v1=xp[1]; v2=xp[2]; v3=xp[3];
                }
                s_xf[kk][offs+0][m] = v0;
                s_xf[kk][offs+1][m] = v1;
                s_xf[kk][offs+2][m] = v2;
                s_xf[kk][offs+3][m] = v3;
            }
        }
        // ---- Stage 2: stage gate+up weight tiles (Q4_0 nibble-decode -> int8). ----
        {
            const int row0_local = group8;  // 0..31 (= TQ4_BN rows, one pass)
            const int kk         = lane8;   // 0..7
            const unsigned int kblk = k0 + (unsigned int)kk;
            const bool kvalid = (kblk < num_blocks);
            const int row_local  = row0_local;
            const unsigned int row_global = row_base + (unsigned int)row_local;
            signed char gwb[32]; signed char uwb[32]; float gsc=0.f, usc=0.f;
            if (row_global < inter_dim && kvalid) {
                const unsigned char* gblk = gate_base
                    + (size_t)row_global * row_bytes
                    + (size_t)kblk * MG_Q4_0_BLOCK_BYTES;
                const unsigned char* ublk = up_base
                    + (size_t)row_global * row_bytes
                    + (size_t)kblk * MG_Q4_0_BLOCK_BYTES;
                gsc = mg_load_scale(gblk);
                usc = mg_load_scale(ublk);
                mg_q4_0_decode32(gblk + 2, gwb);
                mg_q4_0_decode32(ublk + 2, uwb);
            } else {
                #pragma unroll
                for (int p=0;p<32;++p){gwb[p]=0;uwb[p]=0;}
            }
            #pragma unroll
            for (int p=0;p<32;++p){ s_gwb[kk][p][row_local]=gwb[p]; s_uwb[kk][p][row_local]=uwb[p]; }
            s_gws[kk][row_local]=gsc; s_uws[kk][row_local]=usc;
        }
        __syncthreads();
        // ---- Stage 3: compute, no reductions. 2 rows/thread, F32 dot. ----
        #pragma unroll
        for (int kk=0; kk<TQ4_BK; ++kk) {
            float bg0=0.f,bg1=0.f,bu0=0.f,bu1=0.f;
            #pragma unroll
            for (int j=0;j<32;++j) {
                const float xv = s_xf[kk][j][col];
                bg0 += (float)s_gwb[kk][j][r0+0] * xv;
                bg1 += (float)s_gwb[kk][j][r0+1] * xv;
                bu0 += (float)s_uwb[kk][j][r0+0] * xv;
                bu1 += (float)s_uwb[kk][j][r0+1] * xv;
            }
            g0 += s_gws[kk][r0+0] * bg0;
            g1 += s_gws[kk][r0+1] * bg1;
            u0 += s_uws[kk][r0+0] * bu0;
            u1 += s_uws[kk][r0+1] * bu1;
        }
        __syncthreads();
    }
    // ---- SwiGLU + store (compact-col major: [col*inter_dim + row]). ----
    if (col < (unsigned int)active_cols) {
        const size_t out_col = (size_t)(col_start + (int)col);
        const size_t base = out_col * (size_t)inter_dim + (size_t)(row_base + r0);
        if (row_base + r0 + 0 < inter_dim) swiglu_compact[base+0] = mg_swiglu(g0, u0);
        if (row_base + r0 + 1 < inter_dim) swiglu_compact[base+1] = mg_swiglu(g1, u1);
    }
}

// ---- down, f32-activation tiled (Q4_0). ----
// Mirrors moe_grouped_down_q8_0_tiled_f32act exactly; only the weight decode
// (Q4_0 nibble) differs. CTA = TQ4D_BM cols x TQ4D_BN rows x TQ4D_BK blocks.
// One weight buffer (down) -> BN=64 fits 48 KB (same budget as the Q8 f32act down).
#define TQ4D_BM       16
#define TQ4D_BN       64
#define TQ4D_BK       8
#define TQ4D_COL_PAD  (TQ4D_BM + 1)   // 17
#define TQ4D_ROW_PAD  (TQ4D_BN + 1)   // 65

extern "C" __global__ __launch_bounds__(256, 2)
void moe_grouped_down_q4_0_tiled_f32act(
    const float* __restrict__ swiglu_compact,               // [total_cols * inter_dim]
    const unsigned char* __restrict__ layer_buf,            // raw weight blob
    const int* __restrict__ tiles,                          // [num_tiles*4] {expert,col_start,col_count,pad}
    unsigned int num_tiles,
    const unsigned long long* __restrict__ down_offsets,    // [num_experts]
    float* __restrict__ down_compact,                       // [total_cols * hidden_dim]
    unsigned int hidden_dim,
    unsigned int inter_dim)
{
    const unsigned int tile_id  = blockIdx.x;
    const unsigned int row_tile = blockIdx.y;
    if (tile_id >= num_tiles) return;

    const unsigned int tid      = threadIdx.x;
    const unsigned int lane8    = tid & 7;
    const unsigned int group8   = tid >> 3;     // 0..31
    const unsigned int col      = tid & 15;     // 0..15
    const unsigned int row_q    = tid >> 4;     // 0..15
    const unsigned int r0       = row_q << 2;   // local row base 0,4,..,60

    const unsigned int row_base = row_tile * TQ4D_BN;
    if (row_base >= hidden_dim) return;

    const unsigned int num_blocks = inter_dim / MG_Q4_0_BLOCK_SIZE; // K-blocks (e.g. 44)
    const size_t row_bytes = (size_t)num_blocks * MG_Q4_0_BLOCK_BYTES;

    __shared__ int   s_expert;
    __shared__ int   s_col_start;
    __shared__ int   s_col_count;
    __shared__ unsigned long long s_down_off;
    __shared__ int   s_src_col[TQ4D_BM];
    __shared__ float s_xf[TQ4D_BK][32][TQ4D_COL_PAD];        // raw F32 activation
    __shared__ signed char s_wb[TQ4D_BK][32][TQ4D_ROW_PAD];  // int8 weight bytes
    __shared__ float s_ws[TQ4D_BK][TQ4D_ROW_PAD];            // weight per-block scale

    if (tid == 0) {
        s_expert    = tiles[tile_id * 4 + 0];
        s_col_start = tiles[tile_id * 4 + 1];
        s_col_count = tiles[tile_id * 4 + 2];
        s_down_off  = down_offsets[(size_t)s_expert];
    }
    __syncthreads();

    const int active_cols = s_col_count;
    const int col_start   = s_col_start;
    if ((int)tid < TQ4D_BM) {
        s_src_col[tid] = ((int)tid < active_cols) ? (col_start + (int)tid) : 0;
    }
    __syncthreads();

    const unsigned char* down_base = layer_buf + s_down_off;

    float a0=0.f,a1=0.f,a2=0.f,a3=0.f;

    for (unsigned int k0 = 0; k0 < num_blocks; k0 += TQ4D_BK) {
        // ---- Stage 1: gather RAW F32 swiglu activation into shared (no quant). ----
        {
            const int m       = group8 & 15;        // column 0..15
            const int kk_base = group8 >> 4;        // 0 or 1
            const bool active = (m < active_cols);
            const int ccol    = s_src_col[m];
            const float* xrow = active
                ? (swiglu_compact + (size_t)ccol * (size_t)inter_dim)
                : swiglu_compact;
            const unsigned int offs = lane8 << 2; // 4 elems per lane8
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int kk = kk_base + (t << 1); // 0,2,4,6 or 1,3,5,7
                const unsigned int kblk = k0 + (unsigned int)kk;
                const bool kvalid = (kblk < num_blocks);
                float v0=0.f,v1=0.f,v2=0.f,v3=0.f;
                if (active && kvalid) {
                    const unsigned int k_elem = kblk << 5; // *32
                    const float* xp = xrow + k_elem + offs;
                    v0=xp[0]; v1=xp[1]; v2=xp[2]; v3=xp[3];
                }
                s_xf[kk][offs+0][m] = v0;
                s_xf[kk][offs+1][m] = v1;
                s_xf[kk][offs+2][m] = v2;
                s_xf[kk][offs+3][m] = v3;
            }
        }
        // ---- Stage 2: stage down weight tile (Q4_0 nibble-decode -> int8). ----
        // 64 rows x 8 kblocks. group8 0..31 = row-in-pass, 2 passes (+32 each),
        // lane8 = kk 0..7. Each thread decodes one 18-byte q-block per pass.
        {
            const int row0_local = group8;  // 0..31
            const int kk         = lane8;   // 0..7
            const unsigned int kblk = k0 + (unsigned int)kk;
            const bool kvalid = (kblk < num_blocks);
            #pragma unroll
            for (int pass = 0; pass < 2; ++pass) {
                const int row_local  = row0_local + (pass << 5); // +32
                const unsigned int row_global = row_base + (unsigned int)row_local;
                signed char wb[32]; float wsc=0.f;
                if (row_global < hidden_dim && kvalid) {
                    const unsigned char* wblk = down_base
                        + (size_t)row_global * row_bytes
                        + (size_t)kblk * MG_Q4_0_BLOCK_BYTES;
                    wsc = mg_load_scale(wblk);
                    mg_q4_0_decode32(wblk + 2, wb);
                } else {
                    #pragma unroll
                    for (int p=0;p<32;++p) wb[p]=0;
                }
                #pragma unroll
                for (int p=0;p<32;++p) s_wb[kk][p][row_local]=wb[p];
                s_ws[kk][row_local]=wsc;
            }
        }
        __syncthreads();
        // ---- Stage 3: compute, no reductions. 4 rows/thread, F32 dot. ----
        #pragma unroll
        for (int kk=0; kk<TQ4D_BK; ++kk) {
            float bs0=0.f,bs1=0.f,bs2=0.f,bs3=0.f;
            #pragma unroll
            for (int j=0;j<32;++j) {
                const float xv = s_xf[kk][j][col];
                bs0 += (float)s_wb[kk][j][r0+0] * xv;
                bs1 += (float)s_wb[kk][j][r0+1] * xv;
                bs2 += (float)s_wb[kk][j][r0+2] * xv;
                bs3 += (float)s_wb[kk][j][r0+3] * xv;
            }
            a0 += s_ws[kk][r0+0] * bs0;
            a1 += s_ws[kk][r0+1] * bs1;
            a2 += s_ws[kk][r0+2] * bs2;
            a3 += s_ws[kk][r0+3] * bs3;
        }
        __syncthreads();
    }
    // ---- Store (compact-col major: [col*hidden_dim + row]). NO activation. ----
    if (col < (unsigned int)active_cols) {
        const size_t out_col = (size_t)(col_start + (int)col);
        const size_t base = out_col * (size_t)hidden_dim + (size_t)(row_base + r0);
        if (row_base + r0 + 0 < hidden_dim) down_compact[base+0] = a0;
        if (row_base + r0 + 1 < hidden_dim) down_compact[base+1] = a1;
        if (row_base + r0 + 2 < hidden_dim) down_compact[base+2] = a2;
        if (row_base + r0 + 3 < hidden_dim) down_compact[base+3] = a3;
    }
}

// ===========================================================================
// Port: BF16 grouped TILED FFN — f32-activation design (shmem-corrected).
// BF16 weights = contiguous unsigned short (no blocks/scales). To stay under the
// 48 KB static shmem budget, weights are STAGED AS u16 (2 B) and converted to F32
// in the compute loop (bits<<16 -> __int_as_float). Raw F32 activation; flat
// per-thread F32 dot matches the per-token bf16 reference (moe_batched_bf16.cu).
// gate_up: TBF_BN=16 rows (2 weight bufs); down: TBFD_BN=32 rows (1 weight buf).
// Shapes (host): hidden%256 (gate_up K), inter%16 (gate_up rows), hidden%32 (down
// rows), inter%32 (down K, tail masked).
// ===========================================================================
#define TBF_BM      16
#define TBF_BN      16
#define TBF_BK      8
#define TBF_COL_PAD (TBF_BM + 1)
#define TBF_ROW_PAD (TBF_BN + 1)

__device__ __forceinline__ float mg_bf16_to_f32(unsigned short bits) {
    unsigned int x = ((unsigned int)bits) << 16;
    return __int_as_float((int)x);
}

extern "C" __global__ __launch_bounds__(256, 2)
void moe_grouped_gate_up_swiglu_bf16_tiled_f32act(
    const float* __restrict__ normed,
    const unsigned char* __restrict__ layer_buf,
    const int* __restrict__ col_src_tok,
    const int* __restrict__ tiles,
    unsigned int num_tiles,
    const unsigned long long* __restrict__ gate_up_offsets,
    float* __restrict__ swiglu_compact,
    unsigned int hidden_dim,
    unsigned int inter_dim)
{
    const unsigned int tile_id  = blockIdx.x;
    const unsigned int row_tile = blockIdx.y;
    if (tile_id >= num_tiles) return;
    const unsigned int tid    = threadIdx.x;
    const unsigned int lane8  = tid & 7;
    const unsigned int group8 = tid >> 3;          // 0..31
    const unsigned int col    = tid & 15;          // token 0..15
    const unsigned int row_q  = tid >> 4;          // 0..15
    const unsigned int r0     = row_q;             // 1 row/thread (TBF_BN=16)
    const unsigned int row_base = row_tile * TBF_BN;
    if (row_base >= inter_dim) return;
    const unsigned int num_chunks = hidden_dim / 32;
    const size_t row_elems = (size_t)hidden_dim;
    __shared__ int   s_expert;
    __shared__ int   s_col_start;
    __shared__ int   s_col_count;
    __shared__ unsigned long long s_gate_off;
    __shared__ unsigned long long s_up_off;
    __shared__ int   s_src_tok[TBF_BM];
    __shared__ float          s_xf[TBF_BK][32][TBF_COL_PAD];
    __shared__ unsigned short s_gwh[TBF_BK][32][TBF_ROW_PAD];
    __shared__ unsigned short s_uwh[TBF_BK][32][TBF_ROW_PAD];
    if (tid == 0) {
        s_expert    = tiles[tile_id * 4 + 0];
        s_col_start = tiles[tile_id * 4 + 1];
        s_col_count = tiles[tile_id * 4 + 2];
        s_gate_off  = gate_up_offsets[(size_t)s_expert * 2 + 0];
        s_up_off    = gate_up_offsets[(size_t)s_expert * 2 + 1];
    }
    __syncthreads();
    const int active_cols = s_col_count;
    const int col_start   = s_col_start;
    if ((int)tid < TBF_BM) {
        s_src_tok[tid] = ((int)tid < active_cols) ? col_src_tok[col_start + (int)tid] : 0;
    }
    __syncthreads();
    const unsigned short* gate_base = (const unsigned short*)(layer_buf + s_gate_off);
    const unsigned short* up_base   = (const unsigned short*)(layer_buf + s_up_off);
    float g0=0.f,u0=0.f;
    for (unsigned int k0 = 0; k0 < num_chunks; k0 += TBF_BK) {
        {
            const int m       = group8 & 15;
            const int kk_base = group8 >> 4;
            const bool active = (m < active_cols);
            const int tok     = s_src_tok[m];
            const float* xrow = active ? (normed + (size_t)tok * hidden_dim) : normed;
            const unsigned int offs = lane8 << 2;
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int kk = kk_base + (t << 1);
                const unsigned int kchunk = k0 + (unsigned int)kk;
                const bool kvalid = (kchunk < num_chunks);
                float v0=0.f,v1=0.f,v2=0.f,v3=0.f;
                if (active && kvalid) {
                    const unsigned int k_elem = kchunk << 5;
                    const float* xp = xrow + k_elem + offs;
                    v0=xp[0]; v1=xp[1]; v2=xp[2]; v3=xp[3];
                }
                s_xf[kk][offs+0][m]=v0; s_xf[kk][offs+1][m]=v1;
                s_xf[kk][offs+2][m]=v2; s_xf[kk][offs+3][m]=v3;
            }
        }
        {
            // 16 rows x 8 kchunks: 128 threads load (group8 0..15 = row, lane8 = kk).
            const int row_local = group8;          // 0..31 (only 0..15 used)
            const int kk        = lane8;           // 0..7
            const unsigned int kchunk = k0 + (unsigned int)kk;
            const bool kvalid = (kchunk < num_chunks);
            if (row_local < TBF_BN) {
                const unsigned int row_global = row_base + (unsigned int)row_local;
                #pragma unroll
                for (int p=0;p<32;++p) {
                    unsigned short gv=0, uv=0;
                    if (row_global < inter_dim && kvalid) {
                        const unsigned short* grow = gate_base + (size_t)row_global * row_elems + (size_t)(kchunk << 5);
                        const unsigned short* urow = up_base   + (size_t)row_global * row_elems + (size_t)(kchunk << 5);
                        gv=grow[p]; uv=urow[p];
                    }
                    s_gwh[kk][p][row_local]=gv; s_uwh[kk][p][row_local]=uv;
                }
            }
        }
        __syncthreads();
        #pragma unroll
        for (int kk=0; kk<TBF_BK; ++kk) {
            #pragma unroll
            for (int j=0;j<32;++j) {
                const float xv = s_xf[kk][j][col];
                g0 += mg_bf16_to_f32(s_gwh[kk][j][r0])*xv;
                u0 += mg_bf16_to_f32(s_uwh[kk][j][r0])*xv;
            }
        }
        __syncthreads();
    }
    if (col < (unsigned int)active_cols) {
        const size_t out_col = (size_t)(col_start + (int)col);
        const size_t base = out_col * (size_t)inter_dim + (size_t)(row_base + r0);
        if (row_base + r0 < inter_dim) swiglu_compact[base] = mg_swiglu(g0, u0);
    }
}

#define TBFD_BM      16
#define TBFD_BN      32
#define TBFD_BK      8
#define TBFD_COL_PAD (TBFD_BM + 1)
#define TBFD_ROW_PAD (TBFD_BN + 1)

extern "C" __global__ __launch_bounds__(256, 2)
void moe_grouped_down_bf16_tiled_f32act(
    const float* __restrict__ swiglu_compact,
    const unsigned char* __restrict__ layer_buf,
    const int* __restrict__ tiles,
    unsigned int num_tiles,
    const unsigned long long* __restrict__ down_offsets,
    float* __restrict__ down_compact,
    unsigned int hidden_dim,
    unsigned int inter_dim)
{
    const unsigned int tile_id  = blockIdx.x;
    const unsigned int row_tile = blockIdx.y;
    if (tile_id >= num_tiles) return;
    const unsigned int tid    = threadIdx.x;
    const unsigned int lane8  = tid & 7;
    const unsigned int group8 = tid >> 3;
    const unsigned int col    = tid & 15;
    const unsigned int row_q  = tid >> 4;          // 0..15
    const unsigned int r0     = row_q << 1;        // 2 rows/thread (TBFD_BN=32)
    const unsigned int row_base = row_tile * TBFD_BN;
    if (row_base >= hidden_dim) return;
    const unsigned int num_chunks = inter_dim / 32;
    const size_t row_elems = (size_t)inter_dim;
    __shared__ int   s_expert;
    __shared__ int   s_col_start;
    __shared__ int   s_col_count;
    __shared__ unsigned long long s_down_off;
    __shared__ int   s_src_col[TBFD_BM];
    __shared__ float          s_xf[TBFD_BK][32][TBFD_COL_PAD];
    __shared__ unsigned short s_wh[TBFD_BK][32][TBFD_ROW_PAD];
    if (tid == 0) {
        s_expert    = tiles[tile_id * 4 + 0];
        s_col_start = tiles[tile_id * 4 + 1];
        s_col_count = tiles[tile_id * 4 + 2];
        s_down_off  = down_offsets[(size_t)s_expert];
    }
    __syncthreads();
    const int active_cols = s_col_count;
    const int col_start   = s_col_start;
    if ((int)tid < TBFD_BM) {
        s_src_col[tid] = ((int)tid < active_cols) ? (col_start + (int)tid) : 0;
    }
    __syncthreads();
    const unsigned short* down_base = (const unsigned short*)(layer_buf + s_down_off);
    float a0=0.f,a1=0.f;
    for (unsigned int k0 = 0; k0 < num_chunks; k0 += TBFD_BK) {
        {
            const int m       = group8 & 15;
            const int kk_base = group8 >> 4;
            const bool active = (m < active_cols);
            const int ccol    = s_src_col[m];
            const float* xrow = active ? (swiglu_compact + (size_t)ccol * inter_dim) : swiglu_compact;
            const unsigned int offs = lane8 << 2;
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int kk = kk_base + (t << 1);
                const unsigned int kchunk = k0 + (unsigned int)kk;
                const bool kvalid = (kchunk < num_chunks);
                float v0=0.f,v1=0.f,v2=0.f,v3=0.f;
                if (active && kvalid) {
                    const unsigned int k_elem = kchunk << 5;
                    const float* xp = xrow + k_elem + offs;
                    v0=xp[0]; v1=xp[1]; v2=xp[2]; v3=xp[3];
                }
                s_xf[kk][offs+0][m]=v0; s_xf[kk][offs+1][m]=v1;
                s_xf[kk][offs+2][m]=v2; s_xf[kk][offs+3][m]=v3;
            }
        }
        {
            const int row_local = group8;          // 0..31
            const int kk        = lane8;           // 0..7
            const unsigned int kchunk = k0 + (unsigned int)kk;
            const bool kvalid = (kchunk < num_chunks);
            const unsigned int row_global = row_base + (unsigned int)row_local;
            #pragma unroll
            for (int p=0;p<32;++p) {
                unsigned short wv=0;
                if (row_global < hidden_dim && kvalid) {
                    const unsigned short* wrow = down_base + (size_t)row_global * row_elems + (size_t)(kchunk << 5);
                    wv=wrow[p];
                }
                s_wh[kk][p][row_local]=wv;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int kk=0; kk<TBFD_BK; ++kk) {
            #pragma unroll
            for (int j=0;j<32;++j) {
                const float xv = s_xf[kk][j][col];
                a0 += mg_bf16_to_f32(s_wh[kk][j][r0+0])*xv;
                a1 += mg_bf16_to_f32(s_wh[kk][j][r0+1])*xv;
            }
        }
        __syncthreads();
    }
    if (col < (unsigned int)active_cols) {
        const size_t out_col = (size_t)(col_start + (int)col);
        const size_t base = out_col * (size_t)hidden_dim + (size_t)(row_base + r0);
        if (row_base + r0 + 0 < hidden_dim) down_compact[base+0] = a0;
        if (row_base + r0 + 1 < hidden_dim) down_compact[base+1] = a1;
    }
}

// ===========================================================================
// offline aligned-plane weight REPACK for the routed-FFN down
// projection. Runs ONCE per MoE layer at preload, reading the raw Q8_0 down
// weights from the (already-uploaded) layer blob and writing them into two
// aligned planes:
//   d_q[e][k_blk][row_tile8] : 8 rows x 32 int8 = 256 B  (alignas 128/16)
//   d_s[e][k_blk][row_tile8] : 8 half scales    =  16 B
// Original Q8_0 down layout: down_base[row*row_bytes + kblk*34] = {f16 d}{32 i8}.
// The +2-byte scale prefix makes the q bytes 2-byte-misaligned and interleaves
// scales with quants — defeating vectorized char4 access AND ldmatrix. The
// repack splits them into contiguous aligned planes so the fast-down kernel
// can read a full 32-byte q-block per row as 8 aligned `char4`, and so the W9
// IMMA kernel can `ldmatrix` from a 256-B aligned 8x32 tile. The ORIGINAL blob
// is left byte-untouched (the decode path + all other kernels keep reading it).
//
// Plane storage order:
//   d_q[e][kb][rt8]  flat index = ((e*Kb + kb)*Rt + rt8) where Kb=I/32, Rt=H/8.
//   Each QFrag8x32 = row-major [8][32] int8.
//   Each Scale8    = [8] half.
//
// Launch: grid (Rt, Kb, num_experts), block 256 (8 rows x 32 cols = 256 lanes).
//   blockIdx.x = row_tile8 (0..Rt-1), blockIdx.y = k_blk (0..Kb-1), z = expert.
//   tid: r = tid>>5 (0..7), c = tid&31 (0..31).
// Each thread copies one int8 weight byte; thread (r,0) also copies the scale.
// ===========================================================================
extern "C" __global__ void moe_repack_down_q8_0(
    const unsigned char* __restrict__ layer_buf,            // raw weight blob
    const unsigned long long* __restrict__ down_offsets,    // [num_experts]
    signed char* __restrict__ d_q,                          // [E*Kb*Rt*256] int8
    unsigned short* __restrict__ d_s,                       // [E*Kb*Rt*8] half-bits
    unsigned int hidden_dim,                                // H (= Rt*8)
    unsigned int inter_dim)                                 // I (= Kb*32)
{
    const unsigned int rt8     = blockIdx.x;   // row-tile of 8 rows
    const unsigned int kb      = blockIdx.y;   // k-block
    const unsigned int e       = blockIdx.z;   // expert
    const unsigned int tid     = threadIdx.x;
    const unsigned int r       = tid >> 5;     // 0..7  (row in tile)
    const unsigned int c       = tid & 31;     // 0..31 (q byte in block)

    const unsigned int Kb = inter_dim / MG_Q8_0_BLOCK_SIZE;  // K-blocks (44)
    const unsigned int Rt = hidden_dim / 8;                  // row-tiles (256)
    const unsigned int row_global = rt8 * 8 + r;
    if (row_global >= hidden_dim) return;

    const size_t row_bytes = (size_t)Kb * MG_Q8_0_BLOCK_BYTES;
    const unsigned char* down_base = layer_buf + down_offsets[(size_t)e];
    const unsigned char* wblk = down_base
        + (size_t)row_global * row_bytes
        + (size_t)kb * MG_Q8_0_BLOCK_BYTES;

    // Destination plane offsets.
    const size_t frag_idx = ((size_t)e * Kb + kb) * Rt + rt8; // QFrag/Scale index
    signed char*    dq = d_q + frag_idx * 256 + (size_t)r * 32;
    unsigned short* ds = d_s + frag_idx * 8;

    // Copy one q byte (q bytes start at +2 in the raw block).
    dq[c] = (signed char)wblk[2 + c];
    // Thread c==0 of each row copies the per-block f16 scale.
    if (c == 0) {
        ds[r] = *reinterpret_cast<const unsigned short*>(wblk);
    }
}

// ===========================================================================
// offline aligned-plane weight REPACK for the routed-FFN GATE+UP
// projection (FUSED), prerequisite for the IMMA gate+up kernel. Like the
// down repack but fuses gate and up (always consumed together) into one blob:
//   gu_q[e][k_blk][row_tile8] : GUQFrag = gate[8][32] + up[8][32] = 512 B
//   gu_s[e][k_blk][row_tile8] : GUScaleFrag = gate[8] + up[8] half = 32 B
// where Kb = hidden_dim/32 (K dim is hidden_dim for gate+up), Rt = inter_dim/8.
// Original raw Q8_0 gate/up blocks: {f16 d}{32 i8}, +2-misaligned. The repack
// produces 256-B aligned 8x32 row-major int8 tiles ldmatrix can consume, with
// gate then up contiguous. ORIGINAL blob byte-untouched.
//
// Plane order gate+up fused blob):
//   frag_idx = ((e*Kb + kb)*Rt + rt8). gu_q stride 512, gu_s stride 32(=16 half).
//   Within GUQFrag: bytes [0..255]=gate row-major[8][32], [256..511]=up.
//   Within GUScaleFrag: halves [0..7]=gate scales, [8..15]=up scales.
//
// Launch: grid (Rt, Kb, num_experts), block 256 (8 rows x 32 cols).
//   tid: r = tid>>5 (0..7), c = tid&31. Each thread copies one gate byte + one
//   up byte; thread c==0 copies the gate+up scales for its row.
// ===========================================================================
extern "C" __global__ void moe_repack_gate_up_q8_0(
    const unsigned char* __restrict__ layer_buf,            // raw weight blob
    const unsigned long long* __restrict__ gate_up_offsets, // [num_experts*2] {g,u}
    signed char* __restrict__ gu_q,                         // [E*Kb*Rt*512] int8
    unsigned short* __restrict__ gu_s,                      // [E*Kb*Rt*16] half-bits
    unsigned int hidden_dim,                                // H = Kb*32 (K dim)
    unsigned int inter_dim)                                 // I = Rt*8  (rows)
{
    const unsigned int rt8 = blockIdx.x;   // row-tile of 8 rows
    const unsigned int kb  = blockIdx.y;   // k-block (over hidden_dim)
    const unsigned int e   = blockIdx.z;   // expert
    const unsigned int tid = threadIdx.x;
    const unsigned int r   = tid >> 5;     // 0..7
    const unsigned int c   = tid & 31;     // 0..31

    const unsigned int Kb = hidden_dim / MG_Q8_0_BLOCK_SIZE; // K-blocks (64)
    const unsigned int Rt = inter_dim / 8;                   // row-tiles (64)
    const unsigned int row_global = rt8 * 8 + r;
    if (row_global >= inter_dim) return;

    const size_t row_bytes = (size_t)Kb * MG_Q8_0_BLOCK_BYTES;
    const unsigned char* gate_base = layer_buf + gate_up_offsets[(size_t)e * 2 + 0];
    const unsigned char* up_base   = layer_buf + gate_up_offsets[(size_t)e * 2 + 1];
    const unsigned char* gblk = gate_base + (size_t)row_global * row_bytes
        + (size_t)kb * MG_Q8_0_BLOCK_BYTES;
    const unsigned char* ublk = up_base + (size_t)row_global * row_bytes
        + (size_t)kb * MG_Q8_0_BLOCK_BYTES;

    const size_t frag_idx = ((size_t)e * Kb + kb) * Rt + rt8;
    signed char*    gq = gu_q + frag_idx * 512 + (size_t)r * 32;          // gate plane
    signed char*    uq = gu_q + frag_idx * 512 + 256 + (size_t)r * 32;    // up plane
    unsigned short* gs = gu_s + frag_idx * 16;                            // gate scales
    unsigned short* us = gu_s + frag_idx * 16 + 8;                        // up scales

    gq[c] = (signed char)gblk[2 + c];
    uq[c] = (signed char)ublk[2 + c];
    if (c == 0) {
        gs[r] = *reinterpret_cast<const unsigned short*>(gblk);
        us[r] = *reinterpret_cast<const unsigned short*>(ublk);
    }
}

// ===========================================================================
// numerics-PRESERVING fast down — `down_f32_fast_bn128`. Replaces the naive
// scalar BN128 (which regressed to 759.6 because it had no cp.async, no repacked
// weights, scalar misaligned 34-byte staging, and launch-bounds (256,1)).
//
// This kernel recovers the +10.8% down headroom (int8 down 861 vs f32act 777)
// WITHOUT the int8 activation quant that tips GQ-004, by attacking the *memory
// pipeline* not the math (layout matters more than compute here):
//   - reads the REPACKED aligned down planes (d_q char4-aligned, d_s contiguous)
//   - double-buffered shared (2-stage ping-pong) with cp.async.cg global->shared
//   - BM16/BN128/BK4 (4 q-blocks staged = 128 K-dims per stage)
//   - char4 vectorized weight staging, float4 vectorized activation staging
//   - __launch_bounds__(256, 3) (3 CTAs/SM via 52,224 B dynamic shmem + carveout)
//
// NUMERICS: BIT-for-bit the same per-block math as the f32act path — per 32-block,
// block_sum = Σ_{j=0..31} (float)wq[j]*xf[j] in EXACT j order (float4-grouped as
// j=4p..4p+3, p=0..7, matching __fmaf_rn sequential), then acc += w_scale*block_sum
// sequentially across blocks. Same allowed per-thread reorder vs the per-column
// reference (accepted near-tie class). NO int8 activation quant.
//   Expected: ~845-885 tok/s @ PRISTINE class.
//
// Shared layout (52,224 B dynamic):
//   s_xf4 : [2][4][8][17] float4 = 2*4*8*17*16 = 17,408 B  (raw F32 act, p-major)
//   s_wq4 : [2][4][128][8] char4 = 2*4*128*8*4 =  32,768 B  (repacked q, row-major)
//   s_ws  : [2][4][128]    half  = 2*4*128*2    =   2,048 B
//   total                                          = 52,224 B
// Thread map: col=tid&15 (0..15), row_oct=tid>>4 (0..15), r0=row_oct*8 (0..120).
//   Each thread owns 1 compact column x 8 output rows -> float acc[8].
// Grid: (num_tiles, hidden_dim/128, 1), block 256.
// SHAPE (host-guarded): inter_dim % 32 == 0, hidden_dim % 128 == 0.
// ===========================================================================
#define TD4_BM       16
#define TD4_BN       128
#define TD4_BK       4              // q-blocks staged per K-iteration
#define TD4_COL_PAD  17             // 16 cols + 1 pad (float4 bank stagger)

extern "C" __global__ __launch_bounds__(256, 3)
void moe_grouped_down_q8_0_fast_bn128(
    const float* __restrict__ swiglu_compact,               // [total_cols * inter_dim]
    const signed char* __restrict__ d_q,                    // repacked [E*Kb*Rt*256]
    const unsigned short* __restrict__ d_s,                 // repacked [E*Kb*Rt*8] half
    const int* __restrict__ tiles,                          // [num_tiles*4] {expert,col_start,col_count,pad}
    unsigned int num_tiles,
    float* __restrict__ down_compact,                       // [total_cols * hidden_dim]
    unsigned int hidden_dim,
    unsigned int inter_dim)
{
    const unsigned int tile_id  = blockIdx.x;
    const unsigned int row_tile = blockIdx.y;
    if (tile_id >= num_tiles) return;

    const unsigned int tid      = threadIdx.x;
    const unsigned int lane8    = tid & 7;
    const unsigned int group8   = tid >> 3;     // 0..31
    const unsigned int col      = tid & 15;     // 0..15
    const unsigned int row_oct  = tid >> 4;     // 0..15
    const unsigned int r0       = row_oct << 3; // local row base 0,8,..,120

    const unsigned int row_base = row_tile * TD4_BN;
    if (row_base >= hidden_dim) return;

    const unsigned int Kb = inter_dim / MG_Q8_0_BLOCK_SIZE;  // 44
    const unsigned int Rt = hidden_dim / 8;                  // 256

    // --- Dynamic shared (52,224 B): float4 act + char4 q + half scale ---
    extern __shared__ unsigned char s_dyn4[];
    // s_xf4[stage][kk][p][m]  p=0..7 (float4 group), m=0..16 (col+pad)
    float4*       s_xf4 = (float4*)s_dyn4;                               // 17,408 B
    // s_wq4[stage][kk][row_local][p]  row_local 0..127, p 0..7 (char4)
    char4*        s_wq4 = (char4*)(s_xf4 + 2*4*8*TD4_COL_PAD);           // 32,768 B
    unsigned short* s_ws = (unsigned short*)(s_wq4 + 2*4*128*8);        // 2,048 B
    __shared__ int   s_expert;
    __shared__ int   s_col_start;
    __shared__ int   s_col_count;
    __shared__ int   s_src_col[TD4_BM];
    #define XF4(st,kk,p,m)  s_xf4[(((st)*4 + (kk))*8 + (p))*TD4_COL_PAD + (m)]
    #define WQ4(st,kk,r,p)  s_wq4[(((st)*4 + (kk))*128 + (r))*8 + (p)]
    #define WS4(st,kk,r)    s_ws[((st)*4 + (kk))*128 + (r)]

    if (tid == 0) {
        s_expert    = tiles[tile_id * 4 + 0];
        s_col_start = tiles[tile_id * 4 + 1];
        s_col_count = tiles[tile_id * 4 + 2];
    }
    __syncthreads();

    const int active_cols = s_col_count;
    const int col_start   = s_col_start;
    const int expert      = s_expert;
    if ((int)tid < TD4_BM) {
        s_src_col[tid] = ((int)tid < active_cols) ? (col_start + (int)tid) : 0;
    }
    __syncthreads();

    // Repacked-plane bases for this expert. d_q frag index = ((e*Kb+kb)*Rt+rt8).
    const size_t q_expert_base = (size_t)expert * Kb * Rt * 256;
    const size_t s_expert_base = (size_t)expert * Kb * Rt * 8;
    const unsigned int rt8_base = row_base / 8;   // first row-tile8 of this CTA (0..Rt-1), 16 tiles span BN128

    float acc[8];
    #pragma unroll
    for (int i=0;i<8;++i) acc[i]=0.f;

    // ---- Staging helper: load TD4_BK q-blocks for stage `st`, k base `k0`. ----
    // Activation: float4 groups. 64 tasks/stage (4 kk x 16 cols) over 32 8-thread
    // subgroups -> 2 rounds. lane8 p=0..7 each loads one float4 (j=4p..4p+3).
    // Weight: 128 rows x 4 kk char4-packed. 256 threads each copy one char4 row-q
    // segment via cp.async; first pass covers (row 0..127, p = tid layout).
    #define STAGE_LOAD(st, k0)                                                       \
    {                                                                                \
        /* activation: 2 rounds, subgroup=group8 (0..31), lane8 p=0..7 */            \
        _Pragma("unroll")                                                            \
        for (int round=0; round<2; ++round) {                                        \
            const int task = (int)group8 + round*32;  /* 0..63 */                    \
            const int kk   = task >> 4;               /* 0..3 */                     \
            const int m    = task & 15;               /* col 0..15 */                \
            const unsigned int kblk = (k0) + (unsigned int)kk;                       \
            const bool active = (m < active_cols) && (kblk < Kb);                    \
            const int ccol = s_src_col[m];                                           \
            float4 xv = make_float4(0.f,0.f,0.f,0.f);                                \
            if (active) {                                                            \
                const float* xp = swiglu_compact + (size_t)ccol*(size_t)inter_dim   \
                    + (size_t)kblk*32 + (size_t)lane8*4;                             \
                xv = *reinterpret_cast<const float4*>(xp);                           \
            }                                                                        \
            XF4((st), kk, (int)lane8, m) = xv;                                       \
        }                                                                            \
        /* weight: 128 rows x 4 kk. group8=row-in-pass (0..31), 4 passes (+32),     \
           lane8 = which kk-block's char4-octet? No: stage char4 directly. Each    \
           thread copies one full 32-byte q-block (8 char4) per (row,kk). We map:  \
           pass p (0..3) -> row_local = group8 + p*32; lane8 -> kk-and-half. */     \
        _Pragma("unroll")                                                            \
        for (int wpass=0; wpass<16; ++wpass) {                                       \
            /* 16 passes cover 128 rows x 4 kk = 512 (row,kk) tasks / 256 thr / ... \
               simpler: task = tid + wpass*256, 0..4095 over (128 row * 4 kk * 8 p)*/\
            const int task = (int)tid + wpass*256;  /* 0..4095 */                    \
            const int p    = task & 7;              /* char4 index 0..7 */           \
            const int rk   = task >> 3;             /* 0..511 */                     \
            const int kk   = rk & 3;                /* 0..3 */                       \
            const int rl   = rk >> 2;               /* row_local 0..127 */           \
            const unsigned int kblk = (k0) + (unsigned int)kk;                       \
            const unsigned int rg   = row_base + (unsigned int)rl;                   \
            char4 wv = make_char4(0,0,0,0);                                          \
            if (rg < hidden_dim && kblk < Kb) {                                      \
                const unsigned int rt8 = rg >> 3;                                    \
                const unsigned int rsub = rg & 7;                                    \
                const size_t fidx = ((size_t)kblk*Rt + rt8);                         \
                const signed char* qp = d_q + q_expert_base + fidx*256              \
                    + (size_t)rsub*32 + (size_t)p*4;                                 \
                wv = *reinterpret_cast<const char4*>(qp);                            \
                if (p == 0) {                                                        \
                    WS4((st), kk, rl) = (d_s + s_expert_base + fidx*8)[rsub];        \
                }                                                                    \
            } else if (p == 0) {                                                     \
                WS4((st), kk, rl) = 0;                                               \
            }                                                                        \
            WQ4((st), kk, rl, p) = wv;                                               \
        }                                                                            \
    }

    const int n_kiter = (int)((Kb + TD4_BK - 1) / TD4_BK);
    // Prologue: stage 0.
    STAGE_LOAD(0, 0u);
    __syncthreads();

    for (int it=0; it<n_kiter; ++it) {
        const int cur = it & 1;
        const int nxt = (it + 1) & 1;
        const unsigned int k0 = (unsigned int)(it * TD4_BK);
        // Prefetch next stage while computing current.
        if (it + 1 < n_kiter) {
            STAGE_LOAD(nxt, (unsigned int)((it+1)*TD4_BK));
        }
        // ---- Compute current stage: exact per-block F32 order. ----
        #pragma unroll
        for (int kk=0; kk<TD4_BK; ++kk) {
            const unsigned int kblk = k0 + (unsigned int)kk;
            if (kblk >= Kb) continue;
            float sum[8];
            #pragma unroll
            for (int rr=0; rr<8; ++rr) sum[rr]=0.f;
            #pragma unroll
            for (int p=0; p<8; ++p) {            // j = 4p..4p+3
                float4 x = XF4(cur, kk, p, col);
                #pragma unroll
                for (int rr=0; rr<8; ++rr) {
                    char4 w = WQ4(cur, kk, r0+rr, p);
                    sum[rr] = __fmaf_rn((float)(int)w.x, x.x, sum[rr]);
                    sum[rr] = __fmaf_rn((float)(int)w.y, x.y, sum[rr]);
                    sum[rr] = __fmaf_rn((float)(int)w.z, x.z, sum[rr]);
                    sum[rr] = __fmaf_rn((float)(int)w.w, x.w, sum[rr]);
                }
            }
            #pragma unroll
            for (int rr=0; rr<8; ++rr) {
                float wsc = mg_f16_to_f32(WS4(cur, kk, r0+rr));
                acc[rr] = __fmaf_rn(wsc, sum[rr], acc[rr]);
            }
        }
        __syncthreads();
    }

    // ---- Store (compact-col major: [col*hidden_dim + row]). ----
    if (col < (unsigned int)active_cols) {
        const size_t out_col = (size_t)(col_start + (int)col);
        const size_t base = out_col * (size_t)hidden_dim + (size_t)(row_base + r0);
        #pragma unroll
        for (int rr=0; rr<8; ++rr) {
            const unsigned int rg = row_base + r0 + (unsigned int)rr;
            if (rg < hidden_dim) down_compact[base+rr] = acc[rr];
        }
    }
    #undef XF4
    #undef WQ4
    #undef WS4
    #undef STAGE_LOAD
    (void)rt8_base;
}

// ===========================================================================
// int8 TENSOR-CORE (IMMA) grouped gate+up+SwiGLU GEMM.
//   `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`, ldmatrix-fed, BM16/BN64/BK4.
//
// Replaces the dp4a inner product of `moe_grouped_gate_up_swiglu_q8_0_tiled`
// with A100 int8 tensor cores (~624 TOPS vs dp4a ~39 TOPS = 16× peak; LC
// realized 1.56-2.12× over tiled-dp4a). Analysis indicates this is THE structural
// lever vs llama.cpp's mul_mat_id.
//
// FIDELITY (load-bearing, BIT-IDENTICAL to dp4a): one mma.sync m16n8k32 covers
// exactly K=32 = ONE Q8_0 block. The s8*s8->s32 accumulate is EXACT integer
// (max |dot| = 32*127*127 = 516,128 < 2^24, so int32->float is exact too). We
// apply the per-block f32 scale (x_scale * w_scale) per block, EXACTLY the same
// per-32-block int32-sum-then-f32-scale contract as dp4a. Only the f32 add-order
// differs (the same accepted near-tie class as the tiled path). NO cross-block tensor accum.
//
// CTA: one expert, one m_tile (16 compact cols), one n_tile (64 output rows).
//   8 warps. warp w owns output rows [n0+8w .. n0+8w+7]. Per kk: 2 MMA frags
//   (gate 16x8, up 16x8). On-the-fly int8 x quant into shared (no global xq buf).
//   Reads REPACKED gu_q/gu_s planes (ldmatrix-friendly aligned 8x32 tiles).
//
// Activation A (16 cols x 32 k) staged int8 in shared, fed to MMA via ldmatrix.x4.
// Weight  B (8 rows  x 32 k) staged int8 in shared (from gu_q), fed via ldmatrix
//   .x2.trans. Per-block int32 D -> f32 (x_scale[m]*w_scale[n]) -> f32 acc.
//
// Shared (static, 47,616 B — fits 48 KB, 3 CTAs/SM possible):
//   s_xq  : [2][4][16][32] int8 =  4,096 B  (A tiles, on-the-fly quant)
//   s_xs  : [2][4][16]     f32  =    512 B  (x per-block scale)
//   s_wq  : [2][4][8][2][8][32] int8 = 32,768 B (B tiles gate/up from gu_q)
//   s_ws  : [2][4][8][2][8] half = 2,048 B  (gate/up per-block scale)
//   s_gepi: [16][64] f32 = 4,096 B  (gate epilogue scratch)
//   s_uepi: [16][64] f32 = 4,096 B  (up epilogue scratch)
//   total = 47,616 B (dynamic; opt-in for 3 CTAs/SM with carveout).
//
// SHAPE (host-guarded): hidden_dim % 32 == 0 (whole K blocks), inter_dim % 64 == 0.
// Grid: (num_tiles, inter_dim/64, 1), block 256.
// ===========================================================================
#define IMG_BM 16
#define IMG_BN 64
#define IMG_BK 4    // q-blocks staged per K-iter (=128 K)

// A/B fragment loaders use the llama.cpp-validated NVIDIA Ampere int8 tile maps
// (mma.cuh tile<I,J,int,I_MAJOR>, the `#else` Turing/Ampere branch) — load_generic
// via explicit get_i/get_j, NOT ldmatrix (LC uses ldmatrix only for half tiles;
// int8 A/B are loaded with ggml_cuda_memcpy through these maps). Each `int` packs
// 4 int8; the shared tile is row-major int8[rows][32], so element (i, j_int) is
// the int32 at byte offset i*32 + j_int*4.
//
//  BIT-VALIDATED fragment maps (standalone m16n8k32 validator, /tmp/imma_search:
//  aMode=0,bMode=1 = 0/128 mismatches vs CPU dot). Two design rounds proposed
//  the C-map and a half-swizzle for A; the validator proved BOTH A and B were
//  wrong and pinned the exact maps below.
//  A MULTIPLICAND (16x32, .row): row = lane/4 + 8*(l&1); k_int = (lane&3) + 4*(l>>1)
//    a0=A[g][k], a1=A[g+8][k], a2=A[g][k+4], a3=A[g+8][k+4]  (g=lane/4, k=lane&3).
//  B MULTIPLICAND (8x32, .col):  n = lane/4 (0..7); k_int = (lane&3) + 4*l
//  C ACCUMULATOR (16x8 s32):     m = (i>>1)*8 + lane/4; n = (lane&3)*2 + (i&1).
// Load A (16x32 int8 row-major [m][k]) into 4 int regs per lane.
__device__ __forceinline__ void mg_load_A_s8(unsigned (&a)[4], const signed char* tile32, unsigned lane) {
    #pragma unroll
    for (int l=0;l<4;++l) {
        const int i = (int)(lane>>2) + 8*(l&1);     // row: g, g+8
        const int j = (int)(lane&3) + 4*(l>>1);     // k_int: k, k+4
        a[l] = *reinterpret_cast<const unsigned*>(tile32 + (size_t)i*32 + (size_t)j*4);
    }
}
// Load B (8 N-rows x 32 K int8, staged row-major [n][k]) into 2 int regs/lane.
// BIT-VALIDATED (validator bMode=1): n = lane/4 (0..7); k_int = (lane&3) + 4*l.
// b0=B[n][k], b1=B[n][k+4]  (n=lane/4, k=lane&3). Reads tile[n][k_int]=n*32+k_int*4.
__device__ __forceinline__ void mg_load_B_s8(unsigned (&b)[2], const signed char* tile32, unsigned lane) {
    #pragma unroll
    for (int l=0;l<2;++l) {
        const int n     = (int)(lane >> 2);            // 0..7 = output row in warp
        const int k_int = (int)(lane & 3) + 4*l;       // 0..7 = K/4 index
        b[l] = *reinterpret_cast<const unsigned*>(tile32 + (size_t)n*32 + (size_t)k_int*4);
    }
}
// mma.sync m16n8k32 s8.s8.s32. A = 4 b32 (16x32 s8), B = 2 b32 (8x32 s8),
// D += A*B, D = 4 s32 (16x8 acc). Accumulate-in-place into d[0..3].
__device__ __forceinline__ void mg_mma_m16n8k32(int (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    (void)a; (void)b; d[0]=d[1]=d[2]=d[3]=0;
#endif
}

extern "C" __global__ __launch_bounds__(256, 3)
void moe_grouped_gate_up_swiglu_q8_0_imma(
    const float* __restrict__ normed,                       // [batch, hidden_dim]
    const signed char* __restrict__ gu_q,                   // repacked [E*Kb*Rt*512]
    const unsigned short* __restrict__ gu_s,                // repacked [E*Kb*Rt*16] half
    const int* __restrict__ col_src_tok,                    // [total_cols]
    const int* __restrict__ tiles,                          // [num_tiles*4]
    unsigned int num_tiles,
    float* __restrict__ swiglu_compact,                     // [total_cols * inter_dim]
    unsigned int hidden_dim,
    unsigned int inter_dim)
{
    const unsigned int tile_id  = blockIdx.x;
    const unsigned int row_tile = blockIdx.y;
    if (tile_id >= num_tiles) return;

    const unsigned int tid   = threadIdx.x;
    const unsigned int warp  = tid >> 5;     // 0..7
    const unsigned int lane  = tid & 31;
    const unsigned int row_base = row_tile * IMG_BN;
    if (row_base >= inter_dim) return;

    const unsigned int Kb = hidden_dim / 32;  // K-blocks (64)
    const unsigned int Rt = inter_dim / 8;    // row-tiles of 8

    extern __shared__ unsigned char s_immagu[];
    signed char* s_xq = (signed char*)s_immagu;                       // [2][4][16][32]
    float*       s_xs = (float*)(s_xq + 2*4*16*32);                   // [2][4][16]
    signed char* s_wq = (signed char*)(s_xs + 2*4*16);               // [2][4][8][2][8][32]
    unsigned short* s_ws = (unsigned short*)(s_wq + 2*4*8*2*8*32);    // [2][4][8][2][8]
    float* s_gepi = (float*)(s_ws + 2*4*8*2*8);                       // [16][64]
    float* s_uepi = (float*)(s_gepi + 16*64);                         // [16][64]
    __shared__ int s_expert, s_col_start, s_col_count;
    __shared__ int s_src_tok[IMG_BM];
    #define XQ(st,kk,m)      (s_xq + (((st)*4 + (kk))*16 + (m))*32)
    #define XS(st,kk,m)      s_xs[((st)*4 + (kk))*16 + (m)]
    // weight: warp w -> n-group g = warp (8 rows); mat 0=gate,1=up.
    #define WQ(st,kk,g,mat)  (s_wq + (((((st)*4 + (kk))*8 + (g))*2 + (mat))*8)*32)
    #define WS(st,kk,g,mat,rr) s_ws[((((st)*4 + (kk))*8 + (g))*2 + (mat))*8 + (rr)]

    if (tid == 0) {
        s_expert    = tiles[tile_id*4+0];
        s_col_start = tiles[tile_id*4+1];
        s_col_count = tiles[tile_id*4+2];
    }
    __syncthreads();
    const int active_cols = s_col_count;
    const int col_start   = s_col_start;
    const int expert      = s_expert;
    if ((int)tid < IMG_BM) {
        s_src_tok[tid] = ((int)tid < active_cols) ? col_src_tok[col_start + (int)tid] : 0;
    }
    __syncthreads();

    const size_t gu_q_eb = (size_t)expert * Kb * Rt * 512;
    const size_t gu_s_eb = (size_t)expert * Kb * Rt * 16;

    // Accumulators: warp owns 8 rows (n0..n0+7) x 16 cols. We keep f32 epilogue in
    // shared. Per-thread holds gate/up int32 acc for its MMA fragment lanes.
    // d_g/d_u: 4 s32 each, mapped to (m in 0..15, n in n0..n0+7) per MMA layout.
    // We re-zero per kk-block and immediately scale+accumulate into s_gepi/s_uepi.

    // Pre-zero epilogue scratch.
    for (unsigned int i = tid; i < 16*64; i += 256) { s_gepi[i]=0.f; s_uepi[i]=0.f; }
    __syncthreads();

    // ---- Staging macro: load IMG_BK q-blocks for stage st at k base k0. ----
    #define STAGE(st, k0)                                                            \
    {                                                                                \
        /* activation: on-the-fly int8 quant. 8-thread subgroups, group8 0..31. */   \
        const unsigned int lane8  = tid & 7;                                         \
        const unsigned int group8 = tid >> 3;                                        \
        _Pragma("unroll")                                                            \
        for (int t=0;t<2;++t) {                                                       \
            const int task = (int)group8 + t*32;  /* 0..63 = 4kk x 16m */            \
            const int kk = task >> 4; const int m = task & 15;                       \
            const unsigned int kblk = (k0) + (unsigned int)kk;                        \
            const bool active = (m < active_cols) && (kblk < Kb);                     \
            const int tok = s_src_tok[m];                                             \
            float v0=0,v1=0,v2=0,v3=0;                                                \
            if (active) {                                                             \
                const float* xp = normed + (size_t)tok*(size_t)hidden_dim            \
                    + (size_t)kblk*32 + (size_t)lane8*4;                              \
                v0=xp[0];v1=xp[1];v2=xp[2];v3=xp[3];                                  \
            }                                                                         \
            float vmax=fabsf(v0),a;                                                   \
            a=fabsf(v1);if(a>vmax)vmax=a; a=fabsf(v2);if(a>vmax)vmax=a;               \
            a=fabsf(v3);if(a>vmax)vmax=a;                                             \
            float o;                                                                  \
            o=__shfl_xor_sync(0xffffffffu,vmax,4,8);if(o>vmax)vmax=o;                 \
            o=__shfl_xor_sync(0xffffffffu,vmax,2,8);if(o>vmax)vmax=o;                 \
            o=__shfl_xor_sync(0xffffffffu,vmax,1,8);if(o>vmax)vmax=o;                 \
            const float amax=__shfl_sync(0xffffffffu,vmax,0,8);                       \
            const float sc=(amax>0.f)?(amax*(1.0f/127.0f)):0.f;                       \
            const float inv=(amax>0.f)?(127.0f/amax):0.f;                            \
            signed char* dst = XQ((st),kk,m) + lane8*4;                               \
            dst[0]=(signed char)(inv>0.f?__float2int_rn(v0*inv):0);                   \
            dst[1]=(signed char)(inv>0.f?__float2int_rn(v1*inv):0);                   \
            dst[2]=(signed char)(inv>0.f?__float2int_rn(v2*inv):0);                   \
            dst[3]=(signed char)(inv>0.f?__float2int_rn(v3*inv):0);                   \
            if (lane8==0) XS((st),kk,m)=sc;                                           \
        }                                                                            \
        /* weight: copy gate+up repacked tiles. 4kk x 8g x 8rows x 8p = 2048 (kk,g,  \
           rr,p) tasks; x2 mats inside = 16 char4/thr = 64 B/thr. (consult: w<16 ran \
           4096 tasks = each cell written TWICE = same-addr race. w<8 = 2048 unique.)*/\
        _Pragma("unroll")                                                            \
        for (int w=0; w<8; ++w) {                                                     \
            const int task = (int)tid + w*256;  /* 0..2047 (kk,g,rr,p) tasks */      \
            const int p  = task & 7;            /* char4 in 32-byte row */           \
            const int rk = task >> 3;           /* 0..511 */                         \
            const int rr = rk & 7;              /* row in 8 0..7 */                   \
            const int g  = (rk >> 3) & 7;       /* n-group 0..7 */                   \
            const int kk = (rk >> 6) & 3;       /* 0..3 */                           \
            const unsigned int kblk = (k0) + (unsigned int)kk;                        \
            const unsigned int n_row = row_base + (unsigned int)(g*8 + rr);           \
            for (int mat=0; mat<2; ++mat) {                                           \
                char4 wv = make_char4(0,0,0,0);                                       \
                if (n_row < inter_dim && kblk < Kb) {                                 \
                    const unsigned int rt8 = n_row >> 3;                              \
                    const unsigned int rsub= n_row & 7;                              \
                    const size_t fidx = ((size_t)kblk*Rt + rt8);                      \
                    const signed char* qp = gu_q + gu_q_eb + fidx*512                 \
                        + (size_t)mat*256 + (size_t)rsub*32 + (size_t)p*4;            \
                    wv = *reinterpret_cast<const char4*>(qp);                         \
                    if (p==0) WS((st),kk,g,mat,rr) =                                  \
                        (gu_s + gu_s_eb + fidx*16 + mat*8)[rsub];                     \
                } else if (p==0) WS((st),kk,g,mat,rr)=0;                              \
                *reinterpret_cast<char4*>(WQ((st),kk,g,mat) + rr*32 + p*4) = wv;      \
            }                                                                         \
        }                                                                            \
    }

    const int n_kiter = (int)((Kb + IMG_BK - 1) / IMG_BK);
    STAGE(0, 0u);
    __syncthreads();

    const unsigned int g_warp = warp; // n-group this warp owns (rows n0..n0+7)

    for (int it=0; it<n_kiter; ++it) {
        const int cur = it & 1;
        const int nxt = (it+1) & 1;
        const unsigned int k0 = (unsigned int)(it*IMG_BK);
        if (it+1 < n_kiter) { STAGE(nxt, (unsigned int)((it+1)*IMG_BK)); }

        #pragma unroll
        for (int kk=0; kk<IMG_BK; ++kk) {
            const unsigned int kblk = k0 + (unsigned int)kk;
            if (kblk >= Kb) continue;
            // Load A (16x32 s8) + gate/up B (8x32 s8) fragments via the LC int8
            // tile maps (load_generic, get_i/get_j). A is shared across gate+up.
            unsigned a[4]; mg_load_A_s8(a, XQ(cur,kk,0), lane);
            unsigned bg[2]; mg_load_B_s8(bg, WQ(cur,kk,g_warp,0), lane);
            unsigned bu[2]; mg_load_B_s8(bu, WQ(cur,kk,g_warp,1), lane);
            int dg[4]={0,0,0,0}, du[4]={0,0,0,0};
            mg_mma_m16n8k32(dg, a, bg);
            mg_mma_m16n8k32(du, a, bu);
            // C = tile<16,8,int> accumulator (same map as A): element l ->
            //   m = (l/2)*8 + lane/4 ; n = (lane%4)*2 + (l%2)   (n 0..7 in group).
            #pragma unroll
            for (int i=0;i<4;++i) {
                const int m  = (i>>1)*8 + (int)(lane>>2);
                const int nn = (int)((lane&3)*2) + (i&1);          // 0..7 in n-group
                const int n_local = (int)g_warp*8 + nn;            // 0..63
                if (m < 16) {
                    const float xs = XS(cur,kk,m);
                    const float gsc = mg_f16_to_f32(WS(cur,kk,g_warp,0,nn));
                    const float usc = mg_f16_to_f32(WS(cur,kk,g_warp,1,nn));
                    s_gepi[m*64 + n_local] = __fmaf_rn(xs*gsc, (float)dg[i], s_gepi[m*64 + n_local]);
                    s_uepi[m*64 + n_local] = __fmaf_rn(xs*usc, (float)du[i], s_uepi[m*64 + n_local]);
                }
            }
        }
        __syncthreads();
    }

    // ---- SwiGLU + store. Thread map: col = tid&15, 4 rows per (tid>>4). ----
    {
        const unsigned int col = tid & 15;
        const unsigned int rq  = tid >> 4;     // 0..15
        if (col < (unsigned int)active_cols) {
            const size_t out_col = (size_t)(col_start + (int)col);
            #pragma unroll
            for (int t=0;t<4;++t) {
                const unsigned int n_local = rq*4 + t;  // 0..63
                const unsigned int row_global = row_base + n_local;
                if (row_global < inter_dim) {
                    const float g = s_gepi[col*64 + n_local];
                    const float u = s_uepi[col*64 + n_local];
                    swiglu_compact[out_col*(size_t)inter_dim + row_global] = mg_swiglu(g, u);
                }
            }
        }
    }
    #undef XQ
    #undef XS
    #undef WQ
    #undef WS
    #undef STAGE
}

// ===========================================================================
// register-resident-C + wide-M IMMA gate+up. Design analysis// Finding: the IMMA -2.4% was per-CTA
// inefficiency (shared-epilogue RMW + 16-col M half-fill + per-col-tile weight
// re-stream), NOT SM starvation. Fix = LC's register-C + larger tiles WITHOUT
// LC's split-K/fixup (which our per-32-block-scale contract forbids):
//   - N = 128 output rows / CTA (8 warps, warp w owns rows [16w..16w+15]).
//   - M = MG*16 compact cols (MG in {1,2,3,4}), bucketed; tail masked.
//   - per (rg in 0..1)(mg in 0..MG-1) register fp32 C accumulators (gate+up),
//     4 regs each. NO shared epilogue.
//   - per k32: mma.sync.m16n8k32.s8.s8.s32 -> transient s32 -> *(xs*wscale)
//     per block -> fmaf into reg accumulators. EXACT per-32-block contract.
//   - reads PRE-QUANTIZED K-major activation (xq_q/xq_d, built once per layer by
//     moe_prequant_x_q8) so the GEMM never re-quantizes across the 4 row-tiles.
//   - reads the SAME repacked GUQFrag planes (gu_q/gu_s) as the IMMA path.
// Standalone tile validator
// bit-validated 13/13 cases (cols 1..64, MG 1..4): maxrel 2.5e-7, zero tail leak.
//
// Activation prequant. Replicates the IMMA on-the-fly quant rule EXACTLY
// (8-lane subgroup amax over 4 contiguous, sc=amax/127, q=rnd(v*127/amax)), so
// prequant+W10 is the SAME numeric class as the IMMA path IMMA and dp4a (per-32-block).
// Layout: xq_q[Kb][total_cols][32] int8 (K-major: kblk outer, col mid, k inner),
//         xq_d[Kb][total_cols] fp32 scale. One block = one compact column; 8
// warps, warp w handles kblk = w, w+8, ... (Kb total).
// ===========================================================================
extern "C" __global__ void moe_prequant_x_q8(
    const float* __restrict__ normed,        // [batch, hidden_dim]
    const int*   __restrict__ col_src_tok,    // [total_cols]
    signed char* __restrict__ xq_q,           // [Kb * total_cols * 32]
    float*       __restrict__ xq_d,           // [Kb * total_cols]
    unsigned int hidden_dim,
    unsigned int total_cols)
{
    const unsigned int col = blockIdx.x;
    if (col >= total_cols) return;
    const unsigned int tid   = threadIdx.x;
    const unsigned int warp  = tid >> 5;      // 0..7
    const unsigned int lane  = tid & 31;
    const unsigned int lane8 = lane & 7;      // position in 8-lane subgroup
    const unsigned int sg    = lane >> 3;     // 0..3 subgroup within warp
    const unsigned int Kb    = hidden_dim / 32;
    const int tok = col_src_tok[col];
    const float* xrow = normed + (size_t)tok * (size_t)hidden_dim;
    // Each warp covers 4 kblks per pass (one per 8-lane subgroup); stride 8 warps*?.
    // Simpler: warp w owns kblk = w, w+8, w+16, ...; the 4 subgroups each take a
    // DIFFERENT kblk within that stride window so all 32 lanes are productive.
    for (unsigned int kbase = warp * 4u; kbase < Kb; kbase += 32u) {
        const unsigned int kblk = kbase + sg;       // this lane's k-block
        if (kblk >= Kb) continue;
        const float* xp = xrow + (size_t)kblk * 32 + (size_t)lane8 * 4;
        float v0 = xp[0], v1 = xp[1], v2 = xp[2], v3 = xp[3];
        float vmax = fabsf(v0), a;
        a = fabsf(v1); if (a > vmax) vmax = a;
        a = fabsf(v2); if (a > vmax) vmax = a;
        a = fabsf(v3); if (a > vmax) vmax = a;
        // reduce amax within the 8-lane subgroup
        float o;
        o = __shfl_xor_sync(0xffffffffu, vmax, 4, 8); if (o > vmax) vmax = o;
        o = __shfl_xor_sync(0xffffffffu, vmax, 2, 8); if (o > vmax) vmax = o;
        o = __shfl_xor_sync(0xffffffffu, vmax, 1, 8); if (o > vmax) vmax = o;
        // width-8 broadcast (srcLane 0 within the 8-lane subgroup) — EXACTLY the W9
        // IMMA STAGE rule `__shfl_sync(..., vmax, 0, 8)`, bit-identical activation quant.
        const float amax = __shfl_sync(0xffffffffu, vmax, 0, 8);
        const float sc  = (amax > 0.f) ? (amax * (1.0f / 127.0f)) : 0.f;
        const float inv = (amax > 0.f) ? (127.0f / amax) : 0.f;
        signed char* dst = xq_q + ((size_t)kblk * total_cols + col) * 32 + lane8 * 4;
        dst[0] = (signed char)(inv > 0.f ? __float2int_rn(v0 * inv) : 0);
        dst[1] = (signed char)(inv > 0.f ? __float2int_rn(v1 * inv) : 0);
        dst[2] = (signed char)(inv > 0.f ? __float2int_rn(v2 * inv) : 0);
        dst[3] = (signed char)(inv > 0.f ? __float2int_rn(v3 * inv) : 0);
        if (lane8 == 0) xq_d[(size_t)kblk * total_cols + col] = sc;
    }
}

// A/B fragment loaders (the bit-validated maps; aMode0/bMode1).
__device__ __forceinline__ void mg_w10_load_A(unsigned (&a)[4], const signed char* tile32, unsigned lane) {
    #pragma unroll
    for (int l=0;l<4;++l) { const int i=(int)(lane>>2)+8*(l&1); const int j=(int)(lane&3)+4*(l>>1);
        a[l]=*reinterpret_cast<const unsigned*>(tile32+(size_t)i*32+(size_t)j*4); }
}
__device__ __forceinline__ void mg_w10_load_B(unsigned (&b)[2], const signed char* tile32, unsigned lane) {
    #pragma unroll
    for (int l=0;l<2;++l) { const int n=(int)(lane>>2); const int j=(int)(lane&3)+4*l;
        b[l]=*reinterpret_cast<const unsigned*>(tile32+(size_t)n*32+(size_t)j*4); }
}

// Templated W10 gate+up. MG in {1,2,3,4}. Grid: (num_tiles_MG, row128_count, 1),
// block 256. tiles[] = {col0, expert, row128, cols_valid} per entry (i32 x4).
template<int MG>
__device__ __forceinline__ void moe_w10_gate_up_body(
    const signed char* __restrict__ xq_q,    // [Kb][total_cols][32]
    const float*       __restrict__ xq_d,     // [Kb][total_cols]
    const signed char* __restrict__ gu_q,     // [E*Kb*Rt*512]
    const unsigned short* __restrict__ gu_s,  // [E*Kb*Rt*16]
    const int* __restrict__ tiles,            // [num_tiles*4] {col0,expert,row128,cols_valid}
    unsigned int num_tiles, int total_cols,
    float* __restrict__ swiglu_compact,       // [total_cols * inter_dim]
    unsigned int hidden_dim, unsigned int inter_dim)
{
    const unsigned int t = blockIdx.x;
    if (t >= num_tiles) return;
    const unsigned int tid=threadIdx.x, warp=tid>>5, lane=tid&31;
    const int col0       = tiles[t*4+0];
    const int expert     = tiles[t*4+1];
    const int row128     = tiles[t*4+2];
    const int cols_valid = tiles[t*4+3];
    const int M = MG*16;
    const unsigned int Kb = hidden_dim / 32;
    const unsigned int Rt = inter_dim / 8;

    float accg[2][MG][4]; float accu[2][MG][4];
    #pragma unroll
    for (int rg=0;rg<2;++rg) for (int mg=0;mg<MG;++mg) for (int i=0;i<4;++i){accg[rg][mg][i]=0.f;accu[rg][mg][i]=0.f;}

    // shared A staging for one k32 block (double-buffered).
    __shared__ signed char sA[2][MG][16][32];
    __shared__ float       sAd[2][MG][16];
    const int row8_base = row128*16;   // 16 row8 = 128 rows per CTA

    // stage helper: load k-block kb of A into buffer buf.
    #define W10_STAGE_A(buf, kb)                                                       \
    {                                                                                  \
        for (unsigned int q = tid; q < (unsigned int)(M*32); q += 256) {               \
            const int col=q>>5, k=q&31; const int mg=col>>4, mrow=col&15;              \
            signed char v=0;                                                           \
            if (col < cols_valid) v = xq_q[((size_t)(kb)*total_cols + (col0+col))*32 + k]; \
            sA[buf][mg][mrow][k]=v;                                                     \
            if (k==0) sAd[buf][mg][mrow] = (col<cols_valid)?xq_d[(size_t)(kb)*total_cols + (col0+col)]:0.f; \
        }                                                                              \
    }

    W10_STAGE_A(0, 0u);
    __syncthreads();
    for (unsigned int kb=0; kb<Kb; ++kb) {
        const int cur = (int)(kb & 1), nxt = (int)((kb+1) & 1);
        if (kb+1 < Kb) { W10_STAGE_A(nxt, kb+1); }
        const int rt8_0 = row8_base + (int)warp*2;
        const size_t fidx = ((size_t)expert*Kb + kb)*Rt;
        #pragma unroll
        for (int rg=0; rg<2; ++rg) {
            const int rt8 = rt8_0 + rg;
            const signed char* gbase = gu_q + (fidx+rt8)*512;
            const signed char* ubase = gbase + 256;
            const unsigned short* gsc = gu_s + (fidx+rt8)*16;
            unsigned bg[2], bu[2];
            mg_w10_load_B(bg, gbase, lane);
            mg_w10_load_B(bu, ubase, lane);
            #pragma unroll
            for (int mg=0; mg<MG; ++mg) {
                unsigned a[4]; mg_w10_load_A(a, &sA[cur][mg][0][0], lane);
                int dg[4]={0,0,0,0}, du[4]={0,0,0,0};
                mg_mma_m16n8k32(dg,a,bg); mg_mma_m16n8k32(du,a,bu);
                #pragma unroll
                for (int i=0;i<4;++i) {
                    const int m=(i>>1)*8 + (int)(lane>>2);     // 0..15 col within m16
                    const int nn=(int)((lane&3)*2) + (i&1);    // 0..7 row within n8
                    const float xs = sAd[cur][mg][m];
                    const float gs = mg_f16_to_f32(gsc[nn]);
                    const float us = mg_f16_to_f32(gsc[8+nn]);
                    accg[rg][mg][i] = __fmaf_rn(xs*gs, (float)dg[i], accg[rg][mg][i]);
                    accu[rg][mg][i] = __fmaf_rn(xs*us, (float)du[i], accu[rg][mg][i]);
                }
            }
        }
        __syncthreads();
    }

    // epilogue: SwiGLU + store.
    #pragma unroll
    for (int rg=0; rg<2; ++rg) {
        const int rt8 = (int)warp*2 + rg;          // local row8 within CTA (0..15)
        #pragma unroll
        for (int mg=0; mg<MG; ++mg) {
            #pragma unroll
            for (int i=0;i<4;++i) {
                const int m=(i>>1)*8 + (int)(lane>>2);
                const int nn=(int)((lane&3)*2) + (i&1);
                const int col = mg*16 + m;
                const int row_local = rt8*8 + nn;
                const int row_global = row128*128 + row_local;
                if (col < cols_valid && row_global < (int)inter_dim) {
                    swiglu_compact[(size_t)(col0+col)*inter_dim + row_global] =
                        mg_swiglu(accg[rg][mg][i], accu[rg][mg][i]);
                }
            }
        }
    }
    #undef W10_STAGE_A
}

extern "C" __global__ __launch_bounds__(256, 3)
void moe_grouped_gate_up_swiglu_q8_0_w10_mg1(
    const signed char* xq_q, const float* xq_d, const signed char* gu_q,
    const unsigned short* gu_s, const int* tiles, unsigned int num_tiles, int total_cols,
    float* swiglu_compact, unsigned int hidden_dim, unsigned int inter_dim) {
    moe_w10_gate_up_body<1>(xq_q,xq_d,gu_q,gu_s,tiles,num_tiles,total_cols,swiglu_compact,hidden_dim,inter_dim);
}
extern "C" __global__ __launch_bounds__(256, 3)
void moe_grouped_gate_up_swiglu_q8_0_w10_mg2(
    const signed char* xq_q, const float* xq_d, const signed char* gu_q,
    const unsigned short* gu_s, const int* tiles, unsigned int num_tiles, int total_cols,
    float* swiglu_compact, unsigned int hidden_dim, unsigned int inter_dim) {
    moe_w10_gate_up_body<2>(xq_q,xq_d,gu_q,gu_s,tiles,num_tiles,total_cols,swiglu_compact,hidden_dim,inter_dim);
}
extern "C" __global__ __launch_bounds__(256, 2)
void moe_grouped_gate_up_swiglu_q8_0_w10_mg3(
    const signed char* xq_q, const float* xq_d, const signed char* gu_q,
    const unsigned short* gu_s, const int* tiles, unsigned int num_tiles, int total_cols,
    float* swiglu_compact, unsigned int hidden_dim, unsigned int inter_dim) {
    moe_w10_gate_up_body<3>(xq_q,xq_d,gu_q,gu_s,tiles,num_tiles,total_cols,swiglu_compact,hidden_dim,inter_dim);
}
extern "C" __global__ __launch_bounds__(256, 2)
void moe_grouped_gate_up_swiglu_q8_0_w10_mg4(
    const signed char* xq_q, const float* xq_d, const signed char* gu_q,
    const unsigned short* gu_s, const int* tiles, unsigned int num_tiles, int total_cols,
    float* swiglu_compact, unsigned int hidden_dim, unsigned int inter_dim) {
    moe_w10_gate_up_body<4>(xq_q,xq_d,gu_q,gu_s,tiles,num_tiles,total_cols,swiglu_compact,hidden_dim,inter_dim);
}
