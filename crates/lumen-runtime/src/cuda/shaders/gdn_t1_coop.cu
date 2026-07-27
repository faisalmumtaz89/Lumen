// ==========================================================================
// gdn_t1_coop_all: the whole T=1 GDN post-projection chain in ONE cooperative
// launch — conv1d+SiLU+L2norm, gate transform, delta-rule state update, and
// output RMSNorm+gate — separated by grid.sync() instead of kernel boundaries.
//
// WHY THIS SHAPE
//
// The GDN recurrence is 1.23 ms/token = 51 us per GDN layer while moving only
// ~96 MiB of state across all 24 layers. Two earlier attempts explain what does
// and does not work here:
//
//   * Four-warp CTA grouping (gdn_prefill_fused_v3_t1_w4) gave +0.6%. It cut
//     CTA COUNT 4x but kept all 4096 warps, kept the per-column Q/K reloads,
//     and left the other three launches standing.
//   * Fusing conv+SiLU+L2norm gave +1.0% by removing one launch of four.
//
// So neither CTA count nor a single launch boundary is the binding cost — the
// combination of warp count, redundant Q/K traffic and four serialized
// dispatches is. This kernel attacks all three at once:
//
//   Phase A (CTAs 0-63):  conv1d + SiLU + L2 normalize Q/K; CTAs 0-31 also
//                         compute one head's alpha/beta gates.
//   grid.sync()
//   Phase B (256 CTAs):   delta-rule state update, 8 CTAs per head. Warp w
//                         handles columns tile*16 + w + 4*j for j=0..3, so a
//                         lane loads its Q/K float4 ONCE and reuses it across
//                         four columns — removing the reload that limited the
//                         w4 kernel. Two columns are interleaved to expose two
//                         independent load/reduce chains (the useful part of
//                         R2) while staying under ~48 registers/thread.
//   grid.sync()
//   Phase C (CTAs 0-31):  per-head RMSNorm of the raw output and the SiLU gate.
//
// State layout is UNCHANGED: F32 [head][value_column][key] = [32][128][128],
// one read and one write per element, no transpose and no migration.
//
// Numerics: each phase performs the same arithmetic in the same order as the
// kernel it replaces — same conv taps and SiLU, same sum-of-squares reduction
// and 1e-12 floor for L2, the same warp_reduce delta-rule sequence with q_scale
// applied identically, and the same RMSNorm+SiLU-gate expression. Only the
// synchronization mechanism changes, so the audited harness's token-identity
// check is the correctness gate.
//
// Launch: cooperative, grid 256, block 128. Requires the device to support
// cooperative launch and the grid to be co-resident — the caller checks
// occupancy and falls back to the four-kernel chain when it is not.
//
// NVRTC-compatible: no system includes, extern "C" linkage. Uses the PTX
// barrier intrinsic rather than <cooperative_groups.h> so no host headers are
// required at NVRTC compile time.

#define COOP_BLOCK      128
#define COOP_WARPS      (COOP_BLOCK / 32)
#define COOP_GRID       256
#define COOP_COLS_PER_W 4          // columns each warp owns in phase B
#define Q4_EPS          1e-12f

__device__ __forceinline__ float warp_reduce_sum_coop(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 16);
    v += __shfl_xor_sync(0xffffffff, v, 8);
    v += __shfl_xor_sync(0xffffffff, v, 4);
    v += __shfl_xor_sync(0xffffffff, v, 2);
    v += __shfl_xor_sync(0xffffffff, v, 1);
    return v;
}

// TRUE grid-wide barrier (sense-reversing, atomic-counter).
//
// `barrier.sync 0` is a BLOCK barrier — using it here would let CTAs race
// between phases and silently corrupt state rather than merely run slowly.
// cooperative_groups is unavailable under NVRTC without host headers, so this
// is the standard counter barrier that this_grid().sync() compiles to.
//
// SAFETY: this deadlocks unless the ENTIRE grid is co-resident, which is
// exactly what a cooperative launch guarantees. The host side must verify
// occupancy via cuOccupancyMaxActiveBlocksPerMultiprocessor and launch with
// cuLaunchCooperativeKernel; it falls back to the four-kernel chain otherwise.
//
// `bar` points at two u32s: [0] = arrival counter, [1] = generation.
__device__ __forceinline__ void grid_sync_coop(unsigned int* bar, unsigned int n_ctas) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();                       // publish this CTA's writes
        const unsigned int gen = atomicAdd(&bar[1], 0u);
        if (atomicAdd(&bar[0], 1u) == n_ctas - 1u) {
            atomicExch(&bar[0], 0u);           // last arrival resets and flips
            atomicAdd(&bar[1], 1u);
        } else {
            // Plain spin: __nanosleep is unavailable under NVRTC without an
            // explicit arch flag (this is the NVRTC_ERROR_COMPILATION at line
            // 90 that made the kernel load as None in round 33).
            while (atomicAdd(&bar[1], 0u) == gen) {
                __threadfence();
            }
        }
        __threadfence();                       // observe everyone else's writes
    }
    __syncthreads();
}

extern "C" __global__ void gdn_t1_coop_all(
    const float* __restrict__ qkv_in,     // [qkv_dim] raw projection output
    float* __restrict__ conv_state,       // [buf_slots, conv_dim] R/W
    const float* __restrict__ conv_w,     // [conv_dim, kernel_size]
    float* __restrict__ conv_out,         // [qkv_dim] post conv+SiLU+L2
    const float* __restrict__ dt_bias,    // [n_heads]
    const float* __restrict__ ssm_a,      // [n_heads]
    const float* __restrict__ alpha_raw,  // [n_heads]
    const float* __restrict__ beta_raw,   // [n_heads]
    float* __restrict__ alpha_buf,        // [n_heads]
    float* __restrict__ beta_buf,         // [n_heads]
    float* __restrict__ h_state,          // [n_heads, val_dim, key_dim]
    float* __restrict__ raw_out,          // [n_heads, val_dim]
    const float* __restrict__ norm_w,     // [val_dim]
    const float* __restrict__ z_gate,     // [n_heads * val_dim]
    float* __restrict__ normed_out,       // [n_heads * val_dim]
    unsigned int* __restrict__ barrier,   // [2] u32 scratch, zeroed by host
    unsigned int n_heads,
    unsigned int n_kv_heads,
    unsigned int key_dim,
    unsigned int val_dim,
    unsigned int qk_dim,
    unsigned int qkv_dim,
    unsigned int kernel_size,
    unsigned int state_pos,
    float norm_eps)
{
    const unsigned int tid  = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const unsigned int blk  = blockIdx.x;
    const unsigned int buf_slots = kernel_size - 1;

    // ---------------- Phase A: conv1d + SiLU + L2(Q,K), and gates ----------
    if (blk < 2u * n_kv_heads + ((qkv_dim - 2u * qk_dim) + COOP_BLOCK - 1) / COOP_BLOCK) {
        unsigned int base, count;
        bool normalize;
        if (blk < 2u * n_kv_heads) {
            base = (blk / n_kv_heads) * qk_dim + (blk % n_kv_heads) * key_dim;
            count = key_dim;
            normalize = true;
        } else {
            base = 2u * qk_dim + (blk - 2u * n_kv_heads) * COOP_BLOCK;
            count = (qkv_dim > base) ? (qkv_dim - base) : 0u;
            if (count > COOP_BLOCK) count = COOP_BLOCK;
            normalize = false;
        }

        float ss = 0.0f;
        for (unsigned int i = tid; i < count; i += COOP_BLOCK) {
            const unsigned int gid = base + i;
            const float inp = qkv_in[gid];
            float sum = 0.0f;
            for (unsigned int tap = 0; tap < buf_slots; tap++) {
                const unsigned int slot = (state_pos + tap) % buf_slots;
                sum += conv_w[gid * kernel_size + tap] * conv_state[slot * qkv_dim + gid];
            }
            sum += conv_w[gid * kernel_size + buf_slots] * inp;
            conv_state[state_pos * qkv_dim + gid] = inp;
            const float act = sum / (1.0f + expf(-sum));
            conv_out[gid] = act;
            if (normalize) ss += act * act;
        }

        if (normalize) {
            __shared__ float red[COOP_WARPS];
            __shared__ float inv_n;
            ss = warp_reduce_sum_coop(ss);
            if (lane == 0) red[warp] = ss;
            __syncthreads();
            if (tid == 0) {
                float t = 0.0f;
                #pragma unroll
                for (int i = 0; i < COOP_WARPS; i++) t += red[i];
                const float n = sqrtf(t);
                inv_n = (n > Q4_EPS) ? (1.0f / n) : (1.0f / Q4_EPS);
            }
            __syncthreads();
            const float s = inv_n;
            for (unsigned int i = tid; i < count; i += COOP_BLOCK) conv_out[base + i] *= s;
        }
    }

    // gates: one head per low CTA, thread 0 (same formula as the batched kernel)
    if (blk < n_heads && tid == 0) {
        // EXACTLY gdn_compute_gates_batched: dt_bias is added BEFORE the
        // softplus, and the softplus uses logf(1+expf(x)) rather than log1pf.
        // My first draft did neither, which would have diverged the recurrence
        // silently — the harness would have reported "wrong output" with no
        // pointer to the cause.
        const float sp_input = alpha_raw[blk] + dt_bias[blk];
        const float sp = (sp_input > 20.0f) ? sp_input : logf(1.0f + expf(sp_input));
        alpha_buf[blk] = expf(ssm_a[blk] * sp);
        beta_buf[blk] = 1.0f / (1.0f + expf(-beta_raw[blk]));
    }

    grid_sync_coop(barrier, gridDim.x);

    // ---------------- Phase B: delta-rule state update ---------------------
    {
        const unsigned int head = blk >> 3;          // 8 CTAs per head
        const unsigned int tile = blk & 7u;
        if (head < n_heads) {
            const unsigned int kv_head = head % n_kv_heads;
            const float q_scale = rsqrtf((float)key_dim);
            const float a = alpha_buf[head];
            const float b = beta_buf[head];
            const unsigned int k_base = lane * 4u;

            // Q/K loaded ONCE per lane and reused across this warp's columns —
            // the reload the w4 kernel kept paying.
            const float* cq = conv_out + kv_head * key_dim + k_base;
            const float* ck = conv_out + qk_dim + kv_head * key_dim + k_base;
            const float qn0 = cq[0], qn1 = cq[1], qn2 = cq[2], qn3 = cq[3];
            const float kn0 = ck[0], kn1 = ck[1], kn2 = ck[2], kn3 = ck[3];

            #pragma unroll
            for (int j = 0; j < COOP_COLS_PER_W; j++) {
                const unsigned int vj = tile * 16u + warp + 4u * (unsigned int)j;
                if (vj >= val_dim) continue;
                float* h_row = h_state + (size_t)head * val_dim * key_dim + (size_t)vj * key_dim;
                float s0 = h_row[k_base + 0], s1 = h_row[k_base + 1];
                float s2 = h_row[k_base + 2], s3 = h_row[k_base + 3];

                const float v_val = conv_out[2u * qk_dim + head * val_dim + vj];
                float d0 = a * s0, d1 = a * s1, d2 = a * s2, d3 = a * s3;
                const float retrieval =
                    warp_reduce_sum_coop(d0 * kn0 + d1 * kn1 + d2 * kn2 + d3 * kn3);
                const float v_delta = b * (v_val - retrieval);
                s0 = d0 + kn0 * v_delta; s1 = d1 + kn1 * v_delta;
                s2 = d2 + kn2 * v_delta; s3 = d3 + kn3 * v_delta;
                const float o =
                    warp_reduce_sum_coop(s0 * qn0 + s1 * qn1 + s2 * qn2 + s3 * qn3) * q_scale;
                if (lane == 0) raw_out[head * val_dim + vj] = o;
                h_row[k_base + 0] = s0; h_row[k_base + 1] = s1;
                h_row[k_base + 2] = s2; h_row[k_base + 3] = s3;
            }
        }
    }

    grid_sync_coop(barrier, gridDim.x);

    // ---------------- Phase C: output RMSNorm + SiLU gate ------------------
    if (blk < n_heads) {
        const unsigned int head = blk;
        const float* r = raw_out + head * val_dim;
        float* o = normed_out + head * val_dim;
        const float* z = z_gate + head * val_dim;

        float ss = 0.0f;
        for (unsigned int i = tid; i < val_dim; i += COOP_BLOCK) ss += r[i] * r[i];
        __shared__ float red2[COOP_WARPS];
        __shared__ float inv_rms;
        ss = warp_reduce_sum_coop(ss);
        if (lane == 0) red2[warp] = ss;
        __syncthreads();
        if (tid == 0) {
            float t = 0.0f;
            #pragma unroll
            for (int i = 0; i < COOP_WARPS; i++) t += red2[i];
            inv_rms = 1.0f / sqrtf(t / (float)val_dim + norm_eps);
        }
        __syncthreads();
        const float s = inv_rms;
        for (unsigned int i = tid; i < val_dim; i += COOP_BLOCK) {
            const float zv = z[i];
            o[i] = (r[i] * s * norm_w[i]) * (zv / (1.0f + expf(-zv)));
        }
    }
}
