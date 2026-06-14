// Batched shared-expert FFN kernels.
//
// The Qwen3.5-MoE shared expert is a single ALWAYS-active Q4_0 FFN run on every
// token. Batching the routed experts alone still left the shared expert in a
// per-token loop (1384 tok × 40 layers of single-token kernels). The skip-shared
// diagnostic measured the shared-expert loop costs +43% prefill (230.2 → 161.0
// tok/s), making it the #1 residual bottleneck.
//
// These kernels batch the shared expert over all `batch` tokens. They mirror the
// per-token FUSED shared-expert path (fused_glu_gemv_q4_0_prenormed_no_norm +
// moe_shared_dot_f32 + moe_shared_down_q4_0_sigmoid_accum) EXACTLY — same NR=2
// row-tile layout, same Q4_0 de-interleaved nibble dequant (scale*(nibble-8)),
// same F32 per-block-scale accumulation, same warp+cross-warp reduction tree,
// same sigmoid gate — with an added blockIdx.y=token dimension. For a fixed
// (token, row) the produced float is bit-identical to the per-token fused path.
//
// NVRTC-compatible: inline PTX f16->f32, no system includes, extern "C".

#define SB_NR             2
#define SB_BLOCK_DIM      256
#define SB_WARP_SIZE      32
#define SB_Q4_BLOCK_ELEMS 32
#define SB_Q4_BLOCK_BYTES 18

__device__ __forceinline__ float sb_f16_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

__device__ __forceinline__ float sb_warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

// ===========================================================================
// Stage 1: batched fused gate+up+SwiGLU (Q4_0, pre-normalized x).
//
// swiglu_out[tok, r] = silu(gate_r · normed[tok]) * (up_r · normed[tok]).
// Bit-identical to `fused_glu_gemv_q4_0_prenormed_no_norm` per (tok, r).
//
// Grid:  (ceil(inter_dim / SB_NR), batch, 1)
// Block: (SB_BLOCK_DIM, 1, 1)
// Shmem: hidden_dim * 4 bytes
// ===========================================================================
extern "C" __global__ void shared_glu_gemv_q4_0_batched(
    const char*  __restrict__ w_gate,    // [inter_dim, hidden_dim] Q4_0
    const char*  __restrict__ w_up,       // [inter_dim, hidden_dim] Q4_0
    const float* __restrict__ normed,     // [batch, hidden_dim] already RMSNormed
    float*       __restrict__ swiglu_out, // [batch, inter_dim] silu(gate)*up
    unsigned int inter_dim,
    unsigned int hidden_dim,
    unsigned int batch)
{
    extern __shared__ float nx_smem[];

    const unsigned int tok = blockIdx.y;
    if (tok >= batch) return;
    const unsigned int r0 = blockIdx.x * SB_NR;
    const unsigned int warp_id = threadIdx.x / SB_WARP_SIZE;
    const unsigned int lane    = threadIdx.x % SB_WARP_SIZE;
    const unsigned int num_blocks = hidden_dim / SB_Q4_BLOCK_ELEMS;
    const unsigned long long row_bytes = (unsigned long long)num_blocks * SB_Q4_BLOCK_BYTES;

    const float* x_row = normed + (unsigned long long)tok * hidden_dim;
    for (unsigned int i = threadIdx.x; i < hidden_dim; i += SB_BLOCK_DIM) {
        nx_smem[i] = x_row[i];
    }
    __syncthreads();

    float gate_sum[SB_NR];
    float up_sum[SB_NR];
    #pragma unroll
    for (int r = 0; r < SB_NR; r++) { gate_sum[r] = 0.0f; up_sum[r] = 0.0f; }

    for (unsigned int ib = threadIdx.x; ib < num_blocks; ib += SB_BLOCK_DIM) {
        const unsigned int x_base = ib * SB_Q4_BLOCK_ELEMS;
        float xv[32];
        const float4* x4 = (const float4*)(nx_smem + x_base);
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            float4 v = x4[k];
            xv[k * 4 + 0] = v.x; xv[k * 4 + 1] = v.y;
            xv[k * 4 + 2] = v.z; xv[k * 4 + 3] = v.w;
        }
        #pragma unroll
        for (int row = 0; row < SB_NR; row++) {
            if (r0 + row >= inter_dim) break;
            const char* gp = w_gate + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * SB_Q4_BLOCK_BYTES;
            unsigned short g_scale_bits = (unsigned short)(unsigned char)gp[0]
                | ((unsigned short)(unsigned char)gp[1] << 8);
            float g_scale = sb_f16_to_f32(g_scale_bits);
            const unsigned char* gq = (const unsigned char*)(gp + 2);
            const char* up_ = w_up + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * SB_Q4_BLOCK_BYTES;
            unsigned short u_scale_bits = (unsigned short)(unsigned char)up_[0]
                | ((unsigned short)(unsigned char)up_[1] << 8);
            float u_scale = sb_f16_to_f32(u_scale_bits);
            const unsigned char* uq = (const unsigned char*)(up_ + 2);
            float g_block = 0.0f, u_block = 0.0f;
            #pragma unroll
            for (int b = 0; b < 16; b++) {
                unsigned char gb = gq[b];
                unsigned char ub = uq[b];
                float gq_lo = (float)(gb & 0x0F) - 8.0f;
                float gq_hi = (float)(gb >> 4)   - 8.0f;
                float uq_lo = (float)(ub & 0x0F) - 8.0f;
                float uq_hi = (float)(ub >> 4)   - 8.0f;
                g_block += gq_lo * xv[b] + gq_hi * xv[b + 16];
                u_block += uq_lo * xv[b] + uq_hi * xv[b + 16];
            }
            gate_sum[row] += g_scale * g_block;
            up_sum[row]   += u_scale * u_block;
        }
    }

    const unsigned int num_warps = SB_BLOCK_DIM / SB_WARP_SIZE;
    #pragma unroll
    for (int r = 0; r < SB_NR; r++) gate_sum[r] = sb_warp_reduce_sum(gate_sum[r]);
    __syncthreads();
    float* reduce_smem = nx_smem;
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < SB_NR; r++) reduce_smem[r * num_warps + warp_id] = gate_sum[r];
    }
    __syncthreads();
    float final_gate[SB_NR];
    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < SB_NR; r++) {
            float val = (lane < num_warps) ? reduce_smem[r * num_warps + lane] : 0.0f;
            final_gate[r] = sb_warp_reduce_sum(val);
        }
    }
    __syncthreads();
    #pragma unroll
    for (int r = 0; r < SB_NR; r++) up_sum[r] = sb_warp_reduce_sum(up_sum[r]);
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < SB_NR; r++) reduce_smem[r * num_warps + warp_id] = up_sum[r];
    }
    __syncthreads();
    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < SB_NR; r++) {
            if (r0 + r < inter_dim) {
                float val = (lane < num_warps) ? reduce_smem[r * num_warps + lane] : 0.0f;
                val = sb_warp_reduce_sum(val);
                if (lane == 0) {
                    float g = final_gate[r];
                    float silu_g = g / (1.0f + expf(-g));
                    swiglu_out[(unsigned long long)tok * inter_dim + (r0 + r)] = silu_g * val;
                }
            }
        }
    }
}

// ===========================================================================
// Stage 2: batched F32 gate-input dot → per-token logit.
//
// logit[tok] = Σ_j w[j] * normed[tok, j]. Bit-identical to moe_shared_dot_f32
// per token (same shfl_down warp reduction, same accumulation order).
//
// Grid:  (1, batch, 1)
// Block: (SB_BLOCK_DIM, 1, 1)
// ===========================================================================
extern "C" __global__ void shared_dot_f32_batched(
    const float* __restrict__ w,        // [hidden_dim]
    const float* __restrict__ normed,   // [batch, hidden_dim]
    float*       __restrict__ logit,    // [batch]
    unsigned int in_dim,
    unsigned int batch)
{
    const unsigned int tok = blockIdx.y;
    if (tok >= batch) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31;
    const unsigned int warp_id = tid >> 5;

    const float* x_row = normed + (unsigned long long)tok * in_dim;
    float partial = 0.0f;
    for (unsigned int j = tid; j < in_dim; j += SB_BLOCK_DIM) {
        partial += w[j] * x_row[j];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(0xffffffff, partial, offset);
    }
    __shared__ float warp_partial[SB_BLOCK_DIM / 32];
    if (lane == 0) warp_partial[warp_id] = partial;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane < (SB_BLOCK_DIM / 32)) ? warp_partial[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_down_sync(0xffffffff, v, offset);
        }
        if (lane == 0) logit[tok] = v;
    }
}

// ===========================================================================
// Stage 3: batched down-matvec + per-token sigmoid-gated accumulate.
//
// x_out[tok, row] += sigmoid(logit[tok]) * Σ_b scale[b] * dot(W_down[row,b], swiglu[tok,b]).
// Bit-identical to moe_shared_down_q4_0_sigmoid_accum per (tok, row).
//
// Grid:  (hidden_dim, batch, 1)
// Block: (SB_BLOCK_DIM, 1, 1)
// ===========================================================================
extern "C" __global__ void shared_down_q4_0_sigmoid_accum_batched(
    const char*  __restrict__ w_down,     // [hidden_dim, inter_dim] Q4_0
    const float* __restrict__ swiglu,     // [batch, inter_dim]
    const float* __restrict__ logit,      // [batch] pre-sigmoid
    float*       __restrict__ x_out,      // [batch, hidden_dim] in/out
    unsigned int hidden_dim,
    unsigned int inter_dim,
    unsigned int batch)
{
    const unsigned int row = blockIdx.x;
    const unsigned int tok = blockIdx.y;
    if (row >= hidden_dim || tok >= batch) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & (SB_WARP_SIZE - 1);
    const unsigned int warp_id = tid / SB_WARP_SIZE;

    const unsigned int num_blocks = inter_dim / SB_Q4_BLOCK_ELEMS;
    const unsigned long long row_bytes = (unsigned long long)num_blocks * SB_Q4_BLOCK_BYTES;
    const char* row_ptr = w_down + (unsigned long long)row * row_bytes;
    const float* swig = swiglu + (unsigned long long)tok * inter_dim;

    float partial = 0.0f;
    for (unsigned int b = tid; b < num_blocks; b += SB_BLOCK_DIM) {
        const char* block_ptr = row_ptr + b * SB_Q4_BLOCK_BYTES;
        unsigned short scale_bits = (unsigned short)(unsigned char)block_ptr[0]
            | ((unsigned short)(unsigned char)block_ptr[1] << 8);
        float scale = sb_f16_to_f32(scale_bits);
        const unsigned int x_base = b * SB_Q4_BLOCK_ELEMS;
        const unsigned char* qp = (const unsigned char*)(block_ptr + 2);
        float block_sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < 16; k++) {
            unsigned char by = qp[k];
            float q_lo = (float)(by & 0x0F) - 8.0f;
            float q_hi = (float)(by >> 4)   - 8.0f;
            block_sum += q_lo * swig[x_base + k] + q_hi * swig[x_base + k + 16];
        }
        partial += scale * block_sum;
    }
    partial = sb_warp_reduce_sum(partial);
    __shared__ float warp_sums[SB_BLOCK_DIM / SB_WARP_SIZE];
    if (lane == 0) warp_sums[warp_id] = partial;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane < (SB_BLOCK_DIM / SB_WARP_SIZE)) ? warp_sums[lane] : 0.0f;
        v = sb_warp_reduce_sum(v);
        if (lane == 0) {
            const float s = logit[tok];
            const float g = 1.0f / (1.0f + expf(-s));
            x_out[(unsigned long long)tok * hidden_dim + row] += g * v;
        }
    }
}

// ===========================================================================
// Stage 3 (no-gate variant): batched down-matvec + plain residual accumulate.
// x_out[tok, row] += Σ_b scale[b] * dot(W_down[row,b], swiglu[tok,b]).
// Bit-identical to moe_shared_down_q4_0_residual_accum per (tok, row).
// ===========================================================================
extern "C" __global__ void shared_down_q4_0_residual_accum_batched(
    const char*  __restrict__ w_down,     // [hidden_dim, inter_dim] Q4_0
    const float* __restrict__ swiglu,     // [batch, inter_dim]
    float*       __restrict__ x_out,      // [batch, hidden_dim] in/out
    unsigned int hidden_dim,
    unsigned int inter_dim,
    unsigned int batch)
{
    const unsigned int row = blockIdx.x;
    const unsigned int tok = blockIdx.y;
    if (row >= hidden_dim || tok >= batch) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & (SB_WARP_SIZE - 1);
    const unsigned int warp_id = tid / SB_WARP_SIZE;

    const unsigned int num_blocks = inter_dim / SB_Q4_BLOCK_ELEMS;
    const unsigned long long row_bytes = (unsigned long long)num_blocks * SB_Q4_BLOCK_BYTES;
    const char* row_ptr = w_down + (unsigned long long)row * row_bytes;
    const float* swig = swiglu + (unsigned long long)tok * inter_dim;

    float partial = 0.0f;
    for (unsigned int b = tid; b < num_blocks; b += SB_BLOCK_DIM) {
        const char* block_ptr = row_ptr + b * SB_Q4_BLOCK_BYTES;
        unsigned short scale_bits = (unsigned short)(unsigned char)block_ptr[0]
            | ((unsigned short)(unsigned char)block_ptr[1] << 8);
        float scale = sb_f16_to_f32(scale_bits);
        const unsigned int x_base = b * SB_Q4_BLOCK_ELEMS;
        const unsigned char* qp = (const unsigned char*)(block_ptr + 2);
        float block_sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < 16; k++) {
            unsigned char by = qp[k];
            float q_lo = (float)(by & 0x0F) - 8.0f;
            float q_hi = (float)(by >> 4)   - 8.0f;
            block_sum += q_lo * swig[x_base + k] + q_hi * swig[x_base + k + 16];
        }
        partial += scale * block_sum;
    }
    partial = sb_warp_reduce_sum(partial);
    __shared__ float warp_sums[SB_BLOCK_DIM / SB_WARP_SIZE];
    if (lane == 0) warp_sums[warp_id] = partial;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane < (SB_BLOCK_DIM / SB_WARP_SIZE)) ? warp_sums[lane] : 0.0f;
        v = sb_warp_reduce_sum(v);
        if (lane == 0) {
            x_out[(unsigned long long)tok * hidden_dim + row] += v;
        }
    }
}

// ===========================================================================
// TILED shared-expert FFN — f32-activation design.
//
// The shared expert is a DENSE Q4_0 FFN over ALL batch tokens (no routing), so
// the same shmem-staged, per-thread-accumulator, no-cross-thread-reduction tiled
// design used for the routed FFN applies directly with the token batch as
// the "columns" — no expert/tile-list machinery (every CTA processes a contiguous
// token block). RAW F32 activation + Q4_0 nibble-decode int8 weight + per-block
// f32 scale + F32 dot = the per-token reference math (only per-thread sequential
// cross-block accumulation differs from the warp-tree reduction = the PRISTINE
// near-tie class). Replaces the per-(row,token) matvec shared_*_batched kernels.
//
// Shape guards (host): hidden_dim % 256 == 0 (gate_up K-blocks SBT_BK=8), inter_dim
// % 32 == 0 (whole down q-blocks; down K-tail masked), hidden_dim % SBTD_BN(64)
// == 0 (down rows), inter_dim % SBT_BN(32) == 0 (gate_up rows).
// ===========================================================================
#define SB_Q4_DECODE 1  // marker

// Decode a Q4_0 block's 32 quants into 32 signed int8 (K-order, de-interleaved).
__device__ __forceinline__ void sb_q4_0_decode32(const unsigned char* qs, signed char* dst) {
    #pragma unroll
    for (int idx = 0; idx < 16; ++idx) {
        const unsigned char b = qs[idx];
        dst[idx]      = (signed char)((int)(b & 0x0F) - 8);
        dst[idx + 16] = (signed char)((int)(b >> 4)   - 8);
    }
}

// ---- gate+up+SwiGLU, f32-activation tiled (Q4_0 dense, all tokens). ----
// CTA = SBT_BM tokens x SBT_BN rows x SBT_BK k-blocks, 256 threads, 1 tok x 2 rows
// per thread. Grid: (ceil(batch/SBT_BM), inter_dim/SBT_BN, 1).
#define SBT_BM      16
#define SBT_BN      32
#define SBT_BK      8
#define SBT_COL_PAD (SBT_BM + 1)   // 17
#define SBT_ROW_PAD (SBT_BN + 1)   // 33

extern "C" __global__ __launch_bounds__(256, 2)
void shared_glu_gemv_q4_0_batched_tiled_f32act(
    const char*  __restrict__ w_gate,    // [inter_dim, hidden_dim] Q4_0
    const char*  __restrict__ w_up,      // [inter_dim, hidden_dim] Q4_0
    const float* __restrict__ normed,    // [batch, hidden_dim] already RMSNormed
    float*       __restrict__ swiglu_out,// [batch, inter_dim]
    unsigned int inter_dim,
    unsigned int hidden_dim,
    unsigned int batch)
{
    const unsigned int tok_tile = blockIdx.x;
    const unsigned int row_tile = blockIdx.y;
    const unsigned int tok_base = tok_tile * SBT_BM;
    const unsigned int row_base = row_tile * SBT_BN;
    if (tok_base >= batch || row_base >= inter_dim) return;

    const unsigned int tid    = threadIdx.x;
    const unsigned int lane8  = tid & 7;
    const unsigned int group8 = tid >> 3;     // 0..31
    const unsigned int col    = tid & 15;     // token-in-tile 0..15
    const unsigned int row_q  = tid >> 4;     // 0..15
    const unsigned int r0     = row_q << 1;   // local row 0,2,..,30

    const unsigned int num_blocks = hidden_dim / SB_Q4_BLOCK_ELEMS;
    const unsigned long long row_bytes = (unsigned long long)num_blocks * SB_Q4_BLOCK_BYTES;

    __shared__ float       s_xf[SBT_BK][32][SBT_COL_PAD];
    __shared__ signed char s_gwb[SBT_BK][32][SBT_ROW_PAD];
    __shared__ signed char s_uwb[SBT_BK][32][SBT_ROW_PAD];
    __shared__ float       s_gws[SBT_BK][SBT_ROW_PAD];
    __shared__ float       s_uws[SBT_BK][SBT_ROW_PAD];

    const int active_toks = (int)batch - (int)tok_base; // tokens valid in this tile
    float g0=0.f,g1=0.f,u0=0.f,u1=0.f;

    for (unsigned int k0 = 0; k0 < num_blocks; k0 += SBT_BK) {
        // Stage 1: gather raw F32 activation.
        {
            const int m       = group8 & 15;       // token 0..15
            const int kk_base = group8 >> 4;        // 0 or 1
            const bool active = (m < active_toks);
            const unsigned int tok = tok_base + (unsigned int)m;
            const float* xrow = active ? (normed + (unsigned long long)tok * hidden_dim) : normed;
            const unsigned int offs = lane8 << 2;
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int kk = kk_base + (t << 1);
                const unsigned int kblk = k0 + (unsigned int)kk;
                const bool kvalid = (kblk < num_blocks);
                float v0=0.f,v1=0.f,v2=0.f,v3=0.f;
                if (active && kvalid) {
                    const unsigned int k_elem = kblk << 5;
                    const float* xp = xrow + k_elem + offs;
                    v0=xp[0]; v1=xp[1]; v2=xp[2]; v3=xp[3];
                }
                s_xf[kk][offs+0][m]=v0; s_xf[kk][offs+1][m]=v1;
                s_xf[kk][offs+2][m]=v2; s_xf[kk][offs+3][m]=v3;
            }
        }
        // Stage 2: stage gate+up weight tiles (nibble decode).
        {
            const int row_local = group8;   // 0..31
            const int kk        = lane8;    // 0..7
            const unsigned int kblk = k0 + (unsigned int)kk;
            const bool kvalid = (kblk < num_blocks);
            const unsigned int row_global = row_base + (unsigned int)row_local;
            signed char gwb[32], uwb[32]; float gsc=0.f, usc=0.f;
            if (row_global < inter_dim && kvalid) {
                const char* gblk = w_gate + (unsigned long long)row_global * row_bytes
                    + (unsigned long long)kblk * SB_Q4_BLOCK_BYTES;
                const char* ublk = w_up + (unsigned long long)row_global * row_bytes
                    + (unsigned long long)kblk * SB_Q4_BLOCK_BYTES;
                unsigned short gb = (unsigned short)(unsigned char)gblk[0]
                    | ((unsigned short)(unsigned char)gblk[1] << 8);
                unsigned short ub = (unsigned short)(unsigned char)ublk[0]
                    | ((unsigned short)(unsigned char)ublk[1] << 8);
                gsc = sb_f16_to_f32(gb); usc = sb_f16_to_f32(ub);
                sb_q4_0_decode32((const unsigned char*)(gblk + 2), gwb);
                sb_q4_0_decode32((const unsigned char*)(ublk + 2), uwb);
            } else {
                #pragma unroll
                for (int p=0;p<32;++p){gwb[p]=0;uwb[p]=0;}
            }
            #pragma unroll
            for (int p=0;p<32;++p){ s_gwb[kk][p][row_local]=gwb[p]; s_uwb[kk][p][row_local]=uwb[p]; }
            s_gws[kk][row_local]=gsc; s_uws[kk][row_local]=usc;
        }
        __syncthreads();
        // Stage 3: compute.
        #pragma unroll
        for (int kk=0; kk<SBT_BK; ++kk) {
            float bg0=0.f,bg1=0.f,bu0=0.f,bu1=0.f;
            #pragma unroll
            for (int j=0;j<32;++j) {
                const float xv = s_xf[kk][j][col];
                bg0 += (float)s_gwb[kk][j][r0+0]*xv;
                bg1 += (float)s_gwb[kk][j][r0+1]*xv;
                bu0 += (float)s_uwb[kk][j][r0+0]*xv;
                bu1 += (float)s_uwb[kk][j][r0+1]*xv;
            }
            g0 += s_gws[kk][r0+0]*bg0; g1 += s_gws[kk][r0+1]*bg1;
            u0 += s_uws[kk][r0+0]*bu0; u1 += s_uws[kk][r0+1]*bu1;
        }
        __syncthreads();
    }
    // SwiGLU + store.
    if (col < (unsigned int)active_toks) {
        const unsigned int tok = tok_base + col;
        const unsigned long long base = (unsigned long long)tok * inter_dim + (row_base + r0);
        if (row_base + r0 + 0 < inter_dim) { float sg=g0/(1.0f+expf(-g0)); swiglu_out[base+0]=sg*u0; }
        if (row_base + r0 + 1 < inter_dim) { float sg=g1/(1.0f+expf(-g1)); swiglu_out[base+1]=sg*u1; }
    }
}

// ---- down + sigmoid-gated accum, f32-activation tiled (Q4_0 dense). ----
// CTA = SBTD_BM tokens x SBTD_BN rows x SBTD_BK k-blocks, 1 tok x 4 rows/thread.
// gate_mode: 0 = plain residual accum, 1 = sigmoid(logit[tok]) gated.
#define SBTD_BM      16
#define SBTD_BN      64
#define SBTD_BK      8
#define SBTD_COL_PAD (SBTD_BM + 1)   // 17
#define SBTD_ROW_PAD (SBTD_BN + 1)   // 65

extern "C" __global__ __launch_bounds__(256, 2)
void shared_down_q4_0_accum_batched_tiled_f32act(
    const char*  __restrict__ w_down,    // [hidden_dim, inter_dim] Q4_0
    const float* __restrict__ swiglu,    // [batch, inter_dim]
    const float* __restrict__ logit,     // [batch] pre-sigmoid (used iff gate_mode==1)
    float*       __restrict__ x_out,     // [batch, hidden_dim] in/out
    unsigned int hidden_dim,
    unsigned int inter_dim,
    unsigned int batch,
    unsigned int gate_mode)
{
    const unsigned int tok_tile = blockIdx.x;
    const unsigned int row_tile = blockIdx.y;
    const unsigned int tok_base = tok_tile * SBTD_BM;
    const unsigned int row_base = row_tile * SBTD_BN;
    if (tok_base >= batch || row_base >= hidden_dim) return;

    const unsigned int tid    = threadIdx.x;
    const unsigned int lane8  = tid & 7;
    const unsigned int group8 = tid >> 3;     // 0..31
    const unsigned int col    = tid & 15;     // token-in-tile 0..15
    const unsigned int row_q  = tid >> 4;     // 0..15
    const unsigned int r0     = row_q << 2;   // local row 0,4,..,60

    const unsigned int num_blocks = inter_dim / SB_Q4_BLOCK_ELEMS;
    const unsigned long long row_bytes = (unsigned long long)num_blocks * SB_Q4_BLOCK_BYTES;

    __shared__ float       s_xf[SBTD_BK][32][SBTD_COL_PAD];
    __shared__ signed char s_wb[SBTD_BK][32][SBTD_ROW_PAD];
    __shared__ float       s_ws[SBTD_BK][SBTD_ROW_PAD];

    const int active_toks = (int)batch - (int)tok_base;
    float a0=0.f,a1=0.f,a2=0.f,a3=0.f;

    for (unsigned int k0 = 0; k0 < num_blocks; k0 += SBTD_BK) {
        // Stage 1: gather raw F32 swiglu activation.
        {
            const int m       = group8 & 15;
            const int kk_base = group8 >> 4;
            const bool active = (m < active_toks);
            const unsigned int tok = tok_base + (unsigned int)m;
            const float* xrow = active ? (swiglu + (unsigned long long)tok * inter_dim) : swiglu;
            const unsigned int offs = lane8 << 2;
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int kk = kk_base + (t << 1);
                const unsigned int kblk = k0 + (unsigned int)kk;
                const bool kvalid = (kblk < num_blocks);
                float v0=0.f,v1=0.f,v2=0.f,v3=0.f;
                if (active && kvalid) {
                    const unsigned int k_elem = kblk << 5;
                    const float* xp = xrow + k_elem + offs;
                    v0=xp[0]; v1=xp[1]; v2=xp[2]; v3=xp[3];
                }
                s_xf[kk][offs+0][m]=v0; s_xf[kk][offs+1][m]=v1;
                s_xf[kk][offs+2][m]=v2; s_xf[kk][offs+3][m]=v3;
            }
        }
        // Stage 2: stage down weight tile (nibble decode), 2 passes x 32 rows.
        {
            const int row0_local = group8;
            const int kk         = lane8;
            const unsigned int kblk = k0 + (unsigned int)kk;
            const bool kvalid = (kblk < num_blocks);
            #pragma unroll
            for (int pass = 0; pass < 2; ++pass) {
                const int row_local  = row0_local + (pass << 5);
                const unsigned int row_global = row_base + (unsigned int)row_local;
                signed char wb[32]; float wsc=0.f;
                if (row_global < hidden_dim && kvalid) {
                    const char* wblk = w_down + (unsigned long long)row_global * row_bytes
                        + (unsigned long long)kblk * SB_Q4_BLOCK_BYTES;
                    unsigned short sb = (unsigned short)(unsigned char)wblk[0]
                        | ((unsigned short)(unsigned char)wblk[1] << 8);
                    wsc = sb_f16_to_f32(sb);
                    sb_q4_0_decode32((const unsigned char*)(wblk + 2), wb);
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
        // Stage 3: compute.
        #pragma unroll
        for (int kk=0; kk<SBTD_BK; ++kk) {
            float bs0=0.f,bs1=0.f,bs2=0.f,bs3=0.f;
            #pragma unroll
            for (int j=0;j<32;++j) {
                const float xv = s_xf[kk][j][col];
                bs0 += (float)s_wb[kk][j][r0+0]*xv;
                bs1 += (float)s_wb[kk][j][r0+1]*xv;
                bs2 += (float)s_wb[kk][j][r0+2]*xv;
                bs3 += (float)s_wb[kk][j][r0+3]*xv;
            }
            a0 += s_ws[kk][r0+0]*bs0; a1 += s_ws[kk][r0+1]*bs1;
            a2 += s_ws[kk][r0+2]*bs2; a3 += s_ws[kk][r0+3]*bs3;
        }
        __syncthreads();
    }
    // Sigmoid-gated (or plain) accum into x_out.
    if (col < (unsigned int)active_toks) {
        const unsigned int tok = tok_base + col;
        float gate = 1.0f;
        if (gate_mode == 1) { const float s = logit[tok]; gate = 1.0f / (1.0f + expf(-s)); }
        const unsigned long long base = (unsigned long long)tok * hidden_dim + (row_base + r0);
        if (row_base + r0 + 0 < hidden_dim) x_out[base+0] += gate * a0;
        if (row_base + r0 + 1 < hidden_dim) x_out[base+1] += gate * a1;
        if (row_base + r0 + 2 < hidden_dim) x_out[base+2] += gate * a2;
        if (row_base + r0 + 3 < hidden_dim) x_out[base+3] += gate * a3;
    }
}
