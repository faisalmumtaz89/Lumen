// Native prefill attention for the full-attention layers: 24 query heads, 4 KV heads (query head h reads
// KV head h / 6), head size 256, RoPE over the first 64 dimensions with NeoX pairing (d, d + 32), causal
// from any start position p0. Compiled after native_prefill_common.cu as one NVRTC module for
// compute_120a, with no header.
//
// The KV cache stays F32, [4][max_seq][256] per layer. Attention reads a BF16 copy of it in the same
// layout (the staging): the prep kernel writes both for the new positions [p0, p0 + T), and
// native_kv_to_bf16 rounds the older positions [0, p0) to nearest even.
//
// Layouts: qg [T][24][512] BF16 (per head the query, then its gate), k and v [T][4][256] BF16, the RoPE
// table [pos][64] F32 (cos of pairs 0..31, then their sin), q and the attention output [T][24][256] BF16.

#define NATIVE_ATTN_HEADS 24
#define NATIVE_ATTN_KV_HEADS 4
#define NATIVE_ATTN_DIM 256
#define NATIVE_ATTN_ROT 64
// Query rows and keys per attention tile, and the BF16 row stride of a shared tile (8 elements of padding
// keep ldmatrix free of bank conflicts).
#define NATIVE_ATTN_TILE 64
#define NATIVE_ATTN_LD 264

// RoPE table rows [0, n_pos): the angle arithmetic of the F32 prefill's NeoX RoPE over 64 dimensions.
// Block 256, one thread per (position, pair).
extern "C" __global__ void native_rope_table(float* __restrict__ cs, unsigned int n_pos, float theta_base)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_pos * (NATIVE_ATTN_ROT / 2)) return;
    unsigned int pos = idx / (NATIVE_ATTN_ROT / 2);
    unsigned int d = idx % (NATIVE_ATTN_ROT / 2);
    unsigned int actual_rot = NATIVE_ATTN_ROT;

    float freq = 1.0f / powf(theta_base, (float)(2 * d) / (float)actual_rot);
    float angle = (float)pos * freq;
    cs[(unsigned long long)pos * NATIVE_ATTN_ROT + d] = cosf(angle);
    cs[(unsigned long long)pos * NATIVE_ATTN_ROT + NATIVE_ATTN_ROT / 2 + d] = sinf(angle);
}

// Per-head Gemma RMSNorm and RoPE of the queries and keys, the gate split from the queries, and the new
// KV rows. Grid (T, 28): y < 24 is query head y, y >= 24 is KV head y - 24. Block 128; thread t owns
// columns 2t and 2t + 1.
//   sum   = fma(x0, x0, x1 * x1) per thread; warp butterfly with offsets 16, 8, 4, 2, 1;
//           head sum = (warp0 + warp2) + (warp1 + warp3)
//   rstd  = rsqrt.approx.ftz(div.full(sum, 256) + eps)
//   n     = bf16((rstd * x) * w1), w1 the stored F32 (w + 1)
//   RoPE  of the pair (n1, n2) = (n[d], n[d + 32]), d < 32, at position p0 + t:
//         bf16(fma(n1, cos, -(n2 * sin))), bf16(fma(n1, sin, n2 * cos)); columns 64..255 keep n.
// The key's BF16 result goes to the staging and, widened exactly, to the F32 cache at row p0 + t; the
// value's BF16 input likewise. Nothing else is written.
extern "C" __global__ void __launch_bounds__(128) native_attn_prep(
    const unsigned short* __restrict__ qg,
    const unsigned short* __restrict__ k,
    const unsigned short* __restrict__ v,
    const float* __restrict__ q_w1,
    const float* __restrict__ k_w1,
    const float* __restrict__ cs,
    float eps,
    unsigned int p0,
    unsigned int max_seq,
    unsigned short* __restrict__ q_out,
    unsigned short* __restrict__ gate_out,
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    unsigned short* __restrict__ k_stage,
    unsigned short* __restrict__ v_stage)
{
    __shared__ float part[4];
    const unsigned int t = blockIdx.x;
    const unsigned int head = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int c = 2 * tid;
    const bool is_k = head >= NATIVE_ATTN_HEADS;
    const unsigned int hk = head - NATIVE_ATTN_HEADS;
    const unsigned short* in = is_k
        ? k + ((unsigned long long)t * NATIVE_ATTN_KV_HEADS + hk) * NATIVE_ATTN_DIM
        : qg + ((unsigned long long)t * NATIVE_ATTN_HEADS + head) * (2 * NATIVE_ATTN_DIM);
    const float* w1 = is_k ? k_w1 : q_w1;

    const unsigned int xp = *reinterpret_cast<const unsigned int*>(in + c);
    const float x0 = native_lo(xp);
    const float x1 = native_hi(xp);
    float sum = __fmaf_rn(x0, x0, __fmul_rn(x1, x1));
    for (int o = 16; o > 0; o >>= 1) {
        sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, o));
    }
    if (tid % 32 == 0) {
        part[tid / 32] = sum;
    }
    __syncthreads();
    const float total = __fadd_rn(__fadd_rn(part[0], part[2]), __fadd_rn(part[1], part[3]));
    const float rstd = native_rsqrt_approx_ftz(__fadd_rn(native_div_full(total, 256.0f), eps));
    unsigned int y = native_bf16x2_rn(__fmul_rn(__fmul_rn(rstd, x0), w1[c]), __fmul_rn(__fmul_rn(rstd, x1), w1[c + 1]));

    // Columns 0..63 are warp 0: lanes 0..15 hold the first of each pair, lanes 16..31 the second.
    if (c < NATIVE_ATTN_ROT) {
        const unsigned int other = __shfl_xor_sync(0xffffffffu, y, 16);
        const float* row = cs + (unsigned long long)(p0 + t) * NATIVE_ATTN_ROT;
        const unsigned int d = c % (NATIVE_ATTN_ROT / 2);
        float r[2];
#pragma unroll
        for (int e = 0; e < 2; e++) {
            const float self = e ? native_hi(y) : native_lo(y);
            const float peer = e ? native_hi(other) : native_lo(other);
            const float cv = row[d + e];
            const float sv = row[NATIVE_ATTN_ROT / 2 + d + e];
            r[e] = c < NATIVE_ATTN_ROT / 2 ? __fmaf_rn(self, cv, -__fmul_rn(peer, sv))
                                           : __fmaf_rn(peer, sv, __fmul_rn(self, cv));
        }
        y = native_bf16x2_rn(r[0], r[1]);
    }

    if (!is_k) {
        const unsigned long long o = ((unsigned long long)t * NATIVE_ATTN_HEADS + head) * NATIVE_ATTN_DIM + c;
        *reinterpret_cast<unsigned int*>(q_out + o) = y;
        *reinterpret_cast<unsigned int*>(gate_out + o) = *reinterpret_cast<const unsigned int*>(in + NATIVE_ATTN_DIM + c);
        return;
    }
    const unsigned long long kv_row = (unsigned long long)hk * max_seq + p0 + t;
    const unsigned long long o = kv_row * NATIVE_ATTN_DIM + c;
    const unsigned int vp =
        *reinterpret_cast<const unsigned int*>(v + ((unsigned long long)t * NATIVE_ATTN_KV_HEADS + hk) * NATIVE_ATTN_DIM + c);
    float2 kf;
    kf.x = native_lo(y);
    kf.y = native_hi(y);
    float2 vf;
    vf.x = native_lo(vp);
    vf.y = native_hi(vp);
    *reinterpret_cast<unsigned int*>(k_stage + o) = y;
    *reinterpret_cast<float2*>(k_cache + o) = kf;
    *reinterpret_cast<unsigned int*>(v_stage + o) = vp;
    *reinterpret_cast<float2*>(v_cache + o) = vf;
}

// Four F32 values rounded to nearest even as BF16, in order.
__device__ __forceinline__ uint2 native_bf16x4_rn(float4 a)
{
    uint2 r;
    r.x = native_bf16x2_rn(a.x, a.y);
    r.y = native_bf16x2_rn(a.z, a.w);
    return r;
}

// Staging of the cache positions [0, len) of all 4 KV heads, F32 -> BF16 to nearest even. Block 256,
// grid len: one thread per 4 values.
extern "C" __global__ void __launch_bounds__(256) native_kv_to_bf16(
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    unsigned short* __restrict__ k_stage,
    unsigned short* __restrict__ v_stage,
    unsigned int len,
    unsigned int max_seq)
{
    const unsigned long long per_head = (unsigned long long)len * (NATIVE_ATTN_DIM / 4);
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NATIVE_ATTN_KV_HEADS * per_head) return;
    const unsigned long long hk = i / per_head;
    const unsigned long long off = (i % per_head) * 4 + hk * max_seq * NATIVE_ATTN_DIM;
    const float4 a = *reinterpret_cast<const float4*>(k_cache + off);
    const float4 b = *reinterpret_cast<const float4*>(v_cache + off);
    *reinterpret_cast<uint2*>(k_stage + off) = native_bf16x4_rn(a);
    *reinterpret_cast<uint2*>(v_stage + off) = native_bf16x4_rn(b);
}

// Warp-level BF16 tensor-core pieces (mma.m16n8k16, ldmatrix, cp.async). Fragment layouts, g = lane / 4,
// t = lane % 4:
//   A (16x16, row):  a0 (g, 2t..2t+1)  a1 (g+8, 2t..)  a2 (g, 2t+8..)  a3 (g+8, 2t+8..)
//   B (16x8,  col):  b0 (k=2t..2t+1, n=g)  b1 (k=2t+8.., n=g)
//   C (16x8,  F32):  c0 (g, 2t) c1 (g, 2t+1) c2 (g+8, 2t) c3 (g+8, 2t+1)

__device__ __forceinline__ unsigned int native_smem_addr(const void* p)
{
    return (unsigned int)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void native_ldsm_x4(unsigned int (&r)[4], unsigned int addr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

__device__ __forceinline__ void native_ldsm_x4_t(unsigned int (&r)[4], unsigned int addr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

// A fragment (16x16) of a row-major [m][k] tile at (m0, k0).
__device__ __forceinline__ void native_lda(unsigned int (&a)[4], const unsigned short* base, int m0, int k0)
{
    const int lane = threadIdx.x & 31;
    native_ldsm_x4(a, native_smem_addr(base + (m0 + (lane & 15)) * NATIVE_ATTN_LD + k0 + ((lane >> 4) << 3)));
}

// Two B fragments (k 16 x n 16: n-tiles n0 in b[0], b[1] and n0 + 8 in b[2], b[3]) of B(k, n) stored [n][k].
__device__ __forceinline__ void native_ldb_nk(unsigned int (&b)[4], const unsigned short* base, int n0, int k0)
{
    const int lane = threadIdx.x & 31;
    native_ldsm_x4(
        b, native_smem_addr(base + (n0 + (lane & 7) + ((lane >> 4) << 3)) * NATIVE_ATTN_LD + k0 + (((lane >> 3) & 1) << 3)));
}

// The same two B fragments of B(k, n) stored [k][n].
__device__ __forceinline__ void native_ldb_kn(unsigned int (&b)[4], const unsigned short* base, int n0, int k0)
{
    const int lane = threadIdx.x & 31;
    native_ldsm_x4_t(
        b, native_smem_addr(base + (k0 + (lane & 7) + (((lane >> 3) & 1) << 3)) * NATIVE_ATTN_LD + n0 + ((lane >> 4) << 3)));
}

__device__ __forceinline__ void native_mma_bf16(float (&c)[4], const unsigned int (&a)[4], unsigned int b0, unsigned int b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 16-byte asynchronous copy global -> shared; src_bytes 0 writes zeros.
__device__ __forceinline__ void native_cp_async16(void* smem, const void* gmem, int src_bytes)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(native_smem_addr(smem)), "l"(gmem), "r"(src_bytes));
}

__device__ __forceinline__ void native_cp_async_commit()
{
    asm volatile("cp.async.commit_group;" ::);
}

__device__ __forceinline__ void native_cp_async_wait_all()
{
    asm volatile("cp.async.wait_all;" ::);
}

__device__ __forceinline__ float native_ex2_approx_ftz(float x)
{
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// Rows [0, NATIVE_ATTN_TILE) x 256 BF16 of `src` (row stride 256) into a shared tile; rows at or past
// `valid` are written as zeros without reading `src` there.
__device__ __forceinline__ void native_load_tile(unsigned short* dst, const unsigned short* src, int valid)
{
    for (int i = threadIdx.x; i < NATIVE_ATTN_TILE * (NATIVE_ATTN_DIM / 8); i += 128) {
        const int r = i / (NATIVE_ATTN_DIM / 8);
        const int c8 = (i % (NATIVE_ATTN_DIM / 8)) * 8;
        const bool ok = r < valid;
        native_cp_async16(&dst[r * NATIVE_ATTN_LD + c8], src + (unsigned long long)(ok ? r : 0) * NATIVE_ATTN_DIM + c8, ok ? 16 : 0);
    }
}

// Causal GQA attention over the staged keys [0, p0 + T), in F32 accumulation on BF16 tensor cores.
// Grid (24, ceil(T / 64)), block 128 (4 warps x 16 query rows), dynamic shared memory 3 x 64 x 264 x 2 =
// 101,376 bytes (the query, key and value tiles). Tiles are issued longest first. Per key tile:
//   s      = q . k (raw); keys after the query's position get s = -inf
//   m      = max(m, max over the tile of s)
//   p      = bf16(ex2.approx.ftz(fma(s, c, -(m * c)))), c = scale * log2(e)
//   rescale the running output and denominator by ex2.approx.ftz(fma(m_old, c, -(m * c)))
//   l      = l * rescale + sum of the rounded p; o += p . v
// out = bf16(o * rcp.approx.ftz(l)).
extern "C" __global__ void __launch_bounds__(128, 1) native_attn_prefill(
    const unsigned short* __restrict__ q,
    const unsigned short* __restrict__ k_stage,
    const unsigned short* __restrict__ v_stage,
    unsigned short* __restrict__ out,
    unsigned int t_len,
    unsigned int p0,
    unsigned int max_seq,
    float scale_log2)
{
    extern __shared__ __align__(16) unsigned char native_attn_smem[];
    unsigned short* sq = reinterpret_cast<unsigned short*>(native_attn_smem);
    unsigned short* sk = sq + NATIVE_ATTN_TILE * NATIVE_ATTN_LD;
    unsigned short* sv = sk + NATIVE_ATTN_TILE * NATIVE_ATTN_LD;
    const float neg_inf = __int_as_float(0xff800000);

    const int mt = gridDim.y - 1 - blockIdx.y;
    const int h = blockIdx.x;
    const int hk = h / 6;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, tq = lane & 3;
    const int m0 = mt * NATIVE_ATTN_TILE;
    const int rows = min(NATIVE_ATTN_TILE, (int)t_len - m0);
    const int lkv = (int)(p0 + t_len);
    const int kv_end = min((int)p0 + m0 + rows, lkv);
    const int ntiles = (kv_end + NATIVE_ATTN_TILE - 1) / NATIVE_ATTN_TILE;
    const unsigned short* kbase = k_stage + (unsigned long long)hk * max_seq * NATIVE_ATTN_DIM;
    const unsigned short* vbase = v_stage + (unsigned long long)hk * max_seq * NATIVE_ATTN_DIM;

    for (int i = tid; i < NATIVE_ATTN_TILE * (NATIVE_ATTN_DIM / 8); i += 128) {
        const int r = i / (NATIVE_ATTN_DIM / 8);
        const int c8 = (i % (NATIVE_ATTN_DIM / 8)) * 8;
        const bool ok = r < rows;
        const unsigned long long src = ((unsigned long long)(m0 + (ok ? r : 0)) * NATIVE_ATTN_HEADS + h) * NATIVE_ATTN_DIM + c8;
        native_cp_async16(&sq[r * NATIVE_ATTN_LD + c8], q + src, ok ? 16 : 0);
    }
    native_load_tile(sk, kbase, lkv);
    native_cp_async_commit();

    const int r0 = warp * 16;
    const int qp[2] = {(int)p0 + m0 + r0 + g, (int)p0 + m0 + r0 + g + 8};
    float o[NATIVE_ATTN_DIM / 8][4];
#pragma unroll
    for (int a = 0; a < NATIVE_ATTN_DIM / 8; a++)
#pragma unroll
        for (int e = 0; e < 4; e++) o[a][e] = 0.0f;
    float mrow[2] = {neg_inf, neg_inf};
    float lrow[2] = {0.0f, 0.0f};

    for (int j = 0; j < ntiles; j++) {
        const int kv0 = j * NATIVE_ATTN_TILE;
        native_cp_async_wait_all();
        __syncthreads(); // K_j (and Q) visible; every warp is done with V_{j-1}
        native_load_tile(sv, vbase + (unsigned long long)kv0 * NATIVE_ATTN_DIM, lkv - kv0);
        native_cp_async_commit();
        float sc[NATIVE_ATTN_TILE / 8][4];
#pragma unroll
        for (int a = 0; a < NATIVE_ATTN_TILE / 8; a++)
#pragma unroll
            for (int e = 0; e < 4; e++) sc[a][e] = 0.0f;
#pragma unroll
        for (int k0 = 0; k0 < NATIVE_ATTN_DIM; k0 += 16) {
            unsigned int a[4];
            native_lda(a, sq, r0, k0);
#pragma unroll
            for (int np = 0; np < NATIVE_ATTN_TILE / 16; np++) {
                unsigned int bb[4];
                native_ldb_nk(bb, sk, np * 16, k0);
                native_mma_bf16(sc[2 * np], a, bb[0], bb[1]);
                native_mma_bf16(sc[2 * np + 1], a, bb[2], bb[3]);
            }
        }
        float mx[2] = {mrow[0], mrow[1]};
#pragma unroll
        for (int nt = 0; nt < NATIVE_ATTN_TILE / 8; nt++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int key = kv0 + nt * 8 + 2 * tq + (e & 1), r = e >> 1;
                const bool allowed = key <= qp[r];
                sc[nt][e] = allowed ? sc[nt][e] : neg_inf;
                mx[r] = fmaxf(mx[r], sc[nt][e]);
            }
        float alpha[2];
        float mc[2];
#pragma unroll
        for (int r = 0; r < 2; r++) {
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
            mc[r] = -__fmul_rn(mx[r], scale_log2);
            alpha[r] = native_ex2_approx_ftz(__fmaf_rn(mrow[r], scale_log2, mc[r]));
            mrow[r] = mx[r];
        }
        unsigned int pa[NATIVE_ATTN_TILE / 16][4];
        float ls[2] = {0.0f, 0.0f};
#pragma unroll
        for (int nt = 0; nt < NATIVE_ATTN_TILE / 8; nt++) {
            const unsigned int lo = native_bf16x2_rn(native_ex2_approx_ftz(__fmaf_rn(sc[nt][0], scale_log2, mc[0])),
                                                     native_ex2_approx_ftz(__fmaf_rn(sc[nt][1], scale_log2, mc[0])));
            const unsigned int hi = native_bf16x2_rn(native_ex2_approx_ftz(__fmaf_rn(sc[nt][2], scale_log2, mc[1])),
                                                     native_ex2_approx_ftz(__fmaf_rn(sc[nt][3], scale_log2, mc[1])));
            ls[0] = __fadd_rn(ls[0], __fadd_rn(native_lo(lo), native_hi(lo)));
            ls[1] = __fadd_rn(ls[1], __fadd_rn(native_lo(hi), native_hi(hi)));
            pa[nt >> 1][(nt & 1) * 2 + 0] = lo;
            pa[nt >> 1][(nt & 1) * 2 + 1] = hi;
        }
#pragma unroll
        for (int r = 0; r < 2; r++) lrow[r] = __fadd_rn(__fmul_rn(lrow[r], alpha[r]), ls[r]);
#pragma unroll
        for (int a = 0; a < NATIVE_ATTN_DIM / 8; a++) {
            o[a][0] = __fmul_rn(o[a][0], alpha[0]);
            o[a][1] = __fmul_rn(o[a][1], alpha[0]);
            o[a][2] = __fmul_rn(o[a][2], alpha[1]);
            o[a][3] = __fmul_rn(o[a][3], alpha[1]);
        }
        native_cp_async_wait_all();
        __syncthreads(); // V_j visible; every warp is done with K_j
        if (j + 1 < ntiles) {
            const int kv1 = kv0 + NATIVE_ATTN_TILE;
            native_load_tile(sk, kbase + (unsigned long long)kv1 * NATIVE_ATTN_DIM, lkv - kv1);
            native_cp_async_commit();
        }
#pragma unroll
        for (int kk = 0; kk < NATIVE_ATTN_TILE / 16; kk++)
#pragma unroll
            for (int np = 0; np < NATIVE_ATTN_DIM / 16; np++) {
                unsigned int bb[4];
                native_ldb_kn(bb, sv, np * 16, kk * 16);
                native_mma_bf16(o[2 * np], pa[kk], bb[0], bb[1]);
                native_mma_bf16(o[2 * np + 1], pa[kk], bb[2], bb[3]);
            }
    }
    float inv[2];
#pragma unroll
    for (int r = 0; r < 2; r++) {
        float l = lrow[r];
        l = __fadd_rn(l, __shfl_xor_sync(0xffffffffu, l, 1));
        l = __fadd_rn(l, __shfl_xor_sync(0xffffffffu, l, 2));
        inv[r] = native_rcp_approx_ftz(l);
    }
#pragma unroll
    for (int a = 0; a < NATIVE_ATTN_DIM / 8; a++)
#pragma unroll
        for (int r = 0; r < 2; r++) {
            const int i = r0 + g + r * 8;
            const int d = a * 8 + 2 * tq;
            if (i < rows) {
                *reinterpret_cast<unsigned int*>(out + ((unsigned long long)(m0 + i) * NATIVE_ATTN_HEADS + h) * NATIVE_ATTN_DIM + d) =
                    native_bf16x2_rn(__fmul_rn(o[a][2 * r], inv[r]), __fmul_rn(o[a][2 * r + 1], inv[r]));
            }
        }
}
