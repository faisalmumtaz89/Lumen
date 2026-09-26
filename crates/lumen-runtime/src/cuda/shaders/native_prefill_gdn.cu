// Native prefill GDN kernels: the causal conv1d of every token at once, and the gated delta rule over
// 64-token chunks on BF16 tensor cores (mma.sync m16n8k16, F32 accumulation). Compiled after
// native_prefill_common.cu, with no header.
//
// Geometry: 48 value heads, 16 key heads (value head h reads key head h % 16), head size 128, conv kernel 4
// over 10240 channels (q 2048 | k 2048 | v 6144). The layer state is decode's, updated in place:
//   conv ring  [3][10240] F32; slot (state_pos + s) % 3 holds the input of token s (s < 0: earlier tokens)
//   h_state    [48][128][128] F32, [head][value][key]
//
// One layer is three launches:
//   native_gdn_conv_silu_l2      per channel and 32 tokens: the conv (taps oldest first, F32 FMAs), SiLU
//                                x / (1 + e^-x) to BF16; q and k heads L2-normalized over the rounded values
//                                with scale 1 / max(norm, 1e-12), BF16 again. Writes the ring's new inputs.
//   native_gdn_chunk_intra       per (chunk, value head): gate = ssm_a * softplus(bf16(a) + dt_bias) (the
//                                identity above 20) and its in-chunk cumulative sum G; beta = bf16(sigmoid(bf16(b)));
//                                A = beta_i k_i.k_j e^(G_i - G_j) (j < i); X = (I + A)^-1 by F32 forward
//                                substitution; W = X (k beta e^G), U = X (v beta), Aqk = q_i.k_j e^(G_i - G_j) (j <= i).
//   native_gdn_chunk_state_bv48  per (value head, 48-row slice of the state), chunk by chunk:
//                                v_new = U - W S^T; o = (e^G q S^T + Aqk v_new) * rsqrt(128);
//                                S <- e^(G_last) S + (v_new e^(G_last - G))^T k.
// BF16 is held at: a, b, the conv output and the normalized q/k, beta, X, k beta e^G, v beta, W, U, the chunk's
// state snapshot in products, v_new, v_new e^(G_last - G), Aqk and the output. The state, gates, G, every
// accumulation and the solve are F32. Scratch layouts: G [T][48] F32; W, U [T][48][128], Aqk [T][48][64] and
// the output [T][48][128] BF16; a and b are the F32 rows [T][96] (a in columns 0-47, b in 48-95).

#define NATIVE_GDN_H 48
#define NATIVE_GDN_HK 16
#define NATIVE_GDN_D 128
#define NATIVE_GDN_QK 2048
#define NATIVE_GDN_CONV 10240
#define NATIVE_GDN_AB 96
#define NATIVE_GDN_BT 64
#define NATIVE_GDN_CONV_TT 32
#define NATIVE_GDN_LDK 136 // BF16 row stride of the 128-wide tiles (conflict-free ldmatrix)
#define NATIVE_GDN_LDA 72  // BF16 row stride of the 64-wide tiles
#define NATIVE_GDN_LDE 68  // F32 row stride of the parity-split A
#define NATIVE_GDN_ODD 36  // offset of A's odd columns within its row
#define NATIVE_GDN_BV 48
#define NATIVE_GDN_LDV 56  // BF16 row stride of the 48-wide value tiles

__device__ __forceinline__ float native_gdn_rbf(float x)
{
    return native_lo(native_bf16x2_rn(x, 0.0f));
}

__device__ __forceinline__ float native_gdn_warp_sum(float v)
{
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    return v;
}

__device__ __forceinline__ unsigned int native_gdn_smem(const void* p)
{
    return (unsigned int)__cvta_generic_to_shared(p);
}

// ---------------------------------------------------------------------------------------------------------
// Tensor-core fragments of mma.m16n8k16 (g = lane / 4, t = lane % 4):
//   A (16x16, row): a0 (g, 2t..2t+1)  a1 (g+8, 2t..)  a2 (g, 2t+8..)  a3 (g+8, 2t+8..)
//   B (16x8, col):  b0 (k 2t..2t+1, n g)  b1 (k 2t+8.., n g)
//   C (16x8, F32):  c0 (g, 2t) c1 (g, 2t+1) c2 (g+8, 2t) c3 (g+8, 2t+1)
// Each loader takes the shared-memory base of a row-major BF16 tile and its row stride in elements.

__device__ __forceinline__ void native_gdn_ldsm(unsigned int (&r)[4], unsigned int addr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

__device__ __forceinline__ void native_gdn_ldsm_t(unsigned int (&r)[4], unsigned int addr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

// A fragment of the [m][k] tile at (m0, k0).
__device__ __forceinline__ void native_gdn_lda(unsigned int (&a)[4], const unsigned short* base, int ld, int m0,
                                               int k0)
{
    const int lane = threadIdx.x & 31;
    native_gdn_ldsm(a, native_gdn_smem(base + (m0 + (lane & 15)) * ld + k0 + ((lane >> 4) << 3)));
}

// A fragment of A(m, k) = tile[k][m].
__device__ __forceinline__ void native_gdn_lda_t(unsigned int (&a)[4], const unsigned short* base, int ld, int m0,
                                                 int k0)
{
    const int lane = threadIdx.x & 31;
    native_gdn_ldsm_t(a, native_gdn_smem(base + (k0 + (lane & 7) + ((lane >> 4) << 3)) * ld + m0
                                         + (((lane >> 3) & 1) << 3)));
}

// B fragments of B(k, n) = tile[n][k] for n-tiles n0 (b[0], b[1]) and n0 + 8 (b[2], b[3]).
__device__ __forceinline__ void native_gdn_ldb_nk(unsigned int (&b)[4], const unsigned short* base, int ld, int n0,
                                                  int k0)
{
    const int lane = threadIdx.x & 31;
    native_gdn_ldsm(b, native_gdn_smem(base + (n0 + (lane & 7) + ((lane >> 4) << 3)) * ld + k0
                                       + (((lane >> 3) & 1) << 3)));
}

// B fragments of B(k, n) = tile[k][n], in the order of native_gdn_ldb_nk.
__device__ __forceinline__ void native_gdn_ldb_kn(unsigned int (&b)[4], const unsigned short* base, int ld, int n0,
                                                  int k0)
{
    const int lane = threadIdx.x & 31;
    native_gdn_ldsm_t(b, native_gdn_smem(base + (k0 + (lane & 7) + (((lane >> 3) & 1) << 3)) * ld + n0
                                         + ((lane >> 4) << 3)));
}

__device__ __forceinline__ void native_gdn_mma(float (&c)[4], const unsigned int (&a)[4], unsigned int b0,
                                               unsigned int b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Asynchronous global -> shared copies; a source size of 0 fills the destination with zeros.
__device__ __forceinline__ void native_gdn_cp16(void* smem, const void* gmem, int src_bytes)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(native_gdn_smem(smem)), "l"(gmem),
                 "r"(src_bytes));
}

__device__ __forceinline__ void native_gdn_cp4(void* smem, const void* gmem, int src_bytes)
{
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(native_gdn_smem(smem)), "l"(gmem),
                 "r"(src_bytes));
}

__device__ __forceinline__ void native_gdn_cp_commit()
{
    asm volatile("cp.async.commit_group;");
}

template <int N> __device__ __forceinline__ void native_gdn_cp_wait()
{
    asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

// ---------------------------------------------------------------------------------------------------------
// Conv + SiLU (+ L2 on the q/k heads). Grid (80 heads, ceil(T / 32)), block 128: thread = channel of the head,
// 32 consecutive tokens per block. Only the first token block reads the ring (32 >= 3 taps back), and each of
// its threads writes the ring's new inputs of its own channel after reading them, so the update is in place.
extern "C" __global__ void __launch_bounds__(128) native_gdn_conv_silu_l2(
    const unsigned short* __restrict__ qkv, // [T][10240] BF16
    float* ring,                            // [3][10240] F32, read then updated
    const float* __restrict__ conv_w,       // [10240][4] F32
    unsigned short* __restrict__ out,       // [T][10240] BF16
    int T,
    int state_pos)
{
    __shared__ float part[NATIVE_GDN_CONV_TT][4];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int head = blockIdx.x, c = head * NATIVE_GDN_D + tid, t0 = blockIdx.y * NATIVE_GDN_CONV_TT;
    const bool is_qk = head < 2 * NATIVE_GDN_HK;
    const float w0 = conv_w[c * 4 + 0], w1 = conv_w[c * 4 + 1], w2 = conv_w[c * 4 + 2], w3 = conv_w[c * 4 + 3];
    float xm[3];
#pragma unroll
    for (int k = 0; k < 3; k++) {
        const int s = t0 - 3 + k;
        xm[k] = s >= 0 ? native_bf16_to_f32(qkv[(size_t)s * NATIVE_GDN_CONV + c])
                       : ring[((state_pos + s + 3) % 3) * NATIVE_GDN_CONV + c];
    }
    float y[NATIVE_GDN_CONV_TT];
#pragma unroll
    for (int i = 0; i < NATIVE_GDN_CONV_TT; i++) {
        const int t = t0 + i;
        const float x = t < T ? native_bf16_to_f32(qkv[(size_t)t * NATIVE_GDN_CONV + c]) : 0.0f;
        float s = __fmaf_rn(w0, xm[0], 0.0f);
        s = __fmaf_rn(w1, xm[1], s);
        s = __fmaf_rn(w2, xm[2], s);
        s = __fmaf_rn(w3, x, s);
        y[i] = t < T ? native_gdn_rbf(s / (1.0f + expf(-s))) : 0.0f;
        xm[0] = xm[1];
        xm[1] = xm[2];
        xm[2] = x;
    }
    if (is_qk) {
        // Each thread holds one element of the head: warp sums, then the four warps' sums combined as
        // (w0 + w2) + (w1 + w3), the order of decode's L2 norm.
#pragma unroll
        for (int i = 0; i < NATIVE_GDN_CONV_TT; i++) {
            const float ss = native_gdn_warp_sum(__fmul_rn(y[i], y[i]));
            if (lane == 0) {
                part[i][warp] = ss;
            }
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < NATIVE_GDN_CONV_TT; i++) {
            const int t = t0 + i;
            if (t < T) {
                const float ss = (part[i][0] + part[i][2]) + (part[i][1] + part[i][3]);
                const float norm = sqrtf(ss);
                const float scale = norm > 1e-12f ? 1.0f / norm : 1.0f / 1e-12f;
                out[(size_t)t * NATIVE_GDN_CONV + c] = (unsigned short)native_bf16x2_rn(y[i] * scale, 0.0f);
            }
        }
    } else {
#pragma unroll
        for (int i = 0; i < NATIVE_GDN_CONV_TT; i++) {
            const int t = t0 + i;
            if (t < T) {
                out[(size_t)t * NATIVE_GDN_CONV + c] = (unsigned short)native_bf16x2_rn(y[i], 0.0f);
            }
        }
    }
    if (blockIdx.y == 0) {
        // Token s goes to slot (state_pos + s) % 3; slots no new token reaches (T < 3) keep their input.
#pragma unroll
        for (int k = 1; k <= 3; k++) {
            const int s = T - k;
            if (s >= 0) {
                ring[((state_pos + s) % 3) * NATIVE_GDN_CONV + c] =
                    native_bf16_to_f32(qkv[(size_t)s * NATIVE_GDN_CONV + c]);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// In-chunk kernel. Grid (ceil(T / 64), 48), block 128 (4 warps), two blocks per multiprocessor.
//   * X = (I + A)^-1 by column-parallel forward substitution in registers: the thread pair (j, half) owns column
//     j; half h keeps x_k for k = 2k' + h and sums its parity's terms from float4 reads of A shared by all lanes;
//     the halves combine by one shuffle per row. A is stored parity-split (even columns at [0, 32), odd at
//     [36, 68) of a 68-float row), so both halves read unit-stride and in different banks.
//   * The 64 gates are summed by a warp scan.
//   * The 20 lower-triangle 16x16 blocks of K K^T and Q K^T are dealt 5 per warp; W and U split by output columns.
//   * V is prefetched into registers and stored as V beta once the solve has freed its shared memory.
struct native_gdn_intra_smem {
    unsigned short k[NATIVE_GDN_BT * NATIVE_GDN_LDK]; // K, then K beta e^G in place
    union {
        unsigned short q[NATIVE_GDN_BT * NATIVE_GDN_LDK];  // Q
        float a[NATIVE_GDN_BT * NATIVE_GDN_LDE];           // then A (strictly lower), parity-split
        unsigned short vb[NATIVE_GDN_BT * NATIVE_GDN_LDK]; // then V beta
    } r2;
    unsigned short x[NATIVE_GDN_BT * NATIVE_GDN_LDA]; // X in BF16
    float g[NATIVE_GDN_BT], b[NATIVE_GDN_BT], eg[NATIVE_GDN_BT];
};
static_assert(sizeof(native_gdn_intra_smem) == 44800, "native_gdn_chunk_intra's dynamic shared memory");

extern "C" __global__ void __launch_bounds__(128, 2) native_gdn_chunk_intra(
    const unsigned short* __restrict__ cv, // [T][10240] BF16, the conv output
    const float* __restrict__ ab,          // [T][96] F32
    const float* __restrict__ dt_bias,     // [48]
    const float* __restrict__ ssm_a,       // [48], -exp(A_log)
    float* __restrict__ gc,                // [T][48] G
    unsigned short* __restrict__ wg,       // [T][48][128] W
    unsigned short* __restrict__ ug,       // [T][48][128] U
    unsigned short* __restrict__ aqk,      // [T][48][64]
    int T)
{
    extern __shared__ __align__(16) unsigned char native_gdn_smem_raw[];
    native_gdn_intra_smem& s = *reinterpret_cast<native_gdn_intra_smem*>(native_gdn_smem_raw);
    const int h = blockIdx.y, kh = h % NATIVE_GDN_HK;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, tq = lane & 3;
    const int t0 = blockIdx.x * NATIVE_GDN_BT, n = min(NATIVE_GDN_BT, T - t0);

    for (int i = tid; i < NATIVE_GDN_BT * 16; i += 128) {
        const int r = i >> 4, c8 = (i & 15) * 8;
        const bool ok = r < n;
        const unsigned short* row = cv + (size_t)(ok ? t0 + r : 0) * NATIVE_GDN_CONV;
        native_gdn_cp16(&s.k[r * NATIVE_GDN_LDK + c8], row + NATIVE_GDN_QK + kh * NATIVE_GDN_D + c8, ok ? 16 : 0);
        native_gdn_cp16(&s.r2.q[r * NATIVE_GDN_LDK + c8], row + kh * NATIVE_GDN_D + c8, ok ? 16 : 0);
    }
    native_gdn_cp_commit();
    // V rows tid / 16 + 8m, 8 columns at (tid % 16) * 8, kept in registers until the solve is done.
    uint4 vreg[8];
#pragma unroll
    for (int m = 0; m < 8; m++) {
        const int r = (tid >> 4) + 8 * m, c8 = (tid & 15) * 8;
        vreg[m] = r < n ? *reinterpret_cast<const uint4*>(cv + (size_t)(t0 + r) * NATIVE_GDN_CONV
                                                          + 2 * NATIVE_GDN_QK + h * NATIVE_GDN_D + c8)
                        : make_uint4(0, 0, 0, 0);
    }
    if (tid < NATIVE_GDN_BT) {
        float gate = 0.0f, beta = 0.0f;
        if (tid < n) {
            const float* abrow = ab + (size_t)(t0 + tid) * NATIVE_GDN_AB;
            const float sp_input = native_gdn_rbf(abrow[h]) + dt_bias[h];
            const float sp = sp_input > 20.0f ? sp_input : logf(1.0f + expf(sp_input));
            gate = ssm_a[h] * sp;
            beta = native_gdn_rbf(1.0f / (1.0f + expf(-native_gdn_rbf(abrow[NATIVE_GDN_H + h]))));
        }
        s.g[tid] = gate;
        s.b[tid] = beta;
    }
    __syncthreads();
    if (warp == 0) { // inclusive sum of the 64 gates: lane owns 2 lane and 2 lane + 1
        const float a0 = s.g[2 * lane], a1 = a0 + s.g[2 * lane + 1];
        float x = a1;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const float y = __shfl_up_sync(0xffffffffu, x, o);
            if (lane >= o) {
                x += y;
            }
        }
        s.g[2 * lane] = (x - a1) + a0;
        s.g[2 * lane + 1] = x;
    }
    __syncthreads();
    if (tid < NATIVE_GDN_BT) {
        s.eg[tid] = expf(s.g[tid]);
        if (tid < n) {
            gc[(size_t)(t0 + tid) * NATIVE_GDN_H + h] = s.g[tid];
        }
    }
    native_gdn_cp_wait<0>();
    __syncthreads();

    // K K^T and Q K^T: 20 jobs (10 lower-triangle 16x16 blocks x {K, Q}), 5 per warp.
    float acc[5][2][4];
#pragma unroll
    for (int jj = 0; jj < 5; jj++) {
        const int job = warp + 4 * jj, blk = job >> 1, isq = job & 1;
        const int m = blk >= 6 ? 3 : (blk >= 3 ? 2 : (blk >= 1 ? 1 : 0)), c = blk - m * (m + 1) / 2;
        const unsigned short* asrc = isq ? s.r2.q : s.k;
#pragma unroll
        for (int e = 0; e < 4; e++) {
            acc[jj][0][e] = acc[jj][1][e] = 0.0f;
        }
#pragma unroll
        for (int k0 = 0; k0 < NATIVE_GDN_D; k0 += 16) {
            unsigned int a[4], bb[4];
            native_gdn_lda(a, asrc, NATIVE_GDN_LDK, 16 * m, k0);
            native_gdn_ldb_nk(bb, s.k, NATIVE_GDN_LDK, 16 * c, k0);
            native_gdn_mma(acc[jj][0], a, bb[0], bb[1]);
            native_gdn_mma(acc[jj][1], a, bb[2], bb[3]);
        }
    }
    __syncthreads(); // every read of Q, and of K as an operand, is done
#pragma unroll
    for (int jj = 0; jj < 5; jj++) {
        const int job = warp + 4 * jj, blk = job >> 1, isq = job & 1;
        const int m = blk >= 6 ? 3 : (blk >= 3 ? 2 : (blk >= 1 ? 1 : 0)), c = blk - m * (m + 1) / 2;
#pragma unroll
        for (int nt = 0; nt < 2; nt++) {
#pragma unroll
            for (int hf = 0; hf < 2; hf++) {
                const int i = 16 * m + g + hf * 8, j = 16 * c + 8 * nt + 2 * tq;
                float q2[2] = {0.0f, 0.0f};
#pragma unroll
                for (int u = 0; u < 2; u++) {
                    const int jc = j + u;
                    if (jc <= i) {
                        const float ex = expf(s.g[i] - s.g[jc]);
                        if (isq) {
                            q2[u] = acc[jj][nt][hf * 2 + u] * ex;
                        } else if (jc < i) {
                            const float a = acc[jj][nt][hf * 2 + u] * ex;
                            s.r2.a[i * NATIVE_GDN_LDE + (jc & 1) * NATIVE_GDN_ODD + (jc >> 1)] = a * s.b[i];
                        }
                    }
                }
                if (isq && t0 + i < T) {
                    *reinterpret_cast<unsigned int*>(&aqk[((size_t)(t0 + i) * NATIVE_GDN_H + h) * NATIVE_GDN_BT + j]) =
                        native_bf16x2_rn(q2[0], q2[1]);
                }
            }
        }
    }
    // The upper-triangle blocks of Aqk are zero: (0,1) (0,2) (0,3) (1,2) (1,3) (2,3).
    for (int idx = tid; idx < 6 * 16 * 8; idx += 128) {
        const int bk = idx >> 7, r = (idx >> 3) & 15, pr = idx & 7;
        const int m = bk < 3 ? 0 : (bk < 5 ? 1 : 2), c = bk < 3 ? bk + 1 : (bk < 5 ? bk - 1 : 3);
        const int i = 16 * m + r, j = 16 * c + 2 * pr;
        if (t0 + i < T) {
            *reinterpret_cast<unsigned int*>(&aqk[((size_t)(t0 + i) * NATIVE_GDN_H + h) * NATIVE_GDN_BT + j]) = 0u;
        }
    }
    for (int i = tid; i < NATIVE_GDN_BT * NATIVE_GDN_D / 2; i += 128) {
        const int r = i / (NATIVE_GDN_D / 2), d = (i % (NATIVE_GDN_D / 2)) * 2;
        unsigned int* p = reinterpret_cast<unsigned int*>(&s.k[r * NATIVE_GDN_LDK + d]);
        const float bb = s.b[r], e = s.eg[r];
        *p = native_bf16x2_rn((native_lo(*p) * bb) * e, (native_hi(*p) * bb) * e);
    }
    __syncthreads();

    // X column j by the thread pair (j, half). Row i: x_i = -sum_{k<i} A[i][k] x_k (x_j = 1, x_i = 0 above the
    // diagonal). Half h sums its parity's terms k = 2k' + h: k' < i / 2 for both halves, and k = i - 1 (even)
    // for half 0 when i is odd.
    {
        const int j = tid >> 1, half = tid & 1;
        float xs[NATIVE_GDN_BT / 2];
#pragma unroll
        for (int e = 0; e < NATIVE_GDN_BT / 2; e++) {
            xs[e] = 0.0f;
        }
#pragma unroll
        for (int i = 0; i < NATIVE_GDN_BT; i++) {
            const float* arow = s.r2.a + i * NATIVE_GDN_LDE + half * NATIVE_GDN_ODD;
            const int nk = i >> 1;
            float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
#pragma unroll
            for (int k4 = 0; k4 < nk / 4; k4++) {
                const float4 a4 = *reinterpret_cast<const float4*>(arow + 4 * k4);
                c0 = __fmaf_rn(a4.x, xs[4 * k4], c0);
                c1 = __fmaf_rn(a4.y, xs[4 * k4 + 1], c1);
                c2 = __fmaf_rn(a4.z, xs[4 * k4 + 2], c2);
                c3 = __fmaf_rn(a4.w, xs[4 * k4 + 3], c3);
            }
#pragma unroll
            for (int k = (nk / 4) * 4; k < nk; k++) {
                c0 = __fmaf_rn(arow[k], xs[k], c0);
            }
            float part = (c0 + c1) + (c2 + c3);
            if (i & 1) {
                const float extra = __fmul_rn(s.r2.a[i * NATIVE_GDN_LDE + nk], xs[nk]); // k = i - 1: half 0's term
                part += half == 0 ? extra : 0.0f;
            }
            const float tot = part + __shfl_xor_sync(0xffffffffu, part, 1);
            const float xi = i < j ? 0.0f : (i == j ? 1.0f : -tot);
            xs[i >> 1] = (i & 1) == half ? xi : xs[i >> 1];
        }
#pragma unroll
        for (int e = 0; e < NATIVE_GDN_BT / 2; e++) {
            s.x[(2 * e + half) * NATIVE_GDN_LDA + j] = (unsigned short)native_bf16x2_rn(xs[e], 0.0f);
        }
    }
    __syncthreads(); // X is complete; region 2 is free

    // V beta from the prefetched registers.
#pragma unroll
    for (int m = 0; m < 8; m++) {
        const int r = (tid >> 4) + 8 * m, c8 = (tid & 15) * 8;
        const float bb = s.b[r];
        const unsigned int pv[4] = {vreg[m].x, vreg[m].y, vreg[m].z, vreg[m].w};
        unsigned int po[4];
#pragma unroll
        for (int e = 0; e < 4; e++) {
            po[e] = native_bf16x2_rn(native_lo(pv[e]) * bb, native_hi(pv[e]) * bb);
        }
        *reinterpret_cast<uint4*>(&s.r2.vb[r * NATIVE_GDN_LDK + c8]) = make_uint4(po[0], po[1], po[2], po[3]);
    }
    __syncthreads();

    // W = X (K beta e^G), U = X (V beta): warp w owns output columns 32w..32w+31 of all 64 rows.
#pragma unroll
    for (int pass = 0; pass < 2; pass++) {
        const unsigned short* src = pass == 0 ? s.k : s.r2.vb;
        unsigned short* dst = pass == 0 ? wg : ug;
        float o[4][4][4];
#pragma unroll
        for (int a = 0; a < 4; a++) {
#pragma unroll
            for (int b = 0; b < 4; b++) {
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    o[a][b][e] = 0.0f;
                }
            }
        }
#pragma unroll
        for (int mt = 0; mt < 4; mt++) {
#pragma unroll
            for (int kb = 0; kb <= mt; kb++) {
                unsigned int a[4];
                native_gdn_lda(a, s.x, NATIVE_GDN_LDA, 16 * mt, 16 * kb);
#pragma unroll
                for (int np = 0; np < 2; np++) {
                    unsigned int bb[4];
                    native_gdn_ldb_kn(bb, src, NATIVE_GDN_LDK, 32 * warp + 16 * np, 16 * kb);
                    native_gdn_mma(o[mt][2 * np], a, bb[0], bb[1]);
                    native_gdn_mma(o[mt][2 * np + 1], a, bb[2], bb[3]);
                }
            }
        }
#pragma unroll
        for (int mt = 0; mt < 4; mt++) {
#pragma unroll
            for (int nt = 0; nt < 4; nt++) {
#pragma unroll
                for (int hf = 0; hf < 2; hf++) {
                    const int i = 16 * mt + g + hf * 8, d = 32 * warp + 8 * nt + 2 * tq;
                    if (t0 + i < T) {
                        *reinterpret_cast<unsigned int*>(&dst[((size_t)(t0 + i) * NATIVE_GDN_H + h) * NATIVE_GDN_D + d]) =
                            native_bf16x2_rn(o[mt][nt][2 * hf], o[mt][nt][2 * hf + 1]);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// Inter-chunk recurrence fused with the output. Grid (3, 48), block 128: block (x, h) owns state rows
// [48x, 48x + 48) of head h (value index, the last slice holding 32), all 128 key columns. Warp w computes
// output rows 16w..16w+15 of each chunk and state columns 32w..32w+31. The state stays in registers across
// the chunks; each block reads its slice before writing it and the slices are disjoint, so the update is in
// place. Rows past 128 in the last slice stay zero and are never stored.
struct native_gdn_state_smem {
    unsigned short w[NATIVE_GDN_BT * NATIVE_GDN_LDK], q[NATIVE_GDN_BT * NATIVE_GDN_LDK], k[NATIVE_GDN_BT * NATIVE_GDN_LDK];
    unsigned short u[NATIVE_GDN_BT * NATIVE_GDN_LDV], vn[NATIVE_GDN_BT * NATIVE_GDN_LDV], vd[NATIVE_GDN_BT * NATIVE_GDN_LDV];
    unsigned short a[NATIVE_GDN_BT * NATIVE_GDN_LDA];
    unsigned short sb[NATIVE_GDN_BV * NATIVE_GDN_LDK]; // the chunk's state snapshot, [value][key]
    float g[NATIVE_GDN_BT];
};
static_assert(sizeof(native_gdn_state_smem) == 96256, "native_gdn_chunk_state_bv48's dynamic shared memory");

extern "C" __global__ void __launch_bounds__(128) native_gdn_chunk_state_bv48(
    const unsigned short* __restrict__ cv,  // [T][10240] BF16
    const float* __restrict__ gc,           // [T][48]
    const unsigned short* __restrict__ wg,  // [T][48][128]
    const unsigned short* __restrict__ ug,  // [T][48][128]
    const unsigned short* __restrict__ aqk, // [T][48][64]
    float* state,                           // [48][128][128] F32, read then updated
    unsigned short* __restrict__ out,       // [T][48][128] BF16
    int T)
{
    constexpr int MT = NATIVE_GDN_BV / 16, NTV = NATIVE_GDN_BV / 8, LDV = NATIVE_GDN_LDV;
    extern __shared__ __align__(16) unsigned char native_gdn_smem_raw[];
    native_gdn_state_smem& s = *reinterpret_cast<native_gdn_state_smem*>(native_gdn_smem_raw);
    const int h = blockIdx.y, kh = h % NATIVE_GDN_HK, vj0 = blockIdx.x * NATIVE_GDN_BV;
    const int vvalid = min(NATIVE_GDN_BV, NATIVE_GDN_D - vj0);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, tq = lane & 3;
    const int nchunks = (T + NATIVE_GDN_BT - 1) / NATIVE_GDN_BT;
    // Lumen's rsqrtf(128): the hardware's approximate reciprocal square root, of the block size (128, the
    // head size) so that the assembler cannot fold it into the correctly rounded constant, one ulp away.
    float qscale;
    asm("rsqrt.approx.f32 %0, %1;" : "=f"(qscale) : "f"((float)blockDim.x));

    float S[MT][4][4];
#pragma unroll
    for (int mi = 0; mi < MT; mi++) {
#pragma unroll
        for (int ni = 0; ni < 4; ni++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int vj = 16 * mi + g + ((e >> 1) << 3), ki = 32 * warp + 8 * ni + 2 * tq + (e & 1);
                S[mi][ni][e] = vj < vvalid ? state[((size_t)h * NATIVE_GDN_D + vj0 + vj) * NATIVE_GDN_D + ki] : 0.0f;
            }
        }
    }

    auto load_wqu = [&](int c) {
        const int t0 = c * NATIVE_GDN_BT, n = min(NATIVE_GDN_BT, T - t0);
        for (int i = tid; i < NATIVE_GDN_BT * 16; i += 128) {
            const int r = i >> 4, c8 = (i & 15) * 8;
            const bool ok = r < n;
            const int row = ok ? t0 + r : 0;
            native_gdn_cp16(&s.w[r * NATIVE_GDN_LDK + c8], wg + ((size_t)row * NATIVE_GDN_H + h) * NATIVE_GDN_D + c8,
                            ok ? 16 : 0);
            native_gdn_cp16(&s.q[r * NATIVE_GDN_LDK + c8], cv + (size_t)row * NATIVE_GDN_CONV + kh * NATIVE_GDN_D + c8,
                            ok ? 16 : 0);
        }
        for (int i = tid; i < NATIVE_GDN_BT * (NATIVE_GDN_BV / 8); i += 128) {
            const int r = i / (NATIVE_GDN_BV / 8), c8 = (i % (NATIVE_GDN_BV / 8)) * 8;
            const bool ok = r < n && c8 < vvalid;
            const int row = ok ? t0 + r : 0, col = ok ? vj0 + c8 : 0;
            native_gdn_cp16(&s.u[r * LDV + c8], ug + ((size_t)row * NATIVE_GDN_H + h) * NATIVE_GDN_D + col, ok ? 16 : 0);
        }
        if (tid < NATIVE_GDN_BT) {
            const bool ok = tid < n;
            native_gdn_cp4(&s.g[tid], gc + (size_t)(ok ? t0 + tid : 0) * NATIVE_GDN_H + h, ok ? 4 : 0);
        }
    };
    auto load_ka = [&](int c) {
        const int t0 = c * NATIVE_GDN_BT, n = min(NATIVE_GDN_BT, T - t0);
        for (int i = tid; i < NATIVE_GDN_BT * 16; i += 128) {
            const int r = i >> 4, c8 = (i & 15) * 8;
            const bool ok = r < n;
            const int row = ok ? t0 + r : 0;
            native_gdn_cp16(&s.k[r * NATIVE_GDN_LDK + c8],
                            cv + (size_t)row * NATIVE_GDN_CONV + NATIVE_GDN_QK + kh * NATIVE_GDN_D + c8, ok ? 16 : 0);
        }
        for (int i = tid; i < NATIVE_GDN_BT * 8; i += 128) {
            const int r = i >> 3, c8 = (i & 7) * 8;
            const bool ok = r < n;
            const int row = ok ? t0 + r : 0;
            native_gdn_cp16(&s.a[r * NATIVE_GDN_LDA + c8], aqk + ((size_t)row * NATIVE_GDN_H + h) * NATIVE_GDN_BT + c8,
                            ok ? 16 : 0);
        }
    };

    // Two copy groups per chunk, {W, Q, U, G} and {K, Aqk}: the loop head waits for the first only, and the
    // chunk's K and Aqk land while the snapshot and the W/Q products run.
    load_wqu(0);
    native_gdn_cp_commit();
    load_ka(0);
    native_gdn_cp_commit();
    const int r0 = warp * 16;
    for (int c = 0; c < nchunks; c++) {
        const int t0 = c * NATIVE_GDN_BT, n = min(NATIVE_GDN_BT, T - t0);
        native_gdn_cp_wait<1>();
        __syncthreads();
        // The chunk-start state in BF16, row vj, column ki.
#pragma unroll
        for (int mi = 0; mi < MT; mi++) {
#pragma unroll
            for (int ni = 0; ni < 4; ni++) {
#pragma unroll
                for (int hf = 0; hf < 2; hf++) {
                    const int vj = 16 * mi + g + hf * 8, ki = 32 * warp + 8 * ni + 2 * tq;
                    *reinterpret_cast<unsigned int*>(&s.sb[vj * NATIVE_GDN_LDK + ki]) =
                        native_bf16x2_rn(S[mi][ni][2 * hf], S[mi][ni][2 * hf + 1]);
                }
            }
        }
        __syncthreads();
        // P = W S^T and Qo = Q S^T for this warp's 16 rows.
        float P[NTV][4], O[NTV][4];
#pragma unroll
        for (int a = 0; a < NTV; a++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                P[a][e] = O[a][e] = 0.0f;
            }
        }
#pragma unroll
        for (int k0 = 0; k0 < NATIVE_GDN_D; k0 += 16) {
            unsigned int aw[4], aq[4];
            native_gdn_lda(aw, s.w, NATIVE_GDN_LDK, r0, k0);
            native_gdn_lda(aq, s.q, NATIVE_GDN_LDK, r0, k0);
#pragma unroll
            for (int np = 0; np < NTV / 2; np++) {
                unsigned int bb[4];
                native_gdn_ldb_nk(bb, s.sb, NATIVE_GDN_LDK, np * 16, k0);
                native_gdn_mma(P[2 * np], aw, bb[0], bb[1]);
                native_gdn_mma(P[2 * np + 1], aw, bb[2], bb[3]);
                native_gdn_mma(O[2 * np], aq, bb[0], bb[1]);
                native_gdn_mma(O[2 * np + 1], aq, bb[2], bb[3]);
            }
        }
        const float glast = s.g[n - 1];
#pragma unroll
        for (int nt = 0; nt < NTV; nt++) {
#pragma unroll
            for (int hf = 0; hf < 2; hf++) {
                const int i = r0 + g + hf * 8, vj = nt * 8 + 2 * tq;
                const float gi = s.g[i];
                const float dec = expf(glast - gi), eo = expf(gi);
                const unsigned int uu = *reinterpret_cast<const unsigned int*>(&s.u[i * LDV + vj]);
                const float v0 = native_lo(uu) - P[nt][2 * hf], v1 = native_hi(uu) - P[nt][2 * hf + 1];
                *reinterpret_cast<unsigned int*>(&s.vn[i * LDV + vj]) = native_bf16x2_rn(v0, v1);
                *reinterpret_cast<unsigned int*>(&s.vd[i * LDV + vj]) = native_bf16x2_rn(v0 * dec, v1 * dec);
                O[nt][2 * hf] *= eo;
                O[nt][2 * hf + 1] *= eo;
            }
        }
        native_gdn_cp_wait<0>(); // this chunk's K and Aqk
        __syncthreads();
        if (c + 1 < nchunks) {
            load_wqu(c + 1);
            native_gdn_cp_commit();
        }
        // S <- e^(G_last) S + VD^T K on this warp's 32 key columns.
        const float egl = expf(glast);
#pragma unroll
        for (int mi = 0; mi < MT; mi++) {
#pragma unroll
            for (int ni = 0; ni < 4; ni++) {
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    S[mi][ni][e] *= egl;
                }
            }
        }
#pragma unroll
        for (int kt = 0; kt < NATIVE_GDN_BT; kt += 16) {
            unsigned int aa[MT][4];
#pragma unroll
            for (int mi = 0; mi < MT; mi++) {
                native_gdn_lda_t(aa[mi], s.vd, LDV, mi * 16, kt);
            }
#pragma unroll
            for (int np = 0; np < 2; np++) {
                unsigned int bb[4];
                native_gdn_ldb_kn(bb, s.k, NATIVE_GDN_LDK, 32 * warp + np * 16, kt);
#pragma unroll
                for (int mi = 0; mi < MT; mi++) {
                    native_gdn_mma(S[mi][2 * np], aa[mi], bb[0], bb[1]);
                    native_gdn_mma(S[mi][2 * np + 1], aa[mi], bb[2], bb[3]);
                }
            }
        }
        // O += Aqk V_new; Aqk is zero above the diagonal, so k-blocks 0..warp.
        for (int kb = 0; kb <= warp; kb++) {
            unsigned int a[4];
            native_gdn_lda(a, s.a, NATIVE_GDN_LDA, r0, kb * 16);
#pragma unroll
            for (int np = 0; np < NTV / 2; np++) {
                unsigned int bb[4];
                native_gdn_ldb_kn(bb, s.vn, LDV, np * 16, kb * 16);
                native_gdn_mma(O[2 * np], a, bb[0], bb[1]);
                native_gdn_mma(O[2 * np + 1], a, bb[2], bb[3]);
            }
        }
#pragma unroll
        for (int nt = 0; nt < NTV; nt++) {
#pragma unroll
            for (int hf = 0; hf < 2; hf++) {
                const int i = r0 + g + hf * 8, vj = nt * 8 + 2 * tq;
                if (i < n && vj < vvalid) {
                    *reinterpret_cast<unsigned int*>(&out[((size_t)(t0 + i) * NATIVE_GDN_H + h) * NATIVE_GDN_D + vj0 + vj]) =
                        native_bf16x2_rn(O[nt][2 * hf] * qscale, O[nt][2 * hf + 1] * qscale);
                }
            }
        }
        __syncthreads();
        if (c + 1 < nchunks) {
            load_ka(c + 1);
            native_gdn_cp_commit();
        }
    }
#pragma unroll
    for (int mi = 0; mi < MT; mi++) {
#pragma unroll
        for (int ni = 0; ni < 4; ni++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int vj = 16 * mi + g + ((e >> 1) << 3), ki = 32 * warp + 8 * ni + 2 * tq + (e & 1);
                if (vj < vvalid) {
                    state[((size_t)h * NATIVE_GDN_D + vj0 + vj) * NATIVE_GDN_D + ki] = S[mi][ni][e];
                }
            }
        }
    }
}
