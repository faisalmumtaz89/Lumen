// The VAE decoder's convolutions, as implicit GEMMs on TF32 tensor cores.
//
// The reference runs its convolutions through cuDNN in f32 with TF32 products
// (torch's default `allow_tf32`): each operand is rounded to TF32 and the
// products are summed in f32. These kernels do the same, taking the kernel
// taps in order and, within a tap, the input channels in order, 8 terms to a
// tensor-core product.
//
// A convolution takes two launches:
//   - to_channels_last_tf32  copies input rows `[C, n]` (channel planes
//                            `stride` apart) to `[n, C]`, rounding each value
//                            to TF32, so the input channels of one pixel are
//                            contiguous and a 16-wide k slice is one 64-byte
//                            run;
//   - conv_tf32              `y[oc, p] = bias[oc] + Σ_k w[oc, k] · x(p, k)`
//                            over a range of output pixels, where
//                            `k = (ky * ks + kx) * in_c + ic` and `x(p, k)` is
//                            the channels-last input at the pixel tap (ky, kx)
//                            reaches, zero outside the image. The weights come
//                            already in that k order and rounded to TF32.
//
// Every output value is one accumulation over k in that fixed order, whichever
// block, tile or band of rows computes it, so a decode in bands produces the
// same bits as a decode of the whole image.
//
// Requires sm_80+ (cp.async, ldmatrix, mma.sync TF32). NVRTC-compatible: no
// includes, extern "C" linkage.

// Output channels x pixels per block, k per tile, tiles in flight.
#define CV_BM 128
#define CV_BN 128
#define CV_BK 16
#define CV_STAGES 3
#define CV_THREADS 128

__device__ __forceinline__ unsigned int cv_smem_addr(const void* p)
{
    return (unsigned int)__cvta_generic_to_shared(p);
}

// 16 bytes from global into shared memory in the background; zero-filled
// when `inside` is false.
__device__ __forceinline__ void cv_cp_async16(float* dst, const float* src, bool inside)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(cv_smem_addr(dst)), "l"(src), "r"(inside ? 16u : 0u));
}

__device__ __forceinline__ void cv_ldmatrix_x4(unsigned int& r0, unsigned int& r1, unsigned int& r2,
                                               unsigned int& r3, const float* p)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(cv_smem_addr(p)));
}

// Element offset of (row, 4-float chunk) in a tile of 16-float rows whose
// chunks are XOR-swizzled by `row / 2`: the eight rows an `ldmatrix` reads and
// the two rows eight threads copy each fall on distinct banks.
__device__ __forceinline__ unsigned int cv_swz(unsigned int row, unsigned int chunk)
{
    return row * CV_BK + ((chunk ^ ((row >> 1) & 3u)) << 2);
}

// ---------------------------------------------------------------------------
// [C, n] (planes `stride` apart) -> [n, C], rounded to TF32 (round to nearest,
// ties away from zero, as cvt.rna does). 32x32 tiles through shared memory,
// 256 threads, so both the read and the write are coalesced.
// ---------------------------------------------------------------------------
extern "C" __global__ void to_channels_last_tf32(
    const float* __restrict__ x,
    float* __restrict__ xt,
    unsigned int c,
    unsigned int n,
    unsigned long long stride)
{
    __shared__ float tile[32][33];
    unsigned int c0 = blockIdx.y * 32, p0 = blockIdx.x * 32;
    unsigned int tx = threadIdx.x & 31, ty = threadIdx.x >> 5;
    for (unsigned int i = ty; i < 32; i += 8) {
        unsigned int cc = c0 + i, p = p0 + tx;
        if (cc < c && p < n) {
            unsigned int r;
            asm("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(r) : "f"(x[cc * stride + p]));
            tile[i][tx] = __uint_as_float(r);
        }
    }
    __syncthreads();
    for (unsigned int i = ty; i < 32; i += 8) {
        unsigned int p = p0 + i, cc = c0 + tx;
        if (cc < c && p < n) xt[(unsigned long long)p * c + cc] = tile[tx][i];
    }
}

// ---------------------------------------------------------------------------
// One ks x ks convolution (stride 1, padding ks / 2) over output pixels
// [p_first, p_end) of one image, `p = y * w + x`.
//
// `xt` holds input rows `row0..` channels-last, `[rows, w, in_c]`; `w_tf32` is
// `[out_c, ks * ks * in_c]`; `y` is `[out_c, h * w]`. `in_c` is a multiple of
// 16, so a 16-wide k slice is 16 channels of one tap.
//
// A block computes 128 output channels x 128 pixels with four warps, each
// 64 x 64 as 4 x 8 m16n8k8 products. Each k tile is copied by cp.async, four
// neighbouring threads to a 64-byte row, three tiles in flight.
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(CV_THREADS)
conv_tf32(
    const float* __restrict__ xt,
    const float* __restrict__ w_tf32,
    const float* __restrict__ bias,
    float* __restrict__ y,
    unsigned int in_c,
    unsigned int out_c,
    unsigned int h,
    unsigned int w,
    unsigned int ks,
    unsigned int row0,
    unsigned int p_first,
    unsigned int p_end)
{
    __shared__ __align__(128) float s_w[CV_STAGES * CV_BM * CV_BK];
    __shared__ __align__(128) float s_x[CV_STAGES * CV_BN * CV_BK];

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31;
    const unsigned int warp = tid >> 5;
    const unsigned int k_total = in_c * ks * ks;
    const unsigned int pad = ks / 2;
    const unsigned int m0 = blockIdx.y * CV_BM;
    const unsigned int p0 = p_first + blockIdx.x * CV_BN;

    // This thread copies chunk `chunk` of rows `r_base + 32 j`, j < 4, of both
    // tiles: output channels for the weights, pixels for the input.
    const unsigned int chunk = tid & 3;
    const unsigned int r_base = tid >> 2;
    int py[4], px[4];
    bool p_in[4], m_in[4];
    #pragma unroll
    for (unsigned int j = 0; j < 4; j++) {
        unsigned int p = p0 + r_base + 32 * j;
        p_in[j] = p < p_end;
        py[j] = p_in[j] ? (int)(p / w) : 0;
        px[j] = p_in[j] ? (int)(p % w) : 0;
        m_in[j] = m0 + r_base + 32 * j < out_c;
    }

    auto load_tile = [&](unsigned int stage, unsigned int k0) {
        float* sw = s_w + stage * CV_BM * CV_BK;
        float* sx = s_x + stage * CV_BN * CV_BK;
        unsigned int tap = k0 / in_c;
        unsigned int c0 = k0 - tap * in_c;
        int dy = (int)(tap / ks) - (int)pad;
        int dx = (int)(tap % ks) - (int)pad;
        #pragma unroll
        for (unsigned int j = 0; j < 4; j++) {
            unsigned int r = r_base + 32 * j;
            const float* wsrc =
                w_tf32 + (m_in[j] ? (unsigned long long)(m0 + r) * k_total + k0 + 4 * chunk : 0);
            cv_cp_async16(sw + cv_swz(r, chunk), wsrc, m_in[j]);
            int yy = py[j] + dy, xx = px[j] + dx;
            bool inside = p_in[j] && yy >= 0 && yy < (int)h && xx >= 0 && xx < (int)w;
            const float* xsrc =
                xt + (inside ? ((unsigned long long)(yy - (int)row0) * w + xx) * in_c + c0 + 4 * chunk : 0);
            cv_cp_async16(sx + cv_swz(r, chunk), xsrc, inside);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };

    const unsigned int wm = (warp >> 1) * 64;
    const unsigned int wn = (warp & 1) * 64;
    float acc[4][8][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++)
            #pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;

    // ldmatrix.x4 rows: lanes 0-7 address rows 0-7 at k 0-3, lanes 8-15 rows
    // 8-15 at k 0-3, lanes 16-23 rows 0-7 at k 4-7, lanes 24-31 rows 8-15 at
    // k 4-7. With 32-bit elements each 8x8 b16 matrix is an 8x4 f32 tile and a
    // lane receives (row lane/4, k lane%4). For the weights that is a0..a3 of
    // an m16k8 fragment; for the pixels it is b0 of tiles j, j+1 then b1 of
    // tiles j, j+1.
    const unsigned int l_row = (lane & 7) + ((lane >> 3) & 1) * 8;
    const unsigned int l_chunk = lane >> 4;

    const unsigned int tiles = k_total / CV_BK;
    for (unsigned int s = 0; s < CV_STAGES - 1; s++) {
        if (s < tiles) load_tile(s, s * CV_BK);
        else asm volatile("cp.async.commit_group;\n" ::);
    }
    for (unsigned int t = 0; t < tiles; t++) {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(CV_STAGES - 2));
        __syncthreads();
        // The stage refilled here was last read in iteration t - 1, which
        // every warp finished before the barrier above.
        if (t + CV_STAGES - 1 < tiles) {
            load_tile((t + CV_STAGES - 1) % CV_STAGES, (t + CV_STAGES - 1) * CV_BK);
        } else {
            asm volatile("cp.async.commit_group;\n" ::);
        }
        const float* sw = s_w + (t % CV_STAGES) * CV_BM * CV_BK;
        const float* sx = s_x + (t % CV_STAGES) * CV_BN * CV_BK;
        #pragma unroll
        for (unsigned int ks8 = 0; ks8 < CV_BK; ks8 += 8) {
            unsigned int a[4][4], b[8][2];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                cv_ldmatrix_x4(a[i][0], a[i][1], a[i][2], a[i][3],
                               sw + cv_swz(wm + i * 16 + l_row, ks8 / 4 + l_chunk));
            }
            #pragma unroll
            for (int j = 0; j < 8; j += 2) {
                cv_ldmatrix_x4(b[j][0], b[j + 1][0], b[j][1], b[j + 1][1],
                               sx + cv_swz(wn + j * 8 + l_row, ks8 / 4 + l_chunk));
            }
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+f"(acc[i][j][0]), "+f"(acc[i][j][1]), "+f"(acc[i][j][2]), "+f"(acc[i][j][3])
                        : "r"(a[i][0]), "r"(a[i][1]), "r"(a[i][2]), "r"(a[i][3]), "r"(b[j][0]), "r"(b[j][1]));
                }
            }
        }
    }

    // C fragment: rows (lane/4, lane/4 + 8), pixels 2 (lane%4) + {0, 1}.
    const unsigned long long hw = (unsigned long long)h * w;
    const unsigned int g = lane >> 2, t4 = lane & 3;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            unsigned int p = p0 + wn + j * 8 + t4 * 2;
            #pragma unroll
            for (int half = 0; half < 2; half++) {
                unsigned int m = m0 + wm + i * 16 + g + half * 8;
                if (m >= out_c) continue;
                float bm = bias[m];
                float* row = y + m * hw;
                if (p < p_end) row[p] = acc[i][j][half * 2] + bm;
                if (p + 1 < p_end) row[p + 1] = acc[i][j][half * 2 + 1] + bm;
            }
        }
    }
}
