// Native prefill producers: the Gemma RMSNorms over the hidden size (with and without the residual
// add) quantizing their BF16 output to FP8 or FP4, the gated RMSNorm of the GDN output quantizing to
// FP8, the final residual row, and the BF16 embedding row gather.
//
// Gemma RMSNorm over H = 5120: 64 threads per row, 2 rows per 128-thread block; thread t of a row owns
// columns v1 * 512 + t * 8 + v0 (v1 = 0..9, v0 = 0..7).
//   h         = f32(x) + f32(r) (or f32(x) without the add); the residual becomes bf16(h)
//   sum       = fma(h, h, sum) over the thread's 80 values in (v1, v0) order, from 0
//   warp sum  = butterfly with offsets 1, 2, 4, 8, 16; row sum = warp(2r) + warp(2r + 1)
//   rstd      = rsqrt.approx.ftz(eps + sum / 5120)
//   out       = bf16((h * rstd) * w1), w1 the stored F32 (w + 1)
// The quantizers read the BF16 output.

#define NATIVE_NORM_VEC 8
#define NATIVE_NORM_TPR 64
#define NATIVE_NORM_NV (NATIVE_HIDDEN / (NATIVE_NORM_VEC * NATIVE_NORM_TPR))

__device__ __forceinline__ unsigned int native_norm_col(unsigned int t, int v1)
{
    return v1 * (NATIVE_NORM_VEC * NATIVE_NORM_TPR) + t * NATIVE_NORM_VEC;
}

// One row's norm, BF16 output pairs in out[v1][0..3]. Rows past M (valid false) read zeros, write nothing
// and take part in the block barrier. With ADD, resid holds the residual on entry and bf16(h) on exit.
template <bool ADD>
__device__ __forceinline__ void native_gemma_rmsnorm(
    const unsigned short* __restrict__ x,
    unsigned short* __restrict__ resid,
    const float* __restrict__ w1,
    float eps,
    unsigned int row,
    bool valid,
    unsigned int (&out)[NATIVE_NORM_NV][4])
{
    __shared__ float red[4];
    const unsigned int tid = threadIdx.x;
    const unsigned int t = tid % NATIVE_NORM_TPR;
    const unsigned int rib = tid / NATIVE_NORM_TPR;
    float h[NATIVE_NORM_NV][NATIVE_NORM_VEC];
    float sum = 0.0f;
#pragma unroll
    for (int v1 = 0; v1 < NATIVE_NORM_NV; v1++) {
        const unsigned long long off = (unsigned long long)row * NATIVE_HIDDEN + native_norm_col(t, v1);
        uint4 xv = make_uint4(0, 0, 0, 0);
        uint4 rv = make_uint4(0, 0, 0, 0);
        if (valid) {
            xv = *reinterpret_cast<const uint4*>(x + off);
            if (ADD) {
                rv = *reinterpret_cast<const uint4*>(resid + off);
            }
        }
        const unsigned int xp[4] = {xv.x, xv.y, xv.z, xv.w};
        const unsigned int rp[4] = {rv.x, rv.y, rv.z, rv.w};
        unsigned int ro[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            if (ADD) {
                h[v1][2 * j] = __fadd_rn(native_lo(xp[j]), native_lo(rp[j]));
                h[v1][2 * j + 1] = __fadd_rn(native_hi(xp[j]), native_hi(rp[j]));
                ro[j] = native_bf16x2_rn(h[v1][2 * j], h[v1][2 * j + 1]);
            } else {
                h[v1][2 * j] = native_lo(xp[j]);
                h[v1][2 * j + 1] = native_hi(xp[j]);
            }
        }
        if (ADD && valid) {
            *reinterpret_cast<uint4*>(resid + off) = make_uint4(ro[0], ro[1], ro[2], ro[3]);
        }
#pragma unroll
        for (int v0 = 0; v0 < NATIVE_NORM_VEC; v0++) {
            sum = __fmaf_rn(h[v1][v0], h[v1][v0], sum);
        }
    }
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, o));
    }
    const unsigned int warp = tid / 32;
    if (tid % 32 == 0) {
        red[rib * 2 + warp % 2] = sum;
    }
    __syncthreads();
    const float total = __fadd_rn(red[rib * 2], red[rib * 2 + 1]);
    const float rstd = native_rsqrt_approx_ftz(__fadd_rn(eps, __fdiv_rn(total, (float)NATIVE_HIDDEN)));
#pragma unroll
    for (int v1 = 0; v1 < NATIVE_NORM_NV; v1++) {
        const unsigned int c = native_norm_col(t, v1);
        const float4 wa = *reinterpret_cast<const float4*>(w1 + c);
        const float4 wb = *reinterpret_cast<const float4*>(w1 + c + 4);
        const float wv[8] = {wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z, wb.w};
#pragma unroll
        for (int j = 0; j < 4; j++) {
            out[v1][j] = native_bf16x2_rn(__fmul_rn(__fmul_rn(h[v1][2 * j], rstd), wv[2 * j]),
                                          __fmul_rn(__fmul_rn(h[v1][2 * j + 1], rstd), wv[2 * j + 1]));
        }
    }
}

// The BF16 output (if `normed` is not null) and its static FP8 codes.
template <bool ADD>
__device__ __forceinline__ void native_rmsnorm_fp8_body(
    const unsigned short* __restrict__ x,
    unsigned short* __restrict__ resid,
    const float* __restrict__ w1,
    float eps,
    unsigned int m,
    float input_scale,
    unsigned char* __restrict__ q8,
    unsigned short* __restrict__ normed)
{
    const unsigned int t = threadIdx.x % NATIVE_NORM_TPR;
    const unsigned int row = blockIdx.x * 2 + threadIdx.x / NATIVE_NORM_TPR;
    const bool valid = row < m;
    unsigned int o[NATIVE_NORM_NV][4];
    native_gemma_rmsnorm<ADD>(x, resid, w1, eps, row, valid, o);
    if (!valid) {
        return;
    }
    const float inv = native_div_full(1.0f, input_scale);
#pragma unroll
    for (int v1 = 0; v1 < NATIVE_NORM_NV; v1++) {
        const unsigned long long off = (unsigned long long)row * NATIVE_HIDDEN + native_norm_col(t, v1);
        if (normed) {
            *reinterpret_cast<uint4*>(normed + off) = make_uint4(o[v1][0], o[v1][1], o[v1][2], o[v1][3]);
        }
        *reinterpret_cast<uint2*>(q8 + off) = native_fp8x8(o[v1], inv);
    }
}

// Layer 0's input norm: x is the embedded row itself (no residual add). Grid ceil(M / 2), block 128.
extern "C" __global__ void __launch_bounds__(128) native_rmsnorm_fp8(
    const unsigned short* __restrict__ x,
    const float* __restrict__ w1,
    float eps,
    unsigned int m,
    float input_scale,
    unsigned char* __restrict__ q8,
    unsigned short* __restrict__ normed)
{
    native_rmsnorm_fp8_body<false>(x, nullptr, w1, eps, m, input_scale, q8, normed);
}

// resid <- bf16(x + resid), then the norm of the unrounded sum to BF16 and FP8. Grid ceil(M / 2), block 128.
extern "C" __global__ void __launch_bounds__(128) native_add_rmsnorm_fp8(
    const unsigned short* __restrict__ x,
    unsigned short* __restrict__ resid,
    const float* __restrict__ w1,
    float eps,
    unsigned int m,
    float input_scale,
    unsigned char* __restrict__ q8,
    unsigned short* __restrict__ normed)
{
    native_rmsnorm_fp8_body<true>(x, resid, w1, eps, m, input_scale, q8, normed);
}

// resid <- bf16(x + resid), then the norm to BF16 and NVFP4 with global scale s (codes [M][H/2], swizzled
// scales). A 16-element scale block spans threads 2b and 2b + 1 of one warp, which exchange their lane-wise
// maxima. Grid pad128(M) / 2, block 128: rows [M, pad128(M)) only zero their scale bytes.
extern "C" __global__ void __launch_bounds__(128) native_add_rmsnorm_fp4(
    const unsigned short* __restrict__ x,
    unsigned short* __restrict__ resid,
    const float* __restrict__ w1,
    float eps,
    unsigned int m,
    float s,
    unsigned char* __restrict__ q,
    unsigned char* __restrict__ sf)
{
    const unsigned int t = threadIdx.x % NATIVE_NORM_TPR;
    const unsigned int row = blockIdx.x * 2 + threadIdx.x / NATIVE_NORM_TPR;
    const bool valid = row < m;
    unsigned int o[NATIVE_NORM_NV][4];
    native_gemma_rmsnorm<true>(x, resid, w1, eps, row, valid, o);
#pragma unroll
    for (int v1 = 0; v1 < NATIVE_NORM_NV; v1++) {
        const unsigned int c = native_norm_col(t, v1);
        const unsigned int mine = native_absmax_pairs(o[v1], 4);
        const unsigned int other = __shfl_xor_sync(0xffffffffu, mine, 1);
        unsigned int sfb;
        const float scale = native_fp4_block_scale(native_fp4_vec_max(native_max_bf16x2(other, mine)), s, &sfb);
        if (valid) {
            *reinterpret_cast<unsigned int*>(q + (unsigned long long)row * (NATIVE_HIDDEN / 2) + c / 2) =
                native_fp4x8(o[v1], scale);
        }
        if (t % 2 == 0 && row < native_pad128(m)) {
            sf[native_sf_offset(row, c / 16, NATIVE_HIDDEN / 16)] = valid ? (unsigned char)sfb : 0;
        }
    }
}

// Gated RMSNorm of the GDN output, per 128-wide head row: y = ((rstd * x) * w) * (sigmoid(z) * z), rounded
// to BF16, then static FP8. `rows` = tokens x 48 rows of x (core) and z, both [rows][128] BF16; w is the
// F32 norm weight. The reduction takes one of two shapes, chosen by the caller:
//   lanes_per_row 32 (one row per warp): lane l owns columns 4l..4l+3, butterfly offsets 16, 8, 4, 2, 1;
//   lanes_per_row 16 (two rows per warp): lane l of a half owns columns 8l..8l+7, offsets 8, 4, 2, 1;
// each lane sums its squares as x1 * x1, then fma over x0, x2, x3, ... in column order.
//   rstd = rsqrt.approx.ftz(eps + div.full(sum, 128))
// Block 128; grid ceil(rows / 4) with 32 lanes per row, ceil(rows / 8) with 16.
extern "C" __global__ void __launch_bounds__(128) native_gdn_norm_gate_fp8(
    const unsigned short* __restrict__ x,
    const unsigned short* __restrict__ z,
    const float* __restrict__ w,
    float eps,
    unsigned int rows,
    unsigned int lanes_per_row,
    float input_scale,
    unsigned char* __restrict__ q8)
{
    const unsigned int lane = threadIdx.x % 32;
    const unsigned int warp = blockIdx.x * 4 + threadIdx.x / 32;
    const bool wide = lanes_per_row == 32;
    const unsigned int row = wide ? warp : warp * 2 + lane / 16;
    const unsigned int l = wide ? lane : lane % 16;
    const unsigned int n = wide ? 4 : 8;
    const unsigned int col = l * n;
    const bool valid = row < rows;
    const unsigned long long off = (unsigned long long)row * 128 + col;
    unsigned int xp[4] = {0, 0, 0, 0};
    unsigned int zp[4] = {0, 0, 0, 0};
    if (valid) {
        if (wide) {
            const uint2 a = *reinterpret_cast<const uint2*>(x + off);
            const uint2 b = *reinterpret_cast<const uint2*>(z + off);
            xp[0] = a.x;
            xp[1] = a.y;
            zp[0] = b.x;
            zp[1] = b.y;
        } else {
            const uint4 a = *reinterpret_cast<const uint4*>(x + off);
            const uint4 b = *reinterpret_cast<const uint4*>(z + off);
            xp[0] = a.x;
            xp[1] = a.y;
            xp[2] = a.z;
            xp[3] = a.w;
            zp[0] = b.x;
            zp[1] = b.y;
            zp[2] = b.z;
            zp[3] = b.w;
        }
    }
    float xv[8];
#pragma unroll
    for (int j = 0; j < 4; j++) {
        xv[2 * j] = native_lo(xp[j]);
        xv[2 * j + 1] = native_hi(xp[j]);
    }
    float sum = __fmul_rn(xv[1], xv[1]);
    sum = __fmaf_rn(xv[0], xv[0], sum);
#pragma unroll
    for (int i = 2; i < 8; i++) {
        if (i < n) {
            sum = __fmaf_rn(xv[i], xv[i], sum);
        }
    }
    for (int o = wide ? 16 : 8; o > 0; o >>= 1) {
        sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, o));
    }
    const float rstd = native_rsqrt_approx_ftz(__fadd_rn(eps, native_div_full(sum, 128.0f)));
    if (!valid) {
        return;
    }
    const float inv = native_div_full(1.0f, input_scale);
    unsigned int yp[4] = {0, 0, 0, 0};
#pragma unroll
    for (int j = 0; j < 4; j++) {
        if (2 * j < n) {
            float y[2];
#pragma unroll
            for (int k = 0; k < 2; k++) {
                const int i = 2 * j + k;
                const float zi = k ? native_hi(zp[j]) : native_lo(zp[j]);
                const float gate = __fmul_rn(native_sigmoid(zi), zi);
                y[k] = __fmul_rn(__fmul_rn(__fmul_rn(rstd, xv[i]), w[col + i]), gate);
            }
            yp[j] = native_bf16x2_rn(y[0], y[1]);
        }
    }
    const uint2 codes = native_fp8x8(yp, inv);
    if (wide) {
        *reinterpret_cast<unsigned int*>(q8 + off) = codes.x;
    } else {
        *reinterpret_cast<uint2*>(q8 + off) = codes;
    }
}

// out[i] = f32(x[row][i]) + f32(resid[row][i]) over H: the last position's hidden state, unrounded.
// Grid H / 256, block 256.
extern "C" __global__ void __launch_bounds__(256) native_final_row_f32(
    const unsigned short* __restrict__ x,
    const unsigned short* __restrict__ resid,
    unsigned int row,
    float* __restrict__ out)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NATIVE_HIDDEN) {
        const unsigned long long off = (unsigned long long)row * NATIVE_HIDDEN + i;
        out[i] = __fadd_rn(native_bf16_to_f32(x[off]), native_bf16_to_f32(resid[off]));
    }
}

// out[t][..] = table[ids[t]][..]: the BF16 embedding rows of m tokens, copied 16 bytes per thread.
// Grid ceil(m * H / 8 / 256), block 256.
extern "C" __global__ void __launch_bounds__(256) native_embed_gather_bf16(
    const unsigned short* __restrict__ table,
    const unsigned int* __restrict__ ids,
    unsigned int m,
    unsigned short* __restrict__ out)
{
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int per_row = NATIVE_HIDDEN / 8;
    if (i < (unsigned long long)m * per_row) {
        const unsigned int row = (unsigned int)(i / per_row);
        const unsigned int c = (unsigned int)(i % per_row) * 8;
        const uint4 v = *reinterpret_cast<const uint4*>(table + (unsigned long long)ids[row] * NATIVE_HIDDEN + c);
        *reinterpret_cast<uint4*>(out + (unsigned long long)row * NATIVE_HIDDEN + c) = v;
    }
}
