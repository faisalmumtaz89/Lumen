// Native prefill producers: SwiGLU to NVFP4 and the attention output's sigmoid gate to FP8.

// silu(g) * u of the 48 GDN-layer MLPs, the fast-math arithmetic:
//   d = 1 + ex2(g * -log2 e); silu = g / d (approximate division); out = u * silu
// with flush-to-zero multiplies and adds.
__device__ __forceinline__ float native_silu_mul_fast(float g, float u)
{
    float t, e, d, s, r;
    asm("mul.rn.ftz.f32 %0, %1, 0fBFB8AA3B;" : "=f"(t) : "f"(g));
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(t));
    asm("add.rn.ftz.f32 %0, %1, 0f3F800000;" : "=f"(d) : "f"(e));
    asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(s) : "f"(g), "f"(d));
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(u), "f"(s));
    return r;
}

// silu(g) * u of the 16 attention-layer MLPs, the IEEE arithmetic: g / (1 + expf(-g)) * u, with expf as the
// standard library's sequence (range reduction to 2^j * 2^f, ex2.approx.ftz of the reduced argument) whose
// final scaling is fused with the + 1, then a correctly rounded division.
__device__ __forceinline__ float native_silu_mul_precise(float g, float u)
{
    const float a = __fmaf_rn(g, __uint_as_float(0xBBBB989Du), 0.5f);
    float c, j;
    asm("cvt.sat.f32.f32 %0, %1;" : "=f"(c) : "f"(a));
    asm("fma.rm.f32 %0, %1, 0f437C0000, 0f4B400001;" : "=f"(j) : "f"(c));
    const float k = __fadd_rn(j, __uint_as_float(0xCB40007Fu));
    const float f = __fmaf_rn(g, __uint_as_float(0xB2A57060u), __fmaf_rn(g, __uint_as_float(0xBFB8AA3Bu), -k));
    float e;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(f));
    const float d = __fmaf_rn(e, __uint_as_float(__float_as_uint(j) << 23), 1.0f);
    return __fmul_rn(u, __fdiv_rn(g, d));
}

// gu [M][2 * I] BF16 (gate columns [0, I), up columns [I, 2I)) -> BF16 silu(g) * u -> NVFP4 with global
// scale s: codes [M][I / 2], swizzled scales. One thread per 16-element scale block; threads of rows
// [M, pad128(M)) only zero their scale byte. Block 256, grid ceil(pad128(M) * I / 16 / 256).
template <bool FAST>
__device__ __forceinline__ void native_silu_mul_fp4_body(
    const unsigned short* __restrict__ gu,
    unsigned int m,
    float s,
    unsigned char* __restrict__ q,
    unsigned char* __restrict__ sf)
{
    const unsigned int nb = NATIVE_INTER / 16;
    const unsigned long long t = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (unsigned long long)native_pad128(m) * nb) {
        return;
    }
    const unsigned int row = (unsigned int)(t / nb);
    const unsigned int b = (unsigned int)(t % nb);
    if (row >= m) {
        sf[native_sf_offset(row, b, nb)] = 0;
        return;
    }
    const unsigned short* g = gu + (unsigned long long)row * 2 * NATIVE_INTER + b * 16;
    const uint4 g0 = *reinterpret_cast<const uint4*>(g);
    const uint4 g1 = *reinterpret_cast<const uint4*>(g + 8);
    const uint4 u0 = *reinterpret_cast<const uint4*>(g + NATIVE_INTER);
    const uint4 u1 = *reinterpret_cast<const uint4*>(g + NATIVE_INTER + 8);
    const unsigned int gp[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
    const unsigned int up[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
    unsigned int v[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const float lo = FAST ? native_silu_mul_fast(native_lo(gp[i]), native_lo(up[i]))
                              : native_silu_mul_precise(native_lo(gp[i]), native_lo(up[i]));
        const float hi = FAST ? native_silu_mul_fast(native_hi(gp[i]), native_hi(up[i]))
                              : native_silu_mul_precise(native_hi(gp[i]), native_hi(up[i]));
        v[i] = native_bf16x2_rn(lo, hi);
    }
    unsigned int sfb;
    const float scale = native_fp4_block_scale(native_fp4_vec_max(native_absmax_pairs(v, 8)), s, &sfb);
    *reinterpret_cast<uint2*>(q + (unsigned long long)row * (NATIVE_INTER / 2) + b * 8) =
        make_uint2(native_fp4x8(v, scale), native_fp4x8(v + 4, scale));
    sf[native_sf_offset(row, b, nb)] = (unsigned char)sfb;
}

extern "C" __global__ void __launch_bounds__(256) native_silu_mul_fp4_fast(
    const unsigned short* __restrict__ gu,
    unsigned int m,
    float s,
    unsigned char* __restrict__ q,
    unsigned char* __restrict__ sf)
{
    native_silu_mul_fp4_body<true>(gu, m, s, q, sf);
}

extern "C" __global__ void __launch_bounds__(256) native_silu_mul_fp4_precise(
    const unsigned short* __restrict__ gu,
    unsigned int m,
    float s,
    unsigned char* __restrict__ q,
    unsigned char* __restrict__ sf)
{
    native_silu_mul_fp4_body<false>(gu, m, s, q, sf);
}

// o [M][6144] BF16 attention output, gate [M][6144] BF16: bf16(sigmoid(gate) * o), then static FP8.
// One thread per 8 elements; block 256, grid ceil(M * 768 / 256).
extern "C" __global__ void __launch_bounds__(256) native_sigmoid_gate_fp8(
    const unsigned short* __restrict__ o,
    const unsigned short* __restrict__ gate,
    unsigned int m,
    float input_scale,
    unsigned char* __restrict__ q8)
{
    const unsigned long long t = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (unsigned long long)m * 768) {
        return;
    }
    const uint4 ov = *reinterpret_cast<const uint4*>(o + t * 8);
    const uint4 gv = *reinterpret_cast<const uint4*>(gate + t * 8);
    const unsigned int op[4] = {ov.x, ov.y, ov.z, ov.w};
    const unsigned int gp[4] = {gv.x, gv.y, gv.z, gv.w};
    unsigned int v[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        v[i] = native_bf16x2_rn(__fmul_rn(native_sigmoid(native_lo(gp[i])), native_lo(op[i])),
                                __fmul_rn(native_sigmoid(native_hi(gp[i])), native_hi(op[i])));
    }
    *reinterpret_cast<uint2*>(q8 + t * 8) = native_fp8x8(v, native_div_full(1.0f, input_scale));
}
