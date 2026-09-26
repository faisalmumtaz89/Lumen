// NVFP4 matrix-vector multiply: out = W x for one NVFP4 plane, one warp per row.
//
// Plane layout (the converter's): weight nibbles [out_dim * in_dim/2] | E4M3 block scales
// [out_dim * in_dim/16] | global scale (F32 LE, 4 bytes). Each 16-weight group's value is
// e2m1(code) * (e4m3(block_scale) * global_scale).
//
// Mapping and accumulation order. Lane `l` owns groups l, l+32, l+64, ... of the row, ascending, into one
// running `acc`; a group's 16 products are summed in code order into a group partial that is multiplied by
// the group's folded scale and added to `acc`; the lanes are then reduced with `shfl_down` 16..1. Nothing is
// reassociated across groups, so every kernel in this file that walks a row this way produces the same bits.
//
// Loads. A group arrives in few, wide instructions so the decode stays inside the per-weight instruction
// budget of a bandwidth-bound matvec:
//   * its 8 weight bytes as two `uchar4`;
//   * its 16 activations straight from global memory as four `float4` (the warp's 32 lanes read 2 KB of
//     contiguous x per step, already coalesced, so there is no shared-memory stage and no shared-memory
//     footprint to cap occupancy);
//   * the E2M1 decode built arithmetically from the code's bits (below).
//
// NVRTC-compatible: no system includes, extern "C" linkage, no __constant__, no #if __CUDA_ARCH__.
//
// Grid: (ceil(out_dim / (NVFP4W_THREADS/32)), 1, 1)   Block: (NVFP4W_THREADS, 1, 1), no shared memory.
//   One WARP per row.

#ifndef NVFP4W_THREADS
#define NVFP4W_THREADS 128
#endif

#define NVFP4W_GROUP 16
#define NVFP4W_BYTES_PER_GROUP 8

// E2M1 decode by construction. For a 4-bit code `c`, the float whose bits are
//     ((c & 8) << 28) | ((c & 7) << 22)
// has exponent field (c>>1)&3 and top mantissa bit c&1, and equals the code's value times 2^-126 EXACTLY: an
// exponent of 1..3 is 2^(e-127)(1+m/2), and exponent 0 is the subnormal m*2^-127, i.e. 0.5 * 2^-126 for code 1
// and zero for code 0. Multiplying by 2^126 is exact, so this returns the format table's float in a few integer
// ops and one multiply — no shuffle, no branch, no load — for every code except 0x8, which decodes to -0.0
// where the table has +0.0. Every sum starts at +0.0, and adding a signed-zero product never changes a sum
// (+0 + -0 = +0, x + -0 = x), so the matvec output is unaffected.
//
// It REQUIRES subnormals to be kept (no flush-to-zero, so never built with fast-math): code 1 passes through a
// subnormal. The matvec test against the host decoder in `lumen_format::planar_dequant` covers code 1 and
// would catch a build that flushes it.
//
// Unlike a 16-entry shuffle table, it issues no `shfl` per weight, so the decode does not contend for the
// MIO queue with the loads.
#define NVFP4W_TWO_POW_126 8.5070591730234616e37f

extern "C" __device__ __forceinline__ float nvfp4w_e2m1_lo(unsigned int byte)
{
    return __uint_as_float(((byte & 0x8u) << 28) | ((byte & 0x7u) << 22)) * NVFP4W_TWO_POW_126;
}

extern "C" __device__ __forceinline__ float nvfp4w_e2m1_hi(unsigned int byte)
{
    return __uint_as_float(((byte & 0x80u) << 24) | ((byte & 0x70u) << 18)) * NVFP4W_TWO_POW_126;
}

extern "C" __device__ __forceinline__ float nvfp4w_e4m3_to_f32(unsigned int code)
{
    unsigned int c = code & 0xFFu;
    unsigned int sign = (c & 0x80u) << 24;
    unsigned int exp = (c >> 3) & 0xFu;
    unsigned int mantissa = c & 7u;
    if (exp == 15u && mantissa == 7u) {
        return __uint_as_float(0x7FC00000u);   // the format's only NaN
    }
    if (exp == 0u) {
        float sub = ((float)mantissa) / 512.0f;   // a DIVISION: a subnormal has no implicit one
        return __uint_as_float(__float_as_uint(sub) | sign);
    }
    return __uint_as_float(sign | ((exp + 120u) << 23) | (mantissa << 20));
}

// One 16-weight group of one row: sixteen `s += value * x` in code order.
extern "C" __device__ __forceinline__ float nvfp4w_group_dot(
    const uchar4 wb, const uchar4 wc, const float4 xa, const float4 xb, const float4 xc, const float4 xd)
{
    float s = 0.0f;
    s += nvfp4w_e2m1_lo((unsigned int)wb.x) * xa.x;
    s += nvfp4w_e2m1_hi((unsigned int)wb.x) * xa.y;
    s += nvfp4w_e2m1_lo((unsigned int)wb.y) * xa.z;
    s += nvfp4w_e2m1_hi((unsigned int)wb.y) * xa.w;
    s += nvfp4w_e2m1_lo((unsigned int)wb.z) * xb.x;
    s += nvfp4w_e2m1_hi((unsigned int)wb.z) * xb.y;
    s += nvfp4w_e2m1_lo((unsigned int)wb.w) * xb.z;
    s += nvfp4w_e2m1_hi((unsigned int)wb.w) * xb.w;
    s += nvfp4w_e2m1_lo((unsigned int)wc.x) * xc.x;
    s += nvfp4w_e2m1_hi((unsigned int)wc.x) * xc.y;
    s += nvfp4w_e2m1_lo((unsigned int)wc.y) * xc.z;
    s += nvfp4w_e2m1_hi((unsigned int)wc.y) * xc.w;
    s += nvfp4w_e2m1_lo((unsigned int)wc.z) * xd.x;
    s += nvfp4w_e2m1_hi((unsigned int)wc.z) * xd.y;
    s += nvfp4w_e2m1_lo((unsigned int)wc.w) * xd.z;
    s += nvfp4w_e2m1_hi((unsigned int)wc.w) * xd.w;
    return s;
}

extern "C" __device__ __forceinline__ float nvfp4w_global_scale(
    const unsigned char* plane, unsigned int out_dim, unsigned int in_dim)
{
    const unsigned char* gs = plane + (size_t)out_dim * (in_dim / 2u) + (size_t)out_dim * (in_dim / NVFP4W_GROUP);
    return __uint_as_float((unsigned int)gs[0] | ((unsigned int)gs[1] << 8) |
                           ((unsigned int)gs[2] << 16) | ((unsigned int)gs[3] << 24));
}

// One row's dot product by one warp, reduced: the result is valid in lane 0.
//
// Lane `l` owns group `l + 32k`, so a warp's 32 lanes cover 32 consecutive groups per step: 256 contiguous
// weight bytes and 2 KB of contiguous x.
extern "C" __device__ __forceinline__ float nvfp4w_row_dot(
    const unsigned char* __restrict__ plane,
    const float* __restrict__ x,
    unsigned int row,
    unsigned int out_dim,
    unsigned int in_dim,
    float global_scale)
{
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int row_bytes = in_dim / 2u;
    const unsigned int scale_bytes = in_dim / NVFP4W_GROUP;
    const unsigned char* wrow = plane + (size_t)row * row_bytes;
    const unsigned char* srow = plane + (size_t)out_dim * row_bytes + (size_t)row * scale_bytes;
    const unsigned int ngroup = in_dim / NVFP4W_GROUP;
    float acc = 0.0f;
    for (unsigned int g = lane; g < ngroup; g += 32u) {
        const uchar4 wb = *reinterpret_cast<const uchar4*>(wrow + (size_t)g * NVFP4W_BYTES_PER_GROUP);
        const uchar4 wc = *reinterpret_cast<const uchar4*>(wrow + (size_t)g * NVFP4W_BYTES_PER_GROUP + 4u);
        const float4 xa = *reinterpret_cast<const float4*>(x + (size_t)g * NVFP4W_GROUP);
        const float4 xb = *reinterpret_cast<const float4*>(x + (size_t)g * NVFP4W_GROUP + 4u);
        const float4 xc = *reinterpret_cast<const float4*>(x + (size_t)g * NVFP4W_GROUP + 8u);
        const float4 xd = *reinterpret_cast<const float4*>(x + (size_t)g * NVFP4W_GROUP + 12u);
        const float folded = nvfp4w_e4m3_to_f32((unsigned int)srow[g]) * global_scale;
        acc += nvfp4w_group_dot(wb, wc, xa, xb, xc, xd) * folded;
    }
    float v = acc;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xFFFFFFFFu, v, off);
    }
    return v;
}

extern "C" __global__ __launch_bounds__(NVFP4W_THREADS, 1) void matvec_nvfp4_wide_f32(
    const unsigned char* __restrict__ plane,
    const float* __restrict__ x,               // [in_dim] f32, read from GLOBAL
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim)
{
    const unsigned int NWARP = NVFP4W_THREADS / 32u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const float global_scale = nvfp4w_global_scale(plane, out_dim, in_dim);
    const unsigned int stride = gridDim.x * NWARP;
    for (unsigned int row = blockIdx.x * NWARP + warp; row < out_dim; row += stride) {
        const float v = nvfp4w_row_dot(plane, x, row, out_dim, in_dim, global_scale);
        if (lane == 0u) {
            out[row] = v;
        }
    }
}

// The same matvec with the residual added on the store: out[i] = (W x)[i] + residual[i].
//
// Replaces matvec -> `residual_add` (x += residual) -> copy with one launch that writes the layer output
// directly. The row's value is exactly the plain kernel's, and float addition is commutative, so the stored
// sum is bit-identical to the separate add.
extern "C" __global__ __launch_bounds__(NVFP4W_THREADS, 1) void matvec_nvfp4_wide_residual_f32(
    const unsigned char* __restrict__ plane,
    const float* __restrict__ x,
    const float* __restrict__ residual,        // [out_dim]
    float* __restrict__ out,                   // [out_dim]
    unsigned int out_dim,
    unsigned int in_dim)
{
    const unsigned int NWARP = NVFP4W_THREADS / 32u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const float global_scale = nvfp4w_global_scale(plane, out_dim, in_dim);
    const unsigned int stride = gridDim.x * NWARP;
    for (unsigned int row = blockIdx.x * NWARP + warp; row < out_dim; row += stride) {
        const float v = nvfp4w_row_dot(plane, x, row, out_dim, in_dim, global_scale);
        if (lane == 0u) {
            out[row] = residual[row] + v;
        }
    }
}

// FFN gate + up + SwiGLU in ONE launch: out[i] = silu(gate_i . x) * (up_i . x).
//
// A warp computes row i of the gate plane AND row i of the up plane, so each activation load serves both, and
// writes the SwiGLU of the pair — replacing two matvec launches and the SwiGLU launch. Each plane's row is
// walked exactly as `matvec_nvfp4_wide_f32` walks it (same lane-to-group mapping, same add order, same
// reduction) and the SwiGLU is `swiglu_inplace`'s expression verbatim, so the output is bit-identical to
// gate matvec -> up matvec -> swiglu_inplace.
//
// Grid: (ceil(out_dim / (NVFP4W_THREADS/32)), 1, 1)   Block: (NVFP4W_THREADS, 1, 1), no shared memory.
extern "C" __global__ __launch_bounds__(NVFP4W_THREADS, 1) void matvec_nvfp4_wide_glu_f32(
    const unsigned char* __restrict__ gate_plane,
    const unsigned char* __restrict__ up_plane,
    const float* __restrict__ x,               // [in_dim]
    float* __restrict__ out,                   // [out_dim] = silu(gate) * up
    unsigned int out_dim,
    unsigned int in_dim)
{
    const unsigned int NWARP = NVFP4W_THREADS / 32u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int row_bytes = in_dim / 2u;
    const unsigned int scale_bytes = in_dim / NVFP4W_GROUP;
    const unsigned int weight_plane = out_dim * row_bytes;
    const float gate_global = nvfp4w_global_scale(gate_plane, out_dim, in_dim);
    const float up_global = nvfp4w_global_scale(up_plane, out_dim, in_dim);

    const unsigned int stride = gridDim.x * NWARP;
    for (unsigned int row = blockIdx.x * NWARP + warp; row < out_dim; row += stride) {
        const unsigned char* grow = gate_plane + (size_t)row * row_bytes;
        const unsigned char* urow = up_plane + (size_t)row * row_bytes;
        const unsigned char* gsrow = gate_plane + weight_plane + (size_t)row * scale_bytes;
        const unsigned char* usrow = up_plane + weight_plane + (size_t)row * scale_bytes;
        const unsigned int ngroup = in_dim / NVFP4W_GROUP;
        float acc_g = 0.0f;
        float acc_u = 0.0f;
        for (unsigned int g = lane; g < ngroup; g += 32u) {
            const float4 xa = *reinterpret_cast<const float4*>(x + (size_t)g * NVFP4W_GROUP);
            const float4 xb = *reinterpret_cast<const float4*>(x + (size_t)g * NVFP4W_GROUP + 4u);
            const float4 xc = *reinterpret_cast<const float4*>(x + (size_t)g * NVFP4W_GROUP + 8u);
            const float4 xd = *reinterpret_cast<const float4*>(x + (size_t)g * NVFP4W_GROUP + 12u);
            const uchar4 gb = *reinterpret_cast<const uchar4*>(grow + (size_t)g * NVFP4W_BYTES_PER_GROUP);
            const uchar4 gc = *reinterpret_cast<const uchar4*>(grow + (size_t)g * NVFP4W_BYTES_PER_GROUP + 4u);
            const uchar4 ub = *reinterpret_cast<const uchar4*>(urow + (size_t)g * NVFP4W_BYTES_PER_GROUP);
            const uchar4 uc = *reinterpret_cast<const uchar4*>(urow + (size_t)g * NVFP4W_BYTES_PER_GROUP + 4u);
            const float gfold = nvfp4w_e4m3_to_f32((unsigned int)gsrow[g]) * gate_global;
            const float ufold = nvfp4w_e4m3_to_f32((unsigned int)usrow[g]) * up_global;
            acc_g += nvfp4w_group_dot(gb, gc, xa, xb, xc, xd) * gfold;
            acc_u += nvfp4w_group_dot(ub, uc, xa, xb, xc, xd) * ufold;
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc_g += __shfl_down_sync(0xFFFFFFFFu, acc_g, off);
            acc_u += __shfl_down_sync(0xFFFFFFFFu, acc_u, off);
        }
        if (lane == 0u) {
            // swiglu_inplace, verbatim.
            const float g = acc_g;
            const float silu_g = g / (1.0f + expf(-g));
            out[row] = silu_g * acc_u;
        }
    }
}
