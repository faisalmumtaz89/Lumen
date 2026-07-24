// ==========================================================================
// Fused gate+up+SwiGLU decode GEMV on the Q8 split layout -- llama mmvq port.
//
// Consult §2.7 gate/up fusion, built on the mmvq dp4a work-decomposition (NOT
// the scalar fused_glu_gemv.cu, which was measured ~6x slower per call than the
// dp4a path and is disabled by default on quantised dense). Replaces the FFN
// decode sub-sequence
//
//     gate matvec (matvec_q8_split_q8_1_mmvq -> scratch.gate)
//   + up   matvec (matvec_q8_split_q8_1_mmvq -> scratch.up)
//   + SwiGLU       (swiglu_inplace: scratch.gate = silu(gate) * up)
//
// with ONE kernel that:
//   * reads the shared pre-quantized Q8_1 activation ONCE (the same q8_1 buffer
//     the two separate mmvq matvecs consume) -- one activation-fragment load
//     feeds BOTH the gate and up dot products;
//   * computes gate_dot and up_dot with the SAME 4-lane-per-Q8-block mmvq
//     striping (frag = tid&3, ib0 = tid>>2, 2 dp4a/lane) and the SAME llama
//     lane-preserving cross-warp reduction (one shuffle tree each) as
//     matvec_q8_split_q8_1_mmvq -- two independent weight streams W_gate, W_up;
//   * applies SwiGLU in-register in the epilogue: out[row] = silu(gate)*up,
//     writing silu(gate)*up directly to scratch.gate.
//
// This removes ONE matvec launch + ONE SwiGLU launch per FFN per layer and the
// scratch.up global round-trip, and reuses the activation headers/quants for
// both streams (consult §2.7). The down projection is unchanged.
//
// PRECISION: byte-identical to the separate mmvq gate+up+SwiGLU path. The gate
// and up dot products use the identical `.rn`-pinned mmvq epilogue + reduction
// as matvec_q8_split_q8_1_mmvq, so gate_dot / up_dot match those kernels' F32
// outputs bit-for-bit (the separate path's F32 global round-trip of gate_dot /
// up_dot is lossless). The SwiGLU is the identical `silu_g = g/(1+expf(-g));
// out = silu_g*up` as swiglu_inplace (activations.cu). Relative to the mmvq
// path it is byte-identical; relative to the OFF (non-mmvq) path it carries
// exactly the same mmvq near-tie as the split matvec -- no NEW divergence. Same
// GQ + MoE-router gate requirement as the rest of the mmvq family.
//
// Layout (both weight streams, Q8 split / SoA):
//   Per row: [f16 scale * nb][int8 quant[32] * nb], stride 34*nb, scales @ 0,
//   quants @ 2*nb.  Q8_1 activation: 36-byte blocks, scale @ 0, quants @ +4.
//   Q8_0 zero-point 0 (no sum correction). nb even (enforced by the split repack).
//
// Grid:  (inter_dim, 1, 1)   -- ONE output row per CTA
// Block: (128, 1, 1)         -- 4 warps
//
// Requires compute capability >= 6.1 for dp4a (Pascal+).
// NVRTC-compatible: no system includes, extern "C" linkage.
// ==========================================================================

#define NW       32
#define THREADS_PER_BLOCK 128  // 4 warps, ONE output row per CTA
#define NWARPS   (THREADS_PER_BLOCK / NW)  // 4
#define Q8_1_BYTES        36   // 2B f16 scale + 2B f16 sum + 32B int8 data

__device__ __forceinline__ float f16_bits_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

__device__ __forceinline__ int dp4a_s32(int a, int b, int c) {
    int d;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// THE LOCK -- pinned add.rn (mirrors matvec_q4_split_q8_1_locked.cu).
__device__ __forceinline__ float fadd_rn_locked(float a, float b) {
    float o;
    asm volatile("add.rn.f32 %0, %1, %2;" : "=f"(o) : "f"(a), "f"(b));
    return o;
}

// LOCKED per-fragment Q8 epilogue: acc += w_scale * (x_scale * float(si)).
// Identical to matvec_q8_split_q8_1_mmvq -> byte-identical dot accumulation.
__device__ __forceinline__ float q8_frag_epilogue_locked(
    float acc, int si, float w_scale, float x_scale)
{
    asm volatile(
        "{\n\t"
        "  .reg .f32 fsi, prod;\n\t"
        "  cvt.rn.f32.s32 fsi, %1;\n\t"
        "  mul.rn.f32     prod, %3, fsi;\n\t"
        "  fma.rn.f32     %0, %2, prod, %0;\n\t"
        "}\n\t"
        : "+f"(acc)
        : "r"(si), "f"(w_scale), "f"(x_scale));
    return acc;
}

__device__ __forceinline__ float shfl_xor_f32_locked(float v, int m) {
    unsigned in = __float_as_uint(v), out;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, %3;"
                 : "=r"(out) : "r"(in), "r"(m), "r"(0xffffffffu));
    return __uint_as_float(out);
}

__device__ __forceinline__ float warp_allreduce_sum_locked(float v) {
    v = fadd_rn_locked(v, shfl_xor_f32_locked(v, 16));
    v = fadd_rn_locked(v, shfl_xor_f32_locked(v, 8));
    v = fadd_rn_locked(v, shfl_xor_f32_locked(v, 4));
    v = fadd_rn_locked(v, shfl_xor_f32_locked(v, 2));
    v = fadd_rn_locked(v, shfl_xor_f32_locked(v, 1));
    return v;
}

// ==========================================================================
// Fused gate+up+SwiGLU mmvq. grid = inter_dim (1 row/CTA), block = 128.
// ==========================================================================
extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 2) void fused_glu_gemv_q8_split_mmvq(
    const char* __restrict__ weight_gate_split,  // [inter_dim * nb * 34] Q8 split
    const char* __restrict__ weight_up_split,    // [inter_dim * nb * 34] Q8 split
    const char* __restrict__ input_q8_1,         // [nb * 36] Q8_1 pre-quantized input
    float* __restrict__ output,                  // [inter_dim] silu(gate) * up
    unsigned int inter_dim,
    unsigned int in_dim)
{
    unsigned int row = blockIdx.x;            // ONE row per CTA
    if (row >= inter_dim) return;             // uniform (grid == inter_dim), safe

    unsigned int tid     = threadIdx.x;
    unsigned int warp_id = tid / NW;
    unsigned int lane    = tid % NW;
    int frag = (int)(tid & 3u);               // 0..3
    unsigned int ib0 = tid >> 2;              // 0..31

    unsigned int nb = in_dim >> 5;            // in_dim / 32
    unsigned long long row_bytes = (unsigned long long)nb * 34ULL;
    unsigned long long scales_bytes_per_row = (unsigned long long)nb * 2ULL;

    const char* gate_base   = weight_gate_split + (unsigned long long)row * row_bytes;
    const char* gate_quants = gate_base + scales_bytes_per_row;
    const char* up_base     = weight_up_split + (unsigned long long)row * row_bytes;
    const char* up_quants   = up_base + scales_bytes_per_row;

    float g_tmp = 0.0f;
    float u_tmp = 0.0f;

    for (unsigned int ib = ib0; ib < nb; ib += 32) {
        // Shared activation fragment (loaded ONCE, feeds both gate and up).
        const int* xq = (const int*)(input_q8_1 + (unsigned long long)ib * 36ULL + 4) + 2 * frag;
        int xw0 = xq[0];
        int xw1 = xq[1];
        float x_scale = f16_bits_to_f32(
            *(const unsigned short*)(input_q8_1 + (unsigned long long)ib * 36ULL));

        // Gate stream.
        const int* gwq = (const int*)(gate_quants + (unsigned long long)ib * 32ULL) + 2 * frag;
        int gsi = dp4a_s32(gwq[0], xw0, 0);
        gsi     = dp4a_s32(gwq[1], xw1, gsi);
        float g_scale = f16_bits_to_f32(
            *(const unsigned short*)(gate_base + (unsigned long long)ib * 2ULL));
        g_tmp = q8_frag_epilogue_locked(g_tmp, gsi, g_scale, x_scale);

        // Up stream (reuses xw0/xw1/x_scale).
        const int* uwq = (const int*)(up_quants + (unsigned long long)ib * 32ULL) + 2 * frag;
        int usi = dp4a_s32(uwq[0], xw0, 0);
        usi     = dp4a_s32(uwq[1], xw1, usi);
        float u_scale = f16_bits_to_f32(
            *(const unsigned short*)(up_base + (unsigned long long)ib * 2ULL));
        u_tmp = q8_frag_epilogue_locked(u_tmp, usi, u_scale, x_scale);
    }

    // ---- llama lane-preserving cross-warp reduction (both streams) ----
    __shared__ float pg[NWARPS - 1][NW];  // [3][32] gate partials
    __shared__ float pu[NWARPS - 1][NW];  // [3][32] up partials

    if (warp_id > 0) {
        pg[warp_id - 1][lane] = g_tmp;
        pu[warp_id - 1][lane] = u_tmp;
    }
    __syncthreads();

    if (warp_id > 0) {
        return;
    }

    g_tmp = fadd_rn_locked(g_tmp, pg[0][lane]);
    g_tmp = fadd_rn_locked(g_tmp, pg[1][lane]);
    g_tmp = fadd_rn_locked(g_tmp, pg[2][lane]);
    u_tmp = fadd_rn_locked(u_tmp, pu[0][lane]);
    u_tmp = fadd_rn_locked(u_tmp, pu[1][lane]);
    u_tmp = fadd_rn_locked(u_tmp, pu[2][lane]);

    g_tmp = warp_allreduce_sum_locked(g_tmp);
    u_tmp = warp_allreduce_sum_locked(u_tmp);

    if (lane == 0) {
        // SwiGLU, identical to swiglu_inplace (activations.cu): silu(gate)*up.
        float silu_g = g_tmp / (1.0f + expf(-g_tmp));
        output[row] = silu_g * u_tmp;
    }
}
