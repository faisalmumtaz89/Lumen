// ==========================================================================
// W4A16 tensor-core decode matvec over the tile-major Q4_0 layout.
//
// Consumes the two-plane layout produced by `q4_marlin::pack_q4_marlin`
// (see that module for the exact bit-level spec):
//   * nibble plane: K16xN64 tiles of 128 u32 words, fragment-native order
//   * scale plane:  group-32 FP16 bits, [K/32, N] with 8x8-transposed
//     64-column chunks (so the two scales a lane needs are one u32 load)
//
// Math per output n:  out[n] = sum_k fp16(scale[g][n] * (q(n,k) - 8)) * x[k]
// dequantized to FP16 exactly as the source contract defines the value, then
// accumulated through mma.sync.m16n8k16 f16*f16 with FP32 accumulators —
// NEAR-TIE numeric class vs. the dp4a path (different accumulation order and
// activation precision), never routed while byte-identity gates apply.
//
// The M=1 GEMV is padded to a logical M=8 batch: dequantized weights ride the
// PTX A operand (16 output columns per MMA), the activation vector rides the
// B operand with only batch-column 0 populated. Tensor-core row waste is not
// the limiting physics — the kernel is bandwidth-bound (~28 flop/B issued vs
// A100's ~186 flop/B balance point).
//
// Nibble-order -> fragment mapping (from the pack spec, order [0,2,4,6,1,3,5,7]):
//   (word >> 0) & 0x000f000f -> {v0,v1} = A-frag a0 (n,   k+{0,1})
//   (word >> 4) & 0x000f000f -> {v2,v3} = A-frag a2 (n,   k+{8,9})
//   (word >> 8) & 0x000f000f -> {v4,v5} = A-frag a1 (n+8, k+{0,1})
//   (word >>12) & 0x000f000f -> {v6,v7} = A-frag a3 (n+8, k+{8,9})
// The int4->fp16 expansion trick (0x6400 bias, subtract 1032) follows the
// Marlin kernel lineage (IST-DASLab/marlin, vLLM — Apache-2.0).
//
// Kernels:
//   lumen_marlin_m1_g32_m8_bn128_probe — correctness bring-up: direct global
//     loads, no pipeline. One CTA per 128 outputs, full-K loop. Used by the
//     isolated P2 fragment/GEMV oracle only; never dispatched by the engine.
//
// Requires SM80 (cp.async lives in the pipelined variants; mma m16n8k16
// f16f16f32 itself is SM80+). NVRTC-compatible: no includes, extern "C".
// ==========================================================================

#define MARLIN_THREADS 256
#define MARLIN_LOP3_AND_OR 0xEA // (a & b) | c

__device__ __forceinline__ unsigned int lop3_and_or(
    unsigned int a, unsigned int b, unsigned int c)
{
    unsigned int d;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// {1024+qlo, 1024+qhi} - {1032,1032} = {qlo-8, qhi-8} exactly (f16 integers).
__device__ __forceinline__ unsigned int f16x2_sub_bias(unsigned int biased)
{
    unsigned int d;
    asm("sub.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(biased), "r"(0x64086408u));
    return d;
}

__device__ __forceinline__ unsigned int f16x2_mul(unsigned int a, unsigned int b)
{
    unsigned int d;
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}

__device__ __forceinline__ unsigned int prmt(unsigned int a, unsigned int sel)
{
    unsigned int d;
    asm("prmt.b32 %0, %1, %1, %2;" : "=r"(d) : "r"(a), "r"(sel));
    return d;
}

// Expand one packed word into the four A-fragment registers, applying the
// per-column group scales (s_n for columns n, s_n8 for columns n+8).
__device__ __forceinline__ void dequant_frag_a(
    unsigned int word, unsigned int s_n2, unsigned int s_n8_2,
    unsigned int &a0, unsigned int &a1, unsigned int &a2, unsigned int &a3)
{
    const unsigned int MASK = 0x000f000fu;
    const unsigned int BIAS = 0x64006400u;
    unsigned int t0 = lop3_and_or(word, MASK, BIAS);        // {v0,v1} -> a0
    unsigned int t2 = lop3_and_or(word >> 4, MASK, BIAS);   // {v2,v3} -> a2
    unsigned int t1 = lop3_and_or(word >> 8, MASK, BIAS);   // {v4,v5} -> a1
    unsigned int t3 = lop3_and_or(word >> 12, MASK, BIAS);  // {v6,v7} -> a3
    a0 = f16x2_mul(f16x2_sub_bias(t0), s_n2);
    a2 = f16x2_mul(f16x2_sub_bias(t2), s_n2);
    a1 = f16x2_mul(f16x2_sub_bias(t1), s_n8_2);
    a3 = f16x2_mul(f16x2_sub_bias(t3), s_n8_2);
}

__device__ __forceinline__ void mma_m16n8k16_f32(
    float &c0, float &c1, float &c2, float &c3,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// --------------------------------------------------------------------------
// Correctness probe: direct-load GEMV, one CTA per 128 outputs.
//   q_words   : nibble plane (see layout spec)
//   scale_bits: scale plane, [K/32, N] chunk-transposed FP16 bits
//   x_f16     : activations, K halves
//   out_f32   : N floats
// Requires N % 128 == 0, K % 32 == 0.
// --------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(MARLIN_THREADS, 1)
void lumen_marlin_m1_g32_m8_bn128_probe(
    const unsigned int* __restrict__ q_words,
    const unsigned short* __restrict__ scale_bits,
    const unsigned short* __restrict__ x_f16,
    float* __restrict__ out_f32,
    unsigned int n_dim,
    unsigned int k_dim)
{
    const unsigned int warp = threadIdx.x / 32u;
    const unsigned int lane = threadIdx.x % 32u;
    const unsigned int n0 = blockIdx.x * 128u;      // CTA's first output col
    const unsigned int wcol = n0 + 16u * warp;      // warp's first output col
    const unsigned int wp = warp % 4u;              // packing warp inside tile
    const unsigned int chunk64 = wcol / 64u;        // 64-col chunk index
    const unsigned int tiles_per_krow = n_dim / 64u;
    const unsigned int c = lane / 4u;

    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;

    unsigned int s_n2 = 0u, s_n8_2 = 0u;
    for (unsigned int k0 = 0; k0 < k_dim; k0 += 16u) {
        if ((k0 & 31u) == 0u) {
            // Two adjacent u16 scales (columns n and n+8) in one u32 load.
            const unsigned int g = k0 >> 5;
            const unsigned int sidx = g * n_dim + chunk64 * 64u + 8u * c + 2u * wp;
            const unsigned int sv = *(const unsigned int*)(scale_bits + sidx);
            s_n2 = prmt(sv, 0x1010u);   // {lo, lo}
            s_n8_2 = prmt(sv, 0x3232u); // {hi, hi}
        }

        const unsigned int tile_id = (k0 >> 4) * tiles_per_krow + wcol / 64u;
        const unsigned int word = q_words[tile_id * 128u + lane * 4u + wp];

        unsigned int a0, a1, a2, a3;
        dequant_frag_a(word, s_n2, s_n8_2, a0, a1, a2, a3);

        // B fragment: batch column = lane/4; only column 0 is real.
        unsigned int b0 = 0u, b1 = 0u;
        if (lane < 4u) {
            b0 = *(const unsigned int*)(x_f16 + k0 + 2u * lane);
            b1 = *(const unsigned int*)(x_f16 + k0 + 2u * lane + 8u);
        }

        mma_m16n8k16_f32(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
    }

    // Batch column 0 lives in c0/c2 of lanes with lane%4 == 0.
    if ((lane & 3u) == 0u) {
        out_f32[wcol + c] = c0;
        out_f32[wcol + c + 8u] = c2;
    }
}
