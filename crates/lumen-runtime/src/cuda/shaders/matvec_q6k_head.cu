// Q6_K output-head GEMV against pre-quantized Q8_1 input.
//
// The Q4_0-preset GGUF this engine converts from keeps `output.weight` in
// Q6_K (6.5625 bits/weight); serving it directly instead of requantizing to
// Q8_0 removes ~0.3 GB of head reads per decoded token at source fidelity.
//
// Consumes a lossless SPLIT of GGML Q6_K superblocks (256 elements each,
// 210 bytes: ql[128] low-4s, qh[64] upper-2s, sc[16] int8 sub-scales, d f16)
// into four per-row planes so every load is naturally aligned:
//   ql plane: 128 B/superblock   qh plane: 64 B/superblock
//   sc plane:  16 B/superblock   d  plane:  2 B/superblock
// Bytes per weight unchanged (6.5625 bpw). Dequant identity (GGML):
//   for half n in {0,128}, l in 0..32, band b in 0..4:
//     q = compose(ql, qh) - 32;  y[n + 32*b + l] = d * sc[n/16 + 2*b + l/16] * q
//
// GEMV: one warp per row; each lane iterates 16-element runs (4 dp4a for the
// values, 4 dp4a against 0x01010101 for the run sum -> -32 zero-point
// correction), Q8_1 per-32-block scale applied per run. Deterministic
// reduction order (fixed lane ownership + xor tree).
//
// NR rows per CTA via 32-thread warps; NR=4 -> 128-thread CTA.

#ifndef Q6_NR
#define Q6_NR 2
#endif
#define Q6_THREADS (Q6_NR * 32)

__device__ __forceinline__ float q6_f16_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

__device__ __forceinline__ int q6_dp4a(int a, int b, int c) {
    int d;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

__device__ __forceinline__ float q6_warp_reduce(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v;
}

extern "C" __global__ __launch_bounds__(Q6_THREADS, 2)
void matvec_q6k_split_q8_1(
    const unsigned char* __restrict__ ql_plane, // [rows][SB*128]
    const unsigned char* __restrict__ qh_plane, // [rows][SB*64]
    const signed char* __restrict__ sc_plane,   // [rows][SB*16]
    const unsigned short* __restrict__ d_plane, // [rows][SB]
    const char* __restrict__ input_q8_1,        // [K/32] 36-byte blocks
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim)
{
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int row = blockIdx.x * Q6_NR + warp;
    if (row >= out_dim) return;

    const unsigned int nsb = in_dim >> 8; // superblocks per row
    const unsigned char* ql_row = ql_plane + (unsigned long long)row * nsb * 128u;
    const unsigned char* qh_row = qh_plane + (unsigned long long)row * nsb * 64u;
    const signed char* sc_row = sc_plane + (unsigned long long)row * nsb * 16u;
    const unsigned short* d_row = d_plane + (unsigned long long)row * nsb;

    float acc = 0.0f;

    // 16 runs of 16 elements per superblock; lane owns runs strided by 32.
    const unsigned int total_runs = nsb * 16u;
    for (unsigned int run = lane; run < total_runs; run += 32u) {
        const unsigned int sb = run >> 4;
        const unsigned int r = run & 15u;          // run inside superblock
        const unsigned int half = r >> 3;          // 0 or 1 (128-el half)
        const unsigned int band = (r >> 1) & 3u;   // 0..4 (y offset 32*band)
        const unsigned int l16 = (r & 1u) * 16u;   // l base 0 or 16

        // Element positions in the full row: sb*256 + half*128 + band*32 + l16 .. +15
        const unsigned int pos = sb * 256u + half * 128u + band * 32u + l16;

        // ql source: half*64 + (band%2)*32 + l16, nibble = band/2.
        const unsigned char* ql = ql_row + sb * 128u + half * 64u + (band & 1u) * 32u + l16;
        const unsigned char* qh = qh_row + sb * 64u + half * 32u + l16;
        const unsigned int hshift = band * 2u;     // qh bit pair per band
        const int use_hi = (int)(band >> 1);       // ql high nibble for bands 2,3

        const float d = q6_f16_to_f32(d_row[sb]);
        const float sc = (float)sc_row[sb * 16u + half * 8u + band * 2u + (l16 >> 4)];

        // Q8_1 input block covering these 16 elements (pos%32 selects half).
        const char* xb = input_q8_1 + (unsigned long long)(pos >> 5) * 36u;
        const float x_scale = q6_f16_to_f32(*(const unsigned short*)xb);
        const int* xq = (const int*)(xb + 4) + ((pos & 31u) >> 2);

        int dot = 0;
        int s16 = 0;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            const unsigned int qlw = *(const unsigned int*)(ql + 4 * k);
            const unsigned int qhw = *(const unsigned int*)(qh + 4 * k);
            const unsigned int lo = use_hi ? ((qlw >> 4) & 0x0F0F0F0Fu)
                                           : (qlw & 0x0F0F0F0Fu);
            const unsigned int hi = ((qhw >> hshift) & 0x03030303u) << 4;
            const int q = (int)(lo | hi);          // 0..63 per byte
            const int xv = xq[k];
            dot = q6_dp4a(q, xv, dot);
            s16 = q6_dp4a(0x01010101, xv, s16);
        }
        // value = d*sc*(q-32); dot over run = d*sc*x_scale*(dot - 32*s16)
        acc += d * sc * x_scale * (float)(dot - 32 * s16);
    }

    acc = q6_warp_reduce(acc);
    if (lane == 0) {
        out[row] = acc;
    }
}
