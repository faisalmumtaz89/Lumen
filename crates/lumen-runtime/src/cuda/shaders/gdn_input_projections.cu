// GDN input projections in ONE launch: the F32 alpha and beta gates, the FP8 qkv projection and the FP8
// output-gate (z) projection, all of which read the same normed activation.
//
// Served separately they are three launches per GDN layer — `matvec_f32_gates_banked` (few rows,
// launch-latency bound), `matvec_fp8_f32` on qkv and `matvec_fp8_f32` on z. Here block ranges select the
// matrix: blocks [0, 2*n_heads) are gate rows (one row per block, alpha then beta), then qkv rows, then z rows
// (four rows per block, one per warp). The gate blocks come FIRST so their latency-bound rows start while the
// FP8 rows stream.
//
// Every row runs exactly the code its own kernel runs — `gates_row_dot` for a gate row, `fp8_row_dot` +
// `fp8_warp_reduce` + the per-tensor scale for an FP8 row — so each output is bit-identical to the separate
// launches, except that the separate `matvec_f32_gates_banked` is built with fast-math and flushes denormals
// to zero, which this kernel does not.
//
// Compiled as MATVEC_FP8 + MATVEC_F32_GATES + this file, so it shares their device functions.
//
// Grid: (2*n_heads + ceil(qkv_dim / 4) + ceil(z_dim / 4), 1, 1)   Block: (128, 1, 1), 1 KB + 16 B static shared.

extern "C" __global__ __launch_bounds__(128, 1) void gdn_input_projections_f32(
    const unsigned char* __restrict__ wqkv,    // FP8 plane [qkv_dim, in_dim] | scale
    const unsigned char* __restrict__ wz,      // FP8 plane [z_dim, in_dim] | scale
    const float* __restrict__ w_alpha,         // F32 [n_heads, in_dim]
    const float* __restrict__ w_beta,          // F32 [n_heads, in_dim]
    const float* __restrict__ x,               // [in_dim]
    float* __restrict__ qkv,                   // [qkv_dim]
    float* __restrict__ z,                     // [z_dim]
    float* __restrict__ alpha,                 // [n_heads]
    float* __restrict__ beta,                  // [n_heads]
    unsigned int qkv_dim,
    unsigned int z_dim,
    unsigned int n_heads,
    unsigned int in_dim)
{
    __shared__ float lut[256];
    __shared__ float warp_partial[GATES_THREADS / 32];
    fp8_fill_lut(lut);

    unsigned int b = blockIdx.x;
    if (b < 2u * n_heads) {
        const unsigned int is_beta = b >= n_heads;
        const unsigned int row = is_beta ? (b - n_heads) : b;
        const float* w = (is_beta ? w_beta : w_alpha) + (unsigned long long)row * in_dim;
        const float total = gates_row_dot(w, x, in_dim, warp_partial);
        if (threadIdx.x == 0) {
            (is_beta ? beta : alpha)[row] = total;
        }
        return;
    }
    b -= 2u * n_heads;

    const unsigned int NWARP = 128u / 32u;
    const unsigned int qkv_blocks = (qkv_dim + NWARP - 1u) / NWARP;
    const unsigned char* plane = wqkv;
    float* out = qkv;
    unsigned int out_dim = qkv_dim;
    if (b >= qkv_blocks) {
        b -= qkv_blocks;
        plane = wz;
        out = z;
        out_dim = z_dim;
    }
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int i = b * NWARP + warp;
    if (i >= out_dim) {
        return;
    }
    const float weight_scale = fp8_plane_scale(plane, out_dim, in_dim);
    const float acc = fp8_row_dot(plane + (size_t)i * in_dim, x, lut, in_dim);
    const float v = fp8_warp_reduce(acc);
    if (lane == 0u) {
        out[i] = weight_scale * v;
    }
}
