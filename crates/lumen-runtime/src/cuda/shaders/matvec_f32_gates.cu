// Banked F32 matvec for the GDN ssm_alpha/ssm_beta gate projections
// (source-fidelity artifacts keep the gates in their F32 source form).
//
// Both gates are tiny ([num_heads, hidden] = [48, 5120] on Qwen3.8-27B:
// ~1 MB each), so the cost of serving them is launch- and latency-bound,
// not bandwidth-bound. Dispatching them as two cuBLAS SGEMVs costs two
// fixed cuBLAS launch overheads per GDN layer; this kernel issues BOTH
// projections in ONE launch.
//
// Grid: 2 * out_dim CTAs (blockIdx.x < out_dim -> alpha row, else beta
// row). Block: 128 threads. Deterministic reduction (fixed lane ownership,
// xor tree + fixed warp-partial order).

#define GATES_THREADS 128

__device__ __forceinline__ float gates_warp_reduce(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v;
}

extern "C" __global__ __launch_bounds__(GATES_THREADS, 4)
void matvec_f32_gates_banked(
    const float* __restrict__ w_alpha, // [out_dim, in_dim] row-major F32
    const float* __restrict__ w_beta,  // [out_dim, in_dim] row-major F32
    const float* __restrict__ x,       // [in_dim] F32 activation
    float* __restrict__ out_alpha,     // [out_dim]
    float* __restrict__ out_beta,      // [out_dim]
    unsigned int out_dim,
    unsigned int in_dim)
{
    __shared__ float warp_partial[GATES_THREADS / 32];
    const unsigned int is_beta = blockIdx.x >= out_dim;
    const unsigned int row = is_beta ? (blockIdx.x - out_dim) : blockIdx.x;
    if (row >= out_dim) return;

    const float* w = (is_beta ? w_beta : w_alpha) + (unsigned long long)row * in_dim;

    float acc = 0.0f;
    for (unsigned int i = threadIdx.x; i < in_dim; i += GATES_THREADS) {
        acc = fmaf(w[i], x[i], acc);
    }
    acc = gates_warp_reduce(acc);
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    if (lane == 0) warp_partial[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        #pragma unroll
        for (int w2 = 0; w2 < GATES_THREADS / 32; w2++) total += warp_partial[w2];
        (is_beta ? out_beta : out_alpha)[row] = total;
    }
}
