// Causal row softmax over a block of attention scores, in place, exact F32.
//
// The tiled prefill attention computes S = Q·Kᵀ for a block of query rows with
// cuBLAS SGEMM, runs this kernel, then O = P·V with a second SGEMM. Row r of
// head g holds kv_len scores; the query it belongs to sits at absolute
// position pos_start + r, so keys j > pos_start + r are masked. Masked
// entries are written as 0 so the P·V GEMM can run over the full kv_len.
//
// Grid: (rows, heads). Block: 128 threads. One block per row; the max and
// the sum are warp xor trees folded across the four warps in a fixed order.
#define SMX_THREADS 128u
#define SMX_NEG_INF (-3.402823466e+38f)

__device__ __forceinline__ float smx_warp_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 16));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 8));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 4));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    return v;
}

__device__ __forceinline__ float smx_warp_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v;
}

extern "C" __global__ void attn_softmax_causal_rows(
    float* __restrict__ s,          // [heads][rows][s_ld] scores in, probabilities out
    unsigned int rows,
    unsigned int kv_len,
    unsigned int s_ld,
    unsigned int pos_start,         // absolute position of row 0
    float scale)
{
    __shared__ float part[4];
    unsigned int r = blockIdx.x;
    unsigned int g = blockIdx.y;
    if (r >= rows) return;
    unsigned int tid = threadIdx.x;
    unsigned int lane = tid & 31u;
    unsigned int warp = tid >> 5;
    unsigned int valid = pos_start + r + 1u;
    if (valid > kv_len) valid = kv_len;
    float* row = s + ((unsigned long long)g * rows + r) * (unsigned long long)s_ld;

    float m = SMX_NEG_INF;
    for (unsigned int j = tid; j < valid; j += SMX_THREADS) {
        m = fmaxf(m, row[j] * scale);
    }
    m = smx_warp_max(m);
    if (lane == 0) part[warp] = m;
    __syncthreads();
    m = fmaxf(fmaxf(part[0], part[1]), fmaxf(part[2], part[3]));
    __syncthreads();

    float sum = 0.0f;
    for (unsigned int j = tid; j < valid; j += SMX_THREADS) {
        float p = expf(row[j] * scale - m);
        row[j] = p;
        sum += p;
    }
    sum = smx_warp_sum(sum);
    if (lane == 0) part[warp] = sum;
    __syncthreads();
    sum = ((part[0] + part[1]) + (part[2] + part[3]));
    float inv = 1.0f / sum;
    for (unsigned int j = tid; j < valid; j += SMX_THREADS) {
        row[j] *= inv;
    }
    for (unsigned int j = valid + tid; j < kv_len; j += SMX_THREADS) {
        row[j] = 0.0f;
    }
}
