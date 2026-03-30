// Batched RMSNorm kernel for CUDA prefill.
//
// RMSNorm(x, weight, eps) = x[i] * weight[i] / sqrt(mean(x^2) + eps)
//
// Operates on a [batch, dim] matrix, normalizing each row independently.
// Launch: grid_dim=(batch, 1, 1), block_dim=(block_size, 1, 1)
// Shared memory: (block_size / 32) * sizeof(float) bytes.
//
// Identical reduction to norm.cu but batched: one threadblock per row.
// The kernel name is `rmsnorm_batched` (matching the existing KernelSet field).

// Warp-level sum reduction using butterfly shuffle.
//
// Reduces a float value across all 32 lanes in a warp using __shfl_xor_sync.
// Requires all 32 lanes to participate (full warp mask 0xffffffff).
__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

// Batched RMSNorm: normalize each row of [batch, dim] independently.
//
// Each threadblock processes exactly one row (batch_idx = blockIdx.x).
// The reduction uses a two-level hierarchy:
//   1. Each thread accumulates sum-of-squares for its strided elements.
//   2. Warp shuffle reduces within each warp.
//   3. Shared memory collects per-warp results.
//   4. First warp reduces across all warps.
//   5. Thread 0 computes the final RMS scale and broadcasts via shared memory.
//   6. All threads apply normalization to their elements.
//
// Block size should be a power of 2, up to 1024. The caller sets block_size
// via rmsnorm_block_size(dim) = min(dim, 1024) rounded down to warp multiple.
extern "C" __global__ void rmsnorm_batched(
    const float* __restrict__ x,       // [batch, dim]
    const float* __restrict__ weight,  // [dim]
    float* __restrict__ out,           // [batch, dim]
    float eps,
    unsigned int dim)
{
    // Shared memory for per-warp partial sums (max 32 warps per block).
    extern __shared__ float shared[];

    unsigned int batch_idx = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;
    unsigned int warp_id = tid >> 5;       // tid / 32
    unsigned int lane_id = tid & 31u;      // tid % 32
    unsigned int num_warps = block_size >> 5;

    // Address the correct row using 64-bit arithmetic to avoid overflow
    // when batch_idx * dim exceeds 2^32 (e.g., batch=256, dim=32768).
    const float* row_in = x + (unsigned long long)batch_idx * dim;
    float* row_out = out + (unsigned long long)batch_idx * dim;

    // Phase 1: Each thread accumulates sum-of-squares for its strided elements.
    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < dim; i += block_size) {
        float val = row_in[i];
        sum_sq += val * val;
    }

    // Phase 2: Warp-level reduction.
    sum_sq = warp_reduce_sum(sum_sq);

    // Phase 3: Store per-warp results to shared memory.
    if (lane_id == 0) {
        shared[warp_id] = sum_sq;
    }
    __syncthreads();

    // Phase 4: First warp reduces across all warp partial sums.
    float total = 0.0f;
    if (warp_id == 0) {
        total = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        total = warp_reduce_sum(total);
    }

    // Phase 5: Compute RMS scale and broadcast via shared memory.
    // rms = 1 / sqrt(mean(x^2) + eps) where mean = sum_sq / dim.
    if (tid == 0) {
        float rms = 1.0f / sqrtf(total / (float)dim + eps);
        shared[0] = rms;
    }
    __syncthreads();

    float rms = shared[0];

    // Phase 6: Apply normalization: out[i] = x[i] * rms * weight[i].
    for (unsigned int i = tid; i < dim; i += block_size) {
        row_out[i] = row_in[i] * rms * weight[i];
    }
}
