// Elementwise activation kernels for CUDA.
//
// All kernels use extern "C" linkage for NVRTC compatibility.
// No system headers required -- only built-in CUDA math functions.

// Warp-level max reduction using butterfly shuffle.
__device__ __forceinline__ float warp_reduce_max(float val) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 16));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 8));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 4));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 2));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 1));
    return val;
}

// Warp-level sum reduction using butterfly shuffle.
__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

// ---------- SwiGLU: fused silu(gate) * up, in-place on gate ----------
//
// SiLU(x) = x / (1 + exp(-x)), also called swish.
// SwiGLU(gate, up) = SiLU(gate[i]) * up[i], written to gate[i].
//
// Grid: 1D, one thread per element.
extern "C" __global__ void swiglu_inplace(
    float* __restrict__ gate,
    const float* __restrict__ up,
    unsigned int n)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = gate[idx];
    // SiLU: g / (1 + exp(-g))
    float silu_g = g / (1.0f + expf(-g));
    gate[idx] = silu_g * up[idx];
}

// ---------- Residual addition: x[i] += residual[i], in-place ----------
//
// Grid: 1D, one thread per element.
extern "C" __global__ void residual_add(
    float* __restrict__ x,
    const float* __restrict__ residual,
    unsigned int n)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    x[idx] += residual[idx];
}

// ---------- Residual add + copy: dst[i] = a[i] + b[i] ----------
//
// Writes a[i] + b[i] to a SEPARATE output buffer (not in-place on either input).
// Fuses residual_add + memcpy_dtod into a single kernel dispatch.
// Used in the graph decode pipeline: replaces attn_proj += down; memcpy(x_gpu, attn_proj).
//
// Grid: 1D, one thread per element.
extern "C" __global__ void residual_add_copy(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ dst,
    unsigned int n)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    dst[idx] = a[idx] + b[idx];
}

// ---------- Softmax with max-subtraction for numerical stability ----------
//
// Single-block kernel: launches one block of up to 1024 threads.
// Three-phase approach:
//   1. Find max (strided accumulation + shared-memory reduction)
//   2. Subtract max, compute exp, accumulate sum
//   3. Normalize by 1/sum
//
// Shared memory: (blockDim.x / 32) floats for warp partial sums.
extern "C" __global__ void softmax_inplace(
    float* __restrict__ scores,
    unsigned int n)
{
    extern __shared__ float shared[];

    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;
    unsigned int warp_id = tid >> 5;
    unsigned int lane_id = tid & 31u;
    unsigned int num_warps = block_size >> 5;

    // Phase 1: Find max across all elements.
    float local_max = -3.402823466e+38f;  // -FLT_MAX
    for (unsigned int i = tid; i < n; i += block_size) {
        local_max = fmaxf(local_max, scores[i]);
    }

    local_max = warp_reduce_max(local_max);
    if (lane_id == 0) {
        shared[warp_id] = local_max;
    }
    __syncthreads();

    float global_max = -3.402823466e+38f;
    if (warp_id == 0) {
        global_max = (lane_id < num_warps) ? shared[lane_id] : -3.402823466e+38f;
        global_max = warp_reduce_max(global_max);
    }
    if (tid == 0) {
        shared[0] = global_max;
    }
    __syncthreads();
    global_max = shared[0];

    // Phase 2: Subtract max, compute exp, accumulate sum.
    float local_sum = 0.0f;
    for (unsigned int i = tid; i < n; i += block_size) {
        float val = expf(scores[i] - global_max);
        scores[i] = val;
        local_sum += val;
    }

    local_sum = warp_reduce_sum(local_sum);
    if (lane_id == 0) {
        shared[warp_id] = local_sum;
    }
    __syncthreads();

    float total_sum = 0.0f;
    if (warp_id == 0) {
        total_sum = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        total_sum = warp_reduce_sum(total_sum);
    }
    if (tid == 0) {
        shared[0] = total_sum;
    }
    __syncthreads();
    total_sum = shared[0];

    // Phase 3: Normalize.
    float inv_sum = 1.0f / total_sum;
    for (unsigned int i = tid; i < n; i += block_size) {
        scores[i] *= inv_sum;
    }
}

// In-place SiLU activation: x[i] = x[i] * sigmoid(x[i]) = x[i] / (1 + exp(-x[i]))
// Used by GDN layer after Conv1D.
// Grid: ceil(n / 256), Block: 256
extern "C" __global__ void silu_inplace(
    float* __restrict__ x,
    unsigned int n)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = x[idx];
    x[idx] = v / (1.0f + expf(-v));
}

// Elementwise multiply: out[i] = silu(a[i]) * b[i]
// Used by GDN attention gating: silu(gate) * normed_output.
// Grid: ceil(n / 256), Block: 256
extern "C" __global__ void silu_elementwise_mul(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    unsigned int n)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float va = a[idx];
    out[idx] = (va / (1.0f + expf(-va))) * b[idx];
}
