// Batched elementwise kernels for CUDA prefill.
//
// These kernels operate on [batch, dim] activation matrices. They are the
// batched counterparts of the single-vector kernels in activations.cu.
// Separated into their own compilation unit to keep prefill_kernels.cu
// focused on embed/norm/rope/kv operations.
//
// NVRTC-compatible: no system includes, extern "C" linkage.

// ---------- Batched SwiGLU: gate = silu(gate) * up ----------
//
// SiLU(x) = x / (1 + exp(-x)), also called swish.
// SwiGLU(gate, up) = SiLU(gate[i]) * up[i], written to gate[i].
//
// Grid: 1D, one thread per element in [batch * inter_dim].
// Both gate and up are flattened [batch * inter_dim] arrays -- the batch
// and dimension structure is irrelevant because the operation is purely
// elementwise.

extern "C" __global__ void swiglu_batched(
    float* __restrict__ gate,      // [batch * dim], modified in-place
    const float* __restrict__ up,  // [batch * dim]
    unsigned int total)            // batch * dim
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    float g = gate[idx];
    // SiLU: g / (1 + exp(-g))
    float silu_g = g / (1.0f + expf(-g));
    gate[idx] = silu_g * up[idx];
}

// ---------- Batched residual add: x += residual ----------
//
// x[i] += residual[i] for all batch * dim elements.
// Grid: 1D, one thread per element.

extern "C" __global__ void residual_add_batched(
    float* __restrict__ x,              // [batch * dim], modified in-place
    const float* __restrict__ residual,  // [batch * dim]
    unsigned int total)                 // batch * dim
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    x[idx] += residual[idx];
}

// ---------- Extract single row from batch matrix ----------
//
// Copies row `row_idx` from a [batch, cols] matrix to a [cols] vector.
// Used as the bridge between batched GEMM projections and the sequential
// per-token attention path: extract Q for one token, run attention_decode,
// scatter the result back.
//
// Grid: 1D, one thread per column element.

extern "C" __global__ void extract_row(
    const float* __restrict__ matrix,  // [batch, cols]
    float* __restrict__ row,           // [cols]
    unsigned int row_idx,
    unsigned int cols)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cols) return;

    row[idx] = matrix[(unsigned long long)row_idx * cols + idx];
}

// ---------- Scatter single row into batch matrix ----------
//
// Copies a [cols] vector into row `row_idx` of a [batch, cols] matrix.
// Inverse of extract_row -- scatters attention output back into the
// batched activation matrix.
//
// Grid: 1D, one thread per column element.

extern "C" __global__ void scatter_row(
    float* __restrict__ matrix,       // [batch, cols]
    const float* __restrict__ row,    // [cols]
    unsigned int row_idx,
    unsigned int cols)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cols) return;

    matrix[(unsigned long long)row_idx * cols + idx] = row[idx];
}
