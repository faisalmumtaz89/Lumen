// KV cache write kernel for CUDA.
//
// Writes a single token's K or V data into the GPU-resident KV cache at the
// specified position. The cache uses head-first layout:
//   cache[num_kv_heads][max_seq_len][head_dim]
//
// Input `data` is a contiguous vector of shape [num_kv_heads * head_dim]
// (one token's worth of K or V). Each head's slice of head_dim elements is
// scatter-written to the correct position in the head-first layout.
//
// Thread mapping: one thread per element in data (total = num_kv_heads * head_dim).

extern "C" __global__ void kv_cache_write(
    float* __restrict__ cache,
    const float* __restrict__ data,
    unsigned int pos,
    unsigned int num_kv_heads,
    unsigned int max_seq_len,
    unsigned int head_dim)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = num_kv_heads * head_dim;
    if (idx >= total) return;

    // Decompose flat index into (head, dim_offset).
    unsigned int head = idx / head_dim;
    unsigned int dim_offset = idx % head_dim;

    // Head-first layout: cache[head][pos][dim] = data[head * head_dim + dim]
    unsigned int cache_idx = head * max_seq_len * head_dim + pos * head_dim + dim_offset;
    cache[cache_idx] = data[idx];
}
