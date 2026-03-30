// Batched KV cache write kernel for prefill.
//
// Writes N tokens' K or V data to the GPU-resident KV cache in a single
// dispatch. Replaces N sequential kv_cache_write launches with one launch
// of batch * num_kv_heads * head_dim threads.
//
// Input `data` is a contiguous [batch, num_kv_heads * head_dim] matrix
// (row-major: each row is one token's K or V vector). The kernel scatters
// each element to the correct position in the head-first cache layout:
//   cache[num_kv_heads][max_seq_len][head_dim]
//
// Thread mapping: one thread per element in data (total = batch * num_kv_heads * head_dim).
// Positions written: pos_start, pos_start+1, ..., pos_start+batch-1.
//
// Dispatch: grid = ceil(total / 256), block = 256.

extern "C" __global__ void kv_cache_write_batch(
    float* __restrict__ cache,         // [num_kv_heads, max_seq_len, head_dim]
    const float* __restrict__ data,    // [batch, num_kv_heads * head_dim]
    unsigned int pos_start,            // first position to write
    unsigned int batch,                // number of tokens
    unsigned int num_kv_heads,
    unsigned int max_seq_len,
    unsigned int head_dim)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int kv_dim = num_kv_heads * head_dim;
    unsigned int total = batch * kv_dim;
    if (idx >= total) return;

    // Decompose flat index into (token, head, dim_offset).
    unsigned int token = idx / kv_dim;
    unsigned int within_token = idx % kv_dim;
    unsigned int head = within_token / head_dim;
    unsigned int dim_offset = within_token % head_dim;

    unsigned int pos = pos_start + token;

    // Head-first layout: cache[head][pos][dim] = data[token][head * head_dim + dim]
    unsigned int cache_idx = head * max_seq_len * head_dim + pos * head_dim + dim_offset;
    cache[cache_idx] = data[idx];
}
