// ============================================================================
// 16-bit KV cache writers and the widening read for CUDA.
//
// The cache stores key/value projections as IEEE half (`unsigned short` bit
// patterns) in the same head-first layout as the F32 cache:
// [num_kv_heads, max_seq_len, head_dim]. Activations flow in F32; the writers
// round to half on store (round to nearest even, `cvt.rn.f16.f32`), the decode
// readers widen on load (exact), and the prefill readers work on a widened F32
// copy produced by `kv_cache_widen_f16`.
//
// Overflow: half holds |x| < 65,520 (values in [65,504, 65,520) round down to
// 65,504; anything larger, and any non-finite input, becomes ±Inf). Every
// writer counts such inputs in `overflow_count` so the host can refuse to
// continue with a poisoned cache instead of producing NaN logits. The count
// is exact (one atomic per offending element) and stays zero on every input
// the cache is meant for.
//
// NVRTC-compatible: no system includes, extern "C" linkage, PTX conversions.
// ============================================================================

#define KVF16_HALF_LIMIT 65520.0f

// F32 -> half bits, round to nearest even (one instruction on SM 53+).
__device__ __forceinline__ unsigned short kvf16_f32_to_bits(float val) {
    unsigned short result;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(result) : "f"(val));
    return result;
}

// Half bits -> F32, exact.
__device__ __forceinline__ float kvf16_bits_to_f32(unsigned short h) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(h));
    return result;
}

// Store one value as half and count it if it does not fit. `fabsf(v) <
// limit` is false for NaN and ±Inf too, so those are counted as well.
__device__ __forceinline__ void kvf16_store(
    unsigned short* __restrict__ dst,
    float v,
    unsigned int* __restrict__ overflow_count)
{
    if (!(fabsf(v) < KVF16_HALF_LIMIT)) {
        atomicAdd(overflow_count, 1u);
    }
    *dst = kvf16_f32_to_bits(v);
}

// ---------------------------------------------------------------------------
// kv_cache_write_f16: one token's K or V, [num_kv_heads * head_dim] F32, into
// position `pos`. Grid: ceil(num_kv_heads * head_dim / block).
// ---------------------------------------------------------------------------
extern "C" __global__ void kv_cache_write_f16(
    unsigned short* __restrict__ cache,       // [num_kv_heads, max_seq_len, head_dim]
    const float* __restrict__ data,           // [num_kv_heads * head_dim]
    unsigned int* __restrict__ overflow_count,
    unsigned int pos,
    unsigned int num_kv_heads,
    unsigned int max_seq_len,
    unsigned int head_dim)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = num_kv_heads * head_dim;
    if (idx >= total) return;

    unsigned int head = idx / head_dim;
    unsigned int dim_offset = idx % head_dim;

    unsigned long long cache_idx = (unsigned long long)head * max_seq_len * head_dim
                                 + (unsigned long long)pos * head_dim
                                 + dim_offset;
    kvf16_store(cache + cache_idx, data[idx], overflow_count);
}

// ---------------------------------------------------------------------------
// kv_cache_write_batch_f16: `batch` tokens' K or V, [batch, num_kv_heads *
// head_dim] F32 row-major, into positions pos_start .. pos_start + batch - 1.
// Grid: ceil(batch * num_kv_heads * head_dim / block). The F32 batch writer's
// addressing, exactly (prefill_kernels.cu kv_cache_write_batch).
// ---------------------------------------------------------------------------
extern "C" __global__ void kv_cache_write_batch_f16(
    unsigned short* __restrict__ cache,       // [num_kv_heads, max_seq_len, head_dim]
    const float* __restrict__ data,           // [batch, num_kv_heads * head_dim]
    unsigned int* __restrict__ overflow_count,
    unsigned int pos_start,
    unsigned int batch,
    unsigned int num_kv_heads,
    unsigned int max_seq_len,
    unsigned int head_dim)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int kv_dim = num_kv_heads * head_dim;
    unsigned int total = batch * kv_dim;
    if (idx >= total) return;

    unsigned int token = idx / kv_dim;
    unsigned int within_token = idx % kv_dim;
    unsigned int head = within_token / head_dim;
    unsigned int dim_offset = within_token % head_dim;

    unsigned int pos = pos_start + token;
    unsigned long long cache_idx = (unsigned long long)head * max_seq_len * head_dim
                                 + (unsigned long long)pos * head_dim
                                 + dim_offset;
    kvf16_store(cache + cache_idx, data[idx], overflow_count);
}

// ---------------------------------------------------------------------------
// kv_cache_widen_f16: positions 0 .. count - 1 of every KV head, widened to
// F32 into a contiguous [num_kv_heads, count, head_dim] buffer — the F32
// cache layout with `max_seq_len` = `count`, so an F32 reader takes it with
// that stride and no other change. Grid: ceil(num_kv_heads * count * head_dim
// / block). Widening is exact.
// ---------------------------------------------------------------------------
extern "C" __global__ void kv_cache_widen_f16(
    const unsigned short* __restrict__ cache, // [num_kv_heads, max_seq_len, head_dim]
    float* __restrict__ out,                  // [num_kv_heads, count, head_dim]
    unsigned int num_kv_heads,
    unsigned int count,
    unsigned int max_seq_len,
    unsigned int head_dim)
{
    unsigned long long gid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long per_head = (unsigned long long)count * head_dim;
    unsigned long long total = (unsigned long long)num_kv_heads * per_head;
    if (gid >= total) return;

    unsigned int head = (unsigned int)(gid / per_head);
    unsigned long long within = gid % per_head;      // pos * head_dim + d

    unsigned long long cache_idx = (unsigned long long)head * max_seq_len * head_dim + within;
    out[gid] = kvf16_bits_to_f32(cache[cache_idx]);
}
