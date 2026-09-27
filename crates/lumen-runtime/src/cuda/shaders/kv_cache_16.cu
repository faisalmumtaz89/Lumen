// ============================================================================
// 16-bit KV cache writers and the widening read for CUDA.
//
// The cache stores key/value projections as IEEE half or as bfloat16
// (`unsigned short` bit patterns) in the same head-first layout as the F32
// cache: [num_kv_heads, max_seq_len, head_dim]. Activations flow in F32; the
// writers round on store (round to nearest even), the decode readers widen on
// load (exact), and the prefill readers work on a widened F32 copy produced by
// `kv_cache_widen_f16` / `kv_cache_widen_bf16`. The `_bf16` kernels are the
// `_f16` kernels with the other conversion.
//
// Overflow: half holds |x| < 65,520 (values in [65,504, 65,520) round down to
// 65,504; anything larger, and any non-finite input, becomes ±Inf). bfloat16
// holds the F32 range, so only a non-finite input, or a finite one within half
// a unit of the largest float, becomes non-finite. Every writer counts the
// values it stores as ±Inf or NaN in `overflow_count` so the host can refuse
// to continue with a poisoned cache instead of producing NaN logits. The count
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

// F32 -> bfloat16 bits, round to nearest even (the rounding of
// `cvt.rn.bf16.f32`, in integer arithmetic so no architecture floor applies).
// A NaN stays a NaN.
__device__ __forceinline__ unsigned short kvbf16_f32_to_bits(float val) {
    const unsigned int u = __float_as_uint(val);
    if ((u & 0x7fffffffu) > 0x7f800000u) {
        return (unsigned short)((u >> 16) | 0x0040u);
    }
    return (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// bfloat16 bits -> F32, exact.
__device__ __forceinline__ float kvbf16_bits_to_f32(unsigned short b) {
    return __uint_as_float((unsigned int)b << 16);
}

// Store one value as bfloat16 and count it if it is stored as ±Inf or NaN.
__device__ __forceinline__ void kvbf16_store(
    unsigned short* __restrict__ dst,
    float v,
    unsigned int* __restrict__ overflow_count)
{
    const unsigned short b = kvbf16_f32_to_bits(v);
    if ((b & 0x7f80u) == 0x7f80u) {
        atomicAdd(overflow_count, 1u);
    }
    *dst = b;
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

// The three kernels, defined once for both 16-bit formats: SUF names the
// format (`f16` or `bf16`), STORE rounds, stores and counts one value, WIDEN
// widens one stored value.
//
// kv_cache_write_<SUF>: one token's K or V, [num_kv_heads * head_dim] F32, into
// position `pos`. Grid: ceil(num_kv_heads * head_dim / block).
//
// kv_cache_write_batch_<SUF>: `batch` tokens' K or V, [batch, num_kv_heads *
// head_dim] F32 row-major, into positions pos_start .. pos_start + batch - 1.
// Grid: ceil(batch * num_kv_heads * head_dim / block). The F32 batch writer's
// addressing, exactly (prefill_kernels.cu kv_cache_write_batch).
//
// kv_cache_widen_<SUF>: positions 0 .. count - 1 of every KV head, widened to
// F32 into a contiguous [num_kv_heads, count, head_dim] buffer — the F32 cache
// layout with `max_seq_len` = `count`, so an F32 reader takes it with that
// stride and no other change. Grid: ceil(num_kv_heads * count * head_dim /
// block). Widening is exact.
#define KV16_KERNELS(SUF, STORE, WIDEN)                                                     \
extern "C" __global__ void kv_cache_write_##SUF(                                            \
    unsigned short* __restrict__ cache,       /* [num_kv_heads, max_seq_len, head_dim] */   \
    const float* __restrict__ data,           /* [num_kv_heads * head_dim] */               \
    unsigned int* __restrict__ overflow_count,                                              \
    unsigned int pos,                                                                       \
    unsigned int num_kv_heads,                                                              \
    unsigned int max_seq_len,                                                               \
    unsigned int head_dim)                                                                  \
{                                                                                           \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;                               \
    unsigned int total = num_kv_heads * head_dim;                                           \
    if (idx >= total) return;                                                               \
                                                                                            \
    unsigned int head = idx / head_dim;                                                     \
    unsigned int dim_offset = idx % head_dim;                                               \
                                                                                            \
    unsigned long long cache_idx = (unsigned long long)head * max_seq_len * head_dim        \
                                 + (unsigned long long)pos * head_dim                       \
                                 + dim_offset;                                              \
    STORE(cache + cache_idx, data[idx], overflow_count);                                    \
}                                                                                           \
                                                                                            \
extern "C" __global__ void kv_cache_write_batch_##SUF(                                      \
    unsigned short* __restrict__ cache,       /* [num_kv_heads, max_seq_len, head_dim] */   \
    const float* __restrict__ data,           /* [batch, num_kv_heads * head_dim] */        \
    unsigned int* __restrict__ overflow_count,                                              \
    unsigned int pos_start,                                                                 \
    unsigned int batch,                                                                     \
    unsigned int num_kv_heads,                                                              \
    unsigned int max_seq_len,                                                               \
    unsigned int head_dim)                                                                  \
{                                                                                           \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;                               \
    unsigned int kv_dim = num_kv_heads * head_dim;                                          \
    unsigned int total = batch * kv_dim;                                                    \
    if (idx >= total) return;                                                               \
                                                                                            \
    unsigned int token = idx / kv_dim;                                                      \
    unsigned int within_token = idx % kv_dim;                                               \
    unsigned int head = within_token / head_dim;                                            \
    unsigned int dim_offset = within_token % head_dim;                                      \
                                                                                            \
    unsigned int pos = pos_start + token;                                                   \
    unsigned long long cache_idx = (unsigned long long)head * max_seq_len * head_dim        \
                                 + (unsigned long long)pos * head_dim                       \
                                 + dim_offset;                                              \
    STORE(cache + cache_idx, data[idx], overflow_count);                                    \
}                                                                                           \
                                                                                            \
extern "C" __global__ void kv_cache_widen_##SUF(                                            \
    const unsigned short* __restrict__ cache, /* [num_kv_heads, max_seq_len, head_dim] */   \
    float* __restrict__ out,                  /* [num_kv_heads, count, head_dim] */         \
    unsigned int num_kv_heads,                                                              \
    unsigned int count,                                                                     \
    unsigned int max_seq_len,                                                               \
    unsigned int head_dim)                                                                  \
{                                                                                           \
    unsigned long long gid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;     \
    unsigned long long per_head = (unsigned long long)count * head_dim;                     \
    unsigned long long total = (unsigned long long)num_kv_heads * per_head;                 \
    if (gid >= total) return;                                                               \
                                                                                            \
    unsigned int head = (unsigned int)(gid / per_head);                                     \
    unsigned long long within = gid % per_head;      /* pos * head_dim + d */               \
                                                                                            \
    unsigned long long cache_idx = (unsigned long long)head * max_seq_len * head_dim + within; \
    out[gid] = WIDEN(cache[cache_idx]);                                                     \
}

KV16_KERNELS(f16, kvf16_store, kvf16_bits_to_f32)
KV16_KERNELS(bf16, kvbf16_store, kvbf16_bits_to_f32)
