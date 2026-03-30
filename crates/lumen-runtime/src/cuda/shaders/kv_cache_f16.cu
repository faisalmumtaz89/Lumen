// ============================================================================
// F16 KV cache kernels for CUDA.
//
// The KV cache stores key/value projections in half-precision to halve memory
// bandwidth. Activations flow in f32; conversion happens on write (f32->f16)
// and read (f16->f32).
//
// Cache layout: [num_kv_heads, max_seq_len, head_dim] stored as unsigned short
// (f16 bits). This head-first layout groups all positions for a single KV head
// contiguously, enabling efficient sequential access during attention scoring.
//
// NVRTC-compatible: inline f32<->f16 conversion, no cuda_fp16.h.
// ============================================================================

// ---------------------------------------------------------------------------
// f32 -> f16 bit conversion (IEEE 754, handles overflow/underflow/rounding)
// ---------------------------------------------------------------------------

/// Hardware f32->f16 conversion via PTX (single instruction on SM 53+).
/// NVRTC-compatible: inline PTX requires no headers or include paths.
__device__ __forceinline__ unsigned short f32_to_f16_bits(float val) {
    unsigned short result;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(result) : "f"(val));
    return result;
}

/// Hardware f16->f32 conversion via PTX (single instruction on SM 53+).
/// Replaces ~15 ALU software bit-manipulation with the native CVT instruction.
/// NVRTC-compatible: inline PTX requires no headers or include paths.
__device__ __forceinline__ float f16_bits_to_f32(unsigned short h) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(h));
    return result;
}

// ============================================================================
// kv_cache_write_f16: Write f32 K/V data into F16 cache at a given position.
//
// Cache layout: [num_kv_heads, max_seq_len, head_dim] as f16.
// Input data: [num_kv_heads * head_dim] f32 (one token's worth of K or V).
//
// Each thread converts and writes one element.
// Dispatch: grid = ceil(num_kv_heads * head_dim / 256), block = 256
// ============================================================================
extern "C" __global__ void kv_cache_write_f16(
    unsigned short*       __restrict__ cache,   // [num_kv_heads, max_seq_len, head_dim] f16
    const float*          __restrict__ data,    // [num_kv_heads * head_dim] f32
    unsigned int pos,
    unsigned int num_kv_heads,
    unsigned int max_seq_len,
    unsigned int head_dim)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_elems = num_kv_heads * head_dim;
    if (gid >= total_elems) return;

    unsigned int head = gid / head_dim;
    unsigned int d = gid % head_dim;

    // Head-first layout: cache[head][pos][d]
    unsigned long long cache_idx = (unsigned long long)head * max_seq_len * head_dim
                                 + (unsigned long long)pos * head_dim
                                 + d;

    cache[cache_idx] = f32_to_f16_bits(data[gid]);
}

// ============================================================================
// kv_cache_read_f16: Read F16 cache into f32 buffer for a range of positions.
//
// Reads cache[head][pos_start..pos_start+count][d] and writes contiguous f32.
// Output: [count * head_dim] f32 for one KV head.
//
// Dispatch: grid = ceil(count * head_dim / 256), block = 256
// ============================================================================
extern "C" __global__ void kv_cache_read_f16(
    const unsigned short* __restrict__ cache,   // [num_kv_heads, max_seq_len, head_dim] f16
    float*                __restrict__ out,     // [count * head_dim] f32
    unsigned int head,
    unsigned int pos_start,
    unsigned int count,
    unsigned int max_seq_len,
    unsigned int head_dim)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_elems = count * head_dim;
    if (gid >= total_elems) return;

    unsigned int t = gid / head_dim;     // position index within [0, count)
    unsigned int d = gid % head_dim;

    unsigned long long cache_idx = (unsigned long long)head * max_seq_len * head_dim
                                 + (unsigned long long)(pos_start + t) * head_dim
                                 + d;

    out[gid] = f16_bits_to_f32(cache[cache_idx]);
}
