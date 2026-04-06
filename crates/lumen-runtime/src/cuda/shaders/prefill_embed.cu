// Batched token embedding lookup kernels for CUDA prefill.
//
// Gathers embeddings for a batch of token IDs in a single dispatch,
// replacing N sequential embed_token calls with one kernel launch.
//
// Grid: 1D, one thread per element in [batch * hidden_dim].
// Each thread copies one element from the embedding table to the output.
//
// Supports F32, Q8_0, F16, and Q4_0 embedding tables. All output F32.
//
// NVRTC-compatible: no system includes, extern "C" linkage.

// ---------- f16 bit conversion ----------
//
// IEEE 754 half-precision to single-precision via bit manipulation.
// Avoids cuda_fp16.h dependency (requires NVRTC include path config).

// Hardware f16->f32 conversion via PTX (single instruction on SM 53+).
// NVRTC-compatible: inline PTX requires no headers or include paths.
__device__ __forceinline__ float f16_bits_to_f32_embed(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

// ============================================================================
// embed_batch_f32: F32 batched token embedding lookup
//
// Copies embedding[token_ids[tok] * hidden_dim .. +hidden_dim] for each tok.
// ============================================================================

extern "C" __global__ void embed_batch_f32(
    const float* __restrict__ embedding_table,  // [vocab_size, hidden_dim]
    const unsigned int* __restrict__ token_ids, // [batch]
    float* __restrict__ output,                 // [batch, hidden_dim]
    unsigned int batch,
    unsigned int hidden_dim)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = batch * hidden_dim;
    if (idx >= total) return;

    unsigned int tok = idx / hidden_dim;
    unsigned int dim = idx % hidden_dim;
    unsigned int token_id = token_ids[tok];
    output[idx] = embedding_table[(unsigned long long)token_id * hidden_dim + dim];
}

// ============================================================================
// embed_batch_q8_0: Q8_0 batched token embedding lookup
//
// Q8_0 block: 34 bytes = 2-byte f16 scale + 32 int8 values.
// Dequant: out = scale * (float)(int8)quant[elem_in_block]
// ============================================================================

extern "C" __global__ void embed_batch_q8_0(
    const char* __restrict__ embedding_q8,      // Q8_0 packed embedding table
    const unsigned int* __restrict__ token_ids,  // [batch]
    float* __restrict__ output,                  // [batch, hidden_dim]
    unsigned int batch,
    unsigned int hidden_dim)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = batch * hidden_dim;
    if (idx >= total) return;

    unsigned int tok = idx / hidden_dim;
    unsigned int dim = idx % hidden_dim;
    unsigned int token_id = token_ids[tok];

    unsigned int global_elem = token_id * hidden_dim + dim;
    unsigned int block_idx = global_elem >> 5;       // / 32
    unsigned int elem_in_block = global_elem & 31u;  // % 32

    const char* block_ptr = embedding_q8 + block_idx * 34;

    // Read f16 scale (little-endian) and convert to f32.
    unsigned short scale_bits = (unsigned short)(unsigned char)block_ptr[0]
                              | ((unsigned short)(unsigned char)block_ptr[1] << 8);
    float scale = f16_bits_to_f32_embed(scale_bits);

    float val = (float)(signed char)block_ptr[2 + elem_in_block];
    output[idx] = val * scale;
}

// ============================================================================
// embed_batch_f16: F16 batched token embedding lookup
//
// Each element is 2-byte IEEE 754 half-precision. Dequantizes to F32.
// ============================================================================

extern "C" __global__ void embed_batch_f16(
    const unsigned short* __restrict__ embedding_f16,  // [vocab_size, hidden_dim] as f16
    const unsigned int* __restrict__ token_ids,        // [batch]
    float* __restrict__ output,                        // [batch, hidden_dim]
    unsigned int batch,
    unsigned int hidden_dim)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = batch * hidden_dim;
    if (idx >= total) return;

    unsigned int tok = idx / hidden_dim;
    unsigned int dim = idx % hidden_dim;
    unsigned int token_id = token_ids[tok];

    output[idx] = f16_bits_to_f32_embed(
        embedding_f16[(unsigned long long)token_id * hidden_dim + dim]
    );
}

// ============================================================================
// embed_batch_q4_0: Q4_0 batched token embedding lookup
//
// Q4_0 block: 18 bytes = 2-byte f16 scale + 16 bytes (32 x 4-bit unsigned).
// Nibble layout (GGML de-interleaved):
//   Elements 0-15: lo nibbles of bytes 0-15
//   Elements 16-31: hi nibbles of bytes 0-15
// Dequant: val = scale * ((float)nibble - 8.0f)
// ============================================================================

extern "C" __global__ void embed_batch_q4_0(
    const char* __restrict__ embedding_q4,      // Q4_0 packed embedding table
    const unsigned int* __restrict__ token_ids,  // [batch]
    float* __restrict__ output,                  // [batch, hidden_dim]
    unsigned int batch,
    unsigned int hidden_dim)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = batch * hidden_dim;
    if (idx >= total) return;

    unsigned int tok = idx / hidden_dim;
    unsigned int dim = idx % hidden_dim;
    unsigned int token_id = token_ids[tok];

    unsigned int global_elem = (unsigned long long)token_id * hidden_dim + dim;
    unsigned int block_idx = global_elem >> 5;        // / 32
    unsigned int elem_in_block = global_elem & 31u;   // % 32

    const char* block_ptr = embedding_q4 + block_idx * 18;

    // Read f16 scale (little-endian) and convert to f32.
    unsigned short scale_bits = (unsigned short)(unsigned char)block_ptr[0]
                              | ((unsigned short)(unsigned char)block_ptr[1] << 8);
    float scale = f16_bits_to_f32_embed(scale_bits);

    // GGML de-interleaved layout: elements 0-15 = lo nibbles, elements 16-31 = hi nibbles.
    unsigned int byte_idx = (elem_in_block < 16u) ? elem_in_block : (elem_in_block - 16u);
    unsigned char byte_val = (unsigned char)block_ptr[2 + byte_idx];
    unsigned int nibble = (elem_in_block < 16u) ? (byte_val & 0x0Fu) : ((byte_val >> 4) & 0x0Fu);
    output[idx] = scale * ((float)nibble - 8.0f);
}
