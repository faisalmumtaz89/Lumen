// ==========================================================================
// Cooperative 4-thread-per-Q8_0-block matvec with pre-quantized Q8_1 input.
//
// Matches llama.cpp's mul_mat_vec_q architecture: instead of 1 thread
// processing an entire Q8_0 block (8 dp4a calls, 32 byte loads), 4 threads
// cooperate on each block (2 dp4a calls per thread, 8 byte loads each).
//
// Why this matters:
//   - Current kernel: 1 thread/block, 8 dp4a + 8 scale loads + 32 weight bytes
//     Each thread has a long serial dependency chain (8 sequential dp4a).
//   - Cooperative: 4 threads/block, 2 dp4a each.
//     Shorter dependency chain per thread (2 dp4a), better ILP.
//     The partial sums from 4 threads get merged in warp reduction.
//
// Architecture:
//   - 1 output row per thread block (not NR=2).
//   - 128 threads per block (4 warps).
//   - TPB=4 threads per Q8_0 block -> 32 blocks per iteration (128/4=32).
//   - VDR=2: each thread loads 2 int32 words (8 bytes of quants) per block.
//   - 2 dp4a calls per thread per block iteration.
//   - Warp reduction + shmem cross-warp reduction for the full row sum.
//
// Two kernels:
//   1. matvec_q8_coop_q8_1:          W*x -> out
//   2. matvec_q8_coop_q8_1_residual: W*x + residual -> out
//
// Two variants each:
//   - Q8Raw (native Q8_0, 34-byte blocks): quants at offset +2 (NOT aligned)
//   - Q8Aligned (repacked 36-byte blocks): quants at offset +4 (4-byte aligned)
//
// Grid:  (out_dim, 1, 1)  -- one block per output row
// Block: (128, 1, 1)       -- 4 warps
//
// Q8_0 block layout (34 bytes):
//   bytes [0..1]: f16 scale
//   bytes [2..33]: 32 x int8 quants (NOT 4-byte aligned)
//   Each thread loads VDR=2 int32 words = 8 bytes of quants at its position.
//
// Q8Aligned block layout (36 bytes):
//   bytes [0..1]: f16 scale
//   bytes [2..3]: 2-byte padding
//   bytes [4..35]: 32 x int8 quants (4-byte aligned)
//
// Q8_1 block layout (36 bytes):
//   bytes [0..1]: f16 scale
//   bytes [2..3]: f16 sum (unused for Q8_0 which has zero_point=0)
//   bytes [4..35]: 32 x int8 quants (4-byte aligned)
//
// Requires compute capability >= 6.1 for __dp4a() (Pascal+).
// in_dim must be a multiple of 32 (Q8_0 block size).
//
// NVRTC-compatible: no system includes, extern "C" linkage.
// ==========================================================================

#define WARP_SIZE  32
#define THREADS    128    // 4 warps per block
#define NWARPS     (THREADS / WARP_SIZE)  // 4

// Cooperative parameters (matching llama.cpp):
#define QI8_0      8      // Q8_0 has 8 int32 words of quant data (32 bytes / 4)
#define VDR        2      // Values decoded per reduction step: 2 int32 words per thread
#define TPB        (QI8_0 / VDR)   // 4 threads per Q8_0 block

#define Q8_0_BYTES   34   // Native Q8_0: 2B f16 scale + 32B int8 quants
#define Q8_1_BYTES   36   // Q8_1: 2B f16 scale + 2B f16 sum + 32B int8 quants
#define Q8A_BYTES    36   // Q8Aligned: 2B f16 scale + 2B pad + 32B int8 quants

// Hardware f16->f32 conversion via PTX (single cycle on SM 53+).
__device__ __forceinline__ float f16_bits_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

// Warp-level reduction: sum all 32 lanes via butterfly shuffle.
__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

// ==========================================================================
// Kernel 1: Cooperative Q8_0 (native 34B) x Q8_1 -> F32 output.
//
// Grid:  (out_dim, 1, 1)  -- one block per row
// Block: (128, 1, 1)       -- 4 warps
//
// Each iteration: 128/4 = 32 Q8_0 blocks processed cooperatively.
// Each thread processes VDR=2 int32 words (8 bytes) of one Q8_0 block.
// ==========================================================================
extern "C" __global__ __launch_bounds__(THREADS, 1) void matvec_q8_coop_q8_1(
    const char* __restrict__ weight_q8,  // [out_dim * nb * 34] Q8_0 weight bytes
    const char* __restrict__ input_q8_1, // [nb * 36] Q8_1 pre-quantized input
    float* __restrict__ out,             // [out_dim] F32 output
    unsigned int out_dim,
    unsigned int in_dim)
{
    const unsigned int row = blockIdx.x;
    if (row >= out_dim) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int nb = in_dim >> 5;  // blocks per row = in_dim / 32
    const unsigned long long row_bytes = (unsigned long long)nb * Q8_0_BYTES;

    // Thread's position within its Q8_0 block.
    const unsigned int kqs = VDR * (tid % TPB);  // offset in int32 words: 0, 2, 4, 6

    // Thread's starting block index (each group of TPB threads shares a block).
    const unsigned int blocks_per_iter = THREADS / TPB;  // 32

    float sum = 0.0f;

    for (unsigned int ib = tid / TPB; ib < nb; ib += blocks_per_iter) {

        // --- Load Q8_0 weight block ---
        const char* w_block = weight_q8
            + (unsigned long long)row * row_bytes
            + (unsigned long long)ib * Q8_0_BYTES;

        // Read f16 weight scale (bytes 0-1, little-endian).
        unsigned short w_scale_bits = (unsigned short)(unsigned char)w_block[0]
                                    | ((unsigned short)(unsigned char)w_block[1] << 8);
        float w_scale = f16_bits_to_f32(w_scale_bits);

        // Load VDR=2 int32 weight words from quant data at offset +2 (NOT aligned).
        // Byte-level load + manual packing (34-byte Q8_0 blocks are never aligned).
        const unsigned char* wq = (const unsigned char*)(w_block + 2);
        int w0, w1;

        // Each int32 word = 4 consecutive int8 quants.
        // kqs selects which 2 words this thread handles.
        {
            const unsigned char* base0 = wq + kqs * 4;
            w0 = (int)(signed char)base0[0]
               | ((int)(signed char)base0[1] << 8)
               | ((int)(signed char)base0[2] << 16)
               | ((int)(signed char)base0[3] << 24);
        }
        {
            const unsigned char* base1 = wq + (kqs + 1) * 4;
            w1 = (int)(signed char)base1[0]
               | ((int)(signed char)base1[1] << 8)
               | ((int)(signed char)base1[2] << 16)
               | ((int)(signed char)base1[3] << 24);
        }

        // --- Load Q8_1 input block ---
        const char* x_block = input_q8_1 + (unsigned long long)ib * Q8_1_BYTES;

        unsigned short x_scale_bits = (unsigned short)(unsigned char)x_block[0]
                                    | ((unsigned short)(unsigned char)x_block[1] << 8);
        float x_scale = f16_bits_to_f32(x_scale_bits);

        // Q8_1 quant data at offset +4 is 4-byte aligned -> native int* loads.
        const int* xq = (const int*)(x_block + 4);
        int x0 = xq[kqs];
        int x1 = xq[kqs + 1];

        // 2 dp4a calls: 8 multiply-accumulates total.
        int acc = __dp4a(w0, x0, 0);
        acc = __dp4a(w1, x1, acc);

        sum += w_scale * x_scale * (float)acc;
    }

    // --- Warp reduction: sum across 32 lanes ---
    sum = warp_reduce_sum(sum);

    // --- Cross-warp reduction via shmem ---
    __shared__ float shmem[NWARPS];
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane = tid % WARP_SIZE;

    if (lane == 0) {
        shmem[warp_id] = sum;
    }
    __syncthreads();

    // Warp 0 does final reduction across 4 warps.
    if (warp_id == 0 && lane < NWARPS) {
        float val = shmem[lane];
        // Reduce 4 values within warp 0.
        val += __shfl_xor_sync(0xffffffff, val, 2);
        val += __shfl_xor_sync(0xffffffff, val, 1);
        if (lane == 0) {
            out[row] = val;
        }
    }
}

// ==========================================================================
// Kernel 2: Cooperative Q8_0 (native 34B) x Q8_1 + residual -> F32 output.
//
// Same as matvec_q8_coop_q8_1 but with fused residual addition.
// ==========================================================================
extern "C" __global__ __launch_bounds__(THREADS, 1) void matvec_q8_coop_q8_1_residual(
    const char* __restrict__ weight_q8,  // [out_dim * nb * 34] Q8_0 weight bytes
    const char* __restrict__ input_q8_1, // [nb * 36] Q8_1 pre-quantized input
    const float* __restrict__ residual,  // [out_dim] F32 residual
    float* __restrict__ out,             // [out_dim] F32 output
    unsigned int out_dim,
    unsigned int in_dim)
{
    const unsigned int row = blockIdx.x;
    if (row >= out_dim) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int nb = in_dim >> 5;
    const unsigned long long row_bytes = (unsigned long long)nb * Q8_0_BYTES;

    const unsigned int kqs = VDR * (tid % TPB);
    const unsigned int blocks_per_iter = THREADS / TPB;

    float sum = 0.0f;

    for (unsigned int ib = tid / TPB; ib < nb; ib += blocks_per_iter) {

        const char* w_block = weight_q8
            + (unsigned long long)row * row_bytes
            + (unsigned long long)ib * Q8_0_BYTES;

        unsigned short w_scale_bits = (unsigned short)(unsigned char)w_block[0]
                                    | ((unsigned short)(unsigned char)w_block[1] << 8);
        float w_scale = f16_bits_to_f32(w_scale_bits);

        const unsigned char* wq = (const unsigned char*)(w_block + 2);
        int w0, w1;
        {
            const unsigned char* base0 = wq + kqs * 4;
            w0 = (int)(signed char)base0[0]
               | ((int)(signed char)base0[1] << 8)
               | ((int)(signed char)base0[2] << 16)
               | ((int)(signed char)base0[3] << 24);
        }
        {
            const unsigned char* base1 = wq + (kqs + 1) * 4;
            w1 = (int)(signed char)base1[0]
               | ((int)(signed char)base1[1] << 8)
               | ((int)(signed char)base1[2] << 16)
               | ((int)(signed char)base1[3] << 24);
        }

        const char* x_block = input_q8_1 + (unsigned long long)ib * Q8_1_BYTES;

        unsigned short x_scale_bits = (unsigned short)(unsigned char)x_block[0]
                                    | ((unsigned short)(unsigned char)x_block[1] << 8);
        float x_scale = f16_bits_to_f32(x_scale_bits);

        const int* xq = (const int*)(x_block + 4);
        int x0 = xq[kqs];
        int x1 = xq[kqs + 1];

        int acc = __dp4a(w0, x0, 0);
        acc = __dp4a(w1, x1, acc);

        sum += w_scale * x_scale * (float)acc;
    }

    sum = warp_reduce_sum(sum);

    __shared__ float shmem[NWARPS];
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane = tid % WARP_SIZE;

    if (lane == 0) {
        shmem[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0 && lane < NWARPS) {
        float val = shmem[lane];
        val += __shfl_xor_sync(0xffffffff, val, 2);
        val += __shfl_xor_sync(0xffffffff, val, 1);
        if (lane == 0) {
            out[row] = val + residual[row];
        }
    }
}

// ==========================================================================
// Kernel 3: Cooperative Q8Aligned (36B) x Q8_1 -> F32 output.
//
// Same cooperative architecture but for repacked 36-byte blocks where
// quant data at offset +4 is 4-byte aligned. Uses native int* loads
// for weights (eliminates byte-level packing overhead).
//
// Grid:  (out_dim, 1, 1)  -- one block per row
// Block: (128, 1, 1)       -- 4 warps
// ==========================================================================
extern "C" __global__ __launch_bounds__(THREADS, 1) void matvec_q8a_coop_q8_1(
    const char* __restrict__ weight_q8a,  // [out_dim * nb * 36] Q8Aligned bytes
    const char* __restrict__ input_q8_1,  // [nb * 36] Q8_1 pre-quantized input
    float* __restrict__ out,              // [out_dim] F32 output
    unsigned int out_dim,
    unsigned int in_dim)
{
    const unsigned int row = blockIdx.x;
    if (row >= out_dim) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int nb = in_dim >> 5;
    const unsigned long long row_bytes = (unsigned long long)nb * Q8A_BYTES;

    const unsigned int kqs = VDR * (tid % TPB);  // 0, 2, 4, 6
    const unsigned int blocks_per_iter = THREADS / TPB;  // 32

    float sum = 0.0f;

    for (unsigned int ib = tid / TPB; ib < nb; ib += blocks_per_iter) {

        // --- Load Q8Aligned weight block ---
        const char* w_block = weight_q8a
            + (unsigned long long)row * row_bytes
            + (unsigned long long)ib * Q8A_BYTES;

        unsigned short w_scale_bits = (unsigned short)(unsigned char)w_block[0]
                                    | ((unsigned short)(unsigned char)w_block[1] << 8);
        float w_scale = f16_bits_to_f32(w_scale_bits);

        // Native int* loads for weight quant data (4-byte aligned at +4).
        const int* wq = (const int*)(w_block + 4);
        int w0 = wq[kqs];
        int w1 = wq[kqs + 1];

        // --- Load Q8_1 input block ---
        const char* x_block = input_q8_1 + (unsigned long long)ib * Q8_1_BYTES;

        unsigned short x_scale_bits = (unsigned short)(unsigned char)x_block[0]
                                    | ((unsigned short)(unsigned char)x_block[1] << 8);
        float x_scale = f16_bits_to_f32(x_scale_bits);

        const int* xq = (const int*)(x_block + 4);
        int x0 = xq[kqs];
        int x1 = xq[kqs + 1];

        // 2 dp4a calls: 8 multiply-accumulates total.
        int acc = __dp4a(w0, x0, 0);
        acc = __dp4a(w1, x1, acc);

        sum += w_scale * x_scale * (float)acc;
    }

    // --- Warp reduction ---
    sum = warp_reduce_sum(sum);

    // --- Cross-warp reduction via shmem ---
    __shared__ float shmem[NWARPS];
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane = tid % WARP_SIZE;

    if (lane == 0) {
        shmem[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0 && lane < NWARPS) {
        float val = shmem[lane];
        val += __shfl_xor_sync(0xffffffff, val, 2);
        val += __shfl_xor_sync(0xffffffff, val, 1);
        if (lane == 0) {
            out[row] = val;
        }
    }
}

// ==========================================================================
// Kernel 4: Cooperative Q8Aligned (36B) x Q8_1 + residual -> F32 output.
//
// Same as matvec_q8a_coop_q8_1 but with fused residual addition.
// ==========================================================================
extern "C" __global__ __launch_bounds__(THREADS, 1) void matvec_q8a_coop_q8_1_residual(
    const char* __restrict__ weight_q8a,  // [out_dim * nb * 36] Q8Aligned bytes
    const char* __restrict__ input_q8_1,  // [nb * 36] Q8_1 pre-quantized input
    const float* __restrict__ residual,   // [out_dim] F32 residual
    float* __restrict__ out,              // [out_dim] F32 output
    unsigned int out_dim,
    unsigned int in_dim)
{
    const unsigned int row = blockIdx.x;
    if (row >= out_dim) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int nb = in_dim >> 5;
    const unsigned long long row_bytes = (unsigned long long)nb * Q8A_BYTES;

    const unsigned int kqs = VDR * (tid % TPB);
    const unsigned int blocks_per_iter = THREADS / TPB;

    float sum = 0.0f;

    for (unsigned int ib = tid / TPB; ib < nb; ib += blocks_per_iter) {

        const char* w_block = weight_q8a
            + (unsigned long long)row * row_bytes
            + (unsigned long long)ib * Q8A_BYTES;

        unsigned short w_scale_bits = (unsigned short)(unsigned char)w_block[0]
                                    | ((unsigned short)(unsigned char)w_block[1] << 8);
        float w_scale = f16_bits_to_f32(w_scale_bits);

        const int* wq = (const int*)(w_block + 4);
        int w0 = wq[kqs];
        int w1 = wq[kqs + 1];

        const char* x_block = input_q8_1 + (unsigned long long)ib * Q8_1_BYTES;

        unsigned short x_scale_bits = (unsigned short)(unsigned char)x_block[0]
                                    | ((unsigned short)(unsigned char)x_block[1] << 8);
        float x_scale = f16_bits_to_f32(x_scale_bits);

        const int* xq = (const int*)(x_block + 4);
        int x0 = xq[kqs];
        int x1 = xq[kqs + 1];

        int acc = __dp4a(w0, x0, 0);
        acc = __dp4a(w1, x1, acc);

        sum += w_scale * x_scale * (float)acc;
    }

    sum = warp_reduce_sum(sum);

    __shared__ float shmem[NWARPS];
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane = tid % WARP_SIZE;

    if (lane == 0) {
        shmem[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0 && lane < NWARPS) {
        float val = shmem[lane];
        val += __shfl_xor_sync(0xffffffff, val, 2);
        val += __shfl_xor_sync(0xffffffff, val, 1);
        if (lane == 0) {
            out[row] = val + residual[row];
        }
    }
}
