// BF16 matvec for the lm_head dispatch (replaces
// cuBLAS HGEMV-BF16 persistent-CTA path with a batch=1-purpose kernel).
//
// Templated mul_mat_vec_f<T, type_acc, ncols_dst, block_size, has_fusion,
// is_multi_token_id> with T=nv_bfloat16, ncols_dst=1, block_size=128,
// has_fusion=false. The BF16 path packs two bf16 elements into nv_bfloat162
// for paired loads, casts each to F32, then F32-MADs against paired F32
// activations. The reduction is standard warp_reduce_sum + shared-mem
// inter-warp reduce.
//
// This kernel targets the SINGLE largest call in BF16 decode: output_proj
// (vocab × hidden). measures cuBLAS HGEMV-BF16 at 1218 µs/call
// × 2245 inst = 125.5 ms / 64-tok = 16.7% TPOT. The purpose-built batch=1
// kernel skips cuBLAS persistent-CTA setup cost.
//
// NVRTC notes:
//   - We avoid <cuda_bf16.h> for compute_61 compatibility. Instead we
//     reinterpret bf16 bits as raw unsigned short; BF16 is the upper 16 bits
//     of an IEEE F32. Casting bf16 -> f32 is a simple bit-shift to the high
//     half of an f32 word, which matches __bfloat162float exactly.
//   - All loads use unsigned int (32-bit) for nv_bfloat162 = (bf16, bf16)
//     packed. The 2 bf16 values are extracted as the low and high 16-bit
//     halves; each is shifted left by 16 to form a bit-equivalent F32.
//   - F32 accumulator (type_acc = float). Standard BF16 type_acc on
//     Ampere/Ada/Hopper.
//
// Grid: (nrows_x, 1, 1)            rpb=1
// Block: (32, 4, 1)                warp_size=32, NWARPS=4 -> 128 threads/CTA
// Shared mem: (NWARPS-1) * WARP_SIZE * sizeof(float) = 3*32*4 = 384 bytes.

typedef unsigned char       uint8_t;
typedef int                 int32_t;
typedef unsigned int        uint32_t;
typedef unsigned long long  uint64_t;

#define WARP_SIZE 32
#define NWARPS 4
#define BLOCK_DIM (WARP_SIZE * NWARPS)  // 128

// BF16 -> F32: shift the 16-bit raw to the upper half of an f32 word.
// Equivalent to __bfloat162float(*reinterpret_cast<const nv_bfloat16*>(&bits)).
__device__ __forceinline__ float bf16_bits_to_f32(unsigned short bits) {
    union { unsigned int u; float f; } cv;
    cv.u = ((unsigned int)bits) << 16;
    return cv.f;
}

// Warp-level sum-reduction (5 butterfly shuffles).
__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val,  8);
    val += __shfl_xor_sync(0xffffffff, val,  4);
    val += __shfl_xor_sync(0xffffffff, val,  2);
    val += __shfl_xor_sync(0xffffffff, val,  1);
    return val;
}

// ============================================================================
// mul_mat_vec_f_bf16
//
// BF16 matvec: mul_mat_vec_f<nv_bfloat16, float, 1, 128, false, false>.
//
// Args:
//   x         : [nrows_x * stride_row] bf16 weights, row-major
//   y         : [ncols_x] F32 activation vector
//   dst       : [nrows_x] F32 output
//   ncols2    : ncols_x / 2 (the kernel reads bf16 in pairs)
//   stride_row: row stride in bf16 ELEMENTS (= ncols_x for contiguous weights)
//
// For a contiguous (V, K) bf16 weight tensor of shape `[nrows_x=V, ncols_x=K]`,
// stride_row should equal K. ncols2 should equal K/2 (kernel asserts K is even
// — for vocab_size × 4096 hidden, this is always true).
// ============================================================================
extern "C" __global__ void mul_mat_vec_f_bf16(
    const unsigned short* __restrict__ x,  // [nrows_x * stride_row] bf16 bits
    const float * __restrict__         y,  // [ncols_x] F32 activation
    float * __restrict__               dst, // [nrows_x] F32 output
    const int                          ncols2,      // = ncols_x / 2
    const int                          stride_row)  // = ncols_x for contig.
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x + threadIdx.y * WARP_SIZE;  // [0..128)

    // Pointer to this row's bf16 weights.
    const unsigned short* x_row = x + (unsigned long long)row * (unsigned long long)stride_row;

    // Cast to (unsigned int *) for paired bf162 loads.
    const unsigned int * x2 = (const unsigned int *) x_row;
    const float2 *       y2 = (const float2 *) y;

    float sumf = 0.0f;

    // Main loop: each thread reads ncols2 / 128 packed bf162 chunks.
    // Striding by BLOCK_DIM (=128) keeps coalesced access patterns.
    for (int col2 = tid; col2 < ncols2; col2 += BLOCK_DIM) {
        const unsigned int xv = x2[col2];
        const float2       yv = y2[col2];

        // Unpack bf162: low 16 bits = element[0], high 16 bits = element[1].
        // (nv_bfloat162.x is element[0], .y is element[1]; in memory, .x is
        //  at the lower address — confirmed by NVIDIA's bf16/half2 layout.)
        const unsigned short xlo = (unsigned short)(xv & 0xFFFF);
        const unsigned short xhi = (unsigned short)((xv >> 16) & 0xFFFF);

        const float xf0 = bf16_bits_to_f32(xlo);
        const float xf1 = bf16_bits_to_f32(xhi);

        // F32 fused multiply-add (ggml_cuda_mad equivalent).
        sumf = __fmaf_rn(xf0, yv.x, sumf);
        sumf = __fmaf_rn(xf1, yv.y, sumf);
    }

    // ----- Reduction stage -----
    //
    // Step 1: warp-internal reduction.
    sumf = warp_reduce_sum(sumf);

    // Step 2: inter-warp via shared mem (block_size > warp_size).
    // `__shared__ float buf_iw[warp_size]`. The non-warp-0 warps write
    // `sumf` (their warp-reduced result) into `buf_iw[warp_id]`. Then warp 0
    // reads `buf_iw[lane]` for lane in [0, NWARPS) and re-reduces.
    __shared__ float buf_iw[WARP_SIZE];

    if (threadIdx.x == 0) {
        buf_iw[threadIdx.y] = sumf;
    }
    __syncthreads();

    if (threadIdx.y != 0) {
        return;
    }

    // Warp 0 only from here.
    sumf = (threadIdx.x < NWARPS) ? buf_iw[threadIdx.x] : 0.0f;
    sumf = warp_reduce_sum(sumf);

    // Step 3: write tid==0 to dst[row].
    if (threadIdx.x != 0) {
        return;
    }
    dst[row] = sumf;
}

// ============================================================================
// mul_mat_vec_f_bf16_n128
//
// Alternate variant: same algorithm but with stride_row baked as ncols_x,
// for the contiguous case. Same kernel name suffix discriminator so the
// dispatch wiring can pick whichever variant matches the weight layout
// without re-compiling.
//
// (Currently identical to mul_mat_vec_f_bf16 — kept as a stub for future
// specialization, e.g. when stride_row != ncols (transposed weights).)
// ============================================================================
extern "C" __global__ void mul_mat_vec_f_bf16_n128(
    const unsigned short* __restrict__ x,
    const float * __restrict__         y,
    float * __restrict__               dst,
    const int                          ncols2,
    const int                          stride_row)
{
    // Forward to main kernel via duplication (NVRTC has no templates without
    // <type_traits>). Identical body — exists so the dispatch side can pick a
    // specialized variant without recompiling.
    const int row = blockIdx.x;
    const int tid = threadIdx.x + threadIdx.y * WARP_SIZE;

    const unsigned short* x_row = x + (unsigned long long)row * (unsigned long long)stride_row;
    const unsigned int * x2 = (const unsigned int *) x_row;
    const float2 *       y2 = (const float2 *) y;

    float sumf = 0.0f;

    for (int col2 = tid; col2 < ncols2; col2 += BLOCK_DIM) {
        const unsigned int xv = x2[col2];
        const float2       yv = y2[col2];

        const unsigned short xlo = (unsigned short)(xv & 0xFFFF);
        const unsigned short xhi = (unsigned short)((xv >> 16) & 0xFFFF);

        const float xf0 = bf16_bits_to_f32(xlo);
        const float xf1 = bf16_bits_to_f32(xhi);

        sumf = __fmaf_rn(xf0, yv.x, sumf);
        sumf = __fmaf_rn(xf1, yv.y, sumf);
    }

    sumf = warp_reduce_sum(sumf);

    __shared__ float buf_iw[WARP_SIZE];
    if (threadIdx.x == 0) {
        buf_iw[threadIdx.y] = sumf;
    }
    __syncthreads();

    if (threadIdx.y != 0) return;

    sumf = (threadIdx.x < NWARPS) ? buf_iw[threadIdx.x] : 0.0f;
    sumf = warp_reduce_sum(sumf);

    if (threadIdx.x != 0) return;
    dst[row] = sumf;
}
