// GPU-side argmax: finds the index of the maximum value in a float array.
//
// Tie semantics (CORR-011): on exact-equal values this returns the LOWEST
// index, matching the CPU `argmax` in sampling.rs, `argmax_excluding`, and
// llama.cpp. The per-thread strided scan keeps the lowest index within a
// thread (strict `>` over increasing `i`); the warp/block reductions break
// ties by `other_idx < best_idx`. Because "max value, then min index" is an
// associative+commutative reduction operator, every lane converges to the
// global (max-value, min-index) pair regardless of lane assignment. NaN is
// never selected (`>` and `<` on NaN are false), preserving prior behaviour.
//
// Two-phase reduction:
// Phase 1: Each block reduces BLOCK_SIZE elements, writes (max_val, max_idx) to shared mem,
//          then reduces within the block to produce one (val, idx) pair per block.
//          Grid writes partial results to global arrays.
// Phase 2: A single block reduces the partial results to the final argmax.
//
// For vocab_size <= 262144 (256K), a single-block approach with 1024 threads suffices:
// each thread reduces vocab_size/1024 elements, then warp + block reduction.
//
// Grid: (1, 1, 1)   Block: (1024, 1, 1)
// Parameters: data[n], result[1] (output: index of max), n (number of elements)

extern "C" __global__ void argmax_f32(
    const float* __restrict__ data,
    unsigned int* __restrict__ result,
    unsigned int n)
{
    __shared__ float s_val[32];   // one per warp
    __shared__ unsigned int s_idx[32];

    float best_val = -3.402823466e+38f;
    unsigned int best_idx = 0;

    // Each thread strides through the array
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = data[i];
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }

    // Warp-level reduction (lowest-index tie-break: keep smaller idx on equal val)
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_val = __shfl_xor_sync(0xffffffff, best_val, offset);
        unsigned int other_idx = __shfl_xor_sync(0xffffffff, best_idx, offset);
        if (other_val > best_val || (other_val == best_val && other_idx < best_idx)) {
            best_val = other_val;
            best_idx = other_idx;
        }
    }

    unsigned int lane = threadIdx.x & 31;
    unsigned int warp_id = threadIdx.x >> 5;

    if (lane == 0) {
        s_val[warp_id] = best_val;
        s_idx[warp_id] = best_idx;
    }
    __syncthreads();

    // Final reduction by warp 0
    if (warp_id == 0) {
        unsigned int num_warps = blockDim.x >> 5;
        best_val = (lane < num_warps) ? s_val[lane] : -3.402823466e+38f;
        best_idx = (lane < num_warps) ? s_idx[lane] : 0;

        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_val = __shfl_xor_sync(0xffffffff, best_val, offset);
            unsigned int other_idx = __shfl_xor_sync(0xffffffff, best_idx, offset);
            if (other_val > best_val || (other_val == best_val && other_idx < best_idx)) {
                best_val = other_val;
                best_idx = other_idx;
            }
        }

        if (lane == 0) {
            result[0] = best_idx;
        }
    }
}

// ============================================================================
// Tiled two-phase argmax (LUMEN_CUDA_ARGMAX_TILED=1, default-OFF).
//
// The single-block kernel above reads the whole logits vector (vocab 248320 =
// ~1 MB) from ONE SM — single-SM read bandwidth, ~128 us in-bracket on A100.
// Phase 1 spreads the read across NUM_TILES blocks (whole-GPU bandwidth);
// phase 2 reduces the NUM_TILES (val, idx) partials in one tiny block.
//
// SEMANTICS IDENTICAL to argmax_f32 (CORR-011): the reduction operator is
// "max value, then min index", associative + commutative, so the tiled
// grouping produces the SAME (max, min-index) pair — output byte-identical.
// NaN never selected (same comparators). Each phase-1 block scans a
// CONTIGUOUS tile with the same strict-> strided loop, so per-thread lowest-
// index behavior is preserved within tiles; cross-tile ties resolve by
// other_idx < best_idx exactly as the warp reduction does.
// ============================================================================

extern "C" __global__ void argmax_f32_tile_phase1(
    const float* __restrict__ data,
    float* __restrict__ partial_val,      // [gridDim.x]
    unsigned int* __restrict__ partial_idx, // [gridDim.x]
    unsigned int n)
{
    __shared__ float s_val[32];
    __shared__ unsigned int s_idx[32];

    // Contiguous tile per block: [tile_lo, tile_hi)
    unsigned int tile = (n + gridDim.x - 1) / gridDim.x;
    unsigned int lo = blockIdx.x * tile;
    unsigned int hi = lo + tile < n ? lo + tile : n;

    float best_val = -3.402823466e+38f;
    unsigned int best_idx = 0;

    for (unsigned int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
        float v = data[i];
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_val = __shfl_xor_sync(0xffffffff, best_val, offset);
        unsigned int other_idx = __shfl_xor_sync(0xffffffff, best_idx, offset);
        if (other_val > best_val || (other_val == best_val && other_idx < best_idx)) {
            best_val = other_val;
            best_idx = other_idx;
        }
    }

    unsigned int lane = threadIdx.x & 31;
    unsigned int warp_id = threadIdx.x >> 5;
    if (lane == 0) {
        s_val[warp_id] = best_val;
        s_idx[warp_id] = best_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        unsigned int num_warps = blockDim.x >> 5;
        best_val = (lane < num_warps) ? s_val[lane] : -3.402823466e+38f;
        best_idx = (lane < num_warps) ? s_idx[lane] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_val = __shfl_xor_sync(0xffffffff, best_val, offset);
            unsigned int other_idx = __shfl_xor_sync(0xffffffff, best_idx, offset);
            if (other_val > best_val || (other_val == best_val && other_idx < best_idx)) {
                best_val = other_val;
                best_idx = other_idx;
            }
        }
        if (lane == 0) {
            partial_val[blockIdx.x] = best_val;
            partial_idx[blockIdx.x] = best_idx;
        }
    }
}

// Phase 2: one warp reduces the partials. num_partials <= 128.
extern "C" __global__ void argmax_f32_tile_phase2(
    const float* __restrict__ partial_val,
    const unsigned int* __restrict__ partial_idx,
    unsigned int* __restrict__ result,
    unsigned int num_partials)
{
    float best_val = -3.402823466e+38f;
    unsigned int best_idx = 0;
    for (unsigned int i = threadIdx.x; i < num_partials; i += 32) {
        float v = partial_val[i];
        unsigned int idx = partial_idx[i];
        if (v > best_val || (v == best_val && idx < best_idx)) {
            best_val = v;
            best_idx = idx;
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_val = __shfl_xor_sync(0xffffffff, best_val, offset);
        unsigned int other_idx = __shfl_xor_sync(0xffffffff, best_idx, offset);
        if (other_val > best_val || (other_val == best_val && other_idx < best_idx)) {
            best_val = other_val;
            best_idx = other_idx;
        }
    }
    if (threadIdx.x == 0) {
        result[0] = best_idx;
    }
}
