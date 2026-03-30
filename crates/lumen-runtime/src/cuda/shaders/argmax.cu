// GPU-side argmax: finds the index of the maximum value in a float array.
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

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_val = __shfl_xor_sync(0xffffffff, best_val, offset);
        unsigned int other_idx = __shfl_xor_sync(0xffffffff, best_idx, offset);
        if (other_val > best_val) {
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
            if (other_val > best_val) {
                best_val = other_val;
                best_idx = other_idx;
            }
        }

        if (lane == 0) {
            result[0] = best_idx;
        }
    }
}
