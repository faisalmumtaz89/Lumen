// Tiled GEMM: C[M,N] = A[M,K] * B^T[N,K]
//
// For prefill: M=batch_size, K=hidden_dim, N=out_dim
// A = activation matrix [M, K] row-major
// B = weight matrix [N, K] row-major (transposed during compute: B^T gives [K, N])
//
// Classic tiled GEMM with shared memory for both A and B tiles.
// Tile size: 32x32 with K-dimension tiles of 32.
// Each thread block computes one TILE_M x TILE_N output tile.
//
// Shared memory tiles use +1 column padding (e.g. As[32][33]) to eliminate
// 32-way bank conflicts during the K-loop accumulation. Without padding,
// all 32 threads in a warp read the same shared memory bank on every
// iteration of the inner product loop, serializing access. The extra
// column shifts each row by one bank, giving conflict-free access.
//
// NVRTC-compatible: no system includes, extern "C" linkage.

#define TILE_M 32
#define TILE_N 32
#define TILE_K 32
#define SMEM_PAD 1   // +1 padding to avoid 32-way shared memory bank conflicts
#define BLOCK_SIZE_X 32  // threads per block in N-dimension
#define BLOCK_SIZE_Y 32  // threads per block in M-dimension (matches TILE_M)

// Tiled GEMM: C = A * B^T
// A: [M, K] row-major (activation matrix, batch_size x hidden_dim)
// B: [N, K] row-major (weight matrix, out_dim x hidden_dim; transposed implicitly)
// C: [M, N] row-major (output, batch_size x out_dim)
//
// Grid:  (ceil(N/TILE_N), ceil(M/TILE_M), 1)
// Block: (BLOCK_SIZE_X, BLOCK_SIZE_Y, 1)
//
// Each thread computes one element of C. The thread at (tx, ty) within a block
// computes C[block_row + ty][block_col + tx].
//
// Shared memory holds one TILE_M x TILE_K tile of A and one TILE_N x TILE_K
// tile of B per K-loop iteration. Both tiles are loaded cooperatively by all
// threads in the block.
extern "C" __global__ void gemm_f32(
    const float* __restrict__ A,  // [M, K]
    const float* __restrict__ B,  // [N, K] (row-major, used as B^T)
    float* __restrict__ C,        // [M, N]
    unsigned int M,
    unsigned int N,
    unsigned int K)
{
    // Thread indices within the block.
    unsigned int tx = threadIdx.x; // column within tile (0..TILE_N-1)
    unsigned int ty = threadIdx.y; // row within tile (0..TILE_M-1)

    // Global row and column of the C element this thread computes.
    unsigned int row = blockIdx.y * TILE_M + ty;
    unsigned int col = blockIdx.x * TILE_N + tx;

    // Shared memory for tiles of A and B.
    // +1 column padding eliminates 32-way bank conflicts in the K-loop.
    __shared__ float As[TILE_M][TILE_K + SMEM_PAD];
    __shared__ float Bs[TILE_N][TILE_K + SMEM_PAD];

    float sum = 0.0f;

    // Number of K-dimension tiles (rounded up).
    unsigned int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    for (unsigned int t = 0; t < num_k_tiles; t++) {
        // Load one element of the A tile into shared memory.
        // A tile covers rows [block_row..block_row+TILE_M-1],
        //                cols [t*TILE_K..t*TILE_K+TILE_K-1]
        unsigned int a_col = t * TILE_K + tx;
        if (row < M && a_col < K) {
            As[ty][tx] = A[(unsigned long long)row * K + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load one element of the B tile into shared memory.
        // B is [N, K] row-major. We need B^T[K, N], so we load
        // B[col][t*TILE_K + ty] = B^T[t*TILE_K + ty][col].
        unsigned int b_col = t * TILE_K + ty;
        if (col < N && b_col < K) {
            Bs[tx][ty] = B[(unsigned long long)col * K + b_col];
        } else {
            Bs[tx][ty] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product for this K-tile.
        // C[row][col] += sum_k(As[ty][k] * Bs[tx][k])
        // Note: Bs is stored as [TILE_N][TILE_K+1], and we index Bs[tx][k]
        // which gives us B^T[k][col] = B[col][k].
        #pragma unroll
        for (unsigned int k = 0; k < TILE_K; k++) {
            sum += As[ty][k] * Bs[tx][k];
        }

        __syncthreads();
    }

    // Write the result. Bounds check prevents out-of-tile writes when
    // M or N is not a multiple of TILE_M/TILE_N.
    if (row < M && col < N) {
        C[(unsigned long long)row * N + col] = sum;
    }
}

// Tiled GEMM with fused residual addition: C = A * B^T + residual
// residual: [M, N] row-major, added element-wise to the GEMM output.
//
// Used for output projection + residual in the prefill path.
extern "C" __global__ void gemm_f32_residual(
    const float* __restrict__ A,         // [M, K]
    const float* __restrict__ B,         // [N, K] (row-major, used as B^T)
    const float* __restrict__ residual,  // [M, N]
    float* __restrict__ C,               // [M, N]
    unsigned int M,
    unsigned int N,
    unsigned int K)
{
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int row = blockIdx.y * TILE_M + ty;
    unsigned int col = blockIdx.x * TILE_N + tx;

    __shared__ float As[TILE_M][TILE_K + SMEM_PAD];
    __shared__ float Bs[TILE_N][TILE_K + SMEM_PAD];

    float sum = 0.0f;
    unsigned int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    for (unsigned int t = 0; t < num_k_tiles; t++) {
        unsigned int a_col = t * TILE_K + tx;
        if (row < M && a_col < K) {
            As[ty][tx] = A[(unsigned long long)row * K + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        unsigned int b_col = t * TILE_K + ty;
        if (col < N && b_col < K) {
            Bs[tx][ty] = B[(unsigned long long)col * K + b_col];
        } else {
            Bs[tx][ty] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (unsigned int k = 0; k < TILE_K; k++) {
            sum += As[ty][k] * Bs[tx][k];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        unsigned long long idx = (unsigned long long)row * N + col;
        C[idx] = sum + residual[idx];
    }
}
