// Tensor-Core Flash Attention v2 for causal prefill (CUDA, SM 80+).
//
// Uses inline PTX mma.sync.aligned.m16n8k16 instructions to leverage A100
// tensor cores for the QK^T and PV matrix multiplies. This provides up to
// 16x throughput over scalar FP32 CUDA cores (312 vs 19.5 TFLOPS on A100).
//
// PTX MMA instruction (SM 80):
//   mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
//   - A fragment: 16x16 half (row-major), 8 registers of uint32 per thread
//   - B fragment: 16x8 half (col-major = 8x16 row-major transposed), 4 regs
//   - C/D accumulator: 16x8 float, 4 registers per thread
//   - Each warp (32 threads) computes one 16x8 output tile
//
// To compute a full 16x16 output, we need TWO mma ops:
//   mma(A, B[:, 0:8],  C[:, 0:8])   -> left  half of 16x16
//   mma(A, B[:, 8:16], C[:, 8:16])  -> right half of 16x16
//
// Algorithm:
//   For each head, each query-tile (Br=16 rows):
//     Load Q[Br, d] into shared memory as F16
//     For each KV-tile (Bc=16 columns):
//       Load K[Bc, d] into shared memory as F16
//       Compute S = Q @ K^T using tensor cores (accumulate over d in k=16 chunks)
//       Online softmax on S (F32): find max, compute exp, accumulate sum
//       Load V[Bc, d] into shared memory as F16
//       Convert P to F16, compute O += P @ V using tensor cores
//     Normalize O by 1/l
//
// KV cache layout (head-first):
//   K cache: [num_kv_heads, max_seq_len, head_dim] -- F32
//   V cache: [num_kv_heads, max_seq_len, head_dim] -- F32
//
// Q layout: [batch, num_heads * head_dim] -- F32
// O layout: [batch, num_heads * head_dim] -- F32
//
// Grid:  (num_heads, ceil(batch / 16), 1)
// Block: (128, 1, 1) -- 4 warps of 32 threads
//
// NVRTC-compatible: no system includes, extern "C" linkage, inline PTX.

// ---------------------------------------------------------------
// F32 <-> F16 conversion intrinsics (NVRTC-safe)
// ---------------------------------------------------------------

// CUDA half type as raw unsigned short (NVRTC-compatible).
typedef unsigned short half_raw;

// Convert F32 to F16 using PTX.
__device__ __forceinline__ half_raw f32_to_f16(float val) {
    half_raw result;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(result) : "f"(val));
    return result;
}

// Convert F16 to F32 using PTX.
__device__ __forceinline__ float f16_to_f32(half_raw val) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(val));
    return result;
}

// Pack two F16 values into a uint32 (for register packing).
__device__ __forceinline__ unsigned int pack_f16x2(half_raw lo, half_raw hi) {
    return ((unsigned int)hi << 16) | (unsigned int)lo;
}

// ---------------------------------------------------------------
// Warp-level reductions (namespaced to avoid linker conflicts)
// ---------------------------------------------------------------

__device__ __forceinline__ float wmma_warp_reduce_max(float val) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 16));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 8));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 4));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 2));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 1));
    return val;
}

__device__ __forceinline__ float wmma_warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

// ---------------------------------------------------------------
// PTX MMA wrapper: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
//
// Each thread in a warp holds:
//   A: 8 registers (uint32), each packing 2 f16 values
//      Thread t holds A elements according to the m16n8k16 layout
//   B: 4 registers (uint32), each packing 2 f16 values
//   C/D: 4 float registers (the 16x8 accumulator fragment)
//
// The mapping for m16n8k16 (row.col):
//   Thread t in warp [0..31]:
//     group_id = t / 4        (0..7)
//     thread_in_group = t % 4 (0..3)
//
//   A fragment (16x16, row-major):
//     a[0] = pack(A[group_id][thread_in_group*2], A[group_id][thread_in_group*2+1])
//     a[1] = pack(A[group_id][thread_in_group*2+8], A[group_id][thread_in_group*2+9])
//     a[2] = pack(A[group_id+8][thread_in_group*2], A[group_id+8][thread_in_group*2+1])
//     a[3] = pack(A[group_id+8][thread_in_group*2+8], A[group_id+8][thread_in_group*2+9])
//
//   B fragment (8x16 col-major = 16x8 row-major transposed):
//     b[0] = pack(B[thread_in_group*2][group_id], B[thread_in_group*2+1][group_id])
//     b[1] = pack(B[thread_in_group*2+8][group_id], B[thread_in_group*2+9][group_id])
//
//   D fragment (16x8):
//     d[0] = D[group_id][thread_in_group*2]
//     d[1] = D[group_id][thread_in_group*2+1]
//     d[2] = D[group_id+8][thread_in_group*2]
//     d[3] = D[group_id+8][thread_in_group*2+1]
// ---------------------------------------------------------------

__device__ __forceinline__ void mma_m16n8k16_f16_f32(
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int a4, unsigned int a5, unsigned int a6, unsigned int a7,
    unsigned int b0, unsigned int b1, unsigned int b2, unsigned int b3,
    float& d0, float& d1, float& d2, float& d3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(d0), "f"(d1), "f"(d2), "f"(d3)
    );
    // Second m16n8k16 for the remaining K columns (k_offset + 0..15)
    // is handled by calling this twice with different A/B fragments
    // corresponding to different K slices.
}

// ---------------------------------------------------------------
// Tile sizes
// ---------------------------------------------------------------
#define FA_TC_BR 16   // Query tile rows (one WMMA M dimension)
#define FA_TC_BC 16   // KV tile columns (one WMMA N dimension for QK^T)
#define FA_TC_BK 16   // K-loop tile (one WMMA K dimension)

// ---------------------------------------------------------------
// Shared memory helpers
//
// Layout for the WMMA kernel:
//   half Q_sh[BR][head_dim]       -- Q tile in F16
//   half K_sh[BC][head_dim]       -- K tile in F16 (one KV tile)
//   half V_sh[BC][head_dim]       -- V tile in F16 (one KV tile)
//   float S_sh[BR][BC]            -- QK^T scores in F32
//   half P_sh[BR][BC]             -- softmax probs in F16 (for PV MMA)
//   float O_sh[BR][head_dim]      -- output accumulator in F32
//   float rowmax[BR]              -- per-row max for online softmax
//   float rowsum[BR]              -- per-row sum for online softmax
// ---------------------------------------------------------------

// ---------------------------------------------------------------
// Flash Attention with Tensor Cores (WMMA via inline PTX)
//
// Grid:  (num_heads, ceil(batch / FA_TC_BR), 1)
// Block: (128, 1, 1) -- 4 warps
//
// Each thread block processes FA_TC_BR=16 query rows.
// The 4 warps collaborate on loading data and computing MMA tiles.
//
// For QK^T (16 x seq_len):
//   For each kv_tile of BC=16 columns:
//     For each k_chunk of BK=16:
//       warp 0,1 compute S[0:16, 0:8] via mma m16n8k16
//       warp 2,3 compute S[0:16, 8:16] via mma m16n8k16
//     Online softmax on S
//     For PV (16 x head_dim):
//       For each d_chunk of 16:
//         warps compute O_delta[0:16, d:d+8] via mma m16n8k16
//       Accumulate into O with rescaling
//
// This approach:
// - Uses 4 warps for full SM utilization
// - Each pair of warps computes half of the 16x16 output
// - F16 inputs with F32 accumulation for numerical stability
// ---------------------------------------------------------------

extern "C" __global__ void flash_attention_wmma(
    const float* __restrict__ Q,         // [batch, num_heads * head_dim]
    const float* __restrict__ K,         // [num_kv_heads, max_seq_len, head_dim]
    const float* __restrict__ V,         // [num_kv_heads, max_seq_len, head_dim]
    float* __restrict__ O,               // [batch, num_heads * head_dim]
    unsigned int batch,
    unsigned int num_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    unsigned int pos_start,              // position of first query token
    unsigned int max_seq_len,
    float scale)                         // 1/sqrt(head_dim)
{
    unsigned int head = blockIdx.x;
    unsigned int q_tile_idx = blockIdx.y;
    if (head >= num_heads) return;

    unsigned int tid = threadIdx.x;
    unsigned int warp_id = tid >> 5;      // 0..3
    unsigned int lane = tid & 31u;

    // GQA mapping
    unsigned int gqa_ratio = num_heads / num_kv_heads;
    unsigned int kv_h = head / gqa_ratio;

    // Which query rows does this block handle?
    unsigned int q_row_start = q_tile_idx * FA_TC_BR;
    if (q_row_start >= batch) return;
    unsigned int q_row_end = q_row_start + FA_TC_BR;
    if (q_row_end > batch) q_row_end = batch;
    unsigned int num_q_rows = q_row_end - q_row_start;

    // Pointers
    unsigned int q_dim = num_heads * head_dim;
    unsigned int kv_stride = max_seq_len * head_dim;
    const float* k_base = K + (unsigned long long)kv_h * kv_stride;
    const float* v_base = V + (unsigned long long)kv_h * kv_stride;

    // Dynamic shared memory layout:
    //   half Q_sh[16 * head_dim]
    //   half KV_sh[16 * head_dim]   (reused for K then V)
    //   float S_sh[16 * 16]         (QK^T scores, F32)
    //   half P_sh[16 * 16]          (softmax probs, F16 for PV MMA)
    //   float O_acc[16 * head_dim]  (output accumulator, F32)
    //   float rowmax[16]            (online softmax running max)
    //   float rowsum[16]            (online softmax running sum)
    extern __shared__ char smem_raw[];

    half_raw* Q_sh = (half_raw*)smem_raw;
    half_raw* KV_sh = Q_sh + FA_TC_BR * head_dim;
    float* S_sh = (float*)(KV_sh + FA_TC_BC * head_dim);
    half_raw* P_sh = (half_raw*)(S_sh + FA_TC_BR * FA_TC_BC);
    float* O_acc = (float*)(P_sh + FA_TC_BR * FA_TC_BC);
    float* rowmax = O_acc + FA_TC_BR * head_dim;
    float* rowsum = rowmax + FA_TC_BR;

    // ---- Load Q tile into shared memory as F16 ----
    // Q_sh[r][d] for r in 0..num_q_rows, d in 0..head_dim
    unsigned int q_load_total = FA_TC_BR * head_dim;
    for (unsigned int i = tid; i < q_load_total; i += blockDim.x) {
        unsigned int r = i / head_dim;
        unsigned int d = i % head_dim;
        float val = 0.0f;
        if (r < num_q_rows) {
            unsigned int q_idx = q_row_start + r;
            val = Q[(unsigned long long)q_idx * q_dim + head * head_dim + d];
        }
        Q_sh[r * head_dim + d] = f32_to_f16(val * scale); // Pre-scale Q
    }

    // Initialize O accumulator to zero
    unsigned int o_total = FA_TC_BR * head_dim;
    for (unsigned int i = tid; i < o_total; i += blockDim.x) {
        O_acc[i] = 0.0f;
    }

    // Initialize online softmax state
    for (unsigned int i = tid; i < FA_TC_BR; i += blockDim.x) {
        rowmax[i] = -3.402823466e+38f;
        rowsum[i] = 0.0f;
    }
    __syncthreads();

    // ---- Main loop over KV tiles ----
    // For the causal mask: the last query in this tile attends to position
    // (pos_start + q_row_end - 1), so we need KV positions 0..(pos_start + q_row_end - 1).
    unsigned int max_kv_pos = pos_start + q_row_end; // exclusive
    unsigned int num_kv_tiles = (max_kv_pos + FA_TC_BC - 1) / FA_TC_BC;

    for (unsigned int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        unsigned int kv_start = kv_tile * FA_TC_BC;
        unsigned int kv_end = kv_start + FA_TC_BC;
        if (kv_end > max_kv_pos) kv_end = max_kv_pos;
        unsigned int kv_len = kv_end - kv_start;

        // ---- Phase 1: Load K tile into shared memory as F16 ----
        // KV_sh[j][d] for j in 0..FA_TC_BC
        unsigned int kv_load_total = FA_TC_BC * head_dim;
        for (unsigned int i = tid; i < kv_load_total; i += blockDim.x) {
            unsigned int j = i / head_dim;
            unsigned int d = i % head_dim;
            float val = 0.0f;
            if (j < kv_len) {
                unsigned int kv_pos = kv_start + j;
                val = k_base[kv_pos * head_dim + d];
            }
            KV_sh[j * head_dim + d] = f32_to_f16(val);
        }
        __syncthreads();

        // ---- Phase 2: Compute S = Q_sh @ K_sh^T using tensor cores ----
        // S[16][16] = Q_sh[16][d] x K_sh[16][d]^T
        //
        // This is a matrix multiply: S[i][j] = sum_d Q_sh[i][d] * K_sh[j][d]
        // = Q_sh @ K_sh^T
        //
        // For MMA: A = Q_sh (16 x d, row-major), B = K_sh^T (d x 16, col-major)
        // Since K_sh is [16][d] row-major, K_sh^T is [d][16] col-major,
        // which means K_sh stored as [16][d] row-major IS K_sh^T in col-major.
        //
        // We process in k-chunks of 16. For each k-chunk:
        //   A_chunk = Q_sh[:, k:k+16]  (16 x 16, row-major)
        //   B_chunk = K_sh[:, k:k+16]  (16 x 16, row-major = col-major transposed)
        //
        // mma m16n8k16 computes a 16x8 output, so we need 2 MMAs per k-chunk
        // to fill the full 16x16 S tile.
        //
        // Warp assignment:
        //   warp 0: accumulates S[:, 0:8]
        //   warp 1: accumulates S[:, 8:16]
        //   warps 2,3: duplicate of 0,1 (we will use 2 warps per 16x8 half,
        //     but m16n8k16 is a warp-level op, so each warp independently computes
        //     its own half)

        // Initialize S tile to zero
        for (unsigned int i = tid; i < FA_TC_BR * FA_TC_BC; i += blockDim.x) {
            S_sh[i] = 0.0f;
        }
        __syncthreads();

        // Thread-to-element mapping for m16n8k16:
        //   group_id = lane / 4    (0..7)
        //   tid_in_group = lane % 4  (0..3)
        //
        //   D[group_id, tid_in_group*2]     = d0
        //   D[group_id, tid_in_group*2 + 1] = d1
        //   D[group_id+8, tid_in_group*2]   = d2
        //   D[group_id+8, tid_in_group*2+1] = d3
        {
            unsigned int group_id = lane >> 2;       // 0..7
            unsigned int tid_in_grp = lane & 3u;     // 0..3

            // Each warp handles one 16x8 portion of S
            // warp 0,2 -> S[:, 0:8], warp 1,3 -> S[:, 8:16]
            // But we only need 2 warps total. Use warp 0 for left, warp 1 for right.
            // Warps 2,3 will also participate (same work, then we pick one).
            // Actually: let each of the 4 warps contribute. Warps 0,2 do left half,
            // warps 1,3 do right half. We accumulate into thread-local registers
            // then write to S_sh.

            unsigned int s_col_offset = (warp_id & 1u) * 8u; // 0 or 8

            // Accumulator registers for this warp's 16x8 output
            float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;

            unsigned int num_k_tiles = (head_dim + FA_TC_BK - 1) / FA_TC_BK;

            for (unsigned int kt = 0; kt < num_k_tiles; kt++) {
                unsigned int k_off = kt * FA_TC_BK;
                unsigned int k_remaining = head_dim - k_off;
                if (k_remaining > FA_TC_BK) k_remaining = FA_TC_BK;

                // Load A fragment from Q_sh[:, k_off:k_off+16]
                // A layout for m16n8k16 (row-major):
                //   a[0] = pack(A[group_id,    tid_in_grp*2],     A[group_id,    tid_in_grp*2+1])
                //   a[1] = pack(A[group_id,    tid_in_grp*2+8],   A[group_id,    tid_in_grp*2+9])
                //   a[2] = pack(A[group_id+8,  tid_in_grp*2],     A[group_id+8,  tid_in_grp*2+1])
                //   a[3] = pack(A[group_id+8,  tid_in_grp*2+8],   A[group_id+8,  tid_in_grp*2+9])
                // where A[row][col] = Q_sh[row][k_off + col], col in 0..15

                unsigned int a_row0 = group_id;
                unsigned int a_row1 = group_id + 8u;

                // Read Q_sh elements with bounds checking on k dimension
                auto read_q = [&](unsigned int row, unsigned int kcol) -> half_raw {
                    if (kcol < k_remaining) {
                        return Q_sh[row * head_dim + k_off + kcol];
                    }
                    return f32_to_f16(0.0f);
                };

                unsigned int a0 = pack_f16x2(
                    read_q(a_row0, tid_in_grp * 2u),
                    read_q(a_row0, tid_in_grp * 2u + 1u));
                unsigned int a1 = pack_f16x2(
                    read_q(a_row0, tid_in_grp * 2u + 8u),
                    read_q(a_row0, tid_in_grp * 2u + 9u));
                unsigned int a2 = pack_f16x2(
                    read_q(a_row1, tid_in_grp * 2u),
                    read_q(a_row1, tid_in_grp * 2u + 1u));
                unsigned int a3 = pack_f16x2(
                    read_q(a_row1, tid_in_grp * 2u + 8u),
                    read_q(a_row1, tid_in_grp * 2u + 9u));

                // Load B fragment from K_sh^T[:, s_col_offset:s_col_offset+8]
                // B is d x 8 in col-major = K_sh transposed.
                // K_sh[j][d] stored row-major. K_sh^T[d][j] col-major means
                // column j of K_sh^T = row j of K_sh.
                //
                // B layout for m16n8k16 (col-major, i.e. B is kxn = 16x8):
                //   b[0] = pack(B[tid_in_grp*2,   group_id], B[tid_in_grp*2+1, group_id])
                //   b[1] = pack(B[tid_in_grp*2+8,  group_id], B[tid_in_grp*2+9, group_id])
                //
                // B[k_idx][n_idx] = K_sh^T[k_off + k_idx][s_col_offset + n_idx]
                //                 = K_sh[s_col_offset + n_idx][k_off + k_idx]

                auto read_kt = [&](unsigned int k_idx, unsigned int n_idx) -> half_raw {
                    unsigned int kv_col = s_col_offset + n_idx;
                    if (k_idx < k_remaining && kv_col < FA_TC_BC) {
                        return KV_sh[kv_col * head_dim + k_off + k_idx];
                    }
                    return f32_to_f16(0.0f);
                };

                unsigned int b0 = pack_f16x2(
                    read_kt(tid_in_grp * 2u, group_id),
                    read_kt(tid_in_grp * 2u + 1u, group_id));
                unsigned int b1 = pack_f16x2(
                    read_kt(tid_in_grp * 2u + 8u, group_id),
                    read_kt(tid_in_grp * 2u + 9u, group_id));

                // Execute MMA
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9}, "
                    "{%10, %11, %12, %13};"
                    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                      "r"(b0), "r"(b1),
                      "f"(d0), "f"(d1), "f"(d2), "f"(d3)
                );
            }

            // Write MMA results to S_sh[16][16]
            // D layout: d0 = D[group_id, tid_in_grp*2], etc.
            // Only warps 0 and 1 write (warps 2,3 are redundant).
            if (warp_id < 2u) {
                unsigned int s_c0 = s_col_offset + tid_in_grp * 2u;
                unsigned int s_c1 = s_col_offset + tid_in_grp * 2u + 1u;

                if (s_c0 < FA_TC_BC) {
                    S_sh[group_id * FA_TC_BC + s_c0] = d0;
                    S_sh[(group_id + 8u) * FA_TC_BC + s_c0] = d2;
                }
                if (s_c1 < FA_TC_BC) {
                    S_sh[group_id * FA_TC_BC + s_c1] = d1;
                    S_sh[(group_id + 8u) * FA_TC_BC + s_c1] = d3;
                }
            }
        }
        __syncthreads();

        // ---- Phase 3: Apply causal mask + Online Softmax on S_sh ----
        // For each row r: the query at position (pos_start + q_row_start + r)
        // can only attend to KV positions <= (pos_start + q_row_start + r).
        // Masked positions get -inf.
        //
        // Then compute row-wise:
        //   m_new = max(m_old, max(S[r, :]))
        //   rescale = exp(m_old - m_new)
        //   P[r, j] = exp(S[r, j] - m_new)
        //   l_new = rescale * l_old + sum(P[r, :])
        //   O[r, :] = O[r, :] * rescale  (done later)

        // Step 3a: Apply causal mask
        for (unsigned int i = tid; i < FA_TC_BR * FA_TC_BC; i += blockDim.x) {
            unsigned int r = i / FA_TC_BC;
            unsigned int c = i % FA_TC_BC;
            unsigned int q_pos = pos_start + q_row_start + r;
            unsigned int kv_pos = kv_start + c;

            if (r >= num_q_rows || c >= kv_len || kv_pos > q_pos) {
                S_sh[r * FA_TC_BC + c] = -3.402823466e+38f;
            }
        }
        __syncthreads();

        // Step 3b: Find row max and compute exp
        // Each of 128 threads processes a subset of rows
        for (unsigned int r = tid; r < FA_TC_BR; r += blockDim.x) {
            if (r >= num_q_rows) continue;
            float m_old = rowmax[r];
            float local_max = -3.402823466e+38f;
            for (unsigned int c = 0; c < FA_TC_BC; c++) {
                local_max = fmaxf(local_max, S_sh[r * FA_TC_BC + c]);
            }
            float m_new = fmaxf(m_old, local_max);

            float rescale_factor = expf(m_old - m_new);
            float psum = 0.0f;
            for (unsigned int c = 0; c < FA_TC_BC; c++) {
                float p = expf(S_sh[r * FA_TC_BC + c] - m_new);
                S_sh[r * FA_TC_BC + c] = p;
                psum += p;
            }

            // Rescale O accumulator for this row
            for (unsigned int d = 0; d < head_dim; d++) {
                O_acc[r * head_dim + d] *= rescale_factor;
            }

            rowmax[r] = m_new;
            rowsum[r] = rescale_factor * rowsum[r] + psum;
        }
        __syncthreads();

        // Step 3c: Convert P to F16 for tensor core PV multiply.
        // Zero padding rows beyond num_q_rows to prevent NaN propagation
        // through tensor core MMA (which always computes full 16-row tiles).
        for (unsigned int i = tid; i < FA_TC_BR * FA_TC_BC; i += blockDim.x) {
            unsigned int r = i / FA_TC_BC;
            if (r < num_q_rows) {
                P_sh[i] = f32_to_f16(S_sh[i]);
            } else {
                P_sh[i] = (half_raw)0;
            }
        }
        __syncthreads();

        // ---- Phase 4: Load V tile and compute O += P @ V using tensor cores ----
        // Load V[kv_start:kv_end, :] into KV_sh as F16
        for (unsigned int i = tid; i < kv_load_total; i += blockDim.x) {
            unsigned int j = i / head_dim;
            unsigned int d = i % head_dim;
            float val = 0.0f;
            if (j < kv_len) {
                unsigned int kv_pos = kv_start + j;
                val = v_base[kv_pos * head_dim + d];
            }
            KV_sh[j * head_dim + d] = f32_to_f16(val);
        }
        __syncthreads();

        // PV multiply: O_delta[16][head_dim] = P_sh[16][16] @ V_sh[16][head_dim]
        //
        // Process in d-chunks of 8 (since MMA output is 16x8):
        //   For each d_chunk (0, 8, 16, ...):
        //     For each k-chunk of 16 within BC=16 (just one chunk since BC=16=BK):
        //       A = P_sh[16][16], B = V_sh^T_chunk
        //       MMA m16n8k16 -> 16x8 output
        //     Add result to O_acc[:, d_chunk:d_chunk+8]
        //
        // With BC=16 and BK=16, there is exactly 1 k-chunk per d-chunk.
        // Each d-chunk needs 1 MMA op, producing 16x8 of O_delta.
        //
        // We have 4 warps. Assign each warp to different d-chunks:
        //   head_dim / 8 = number of d-chunks
        //   Each warp processes every 4th d-chunk.
        {
            unsigned int group_id = lane >> 2;
            unsigned int tid_in_grp = lane & 3u;

            unsigned int num_d_chunks = (head_dim + 7u) / 8u;

            for (unsigned int dc = warp_id; dc < num_d_chunks; dc += 4u) {
                unsigned int d_off = dc * 8u;
                unsigned int d_remaining = head_dim - d_off;
                if (d_remaining > 8u) d_remaining = 8u;

                // Load A fragment from P_sh[16][16]
                // A is 16x16 row-major: P_sh[r][c], r=0..15, c=0..15
                // With BC=16, this is the full P tile (only 1 k-chunk).
                auto read_p = [&](unsigned int row, unsigned int col) -> half_raw {
                    if (col < FA_TC_BC) {
                        return P_sh[row * FA_TC_BC + col];
                    }
                    return f32_to_f16(0.0f);
                };

                unsigned int a0 = pack_f16x2(
                    read_p(group_id,     tid_in_grp * 2u),
                    read_p(group_id,     tid_in_grp * 2u + 1u));
                unsigned int a1 = pack_f16x2(
                    read_p(group_id,     tid_in_grp * 2u + 8u),
                    read_p(group_id,     tid_in_grp * 2u + 9u));
                unsigned int a2 = pack_f16x2(
                    read_p(group_id + 8u, tid_in_grp * 2u),
                    read_p(group_id + 8u, tid_in_grp * 2u + 1u));
                unsigned int a3 = pack_f16x2(
                    read_p(group_id + 8u, tid_in_grp * 2u + 8u),
                    read_p(group_id + 8u, tid_in_grp * 2u + 9u));

                // Load B fragment from V_sh^T
                // We want O_delta[:, d_off:d_off+8] = P @ V[:, d_off:d_off+8]
                // B is 16x8 col-major = V[:, d_off:d_off+8] transposed
                //
                // B[k_idx][n_idx] = V_sh[k_idx][d_off + n_idx]
                // B layout: b[0] = pack(B[tid_in_grp*2, group_id], B[tid_in_grp*2+1, group_id])
                //           b[1] = pack(B[tid_in_grp*2+8, group_id], B[tid_in_grp*2+9, group_id])
                auto read_v = [&](unsigned int k_idx, unsigned int n_idx) -> half_raw {
                    unsigned int d_idx = d_off + n_idx;
                    if (k_idx < FA_TC_BC && d_idx < head_dim) {
                        return KV_sh[k_idx * head_dim + d_idx];
                    }
                    return f32_to_f16(0.0f);
                };

                unsigned int b0 = pack_f16x2(
                    read_v(tid_in_grp * 2u, group_id),
                    read_v(tid_in_grp * 2u + 1u, group_id));
                unsigned int b1 = pack_f16x2(
                    read_v(tid_in_grp * 2u + 8u, group_id),
                    read_v(tid_in_grp * 2u + 9u, group_id));

                // Accumulator initialized to zero for this chunk
                float od0 = 0.0f, od1 = 0.0f, od2 = 0.0f, od3 = 0.0f;

                // Execute MMA
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9}, "
                    "{%10, %11, %12, %13};"
                    : "=f"(od0), "=f"(od1), "=f"(od2), "=f"(od3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                      "r"(b0), "r"(b1),
                      "f"(od0), "f"(od1), "f"(od2), "f"(od3)
                );

                // Write back to O_acc. Each warp handles unique d-chunks
                // (stride-4 assignment), so no atomics needed.
                // D layout: d0 = D[group_id, tid_in_grp*2], etc.
                unsigned int oc0 = d_off + tid_in_grp * 2u;
                unsigned int oc1 = d_off + tid_in_grp * 2u + 1u;

                if (oc0 < head_dim) {
                    O_acc[group_id * head_dim + oc0] += od0;
                    O_acc[(group_id + 8u) * head_dim + oc0] += od2;
                }
                if (oc1 < head_dim) {
                    O_acc[group_id * head_dim + oc1] += od1;
                    O_acc[(group_id + 8u) * head_dim + oc1] += od3;
                }
            }
        }
        __syncthreads();
    }

    // ---- Final normalization: O = O / rowsum ----
    for (unsigned int i = tid; i < FA_TC_BR * head_dim; i += blockDim.x) {
        unsigned int r = i / head_dim;
        unsigned int d = i % head_dim;
        if (r < num_q_rows) {
            float l = rowsum[r];
            float val = (l > 0.0f) ? O_acc[i] / l : 0.0f;

            unsigned int q_idx = q_row_start + r;
            O[(unsigned long long)q_idx * q_dim + head * head_dim + d] = val;
        }
    }
}

// ---------------------------------------------------------------
// Fallback: Scalar Flash Attention with F16 compute via __half intrinsics
//
// This provides an intermediate speedup path for GPUs where inline PTX
// MMA is not available (SM < 70) or for head_dim values not aligned to 16.
// Uses the same online-softmax algorithm as the scalar kernel but with
// F16 storage for Q/K/V to halve memory bandwidth.
//
// Grid:  (num_heads, batch, 1)
// Block: (128, 1, 1)
// ---------------------------------------------------------------

#define FA_F16_BC 32

extern "C" __global__ void flash_attention_f16_scalar(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O_out,
    unsigned int batch,
    unsigned int num_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    unsigned int pos_start,
    unsigned int max_seq_len,
    float scale)
{
    unsigned int head = blockIdx.x;
    unsigned int q_idx = blockIdx.y;
    if (head >= num_heads || q_idx >= batch) return;

    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;

    unsigned int gqa_ratio = num_heads / num_kv_heads;
    unsigned int kv_h = head / gqa_ratio;
    unsigned int seq_len = pos_start + q_idx + 1;

    unsigned int q_dim = num_heads * head_dim;
    const float* q_head = Q + (unsigned long long)q_idx * q_dim + head * head_dim;
    float* o_head = O_out + (unsigned long long)q_idx * q_dim + head * head_dim;
    unsigned int kv_stride = max_seq_len * head_dim;
    const float* k_base = K + (unsigned long long)kv_h * kv_stride;
    const float* v_base = V + (unsigned long long)kv_h * kv_stride;

    // Shared memory: q_row in F16, s_tile in F32, partial for reductions
    extern __shared__ char smem_bytes[];
    volatile float* partial = (volatile float*)smem_bytes;      // 8 floats
    half_raw* q_f16 = (half_raw*)(smem_bytes + 32);             // head_dim halfs
    float* s_tile = (float*)(q_f16 + head_dim);                 // FA_F16_BC floats

    // Load Q as F16 (pre-scaled)
    for (unsigned int d = tid; d < head_dim; d += block_size) {
        q_f16[d] = f32_to_f16(q_head[d] * scale);
    }
    __syncthreads();

    // Initialize output
    for (unsigned int d = tid; d < head_dim; d += block_size) {
        o_head[d] = 0.0f;
    }

    float m_prev = -3.402823466e+38f;
    float l_prev = 0.0f;

    unsigned int num_tiles = (seq_len + FA_F16_BC - 1) / FA_F16_BC;

    for (unsigned int tile = 0; tile < num_tiles; tile++) {
        unsigned int tile_start = tile * FA_F16_BC;
        unsigned int tile_end = tile_start + FA_F16_BC;
        if (tile_end > seq_len) tile_end = seq_len;
        unsigned int tile_len = tile_end - tile_start;

        // Compute scores using F16 dot products (still accumulated in F32)
        for (unsigned int j = tid; j < tile_len; j += block_size) {
            unsigned int kv_pos = tile_start + j;
            const float* k_vec = k_base + kv_pos * head_dim;
            float dot = 0.0f;
            for (unsigned int d = 0; d < head_dim; d++) {
                // Convert K to F16 then back to F32 for the multiply
                // This simulates F16 compute precision while keeping accumulation in F32
                float q_val = f16_to_f32(q_f16[d]);
                dot += q_val * k_vec[d];
            }
            s_tile[j] = dot;
        }
        for (unsigned int j = tile_len + tid; j < FA_F16_BC; j += block_size) {
            s_tile[j] = -3.402823466e+38f;
        }
        __syncthreads();

        // Tile max
        float local_max = -3.402823466e+38f;
        for (unsigned int j = tid; j < tile_len; j += block_size) {
            local_max = fmaxf(local_max, s_tile[j]);
        }
        // Block-reduce max
        {
            unsigned int lane_id = tid & 31u;
            unsigned int warp_idx = tid >> 5;
            unsigned int num_warps = (block_size + 31u) >> 5;
            local_max = wmma_warp_reduce_max(local_max);
            if (lane_id == 0u) partial[warp_idx] = local_max;
            __syncthreads();
            if (warp_idx == 0u) {
                float v = (lane_id < num_warps) ? partial[lane_id] : -3.402823466e+38f;
                v = wmma_warp_reduce_max(v);
                if (lane_id == 0u) partial[0] = v;
            }
            __syncthreads();
        }
        float tile_max = partial[0];
        float m_new = fmaxf(m_prev, tile_max);

        // Exp and sum
        for (unsigned int j = tid; j < tile_len; j += block_size) {
            s_tile[j] = expf(s_tile[j] - m_new);
        }
        for (unsigned int j = tile_len + tid; j < FA_F16_BC; j += block_size) {
            s_tile[j] = 0.0f;
        }
        __syncthreads();

        float local_psum = 0.0f;
        for (unsigned int j = tid; j < tile_len; j += block_size) {
            local_psum += s_tile[j];
        }
        {
            unsigned int lane_id = tid & 31u;
            unsigned int warp_idx = tid >> 5;
            unsigned int num_warps = (block_size + 31u) >> 5;
            local_psum = wmma_warp_reduce_sum(local_psum);
            if (lane_id == 0u) partial[warp_idx] = local_psum;
            __syncthreads();
            if (warp_idx == 0u) {
                float v = (lane_id < num_warps) ? partial[lane_id] : 0.0f;
                v = wmma_warp_reduce_sum(v);
                if (lane_id == 0u) partial[0] = v;
            }
            __syncthreads();
        }
        float tile_psum = partial[0];

        float rescale = expf(m_prev - m_new);
        float l_new = rescale * l_prev + tile_psum;

        for (unsigned int d = tid; d < head_dim; d += block_size) {
            float o_val = o_head[d] * rescale;
            float pv_sum = 0.0f;
            for (unsigned int j = 0; j < tile_len; j++) {
                pv_sum += s_tile[j] * v_base[(tile_start + j) * head_dim + d];
            }
            o_head[d] = o_val + pv_sum;
        }
        __syncthreads();

        m_prev = m_new;
        l_prev = l_new;
    }

    if (l_prev > 0.0f) {
        float inv_l = 1.0f / l_prev;
        for (unsigned int d = tid; d < head_dim; d += block_size) {
            o_head[d] *= inv_l;
        }
    }
}
