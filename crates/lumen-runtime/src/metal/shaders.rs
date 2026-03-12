//! Metal Shading Language (MSL) kernel source code.
//!
//! All GPU compute kernels are compiled at runtime from this source string
//! via `MTLDevice.newLibraryWithSource()`. Keeping shaders as a Rust string
//! constant avoids file I/O at init time and keeps the single-binary deployment
//! model intact.
//!
//! Kernels are hyper-optimized for Apple Silicon M-series:
//! - SIMD group reductions (simd_sum / simd_max) for fast parallel sums/max
//! - Threadgroup memory tiling for input vector reuse across output rows
//! - 32-wide SIMD groups (Apple GPU architecture)
//! - Fused operations where profitable
//!
//! Compatibility note: `thread_index_in_simdgroup` and `simdgroup_index_in_threadgroup`
//! are passed as kernel function arguments with [[attribute]] syntax for broad
//! Metal version compatibility.

#[cfg(target_os = "macos")]
pub const METAL_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// FOR_UNROLL: Hint the compiler to fully unroll the following loop.
// Matches llama.cpp convention for simdgroup MMA inner loops.
#define FOR_UNROLL _Pragma("clang loop unroll(full)") for

// ============================================================================
// matmul_f32: Matrix-vector multiply (the hot kernel, ~90% of compute)
//
// W: [out_dim, in_dim] row-major weights
// x: [in_dim] input vector
// out: [out_dim] output vector
//
// Strategy: One threadgroup per output row. Each thread in the threadgroup
// processes a chunk of the dot product, then we reduce via simd_sum.
// ============================================================================

kernel void matmul_f32(
    device const float* W      [[buffer(0)]],
    device const float* x      [[buffer(1)]],
    device float*       out    [[buffer(2)]],
    constant uint&      in_dim [[buffer(3)]],
    uint row                   [[threadgroup_position_in_grid]],
    uint tid                   [[thread_index_in_threadgroup]],
    uint tg_size               [[threads_per_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_group            [[simdgroup_index_in_threadgroup]])
{
    // Each thread accumulates its portion of the dot product
    device const float* w_row = W + row * in_dim;
    float sum = 0.0f;

    // Process elements in strides of threadgroup size
    for (uint j = tid; j < in_dim; j += tg_size) {
        sum += w_row[j] * x[j];
    }

    // SIMD group reduction (hardware-accelerated on Apple GPU)
    sum = simd_sum(sum);

    // Only the first thread in each SIMD group has the partial sum.
    // We need to reduce across SIMD groups within the threadgroup.
    threadgroup float partial_sums[32]; // max 1024 threads / 32 per simd group

    if (simd_lane == 0) {
        partial_sums[simd_group] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction by the first SIMD group
    uint num_simd_groups = (tg_size + 31) / 32;
    if (simd_group == 0) {
        float val = (simd_lane < num_simd_groups) ? partial_sums[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            out[row] = val;
        }
    }
}

// ============================================================================
// matmul_bytes_f32: Matrix-vector multiply reading weights from raw LE bytes.
//
// Same algorithm as matmul_f32, but w_bytes is a byte buffer containing
// LE-encoded f32 values. On Apple Silicon (little-endian), we can just cast.
// ============================================================================

kernel void matmul_bytes_f32(
    device const uchar* w_bytes [[buffer(0)]],
    device const float* x       [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant uint&      in_dim  [[buffer(3)]],
    uint row                    [[threadgroup_position_in_grid]],
    uint tid                    [[thread_index_in_threadgroup]],
    uint tg_size                [[threads_per_threadgroup]],
    uint simd_lane              [[thread_index_in_simdgroup]],
    uint simd_group             [[simdgroup_index_in_threadgroup]])
{
    // Cast the byte pointer to float pointer for this row
    // Apple Silicon is little-endian, so LE f32 bytes == native f32.
    device const float* w_row = (device const float*)(w_bytes + row * in_dim * 4);

    float sum = 0.0f;
    for (uint j = tid; j < in_dim; j += tg_size) {
        sum += w_row[j] * x[j];
    }

    sum = simd_sum(sum);

    threadgroup float partial_sums[32];

    if (simd_lane == 0) {
        partial_sums[simd_group] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simd_groups = (tg_size + 31) / 32;
    if (simd_group == 0) {
        float val = (simd_lane < num_simd_groups) ? partial_sums[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            out[row] = val;
        }
    }
}

// ============================================================================
// matmul_f32_deferred: Deferred-reduction F32 matvec (4 rows/TG, 128 threads)
//
// Same deferred-reduction pattern as dequant_matmul_q8_0_deferred but for F32
// weights — no dequantization needed. Each thread accumulates locally across
// a stride of the input dimension, then ONE simd_sum + ONE cross-SG shmem
// reduce = 2 sync points total.
//
// NR0=4 rows per threadgroup (4 SIMD groups x 32 threads = 128 threads).
// Each SG handles one output row and strides across in_dim with step=32.
// x-vector is reused across all 4 rows (4x bandwidth reduction vs 1 row/TG).
//
// Dispatch: threadgroups = ceil(out_dim/4), threads_per_threadgroup = 128
// Threadgroup memory: NR0 * 32 * sizeof(float) = 512 bytes
// ============================================================================

kernel void matmul_f32_deferred(
    device const float* W       [[buffer(0)]],
    device const float* x       [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant uint&      in_dim  [[buffer(3)]],
    constant uint&      out_dim [[buffer(4)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint NR0 = 4;   // rows per threadgroup
    const uint NSG = 4;   // simdgroups per threadgroup
    const uint NW  = 32;  // threads per simdgroup

    const uint r0 = tgpig * NR0;  // first output row for this threadgroup

    // Global thread index within threadgroup: sgitg * 32 + tiisg
    const uint tid = sgitg * NW + tiisg;
    const uint total_threads = NSG * NW;  // 128

    // Each thread accumulates a partial dot product for all NR0 rows
    float sumf[NR0] = { 0.f, 0.f, 0.f, 0.f };

    // Stride across in_dim: thread tid processes elements tid, tid+128, tid+256, ...
    for (uint j = tid; j < in_dim; j += total_threads) {
        float xv = x[j];
        for (uint row = 0; row < NR0; ++row) {
            if (r0 + row < out_dim) {
                sumf[row] += W[(r0 + row) * in_dim + j] * xv;
            }
        }
    }

    // Final reduction: 1 simd_sum + cross-SG shmem reduce
    threadgroup float shmem[NR0 * NW];  // 512 bytes

    // Initialize shmem (SG 0 zeros its slots)
    for (uint row = 0; row < NR0; ++row) {
        if (sgitg == 0) {
            shmem[row * NW + tiisg] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each SG writes its reduced sum
    for (uint row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem[row * NW + sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // SG 0 does the final reduction and writes output
    for (uint row = 0; row < NR0 && r0 + row < out_dim; ++row) {
        float tot = simd_sum(shmem[row * NW + tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            out[r0 + row] = tot;
        }
    }
}
// ============================================================================
// F16 (half-precision) decode matvec kernels.
//
// IEEE 754 half-precision weights: each weight is 2 bytes, no block structure,
// no scale factors. Simpler than Q8_0 — just raw half values.
// row_bytes = in_dim * 2 (sizeof(half)).
//
// All variants use the NR0=2 deferred-reduction pattern (4 SGs, 128 threads,
// 2 rows per TG) matching the Q8_0 NR2 dispatch layout.
// Weights read as half, accumulated in float for precision.
// ============================================================================

// matmul_f16_deferred_nr2: Basic F16 matvec, NR0=2 deferred reduction.
// out[row] = dot(W_f16_row, x)
// Dispatch: threadgroups = ceil(out_dim/2), threads_per_threadgroup = 128
kernel void matmul_f16_deferred_nr2(
    device const half*  weights [[buffer(0)]],
    device const float* x       [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant uint&      in_dim  [[buffer(3)]],
    constant uint&      out_dim [[buffer(4)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint NR0 = 2;
    const uint NSG = 4;
    const uint NW = 32;

    const uint r0 = tgpig * NR0;

    // Global thread index for striding across in_dim
    const uint tid = sgitg * NW + tiisg;
    const uint total_threads = NSG * NW;  // 128

    float sumf[NR0] = { 0.f, 0.f };

    for (uint j = tid; j < in_dim; j += total_threads) {
        float xv = x[j];
        for (uint row = 0; row < NR0; ++row) {
            if (r0 + row < out_dim) {
                sumf[row] += float(weights[(r0 + row) * in_dim + j]) * xv;
            }
        }
    }

    // Final reduction: simd_sum + cross-SG shmem reduce
    threadgroup float shmem[NR0 * NW];

    for (uint row = 0; row < NR0; ++row) {
        if (sgitg == 0) {
            shmem[row * NW + tiisg] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem[row * NW + sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0 && r0 + row < out_dim; ++row) {
        float tot = simd_sum(shmem[row * NW + tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            out[r0 + row] = tot;
        }
    }
}

// matmul_f16_deferred_residual_nr2: F16 matvec + residual add.
// out[row] = dot(W_f16_row, x) + residual[row]
// Dispatch: threadgroups = ceil(out_dim/2), threads_per_threadgroup = 128
kernel void matmul_f16_deferred_residual_nr2(
    device const half*  weights   [[buffer(0)]],
    device const float* x         [[buffer(1)]],
    device float*       out       [[buffer(2)]],
    constant uint&      in_dim    [[buffer(3)]],
    device const float* residual  [[buffer(4)]],
    constant uint&      out_dim   [[buffer(5)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint NR0 = 2;
    const uint NSG = 4;
    const uint NW = 32;

    const uint r0 = tgpig * NR0;
    const uint tid = sgitg * NW + tiisg;
    const uint total_threads = NSG * NW;

    float sumf[NR0] = { 0.f, 0.f };

    for (uint j = tid; j < in_dim; j += total_threads) {
        float xv = x[j];
        for (uint row = 0; row < NR0; ++row) {
            if (r0 + row < out_dim) {
                sumf[row] += float(weights[(r0 + row) * in_dim + j]) * xv;
            }
        }
    }

    threadgroup float shmem[NR0 * NW];

    for (uint row = 0; row < NR0; ++row) {
        if (sgitg == 0) {
            shmem[row * NW + tiisg] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem[row * NW + sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0 && r0 + row < out_dim; ++row) {
        float tot = simd_sum(shmem[row * NW + tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            out[r0 + row] = tot + residual[r0 + row];
        }
    }
}

// matmul_f16_deferred_bias_nr2: F16 matvec + fused QKV bias.
// Buffer layout matches dequant_matmul_q8_0_deferred_bias_nr2.
// Dispatch: threadgroups = ceil(out_dim/2), threads_per_threadgroup = 128
kernel void matmul_f16_deferred_bias_nr2(
    device const half*  weights [[buffer(0)]],
    device const float* x       [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant uint&      in_dim  [[buffer(3)]],
    constant uint&      out_dim [[buffer(4)]],
    device const float* bias_q  [[buffer(5)]],
    device const float* bias_k  [[buffer(6)]],
    device const float* bias_v  [[buffer(7)]],
    constant uint&      q_dim   [[buffer(8)]],
    constant uint&      qk_dim  [[buffer(9)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint NR0 = 2;
    const uint NSG = 4;
    const uint NW = 32;

    const uint r0 = tgpig * NR0;
    const uint tid = sgitg * NW + tiisg;
    const uint total_threads = NSG * NW;

    float sumf[NR0] = { 0.f, 0.f };

    for (uint j = tid; j < in_dim; j += total_threads) {
        float xv = x[j];
        for (uint row = 0; row < NR0; ++row) {
            if (r0 + row < out_dim) {
                sumf[row] += float(weights[(r0 + row) * in_dim + j]) * xv;
            }
        }
    }

    threadgroup float shmem[NR0 * NW];

    for (uint row = 0; row < NR0; ++row) {
        if (sgitg == 0) {
            shmem[row * NW + tiisg] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem[row * NW + sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0 && r0 + row < out_dim; ++row) {
        float tot = simd_sum(shmem[row * NW + tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            uint r = r0 + row;
            float b;
            if (r < q_dim) b = bias_q[r];
            else if (r < qk_dim) b = bias_k[r - q_dim];
            else b = bias_v[r - qk_dim];
            out[r] = tot + b;
        }
    }
}

// rmsnorm_matmul_f16_deferred_nr2: Fused RMSNorm + F16 matvec NR2.
// Single-pass: computes dot(W, x * norm_w) and sum(x^2) simultaneously,
// then applies scale * dot at the end. Saves 1 RMSNorm dispatch + 1 barrier.
// Dispatch: threadgroups = ceil(out_dim/2), threads_per_threadgroup = 128
kernel void rmsnorm_matmul_f16_deferred_nr2(
    device const half*  weights   [[buffer(0)]],
    device const float* x         [[buffer(1)]],
    device float*       out       [[buffer(2)]],
    constant uint&      in_dim    [[buffer(3)]],
    constant uint&      out_dim   [[buffer(4)]],
    device const uchar* norm_w    [[buffer(5)]],
    constant float&     eps       [[buffer(6)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint NR0 = 2;
    const uint NSG = 4;
    const uint NW = 32;

    device const float* norm_weight = (device const float*)norm_w;

    const uint r0 = tgpig * NR0;
    const uint tid = sgitg * NW + tiisg;
    const uint total_threads = NSG * NW;

    float sumf[NR0] = { 0.f, 0.f };
    float ss = 0.0f;  // sum of squares for RMSNorm

    for (uint j = tid; j < in_dim; j += total_threads) {
        float xi = x[j];
        ss += xi * xi;
        float normed_xj = xi * norm_weight[j];
        for (uint row = 0; row < NR0; ++row) {
            if (r0 + row < out_dim) {
                sumf[row] += float(weights[(r0 + row) * in_dim + j]) * normed_xj;
            }
        }
    }

    // Reduce sum-of-squares across all 128 threads
    ss = simd_sum(ss);

    threadgroup float shmem[NR0 * NW];

    if (tiisg == 0) {
        shmem[sgitg] = ss;
    }

    for (uint row = 0; row < NR0; ++row) {
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute RMSNorm scale from reduced sum-of-squares
    threadgroup float rms_scale_shared;
    if (sgitg == 0) {
        float total_ss = (tiisg < NSG) ? shmem[tiisg] : 0.0f;
        total_ss = simd_sum(total_ss);
        if (tiisg == 0) {
            rms_scale_shared = rsqrt(total_ss / float(in_dim) + eps);
        }
    }

    // Write dot product partial sums to shmem (reuse space)
    for (uint row = 0; row < NR0; ++row) {
        if (sgitg == 0) {
            shmem[row * NW + tiisg] = 0.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem[row * NW + sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms_scale = rms_scale_shared;

    for (uint row = 0; row < NR0 && r0 + row < out_dim; ++row) {
        float tot = simd_sum(shmem[row * NW + tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            out[r0 + row] = tot * rms_scale;
        }
    }
}

// rmsnorm_matmul_f16_deferred_residual_nr2: Fused RMSNorm + F16 matvec + residual.
// Single-pass: computes dot(W, x * norm_w) and sum(x^2) simultaneously,
// then applies scale * dot + residual at the end.
// Dispatch: threadgroups = ceil(out_dim/2), threads_per_threadgroup = 128
kernel void rmsnorm_matmul_f16_deferred_residual_nr2(
    device const half*  weights   [[buffer(0)]],
    device const float* x         [[buffer(1)]],
    device float*       out       [[buffer(2)]],
    constant uint&      in_dim    [[buffer(3)]],
    device const float* residual  [[buffer(4)]],
    constant uint&      out_dim   [[buffer(5)]],
    device const uchar* norm_w    [[buffer(6)]],
    constant float&     eps       [[buffer(7)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint NR0 = 2;
    const uint NSG = 4;
    const uint NW = 32;

    device const float* norm_weight = (device const float*)norm_w;

    const uint r0 = tgpig * NR0;
    const uint tid = sgitg * NW + tiisg;
    const uint total_threads = NSG * NW;

    float sumf[NR0] = { 0.f, 0.f };
    float ss = 0.0f;

    for (uint j = tid; j < in_dim; j += total_threads) {
        float xi = x[j];
        ss += xi * xi;
        float normed_xj = xi * norm_weight[j];
        for (uint row = 0; row < NR0; ++row) {
            if (r0 + row < out_dim) {
                sumf[row] += float(weights[(r0 + row) * in_dim + j]) * normed_xj;
            }
        }
    }

    ss = simd_sum(ss);

    threadgroup float shmem[NR0 * NW];

    if (tiisg == 0) {
        shmem[sgitg] = ss;
    }

    for (uint row = 0; row < NR0; ++row) {
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float rms_scale_shared;
    if (sgitg == 0) {
        float total_ss = (tiisg < NSG) ? shmem[tiisg] : 0.0f;
        total_ss = simd_sum(total_ss);
        if (tiisg == 0) {
            rms_scale_shared = rsqrt(total_ss / float(in_dim) + eps);
        }
    }

    for (uint row = 0; row < NR0; ++row) {
        if (sgitg == 0) {
            shmem[row * NW + tiisg] = 0.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem[row * NW + sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms_scale = rms_scale_shared;

    for (uint row = 0; row < NR0 && r0 + row < out_dim; ++row) {
        float tot = simd_sum(shmem[row * NW + tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            out[r0 + row] = tot * rms_scale + residual[r0 + row];
        }
    }
}


// ============================================================================
// rmsnorm: RMS Normalization
//
// out[i] = x[i] * weight[i] / sqrt(mean(x^2) + eps)
//
// Two-pass: first compute sum of squares, then normalize.
// Uses simd_sum for fast reduction.
// ============================================================================

kernel void rmsnorm(
    device const float* x       [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant uint&      dim     [[buffer(3)]],
    constant float&     eps     [[buffer(4)]],
    uint tid                    [[thread_index_in_threadgroup]],
    uint tg_size                [[threads_per_threadgroup]],
    uint simd_lane              [[thread_index_in_simdgroup]],
    uint simd_group             [[simdgroup_index_in_threadgroup]])
{
    // Pass 1: compute sum of squares
    float ss = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        float v = x[i];
        ss += v * v;
    }

    ss = simd_sum(ss);

    threadgroup float partial_sums[32];

    if (simd_lane == 0) {
        partial_sums[simd_group] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce across SIMD groups
    uint num_simd_groups = (tg_size + 31) / 32;
    threadgroup float total_ss;
    if (simd_group == 0) {
        float val = (simd_lane < num_simd_groups) ? partial_sums[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            total_ss = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = rsqrt(total_ss / float(dim) + eps);

    // Pass 2: normalize and scale by weight
    for (uint i = tid; i < dim; i += tg_size) {
        out[i] = x[i] * scale * weight[i];
    }
}

// ============================================================================
// rmsnorm_bytes: RMS Normalization with byte-encoded weights.
// ============================================================================

kernel void rmsnorm_bytes(
    device const float* x          [[buffer(0)]],
    device const uchar* w_bytes    [[buffer(1)]],
    device float*       out        [[buffer(2)]],
    constant uint&      dim        [[buffer(3)]],
    constant float&     eps        [[buffer(4)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_group                [[simdgroup_index_in_threadgroup]])
{
    device const float* weight = (device const float*)w_bytes;

    float ss = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        float v = x[i];
        ss += v * v;
    }

    ss = simd_sum(ss);

    threadgroup float partial_sums[32];

    if (simd_lane == 0) {
        partial_sums[simd_group] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simd_groups = (tg_size + 31) / 32;
    threadgroup float total_ss;
    if (simd_group == 0) {
        float val = (simd_lane < num_simd_groups) ? partial_sums[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            total_ss = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = rsqrt(total_ss / float(dim) + eps);

    for (uint i = tid; i < dim; i += tg_size) {
        out[i] = x[i] * scale * weight[i];
    }
}

// ============================================================================
// dequant_matmul_q8_0: Fused Q8_0 dequantization + matrix-vector multiply
//
// w_q8:   Q8_0 weight data for [out_dim, in_dim] matrix
//         Each row: ceil(in_dim/32) blocks, each block = 2 bytes f16 scale + 32 bytes int8
// x:      [in_dim] input vector (f32)
// out:    [out_dim] output vector (f32)
// in_dim: number of elements per row (NOT byte stride)
// out_dim: total number of output rows (for bounds checking in multi-row dispatch)
//
// Strategy: One threadgroup per output row (same dispatch pattern as matmul_f32).
// Each thread processes multiple Q8_0 blocks in a strided pattern, accumulating
// the dot product with on-the-fly dequantization. Reduced via simd_sum.
//
// Q8_0 block layout (34 bytes):
//   [f16 scale (2 bytes)] [32 x int8 quantized values (32 bytes)]
//   dequantized value = scale * (float)int8_val
// ============================================================================

kernel void dequant_matmul_q8_0(
    device const uchar* w_q8    [[buffer(0)]],
    device const float* x       [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant uint&      in_dim  [[buffer(3)]],
    uint row                    [[threadgroup_position_in_grid]],
    uint lane                   [[thread_index_in_simdgroup]])
{
    // Single SIMD group (32 threads) per output row.
    // Each lane handles element [lane] of every Q8_0 block — perfect 1:1 mapping
    // since Q8_0 has exactly 32 elements per block = Apple SIMD width.
    // No threadgroup memory or barriers needed.
    const uint Q8_BLOCK_SIZE = 34;  // 2 bytes f16 scale + 32 bytes int8 data

    uint num_blocks = in_dim >> 5;  // in_dim / 32
    uint row_bytes = num_blocks * Q8_BLOCK_SIZE;
    device const uchar* row_ptr = w_q8 + row * row_bytes;

    float sum = 0.0f;

    // 4x unrolled: process 4 Q8_0 blocks per iteration for ILP
    uint b = 0;
    for (; b + 3 < num_blocks; b += 4) {
        device const uchar* bp0 = row_ptr + b * Q8_BLOCK_SIZE;
        device const uchar* bp1 = bp0 + Q8_BLOCK_SIZE;
        device const uchar* bp2 = bp1 + Q8_BLOCK_SIZE;
        device const uchar* bp3 = bp2 + Q8_BLOCK_SIZE;
        half s0 = as_type<half>(*(device const ushort*)bp0);
        half s1 = as_type<half>(*(device const ushort*)bp1);
        half s2 = as_type<half>(*(device const ushort*)bp2);
        half s3 = as_type<half>(*(device const ushort*)bp3);
        float v0 = float(((device const char*)(bp0 + 2))[lane]) * x[(b << 5) + lane];
        float v1 = float(((device const char*)(bp1 + 2))[lane]) * x[((b+1) << 5) + lane];
        float v2 = float(((device const char*)(bp2 + 2))[lane]) * x[((b+2) << 5) + lane];
        float v3 = float(((device const char*)(bp3 + 2))[lane]) * x[((b+3) << 5) + lane];
        sum += float(s0) * simd_sum(v0) + float(s1) * simd_sum(v1)
             + float(s2) * simd_sum(v2) + float(s3) * simd_sum(v3);
    }
    // Handle remaining blocks
    for (; b < num_blocks; b++) {
        device const uchar* bp = row_ptr + b * Q8_BLOCK_SIZE;
        half scale = as_type<half>(*(device const ushort*)bp);
        float val = float(((device const char*)(bp + 2))[lane]) * x[(b << 5) + lane];
        sum += float(scale) * simd_sum(val);
    }

    if (lane == 0) {
        out[row] = sum;
    }
}

// ============================================================================
// dequant_matmul_q8_0_residual: Fused Q8_0 matmul + residual add
// out[row] = dot(w_q8_row, x) + residual[row]
// Eliminates separate add_residual dispatch.
// ============================================================================

kernel void dequant_matmul_q8_0_residual(
    device const uchar* w_q8      [[buffer(0)]],
    device const float* x         [[buffer(1)]],
    device float*       out       [[buffer(2)]],
    constant uint&      in_dim    [[buffer(3)]],
    device const float* residual  [[buffer(4)]],
    uint row                      [[threadgroup_position_in_grid]],
    uint lane                     [[thread_index_in_simdgroup]])
{
    const uint Q8_BLOCK_SIZE = 34;
    uint num_blocks = in_dim >> 5;
    device const uchar* row_ptr = w_q8 + row * num_blocks * Q8_BLOCK_SIZE;

    float sum = 0.0f;
    uint b = 0;
    for (; b + 3 < num_blocks; b += 4) {
        device const uchar* bp0 = row_ptr + b * Q8_BLOCK_SIZE;
        device const uchar* bp1 = bp0 + Q8_BLOCK_SIZE;
        device const uchar* bp2 = bp1 + Q8_BLOCK_SIZE;
        device const uchar* bp3 = bp2 + Q8_BLOCK_SIZE;
        half s0 = as_type<half>(*(device const ushort*)bp0);
        half s1 = as_type<half>(*(device const ushort*)bp1);
        half s2 = as_type<half>(*(device const ushort*)bp2);
        half s3 = as_type<half>(*(device const ushort*)bp3);
        float v0 = float(((device const char*)(bp0 + 2))[lane]) * x[(b << 5) + lane];
        float v1 = float(((device const char*)(bp1 + 2))[lane]) * x[((b+1) << 5) + lane];
        float v2 = float(((device const char*)(bp2 + 2))[lane]) * x[((b+2) << 5) + lane];
        float v3 = float(((device const char*)(bp3 + 2))[lane]) * x[((b+3) << 5) + lane];
        sum += float(s0) * simd_sum(v0) + float(s1) * simd_sum(v1)
             + float(s2) * simd_sum(v2) + float(s3) * simd_sum(v3);
    }
    for (; b < num_blocks; b++) {
        device const uchar* bp = row_ptr + b * Q8_BLOCK_SIZE;
        half scale = as_type<half>(*(device const ushort*)bp);
        float val = float(((device const char*)(bp + 2))[lane]) * x[(b << 5) + lane];
        sum += float(scale) * simd_sum(val);
    }

    if (lane == 0) {
        out[row] = sum + residual[row];
    }
}

// ============================================================================
// dequant_matmul_q8_0_multirow: Multi-row Q8_0 matmul (2 rows per threadgroup)
//
// Same computation as dequant_matmul_q8_0 but processes 2 output rows per
// threadgroup using 2 SIMD groups (64 threads). Both SIMD groups load the
// SAME x-vector blocks from device memory, halving x-bandwidth compared to
// dispatching separate threadgroups per row.
//
// Dispatch: threadgroups = ceil(out_dim/2), threads_per_threadgroup = 64
// SIMD group 0 -> row 2*gid, SIMD group 1 -> row 2*gid+1
// ============================================================================

kernel void dequant_matmul_q8_0_multirow(
    device const uchar* w_q8    [[buffer(0)]],
    device const float* x       [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant uint&      in_dim  [[buffer(3)]],
    constant uint&      out_dim [[buffer(4)]],
    uint row_pair               [[threadgroup_position_in_grid]],
    uint lane                   [[thread_index_in_simdgroup]],
    uint sg                     [[simdgroup_index_in_threadgroup]])
{
    // Each SIMD group handles one row: sg0 -> row 2*row_pair, sg1 -> row 2*row_pair+1
    uint row = row_pair * 2 + sg;
    if (row >= out_dim) return;

    const uint Q8_BLOCK_SIZE = 34;
    uint num_blocks = in_dim >> 5;
    uint row_bytes = num_blocks * Q8_BLOCK_SIZE;
    device const uchar* row_ptr = w_q8 + row * row_bytes;

    float sum = 0.0f;

    // 4x unrolled: process 4 Q8_0 blocks per iteration for ILP
    uint b = 0;
    for (; b + 3 < num_blocks; b += 4) {
        device const uchar* bp0 = row_ptr + b * Q8_BLOCK_SIZE;
        device const uchar* bp1 = bp0 + Q8_BLOCK_SIZE;
        device const uchar* bp2 = bp1 + Q8_BLOCK_SIZE;
        device const uchar* bp3 = bp2 + Q8_BLOCK_SIZE;
        half s0 = as_type<half>(*(device const ushort*)bp0);
        half s1 = as_type<half>(*(device const ushort*)bp1);
        half s2 = as_type<half>(*(device const ushort*)bp2);
        half s3 = as_type<half>(*(device const ushort*)bp3);
        float v0 = float(((device const char*)(bp0 + 2))[lane]) * x[(b << 5) + lane];
        float v1 = float(((device const char*)(bp1 + 2))[lane]) * x[((b+1) << 5) + lane];
        float v2 = float(((device const char*)(bp2 + 2))[lane]) * x[((b+2) << 5) + lane];
        float v3 = float(((device const char*)(bp3 + 2))[lane]) * x[((b+3) << 5) + lane];
        sum += float(s0) * simd_sum(v0) + float(s1) * simd_sum(v1)
             + float(s2) * simd_sum(v2) + float(s3) * simd_sum(v3);
    }
    // Handle remaining blocks
    for (; b < num_blocks; b++) {
        device const uchar* bp = row_ptr + b * Q8_BLOCK_SIZE;
        half scale = as_type<half>(*(device const ushort*)bp);
        float val = float(((device const char*)(bp + 2))[lane]) * x[(b << 5) + lane];
        sum += float(scale) * simd_sum(val);
    }

    if (lane == 0) {
        out[row] = sum;
    }
}

// ============================================================================
// dequant_matmul_q8_0_residual_multirow: Multi-row Q8_0 matmul + residual add
// out[row] = dot(w_q8_row, x) + residual[row]
// Same multi-row strategy: 2 rows per threadgroup, 2 SIMD groups.
// ============================================================================

kernel void dequant_matmul_q8_0_residual_multirow(
    device const uchar* w_q8      [[buffer(0)]],
    device const float* x         [[buffer(1)]],
    device float*       out       [[buffer(2)]],
    constant uint&      in_dim    [[buffer(3)]],
    device const float* residual  [[buffer(4)]],
    constant uint&      out_dim   [[buffer(5)]],
    uint row_pair                 [[threadgroup_position_in_grid]],
    uint lane                     [[thread_index_in_simdgroup]],
    uint sg                       [[simdgroup_index_in_threadgroup]])
{
    uint row = row_pair * 2 + sg;
    if (row >= out_dim) return;

    const uint Q8_BLOCK_SIZE = 34;
    uint num_blocks = in_dim >> 5;
    device const uchar* row_ptr = w_q8 + row * num_blocks * Q8_BLOCK_SIZE;

    float sum = 0.0f;
    uint b = 0;
    for (; b + 3 < num_blocks; b += 4) {
        device const uchar* bp0 = row_ptr + b * Q8_BLOCK_SIZE;
        device const uchar* bp1 = bp0 + Q8_BLOCK_SIZE;
        device const uchar* bp2 = bp1 + Q8_BLOCK_SIZE;
        device const uchar* bp3 = bp2 + Q8_BLOCK_SIZE;
        half s0 = as_type<half>(*(device const ushort*)bp0);
        half s1 = as_type<half>(*(device const ushort*)bp1);
        half s2 = as_type<half>(*(device const ushort*)bp2);
        half s3 = as_type<half>(*(device const ushort*)bp3);
        float v0 = float(((device const char*)(bp0 + 2))[lane]) * x[(b << 5) + lane];
        float v1 = float(((device const char*)(bp1 + 2))[lane]) * x[((b+1) << 5) + lane];
        float v2 = float(((device const char*)(bp2 + 2))[lane]) * x[((b+2) << 5) + lane];
        float v3 = float(((device const char*)(bp3 + 2))[lane]) * x[((b+3) << 5) + lane];
        sum += float(s0) * simd_sum(v0) + float(s1) * simd_sum(v1)
             + float(s2) * simd_sum(v2) + float(s3) * simd_sum(v3);
    }
    for (; b < num_blocks; b++) {
        device const uchar* bp = row_ptr + b * Q8_BLOCK_SIZE;
        half scale = as_type<half>(*(device const ushort*)bp);
        float val = float(((device const char*)(bp + 2))[lane]) * x[(b << 5) + lane];
        sum += float(scale) * simd_sum(val);
    }

    if (lane == 0) {
        out[row] = sum + residual[row];
    }
}

// ============================================================================
// dequant_matmul_q8_0_4row: 4-row Q8_0 matmul (4 rows per threadgroup)
//
// Same computation as dequant_matmul_q8_0 but processes 4 output rows per
// threadgroup using 4 SIMD groups (128 threads). All SIMD groups load the
// SAME x-vector blocks, quartering x-bandwidth vs single-row dispatch.
//
// Dispatch: threadgroups = ceil(out_dim/4), threads_per_threadgroup = 128
// SIMD group 0..3 -> rows 4*gid+0 .. 4*gid+3
// ============================================================================

kernel void dequant_matmul_q8_0_4row(
    device const uchar* w_q8    [[buffer(0)]],
    device const float* x       [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant uint&      in_dim  [[buffer(3)]],
    constant uint&      out_dim [[buffer(4)]],
    uint row_group              [[threadgroup_position_in_grid]],
    uint lane                   [[thread_index_in_simdgroup]],
    uint sg                     [[simdgroup_index_in_threadgroup]])
{
    uint row = row_group * 4 + sg;
    if (row >= out_dim) return;

    const uint Q8_BLOCK_SIZE = 34;
    uint num_blocks = in_dim >> 5;
    uint row_bytes = num_blocks * Q8_BLOCK_SIZE;
    device const uchar* row_ptr = w_q8 + row * row_bytes;

    float sum = 0.0f;

    uint b = 0;
    for (; b + 3 < num_blocks; b += 4) {
        device const uchar* bp0 = row_ptr + b * Q8_BLOCK_SIZE;
        device const uchar* bp1 = bp0 + Q8_BLOCK_SIZE;
        device const uchar* bp2 = bp1 + Q8_BLOCK_SIZE;
        device const uchar* bp3 = bp2 + Q8_BLOCK_SIZE;
        half s0 = as_type<half>(*(device const ushort*)bp0);
        half s1 = as_type<half>(*(device const ushort*)bp1);
        half s2 = as_type<half>(*(device const ushort*)bp2);
        half s3 = as_type<half>(*(device const ushort*)bp3);
        float v0 = float(((device const char*)(bp0 + 2))[lane]) * x[(b << 5) + lane];
        float v1 = float(((device const char*)(bp1 + 2))[lane]) * x[((b+1) << 5) + lane];
        float v2 = float(((device const char*)(bp2 + 2))[lane]) * x[((b+2) << 5) + lane];
        float v3 = float(((device const char*)(bp3 + 2))[lane]) * x[((b+3) << 5) + lane];
        sum += float(s0) * simd_sum(v0) + float(s1) * simd_sum(v1)
             + float(s2) * simd_sum(v2) + float(s3) * simd_sum(v3);
    }
    for (; b < num_blocks; b++) {
        device const uchar* bp = row_ptr + b * Q8_BLOCK_SIZE;
        half scale = as_type<half>(*(device const ushort*)bp);
        float val = float(((device const char*)(bp + 2))[lane]) * x[(b << 5) + lane];
        sum += float(scale) * simd_sum(val);
    }

    if (lane == 0) {
        out[row] = sum;
    }
}

// ============================================================================
// dequant_matmul_q8_0_residual_4row: 4-row Q8_0 matmul + residual add
// out[row] = dot(w_q8_row, x) + residual[row]
// Same 4-row strategy: 4 rows per threadgroup, 4 SIMD groups (128 threads).
// ============================================================================

kernel void dequant_matmul_q8_0_residual_4row(
    device const uchar* w_q8      [[buffer(0)]],
    device const float* x         [[buffer(1)]],
    device float*       out       [[buffer(2)]],
    constant uint&      in_dim    [[buffer(3)]],
    device const float* residual  [[buffer(4)]],
    constant uint&      out_dim   [[buffer(5)]],
    uint row_group                [[threadgroup_position_in_grid]],
    uint lane                     [[thread_index_in_simdgroup]],
    uint sg                       [[simdgroup_index_in_threadgroup]])
{
    uint row = row_group * 4 + sg;
    if (row >= out_dim) return;

    const uint Q8_BLOCK_SIZE = 34;
    uint num_blocks = in_dim >> 5;
    device const uchar* row_ptr = w_q8 + row * num_blocks * Q8_BLOCK_SIZE;

    float sum = 0.0f;
    uint b = 0;
    for (; b + 3 < num_blocks; b += 4) {
        device const uchar* bp0 = row_ptr + b * Q8_BLOCK_SIZE;
        device const uchar* bp1 = bp0 + Q8_BLOCK_SIZE;
        device const uchar* bp2 = bp1 + Q8_BLOCK_SIZE;
        device const uchar* bp3 = bp2 + Q8_BLOCK_SIZE;
        half s0 = as_type<half>(*(device const ushort*)bp0);
        half s1 = as_type<half>(*(device const ushort*)bp1);
        half s2 = as_type<half>(*(device const ushort*)bp2);
        half s3 = as_type<half>(*(device const ushort*)bp3);
        float v0 = float(((device const char*)(bp0 + 2))[lane]) * x[(b << 5) + lane];
        float v1 = float(((device const char*)(bp1 + 2))[lane]) * x[((b+1) << 5) + lane];
        float v2 = float(((device const char*)(bp2 + 2))[lane]) * x[((b+2) << 5) + lane];
        float v3 = float(((device const char*)(bp3 + 2))[lane]) * x[((b+3) << 5) + lane];
        sum += float(s0) * simd_sum(v0) + float(s1) * simd_sum(v1)
             + float(s2) * simd_sum(v2) + float(s3) * simd_sum(v3);
    }
    for (; b < num_blocks; b++) {
        device const uchar* bp = row_ptr + b * Q8_BLOCK_SIZE;
        half scale = as_type<half>(*(device const ushort*)bp);
        float val = float(((device const char*)(bp + 2))[lane]) * x[(b << 5) + lane];
        sum += float(scale) * simd_sum(val);
    }

    if (lane == 0) {
        out[row] = sum + residual[row];
    }
}

// ============================================================================
// dequant_matmul_q8_0_deferred: Deferred-reduction Q8_0 matvec (llama.cpp pattern)
//
// Key insight: Instead of calling simd_sum() once per Q8_0 block (64 syncs for
// hidden_dim=2048), each thread processes NQ=8 elements from the SAME block
// and accumulates locally. Final reduction: 1 simd_sum + 1 shmem cross-SG
// reduce = 2 sync points total (32x fewer than the 4-row kernel).
//
// Thread mapping (NQ=8, 32 threads per SIMD group):
//   ix = tiisg / 4  -> 0..7 (which of 8 blocks in the stride)
//   il = tiisg % 4  -> 0..3 (which sub-chunk of 8 within the 32-element block)
//   4 threads collectively process one 32-element block (4 x 8 = 32)
//
// NR0=4 rows per threadgroup (4 SIMD groups x 32 threads = 128 threads).
// Stride = NSG * NQ = 4 * 8 = 32 blocks per outer iteration.
//
// Dispatch: threadgroups = ceil(out_dim/4), threads_per_threadgroup = 128
// Threadgroup memory: NR0 * 32 * sizeof(float) = 512 bytes (for cross-SG reduce)
// ============================================================================

kernel void dequant_matmul_q8_0_deferred(
    device const uchar* weights [[buffer(0)]],
    device const float* x       [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant uint&      in_dim  [[buffer(3)]],
    constant uint&      out_dim [[buffer(4)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint NR0 = 4;
    const uint NSG = 4;
    const uint NQ = 8;
    const uint NW = 32;
    const uint Q8_BLOCK_SIZE = 34; // 2 bytes scale (f16) + 32 bytes data (int8)

    const uint nb = in_dim >> 5;  // number of Q8_0 blocks per row = in_dim / 32
    const uint row_bytes = nb * Q8_BLOCK_SIZE;

    const uint r0 = tgpig * NR0;  // first output row for this threadgroup

    // Thread mapping within SIMD group
    const uint ix = tiisg / (NW / NQ);  // = tiisg / 4 -> 0..7 (block index in stride)
    const uint il = tiisg % (NW / NQ);  // = tiisg % 4 -> 0..3 (sub-chunk index)

    const uint ib0 = sgitg * NQ + ix;   // starting block for this thread

    // Pointers to weight rows
    device const uchar* ax[NR0];
    for (uint row = 0; row < NR0; ++row) {
        ax[row] = weights + (r0 + row) * row_bytes;
    }

    float sumf[NR0] = { 0.f, 0.f, 0.f, 0.f };

    float yl[NQ];

    device const float* yb = x + ib0 * 32 + il * NQ;

    // Main loop: each thread processes NQ=8 elements per block, accumulates locally
    for (uint ib = ib0; ib < nb; ib += NSG * NQ) {
        // Load 8 x-values into registers
        #pragma clang loop unroll(full)
        for (uint i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        // Process all NR0 rows with these x-values
        for (uint row = 0; row < NR0; ++row) {
            if (r0 + row >= out_dim) break;

            // Point to this block in the weight row
            device const uchar* bp = ax[row] + ib * Q8_BLOCK_SIZE;
            float scale = float(as_type<half>(*(device const ushort*)bp));
            device const char* qs = (device const char*)(bp + 2) + il * NQ;

            float sumq = 0.f;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < NQ; ++i) {
                sumq += float(qs[i]) * yl[i];
            }
            sumf[row] += sumq * scale;
        }

        yb += NSG * NQ * 32;
    }

    // Final reduction: 1 simd_sum + cross-SG shmem reduce
    threadgroup float shmem[NR0 * NW];  // NR0 * 32 floats = 512 bytes

    // Initialize shmem (SG 0 zeros its slots)
    for (uint row = 0; row < NR0; ++row) {
        if (sgitg == 0) {
            shmem[row * NW + tiisg] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each SG writes its reduced sum
    for (uint row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem[row * NW + sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // SG 0 does the final reduction and writes output
    for (uint row = 0; row < NR0 && r0 + row < out_dim; ++row) {
        float tot = simd_sum(shmem[row * NW + tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            out[r0 + row] = tot;
        }
    }
}

// ============================================================================
// dequant_matmul_q8_0_deferred_residual: Deferred-reduction Q8_0 matvec + residual
// out[row] = dot(w_q8_row, x) + residual[row]
// Same deferred-reduction pattern as above, with fused residual addition.
// ============================================================================

kernel void dequant_matmul_q8_0_deferred_residual(
    device const uchar* weights   [[buffer(0)]],
    device const float* x         [[buffer(1)]],
    device float*       out       [[buffer(2)]],
    constant uint&      in_dim    [[buffer(3)]],
    device const float* residual  [[buffer(4)]],
    constant uint&      out_dim   [[buffer(5)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint NR0 = 4;
    const uint NSG = 4;
    const uint NQ = 8;
    const uint NW = 32;
    const uint Q8_BLOCK_SIZE = 34;

    const uint nb = in_dim >> 5;
    const uint row_bytes = nb * Q8_BLOCK_SIZE;

    const uint r0 = tgpig * NR0;

    const uint ix = tiisg / (NW / NQ);
    const uint il = tiisg % (NW / NQ);

    const uint ib0 = sgitg * NQ + ix;

    device const uchar* ax[NR0];
    for (uint row = 0; row < NR0; ++row) {
        ax[row] = weights + (r0 + row) * row_bytes;
    }

    float sumf[NR0] = { 0.f, 0.f, 0.f, 0.f };

    float yl[NQ];

    device const float* yb = x + ib0 * 32 + il * NQ;

    for (uint ib = ib0; ib < nb; ib += NSG * NQ) {
        #pragma clang loop unroll(full)
        for (uint i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        for (uint row = 0; row < NR0; ++row) {
            if (r0 + row >= out_dim) break;

            device const uchar* bp = ax[row] + ib * Q8_BLOCK_SIZE;
            float scale = float(as_type<half>(*(device const ushort*)bp));
            device const char* qs = (device const char*)(bp + 2) + il * NQ;

            float sumq = 0.f;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < NQ; ++i) {
                sumq += float(qs[i]) * yl[i];
            }
            sumf[row] += sumq * scale;
        }

        yb += NSG * NQ * 32;
    }

    threadgroup float shmem[NR0 * NW];

    for (uint row = 0; row < NR0; ++row) {
        if (sgitg == 0) {
            shmem[row * NW + tiisg] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem[row * NW + sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0 && r0 + row < out_dim; ++row) {
        float tot = simd_sum(shmem[row * NW + tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            out[r0 + row] = tot + residual[r0 + row];
        }
    }
}

// ============================================================================
// dequant_matmul_q8_0_deferred_bias: Deferred-reduction Q8_0 matvec + fused bias
// out[row] = dot(w_q8_row, x) + bias[row]
//
// Same deferred-reduction pattern as dequant_matmul_q8_0_deferred, with fused
// bias addition. The bias is split across Q/K/V sections:
//   row in [0, q_dim)            -> bias_q[row]
//   row in [q_dim, q_dim+kv_dim) -> bias_k[row - q_dim]
//   row in [q_dim+kv_dim, ...)   -> bias_v[row - q_dim - kv_dim]
//
// Eliminates 3 separate bias_add dispatches per layer for Qwen2-family models.
// Dispatch: threadgroups = ceil(out_dim/4), threads_per_threadgroup = 128
// ============================================================================

kernel void dequant_matmul_q8_0_deferred_bias(
    device const uchar* weights [[buffer(0)]],
    device const float* x       [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant uint&      in_dim  [[buffer(3)]],
    constant uint&      out_dim [[buffer(4)]],
    device const float* bias_q  [[buffer(5)]],
    device const float* bias_k  [[buffer(6)]],
    device const float* bias_v  [[buffer(7)]],
    constant uint&      q_dim   [[buffer(8)]],
    constant uint&      qk_dim  [[buffer(9)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint NR0 = 4;
    const uint NSG = 4;
    const uint NQ = 8;
    const uint NW = 32;
    const uint Q8_BLOCK_SIZE = 34;

    const uint nb = in_dim >> 5;
    const uint row_bytes = nb * Q8_BLOCK_SIZE;

    const uint r0 = tgpig * NR0;

    const uint ix = tiisg / (NW / NQ);
    const uint il = tiisg % (NW / NQ);

    const uint ib0 = sgitg * NQ + ix;

    device const uchar* ax[NR0];
    for (uint row = 0; row < NR0; ++row) {
        ax[row] = weights + (r0 + row) * row_bytes;
    }

    float sumf[NR0] = { 0.f, 0.f, 0.f, 0.f };

    float yl[NQ];

    device const float* yb = x + ib0 * 32 + il * NQ;

    for (uint ib = ib0; ib < nb; ib += NSG * NQ) {
        #pragma clang loop unroll(full)
        for (uint i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        for (uint row = 0; row < NR0; ++row) {
            if (r0 + row >= out_dim) break;

            device const uchar* bp = ax[row] + ib * Q8_BLOCK_SIZE;
            float scale = float(as_type<half>(*(device const ushort*)bp));
            device const char* qs = (device const char*)(bp + 2) + il * NQ;

            float sumq = 0.f;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < NQ; ++i) {
                sumq += float(qs[i]) * yl[i];
            }
            sumf[row] += sumq * scale;
        }

        yb += NSG * NQ * 32;
    }

    threadgroup float shmem[NR0 * NW];

    for (uint row = 0; row < NR0; ++row) {
        if (sgitg == 0) {
            shmem[row * NW + tiisg] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem[row * NW + sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0 && r0 + row < out_dim; ++row) {
        float tot = simd_sum(shmem[row * NW + tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            uint r = r0 + row;
            float b;
            if (r < q_dim) b = bias_q[r];
            else if (r < qk_dim) b = bias_k[r - q_dim];
            else b = bias_v[r - qk_dim];
            out[r] = tot + b;
        }
    }
}

// ============================================================================
// NR0=2 variants of the deferred-reduction Q8_0 matvec kernels.
//
// Same algorithm as the NR0=4 variants above, but with NR0=2 (2 rows per TG).
// This doubles the threadgroup count, improving GPU occupancy for small output
// dimensions where NR0=4 produces too few TGs to saturate all GPU cores.
//
// Trade-off: less x-vector reuse per TG (2 rows share x vs 4 rows), but more
// TGs for better wave scheduling. Shmem per TG: 2*32*4 = 256 bytes (vs 512).
//
// Dispatch: threadgroups = ceil(out_dim/2), threads_per_threadgroup = 128
// ============================================================================

kernel void dequant_matmul_q8_0_deferred_nr2(
    device const uchar* weights [[buffer(0)]],
    device const float* x       [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant uint&      in_dim  [[buffer(3)]],
    constant uint&      out_dim [[buffer(4)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint NR0 = 2;
    const uint NSG = 4;
    const uint NQ = 8;
    const uint NW = 32;
    const uint Q8_BLOCK_SIZE = 34;

    const uint nb = in_dim >> 5;
    const uint row_bytes = nb * Q8_BLOCK_SIZE;

    const uint r0 = tgpig * NR0;

    const uint ix = tiisg / (NW / NQ);
    const uint il = tiisg % (NW / NQ);

    const uint ib0 = sgitg * NQ + ix;

    device const uchar* ax[NR0];
    for (uint row = 0; row < NR0; ++row) {
        ax[row] = weights + (r0 + row) * row_bytes;
    }

    float sumf[NR0] = { 0.f, 0.f };

    float yl[NQ];

    device const float* yb = x + ib0 * 32 + il * NQ;

    for (uint ib = ib0; ib < nb; ib += NSG * NQ) {
        #pragma clang loop unroll(full)
        for (uint i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        for (uint row = 0; row < NR0; ++row) {
            if (r0 + row >= out_dim) break;

            device const uchar* bp = ax[row] + ib * Q8_BLOCK_SIZE;
            float scale = float(as_type<half>(*(device const ushort*)bp));
            device const char* qs = (device const char*)(bp + 2) + il * NQ;

            float sumq = 0.f;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < NQ; ++i) {
                sumq += float(qs[i]) * yl[i];
            }
            sumf[row] += sumq * scale;
        }

        yb += NSG * NQ * 32;
    }

    threadgroup float shmem[NR0 * NW];

    for (uint row = 0; row < NR0; ++row) {
        if (sgitg == 0) {
            shmem[row * NW + tiisg] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem[row * NW + sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0 && r0 + row < out_dim; ++row) {
        float tot = simd_sum(shmem[row * NW + tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            out[r0 + row] = tot;
        }
    }
}

kernel void dequant_matmul_q8_0_deferred_residual_nr2(
    device const uchar* weights   [[buffer(0)]],
    device const float* x         [[buffer(1)]],
    device float*       out       [[buffer(2)]],
    constant uint&      in_dim    [[buffer(3)]],
    device const float* residual  [[buffer(4)]],
    constant uint&      out_dim   [[buffer(5)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint NR0 = 2;
    const uint NSG = 4;
    const uint NQ = 8;
    const uint NW = 32;
    const uint Q8_BLOCK_SIZE = 34;

    const uint nb = in_dim >> 5;
    const uint row_bytes = nb * Q8_BLOCK_SIZE;

    const uint r0 = tgpig * NR0;

    const uint ix = tiisg / (NW / NQ);
    const uint il = tiisg % (NW / NQ);

    const uint ib0 = sgitg * NQ + ix;

    device const uchar* ax[NR0];
    for (uint row = 0; row < NR0; ++row) {
        ax[row] = weights + (r0 + row) * row_bytes;
    }

    float sumf[NR0] = { 0.f, 0.f };

    float yl[NQ];

    device const float* yb = x + ib0 * 32 + il * NQ;

    for (uint ib = ib0; ib < nb; ib += NSG * NQ) {
        #pragma clang loop unroll(full)
        for (uint i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        for (uint row = 0; row < NR0; ++row) {
            if (r0 + row >= out_dim) break;

            device const uchar* bp = ax[row] + ib * Q8_BLOCK_SIZE;
            float scale = float(as_type<half>(*(device const ushort*)bp));
            device const char* qs = (device const char*)(bp + 2) + il * NQ;

            float sumq = 0.f;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < NQ; ++i) {
                sumq += float(qs[i]) * yl[i];
            }
            sumf[row] += sumq * scale;
        }

        yb += NSG * NQ * 32;
    }

    threadgroup float shmem[NR0 * NW];

    for (uint row = 0; row < NR0; ++row) {
        if (sgitg == 0) {
            shmem[row * NW + tiisg] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem[row * NW + sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0 && r0 + row < out_dim; ++row) {
        float tot = simd_sum(shmem[row * NW + tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            out[r0 + row] = tot + residual[r0 + row];
        }
    }
}

kernel void dequant_matmul_q8_0_deferred_bias_nr2(
    device const uchar* weights [[buffer(0)]],
    device const float* x       [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant uint&      in_dim  [[buffer(3)]],
    constant uint&      out_dim [[buffer(4)]],
    device const float* bias_q  [[buffer(5)]],
    device const float* bias_k  [[buffer(6)]],
    device const float* bias_v  [[buffer(7)]],
    constant uint&      q_dim   [[buffer(8)]],
    constant uint&      qk_dim  [[buffer(9)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint NR0 = 2;
    const uint NSG = 4;
    const uint NQ = 8;
    const uint NW = 32;
    const uint Q8_BLOCK_SIZE = 34;

    const uint nb = in_dim >> 5;
    const uint row_bytes = nb * Q8_BLOCK_SIZE;

    const uint r0 = tgpig * NR0;

    const uint ix = tiisg / (NW / NQ);
    const uint il = tiisg % (NW / NQ);

    const uint ib0 = sgitg * NQ + ix;

    device const uchar* ax[NR0];
    for (uint row = 0; row < NR0; ++row) {
        ax[row] = weights + (r0 + row) * row_bytes;
    }

    float sumf[NR0] = { 0.f, 0.f };

    float yl[NQ];

    device const float* yb = x + ib0 * 32 + il * NQ;

    for (uint ib = ib0; ib < nb; ib += NSG * NQ) {
        #pragma clang loop unroll(full)
        for (uint i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        for (uint row = 0; row < NR0; ++row) {
            if (r0 + row >= out_dim) break;

            device const uchar* bp = ax[row] + ib * Q8_BLOCK_SIZE;
            float scale = float(as_type<half>(*(device const ushort*)bp));
            device const char* qs = (device const char*)(bp + 2) + il * NQ;

            float sumq = 0.f;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < NQ; ++i) {
                sumq += float(qs[i]) * yl[i];
            }
            sumf[row] += sumq * scale;
        }

        yb += NSG * NQ * 32;
    }

    threadgroup float shmem[NR0 * NW];

    for (uint row = 0; row < NR0; ++row) {
        if (sgitg == 0) {
            shmem[row * NW + tiisg] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem[row * NW + sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0 && r0 + row < out_dim; ++row) {
        float tot = simd_sum(shmem[row * NW + tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            uint r = r0 + row;
            float b;
            if (r < q_dim) b = bias_q[r];
            else if (r < qk_dim) b = bias_k[r - q_dim];
            else b = bias_v[r - qk_dim];
            out[r] = tot + b;
        }
    }
}

kernel void dequant_matmul_q8_0_deferred_residual_copy_nr2(
    device const uchar* weights   [[buffer(0)]],
    device const float* x         [[buffer(1)]],
    device       float* accum     [[buffer(2)]],
    constant     uint&  in_dim    [[buffer(3)]],
    device       float* copy_dst  [[buffer(4)]],
    constant     uint&  out_dim   [[buffer(5)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint NR0 = 2;
    const uint NSG = 4;
    const uint NQ = 8;
    const uint NW = 32;
    const uint Q8_BLOCK_SIZE = 34;

    const uint nb = in_dim >> 5;
    const uint row_bytes = nb * Q8_BLOCK_SIZE;

    const uint r0 = tgpig * NR0;

    const uint ix = tiisg / (NW / NQ);
    const uint il = tiisg % (NW / NQ);

    const uint ib0 = sgitg * NQ + ix;

    device const uchar* ax[NR0];
    for (uint row = 0; row < NR0; ++row) {
        ax[row] = weights + (r0 + row) * row_bytes;
    }

    float sumf[NR0] = { 0.f, 0.f };

    float yl[NQ];

    device const float* yb = x + ib0 * 32 + il * NQ;

    for (uint ib = ib0; ib < nb; ib += NSG * NQ) {
        #pragma clang loop unroll(full)
        for (uint i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        for (uint row = 0; row < NR0; ++row) {
            if (r0 + row >= out_dim) break;

            device const uchar* bp = ax[row] + ib * Q8_BLOCK_SIZE;
            float scale = float(as_type<half>(*(device const ushort*)bp));
            device const char* qs = (device const char*)(bp + 2) + il * NQ;

            float sumq = 0.f;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < NQ; ++i) {
                sumq += float(qs[i]) * yl[i];
            }
            sumf[row] += sumq * scale;
        }

        yb += NSG * NQ * 32;
    }

    threadgroup float shmem[NR0 * NW];

    for (uint row = 0; row < NR0; ++row) {
        if (sgitg == 0) {
            shmem[row * NW + tiisg] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem[row * NW + sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0 && r0 + row < out_dim; ++row) {
        float tot = simd_sum(shmem[row * NW + tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            float val = tot + accum[r0 + row];
            accum[r0 + row] = val;
            copy_dst[r0 + row] = val;
        }
    }
}

// ============================================================================
// dequant_matmul_q8_0_silu_deferred_residual_copy_nr2:
// Same as dequant_matmul_q8_0_deferred_residual_copy_nr2 but applies
// silu(gate[i]) * x[i] inline during x-vector loading. This eliminates
// the separate silu_elementwise_mul dispatch and barrier.
//
// buffer(0): weights   [out_dim, in_dim] Q8_0
// buffer(1): x         [in_dim] float -- values to gate
// buffer(2): accum     [out_dim] float -- R/W (residual accumulator)
// buffer(3): in_dim (uint)
// buffer(4): copy_dst  [out_dim] float -- write copy of accum
// buffer(5): out_dim (uint)
// buffer(6): gate      [in_dim] float -- gate values (silu applied)
//
// Dispatch: threadgroups = ceil(out_dim/2), threads_per_threadgroup = 128
// ============================================================================

kernel void dequant_matmul_q8_0_silu_deferred_residual_copy_nr2(
    device const uchar* weights   [[buffer(0)]],
    device const float* x         [[buffer(1)]],
    device       float* accum     [[buffer(2)]],
    constant     uint&  in_dim    [[buffer(3)]],
    device       float* copy_dst  [[buffer(4)]],
    constant     uint&  out_dim   [[buffer(5)]],
    device const float* gate      [[buffer(6)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint NR0 = 2;
    const uint NSG = 4;
    const uint NQ = 8;
    const uint NW = 32;
    const uint Q8_BLOCK_SIZE = 34;

    const uint nb = in_dim >> 5;
    const uint row_bytes = nb * Q8_BLOCK_SIZE;

    const uint r0 = tgpig * NR0;

    const uint ix = tiisg / (NW / NQ);
    const uint il = tiisg % (NW / NQ);

    const uint ib0 = sgitg * NQ + ix;

    device const uchar* ax[NR0];
    for (uint row = 0; row < NR0; ++row) {
        ax[row] = weights + (r0 + row) * row_bytes;
    }

    float sumf[NR0] = { 0.f, 0.f };

    float yl[NQ];

    device const float* xb = x + ib0 * 32 + il * NQ;
    device const float* gb = gate + ib0 * 32 + il * NQ;

    for (uint ib = ib0; ib < nb; ib += NSG * NQ) {
        // Fused silu(gate) * x: load gate and x, compute inline
        #pragma clang loop unroll(full)
        for (uint i = 0; i < NQ; ++i) {
            float g = gb[i];
            float sigmoid = 1.0f / (1.0f + exp(-g));
            yl[i] = g * sigmoid * xb[i];
        }

        for (uint row = 0; row < NR0; ++row) {
            if (r0 + row >= out_dim) break;

            device const uchar* bp = ax[row] + ib * Q8_BLOCK_SIZE;
            float scale = float(as_type<half>(*(device const ushort*)bp));
            device const char* qs = (device const char*)(bp + 2) + il * NQ;

            float sumq = 0.f;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < NQ; ++i) {
                sumq += float(qs[i]) * yl[i];
            }
            sumf[row] += sumq * scale;
        }

        xb += NSG * NQ * 32;
        gb += NSG * NQ * 32;
    }

    threadgroup float shmem[NR0 * NW];

    for (uint row = 0; row < NR0; ++row) {
        if (sgitg == 0) {
            shmem[row * NW + tiisg] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem[row * NW + sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0 && r0 + row < out_dim; ++row) {
        float tot = simd_sum(shmem[row * NW + tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            float val = tot + accum[r0 + row];
            accum[r0 + row] = val;
            copy_dst[r0 + row] = val;
        }
    }
}

// ============================================================================
// Fused RMSNorm + Q8_0 matvec NR2 variants.
//
// These kernels eliminate the separate RMSNorm dispatch by computing the
// RMSNorm scale factor as a preamble, then applying x[i]*scale*norm_w[i]
// inline during the matvec x-vector load.
//
// Phase 1 (preamble): All 128 threads cooperatively compute sum(x^2),
//   reduce via simd_sum + shmem, compute scale = rsqrt(ss/dim + eps).
//   Cost: ~dim/128 iterations + 2 reductions.  For dim=2048: 16 iters.
//
// Phase 2 (matvec): Standard deferred-reduction pattern with NR0=2,
//   but loads yl[i] = x[idx] * scale * norm_w[idx] instead of just x[idx].
//
// Saves per invocation: 1 RMSNorm dispatch, 1 barrier, 1 full write + read
//   of the normed intermediate buffer (dim * 4 bytes).
//
// Dispatch: threadgroups = ceil(out_dim/2), threads_per_threadgroup = 128
// ============================================================================

kernel void rmsnorm_dequant_matmul_q8_0_deferred_nr2(
    device const uchar* weights   [[buffer(0)]],
    device const float* x         [[buffer(1)]],
    device float*       out       [[buffer(2)]],
    constant uint&      in_dim    [[buffer(3)]],
    constant uint&      out_dim   [[buffer(4)]],
    device const uchar* norm_w    [[buffer(5)]],    // RMSNorm weight [in_dim] as f32 bytes
    constant float&     eps       [[buffer(6)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    // Single-pass fused RMSNorm + Q8_0 matvec NR2.
    //
    // Key insight: dot(W, norm_x) = scale * dot(W, x * norm_w)
    // where scale = rsqrt(sum(x^2)/dim + eps).
    //
    // We compute dot(W, x * norm_w) AND sum(x^2) in the SAME loop,
    // then multiply the final result by scale. Zero preamble overhead.
    //
    // Each thread reads x exactly ONCE, accumulating both the dot product
    // contribution and the sum-of-squares simultaneously.

    const uint NR0 = 2;
    const uint NSG = 4;
    const uint NQ = 8;
    const uint NW = 32;
    const uint Q8_BLOCK_SIZE = 34;

    device const float* norm_weight = (device const float*)norm_w;

    const uint nb = in_dim >> 5;
    const uint row_bytes = nb * Q8_BLOCK_SIZE;

    const uint r0 = tgpig * NR0;

    const uint ix = tiisg / (NW / NQ);
    const uint il = tiisg % (NW / NQ);

    const uint ib0 = sgitg * NQ + ix;

    device const uchar* ax[NR0];
    for (uint row = 0; row < NR0; ++row) {
        ax[row] = weights + (r0 + row) * row_bytes;
    }

    float sumf[NR0] = { 0.f, 0.f };
    float ss = 0.0f;  // sum of squares for RMSNorm

    float yl[NQ];

    device const float* yb = x + ib0 * 32 + il * NQ;
    device const float* nwb = norm_weight + ib0 * 32 + il * NQ;

    for (uint ib = ib0; ib < nb; ib += NSG * NQ) {
        // Load x * norm_w (without scale) and accumulate x^2 simultaneously
        #pragma clang loop unroll(full)
        for (uint i = 0; i < NQ; ++i) {
            float xi = yb[i];
            ss += xi * xi;
            yl[i] = xi * nwb[i];
        }

        for (uint row = 0; row < NR0; ++row) {
            if (r0 + row >= out_dim) break;

            device const uchar* bp = ax[row] + ib * Q8_BLOCK_SIZE;
            float scale = float(as_type<half>(*(device const ushort*)bp));
            device const char* qs = (device const char*)(bp + 2) + il * NQ;

            float sumq = 0.f;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < NQ; ++i) {
                sumq += float(qs[i]) * yl[i];
            }
            sumf[row] += sumq * scale;
        }

        yb += NSG * NQ * 32;
        nwb += NSG * NQ * 32;
    }

    // Reduce sum-of-squares across all 128 threads
    ss = simd_sum(ss);

    threadgroup float shmem[NR0 * NW];

    if (tiisg == 0) {
        shmem[sgitg] = ss;
    }

    // Also reduce dot products
    for (uint row = 0; row < NR0; ++row) {
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute RMSNorm scale from reduced sum-of-squares
    threadgroup float rms_scale_shared;
    if (sgitg == 0) {
        float total_ss = (tiisg < NSG) ? shmem[tiisg] : 0.0f;
        total_ss = simd_sum(total_ss);
        if (tiisg == 0) {
            rms_scale_shared = rsqrt(total_ss / float(in_dim) + eps);
        }
    }

    // Write dot product partial sums to shmem (reuse space)
    for (uint row = 0; row < NR0; ++row) {
        if (sgitg == 0) {
            shmem[row * NW + tiisg] = 0.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem[row * NW + sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms_scale = rms_scale_shared;

    // Final reduction of dot products, multiply by RMSNorm scale
    for (uint row = 0; row < NR0 && r0 + row < out_dim; ++row) {
        float tot = simd_sum(shmem[row * NW + tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            out[r0 + row] = tot * rms_scale;
        }
    }
}

// Fused RMSNorm + Q8_0 matvec + residual NR2.
// Single-pass: computes dot(W, x * norm_w) and sum(x^2) simultaneously,
// then applies scale * dot + residual at the end.
kernel void rmsnorm_dequant_matmul_q8_0_deferred_residual_nr2(
    device const uchar* weights   [[buffer(0)]],
    device const float* x         [[buffer(1)]],
    device float*       out       [[buffer(2)]],
    constant uint&      in_dim    [[buffer(3)]],
    device const float* residual  [[buffer(4)]],
    constant uint&      out_dim   [[buffer(5)]],
    device const uchar* norm_w    [[buffer(6)]],
    constant float&     eps       [[buffer(7)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint NR0 = 2;
    const uint NSG = 4;
    const uint NQ = 8;
    const uint NW = 32;
    const uint Q8_BLOCK_SIZE = 34;

    device const float* norm_weight = (device const float*)norm_w;

    const uint nb = in_dim >> 5;
    const uint row_bytes = nb * Q8_BLOCK_SIZE;

    const uint r0 = tgpig * NR0;

    const uint ix = tiisg / (NW / NQ);
    const uint il = tiisg % (NW / NQ);

    const uint ib0 = sgitg * NQ + ix;

    device const uchar* ax[NR0];
    for (uint row = 0; row < NR0; ++row) {
        ax[row] = weights + (r0 + row) * row_bytes;
    }

    float sumf[NR0] = { 0.f, 0.f };
    float ss = 0.0f;

    float yl[NQ];

    device const float* yb = x + ib0 * 32 + il * NQ;
    device const float* nwb = norm_weight + ib0 * 32 + il * NQ;

    for (uint ib = ib0; ib < nb; ib += NSG * NQ) {
        #pragma clang loop unroll(full)
        for (uint i = 0; i < NQ; ++i) {
            float xi = yb[i];
            ss += xi * xi;
            yl[i] = xi * nwb[i];
        }

        for (uint row = 0; row < NR0; ++row) {
            if (r0 + row >= out_dim) break;

            device const uchar* bp = ax[row] + ib * Q8_BLOCK_SIZE;
            float scale = float(as_type<half>(*(device const ushort*)bp));
            device const char* qs = (device const char*)(bp + 2) + il * NQ;

            float sumq = 0.f;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < NQ; ++i) {
                sumq += float(qs[i]) * yl[i];
            }
            sumf[row] += sumq * scale;
        }

        yb += NSG * NQ * 32;
        nwb += NSG * NQ * 32;
    }

    ss = simd_sum(ss);

    threadgroup float shmem[NR0 * NW];

    if (tiisg == 0) {
        shmem[sgitg] = ss;
    }

    for (uint row = 0; row < NR0; ++row) {
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float rms_scale_shared;
    if (sgitg == 0) {
        float total_ss = (tiisg < NSG) ? shmem[tiisg] : 0.0f;
        total_ss = simd_sum(total_ss);
        if (tiisg == 0) {
            rms_scale_shared = rsqrt(total_ss / float(in_dim) + eps);
        }
    }

    for (uint row = 0; row < NR0; ++row) {
        if (sgitg == 0) {
            shmem[row * NW + tiisg] = 0.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem[row * NW + sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms_scale = rms_scale_shared;

    for (uint row = 0; row < NR0 && r0 + row < out_dim; ++row) {
        float tot = simd_sum(shmem[row * NW + tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            out[r0 + row] = tot * rms_scale + residual[r0 + row];
        }
    }
}

// ============================================================================
// Fused RMSNorm + Q8_0 matvec — 2SG variant
//
// Same single-pass fusion as rmsnorm_dequant_matmul_q8_0_deferred_nr2:
//   dot(W, norm_x) = scale * dot(W, x * norm_w)
//   where scale = rsqrt(sum(x^2)/dim + eps)
//
// Key difference from NR2: each SG independently processes ALL blocks (stride NQ),
// so each SG computes the FULL sum-of-squares. No cross-SG reduction needed for ss.
// Dot products: each SG owns 4 rows, only needs simd_sum (no threadgroup barrier).
//
// 2 SGs, 64 threads/TG, 8 rows/TG, ZERO threadgroup barriers for dot products.
// One threadgroup barrier needed for ss -> rms_scale broadcast between SGs.
//
// Dispatch: threadgroups = ceil(out_dim/8), threads_per_threadgroup = 64
// ============================================================================

kernel void rmsnorm_dequant_matmul_q8_0_2sg(
    device const uchar* weights   [[buffer(0)]],
    device const float* x         [[buffer(1)]],
    device float*       out       [[buffer(2)]],
    constant uint&      in_dim    [[buffer(3)]],
    constant uint&      out_dim   [[buffer(4)]],
    device const uchar* norm_w    [[buffer(5)]],
    constant float&     eps       [[buffer(6)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint NR = 4;
    const uint NQ = 8;
    const uint NW = 32;
    const uint Q8_BLOCK_SIZE = 34;

    device const float* norm_weight = (device const float*)norm_w;

    const uint nb = in_dim >> 5;
    const uint row_bytes = nb * Q8_BLOCK_SIZE;

    const uint r0 = tgpig * 8 + sgitg * NR;

    const uint ix = tiisg / (NW / NQ);
    const uint il = tiisg % (NW / NQ);

    const uint ib0 = ix;

    device const uchar* ax[NR];
    for (uint row = 0; row < NR; ++row) {
        ax[row] = weights + (r0 + row) * row_bytes;
    }

    float sumf[NR] = { 0.f, 0.f, 0.f, 0.f };
    float ss = 0.0f;

    float yl[NQ];

    device const float* yb = x + ib0 * 32 + il * NQ;
    device const float* nwb = norm_weight + ib0 * 32 + il * NQ;

    for (uint ib = ib0; ib < nb; ib += NQ) {
        #pragma clang loop unroll(full)
        for (uint i = 0; i < NQ; ++i) {
            float xi = yb[i];
            ss += xi * xi;
            yl[i] = xi * nwb[i];
        }

        for (uint row = 0; row < NR; ++row) {
            if (r0 + row >= out_dim) break;

            device const uchar* bp = ax[row] + ib * Q8_BLOCK_SIZE;
            float scale = float(as_type<half>(*(device const ushort*)bp));
            device const char* qs = (device const char*)(bp + 2) + il * NQ;

            float sumq = 0.f;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < NQ; ++i) {
                sumq += float(qs[i]) * yl[i];
            }
            sumf[row] += sumq * scale;
        }

        yb += NQ * 32;
        nwb += NQ * 32;
    }

    // Each SG independently computes the full ss — just simd_sum within SG
    ss = simd_sum(ss);

    // Compute RMSNorm scale (both SGs have identical ss, but use SG0 as canonical)
    // Broadcast via threadgroup memory with a single barrier
    threadgroup float rms_scale_shared;
    if (sgitg == 0 && tiisg == 0) {
        rms_scale_shared = rsqrt(ss / float(in_dim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rms_scale = rms_scale_shared;

    // Final reduction: simd_sum for each row, then write
    for (uint row = 0; row < NR && r0 + row < out_dim; ++row) {
        sumf[row] = simd_sum(sumf[row]);
        if (tiisg == 0) {
            out[r0 + row] = sumf[row] * rms_scale;
        }
    }
}

// ============================================================================
// Fused RMSNorm + Q8_0 matvec + residual — 2SG variant
// out[row] = dot(W, norm(x)) + residual[row]
// Same pattern as above with residual addition at the end.
//
// Dispatch: threadgroups = ceil(out_dim/8), threads_per_threadgroup = 64
// ============================================================================

kernel void rmsnorm_dequant_matmul_q8_0_2sg_residual(
    device const uchar* weights   [[buffer(0)]],
    device const float* x         [[buffer(1)]],
    device float*       out       [[buffer(2)]],
    constant uint&      in_dim    [[buffer(3)]],
    device const float* residual  [[buffer(4)]],
    constant uint&      out_dim   [[buffer(5)]],
    device const uchar* norm_w    [[buffer(6)]],
    constant float&     eps       [[buffer(7)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint NR = 4;
    const uint NQ = 8;
    const uint NW = 32;
    const uint Q8_BLOCK_SIZE = 34;

    device const float* norm_weight = (device const float*)norm_w;

    const uint nb = in_dim >> 5;
    const uint row_bytes = nb * Q8_BLOCK_SIZE;

    const uint r0 = tgpig * 8 + sgitg * NR;

    const uint ix = tiisg / (NW / NQ);
    const uint il = tiisg % (NW / NQ);

    const uint ib0 = ix;

    device const uchar* ax[NR];
    for (uint row = 0; row < NR; ++row) {
        ax[row] = weights + (r0 + row) * row_bytes;
    }

    float sumf[NR] = { 0.f, 0.f, 0.f, 0.f };
    float ss = 0.0f;

    float yl[NQ];

    device const float* yb = x + ib0 * 32 + il * NQ;
    device const float* nwb = norm_weight + ib0 * 32 + il * NQ;

    for (uint ib = ib0; ib < nb; ib += NQ) {
        #pragma clang loop unroll(full)
        for (uint i = 0; i < NQ; ++i) {
            float xi = yb[i];
            ss += xi * xi;
            yl[i] = xi * nwb[i];
        }

        for (uint row = 0; row < NR; ++row) {
            if (r0 + row >= out_dim) break;

            device const uchar* bp = ax[row] + ib * Q8_BLOCK_SIZE;
            float scale = float(as_type<half>(*(device const ushort*)bp));
            device const char* qs = (device const char*)(bp + 2) + il * NQ;

            float sumq = 0.f;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < NQ; ++i) {
                sumq += float(qs[i]) * yl[i];
            }
            sumf[row] += sumq * scale;
        }

        yb += NQ * 32;
        nwb += NQ * 32;
    }

    ss = simd_sum(ss);

    threadgroup float rms_scale_shared;
    if (sgitg == 0 && tiisg == 0) {
        rms_scale_shared = rsqrt(ss / float(in_dim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rms_scale = rms_scale_shared;

    for (uint row = 0; row < NR && r0 + row < out_dim; ++row) {
        sumf[row] = simd_sum(sumf[row]);
        if (tiisg == 0) {
            out[r0 + row] = sumf[row] * rms_scale + residual[r0 + row];
        }
    }
}

// Fused RMSNorm + FFN Gate+Up+SwiGLU Q8_0 deferred.
// Single-pass: computes dot(gate/up, x * norm_w) and sum(x^2) simultaneously.
// After loop, computes rms_scale, multiplies gate/up sums by scale, applies SwiGLU.
kernel void rmsnorm_ffn_fused_gate_up_swiglu_q8_0_deferred(
    device const uchar* w_gate_q8   [[buffer(0)]],   // gate weights Q8_0 [inter_dim, hidden_dim]
    device const float* x           [[buffer(1)]],   // input [hidden_dim] (NOT normed)
    device float*       out         [[buffer(2)]],   // output [inter_dim] (SwiGLU result)
    constant uint&      in_dim      [[buffer(3)]],   // hidden_dim
    constant uint&      out_dim     [[buffer(4)]],   // inter_dim
    device const uchar* w_up_q8     [[buffer(5)]],   // up weights Q8_0 [inter_dim, hidden_dim]
    device const uchar* norm_w      [[buffer(6)]],   // RMSNorm weight [hidden_dim] as f32 bytes
    constant float&     eps         [[buffer(7)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    uint row = tgpig;
    if (row >= out_dim) return;

    device const float* norm_weight = (device const float*)norm_w;

    const uint Q8_BLOCK_SIZE = 34;
    uint num_blocks = in_dim >> 5;
    uint row_bytes = num_blocks * Q8_BLOCK_SIZE;

    uint ix = tiisg / 4;
    uint il = tiisg % 4;

    device const uchar* gate_row_ptr = w_gate_q8 + row * row_bytes;
    device const uchar* up_row_ptr   = w_up_q8   + row * row_bytes;

    float gate_sum = 0.0f;
    float up_sum   = 0.0f;
    float ss       = 0.0f;

    // Single-pass: accumulate gate/up dot products (unscaled) AND sum(x^2)
    for (uint ib = sgitg * 8 + ix; ib < num_blocks; ib += 32) {
        uint x_base = ib * 32 + il * 8;
        float yl[8];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; i++) {
            float xi = x[x_base + i];
            ss += xi * xi;
            yl[i] = xi * norm_weight[x_base + i];
        }

        device const uchar* gate_bp = gate_row_ptr + ib * Q8_BLOCK_SIZE;
        half gate_scale = as_type<half>(*(device const ushort*)gate_bp);
        device const char* gate_qs = (device const char*)(gate_bp + 2) + il * 8;
        float gate_dot = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; i++) {
            gate_dot += float(gate_qs[i]) * yl[i];
        }
        gate_sum += gate_dot * float(gate_scale);

        device const uchar* up_bp = up_row_ptr + ib * Q8_BLOCK_SIZE;
        half up_scale = as_type<half>(*(device const ushort*)up_bp);
        device const char* up_qs = (device const char*)(up_bp + 2) + il * 8;
        float up_dot = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; i++) {
            up_dot += float(up_qs[i]) * yl[i];
        }
        up_sum += up_dot * float(up_scale);
    }

    // Reduce all three quantities
    gate_sum = simd_sum(gate_sum);
    up_sum   = simd_sum(up_sum);
    ss       = simd_sum(ss);

    threadgroup float shmem[12];  // [0..3] gate, [4..7] up, [8..11] ss

    if (tiisg == 0) {
        shmem[sgitg]     = gate_sum;
        shmem[sgitg + 4] = up_sum;
        shmem[sgitg + 8] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0 && tiisg == 0) {
        float g = shmem[0] + shmem[1] + shmem[2] + shmem[3];
        float u = shmem[4] + shmem[5] + shmem[6] + shmem[7];
        float total_ss = shmem[8] + shmem[9] + shmem[10] + shmem[11];
        float rms_scale = rsqrt(total_ss / float(in_dim) + eps);
        // Apply scale to both gate and up sums
        g *= rms_scale;
        u *= rms_scale;
        float sigmoid = 1.0f / (1.0f + exp(-g));
        out[row] = g * sigmoid * u;
    }
}

// ============================================================================
// rmsnorm_ffn_fused_gate_up_swiglu_q8_0_8row: 8-row fused RMSNorm + FFN
//
// Same computation as rmsnorm_ffn_fused_gate_up_swiglu_q8_0_deferred but uses
// the 8-row independent SG pattern: 8 SGs (256 threads), each SG independently
// owns 1 output row. Zero threadgroup barriers. Zero shared memory for reduction.
//
// Key insight: Each SG independently walks the ENTIRE x-vector for its row's
// gate+up dot products. Since every SG reads all of x, each SG also computes
// the complete sum-of-squares for RMSNorm -- no cross-SG communication needed.
//
// Benefits vs deferred (1 row/TG, 4 SGs cooperating):
// - Zero threadgroup barriers (deferred needs 2)
// - Zero shared memory for reduction
// - 8x fewer threadgroups (1024 vs 8192 for inter_dim=8192)
// - Better x-vector cache reuse (8 SGs read same x from L1/L2)
//
// Trade-off: Each SG reads x independently (8x bandwidth), but for small
// hidden_dim (2048 = 8 KB), x fits in L1 cache and subsequent SG reads hit cache.
//
// Dispatch: ceil(inter_dim/8) threadgroups, 256 threads each (8 simdgroups)
// ============================================================================

kernel void rmsnorm_ffn_fused_gate_up_swiglu_q8_0_8row(
    device const uchar* w_gate_q8   [[buffer(0)]],   // gate weights Q8_0 [inter_dim, hidden_dim]
    device const float* x           [[buffer(1)]],   // input [hidden_dim] (NOT normed)
    device float*       out         [[buffer(2)]],   // output [inter_dim] (SwiGLU result)
    constant uint&      in_dim      [[buffer(3)]],   // hidden_dim
    constant uint&      out_dim     [[buffer(4)]],   // inter_dim
    device const uchar* w_up_q8     [[buffer(5)]],   // up weights Q8_0 [inter_dim, hidden_dim]
    device const uchar* norm_w      [[buffer(6)]],   // RMSNorm weight [hidden_dim] as f32 bytes
    constant float&     eps         [[buffer(7)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    // Each SG independently owns 1 output row. 8 SGs per TG = 8 rows per TG.
    // Uses deferred reduction (accumulate locally, one simd_sum at end).
    // Zero threadgroup barriers, zero shared memory.
    uint row = tgpig * 8 + sgitg;
    if (row >= out_dim) return;

    device const float* norm_weight = (device const float*)norm_w;

    const uint NQ = 8;
    const uint Q8_BLOCK_SIZE = 34;
    uint num_blocks = in_dim >> 5;
    uint row_bytes = num_blocks * Q8_BLOCK_SIZE;

    // Thread mapping within SG (same as deferred pattern)
    uint ix = tiisg / 4;   // 0..7: which block in stride of 8
    uint il = tiisg % 4;   // 0..3: which 8-byte quarter within 32-byte data

    device const uchar* gate_row_ptr = w_gate_q8 + row * row_bytes;
    device const uchar* up_row_ptr   = w_up_q8   + row * row_bytes;

    float gate_sum = 0.0f;
    float up_sum   = 0.0f;
    float ss       = 0.0f;

    // Each SG independently processes ALL blocks for its row.
    // Stride = NQ = 8 blocks per iteration (single SG, no cross-SG stride).
    for (uint ib = ix; ib < num_blocks; ib += NQ) {
        uint x_base = ib * 32 + il * 8;
        float yl[8];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; i++) {
            float xi = x[x_base + i];
            ss += xi * xi;
            yl[i] = xi * norm_weight[x_base + i];
        }

        // Gate weights
        device const uchar* gate_bp = gate_row_ptr + ib * Q8_BLOCK_SIZE;
        half gate_scale = as_type<half>(*(device const ushort*)gate_bp);
        device const char* gate_qs = (device const char*)(gate_bp + 2) + il * 8;
        float gate_dot = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; i++) {
            gate_dot += float(gate_qs[i]) * yl[i];
        }
        gate_sum += gate_dot * float(gate_scale);

        // Up weights
        device const uchar* up_bp = up_row_ptr + ib * Q8_BLOCK_SIZE;
        half up_scale = as_type<half>(*(device const ushort*)up_bp);
        device const char* up_qs = (device const char*)(up_bp + 2) + il * 8;
        float up_dot = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; i++) {
            up_dot += float(up_qs[i]) * yl[i];
        }
        up_sum += up_dot * float(up_scale);
    }

    // Final reduction: ONE simd_sum each (deferred pattern payoff)
    gate_sum = simd_sum(gate_sum);
    up_sum   = simd_sum(up_sum);
    ss       = simd_sum(ss);

    // Apply RMSNorm scale + SwiGLU
    if (tiisg == 0) {
        float rms_scale = rsqrt(ss / float(in_dim) + eps);
        float g = gate_sum * rms_scale;
        float u = up_sum * rms_scale;
        float sigmoid = 1.0f / (1.0f + exp(-g));
        out[row] = g * sigmoid * u;
    }
}

// ============================================================================
// Fused RMSNorm + Q4_0 matvec NR2
//
// Same single-pass fusion as rmsnorm_dequant_matmul_q8_0_deferred_nr2:
//   dot(W, norm_x) = scale * dot(W, x * norm_w)
//   where scale = rsqrt(sum(x^2)/dim + eps)
//
// Q4_0 block: 18 bytes = 2B f16 scale + 16B packed nibbles (32 elements).
// De-interleaved: byte j has lo -> element j, hi -> element j+16.
// Thread processes 4 bytes = 8 elements (4 lo + 4 hi).
//
// Dispatch: threadgroups = ceil(out_dim/2), threads_per_threadgroup = 128
// ============================================================================

kernel void rmsnorm_dequant_matmul_q4_0_deferred_nr2(
    device const uchar* weights   [[buffer(0)]],
    device const float* x         [[buffer(1)]],
    device float*       out       [[buffer(2)]],
    constant uint&      in_dim    [[buffer(3)]],
    constant uint&      out_dim   [[buffer(4)]],
    device const uchar* norm_w    [[buffer(5)]],    // RMSNorm weight [in_dim] as f32 bytes
    constant float&     eps       [[buffer(6)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    // Single-pass fused RMSNorm + Q4_0 matvec NR2.
    // Computes dot(W, x * norm_w) AND sum(x^2) in the SAME loop.

    const uint NR0 = 2;
    const uint NSG = 4;
    const uint NQ = 8;
    const uint NW = 32;
    const uint Q4_BS = 18;  // Q4_0 block size: 2B scale + 16B nibbles

    device const float* norm_weight = (device const float*)norm_w;

    const uint nb = in_dim >> 5;
    const uint row_bytes = nb * Q4_BS;

    const uint r0 = tgpig * NR0;

    const uint ix = tiisg / (NW / NQ);
    const uint il = tiisg % (NW / NQ);

    const uint ib0 = sgitg * NQ + ix;

    device const uchar* ax[NR0];
    for (uint row = 0; row < NR0; ++row) {
        ax[row] = weights + (r0 + row) * row_bytes;
    }

    float sumf[NR0] = { 0.f, 0.f };
    float ss = 0.0f;  // sum of squares for RMSNorm

    device const float* yb_lo = x + ib0 * 32 + il * 4;
    device const float* yb_hi = x + ib0 * 32 + il * 4 + 16;
    device const float* nwb_lo = norm_weight + ib0 * 32 + il * 4;
    device const float* nwb_hi = norm_weight + ib0 * 32 + il * 4 + 16;

    for (uint ib = ib0; ib < nb; ib += NSG * NQ) {
        // Load x * norm_w for lo and hi nibble positions, accumulate x^2
        float yl_lo[4], yl_hi[4];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            float xi_lo = yb_lo[i];
            float xi_hi = yb_hi[i];
            ss += xi_lo * xi_lo + xi_hi * xi_hi;
            yl_lo[i] = xi_lo * nwb_lo[i];
            yl_hi[i] = xi_hi * nwb_hi[i];
        }

        for (uint row = 0; row < NR0; ++row) {
            if (r0 + row >= out_dim) break;

            device const uchar* bp = ax[row] + ib * Q4_BS;
            float scale = float(as_type<half>(*(device const ushort*)bp));
            device const uchar* qdata = (device const uchar*)(bp + 2) + il * 4;

            float sumq = 0.f;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; ++i) {
                uchar byte_val = qdata[i];
                float lo = float(byte_val & 0x0F) - 8.0f;
                float hi = float(byte_val >> 4) - 8.0f;
                sumq += lo * yl_lo[i] + hi * yl_hi[i];
            }
            sumf[row] += sumq * scale;
        }

        yb_lo += NSG * NQ * 32;
        yb_hi += NSG * NQ * 32;
        nwb_lo += NSG * NQ * 32;
        nwb_hi += NSG * NQ * 32;
    }

    // Reduce sum-of-squares across all 128 threads
    ss = simd_sum(ss);

    threadgroup float shmem[NR0 * NW];

    if (tiisg == 0) {
        shmem[sgitg] = ss;
    }

    // Also reduce dot products
    for (uint row = 0; row < NR0; ++row) {
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute RMSNorm scale from reduced sum-of-squares
    threadgroup float rms_scale_shared;
    if (sgitg == 0) {
        float total_ss = (tiisg < NSG) ? shmem[tiisg] : 0.0f;
        total_ss = simd_sum(total_ss);
        if (tiisg == 0) {
            rms_scale_shared = rsqrt(total_ss / float(in_dim) + eps);
        }
    }

    // Write dot product partial sums to shmem (reuse space)
    for (uint row = 0; row < NR0; ++row) {
        if (sgitg == 0) {
            shmem[row * NW + tiisg] = 0.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem[row * NW + sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms_scale = rms_scale_shared;

    // Final reduction of dot products, multiply by RMSNorm scale
    for (uint row = 0; row < NR0 && r0 + row < out_dim; ++row) {
        float tot = simd_sum(shmem[row * NW + tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            out[r0 + row] = tot * rms_scale;
        }
    }
}

// Fused RMSNorm + Q4_0 matvec + residual NR2.
// Single-pass: computes dot(W, x * norm_w) and sum(x^2) simultaneously,
// then applies scale * dot + residual at the end.
kernel void rmsnorm_dequant_matmul_q4_0_deferred_residual_nr2(
    device const uchar* weights   [[buffer(0)]],
    device const float* x         [[buffer(1)]],
    device float*       out       [[buffer(2)]],
    constant uint&      in_dim    [[buffer(3)]],
    device const float* residual  [[buffer(4)]],
    constant uint&      out_dim   [[buffer(5)]],
    device const uchar* norm_w    [[buffer(6)]],
    constant float&     eps       [[buffer(7)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint NR0 = 2;
    const uint NSG = 4;
    const uint NQ = 8;
    const uint NW = 32;
    const uint Q4_BS = 18;  // Q4_0 block size: 2B scale + 16B nibbles

    device const float* norm_weight = (device const float*)norm_w;

    const uint nb = in_dim >> 5;
    const uint row_bytes = nb * Q4_BS;

    const uint r0 = tgpig * NR0;

    const uint ix = tiisg / (NW / NQ);
    const uint il = tiisg % (NW / NQ);

    const uint ib0 = sgitg * NQ + ix;

    device const uchar* ax[NR0];
    for (uint row = 0; row < NR0; ++row) {
        ax[row] = weights + (r0 + row) * row_bytes;
    }

    float sumf[NR0] = { 0.f, 0.f };
    float ss = 0.0f;

    device const float* yb_lo = x + ib0 * 32 + il * 4;
    device const float* yb_hi = x + ib0 * 32 + il * 4 + 16;
    device const float* nwb_lo = norm_weight + ib0 * 32 + il * 4;
    device const float* nwb_hi = norm_weight + ib0 * 32 + il * 4 + 16;

    for (uint ib = ib0; ib < nb; ib += NSG * NQ) {
        float yl_lo[4], yl_hi[4];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            float xi_lo = yb_lo[i];
            float xi_hi = yb_hi[i];
            ss += xi_lo * xi_lo + xi_hi * xi_hi;
            yl_lo[i] = xi_lo * nwb_lo[i];
            yl_hi[i] = xi_hi * nwb_hi[i];
        }

        for (uint row = 0; row < NR0; ++row) {
            if (r0 + row >= out_dim) break;

            device const uchar* bp = ax[row] + ib * Q4_BS;
            float scale = float(as_type<half>(*(device const ushort*)bp));
            device const uchar* qdata = (device const uchar*)(bp + 2) + il * 4;

            float sumq = 0.f;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; ++i) {
                uchar byte_val = qdata[i];
                float lo = float(byte_val & 0x0F) - 8.0f;
                float hi = float(byte_val >> 4) - 8.0f;
                sumq += lo * yl_lo[i] + hi * yl_hi[i];
            }
            sumf[row] += sumq * scale;
        }

        yb_lo += NSG * NQ * 32;
        yb_hi += NSG * NQ * 32;
        nwb_lo += NSG * NQ * 32;
        nwb_hi += NSG * NQ * 32;
    }

    ss = simd_sum(ss);

    threadgroup float shmem[NR0 * NW];

    if (tiisg == 0) {
        shmem[sgitg] = ss;
    }

    for (uint row = 0; row < NR0; ++row) {
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float rms_scale_shared;
    if (sgitg == 0) {
        float total_ss = (tiisg < NSG) ? shmem[tiisg] : 0.0f;
        total_ss = simd_sum(total_ss);
        if (tiisg == 0) {
            rms_scale_shared = rsqrt(total_ss / float(in_dim) + eps);
        }
    }

    for (uint row = 0; row < NR0; ++row) {
        if (sgitg == 0) {
            shmem[row * NW + tiisg] = 0.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem[row * NW + sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms_scale = rms_scale_shared;

    for (uint row = 0; row < NR0 && r0 + row < out_dim; ++row) {
        float tot = simd_sum(shmem[row * NW + tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            out[r0 + row] = tot * rms_scale + residual[r0 + row];
        }
    }
}

// ============================================================================
// Fused RMSNorm + FFN Gate+Up+SwiGLU Q4_0 deferred.
// Single-pass: computes dot(gate/up, x * norm_w) and sum(x^2) simultaneously.
// After loop, computes rms_scale, multiplies gate/up sums by scale, applies SwiGLU.
//
// 128 threads (4 simdgroups), 1 output row per threadgroup.
// Q4_0 nibble dequantization for both gate and up weights.
//
// Dispatch: inter_dim threadgroups, 128 threads each
// ============================================================================

kernel void rmsnorm_ffn_fused_gate_up_swiglu_q4_0_deferred(
    device const uchar* w_gate_q4   [[buffer(0)]],   // gate weights Q4_0 [inter_dim, hidden_dim]
    device const float* x           [[buffer(1)]],   // input [hidden_dim] (NOT normed)
    device float*       out         [[buffer(2)]],   // output [inter_dim] (SwiGLU result)
    constant uint&      in_dim      [[buffer(3)]],   // hidden_dim
    constant uint&      out_dim     [[buffer(4)]],   // inter_dim
    device const uchar* w_up_q4     [[buffer(5)]],   // up weights Q4_0 [inter_dim, hidden_dim]
    device const uchar* norm_w      [[buffer(6)]],   // RMSNorm weight [hidden_dim] as f32 bytes
    constant float&     eps         [[buffer(7)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    uint row = tgpig;
    if (row >= out_dim) return;

    device const float* norm_weight = (device const float*)norm_w;

    const uint Q4_BS = 18;  // Q4_0 block size: 2B scale + 16B nibbles
    uint num_blocks = in_dim >> 5;
    uint row_bytes = num_blocks * Q4_BS;

    uint ix = tiisg / 4;
    uint il = tiisg % 4;

    device const uchar* gate_row_ptr = w_gate_q4 + row * row_bytes;
    device const uchar* up_row_ptr   = w_up_q4   + row * row_bytes;

    float gate_sum = 0.0f;
    float up_sum   = 0.0f;
    float ss       = 0.0f;

    // Single-pass: accumulate gate/up dot products (unscaled) AND sum(x^2)
    for (uint ib = sgitg * 8 + ix; ib < num_blocks; ib += 32) {
        uint block_base = ib * 32;
        float yl_lo[4], yl_hi[4];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; i++) {
            float xi_lo = x[block_base + il * 4 + i];
            float xi_hi = x[block_base + il * 4 + 16 + i];
            ss += xi_lo * xi_lo + xi_hi * xi_hi;
            yl_lo[i] = xi_lo * norm_weight[block_base + il * 4 + i];
            yl_hi[i] = xi_hi * norm_weight[block_base + il * 4 + 16 + i];
        }

        // Gate weights
        device const uchar* gate_bp = gate_row_ptr + ib * Q4_BS;
        float gate_scale = float(as_type<half>(*(device const ushort*)gate_bp));
        device const uchar* gate_qdata = (device const uchar*)(gate_bp + 2) + il * 4;
        float gate_dot = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; i++) {
            uchar byte_val = gate_qdata[i];
            float lo = float(byte_val & 0x0F) - 8.0f;
            float hi = float(byte_val >> 4) - 8.0f;
            gate_dot += lo * yl_lo[i] + hi * yl_hi[i];
        }
        gate_sum += gate_dot * gate_scale;

        // Up weights
        device const uchar* up_bp = up_row_ptr + ib * Q4_BS;
        float up_scale = float(as_type<half>(*(device const ushort*)up_bp));
        device const uchar* up_qdata = (device const uchar*)(up_bp + 2) + il * 4;
        float up_dot = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; i++) {
            uchar byte_val = up_qdata[i];
            float lo = float(byte_val & 0x0F) - 8.0f;
            float hi = float(byte_val >> 4) - 8.0f;
            up_dot += lo * yl_lo[i] + hi * yl_hi[i];
        }
        up_sum += up_dot * up_scale;
    }

    // Reduce all three quantities
    gate_sum = simd_sum(gate_sum);
    up_sum   = simd_sum(up_sum);
    ss       = simd_sum(ss);

    threadgroup float shmem[12];  // [0..3] gate, [4..7] up, [8..11] ss

    if (tiisg == 0) {
        shmem[sgitg]     = gate_sum;
        shmem[sgitg + 4] = up_sum;
        shmem[sgitg + 8] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0 && tiisg == 0) {
        float g = shmem[0] + shmem[1] + shmem[2] + shmem[3];
        float u = shmem[4] + shmem[5] + shmem[6] + shmem[7];
        float total_ss = shmem[8] + shmem[9] + shmem[10] + shmem[11];
        float rms_scale = rsqrt(total_ss / float(in_dim) + eps);
        // Apply scale to both gate and up sums
        g *= rms_scale;
        u *= rms_scale;
        float sigmoid = 1.0f / (1.0f + exp(-g));
        out[row] = g * sigmoid * u;
    }
}

// ============================================================================
// rmsnorm_ffn_fused_gate_up_swiglu_q4_0_8row: 8-row fused RMSNorm + FFN
//
// Same computation as rmsnorm_ffn_fused_gate_up_swiglu_q4_0_deferred but uses
// the 8-row independent SG pattern: 8 SGs (256 threads), each SG independently
// owns 1 output row. Zero threadgroup barriers. Zero shared memory for reduction.
//
// Each SG independently walks the ENTIRE x-vector for its row's gate+up dot
// products. Since every SG reads all of x, each SG also computes the complete
// sum-of-squares for RMSNorm -- no cross-SG communication needed.
//
// Q4_0 nibble dequantization for both gate and up weights.
//
// Dispatch: ceil(inter_dim/8) threadgroups, 256 threads each (8 simdgroups)
// ============================================================================

kernel void rmsnorm_ffn_fused_gate_up_swiglu_q4_0_8row(
    device const uchar* w_gate_q4   [[buffer(0)]],   // gate weights Q4_0 [inter_dim, hidden_dim]
    device const float* x           [[buffer(1)]],   // input [hidden_dim] (NOT normed)
    device float*       out         [[buffer(2)]],   // output [inter_dim] (SwiGLU result)
    constant uint&      in_dim      [[buffer(3)]],   // hidden_dim
    constant uint&      out_dim     [[buffer(4)]],   // inter_dim
    device const uchar* w_up_q4     [[buffer(5)]],   // up weights Q4_0 [inter_dim, hidden_dim]
    device const uchar* norm_w      [[buffer(6)]],   // RMSNorm weight [hidden_dim] as f32 bytes
    constant float&     eps         [[buffer(7)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    // Each SG independently owns 1 output row. 8 SGs per TG = 8 rows per TG.
    // Uses deferred reduction (accumulate locally, one simd_sum at end).
    // Zero threadgroup barriers, zero shared memory.
    uint row = tgpig * 8 + sgitg;
    if (row >= out_dim) return;

    device const float* norm_weight = (device const float*)norm_w;

    const uint NQ = 8;
    const uint Q4_BS = 18;  // Q4_0 block size: 2B scale + 16B nibbles
    uint num_blocks = in_dim >> 5;
    uint row_bytes = num_blocks * Q4_BS;

    // Thread mapping within SG (same as deferred pattern)
    uint ix = tiisg / 4;   // 0..7: which block in stride of 8
    uint il = tiisg % 4;   // 0..3: which 4-byte quarter within 16-byte data

    device const uchar* gate_row_ptr = w_gate_q4 + row * row_bytes;
    device const uchar* up_row_ptr   = w_up_q4   + row * row_bytes;

    float gate_sum = 0.0f;
    float up_sum   = 0.0f;
    float ss       = 0.0f;

    // Each SG independently processes ALL blocks for its row.
    // Stride = NQ = 8 blocks per iteration (single SG, no cross-SG stride).
    for (uint ib = ix; ib < num_blocks; ib += NQ) {
        uint block_base = ib * 32;
        float yl_lo[4], yl_hi[4];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; i++) {
            float xi_lo = x[block_base + il * 4 + i];
            float xi_hi = x[block_base + il * 4 + 16 + i];
            ss += xi_lo * xi_lo + xi_hi * xi_hi;
            yl_lo[i] = xi_lo * norm_weight[block_base + il * 4 + i];
            yl_hi[i] = xi_hi * norm_weight[block_base + il * 4 + 16 + i];
        }

        // Gate weights
        device const uchar* gate_bp = gate_row_ptr + ib * Q4_BS;
        float gate_scale = float(as_type<half>(*(device const ushort*)gate_bp));
        device const uchar* gate_qdata = (device const uchar*)(gate_bp + 2) + il * 4;
        float gate_dot = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; i++) {
            uchar byte_val = gate_qdata[i];
            float lo = float(byte_val & 0x0F) - 8.0f;
            float hi = float(byte_val >> 4) - 8.0f;
            gate_dot += lo * yl_lo[i] + hi * yl_hi[i];
        }
        gate_sum += gate_dot * gate_scale;

        // Up weights
        device const uchar* up_bp = up_row_ptr + ib * Q4_BS;
        float up_scale = float(as_type<half>(*(device const ushort*)up_bp));
        device const uchar* up_qdata = (device const uchar*)(up_bp + 2) + il * 4;
        float up_dot = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; i++) {
            uchar byte_val = up_qdata[i];
            float lo = float(byte_val & 0x0F) - 8.0f;
            float hi = float(byte_val >> 4) - 8.0f;
            up_dot += lo * yl_lo[i] + hi * yl_hi[i];
        }
        up_sum += up_dot * up_scale;
    }

    // Final reduction: ONE simd_sum each (deferred pattern payoff)
    gate_sum = simd_sum(gate_sum);
    up_sum   = simd_sum(up_sum);
    ss       = simd_sum(ss);

    // Apply RMSNorm scale + SwiGLU
    if (tiisg == 0) {
        float rms_scale = rsqrt(ss / float(in_dim) + eps);
        float g = gate_sum * rms_scale;
        float u = up_sum * rms_scale;
        float sigmoid = 1.0f / (1.0f + exp(-g));
        out[row] = g * sigmoid * u;
    }
}

// ============================================================================
// rmsnorm_ffn_fused_gate_up_swiglu_f16_deferred: Fused RMSNorm + F16 FFN
//
// Single-pass: accumulates dot(gate, x*norm_w), dot(up, x*norm_w),
// and sum(x^2) simultaneously, then applies rms_scale + SwiGLU at the end.
// Saves 1 RMSNorm dispatch + 1 barrier + normed_buf write/read + 2 matvec
// dispatches + 1 SwiGLU dispatch.
//
// F16 weights: dense half-precision, no block structure.
// Each row of gate/up weights is in_dim contiguous half values.
//
// 128 threads (4 simdgroups), 1 output row per threadgroup.
//
// Dispatch: inter_dim threadgroups, 128 threads each
// ============================================================================

kernel void rmsnorm_ffn_fused_gate_up_swiglu_f16_deferred(
    device const half*  w_gate_f16  [[buffer(0)]],   // gate weights F16 [inter_dim, hidden_dim]
    device const float* x           [[buffer(1)]],   // input [hidden_dim] (NOT normed)
    device float*       out         [[buffer(2)]],   // output [inter_dim] (SwiGLU result)
    constant uint&      in_dim      [[buffer(3)]],   // hidden_dim
    constant uint&      out_dim     [[buffer(4)]],   // inter_dim
    device const half*  w_up_f16    [[buffer(5)]],   // up weights F16 [inter_dim, hidden_dim]
    device const uchar* norm_w      [[buffer(6)]],   // RMSNorm weight [hidden_dim] as f32 bytes
    constant float&     eps         [[buffer(7)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    uint row = tgpig;
    if (row >= out_dim) return;

    device const float* norm_weight = (device const float*)norm_w;

    const uint NSG = 4;
    const uint NW = 32;
    const uint tid = sgitg * NW + tiisg;
    const uint total_threads = NSG * NW;  // 128

    device const half* gate_row = w_gate_f16 + row * in_dim;
    device const half* up_row   = w_up_f16   + row * in_dim;

    float gate_sum = 0.0f;
    float up_sum   = 0.0f;
    float ss       = 0.0f;

    // Single-pass: accumulate gate/up dot products (with norm_w) AND sum(x^2)
    for (uint j = tid; j < in_dim; j += total_threads) {
        float xi = x[j];
        ss += xi * xi;
        float normed_xj = xi * norm_weight[j];
        gate_sum += float(gate_row[j]) * normed_xj;
        up_sum   += float(up_row[j])   * normed_xj;
    }

    // Final reduction: simd_sum within each simdgroup
    gate_sum = simd_sum(gate_sum);
    up_sum   = simd_sum(up_sum);
    ss       = simd_sum(ss);

    // Cross-simdgroup reduction via shared memory
    threadgroup float shmem[12];  // [0..3] gate, [4..7] up, [8..11] ss

    if (tiisg == 0) {
        shmem[sgitg]     = gate_sum;
        shmem[sgitg + 4] = up_sum;
        shmem[sgitg + 8] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0 && tiisg == 0) {
        float g = shmem[0] + shmem[1] + shmem[2] + shmem[3];
        float u = shmem[4] + shmem[5] + shmem[6] + shmem[7];
        float total_ss = shmem[8] + shmem[9] + shmem[10] + shmem[11];
        float rms_scale = rsqrt(total_ss / float(in_dim) + eps);
        // Apply scale to both gate and up sums
        g *= rms_scale;
        u *= rms_scale;
        float sigmoid = 1.0f / (1.0f + exp(-g));
        out[row] = g * sigmoid * u;
    }
}

// ============================================================================
// ============================================================================
// dequant_matmul_q8_0_2sg: MLX-style 2-SG independent row ownership matvec
//
// Key insight from MLX: Instead of 4 SGs cooperatively computing the SAME rows
// (requiring cross-SG reduction via shmem + 2 barriers), use 2 SGs where each
// SG independently owns 4 output rows. Zero threadgroup barriers. Zero shared
// memory for reduction.
//
// Thread mapping (same as deferred pattern):
//   ix = tiisg / 4  -> 0..7 (which of 8 blocks in the stride)
//   il = tiisg % 4  -> 0..3 (which sub-chunk of 8 within the 32-element block)
//   4 threads collectively process one 32-element Q8_0 block (4 x 8 = 32)
//
// Each SG independently processes ALL blocks for its 4 rows.
// Stride = NQ = 8 blocks per outer iteration (each SG is self-contained).
//
// Dispatch: threadgroups = ceil(out_dim/8), threads_per_threadgroup = 64
// Threadgroup memory: ZERO
// ============================================================================

kernel void dequant_matmul_q8_0_2sg(
    device const uchar* weights [[buffer(0)]],
    device const float* x       [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant uint&      in_dim  [[buffer(3)]],
    constant uint&      out_dim [[buffer(4)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint NR = 4;          // rows per simdgroup
    const uint NQ = 8;          // elements per thread per iteration
    const uint NW = 32;         // SIMD width
    const uint Q8_BLOCK_SIZE = 34; // 2 bytes scale (f16) + 32 bytes data (int8)

    const uint nb = in_dim >> 5;   // number of Q8_0 blocks per row = in_dim / 32
    const uint row_bytes = nb * Q8_BLOCK_SIZE;

    // Each SG owns 4 rows: SG 0 -> rows 0-3, SG 1 -> rows 4-7
    const uint r0 = tgpig * 8 + sgitg * NR;

    // Thread mapping within SIMD group (same as deferred)
    const uint ix = tiisg / (NW / NQ);  // = tiisg / 4 -> 0..7 (block index in stride)
    const uint il = tiisg % (NW / NQ);  // = tiisg % 4 -> 0..3 (sub-chunk index)

    const uint ib0 = ix;               // starting block (no SG offset, each SG independent)

    // Pointers to weight rows
    device const uchar* ax[NR];
    for (uint row = 0; row < NR; ++row) {
        ax[row] = weights + (r0 + row) * row_bytes;
    }

    float sumf[NR] = { 0.f, 0.f, 0.f, 0.f };

    float yl[NQ];

    device const float* yb = x + ib0 * 32 + il * NQ;

    // Main loop: each SG independently processes all blocks for its 4 rows
    // Stride = NQ = 8 blocks per iteration (single SG, not NSG*NQ)
    for (uint ib = ib0; ib < nb; ib += NQ) {
        // Load 8 x-values into registers
        #pragma clang loop unroll(full)
        for (uint i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        // Process all NR rows with these x-values
        for (uint row = 0; row < NR; ++row) {
            if (r0 + row >= out_dim) break;

            // Point to this block in the weight row
            device const uchar* bp = ax[row] + ib * Q8_BLOCK_SIZE;
            float scale = float(as_type<half>(*(device const ushort*)bp));
            device const char* qs = (device const char*)(bp + 2) + il * NQ;

            float sumq = 0.f;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < NQ; ++i) {
                sumq += float(qs[i]) * yl[i];
            }
            sumf[row] += sumq * scale;
        }

        yb += NQ * 32;
    }

    // Final reduction: just simd_sum, NO cross-SG reduction needed
    for (uint row = 0; row < NR && r0 + row < out_dim; ++row) {
        sumf[row] = simd_sum(sumf[row]);
        if (tiisg == 0) {
            out[r0 + row] = sumf[row];
        }
    }
}

// ============================================================================
// dequant_matmul_q8_0_2sg_residual: MLX-style 2-SG matvec + residual add
// out[row] = dot(w_q8_row, x) + residual[row]
// Same 2-SG independent ownership pattern, with fused residual addition.
//
// Dispatch: threadgroups = ceil(out_dim/8), threads_per_threadgroup = 64
// Threadgroup memory: ZERO
// ============================================================================

kernel void dequant_matmul_q8_0_2sg_residual(
    device const uchar* weights   [[buffer(0)]],
    device const float* x         [[buffer(1)]],
    device float*       out       [[buffer(2)]],
    constant uint&      in_dim    [[buffer(3)]],
    device const float* residual  [[buffer(4)]],
    constant uint&      out_dim   [[buffer(5)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint NR = 4;
    const uint NQ = 8;
    const uint NW = 32;
    const uint Q8_BLOCK_SIZE = 34;

    const uint nb = in_dim >> 5;
    const uint row_bytes = nb * Q8_BLOCK_SIZE;

    const uint r0 = tgpig * 8 + sgitg * NR;

    const uint ix = tiisg / (NW / NQ);
    const uint il = tiisg % (NW / NQ);

    const uint ib0 = ix;

    device const uchar* ax[NR];
    for (uint row = 0; row < NR; ++row) {
        ax[row] = weights + (r0 + row) * row_bytes;
    }

    float sumf[NR] = { 0.f, 0.f, 0.f, 0.f };

    float yl[NQ];

    device const float* yb = x + ib0 * 32 + il * NQ;

    for (uint ib = ib0; ib < nb; ib += NQ) {
        #pragma clang loop unroll(full)
        for (uint i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        for (uint row = 0; row < NR; ++row) {
            if (r0 + row >= out_dim) break;

            device const uchar* bp = ax[row] + ib * Q8_BLOCK_SIZE;
            float scale = float(as_type<half>(*(device const ushort*)bp));
            device const char* qs = (device const char*)(bp + 2) + il * NQ;

            float sumq = 0.f;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < NQ; ++i) {
                sumq += float(qs[i]) * yl[i];
            }
            sumf[row] += sumq * scale;
        }

        yb += NQ * 32;
    }

    // Final reduction: just simd_sum + residual add, NO cross-SG reduction
    for (uint row = 0; row < NR && r0 + row < out_dim; ++row) {
        sumf[row] = simd_sum(sumf[row]);
        if (tiisg == 0) {
            out[r0 + row] = sumf[row] + residual[r0 + row];
        }
    }
}

// ============================================================================
// ffn_fused_gate_up_swiglu_q8_0_2sg: MLX-style 2-SG fused Gate+Up+SwiGLU
//
// Same 2-SG independent ownership but fuses gate+up+SwiGLU:
// Each SG owns 4 output rows. For each row, compute both gate and up dot
// products, then apply SwiGLU = silu(gate) * up.
//
// 64 threads (2 SGs), 8 output rows per TG.
// Zero threadgroup barriers. Zero shared memory for reduction.
//
// Dispatch: threadgroups = ceil(inter_dim/8), threads_per_threadgroup = 64
// ============================================================================

kernel void ffn_fused_gate_up_swiglu_q8_0_2sg(
    device const uchar* w_gate_q8   [[buffer(0)]],   // gate weights Q8_0 [inter_dim, hidden_dim]
    device const float* x           [[buffer(1)]],   // normed input [hidden_dim]
    device float*       out         [[buffer(2)]],   // output [inter_dim] (SwiGLU result)
    constant uint&      in_dim      [[buffer(3)]],   // hidden_dim
    constant uint&      out_dim     [[buffer(4)]],   // inter_dim
    device const uchar* w_up_q8     [[buffer(5)]],   // up weights Q8_0 [inter_dim, hidden_dim]
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint NR = 4;          // rows per simdgroup
    const uint NQ = 8;          // elements per thread per iteration
    const uint NW = 32;         // SIMD width
    const uint Q8_BLOCK_SIZE = 34;

    const uint nb = in_dim >> 5;   // number of Q8_0 blocks per row
    const uint row_bytes = nb * Q8_BLOCK_SIZE;

    const uint r0 = tgpig * 8 + sgitg * NR;

    const uint ix = tiisg / (NW / NQ);
    const uint il = tiisg % (NW / NQ);

    const uint ib0 = ix;

    float gate_sum[NR] = { 0.f, 0.f, 0.f, 0.f };
    float up_sum[NR]   = { 0.f, 0.f, 0.f, 0.f };

    float yl[NQ];

    device const float* yb = x + ib0 * 32 + il * NQ;

    for (uint ib = ib0; ib < nb; ib += NQ) {
        // Load 8 x-values into registers (shared between gate and up)
        #pragma clang loop unroll(full)
        for (uint i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        for (uint row = 0; row < NR; ++row) {
            if (r0 + row >= out_dim) break;

            uint global_row = r0 + row;

            // Gate weights
            device const uchar* gate_bp = w_gate_q8 + global_row * row_bytes + ib * Q8_BLOCK_SIZE;
            float gate_scale = float(as_type<half>(*(device const ushort*)gate_bp));
            device const char* gate_qs = (device const char*)(gate_bp + 2) + il * NQ;

            // Up weights
            device const uchar* up_bp = w_up_q8 + global_row * row_bytes + ib * Q8_BLOCK_SIZE;
            float up_scale = float(as_type<half>(*(device const ushort*)up_bp));
            device const char* up_qs = (device const char*)(up_bp + 2) + il * NQ;

            float gate_dot = 0.f;
            float up_dot   = 0.f;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < NQ; ++i) {
                float xi = yl[i];
                gate_dot += float(gate_qs[i]) * xi;
                up_dot   += float(up_qs[i])   * xi;
            }
            gate_sum[row] += gate_dot * gate_scale;
            up_sum[row]   += up_dot   * up_scale;
        }

        yb += NQ * 32;
    }

    // Final reduction: simd_sum + SwiGLU, NO cross-SG reduction
    for (uint row = 0; row < NR && r0 + row < out_dim; ++row) {
        float g = simd_sum(gate_sum[row]);
        float u = simd_sum(up_sum[row]);
        if (tiisg == 0) {
            float sigmoid = 1.0f / (1.0f + exp(-g));
            out[r0 + row] = g * sigmoid * u;
        }
    }
}

// ============================================================================
// dequant_matmul_q8_0_8row: 8-row Q8_0 matmul (8 rows per threadgroup)
//
// Processes 8 output rows per threadgroup using 8 SIMD groups (256 threads).
// Maximal x-vector reuse: 8x less x-bandwidth vs single-row dispatch.
//
// Dispatch: threadgroups = ceil(out_dim/8), threads_per_threadgroup = 256
// SIMD group 0..7 -> rows 8*gid+0 .. 8*gid+7
// ============================================================================

kernel void dequant_matmul_q8_0_8row(
    device const uchar* w_q8    [[buffer(0)]],
    device const float* x       [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant uint&      in_dim  [[buffer(3)]],
    constant uint&      out_dim [[buffer(4)]],
    uint row_group              [[threadgroup_position_in_grid]],
    uint lane                   [[thread_index_in_simdgroup]],
    uint sg                     [[simdgroup_index_in_threadgroup]])
{
    uint row = row_group * 8 + sg;
    if (row >= out_dim) return;

    const uint Q8_BLOCK_SIZE = 34;
    uint num_blocks = in_dim >> 5;
    uint row_bytes = num_blocks * Q8_BLOCK_SIZE;
    device const uchar* row_ptr = w_q8 + row * row_bytes;

    float sum = 0.0f;

    uint b = 0;
    for (; b + 3 < num_blocks; b += 4) {
        device const uchar* bp0 = row_ptr + b * Q8_BLOCK_SIZE;
        device const uchar* bp1 = bp0 + Q8_BLOCK_SIZE;
        device const uchar* bp2 = bp1 + Q8_BLOCK_SIZE;
        device const uchar* bp3 = bp2 + Q8_BLOCK_SIZE;
        half s0 = as_type<half>(*(device const ushort*)bp0);
        half s1 = as_type<half>(*(device const ushort*)bp1);
        half s2 = as_type<half>(*(device const ushort*)bp2);
        half s3 = as_type<half>(*(device const ushort*)bp3);
        float v0 = float(((device const char*)(bp0 + 2))[lane]) * x[(b << 5) + lane];
        float v1 = float(((device const char*)(bp1 + 2))[lane]) * x[((b+1) << 5) + lane];
        float v2 = float(((device const char*)(bp2 + 2))[lane]) * x[((b+2) << 5) + lane];
        float v3 = float(((device const char*)(bp3 + 2))[lane]) * x[((b+3) << 5) + lane];
        sum += float(s0) * simd_sum(v0) + float(s1) * simd_sum(v1)
             + float(s2) * simd_sum(v2) + float(s3) * simd_sum(v3);
    }
    for (; b < num_blocks; b++) {
        device const uchar* bp = row_ptr + b * Q8_BLOCK_SIZE;
        half scale = as_type<half>(*(device const ushort*)bp);
        float val = float(((device const char*)(bp + 2))[lane]) * x[(b << 5) + lane];
        sum += float(scale) * simd_sum(val);
    }

    if (lane == 0) {
        out[row] = sum;
    }
}

// ============================================================================
// dequant_matmul_q8_0_residual_8row: 8-row Q8_0 matmul + residual add
// out[row] = dot(w_q8_row, x) + residual[row]
// Same 8-row strategy: 8 rows per threadgroup, 8 SIMD groups (256 threads).
// ============================================================================

kernel void dequant_matmul_q8_0_residual_8row(
    device const uchar* w_q8      [[buffer(0)]],
    device const float* x         [[buffer(1)]],
    device float*       out       [[buffer(2)]],
    constant uint&      in_dim    [[buffer(3)]],
    device const float* residual  [[buffer(4)]],
    constant uint&      out_dim   [[buffer(5)]],
    uint row_group                [[threadgroup_position_in_grid]],
    uint lane                     [[thread_index_in_simdgroup]],
    uint sg                       [[simdgroup_index_in_threadgroup]])
{
    uint row = row_group * 8 + sg;
    if (row >= out_dim) return;

    const uint Q8_BLOCK_SIZE = 34;
    uint num_blocks = in_dim >> 5;
    device const uchar* row_ptr = w_q8 + row * num_blocks * Q8_BLOCK_SIZE;

    float sum = 0.0f;
    uint b = 0;
    for (; b + 3 < num_blocks; b += 4) {
        device const uchar* bp0 = row_ptr + b * Q8_BLOCK_SIZE;
        device const uchar* bp1 = bp0 + Q8_BLOCK_SIZE;
        device const uchar* bp2 = bp1 + Q8_BLOCK_SIZE;
        device const uchar* bp3 = bp2 + Q8_BLOCK_SIZE;
        half s0 = as_type<half>(*(device const ushort*)bp0);
        half s1 = as_type<half>(*(device const ushort*)bp1);
        half s2 = as_type<half>(*(device const ushort*)bp2);
        half s3 = as_type<half>(*(device const ushort*)bp3);
        float v0 = float(((device const char*)(bp0 + 2))[lane]) * x[(b << 5) + lane];
        float v1 = float(((device const char*)(bp1 + 2))[lane]) * x[((b+1) << 5) + lane];
        float v2 = float(((device const char*)(bp2 + 2))[lane]) * x[((b+2) << 5) + lane];
        float v3 = float(((device const char*)(bp3 + 2))[lane]) * x[((b+3) << 5) + lane];
        sum += float(s0) * simd_sum(v0) + float(s1) * simd_sum(v1)
             + float(s2) * simd_sum(v2) + float(s3) * simd_sum(v3);
    }
    for (; b < num_blocks; b++) {
        device const uchar* bp = row_ptr + b * Q8_BLOCK_SIZE;
        half scale = as_type<half>(*(device const ushort*)bp);
        float val = float(((device const char*)(bp + 2))[lane]) * x[(b << 5) + lane];
        sum += float(scale) * simd_sum(val);
    }

    if (lane == 0) {
        out[row] = sum + residual[row];
    }
}

// ============================================================================
// rope: Rotary Position Embeddings
//
// Applies rotation to interleaved (even, odd) pairs:
//   new_even = even * cos - odd * sin
//   new_odd  = even * sin + odd * cos
// ============================================================================

kernel void rope(
    device float*       vec          [[buffer(0)]],
    device const float* cos_table    [[buffer(1)]],
    device const float* sin_table    [[buffer(2)]],
    constant uint&      num_heads    [[buffer(3)]],
    constant uint&      head_dim     [[buffer(4)]],
    constant uint&      half_dim     [[buffer(5)]],
    constant uint&      pos_offset   [[buffer(6)]],
    uint gid                         [[thread_position_in_grid]])
{
    // gid ranges over total_half_elements = num_heads * half_dim
    uint head = gid / half_dim;
    uint i    = gid % half_dim;

    if (head >= num_heads) return;

    uint idx0 = head * head_dim + 2 * i;
    uint idx1 = idx0 + 1;

    float v0 = vec[idx0];
    float v1 = vec[idx1];

    uint cos_idx = pos_offset + i;
    float c = cos_table[cos_idx];
    float s = sin_table[cos_idx];

    vec[idx0] = v0 * c - v1 * s;
    vec[idx1] = v0 * s + v1 * c;
}

// ============================================================================
// rope_neox: NeoX-style Rotary Position Embeddings
//
// Pairs dimensions as (i, i + half_dim) instead of (2*i, 2*i+1):
//   x0 = vec[head*head_dim + i]
//   x1 = vec[head*head_dim + i + half_dim]
//   new_x0 = x0 * cos - x1 * sin
//   new_x1 = x0 * sin + x1 * cos
//
// Used by Qwen3.5 (IMROPE) and other NeoX-family models.
// ============================================================================

kernel void rope_neox(
    device float*       vec          [[buffer(0)]],
    device const float* cos_table    [[buffer(1)]],
    device const float* sin_table    [[buffer(2)]],
    constant uint&      num_heads    [[buffer(3)]],
    constant uint&      head_dim     [[buffer(4)]],
    constant uint&      half_dim     [[buffer(5)]],
    constant uint&      pos_offset   [[buffer(6)]],
    uint gid                         [[thread_position_in_grid]])
{
    // gid ranges over total_half_elements = num_heads * half_dim
    uint head = gid / half_dim;
    uint i    = gid % half_dim;

    if (head >= num_heads) return;

    uint base = head * head_dim;
    uint idx0 = base + i;            // first half: [0, half_dim)
    uint idx1 = base + i + half_dim; // second half: [half_dim, 2*half_dim)

    float v0 = vec[idx0];
    float v1 = vec[idx1];

    uint cos_idx = pos_offset + i;
    float c = cos_table[cos_idx];
    float s = sin_table[cos_idx];

    vec[idx0] = v0 * c - v1 * s;
    vec[idx1] = v0 * s + v1 * c;
}

// ============================================================================
// swiglu: Fused SiLU(gate) * up
//
// gate[i] = silu(gate[i]) * up[i]
// silu(x) = x / (1 + exp(-x))
// ============================================================================

kernel void swiglu(
    device float*       gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    constant uint&      dim  [[buffer(2)]],
    uint gid                 [[thread_position_in_grid]])
{
    if (gid >= dim) return;
    float g = gate[gid];
    float sigmoid = 1.0f / (1.0f + exp(-g));
    gate[gid] = g * sigmoid * up[gid];
}

// ============================================================================
// softmax: Softmax with max-subtraction (best practice 4.1)
//
// Three-phase approach using threadgroup reduction:
// 1. Find max
// 2. Subtract max, exp, sum
// 3. Normalize
// ============================================================================

kernel void softmax(
    device float*  data   [[buffer(0)]],
    constant uint& len    [[buffer(1)]],
    uint tid              [[thread_index_in_threadgroup]],
    uint tg_size          [[threads_per_threadgroup]],
    uint simd_lane        [[thread_index_in_simdgroup]],
    uint simd_group       [[simdgroup_index_in_threadgroup]])
{
    // Phase 1: Find max
    float local_max = -INFINITY;
    for (uint i = tid; i < len; i += tg_size) {
        local_max = max(local_max, data[i]);
    }

    local_max = simd_max(local_max);

    threadgroup float partial_max[32];

    if (simd_lane == 0) {
        partial_max[simd_group] = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simd_groups = (tg_size + 31) / 32;
    threadgroup float global_max;
    if (simd_group == 0) {
        float val = (simd_lane < num_simd_groups) ? partial_max[simd_lane] : -INFINITY;
        val = simd_max(val);
        if (simd_lane == 0) {
            global_max = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: exp(x - max) and sum
    float local_sum = 0.0f;
    for (uint i = tid; i < len; i += tg_size) {
        float e = exp(data[i] - global_max);
        data[i] = e;
        local_sum += e;
    }

    local_sum = simd_sum(local_sum);

    threadgroup float partial_sums[32];
    if (simd_lane == 0) {
        partial_sums[simd_group] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float global_sum;
    if (simd_group == 0) {
        float val = (simd_lane < num_simd_groups) ? partial_sums[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            global_sum = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Normalize
    float inv_sum = 1.0f / global_sum;
    for (uint i = tid; i < len; i += tg_size) {
        data[i] *= inv_sum;
    }
}

// ============================================================================
// attention_scores: Compute Q . K^T dot products for a single head
//
// q:       [head_dim]
// k_cache: [seq_len * kv_dim] -- all KV heads flattened
// scores:  [seq_len] output
// ============================================================================

kernel void attention_scores(
    device const float* q          [[buffer(0)]],
    device const half*  k_cache    [[buffer(1)]],
    device float*       scores     [[buffer(2)]],
    constant uint&      head_dim   [[buffer(3)]],
    constant uint&      kv_dim     [[buffer(4)]],
    constant uint&      kv_head    [[buffer(5)]],
    constant float&     scale      [[buffer(6)]],
    constant uint&      seq_len    [[buffer(7)]],
    uint t                         [[threadgroup_position_in_grid]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_group                [[simdgroup_index_in_threadgroup]])
{
    if (t >= seq_len) return;

    device const half* k_vec = k_cache + t * kv_dim + kv_head * head_dim;

    float dot = 0.0f;
    for (uint d = tid; d < head_dim; d += tg_size) {
        dot += q[d] * float(k_vec[d]);
    }

    dot = simd_sum(dot);

    threadgroup float partial_sums[32];

    if (simd_lane == 0) {
        partial_sums[simd_group] = dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simd_groups = (tg_size + 31) / 32;
    if (simd_group == 0) {
        float val = (simd_lane < num_simd_groups) ? partial_sums[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            scores[t] = val * scale;
        }
    }
}

// ============================================================================
// attention_output: Weighted sum of V vectors
//
// scores:  [seq_len] -- softmax attention weights
// v_cache: [seq_len * kv_dim] -- all KV heads flattened
// out:     [head_dim] -- output for one head
// ============================================================================

kernel void attention_output(
    device const float* scores     [[buffer(0)]],
    device const half*  v_cache    [[buffer(1)]],
    device float*       out        [[buffer(2)]],
    constant uint&      head_dim   [[buffer(3)]],
    constant uint&      kv_dim     [[buffer(4)]],
    constant uint&      kv_head    [[buffer(5)]],
    constant uint&      seq_len    [[buffer(6)]],
    constant uint&      max_seq_len [[buffer(7)]],
    uint d                         [[thread_position_in_grid]])
{
    if (d >= head_dim) return;

    // V cache transposed: [kv_dim, max_seq_len], contiguous along time (f16)
    device const half* v_row = v_cache + (kv_head * head_dim + d) * max_seq_len;
    float sum = 0.0f;
    for (uint t = 0; t < seq_len; t++) {
        sum += scores[t] * float(v_row[t]);
    }
    out[d] = sum;
}

// ============================================================================
// write_kv_cache: Write K,V projections directly into GPU-resident KV cache
//
// Copies kv_dim floats from k_new/v_new into the correct position in the
// persistent GPU KV cache buffers. Avoids CPU round-trip entirely.
// ============================================================================

kernel void write_kv_cache(
    device const float* k_new       [[buffer(0)]],
    device const float* v_new       [[buffer(1)]],
    device half*        k_cache     [[buffer(2)]],
    device half*        v_cache     [[buffer(3)]],
    constant uint&      kv_dim      [[buffer(4)]],
    constant uint&      seq_pos     [[buffer(5)]],
    constant uint&      max_seq_len [[buffer(6)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid >= kv_dim) return;
    // K cache: row-major [max_seq_len, kv_dim]
    k_cache[seq_pos * kv_dim + gid] = half(k_new[gid]);
    // V cache: transposed [kv_dim, max_seq_len] for contiguous attention reads
    v_cache[gid * max_seq_len + seq_pos] = half(v_new[gid]);
}

// ============================================================================
// fused_rope_kv_write: Fused RoPE Q + RoPE K + KV cache write
//
// Combines 3 separate dispatches into 1 for decode:
//   Region 0: Apply standard RoPE to Q vectors (interleaved pairs)
//   Region 1: Apply standard RoPE to K vectors + write K to cache
//   Region 2: Write V to cache (no RoPE needed)
//
// Grid is flat: total_threads = num_q_heads*half_dim + num_kv_heads*half_dim + kv_dim
// The kernel partitions gid space into 3 contiguous regions using offsets.
// All 3 regions operate on non-overlapping buffers, so no barriers are needed.
//
// Saves 2 dispatches per layer (64 dispatches for 32-layer models).
// ============================================================================

kernel void fused_rope_kv_write(
    device float*       q_vec        [[buffer(0)]],   // qkv_buf + q_byte_off
    device float*       k_vec        [[buffer(1)]],   // qkv_buf + k_byte_off
    device const float* v_vec        [[buffer(2)]],   // qkv_buf + v_byte_off
    device const float* cos_table    [[buffer(3)]],
    device const float* sin_table    [[buffer(4)]],
    device half*        k_cache      [[buffer(5)]],
    device half*        v_cache      [[buffer(6)]],
    constant uint&      num_q_heads  [[buffer(7)]],
    constant uint&      num_kv_heads [[buffer(8)]],
    constant uint&      head_dim     [[buffer(9)]],
    constant uint&      half_dim     [[buffer(10)]],
    constant uint&      pos_offset   [[buffer(11)]],
    constant uint&      kv_dim       [[buffer(12)]],
    constant uint&      seq_pos      [[buffer(13)]],
    constant uint&      max_seq_len  [[buffer(14)]],
    uint gid                         [[thread_position_in_grid]])
{
    // Region boundaries
    uint q_region_size  = num_q_heads * half_dim;   // Q RoPE region
    uint k_region_size  = num_kv_heads * half_dim;  // K RoPE region

    if (gid < q_region_size) {
        // --- Region 0: Q RoPE (standard interleaved) ---
        uint head = gid / half_dim;
        uint i    = gid % half_dim;
        if (head >= num_q_heads) return;

        uint idx0 = head * head_dim + 2 * i;
        uint idx1 = idx0 + 1;

        float v0 = q_vec[idx0];
        float v1 = q_vec[idx1];

        uint cos_idx = pos_offset + i;
        float c = cos_table[cos_idx];
        float s = sin_table[cos_idx];

        q_vec[idx0] = v0 * c - v1 * s;
        q_vec[idx1] = v0 * s + v1 * c;
    }
    else if (gid < q_region_size + k_region_size) {
        // --- Region 1: K RoPE (standard interleaved) + K cache write ---
        uint local_gid = gid - q_region_size;
        uint head = local_gid / half_dim;
        uint i    = local_gid % half_dim;
        if (head >= num_kv_heads) return;

        uint idx0 = head * head_dim + 2 * i;
        uint idx1 = idx0 + 1;

        float v0 = k_vec[idx0];
        float v1 = k_vec[idx1];

        uint cos_idx = pos_offset + i;
        float c = cos_table[cos_idx];
        float s = sin_table[cos_idx];

        float new0 = v0 * c - v1 * s;
        float new1 = v0 * s + v1 * c;

        k_vec[idx0] = new0;
        k_vec[idx1] = new1;

        // Write both rotated elements to K cache (row-major [max_seq_len, kv_dim]) as f16
        k_cache[seq_pos * kv_dim + idx0] = half(new0);
        k_cache[seq_pos * kv_dim + idx1] = half(new1);
    }
    else {
        // --- Region 2: V cache write (no RoPE) ---
        uint local_gid = gid - q_region_size - k_region_size;
        if (local_gid >= kv_dim) return;

        // V cache: transposed [kv_dim, max_seq_len] as f16
        v_cache[local_gid * max_seq_len + seq_pos] = half(v_vec[local_gid]);
    }
}

// ============================================================================
// fused_rope_neox_kv_write: NeoX-style variant of fused RoPE + KV write
//
// Same as fused_rope_kv_write but uses NeoX dimension pairing:
//   (i, i + half_dim) instead of (2*i, 2*i+1)
// ============================================================================

kernel void fused_rope_neox_kv_write(
    device float*       q_vec        [[buffer(0)]],
    device float*       k_vec        [[buffer(1)]],
    device const float* v_vec        [[buffer(2)]],
    device const float* cos_table    [[buffer(3)]],
    device const float* sin_table    [[buffer(4)]],
    device half*        k_cache      [[buffer(5)]],
    device half*        v_cache      [[buffer(6)]],
    constant uint&      num_q_heads  [[buffer(7)]],
    constant uint&      num_kv_heads [[buffer(8)]],
    constant uint&      head_dim     [[buffer(9)]],
    constant uint&      half_dim     [[buffer(10)]],
    constant uint&      pos_offset   [[buffer(11)]],
    constant uint&      kv_dim       [[buffer(12)]],
    constant uint&      seq_pos      [[buffer(13)]],
    constant uint&      max_seq_len  [[buffer(14)]],
    uint gid                         [[thread_position_in_grid]])
{
    uint q_region_size  = num_q_heads * half_dim;
    uint k_region_size  = num_kv_heads * half_dim;

    if (gid < q_region_size) {
        // --- Region 0: Q RoPE (NeoX half-offset) ---
        uint head = gid / half_dim;
        uint i    = gid % half_dim;
        if (head >= num_q_heads) return;

        uint base = head * head_dim;
        uint idx0 = base + i;
        uint idx1 = base + i + half_dim;

        float v0 = q_vec[idx0];
        float v1 = q_vec[idx1];

        uint cos_idx = pos_offset + i;
        float c = cos_table[cos_idx];
        float s = sin_table[cos_idx];

        q_vec[idx0] = v0 * c - v1 * s;
        q_vec[idx1] = v0 * s + v1 * c;
    }
    else if (gid < q_region_size + k_region_size) {
        // --- Region 1: K RoPE (NeoX half-offset) + K cache write ---
        uint local_gid = gid - q_region_size;
        uint head = local_gid / half_dim;
        uint i    = local_gid % half_dim;
        if (head >= num_kv_heads) return;

        uint base = head * head_dim;
        uint idx0 = base + i;
        uint idx1 = base + i + half_dim;

        float v0 = k_vec[idx0];
        float v1 = k_vec[idx1];

        uint cos_idx = pos_offset + i;
        float c = cos_table[cos_idx];
        float s = sin_table[cos_idx];

        float new0 = v0 * c - v1 * s;
        float new1 = v0 * s + v1 * c;

        k_vec[idx0] = new0;
        k_vec[idx1] = new1;

        // Write both rotated elements to K cache (row-major [max_seq_len, kv_dim]) as f16
        k_cache[seq_pos * kv_dim + idx0] = half(new0);
        k_cache[seq_pos * kv_dim + idx1] = half(new1);
    }
    else {
        // --- Region 2: V cache write (no RoPE) ---
        uint local_gid = gid - q_region_size - k_region_size;
        if (local_gid >= kv_dim) return;

        // V cache: transposed [kv_dim, max_seq_len] as f16
        v_cache[local_gid * max_seq_len + seq_pos] = half(v_vec[local_gid]);
    }
}

// ============================================================================
// multi_head_attention: Fused all-heads attention in a single dispatch
//
// Computes Q.K^T scores, softmax, and V weighted sum for ALL heads
// simultaneously. Each threadgroup handles one query head. Supports GQA
// via kv_head = head / gqa_ratio mapping.
//
// Dispatch: threadgroups = (num_heads, 1, 1), threads_per_threadgroup = (tg_size, 1, 1)
//
// The kernel performs three phases per head:
// 1. Dot product: score[t] = sum_d(q[d] * k_cache[t, kv_head, d]) * scale
// 2. Softmax: in-place over scores (max-subtract for numerical stability)
// 3. Output: out[d] = sum_t(score[t] * v_cache[t, kv_head, d])
//
// Uses device-memory scores buffer instead of threadgroup memory to avoid
// threadgroup size limits and compiler issues.
// ============================================================================

kernel void multi_head_attention(
    device const float* q              [[buffer(0)]],
    device const half*  k_cache        [[buffer(1)]],
    device const half*  v_cache        [[buffer(2)]],
    device float*       attn_out       [[buffer(3)]],
    device float*       scores_scratch [[buffer(4)]],
    constant uint&      num_heads      [[buffer(5)]],
    constant uint&      num_kv_heads   [[buffer(6)]],
    constant uint&      head_dim       [[buffer(7)]],
    constant uint&      kv_dim         [[buffer(8)]],
    constant uint&      seq_len        [[buffer(9)]],
    constant float&     scale          [[buffer(10)]],
    constant uint&      max_seq_len    [[buffer(11)]],
    uint head                          [[threadgroup_position_in_grid]],
    uint tid                           [[thread_index_in_threadgroup]],
    uint tg_size                       [[threads_per_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_group                    [[simdgroup_index_in_threadgroup]])
{
    if (head >= num_heads) return;

    // GQA mapping: multiple query heads share the same KV head
    uint gqa_ratio = num_heads / num_kv_heads;
    uint kv_head = head / gqa_ratio;

    // Pointers for this head's Q and output
    device const float* q_head = q + head * head_dim;
    device float* out_head = attn_out + head * head_dim;

    // Per-head scores stored in device memory (indexed by head * max_seq_capacity + t)
    // The host allocates scores_scratch as [num_heads * max_seq_len] floats.
    device float* scores = scores_scratch + head * seq_len;

    // ---- Phase 1: Compute attention scores (read K as half, accumulate in f32) ----
    for (uint t = tid; t < seq_len; t += tg_size) {
        device const half* k_vec = k_cache + t * kv_dim + kv_head * head_dim;
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dot += q_head[d] * float(k_vec[d]);
        }
        scores[t] = dot * scale;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // ---- Phase 2: Softmax over scores[0..seq_len] ----

    // 2a: Find max
    float local_max = -INFINITY;
    for (uint t = tid; t < seq_len; t += tg_size) {
        local_max = max(local_max, scores[t]);
    }
    local_max = simd_max(local_max);

    threadgroup float partial_max[32];
    if (simd_lane == 0) {
        partial_max[simd_group] = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simd_groups = (tg_size + 31) / 32;
    threadgroup float global_max;
    if (simd_group == 0) {
        float val = (simd_lane < num_simd_groups) ? partial_max[simd_lane] : -INFINITY;
        val = simd_max(val);
        if (simd_lane == 0) {
            global_max = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2b: exp(x - max) and sum
    float local_sum = 0.0f;
    for (uint t = tid; t < seq_len; t += tg_size) {
        float e = exp(scores[t] - global_max);
        scores[t] = e;
        local_sum += e;
    }
    local_sum = simd_sum(local_sum);

    threadgroup float partial_sums[32];
    if (simd_lane == 0) {
        partial_sums[simd_group] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float global_sum;
    if (simd_group == 0) {
        float val = (simd_lane < num_simd_groups) ? partial_sums[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            global_sum = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2c: Normalize
    float inv_sum = 1.0f / global_sum;
    for (uint t = tid; t < seq_len; t += tg_size) {
        scores[t] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // ---- Phase 3: Weighted sum of V (transposed layout: [kv_dim, max_seq_len], f16) ----
    for (uint d = tid; d < head_dim; d += tg_size) {
        // V is stored as [kv_dim, max_seq_len]: contiguous along time dimension (f16)
        device const half* v_row = v_cache + (kv_head * head_dim + d) * max_seq_len;
        float sum = 0.0f;
        for (uint t = 0; t < seq_len; t++) {
            sum += scores[t] * float(v_row[t]);
        }
        out_head[d] = sum;
    }
}

// ============================================================================
// fused_rope_kv_mha: Fused RoPE + KV cache write + Multi-Head Attention
//
// Combines fused_rope_kv_write + multi_head_attention into a single dispatch,
// eliminating 2 memory barriers and 1 dispatch per layer.
//
// Each threadgroup handles one Q head:
//   Phase 0: Apply RoPE to Q[head], RoPE to K[kv_head], write K/V to cache
//   Phase 1: Compute attention scores (Q dot K_cache for all positions)
//   Phase 2: Softmax over scores
//   Phase 3: Weighted sum of V_cache
//
// For GQA (num_q_heads > num_kv_heads), all Q heads in the same KV group
// redundantly write K/V to cache (benign race: identical values).
//
// Only used for short sequences (seq_len < FLASH_DECODE_THRESHOLD = 257).
// Dispatch: threadgroups = num_q_heads, threads = min(256, max(head_dim, seq_len+1))
// ============================================================================

kernel void fused_rope_kv_mha(
    device float*       q_vec        [[buffer(0)]],   // qkv_buf + q_byte_off [q_dim]
    device float*       k_vec        [[buffer(1)]],   // qkv_buf + k_byte_off [kv_dim]
    device const float* v_vec        [[buffer(2)]],   // qkv_buf + v_byte_off [kv_dim]
    device const float* cos_table    [[buffer(3)]],
    device const float* sin_table    [[buffer(4)]],
    device half*        k_cache      [[buffer(5)]],   // [max_seq_len, kv_dim] row-major
    device half*        v_cache      [[buffer(6)]],   // [kv_dim, max_seq_len] transposed
    device float*       attn_out     [[buffer(7)]],   // [q_dim]
    device float*       scores_scratch [[buffer(8)]], // [num_q_heads * (seq_pos+1)]
    constant uint&      num_q_heads  [[buffer(9)]],
    constant uint&      num_kv_heads [[buffer(10)]],
    constant uint&      head_dim     [[buffer(11)]],
    constant uint&      half_dim     [[buffer(12)]],  // head_dim / 2 (for RoPE)
    constant uint&      pos_offset   [[buffer(13)]],  // seq_pos * half_dim (cos/sin table offset)
    constant uint&      kv_dim       [[buffer(14)]],
    constant uint&      seq_pos      [[buffer(15)]],  // current position (0-indexed)
    constant float&     attn_scale   [[buffer(16)]],
    constant uint&      max_seq_len  [[buffer(17)]],
    uint head                        [[threadgroup_position_in_grid]],
    uint tid                         [[thread_index_in_threadgroup]],
    uint tg_size                     [[threads_per_threadgroup]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_group                  [[simdgroup_index_in_threadgroup]])
{
    if (head >= num_q_heads) return;

    uint gqa_ratio = num_q_heads / num_kv_heads;
    uint kv_head = head / gqa_ratio;
    uint new_seq_len = seq_pos + 1;

    // ---- Phase 0: RoPE + KV cache write ----

    // Apply RoPE to Q[head] (in-place)
    for (uint i = tid; i < half_dim; i += tg_size) {
        uint idx0 = head * head_dim + 2 * i;
        uint idx1 = idx0 + 1;
        float v0 = q_vec[idx0];
        float v1 = q_vec[idx1];
        uint cos_idx = pos_offset + i;
        float c = cos_table[cos_idx];
        float s = sin_table[cos_idx];
        q_vec[idx0] = v0 * c - v1 * s;
        q_vec[idx1] = v0 * s + v1 * c;
    }

    // Apply RoPE to K[kv_head] + write to K cache
    // All Q heads in the same KV group do this redundantly (benign race).
    for (uint i = tid; i < half_dim; i += tg_size) {
        uint idx0 = kv_head * head_dim + 2 * i;
        uint idx1 = idx0 + 1;
        float v0 = k_vec[idx0];
        float v1 = k_vec[idx1];
        uint cos_idx = pos_offset + i;
        float c = cos_table[cos_idx];
        float s = sin_table[cos_idx];
        float new0 = v0 * c - v1 * s;
        float new1 = v0 * s + v1 * c;
        k_vec[idx0] = new0;
        k_vec[idx1] = new1;
        k_cache[seq_pos * kv_dim + idx0] = half(new0);
        k_cache[seq_pos * kv_dim + idx1] = half(new1);
    }

    // Write V to cache (transposed layout: [kv_dim, max_seq_len])
    for (uint d = tid; d < head_dim; d += tg_size) {
        uint v_idx = kv_head * head_dim + d;
        v_cache[v_idx * max_seq_len + seq_pos] = half(v_vec[v_idx]);
    }

    // Ensure K/V cache writes are visible before reading in attention phase
    threadgroup_barrier(mem_flags::mem_device);

    // ---- Phase 1: Attention scores ----
    device const float* q_head_ptr = q_vec + head * head_dim;
    device float* out_head = attn_out + head * head_dim;
    device float* scores = scores_scratch + head * new_seq_len;

    for (uint t = tid; t < new_seq_len; t += tg_size) {
        device const half* k_t = k_cache + t * kv_dim + kv_head * head_dim;
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dot += q_head_ptr[d] * float(k_t[d]);
        }
        scores[t] = dot * attn_scale;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // ---- Phase 2: Softmax ----

    // 2a: Find max
    float local_max = -INFINITY;
    for (uint t = tid; t < new_seq_len; t += tg_size) {
        local_max = max(local_max, scores[t]);
    }
    local_max = simd_max(local_max);

    threadgroup float partial_max[32];
    if (simd_lane == 0) {
        partial_max[simd_group] = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simd_groups = (tg_size + 31) / 32;
    threadgroup float global_max_val;
    if (simd_group == 0) {
        float val = (simd_lane < num_simd_groups) ? partial_max[simd_lane] : -INFINITY;
        val = simd_max(val);
        if (simd_lane == 0) {
            global_max_val = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2b: exp(x - max) and sum
    float local_sum = 0.0f;
    for (uint t = tid; t < new_seq_len; t += tg_size) {
        float e = exp(scores[t] - global_max_val);
        scores[t] = e;
        local_sum += e;
    }
    local_sum = simd_sum(local_sum);

    threadgroup float partial_sums[32];
    if (simd_lane == 0) {
        partial_sums[simd_group] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float global_sum_val;
    if (simd_group == 0) {
        float val = (simd_lane < num_simd_groups) ? partial_sums[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            global_sum_val = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2c: Normalize
    float inv_sum = 1.0f / global_sum_val;
    for (uint t = tid; t < new_seq_len; t += tg_size) {
        scores[t] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // ---- Phase 3: Weighted sum of V ----
    for (uint d = tid; d < head_dim; d += tg_size) {
        device const half* v_row = v_cache + (kv_head * head_dim + d) * max_seq_len;
        float sum = 0.0f;
        for (uint t = 0; t < new_seq_len; t++) {
            sum += scores[t] * float(v_row[t]);
        }
        out_head[d] = sum;
    }
}


// ============================================================================
// Flash Decoding: Two-phase tiled attention for decode (single query)
//
// Phase 1 (flash_decode_attention):
//   Grid: (num_heads * num_kv_tiles, 1, 1) -- flattened 1D
//   Each threadgroup processes one (head, kv_tile) pair.
//   Uses online softmax: maintains running (max, sum, weighted_v).
//   Writes partial results to an intermediate buffer.
//
// Phase 2 (flash_decode_reduce):
//   Grid: (num_heads, 1, 1)
//   Merges partial results across all KV tiles for each head using
//   log-sum-exp combination to produce final attention output.
//
// This provides parallelism across the KV sequence dimension, which is
// critical for long sequences where a single threadgroup per head
// cannot saturate the GPU.
// ============================================================================

// Partial result layout per (head, tile):
//   [head_dim floats of weighted_v] [1 float max] [1 float sum]
// Total stride per entry = head_dim + 2

kernel void flash_decode_attention(
    device const float* q              [[buffer(0)]],
    device const half*  k_cache        [[buffer(1)]],
    device const half*  v_cache        [[buffer(2)]],
    device float*       partial_out    [[buffer(3)]],
    constant uint&      num_heads      [[buffer(4)]],
    constant uint&      num_kv_heads   [[buffer(5)]],
    constant uint&      head_dim       [[buffer(6)]],
    constant uint&      kv_dim         [[buffer(7)]],
    constant uint&      seq_len        [[buffer(8)]],
    constant float&     scale          [[buffer(9)]],
    constant uint&      tile_kv        [[buffer(10)]],
    constant uint&      num_tiles      [[buffer(11)]],
    constant uint&      max_seq_len    [[buffer(12)]],
    uint tg_flat                       [[threadgroup_position_in_grid]],
    uint tid                           [[thread_index_in_threadgroup]],
    uint tg_size                       [[threads_per_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_group                    [[simdgroup_index_in_threadgroup]])
{
    // Flatten 2D grid: tg_flat = head * num_tiles + tile_idx
    uint head = tg_flat / num_tiles;
    uint tile_idx = tg_flat % num_tiles;
    if (head >= num_heads || tile_idx >= num_tiles) return;

    // GQA mapping
    uint gqa_ratio = num_heads / num_kv_heads;
    uint kv_head = head / gqa_ratio;

    // KV range for this tile
    uint t_start = tile_idx * tile_kv;
    uint t_end = min(t_start + tile_kv, seq_len);
    if (t_start >= seq_len) {
        // This tile is entirely beyond seq_len; write sentinel values
        uint partial_stride = head_dim + 2;
        uint partial_base = (head * num_tiles + tile_idx) * partial_stride;
        for (uint d = tid; d < head_dim; d += tg_size) {
            partial_out[partial_base + d] = 0.0f;
        }
        if (tid == 0) {
            partial_out[partial_base + head_dim]     = -INFINITY;  // max
            partial_out[partial_base + head_dim + 1] = 0.0f;       // sum
        }
        return;
    }

    // Pointer to this head's query
    device const float* q_head = q + head * head_dim;

    // Phase 1: Compute Q*K scores for this tile and online softmax

    // Step 1a: Compute all scores and find local max
    // Use threadgroup memory for scores within this tile
    threadgroup float tile_scores[256];  // max tile_kv; we cap tile_kv at 256

    // Each thread computes dot products for positions it owns (read K as half, accumulate in f32)
    float local_max = -INFINITY;
    uint tile_len = t_end - t_start;
    for (uint i = tid; i < tile_len; i += tg_size) {
        uint t = t_start + i;
        device const half* k_vec = k_cache + t * kv_dim + kv_head * head_dim;
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dot += q_head[d] * float(k_vec[d]);
        }
        float s = dot * scale;
        tile_scores[i] = s;
        local_max = max(local_max, s);
    }

    // Reduce max across threadgroup
    local_max = simd_max(local_max);
    threadgroup float partial_max[32];
    if (simd_lane == 0) {
        partial_max[simd_group] = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simd_groups = (tg_size + 31) / 32;
    threadgroup float global_max_val;
    if (simd_group == 0) {
        float val = (simd_lane < num_simd_groups) ? partial_max[simd_lane] : -INFINITY;
        val = simd_max(val);
        if (simd_lane == 0) global_max_val = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 1b: exp(score - max) and sum
    float local_sum = 0.0f;
    for (uint i = tid; i < tile_len; i += tg_size) {
        float e = exp(tile_scores[i] - global_max_val);
        tile_scores[i] = e;
        local_sum += e;
    }

    local_sum = simd_sum(local_sum);
    threadgroup float partial_sums[32];
    if (simd_lane == 0) {
        partial_sums[simd_group] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float global_sum_val;
    if (simd_group == 0) {
        float val = (simd_lane < num_simd_groups) ? partial_sums[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) global_sum_val = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 1c: Weighted sum of V for this tile (unnormalized -- store raw exp-weighted sum)
    uint partial_stride = head_dim + 2;
    uint partial_base = (head * num_tiles + tile_idx) * partial_stride;

    for (uint d = tid; d < head_dim; d += tg_size) {
        // V cache transposed: [kv_dim, max_seq_len], contiguous along time (f16)
        device const half* v_row = v_cache + (kv_head * head_dim + d) * max_seq_len + t_start;
        float wsum = 0.0f;
        for (uint i = 0; i < tile_len; i++) {
            wsum += tile_scores[i] * float(v_row[i]);
        }
        partial_out[partial_base + d] = wsum;
    }

    // Store max and sum for this tile
    if (tid == 0) {
        partial_out[partial_base + head_dim]     = global_max_val;
        partial_out[partial_base + head_dim + 1] = global_sum_val;
    }
}

// Phase 2: Reduce partial results across KV tiles
// Parallel reduction -- all threads participate in finding global max
// and computing rescaled sum, eliminating the serial thread-0 bottleneck.
// Tile rescale factors are pre-computed into threadgroup memory so Phase 3
// reads from fast SRAM instead of recomputing exp() from device memory.
kernel void flash_decode_reduce(
    device const float* partial_in     [[buffer(0)]],  // [num_heads * num_tiles * (head_dim+2)]
    device float*       attn_out       [[buffer(1)]],  // [num_heads * head_dim]
    constant uint&      num_heads      [[buffer(2)]],
    constant uint&      head_dim       [[buffer(3)]],
    constant uint&      num_tiles      [[buffer(4)]],
    uint head                          [[threadgroup_position_in_grid]],
    uint tid                           [[thread_index_in_threadgroup]],
    uint tg_size                       [[threads_per_threadgroup]],
    uint tiisg                         [[thread_index_in_simdgroup]],
    uint sgitg                         [[simdgroup_index_in_threadgroup]])
{
    if (head >= num_heads) return;

    uint partial_stride = head_dim + 2;
    uint num_simd_groups = (tg_size + 31) / 32;

    // --- Phase 1: Parallel global max across all tiles ---
    // Each thread reads tiles at stride tg_size, then SIMD + cross-SG reduction
    float local_max = -INFINITY;
    for (uint t = tid; t < num_tiles; t += tg_size) {
        uint base = (head * num_tiles + t) * partial_stride;
        local_max = max(local_max, partial_in[base + head_dim]);
    }
    // Intra-simdgroup max
    local_max = simd_max(local_max);

    // Cross-simdgroup max via threadgroup memory
    threadgroup float sg_max[8];  // up to 8 simdgroups (256 threads / 32)
    if (tiisg == 0) sg_max[sgitg] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float global_max_shared;
    if (sgitg == 0) {
        float v = (tiisg < num_simd_groups) ? sg_max[tiisg] : -INFINITY;
        v = simd_max(v);
        if (tiisg == 0) global_max_shared = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gmax = global_max_shared;

    // --- Phase 2: Parallel rescale + global sum ---
    // Each thread computes rescale factors for its tiles, stores in threadgroup mem
    threadgroup float tile_rescale[1024];  // max tiles supported
    float local_sum = 0.0f;
    for (uint t = tid; t < num_tiles; t += tg_size) {
        uint base = (head * num_tiles + t) * partial_stride;
        float tile_max = partial_in[base + head_dim];
        float tile_sum = partial_in[base + head_dim + 1];
        float rescale = (tile_sum > 0.0f) ? exp(tile_max - gmax) : 0.0f;
        tile_rescale[t] = rescale;
        local_sum += rescale * tile_sum;
    }
    // Intra-simdgroup sum
    local_sum = simd_sum(local_sum);

    // Cross-simdgroup sum via threadgroup memory
    threadgroup float sg_sum[8];
    if (tiisg == 0) sg_sum[sgitg] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float global_sum_shared;
    if (sgitg == 0) {
        float v = (tiisg < num_simd_groups) ? sg_sum[tiisg] : 0.0f;
        v = simd_sum(v);
        if (tiisg == 0) global_sum_shared = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_sum = (global_sum_shared > 0.0f) ? (1.0f / global_sum_shared) : 0.0f;

    // --- Phase 3: Combine weighted V across tiles ---
    // tile_rescale is in threadgroup memory (fast SRAM), no redundant exp() needed
    device float* out_head = attn_out + head * head_dim;

    for (uint d = tid; d < head_dim; d += tg_size) {
        float combined = 0.0f;
        for (uint t = 0; t < num_tiles; t++) {
            uint base = (head * num_tiles + t) * partial_stride;
            float tile_v = partial_in[base + d];
            combined += tile_v * tile_rescale[t];
        }
        out_head[d] = combined * inv_sum;
    }
}

// ============================================================================
// add_residual: Element-wise vector addition dst[i] += src[i]
// ============================================================================

kernel void add_residual(
    device float*       dst [[buffer(0)]],
    device const float* src [[buffer(1)]],
    constant uint&      dim [[buffer(2)]],
    uint gid                [[thread_position_in_grid]])
{
    if (gid >= dim) return;
    dst[gid] += src[gid];
}

// ============================================================================
// embed_token: Token embedding lookup
//
// Copies embedding[token_id * hidden_dim .. (token_id+1) * hidden_dim]
// to the output buffer.
// ============================================================================

kernel void embed_token(
    device const float* embedding  [[buffer(0)]],
    device float*       out        [[buffer(1)]],
    constant uint&      token_id   [[buffer(2)]],
    constant uint&      hidden_dim [[buffer(3)]],
    uint gid                       [[thread_position_in_grid]])
{
    if (gid >= hidden_dim) return;
    out[gid] = embedding[token_id * hidden_dim + gid];
}

// ============================================================================
// embed_token_f16: F16 (half-precision) token embedding lookup (decode)
//
// Reads half-precision embedding table, outputs F32.
// ============================================================================

kernel void embed_token_f16(
    device const half* embedding_f16 [[buffer(0)]],
    device float*      out           [[buffer(1)]],
    constant uint&     token_id      [[buffer(2)]],
    constant uint&     hidden_dim    [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= hidden_dim) return;
    out[gid] = float(embedding_f16[token_id * hidden_dim + gid]);
}


// ============================================================================
// embed_token_q8_0: Q8_0 quantized token embedding lookup (decode)
//
// Q8_0 block: 34 bytes = 2-byte f16 scale + 32 int8 values
// Dequantizes on-the-fly: out[gid] = scale * (int8)quant[elem]
// ============================================================================

kernel void embed_token_q8_0(
    device const char* embedding_q8  [[buffer(0)]],
    device float*      out           [[buffer(1)]],
    constant uint&     token_id      [[buffer(2)]],
    constant uint&     hidden_dim    [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= hidden_dim) return;
    uint global_elem = token_id * hidden_dim + gid;
    uint block_idx = global_elem >> 5;       // / 32
    uint elem_in_block = global_elem & 31;   // % 32

    device const char* block_ptr = embedding_q8 + block_idx * 34;
    float scale = float(*(device const half*)block_ptr);
    float val = float(block_ptr[2 + elem_in_block]);
    out[gid] = val * scale;
}

// ============================================================================
// embed_token_q4_0: Q4_0 quantized token embedding lookup (decode)
//
// Q4_0 block: 18 bytes = 2-byte f16 scale + 16 bytes (32 packed 4-bit values)
// GGML de-interleaved nibble ordering:
//   elements 0-15  = lo nibble of bytes 0-15
//   elements 16-31 = hi nibble of bytes 0-15
// Values are unsigned 0-15, subtract 8 for signed range [-8, 7]
// ============================================================================

kernel void embed_token_q4_0(
    device const char* embedding_q4  [[buffer(0)]],
    device float*      out           [[buffer(1)]],
    constant uint&     token_id      [[buffer(2)]],
    constant uint&     hidden_dim    [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= hidden_dim) return;
    uint global_elem = token_id * hidden_dim + gid;
    uint block_idx = global_elem >> 5;
    uint elem_in_block = global_elem & 31;

    device const uchar* block_ptr = (device const uchar*)(embedding_q4 + block_idx * 18);
    float scale = float(*(device const half*)block_ptr);

    // De-interleaved: elem 0-15 use lo nibble, elem 16-31 use hi nibble
    uint byte_idx = (elem_in_block < 16) ? elem_in_block : (elem_in_block - 16);
    uchar packed = block_ptr[2 + byte_idx];
    int nibble = (elem_in_block < 16) ? (int)(packed & 0xF) - 8 : (int)(packed >> 4) - 8;
    out[gid] = float(nibble) * scale;
}

// ============================================================================
// BATCHED PREFILL KERNELS
//
// These kernels process multiple tokens at once (mat-mat GEMM) for prompt
// processing. Dramatically more efficient than dispatching mat-vec per token
// because:
//   1. Weights are loaded once from memory and reused across all batch tokens
//   2. Threadgroup memory tiling achieves high ALU utilization
//   3. Orders of magnitude fewer GPU dispatches
// ============================================================================

// ============================================================================
// tiled_matmul_f32: Tiled matrix-matrix multiply using threadgroup memory
//
// C[M,N] = A[M,K] * B^T[N,K]    (B stored row-major as [N,K])
//
// For prefill: M = batch_size, K = in_dim, N = out_dim
// A = x_batch (token activations), B = weights (each row is one output dim)
//
// Tiling strategy optimized for Apple Silicon M-series:
// - TILE_M = 32, TILE_N = 32 (matches SIMD width of 32)
// - Threadgroup: 16x16 threads, each computes a 2x2 sub-tile of C
// - K dimension tiled in chunks of TILE_K = 16
// - Threadgroup memory: 2 tiles of [32 x 16] = 2 x 2048 = 4096 bytes
//   Well within the 32KB threadgroup memory limit
// ============================================================================

// ============================================================================
// Tile constants for simdgroup MMA kernels
// ============================================================================
constant constexpr uint TILE_M = 32;
constant constexpr uint TILE_N = 32;
constant constexpr uint TILE_K = 32;
constant constexpr uint SG_ROWS = 2;   // simdgroup grid rows
constant constexpr uint SG_COLS = 2;   // simdgroup grid columns
constant constexpr uint NUM_SG = 4;    // total simdgroups (SG_ROWS * SG_COLS)
constant constexpr uint TG_SIZE = NUM_SG * 32;  // 128 threads

// ============================================================================
// Function constants for GEMM boundary elimination
//
// When BC_M/BC_N/BC_K are false, the Metal compiler dead-code-eliminates all
// boundary checks in the GEMM inner loop. For aligned dimensions (M%32==0,
// N%32==0, K%32==0), this removes per-element bounds checking from the hot path.
// Index 10-12 chosen to avoid conflicts with future constants.
// ============================================================================
constant bool FC_BC_M [[function_constant(10)]];  // true if M may not align to TILE_M
constant bool FC_BC_N [[function_constant(11)]];  // true if N may not align to TILE_N
constant bool FC_BC_K [[function_constant(12)]];  // true if K may not align to TILE_K

// ============================================================================
// tiled_matmul_f32: Tiled GEMM using simdgroup MMA with half-precision tiles
//
// C[M,N] = A[M,K] * B^T[N,K]    (B stored row-major as [N,K])
//
// 128 threads = 4 simdgroups arranged as 2x2 over the 32x32 output tile.
// Each simdgroup computes a 16x16 sub-tile using 2x2 grid of 8x8 MMA ops.
// Threadgroup memory stores tiles as half to halve bandwidth.
// Mixed-precision MMA: half inputs, float accumulators.
// ============================================================================

kernel void tiled_matmul_f32(
    device const float* A       [[buffer(0)]],   // [M, K] row-major
    device const float* B       [[buffer(1)]],   // [N, K] row-major
    device float*       C       [[buffer(2)]],   // [M, N] row-major
    constant uint&      M       [[buffer(3)]],
    constant uint&      N       [[buffer(4)]],
    constant uint&      K       [[buffer(5)]],
    uint2 tg_pos                [[threadgroup_position_in_grid]],
    ushort tiitg                [[thread_index_in_threadgroup]],
    ushort sgitg                [[simdgroup_index_in_threadgroup]])
{
    // shmem: sa[64,32] + sb[64,32] = 4096 halfs = 8192 bytes for loading
    // Reused as float staging for boundary store: 16 * 256 = 4096 floats = 16384 bytes
    // Declare as 8192 halfs = 16384 bytes to cover both uses
    threadgroup half shmem[2048];
    threadgroup half* sa = shmem;
    threadgroup half* sb = shmem + TILE_M * TILE_K;

    uint tile_m_start = tg_pos.y * TILE_M;
    uint tile_n_start = tg_pos.x * TILE_N;

    ushort sg_r = sgitg / SG_COLS;  // 0..3
    ushort sg_c = sgitg % SG_COLS;  // 0..3

    simdgroup_float8x8 mc[2][2];
    FOR_UNROLL (ushort i = 0; i < 2; i++)
        FOR_UNROLL (ushort j = 0; j < 2; j++)
            mc[i][j] = make_filled_simdgroup_matrix<float, 8>(0.f);

    uint num_k_tiles = (K + TILE_K - 1) / TILE_K;

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        uint k_base = kt * TILE_K;

        // Load A tile: sa[row, k] = A[m, k]  (32x32 = 1024 elements, 128 threads, 8 per thread)
        // 2D thread mapping: 4 threads per row, each loads 8 consecutive floats via float4
        {
            ushort row = tiitg >> 2;           // tiitg / 4, range 0..31
            ushort col_group = tiitg & 3;      // tiitg % 4, range 0..3
            ushort k_start = col_group << 3;   // col_group * 8

            uint gm = tile_m_start + row;
            uint gk = k_base + k_start;
            threadgroup half* sa_ptr = sa + row * TILE_K + k_start;

            if (gm < M && gk + 7 < K) {
                // Fast path: all 8 elements valid, coalesced float4 loads
                device const float4* a_ptr = (device const float4*)(A + gm * K + gk);
                float4 v0 = a_ptr[0];
                float4 v1 = a_ptr[1];
                sa_ptr[0] = (half)v0.x; sa_ptr[1] = (half)v0.y;
                sa_ptr[2] = (half)v0.z; sa_ptr[3] = (half)v0.w;
                sa_ptr[4] = (half)v1.x; sa_ptr[5] = (half)v1.y;
                sa_ptr[6] = (half)v1.z; sa_ptr[7] = (half)v1.w;
            } else {
                // Boundary: element-wise with bounds check
                FOR_UNROLL (ushort i = 0; i < 8; i++) {
                    uint gk_i = gk + i;
                    sa_ptr[i] = (gm < M && gk_i < K) ? (half)A[gm * K + gk_i] : (half)0.0h;
                }
            }
        }

        // Load B tile row-major: sb[n_local, k_local] = B[n, k]
        for (ushort idx = tiitg; idx < TILE_N * TILE_K; idx += TG_SIZE) {
            ushort n_local = idx / TILE_K;
            ushort k_local = idx % TILE_K;
            uint gn = tile_n_start + n_local;
            uint gk = k_base + k_local;
            sb[n_local * TILE_K + k_local] = (gn < N && gk < K) ? (half)B[gn * K + gk] : (half)0.0h;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // MMA: each simdgroup computes its 16x16 sub-tile
        FOR_UNROLL (ushort ks = 0; ks < TILE_K; ks += 8) {
            simdgroup_half8x8 ma[2];
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(ma[0], sa + (sg_r * 16 + 0) * TILE_K + ks, TILE_K);
            simdgroup_load(ma[1], sa + (sg_r * 16 + 8) * TILE_K + ks, TILE_K);

            simdgroup_half8x8 mb[2];
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(mb[0], sb + (sg_c * 16 + 0) * TILE_K + ks, TILE_K, ulong2(0,0), true);
            simdgroup_load(mb[1], sb + (sg_c * 16 + 8) * TILE_K + ks, TILE_K, ulong2(0,0), true);

            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_multiply_accumulate(mc[0][0], ma[0], mb[0], mc[0][0]);
            simdgroup_multiply_accumulate(mc[0][1], ma[0], mb[1], mc[0][1]);
            simdgroup_multiply_accumulate(mc[1][0], ma[1], mb[0], mc[1][0]);
            simdgroup_multiply_accumulate(mc[1][1], ma[1], mb[1], mc[1][1]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store results
    uint sg_m_base = tile_m_start + sg_r * 16;
    uint sg_n_base = tile_n_start + sg_c * 16;

    if (sg_m_base + 16 <= M && sg_n_base + 16 <= N) {
        FOR_UNROLL (ushort i = 0; i < 2; i++)
            FOR_UNROLL (ushort j = 0; j < 2; j++)
                simdgroup_store(mc[i][j], C + (sg_m_base + i * 8) * N + (sg_n_base + j * 8), N);
    } else {
        threadgroup float* sc = (threadgroup float*)shmem;
        threadgroup float* my_sc = sc + sgitg * 256;
        FOR_UNROLL (ushort i = 0; i < 2; i++)
            FOR_UNROLL (ushort j = 0; j < 2; j++)
                simdgroup_store(mc[i][j], my_sc + (i * 8) * 16 + (j * 8), 16);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ushort lane = tiitg % 32;
        for (ushort idx = lane; idx < 256; idx += 32) {
            ushort local_m = idx / 16;
            ushort local_n = idx % 16;
            uint gm = sg_m_base + local_m;
            uint gn = sg_n_base + local_n;
            if (gm < M && gn < N)
                C[gm * N + gn] = my_sc[local_m * 16 + local_n];
        }
    }
}

// ============================================================================
// tiled_matmul_bytes_f32: Tiled GEMM with byte-encoded F32 weights
//
// Same as tiled_matmul_f32 but B is a byte buffer of LE f32.
// Uses simdgroup MMA with half-precision threadgroup tiles.
// ============================================================================

kernel void tiled_matmul_bytes_f32(
    device const float* A        [[buffer(0)]],
    device const uchar* B_bytes  [[buffer(1)]],
    device float*       C        [[buffer(2)]],
    constant uint&      M        [[buffer(3)]],
    constant uint&      N        [[buffer(4)]],
    constant uint&      K        [[buffer(5)]],
    uint2 tg_pos                 [[threadgroup_position_in_grid]],
    ushort tiitg                 [[thread_index_in_threadgroup]],
    ushort sgitg                 [[simdgroup_index_in_threadgroup]])
{
    device const float* B = (device const float*)B_bytes;

    threadgroup half shmem[2048];
    threadgroup half* sa = shmem;
    threadgroup half* sb = shmem + TILE_M * TILE_K;

    uint tile_m_start = tg_pos.y * TILE_M;
    uint tile_n_start = tg_pos.x * TILE_N;

    ushort sg_r = sgitg / SG_COLS;
    ushort sg_c = sgitg % SG_COLS;

    simdgroup_float8x8 mc[2][2];
    FOR_UNROLL (ushort i = 0; i < 2; i++)
        FOR_UNROLL (ushort j = 0; j < 2; j++)
            mc[i][j] = make_filled_simdgroup_matrix<float, 8>(0.f);

    uint num_k_tiles = (K + TILE_K - 1) / TILE_K;

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        uint k_base = kt * TILE_K;

        // Load A tile: vectorized 2D mapping, 4 threads per row, 8 elements each via float4
        {
            ushort row = tiitg >> 2;           // tiitg / 4, range 0..31
            ushort col_group = tiitg & 3;      // tiitg % 4, range 0..3
            ushort k_start = col_group << 3;   // col_group * 8

            uint gm = tile_m_start + row;
            uint gk = k_base + k_start;
            threadgroup half* sa_ptr = sa + row * TILE_K + k_start;

            if (gm < M && gk + 7 < K) {
                device const float4* a_ptr = (device const float4*)(A + gm * K + gk);
                float4 v0 = a_ptr[0];
                float4 v1 = a_ptr[1];
                sa_ptr[0] = (half)v0.x; sa_ptr[1] = (half)v0.y;
                sa_ptr[2] = (half)v0.z; sa_ptr[3] = (half)v0.w;
                sa_ptr[4] = (half)v1.x; sa_ptr[5] = (half)v1.y;
                sa_ptr[6] = (half)v1.z; sa_ptr[7] = (half)v1.w;
            } else {
                FOR_UNROLL (ushort i = 0; i < 8; i++) {
                    uint gk_i = gk + i;
                    sa_ptr[i] = (gm < M && gk_i < K) ? (half)A[gm * K + gk_i] : (half)0.0h;
                }
            }
        }

        for (ushort idx = tiitg; idx < TILE_N * TILE_K; idx += TG_SIZE) {
            ushort n_local = idx / TILE_K;
            ushort k_local = idx % TILE_K;
            uint gn = tile_n_start + n_local;
            uint gk = k_base + k_local;
            sb[n_local * TILE_K + k_local] = (gn < N && gk < K) ? (half)B[gn * K + gk] : (half)0.0h;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (ushort ks = 0; ks < TILE_K; ks += 8) {
            simdgroup_half8x8 ma[2];
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(ma[0], sa + (sg_r * 16 + 0) * TILE_K + ks, TILE_K);
            simdgroup_load(ma[1], sa + (sg_r * 16 + 8) * TILE_K + ks, TILE_K);

            simdgroup_half8x8 mb[2];
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(mb[0], sb + (sg_c * 16 + 0) * TILE_K + ks, TILE_K, ulong2(0,0), true);
            simdgroup_load(mb[1], sb + (sg_c * 16 + 8) * TILE_K + ks, TILE_K, ulong2(0,0), true);

            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_multiply_accumulate(mc[0][0], ma[0], mb[0], mc[0][0]);
            simdgroup_multiply_accumulate(mc[0][1], ma[0], mb[1], mc[0][1]);
            simdgroup_multiply_accumulate(mc[1][0], ma[1], mb[0], mc[1][0]);
            simdgroup_multiply_accumulate(mc[1][1], ma[1], mb[1], mc[1][1]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint sg_m_base = tile_m_start + sg_r * 16;
    uint sg_n_base = tile_n_start + sg_c * 16;

    if (sg_m_base + 16 <= M && sg_n_base + 16 <= N) {
        FOR_UNROLL (ushort i = 0; i < 2; i++)
            FOR_UNROLL (ushort j = 0; j < 2; j++)
                simdgroup_store(mc[i][j], C + (sg_m_base + i * 8) * N + (sg_n_base + j * 8), N);
    } else {
        threadgroup float* sc = (threadgroup float*)shmem;
        threadgroup float* my_sc = sc + sgitg * 256;
        FOR_UNROLL (ushort i = 0; i < 2; i++)
            FOR_UNROLL (ushort j = 0; j < 2; j++)
                simdgroup_store(mc[i][j], my_sc + (i * 8) * 16 + (j * 8), 16);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ushort lane = tiitg % 32;
        for (ushort idx = lane; idx < 256; idx += 32) {
            ushort local_m = idx / 16;
            ushort local_n = idx % 16;
            uint gm = sg_m_base + local_m;
            uint gn = sg_n_base + local_n;
            if (gm < M && gn < N)
                C[gm * N + gn] = my_sc[local_m * 16 + local_n];
        }
    }
}

// ============================================================================
// dequant_tiled_matmul_q8_0: Fused Q8_0 dequantization + tiled GEMM via MMA
//
// Y[M,N] = X[M,K] * dequant(W_q8[N,K_bytes])^T
//
// Uses simdgroup MMA with half-precision tiles. Dequantized Q8_0 values
// naturally fit in half precision (int8 * f16_scale).
//
// Q8_0 block layout (34 bytes per 32 elements):
//   [f16 scale (2 bytes)] [32 x int8 values (32 bytes)]
// ============================================================================

constant constexpr uint Q8B_GROUP_SIZE = 32;
constant constexpr uint Q8B_BLOCK_SIZE = 34;

kernel void dequant_tiled_matmul_q8_0(
    device const uchar* W_q8     [[buffer(0)]],   // Q8_0 weights [N, K_bytes]
    device const float* X        [[buffer(1)]],   // [M, K] input batch
    device float*       Y        [[buffer(2)]],   // [M, N] output batch
    constant uint&      M        [[buffer(3)]],   // batch size
    constant uint&      N        [[buffer(4)]],   // output dim
    constant uint&      K        [[buffer(5)]],   // input dim (elements, not bytes)
    uint2 tg_pos                 [[threadgroup_position_in_grid]],
    ushort tiitg                 [[thread_index_in_threadgroup]],
    ushort sgitg                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup half shmem[2048];
    threadgroup half* sa = shmem;
    threadgroup half* sb = shmem + TILE_M * TILE_K;

    uint tile_m_start = tg_pos.y * TILE_M;
    uint tile_n_start = tg_pos.x * TILE_N;

    ushort sg_r = sgitg / SG_COLS;
    ushort sg_c = sgitg % SG_COLS;

    simdgroup_float8x8 mc[2][2];
    FOR_UNROLL (ushort i = 0; i < 2; i++)
        FOR_UNROLL (ushort j = 0; j < 2; j++)
            mc[i][j] = make_filled_simdgroup_matrix<float, 8>(0.f);

    uint num_blocks_per_row = (K + Q8B_GROUP_SIZE - 1) / Q8B_GROUP_SIZE;
    uint row_bytes = num_blocks_per_row * Q8B_BLOCK_SIZE;

    uint num_k_tiles = (K + TILE_K - 1) / TILE_K;

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        uint k_base = kt * TILE_K;

        // Load A tile: vectorized 2D mapping, 4 threads per row, 8 elements each via float4
        // FC_BC_M/FC_BC_K: when false (aligned), compiler eliminates boundary checks
        {
            ushort row = tiitg >> 2;           // tiitg / 4, range 0..31
            ushort col_group = tiitg & 3;      // tiitg % 4, range 0..3
            ushort k_start = col_group << 3;   // col_group * 8

            uint gm = tile_m_start + row;
            uint gk = k_base + k_start;
            threadgroup half* sa_ptr = sa + row * TILE_K + k_start;

            if (FC_BC_M || FC_BC_K) {
                // Boundary-checked path (compiled in only when dimensions may be unaligned)
                if (gm < M && gk + 7 < K) {
                    device const float4* x_ptr = (device const float4*)(X + gm * K + gk);
                    float4 v0 = x_ptr[0];
                    float4 v1 = x_ptr[1];
                    sa_ptr[0] = (half)v0.x; sa_ptr[1] = (half)v0.y;
                    sa_ptr[2] = (half)v0.z; sa_ptr[3] = (half)v0.w;
                    sa_ptr[4] = (half)v1.x; sa_ptr[5] = (half)v1.y;
                    sa_ptr[6] = (half)v1.z; sa_ptr[7] = (half)v1.w;
                } else {
                    FOR_UNROLL (ushort i = 0; i < 8; i++) {
                        uint gk_i = gk + i;
                        sa_ptr[i] = (gm < M && gk_i < K) ? (half)X[gm * K + gk_i] : (half)0.0h;
                    }
                }
            } else {
                // Fast path: no bounds checks needed (M aligned to TILE_M, K aligned to TILE_K)
                device const float4* x_ptr = (device const float4*)(X + gm * K + gk);
                float4 v0 = x_ptr[0];
                float4 v1 = x_ptr[1];
                sa_ptr[0] = (half)v0.x; sa_ptr[1] = (half)v0.y;
                sa_ptr[2] = (half)v0.z; sa_ptr[3] = (half)v0.w;
                sa_ptr[4] = (half)v1.x; sa_ptr[5] = (half)v1.y;
                sa_ptr[6] = (half)v1.z; sa_ptr[7] = (half)v1.w;
            }
        }

        // Load B tile: vectorized Q8_0 dequant (TILE_K=32 = Q8_0 block size)
        // 32 rows x 1 block/row = 32 blocks. 128 threads => 4 threads/row.
        // Each thread loads 8 consecutive int8 values (2 x uint) + 1 scale read.
        // FC_BC_N/FC_BC_K: when false (aligned), compiler eliminates boundary checks
        {
            uint block_idx = k_base / Q8B_GROUP_SIZE;
            ushort n_local = tiitg / 4;     // 0..31 (which row)
            ushort t_in_row = tiitg % 4;    // 0..3 (which quarter of the 32 elements)
            ushort k_offset = t_in_row * 8; // starting k position within block
            uint gn = tile_n_start + n_local;

            threadgroup half* sb_row = sb + n_local * TILE_K + k_offset;

            if (FC_BC_N || FC_BC_K) {
                // Boundary-checked path
                if (gn < N && k_base + k_offset < K) {
                    uint block_offset = gn * row_bytes + block_idx * Q8B_BLOCK_SIZE;
                    half scale = as_type<half>(*(device const ushort*)(W_q8 + block_offset));
                    device const char* qdata = (device const char*)(W_q8 + block_offset + 2 + k_offset);

                    uint k_remaining = K - (k_base + k_offset);
                    ushort count = (k_remaining >= 8) ? 8 : (ushort)k_remaining;

                    if (count == 8) {
                        sb_row[0] = scale * (half)qdata[0];
                        sb_row[1] = scale * (half)qdata[1];
                        sb_row[2] = scale * (half)qdata[2];
                        sb_row[3] = scale * (half)qdata[3];
                        sb_row[4] = scale * (half)qdata[4];
                        sb_row[5] = scale * (half)qdata[5];
                        sb_row[6] = scale * (half)qdata[6];
                        sb_row[7] = scale * (half)qdata[7];
                    } else {
                        for (ushort i = 0; i < count; i++) {
                            sb_row[i] = scale * (half)qdata[i];
                        }
                        for (ushort i = count; i < 8; i++) {
                            sb_row[i] = (half)0.0h;
                        }
                    }
                } else {
                    FOR_UNROLL (ushort i = 0; i < 8; i++) {
                        sb_row[i] = (half)0.0h;
                    }
                }
            } else {
                // Fast path: no bounds checks (N aligned to TILE_N, K aligned to TILE_K)
                uint block_offset = gn * row_bytes + block_idx * Q8B_BLOCK_SIZE;
                half scale = as_type<half>(*(device const ushort*)(W_q8 + block_offset));
                device const char* qdata = (device const char*)(W_q8 + block_offset + 2 + k_offset);

                sb_row[0] = scale * (half)qdata[0];
                sb_row[1] = scale * (half)qdata[1];
                sb_row[2] = scale * (half)qdata[2];
                sb_row[3] = scale * (half)qdata[3];
                sb_row[4] = scale * (half)qdata[4];
                sb_row[5] = scale * (half)qdata[5];
                sb_row[6] = scale * (half)qdata[6];
                sb_row[7] = scale * (half)qdata[7];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (ushort ks = 0; ks < TILE_K; ks += 8) {
            simdgroup_half8x8 ma[2];
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(ma[0], sa + (sg_r * 16 + 0) * TILE_K + ks, TILE_K);
            simdgroup_load(ma[1], sa + (sg_r * 16 + 8) * TILE_K + ks, TILE_K);

            simdgroup_half8x8 mb[2];
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(mb[0], sb + (sg_c * 16 + 0) * TILE_K + ks, TILE_K, ulong2(0,0), true);
            simdgroup_load(mb[1], sb + (sg_c * 16 + 8) * TILE_K + ks, TILE_K, ulong2(0,0), true);

            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_multiply_accumulate(mc[0][0], ma[0], mb[0], mc[0][0]);
            simdgroup_multiply_accumulate(mc[0][1], ma[0], mb[1], mc[0][1]);
            simdgroup_multiply_accumulate(mc[1][0], ma[1], mb[0], mc[1][0]);
            simdgroup_multiply_accumulate(mc[1][1], ma[1], mb[1], mc[1][1]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint sg_m_base = tile_m_start + sg_r * 16;
    uint sg_n_base = tile_n_start + sg_c * 16;

    if (FC_BC_M || FC_BC_N) {
        // Boundary-checked store path
        if (sg_m_base + 16 <= M && sg_n_base + 16 <= N) {
            FOR_UNROLL (ushort i = 0; i < 2; i++)
                FOR_UNROLL (ushort j = 0; j < 2; j++)
                    simdgroup_store(mc[i][j], Y + (sg_m_base + i * 8) * N + (sg_n_base + j * 8), N);
        } else {
            threadgroup float* sc = (threadgroup float*)shmem;
            threadgroup float* my_sc = sc + sgitg * 256;
            FOR_UNROLL (ushort i = 0; i < 2; i++)
                FOR_UNROLL (ushort j = 0; j < 2; j++)
                    simdgroup_store(mc[i][j], my_sc + (i * 8) * 16 + (j * 8), 16);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            ushort lane = tiitg % 32;
            for (ushort idx = lane; idx < 256; idx += 32) {
                ushort local_m = idx / 16;
                ushort local_n = idx % 16;
                uint gm = sg_m_base + local_m;
                uint gn = sg_n_base + local_n;
                if (gm < M && gn < N) {
                    Y[gm * N + gn] = my_sc[local_m * 16 + local_n];
                }
            }
        }
    } else {
        // Fast path: direct store (M and N aligned to tile dimensions)
        FOR_UNROLL (ushort i = 0; i < 2; i++)
            FOR_UNROLL (ushort j = 0; j < 2; j++)
                simdgroup_store(mc[i][j], Y + (sg_m_base + i * 8) * N + (sg_n_base + j * 8), N);
    }
}

constant constexpr uint TILE_K_64 = 64;

// ============================================================================
// dequant_tiled_matmul_q8_0_k64: TILE_K=64 variant for fewer barriers
//
// Same algorithm as dequant_tiled_matmul_q8_0 but processes 2 Q8_0 blocks per
// K-step instead of 1. This halves the number of threadgroup barriers (from
// K/32 to K/64) and K-loop iterations, improving efficiency for large-K GEMMs.
//
// Shared memory: sa[32*64] + sb[32*64] = 4096 halfs = 8192 bytes
// (still well within 32KB threadgroup limit; occupancy >= 2 TG/CU)
//
// A-tile loading: 128 threads, 32 rows x 4 threads/row, each loads 16 elems
// (two float4 reads). 4 threads * 16 = 64 = TILE_K_64.
//
// B-tile loading: 2-pass approach. Each pass loads one Q8_0 block (32 elems)
// across 32 rows using 128 threads mapped as 32 rows x 4 threads/row.
// ============================================================================

kernel void dequant_tiled_matmul_q8_0_k64(
    device const uchar* W_q8     [[buffer(0)]],   // Q8_0 weights [N, K_bytes]
    device const float* X        [[buffer(1)]],   // [M, K] input batch
    device float*       Y        [[buffer(2)]],   // [M, N] output batch
    constant uint&      M        [[buffer(3)]],   // batch size
    constant uint&      N        [[buffer(4)]],   // output dim
    constant uint&      K        [[buffer(5)]],   // input dim (elements, not bytes)
    uint2 tg_pos                 [[threadgroup_position_in_grid]],
    ushort tiitg                 [[thread_index_in_threadgroup]],
    ushort sgitg                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup half shmem_k64[4096];  // sa[32*64] + sb[32*64]
    threadgroup half* sa = shmem_k64;
    threadgroup half* sb = shmem_k64 + TILE_M * TILE_K_64;

    uint tile_m_start = tg_pos.y * TILE_M;
    uint tile_n_start = tg_pos.x * TILE_N;

    ushort sg_r = sgitg / SG_COLS;
    ushort sg_c = sgitg % SG_COLS;

    simdgroup_float8x8 mc[2][2];
    FOR_UNROLL (ushort i = 0; i < 2; i++)
        FOR_UNROLL (ushort j = 0; j < 2; j++)
            mc[i][j] = make_filled_simdgroup_matrix<float, 8>(0.f);

    uint num_blocks_per_row = (K + Q8B_GROUP_SIZE - 1) / Q8B_GROUP_SIZE;
    uint row_bytes = num_blocks_per_row * Q8B_BLOCK_SIZE;

    uint num_k_tiles = (K + TILE_K_64 - 1) / TILE_K_64;

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        uint k_base = kt * TILE_K_64;

        // Load A tile: 32 rows x 64 cols
        // FC_BC_M/FC_BC_K: when false (aligned), compiler eliminates boundary checks
        {
            ushort row = tiitg >> 2;           // tiitg / 4, range 0..31
            ushort col_group = tiitg & 3;      // tiitg % 4, range 0..3
            ushort k_start = col_group << 4;   // col_group * 16

            uint gm = tile_m_start + row;
            uint gk = k_base + k_start;
            threadgroup half* sa_ptr = sa + row * TILE_K_64 + k_start;

            if (FC_BC_M || FC_BC_K) {
                if (gm < M && gk + 15 < K) {
                    device const float4* x_ptr = (device const float4*)(X + gm * K + gk);
                    float4 v0 = x_ptr[0];
                    float4 v1 = x_ptr[1];
                    float4 v2 = x_ptr[2];
                    float4 v3 = x_ptr[3];
                    sa_ptr[0]  = (half)v0.x; sa_ptr[1]  = (half)v0.y;
                    sa_ptr[2]  = (half)v0.z; sa_ptr[3]  = (half)v0.w;
                    sa_ptr[4]  = (half)v1.x; sa_ptr[5]  = (half)v1.y;
                    sa_ptr[6]  = (half)v1.z; sa_ptr[7]  = (half)v1.w;
                    sa_ptr[8]  = (half)v2.x; sa_ptr[9]  = (half)v2.y;
                    sa_ptr[10] = (half)v2.z; sa_ptr[11] = (half)v2.w;
                    sa_ptr[12] = (half)v3.x; sa_ptr[13] = (half)v3.y;
                    sa_ptr[14] = (half)v3.z; sa_ptr[15] = (half)v3.w;
                } else {
                    FOR_UNROLL (ushort i = 0; i < 16; i++) {
                        uint gk_i = gk + i;
                        sa_ptr[i] = (gm < M && gk_i < K) ? (half)X[gm * K + gk_i] : (half)0.0h;
                    }
                }
            } else {
                device const float4* x_ptr = (device const float4*)(X + gm * K + gk);
                float4 v0 = x_ptr[0];
                float4 v1 = x_ptr[1];
                float4 v2 = x_ptr[2];
                float4 v3 = x_ptr[3];
                sa_ptr[0]  = (half)v0.x; sa_ptr[1]  = (half)v0.y;
                sa_ptr[2]  = (half)v0.z; sa_ptr[3]  = (half)v0.w;
                sa_ptr[4]  = (half)v1.x; sa_ptr[5]  = (half)v1.y;
                sa_ptr[6]  = (half)v1.z; sa_ptr[7]  = (half)v1.w;
                sa_ptr[8]  = (half)v2.x; sa_ptr[9]  = (half)v2.y;
                sa_ptr[10] = (half)v2.z; sa_ptr[11] = (half)v2.w;
                sa_ptr[12] = (half)v3.x; sa_ptr[13] = (half)v3.y;
                sa_ptr[14] = (half)v3.z; sa_ptr[15] = (half)v3.w;
            }
        }

        // Load B tile: 32 rows x 64 cols = 2 Q8_0 blocks per row
        // FC_BC_N/FC_BC_K: when false (aligned), compiler eliminates boundary checks
        {
            ushort n_local = tiitg >> 2;       // 0..31 (which row)
            ushort t_in_row = tiitg & 3;       // 0..3 (quarter of 32 elements)
            uint gn = tile_n_start + n_local;

            // Pass 0: first Q8_0 block (k_base + 0..31)
            {
                uint block_idx = k_base / Q8B_GROUP_SIZE;
                ushort k_offset = t_in_row * 8;
                threadgroup half* sb_row = sb + n_local * TILE_K_64 + k_offset;

                if (FC_BC_N || FC_BC_K) {
                    if (gn < N && k_base + k_offset < K) {
                        uint block_offset = gn * row_bytes + block_idx * Q8B_BLOCK_SIZE;
                        half scale = as_type<half>(*(device const ushort*)(W_q8 + block_offset));
                        device const char* qdata = (device const char*)(W_q8 + block_offset + 2 + k_offset);

                        uint k_remaining = K - (k_base + k_offset);
                        ushort count = (k_remaining >= 8) ? 8 : (ushort)k_remaining;

                        if (count == 8) {
                            sb_row[0] = scale * (half)qdata[0];
                            sb_row[1] = scale * (half)qdata[1];
                            sb_row[2] = scale * (half)qdata[2];
                            sb_row[3] = scale * (half)qdata[3];
                            sb_row[4] = scale * (half)qdata[4];
                            sb_row[5] = scale * (half)qdata[5];
                            sb_row[6] = scale * (half)qdata[6];
                            sb_row[7] = scale * (half)qdata[7];
                        } else {
                            for (ushort i = 0; i < count; i++) {
                                sb_row[i] = scale * (half)qdata[i];
                            }
                            for (ushort i = count; i < 8; i++) {
                                sb_row[i] = (half)0.0h;
                            }
                        }
                    } else {
                        FOR_UNROLL (ushort i = 0; i < 8; i++) {
                            sb_row[i] = (half)0.0h;
                        }
                    }
                } else {
                    uint block_offset = gn * row_bytes + block_idx * Q8B_BLOCK_SIZE;
                    half scale = as_type<half>(*(device const ushort*)(W_q8 + block_offset));
                    device const char* qdata = (device const char*)(W_q8 + block_offset + 2 + k_offset);
                    sb_row[0] = scale * (half)qdata[0];
                    sb_row[1] = scale * (half)qdata[1];
                    sb_row[2] = scale * (half)qdata[2];
                    sb_row[3] = scale * (half)qdata[3];
                    sb_row[4] = scale * (half)qdata[4];
                    sb_row[5] = scale * (half)qdata[5];
                    sb_row[6] = scale * (half)qdata[6];
                    sb_row[7] = scale * (half)qdata[7];
                }
            }

            // Pass 1: second Q8_0 block (k_base + 32..63)
            {
                uint block_idx = (k_base + 32) / Q8B_GROUP_SIZE;
                ushort k_offset = t_in_row * 8;
                threadgroup half* sb_row = sb + n_local * TILE_K_64 + 32 + k_offset;

                if (FC_BC_N || FC_BC_K) {
                    if (gn < N && k_base + 32 + k_offset < K) {
                        uint block_offset = gn * row_bytes + block_idx * Q8B_BLOCK_SIZE;
                        half scale = as_type<half>(*(device const ushort*)(W_q8 + block_offset));
                        device const char* qdata = (device const char*)(W_q8 + block_offset + 2 + k_offset);

                        uint k_remaining = K - (k_base + 32 + k_offset);
                        ushort count = (k_remaining >= 8) ? 8 : (ushort)k_remaining;

                        if (count == 8) {
                            sb_row[0] = scale * (half)qdata[0];
                            sb_row[1] = scale * (half)qdata[1];
                            sb_row[2] = scale * (half)qdata[2];
                            sb_row[3] = scale * (half)qdata[3];
                            sb_row[4] = scale * (half)qdata[4];
                            sb_row[5] = scale * (half)qdata[5];
                            sb_row[6] = scale * (half)qdata[6];
                            sb_row[7] = scale * (half)qdata[7];
                        } else {
                            for (ushort i = 0; i < count; i++) {
                                sb_row[i] = scale * (half)qdata[i];
                            }
                            for (ushort i = count; i < 8; i++) {
                                sb_row[i] = (half)0.0h;
                            }
                        }
                    } else {
                        FOR_UNROLL (ushort i = 0; i < 8; i++) {
                            sb_row[i] = (half)0.0h;
                        }
                    }
                } else {
                    uint block_offset = gn * row_bytes + block_idx * Q8B_BLOCK_SIZE;
                    half scale = as_type<half>(*(device const ushort*)(W_q8 + block_offset));
                    device const char* qdata = (device const char*)(W_q8 + block_offset + 2 + k_offset);
                    sb_row[0] = scale * (half)qdata[0];
                    sb_row[1] = scale * (half)qdata[1];
                    sb_row[2] = scale * (half)qdata[2];
                    sb_row[3] = scale * (half)qdata[3];
                    sb_row[4] = scale * (half)qdata[4];
                    sb_row[5] = scale * (half)qdata[5];
                    sb_row[6] = scale * (half)qdata[6];
                    sb_row[7] = scale * (half)qdata[7];
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // MMA: 8 iterations (TILE_K_64=64, step by 8)
        FOR_UNROLL (ushort ks = 0; ks < TILE_K_64; ks += 8) {
            simdgroup_half8x8 ma[2];
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(ma[0], sa + (sg_r * 16 + 0) * TILE_K_64 + ks, TILE_K_64);
            simdgroup_load(ma[1], sa + (sg_r * 16 + 8) * TILE_K_64 + ks, TILE_K_64);

            simdgroup_half8x8 mb[2];
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(mb[0], sb + (sg_c * 16 + 0) * TILE_K_64 + ks, TILE_K_64, ulong2(0,0), true);
            simdgroup_load(mb[1], sb + (sg_c * 16 + 8) * TILE_K_64 + ks, TILE_K_64, ulong2(0,0), true);

            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_multiply_accumulate(mc[0][0], ma[0], mb[0], mc[0][0]);
            simdgroup_multiply_accumulate(mc[0][1], ma[0], mb[1], mc[0][1]);
            simdgroup_multiply_accumulate(mc[1][0], ma[1], mb[0], mc[1][0]);
            simdgroup_multiply_accumulate(mc[1][1], ma[1], mb[1], mc[1][1]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store results
    uint sg_m_base = tile_m_start + sg_r * 16;
    uint sg_n_base = tile_n_start + sg_c * 16;

    if (FC_BC_M || FC_BC_N) {
        if (sg_m_base + 16 <= M && sg_n_base + 16 <= N) {
            FOR_UNROLL (ushort i = 0; i < 2; i++)
                FOR_UNROLL (ushort j = 0; j < 2; j++)
                    simdgroup_store(mc[i][j], Y + (sg_m_base + i * 8) * N + (sg_n_base + j * 8), N);
        } else {
            threadgroup float* sc = (threadgroup float*)shmem_k64;
            threadgroup float* my_sc = sc + sgitg * 256;
            FOR_UNROLL (ushort i = 0; i < 2; i++)
                FOR_UNROLL (ushort j = 0; j < 2; j++)
                    simdgroup_store(mc[i][j], my_sc + (i * 8) * 16 + (j * 8), 16);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            ushort lane = tiitg % 32;
            for (ushort idx = lane; idx < 256; idx += 32) {
                ushort local_m = idx / 16;
                ushort local_n = idx % 16;
                uint gm = sg_m_base + local_m;
                uint gn = sg_n_base + local_n;
                if (gm < M && gn < N) {
                    Y[gm * N + gn] = my_sc[local_m * 16 + local_n];
                }
            }
        }
    } else {
        FOR_UNROLL (ushort i = 0; i < 2; i++)
            FOR_UNROLL (ushort j = 0; j < 2; j++)
                simdgroup_store(mc[i][j], Y + (sg_m_base + i * 8) * N + (sg_n_base + j * 8), N);
    }
}

// ============================================================================
// dequant_tiled_matmul_q8_0_k64_residual_batched: K64 variant + residual add
//
// Y[m,n] = X[M,K] * dequant(W_q8[N,K_bytes])^T + R[m,n]
// Same as dequant_tiled_matmul_q8_0_k64 but fuses residual add at writeback.
// ============================================================================

kernel void dequant_tiled_matmul_q8_0_k64_residual_batched(
    device const uchar* W_q8     [[buffer(0)]],   // Q8_0 weights [N, K_bytes]
    device const float* X        [[buffer(1)]],   // [M, K] input batch
    device float*       Y        [[buffer(2)]],   // [M, N] output batch
    constant uint&      M        [[buffer(3)]],   // batch size
    constant uint&      N        [[buffer(4)]],   // output dim
    constant uint&      K        [[buffer(5)]],   // input dim (elements, not bytes)
    device const float* R        [[buffer(6)]],   // [M, N] residual to add
    uint2 tg_pos                 [[threadgroup_position_in_grid]],
    ushort tiitg                 [[thread_index_in_threadgroup]],
    ushort sgitg                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup half shmem_k64r[4096];
    threadgroup half* sa = shmem_k64r;
    threadgroup half* sb = shmem_k64r + TILE_M * TILE_K_64;

    uint tile_m_start = tg_pos.y * TILE_M;
    uint tile_n_start = tg_pos.x * TILE_N;

    ushort sg_r = sgitg / SG_COLS;
    ushort sg_c = sgitg % SG_COLS;

    simdgroup_float8x8 mc[2][2];
    FOR_UNROLL (ushort i = 0; i < 2; i++)
        FOR_UNROLL (ushort j = 0; j < 2; j++)
            mc[i][j] = make_filled_simdgroup_matrix<float, 8>(0.f);

    uint num_blocks_per_row = (K + Q8B_GROUP_SIZE - 1) / Q8B_GROUP_SIZE;
    uint row_bytes = num_blocks_per_row * Q8B_BLOCK_SIZE;

    uint num_k_tiles = (K + TILE_K_64 - 1) / TILE_K_64;

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        uint k_base = kt * TILE_K_64;

        // Load A tile: 32 rows x 64 cols
        // FC_BC_M/FC_BC_K: when false (aligned), compiler eliminates boundary checks
        {
            ushort row = tiitg >> 2;
            ushort col_group = tiitg & 3;
            ushort k_start = col_group << 4;   // col_group * 16

            uint gm = tile_m_start + row;
            uint gk = k_base + k_start;
            threadgroup half* sa_ptr = sa + row * TILE_K_64 + k_start;

            if (FC_BC_M || FC_BC_K) {
                if (gm < M && gk + 15 < K) {
                    device const float4* x_ptr = (device const float4*)(X + gm * K + gk);
                    float4 v0 = x_ptr[0];
                    float4 v1 = x_ptr[1];
                    float4 v2 = x_ptr[2];
                    float4 v3 = x_ptr[3];
                    sa_ptr[0]  = (half)v0.x; sa_ptr[1]  = (half)v0.y;
                    sa_ptr[2]  = (half)v0.z; sa_ptr[3]  = (half)v0.w;
                    sa_ptr[4]  = (half)v1.x; sa_ptr[5]  = (half)v1.y;
                    sa_ptr[6]  = (half)v1.z; sa_ptr[7]  = (half)v1.w;
                    sa_ptr[8]  = (half)v2.x; sa_ptr[9]  = (half)v2.y;
                    sa_ptr[10] = (half)v2.z; sa_ptr[11] = (half)v2.w;
                    sa_ptr[12] = (half)v3.x; sa_ptr[13] = (half)v3.y;
                    sa_ptr[14] = (half)v3.z; sa_ptr[15] = (half)v3.w;
                } else {
                    FOR_UNROLL (ushort i = 0; i < 16; i++) {
                        uint gk_i = gk + i;
                        sa_ptr[i] = (gm < M && gk_i < K) ? (half)X[gm * K + gk_i] : (half)0.0h;
                    }
                }
            } else {
                device const float4* x_ptr = (device const float4*)(X + gm * K + gk);
                float4 v0 = x_ptr[0];
                float4 v1 = x_ptr[1];
                float4 v2 = x_ptr[2];
                float4 v3 = x_ptr[3];
                sa_ptr[0]  = (half)v0.x; sa_ptr[1]  = (half)v0.y;
                sa_ptr[2]  = (half)v0.z; sa_ptr[3]  = (half)v0.w;
                sa_ptr[4]  = (half)v1.x; sa_ptr[5]  = (half)v1.y;
                sa_ptr[6]  = (half)v1.z; sa_ptr[7]  = (half)v1.w;
                sa_ptr[8]  = (half)v2.x; sa_ptr[9]  = (half)v2.y;
                sa_ptr[10] = (half)v2.z; sa_ptr[11] = (half)v2.w;
                sa_ptr[12] = (half)v3.x; sa_ptr[13] = (half)v3.y;
                sa_ptr[14] = (half)v3.z; sa_ptr[15] = (half)v3.w;
            }
        }

        // Load B tile: 2-pass Q8_0 dequant (2 blocks per row)
        // FC_BC_N/FC_BC_K: when false (aligned), compiler eliminates boundary checks
        {
            ushort n_local = tiitg >> 2;
            ushort t_in_row = tiitg & 3;
            uint gn = tile_n_start + n_local;

            // Pass 0: first block (k_base + 0..31)
            {
                uint block_idx = k_base / Q8B_GROUP_SIZE;
                ushort k_offset = t_in_row * 8;
                threadgroup half* sb_row = sb + n_local * TILE_K_64 + k_offset;

                if (FC_BC_N || FC_BC_K) {
                    if (gn < N && k_base + k_offset < K) {
                        uint block_offset = gn * row_bytes + block_idx * Q8B_BLOCK_SIZE;
                        half scale = as_type<half>(*(device const ushort*)(W_q8 + block_offset));
                        device const char* qdata = (device const char*)(W_q8 + block_offset + 2 + k_offset);

                        uint k_remaining = K - (k_base + k_offset);
                        ushort count = (k_remaining >= 8) ? 8 : (ushort)k_remaining;

                        if (count == 8) {
                            sb_row[0] = scale * (half)qdata[0];
                            sb_row[1] = scale * (half)qdata[1];
                            sb_row[2] = scale * (half)qdata[2];
                            sb_row[3] = scale * (half)qdata[3];
                            sb_row[4] = scale * (half)qdata[4];
                            sb_row[5] = scale * (half)qdata[5];
                            sb_row[6] = scale * (half)qdata[6];
                            sb_row[7] = scale * (half)qdata[7];
                        } else {
                            for (ushort i = 0; i < count; i++) {
                                sb_row[i] = scale * (half)qdata[i];
                            }
                            for (ushort i = count; i < 8; i++) {
                                sb_row[i] = (half)0.0h;
                            }
                        }
                    } else {
                        FOR_UNROLL (ushort i = 0; i < 8; i++) {
                            sb_row[i] = (half)0.0h;
                        }
                    }
                } else {
                    uint block_offset = gn * row_bytes + block_idx * Q8B_BLOCK_SIZE;
                    half scale = as_type<half>(*(device const ushort*)(W_q8 + block_offset));
                    device const char* qdata = (device const char*)(W_q8 + block_offset + 2 + k_offset);
                    sb_row[0] = scale * (half)qdata[0];
                    sb_row[1] = scale * (half)qdata[1];
                    sb_row[2] = scale * (half)qdata[2];
                    sb_row[3] = scale * (half)qdata[3];
                    sb_row[4] = scale * (half)qdata[4];
                    sb_row[5] = scale * (half)qdata[5];
                    sb_row[6] = scale * (half)qdata[6];
                    sb_row[7] = scale * (half)qdata[7];
                }
            }

            // Pass 1: second block (k_base + 32..63)
            {
                uint block_idx = (k_base + 32) / Q8B_GROUP_SIZE;
                ushort k_offset = t_in_row * 8;
                threadgroup half* sb_row = sb + n_local * TILE_K_64 + 32 + k_offset;

                if (FC_BC_N || FC_BC_K) {
                    if (gn < N && k_base + 32 + k_offset < K) {
                        uint block_offset = gn * row_bytes + block_idx * Q8B_BLOCK_SIZE;
                        half scale = as_type<half>(*(device const ushort*)(W_q8 + block_offset));
                        device const char* qdata = (device const char*)(W_q8 + block_offset + 2 + k_offset);

                        uint k_remaining = K - (k_base + 32 + k_offset);
                        ushort count = (k_remaining >= 8) ? 8 : (ushort)k_remaining;

                        if (count == 8) {
                            sb_row[0] = scale * (half)qdata[0];
                            sb_row[1] = scale * (half)qdata[1];
                            sb_row[2] = scale * (half)qdata[2];
                            sb_row[3] = scale * (half)qdata[3];
                            sb_row[4] = scale * (half)qdata[4];
                            sb_row[5] = scale * (half)qdata[5];
                            sb_row[6] = scale * (half)qdata[6];
                            sb_row[7] = scale * (half)qdata[7];
                        } else {
                            for (ushort i = 0; i < count; i++) {
                                sb_row[i] = scale * (half)qdata[i];
                            }
                            for (ushort i = count; i < 8; i++) {
                                sb_row[i] = (half)0.0h;
                            }
                        }
                    } else {
                        FOR_UNROLL (ushort i = 0; i < 8; i++) {
                            sb_row[i] = (half)0.0h;
                        }
                    }
                } else {
                    uint block_offset = gn * row_bytes + block_idx * Q8B_BLOCK_SIZE;
                    half scale = as_type<half>(*(device const ushort*)(W_q8 + block_offset));
                    device const char* qdata = (device const char*)(W_q8 + block_offset + 2 + k_offset);
                    sb_row[0] = scale * (half)qdata[0];
                    sb_row[1] = scale * (half)qdata[1];
                    sb_row[2] = scale * (half)qdata[2];
                    sb_row[3] = scale * (half)qdata[3];
                    sb_row[4] = scale * (half)qdata[4];
                    sb_row[5] = scale * (half)qdata[5];
                    sb_row[6] = scale * (half)qdata[6];
                    sb_row[7] = scale * (half)qdata[7];
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (ushort ks = 0; ks < TILE_K_64; ks += 8) {
            simdgroup_half8x8 ma[2];
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(ma[0], sa + (sg_r * 16 + 0) * TILE_K_64 + ks, TILE_K_64);
            simdgroup_load(ma[1], sa + (sg_r * 16 + 8) * TILE_K_64 + ks, TILE_K_64);

            simdgroup_half8x8 mb[2];
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(mb[0], sb + (sg_c * 16 + 0) * TILE_K_64 + ks, TILE_K_64, ulong2(0,0), true);
            simdgroup_load(mb[1], sb + (sg_c * 16 + 8) * TILE_K_64 + ks, TILE_K_64, ulong2(0,0), true);

            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_multiply_accumulate(mc[0][0], ma[0], mb[0], mc[0][0]);
            simdgroup_multiply_accumulate(mc[0][1], ma[0], mb[1], mc[0][1]);
            simdgroup_multiply_accumulate(mc[1][0], ma[1], mb[0], mc[1][0]);
            simdgroup_multiply_accumulate(mc[1][1], ma[1], mb[1], mc[1][1]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store results with residual add: Y[m,n] = GEMM[m,n] + R[m,n]
    uint sg_m_base = tile_m_start + sg_r * 16;
    uint sg_n_base = tile_n_start + sg_c * 16;

    {
        threadgroup float* sc = (threadgroup float*)shmem_k64r;
        threadgroup float* my_sc = sc + sgitg * 256;
        FOR_UNROLL (ushort i = 0; i < 2; i++)
            FOR_UNROLL (ushort j = 0; j < 2; j++)
                simdgroup_store(mc[i][j], my_sc + (i * 8) * 16 + (j * 8), 16);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ushort lane = tiitg % 32;
        if (FC_BC_M || FC_BC_N) {
            for (ushort idx = lane; idx < 256; idx += 32) {
                ushort local_m = idx / 16;
                ushort local_n = idx % 16;
                uint gm = sg_m_base + local_m;
                uint gn = sg_n_base + local_n;
                if (gm < M && gn < N) {
                    Y[gm * N + gn] = my_sc[local_m * 16 + local_n] + R[gm * N + gn];
                }
            }
        } else {
            for (ushort idx = lane; idx < 256; idx += 32) {
                ushort local_m = idx / 16;
                ushort local_n = idx % 16;
                uint gm = sg_m_base + local_m;
                uint gn = sg_n_base + local_n;
                Y[gm * N + gn] = my_sc[local_m * 16 + local_n] + R[gm * N + gn];
            }
        }
    }
}

// ============================================================================
// dequant_tiled_matmul_q8_0_splitk: Split-K variant for GPU core saturation
//
// Same computation as dequant_tiled_matmul_q8_0 but splits the K (reduction)
// dimension across multiple threadgroups via a 3D dispatch grid:
//   (ceil(N/TILE_N), ceil(M/TILE_M), k_splits)
//
// Each threadgroup computes a partial result for its K range, writing to
// Y_partial[split_idx * M * N + m * N + n]. A separate reduce_splitk kernel
// sums the partial results into the final output.
//
// This increases threadgroup count for small M*N GEMMs (e.g. pp128 with
// TinyLlama), saturating more GPU cores.
// ============================================================================

kernel void dequant_tiled_matmul_q8_0_splitk(
    device const uchar* W_q8     [[buffer(0)]],   // Q8_0 weights [N, K_bytes]
    device const float* X        [[buffer(1)]],   // [M, K] input batch
    device float*       Y_partial [[buffer(2)]],  // [k_splits, M, N] partial results
    constant uint&      M        [[buffer(3)]],   // batch size
    constant uint&      N        [[buffer(4)]],   // output dim
    constant uint&      K        [[buffer(5)]],   // input dim (elements)
    constant uint&      k_splits [[buffer(6)]],   // number of K splits
    uint3 tg_pos                 [[threadgroup_position_in_grid]],
    ushort tiitg                 [[thread_index_in_threadgroup]],
    ushort sgitg                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup half shmem[2048];
    threadgroup half* sa = shmem;
    threadgroup half* sb = shmem + TILE_M * TILE_K;

    uint tile_n_start = tg_pos.x * TILE_N;
    uint tile_m_start = tg_pos.y * TILE_M;
    uint split_idx    = tg_pos.z;

    // Compute K range for this split, aligned to TILE_K (which equals Q8_0 group size)
    uint k_per_split = ((K + k_splits - 1) / k_splits);
    k_per_split = ((k_per_split + TILE_K - 1) / TILE_K) * TILE_K;
    uint k_start = split_idx * k_per_split;
    uint k_end = min(k_start + k_per_split, K);
    if (k_start >= K) {
        // This split has no work; zero its output tile
        uint partial_offset = split_idx * M * N;
        for (ushort idx = tiitg; idx < TILE_M * TILE_N; idx += TG_SIZE) {
            uint m_local = idx / TILE_N;
            uint n_local = idx % TILE_N;
            uint gm = tile_m_start + m_local;
            uint gn = tile_n_start + n_local;
            if (gm < M && gn < N)
                Y_partial[partial_offset + gm * N + gn] = 0.0f;
        }
        return;
    }

    ushort sg_r = sgitg / SG_COLS;
    ushort sg_c = sgitg % SG_COLS;

    simdgroup_float8x8 mc[2][2];
    FOR_UNROLL (ushort i = 0; i < 2; i++)
        FOR_UNROLL (ushort j = 0; j < 2; j++)
            mc[i][j] = make_filled_simdgroup_matrix<float, 8>(0.f);

    uint num_blocks_per_row = (K + Q8B_GROUP_SIZE - 1) / Q8B_GROUP_SIZE;
    uint row_bytes = num_blocks_per_row * Q8B_BLOCK_SIZE;

    uint first_kt = k_start / TILE_K;
    uint last_kt = (k_end + TILE_K - 1) / TILE_K;

    for (uint kt = first_kt; kt < last_kt; kt++) {
        uint k_base = kt * TILE_K;

        // Load A tile: vectorized 2D mapping, 4 threads per row, 8 elements each via float4
        // (Split-K variant: also checks gk is within [k_start, k_end) range)
        {
            ushort row = tiitg >> 2;           // tiitg / 4, range 0..31
            ushort col_group = tiitg & 3;      // tiitg % 4, range 0..3
            ushort k_local = col_group << 3;   // col_group * 8

            uint gm = tile_m_start + row;
            uint gk = k_base + k_local;
            threadgroup half* sa_ptr = sa + row * TILE_K + k_local;

            if (gm < M && gk + 7 < K && gk >= k_start && gk + 7 < k_end) {
                // Fast path: all 8 elements valid and within split range
                device const float4* x_ptr = (device const float4*)(X + gm * K + gk);
                float4 v0 = x_ptr[0];
                float4 v1 = x_ptr[1];
                sa_ptr[0] = (half)v0.x; sa_ptr[1] = (half)v0.y;
                sa_ptr[2] = (half)v0.z; sa_ptr[3] = (half)v0.w;
                sa_ptr[4] = (half)v1.x; sa_ptr[5] = (half)v1.y;
                sa_ptr[6] = (half)v1.z; sa_ptr[7] = (half)v1.w;
            } else {
                // Boundary: element-wise with full bounds check
                FOR_UNROLL (ushort i = 0; i < 8; i++) {
                    uint gk_i = gk + i;
                    sa_ptr[i] = (gm < M && gk_i < K && gk_i >= k_start && gk_i < k_end) ? (half)X[gm * K + gk_i] : (half)0.0h;
                }
            }
        }

        // Load B tile: vectorized Q8_0 dequant
        {
            uint block_idx = k_base / Q8B_GROUP_SIZE;
            ushort n_local = tiitg / 4;
            ushort t_in_row = tiitg % 4;
            ushort k_offset = t_in_row * 8;
            uint gn = tile_n_start + n_local;

            threadgroup half* sb_row = sb + n_local * TILE_K + k_offset;

            if (gn < N && k_base + k_offset < K && k_base + k_offset >= k_start && k_base + k_offset < k_end) {
                uint block_offset = gn * row_bytes + block_idx * Q8B_BLOCK_SIZE;
                half scale = as_type<half>(*(device const ushort*)(W_q8 + block_offset));
                device const char* qdata = (device const char*)(W_q8 + block_offset + 2 + k_offset);

                uint k_remaining = K - (k_base + k_offset);
                ushort count = (k_remaining >= 8) ? 8 : (ushort)k_remaining;

                if (count == 8) {
                    sb_row[0] = scale * (half)qdata[0];
                    sb_row[1] = scale * (half)qdata[1];
                    sb_row[2] = scale * (half)qdata[2];
                    sb_row[3] = scale * (half)qdata[3];
                    sb_row[4] = scale * (half)qdata[4];
                    sb_row[5] = scale * (half)qdata[5];
                    sb_row[6] = scale * (half)qdata[6];
                    sb_row[7] = scale * (half)qdata[7];
                } else {
                    for (ushort i = 0; i < count; i++) {
                        sb_row[i] = scale * (half)qdata[i];
                    }
                    for (ushort i = count; i < 8; i++) {
                        sb_row[i] = (half)0.0h;
                    }
                }
            } else {
                FOR_UNROLL (ushort i = 0; i < 8; i++) {
                    sb_row[i] = (half)0.0h;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // MMA
        FOR_UNROLL (ushort ks = 0; ks < TILE_K; ks += 8) {
            simdgroup_half8x8 ma[2];
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(ma[0], sa + (sg_r * 16 + 0) * TILE_K + ks, TILE_K);
            simdgroup_load(ma[1], sa + (sg_r * 16 + 8) * TILE_K + ks, TILE_K);

            simdgroup_half8x8 mb[2];
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(mb[0], sb + (sg_c * 16 + 0) * TILE_K + ks, TILE_K, ulong2(0,0), true);
            simdgroup_load(mb[1], sb + (sg_c * 16 + 8) * TILE_K + ks, TILE_K, ulong2(0,0), true);

            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_multiply_accumulate(mc[0][0], ma[0], mb[0], mc[0][0]);
            simdgroup_multiply_accumulate(mc[0][1], ma[0], mb[1], mc[0][1]);
            simdgroup_multiply_accumulate(mc[1][0], ma[1], mb[0], mc[1][0]);
            simdgroup_multiply_accumulate(mc[1][1], ma[1], mb[1], mc[1][1]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store to partial buffer at offset split_idx * M * N
    uint partial_offset = split_idx * M * N;
    uint sg_m_base = tile_m_start + sg_r * 16;
    uint sg_n_base = tile_n_start + sg_c * 16;

    if (sg_m_base + 16 <= M && sg_n_base + 16 <= N) {
        FOR_UNROLL (ushort i = 0; i < 2; i++)
            FOR_UNROLL (ushort j = 0; j < 2; j++)
                simdgroup_store(mc[i][j], Y_partial + partial_offset + (sg_m_base + i * 8) * N + (sg_n_base + j * 8), N);
    } else {
        threadgroup float* sc = (threadgroup float*)shmem;
        threadgroup float* my_sc = sc + sgitg * 256;
        FOR_UNROLL (ushort i = 0; i < 2; i++)
            FOR_UNROLL (ushort j = 0; j < 2; j++)
                simdgroup_store(mc[i][j], my_sc + (i * 8) * 16 + (j * 8), 16);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ushort lane = tiitg % 32;
        for (ushort idx = lane; idx < 256; idx += 32) {
            ushort local_m = idx / 16;
            ushort local_n = idx % 16;
            uint gm = sg_m_base + local_m;
            uint gn = sg_n_base + local_n;
            if (gm < M && gn < N) {
                Y_partial[partial_offset + gm * N + gn] = my_sc[local_m * 16 + local_n];
            }
        }
    }
}

// ============================================================================
// reduce_splitk: Sum partial results from Split-K GEMM
//
// partials: [k_splits, M, N] -- partial sums from each split
// output:   [M, N] -- final result
//
// Simple element-wise reduction: output[i] = sum(partials[s * M*N + i])
// ============================================================================

kernel void reduce_splitk(
    device const float* partials  [[buffer(0)]],
    device float*       output    [[buffer(1)]],
    constant uint&      M         [[buffer(2)]],
    constant uint&      N         [[buffer(3)]],
    constant uint&      k_splits  [[buffer(4)]],
    uint gid                      [[thread_position_in_grid]])
{
    uint total = M * N;
    if (gid >= total) return;

    float sum = 0.0f;
    for (uint s = 0; s < k_splits; s++) {
        sum += partials[s * total + gid];
    }
    output[gid] = sum;
}

// ============================================================================
// rmsnorm_batched: RMS Normalization for a batch of vectors
//
// Each threadgroup normalizes one row: out[batch_idx, :] = normalize(x[batch_idx, :])
//
// Dispatch: threadgroups = batch_size, threads_per_threadgroup = norm_tg_size
// ============================================================================

kernel void rmsnorm_batched(
    device const float* x        [[buffer(0)]],   // [batch_size, dim]
    device const float* weight   [[buffer(1)]],   // [dim]
    device float*       out      [[buffer(2)]],   // [batch_size, dim]
    constant uint&      dim      [[buffer(3)]],
    constant float&     eps      [[buffer(4)]],
    uint batch_idx               [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]],
    uint tg_size                 [[threads_per_threadgroup]],
    uint simd_lane               [[thread_index_in_simdgroup]],
    uint simd_group              [[simdgroup_index_in_threadgroup]])
{
    device const float* x_row   = x + batch_idx * dim;
    device float*       out_row = out + batch_idx * dim;

    float ss = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        float v = x_row[i];
        ss += v * v;
    }

    ss = simd_sum(ss);

    threadgroup float partial_sums[32];
    if (simd_lane == 0) {
        partial_sums[simd_group] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simd_groups = (tg_size + 31) / 32;
    threadgroup float total_ss;
    if (simd_group == 0) {
        float val = (simd_lane < num_simd_groups) ? partial_sums[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            total_ss = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = rsqrt(total_ss / float(dim) + eps);

    for (uint i = tid; i < dim; i += tg_size) {
        out_row[i] = x_row[i] * scale * weight[i];
    }
}

// ============================================================================
// rmsnorm_batched_bytes: Batched RMSNorm with byte-encoded weights
// ============================================================================

kernel void rmsnorm_batched_bytes(
    device const float* x          [[buffer(0)]],
    device const uchar* w_bytes    [[buffer(1)]],
    device float*       out        [[buffer(2)]],
    constant uint&      dim        [[buffer(3)]],
    constant float&     eps        [[buffer(4)]],
    uint batch_idx                 [[threadgroup_position_in_grid]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_group                [[simdgroup_index_in_threadgroup]])
{
    device const float* weight = (device const float*)w_bytes;
    device const float* x_row  = x + batch_idx * dim;
    device float*       out_row = out + batch_idx * dim;

    float ss = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        float v = x_row[i];
        ss += v * v;
    }

    ss = simd_sum(ss);

    threadgroup float partial_sums[32];
    if (simd_lane == 0) {
        partial_sums[simd_group] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simd_groups = (tg_size + 31) / 32;
    threadgroup float total_ss;
    if (simd_group == 0) {
        float val = (simd_lane < num_simd_groups) ? partial_sums[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            total_ss = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = rsqrt(total_ss / float(dim) + eps);

    for (uint i = tid; i < dim; i += tg_size) {
        out_row[i] = x_row[i] * scale * weight[i];
    }
}

// ============================================================================
// rope_batched: Batched Rotary Position Embeddings
//
// Applies RoPE to a batch of head vectors. Each thread handles one
// (even, odd) pair across the batch.
//
// vec:       [batch_size * num_heads * head_dim] flattened
// pos_start: starting position index for this batch
//
// Dispatch: total threads = batch_size * num_heads * half_dim
// ============================================================================

kernel void rope_batched(
    device float*       vec          [[buffer(0)]],
    device const float* cos_table    [[buffer(1)]],
    device const float* sin_table    [[buffer(2)]],
    constant uint&      num_heads    [[buffer(3)]],
    constant uint&      head_dim     [[buffer(4)]],
    constant uint&      half_dim     [[buffer(5)]],
    constant uint&      pos_start    [[buffer(6)]],
    constant uint&      total_dim    [[buffer(7)]],   // num_heads * head_dim per token
    uint gid                         [[thread_position_in_grid]])
{
    // gid ranges over batch_size * num_heads * half_dim
    uint elems_per_token = num_heads * half_dim;
    uint token_idx = gid / elems_per_token;
    uint within_token = gid % elems_per_token;
    uint head = within_token / half_dim;
    uint i = within_token % half_dim;

    if (head >= num_heads) return;

    uint base = token_idx * total_dim + head * head_dim;
    uint idx0 = base + 2 * i;
    uint idx1 = idx0 + 1;

    float v0 = vec[idx0];
    float v1 = vec[idx1];

    uint pos = pos_start + token_idx;
    uint cos_idx = pos * half_dim + i;
    float c = cos_table[cos_idx];
    float s = sin_table[cos_idx];

    vec[idx0] = v0 * c - v1 * s;
    vec[idx1] = v0 * s + v1 * c;
}

// ============================================================================
// rope_batched_neox: NeoX-style Batched Rotary Position Embeddings
//
// Same as rope_batched but pairs (i, i+half_dim) instead of (2*i, 2*i+1).
// Used by Qwen3.5 (IMROPE) and other NeoX-family models.
//
// Dispatch: total threads = batch_size * num_heads * half_dim
// ============================================================================

kernel void rope_batched_neox(
    device float*       vec          [[buffer(0)]],
    device const float* cos_table    [[buffer(1)]],
    device const float* sin_table    [[buffer(2)]],
    constant uint&      num_heads    [[buffer(3)]],
    constant uint&      head_dim     [[buffer(4)]],
    constant uint&      half_dim     [[buffer(5)]],
    constant uint&      pos_start    [[buffer(6)]],
    constant uint&      total_dim    [[buffer(7)]],   // num_heads * head_dim per token
    uint gid                         [[thread_position_in_grid]])
{
    // gid ranges over batch_size * num_heads * half_dim
    uint elems_per_token = num_heads * half_dim;
    uint token_idx = gid / elems_per_token;
    uint within_token = gid % elems_per_token;
    uint head = within_token / half_dim;
    uint i = within_token % half_dim;

    if (head >= num_heads) return;

    uint base = token_idx * total_dim + head * head_dim;
    uint idx0 = base + i;            // first half: [0, half_dim)
    uint idx1 = base + i + half_dim; // second half: [half_dim, 2*half_dim)

    float v0 = vec[idx0];
    float v1 = vec[idx1];

    uint pos = pos_start + token_idx;
    uint cos_idx = pos * half_dim + i;
    float c = cos_table[cos_idx];
    float s = sin_table[cos_idx];

    vec[idx0] = v0 * c - v1 * s;
    vec[idx1] = v0 * s + v1 * c;
}

// ============================================================================
// add_residual_batched: Element-wise addition for batched activations
//
// dst[i] += src[i] for i in 0..len
// ============================================================================

kernel void add_residual_batched(
    device float*       dst  [[buffer(0)]],
    device const float* src  [[buffer(1)]],
    constant uint&      len  [[buffer(2)]],
    uint gid                 [[thread_position_in_grid]])
{
    if (gid >= len) return;
    dst[gid] += src[gid];
}

// ============================================================================
// swiglu_batched: Fused SiLU(gate) * up for batched activations
//
// gate[i] = silu(gate[i]) * up[i] for i in 0..len
// ============================================================================

kernel void swiglu_batched(
    device float*       gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    constant uint&      len  [[buffer(2)]],
    uint gid                 [[thread_position_in_grid]])
{
    if (gid >= len) return;
    float g = gate[gid];
    float sigmoid = 1.0f / (1.0f + exp(-g));
    gate[gid] = g * sigmoid * up[gid];
}

// ============================================================================
// embed_tokens_batched: Batch token embedding lookup
//
// token_ids: [batch_size] array of token IDs
// out:       [batch_size, hidden_dim] output
// ============================================================================

kernel void embed_tokens_batched(
    device const float* embedding   [[buffer(0)]],
    device float*       out         [[buffer(1)]],
    device const uint*  token_ids   [[buffer(2)]],
    constant uint&      hidden_dim  [[buffer(3)]],
    constant uint&      batch_size  [[buffer(4)]],
    uint gid                        [[thread_position_in_grid]])
{
    uint total = batch_size * hidden_dim;
    if (gid >= total) return;

    uint token_idx = gid / hidden_dim;
    uint dim_idx   = gid % hidden_dim;

    uint tok = token_ids[token_idx];
    out[gid] = embedding[tok * hidden_dim + dim_idx];
}

// ============================================================================
// embed_tokens_batched_f16: F16 batch token embedding lookup (prefill)
// ============================================================================

kernel void embed_tokens_batched_f16(
    device const half*  embedding_f16 [[buffer(0)]],
    device float*       out           [[buffer(1)]],
    device const uint*  token_ids     [[buffer(2)]],
    constant uint&      hidden_dim    [[buffer(3)]],
    constant uint&      batch_size    [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = batch_size * hidden_dim;
    if (gid >= total) return;

    uint token_idx = gid / hidden_dim;
    uint dim_idx   = gid % hidden_dim;

    uint tok = token_ids[token_idx];
    out[gid] = float(embedding_f16[tok * hidden_dim + dim_idx]);
}

// ============================================================================
// embed_tokens_batched_q8_0: Q8_0 batch token embedding lookup (prefill)
// ============================================================================

kernel void embed_tokens_batched_q8_0(
    device const char* embedding_q8  [[buffer(0)]],
    device float*      out           [[buffer(1)]],
    device const uint* token_ids     [[buffer(2)]],
    constant uint&     hidden_dim    [[buffer(3)]],
    constant uint&     batch_size    [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = batch_size * hidden_dim;
    if (gid >= total) return;

    uint token_idx = gid / hidden_dim;
    uint dim_idx = gid % hidden_dim;
    uint tok = token_ids[token_idx];

    uint global_elem = tok * hidden_dim + dim_idx;
    uint block_idx = global_elem >> 5;
    uint elem_in_block = global_elem & 31;

    device const char* block_ptr = embedding_q8 + block_idx * 34;
    float scale = float(*(device const half*)block_ptr);
    float val = float(block_ptr[2 + elem_in_block]);
    out[gid] = val * scale;
}

// ============================================================================
// embed_tokens_batched_q4_0: Q4_0 batch token embedding lookup (prefill)
// ============================================================================

kernel void embed_tokens_batched_q4_0(
    device const char* embedding_q4  [[buffer(0)]],
    device float*      out           [[buffer(1)]],
    device const uint* token_ids     [[buffer(2)]],
    constant uint&     hidden_dim    [[buffer(3)]],
    constant uint&     batch_size    [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = batch_size * hidden_dim;
    if (gid >= total) return;

    uint token_idx = gid / hidden_dim;
    uint dim_idx = gid % hidden_dim;
    uint tok = token_ids[token_idx];

    uint global_elem = tok * hidden_dim + dim_idx;
    uint block_idx = global_elem >> 5;
    uint elem_in_block = global_elem & 31;

    device const uchar* block_ptr = (device const uchar*)(embedding_q4 + block_idx * 18);
    float scale = float(*(device const half*)block_ptr);

    // De-interleaved: elem 0-15 use lo nibble, elem 16-31 use hi nibble
    uint byte_idx = (elem_in_block < 16) ? elem_in_block : (elem_in_block - 16);
    uchar packed = block_ptr[2 + byte_idx];
    int nibble = (elem_in_block < 16) ? (int)(packed & 0xF) - 8 : (int)(packed >> 4) - 8;
    out[gid] = float(nibble) * scale;
}

// ============================================================================
// kv_cache_write_batched: Write K or V vectors for a batch to the KV cache
//
// src:  [batch_size, kv_dim] -- projected K or V vectors
// dst:  [max_seq_len, kv_dim] -- KV cache (contiguous)
//
// Writes batch_size vectors to positions [start_pos..start_pos+batch_size]
// ============================================================================

kernel void kv_cache_write_batched(
    device const float* src        [[buffer(0)]],
    device half*        dst        [[buffer(1)]],
    constant uint&      kv_dim     [[buffer(2)]],
    constant uint&      start_pos  [[buffer(3)]],
    constant uint&      batch_size [[buffer(4)]],
    uint gid                       [[thread_position_in_grid]])
{
    uint total = batch_size * kv_dim;
    if (gid >= total) return;

    uint token_idx = gid / kv_dim;
    uint dim_idx   = gid % kv_dim;

    uint dst_pos = start_pos + token_idx;
    dst[dst_pos * kv_dim + dim_idx] = half(src[gid]);
}

// ============================================================================
// v_cache_write_batched: Write V vectors for a batch to the transposed V cache
//
// src:  [batch_size, kv_dim] -- projected V vectors (row-major)
// dst:  [kv_dim, max_seq_len] -- V cache (transposed for contiguous time reads)
//
// Writes batch_size vectors to positions [start_pos..start_pos+batch_size]
// ============================================================================

kernel void v_cache_write_batched(
    device const float* src         [[buffer(0)]],
    device half*        dst         [[buffer(1)]],
    constant uint&      kv_dim      [[buffer(2)]],
    constant uint&      start_pos   [[buffer(3)]],
    constant uint&      batch_size  [[buffer(4)]],
    constant uint&      max_seq_len [[buffer(5)]],
    uint gid                        [[thread_position_in_grid]])
{
    uint total = batch_size * kv_dim;
    if (gid >= total) return;

    uint token_idx = gid / kv_dim;
    uint dim_idx   = gid % kv_dim;

    uint dst_pos = start_pos + token_idx;
    // Transposed layout: dst[dim_idx, dst_pos] = dst[dim_idx * max_seq_len + dst_pos]
    dst[dim_idx * max_seq_len + dst_pos] = half(src[gid]);
}

// ============================================================================
// attention_scores_batched: Batched Q.K^T for prefill with causal mask
//
// For each token t in the batch, computes attention scores against all
// previous positions (0..start_pos+t) with causal masking.
//
// One threadgroup per (time_step, head, token) triple. Reduces the dot
// product within the threadgroup using simd_sum.
// ============================================================================

kernel void attention_scores_batched(
    device const float* q_batch     [[buffer(0)]],   // [batch_size, num_q_heads * head_dim]
    device const half*  k_cache     [[buffer(1)]],   // [max_seq_len, kv_dim] (f16)
    device half*        scores      [[buffer(2)]],   // [batch_size, num_q_heads, max_seq_len] (f16)
    constant uint&      head_dim    [[buffer(3)]],
    constant uint&      kv_dim      [[buffer(4)]],
    constant uint&      num_q_heads [[buffer(5)]],
    constant uint&      num_kv_heads [[buffer(6)]],
    constant float&     scale       [[buffer(7)]],
    constant uint&      start_pos   [[buffer(8)]],
    constant uint&      max_seq_len [[buffer(9)]],
    constant uint&      batch_size  [[buffer(10)]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]],
    uint3 tid_vec                   [[thread_position_in_threadgroup]],
    uint3 tg_size_vec               [[threads_per_threadgroup]],
    uint simd_lane                  [[thread_index_in_simdgroup]],
    uint simd_group                 [[simdgroup_index_in_threadgroup]])
{
    uint tid = tid_vec.x;
    uint tg_size = tg_size_vec.x;
    uint t_step = tg_pos.x;  // which time step to compute score for
    uint head   = tg_pos.y;  // which Q head
    uint token  = tg_pos.z;  // which token in the batch

    // Causal: token at position (start_pos + token) can attend to 0..start_pos+token
    uint attend_len = start_pos + token + 1;
    if (t_step >= attend_len) return;
    if (head >= num_q_heads) return;
    if (token >= batch_size) return;

    uint gqa_ratio = num_q_heads / num_kv_heads;
    uint kv_h = head / gqa_ratio;

    uint q_dim_total = num_q_heads * head_dim;
    device const float* q_vec = q_batch + token * q_dim_total + head * head_dim;
    device const half* k_vec = k_cache + t_step * kv_dim + kv_h * head_dim;

    float dot = 0.0f;
    for (uint d = tid; d < head_dim; d += tg_size) {
        dot += q_vec[d] * float(k_vec[d]);
    }

    dot = simd_sum(dot);

    threadgroup float partial_sums[32];
    if (simd_lane == 0) {
        partial_sums[simd_group] = dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simd_groups = (tg_size + 31) / 32;
    if (simd_group == 0) {
        float val = (simd_lane < num_simd_groups) ? partial_sums[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            uint score_idx = token * num_q_heads * max_seq_len + head * max_seq_len + t_step;
            scores[score_idx] = (half)(val * scale);
        }
    }
}

// ============================================================================
// softmax_batched: Softmax with causal mask for each (token, head) pair
//
// Dispatch: one threadgroup per (head, token) pair.
// ============================================================================

kernel void softmax_batched(
    device half*   scores      [[buffer(0)]],   // f16 scores buffer
    constant uint& max_seq_len [[buffer(1)]],
    constant uint& num_heads   [[buffer(2)]],
    constant uint& start_pos   [[buffer(3)]],
    constant uint& batch_size  [[buffer(4)]],
    uint2 tg_pos               [[threadgroup_position_in_grid]],
    uint2 tid_vec              [[thread_position_in_threadgroup]],
    uint2 tg_size_vec          [[threads_per_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_group            [[simdgroup_index_in_threadgroup]])
{
    uint tid = tid_vec.x;
    uint tg_size = tg_size_vec.x;
    uint head  = tg_pos.x;
    uint token = tg_pos.y;

    if (head >= num_heads) return;
    if (token >= batch_size) return;

    uint attend_len = start_pos + token + 1;
    uint row_offset = token * num_heads * max_seq_len + head * max_seq_len;
    device half* row = scores + row_offset;

    // Phase 1: find max (accumulate in f32 for numerical stability)
    float local_max = -INFINITY;
    for (uint i = tid; i < attend_len; i += tg_size) {
        local_max = max(local_max, (float)row[i]);
    }
    local_max = simd_max(local_max);

    threadgroup float partial_max[32];
    if (simd_lane == 0) {
        partial_max[simd_group] = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simd_groups = (tg_size + 31) / 32;
    threadgroup float global_max;
    if (simd_group == 0) {
        float val = (simd_lane < num_simd_groups) ? partial_max[simd_lane] : -INFINITY;
        val = simd_max(val);
        if (simd_lane == 0) global_max = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: exp(x - max) and sum (f32 accumulation, write back as f16)
    float local_sum = 0.0f;
    for (uint i = tid; i < attend_len; i += tg_size) {
        float e = exp((float)row[i] - global_max);
        row[i] = (half)e;
        local_sum += e;
    }
    local_sum = simd_sum(local_sum);

    threadgroup float partial_sums[32];
    if (simd_lane == 0) {
        partial_sums[simd_group] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float global_sum;
    if (simd_group == 0) {
        float val = (simd_lane < num_simd_groups) ? partial_sums[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) global_sum = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: normalize (read f16, compute in f32, write back f16)
    float inv_sum = 1.0f / global_sum;
    for (uint i = tid; i < attend_len; i += tg_size) {
        row[i] = (half)((float)row[i] * inv_sum);
    }
}

// ============================================================================
// attention_output_batched: Weighted sum of V vectors for batched prefill
//
// For each (token, head, d) element, computes:
//   out[token, head, d] = sum_t scores[token, head, t] * v_cache[t, kv_h, d]
//
// Dispatch: one thread per output element.
// ============================================================================

kernel void attention_output_batched(
    device const half*  scores       [[buffer(0)]],   // f16 scores
    device const half*  v_cache      [[buffer(1)]],   // f16 V cache
    device float*       out          [[buffer(2)]],
    constant uint&      head_dim     [[buffer(3)]],
    constant uint&      kv_dim       [[buffer(4)]],
    constant uint&      num_q_heads  [[buffer(5)]],
    constant uint&      num_kv_heads [[buffer(6)]],
    constant uint&      start_pos    [[buffer(7)]],
    constant uint&      max_seq_len  [[buffer(8)]],
    constant uint&      batch_size   [[buffer(9)]],
    constant uint&      v_max_seq_len [[buffer(10)]],
    uint gid                         [[thread_position_in_grid]])
{
    uint q_dim = num_q_heads * head_dim;
    uint total = batch_size * q_dim;
    if (gid >= total) return;

    uint token = gid / q_dim;
    uint within = gid % q_dim;
    uint head   = within / head_dim;
    uint d      = within % head_dim;

    uint gqa_ratio = num_q_heads / num_kv_heads;
    uint kv_h = head / gqa_ratio;
    uint attend_len = start_pos + token + 1;

    uint scores_base = token * num_q_heads * max_seq_len + head * max_seq_len;

    // V cache transposed: [kv_dim, v_max_seq_len], contiguous along time (f16)
    device const half* v_row = v_cache + (kv_h * head_dim + d) * v_max_seq_len;
    float sum = 0.0f;
    for (uint t = 0; t < attend_len; t++) {
        sum += (float)scores[scores_base + t] * float(v_row[t]);
    }
    out[gid] = sum;
}

// ============================================================================
// attention_scores_tiled: Tiled GEMM for Q * K^T attention scores (prefill)
//
// Per Q-head: Scores[M,N] = Q[M,K] * K_cache[N,K]^T * scale + causal_mask
// Where M=batch_size, N=max_attend_len, K=head_dim
//
// Q layout:     q_batch[token * q_stride + d]  (q_stride = num_q_heads * head_dim)
// K layout:     k_cache[t_step * kv_dim + kv_h * head_dim + d]
// Score layout: scores[token * num_q_heads * max_seq_len + head * max_seq_len + t_step]
//
// Dispatch: (ceil(N/TILE_N), ceil(M/TILE_M), num_q_heads), TG=(128,1,1)
// tg_pos.z = q_head index; GQA maps to kv_head via head / gqa_ratio
// ============================================================================

kernel void attention_scores_tiled(
    device const float* q_batch     [[buffer(0)]],   // [batch_size, num_q_heads * head_dim]
    device const half*  k_cache     [[buffer(1)]],   // [max_seq_len, kv_dim] (f16)
    device half*        scores      [[buffer(2)]],   // [batch_size, num_q_heads, max_seq_len] (f16)
    constant uint&      head_dim    [[buffer(3)]],
    constant uint&      kv_dim      [[buffer(4)]],
    constant uint&      num_q_heads [[buffer(5)]],
    constant uint&      num_kv_heads [[buffer(6)]],
    constant float&     scale       [[buffer(7)]],
    constant uint&      start_pos   [[buffer(8)]],
    constant uint&      max_seq_len [[buffer(9)]],
    constant uint&      batch_size  [[buffer(10)]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]],
    ushort tiitg                    [[thread_index_in_threadgroup]],
    ushort sgitg                    [[simdgroup_index_in_threadgroup]])
{
    // Tile constants: TILE_M=32, TILE_N=32, TILE_K=32, 4 SG (128 threads)
    // shmem: sa[TILE_M * TILE_K] + sb[TILE_N * TILE_K] = 2048 halfs
    threadgroup half shmem_attn[2048];
    threadgroup half* sa = shmem_attn;
    threadgroup half* sb = shmem_attn + TILE_M * TILE_K;

    uint tile_m_start = tg_pos.y * TILE_M;   // token dim
    uint tile_n_start = tg_pos.x * TILE_N;   // time_step dim
    uint q_head = tg_pos.z;                   // Q head index

    if (q_head >= num_q_heads) return;

    // Early exit: if the entire tile is above the causal diagonal, write zeros
    // The minimum attend_len in this tile = start_pos + tile_m_start + 1
    // If tile_n_start >= min_attend_len, the whole tile is masked
    uint min_attend_in_tile = start_pos + tile_m_start + 1;
    if (tile_n_start >= min_attend_in_tile) {
        // Write zeros for the entire tile
        uint score_stride = num_q_heads * max_seq_len;
        for (ushort idx = tiitg; idx < TILE_M * TILE_N; idx += 128) {
            ushort local_m = idx / TILE_N;
            ushort local_n = idx % TILE_N;
            uint gm = tile_m_start + local_m;
            uint gn = tile_n_start + local_n;
            if (gm < batch_size && gn < max_seq_len) {
                scores[gm * score_stride + q_head * max_seq_len + gn] = (half)0.0h;
            }
        }
        return;
    }

    uint gqa_ratio = num_q_heads / num_kv_heads;
    uint kv_h = q_head / gqa_ratio;

    // Q row stride: num_q_heads * head_dim (interleaved heads)
    uint q_stride = num_q_heads * head_dim;
    // Per-head Q base offset
    uint q_head_off = q_head * head_dim;
    // Per-kv-head K base offset
    uint k_head_off = kv_h * head_dim;

    ushort sg_r = sgitg / SG_COLS;
    ushort sg_c = sgitg % SG_COLS;

    simdgroup_float8x8 mc[2][2];
    FOR_UNROLL (ushort i = 0; i < 2; i++)
        FOR_UNROLL (ushort j = 0; j < 2; j++)
            mc[i][j] = make_filled_simdgroup_matrix<float, 8>(0.f);

    uint num_k_tiles = (head_dim + TILE_K - 1) / TILE_K;

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        uint k_base = kt * TILE_K;

        // Load A tile (Q) with pre-scaling: sa[row, k] = Q[token, head, k] * scale
        // Pre-scaling Q eliminates the per-element multiply in the store phase.
        // 128 threads, 32 rows x 32 cols, 4 threads per row, 8 elements each
        {
            ushort row = tiitg >> 2;
            ushort col_group = tiitg & 3;
            ushort k_start = col_group << 3;

            uint gm = tile_m_start + row;    // token index
            uint gk = k_base + k_start;      // head_dim offset
            threadgroup half* sa_ptr = sa + row * TILE_K + k_start;

            if (gm < batch_size && gk + 7 < head_dim) {
                device const float* q_ptr = q_batch + gm * q_stride + q_head_off + gk;
                float4 v0 = *(device const float4*)(q_ptr) * scale;
                float4 v1 = *(device const float4*)(q_ptr + 4) * scale;
                sa_ptr[0] = (half)v0.x; sa_ptr[1] = (half)v0.y;
                sa_ptr[2] = (half)v0.z; sa_ptr[3] = (half)v0.w;
                sa_ptr[4] = (half)v1.x; sa_ptr[5] = (half)v1.y;
                sa_ptr[6] = (half)v1.z; sa_ptr[7] = (half)v1.w;
            } else {
                FOR_UNROLL (ushort i = 0; i < 8; i++) {
                    uint gk_i = gk + i;
                    sa_ptr[i] = (gm < batch_size && gk_i < head_dim)
                        ? (half)(q_batch[gm * q_stride + q_head_off + gk_i] * scale)
                        : (half)0.0h;
                }
            }
        }

        // Load B tile (K): sb[n_local, k_local] = K[t_step, kv_head, k]
        // B is stored as K[t_step, kv_dim] in f16, we need K^T for GEMM
        // sb layout: [TILE_N rows x TILE_K cols], row-major
        {
            ushort row = tiitg >> 2;           // 0..31 = n_local (time_step within tile)
            ushort col_group = tiitg & 3;      // 0..3
            ushort k_start = col_group << 3;   // 0,8,16,24

            uint gn = tile_n_start + row;      // global time_step
            uint gk = k_base + k_start;
            uint max_attend = start_pos + batch_size;  // maximum possible attend length

            threadgroup half* sb_ptr = sb + row * TILE_K + k_start;

            if (gn < max_attend && gk + 7 < head_dim) {
                // K cache is already f16 -- load directly as half4
                device const half* k_ptr = k_cache + gn * kv_dim + k_head_off + gk;
                half4 v0 = *(device const half4*)(k_ptr);
                half4 v1 = *(device const half4*)(k_ptr + 4);
                sb_ptr[0] = v0.x; sb_ptr[1] = v0.y;
                sb_ptr[2] = v0.z; sb_ptr[3] = v0.w;
                sb_ptr[4] = v1.x; sb_ptr[5] = v1.y;
                sb_ptr[6] = v1.z; sb_ptr[7] = v1.w;
            } else {
                FOR_UNROLL (ushort i = 0; i < 8; i++) {
                    uint gk_i = gk + i;
                    sb_ptr[i] = (gn < max_attend && gk_i < head_dim)
                        ? k_cache[gn * kv_dim + k_head_off + gk_i]
                        : (half)0.0h;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // MMA: each simdgroup computes its 16x16 sub-tile
        FOR_UNROLL (ushort ks = 0; ks < TILE_K; ks += 8) {
            simdgroup_half8x8 ma[2];
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(ma[0], sa + (sg_r * 16 + 0) * TILE_K + ks, TILE_K);
            simdgroup_load(ma[1], sa + (sg_r * 16 + 8) * TILE_K + ks, TILE_K);

            simdgroup_half8x8 mb[2];
            simdgroup_barrier(mem_flags::mem_none);
            // B loaded as [N, K] row-major, need transpose for A * B^T
            simdgroup_load(mb[0], sb + (sg_c * 16 + 0) * TILE_K + ks, TILE_K, ulong2(0,0), true);
            simdgroup_load(mb[1], sb + (sg_c * 16 + 8) * TILE_K + ks, TILE_K, ulong2(0,0), true);

            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_multiply_accumulate(mc[0][0], ma[0], mb[0], mc[0][0]);
            simdgroup_multiply_accumulate(mc[0][1], ma[0], mb[1], mc[0][1]);
            simdgroup_multiply_accumulate(mc[1][0], ma[1], mb[0], mc[1][0]);
            simdgroup_multiply_accumulate(mc[1][1], ma[1], mb[1], mc[1][1]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store results with causal mask (scale already applied via pre-scaled Q)
    // Each simdgroup writes its 16x16 sub-tile
    uint sg_m_base = tile_m_start + sg_r * 16;  // token base
    uint sg_n_base = tile_n_start + sg_c * 16;  // time_step base

    // Determine if this simdgroup sub-tile needs causal masking:
    // max attend for any row in sub-tile = start_pos + min(sg_m_base + 15, batch_size - 1) + 1
    // If sg_n_base + 15 < min_attend, all positions are valid (no masking needed)
    uint sg_min_attend = start_pos + sg_m_base + 1;  // attend_len for first row in sub-tile
    bool fully_valid = (sg_n_base + 16 <= sg_min_attend)
                    && (sg_m_base + 16 <= batch_size)
                    && (sg_n_base + 16 <= max_seq_len);

    if (fully_valid) {
        // Fast path: entire sub-tile is valid and in-bounds.
        // Store to threadgroup staging, then write directly to device (no mask check).
        threadgroup float* sc = (threadgroup float*)shmem_attn;
        threadgroup float* my_sc = sc + sgitg * 256;
        FOR_UNROLL (ushort i = 0; i < 2; i++)
            FOR_UNROLL (ushort j = 0; j < 2; j++)
                simdgroup_store(mc[i][j], my_sc + (i * 8) * 16 + (j * 8), 16);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint score_stride = num_q_heads * max_seq_len;
        ushort lane = tiitg % 32;
        for (ushort idx = lane; idx < 256; idx += 32) {
            ushort local_m = idx / 16;
            ushort local_n = idx % 16;
            uint gm = sg_m_base + local_m;
            uint gn = sg_n_base + local_n;
            scores[gm * score_stride + q_head * max_seq_len + gn] = (half)my_sc[local_m * 16 + local_n];
        }
    } else {
        // Boundary path: need per-element causal mask check
        threadgroup float* sc = (threadgroup float*)shmem_attn;
        threadgroup float* my_sc = sc + sgitg * 256;
        FOR_UNROLL (ushort i = 0; i < 2; i++)
            FOR_UNROLL (ushort j = 0; j < 2; j++)
                simdgroup_store(mc[i][j], my_sc + (i * 8) * 16 + (j * 8), 16);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint score_stride = num_q_heads * max_seq_len;
        ushort lane = tiitg % 32;
        for (ushort idx = lane; idx < 256; idx += 32) {
            ushort local_m = idx / 16;
            ushort local_n = idx % 16;
            uint gm = sg_m_base + local_m;
            uint gn = sg_n_base + local_n;

            if (gm < batch_size && gn < max_seq_len) {
                uint attend_len = start_pos + gm + 1;
                float val = (gn < attend_len) ? my_sc[local_m * 16 + local_n] : 0.0f;
                scores[gm * score_stride + q_head * max_seq_len + gn] = (half)val;
            }
        }
    }
}

// ============================================================================
// attention_output_tiled: Tiled GEMM for Scores * V weighted sum (prefill)
//
// Per Q-head: Out[M,N] = Scores[M,K_eff] * V[K_eff,N]
// Where M=batch_size, N=head_dim, K_eff=max_attend_len (after softmax)
//
// Scores layout: scores[token * num_q_heads * max_seq_len + head * max_seq_len + t]
// V layout:      v_cache[t * kv_dim + kv_h * head_dim + d]
// Out layout:    out[token * q_dim + head * head_dim + d]
//
// Dispatch: (ceil(head_dim/TILE_N), ceil(batch_size/TILE_M), num_q_heads), TG=(128,1,1)
// ============================================================================

kernel void attention_output_tiled(
    device const half*  scores       [[buffer(0)]],   // [batch_size, num_q_heads, max_seq_len] (f16)
    device const half*  v_cache      [[buffer(1)]],   // [kv_dim, v_max_seq_len] (transposed, f16)
    device float*       out          [[buffer(2)]],   // [batch_size, num_q_heads * head_dim]
    constant uint&      head_dim     [[buffer(3)]],
    constant uint&      kv_dim       [[buffer(4)]],
    constant uint&      num_q_heads  [[buffer(5)]],
    constant uint&      num_kv_heads [[buffer(6)]],
    constant uint&      start_pos    [[buffer(7)]],
    constant uint&      max_seq_len  [[buffer(8)]],
    constant uint&      batch_size   [[buffer(9)]],
    constant uint&      v_max_seq_len [[buffer(10)]],
    uint3 tg_pos                     [[threadgroup_position_in_grid]],
    ushort tiitg                     [[thread_index_in_threadgroup]],
    ushort sgitg                     [[simdgroup_index_in_threadgroup]])
{
    threadgroup half shmem_attn_out[2048];
    threadgroup half* sa = shmem_attn_out;
    threadgroup half* sb = shmem_attn_out + TILE_M * TILE_K;

    uint tile_m_start = tg_pos.y * TILE_M;   // token dim
    uint tile_n_start = tg_pos.x * TILE_N;   // head_dim output dim
    uint q_head = tg_pos.z;

    if (q_head >= num_q_heads) return;

    uint gqa_ratio = num_q_heads / num_kv_heads;
    uint kv_h = q_head / gqa_ratio;

    uint q_dim = num_q_heads * head_dim;

    // Scores row stride: max_seq_len (per head)
    // V stride between time steps: kv_dim

    ushort sg_r = sgitg / SG_COLS;
    ushort sg_c = sgitg % SG_COLS;

    simdgroup_float8x8 mc[2][2];
    FOR_UNROLL (ushort i = 0; i < 2; i++)
        FOR_UNROLL (ushort j = 0; j < 2; j++)
            mc[i][j] = make_filled_simdgroup_matrix<float, 8>(0.f);

    // K dimension = attend_len (up to start_pos + batch_size)
    uint max_attend = start_pos + batch_size;
    uint num_k_tiles = (max_attend + TILE_K - 1) / TILE_K;

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        uint k_base = kt * TILE_K;

        // Load A tile (Scores): sa[row, k] = scores[token, head, t_step]
        // Scores layout: scores[token * num_q_heads * max_seq_len + q_head * max_seq_len + t]
        {
            ushort row = tiitg >> 2;
            ushort col_group = tiitg & 3;
            ushort k_start = col_group << 3;

            uint gm = tile_m_start + row;
            uint gk = k_base + k_start;
            threadgroup half* sa_ptr = sa + row * TILE_K + k_start;

            // Each token has its own attend_len = start_pos + token + 1
            // But softmax already zeroed beyond attend_len, so we can load freely
            // (just bounds-check against max_seq_len)
            if (gm < batch_size && gk + 7 < max_attend) {
                uint s_base = gm * num_q_heads * max_seq_len + q_head * max_seq_len + gk;
                device const half4* s_ptr = (device const half4*)(scores + s_base);
                half4 v0 = s_ptr[0];
                half4 v1 = s_ptr[1];
                sa_ptr[0] = v0.x; sa_ptr[1] = v0.y;
                sa_ptr[2] = v0.z; sa_ptr[3] = v0.w;
                sa_ptr[4] = v1.x; sa_ptr[5] = v1.y;
                sa_ptr[6] = v1.z; sa_ptr[7] = v1.w;
            } else {
                FOR_UNROLL (ushort i = 0; i < 8; i++) {
                    uint gk_i = gk + i;
                    sa_ptr[i] = (gm < batch_size && gk_i < max_attend)
                        ? scores[gm * num_q_heads * max_seq_len + q_head * max_seq_len + gk_i]
                        : (half)0.0h;
                }
            }
        }

        // Load B tile (V): sb[n_local, k_local] = V[d, t_step] (transposed layout, f16)
        // V cache is [kv_dim, v_max_seq_len]: contiguous along time dimension (f16)
        // B[n, k] = V[(kv_h*head_dim + tile_n_start + n_local), k_base + k_local]
        // sb[n_local, k_local] = v_cache[(kv_h*head_dim + gn) * v_max_seq_len + gk]
        {
            ushort row = tiitg >> 2;           // n_local (head_dim within tile)
            ushort col_group = tiitg & 3;
            ushort k_start = col_group << 3;

            uint gn = tile_n_start + row;      // head_dim index
            uint gk = k_base + k_start;        // time_step index
            threadgroup half* sb_ptr = sb + row * TILE_K + k_start;

            if (gn < head_dim && gk + 7 < max_attend) {
                // Load 8 consecutive V values: contiguous in transposed layout, already f16!
                device const half* v_row = v_cache + (kv_h * head_dim + gn) * v_max_seq_len + gk;
                half4 v0 = *(device const half4*)(v_row);
                half4 v1 = *(device const half4*)(v_row + 4);
                sb_ptr[0] = v0.x; sb_ptr[1] = v0.y;
                sb_ptr[2] = v0.z; sb_ptr[3] = v0.w;
                sb_ptr[4] = v1.x; sb_ptr[5] = v1.y;
                sb_ptr[6] = v1.z; sb_ptr[7] = v1.w;
            } else {
                FOR_UNROLL (ushort i = 0; i < 8; i++) {
                    uint gk_i = gk + i;
                    sb_ptr[i] = (gn < head_dim && gk_i < max_attend)
                        ? v_cache[(kv_h * head_dim + gn) * v_max_seq_len + gk_i]
                        : (half)0.0h;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // MMA
        FOR_UNROLL (ushort ks = 0; ks < TILE_K; ks += 8) {
            simdgroup_half8x8 ma[2];
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(ma[0], sa + (sg_r * 16 + 0) * TILE_K + ks, TILE_K);
            simdgroup_load(ma[1], sa + (sg_r * 16 + 8) * TILE_K + ks, TILE_K);

            simdgroup_half8x8 mb[2];
            simdgroup_barrier(mem_flags::mem_none);
            // B = V transposed: [head_dim, attend_len], load with transpose
            simdgroup_load(mb[0], sb + (sg_c * 16 + 0) * TILE_K + ks, TILE_K, ulong2(0,0), true);
            simdgroup_load(mb[1], sb + (sg_c * 16 + 8) * TILE_K + ks, TILE_K, ulong2(0,0), true);

            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_multiply_accumulate(mc[0][0], ma[0], mb[0], mc[0][0]);
            simdgroup_multiply_accumulate(mc[0][1], ma[0], mb[1], mc[0][1]);
            simdgroup_multiply_accumulate(mc[1][0], ma[1], mb[0], mc[1][0]);
            simdgroup_multiply_accumulate(mc[1][1], ma[1], mb[1], mc[1][1]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store results
    uint sg_m_base = tile_m_start + sg_r * 16;  // token base
    uint sg_n_base = tile_n_start + sg_c * 16;  // head_dim base

    // Fast path: all within bounds
    if (sg_m_base + 16 <= batch_size && sg_n_base + 16 <= head_dim) {
        // Store to staging, then write with stride
        threadgroup float* sc = (threadgroup float*)shmem_attn_out;
        threadgroup float* my_sc = sc + sgitg * 256;
        FOR_UNROLL (ushort i = 0; i < 2; i++)
            FOR_UNROLL (ushort j = 0; j < 2; j++)
                simdgroup_store(mc[i][j], my_sc + (i * 8) * 16 + (j * 8), 16);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        ushort lane = tiitg % 32;
        for (ushort idx = lane; idx < 256; idx += 32) {
            ushort local_m = idx / 16;
            ushort local_n = idx % 16;
            uint gm = sg_m_base + local_m;
            uint gn = sg_n_base + local_n;
            out[gm * q_dim + q_head * head_dim + gn] = my_sc[local_m * 16 + local_n];
        }
    } else {
        // Boundary path
        threadgroup float* sc = (threadgroup float*)shmem_attn_out;
        threadgroup float* my_sc = sc + sgitg * 256;
        FOR_UNROLL (ushort i = 0; i < 2; i++)
            FOR_UNROLL (ushort j = 0; j < 2; j++)
                simdgroup_store(mc[i][j], my_sc + (i * 8) * 16 + (j * 8), 16);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        ushort lane = tiitg % 32;
        for (ushort idx = lane; idx < 256; idx += 32) {
            ushort local_m = idx / 16;
            ushort local_n = idx % 16;
            uint gm = sg_m_base + local_m;
            uint gn = sg_n_base + local_n;
            if (gm < batch_size && gn < head_dim)
                out[gm * q_dim + q_head * head_dim + gn] = my_sc[local_m * 16 + local_n];
        }
    }
}

// ============================================================================
// copy_buffer: Simple buffer copy (src → dst)
//
// Used to copy attn_proj_buf → x_buf between layers within a command buffer,
// avoiding CPU round-trip.
// ============================================================================
// dequant_tiled_matmul_q8_0_residual_batched: Tiled Q8_0 GEMM + residual add
//
// Y[m,n] = X[M,K] * dequant(W_q8[N,K_bytes])^T + R[m,n]
//
// Identical to dequant_tiled_matmul_q8_0 except the writeback adds residual[m,n].
// Eliminates a separate add_residual dispatch + encoder barrier per layer.
// ============================================================================

kernel void dequant_tiled_matmul_q8_0_residual_batched(
    device const uchar* W_q8     [[buffer(0)]],   // Q8_0 weights [N, K_bytes]
    device const float* X        [[buffer(1)]],   // [M, K] input batch
    device float*       Y        [[buffer(2)]],   // [M, N] output batch
    constant uint&      M        [[buffer(3)]],   // batch size
    constant uint&      N        [[buffer(4)]],   // output dim
    constant uint&      K        [[buffer(5)]],   // input dim (elements, not bytes)
    device const float* R        [[buffer(6)]],   // [M, N] residual to add
    uint2 tg_pos                 [[threadgroup_position_in_grid]],
    ushort tiitg                 [[thread_index_in_threadgroup]],
    ushort sgitg                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup half shmem[2048];
    threadgroup half* sa = shmem;
    threadgroup half* sb = shmem + TILE_M * TILE_K;

    uint tile_m_start = tg_pos.y * TILE_M;
    uint tile_n_start = tg_pos.x * TILE_N;

    ushort sg_r = sgitg / SG_COLS;
    ushort sg_c = sgitg % SG_COLS;

    simdgroup_float8x8 mc[2][2];
    FOR_UNROLL (ushort i = 0; i < 2; i++)
        FOR_UNROLL (ushort j = 0; j < 2; j++)
            mc[i][j] = make_filled_simdgroup_matrix<float, 8>(0.f);

    uint num_blocks_per_row = (K + Q8B_GROUP_SIZE - 1) / Q8B_GROUP_SIZE;
    uint row_bytes = num_blocks_per_row * Q8B_BLOCK_SIZE;

    uint num_k_tiles = (K + TILE_K - 1) / TILE_K;

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        uint k_base = kt * TILE_K;

        // Load A tile
        // FC_BC_M/FC_BC_K: when false (aligned), compiler eliminates boundary checks
        {
            ushort row = tiitg >> 2;
            ushort col_group = tiitg & 3;
            ushort k_start = col_group << 3;

            uint gm = tile_m_start + row;
            uint gk = k_base + k_start;
            threadgroup half* sa_ptr = sa + row * TILE_K + k_start;

            if (FC_BC_M || FC_BC_K) {
                if (gm < M && gk + 7 < K) {
                    device const float4* x_ptr = (device const float4*)(X + gm * K + gk);
                    float4 v0 = x_ptr[0];
                    float4 v1 = x_ptr[1];
                    sa_ptr[0] = (half)v0.x; sa_ptr[1] = (half)v0.y;
                    sa_ptr[2] = (half)v0.z; sa_ptr[3] = (half)v0.w;
                    sa_ptr[4] = (half)v1.x; sa_ptr[5] = (half)v1.y;
                    sa_ptr[6] = (half)v1.z; sa_ptr[7] = (half)v1.w;
                } else {
                    FOR_UNROLL (ushort i = 0; i < 8; i++) {
                        uint gk_i = gk + i;
                        sa_ptr[i] = (gm < M && gk_i < K) ? (half)X[gm * K + gk_i] : (half)0.0h;
                    }
                }
            } else {
                device const float4* x_ptr = (device const float4*)(X + gm * K + gk);
                float4 v0 = x_ptr[0];
                float4 v1 = x_ptr[1];
                sa_ptr[0] = (half)v0.x; sa_ptr[1] = (half)v0.y;
                sa_ptr[2] = (half)v0.z; sa_ptr[3] = (half)v0.w;
                sa_ptr[4] = (half)v1.x; sa_ptr[5] = (half)v1.y;
                sa_ptr[6] = (half)v1.z; sa_ptr[7] = (half)v1.w;
            }
        }

        // Load B tile: vectorized Q8_0 dequant
        // FC_BC_N/FC_BC_K: when false (aligned), compiler eliminates boundary checks
        {
            uint block_idx = k_base / Q8B_GROUP_SIZE;
            ushort n_local = tiitg / 4;
            ushort t_in_row = tiitg % 4;
            ushort k_offset = t_in_row * 8;
            uint gn = tile_n_start + n_local;

            threadgroup half* sb_row = sb + n_local * TILE_K + k_offset;

            if (FC_BC_N || FC_BC_K) {
                if (gn < N && k_base + k_offset < K) {
                    uint block_offset = gn * row_bytes + block_idx * Q8B_BLOCK_SIZE;
                    half scale = as_type<half>(*(device const ushort*)(W_q8 + block_offset));
                    device const char* qdata = (device const char*)(W_q8 + block_offset + 2 + k_offset);

                    uint k_remaining = K - (k_base + k_offset);
                    ushort count = (k_remaining >= 8) ? 8 : (ushort)k_remaining;

                    if (count == 8) {
                        sb_row[0] = scale * (half)qdata[0];
                        sb_row[1] = scale * (half)qdata[1];
                        sb_row[2] = scale * (half)qdata[2];
                        sb_row[3] = scale * (half)qdata[3];
                        sb_row[4] = scale * (half)qdata[4];
                        sb_row[5] = scale * (half)qdata[5];
                        sb_row[6] = scale * (half)qdata[6];
                        sb_row[7] = scale * (half)qdata[7];
                    } else {
                        for (ushort i = 0; i < count; i++) {
                            sb_row[i] = scale * (half)qdata[i];
                        }
                        for (ushort i = count; i < 8; i++) {
                            sb_row[i] = (half)0.0h;
                        }
                    }
                } else {
                    FOR_UNROLL (ushort i = 0; i < 8; i++) {
                        sb_row[i] = (half)0.0h;
                    }
                }
            } else {
                uint block_offset = gn * row_bytes + block_idx * Q8B_BLOCK_SIZE;
                half scale = as_type<half>(*(device const ushort*)(W_q8 + block_offset));
                device const char* qdata = (device const char*)(W_q8 + block_offset + 2 + k_offset);

                sb_row[0] = scale * (half)qdata[0];
                sb_row[1] = scale * (half)qdata[1];
                sb_row[2] = scale * (half)qdata[2];
                sb_row[3] = scale * (half)qdata[3];
                sb_row[4] = scale * (half)qdata[4];
                sb_row[5] = scale * (half)qdata[5];
                sb_row[6] = scale * (half)qdata[6];
                sb_row[7] = scale * (half)qdata[7];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (ushort ks = 0; ks < TILE_K; ks += 8) {
            simdgroup_half8x8 ma[2];
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(ma[0], sa + (sg_r * 16 + 0) * TILE_K + ks, TILE_K);
            simdgroup_load(ma[1], sa + (sg_r * 16 + 8) * TILE_K + ks, TILE_K);

            simdgroup_half8x8 mb[2];
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(mb[0], sb + (sg_c * 16 + 0) * TILE_K + ks, TILE_K, ulong2(0,0), true);
            simdgroup_load(mb[1], sb + (sg_c * 16 + 8) * TILE_K + ks, TILE_K, ulong2(0,0), true);

            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_multiply_accumulate(mc[0][0], ma[0], mb[0], mc[0][0]);
            simdgroup_multiply_accumulate(mc[0][1], ma[0], mb[1], mc[0][1]);
            simdgroup_multiply_accumulate(mc[1][0], ma[1], mb[0], mc[1][0]);
            simdgroup_multiply_accumulate(mc[1][1], ma[1], mb[1], mc[1][1]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store results with residual add: Y[m,n] = GEMM[m,n] + R[m,n]
    uint sg_m_base = tile_m_start + sg_r * 16;
    uint sg_n_base = tile_n_start + sg_c * 16;

    // Use threadgroup memory for store + residual add
    {
        threadgroup float* sc = (threadgroup float*)shmem;
        threadgroup float* my_sc = sc + sgitg * 256;
        FOR_UNROLL (ushort i = 0; i < 2; i++)
            FOR_UNROLL (ushort j = 0; j < 2; j++)
                simdgroup_store(mc[i][j], my_sc + (i * 8) * 16 + (j * 8), 16);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ushort lane = tiitg % 32;
        if (FC_BC_M || FC_BC_N) {
            for (ushort idx = lane; idx < 256; idx += 32) {
                ushort local_m = idx / 16;
                ushort local_n = idx % 16;
                uint gm = sg_m_base + local_m;
                uint gn = sg_n_base + local_n;
                if (gm < M && gn < N) {
                    Y[gm * N + gn] = my_sc[local_m * 16 + local_n] + R[gm * N + gn];
                }
            }
        } else {
            for (ushort idx = lane; idx < 256; idx += 32) {
                ushort local_m = idx / 16;
                ushort local_n = idx % 16;
                uint gm = sg_m_base + local_m;
                uint gn = sg_n_base + local_n;
                Y[gm * N + gn] = my_sc[local_m * 16 + local_n] + R[gm * N + gn];
            }
        }
    }
}

// ============================================================================
// tiled_matmul_bytes_f32_residual: Tiled GEMM + residual with byte-encoded F32
//
// C[m,n] = A[M,K] * B^T[N,K] + R[m,n]
// ============================================================================

kernel void tiled_matmul_bytes_f32_residual(
    device const float* A        [[buffer(0)]],
    device const uchar* B_bytes  [[buffer(1)]],
    device float*       C        [[buffer(2)]],
    constant uint&      M        [[buffer(3)]],
    constant uint&      N        [[buffer(4)]],
    constant uint&      K        [[buffer(5)]],
    device const float* R        [[buffer(6)]],
    uint2 tg_pos                 [[threadgroup_position_in_grid]],
    ushort tiitg                 [[thread_index_in_threadgroup]],
    ushort sgitg                 [[simdgroup_index_in_threadgroup]])
{
    device const float* B = (device const float*)B_bytes;

    threadgroup half shmem[2048];
    threadgroup half* sa = shmem;
    threadgroup half* sb = shmem + TILE_M * TILE_K;

    uint tile_m_start = tg_pos.y * TILE_M;
    uint tile_n_start = tg_pos.x * TILE_N;

    ushort sg_r = sgitg / SG_COLS;
    ushort sg_c = sgitg % SG_COLS;

    simdgroup_float8x8 mc[2][2];
    FOR_UNROLL (ushort i = 0; i < 2; i++)
        FOR_UNROLL (ushort j = 0; j < 2; j++)
            mc[i][j] = make_filled_simdgroup_matrix<float, 8>(0.f);

    uint num_k_tiles = (K + TILE_K - 1) / TILE_K;

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        uint k_base = kt * TILE_K;

        {
            ushort row = tiitg >> 2;
            ushort col_group = tiitg & 3;
            ushort k_start = col_group << 3;

            uint gm = tile_m_start + row;
            uint gk = k_base + k_start;
            threadgroup half* sa_ptr = sa + row * TILE_K + k_start;

            if (gm < M && gk + 7 < K) {
                device const float4* a_ptr = (device const float4*)(A + gm * K + gk);
                float4 v0 = a_ptr[0];
                float4 v1 = a_ptr[1];
                sa_ptr[0] = (half)v0.x; sa_ptr[1] = (half)v0.y;
                sa_ptr[2] = (half)v0.z; sa_ptr[3] = (half)v0.w;
                sa_ptr[4] = (half)v1.x; sa_ptr[5] = (half)v1.y;
                sa_ptr[6] = (half)v1.z; sa_ptr[7] = (half)v1.w;
            } else {
                FOR_UNROLL (ushort i = 0; i < 8; i++) {
                    uint gk_i = gk + i;
                    sa_ptr[i] = (gm < M && gk_i < K) ? (half)A[gm * K + gk_i] : (half)0.0h;
                }
            }
        }

        for (ushort idx = tiitg; idx < TILE_N * TILE_K; idx += TG_SIZE) {
            ushort n_local = idx / TILE_K;
            ushort k_local = idx % TILE_K;
            uint gn = tile_n_start + n_local;
            uint gk = k_base + k_local;
            sb[n_local * TILE_K + k_local] = (gn < N && gk < K) ? (half)B[gn * K + gk] : (half)0.0h;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (ushort ks = 0; ks < TILE_K; ks += 8) {
            simdgroup_half8x8 ma[2];
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(ma[0], sa + (sg_r * 16 + 0) * TILE_K + ks, TILE_K);
            simdgroup_load(ma[1], sa + (sg_r * 16 + 8) * TILE_K + ks, TILE_K);

            simdgroup_half8x8 mb[2];
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(mb[0], sb + (sg_c * 16 + 0) * TILE_K + ks, TILE_K, ulong2(0,0), true);
            simdgroup_load(mb[1], sb + (sg_c * 16 + 8) * TILE_K + ks, TILE_K, ulong2(0,0), true);

            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_multiply_accumulate(mc[0][0], ma[0], mb[0], mc[0][0]);
            simdgroup_multiply_accumulate(mc[0][1], ma[0], mb[1], mc[0][1]);
            simdgroup_multiply_accumulate(mc[1][0], ma[1], mb[0], mc[1][0]);
            simdgroup_multiply_accumulate(mc[1][1], ma[1], mb[1], mc[1][1]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint sg_m_base = tile_m_start + sg_r * 16;
    uint sg_n_base = tile_n_start + sg_c * 16;

    {
        threadgroup float* sc = (threadgroup float*)shmem;
        threadgroup float* my_sc = sc + sgitg * 256;
        FOR_UNROLL (ushort i = 0; i < 2; i++)
            FOR_UNROLL (ushort j = 0; j < 2; j++)
                simdgroup_store(mc[i][j], my_sc + (i * 8) * 16 + (j * 8), 16);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ushort lane = tiitg % 32;
        for (ushort idx = lane; idx < 256; idx += 32) {
            ushort local_m = idx / 16;
            ushort local_n = idx % 16;
            uint gm = sg_m_base + local_m;
            uint gn = sg_n_base + local_n;
            if (gm < M && gn < N)
                C[gm * N + gn] = my_sc[local_m * 16 + local_n] + R[gm * N + gn];
        }
    }
}
// ============================================================================
// tiled_matmul_f16: Tiled GEMM with F16 (half-precision) weights
//
// Y[M,N] = X[M,K] * W^T[N,K]    (W stored row-major as [N,K] in half)
//
// Same as tiled_matmul_bytes_f32 but B is a native half buffer.
// Each weight is a 2-byte IEEE 754 half-precision float -- no block structure,
// no scale factors. Simpler than Q8_0 (no dequantization needed).
// Uses simdgroup MMA with half-precision threadgroup tiles.
// ============================================================================

kernel void tiled_matmul_f16(
    device const half*  W        [[buffer(0)]],   // [N, K] weights (half)
    device const float* X        [[buffer(1)]],   // [M, K] input batch (float)
    device float*       Y        [[buffer(2)]],   // [M, N] output (float)
    constant uint&      M        [[buffer(3)]],
    constant uint&      N        [[buffer(4)]],
    constant uint&      K        [[buffer(5)]],
    uint2 tg_pos                 [[threadgroup_position_in_grid]],
    ushort tiitg                 [[thread_index_in_threadgroup]],
    ushort sgitg                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup half shmem[2048];
    threadgroup half* sa = shmem;
    threadgroup half* sb = shmem + TILE_M * TILE_K;

    uint tile_m_start = tg_pos.y * TILE_M;
    uint tile_n_start = tg_pos.x * TILE_N;

    ushort sg_r = sgitg / SG_COLS;
    ushort sg_c = sgitg % SG_COLS;

    simdgroup_float8x8 mc[2][2];
    FOR_UNROLL (ushort i = 0; i < 2; i++)
        FOR_UNROLL (ushort j = 0; j < 2; j++)
            mc[i][j] = make_filled_simdgroup_matrix<float, 8>(0.f);

    uint num_k_tiles = (K + TILE_K - 1) / TILE_K;

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        uint k_base = kt * TILE_K;

        // Load A tile (X): float -> half conversion
        {
            ushort row = tiitg >> 2;
            ushort col_group = tiitg & 3;
            ushort k_start = col_group << 3;

            uint gm = tile_m_start + row;
            uint gk = k_base + k_start;
            threadgroup half* sa_ptr = sa + row * TILE_K + k_start;

            if (gm < M && gk + 7 < K) {
                device const float4* x_ptr = (device const float4*)(X + gm * K + gk);
                float4 v0 = x_ptr[0];
                float4 v1 = x_ptr[1];
                sa_ptr[0] = (half)v0.x; sa_ptr[1] = (half)v0.y;
                sa_ptr[2] = (half)v0.z; sa_ptr[3] = (half)v0.w;
                sa_ptr[4] = (half)v1.x; sa_ptr[5] = (half)v1.y;
                sa_ptr[6] = (half)v1.z; sa_ptr[7] = (half)v1.w;
            } else {
                FOR_UNROLL (ushort i = 0; i < 8; i++) {
                    uint gk_i = gk + i;
                    sa_ptr[i] = (gm < M && gk_i < K) ? (half)X[gm * K + gk_i] : (half)0.0h;
                }
            }
        }

        // Load B tile (W): native half reads -- no dequantization needed
        // Vectorized: 128 threads load 32x32 = 1024 halfs, ~8 per thread
        {
            ushort row = tiitg >> 2;           // 0..31
            ushort col_group = tiitg & 3;      // 0..3
            ushort k_start = col_group << 3;   // col_group * 8

            uint gn = tile_n_start + row;
            uint gk = k_base + k_start;
            threadgroup half* sb_ptr = sb + row * TILE_K + k_start;

            if (gn < N && gk + 7 < K) {
                // Fast path: 8 consecutive halfs via half4 vectorized loads
                device const half4* w_ptr = (device const half4*)(W + gn * K + gk);
                half4 h0 = w_ptr[0];
                half4 h1 = w_ptr[1];
                sb_ptr[0] = h0.x; sb_ptr[1] = h0.y;
                sb_ptr[2] = h0.z; sb_ptr[3] = h0.w;
                sb_ptr[4] = h1.x; sb_ptr[5] = h1.y;
                sb_ptr[6] = h1.z; sb_ptr[7] = h1.w;
            } else {
                FOR_UNROLL (ushort i = 0; i < 8; i++) {
                    uint gk_i = gk + i;
                    sb_ptr[i] = (gn < N && gk_i < K) ? W[gn * K + gk_i] : (half)0.0h;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (ushort ks = 0; ks < TILE_K; ks += 8) {
            simdgroup_half8x8 ma[2];
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(ma[0], sa + (sg_r * 16 + 0) * TILE_K + ks, TILE_K);
            simdgroup_load(ma[1], sa + (sg_r * 16 + 8) * TILE_K + ks, TILE_K);

            simdgroup_half8x8 mb[2];
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(mb[0], sb + (sg_c * 16 + 0) * TILE_K + ks, TILE_K, ulong2(0,0), true);
            simdgroup_load(mb[1], sb + (sg_c * 16 + 8) * TILE_K + ks, TILE_K, ulong2(0,0), true);

            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_multiply_accumulate(mc[0][0], ma[0], mb[0], mc[0][0]);
            simdgroup_multiply_accumulate(mc[0][1], ma[0], mb[1], mc[0][1]);
            simdgroup_multiply_accumulate(mc[1][0], ma[1], mb[0], mc[1][0]);
            simdgroup_multiply_accumulate(mc[1][1], ma[1], mb[1], mc[1][1]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint sg_m_base = tile_m_start + sg_r * 16;
    uint sg_n_base = tile_n_start + sg_c * 16;

    if (sg_m_base + 16 <= M && sg_n_base + 16 <= N) {
        FOR_UNROLL (ushort i = 0; i < 2; i++)
            FOR_UNROLL (ushort j = 0; j < 2; j++)
                simdgroup_store(mc[i][j], Y + (sg_m_base + i * 8) * N + (sg_n_base + j * 8), N);
    } else {
        threadgroup float* sc = (threadgroup float*)shmem;
        threadgroup float* my_sc = sc + sgitg * 256;
        FOR_UNROLL (ushort i = 0; i < 2; i++)
            FOR_UNROLL (ushort j = 0; j < 2; j++)
                simdgroup_store(mc[i][j], my_sc + (i * 8) * 16 + (j * 8), 16);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ushort lane = tiitg % 32;
        for (ushort idx = lane; idx < 256; idx += 32) {
            ushort local_m = idx / 16;
            ushort local_n = idx % 16;
            uint gm = sg_m_base + local_m;
            uint gn = sg_n_base + local_n;
            if (gm < M && gn < N)
                Y[gm * N + gn] = my_sc[local_m * 16 + local_n];
        }
    }
}

// ============================================================================
// tiled_matmul_f16_residual: Tiled GEMM with F16 weights + residual add
//
// Y[m,n] = X[M,K] * W^T[N,K] + R[m,n]
//
// Same as tiled_matmul_f16 but fuses residual addition at writeback.
// Eliminates a separate add_residual dispatch + encoder barrier per layer.
// ============================================================================

kernel void tiled_matmul_f16_residual(
    device const half*  W        [[buffer(0)]],   // [N, K] weights (half)
    device const float* X        [[buffer(1)]],   // [M, K] input batch (float)
    device float*       Y        [[buffer(2)]],   // [M, N] output (float)
    constant uint&      M        [[buffer(3)]],
    constant uint&      N        [[buffer(4)]],
    constant uint&      K        [[buffer(5)]],
    device const float* R        [[buffer(6)]],   // [M, N] residual to add
    uint2 tg_pos                 [[threadgroup_position_in_grid]],
    ushort tiitg                 [[thread_index_in_threadgroup]],
    ushort sgitg                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup half shmem[2048];
    threadgroup half* sa = shmem;
    threadgroup half* sb = shmem + TILE_M * TILE_K;

    uint tile_m_start = tg_pos.y * TILE_M;
    uint tile_n_start = tg_pos.x * TILE_N;

    ushort sg_r = sgitg / SG_COLS;
    ushort sg_c = sgitg % SG_COLS;

    simdgroup_float8x8 mc[2][2];
    FOR_UNROLL (ushort i = 0; i < 2; i++)
        FOR_UNROLL (ushort j = 0; j < 2; j++)
            mc[i][j] = make_filled_simdgroup_matrix<float, 8>(0.f);

    uint num_k_tiles = (K + TILE_K - 1) / TILE_K;

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        uint k_base = kt * TILE_K;

        // Load A tile (X): float -> half conversion
        {
            ushort row = tiitg >> 2;
            ushort col_group = tiitg & 3;
            ushort k_start = col_group << 3;

            uint gm = tile_m_start + row;
            uint gk = k_base + k_start;
            threadgroup half* sa_ptr = sa + row * TILE_K + k_start;

            if (gm < M && gk + 7 < K) {
                device const float4* x_ptr = (device const float4*)(X + gm * K + gk);
                float4 v0 = x_ptr[0];
                float4 v1 = x_ptr[1];
                sa_ptr[0] = (half)v0.x; sa_ptr[1] = (half)v0.y;
                sa_ptr[2] = (half)v0.z; sa_ptr[3] = (half)v0.w;
                sa_ptr[4] = (half)v1.x; sa_ptr[5] = (half)v1.y;
                sa_ptr[6] = (half)v1.z; sa_ptr[7] = (half)v1.w;
            } else {
                FOR_UNROLL (ushort i = 0; i < 8; i++) {
                    uint gk_i = gk + i;
                    sa_ptr[i] = (gm < M && gk_i < K) ? (half)X[gm * K + gk_i] : (half)0.0h;
                }
            }
        }

        // Load B tile (W): native half reads
        {
            ushort row = tiitg >> 2;
            ushort col_group = tiitg & 3;
            ushort k_start = col_group << 3;

            uint gn = tile_n_start + row;
            uint gk = k_base + k_start;
            threadgroup half* sb_ptr = sb + row * TILE_K + k_start;

            if (gn < N && gk + 7 < K) {
                device const half4* w_ptr = (device const half4*)(W + gn * K + gk);
                half4 h0 = w_ptr[0];
                half4 h1 = w_ptr[1];
                sb_ptr[0] = h0.x; sb_ptr[1] = h0.y;
                sb_ptr[2] = h0.z; sb_ptr[3] = h0.w;
                sb_ptr[4] = h1.x; sb_ptr[5] = h1.y;
                sb_ptr[6] = h1.z; sb_ptr[7] = h1.w;
            } else {
                FOR_UNROLL (ushort i = 0; i < 8; i++) {
                    uint gk_i = gk + i;
                    sb_ptr[i] = (gn < N && gk_i < K) ? W[gn * K + gk_i] : (half)0.0h;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (ushort ks = 0; ks < TILE_K; ks += 8) {
            simdgroup_half8x8 ma[2];
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(ma[0], sa + (sg_r * 16 + 0) * TILE_K + ks, TILE_K);
            simdgroup_load(ma[1], sa + (sg_r * 16 + 8) * TILE_K + ks, TILE_K);

            simdgroup_half8x8 mb[2];
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(mb[0], sb + (sg_c * 16 + 0) * TILE_K + ks, TILE_K, ulong2(0,0), true);
            simdgroup_load(mb[1], sb + (sg_c * 16 + 8) * TILE_K + ks, TILE_K, ulong2(0,0), true);

            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_multiply_accumulate(mc[0][0], ma[0], mb[0], mc[0][0]);
            simdgroup_multiply_accumulate(mc[0][1], ma[0], mb[1], mc[0][1]);
            simdgroup_multiply_accumulate(mc[1][0], ma[1], mb[0], mc[1][0]);
            simdgroup_multiply_accumulate(mc[1][1], ma[1], mb[1], mc[1][1]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store results with residual add: Y[m,n] = GEMM[m,n] + R[m,n]
    uint sg_m_base = tile_m_start + sg_r * 16;
    uint sg_n_base = tile_n_start + sg_c * 16;

    {
        threadgroup float* sc = (threadgroup float*)shmem;
        threadgroup float* my_sc = sc + sgitg * 256;
        FOR_UNROLL (ushort i = 0; i < 2; i++)
            FOR_UNROLL (ushort j = 0; j < 2; j++)
                simdgroup_store(mc[i][j], my_sc + (i * 8) * 16 + (j * 8), 16);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ushort lane = tiitg % 32;
        for (ushort idx = lane; idx < 256; idx += 32) {
            ushort local_m = idx / 16;
            ushort local_n = idx % 16;
            uint gm = sg_m_base + local_m;
            uint gn = sg_n_base + local_n;
            if (gm < M && gn < N)
                Y[gm * N + gn] = my_sc[local_m * 16 + local_n] + R[gm * N + gn];
        }
    }
}

// ============================================================================
// tiled_matmul_f16_k64: TILE_K=64 variant for F16 — fewer barriers
//
// Y[M,N] = X[M,K] * W^T[N,K]    (W stored row-major as [N,K] in half)
//
// Processes 64 elements per K-tile iteration instead of 32, halving the
// outer loop count and threadgroup barriers (from K/32 to K/64).
//
// Shared memory: sa[32*64] + sb[32*64] = 4096 halfs = 8192 bytes
//
// A-tile loading: 128 threads, 32 rows x 4 threads/row, each loads 16 elems.
// B-tile loading: 128 threads, 32 rows x 4 threads/row, each loads 16 halfs.
// No dequantization needed — native half reads.
//
// Uses FC_BC function constants for aligned variant optimization.
// ============================================================================

kernel void tiled_matmul_f16_k64(
    device const half*  W        [[buffer(0)]],   // [N, K] weights (half)
    device const float* X        [[buffer(1)]],   // [M, K] input batch (float)
    device float*       Y        [[buffer(2)]],   // [M, N] output (float)
    constant uint&      M        [[buffer(3)]],
    constant uint&      N        [[buffer(4)]],
    constant uint&      K        [[buffer(5)]],
    uint2 tg_pos                 [[threadgroup_position_in_grid]],
    ushort tiitg                 [[thread_index_in_threadgroup]],
    ushort sgitg                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup half shmem_f16_k64[4096];  // sa[32*64] + sb[32*64]
    threadgroup half* sa = shmem_f16_k64;
    threadgroup half* sb = shmem_f16_k64 + TILE_M * TILE_K_64;

    uint tile_m_start = tg_pos.y * TILE_M;
    uint tile_n_start = tg_pos.x * TILE_N;

    ushort sg_r = sgitg / SG_COLS;
    ushort sg_c = sgitg % SG_COLS;

    simdgroup_float8x8 mc[2][2];
    FOR_UNROLL (ushort i = 0; i < 2; i++)
        FOR_UNROLL (ushort j = 0; j < 2; j++)
            mc[i][j] = make_filled_simdgroup_matrix<float, 8>(0.f);

    uint num_k_tiles = (K + TILE_K_64 - 1) / TILE_K_64;

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        uint k_base = kt * TILE_K_64;

        // Load A tile (X): 32 rows x 64 cols, float -> half conversion
        // 128 threads: 32 rows x 4 threads/row, each loads 16 elements
        {
            ushort row = tiitg >> 2;           // 0..31
            ushort col_group = tiitg & 3;      // 0..3
            ushort k_start = col_group << 4;   // col_group * 16

            uint gm = tile_m_start + row;
            uint gk = k_base + k_start;
            threadgroup half* sa_ptr = sa + row * TILE_K_64 + k_start;

            if (FC_BC_M || FC_BC_K) {
                if (gm < M && gk + 15 < K) {
                    device const float4* x_ptr = (device const float4*)(X + gm * K + gk);
                    float4 v0 = x_ptr[0];
                    float4 v1 = x_ptr[1];
                    float4 v2 = x_ptr[2];
                    float4 v3 = x_ptr[3];
                    sa_ptr[0]  = (half)v0.x; sa_ptr[1]  = (half)v0.y;
                    sa_ptr[2]  = (half)v0.z; sa_ptr[3]  = (half)v0.w;
                    sa_ptr[4]  = (half)v1.x; sa_ptr[5]  = (half)v1.y;
                    sa_ptr[6]  = (half)v1.z; sa_ptr[7]  = (half)v1.w;
                    sa_ptr[8]  = (half)v2.x; sa_ptr[9]  = (half)v2.y;
                    sa_ptr[10] = (half)v2.z; sa_ptr[11] = (half)v2.w;
                    sa_ptr[12] = (half)v3.x; sa_ptr[13] = (half)v3.y;
                    sa_ptr[14] = (half)v3.z; sa_ptr[15] = (half)v3.w;
                } else {
                    FOR_UNROLL (ushort i = 0; i < 16; i++) {
                        uint gk_i = gk + i;
                        sa_ptr[i] = (gm < M && gk_i < K) ? (half)X[gm * K + gk_i] : (half)0.0h;
                    }
                }
            } else {
                device const float4* x_ptr = (device const float4*)(X + gm * K + gk);
                float4 v0 = x_ptr[0];
                float4 v1 = x_ptr[1];
                float4 v2 = x_ptr[2];
                float4 v3 = x_ptr[3];
                sa_ptr[0]  = (half)v0.x; sa_ptr[1]  = (half)v0.y;
                sa_ptr[2]  = (half)v0.z; sa_ptr[3]  = (half)v0.w;
                sa_ptr[4]  = (half)v1.x; sa_ptr[5]  = (half)v1.y;
                sa_ptr[6]  = (half)v1.z; sa_ptr[7]  = (half)v1.w;
                sa_ptr[8]  = (half)v2.x; sa_ptr[9]  = (half)v2.y;
                sa_ptr[10] = (half)v2.z; sa_ptr[11] = (half)v2.w;
                sa_ptr[12] = (half)v3.x; sa_ptr[13] = (half)v3.y;
                sa_ptr[14] = (half)v3.z; sa_ptr[15] = (half)v3.w;
            }
        }

        // Load B tile (W): 32 rows x 64 cols, native half reads
        // 128 threads: 32 rows x 4 threads/row, each loads 16 halfs
        {
            ushort row = tiitg >> 2;           // 0..31
            ushort col_group = tiitg & 3;      // 0..3
            ushort k_start = col_group << 4;   // col_group * 16

            uint gn = tile_n_start + row;
            uint gk = k_base + k_start;
            threadgroup half* sb_ptr = sb + row * TILE_K_64 + k_start;

            if (FC_BC_N || FC_BC_K) {
                if (gn < N && gk + 15 < K) {
                    device const half4* w_ptr = (device const half4*)(W + gn * K + gk);
                    half4 h0 = w_ptr[0];
                    half4 h1 = w_ptr[1];
                    half4 h2 = w_ptr[2];
                    half4 h3 = w_ptr[3];
                    sb_ptr[0]  = h0.x; sb_ptr[1]  = h0.y;
                    sb_ptr[2]  = h0.z; sb_ptr[3]  = h0.w;
                    sb_ptr[4]  = h1.x; sb_ptr[5]  = h1.y;
                    sb_ptr[6]  = h1.z; sb_ptr[7]  = h1.w;
                    sb_ptr[8]  = h2.x; sb_ptr[9]  = h2.y;
                    sb_ptr[10] = h2.z; sb_ptr[11] = h2.w;
                    sb_ptr[12] = h3.x; sb_ptr[13] = h3.y;
                    sb_ptr[14] = h3.z; sb_ptr[15] = h3.w;
                } else {
                    FOR_UNROLL (ushort i = 0; i < 16; i++) {
                        uint gk_i = gk + i;
                        sb_ptr[i] = (gn < N && gk_i < K) ? W[gn * K + gk_i] : (half)0.0h;
                    }
                }
            } else {
                device const half4* w_ptr = (device const half4*)(W + gn * K + gk);
                half4 h0 = w_ptr[0];
                half4 h1 = w_ptr[1];
                half4 h2 = w_ptr[2];
                half4 h3 = w_ptr[3];
                sb_ptr[0]  = h0.x; sb_ptr[1]  = h0.y;
                sb_ptr[2]  = h0.z; sb_ptr[3]  = h0.w;
                sb_ptr[4]  = h1.x; sb_ptr[5]  = h1.y;
                sb_ptr[6]  = h1.z; sb_ptr[7]  = h1.w;
                sb_ptr[8]  = h2.x; sb_ptr[9]  = h2.y;
                sb_ptr[10] = h2.z; sb_ptr[11] = h2.w;
                sb_ptr[12] = h3.x; sb_ptr[13] = h3.y;
                sb_ptr[14] = h3.z; sb_ptr[15] = h3.w;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // MMA: 8 iterations (TILE_K_64=64, step by 8)
        FOR_UNROLL (ushort ks = 0; ks < TILE_K_64; ks += 8) {
            simdgroup_half8x8 ma[2];
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(ma[0], sa + (sg_r * 16 + 0) * TILE_K_64 + ks, TILE_K_64);
            simdgroup_load(ma[1], sa + (sg_r * 16 + 8) * TILE_K_64 + ks, TILE_K_64);

            simdgroup_half8x8 mb[2];
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(mb[0], sb + (sg_c * 16 + 0) * TILE_K_64 + ks, TILE_K_64, ulong2(0,0), true);
            simdgroup_load(mb[1], sb + (sg_c * 16 + 8) * TILE_K_64 + ks, TILE_K_64, ulong2(0,0), true);

            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_multiply_accumulate(mc[0][0], ma[0], mb[0], mc[0][0]);
            simdgroup_multiply_accumulate(mc[0][1], ma[0], mb[1], mc[0][1]);
            simdgroup_multiply_accumulate(mc[1][0], ma[1], mb[0], mc[1][0]);
            simdgroup_multiply_accumulate(mc[1][1], ma[1], mb[1], mc[1][1]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store results
    uint sg_m_base = tile_m_start + sg_r * 16;
    uint sg_n_base = tile_n_start + sg_c * 16;

    if (FC_BC_M || FC_BC_N) {
        if (sg_m_base + 16 <= M && sg_n_base + 16 <= N) {
            FOR_UNROLL (ushort i = 0; i < 2; i++)
                FOR_UNROLL (ushort j = 0; j < 2; j++)
                    simdgroup_store(mc[i][j], Y + (sg_m_base + i * 8) * N + (sg_n_base + j * 8), N);
        } else {
            threadgroup float* sc = (threadgroup float*)shmem_f16_k64;
            threadgroup float* my_sc = sc + sgitg * 256;
            FOR_UNROLL (ushort i = 0; i < 2; i++)
                FOR_UNROLL (ushort j = 0; j < 2; j++)
                    simdgroup_store(mc[i][j], my_sc + (i * 8) * 16 + (j * 8), 16);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            ushort lane = tiitg % 32;
            for (ushort idx = lane; idx < 256; idx += 32) {
                ushort local_m = idx / 16;
                ushort local_n = idx % 16;
                uint gm = sg_m_base + local_m;
                uint gn = sg_n_base + local_n;
                if (gm < M && gn < N) {
                    Y[gm * N + gn] = my_sc[local_m * 16 + local_n];
                }
            }
        }
    } else {
        FOR_UNROLL (ushort i = 0; i < 2; i++)
            FOR_UNROLL (ushort j = 0; j < 2; j++)
                simdgroup_store(mc[i][j], Y + (sg_m_base + i * 8) * N + (sg_n_base + j * 8), N);
    }
}

// ============================================================================
// tiled_matmul_f16_k64_residual: TILE_K=64 variant for F16 + residual add
//
// Y[m,n] = X[M,K] * W^T[N,K] + R[m,n]
//
// Same as tiled_matmul_f16_k64 but fuses residual addition at writeback.
// Eliminates a separate add_residual dispatch + encoder barrier per layer.
// ============================================================================

kernel void tiled_matmul_f16_k64_residual(
    device const half*  W        [[buffer(0)]],   // [N, K] weights (half)
    device const float* X        [[buffer(1)]],   // [M, K] input batch (float)
    device float*       Y        [[buffer(2)]],   // [M, N] output (float)
    constant uint&      M        [[buffer(3)]],
    constant uint&      N        [[buffer(4)]],
    constant uint&      K        [[buffer(5)]],
    device const float* R        [[buffer(6)]],   // [M, N] residual to add
    uint2 tg_pos                 [[threadgroup_position_in_grid]],
    ushort tiitg                 [[thread_index_in_threadgroup]],
    ushort sgitg                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup half shmem_f16_k64r[4096];  // sa[32*64] + sb[32*64]
    threadgroup half* sa = shmem_f16_k64r;
    threadgroup half* sb = shmem_f16_k64r + TILE_M * TILE_K_64;

    uint tile_m_start = tg_pos.y * TILE_M;
    uint tile_n_start = tg_pos.x * TILE_N;

    ushort sg_r = sgitg / SG_COLS;
    ushort sg_c = sgitg % SG_COLS;

    simdgroup_float8x8 mc[2][2];
    FOR_UNROLL (ushort i = 0; i < 2; i++)
        FOR_UNROLL (ushort j = 0; j < 2; j++)
            mc[i][j] = make_filled_simdgroup_matrix<float, 8>(0.f);

    uint num_k_tiles = (K + TILE_K_64 - 1) / TILE_K_64;

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        uint k_base = kt * TILE_K_64;

        // Load A tile (X): 32 rows x 64 cols, float -> half conversion
        {
            ushort row = tiitg >> 2;
            ushort col_group = tiitg & 3;
            ushort k_start = col_group << 4;

            uint gm = tile_m_start + row;
            uint gk = k_base + k_start;
            threadgroup half* sa_ptr = sa + row * TILE_K_64 + k_start;

            if (FC_BC_M || FC_BC_K) {
                if (gm < M && gk + 15 < K) {
                    device const float4* x_ptr = (device const float4*)(X + gm * K + gk);
                    float4 v0 = x_ptr[0];
                    float4 v1 = x_ptr[1];
                    float4 v2 = x_ptr[2];
                    float4 v3 = x_ptr[3];
                    sa_ptr[0]  = (half)v0.x; sa_ptr[1]  = (half)v0.y;
                    sa_ptr[2]  = (half)v0.z; sa_ptr[3]  = (half)v0.w;
                    sa_ptr[4]  = (half)v1.x; sa_ptr[5]  = (half)v1.y;
                    sa_ptr[6]  = (half)v1.z; sa_ptr[7]  = (half)v1.w;
                    sa_ptr[8]  = (half)v2.x; sa_ptr[9]  = (half)v2.y;
                    sa_ptr[10] = (half)v2.z; sa_ptr[11] = (half)v2.w;
                    sa_ptr[12] = (half)v3.x; sa_ptr[13] = (half)v3.y;
                    sa_ptr[14] = (half)v3.z; sa_ptr[15] = (half)v3.w;
                } else {
                    FOR_UNROLL (ushort i = 0; i < 16; i++) {
                        uint gk_i = gk + i;
                        sa_ptr[i] = (gm < M && gk_i < K) ? (half)X[gm * K + gk_i] : (half)0.0h;
                    }
                }
            } else {
                device const float4* x_ptr = (device const float4*)(X + gm * K + gk);
                float4 v0 = x_ptr[0];
                float4 v1 = x_ptr[1];
                float4 v2 = x_ptr[2];
                float4 v3 = x_ptr[3];
                sa_ptr[0]  = (half)v0.x; sa_ptr[1]  = (half)v0.y;
                sa_ptr[2]  = (half)v0.z; sa_ptr[3]  = (half)v0.w;
                sa_ptr[4]  = (half)v1.x; sa_ptr[5]  = (half)v1.y;
                sa_ptr[6]  = (half)v1.z; sa_ptr[7]  = (half)v1.w;
                sa_ptr[8]  = (half)v2.x; sa_ptr[9]  = (half)v2.y;
                sa_ptr[10] = (half)v2.z; sa_ptr[11] = (half)v2.w;
                sa_ptr[12] = (half)v3.x; sa_ptr[13] = (half)v3.y;
                sa_ptr[14] = (half)v3.z; sa_ptr[15] = (half)v3.w;
            }
        }

        // Load B tile (W): 32 rows x 64 cols, native half reads
        {
            ushort row = tiitg >> 2;
            ushort col_group = tiitg & 3;
            ushort k_start = col_group << 4;

            uint gn = tile_n_start + row;
            uint gk = k_base + k_start;
            threadgroup half* sb_ptr = sb + row * TILE_K_64 + k_start;

            if (FC_BC_N || FC_BC_K) {
                if (gn < N && gk + 15 < K) {
                    device const half4* w_ptr = (device const half4*)(W + gn * K + gk);
                    half4 h0 = w_ptr[0];
                    half4 h1 = w_ptr[1];
                    half4 h2 = w_ptr[2];
                    half4 h3 = w_ptr[3];
                    sb_ptr[0]  = h0.x; sb_ptr[1]  = h0.y;
                    sb_ptr[2]  = h0.z; sb_ptr[3]  = h0.w;
                    sb_ptr[4]  = h1.x; sb_ptr[5]  = h1.y;
                    sb_ptr[6]  = h1.z; sb_ptr[7]  = h1.w;
                    sb_ptr[8]  = h2.x; sb_ptr[9]  = h2.y;
                    sb_ptr[10] = h2.z; sb_ptr[11] = h2.w;
                    sb_ptr[12] = h3.x; sb_ptr[13] = h3.y;
                    sb_ptr[14] = h3.z; sb_ptr[15] = h3.w;
                } else {
                    FOR_UNROLL (ushort i = 0; i < 16; i++) {
                        uint gk_i = gk + i;
                        sb_ptr[i] = (gn < N && gk_i < K) ? W[gn * K + gk_i] : (half)0.0h;
                    }
                }
            } else {
                device const half4* w_ptr = (device const half4*)(W + gn * K + gk);
                half4 h0 = w_ptr[0];
                half4 h1 = w_ptr[1];
                half4 h2 = w_ptr[2];
                half4 h3 = w_ptr[3];
                sb_ptr[0]  = h0.x; sb_ptr[1]  = h0.y;
                sb_ptr[2]  = h0.z; sb_ptr[3]  = h0.w;
                sb_ptr[4]  = h1.x; sb_ptr[5]  = h1.y;
                sb_ptr[6]  = h1.z; sb_ptr[7]  = h1.w;
                sb_ptr[8]  = h2.x; sb_ptr[9]  = h2.y;
                sb_ptr[10] = h2.z; sb_ptr[11] = h2.w;
                sb_ptr[12] = h3.x; sb_ptr[13] = h3.y;
                sb_ptr[14] = h3.z; sb_ptr[15] = h3.w;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // MMA: 8 iterations (TILE_K_64=64, step by 8)
        FOR_UNROLL (ushort ks = 0; ks < TILE_K_64; ks += 8) {
            simdgroup_half8x8 ma[2];
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(ma[0], sa + (sg_r * 16 + 0) * TILE_K_64 + ks, TILE_K_64);
            simdgroup_load(ma[1], sa + (sg_r * 16 + 8) * TILE_K_64 + ks, TILE_K_64);

            simdgroup_half8x8 mb[2];
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(mb[0], sb + (sg_c * 16 + 0) * TILE_K_64 + ks, TILE_K_64, ulong2(0,0), true);
            simdgroup_load(mb[1], sb + (sg_c * 16 + 8) * TILE_K_64 + ks, TILE_K_64, ulong2(0,0), true);

            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_multiply_accumulate(mc[0][0], ma[0], mb[0], mc[0][0]);
            simdgroup_multiply_accumulate(mc[0][1], ma[0], mb[1], mc[0][1]);
            simdgroup_multiply_accumulate(mc[1][0], ma[1], mb[0], mc[1][0]);
            simdgroup_multiply_accumulate(mc[1][1], ma[1], mb[1], mc[1][1]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store results with residual add: Y[m,n] = GEMM[m,n] + R[m,n]
    uint sg_m_base = tile_m_start + sg_r * 16;
    uint sg_n_base = tile_n_start + sg_c * 16;

    {
        threadgroup float* sc = (threadgroup float*)shmem_f16_k64r;
        threadgroup float* my_sc = sc + sgitg * 256;
        FOR_UNROLL (ushort i = 0; i < 2; i++)
            FOR_UNROLL (ushort j = 0; j < 2; j++)
                simdgroup_store(mc[i][j], my_sc + (i * 8) * 16 + (j * 8), 16);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ushort lane = tiitg % 32;
        if (FC_BC_M || FC_BC_N) {
            for (ushort idx = lane; idx < 256; idx += 32) {
                ushort local_m = idx / 16;
                ushort local_n = idx % 16;
                uint gm = sg_m_base + local_m;
                uint gn = sg_n_base + local_n;
                if (gm < M && gn < N) {
                    Y[gm * N + gn] = my_sc[local_m * 16 + local_n] + R[gm * N + gn];
                }
            }
        } else {
            for (ushort idx = lane; idx < 256; idx += 32) {
                ushort local_m = idx / 16;
                ushort local_n = idx % 16;
                uint gm = sg_m_base + local_m;
                uint gn = sg_n_base + local_n;
                Y[gm * N + gn] = my_sc[local_m * 16 + local_n] + R[gm * N + gn];
            }
        }
    }
}


// ============================================================================
// matmul_bytes_f32_residual: Mat-vec with byte-encoded F32 weights + residual add
//
// out[row] = dot(W_row, x) + residual[row]
// Used in decode path for non-Q8_0 Wo and Down projections.
// ============================================================================

kernel void matmul_bytes_f32_residual(
    device const uchar* w_bytes [[buffer(0)]],
    device const float* x       [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant uint&      in_dim  [[buffer(3)]],
    device const float* residual [[buffer(4)]],
    uint row                    [[threadgroup_position_in_grid]],
    uint tid                    [[thread_index_in_threadgroup]],
    uint tg_size                [[threads_per_threadgroup]],
    uint simd_lane              [[thread_index_in_simdgroup]],
    uint simd_group             [[simdgroup_index_in_threadgroup]])
{
    device const float* w_row = (device const float*)(w_bytes + row * in_dim * 4);

    float sum = 0.0f;
    for (uint j = tid; j < in_dim; j += tg_size) {
        sum += w_row[j] * x[j];
    }

    sum = simd_sum(sum);

    threadgroup float partial_sums[32];

    if (simd_lane == 0) {
        partial_sums[simd_group] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simd_groups = (tg_size + 31) / 32;
    if (simd_group == 0) {
        float val = (simd_lane < num_simd_groups) ? partial_sums[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            out[row] = val + residual[row];
        }
    }
}


// ============================================================================
// Q4_0 QUANTIZATION KERNELS
//
// Q4_0 block layout (18 bytes per 32 elements):
//   [f16 scale (2 bytes)] [16 x uint8 packed nibbles (16 bytes)]
//   Two values per byte: low nibble = (byte & 0xF) - 8
//                         high nibble = ((byte >> 4) & 0xF) - 8
//   dequant value[i] = ((nibble(i) & 0xF) - 8) * scale
// ============================================================================

constant constexpr uint Q4_BLOCK_SIZE = 18;  // 2 bytes f16 scale + 16 bytes nibbles
constant constexpr uint Q4_1_BLOCK_SIZE = 20; // 2 bytes f16 scale + 2 bytes f16 min + 16 bytes nibbles
constant constexpr uint Q4_GROUP_SIZE = 32;  // 32 elements per block

// ============================================================================
// dequant_matmul_q4_0: Fused Q4_0 dequantization + matrix-vector multiply
//
// w_q4:   Q4_0 weight data for [out_dim, in_dim] matrix
//         Each row: ceil(in_dim/32) blocks, each block = 2 bytes f16 scale + 16 bytes nibbles
// x:      [in_dim] input vector (f32)
// out:    [out_dim] output vector (f32)
// in_dim: number of elements per row (NOT byte stride)
//
// Strategy: One SIMD group (32 threads) per output row.
// Each lane handles element [lane] of every Q4_0 block.
// Lane i reads byte qs[i/2], extracts low nibble (i even) or high nibble (i odd).
// ============================================================================

kernel void dequant_matmul_q4_0(
    device const uchar* w_q4    [[buffer(0)]],
    device const float* x       [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant uint&      in_dim  [[buffer(3)]],
    uint row                    [[threadgroup_position_in_grid]],
    uint lane                   [[thread_index_in_simdgroup]])
{
    uint num_blocks = in_dim >> 5;  // in_dim / 32
    uint row_bytes = num_blocks * Q4_BLOCK_SIZE;
    device const uchar* row_ptr = w_q4 + row * row_bytes;

    float sum = 0.0f;

    // De-interleaved nibble mapping: lane 0-15 -> lo nibble, lane 16-31 -> hi nibble
    uint q4_byte = (lane < 16) ? lane : (lane - 16);
    bool q4_hi = (lane >= 16);

    // 4x unrolled: process 4 Q4_0 blocks per iteration for ILP
    uint b = 0;
    for (; b + 3 < num_blocks; b += 4) {
        device const uchar* bp0 = row_ptr + b * Q4_BLOCK_SIZE;
        device const uchar* bp1 = bp0 + Q4_BLOCK_SIZE;
        device const uchar* bp2 = bp1 + Q4_BLOCK_SIZE;
        device const uchar* bp3 = bp2 + Q4_BLOCK_SIZE;
        half s0 = as_type<half>(*(device const ushort*)bp0);
        half s1 = as_type<half>(*(device const ushort*)bp1);
        half s2 = as_type<half>(*(device const ushort*)bp2);
        half s3 = as_type<half>(*(device const ushort*)bp3);
        // Q4_0 dequant: de-interleaved nibble extraction
        uchar byte0 = (bp0 + 2)[q4_byte];
        uchar byte1 = (bp1 + 2)[q4_byte];
        uchar byte2 = (bp2 + 2)[q4_byte];
        uchar byte3 = (bp3 + 2)[q4_byte];
        int q0 = (q4_hi ? (byte0 >> 4) : (byte0 & 0xF)) - 8;
        int q1 = (q4_hi ? (byte1 >> 4) : (byte1 & 0xF)) - 8;
        int q2 = (q4_hi ? (byte2 >> 4) : (byte2 & 0xF)) - 8;
        int q3 = (q4_hi ? (byte3 >> 4) : (byte3 & 0xF)) - 8;
        float v0 = float(q0) * x[(b << 5) + lane];
        float v1 = float(q1) * x[((b+1) << 5) + lane];
        float v2 = float(q2) * x[((b+2) << 5) + lane];
        float v3 = float(q3) * x[((b+3) << 5) + lane];
        sum += float(s0) * simd_sum(v0) + float(s1) * simd_sum(v1)
             + float(s2) * simd_sum(v2) + float(s3) * simd_sum(v3);
    }
    // Handle remaining blocks
    for (; b < num_blocks; b++) {
        device const uchar* bp = row_ptr + b * Q4_BLOCK_SIZE;
        half scale = as_type<half>(*(device const ushort*)bp);
        uchar byte_val = (bp + 2)[q4_byte];
        int q = (q4_hi ? (byte_val >> 4) : (byte_val & 0xF)) - 8;
        float val = float(q) * x[(b << 5) + lane];
        sum += float(scale) * simd_sum(val);
    }

    if (lane == 0) {
        out[row] = sum;
    }
}

// ============================================================================
// dequant_matmul_q4_0_residual: Fused Q4_0 matmul + residual add
// out[row] = dot(w_q4_row, x) + residual[row]
// ============================================================================

kernel void dequant_matmul_q4_0_residual(
    device const uchar* w_q4      [[buffer(0)]],
    device const float* x         [[buffer(1)]],
    device float*       out       [[buffer(2)]],
    constant uint&      in_dim    [[buffer(3)]],
    device const float* residual  [[buffer(4)]],
    uint row                      [[threadgroup_position_in_grid]],
    uint lane                     [[thread_index_in_simdgroup]])
{
    uint num_blocks = in_dim >> 5;
    device const uchar* row_ptr = w_q4 + row * num_blocks * Q4_BLOCK_SIZE;

    // De-interleaved nibble mapping
    uint q4_byte = (lane < 16) ? lane : (lane - 16);
    bool q4_hi = (lane >= 16);

    float sum = 0.0f;
    uint b = 0;
    for (; b + 3 < num_blocks; b += 4) {
        device const uchar* bp0 = row_ptr + b * Q4_BLOCK_SIZE;
        device const uchar* bp1 = bp0 + Q4_BLOCK_SIZE;
        device const uchar* bp2 = bp1 + Q4_BLOCK_SIZE;
        device const uchar* bp3 = bp2 + Q4_BLOCK_SIZE;
        half s0 = as_type<half>(*(device const ushort*)bp0);
        half s1 = as_type<half>(*(device const ushort*)bp1);
        half s2 = as_type<half>(*(device const ushort*)bp2);
        half s3 = as_type<half>(*(device const ushort*)bp3);
        uchar byte0 = (bp0 + 2)[q4_byte];
        uchar byte1 = (bp1 + 2)[q4_byte];
        uchar byte2 = (bp2 + 2)[q4_byte];
        uchar byte3 = (bp3 + 2)[q4_byte];
        int q0 = (q4_hi ? (byte0 >> 4) : (byte0 & 0xF)) - 8;
        int q1 = (q4_hi ? (byte1 >> 4) : (byte1 & 0xF)) - 8;
        int q2 = (q4_hi ? (byte2 >> 4) : (byte2 & 0xF)) - 8;
        int q3 = (q4_hi ? (byte3 >> 4) : (byte3 & 0xF)) - 8;
        float v0 = float(q0) * x[(b << 5) + lane];
        float v1 = float(q1) * x[((b+1) << 5) + lane];
        float v2 = float(q2) * x[((b+2) << 5) + lane];
        float v3 = float(q3) * x[((b+3) << 5) + lane];
        sum += float(s0) * simd_sum(v0) + float(s1) * simd_sum(v1)
             + float(s2) * simd_sum(v2) + float(s3) * simd_sum(v3);
    }
    for (; b < num_blocks; b++) {
        device const uchar* bp = row_ptr + b * Q4_BLOCK_SIZE;
        half scale = as_type<half>(*(device const ushort*)bp);
        uchar byte_val = (bp + 2)[q4_byte];
        int q = (q4_hi ? (byte_val >> 4) : (byte_val & 0xF)) - 8;
        float val = float(q) * x[(b << 5) + lane];
        sum += float(scale) * simd_sum(val);
    }

    if (lane == 0) {
        out[row] = sum + residual[row];
    }
}

// ============================================================================
// dequant_matmul_q4_0_4row: 4-row Q4_0 matmul (4 rows per threadgroup)
//
// Dispatch: threadgroups = ceil(out_dim/4), threads_per_threadgroup = 128
// ============================================================================

kernel void dequant_matmul_q4_0_4row(
    device const uchar* w_q4    [[buffer(0)]],
    device const float* x       [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant uint&      in_dim  [[buffer(3)]],
    constant uint&      out_dim [[buffer(4)]],
    uint row_group              [[threadgroup_position_in_grid]],
    uint lane                   [[thread_index_in_simdgroup]],
    uint sg                     [[simdgroup_index_in_threadgroup]])
{
    uint row = row_group * 4 + sg;
    if (row >= out_dim) return;

    uint num_blocks = in_dim >> 5;
    uint row_bytes = num_blocks * Q4_BLOCK_SIZE;
    device const uchar* row_ptr = w_q4 + row * row_bytes;

    // De-interleaved nibble mapping
    uint q4_byte = (lane < 16) ? lane : (lane - 16);
    bool q4_hi = (lane >= 16);

    float sum = 0.0f;

    uint b = 0;
    for (; b + 3 < num_blocks; b += 4) {
        device const uchar* bp0 = row_ptr + b * Q4_BLOCK_SIZE;
        device const uchar* bp1 = bp0 + Q4_BLOCK_SIZE;
        device const uchar* bp2 = bp1 + Q4_BLOCK_SIZE;
        device const uchar* bp3 = bp2 + Q4_BLOCK_SIZE;
        half s0 = as_type<half>(*(device const ushort*)bp0);
        half s1 = as_type<half>(*(device const ushort*)bp1);
        half s2 = as_type<half>(*(device const ushort*)bp2);
        half s3 = as_type<half>(*(device const ushort*)bp3);
        uchar byte0 = (bp0 + 2)[q4_byte];
        uchar byte1 = (bp1 + 2)[q4_byte];
        uchar byte2 = (bp2 + 2)[q4_byte];
        uchar byte3 = (bp3 + 2)[q4_byte];
        int q0 = (q4_hi ? (byte0 >> 4) : (byte0 & 0xF)) - 8;
        int q1 = (q4_hi ? (byte1 >> 4) : (byte1 & 0xF)) - 8;
        int q2 = (q4_hi ? (byte2 >> 4) : (byte2 & 0xF)) - 8;
        int q3 = (q4_hi ? (byte3 >> 4) : (byte3 & 0xF)) - 8;
        float v0 = float(q0) * x[(b << 5) + lane];
        float v1 = float(q1) * x[((b+1) << 5) + lane];
        float v2 = float(q2) * x[((b+2) << 5) + lane];
        float v3 = float(q3) * x[((b+3) << 5) + lane];
        sum += float(s0) * simd_sum(v0) + float(s1) * simd_sum(v1)
             + float(s2) * simd_sum(v2) + float(s3) * simd_sum(v3);
    }
    for (; b < num_blocks; b++) {
        device const uchar* bp = row_ptr + b * Q4_BLOCK_SIZE;
        half scale = as_type<half>(*(device const ushort*)bp);
        uchar byte_val = (bp + 2)[q4_byte];
        int q = (q4_hi ? (byte_val >> 4) : (byte_val & 0xF)) - 8;
        float val = float(q) * x[(b << 5) + lane];
        sum += float(scale) * simd_sum(val);
    }

    if (lane == 0) {
        out[row] = sum;
    }
}

// ============================================================================
// dequant_matmul_q4_0_residual_4row: 4-row Q4_0 matmul + residual add
// ============================================================================

kernel void dequant_matmul_q4_0_residual_4row(
    device const uchar* w_q4      [[buffer(0)]],
    device const float* x         [[buffer(1)]],
    device float*       out       [[buffer(2)]],
    constant uint&      in_dim    [[buffer(3)]],
    device const float* residual  [[buffer(4)]],
    constant uint&      out_dim   [[buffer(5)]],
    uint row_group                [[threadgroup_position_in_grid]],
    uint lane                     [[thread_index_in_simdgroup]],
    uint sg                       [[simdgroup_index_in_threadgroup]])
{
    uint row = row_group * 4 + sg;
    if (row >= out_dim) return;

    uint num_blocks = in_dim >> 5;
    device const uchar* row_ptr = w_q4 + row * num_blocks * Q4_BLOCK_SIZE;

    // De-interleaved nibble mapping
    uint q4_byte = (lane < 16) ? lane : (lane - 16);
    bool q4_hi = (lane >= 16);

    float sum = 0.0f;
    uint b = 0;
    for (; b + 3 < num_blocks; b += 4) {
        device const uchar* bp0 = row_ptr + b * Q4_BLOCK_SIZE;
        device const uchar* bp1 = bp0 + Q4_BLOCK_SIZE;
        device const uchar* bp2 = bp1 + Q4_BLOCK_SIZE;
        device const uchar* bp3 = bp2 + Q4_BLOCK_SIZE;
        half s0 = as_type<half>(*(device const ushort*)bp0);
        half s1 = as_type<half>(*(device const ushort*)bp1);
        half s2 = as_type<half>(*(device const ushort*)bp2);
        half s3 = as_type<half>(*(device const ushort*)bp3);
        uchar byte0 = (bp0 + 2)[q4_byte];
        uchar byte1 = (bp1 + 2)[q4_byte];
        uchar byte2 = (bp2 + 2)[q4_byte];
        uchar byte3 = (bp3 + 2)[q4_byte];
        int q0 = (q4_hi ? (byte0 >> 4) : (byte0 & 0xF)) - 8;
        int q1 = (q4_hi ? (byte1 >> 4) : (byte1 & 0xF)) - 8;
        int q2 = (q4_hi ? (byte2 >> 4) : (byte2 & 0xF)) - 8;
        int q3 = (q4_hi ? (byte3 >> 4) : (byte3 & 0xF)) - 8;
        float v0 = float(q0) * x[(b << 5) + lane];
        float v1 = float(q1) * x[((b+1) << 5) + lane];
        float v2 = float(q2) * x[((b+2) << 5) + lane];
        float v3 = float(q3) * x[((b+3) << 5) + lane];
        sum += float(s0) * simd_sum(v0) + float(s1) * simd_sum(v1)
             + float(s2) * simd_sum(v2) + float(s3) * simd_sum(v3);
    }
    for (; b < num_blocks; b++) {
        device const uchar* bp = row_ptr + b * Q4_BLOCK_SIZE;
        half scale = as_type<half>(*(device const ushort*)bp);
        uchar byte_val = (bp + 2)[q4_byte];
        int q = (q4_hi ? (byte_val >> 4) : (byte_val & 0xF)) - 8;
        float val = float(q) * x[(b << 5) + lane];
        sum += float(scale) * simd_sum(val);
    }

    if (lane == 0) {
        out[row] = sum + residual[row];
    }
}

// ============================================================================
// dequant_matmul_q4_0_deferred: Deferred-reduction Q4_0 matvec
//
// Mirrors dequant_matmul_q8_0_deferred but with Q4_0 nibble dequantization.
// 128 threads (4 simdgroups), NR0=4 output rows per threadgroup.
// Deferred accumulation: local sums across all blocks, ONE simd_sum at end.
//
// Q4_0 block: 18 bytes = 2-byte f16 scale + 16 bytes packed nibbles (32 elements).
// Each thread handles NQ=8 elements = 4 bytes of packed nibbles.
// Thread mapping: ix = tiisg/4 (block index 0..7), il = tiisg%4 (sub-chunk 0..3).
//
// Dispatch: threadgroups = ceil(out_dim/4), threads_per_threadgroup = 128
// ============================================================================

kernel void dequant_matmul_q4_0_deferred(
    device const uchar* weights [[buffer(0)]],
    device const float* x       [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant uint&      in_dim  [[buffer(3)]],
    constant uint&      out_dim [[buffer(4)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint NR0 = 4;
    const uint NSG = 4;
    const uint NQ = 8;
    const uint NW = 32;

    const uint nb = in_dim >> 5;  // number of Q4_0 blocks per row = in_dim / 32
    const uint row_bytes = nb * Q4_BLOCK_SIZE;

    const uint r0 = tgpig * NR0;  // first output row for this threadgroup

    // Thread mapping within SIMD group
    const uint ix = tiisg / (NW / NQ);  // = tiisg / 4 -> 0..7 (block index in stride)
    const uint il = tiisg % (NW / NQ);  // = tiisg % 4 -> 0..3 (sub-chunk index)

    const uint ib0 = sgitg * NQ + ix;   // starting block for this thread

    // Pointers to weight rows
    device const uchar* ax[NR0];
    for (uint row = 0; row < NR0; ++row) {
        ax[row] = weights + (r0 + row) * row_bytes;
    }

    float sumf[NR0] = { 0.f, 0.f, 0.f, 0.f };

    // Main loop: each thread processes 4 bytes (8 nibbles) per block, accumulates locally
    // De-interleaved: byte j has lo -> element j, hi -> element j+16
    // il selects which 4-byte chunk: bytes il*4 .. il*4+3
    for (uint ib = ib0; ib < nb; ib += NSG * NQ) {
        // Load x-values for the lo and hi nibble elements of our 4 bytes
        uint block_base = ib * 32;
        float yl_lo[4], yl_hi[4];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            yl_lo[i] = x[block_base + il * 4 + i];       // lo nibble elements (0-15)
            yl_hi[i] = x[block_base + il * 4 + 16 + i];  // hi nibble elements (16-31)
        }

        // Process all NR0 rows with these x-values
        for (uint row = 0; row < NR0; ++row) {
            if (r0 + row >= out_dim) break;

            // Point to this block in the weight row
            device const uchar* bp = ax[row] + ib * Q4_BLOCK_SIZE;
            float scale = float(as_type<half>(*(device const ushort*)bp));
            device const uchar* qdata = (device const uchar*)(bp + 2) + il * 4;  // 4 bytes

            float sumq = 0.f;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; ++i) {
                uchar byte_val = qdata[i];
                float lo = float(byte_val & 0x0F) - 8.0f;
                float hi = float(byte_val >> 4) - 8.0f;
                sumq += lo * yl_lo[i] + hi * yl_hi[i];
            }
            sumf[row] += sumq * scale;
        }
    }

    // Final reduction: 1 simd_sum + cross-SG shmem reduce
    threadgroup float shmem[NR0 * NW];  // NR0 * 32 floats = 512 bytes

    // Initialize shmem (SG 0 zeros its slots)
    for (uint row = 0; row < NR0; ++row) {
        if (sgitg == 0) {
            shmem[row * NW + tiisg] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each SG writes its reduced sum
    for (uint row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem[row * NW + sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // SG 0 does the final reduction and writes output
    for (uint row = 0; row < NR0 && r0 + row < out_dim; ++row) {
        float tot = simd_sum(shmem[row * NW + tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            out[r0 + row] = tot;
        }
    }
}

// ============================================================================
// dequant_matmul_q4_1_deferred: Deferred-reduction Q4_1 matvec for decode (M=1).
// Same structure as Q4_0 deferred, but with Q4_1 block layout:
//   [f16 scale (2B)] [f16 min (2B)] [16 x uint8 packed nibbles (16B)] = 20 bytes/block
//   dequant: value[i] = scale * nibble(i) + min
// ============================================================================
kernel void dequant_matmul_q4_1_deferred(
    device const uchar* weights [[buffer(0)]],
    device const float* x       [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant uint&      in_dim  [[buffer(3)]],
    constant uint&      out_dim [[buffer(4)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint NR0 = 4;
    const uint NSG = 4;
    const uint NQ = 8;
    const uint NW = 32;

    const uint nb = in_dim >> 5;  // number of Q4_1 blocks per row = in_dim / 32
    const uint row_bytes = nb * Q4_1_BLOCK_SIZE;

    const uint r0 = tgpig * NR0;

    const uint ix = tiisg / (NW / NQ);
    const uint il = tiisg % (NW / NQ);

    const uint ib0 = sgitg * NQ + ix;

    device const uchar* ax[NR0];
    for (uint row = 0; row < NR0; ++row) {
        ax[row] = weights + (r0 + row) * row_bytes;
    }

    float sumf[NR0] = { 0.f, 0.f, 0.f, 0.f };

    for (uint ib = ib0; ib < nb; ib += NSG * NQ) {
        uint block_base = ib * 32;
        float yl_lo[4], yl_hi[4];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            yl_lo[i] = x[block_base + il * 4 + i];
            yl_hi[i] = x[block_base + il * 4 + 16 + i];
        }

        for (uint row = 0; row < NR0; ++row) {
            if (r0 + row >= out_dim) break;

            device const uchar* bp = ax[row] + ib * Q4_1_BLOCK_SIZE;
            float scale = float(as_type<half>(*(device const ushort*)bp));
            float minval = float(as_type<half>(*(device const ushort*)(bp + 2)));
            device const uchar* qdata = (device const uchar*)(bp + 4) + il * 4;

            float sumq = 0.f;
            float sumy = 0.f;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; ++i) {
                uchar byte_val = qdata[i];
                float lo = float(byte_val & 0x0F);
                float hi = float(byte_val >> 4);
                sumq += lo * yl_lo[i] + hi * yl_hi[i];
                sumy += yl_lo[i] + yl_hi[i];
            }
            sumf[row] += sumq * scale + sumy * minval;
        }
    }

    threadgroup float shmem[NR0 * NW];

    for (uint row = 0; row < NR0; ++row) {
        if (sgitg == 0) {
            shmem[row * NW + tiisg] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem[row * NW + sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0 && r0 + row < out_dim; ++row) {
        float tot = simd_sum(shmem[row * NW + tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            out[r0 + row] = tot;
        }
    }
}

// ============================================================================
// dequant_matmul_q4_0_deferred_residual: Deferred-reduction Q4_0 matvec + residual
// out[row] = dot(w_q4_row, x) + residual[row]
// Same deferred-reduction pattern as above, with fused residual addition.
// ============================================================================

kernel void dequant_matmul_q4_0_deferred_residual(
    device const uchar* weights   [[buffer(0)]],
    device const float* x         [[buffer(1)]],
    device float*       out       [[buffer(2)]],
    constant uint&      in_dim    [[buffer(3)]],
    device const float* residual  [[buffer(4)]],
    constant uint&      out_dim   [[buffer(5)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint NR0 = 4;
    const uint NSG = 4;
    const uint NQ = 8;
    const uint NW = 32;

    const uint nb = in_dim >> 5;
    const uint row_bytes = nb * Q4_BLOCK_SIZE;

    const uint r0 = tgpig * NR0;

    const uint ix = tiisg / (NW / NQ);
    const uint il = tiisg % (NW / NQ);

    const uint ib0 = sgitg * NQ + ix;

    device const uchar* ax[NR0];
    for (uint row = 0; row < NR0; ++row) {
        ax[row] = weights + (r0 + row) * row_bytes;
    }

    float sumf[NR0] = { 0.f, 0.f, 0.f, 0.f };

    for (uint ib = ib0; ib < nb; ib += NSG * NQ) {
        uint block_base = ib * 32;
        float yl_lo[4], yl_hi[4];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            yl_lo[i] = x[block_base + il * 4 + i];
            yl_hi[i] = x[block_base + il * 4 + 16 + i];
        }

        for (uint row = 0; row < NR0; ++row) {
            if (r0 + row >= out_dim) break;

            device const uchar* bp = ax[row] + ib * Q4_BLOCK_SIZE;
            float scale = float(as_type<half>(*(device const ushort*)bp));
            device const uchar* qdata = (device const uchar*)(bp + 2) + il * 4;

            float sumq = 0.f;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; ++i) {
                uchar byte_val = qdata[i];
                float lo = float(byte_val & 0x0F) - 8.0f;
                float hi = float(byte_val >> 4) - 8.0f;
                sumq += lo * yl_lo[i] + hi * yl_hi[i];
            }
            sumf[row] += sumq * scale;
        }
    }

    threadgroup float shmem[NR0 * NW];

    for (uint row = 0; row < NR0; ++row) {
        if (sgitg == 0) {
            shmem[row * NW + tiisg] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem[row * NW + sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0 && r0 + row < out_dim; ++row) {
        float tot = simd_sum(shmem[row * NW + tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            out[r0 + row] = tot + residual[r0 + row];
        }
    }
}

// ============================================================================
// dequant_matmul_q4_0_deferred_bias: Deferred-reduction Q4_0 matvec + fused bias
// out[row] = dot(w_q4_row, x) + bias[row]
//
// Same deferred-reduction pattern as dequant_matmul_q4_0_deferred, with fused
// bias addition. See dequant_matmul_q8_0_deferred_bias for bias section mapping.
//
// Dispatch: threadgroups = ceil(out_dim/4), threads_per_threadgroup = 128
// ============================================================================

kernel void dequant_matmul_q4_0_deferred_bias(
    device const uchar* weights [[buffer(0)]],
    device const float* x       [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant uint&      in_dim  [[buffer(3)]],
    constant uint&      out_dim [[buffer(4)]],
    device const float* bias_q  [[buffer(5)]],
    device const float* bias_k  [[buffer(6)]],
    device const float* bias_v  [[buffer(7)]],
    constant uint&      q_dim   [[buffer(8)]],
    constant uint&      qk_dim  [[buffer(9)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint NR0 = 4;
    const uint NSG = 4;
    const uint NQ = 8;
    const uint NW = 32;

    const uint nb = in_dim >> 5;
    const uint row_bytes = nb * Q4_BLOCK_SIZE;

    const uint r0 = tgpig * NR0;

    const uint ix = tiisg / (NW / NQ);
    const uint il = tiisg % (NW / NQ);

    const uint ib0 = sgitg * NQ + ix;

    device const uchar* ax[NR0];
    for (uint row = 0; row < NR0; ++row) {
        ax[row] = weights + (r0 + row) * row_bytes;
    }

    float sumf[NR0] = { 0.f, 0.f, 0.f, 0.f };

    for (uint ib = ib0; ib < nb; ib += NSG * NQ) {
        uint block_base = ib * 32;
        float yl_lo[4], yl_hi[4];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            yl_lo[i] = x[block_base + il * 4 + i];
            yl_hi[i] = x[block_base + il * 4 + 16 + i];
        }

        for (uint row = 0; row < NR0; ++row) {
            if (r0 + row >= out_dim) break;

            device const uchar* bp = ax[row] + ib * Q4_BLOCK_SIZE;
            float scale = float(as_type<half>(*(device const ushort*)bp));
            device const uchar* qdata = (device const uchar*)(bp + 2) + il * 4;

            float sumq = 0.f;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; ++i) {
                uchar byte_val = qdata[i];
                float lo = float(byte_val & 0x0F) - 8.0f;
                float hi = float(byte_val >> 4) - 8.0f;
                sumq += lo * yl_lo[i] + hi * yl_hi[i];
            }
            sumf[row] += sumq * scale;
        }
    }

    threadgroup float shmem[NR0 * NW];

    for (uint row = 0; row < NR0; ++row) {
        if (sgitg == 0) {
            shmem[row * NW + tiisg] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem[row * NW + sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0 && r0 + row < out_dim; ++row) {
        float tot = simd_sum(shmem[row * NW + tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            uint r = r0 + row;
            float b;
            if (r < q_dim) b = bias_q[r];
            else if (r < qk_dim) b = bias_k[r - q_dim];
            else b = bias_v[r - qk_dim];
            out[r] = tot + b;
        }
    }
}

// ============================================================================
// dequant_matmul_q4_0_deferred_nr2: NR0=2 deferred-reduction Q4_0 matvec
//
// Same structure as dequant_matmul_q8_0_deferred_nr2 but with Q4_0 nibble
// dequantization. 2 output rows per threadgroup = ceil(out_dim/2) TGs.
// More TGs = better GPU occupancy on M3 Ultra 60 cores.
//
// Q4_0 block: 18 bytes = 2-byte f16 scale + 16 bytes packed nibbles (32 elements).
// De-interleaved nibble layout: byte j has lo nibble -> element j, hi nibble -> element j+16.
//
// Dispatch: threadgroups = ceil(out_dim/2), threads_per_threadgroup = 128
// ============================================================================

kernel void dequant_matmul_q4_0_deferred_nr2(
    device const uchar* weights [[buffer(0)]],
    device const float* x       [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant uint&      in_dim  [[buffer(3)]],
    constant uint&      out_dim [[buffer(4)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint NR0 = 2;
    const uint NSG = 4;
    const uint NQ = 8;
    const uint NW = 32;

    const uint nb = in_dim >> 5;  // number of Q4_0 blocks per row = in_dim / 32
    const uint row_bytes = nb * Q4_BLOCK_SIZE;

    const uint r0 = tgpig * NR0;  // first output row for this threadgroup

    // Thread mapping within SIMD group
    const uint ix = tiisg / (NW / NQ);  // = tiisg / 4 -> 0..7 (block index in stride)
    const uint il = tiisg % (NW / NQ);  // = tiisg % 4 -> 0..3 (sub-chunk index)

    const uint ib0 = sgitg * NQ + ix;   // starting block for this thread

    // Pointers to weight rows
    device const uchar* ax[NR0];
    for (uint row = 0; row < NR0; ++row) {
        ax[row] = weights + (r0 + row) * row_bytes;
    }

    float sumf[NR0] = { 0.f, 0.f };

    // Main loop: each thread processes 4 bytes (8 nibbles) per block, accumulates locally
    for (uint ib = ib0; ib < nb; ib += NSG * NQ) {
        // Load x-values for the lo and hi nibble elements of our 4 bytes
        uint block_base = ib * 32;
        float yl_lo[4], yl_hi[4];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            yl_lo[i] = x[block_base + il * 4 + i];       // lo nibble elements (0-15)
            yl_hi[i] = x[block_base + il * 4 + 16 + i];  // hi nibble elements (16-31)
        }

        // Process all NR0 rows with these x-values
        for (uint row = 0; row < NR0; ++row) {
            if (r0 + row >= out_dim) break;

            device const uchar* bp = ax[row] + ib * Q4_BLOCK_SIZE;
            float scale = float(as_type<half>(*(device const ushort*)bp));
            device const uchar* qdata = (device const uchar*)(bp + 2) + il * 4;

            float sumq = 0.f;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; ++i) {
                uchar byte_val = qdata[i];
                float lo = float(byte_val & 0x0F) - 8.0f;
                float hi = float(byte_val >> 4) - 8.0f;
                sumq += lo * yl_lo[i] + hi * yl_hi[i];
            }
            sumf[row] += sumq * scale;
        }
    }

    // Final reduction: 1 simd_sum + cross-SG shmem reduce
    threadgroup float shmem[NR0 * NW];

    for (uint row = 0; row < NR0; ++row) {
        if (sgitg == 0) {
            shmem[row * NW + tiisg] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem[row * NW + sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0 && r0 + row < out_dim; ++row) {
        float tot = simd_sum(shmem[row * NW + tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            out[r0 + row] = tot;
        }
    }
}

// ============================================================================
// dequant_matmul_q4_0_deferred_residual_nr2: NR0=2 Q4_0 matvec + residual
// out[row] = dot(w_q4_row, x) + residual[row]
//
// Dispatch: threadgroups = ceil(out_dim/2), threads_per_threadgroup = 128
// ============================================================================

kernel void dequant_matmul_q4_0_deferred_residual_nr2(
    device const uchar* weights   [[buffer(0)]],
    device const float* x         [[buffer(1)]],
    device float*       out       [[buffer(2)]],
    constant uint&      in_dim    [[buffer(3)]],
    device const float* residual  [[buffer(4)]],
    constant uint&      out_dim   [[buffer(5)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint NR0 = 2;
    const uint NSG = 4;
    const uint NQ = 8;
    const uint NW = 32;

    const uint nb = in_dim >> 5;
    const uint row_bytes = nb * Q4_BLOCK_SIZE;

    const uint r0 = tgpig * NR0;

    const uint ix = tiisg / (NW / NQ);
    const uint il = tiisg % (NW / NQ);

    const uint ib0 = sgitg * NQ + ix;

    device const uchar* ax[NR0];
    for (uint row = 0; row < NR0; ++row) {
        ax[row] = weights + (r0 + row) * row_bytes;
    }

    float sumf[NR0] = { 0.f, 0.f };

    for (uint ib = ib0; ib < nb; ib += NSG * NQ) {
        uint block_base = ib * 32;
        float yl_lo[4], yl_hi[4];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            yl_lo[i] = x[block_base + il * 4 + i];
            yl_hi[i] = x[block_base + il * 4 + 16 + i];
        }

        for (uint row = 0; row < NR0; ++row) {
            if (r0 + row >= out_dim) break;

            device const uchar* bp = ax[row] + ib * Q4_BLOCK_SIZE;
            float scale = float(as_type<half>(*(device const ushort*)bp));
            device const uchar* qdata = (device const uchar*)(bp + 2) + il * 4;

            float sumq = 0.f;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; ++i) {
                uchar byte_val = qdata[i];
                float lo = float(byte_val & 0x0F) - 8.0f;
                float hi = float(byte_val >> 4) - 8.0f;
                sumq += lo * yl_lo[i] + hi * yl_hi[i];
            }
            sumf[row] += sumq * scale;
        }
    }

    threadgroup float shmem[NR0 * NW];

    for (uint row = 0; row < NR0; ++row) {
        if (sgitg == 0) {
            shmem[row * NW + tiisg] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem[row * NW + sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0 && r0 + row < out_dim; ++row) {
        float tot = simd_sum(shmem[row * NW + tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            out[r0 + row] = tot + residual[r0 + row];
        }
    }
}

// ============================================================================
// dequant_matmul_q4_0_deferred_bias_nr2: NR0=2 Q4_0 matvec + fused bias
// out[row] = dot(w_q4_row, x) + bias[row]
// Bias is split across Q/K/V sections (see Q8_0 bias_nr2 for mapping).
//
// Dispatch: threadgroups = ceil(out_dim/2), threads_per_threadgroup = 128
// ============================================================================

kernel void dequant_matmul_q4_0_deferred_bias_nr2(
    device const uchar* weights [[buffer(0)]],
    device const float* x       [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant uint&      in_dim  [[buffer(3)]],
    constant uint&      out_dim [[buffer(4)]],
    device const float* bias_q  [[buffer(5)]],
    device const float* bias_k  [[buffer(6)]],
    device const float* bias_v  [[buffer(7)]],
    constant uint&      q_dim   [[buffer(8)]],
    constant uint&      qk_dim  [[buffer(9)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint NR0 = 2;
    const uint NSG = 4;
    const uint NQ = 8;
    const uint NW = 32;

    const uint nb = in_dim >> 5;
    const uint row_bytes = nb * Q4_BLOCK_SIZE;

    const uint r0 = tgpig * NR0;

    const uint ix = tiisg / (NW / NQ);
    const uint il = tiisg % (NW / NQ);

    const uint ib0 = sgitg * NQ + ix;

    device const uchar* ax[NR0];
    for (uint row = 0; row < NR0; ++row) {
        ax[row] = weights + (r0 + row) * row_bytes;
    }

    float sumf[NR0] = { 0.f, 0.f };

    for (uint ib = ib0; ib < nb; ib += NSG * NQ) {
        uint block_base = ib * 32;
        float yl_lo[4], yl_hi[4];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            yl_lo[i] = x[block_base + il * 4 + i];
            yl_hi[i] = x[block_base + il * 4 + 16 + i];
        }

        for (uint row = 0; row < NR0; ++row) {
            if (r0 + row >= out_dim) break;

            device const uchar* bp = ax[row] + ib * Q4_BLOCK_SIZE;
            float scale = float(as_type<half>(*(device const ushort*)bp));
            device const uchar* qdata = (device const uchar*)(bp + 2) + il * 4;

            float sumq = 0.f;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; ++i) {
                uchar byte_val = qdata[i];
                float lo = float(byte_val & 0x0F) - 8.0f;
                float hi = float(byte_val >> 4) - 8.0f;
                sumq += lo * yl_lo[i] + hi * yl_hi[i];
            }
            sumf[row] += sumq * scale;
        }
    }

    threadgroup float shmem[NR0 * NW];

    for (uint row = 0; row < NR0; ++row) {
        if (sgitg == 0) {
            shmem[row * NW + tiisg] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem[row * NW + sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0 && r0 + row < out_dim; ++row) {
        float tot = simd_sum(shmem[row * NW + tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            uint r = r0 + row;
            float b;
            if (r < q_dim) b = bias_q[r];
            else if (r < qk_dim) b = bias_k[r - q_dim];
            else b = bias_v[r - qk_dim];
            out[r] = tot + b;
        }
    }
}

// ============================================================================
// dequant_tiled_matmul_q4_0: Fused Q4_0 dequantization + tiled GEMM via MMA
//
// Y[M,N] = X[M,K] * dequant(W_q4[N,K_bytes])^T
//
// Uses simdgroup MMA with half-precision tiles. Q4_0 values are dequantized
// to half precision in shared memory before MMA.
//
// Q4_0 block layout (18 bytes per 32 elements):
//   [f16 scale (2 bytes)] [16 x uint8 packed nibbles (16 bytes)]
// ============================================================================

kernel void dequant_tiled_matmul_q4_0(
    device const uchar* W_q4     [[buffer(0)]],   // Q4_0 weights [N, K_bytes]
    device const float* X        [[buffer(1)]],   // [M, K] input batch
    device float*       Y        [[buffer(2)]],   // [M, N] output batch
    constant uint&      M        [[buffer(3)]],   // batch size
    constant uint&      N        [[buffer(4)]],   // output dim
    constant uint&      K        [[buffer(5)]],   // input dim (elements, not bytes)
    uint2 tg_pos                 [[threadgroup_position_in_grid]],
    ushort tiitg                 [[thread_index_in_threadgroup]],
    ushort sgitg                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup half shmem[2048];
    threadgroup half* sa = shmem;
    threadgroup half* sb = shmem + TILE_M * TILE_K;

    uint tile_m_start = tg_pos.y * TILE_M;
    uint tile_n_start = tg_pos.x * TILE_N;

    ushort sg_r = sgitg / SG_COLS;
    ushort sg_c = sgitg % SG_COLS;

    simdgroup_float8x8 mc[2][2];
    FOR_UNROLL (ushort i = 0; i < 2; i++)
        FOR_UNROLL (ushort j = 0; j < 2; j++)
            mc[i][j] = make_filled_simdgroup_matrix<float, 8>(0.f);

    uint num_blocks_per_row = (K + Q4_GROUP_SIZE - 1) / Q4_GROUP_SIZE;
    uint row_bytes = num_blocks_per_row * Q4_BLOCK_SIZE;

    uint num_k_tiles = (K + TILE_K - 1) / TILE_K;

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        uint k_base = kt * TILE_K;

        // Load A tile: vectorized 2D mapping
        {
            ushort row = tiitg >> 2;
            ushort col_group = tiitg & 3;
            ushort k_start = col_group << 3;

            uint gm = tile_m_start + row;
            uint gk = k_base + k_start;
            threadgroup half* sa_ptr = sa + row * TILE_K + k_start;

            if (gm < M && gk + 7 < K) {
                device const float4* x_ptr = (device const float4*)(X + gm * K + gk);
                float4 v0 = x_ptr[0];
                float4 v1 = x_ptr[1];
                sa_ptr[0] = (half)v0.x; sa_ptr[1] = (half)v0.y;
                sa_ptr[2] = (half)v0.z; sa_ptr[3] = (half)v0.w;
                sa_ptr[4] = (half)v1.x; sa_ptr[5] = (half)v1.y;
                sa_ptr[6] = (half)v1.z; sa_ptr[7] = (half)v1.w;
            } else {
                FOR_UNROLL (ushort i = 0; i < 8; i++) {
                    uint gk_i = gk + i;
                    sa_ptr[i] = (gm < M && gk_i < K) ? (half)X[gm * K + gk_i] : (half)0.0h;
                }
            }
        }

        // Load B tile: Q4_0 dequant (de-interleaved nibble ordering)
        // TILE_K=32 = Q4_0 group size, so one block per row per K-tile.
        // 128 threads, 32 rows: 4 threads per row, each loads 8 elements.
        // De-interleaved: elem 0-15 from lo nibbles, elem 16-31 from hi nibbles.
        // k_offset 0,8 -> lo nibbles; k_offset 16,24 -> hi nibbles.
        {
            uint block_idx = k_base / Q4_GROUP_SIZE;
            ushort n_local = tiitg / 4;     // 0..31 (which row)
            ushort t_in_row = tiitg % 4;    // 0..3 (which quarter of the 32 elements)
            ushort k_offset = t_in_row * 8; // starting k position within block
            uint gn = tile_n_start + n_local;

            threadgroup half* sb_row = sb + n_local * TILE_K + k_offset;

            if (gn < N && k_base + k_offset < K) {
                uint block_offset = gn * row_bytes + block_idx * Q4_BLOCK_SIZE;
                half scale = as_type<half>(*(device const ushort*)(W_q4 + block_offset));
                device const uchar* qdata = W_q4 + block_offset + 2;

                uint k_remaining = K - (k_base + k_offset);
                ushort count = (k_remaining >= 8) ? 8 : (ushort)k_remaining;

                // De-interleaved: 8 elements from 8 bytes, one nibble per byte
                ushort byte_start = (k_offset < 16) ? k_offset : (k_offset - 16);
                bool use_hi = (k_offset >= 16);

                if (count == 8) {
                    if (use_hi) {
                        sb_row[0] = scale * (half)(int((qdata[byte_start + 0] >> 4)) - 8);
                        sb_row[1] = scale * (half)(int((qdata[byte_start + 1] >> 4)) - 8);
                        sb_row[2] = scale * (half)(int((qdata[byte_start + 2] >> 4)) - 8);
                        sb_row[3] = scale * (half)(int((qdata[byte_start + 3] >> 4)) - 8);
                        sb_row[4] = scale * (half)(int((qdata[byte_start + 4] >> 4)) - 8);
                        sb_row[5] = scale * (half)(int((qdata[byte_start + 5] >> 4)) - 8);
                        sb_row[6] = scale * (half)(int((qdata[byte_start + 6] >> 4)) - 8);
                        sb_row[7] = scale * (half)(int((qdata[byte_start + 7] >> 4)) - 8);
                    } else {
                        sb_row[0] = scale * (half)(int((qdata[byte_start + 0] & 0xF)) - 8);
                        sb_row[1] = scale * (half)(int((qdata[byte_start + 1] & 0xF)) - 8);
                        sb_row[2] = scale * (half)(int((qdata[byte_start + 2] & 0xF)) - 8);
                        sb_row[3] = scale * (half)(int((qdata[byte_start + 3] & 0xF)) - 8);
                        sb_row[4] = scale * (half)(int((qdata[byte_start + 4] & 0xF)) - 8);
                        sb_row[5] = scale * (half)(int((qdata[byte_start + 5] & 0xF)) - 8);
                        sb_row[6] = scale * (half)(int((qdata[byte_start + 6] & 0xF)) - 8);
                        sb_row[7] = scale * (half)(int((qdata[byte_start + 7] & 0xF)) - 8);
                    }
                } else {
                    for (ushort i = 0; i < count; i++) {
                        ushort elem_idx = k_offset + i;
                        ushort byte_idx = (elem_idx < 16) ? elem_idx : (elem_idx - 16);
                        uchar raw = qdata[byte_idx];
                        int nibble = (elem_idx >= 16) ? int((raw >> 4) & 0xF) : int(raw & 0xF);
                        sb_row[i] = scale * (half)(nibble - 8);
                    }
                    for (ushort i = count; i < 8; i++) {
                        sb_row[i] = (half)0.0h;
                    }
                }
            } else {
                FOR_UNROLL (ushort i = 0; i < 8; i++) {
                    sb_row[i] = (half)0.0h;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (ushort ks = 0; ks < TILE_K; ks += 8) {
            simdgroup_half8x8 ma[2];
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(ma[0], sa + (sg_r * 16 + 0) * TILE_K + ks, TILE_K);
            simdgroup_load(ma[1], sa + (sg_r * 16 + 8) * TILE_K + ks, TILE_K);

            simdgroup_half8x8 mb[2];
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(mb[0], sb + (sg_c * 16 + 0) * TILE_K + ks, TILE_K, ulong2(0,0), true);
            simdgroup_load(mb[1], sb + (sg_c * 16 + 8) * TILE_K + ks, TILE_K, ulong2(0,0), true);

            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_multiply_accumulate(mc[0][0], ma[0], mb[0], mc[0][0]);
            simdgroup_multiply_accumulate(mc[0][1], ma[0], mb[1], mc[0][1]);
            simdgroup_multiply_accumulate(mc[1][0], ma[1], mb[0], mc[1][0]);
            simdgroup_multiply_accumulate(mc[1][1], ma[1], mb[1], mc[1][1]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint sg_m_base = tile_m_start + sg_r * 16;
    uint sg_n_base = tile_n_start + sg_c * 16;

    if (sg_m_base + 16 <= M && sg_n_base + 16 <= N) {
        FOR_UNROLL (ushort i = 0; i < 2; i++)
            FOR_UNROLL (ushort j = 0; j < 2; j++)
                simdgroup_store(mc[i][j], Y + (sg_m_base + i * 8) * N + (sg_n_base + j * 8), N);
    } else {
        threadgroup float* sc = (threadgroup float*)shmem;
        threadgroup float* my_sc = sc + sgitg * 256;
        FOR_UNROLL (ushort i = 0; i < 2; i++)
            FOR_UNROLL (ushort j = 0; j < 2; j++)
                simdgroup_store(mc[i][j], my_sc + (i * 8) * 16 + (j * 8), 16);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ushort lane_id = tiitg % 32;
        for (ushort idx = lane_id; idx < 256; idx += 32) {
            ushort local_m = idx / 16;
            ushort local_n = idx % 16;
            uint gm = sg_m_base + local_m;
            uint gn = sg_n_base + local_n;
            if (gm < M && gn < N) {
                Y[gm * N + gn] = my_sc[local_m * 16 + local_n];
            }
        }
    }
}

// ============================================================================
// dequant_tiled_matmul_q4_0_k64: TILE_K=64 variant for Q4_0 — fewer barriers
//
// Y[M,N] = X[M,K] * dequant(W_q4[N,K_bytes])^T
//
// Processes 2 Q4_0 blocks (64 elements) per K-tile iteration instead of 1,
// halving outer loop count. This halves the number of threadgroup barriers
// (from K/32 to K/64) and K-loop iterations.
//
// Shared memory: sa[32*64] + sb[32*64] = 4096 halfs = 8192 bytes
//
// A-tile loading: 128 threads, 32 rows x 4 threads/row, each loads 16 elems.
//
// B-tile loading: 2-pass approach. Each pass loads one Q4_0 block (32 elems)
// across 32 rows using 128 threads mapped as 32 rows x 4 threads/row.
// Q4_0 uses de-interleaved nibble ordering.
// ============================================================================

kernel void dequant_tiled_matmul_q4_0_k64(
    device const uchar* W_q4     [[buffer(0)]],   // Q4_0 weights [N, K_bytes]
    device const float* X        [[buffer(1)]],   // [M, K] input batch
    device float*       Y        [[buffer(2)]],   // [M, N] output batch
    constant uint&      M        [[buffer(3)]],   // batch size
    constant uint&      N        [[buffer(4)]],   // output dim
    constant uint&      K        [[buffer(5)]],   // input dim (elements, not bytes)
    uint2 tg_pos                 [[threadgroup_position_in_grid]],
    ushort tiitg                 [[thread_index_in_threadgroup]],
    ushort sgitg                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup half shmem_q4_k64[4096];  // sa[32*64] + sb[32*64]
    threadgroup half* sa = shmem_q4_k64;
    threadgroup half* sb = shmem_q4_k64 + TILE_M * TILE_K_64;

    uint tile_m_start = tg_pos.y * TILE_M;
    uint tile_n_start = tg_pos.x * TILE_N;

    ushort sg_r = sgitg / SG_COLS;
    ushort sg_c = sgitg % SG_COLS;

    simdgroup_float8x8 mc[2][2];
    FOR_UNROLL (ushort i = 0; i < 2; i++)
        FOR_UNROLL (ushort j = 0; j < 2; j++)
            mc[i][j] = make_filled_simdgroup_matrix<float, 8>(0.f);

    uint num_blocks_per_row = (K + Q4_GROUP_SIZE - 1) / Q4_GROUP_SIZE;
    uint row_bytes = num_blocks_per_row * Q4_BLOCK_SIZE;

    uint num_k_tiles = (K + TILE_K_64 - 1) / TILE_K_64;

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        uint k_base = kt * TILE_K_64;

        // Load A tile: 32 rows x 64 cols
        // FC_BC_M/FC_BC_K: when false (aligned), compiler eliminates boundary checks
        {
            ushort row = tiitg >> 2;           // tiitg / 4, range 0..31
            ushort col_group = tiitg & 3;      // tiitg % 4, range 0..3
            ushort k_start = col_group << 4;   // col_group * 16

            uint gm = tile_m_start + row;
            uint gk = k_base + k_start;
            threadgroup half* sa_ptr = sa + row * TILE_K_64 + k_start;

            if (FC_BC_M || FC_BC_K) {
                if (gm < M && gk + 15 < K) {
                    device const float4* x_ptr = (device const float4*)(X + gm * K + gk);
                    float4 v0 = x_ptr[0];
                    float4 v1 = x_ptr[1];
                    float4 v2 = x_ptr[2];
                    float4 v3 = x_ptr[3];
                    sa_ptr[0]  = (half)v0.x; sa_ptr[1]  = (half)v0.y;
                    sa_ptr[2]  = (half)v0.z; sa_ptr[3]  = (half)v0.w;
                    sa_ptr[4]  = (half)v1.x; sa_ptr[5]  = (half)v1.y;
                    sa_ptr[6]  = (half)v1.z; sa_ptr[7]  = (half)v1.w;
                    sa_ptr[8]  = (half)v2.x; sa_ptr[9]  = (half)v2.y;
                    sa_ptr[10] = (half)v2.z; sa_ptr[11] = (half)v2.w;
                    sa_ptr[12] = (half)v3.x; sa_ptr[13] = (half)v3.y;
                    sa_ptr[14] = (half)v3.z; sa_ptr[15] = (half)v3.w;
                } else {
                    FOR_UNROLL (ushort i = 0; i < 16; i++) {
                        uint gk_i = gk + i;
                        sa_ptr[i] = (gm < M && gk_i < K) ? (half)X[gm * K + gk_i] : (half)0.0h;
                    }
                }
            } else {
                device const float4* x_ptr = (device const float4*)(X + gm * K + gk);
                float4 v0 = x_ptr[0];
                float4 v1 = x_ptr[1];
                float4 v2 = x_ptr[2];
                float4 v3 = x_ptr[3];
                sa_ptr[0]  = (half)v0.x; sa_ptr[1]  = (half)v0.y;
                sa_ptr[2]  = (half)v0.z; sa_ptr[3]  = (half)v0.w;
                sa_ptr[4]  = (half)v1.x; sa_ptr[5]  = (half)v1.y;
                sa_ptr[6]  = (half)v1.z; sa_ptr[7]  = (half)v1.w;
                sa_ptr[8]  = (half)v2.x; sa_ptr[9]  = (half)v2.y;
                sa_ptr[10] = (half)v2.z; sa_ptr[11] = (half)v2.w;
                sa_ptr[12] = (half)v3.x; sa_ptr[13] = (half)v3.y;
                sa_ptr[14] = (half)v3.z; sa_ptr[15] = (half)v3.w;
            }
        }

        // Load B tile: 32 rows x 64 cols = 2 Q4_0 blocks per row
        // FC_BC_N/FC_BC_K: when false (aligned), compiler eliminates boundary checks
        // Q4_0 de-interleaved nibble ordering: elem 0-15 = lo nibbles, 16-31 = hi nibbles
        {
            ushort n_local = tiitg >> 2;       // 0..31 (which row)
            ushort t_in_row = tiitg & 3;       // 0..3 (quarter of 32 elements)
            uint gn = tile_n_start + n_local;

            // Pass 0: first Q4_0 block (k_base + 0..31)
            {
                uint block_idx = k_base / Q4_GROUP_SIZE;
                ushort k_offset = t_in_row * 8;
                threadgroup half* sb_row = sb + n_local * TILE_K_64 + k_offset;

                if (FC_BC_N || FC_BC_K) {
                    if (gn < N && k_base + k_offset < K) {
                        uint block_offset = gn * row_bytes + block_idx * Q4_BLOCK_SIZE;
                        half scale = as_type<half>(*(device const ushort*)(W_q4 + block_offset));
                        device const uchar* qdata = W_q4 + block_offset + 2;

                        uint k_remaining = K - (k_base + k_offset);
                        ushort count = (k_remaining >= 8) ? 8 : (ushort)k_remaining;

                        ushort byte_start = (k_offset < 16) ? k_offset : (k_offset - 16);
                        bool use_hi = (k_offset >= 16);

                        if (count == 8) {
                            if (use_hi) {
                                sb_row[0] = scale * (half)(int((qdata[byte_start + 0] >> 4)) - 8);
                                sb_row[1] = scale * (half)(int((qdata[byte_start + 1] >> 4)) - 8);
                                sb_row[2] = scale * (half)(int((qdata[byte_start + 2] >> 4)) - 8);
                                sb_row[3] = scale * (half)(int((qdata[byte_start + 3] >> 4)) - 8);
                                sb_row[4] = scale * (half)(int((qdata[byte_start + 4] >> 4)) - 8);
                                sb_row[5] = scale * (half)(int((qdata[byte_start + 5] >> 4)) - 8);
                                sb_row[6] = scale * (half)(int((qdata[byte_start + 6] >> 4)) - 8);
                                sb_row[7] = scale * (half)(int((qdata[byte_start + 7] >> 4)) - 8);
                            } else {
                                sb_row[0] = scale * (half)(int((qdata[byte_start + 0] & 0xF)) - 8);
                                sb_row[1] = scale * (half)(int((qdata[byte_start + 1] & 0xF)) - 8);
                                sb_row[2] = scale * (half)(int((qdata[byte_start + 2] & 0xF)) - 8);
                                sb_row[3] = scale * (half)(int((qdata[byte_start + 3] & 0xF)) - 8);
                                sb_row[4] = scale * (half)(int((qdata[byte_start + 4] & 0xF)) - 8);
                                sb_row[5] = scale * (half)(int((qdata[byte_start + 5] & 0xF)) - 8);
                                sb_row[6] = scale * (half)(int((qdata[byte_start + 6] & 0xF)) - 8);
                                sb_row[7] = scale * (half)(int((qdata[byte_start + 7] & 0xF)) - 8);
                            }
                        } else {
                            for (ushort i = 0; i < count; i++) {
                                ushort elem_idx = k_offset + i;
                                ushort byte_idx = (elem_idx < 16) ? elem_idx : (elem_idx - 16);
                                uchar raw = qdata[byte_idx];
                                int nibble = (elem_idx >= 16) ? int((raw >> 4) & 0xF) : int(raw & 0xF);
                                sb_row[i] = scale * (half)(nibble - 8);
                            }
                            for (ushort i = count; i < 8; i++) {
                                sb_row[i] = (half)0.0h;
                            }
                        }
                    } else {
                        FOR_UNROLL (ushort i = 0; i < 8; i++) {
                            sb_row[i] = (half)0.0h;
                        }
                    }
                } else {
                    uint block_offset = gn * row_bytes + block_idx * Q4_BLOCK_SIZE;
                    half scale = as_type<half>(*(device const ushort*)(W_q4 + block_offset));
                    device const uchar* qdata = W_q4 + block_offset + 2;

                    ushort byte_start = (k_offset < 16) ? k_offset : (k_offset - 16);
                    bool use_hi = (k_offset >= 16);

                    if (use_hi) {
                        sb_row[0] = scale * (half)(int((qdata[byte_start + 0] >> 4)) - 8);
                        sb_row[1] = scale * (half)(int((qdata[byte_start + 1] >> 4)) - 8);
                        sb_row[2] = scale * (half)(int((qdata[byte_start + 2] >> 4)) - 8);
                        sb_row[3] = scale * (half)(int((qdata[byte_start + 3] >> 4)) - 8);
                        sb_row[4] = scale * (half)(int((qdata[byte_start + 4] >> 4)) - 8);
                        sb_row[5] = scale * (half)(int((qdata[byte_start + 5] >> 4)) - 8);
                        sb_row[6] = scale * (half)(int((qdata[byte_start + 6] >> 4)) - 8);
                        sb_row[7] = scale * (half)(int((qdata[byte_start + 7] >> 4)) - 8);
                    } else {
                        sb_row[0] = scale * (half)(int((qdata[byte_start + 0] & 0xF)) - 8);
                        sb_row[1] = scale * (half)(int((qdata[byte_start + 1] & 0xF)) - 8);
                        sb_row[2] = scale * (half)(int((qdata[byte_start + 2] & 0xF)) - 8);
                        sb_row[3] = scale * (half)(int((qdata[byte_start + 3] & 0xF)) - 8);
                        sb_row[4] = scale * (half)(int((qdata[byte_start + 4] & 0xF)) - 8);
                        sb_row[5] = scale * (half)(int((qdata[byte_start + 5] & 0xF)) - 8);
                        sb_row[6] = scale * (half)(int((qdata[byte_start + 6] & 0xF)) - 8);
                        sb_row[7] = scale * (half)(int((qdata[byte_start + 7] & 0xF)) - 8);
                    }
                }
            }

            // Pass 1: second Q4_0 block (k_base + 32..63)
            {
                uint block_idx = (k_base + 32) / Q4_GROUP_SIZE;
                ushort k_offset = t_in_row * 8;
                threadgroup half* sb_row = sb + n_local * TILE_K_64 + 32 + k_offset;

                if (FC_BC_N || FC_BC_K) {
                    if (gn < N && k_base + 32 + k_offset < K) {
                        uint block_offset = gn * row_bytes + block_idx * Q4_BLOCK_SIZE;
                        half scale = as_type<half>(*(device const ushort*)(W_q4 + block_offset));
                        device const uchar* qdata = W_q4 + block_offset + 2;

                        uint k_remaining = K - (k_base + 32 + k_offset);
                        ushort count = (k_remaining >= 8) ? 8 : (ushort)k_remaining;

                        ushort byte_start = (k_offset < 16) ? k_offset : (k_offset - 16);
                        bool use_hi = (k_offset >= 16);

                        if (count == 8) {
                            if (use_hi) {
                                sb_row[0] = scale * (half)(int((qdata[byte_start + 0] >> 4)) - 8);
                                sb_row[1] = scale * (half)(int((qdata[byte_start + 1] >> 4)) - 8);
                                sb_row[2] = scale * (half)(int((qdata[byte_start + 2] >> 4)) - 8);
                                sb_row[3] = scale * (half)(int((qdata[byte_start + 3] >> 4)) - 8);
                                sb_row[4] = scale * (half)(int((qdata[byte_start + 4] >> 4)) - 8);
                                sb_row[5] = scale * (half)(int((qdata[byte_start + 5] >> 4)) - 8);
                                sb_row[6] = scale * (half)(int((qdata[byte_start + 6] >> 4)) - 8);
                                sb_row[7] = scale * (half)(int((qdata[byte_start + 7] >> 4)) - 8);
                            } else {
                                sb_row[0] = scale * (half)(int((qdata[byte_start + 0] & 0xF)) - 8);
                                sb_row[1] = scale * (half)(int((qdata[byte_start + 1] & 0xF)) - 8);
                                sb_row[2] = scale * (half)(int((qdata[byte_start + 2] & 0xF)) - 8);
                                sb_row[3] = scale * (half)(int((qdata[byte_start + 3] & 0xF)) - 8);
                                sb_row[4] = scale * (half)(int((qdata[byte_start + 4] & 0xF)) - 8);
                                sb_row[5] = scale * (half)(int((qdata[byte_start + 5] & 0xF)) - 8);
                                sb_row[6] = scale * (half)(int((qdata[byte_start + 6] & 0xF)) - 8);
                                sb_row[7] = scale * (half)(int((qdata[byte_start + 7] & 0xF)) - 8);
                            }
                        } else {
                            for (ushort i = 0; i < count; i++) {
                                ushort elem_idx = k_offset + i;
                                ushort byte_idx = (elem_idx < 16) ? elem_idx : (elem_idx - 16);
                                uchar raw = qdata[byte_idx];
                                int nibble = (elem_idx >= 16) ? int((raw >> 4) & 0xF) : int(raw & 0xF);
                                sb_row[i] = scale * (half)(nibble - 8);
                            }
                            for (ushort i = count; i < 8; i++) {
                                sb_row[i] = (half)0.0h;
                            }
                        }
                    } else {
                        FOR_UNROLL (ushort i = 0; i < 8; i++) {
                            sb_row[i] = (half)0.0h;
                        }
                    }
                } else {
                    uint block_offset = gn * row_bytes + block_idx * Q4_BLOCK_SIZE;
                    half scale = as_type<half>(*(device const ushort*)(W_q4 + block_offset));
                    device const uchar* qdata = W_q4 + block_offset + 2;

                    ushort byte_start = (k_offset < 16) ? k_offset : (k_offset - 16);
                    bool use_hi = (k_offset >= 16);

                    if (use_hi) {
                        sb_row[0] = scale * (half)(int((qdata[byte_start + 0] >> 4)) - 8);
                        sb_row[1] = scale * (half)(int((qdata[byte_start + 1] >> 4)) - 8);
                        sb_row[2] = scale * (half)(int((qdata[byte_start + 2] >> 4)) - 8);
                        sb_row[3] = scale * (half)(int((qdata[byte_start + 3] >> 4)) - 8);
                        sb_row[4] = scale * (half)(int((qdata[byte_start + 4] >> 4)) - 8);
                        sb_row[5] = scale * (half)(int((qdata[byte_start + 5] >> 4)) - 8);
                        sb_row[6] = scale * (half)(int((qdata[byte_start + 6] >> 4)) - 8);
                        sb_row[7] = scale * (half)(int((qdata[byte_start + 7] >> 4)) - 8);
                    } else {
                        sb_row[0] = scale * (half)(int((qdata[byte_start + 0] & 0xF)) - 8);
                        sb_row[1] = scale * (half)(int((qdata[byte_start + 1] & 0xF)) - 8);
                        sb_row[2] = scale * (half)(int((qdata[byte_start + 2] & 0xF)) - 8);
                        sb_row[3] = scale * (half)(int((qdata[byte_start + 3] & 0xF)) - 8);
                        sb_row[4] = scale * (half)(int((qdata[byte_start + 4] & 0xF)) - 8);
                        sb_row[5] = scale * (half)(int((qdata[byte_start + 5] & 0xF)) - 8);
                        sb_row[6] = scale * (half)(int((qdata[byte_start + 6] & 0xF)) - 8);
                        sb_row[7] = scale * (half)(int((qdata[byte_start + 7] & 0xF)) - 8);
                    }
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // MMA: 8 iterations (TILE_K_64=64, step by 8)
        FOR_UNROLL (ushort ks = 0; ks < TILE_K_64; ks += 8) {
            simdgroup_half8x8 ma[2];
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(ma[0], sa + (sg_r * 16 + 0) * TILE_K_64 + ks, TILE_K_64);
            simdgroup_load(ma[1], sa + (sg_r * 16 + 8) * TILE_K_64 + ks, TILE_K_64);

            simdgroup_half8x8 mb[2];
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(mb[0], sb + (sg_c * 16 + 0) * TILE_K_64 + ks, TILE_K_64, ulong2(0,0), true);
            simdgroup_load(mb[1], sb + (sg_c * 16 + 8) * TILE_K_64 + ks, TILE_K_64, ulong2(0,0), true);

            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_multiply_accumulate(mc[0][0], ma[0], mb[0], mc[0][0]);
            simdgroup_multiply_accumulate(mc[0][1], ma[0], mb[1], mc[0][1]);
            simdgroup_multiply_accumulate(mc[1][0], ma[1], mb[0], mc[1][0]);
            simdgroup_multiply_accumulate(mc[1][1], ma[1], mb[1], mc[1][1]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store results
    uint sg_m_base = tile_m_start + sg_r * 16;
    uint sg_n_base = tile_n_start + sg_c * 16;

    if (FC_BC_M || FC_BC_N) {
        if (sg_m_base + 16 <= M && sg_n_base + 16 <= N) {
            FOR_UNROLL (ushort i = 0; i < 2; i++)
                FOR_UNROLL (ushort j = 0; j < 2; j++)
                    simdgroup_store(mc[i][j], Y + (sg_m_base + i * 8) * N + (sg_n_base + j * 8), N);
        } else {
            threadgroup float* sc = (threadgroup float*)shmem_q4_k64;
            threadgroup float* my_sc = sc + sgitg * 256;
            FOR_UNROLL (ushort i = 0; i < 2; i++)
                FOR_UNROLL (ushort j = 0; j < 2; j++)
                    simdgroup_store(mc[i][j], my_sc + (i * 8) * 16 + (j * 8), 16);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            ushort lane = tiitg % 32;
            for (ushort idx = lane; idx < 256; idx += 32) {
                ushort local_m = idx / 16;
                ushort local_n = idx % 16;
                uint gm = sg_m_base + local_m;
                uint gn = sg_n_base + local_n;
                if (gm < M && gn < N) {
                    Y[gm * N + gn] = my_sc[local_m * 16 + local_n];
                }
            }
        }
    } else {
        FOR_UNROLL (ushort i = 0; i < 2; i++)
            FOR_UNROLL (ushort j = 0; j < 2; j++)
                simdgroup_store(mc[i][j], Y + (sg_m_base + i * 8) * N + (sg_n_base + j * 8), N);
    }
}

// ============================================================================
// dequant_tiled_matmul_q4_0_k64_residual_batched: K64 variant + residual add
//
// Y[m,n] = X[M,K] * dequant(W_q4[N,K_bytes])^T + R[m,n]
// Same as dequant_tiled_matmul_q4_0_k64 but fuses residual add at writeback.
// ============================================================================

kernel void dequant_tiled_matmul_q4_0_k64_residual_batched(
    device const uchar* W_q4     [[buffer(0)]],   // Q4_0 weights [N, K_bytes]
    device const float* X        [[buffer(1)]],   // [M, K] input batch
    device float*       Y        [[buffer(2)]],   // [M, N] output batch
    constant uint&      M        [[buffer(3)]],   // batch size
    constant uint&      N        [[buffer(4)]],   // output dim
    constant uint&      K        [[buffer(5)]],   // input dim (elements, not bytes)
    device const float* R        [[buffer(6)]],   // [M, N] residual to add
    uint2 tg_pos                 [[threadgroup_position_in_grid]],
    ushort tiitg                 [[thread_index_in_threadgroup]],
    ushort sgitg                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup half shmem_q4_k64r[4096];
    threadgroup half* sa = shmem_q4_k64r;
    threadgroup half* sb = shmem_q4_k64r + TILE_M * TILE_K_64;

    uint tile_m_start = tg_pos.y * TILE_M;
    uint tile_n_start = tg_pos.x * TILE_N;

    ushort sg_r = sgitg / SG_COLS;
    ushort sg_c = sgitg % SG_COLS;

    simdgroup_float8x8 mc[2][2];
    FOR_UNROLL (ushort i = 0; i < 2; i++)
        FOR_UNROLL (ushort j = 0; j < 2; j++)
            mc[i][j] = make_filled_simdgroup_matrix<float, 8>(0.f);

    uint num_blocks_per_row = (K + Q4_GROUP_SIZE - 1) / Q4_GROUP_SIZE;
    uint row_bytes = num_blocks_per_row * Q4_BLOCK_SIZE;

    uint num_k_tiles = (K + TILE_K_64 - 1) / TILE_K_64;

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        uint k_base = kt * TILE_K_64;

        // Load A tile: 32 rows x 64 cols
        // FC_BC_M/FC_BC_K: when false (aligned), compiler eliminates boundary checks
        {
            ushort row = tiitg >> 2;
            ushort col_group = tiitg & 3;
            ushort k_start = col_group << 4;   // col_group * 16

            uint gm = tile_m_start + row;
            uint gk = k_base + k_start;
            threadgroup half* sa_ptr = sa + row * TILE_K_64 + k_start;

            if (FC_BC_M || FC_BC_K) {
                if (gm < M && gk + 15 < K) {
                    device const float4* x_ptr = (device const float4*)(X + gm * K + gk);
                    float4 v0 = x_ptr[0];
                    float4 v1 = x_ptr[1];
                    float4 v2 = x_ptr[2];
                    float4 v3 = x_ptr[3];
                    sa_ptr[0]  = (half)v0.x; sa_ptr[1]  = (half)v0.y;
                    sa_ptr[2]  = (half)v0.z; sa_ptr[3]  = (half)v0.w;
                    sa_ptr[4]  = (half)v1.x; sa_ptr[5]  = (half)v1.y;
                    sa_ptr[6]  = (half)v1.z; sa_ptr[7]  = (half)v1.w;
                    sa_ptr[8]  = (half)v2.x; sa_ptr[9]  = (half)v2.y;
                    sa_ptr[10] = (half)v2.z; sa_ptr[11] = (half)v2.w;
                    sa_ptr[12] = (half)v3.x; sa_ptr[13] = (half)v3.y;
                    sa_ptr[14] = (half)v3.z; sa_ptr[15] = (half)v3.w;
                } else {
                    FOR_UNROLL (ushort i = 0; i < 16; i++) {
                        uint gk_i = gk + i;
                        sa_ptr[i] = (gm < M && gk_i < K) ? (half)X[gm * K + gk_i] : (half)0.0h;
                    }
                }
            } else {
                device const float4* x_ptr = (device const float4*)(X + gm * K + gk);
                float4 v0 = x_ptr[0];
                float4 v1 = x_ptr[1];
                float4 v2 = x_ptr[2];
                float4 v3 = x_ptr[3];
                sa_ptr[0]  = (half)v0.x; sa_ptr[1]  = (half)v0.y;
                sa_ptr[2]  = (half)v0.z; sa_ptr[3]  = (half)v0.w;
                sa_ptr[4]  = (half)v1.x; sa_ptr[5]  = (half)v1.y;
                sa_ptr[6]  = (half)v1.z; sa_ptr[7]  = (half)v1.w;
                sa_ptr[8]  = (half)v2.x; sa_ptr[9]  = (half)v2.y;
                sa_ptr[10] = (half)v2.z; sa_ptr[11] = (half)v2.w;
                sa_ptr[12] = (half)v3.x; sa_ptr[13] = (half)v3.y;
                sa_ptr[14] = (half)v3.z; sa_ptr[15] = (half)v3.w;
            }
        }

        // Load B tile: 2-pass Q4_0 dequant (2 blocks per row)
        // FC_BC_N/FC_BC_K: when false (aligned), compiler eliminates boundary checks
        {
            ushort n_local = tiitg >> 2;
            ushort t_in_row = tiitg & 3;
            uint gn = tile_n_start + n_local;

            // Pass 0: first block (k_base + 0..31)
            {
                uint block_idx = k_base / Q4_GROUP_SIZE;
                ushort k_offset = t_in_row * 8;
                threadgroup half* sb_row = sb + n_local * TILE_K_64 + k_offset;

                if (FC_BC_N || FC_BC_K) {
                    if (gn < N && k_base + k_offset < K) {
                        uint block_offset = gn * row_bytes + block_idx * Q4_BLOCK_SIZE;
                        half scale = as_type<half>(*(device const ushort*)(W_q4 + block_offset));
                        device const uchar* qdata = W_q4 + block_offset + 2;

                        uint k_remaining = K - (k_base + k_offset);
                        ushort count = (k_remaining >= 8) ? 8 : (ushort)k_remaining;

                        ushort byte_start = (k_offset < 16) ? k_offset : (k_offset - 16);
                        bool use_hi = (k_offset >= 16);

                        if (count == 8) {
                            if (use_hi) {
                                sb_row[0] = scale * (half)(int((qdata[byte_start + 0] >> 4)) - 8);
                                sb_row[1] = scale * (half)(int((qdata[byte_start + 1] >> 4)) - 8);
                                sb_row[2] = scale * (half)(int((qdata[byte_start + 2] >> 4)) - 8);
                                sb_row[3] = scale * (half)(int((qdata[byte_start + 3] >> 4)) - 8);
                                sb_row[4] = scale * (half)(int((qdata[byte_start + 4] >> 4)) - 8);
                                sb_row[5] = scale * (half)(int((qdata[byte_start + 5] >> 4)) - 8);
                                sb_row[6] = scale * (half)(int((qdata[byte_start + 6] >> 4)) - 8);
                                sb_row[7] = scale * (half)(int((qdata[byte_start + 7] >> 4)) - 8);
                            } else {
                                sb_row[0] = scale * (half)(int((qdata[byte_start + 0] & 0xF)) - 8);
                                sb_row[1] = scale * (half)(int((qdata[byte_start + 1] & 0xF)) - 8);
                                sb_row[2] = scale * (half)(int((qdata[byte_start + 2] & 0xF)) - 8);
                                sb_row[3] = scale * (half)(int((qdata[byte_start + 3] & 0xF)) - 8);
                                sb_row[4] = scale * (half)(int((qdata[byte_start + 4] & 0xF)) - 8);
                                sb_row[5] = scale * (half)(int((qdata[byte_start + 5] & 0xF)) - 8);
                                sb_row[6] = scale * (half)(int((qdata[byte_start + 6] & 0xF)) - 8);
                                sb_row[7] = scale * (half)(int((qdata[byte_start + 7] & 0xF)) - 8);
                            }
                        } else {
                            for (ushort i = 0; i < count; i++) {
                                ushort elem_idx = k_offset + i;
                                ushort byte_idx = (elem_idx < 16) ? elem_idx : (elem_idx - 16);
                                uchar raw = qdata[byte_idx];
                                int nibble = (elem_idx >= 16) ? int((raw >> 4) & 0xF) : int(raw & 0xF);
                                sb_row[i] = scale * (half)(nibble - 8);
                            }
                            for (ushort i = count; i < 8; i++) {
                                sb_row[i] = (half)0.0h;
                            }
                        }
                    } else {
                        FOR_UNROLL (ushort i = 0; i < 8; i++) {
                            sb_row[i] = (half)0.0h;
                        }
                    }
                } else {
                    uint block_offset = gn * row_bytes + block_idx * Q4_BLOCK_SIZE;
                    half scale = as_type<half>(*(device const ushort*)(W_q4 + block_offset));
                    device const uchar* qdata = W_q4 + block_offset + 2;

                    ushort byte_start = (k_offset < 16) ? k_offset : (k_offset - 16);
                    bool use_hi = (k_offset >= 16);

                    if (use_hi) {
                        sb_row[0] = scale * (half)(int((qdata[byte_start + 0] >> 4)) - 8);
                        sb_row[1] = scale * (half)(int((qdata[byte_start + 1] >> 4)) - 8);
                        sb_row[2] = scale * (half)(int((qdata[byte_start + 2] >> 4)) - 8);
                        sb_row[3] = scale * (half)(int((qdata[byte_start + 3] >> 4)) - 8);
                        sb_row[4] = scale * (half)(int((qdata[byte_start + 4] >> 4)) - 8);
                        sb_row[5] = scale * (half)(int((qdata[byte_start + 5] >> 4)) - 8);
                        sb_row[6] = scale * (half)(int((qdata[byte_start + 6] >> 4)) - 8);
                        sb_row[7] = scale * (half)(int((qdata[byte_start + 7] >> 4)) - 8);
                    } else {
                        sb_row[0] = scale * (half)(int((qdata[byte_start + 0] & 0xF)) - 8);
                        sb_row[1] = scale * (half)(int((qdata[byte_start + 1] & 0xF)) - 8);
                        sb_row[2] = scale * (half)(int((qdata[byte_start + 2] & 0xF)) - 8);
                        sb_row[3] = scale * (half)(int((qdata[byte_start + 3] & 0xF)) - 8);
                        sb_row[4] = scale * (half)(int((qdata[byte_start + 4] & 0xF)) - 8);
                        sb_row[5] = scale * (half)(int((qdata[byte_start + 5] & 0xF)) - 8);
                        sb_row[6] = scale * (half)(int((qdata[byte_start + 6] & 0xF)) - 8);
                        sb_row[7] = scale * (half)(int((qdata[byte_start + 7] & 0xF)) - 8);
                    }
                }
            }

            // Pass 1: second block (k_base + 32..63)
            {
                uint block_idx = (k_base + 32) / Q4_GROUP_SIZE;
                ushort k_offset = t_in_row * 8;
                threadgroup half* sb_row = sb + n_local * TILE_K_64 + 32 + k_offset;

                if (FC_BC_N || FC_BC_K) {
                    if (gn < N && k_base + 32 + k_offset < K) {
                        uint block_offset = gn * row_bytes + block_idx * Q4_BLOCK_SIZE;
                        half scale = as_type<half>(*(device const ushort*)(W_q4 + block_offset));
                        device const uchar* qdata = W_q4 + block_offset + 2;

                        uint k_remaining = K - (k_base + 32 + k_offset);
                        ushort count = (k_remaining >= 8) ? 8 : (ushort)k_remaining;

                        ushort byte_start = (k_offset < 16) ? k_offset : (k_offset - 16);
                        bool use_hi = (k_offset >= 16);

                        if (count == 8) {
                            if (use_hi) {
                                sb_row[0] = scale * (half)(int((qdata[byte_start + 0] >> 4)) - 8);
                                sb_row[1] = scale * (half)(int((qdata[byte_start + 1] >> 4)) - 8);
                                sb_row[2] = scale * (half)(int((qdata[byte_start + 2] >> 4)) - 8);
                                sb_row[3] = scale * (half)(int((qdata[byte_start + 3] >> 4)) - 8);
                                sb_row[4] = scale * (half)(int((qdata[byte_start + 4] >> 4)) - 8);
                                sb_row[5] = scale * (half)(int((qdata[byte_start + 5] >> 4)) - 8);
                                sb_row[6] = scale * (half)(int((qdata[byte_start + 6] >> 4)) - 8);
                                sb_row[7] = scale * (half)(int((qdata[byte_start + 7] >> 4)) - 8);
                            } else {
                                sb_row[0] = scale * (half)(int((qdata[byte_start + 0] & 0xF)) - 8);
                                sb_row[1] = scale * (half)(int((qdata[byte_start + 1] & 0xF)) - 8);
                                sb_row[2] = scale * (half)(int((qdata[byte_start + 2] & 0xF)) - 8);
                                sb_row[3] = scale * (half)(int((qdata[byte_start + 3] & 0xF)) - 8);
                                sb_row[4] = scale * (half)(int((qdata[byte_start + 4] & 0xF)) - 8);
                                sb_row[5] = scale * (half)(int((qdata[byte_start + 5] & 0xF)) - 8);
                                sb_row[6] = scale * (half)(int((qdata[byte_start + 6] & 0xF)) - 8);
                                sb_row[7] = scale * (half)(int((qdata[byte_start + 7] & 0xF)) - 8);
                            }
                        } else {
                            for (ushort i = 0; i < count; i++) {
                                ushort elem_idx = k_offset + i;
                                ushort byte_idx = (elem_idx < 16) ? elem_idx : (elem_idx - 16);
                                uchar raw = qdata[byte_idx];
                                int nibble = (elem_idx >= 16) ? int((raw >> 4) & 0xF) : int(raw & 0xF);
                                sb_row[i] = scale * (half)(nibble - 8);
                            }
                            for (ushort i = count; i < 8; i++) {
                                sb_row[i] = (half)0.0h;
                            }
                        }
                    } else {
                        FOR_UNROLL (ushort i = 0; i < 8; i++) {
                            sb_row[i] = (half)0.0h;
                        }
                    }
                } else {
                    uint block_offset = gn * row_bytes + block_idx * Q4_BLOCK_SIZE;
                    half scale = as_type<half>(*(device const ushort*)(W_q4 + block_offset));
                    device const uchar* qdata = W_q4 + block_offset + 2;

                    ushort byte_start = (k_offset < 16) ? k_offset : (k_offset - 16);
                    bool use_hi = (k_offset >= 16);

                    if (use_hi) {
                        sb_row[0] = scale * (half)(int((qdata[byte_start + 0] >> 4)) - 8);
                        sb_row[1] = scale * (half)(int((qdata[byte_start + 1] >> 4)) - 8);
                        sb_row[2] = scale * (half)(int((qdata[byte_start + 2] >> 4)) - 8);
                        sb_row[3] = scale * (half)(int((qdata[byte_start + 3] >> 4)) - 8);
                        sb_row[4] = scale * (half)(int((qdata[byte_start + 4] >> 4)) - 8);
                        sb_row[5] = scale * (half)(int((qdata[byte_start + 5] >> 4)) - 8);
                        sb_row[6] = scale * (half)(int((qdata[byte_start + 6] >> 4)) - 8);
                        sb_row[7] = scale * (half)(int((qdata[byte_start + 7] >> 4)) - 8);
                    } else {
                        sb_row[0] = scale * (half)(int((qdata[byte_start + 0] & 0xF)) - 8);
                        sb_row[1] = scale * (half)(int((qdata[byte_start + 1] & 0xF)) - 8);
                        sb_row[2] = scale * (half)(int((qdata[byte_start + 2] & 0xF)) - 8);
                        sb_row[3] = scale * (half)(int((qdata[byte_start + 3] & 0xF)) - 8);
                        sb_row[4] = scale * (half)(int((qdata[byte_start + 4] & 0xF)) - 8);
                        sb_row[5] = scale * (half)(int((qdata[byte_start + 5] & 0xF)) - 8);
                        sb_row[6] = scale * (half)(int((qdata[byte_start + 6] & 0xF)) - 8);
                        sb_row[7] = scale * (half)(int((qdata[byte_start + 7] & 0xF)) - 8);
                    }
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (ushort ks = 0; ks < TILE_K_64; ks += 8) {
            simdgroup_half8x8 ma[2];
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(ma[0], sa + (sg_r * 16 + 0) * TILE_K_64 + ks, TILE_K_64);
            simdgroup_load(ma[1], sa + (sg_r * 16 + 8) * TILE_K_64 + ks, TILE_K_64);

            simdgroup_half8x8 mb[2];
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(mb[0], sb + (sg_c * 16 + 0) * TILE_K_64 + ks, TILE_K_64, ulong2(0,0), true);
            simdgroup_load(mb[1], sb + (sg_c * 16 + 8) * TILE_K_64 + ks, TILE_K_64, ulong2(0,0), true);

            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_multiply_accumulate(mc[0][0], ma[0], mb[0], mc[0][0]);
            simdgroup_multiply_accumulate(mc[0][1], ma[0], mb[1], mc[0][1]);
            simdgroup_multiply_accumulate(mc[1][0], ma[1], mb[0], mc[1][0]);
            simdgroup_multiply_accumulate(mc[1][1], ma[1], mb[1], mc[1][1]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store results with residual add: Y[m,n] = GEMM[m,n] + R[m,n]
    uint sg_m_base = tile_m_start + sg_r * 16;
    uint sg_n_base = tile_n_start + sg_c * 16;

    {
        threadgroup float* sc = (threadgroup float*)shmem_q4_k64r;
        threadgroup float* my_sc = sc + sgitg * 256;
        FOR_UNROLL (ushort i = 0; i < 2; i++)
            FOR_UNROLL (ushort j = 0; j < 2; j++)
                simdgroup_store(mc[i][j], my_sc + (i * 8) * 16 + (j * 8), 16);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ushort lane = tiitg % 32;
        if (FC_BC_M || FC_BC_N) {
            for (ushort idx = lane; idx < 256; idx += 32) {
                ushort local_m = idx / 16;
                ushort local_n = idx % 16;
                uint gm = sg_m_base + local_m;
                uint gn = sg_n_base + local_n;
                if (gm < M && gn < N) {
                    Y[gm * N + gn] = my_sc[local_m * 16 + local_n] + R[gm * N + gn];
                }
            }
        } else {
            for (ushort idx = lane; idx < 256; idx += 32) {
                ushort local_m = idx / 16;
                ushort local_n = idx % 16;
                uint gm = sg_m_base + local_m;
                uint gn = sg_n_base + local_n;
                Y[gm * N + gn] = my_sc[local_m * 16 + local_n] + R[gm * N + gn];
            }
        }
    }
}

// ============================================================================
// dequant_tiled_matmul_q4_1: Fused Q4_1 dequantization + tiled GEMM via MMA
//
// Y[M,N] = X[M,K] * dequant(W_q41[N,K_bytes])^T
//
// Q4_1 block layout (20 bytes per 32 elements):
//   [f16 scale (2 bytes)] [f16 min (2 bytes)] [16 x uint8 packed nibbles (16 bytes)]
//   dequant value[i] = scale * nibble(i) + min
// ============================================================================

kernel void dequant_tiled_matmul_q4_1(
    device const uchar* W_q4     [[buffer(0)]],   // Q4_1 weights [N, K_bytes]
    device const float* X        [[buffer(1)]],   // [M, K] input batch
    device float*       Y        [[buffer(2)]],   // [M, N] output batch
    constant uint&      M        [[buffer(3)]],   // batch size
    constant uint&      N        [[buffer(4)]],   // output dim
    constant uint&      K        [[buffer(5)]],   // input dim (elements, not bytes)
    uint2 tg_pos                 [[threadgroup_position_in_grid]],
    ushort tiitg                 [[thread_index_in_threadgroup]],
    ushort sgitg                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup half shmem[2048];
    threadgroup half* sa = shmem;
    threadgroup half* sb = shmem + TILE_M * TILE_K;

    uint tile_m_start = tg_pos.y * TILE_M;
    uint tile_n_start = tg_pos.x * TILE_N;

    ushort sg_r = sgitg / SG_COLS;
    ushort sg_c = sgitg % SG_COLS;

    simdgroup_float8x8 mc[2][2];
    FOR_UNROLL (ushort i = 0; i < 2; i++)
        FOR_UNROLL (ushort j = 0; j < 2; j++)
            mc[i][j] = make_filled_simdgroup_matrix<float, 8>(0.f);

    uint num_blocks_per_row = (K + Q4_GROUP_SIZE - 1) / Q4_GROUP_SIZE;
    uint row_bytes = num_blocks_per_row * Q4_1_BLOCK_SIZE;

    uint num_k_tiles = (K + TILE_K - 1) / TILE_K;

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        uint k_base = kt * TILE_K;

        // Load A tile: vectorized 2D mapping
        {
            ushort row = tiitg >> 2;
            ushort col_group = tiitg & 3;
            ushort k_start = col_group << 3;

            uint gm = tile_m_start + row;
            uint gk = k_base + k_start;
            threadgroup half* sa_ptr = sa + row * TILE_K + k_start;

            if (gm < M && gk + 7 < K) {
                device const float4* x_ptr = (device const float4*)(X + gm * K + gk);
                float4 v0 = x_ptr[0];
                float4 v1 = x_ptr[1];
                sa_ptr[0] = (half)v0.x; sa_ptr[1] = (half)v0.y;
                sa_ptr[2] = (half)v0.z; sa_ptr[3] = (half)v0.w;
                sa_ptr[4] = (half)v1.x; sa_ptr[5] = (half)v1.y;
                sa_ptr[6] = (half)v1.z; sa_ptr[7] = (half)v1.w;
            } else {
                FOR_UNROLL (ushort i = 0; i < 8; i++) {
                    uint gk_i = gk + i;
                    sa_ptr[i] = (gm < M && gk_i < K) ? (half)X[gm * K + gk_i] : (half)0.0h;
                }
            }
        }

        // Load B tile: Q4_1 dequant (de-interleaved)
        // TILE_K=32 = Q4_1 group size, so one block per row per K-tile.
        // 128 threads, 32 rows: 4 threads per row, each loads 8 elements.
        {
            uint block_idx = k_base / Q4_GROUP_SIZE;
            ushort n_local = tiitg / 4;
            ushort t_in_row = tiitg % 4;
            ushort k_offset = t_in_row * 8;
            uint gn = tile_n_start + n_local;

            threadgroup half* sb_row = sb + n_local * TILE_K + k_offset;

            if (gn < N && k_base + k_offset < K) {
                uint block_offset = gn * row_bytes + block_idx * Q4_1_BLOCK_SIZE;
                // Q4_1: [f16 scale][f16 min][16B nibbles]
                half scale = as_type<half>(*(device const ushort*)(W_q4 + block_offset));
                half vmin  = as_type<half>(*(device const ushort*)(W_q4 + block_offset + 2));
                device const uchar* qdata = W_q4 + block_offset + 4;

                uint k_remaining = K - (k_base + k_offset);
                ushort count = (k_remaining >= 8) ? 8 : (ushort)k_remaining;

                ushort byte_start = (k_offset < 16) ? k_offset : (k_offset - 16);
                bool use_hi = (k_offset >= 16);

                if (count == 8) {
                    if (use_hi) {
                        sb_row[0] = scale * (half)(qdata[byte_start + 0] >> 4) + vmin;
                        sb_row[1] = scale * (half)(qdata[byte_start + 1] >> 4) + vmin;
                        sb_row[2] = scale * (half)(qdata[byte_start + 2] >> 4) + vmin;
                        sb_row[3] = scale * (half)(qdata[byte_start + 3] >> 4) + vmin;
                        sb_row[4] = scale * (half)(qdata[byte_start + 4] >> 4) + vmin;
                        sb_row[5] = scale * (half)(qdata[byte_start + 5] >> 4) + vmin;
                        sb_row[6] = scale * (half)(qdata[byte_start + 6] >> 4) + vmin;
                        sb_row[7] = scale * (half)(qdata[byte_start + 7] >> 4) + vmin;
                    } else {
                        sb_row[0] = scale * (half)(qdata[byte_start + 0] & 0xF) + vmin;
                        sb_row[1] = scale * (half)(qdata[byte_start + 1] & 0xF) + vmin;
                        sb_row[2] = scale * (half)(qdata[byte_start + 2] & 0xF) + vmin;
                        sb_row[3] = scale * (half)(qdata[byte_start + 3] & 0xF) + vmin;
                        sb_row[4] = scale * (half)(qdata[byte_start + 4] & 0xF) + vmin;
                        sb_row[5] = scale * (half)(qdata[byte_start + 5] & 0xF) + vmin;
                        sb_row[6] = scale * (half)(qdata[byte_start + 6] & 0xF) + vmin;
                        sb_row[7] = scale * (half)(qdata[byte_start + 7] & 0xF) + vmin;
                    }
                } else {
                    for (ushort i = 0; i < count; i++) {
                        ushort elem_idx = k_offset + i;
                        ushort byte_idx = (elem_idx < 16) ? elem_idx : (elem_idx - 16);
                        uchar raw = qdata[byte_idx];
                        ushort nibble = (elem_idx >= 16) ? (raw >> 4) : (raw & 0xF);
                        sb_row[i] = scale * (half)nibble + vmin;
                    }
                    for (ushort i = count; i < 8; i++) {
                        sb_row[i] = (half)0.0h;
                    }
                }
            } else {
                FOR_UNROLL (ushort i = 0; i < 8; i++) {
                    sb_row[i] = (half)0.0h;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (ushort ks = 0; ks < TILE_K; ks += 8) {
            simdgroup_half8x8 ma[2];
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(ma[0], sa + (sg_r * 16 + 0) * TILE_K + ks, TILE_K);
            simdgroup_load(ma[1], sa + (sg_r * 16 + 8) * TILE_K + ks, TILE_K);

            simdgroup_half8x8 mb[2];
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(mb[0], sb + (sg_c * 16 + 0) * TILE_K + ks, TILE_K, ulong2(0,0), true);
            simdgroup_load(mb[1], sb + (sg_c * 16 + 8) * TILE_K + ks, TILE_K, ulong2(0,0), true);

            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_multiply_accumulate(mc[0][0], ma[0], mb[0], mc[0][0]);
            simdgroup_multiply_accumulate(mc[0][1], ma[0], mb[1], mc[0][1]);
            simdgroup_multiply_accumulate(mc[1][0], ma[1], mb[0], mc[1][0]);
            simdgroup_multiply_accumulate(mc[1][1], ma[1], mb[1], mc[1][1]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint sg_m_base = tile_m_start + sg_r * 16;
    uint sg_n_base = tile_n_start + sg_c * 16;

    if (sg_m_base + 16 <= M && sg_n_base + 16 <= N) {
        FOR_UNROLL (ushort i = 0; i < 2; i++)
            FOR_UNROLL (ushort j = 0; j < 2; j++)
                simdgroup_store(mc[i][j], Y + (sg_m_base + i * 8) * N + (sg_n_base + j * 8), N);
    } else {
        threadgroup float* sc = (threadgroup float*)shmem;
        threadgroup float* my_sc = sc + sgitg * 256;
        FOR_UNROLL (ushort i = 0; i < 2; i++)
            FOR_UNROLL (ushort j = 0; j < 2; j++)
                simdgroup_store(mc[i][j], my_sc + (i * 8) * 16 + (j * 8), 16);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ushort lane_id = tiitg % 32;
        for (ushort idx = lane_id; idx < 256; idx += 32) {
            ushort local_m = idx / 16;
            ushort local_n = idx % 16;
            uint gm = sg_m_base + local_m;
            uint gn = sg_n_base + local_n;
            if (gm < M && gn < N) {
                Y[gm * N + gn] = my_sc[local_m * 16 + local_n];
            }
        }
    }
}

// ============================================================================
// dequant_tiled_matmul_q4_1_residual_batched: Tiled Q4_1 GEMM + residual add
//
// Y[m,n] = X[M,K] * dequant(W_q41[N,K_bytes])^T + R[m,n]
//
// Q4_1 block layout (20 bytes per 32 elements):
//   [f16 scale (2 bytes)] [f16 min (2 bytes)] [16 x uint8 packed nibbles (16 bytes)]
//   dequant value[i] = scale * nibble(i) + min
// ============================================================================

kernel void dequant_tiled_matmul_q4_1_residual_batched(
    device const uchar* W_q4     [[buffer(0)]],
    device const float* X        [[buffer(1)]],
    device float*       Y        [[buffer(2)]],
    constant uint&      M        [[buffer(3)]],
    constant uint&      N        [[buffer(4)]],
    constant uint&      K        [[buffer(5)]],
    device const float* R        [[buffer(6)]],
    uint2 tg_pos                 [[threadgroup_position_in_grid]],
    ushort tiitg                 [[thread_index_in_threadgroup]],
    ushort sgitg                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup half shmem[2048];
    threadgroup half* sa = shmem;
    threadgroup half* sb = shmem + TILE_M * TILE_K;

    uint tile_m_start = tg_pos.y * TILE_M;
    uint tile_n_start = tg_pos.x * TILE_N;

    ushort sg_r = sgitg / SG_COLS;
    ushort sg_c = sgitg % SG_COLS;

    simdgroup_float8x8 mc[2][2];
    FOR_UNROLL (ushort i = 0; i < 2; i++)
        FOR_UNROLL (ushort j = 0; j < 2; j++)
            mc[i][j] = make_filled_simdgroup_matrix<float, 8>(0.f);

    uint num_blocks_per_row = (K + Q4_GROUP_SIZE - 1) / Q4_GROUP_SIZE;
    uint row_bytes = num_blocks_per_row * Q4_1_BLOCK_SIZE;

    uint num_k_tiles = (K + TILE_K - 1) / TILE_K;

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        uint k_base = kt * TILE_K;

        // Load A tile
        {
            ushort row = tiitg >> 2;
            ushort col_group = tiitg & 3;
            ushort k_start = col_group << 3;

            uint gm = tile_m_start + row;
            uint gk = k_base + k_start;
            threadgroup half* sa_ptr = sa + row * TILE_K + k_start;

            if (gm < M && gk + 7 < K) {
                device const float4* x_ptr = (device const float4*)(X + gm * K + gk);
                float4 v0 = x_ptr[0];
                float4 v1 = x_ptr[1];
                sa_ptr[0] = (half)v0.x; sa_ptr[1] = (half)v0.y;
                sa_ptr[2] = (half)v0.z; sa_ptr[3] = (half)v0.w;
                sa_ptr[4] = (half)v1.x; sa_ptr[5] = (half)v1.y;
                sa_ptr[6] = (half)v1.z; sa_ptr[7] = (half)v1.w;
            } else {
                FOR_UNROLL (ushort i = 0; i < 8; i++) {
                    uint gk_i = gk + i;
                    sa_ptr[i] = (gm < M && gk_i < K) ? (half)X[gm * K + gk_i] : (half)0.0h;
                }
            }
        }

        // Load B tile: Q4_1 dequant (de-interleaved)
        {
            uint block_idx = k_base / Q4_GROUP_SIZE;
            ushort n_local = tiitg / 4;
            ushort t_in_row = tiitg % 4;
            ushort k_offset = t_in_row * 8;
            uint gn = tile_n_start + n_local;

            threadgroup half* sb_row = sb + n_local * TILE_K + k_offset;

            if (gn < N && k_base + k_offset < K) {
                uint block_offset = gn * row_bytes + block_idx * Q4_1_BLOCK_SIZE;
                half scale = as_type<half>(*(device const ushort*)(W_q4 + block_offset));
                half vmin  = as_type<half>(*(device const ushort*)(W_q4 + block_offset + 2));
                device const uchar* qdata = W_q4 + block_offset + 4;

                uint k_remaining = K - (k_base + k_offset);
                ushort count = (k_remaining >= 8) ? 8 : (ushort)k_remaining;

                ushort byte_start = (k_offset < 16) ? k_offset : (k_offset - 16);
                bool use_hi = (k_offset >= 16);

                if (count == 8) {
                    if (use_hi) {
                        sb_row[0] = scale * (half)(qdata[byte_start + 0] >> 4) + vmin;
                        sb_row[1] = scale * (half)(qdata[byte_start + 1] >> 4) + vmin;
                        sb_row[2] = scale * (half)(qdata[byte_start + 2] >> 4) + vmin;
                        sb_row[3] = scale * (half)(qdata[byte_start + 3] >> 4) + vmin;
                        sb_row[4] = scale * (half)(qdata[byte_start + 4] >> 4) + vmin;
                        sb_row[5] = scale * (half)(qdata[byte_start + 5] >> 4) + vmin;
                        sb_row[6] = scale * (half)(qdata[byte_start + 6] >> 4) + vmin;
                        sb_row[7] = scale * (half)(qdata[byte_start + 7] >> 4) + vmin;
                    } else {
                        sb_row[0] = scale * (half)(qdata[byte_start + 0] & 0xF) + vmin;
                        sb_row[1] = scale * (half)(qdata[byte_start + 1] & 0xF) + vmin;
                        sb_row[2] = scale * (half)(qdata[byte_start + 2] & 0xF) + vmin;
                        sb_row[3] = scale * (half)(qdata[byte_start + 3] & 0xF) + vmin;
                        sb_row[4] = scale * (half)(qdata[byte_start + 4] & 0xF) + vmin;
                        sb_row[5] = scale * (half)(qdata[byte_start + 5] & 0xF) + vmin;
                        sb_row[6] = scale * (half)(qdata[byte_start + 6] & 0xF) + vmin;
                        sb_row[7] = scale * (half)(qdata[byte_start + 7] & 0xF) + vmin;
                    }
                } else {
                    for (ushort i = 0; i < count; i++) {
                        ushort elem_idx = k_offset + i;
                        ushort byte_idx = (elem_idx < 16) ? elem_idx : (elem_idx - 16);
                        uchar raw = qdata[byte_idx];
                        ushort nibble = (elem_idx >= 16) ? (raw >> 4) : (raw & 0xF);
                        sb_row[i] = scale * (half)nibble + vmin;
                    }
                    for (ushort i = count; i < 8; i++) {
                        sb_row[i] = (half)0.0h;
                    }
                }
            } else {
                FOR_UNROLL (ushort i = 0; i < 8; i++) {
                    sb_row[i] = (half)0.0h;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (ushort ks = 0; ks < TILE_K; ks += 8) {
            simdgroup_half8x8 ma[2];
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(ma[0], sa + (sg_r * 16 + 0) * TILE_K + ks, TILE_K);
            simdgroup_load(ma[1], sa + (sg_r * 16 + 8) * TILE_K + ks, TILE_K);

            simdgroup_half8x8 mb[2];
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(mb[0], sb + (sg_c * 16 + 0) * TILE_K + ks, TILE_K, ulong2(0,0), true);
            simdgroup_load(mb[1], sb + (sg_c * 16 + 8) * TILE_K + ks, TILE_K, ulong2(0,0), true);

            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_multiply_accumulate(mc[0][0], ma[0], mb[0], mc[0][0]);
            simdgroup_multiply_accumulate(mc[0][1], ma[0], mb[1], mc[0][1]);
            simdgroup_multiply_accumulate(mc[1][0], ma[1], mb[0], mc[1][0]);
            simdgroup_multiply_accumulate(mc[1][1], ma[1], mb[1], mc[1][1]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store results with residual add
    uint sg_m_base = tile_m_start + sg_r * 16;
    uint sg_n_base = tile_n_start + sg_c * 16;

    {
        threadgroup float* sc = (threadgroup float*)shmem;
        threadgroup float* my_sc = sc + sgitg * 256;
        FOR_UNROLL (ushort i = 0; i < 2; i++)
            FOR_UNROLL (ushort j = 0; j < 2; j++)
                simdgroup_store(mc[i][j], my_sc + (i * 8) * 16 + (j * 8), 16);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ushort lane_id = tiitg % 32;
        for (ushort idx = lane_id; idx < 256; idx += 32) {
            ushort local_m = idx / 16;
            ushort local_n = idx % 16;
            uint gm = sg_m_base + local_m;
            uint gn = sg_n_base + local_n;
            if (gm < M && gn < N) {
                Y[gm * N + gn] = my_sc[local_m * 16 + local_n] + R[gm * N + gn];
            }
        }
    }
}

// ============================================================================
// dequant_tiled_matmul_q4_0_residual_batched: Tiled Q4_0 GEMM + residual add
//
// Y[m,n] = X[M,K] * dequant(W_q4[N,K_bytes])^T + R[m,n]
// ============================================================================

kernel void dequant_tiled_matmul_q4_0_residual_batched(
    device const uchar* W_q4     [[buffer(0)]],
    device const float* X        [[buffer(1)]],
    device float*       Y        [[buffer(2)]],
    constant uint&      M        [[buffer(3)]],
    constant uint&      N        [[buffer(4)]],
    constant uint&      K        [[buffer(5)]],
    device const float* R        [[buffer(6)]],
    uint2 tg_pos                 [[threadgroup_position_in_grid]],
    ushort tiitg                 [[thread_index_in_threadgroup]],
    ushort sgitg                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup half shmem[2048];
    threadgroup half* sa = shmem;
    threadgroup half* sb = shmem + TILE_M * TILE_K;

    uint tile_m_start = tg_pos.y * TILE_M;
    uint tile_n_start = tg_pos.x * TILE_N;

    ushort sg_r = sgitg / SG_COLS;
    ushort sg_c = sgitg % SG_COLS;

    simdgroup_float8x8 mc[2][2];
    FOR_UNROLL (ushort i = 0; i < 2; i++)
        FOR_UNROLL (ushort j = 0; j < 2; j++)
            mc[i][j] = make_filled_simdgroup_matrix<float, 8>(0.f);

    uint num_blocks_per_row = (K + Q4_GROUP_SIZE - 1) / Q4_GROUP_SIZE;
    uint row_bytes = num_blocks_per_row * Q4_BLOCK_SIZE;

    uint num_k_tiles = (K + TILE_K - 1) / TILE_K;

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        uint k_base = kt * TILE_K;

        // Load A tile
        {
            ushort row = tiitg >> 2;
            ushort col_group = tiitg & 3;
            ushort k_start = col_group << 3;

            uint gm = tile_m_start + row;
            uint gk = k_base + k_start;
            threadgroup half* sa_ptr = sa + row * TILE_K + k_start;

            if (gm < M && gk + 7 < K) {
                device const float4* x_ptr = (device const float4*)(X + gm * K + gk);
                float4 v0 = x_ptr[0];
                float4 v1 = x_ptr[1];
                sa_ptr[0] = (half)v0.x; sa_ptr[1] = (half)v0.y;
                sa_ptr[2] = (half)v0.z; sa_ptr[3] = (half)v0.w;
                sa_ptr[4] = (half)v1.x; sa_ptr[5] = (half)v1.y;
                sa_ptr[6] = (half)v1.z; sa_ptr[7] = (half)v1.w;
            } else {
                FOR_UNROLL (ushort i = 0; i < 8; i++) {
                    uint gk_i = gk + i;
                    sa_ptr[i] = (gm < M && gk_i < K) ? (half)X[gm * K + gk_i] : (half)0.0h;
                }
            }
        }

        // Load B tile: Q4_0 dequant (de-interleaved)
        {
            uint block_idx = k_base / Q4_GROUP_SIZE;
            ushort n_local = tiitg / 4;
            ushort t_in_row = tiitg % 4;
            ushort k_offset = t_in_row * 8;
            uint gn = tile_n_start + n_local;

            threadgroup half* sb_row = sb + n_local * TILE_K + k_offset;

            if (gn < N && k_base + k_offset < K) {
                uint block_offset = gn * row_bytes + block_idx * Q4_BLOCK_SIZE;
                half scale = as_type<half>(*(device const ushort*)(W_q4 + block_offset));
                device const uchar* qdata = W_q4 + block_offset + 2;

                uint k_remaining = K - (k_base + k_offset);
                ushort count = (k_remaining >= 8) ? 8 : (ushort)k_remaining;

                ushort byte_start = (k_offset < 16) ? k_offset : (k_offset - 16);
                bool use_hi = (k_offset >= 16);

                if (count == 8) {
                    if (use_hi) {
                        sb_row[0] = scale * (half)(int((qdata[byte_start + 0] >> 4)) - 8);
                        sb_row[1] = scale * (half)(int((qdata[byte_start + 1] >> 4)) - 8);
                        sb_row[2] = scale * (half)(int((qdata[byte_start + 2] >> 4)) - 8);
                        sb_row[3] = scale * (half)(int((qdata[byte_start + 3] >> 4)) - 8);
                        sb_row[4] = scale * (half)(int((qdata[byte_start + 4] >> 4)) - 8);
                        sb_row[5] = scale * (half)(int((qdata[byte_start + 5] >> 4)) - 8);
                        sb_row[6] = scale * (half)(int((qdata[byte_start + 6] >> 4)) - 8);
                        sb_row[7] = scale * (half)(int((qdata[byte_start + 7] >> 4)) - 8);
                    } else {
                        sb_row[0] = scale * (half)(int((qdata[byte_start + 0] & 0xF)) - 8);
                        sb_row[1] = scale * (half)(int((qdata[byte_start + 1] & 0xF)) - 8);
                        sb_row[2] = scale * (half)(int((qdata[byte_start + 2] & 0xF)) - 8);
                        sb_row[3] = scale * (half)(int((qdata[byte_start + 3] & 0xF)) - 8);
                        sb_row[4] = scale * (half)(int((qdata[byte_start + 4] & 0xF)) - 8);
                        sb_row[5] = scale * (half)(int((qdata[byte_start + 5] & 0xF)) - 8);
                        sb_row[6] = scale * (half)(int((qdata[byte_start + 6] & 0xF)) - 8);
                        sb_row[7] = scale * (half)(int((qdata[byte_start + 7] & 0xF)) - 8);
                    }
                } else {
                    for (ushort i = 0; i < count; i++) {
                        ushort elem_idx = k_offset + i;
                        ushort byte_idx = (elem_idx < 16) ? elem_idx : (elem_idx - 16);
                        uchar raw = qdata[byte_idx];
                        int nibble = (elem_idx >= 16) ? int((raw >> 4) & 0xF) : int(raw & 0xF);
                        sb_row[i] = scale * (half)(nibble - 8);
                    }
                    for (ushort i = count; i < 8; i++) {
                        sb_row[i] = (half)0.0h;
                    }
                }
            } else {
                FOR_UNROLL (ushort i = 0; i < 8; i++) {
                    sb_row[i] = (half)0.0h;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (ushort ks = 0; ks < TILE_K; ks += 8) {
            simdgroup_half8x8 ma[2];
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(ma[0], sa + (sg_r * 16 + 0) * TILE_K + ks, TILE_K);
            simdgroup_load(ma[1], sa + (sg_r * 16 + 8) * TILE_K + ks, TILE_K);

            simdgroup_half8x8 mb[2];
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(mb[0], sb + (sg_c * 16 + 0) * TILE_K + ks, TILE_K, ulong2(0,0), true);
            simdgroup_load(mb[1], sb + (sg_c * 16 + 8) * TILE_K + ks, TILE_K, ulong2(0,0), true);

            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_multiply_accumulate(mc[0][0], ma[0], mb[0], mc[0][0]);
            simdgroup_multiply_accumulate(mc[0][1], ma[0], mb[1], mc[0][1]);
            simdgroup_multiply_accumulate(mc[1][0], ma[1], mb[0], mc[1][0]);
            simdgroup_multiply_accumulate(mc[1][1], ma[1], mb[1], mc[1][1]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store results with residual add
    uint sg_m_base = tile_m_start + sg_r * 16;
    uint sg_n_base = tile_n_start + sg_c * 16;

    {
        threadgroup float* sc = (threadgroup float*)shmem;
        threadgroup float* my_sc = sc + sgitg * 256;
        FOR_UNROLL (ushort i = 0; i < 2; i++)
            FOR_UNROLL (ushort j = 0; j < 2; j++)
                simdgroup_store(mc[i][j], my_sc + (i * 8) * 16 + (j * 8), 16);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ushort lane_id = tiitg % 32;
        for (ushort idx = lane_id; idx < 256; idx += 32) {
            ushort local_m = idx / 16;
            ushort local_n = idx % 16;
            uint gm = sg_m_base + local_m;
            uint gn = sg_n_base + local_n;
            if (gm < M && gn < N) {
                Y[gm * N + gn] = my_sc[local_m * 16 + local_n] + R[gm * N + gn];
            }
        }
    }
}

// ============================================================================
// dequant_tiled_matmul_q4_0_splitk: Split-K variant for Q4_0 GEMM
// ============================================================================

kernel void dequant_tiled_matmul_q4_0_splitk(
    device const uchar* W_q4     [[buffer(0)]],
    device const float* X        [[buffer(1)]],
    device float*       Y_partial [[buffer(2)]],
    constant uint&      M        [[buffer(3)]],
    constant uint&      N        [[buffer(4)]],
    constant uint&      K        [[buffer(5)]],
    constant uint&      k_splits [[buffer(6)]],
    uint3 tg_pos                 [[threadgroup_position_in_grid]],
    ushort tiitg                 [[thread_index_in_threadgroup]],
    ushort sgitg                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup half shmem[2048];
    threadgroup half* sa = shmem;
    threadgroup half* sb = shmem + TILE_M * TILE_K;

    uint tile_n_start = tg_pos.x * TILE_N;
    uint tile_m_start = tg_pos.y * TILE_M;
    uint split_idx    = tg_pos.z;

    uint k_per_split = ((K + k_splits - 1) / k_splits);
    k_per_split = ((k_per_split + TILE_K - 1) / TILE_K) * TILE_K;
    uint k_start_val = split_idx * k_per_split;
    uint k_end_val = min(k_start_val + k_per_split, K);
    if (k_start_val >= K) {
        uint partial_offset = split_idx * M * N;
        for (ushort idx = tiitg; idx < TILE_M * TILE_N; idx += TG_SIZE) {
            uint m_local = idx / TILE_N;
            uint n_local = idx % TILE_N;
            uint gm = tile_m_start + m_local;
            uint gn = tile_n_start + n_local;
            if (gm < M && gn < N)
                Y_partial[partial_offset + gm * N + gn] = 0.0f;
        }
        return;
    }

    ushort sg_r = sgitg / SG_COLS;
    ushort sg_c = sgitg % SG_COLS;

    simdgroup_float8x8 mc[2][2];
    FOR_UNROLL (ushort i = 0; i < 2; i++)
        FOR_UNROLL (ushort j = 0; j < 2; j++)
            mc[i][j] = make_filled_simdgroup_matrix<float, 8>(0.f);

    uint num_blocks_per_row = (K + Q4_GROUP_SIZE - 1) / Q4_GROUP_SIZE;
    uint row_bytes = num_blocks_per_row * Q4_BLOCK_SIZE;

    uint first_kt = k_start_val / TILE_K;
    uint last_kt = (k_end_val + TILE_K - 1) / TILE_K;

    for (uint kt = first_kt; kt < last_kt; kt++) {
        uint k_base = kt * TILE_K;

        // Load A tile
        {
            ushort row = tiitg >> 2;
            ushort col_group = tiitg & 3;
            ushort k_local = col_group << 3;

            uint gm = tile_m_start + row;
            uint gk = k_base + k_local;
            threadgroup half* sa_ptr = sa + row * TILE_K + k_local;

            if (gm < M && gk + 7 < K && gk >= k_start_val && gk + 7 < k_end_val) {
                device const float4* x_ptr = (device const float4*)(X + gm * K + gk);
                float4 v0 = x_ptr[0];
                float4 v1 = x_ptr[1];
                sa_ptr[0] = (half)v0.x; sa_ptr[1] = (half)v0.y;
                sa_ptr[2] = (half)v0.z; sa_ptr[3] = (half)v0.w;
                sa_ptr[4] = (half)v1.x; sa_ptr[5] = (half)v1.y;
                sa_ptr[6] = (half)v1.z; sa_ptr[7] = (half)v1.w;
            } else {
                FOR_UNROLL (ushort i = 0; i < 8; i++) {
                    uint gk_i = gk + i;
                    sa_ptr[i] = (gm < M && gk_i < K && gk_i >= k_start_val && gk_i < k_end_val) ? (half)X[gm * K + gk_i] : (half)0.0h;
                }
            }
        }

        // Load B tile: Q4_0 dequant (de-interleaved)
        {
            uint block_idx = k_base / Q4_GROUP_SIZE;
            ushort n_local = tiitg / 4;
            ushort t_in_row = tiitg % 4;
            ushort k_offset = t_in_row * 8;
            uint gn = tile_n_start + n_local;

            threadgroup half* sb_row = sb + n_local * TILE_K + k_offset;

            if (gn < N && k_base + k_offset < K && k_base + k_offset >= k_start_val && k_base + k_offset < k_end_val) {
                uint block_offset = gn * row_bytes + block_idx * Q4_BLOCK_SIZE;
                half scale = as_type<half>(*(device const ushort*)(W_q4 + block_offset));
                device const uchar* qdata = W_q4 + block_offset + 2;

                uint k_remaining = K - (k_base + k_offset);
                ushort count = (k_remaining >= 8) ? 8 : (ushort)k_remaining;

                ushort byte_start = (k_offset < 16) ? k_offset : (k_offset - 16);
                bool use_hi = (k_offset >= 16);

                if (count == 8) {
                    if (use_hi) {
                        sb_row[0] = scale * (half)(int((qdata[byte_start + 0] >> 4)) - 8);
                        sb_row[1] = scale * (half)(int((qdata[byte_start + 1] >> 4)) - 8);
                        sb_row[2] = scale * (half)(int((qdata[byte_start + 2] >> 4)) - 8);
                        sb_row[3] = scale * (half)(int((qdata[byte_start + 3] >> 4)) - 8);
                        sb_row[4] = scale * (half)(int((qdata[byte_start + 4] >> 4)) - 8);
                        sb_row[5] = scale * (half)(int((qdata[byte_start + 5] >> 4)) - 8);
                        sb_row[6] = scale * (half)(int((qdata[byte_start + 6] >> 4)) - 8);
                        sb_row[7] = scale * (half)(int((qdata[byte_start + 7] >> 4)) - 8);
                    } else {
                        sb_row[0] = scale * (half)(int((qdata[byte_start + 0] & 0xF)) - 8);
                        sb_row[1] = scale * (half)(int((qdata[byte_start + 1] & 0xF)) - 8);
                        sb_row[2] = scale * (half)(int((qdata[byte_start + 2] & 0xF)) - 8);
                        sb_row[3] = scale * (half)(int((qdata[byte_start + 3] & 0xF)) - 8);
                        sb_row[4] = scale * (half)(int((qdata[byte_start + 4] & 0xF)) - 8);
                        sb_row[5] = scale * (half)(int((qdata[byte_start + 5] & 0xF)) - 8);
                        sb_row[6] = scale * (half)(int((qdata[byte_start + 6] & 0xF)) - 8);
                        sb_row[7] = scale * (half)(int((qdata[byte_start + 7] & 0xF)) - 8);
                    }
                } else {
                    for (ushort i = 0; i < count; i++) {
                        ushort elem_idx = k_offset + i;
                        ushort byte_idx = (elem_idx < 16) ? elem_idx : (elem_idx - 16);
                        uchar raw = qdata[byte_idx];
                        int nibble = (elem_idx >= 16) ? int((raw >> 4) & 0xF) : int(raw & 0xF);
                        sb_row[i] = scale * (half)(nibble - 8);
                    }
                    for (ushort i = count; i < 8; i++) {
                        sb_row[i] = (half)0.0h;
                    }
                }
            } else {
                FOR_UNROLL (ushort i = 0; i < 8; i++) {
                    sb_row[i] = (half)0.0h;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (ushort ks = 0; ks < TILE_K; ks += 8) {
            simdgroup_half8x8 ma[2];
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(ma[0], sa + (sg_r * 16 + 0) * TILE_K + ks, TILE_K);
            simdgroup_load(ma[1], sa + (sg_r * 16 + 8) * TILE_K + ks, TILE_K);

            simdgroup_half8x8 mb[2];
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(mb[0], sb + (sg_c * 16 + 0) * TILE_K + ks, TILE_K, ulong2(0,0), true);
            simdgroup_load(mb[1], sb + (sg_c * 16 + 8) * TILE_K + ks, TILE_K, ulong2(0,0), true);

            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_multiply_accumulate(mc[0][0], ma[0], mb[0], mc[0][0]);
            simdgroup_multiply_accumulate(mc[0][1], ma[0], mb[1], mc[0][1]);
            simdgroup_multiply_accumulate(mc[1][0], ma[1], mb[0], mc[1][0]);
            simdgroup_multiply_accumulate(mc[1][1], ma[1], mb[1], mc[1][1]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint partial_offset = split_idx * M * N;
    uint sg_m_base = tile_m_start + sg_r * 16;
    uint sg_n_base = tile_n_start + sg_c * 16;

    if (sg_m_base + 16 <= M && sg_n_base + 16 <= N) {
        FOR_UNROLL (ushort i = 0; i < 2; i++)
            FOR_UNROLL (ushort j = 0; j < 2; j++)
                simdgroup_store(mc[i][j], Y_partial + partial_offset + (sg_m_base + i * 8) * N + (sg_n_base + j * 8), N);
    } else {
        threadgroup float* sc = (threadgroup float*)shmem;
        threadgroup float* my_sc = sc + sgitg * 256;
        FOR_UNROLL (ushort i = 0; i < 2; i++)
            FOR_UNROLL (ushort j = 0; j < 2; j++)
                simdgroup_store(mc[i][j], my_sc + (i * 8) * 16 + (j * 8), 16);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ushort lane_id = tiitg % 32;
        for (ushort idx = lane_id; idx < 256; idx += 32) {
            ushort local_m = idx / 16;
            ushort local_n = idx % 16;
            uint gm = sg_m_base + local_m;
            uint gn = sg_n_base + local_n;
            if (gm < M && gn < N) {
                Y_partial[partial_offset + gm * N + gn] = my_sc[local_m * 16 + local_n];
            }
        }
    }
}

// ============================================================================

// add_write: out = a + b (3-operand residual add, fuses residual + copy)
kernel void add_write(
    device const float* a   [[buffer(0)]],
    device const float* b   [[buffer(1)]],
    device float*       out [[buffer(2)]],
    uint gid                [[thread_position_in_grid]])
{
    out[gid] = a[gid] + b[gid];
}

kernel void copy_buffer(
    device const float* src [[buffer(0)]],
    device float*       dst [[buffer(1)]],
    uint gid                [[thread_position_in_grid]])
{
    dst[gid] = src[gid];
}

// ============================================================================
// ffn_fused_gate_up_swiglu_q8_0: Fused Gate+Up+SwiGLU for decode
//
// Fuses 3 operations into a single dispatch:
//   Phase 1: Gate matmul (Q8_0 dequant, reads normed x from device memory)
//   Phase 2: Up matmul (Q8_0 dequant, reads normed x from device memory)
//   Phase 3: SwiGLU (gate * sigmoid(gate) * up) -> write to out
//
// Eliminates:
//   - 1 encoder barrier between gate/up and swiglu dispatches
//   - Device memory write of up_buf (22KB) -- up goes directly to SwiGLU
//   - Device memory read of gate_buf by SwiGLU (22KB)
//   - Device memory read of up_buf by SwiGLU (22KB)
//   - Gate and up values stay in registers, never touch device memory
//
// Dispatch: ceil(inter_dim/4) threadgroups, 128 threads each (4 simdgroups)
// Each simdgroup handles 1 output row.
// Zero threadgroup memory required.
// ============================================================================

kernel void ffn_fused_gate_up_swiglu_q8_0(
    device const uchar* w_gate_q8   [[buffer(0)]],   // gate weights Q8_0 [inter_dim, hidden_dim]
    device const float* x           [[buffer(1)]],   // normed input [hidden_dim]
    device float*       out         [[buffer(2)]],   // output [inter_dim] (SwiGLU result)
    constant uint&      in_dim      [[buffer(3)]],   // hidden_dim
    constant uint&      out_dim     [[buffer(4)]],   // inter_dim
    device const uchar* w_up_q8     [[buffer(5)]],   // up weights Q8_0 [inter_dim, hidden_dim]
    uint row_group                  [[threadgroup_position_in_grid]],
    uint lane                       [[thread_index_in_simdgroup]],
    uint sg                         [[simdgroup_index_in_threadgroup]])
{
    uint row = row_group * 4 + sg;
    if (row >= out_dim) return;

    const uint Q8_BLOCK_SIZE = 34;
    uint num_blocks = in_dim >> 5;  // in_dim / 32
    uint row_bytes = num_blocks * Q8_BLOCK_SIZE;

    // --- Gate matmul: dot(w_gate[row], x) ---
    device const uchar* gate_row_ptr = w_gate_q8 + row * row_bytes;
    float gate_sum = 0.0f;

    uint b = 0;
    for (; b + 3 < num_blocks; b += 4) {
        device const uchar* bp0 = gate_row_ptr + b * Q8_BLOCK_SIZE;
        device const uchar* bp1 = bp0 + Q8_BLOCK_SIZE;
        device const uchar* bp2 = bp1 + Q8_BLOCK_SIZE;
        device const uchar* bp3 = bp2 + Q8_BLOCK_SIZE;
        half s0 = as_type<half>(*(device const ushort*)bp0);
        half s1 = as_type<half>(*(device const ushort*)bp1);
        half s2 = as_type<half>(*(device const ushort*)bp2);
        half s3 = as_type<half>(*(device const ushort*)bp3);
        float v0 = float(((device const char*)(bp0 + 2))[lane]) * x[(b << 5) + lane];
        float v1 = float(((device const char*)(bp1 + 2))[lane]) * x[((b+1) << 5) + lane];
        float v2 = float(((device const char*)(bp2 + 2))[lane]) * x[((b+2) << 5) + lane];
        float v3 = float(((device const char*)(bp3 + 2))[lane]) * x[((b+3) << 5) + lane];
        gate_sum += float(s0) * simd_sum(v0) + float(s1) * simd_sum(v1)
                  + float(s2) * simd_sum(v2) + float(s3) * simd_sum(v3);
    }
    for (; b < num_blocks; b++) {
        device const uchar* bp = gate_row_ptr + b * Q8_BLOCK_SIZE;
        half sc = as_type<half>(*(device const ushort*)bp);
        float val = float(((device const char*)(bp + 2))[lane]) * x[(b << 5) + lane];
        gate_sum += float(sc) * simd_sum(val);
    }

    // --- Up matmul: dot(w_up[row], x) ---
    device const uchar* up_row_ptr = w_up_q8 + row * row_bytes;
    float up_sum = 0.0f;

    b = 0;
    for (; b + 3 < num_blocks; b += 4) {
        device const uchar* bp0 = up_row_ptr + b * Q8_BLOCK_SIZE;
        device const uchar* bp1 = bp0 + Q8_BLOCK_SIZE;
        device const uchar* bp2 = bp1 + Q8_BLOCK_SIZE;
        device const uchar* bp3 = bp2 + Q8_BLOCK_SIZE;
        half s0 = as_type<half>(*(device const ushort*)bp0);
        half s1 = as_type<half>(*(device const ushort*)bp1);
        half s2 = as_type<half>(*(device const ushort*)bp2);
        half s3 = as_type<half>(*(device const ushort*)bp3);
        float v0 = float(((device const char*)(bp0 + 2))[lane]) * x[(b << 5) + lane];
        float v1 = float(((device const char*)(bp1 + 2))[lane]) * x[((b+1) << 5) + lane];
        float v2 = float(((device const char*)(bp2 + 2))[lane]) * x[((b+2) << 5) + lane];
        float v3 = float(((device const char*)(bp3 + 2))[lane]) * x[((b+3) << 5) + lane];
        up_sum += float(s0) * simd_sum(v0) + float(s1) * simd_sum(v1)
                + float(s2) * simd_sum(v2) + float(s3) * simd_sum(v3);
    }
    for (; b < num_blocks; b++) {
        device const uchar* bp = up_row_ptr + b * Q8_BLOCK_SIZE;
        half sc = as_type<half>(*(device const ushort*)bp);
        float val = float(((device const char*)(bp + 2))[lane]) * x[(b << 5) + lane];
        up_sum += float(sc) * simd_sum(val);
    }

    // --- SwiGLU: gate * sigmoid(gate) * up ---
    if (lane == 0) {
        float g = gate_sum;
        float sigmoid = 1.0f / (1.0f + exp(-g));
        out[row] = g * sigmoid * up_sum;
    }
}

// ============================================================================
// ffn_fused_gate_up_swiglu_q8_0_deferred: Deferred-reduction fused FFN
//
// Same computation as ffn_fused_gate_up_swiglu_q8_0 but uses the llama.cpp
// deferred-reduction pattern to minimize simd_sum() synchronization:
//
// Old kernel: simd_sum() on every Q8_0 block (176 calls for hidden_dim=5632)
// New kernel: each thread accumulates 8 elements locally per block, ONE
//             simd_sum() at the end + cross-SG shared memory reduction.
//
// Thread mapping (NQ=8, NW=32, NSG=4, 128 threads/TG):
//   ix = tiisg / 4  -> 0..7 (selects block within stride of 8)
//   il = tiisg % 4  -> 0..3 (selects which 8 bytes within 32-byte block)
//   Stride = NW/4 = 8 blocks per iteration
//
// Each threadgroup processes 1 output row (out_dim threadgroups).
// All 4 simdgroups cooperate on the SAME row via shared memory reduction.
//
// Dispatch: out_dim threadgroups, 128 threads each (4 simdgroups)
// ============================================================================

kernel void ffn_fused_gate_up_swiglu_q8_0_deferred(
    device const uchar* w_gate_q8   [[buffer(0)]],   // gate weights Q8_0 [inter_dim, hidden_dim]
    device const float* x           [[buffer(1)]],   // normed input [hidden_dim]
    device float*       out         [[buffer(2)]],   // output [inter_dim] (SwiGLU result)
    constant uint&      in_dim      [[buffer(3)]],   // hidden_dim
    constant uint&      out_dim     [[buffer(4)]],   // inter_dim
    device const uchar* w_up_q8     [[buffer(5)]],   // up weights Q8_0 [inter_dim, hidden_dim]
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    uint row = tgpig;
    if (row >= out_dim) return;

    const uint Q8_BLOCK_SIZE = 34;
    uint num_blocks = in_dim >> 5;  // in_dim / 32
    uint row_bytes = num_blocks * Q8_BLOCK_SIZE;

    // Thread mapping: each thread handles 8 contiguous elements within a block
    uint ix = tiisg / 4;   // 0..7: which block in stride of 8
    uint il = tiisg % 4;   // 0..3: which 8-byte quarter within 32-byte data

    device const uchar* gate_row_ptr = w_gate_q8 + row * row_bytes;
    device const uchar* up_row_ptr   = w_up_q8   + row * row_bytes;

    float gate_sum = 0.0f;
    float up_sum   = 0.0f;

    // Each simdgroup starts at a different offset and strides by NSG * 8 blocks
    // NSG=4 simdgroups, each strides 8 blocks -> total stride = 32 blocks/iter
    for (uint ib = sgitg * 8 + ix; ib < num_blocks; ib += 32) {
        // Load 8 x-values for this thread's quarter of the block
        uint x_base = ib * 32 + il * 8;
        float yl[8];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; i++) {
            yl[i] = x[x_base + i];
        }

        // Gate weights: read scale + 8 int8 values
        device const uchar* gate_bp = gate_row_ptr + ib * Q8_BLOCK_SIZE;
        half gate_scale = as_type<half>(*(device const ushort*)gate_bp);
        device const char* gate_qs = (device const char*)(gate_bp + 2) + il * 8;
        float gate_dot = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; i++) {
            gate_dot += float(gate_qs[i]) * yl[i];
        }
        gate_sum += gate_dot * float(gate_scale);

        // Up weights: same x-values, different weight row
        device const uchar* up_bp = up_row_ptr + ib * Q8_BLOCK_SIZE;
        half up_scale = as_type<half>(*(device const ushort*)up_bp);
        device const char* up_qs = (device const char*)(up_bp + 2) + il * 8;
        float up_dot = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; i++) {
            up_dot += float(up_qs[i]) * yl[i];
        }
        up_sum += up_dot * float(up_scale);
    }

    // Final reduction: simd_sum within each simdgroup (just 1 call each!)
    gate_sum = simd_sum(gate_sum);
    up_sum   = simd_sum(up_sum);

    // Cross-simdgroup reduction via shared memory
    threadgroup float shmem[8];  // [0..3] for gate, [4..7] for up

    if (tiisg == 0) {
        shmem[sgitg]     = gate_sum;
        shmem[sgitg + 4] = up_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 of simdgroup 0 does final reduction + SwiGLU
    if (sgitg == 0 && tiisg == 0) {
        float g = shmem[0] + shmem[1] + shmem[2] + shmem[3];
        float u = shmem[4] + shmem[5] + shmem[6] + shmem[7];
        float sigmoid = 1.0f / (1.0f + exp(-g));
        out[row] = g * sigmoid * u;
    }
}

// ============================================================================
// ffn_fused_gate_up_swiglu_q4_0: Fused Gate+Up+SwiGLU for Q4_0 decode
//
// Identical structure to Q8_0 fused kernel but with Q4_0 dequantization.
// Q4_0 block: 18 bytes (2 byte f16 scale + 16 bytes of packed 4-bit values)
// Each byte holds 2 elements: low nibble (even lane), high nibble (odd lane).
// Dequant: (float(nibble) - 8.0) * scale
//
// Dispatch: ceil(inter_dim/4) threadgroups, 128 threads each (4 simdgroups)
// Each simdgroup handles 1 output row.
// Zero threadgroup memory required.
// ============================================================================

kernel void ffn_fused_gate_up_swiglu_q4_0(
    device const uchar* w_gate_q4   [[buffer(0)]],   // gate weights Q4_0 [inter_dim, hidden_dim]
    device const float* x           [[buffer(1)]],   // normed input [hidden_dim]
    device float*       out         [[buffer(2)]],   // output [inter_dim] (SwiGLU result)
    constant uint&      in_dim      [[buffer(3)]],   // hidden_dim
    constant uint&      out_dim     [[buffer(4)]],   // inter_dim
    device const uchar* w_up_q4     [[buffer(5)]],   // up weights Q4_0 [inter_dim, hidden_dim]
    uint row_group                  [[threadgroup_position_in_grid]],
    uint lane                       [[thread_index_in_simdgroup]],
    uint sg                         [[simdgroup_index_in_threadgroup]])
{
    uint row = row_group * 4 + sg;
    if (row >= out_dim) return;

    uint num_blocks = in_dim >> 5;  // in_dim / 32
    uint row_bytes = num_blocks * Q4_BLOCK_SIZE;

    // De-interleaved nibble mapping
    uint q4_byte = (lane < 16) ? lane : (lane - 16);
    bool q4_hi = (lane >= 16);

    // --- Gate matmul: dot(w_gate[row], x) ---
    device const uchar* gate_row_ptr = w_gate_q4 + row * row_bytes;
    float gate_sum = 0.0f;

    uint b = 0;
    for (; b + 3 < num_blocks; b += 4) {
        device const uchar* bp0 = gate_row_ptr + b * Q4_BLOCK_SIZE;
        device const uchar* bp1 = bp0 + Q4_BLOCK_SIZE;
        device const uchar* bp2 = bp1 + Q4_BLOCK_SIZE;
        device const uchar* bp3 = bp2 + Q4_BLOCK_SIZE;
        half s0 = as_type<half>(*(device const ushort*)bp0);
        half s1 = as_type<half>(*(device const ushort*)bp1);
        half s2 = as_type<half>(*(device const ushort*)bp2);
        half s3 = as_type<half>(*(device const ushort*)bp3);
        uchar byte0 = (bp0 + 2)[q4_byte];
        uchar byte1 = (bp1 + 2)[q4_byte];
        uchar byte2 = (bp2 + 2)[q4_byte];
        uchar byte3 = (bp3 + 2)[q4_byte];
        int q0 = (q4_hi ? (byte0 >> 4) : (byte0 & 0xF)) - 8;
        int q1 = (q4_hi ? (byte1 >> 4) : (byte1 & 0xF)) - 8;
        int q2 = (q4_hi ? (byte2 >> 4) : (byte2 & 0xF)) - 8;
        int q3 = (q4_hi ? (byte3 >> 4) : (byte3 & 0xF)) - 8;
        float v0 = float(q0) * x[(b << 5) + lane];
        float v1 = float(q1) * x[((b+1) << 5) + lane];
        float v2 = float(q2) * x[((b+2) << 5) + lane];
        float v3 = float(q3) * x[((b+3) << 5) + lane];
        gate_sum += float(s0) * simd_sum(v0) + float(s1) * simd_sum(v1)
                  + float(s2) * simd_sum(v2) + float(s3) * simd_sum(v3);
    }
    for (; b < num_blocks; b++) {
        device const uchar* bp = gate_row_ptr + b * Q4_BLOCK_SIZE;
        half sc = as_type<half>(*(device const ushort*)bp);
        uchar byte_val = (bp + 2)[q4_byte];
        int q = (q4_hi ? (byte_val >> 4) : (byte_val & 0xF)) - 8;
        float val = float(q) * x[(b << 5) + lane];
        gate_sum += float(sc) * simd_sum(val);
    }

    // --- Up matmul: dot(w_up[row], x) ---
    device const uchar* up_row_ptr = w_up_q4 + row * row_bytes;
    float up_sum = 0.0f;

    b = 0;
    for (; b + 3 < num_blocks; b += 4) {
        device const uchar* bp0 = up_row_ptr + b * Q4_BLOCK_SIZE;
        device const uchar* bp1 = bp0 + Q4_BLOCK_SIZE;
        device const uchar* bp2 = bp1 + Q4_BLOCK_SIZE;
        device const uchar* bp3 = bp2 + Q4_BLOCK_SIZE;
        half s0 = as_type<half>(*(device const ushort*)bp0);
        half s1 = as_type<half>(*(device const ushort*)bp1);
        half s2 = as_type<half>(*(device const ushort*)bp2);
        half s3 = as_type<half>(*(device const ushort*)bp3);
        uchar byte0 = (bp0 + 2)[q4_byte];
        uchar byte1 = (bp1 + 2)[q4_byte];
        uchar byte2 = (bp2 + 2)[q4_byte];
        uchar byte3 = (bp3 + 2)[q4_byte];
        int q0 = (q4_hi ? (byte0 >> 4) : (byte0 & 0xF)) - 8;
        int q1 = (q4_hi ? (byte1 >> 4) : (byte1 & 0xF)) - 8;
        int q2 = (q4_hi ? (byte2 >> 4) : (byte2 & 0xF)) - 8;
        int q3 = (q4_hi ? (byte3 >> 4) : (byte3 & 0xF)) - 8;
        float v0 = float(q0) * x[(b << 5) + lane];
        float v1 = float(q1) * x[((b+1) << 5) + lane];
        float v2 = float(q2) * x[((b+2) << 5) + lane];
        float v3 = float(q3) * x[((b+3) << 5) + lane];
        up_sum += float(s0) * simd_sum(v0) + float(s1) * simd_sum(v1)
                + float(s2) * simd_sum(v2) + float(s3) * simd_sum(v3);
    }
    for (; b < num_blocks; b++) {
        device const uchar* bp = up_row_ptr + b * Q4_BLOCK_SIZE;
        half sc = as_type<half>(*(device const ushort*)bp);
        uchar byte_val = (bp + 2)[q4_byte];
        int q = (q4_hi ? (byte_val >> 4) : (byte_val & 0xF)) - 8;
        float val = float(q) * x[(b << 5) + lane];
        up_sum += float(sc) * simd_sum(val);
    }

    // --- SwiGLU: gate * sigmoid(gate) * up ---
    if (lane == 0) {
        float g = gate_sum;
        float sigmoid = 1.0f / (1.0f + exp(-g));
        out[row] = g * sigmoid * up_sum;
    }
}

// ============================================================================
// ffn_fused_gate_up_swiglu_q4_0_deferred: Fused Gate+Up+SwiGLU for Q4_0 decode
// with deferred reduction (mirrors Q8_0 deferred FFN pattern).
//
// 128 threads (4 simdgroups), 1 output row per threadgroup.
// Deferred accumulation: local sums across all blocks, ONE simd_sum at end.
// Q4_0 nibble dequantization for both gate and up weights.
//
// Dispatch: inter_dim threadgroups, 128 threads each
// ============================================================================

kernel void ffn_fused_gate_up_swiglu_q4_0_deferred(
    device const uchar* w_gate_q4   [[buffer(0)]],   // gate weights Q4_0 [inter_dim, hidden_dim]
    device const float* x           [[buffer(1)]],   // normed input [hidden_dim]
    device float*       out         [[buffer(2)]],   // output [inter_dim] (SwiGLU result)
    constant uint&      in_dim      [[buffer(3)]],   // hidden_dim
    constant uint&      out_dim     [[buffer(4)]],   // inter_dim
    device const uchar* w_up_q4     [[buffer(5)]],   // up weights Q4_0 [inter_dim, hidden_dim]
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    uint row = tgpig;
    if (row >= out_dim) return;

    uint num_blocks = in_dim >> 5;  // in_dim / 32
    uint row_bytes = num_blocks * Q4_BLOCK_SIZE;

    // Thread mapping: each thread handles 8 contiguous elements within a block
    uint ix = tiisg / 4;   // 0..7: which block in stride of 8
    uint il = tiisg % 4;   // 0..3: which 4-byte (8-element) quarter within 16-byte data

    device const uchar* gate_row_ptr = w_gate_q4 + row * row_bytes;
    device const uchar* up_row_ptr   = w_up_q4   + row * row_bytes;

    float gate_sum = 0.0f;
    float up_sum   = 0.0f;

    // Each simdgroup starts at a different offset and strides by NSG * 8 blocks
    // NSG=4 simdgroups, each strides 8 blocks -> total stride = 32 blocks/iter
    for (uint ib = sgitg * 8 + ix; ib < num_blocks; ib += 32) {
        // Load x-values for de-interleaved nibble positions
        uint block_base = ib * 32;
        float yl_lo[4], yl_hi[4];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            yl_lo[i] = x[block_base + il * 4 + i];
            yl_hi[i] = x[block_base + il * 4 + 16 + i];
        }

        // Gate weights: read scale + 4 bytes (8 nibbles)
        device const uchar* gate_bp = gate_row_ptr + ib * Q4_BLOCK_SIZE;
        float gate_scale = float(as_type<half>(*(device const ushort*)gate_bp));
        device const uchar* gate_qdata = (device const uchar*)(gate_bp + 2) + il * 4;

        float gate_dot = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            uchar byte_val = gate_qdata[i];
            float lo = float(byte_val & 0x0F) - 8.0f;
            float hi = float(byte_val >> 4) - 8.0f;
            gate_dot += lo * yl_lo[i] + hi * yl_hi[i];
        }
        gate_sum += gate_dot * gate_scale;

        // Up weights: same x-values, different weight row
        device const uchar* up_bp = up_row_ptr + ib * Q4_BLOCK_SIZE;
        float up_scale = float(as_type<half>(*(device const ushort*)up_bp));
        device const uchar* up_qdata = (device const uchar*)(up_bp + 2) + il * 4;

        float up_dot = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            uchar byte_val = up_qdata[i];
            float lo = float(byte_val & 0x0F) - 8.0f;
            float hi = float(byte_val >> 4) - 8.0f;
            up_dot += lo * yl_lo[i] + hi * yl_hi[i];
        }
        up_sum += up_dot * up_scale;
    }

    // Final reduction: simd_sum within each simdgroup (just 1 call each!)
    gate_sum = simd_sum(gate_sum);
    up_sum   = simd_sum(up_sum);

    // Cross-simdgroup reduction via shared memory
    threadgroup float shmem[8];  // [0..3] for gate, [4..7] for up

    if (tiisg == 0) {
        shmem[sgitg]     = gate_sum;
        shmem[sgitg + 4] = up_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 of simdgroup 0 does final reduction + SwiGLU
    if (sgitg == 0 && tiisg == 0) {
        float g = shmem[0] + shmem[1] + shmem[2] + shmem[3];
        float u = shmem[4] + shmem[5] + shmem[6] + shmem[7];
        float sigmoid = 1.0f / (1.0f + exp(-g));
        out[row] = g * sigmoid * u;
    }
}

// ============================================================================
// ffn_fused_gate_up_swiglu_q4_1_deferred: Fused Gate+Up+SwiGLU for Q4_1 decode
// with deferred reduction (mirrors Q4_0 deferred FFN pattern).
//
// Q4_1 block layout: [f16 scale (2B)] [f16 min (2B)] [16 x uint8 nibbles (16B)] = 20 bytes/block
// dequant: value[i] = scale * nibble(i) + min
//
// 128 threads (4 simdgroups), 1 output row per threadgroup.
// Deferred accumulation: local sums across all blocks, ONE simd_sum at end.
//
// Dispatch: inter_dim threadgroups, 128 threads each
// ============================================================================

kernel void ffn_fused_gate_up_swiglu_q4_1_deferred(
    device const uchar* w_gate_q4   [[buffer(0)]],   // gate weights Q4_1 [inter_dim, hidden_dim]
    device const float* x           [[buffer(1)]],   // normed input [hidden_dim]
    device float*       out         [[buffer(2)]],   // output [inter_dim] (SwiGLU result)
    constant uint&      in_dim      [[buffer(3)]],   // hidden_dim
    constant uint&      out_dim     [[buffer(4)]],   // inter_dim
    device const uchar* w_up_q4     [[buffer(5)]],   // up weights Q4_1 [inter_dim, hidden_dim]
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    uint row = tgpig;
    if (row >= out_dim) return;

    uint num_blocks = in_dim >> 5;  // in_dim / 32
    uint row_bytes = num_blocks * Q4_1_BLOCK_SIZE;

    // Thread mapping: each thread handles 8 contiguous elements within a block
    uint ix = tiisg / 4;   // 0..7: which block in stride of 8
    uint il = tiisg % 4;   // 0..3: which 4-byte (8-element) quarter within 16-byte data

    device const uchar* gate_row_ptr = w_gate_q4 + row * row_bytes;
    device const uchar* up_row_ptr   = w_up_q4   + row * row_bytes;

    float gate_sum = 0.0f;
    float up_sum   = 0.0f;

    // Each simdgroup starts at a different offset and strides by NSG * 8 blocks
    // NSG=4 simdgroups, each strides 8 blocks -> total stride = 32 blocks/iter
    for (uint ib = sgitg * 8 + ix; ib < num_blocks; ib += 32) {
        // Load x-values for de-interleaved nibble positions
        uint block_base = ib * 32;
        float yl_lo[4], yl_hi[4];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            yl_lo[i] = x[block_base + il * 4 + i];
            yl_hi[i] = x[block_base + il * 4 + 16 + i];
        }

        // Gate weights: Q4_1 block = [f16 scale, f16 min, 16B nibbles]
        device const uchar* gate_bp = gate_row_ptr + ib * Q4_1_BLOCK_SIZE;
        float gate_scale = float(as_type<half>(*(device const ushort*)gate_bp));
        float gate_min   = float(as_type<half>(*(device const ushort*)(gate_bp + 2)));
        device const uchar* gate_qdata = (device const uchar*)(gate_bp + 4) + il * 4;

        float gate_dot = 0.0f;
        float gate_sumy = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            uchar byte_val = gate_qdata[i];
            float lo = float(byte_val & 0x0F);
            float hi = float(byte_val >> 4);
            gate_dot  += lo * yl_lo[i] + hi * yl_hi[i];
            gate_sumy += yl_lo[i] + yl_hi[i];
        }
        gate_sum += gate_dot * gate_scale + gate_sumy * gate_min;

        // Up weights: same x-values, different weight row
        device const uchar* up_bp = up_row_ptr + ib * Q4_1_BLOCK_SIZE;
        float up_scale = float(as_type<half>(*(device const ushort*)up_bp));
        float up_min   = float(as_type<half>(*(device const ushort*)(up_bp + 2)));
        device const uchar* up_qdata = (device const uchar*)(up_bp + 4) + il * 4;

        float up_dot = 0.0f;
        float up_sumy = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            uchar byte_val = up_qdata[i];
            float lo = float(byte_val & 0x0F);
            float hi = float(byte_val >> 4);
            up_dot  += lo * yl_lo[i] + hi * yl_hi[i];
            up_sumy += yl_lo[i] + yl_hi[i];
        }
        up_sum += up_dot * up_scale + up_sumy * up_min;
    }

    // Final reduction: simd_sum within each simdgroup (just 1 call each!)
    gate_sum = simd_sum(gate_sum);
    up_sum   = simd_sum(up_sum);

    // Cross-simdgroup reduction via shared memory
    threadgroup float shmem[8];  // [0..3] for gate, [4..7] for up

    if (tiisg == 0) {
        shmem[sgitg]     = gate_sum;
        shmem[sgitg + 4] = up_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 of simdgroup 0 does final reduction + SwiGLU
    if (sgitg == 0 && tiisg == 0) {
        float g = shmem[0] + shmem[1] + shmem[2] + shmem[3];
        float u = shmem[4] + shmem[5] + shmem[6] + shmem[7];
        float sigmoid = 1.0f / (1.0f + exp(-g));
        out[row] = g * sigmoid * u;
    }
}

// ============================================================================
// ffn_fused_gate_up_swiglu_f16_deferred: Fused Gate+Up+SwiGLU for F16 decode
// with deferred reduction (mirrors Q8_0 deferred FFN pattern).
//
// F16 weights: dense half-precision, no block structure.
// Each row of gate/up weights is in_dim contiguous half values.
//
// 128 threads (4 simdgroups), 1 output row per threadgroup.
// Deferred accumulation: local sums across entire row, ONE simd_sum at end.
//
// Dispatch: inter_dim threadgroups, 128 threads each
// ============================================================================

kernel void ffn_fused_gate_up_swiglu_f16_deferred(
    device const half*  w_gate_f16  [[buffer(0)]],   // gate weights F16 [inter_dim, hidden_dim]
    device const float* x           [[buffer(1)]],   // normed input [hidden_dim]
    device float*       out         [[buffer(2)]],   // output [inter_dim] (SwiGLU result)
    constant uint&      in_dim      [[buffer(3)]],   // hidden_dim
    constant uint&      out_dim     [[buffer(4)]],   // inter_dim
    device const half*  w_up_f16    [[buffer(5)]],   // up weights F16 [inter_dim, hidden_dim]
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    uint row = tgpig;
    if (row >= out_dim) return;

    const uint NSG = 4;
    const uint NW = 32;
    const uint tid = sgitg * NW + tiisg;
    const uint total_threads = NSG * NW;  // 128

    device const half* gate_row = w_gate_f16 + row * in_dim;
    device const half* up_row   = w_up_f16   + row * in_dim;

    float gate_sum = 0.0f;
    float up_sum   = 0.0f;

    for (uint j = tid; j < in_dim; j += total_threads) {
        float xv = x[j];
        gate_sum += float(gate_row[j]) * xv;
        up_sum   += float(up_row[j])   * xv;
    }

    // Final reduction: simd_sum within each simdgroup
    gate_sum = simd_sum(gate_sum);
    up_sum   = simd_sum(up_sum);

    // Cross-simdgroup reduction via shared memory
    threadgroup float shmem[8];  // [0..3] for gate, [4..7] for up

    if (tiisg == 0) {
        shmem[sgitg]     = gate_sum;
        shmem[sgitg + 4] = up_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 of simdgroup 0 does final reduction + SwiGLU
    if (sgitg == 0 && tiisg == 0) {
        float g = shmem[0] + shmem[1] + shmem[2] + shmem[3];
        float u = shmem[4] + shmem[5] + shmem[6] + shmem[7];
        float sigmoid = 1.0f / (1.0f + exp(-g));
        out[row] = g * sigmoid * u;
    }
}

// ============================================================================
// argmax: GPU-side greedy sampling (single threadgroup, 256 threads)
//
// Finds the index of the maximum element in logits[0..n).
// Eliminates 128 KB logits readback -- only 4 bytes (u32 token ID) returned.
//
// Launch: 1 threadgroup, 256 threads.
// ============================================================================

kernel void argmax(
    device const float* logits [[buffer(0)]],
    device uint*        result [[buffer(1)]],
    constant uint&      n      [[buffer(2)]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    // Each thread finds local max across strided elements
    float max_val = -INFINITY;
    uint max_idx = 0;
    uint tg_size = 256;
    for (uint i = tid; i < n; i += tg_size) {
        float v = logits[i];
        if (v > max_val) { max_val = v; max_idx = i; }
    }

    // SIMD reduction (32-wide on Apple Silicon)
    for (uint offset = 16; offset > 0; offset >>= 1) {
        float other_val = simd_shuffle_down(max_val, offset);
        uint other_idx = simd_shuffle_down(max_idx, offset);
        if (other_val > max_val) { max_val = other_val; max_idx = other_idx; }
    }

    // Cross-simdgroup reduction via threadgroup memory
    threadgroup float tg_vals[8];
    threadgroup uint tg_idxs[8];
    if (simd_lane == 0) {
        tg_vals[simd_group] = max_val;
        tg_idxs[simd_group] = max_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 does final serial reduction across 8 simdgroups
    if (tid == 0) {
        float best = tg_vals[0];
        uint best_idx = tg_idxs[0];
        for (uint i = 1; i < 8; i++) {
            if (tg_vals[i] > best) { best = tg_vals[i]; best_idx = tg_idxs[i]; }
        }
        result[0] = best_idx;
    }
}

// ============================================================================
// bias_add: Element-wise bias addition for single-token decode
//
// Adds a bias vector to the output of a matmul projection:
//   data[i] += bias[i]  for i in 0..n
//
// Used for QKV bias in Qwen2-family models. Zero-cost for models without bias
// (dispatch is conditional on the presence of bias tensors).
// Dispatch: ceil(n / 256) threadgroups, 256 threads each.
// ============================================================================

kernel void bias_add(
    device float*       data [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    constant uint&      n    [[buffer(2)]],
    uint tid                 [[thread_position_in_grid]])
{
    if (tid < n) {
        data[tid] += bias[tid];
    }
}

// ============================================================================
// bias_add_batched: Broadcast bias addition for batched prefill
//
// For a [rows, cols] activation matrix, adds bias[j] to every row:
//   data[i * cols + j] += bias[j]  for all (i, j)
//
// total = rows * cols. The bias vector (length = cols) is broadcast across
// all rows via modular indexing.
// Dispatch: ceil(total / 256) threadgroups, 256 threads each.
// ============================================================================

kernel void bias_add_batched(
    device float*       data  [[buffer(0)]],
    device const float* bias  [[buffer(1)]],
    constant uint&      cols  [[buffer(2)]],
    constant uint&      total [[buffer(3)]],
    uint tid                  [[thread_position_in_grid]])
{
    if (tid < total) {
        data[tid] += bias[tid % cols];
    }
}

// ============================================================================
// deinterleave_qkv: Split fused [M][qkv_dim] output into separate Q, K, V buffers
//
// Fused QKV GEMM produces [M][qkv_dim] where each row is [Q|K|V] concatenated.
// Downstream kernels expect separate [M][q_dim], [M][kv_dim], [M][kv_dim] buffers.
// This kernel copies each component to its target buffer with correct stride.
//
// Thread mapping: 1D grid, total_elements = M * qkv_dim
// Each thread copies one float from the fused buffer to the correct output buffer.
// ============================================================================

kernel void deinterleave_qkv(
    device const float* qkv_fused  [[buffer(0)]],  // [M, qkv_dim] fused output
    device float*       Q          [[buffer(1)]],   // [M, q_dim] output
    device float*       K          [[buffer(2)]],   // [M, kv_dim] output
    device float*       V          [[buffer(3)]],   // [M, kv_dim] output
    constant uint&      M          [[buffer(4)]],   // batch size
    constant uint&      q_dim      [[buffer(5)]],   // Q output dim
    constant uint&      kv_dim     [[buffer(6)]],   // K/V output dim
    constant uint&      qkv_dim    [[buffer(7)]],   // q_dim + 2*kv_dim
    uint tid                       [[thread_position_in_grid]])
{
    uint total = M * qkv_dim;
    if (tid >= total) return;

    uint row = tid / qkv_dim;
    uint col = tid % qkv_dim;
    float val = qkv_fused[tid];

    if (col < q_dim) {
        Q[row * q_dim + col] = val;
    } else if (col < q_dim + kv_dim) {
        K[row * kv_dim + (col - q_dim)] = val;
    } else {
        V[row * kv_dim + (col - q_dim - kv_dim)] = val;
    }
}

// ============================================================================
// MoE (Mixture of Experts) Kernels
// ============================================================================

// ============================================================================
// moe_router_softmax: Compute router logits, apply softmax, select top-K experts.
//
// For a single token's hidden state, computes the dot product against each
// expert's gating vector (one row of gate_weight), applies softmax over all
// experts, selects the top-K experts by weight, and renormalizes the selected
// weights to sum to 1.0.
//
// Optimized for small num_experts (e.g. 8 for Mixtral) with large hidden_dim
// (e.g. 4096). The threadgroup cooperatively reduces the dot products.
//
// buffer(0): hidden_state  [hidden_dim] float            — normalized input
// buffer(1): gate_weight   [num_experts * hidden_dim] float — router weights (row-major)
// buffer(2): expert_ids    [top_k] uint32                — OUTPUT: selected expert indices
// buffer(3): expert_weights [top_k] float                — OUTPUT: normalized routing weights
// buffer(4): hidden_dim (uint)
// buffer(5): num_experts (uint)
// buffer(6): top_k (uint)
//
// grid: (1, 1, 1), threadgroup: (min(256, hidden_dim), 1, 1)
// ============================================================================

kernel void moe_router_softmax(
    device const float* hidden_state   [[buffer(0)]],
    device const float* gate_weight    [[buffer(1)]],
    device uint*        expert_ids     [[buffer(2)]],
    device float*       expert_weights [[buffer(3)]],
    constant uint&      hidden_dim     [[buffer(4)]],
    constant uint&      num_experts    [[buffer(5)]],
    constant uint&      top_k          [[buffer(6)]],
    uint tid                           [[thread_index_in_threadgroup]],
    uint tg_size                       [[threads_per_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_group                    [[simdgroup_index_in_threadgroup]])
{
    // --- Phase 1: Compute logits[e] = dot(gate_weight[e], hidden_state) ---
    // Each expert's logit is a full dot product over hidden_dim elements.
    // We compute all experts serially but parallelize the dot product across
    // the threadgroup (256 threads cooperating on the reduction).

    // Shared memory for intermediate logits (max 64 experts should cover all models)
    threadgroup float logits[256]; // supports up to 256 experts (e.g. Qwen3.5-35B-A3B)
    threadgroup float partial_sums[32]; // for simd cross-group reduction

    uint num_simd_groups = (tg_size + 31) / 32;

    for (uint e = 0; e < num_experts; e++) {
        device const float* w_row = gate_weight + e * hidden_dim;

        // Each thread accumulates partial dot product for this expert
        float sum = 0.0f;
        for (uint j = tid; j < hidden_dim; j += tg_size) {
            sum += w_row[j] * hidden_state[j];
        }

        // SIMD group reduction
        sum = simd_sum(sum);

        if (simd_lane == 0) {
            partial_sums[simd_group] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Final reduction by first SIMD group
        if (simd_group == 0) {
            float val = (simd_lane < num_simd_groups) ? partial_sums[simd_lane] : 0.0f;
            val = simd_sum(val);
            if (simd_lane == 0) {
                logits[e] = val;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // --- Phase 2: Softmax over logits (numerically stable, max-subtraction) ---
    // Only thread 0 performs softmax and top-K since num_experts is small (e.g. 8).
    // Parallelizing over 8 elements would waste threads.
    if (tid == 0) {
        // Find max logit for numerical stability
        float max_logit = logits[0];
        for (uint e = 1; e < num_experts; e++) {
            max_logit = max(max_logit, logits[e]);
        }

        // Compute exp(logit - max) and sum
        float sum_exp = 0.0f;
        for (uint e = 0; e < num_experts; e++) {
            float val = exp(logits[e] - max_logit);
            logits[e] = val;
            sum_exp += val;
        }

        // Normalize to get softmax probabilities
        float inv_sum = 1.0f / sum_exp;
        for (uint e = 0; e < num_experts; e++) {
            logits[e] *= inv_sum;
        }

        // --- Phase 3: Top-K selection via repeated argmax ---
        // For each of the K selections, find the expert with highest softmax
        // weight, record it, then mask it out with -1 so it is not re-selected.
        float renorm_sum = 0.0f;
        for (uint k = 0; k < top_k; k++) {
            float best_val = -1.0f;
            uint best_idx = 0;
            for (uint e = 0; e < num_experts; e++) {
                if (logits[e] > best_val) {
                    best_val = logits[e];
                    best_idx = e;
                }
            }
            expert_ids[k] = best_idx;
            expert_weights[k] = best_val;
            renorm_sum += best_val;
            logits[best_idx] = -1.0f; // mask out selected expert
        }

        // --- Phase 4: Renormalize selected weights to sum to 1.0 ---
        if (renorm_sum > 0.0f) {
            float inv_renorm = 1.0f / renorm_sum;
            for (uint k = 0; k < top_k; k++) {
                expert_weights[k] *= inv_renorm;
            }
        }
    }
}

// ============================================================================
// moe_router_softmax_batched: Batched version for prefill (multiple tokens).
//
// Each threadgroup handles one token (one batch item). The algorithm per token
// is identical to moe_router_softmax above.
//
// buffer(0): hidden_states  [batch_size * hidden_dim] float
// buffer(1): gate_weight    [num_experts * hidden_dim] float
// buffer(2): expert_ids     [batch_size * top_k] uint32    — OUTPUT
// buffer(3): expert_weights [batch_size * top_k] float     — OUTPUT
// buffer(4): hidden_dim (uint)
// buffer(5): num_experts (uint)
// buffer(6): top_k (uint)
// buffer(7): batch_size (uint)
//
// grid: (batch_size, 1, 1), threadgroup: (min(256, hidden_dim), 1, 1)
// ============================================================================

kernel void moe_router_softmax_batched(
    device const float* hidden_states  [[buffer(0)]],
    device const float* gate_weight    [[buffer(1)]],
    device uint*        expert_ids     [[buffer(2)]],
    device float*       expert_weights [[buffer(3)]],
    constant uint&      hidden_dim     [[buffer(4)]],
    constant uint&      num_experts    [[buffer(5)]],
    constant uint&      top_k          [[buffer(6)]],
    constant uint&      batch_size     [[buffer(7)]],
    uint batch_idx                     [[threadgroup_position_in_grid]],
    uint tid                           [[thread_index_in_threadgroup]],
    uint tg_size                       [[threads_per_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_group                    [[simdgroup_index_in_threadgroup]])
{
    if (batch_idx >= batch_size) return;

    // Pointer to this token's hidden state
    device const float* hidden_state = hidden_states + batch_idx * hidden_dim;

    // Output pointers for this token
    device uint*  out_ids     = expert_ids     + batch_idx * top_k;
    device float* out_weights = expert_weights + batch_idx * top_k;

    // --- Phase 1: Compute logits[e] = dot(gate_weight[e], hidden_state) ---
    threadgroup float logits[256]; // supports up to 256 experts (e.g. Qwen3.5-35B-A3B)
    threadgroup float partial_sums[32];

    uint num_simd_groups = (tg_size + 31) / 32;

    for (uint e = 0; e < num_experts; e++) {
        device const float* w_row = gate_weight + e * hidden_dim;

        float sum = 0.0f;
        for (uint j = tid; j < hidden_dim; j += tg_size) {
            sum += w_row[j] * hidden_state[j];
        }

        sum = simd_sum(sum);

        if (simd_lane == 0) {
            partial_sums[simd_group] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_group == 0) {
            float val = (simd_lane < num_simd_groups) ? partial_sums[simd_lane] : 0.0f;
            val = simd_sum(val);
            if (simd_lane == 0) {
                logits[e] = val;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // --- Phase 2: Softmax + Top-K (single thread, num_experts is small) ---
    if (tid == 0) {
        float max_logit = logits[0];
        for (uint e = 1; e < num_experts; e++) {
            max_logit = max(max_logit, logits[e]);
        }

        float sum_exp = 0.0f;
        for (uint e = 0; e < num_experts; e++) {
            float val = exp(logits[e] - max_logit);
            logits[e] = val;
            sum_exp += val;
        }

        float inv_sum = 1.0f / sum_exp;
        for (uint e = 0; e < num_experts; e++) {
            logits[e] *= inv_sum;
        }

        // --- Phase 3: Top-K selection ---
        float renorm_sum = 0.0f;
        for (uint k = 0; k < top_k; k++) {
            float best_val = -1.0f;
            uint best_idx = 0;
            for (uint e = 0; e < num_experts; e++) {
                if (logits[e] > best_val) {
                    best_val = logits[e];
                    best_idx = e;
                }
            }
            out_ids[k] = best_idx;
            out_weights[k] = best_val;
            renorm_sum += best_val;
            logits[best_idx] = -1.0f;
        }

        // --- Phase 4: Renormalize ---
        if (renorm_sum > 0.0f) {
            float inv_renorm = 1.0f / renorm_sum;
            for (uint k = 0; k < top_k; k++) {
                out_weights[k] *= inv_renorm;
            }
        }
    }
}

// ============================================================================
// moe_expert_accum: Weighted accumulation of top-K expert outputs + residual.
//
// After running each selected expert's FFN, this kernel combines the outputs
// using the routing weights and adds the residual connection:
//   output[t] = residual[t] + sum_k(expert_weights[k] * expert_outputs[expert_ids[k] * hidden_dim + t])
//
// The expert_outputs buffer contains ALL num_experts expert outputs at slots
// 0..num_experts (layout: [num_experts, hidden_dim]). The router selects
// top_k of them; expert_ids[k] gives the slot index for selection k.
//
// Purely elementwise -- no threadgroup memory required.
//
// buffer(0): expert_outputs  [num_experts * hidden_dim] float — all expert outputs
// buffer(1): expert_weights  [top_k] float                   — routing weights (sum to 1)
// buffer(2): expert_ids      [top_k] uint32                  — selected expert slot indices
// buffer(3): output          [hidden_dim] float               — OUTPUT: accumulated result
// buffer(4): residual        [hidden_dim] float               — residual to add
// buffer(5): hidden_dim (uint)
// buffer(6): top_k (uint)
//
// grid: (ceil(hidden_dim / 256), 1, 1), threadgroup: (256, 1, 1)
// ============================================================================

kernel void moe_expert_accum(
    device const float* expert_outputs  [[buffer(0)]],
    device const float* expert_weights  [[buffer(1)]],
    device const uint*  expert_ids      [[buffer(2)]],
    device float*       output          [[buffer(3)]],
    device const float* residual        [[buffer(4)]],
    constant uint&      hidden_dim      [[buffer(5)]],
    constant uint&      top_k           [[buffer(6)]],
    uint gid                            [[thread_position_in_grid]])
{
    if (gid >= hidden_dim) return;

    float sum = residual[gid];
    for (uint k = 0; k < top_k; k++) {
        uint e = expert_ids[k];
        sum += expert_weights[k] * expert_outputs[e * hidden_dim + gid];
    }
    output[gid] = sum;
}

// ============================================================================
// moe_expert_accum_batched: Batched version for prefill (multiple tokens).
//
// For each token in the batch, combines top-K expert outputs with routing
// weights and adds the residual. Uses expert_ids to index into the sparse
// expert_outputs buffer.
//
// expert_outputs is written by the dispatch loop with layout:
//   [num_experts, batch_size, hidden_dim]
// where expert e's output for batch item b is at:
//   expert_outputs[e * batch_size * hidden_dim + b * hidden_dim + t]
//
// buffer(0): expert_outputs  [num_experts * batch_size * hidden_dim] float
// buffer(1): expert_weights  [batch_size * top_k] float
// buffer(2): expert_ids      [batch_size * top_k] uint32  — selected expert slot indices
// buffer(3): output          [batch_size * hidden_dim] float  — OUTPUT
// buffer(4): residual        [batch_size * hidden_dim] float
// buffer(5): hidden_dim (uint)
// buffer(6): top_k (uint)
// buffer(7): batch_size (uint)
//
// grid: (ceil(batch_size * hidden_dim / 256), 1, 1), threadgroup: (256, 1, 1)
// ============================================================================

kernel void moe_expert_accum_batched(
    device const float* expert_outputs  [[buffer(0)]],
    device const float* expert_weights  [[buffer(1)]],
    device const uint*  expert_ids      [[buffer(2)]],
    device float*       output          [[buffer(3)]],
    device const float* residual        [[buffer(4)]],
    constant uint&      hidden_dim      [[buffer(5)]],
    constant uint&      top_k           [[buffer(6)]],
    constant uint&      batch_size      [[buffer(7)]],
    uint gid                            [[thread_position_in_grid]])
{
    uint total = batch_size * hidden_dim;
    if (gid >= total) return;

    // Determine which batch item and which element within hidden_dim
    uint b = gid / hidden_dim;
    uint t = gid % hidden_dim;

    float sum = residual[gid];
    for (uint k = 0; k < top_k; k++) {
        uint e = expert_ids[b * top_k + k];
        float w = expert_weights[b * top_k + k];
        // expert_outputs layout: [num_experts, batch_size, hidden_dim]
        sum += w * expert_outputs[e * batch_size * hidden_dim + b * hidden_dim + t];
    }
    output[gid] = sum;
}

// ============================================================================
// moe_router_softmax_biased: Cache-conditional routing bias.
//
// Identical to moe_router_softmax, except after computing each expert's dot
// product logit, a small bias is added for cached experts:
//   logit[e] = dot(hidden_state, gate_weight[e]) + cache_bias_lambda * is_cached[e]
//
// When cache_bias_lambda == 0.0 and all is_cached == 0, output is identical
// to moe_router_softmax.
//
// buffer(0): hidden_state      [hidden_dim] float          — normalized input
// buffer(1): gate_weight       [num_experts * hidden_dim] float — router weights
// buffer(2): expert_ids        [top_k] uint32              — OUTPUT
// buffer(3): expert_weights    [top_k] float               — OUTPUT
// buffer(4): hidden_dim (uint)
// buffer(5): num_experts (uint)
// buffer(6): top_k (uint)
// buffer(7): is_cached         [num_experts] uint8_t       — 1 if cached, 0 if not
// buffer(8): cache_bias_lambda (float)                     — bias magnitude
//
// grid: (1, 1, 1), threadgroup: (min(256, hidden_dim), 1, 1)
// ============================================================================

kernel void moe_router_softmax_biased(
    device const float*   hidden_state      [[buffer(0)]],
    device const float*   gate_weight       [[buffer(1)]],
    device uint*          expert_ids        [[buffer(2)]],
    device float*         expert_weights    [[buffer(3)]],
    constant uint&        hidden_dim        [[buffer(4)]],
    constant uint&        num_experts       [[buffer(5)]],
    constant uint&        top_k             [[buffer(6)]],
    device const uint8_t* is_cached         [[buffer(7)]],
    constant float&       cache_bias_lambda [[buffer(8)]],
    uint tid                               [[thread_index_in_threadgroup]],
    uint tg_size                           [[threads_per_threadgroup]],
    uint simd_lane                         [[thread_index_in_simdgroup]],
    uint simd_group                        [[simdgroup_index_in_threadgroup]])
{
    // --- Phase 1: Compute logits[e] = dot(gate_weight[e], hidden_state) + bias ---
    threadgroup float logits[256]; // supports up to 256 experts (e.g. Qwen3.5-35B-A3B)
    threadgroup float partial_sums[32];

    uint num_simd_groups = (tg_size + 31) / 32;

    for (uint e = 0; e < num_experts; e++) {
        device const float* w_row = gate_weight + e * hidden_dim;

        float sum = 0.0f;
        for (uint j = tid; j < hidden_dim; j += tg_size) {
            sum += w_row[j] * hidden_state[j];
        }

        sum = simd_sum(sum);

        if (simd_lane == 0) {
            partial_sums[simd_group] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_group == 0) {
            float val = (simd_lane < num_simd_groups) ? partial_sums[simd_lane] : 0.0f;
            val = simd_sum(val);
            if (simd_lane == 0) {
                // Add cache bias: cached experts get a small logit boost
                logits[e] = val + cache_bias_lambda * float(is_cached[e]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // --- Phase 2: Softmax (numerically stable, max-subtraction) ---
    if (tid == 0) {
        float max_logit = logits[0];
        for (uint e = 1; e < num_experts; e++) {
            max_logit = max(max_logit, logits[e]);
        }

        float sum_exp = 0.0f;
        for (uint e = 0; e < num_experts; e++) {
            float val = exp(logits[e] - max_logit);
            logits[e] = val;
            sum_exp += val;
        }

        float inv_sum = 1.0f / sum_exp;
        for (uint e = 0; e < num_experts; e++) {
            logits[e] *= inv_sum;
        }

        // --- Phase 3: Top-K selection via repeated argmax ---
        float renorm_sum = 0.0f;
        for (uint k = 0; k < top_k; k++) {
            float best_val = -1.0f;
            uint best_idx = 0;
            for (uint e = 0; e < num_experts; e++) {
                if (logits[e] > best_val) {
                    best_val = logits[e];
                    best_idx = e;
                }
            }
            expert_ids[k] = best_idx;
            expert_weights[k] = best_val;
            renorm_sum += best_val;
            logits[best_idx] = -1.0f;
        }

        // --- Phase 4: Renormalize selected weights to sum to 1.0 ---
        if (renorm_sum > 0.0f) {
            float inv_renorm = 1.0f / renorm_sum;
            for (uint k = 0; k < top_k; k++) {
                expert_weights[k] *= inv_renorm;
            }
        }
    }
}

// ============================================================================
// moe_expert_accum_option_a: Dense accumulation for Option A dispatch.
//
// With Option A, only the top-K selected experts are dispatched. The expert_outputs
// buffer has dense layout: [top_k, hidden_dim] where slot k contains the output
// of the k-th selected expert. No expert_ids indexing is needed for the output
// buffer -- we just iterate 0..top_k directly.
//
// buffer(0): expert_outputs  [top_k * hidden_dim] float — dense top-K outputs
// buffer(1): expert_weights  [top_k] float              — routing weights (sum to 1)
// buffer(2): output          [hidden_dim] float          — OUTPUT: accumulated result
// buffer(3): residual        [hidden_dim] float          — residual to add
// buffer(4): hidden_dim (uint)
// buffer(5): top_k (uint)
//
// grid: (ceil(hidden_dim / 256), 1, 1), threadgroup: (256, 1, 1)
// ============================================================================

kernel void moe_expert_accum_option_a(
    device const float* expert_outputs  [[buffer(0)]],
    device const float* expert_weights  [[buffer(1)]],
    device float*       output          [[buffer(2)]],
    device const float* residual        [[buffer(3)]],
    constant uint&      hidden_dim      [[buffer(4)]],
    constant uint&      top_k           [[buffer(5)]],
    uint gid                            [[thread_position_in_grid]])
{
    if (gid >= hidden_dim) return;

    float sum = residual[gid];
    for (uint k = 0; k < top_k; k++) {
        sum += expert_weights[k] * expert_outputs[k * hidden_dim + gid];
    }
    output[gid] = sum;
}

// ============================================================================
// Batched MoE expert FFN kernels — GPU-side routing, no CPU readback.
//
// These kernels take expert_ids[top_k] and expert_weights[top_k] directly from
// GPU buffers (written by the router kernel in the same command buffer).
// This eliminates the per-layer commit_and_wait + CPU readback that Option A
// previously required.
//
// Two-phase approach per MoE layer:
//   Phase 1: moe_batched_gate_up_swiglu_q4_0 — compute gate+up+swiglu for all top_k experts
//   Phase 2: moe_batched_down_accum_q4_0 — compute down projection + weighted accumulation
// ============================================================================

// Phase 1: Batched gate+up+SwiGLU for top_k experts (Q4_0)
//
// Layout: All expert gate weights are contiguous in the layer buffer at known
// byte offsets. An offset table (expert_offsets) provides the absolute byte offset
// for each of the num_experts gate/up weight pairs:
//   expert_offsets[e * 2 + 0] = gate_off for expert e (byte offset from layer_buf start)
//   expert_offsets[e * 2 + 1] = up_off for expert e
//
// The kernel reads expert_ids[k] (0..num_experts-1) to find which expert to process,
// then uses expert_offsets to locate the weight data.
//
// buffer(0): layer_buf        — full layer weight buffer
// buffer(1): x                — normed input [hidden_dim] float
// buffer(2): swiglu_out       — output [top_k * inter_dim] float (dense packed)
// buffer(3): expert_ids       — [top_k] uint (from router, on GPU)
// buffer(4): expert_offsets   — [num_experts * 2] ulong (gate_off, up_off per expert)
// buffer(5): hidden_dim (uint)
// buffer(6): inter_dim (uint)
// buffer(7): top_k (uint)
//
// Dispatch: (top_k * inter_dim) threadgroups, 128 threads each.
// threadgroup_position_in_grid encodes (expert_slot * inter_dim + row).
// ============================================================================

kernel void moe_batched_gate_up_swiglu_q4_0(
    device const uchar*  layer_buf      [[buffer(0)]],
    device const float*  x              [[buffer(1)]],
    device float*        swiglu_out     [[buffer(2)]],
    device const uint*   expert_ids     [[buffer(3)]],
    device const ulong*  expert_offsets [[buffer(4)]],
    constant uint&       hidden_dim     [[buffer(5)]],
    constant uint&       inter_dim      [[buffer(6)]],
    constant uint&       top_k          [[buffer(7)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    // Decode threadgroup index into (expert_slot, row)
    uint row = tgpig % inter_dim;
    uint expert_slot = tgpig / inter_dim;
    if (expert_slot >= top_k) return;

    // Look up expert ID and weight offsets
    uint expert_id = expert_ids[expert_slot];
    ulong gate_off = expert_offsets[expert_id * 2 + 0];
    ulong up_off   = expert_offsets[expert_id * 2 + 1];

    // Q4_0 row layout
    uint num_blocks = hidden_dim >> 5;  // hidden_dim / 32
    uint row_bytes = num_blocks * 18u;  // Q4_BLOCK_SIZE = 18

    device const uchar* gate_row_ptr = layer_buf + gate_off + row * row_bytes;
    device const uchar* up_row_ptr   = layer_buf + up_off   + row * row_bytes;

    // Thread mapping: each thread handles 8 contiguous elements within a block
    uint ix = tiisg / 4;   // 0..7: which block in stride of 8
    uint il = tiisg % 4;   // 0..3: which 4-byte (8-element) quarter

    float gate_sum = 0.0f;
    float up_sum   = 0.0f;

    for (uint ib = sgitg * 8 + ix; ib < num_blocks; ib += 32) {
        uint block_base = ib * 32;
        float yl_lo[4], yl_hi[4];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            yl_lo[i] = x[block_base + il * 4 + i];
            yl_hi[i] = x[block_base + il * 4 + 16 + i];
        }

        // Gate weights
        device const uchar* gate_bp = gate_row_ptr + ib * 18u;
        float gate_scale = float(as_type<half>(*(device const ushort*)gate_bp));
        device const uchar* gate_qdata = (device const uchar*)(gate_bp + 2) + il * 4;
        float gate_dot = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            uchar bv = gate_qdata[i];
            float lo = float(bv & 0x0F) - 8.0f;
            float hi = float(bv >> 4) - 8.0f;
            gate_dot += lo * yl_lo[i] + hi * yl_hi[i];
        }
        gate_sum += gate_dot * gate_scale;

        // Up weights
        device const uchar* up_bp = up_row_ptr + ib * 18u;
        float up_scale = float(as_type<half>(*(device const ushort*)up_bp));
        device const uchar* up_qdata = (device const uchar*)(up_bp + 2) + il * 4;
        float up_dot = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            uchar bv = up_qdata[i];
            float lo = float(bv & 0x0F) - 8.0f;
            float hi = float(bv >> 4) - 8.0f;
            up_dot += lo * yl_lo[i] + hi * yl_hi[i];
        }
        up_sum += up_dot * up_scale;
    }

    // Final reduction
    gate_sum = simd_sum(gate_sum);
    up_sum   = simd_sum(up_sum);

    threadgroup float shmem[8];
    if (tiisg == 0) {
        shmem[sgitg]     = gate_sum;
        shmem[sgitg + 4] = up_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0 && tiisg == 0) {
        float g = shmem[0] + shmem[1] + shmem[2] + shmem[3];
        float u = shmem[4] + shmem[5] + shmem[6] + shmem[7];
        float sigmoid = 1.0f / (1.0f + exp(-g));
        swiglu_out[expert_slot * inter_dim + row] = g * sigmoid * u;
    }
}

// ============================================================================
// Phase 1b: Batched gate+up+SwiGLU for top_k experts (Q4_1)
//
// Identical to moe_batched_gate_up_swiglu_q4_0 but uses Q4_1 block layout:
//   [f16 scale (2B)] [f16 min (2B)] [16B nibbles] = 20 bytes per 32 elements
// Dequant: value = scale * nibble + min
//
// buffer(0): layer_buf        — full layer weight buffer
// buffer(1): x                — [hidden_dim] float (normed input)
// buffer(2): swiglu_out       — [top_k * inter_dim] float
// buffer(3): expert_ids       — [top_k] uint
// buffer(4): expert_offsets   — [num_experts * 2] ulong (gate_off, up_off per expert)
// buffer(5): hidden_dim (uint)
// buffer(6): inter_dim (uint)
// buffer(7): top_k (uint)
//
// Dispatch: (top_k * inter_dim) threadgroups, 128 threads each.
// threadgroup_position_in_grid encodes (expert_slot * inter_dim + row).
// ============================================================================

kernel void moe_batched_gate_up_swiglu_q4_1(
    device const uchar*  layer_buf      [[buffer(0)]],
    device const float*  x              [[buffer(1)]],
    device float*        swiglu_out     [[buffer(2)]],
    device const uint*   expert_ids     [[buffer(3)]],
    device const ulong*  expert_offsets [[buffer(4)]],
    constant uint&       hidden_dim     [[buffer(5)]],
    constant uint&       inter_dim      [[buffer(6)]],
    constant uint&       top_k          [[buffer(7)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    // Decode threadgroup index into (expert_slot, row)
    uint row = tgpig % inter_dim;
    uint expert_slot = tgpig / inter_dim;
    if (expert_slot >= top_k) return;

    // Look up expert ID and weight offsets
    uint expert_id = expert_ids[expert_slot];
    ulong gate_off = expert_offsets[expert_id * 2 + 0];
    ulong up_off   = expert_offsets[expert_id * 2 + 1];

    // Q4_1 row layout
    uint num_blocks = hidden_dim >> 5;  // hidden_dim / 32
    uint row_bytes = num_blocks * Q4_1_BLOCK_SIZE;  // 20 bytes per block

    device const uchar* gate_row_ptr = layer_buf + gate_off + row * row_bytes;
    device const uchar* up_row_ptr   = layer_buf + up_off   + row * row_bytes;

    // Thread mapping: each thread handles 8 contiguous elements within a block
    uint ix = tiisg / 4;   // 0..7: which block in stride of 8
    uint il = tiisg % 4;   // 0..3: which 4-byte (8-element) quarter

    float gate_sum = 0.0f;
    float up_sum   = 0.0f;

    for (uint ib = sgitg * 8 + ix; ib < num_blocks; ib += 32) {
        uint block_base = ib * 32;
        float yl_lo[4], yl_hi[4];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            yl_lo[i] = x[block_base + il * 4 + i];
            yl_hi[i] = x[block_base + il * 4 + 16 + i];
        }

        // Gate weights: Q4_1 block = [f16 scale, f16 min, 16B nibbles]
        device const uchar* gate_bp = gate_row_ptr + ib * Q4_1_BLOCK_SIZE;
        float gate_scale = float(as_type<half>(*(device const ushort*)gate_bp));
        float gate_min   = float(as_type<half>(*(device const ushort*)(gate_bp + 2)));
        device const uchar* gate_qdata = (device const uchar*)(gate_bp + 4) + il * 4;
        float gate_dot = 0.0f;
        float gate_sumy = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            uchar bv = gate_qdata[i];
            float lo = float(bv & 0x0F);
            float hi = float(bv >> 4);
            gate_dot  += lo * yl_lo[i] + hi * yl_hi[i];
            gate_sumy += yl_lo[i] + yl_hi[i];
        }
        gate_sum += gate_dot * gate_scale + gate_sumy * gate_min;

        // Up weights: same x-values, different weight row
        device const uchar* up_bp = up_row_ptr + ib * Q4_1_BLOCK_SIZE;
        float up_scale = float(as_type<half>(*(device const ushort*)up_bp));
        float up_min   = float(as_type<half>(*(device const ushort*)(up_bp + 2)));
        device const uchar* up_qdata = (device const uchar*)(up_bp + 4) + il * 4;
        float up_dot = 0.0f;
        float up_sumy = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            uchar bv = up_qdata[i];
            float lo = float(bv & 0x0F);
            float hi = float(bv >> 4);
            up_dot  += lo * yl_lo[i] + hi * yl_hi[i];
            up_sumy += yl_lo[i] + yl_hi[i];
        }
        up_sum += up_dot * up_scale + up_sumy * up_min;
    }

    // Final reduction
    gate_sum = simd_sum(gate_sum);
    up_sum   = simd_sum(up_sum);

    threadgroup float shmem[8];
    if (tiisg == 0) {
        shmem[sgitg]     = gate_sum;
        shmem[sgitg + 4] = up_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0 && tiisg == 0) {
        float g = shmem[0] + shmem[1] + shmem[2] + shmem[3];
        float u = shmem[4] + shmem[5] + shmem[6] + shmem[7];
        float sigmoid = 1.0f / (1.0f + exp(-g));
        swiglu_out[expert_slot * inter_dim + row] = g * sigmoid * u;
    }
}
// Phase 1: Batched gate+up+SwiGLU for top_k experts (Q8_0)
//
// Identical structure to moe_batched_gate_up_swiglu_q4_0 but with Q8_0 blocks:
//   [f16 scale (2B)] [32B int8 data] = 34 bytes per 32 elements
// Dequant: value = scale * int8_val
//
// Thread mapping: deferred-reduction pattern with 128 threads (4 simdgroups).
//   ix = tiisg / 4  -> 0..7  (which block in stride of 8)
//   il = tiisg % 4  -> 0..3  (which 8-byte quarter within 32-byte data)
//
// Each threadgroup processes 1 output row for 1 expert.
// gate and up weights share the same x-vector load.
//
// buffer(0): layer_buf        -- full layer weight buffer
// buffer(1): x                -- normed input [hidden_dim] float
// buffer(2): swiglu_out       -- output [top_k * inter_dim] float (dense packed)
// buffer(3): expert_ids       -- [top_k] uint (from router, on GPU)
// buffer(4): expert_offsets   -- [num_experts * 2] ulong (gate_off, up_off per expert)
// buffer(5): hidden_dim (uint)
// buffer(6): inter_dim (uint)
// buffer(7): top_k (uint)
//
// Dispatch: (top_k * inter_dim) threadgroups, 128 threads each.
// threadgroup_position_in_grid encodes (expert_slot * inter_dim + row).
// ============================================================================

kernel void moe_batched_gate_up_swiglu_q8_0(
    device const uchar*  layer_buf      [[buffer(0)]],
    device const float*  x              [[buffer(1)]],
    device float*        swiglu_out     [[buffer(2)]],
    device const uint*   expert_ids     [[buffer(3)]],
    device const ulong*  expert_offsets [[buffer(4)]],
    constant uint&       hidden_dim     [[buffer(5)]],
    constant uint&       inter_dim      [[buffer(6)]],
    constant uint&       top_k          [[buffer(7)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    // Decode threadgroup index into (expert_slot, row)
    uint row = tgpig % inter_dim;
    uint expert_slot = tgpig / inter_dim;
    if (expert_slot >= top_k) return;

    // Look up expert ID and weight offsets
    uint expert_id = expert_ids[expert_slot];
    ulong gate_off = expert_offsets[expert_id * 2 + 0];
    ulong up_off   = expert_offsets[expert_id * 2 + 1];

    // Q8_0 row layout
    const uint Q8_BLOCK_SIZE = 34;
    uint num_blocks = hidden_dim >> 5;  // hidden_dim / 32
    uint row_bytes = num_blocks * Q8_BLOCK_SIZE;

    device const uchar* gate_row_ptr = layer_buf + gate_off + row * row_bytes;
    device const uchar* up_row_ptr   = layer_buf + up_off   + row * row_bytes;

    // Thread mapping: each thread handles 8 contiguous elements within a block
    uint ix = tiisg / 4;   // 0..7: which block in stride of 8
    uint il = tiisg % 4;   // 0..3: which 8-byte quarter within 32-byte data

    float gate_sum = 0.0f;
    float up_sum   = 0.0f;

    // Each simdgroup starts at a different offset and strides by NSG * 8 blocks
    // NSG=4 simdgroups, each strides 8 blocks -> total stride = 32 blocks/iter
    for (uint ib = sgitg * 8 + ix; ib < num_blocks; ib += 32) {
        // Load 8 x-values for this thread's quarter of the block
        uint x_base = ib * 32 + il * 8;
        float yl[8];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; i++) {
            yl[i] = x[x_base + i];
        }

        // Gate weights: read scale + 8 int8 values
        device const uchar* gate_bp = gate_row_ptr + ib * Q8_BLOCK_SIZE;
        half gate_scale = as_type<half>(*(device const ushort*)gate_bp);
        device const char* gate_qs = (device const char*)(gate_bp + 2) + il * 8;
        float gate_dot = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; i++) {
            gate_dot += float(gate_qs[i]) * yl[i];
        }
        gate_sum += gate_dot * float(gate_scale);

        // Up weights: same x-values, different weight row
        device const uchar* up_bp = up_row_ptr + ib * Q8_BLOCK_SIZE;
        half up_scale = as_type<half>(*(device const ushort*)up_bp);
        device const char* up_qs = (device const char*)(up_bp + 2) + il * 8;
        float up_dot = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; i++) {
            up_dot += float(up_qs[i]) * yl[i];
        }
        up_sum += up_dot * float(up_scale);
    }

    // Final reduction: simd_sum within each simdgroup
    gate_sum = simd_sum(gate_sum);
    up_sum   = simd_sum(up_sum);

    // Cross-simdgroup reduction via shared memory
    threadgroup float shmem[8];  // [0..3] for gate, [4..7] for up

    if (tiisg == 0) {
        shmem[sgitg]     = gate_sum;
        shmem[sgitg + 4] = up_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 of simdgroup 0 does final reduction + SwiGLU
    if (sgitg == 0 && tiisg == 0) {
        float g = shmem[0] + shmem[1] + shmem[2] + shmem[3];
        float u = shmem[4] + shmem[5] + shmem[6] + shmem[7];
        float sigmoid = 1.0f / (1.0f + exp(-g));
        swiglu_out[expert_slot * inter_dim + row] = g * sigmoid * u;
    }
}

// ============================================================================
// Phase 2: Batched down projection + weighted accumulation for top_k experts (Q8_0)
//
// For each output element d, computes:
//   output[d] = residual[d] + sum_{k=0}^{top_k-1} expert_weights[k] *
//               dot(w_down[expert_ids[k]][d, :], swiglu_results[k, :])
//
// Q8_0 block: [f16 scale (2B)] [32B int8 data] = 34 bytes per 32 elements
//
// buffer(0): layer_buf        -- full layer weight buffer
// buffer(1): swiglu_in        -- [top_k * inter_dim] float (from phase 1)
// buffer(2): output           -- [hidden_dim] float (x_buf)
// buffer(3): residual         -- [hidden_dim] float (attn_proj_buf)
// buffer(4): expert_ids       -- [top_k] uint
// buffer(5): expert_weights   -- [top_k] float
// buffer(6): down_offsets     -- [num_experts] ulong (byte offset per expert for down weights)
// buffer(7): inter_dim (uint)
// buffer(8): hidden_dim (uint)
// buffer(9): top_k (uint)
//
// Dispatch: ceil(hidden_dim / 4) threadgroups, 128 threads each
// (deferred-reduction: 4 output rows per threadgroup)
// ============================================================================

kernel void moe_batched_down_accum_q8_0(
    device const uchar*  layer_buf       [[buffer(0)]],
    device const float*  swiglu_in       [[buffer(1)]],
    device float*        output          [[buffer(2)]],
    device const float*  residual        [[buffer(3)]],
    device const uint*   expert_ids      [[buffer(4)]],
    device const float*  expert_weights  [[buffer(5)]],
    device const ulong*  down_offsets    [[buffer(6)]],
    constant uint&       inter_dim       [[buffer(7)]],
    constant uint&       hidden_dim      [[buffer(8)]],
    constant uint&       top_k           [[buffer(9)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    // 4 output rows per threadgroup (deferred reduction pattern)
    uint base_row = tgpig * 4;

    const uint Q8_BLOCK_SIZE = 34;
    uint num_blocks = inter_dim >> 5;  // inter_dim / 32
    uint row_bytes = num_blocks * Q8_BLOCK_SIZE;

    // Thread mapping: each thread handles 8 contiguous elements within a block
    uint ix = tiisg / 4;   // 0..7: which block in stride of 8
    uint il = tiisg % 4;   // 0..3: which 8-byte quarter within 32-byte data

    // Accumulate over all top_k experts for each of 4 output rows
    float row_sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint k = 0; k < top_k; k++) {
        uint expert_id = expert_ids[k];
        ulong down_off = down_offsets[expert_id];
        float w_k = expert_weights[k];
        device const float* swiglu_k = swiglu_in + k * inter_dim;

        for (uint r = 0; r < 4; r++) {
            uint row = base_row + r;
            if (row >= hidden_dim) continue;

            device const uchar* row_ptr = layer_buf + down_off + row * row_bytes;

            float dot = 0.0f;
            for (uint ib = sgitg * 8 + ix; ib < num_blocks; ib += 32) {
                uint x_base = ib * 32 + il * 8;
                float yl[8];
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 8; i++) {
                    yl[i] = swiglu_k[x_base + i];
                }

                device const uchar* bp = row_ptr + ib * Q8_BLOCK_SIZE;
                half scale = as_type<half>(*(device const ushort*)bp);
                device const char* qs = (device const char*)(bp + 2) + il * 8;
                float d = 0.0f;
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 8; i++) {
                    d += float(qs[i]) * yl[i];
                }
                dot += d * float(scale);
            }
            row_sums[r] += w_k * dot;
        }
    }

    // Reduce each row_sum across simdgroups
    for (uint r = 0; r < 4; r++) {
        row_sums[r] = simd_sum(row_sums[r]);
    }

    threadgroup float shmem[16]; // 4 rows * 4 simdgroups

    if (tiisg == 0) {
        for (uint r = 0; r < 4; r++) {
            shmem[r * 4 + sgitg] = row_sums[r];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0 && tiisg == 0) {
        for (uint r = 0; r < 4; r++) {
            uint row = base_row + r;
            if (row >= hidden_dim) continue;
            float total = shmem[r * 4] + shmem[r * 4 + 1] + shmem[r * 4 + 2] + shmem[r * 4 + 3];
            output[row] = residual[row] + total;
        }
    }
}

// ============================================================================

// ============================================================================
// Phase 2: Batched down projection + weighted accumulation for top_k experts (Q4_0)
//
// For each output element d, computes:
//   output[d] = residual[d] + sum_{k=0}^{top_k-1} expert_weights[k] *
//               dot(w_down[expert_ids[k]][d, :], swiglu_results[k, :])
//
// buffer(0): layer_buf        — full layer weight buffer
// buffer(1): swiglu_in        — [top_k * inter_dim] float (from phase 1)
// buffer(2): output           — [hidden_dim] float (x_buf)
// buffer(3): residual         — [hidden_dim] float (attn_proj_buf)
// buffer(4): expert_ids       — [top_k] uint
// buffer(5): expert_weights   — [top_k] float
// buffer(6): down_offsets     — [num_experts] ulong (byte offset per expert for down weights)
// buffer(7): inter_dim (uint)
// buffer(8): hidden_dim (uint)
// buffer(9): top_k (uint)
//
// Dispatch: ceil(hidden_dim / 4) threadgroups, 128 threads each
// (deferred-reduction: 4 output rows per threadgroup)
// ============================================================================

kernel void moe_batched_down_accum_q4_0(
    device const uchar*  layer_buf       [[buffer(0)]],
    device const float*  swiglu_in       [[buffer(1)]],
    device float*        output          [[buffer(2)]],
    device const float*  residual        [[buffer(3)]],
    device const uint*   expert_ids      [[buffer(4)]],
    device const float*  expert_weights  [[buffer(5)]],
    device const ulong*  down_offsets    [[buffer(6)]],
    constant uint&       inter_dim       [[buffer(7)]],
    constant uint&       hidden_dim      [[buffer(8)]],
    constant uint&       top_k           [[buffer(9)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    // 4 output rows per threadgroup (deferred reduction pattern)
    uint base_row = tgpig * 4;

    uint num_blocks = inter_dim >> 5;  // inter_dim / 32
    uint row_bytes = num_blocks * 18u;  // Q4_BLOCK_SIZE = 18

    uint ix = tiisg / 4;
    uint il = tiisg % 4;

    // Accumulate over all top_k experts for each of 4 output rows
    float row_sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint k = 0; k < top_k; k++) {
        uint expert_id = expert_ids[k];
        ulong down_off = down_offsets[expert_id];
        float w_k = expert_weights[k];
        device const float* swiglu_k = swiglu_in + k * inter_dim;

        for (uint r = 0; r < 4; r++) {
            uint row = base_row + r;
            if (row >= hidden_dim) continue;

            device const uchar* row_ptr = layer_buf + down_off + row * row_bytes;

            float dot = 0.0f;
            for (uint ib = sgitg * 8 + ix; ib < num_blocks; ib += 32) {
                uint block_base = ib * 32;
                float yl_lo[4], yl_hi[4];
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 4; ++i) {
                    yl_lo[i] = swiglu_k[block_base + il * 4 + i];
                    yl_hi[i] = swiglu_k[block_base + il * 4 + 16 + i];
                }

                device const uchar* bp = row_ptr + ib * 18u;
                float scale = float(as_type<half>(*(device const ushort*)bp));
                device const uchar* qdata = (device const uchar*)(bp + 2) + il * 4;
                float d = 0.0f;
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 4; ++i) {
                    uchar bv = qdata[i];
                    float lo = float(bv & 0x0F) - 8.0f;
                    float hi = float(bv >> 4) - 8.0f;
                    d += lo * yl_lo[i] + hi * yl_hi[i];
                }
                dot += d * scale;
            }
            row_sums[r] += w_k * dot;
        }
    }

    // Reduce each row_sum across simdgroups
    for (uint r = 0; r < 4; r++) {
        row_sums[r] = simd_sum(row_sums[r]);
    }

    threadgroup float shmem[16]; // 4 rows * 4 simdgroups

    if (tiisg == 0) {
        for (uint r = 0; r < 4; r++) {
            shmem[r * 4 + sgitg] = row_sums[r];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0 && tiisg == 0) {
        for (uint r = 0; r < 4; r++) {
            uint row = base_row + r;
            if (row >= hidden_dim) continue;
            float total = shmem[r * 4] + shmem[r * 4 + 1] + shmem[r * 4 + 2] + shmem[r * 4 + 3];
            output[row] = residual[row] + total;
        }
    }
}

// ============================================================================
// Phase 2: Batched down projection + weighted accumulation for top_k experts (Q4_1)
//
// Identical to moe_batched_down_accum_q4_0 but uses Q4_1 block layout:
//   [f16 scale (2B)] [f16 min (2B)] [16B nibbles] = 20 bytes per 32 elements
// Dequant: value = scale * nibble + min
//
// buffer(0): layer_buf        — full layer weight buffer
// buffer(1): swiglu_in        — [top_k * inter_dim] float (from phase 1)
// buffer(2): output           — [hidden_dim] float (x_buf)
// buffer(3): residual         — [hidden_dim] float (attn_proj_buf)
// buffer(4): expert_ids       — [top_k] uint
// buffer(5): expert_weights   — [top_k] float
// buffer(6): down_offsets     — [num_experts] ulong (byte offset per expert for down weights)
// buffer(7): inter_dim (uint)
// buffer(8): hidden_dim (uint)
// buffer(9): top_k (uint)
//
// Dispatch: ceil(hidden_dim / 4) threadgroups, 128 threads each
// (deferred-reduction: 4 output rows per threadgroup)
// ============================================================================

kernel void moe_batched_down_accum_q4_1(
    device const uchar*  layer_buf       [[buffer(0)]],
    device const float*  swiglu_in       [[buffer(1)]],
    device float*        output          [[buffer(2)]],
    device const float*  residual        [[buffer(3)]],
    device const uint*   expert_ids      [[buffer(4)]],
    device const float*  expert_weights  [[buffer(5)]],
    device const ulong*  down_offsets    [[buffer(6)]],
    constant uint&       inter_dim       [[buffer(7)]],
    constant uint&       hidden_dim      [[buffer(8)]],
    constant uint&       top_k           [[buffer(9)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    // 4 output rows per threadgroup (deferred reduction pattern)
    uint base_row = tgpig * 4;

    uint num_blocks = inter_dim >> 5;  // inter_dim / 32
    uint row_bytes = num_blocks * Q4_1_BLOCK_SIZE;  // 20 bytes per block

    uint ix = tiisg / 4;
    uint il = tiisg % 4;

    // Accumulate over all top_k experts for each of 4 output rows
    float row_sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint k = 0; k < top_k; k++) {
        uint expert_id = expert_ids[k];
        ulong down_off = down_offsets[expert_id];
        float w_k = expert_weights[k];
        device const float* swiglu_k = swiglu_in + k * inter_dim;

        for (uint r = 0; r < 4; r++) {
            uint row = base_row + r;
            if (row >= hidden_dim) continue;

            device const uchar* row_ptr = layer_buf + down_off + row * row_bytes;

            float sumq = 0.0f;
            float sumy = 0.0f;
            for (uint ib = sgitg * 8 + ix; ib < num_blocks; ib += 32) {
                // De-interleaved: lo nibble elements at [0..15], hi at [16..31]
                uint block_base = ib * 32;
                float yl_lo[4], yl_hi[4];
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 4; ++i) {
                    yl_lo[i] = swiglu_k[block_base + il * 4 + i];
                    yl_hi[i] = swiglu_k[block_base + il * 4 + 16 + i];
                }

                device const uchar* bp = row_ptr + ib * Q4_1_BLOCK_SIZE;
                float scale = float(as_type<half>(*(device const ushort*)bp));
                float minval = float(as_type<half>(*(device const ushort*)(bp + 2)));
                device const uchar* qdata = (device const uchar*)(bp + 4) + il * 4;
                float dq = 0.0f;
                float dy = 0.0f;
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 4; ++i) {
                    uchar bv = qdata[i];
                    float lo = float(bv & 0x0F);
                    float hi = float(bv >> 4);
                    dq += lo * yl_lo[i] + hi * yl_hi[i];
                    dy += yl_lo[i] + yl_hi[i];
                }
                sumq += dq * scale + dy * minval;
            }
            row_sums[r] += w_k * sumq;
        }
    }

    // Reduce each row_sum across simdgroups
    for (uint r = 0; r < 4; r++) {
        row_sums[r] = simd_sum(row_sums[r]);
    }

    threadgroup float shmem[16]; // 4 rows * 4 simdgroups

    if (tiisg == 0) {
        for (uint r = 0; r < 4; r++) {
            shmem[r * 4 + sgitg] = row_sums[r];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0 && tiisg == 0) {
        for (uint r = 0; r < 4; r++) {
            uint row = base_row + r;
            if (row >= hidden_dim) continue;
            float total = shmem[r * 4] + shmem[r * 4 + 1] + shmem[r * 4 + 2] + shmem[r * 4 + 3];
            output[row] = residual[row] + total;
        }
    }
}

// ============================================================================
// Phase 2+SE: Batched down projection + weighted accumulation + shared expert (Q8_0)
//
// Fuses the routed expert down+accum with the shared expert's down projection,
// sigmoid gating, and residual add into a single kernel. Eliminates 3 separate
// dispatches per MoE layer (shared expert down, gating dot product, sigmoid+add).
//
// For each output element d, computes:
//   output[d] = residual[d]
//             + sum_{k=0}^{top_k-1} expert_weights[k] *
//               dot(w_down[expert_ids[k]][d, :], swiglu_results[k, :])
//             + se_gate * dot(w_se_down[d, :], se_swiglu[:])
//
// where se_gate = sigmoid(se_gate_scalar[0]) is precomputed by a prior dispatch.
// When se_gate_scalar is NULL (no shared expert gating), se_gate = 1.0.
//
// buffer(0):  layer_buf        -- full layer weight buffer
// buffer(1):  swiglu_in        -- [top_k * inter_dim] float (from batched phase 1)
// buffer(2):  output           -- [hidden_dim] float (x_buf)
// buffer(3):  residual         -- [hidden_dim] float (attn_proj_buf)
// buffer(4):  expert_ids       -- [top_k] uint
// buffer(5):  expert_weights   -- [top_k] float
// buffer(6):  down_offsets     -- [num_experts] ulong
// buffer(7):  inter_dim (uint)
// buffer(8):  hidden_dim (uint)
// buffer(9):  top_k (uint)
// buffer(10): se_swiglu        -- [se_inter_dim] float (shared expert gate+up+swiglu output)
// buffer(11): se_down_off (ulong) -- byte offset for shared expert down weights in layer_buf
// buffer(12): se_gate_scalar   -- [1] float (pre-sigmoid gating logit, or post-sigmoid value)
// buffer(13): se_inter_dim (uint) -- shared expert intermediate dimension
// buffer(14): se_use_sigmoid (uint) -- 1 = apply sigmoid to se_gate_scalar, 0 = use as-is
//
// Dispatch: ceil(hidden_dim / 4) threadgroups, 128 threads each
// ============================================================================

kernel void moe_batched_down_accum_shared_q8_0(
    device const uchar*  layer_buf        [[buffer(0)]],
    device const float*  swiglu_in        [[buffer(1)]],
    device float*        output           [[buffer(2)]],
    device const float*  residual         [[buffer(3)]],
    device const uint*   expert_ids       [[buffer(4)]],
    device const float*  expert_weights   [[buffer(5)]],
    device const ulong*  down_offsets     [[buffer(6)]],
    constant uint&       inter_dim        [[buffer(7)]],
    constant uint&       hidden_dim       [[buffer(8)]],
    constant uint&       top_k            [[buffer(9)]],
    device const float*  se_swiglu        [[buffer(10)]],
    device const ulong*  se_down_off_ptr  [[buffer(11)]],
    device const float*  se_gate_scalar   [[buffer(12)]],
    constant uint&       se_inter_dim     [[buffer(13)]],
    constant uint&       se_use_sigmoid   [[buffer(14)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    // 4 output rows per threadgroup (deferred reduction pattern)
    uint base_row = tgpig * 4;

    const uint Q8_BLOCK_SIZE = 34;
    uint num_blocks = inter_dim >> 5;
    uint row_bytes = num_blocks * Q8_BLOCK_SIZE;

    uint ix = tiisg / 4;
    uint il = tiisg % 4;

    // Accumulate routed experts for each of 4 output rows
    float row_sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint k = 0; k < top_k; k++) {
        uint expert_id = expert_ids[k];
        ulong down_off = down_offsets[expert_id];
        float w_k = expert_weights[k];
        device const float* swiglu_k = swiglu_in + k * inter_dim;

        for (uint r = 0; r < 4; r++) {
            uint row = base_row + r;
            if (row >= hidden_dim) continue;

            device const uchar* row_ptr = layer_buf + down_off + row * row_bytes;

            float dot = 0.0f;
            for (uint ib = sgitg * 8 + ix; ib < num_blocks; ib += 32) {
                uint x_base = ib * 32 + il * 8;
                float yl[8];
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 8; i++) {
                    yl[i] = swiglu_k[x_base + i];
                }

                device const uchar* bp = row_ptr + ib * Q8_BLOCK_SIZE;
                half scale = as_type<half>(*(device const ushort*)bp);
                device const char* qs = (device const char*)(bp + 2) + il * 8;
                float d = 0.0f;
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 8; i++) {
                    d += float(qs[i]) * yl[i];
                }
                dot += d * float(scale);
            }
            row_sums[r] += w_k * dot;
        }
    }

    // Shared expert down projection: same Q8_0 layout, using se_swiglu as input
    ulong se_off = se_down_off_ptr[0];
    uint se_num_blocks = se_inter_dim >> 5;
    uint se_row_bytes = se_num_blocks * Q8_BLOCK_SIZE;

    float se_sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint r = 0; r < 4; r++) {
        uint row = base_row + r;
        if (row >= hidden_dim) continue;

        device const uchar* se_row_ptr = layer_buf + se_off + row * se_row_bytes;

        float dot = 0.0f;
        for (uint ib = sgitg * 8 + ix; ib < se_num_blocks; ib += 32) {
            uint x_base = ib * 32 + il * 8;
            float yl[8];
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8; i++) {
                yl[i] = se_swiglu[x_base + i];
            }

            device const uchar* bp = se_row_ptr + ib * Q8_BLOCK_SIZE;
            half scale = as_type<half>(*(device const ushort*)bp);
            device const char* qs = (device const char*)(bp + 2) + il * 8;
            float d = 0.0f;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8; i++) {
                d += float(qs[i]) * yl[i];
            }
            dot += d * float(scale);
        }
        se_sums[r] = dot;
    }

    // Reduce both routed and shared sums across simdgroups
    for (uint r = 0; r < 4; r++) {
        row_sums[r] = simd_sum(row_sums[r]);
        se_sums[r] = simd_sum(se_sums[r]);
    }

    threadgroup float shmem[32]; // 4 rows * 4 simdgroups * 2 (routed + shared)

    if (tiisg == 0) {
        for (uint r = 0; r < 4; r++) {
            shmem[r * 4 + sgitg] = row_sums[r];
            shmem[16 + r * 4 + sgitg] = se_sums[r];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0 && tiisg == 0) {
        // Read shared expert gate value and apply sigmoid if needed
        // se_use_sigmoid=1: apply sigmoid to gate_inp_shexp dot product
        // se_use_sigmoid=0: no gating, shared expert added at full weight (1.0)
        float se_gate = se_use_sigmoid ? (1.0f / (1.0f + exp(-se_gate_scalar[0]))) : 1.0f;

        for (uint r = 0; r < 4; r++) {
            uint row = base_row + r;
            if (row >= hidden_dim) continue;
            float routed_total = shmem[r * 4] + shmem[r * 4 + 1] + shmem[r * 4 + 2] + shmem[r * 4 + 3];
            float se_total = shmem[16 + r * 4] + shmem[16 + r * 4 + 1] + shmem[16 + r * 4 + 2] + shmem[16 + r * 4 + 3];
            output[row] = residual[row] + routed_total + se_gate * se_total;
        }
    }
}

// ============================================================================
// Phase 2+SE: Batched down (Q8_0 routed) + shared expert (Q4_0)
// Mixed-quant variant: routed experts use Q8_0, shared expert uses Q4_0.
// This is the common case for Qwen3.5-35B-A3B where the converter requantizes
// shared expert weights to Q4_0 for memory savings.
// ============================================================================

kernel void moe_batched_down_accum_shared_q8_0_se_q4_0(
    device const uchar*  layer_buf        [[buffer(0)]],
    device const float*  swiglu_in        [[buffer(1)]],
    device float*        output           [[buffer(2)]],
    device const float*  residual         [[buffer(3)]],
    device const uint*   expert_ids       [[buffer(4)]],
    device const float*  expert_weights   [[buffer(5)]],
    device const ulong*  down_offsets     [[buffer(6)]],
    constant uint&       inter_dim        [[buffer(7)]],
    constant uint&       hidden_dim       [[buffer(8)]],
    constant uint&       top_k            [[buffer(9)]],
    device const float*  se_swiglu        [[buffer(10)]],
    device const ulong*  se_down_off_ptr  [[buffer(11)]],
    device const float*  se_gate_scalar   [[buffer(12)]],
    constant uint&       se_inter_dim     [[buffer(13)]],
    constant uint&       se_use_sigmoid   [[buffer(14)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    uint base_row = tgpig * 4;

    // Routed experts: Q8_0
    const uint Q8_BLOCK_SIZE = 34;
    uint num_blocks = inter_dim >> 5;
    uint row_bytes = num_blocks * Q8_BLOCK_SIZE;

    uint ix = tiisg / 4;
    uint il = tiisg % 4;

    float row_sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint k = 0; k < top_k; k++) {
        uint expert_id = expert_ids[k];
        ulong down_off = down_offsets[expert_id];
        float w_k = expert_weights[k];
        device const float* swiglu_k = swiglu_in + k * inter_dim;

        for (uint r = 0; r < 4; r++) {
            uint row = base_row + r;
            if (row >= hidden_dim) continue;

            device const uchar* row_ptr = layer_buf + down_off + row * row_bytes;

            float dot = 0.0f;
            for (uint ib = sgitg * 8 + ix; ib < num_blocks; ib += 32) {
                uint x_base = ib * 32 + il * 8;
                float yl[8];
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 8; i++) {
                    yl[i] = swiglu_k[x_base + i];
                }

                device const uchar* bp = row_ptr + ib * Q8_BLOCK_SIZE;
                half scale = as_type<half>(*(device const ushort*)bp);
                device const char* qs = (device const char*)(bp + 2) + il * 8;
                float d = 0.0f;
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 8; i++) {
                    d += float(qs[i]) * yl[i];
                }
                dot += d * float(scale);
            }
            row_sums[r] += w_k * dot;
        }
    }

    // Shared expert down projection: Q4_0 layout
    ulong se_off = se_down_off_ptr[0];
    uint se_num_blocks = se_inter_dim >> 5;
    uint se_row_bytes = se_num_blocks * 18u;  // Q4_0: 18 bytes per block

    float se_sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint r = 0; r < 4; r++) {
        uint row = base_row + r;
        if (row >= hidden_dim) continue;

        device const uchar* se_row_ptr = layer_buf + se_off + row * se_row_bytes;

        float dot = 0.0f;
        for (uint ib = sgitg * 8 + ix; ib < se_num_blocks; ib += 32) {
            uint block_base = ib * 32;
            float yl_lo[4], yl_hi[4];
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; ++i) {
                yl_lo[i] = se_swiglu[block_base + il * 4 + i];
                yl_hi[i] = se_swiglu[block_base + il * 4 + 16 + i];
            }

            device const uchar* bp = se_row_ptr + ib * 18u;
            float scale = float(as_type<half>(*(device const ushort*)bp));
            device const uchar* qdata = (device const uchar*)(bp + 2) + il * 4;
            float d = 0.0f;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; ++i) {
                uchar bv = qdata[i];
                float lo = float(bv & 0x0F) - 8.0f;
                float hi = float(bv >> 4) - 8.0f;
                d += lo * yl_lo[i] + hi * yl_hi[i];
            }
            dot += d * scale;
        }
        se_sums[r] = dot;
    }

    for (uint r = 0; r < 4; r++) {
        row_sums[r] = simd_sum(row_sums[r]);
        se_sums[r] = simd_sum(se_sums[r]);
    }

    threadgroup float shmem[32];

    if (tiisg == 0) {
        for (uint r = 0; r < 4; r++) {
            shmem[r * 4 + sgitg] = row_sums[r];
            shmem[16 + r * 4 + sgitg] = se_sums[r];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0 && tiisg == 0) {
        // se_use_sigmoid=1: apply sigmoid to gate_inp_shexp dot product
        // se_use_sigmoid=0: no gating, shared expert added at full weight (1.0)
        float se_gate = se_use_sigmoid ? (1.0f / (1.0f + exp(-se_gate_scalar[0]))) : 1.0f;

        for (uint r = 0; r < 4; r++) {
            uint row = base_row + r;
            if (row >= hidden_dim) continue;
            float routed_total = shmem[r * 4] + shmem[r * 4 + 1] + shmem[r * 4 + 2] + shmem[r * 4 + 3];
            float se_total = shmem[16 + r * 4] + shmem[16 + r * 4 + 1] + shmem[16 + r * 4 + 2] + shmem[16 + r * 4 + 3];
            output[row] = residual[row] + routed_total + se_gate * se_total;
        }
    }
}

// ============================================================================
// Phase 2+SE: Batched down + accum + shared expert (Q4_0)
// Same as Q8_0 variant but uses Q4_0 block layout for both routed and shared experts.
// ============================================================================

kernel void moe_batched_down_accum_shared_q4_0(
    device const uchar*  layer_buf        [[buffer(0)]],
    device const float*  swiglu_in        [[buffer(1)]],
    device float*        output           [[buffer(2)]],
    device const float*  residual         [[buffer(3)]],
    device const uint*   expert_ids       [[buffer(4)]],
    device const float*  expert_weights   [[buffer(5)]],
    device const ulong*  down_offsets     [[buffer(6)]],
    constant uint&       inter_dim        [[buffer(7)]],
    constant uint&       hidden_dim       [[buffer(8)]],
    constant uint&       top_k            [[buffer(9)]],
    device const float*  se_swiglu        [[buffer(10)]],
    device const ulong*  se_down_off_ptr  [[buffer(11)]],
    device const float*  se_gate_scalar   [[buffer(12)]],
    constant uint&       se_inter_dim     [[buffer(13)]],
    constant uint&       se_use_sigmoid   [[buffer(14)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    uint base_row = tgpig * 4;

    uint num_blocks = inter_dim >> 5;
    uint row_bytes = num_blocks * 18u;

    uint ix = tiisg / 4;
    uint il = tiisg % 4;

    float row_sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint k = 0; k < top_k; k++) {
        uint expert_id = expert_ids[k];
        ulong down_off = down_offsets[expert_id];
        float w_k = expert_weights[k];
        device const float* swiglu_k = swiglu_in + k * inter_dim;

        for (uint r = 0; r < 4; r++) {
            uint row = base_row + r;
            if (row >= hidden_dim) continue;

            device const uchar* row_ptr = layer_buf + down_off + row * row_bytes;

            float dot = 0.0f;
            for (uint ib = sgitg * 8 + ix; ib < num_blocks; ib += 32) {
                uint block_base = ib * 32;
                float yl_lo[4], yl_hi[4];
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 4; ++i) {
                    yl_lo[i] = swiglu_k[block_base + il * 4 + i];
                    yl_hi[i] = swiglu_k[block_base + il * 4 + 16 + i];
                }

                device const uchar* bp = row_ptr + ib * 18u;
                float scale = float(as_type<half>(*(device const ushort*)bp));
                device const uchar* qdata = (device const uchar*)(bp + 2) + il * 4;
                float d = 0.0f;
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 4; ++i) {
                    uchar bv = qdata[i];
                    float lo = float(bv & 0x0F) - 8.0f;
                    float hi = float(bv >> 4) - 8.0f;
                    d += lo * yl_lo[i] + hi * yl_hi[i];
                }
                dot += d * scale;
            }
            row_sums[r] += w_k * dot;
        }
    }

    // Shared expert down projection (Q4_0)
    ulong se_off = se_down_off_ptr[0];
    uint se_num_blocks = se_inter_dim >> 5;
    uint se_row_bytes = se_num_blocks * 18u;

    float se_sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint r = 0; r < 4; r++) {
        uint row = base_row + r;
        if (row >= hidden_dim) continue;

        device const uchar* se_row_ptr = layer_buf + se_off + row * se_row_bytes;

        float dot = 0.0f;
        for (uint ib = sgitg * 8 + ix; ib < se_num_blocks; ib += 32) {
            uint block_base = ib * 32;
            float yl_lo[4], yl_hi[4];
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; ++i) {
                yl_lo[i] = se_swiglu[block_base + il * 4 + i];
                yl_hi[i] = se_swiglu[block_base + il * 4 + 16 + i];
            }

            device const uchar* bp = se_row_ptr + ib * 18u;
            float scale = float(as_type<half>(*(device const ushort*)bp));
            device const uchar* qdata = (device const uchar*)(bp + 2) + il * 4;
            float d = 0.0f;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; ++i) {
                uchar bv = qdata[i];
                float lo = float(bv & 0x0F) - 8.0f;
                float hi = float(bv >> 4) - 8.0f;
                d += lo * yl_lo[i] + hi * yl_hi[i];
            }
            dot += d * scale;
        }
        se_sums[r] = dot;
    }

    for (uint r = 0; r < 4; r++) {
        row_sums[r] = simd_sum(row_sums[r]);
        se_sums[r] = simd_sum(se_sums[r]);
    }

    threadgroup float shmem[32];

    if (tiisg == 0) {
        for (uint r = 0; r < 4; r++) {
            shmem[r * 4 + sgitg] = row_sums[r];
            shmem[16 + r * 4 + sgitg] = se_sums[r];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0 && tiisg == 0) {
        // se_use_sigmoid=1: apply sigmoid to gate_inp_shexp dot product
        // se_use_sigmoid=0: no gating, shared expert added at full weight (1.0)
        float se_gate = se_use_sigmoid ? (1.0f / (1.0f + exp(-se_gate_scalar[0]))) : 1.0f;

        for (uint r = 0; r < 4; r++) {
            uint row = base_row + r;
            if (row >= hidden_dim) continue;
            float routed_total = shmem[r * 4] + shmem[r * 4 + 1] + shmem[r * 4 + 2] + shmem[r * 4 + 3];
            float se_total = shmem[16 + r * 4] + shmem[16 + r * 4 + 1] + shmem[16 + r * 4 + 2] + shmem[16 + r * 4 + 3];
            output[row] = residual[row] + routed_total + se_gate * se_total;
        }
    }
}

// ============================================================================
// GatedDeltaNet (Linear Attention) kernels for Qwen3.5-35B-A3B decode
//
// These kernels implement the recurrent state-space computation used in
// GatedDeltaNet layers (30 of 40 layers in Qwen3.5-35B-A3B). For single-token
// decode, this is O(1) per token (fixed-size state update) rather than O(n)
// KV cache scan.
//
// Core operations:
//   1. Short causal 1D convolution (kernel_size=4)
//   2. Per-head L2 normalization of Q and K
//   3. Sigmoid gating (beta)
//   4. Delta rule state update: h = (1-beta)*h + beta*outer(k,v)
//   5. Output from state: out = h^T @ q
//   6. SiLU gating (alpha) and elementwise multiply
// ============================================================================

// ============================================================================
// ssm_conv1d_decode: Causal 1D convolution for single-token decode
//
// Maintains a circular buffer of the last (kernel_size-1) token activations.
// For each dimension d:
//   output[d] = sum_{i=0}^{kernel_size-1} kernel_w[d * kernel_size + i] * state[i][d]
// where state includes the current input as the newest entry.
//
// buffer(0): input_val   [dim] float         — current token activation
// buffer(1): conv_state  [(kernel_size-1) * dim] float — circular buffer (R/W)
// buffer(2): kernel_w    [dim * kernel_size] float     — conv weights [dim, kernel_size]
// buffer(3): output      [dim] float                   — OUTPUT
// buffer(4): dim (uint)
// buffer(5): kernel_size (uint)
// buffer(6): state_pos (uint)  — current write position in circular buffer [0..kernel_size-2]
//
// grid: (ceil(dim/256), 1, 1), threadgroup: (256, 1, 1)
// ============================================================================

kernel void ssm_conv1d_decode(
    device const float* input_val  [[buffer(0)]],
    device       float* conv_state [[buffer(1)]],
    device const float* kernel_w   [[buffer(2)]],
    device       float* output     [[buffer(3)]],
    constant     uint&  dim        [[buffer(4)]],
    constant     uint&  kernel_size [[buffer(5)]],
    constant     uint&  state_pos  [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= dim) return;

    float sum = 0.0f;
    uint buf_slots = kernel_size - 1;

    // Taps 0..kernel_size-2: read from circular buffer (oldest to newest)
    for (uint tap = 0; tap < buf_slots; tap++) {
        uint slot = (state_pos + tap) % buf_slots;
        sum += kernel_w[gid * kernel_size + tap] * conv_state[slot * dim + gid];
    }

    // Tap kernel_size-1: current input (newest)
    sum += kernel_w[gid * kernel_size + buf_slots] * input_val[gid];

    output[gid] = sum;

    // Update circular buffer: overwrite oldest entry (at state_pos) with current input
    conv_state[state_pos * dim + gid] = input_val[gid];
}

// ============================================================================
// ssm_conv1d_silu_decode: Fused Conv1D + SiLU for GDN decode
//
// Same as ssm_conv1d_decode but applies SiLU activation to the output:
//   output[gid] = sum * sigmoid(sum) = sum / (1 + exp(-sum))
//
// Eliminates 1 dispatch + 1 barrier per GDN layer vs separate conv1d + silu.
// ============================================================================

kernel void ssm_conv1d_silu_decode(
    device const float* input_val  [[buffer(0)]],
    device       float* conv_state [[buffer(1)]],
    device const float* kernel_w   [[buffer(2)]],
    device       float* output     [[buffer(3)]],
    constant     uint&  dim        [[buffer(4)]],
    constant     uint&  kernel_size [[buffer(5)]],
    constant     uint&  state_pos  [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= dim) return;

    float sum = 0.0f;
    uint buf_slots = kernel_size - 1;

    // Taps 0..kernel_size-2: read from circular buffer (oldest to newest)
    for (uint tap = 0; tap < buf_slots; tap++) {
        uint slot = (state_pos + tap) % buf_slots;
        sum += kernel_w[gid * kernel_size + tap] * conv_state[slot * dim + gid];
    }

    // Tap kernel_size-1: current input (newest)
    sum += kernel_w[gid * kernel_size + buf_slots] * input_val[gid];

    // Fused SiLU: output = sum * sigmoid(sum)
    output[gid] = sum / (1.0f + exp(-sum));

    // Update circular buffer: overwrite oldest entry (at state_pos) with current input
    conv_state[state_pos * dim + gid] = input_val[gid];
}

// ============================================================================
// l2_normalize_heads: Per-head L2 normalization
//
// For each head h: x[h*head_dim .. (h+1)*head_dim] /= max(||x_head||, eps)
//
// buffer(0): x         [n_heads * head_dim] float — modified in-place
// buffer(1): n_heads (uint)
// buffer(2): head_dim (uint)
// buffer(3): eps (float)
//
// grid: (n_heads, 1, 1), threadgroup: (min(head_dim, 256), 1, 1)
// ============================================================================

kernel void l2_normalize_heads(
    device       float* x        [[buffer(0)]],
    constant     uint&  n_heads  [[buffer(1)]],
    constant     uint&  head_dim [[buffer(2)]],
    constant     float& eps      [[buffer(3)]],
    uint head_idx                [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]],
    uint tg_size                 [[threads_per_threadgroup]],
    uint simd_lane               [[thread_index_in_simdgroup]],
    uint simd_group              [[simdgroup_index_in_threadgroup]])
{
    if (head_idx >= n_heads) return;

    device float* head = x + head_idx * head_dim;

    // Pass 1: compute sum of squares
    float ss = 0.0f;
    for (uint i = tid; i < head_dim; i += tg_size) {
        float v = head[i];
        ss += v * v;
    }

    ss = simd_sum(ss);

    threadgroup float partial_sums[8];

    if (simd_lane == 0) {
        partial_sums[simd_group] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simd_groups = (tg_size + 31) / 32;
    threadgroup float total_ss;
    if (simd_group == 0) {
        float val = (simd_lane < num_simd_groups) ? partial_sums[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            total_ss = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float norm = sqrt(total_ss);
    float scale = (norm > eps) ? (1.0f / norm) : (1.0f / eps);

    // Pass 2: normalize
    for (uint i = tid; i < head_dim; i += tg_size) {
        head[i] *= scale;
    }
}

// ============================================================================
// sigmoid_gate: Elementwise sigmoid
//
// output[i] = 1 / (1 + exp(-input[i]))
//
// buffer(0): input  [dim] float
// buffer(1): output [dim] float
// buffer(2): dim (uint)
//
// grid: (ceil(dim/256), 1, 1), threadgroup: (256, 1, 1)
// ============================================================================

kernel void sigmoid_gate(
    device const float* input  [[buffer(0)]],
    device       float* output [[buffer(1)]],
    constant     uint&  dim    [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= dim) return;
    output[gid] = 1.0f / (1.0f + exp(-input[gid]));
}

// ============================================================================
// silu_inplace: In-place SiLU activation
//
// x[i] = x[i] * sigmoid(x[i]) = x[i] / (1 + exp(-x[i]))
//
// buffer(0): x   [dim] float -- modified in-place
// buffer(1): dim (uint)
//
// grid: (ceil(dim/256), 1, 1), threadgroup: (256, 1, 1)
// ============================================================================

kernel void silu_inplace(
    device       float* x   [[buffer(0)]],
    constant     uint&  dim [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= dim) return;
    float val = x[gid];
    x[gid] = val / (1.0f + exp(-val));
}

// ============================================================================
// silu_elementwise_mul: SiLU(alpha) * x elementwise
//
// output[i] = silu(alpha[i]) * x[i]
// silu(a) = a * sigmoid(a) = a / (1 + exp(-a))
//
// buffer(0): alpha  [dim] float — gate values
// buffer(1): x      [dim] float — values to gate
// buffer(2): output [dim] float
// buffer(3): dim (uint)
//
// grid: (ceil(dim/256), 1, 1), threadgroup: (256, 1, 1)
// ============================================================================

kernel void silu_elementwise_mul(
    device const float* alpha  [[buffer(0)]],
    device const float* x      [[buffer(1)]],
    device       float* output [[buffer(2)]],
    constant     uint&  dim    [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= dim) return;
    float a = alpha[gid];
    float sigmoid = 1.0f / (1.0f + exp(-a));
    output[gid] = a * sigmoid * x[gid];
}

// ============================================================================
// gated_delta_net_state_update: Delta rule recurrent state update
//
// For each head h (0..n_heads):
//   h_state[h] = (1 - beta[h]) * h_state[h] + beta[h] * outer(k_norm[h], v[h])
//
// h_state layout: [n_heads, val_dim, key_dim] — transposed for coalesced access
// k_norm layout:  [n_heads * key_dim]
// v layout:       [n_kv_heads * val_dim] — may differ from n_heads (GQA)
//
// Each thread handles one (key_i, val_j) element for ALL heads.
// Grid is 2D: (key_dim, val_dim). Loops over heads.
//
// buffer(0): h_state    [n_heads * val_dim * key_dim] float — R/W
// buffer(1): k_norm     [n_heads * key_dim] float
// buffer(2): v_tokens   [n_kv_heads * val_dim] float
// buffer(3): beta       [n_heads] float — sigmoid-gated interpolation rates
// buffer(4): n_heads (uint)
// buffer(5): key_dim (uint)
// buffer(6): val_dim (uint)
// buffer(7): n_kv_heads (uint)
//
// grid: (key_dim, val_dim, 1), threadgroup: (min(key_dim,16), min(val_dim,16), 1)
// ============================================================================

kernel void gated_delta_net_state_update(
    device       float* h_state    [[buffer(0)]],
    device const float* k_norm     [[buffer(1)]],
    device const float* v_tokens   [[buffer(2)]],
    device const float* beta       [[buffer(3)]],
    constant     uint&  n_heads    [[buffer(4)]],
    constant     uint&  key_dim    [[buffer(5)]],
    constant     uint&  val_dim    [[buffer(6)]],
    constant     uint&  n_kv_heads [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint ki = gid.x;
    uint vj = gid.y;

    if (ki >= key_dim || vj >= val_dim) return;

    uint gqa_ratio = n_heads / n_kv_heads;

    for (uint h = 0; h < n_heads; h++) {
        float b = beta[h];
        float k_val = k_norm[h * key_dim + ki];

        uint kv_head = h / gqa_ratio;
        float v_val = v_tokens[kv_head * val_dim + vj];

        uint state_idx = h * val_dim * key_dim + vj * key_dim + ki;
        h_state[state_idx] = (1.0f - b) * h_state[state_idx] + b * k_val * v_val;
    }
}

// ============================================================================
// gated_delta_net_output: Compute output from recurrent state
//
// For each head h, value dimension j:
//   output[h * val_dim + j] = dot(h_state[h, :, j], q_norm[h, :])
//
// Each threadgroup computes one output element using 32-wide SIMD reduction.
//
// buffer(0): h_state  [n_heads * val_dim * key_dim] float
// buffer(1): q_norm   [n_heads * key_dim] float
// buffer(2): output   [n_heads * val_dim] float
// buffer(3): n_heads (uint)
// buffer(4): key_dim (uint)
// buffer(5): val_dim (uint)
//
// grid: (n_heads * val_dim, 1, 1), threadgroup: (32, 1, 1)
// ============================================================================

kernel void gated_delta_net_output(
    device const float* h_state  [[buffer(0)]],
    device const float* q_norm   [[buffer(1)]],
    device       float* output   [[buffer(2)]],
    constant     uint&  n_heads  [[buffer(3)]],
    constant     uint&  key_dim  [[buffer(4)]],
    constant     uint&  val_dim  [[buffer(5)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint lane  [[thread_index_in_simdgroup]])
{
    uint h = tgpig / val_dim;
    uint j = tgpig % val_dim;
    if (h >= n_heads) return;

    float sum = 0.0f;
    device const float* h_row = h_state + h * val_dim * key_dim + j * key_dim;
    device const float* q_head = q_norm + h * key_dim;

    for (uint i = lane; i < key_dim; i += 32) {
        sum += h_row[i] * q_head[i];
    }

    sum = simd_sum(sum);

    if (lane == 0) {
        output[h * val_dim + j] = sum;
    }
}

// ============================================================================
// gated_delta_net_state_update_v2: Delta rule with independent alpha and beta
//
// For each head h (0..n_heads):
//   h_state[h] = alpha[h] * h_state[h] + beta[h] * outer(k_norm[h], v[h])
//
// Unlike v1 which constrains decay = (1-beta), this version takes separate
// alpha (decay) and beta (mixing) arrays computed from the gating parameters.
//
// h_state layout: [n_heads, val_dim, key_dim] -- transposed for coalesced access
// k_norm layout:  [n_heads * key_dim]
// v layout:       [n_kv_heads * val_dim] -- may differ from n_heads (GQA)
//
// buffer(0): h_state    [n_heads * val_dim * key_dim] float -- R/W
// buffer(1): k_norm     [n_heads * key_dim] float
// buffer(2): v_tokens   [n_kv_heads * val_dim] float
// buffer(3): alpha      [n_heads] float -- decay factors
// buffer(4): beta       [n_heads] float -- mixing rates
// buffer(5): n_heads (uint)
// buffer(6): key_dim (uint)
// buffer(7): val_dim (uint)
// buffer(8): n_kv_heads (uint)
//
// grid: (key_dim, val_dim, 1), threadgroup: (min(key_dim,16), min(val_dim,16), 1)
// ============================================================================

kernel void gated_delta_net_state_update_v2(
    device       float* h_state    [[buffer(0)]],
    device const float* k_norm     [[buffer(1)]],
    device const float* v_tokens   [[buffer(2)]],
    device const float* alpha      [[buffer(3)]],
    device const float* beta       [[buffer(4)]],
    constant     uint&  n_heads    [[buffer(5)]],
    constant     uint&  key_dim    [[buffer(6)]],
    constant     uint&  val_dim    [[buffer(7)]],
    constant     uint&  n_kv_heads [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint ki = gid.x;
    uint vj = gid.y;

    if (ki >= key_dim || vj >= val_dim) return;

    uint gqa_ratio = n_heads / n_kv_heads;

    for (uint h = 0; h < n_heads; h++) {
        float a = alpha[h];
        float b = beta[h];
        uint kv_head = h / gqa_ratio;
        float k_val = k_norm[kv_head * key_dim + ki];     // use kv_head for GQA k_norm
        float v_val = v_tokens[kv_head * val_dim + vj];

        uint state_idx = h * val_dim * key_dim + vj * key_dim + ki;
        h_state[state_idx] = a * h_state[state_idx] + b * k_val * v_val;
    }
}

// ============================================================================
// gdn_compute_gates: Compute GatedDeltaNet gating parameters
//
// Per-head computation (matching NVLabs GatedDeltaNet reference):
//   gate[h] = -exp(A_log[h]) * softplus(gk_proj[h] + dt_bias[h])
//   alpha[h] = exp(gate[h])     // decay factor in (0,1)
//   beta[h]  = sigmoid(gn_proj[h])
//
// GGUF stores ssm_a = -exp(A_log) (already negated and exponentiated).
// Typical values: ssm_a ~ -0.036 (slow decay) to -72 (fast decay).
// gate = ssm_a * softplus(gk_proj + dt_bias), so gate is negative => alpha = exp(gate) in (0,1).
// Reference: gk = -exp(A_log) * softplus(gk_proj + dt_bias)  [NVLabs/GatedDeltaNet]
//
// buffer(0): ssm_dt_bias       [n_heads] float -- dt bias per head
// buffer(1): ssm_a             [n_heads] float -- -exp(A_log) (pre-negated+exponentiated)
// buffer(2): ssm_beta_weight   [n_heads] float -- gn_proj output (pre-sigmoid)
// buffer(3): ssm_alpha_weight  [n_heads] float -- gk_proj output (from matvec)
// buffer(4): alpha_out         [n_heads] float -- OUTPUT: decay factors
// buffer(5): beta_out          [n_heads] float -- OUTPUT: mixing rates
// buffer(6): n_heads (uint)
//
// grid: (ceil(n_heads/256), 1, 1), threadgroup: (256, 1, 1)
// ============================================================================

kernel void gdn_compute_gates(
    device const float* ssm_dt_bias       [[buffer(0)]],
    device const float* ssm_a             [[buffer(1)]],
    device const float* ssm_beta_weight   [[buffer(2)]],
    device const float* ssm_alpha_weight  [[buffer(3)]],
    device       float* alpha_out         [[buffer(4)]],
    device       float* beta_out          [[buffer(5)]],
    constant     uint&  n_heads           [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n_heads) return;

    // softplus(x) = log(1 + exp(x)), numerically stable
    float sp_input = ssm_alpha_weight[gid] + ssm_dt_bias[gid];
    float sp;
    if (sp_input > 20.0f) {
        sp = sp_input;  // softplus(x) ~= x for large x
    } else {
        sp = log(1.0f + exp(sp_input));
    }

    // gate = ssm_a * softplus(gk_proj + dt_bias)
    // GGUF stores ssm_a = -exp(A_log) (already negated+exponentiated).
    // Reference (llama.cpp): gate = ggml_mul(alpha_softplus, ssm_a)
    // Typical values: ssm_a ~ -0.036 (slow decay) to -72 (fast decay).
    // gate is negative => alpha = exp(gate) in (0, 1).
    float gate = ssm_a[gid] * sp;
    alpha_out[gid] = exp(gate);

    // beta = sigmoid(gn_proj)
    float beta_raw = ssm_beta_weight[gid];
    beta_out[gid] = 1.0f / (1.0f + exp(-beta_raw));
}

// ============================================================================
// dequant_matmul_q8_0_dual_gates_nr2: Fused dual alpha+beta matvec + gates
//
// Merges 3 decode dispatches into 1:
//   1. RMSNorm + alpha_raw matvec (rmsnorm_dequant_matmul_q8_0_deferred_nr2)
//   2. RMSNorm + beta_raw matvec  (same kernel, different weights)
//   3. gdn_compute_gates (softplus+exp for alpha, sigmoid for beta)
//
// Reads x_buf ONCE, computes inline RMSNorm, performs dual Q8_0 dot products
// for alpha_raw and beta_raw, then applies gate transforms inline.
//
// Gate math (matching gdn_compute_gates exactly):
//   softplus(x) = x > 20 ? x : log(1 + exp(x))
//   gate = ssm_a[h] * softplus(alpha_raw[h] + dt_bias[h])
//   alpha_out[h] = exp(gate)        // decay factor in (0,1)
//   beta_out[h]  = sigmoid(beta_raw) // mixing rate
//
// buffer(0): weights_alpha  [N * row_bytes] Q8_0 alpha weight matrix
// buffer(1): weights_beta   [N * row_bytes] Q8_0 beta weight matrix
// buffer(2): x              [K] float -- raw input (un-normalized)
// buffer(3): alpha_out      [N] float -- OUTPUT: decay factors
// buffer(4): beta_out       [N] float -- OUTPUT: mixing rates
// buffer(5): norm_w         [K] float (as uchar*) -- RMSNorm weights
// buffer(6): ssm_a          [N] float -- -exp(A_log), pre-negated
// buffer(7): dt_bias        [N] float -- per-head dt bias
// buffer(8): K              (uint) -- hidden_dim
// buffer(9): N              (uint) -- num_heads (output dim)
// buffer(10): eps           (float) -- RMSNorm epsilon
//
// grid: (ceil(N/2), 1, 1), threadgroup: (128, 1, 1)
// ============================================================================

kernel void dequant_matmul_q8_0_dual_gates_nr2(
    device const uchar* weights_alpha [[buffer(0)]],
    device const uchar* weights_beta  [[buffer(1)]],
    device const float* x             [[buffer(2)]],
    device float*       alpha_out     [[buffer(3)]],
    device float*       beta_out      [[buffer(4)]],
    device const uchar* norm_w        [[buffer(5)]],
    device const float* ssm_a         [[buffer(6)]],
    device const float* dt_bias       [[buffer(7)]],
    constant uint&      K             [[buffer(8)]],
    constant uint&      N             [[buffer(9)]],
    constant float&     eps           [[buffer(10)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    // NR2 deferred-reduction with fused RMSNorm + dual dot product + gates.
    // Same single-pass fusion as rmsnorm_dequant_matmul_q8_0_deferred_nr2:
    // dot(W, norm_x) = scale * dot(W, x * norm_w)
    // We compute dot(W_alpha, x*nw) and dot(W_beta, x*nw) AND sum(x^2)
    // in the SAME loop, then multiply by scale and apply gate transforms.

    const uint NR0 = 2;
    const uint NSG = 4;
    const uint NQ = 8;
    const uint NW = 32;
    const uint Q8_BLOCK_SIZE = 34;

    device const float* norm_weight = (device const float*)norm_w;

    const uint nb = K >> 5;
    const uint row_bytes = nb * Q8_BLOCK_SIZE;

    const uint r0 = tgpig * NR0;

    const uint ix = tiisg / (NW / NQ);
    const uint il = tiisg % (NW / NQ);

    const uint ib0 = sgitg * NQ + ix;

    // Weight row pointers for both matrices
    device const uchar* ax_a[NR0];
    device const uchar* ax_b[NR0];
    for (uint row = 0; row < NR0; ++row) {
        ax_a[row] = weights_alpha + (r0 + row) * row_bytes;
        ax_b[row] = weights_beta  + (r0 + row) * row_bytes;
    }

    float sumf_a[NR0] = { 0.f, 0.f };
    float sumf_b[NR0] = { 0.f, 0.f };
    float ss = 0.0f;  // sum of squares for RMSNorm

    float yl[NQ];

    device const float* yb = x + ib0 * 32 + il * NQ;
    device const float* nwb = norm_weight + ib0 * 32 + il * NQ;

    for (uint ib = ib0; ib < nb; ib += NSG * NQ) {
        // Load x * norm_w (without scale) and accumulate x^2 simultaneously
        #pragma clang loop unroll(full)
        for (uint i = 0; i < NQ; ++i) {
            float xi = yb[i];
            ss += xi * xi;
            yl[i] = xi * nwb[i];
        }

        for (uint row = 0; row < NR0; ++row) {
            if (r0 + row >= N) break;

            // Alpha weights
            {
                device const uchar* bp = ax_a[row] + ib * Q8_BLOCK_SIZE;
                float scale = float(as_type<half>(*(device const ushort*)bp));
                device const char* qs = (device const char*)(bp + 2) + il * NQ;

                float sumq = 0.f;
                #pragma clang loop unroll(full)
                for (uint i = 0; i < NQ; ++i) {
                    sumq += float(qs[i]) * yl[i];
                }
                sumf_a[row] += sumq * scale;
            }

            // Beta weights
            {
                device const uchar* bp = ax_b[row] + ib * Q8_BLOCK_SIZE;
                float scale = float(as_type<half>(*(device const ushort*)bp));
                device const char* qs = (device const char*)(bp + 2) + il * NQ;

                float sumq = 0.f;
                #pragma clang loop unroll(full)
                for (uint i = 0; i < NQ; ++i) {
                    sumq += float(qs[i]) * yl[i];
                }
                sumf_b[row] += sumq * scale;
            }
        }

        yb += NSG * NQ * 32;
        nwb += NSG * NQ * 32;
    }

    // Reduce sum-of-squares across all 128 threads
    ss = simd_sum(ss);

    threadgroup float shmem[NR0 * NW];

    if (tiisg == 0) {
        shmem[sgitg] = ss;
    }

    // Also reduce dot products for both alpha and beta
    for (uint row = 0; row < NR0; ++row) {
        sumf_a[row] = simd_sum(sumf_a[row]);
        sumf_b[row] = simd_sum(sumf_b[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute RMSNorm scale from reduced sum-of-squares
    threadgroup float rms_scale_shared;
    if (sgitg == 0) {
        float total_ss = (tiisg < NSG) ? shmem[tiisg] : 0.0f;
        total_ss = simd_sum(total_ss);
        if (tiisg == 0) {
            rms_scale_shared = rsqrt(total_ss / float(K) + eps);
        }
    }

    // Write alpha partial sums to shmem
    for (uint row = 0; row < NR0; ++row) {
        if (sgitg == 0) {
            shmem[row * NW + tiisg] = 0.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem[row * NW + sgitg] = sumf_a[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms_scale = rms_scale_shared;

    // Final reduction of alpha dot products, apply scale + gate transform
    for (uint row = 0; row < NR0 && r0 + row < N; ++row) {
        float tot_a = simd_sum(shmem[row * NW + tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            float alpha_raw = tot_a * rms_scale;

            // Gate: alpha = exp(ssm_a * softplus(alpha_raw + dt_bias))
            float sp_input = alpha_raw + dt_bias[r0 + row];
            float sp;
            if (sp_input > 20.0f) {
                sp = sp_input;  // softplus(x) ~= x for large x
            } else {
                sp = log(1.0f + exp(sp_input));
            }
            float gate = ssm_a[r0 + row] * sp;
            alpha_out[r0 + row] = exp(gate);
        }
    }

    // Now reduce beta partial sums using shmem
    for (uint row = 0; row < NR0; ++row) {
        if (sgitg == 0) {
            shmem[row * NW + tiisg] = 0.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem[row * NW + sgitg] = sumf_b[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction of beta dot products, apply scale + sigmoid
    for (uint row = 0; row < NR0 && r0 + row < N; ++row) {
        float tot_b = simd_sum(shmem[row * NW + tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            float beta_raw = tot_b * rms_scale;

            // Gate: beta = sigmoid(beta_raw)
            beta_out[r0 + row] = 1.0f / (1.0f + exp(-beta_raw));
        }
    }
}

// ============================================================================
// elementwise_mul_f32: Element-wise multiplication of two float arrays
//
// output[i] = a[i] * b[i]
//
// buffer(0): a      [dim] float
// buffer(1): b      [dim] float
// buffer(2): output [dim] float
// buffer(3): dim (uint)
//
// grid: (ceil(dim/256), 1, 1), threadgroup: (256, 1, 1)
// ============================================================================

kernel void elementwise_mul_f32(
    device const float* a      [[buffer(0)]],
    device const float* b      [[buffer(1)]],
    device       float* output [[buffer(2)]],
    constant     uint&  dim    [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= dim) return;
    output[gid] = a[gid] * b[gid];
}

// ============================================================================
// ssm_rms_norm_scale: RMS-normalize then multiply by learned scale per element
//
// For each head h:
//   rms = sqrt(mean(x[h*head_dim..(h+1)*head_dim]^2) + eps)
//   scale_head = h % scale_n_heads  (allows shared scale across heads)
//   output[h*head_dim+i] = (x[h*head_dim+i] / rms) * scale[scale_head*head_dim+i]
//
// buffer(0): x              [n_heads * head_dim] float -- input (not modified)
// buffer(1): scale          [scale_n_heads * head_dim] float -- learned per-element scale
// buffer(2): output         [n_heads * head_dim] float -- OUTPUT
// buffer(3): n_heads        (uint)
// buffer(4): head_dim       (uint)
// buffer(5): eps            (float)
// buffer(6): scale_n_heads  (uint) -- number of heads in scale tensor (1 = shared)
//
// grid: (n_heads, 1, 1), threadgroup: (min(head_dim, 256), 1, 1)
// ============================================================================

kernel void ssm_l2_norm_scale(
    device const float* x              [[buffer(0)]],
    device const float* scale          [[buffer(1)]],
    device       float* output         [[buffer(2)]],
    constant     uint&  n_heads        [[buffer(3)]],
    constant     uint&  head_dim       [[buffer(4)]],
    constant     float& eps            [[buffer(5)]],
    constant     uint&  scale_n_heads  [[buffer(6)]],
    uint head_idx                [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]],
    uint tg_size                 [[threads_per_threadgroup]],
    uint simd_lane               [[thread_index_in_simdgroup]],
    uint simd_group              [[simdgroup_index_in_threadgroup]])
{
    if (head_idx >= n_heads) return;

    device const float* head_in = x + head_idx * head_dim;

    // Pass 1: compute sum of squares
    float ss = 0.0f;
    for (uint i = tid; i < head_dim; i += tg_size) {
        float v = head_in[i];
        ss += v * v;
    }

    ss = simd_sum(ss);

    threadgroup float partial_sums[8];
    if (simd_lane == 0) {
        partial_sums[simd_group] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simd_groups = (tg_size + 31) / 32;
    threadgroup float total_ss;
    if (simd_group == 0) {
        float val = (simd_lane < num_simd_groups) ? partial_sums[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            total_ss = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // RMSNorm: divide by sqrt(mean(x^2) + eps)
    float rms = sqrt(total_ss / (float)head_dim + eps);
    float inv_norm = 1.0f / rms;

    // Pass 2: normalize and scale (modular head indexing for shared scale)
    device float* head_out = output + head_idx * head_dim;
    device const float* head_scale = scale + (head_idx % scale_n_heads) * head_dim;
    for (uint i = tid; i < head_dim; i += tg_size) {
        head_out[i] = head_in[i] * inv_norm * head_scale[i];
    }
}

// ============================================================================
// Fused element-wise kernels for GDN dispatch reduction
// ============================================================================

// sigmoid_mul_fused: Fused sigmoid + element-wise multiply
//
// output[i] = sigmoid(gate[i]) * x[i]
//           = x[i] / (1 + exp(-gate[i]))
//
// Replaces separate sigmoid_gate + elementwise_mul_f32 dispatches.
//
// buffer(0): gate   [dim] float -- raw gate values (pre-sigmoid)
// buffer(1): x      [dim] float -- values to multiply
// buffer(2): output [dim] float
// buffer(3): dim (uint)
//
// grid: (ceil(dim/256), 1, 1), threadgroup: (256, 1, 1)
// ============================================================================

kernel void sigmoid_mul_fused(
    device const float* gate   [[buffer(0)]],
    device const float* x      [[buffer(1)]],
    device       float* output [[buffer(2)]],
    constant     uint&  dim    [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= dim) return;
    float sig = 1.0f / (1.0f + exp(-gate[gid]));
    output[gid] = sig * x[gid];
}

// ============================================================================
// deinterleave_norm_assemble: Fused deinterleave Q+gate, per-head RMSNorm Q/K,
// assemble K|V into qkv_out, and write normalized Q to q_out.
// Replaces 6 separate dispatches + 2 barriers with 1 dispatch.
//
// NOTE: Q is written to q_out (NOT qkv_out) to avoid aliasing with the
// interleaved Q+gate input in qkv_out. The caller must copy q_out to
// qkv_out[0..q_dim] after this kernel completes.
//
// Input: interleaved Q+gate [num_heads * 2 * head_dim] in qgate_interleaved,
//        K [num_kv_heads * head_dim], V [num_kv_heads * head_dim]
// Output: q_out [num_heads * head_dim] (normalized Q, separate buffer)
//         qkv_out: K at [q_dim..q_dim+kv_dim), V at [q_dim+kv_dim..)
//         gate_out [num_heads * head_dim]
//
// Dispatch: (num_heads + num_kv_heads) threadgroups, min(head_dim, 256) threads each.
//   TGs [0, num_heads): deinterleave + RMSNorm Q head, write gate. Copy V if h < num_kv_heads.
//   TGs [num_heads, num_heads+num_kv_heads): RMSNorm K head.
//
// buffer(0):  qgate_interleaved [num_heads * 2 * head_dim] float
// buffer(1):  k_data       [num_kv_heads * head_dim] float
// buffer(2):  v_data       [num_kv_heads * head_dim] float
// buffer(3):  q_norm_w     [head_dim] float -- shared RMSNorm weight for Q
// buffer(4):  k_norm_w     [head_dim] float -- shared RMSNorm weight for K
// buffer(5):  qkv_out      [q_dim + kv_dim + kv_dim] float -- K|V assembled (Q portion unused)
// buffer(6):  gate_out     [num_heads * head_dim] float
// buffer(7):  num_heads    (uint) -- Q/gate heads
// buffer(8):  num_kv_heads (uint) -- K/V heads
// buffer(9):  head_dim     (uint)
// buffer(10): q_dim        (uint) -- num_heads * head_dim
// buffer(11): kv_dim       (uint) -- num_kv_heads * head_dim
// buffer(12): eps          (float) -- RMSNorm epsilon
// buffer(13): q_out        [num_heads * head_dim] float -- normalized Q (separate buf)
// ============================================================================

kernel void deinterleave_norm_assemble(
    device const float* qgate_interleaved [[buffer(0)]],
    device const float* k_data            [[buffer(1)]],
    device const float* v_data            [[buffer(2)]],
    device const float* q_norm_w          [[buffer(3)]],
    device const float* k_norm_w          [[buffer(4)]],
    device       float* qkv_out           [[buffer(5)]],
    device       float* gate_out          [[buffer(6)]],
    constant     uint&  num_heads         [[buffer(7)]],
    constant     uint&  num_kv_heads      [[buffer(8)]],
    constant     uint&  head_dim          [[buffer(9)]],
    constant     uint&  q_dim             [[buffer(10)]],
    constant     uint&  kv_dim            [[buffer(11)]],
    constant     float& eps               [[buffer(12)]],
    device       float* q_out             [[buffer(13)]],
    uint tg_idx                           [[threadgroup_position_in_grid]],
    uint tid                              [[thread_index_in_threadgroup]],
    uint tg_size                          [[threads_per_threadgroup]],
    uint simd_lane                        [[thread_index_in_simdgroup]],
    uint simd_group                       [[simdgroup_index_in_threadgroup]])
{
    uint total_tgs = num_heads + num_kv_heads;
    if (tg_idx >= total_tgs) return;

    threadgroup float partial_sums[32];
    threadgroup float total_ss;

    if (tg_idx < num_heads) {
        // --- Q head: deinterleave + RMSNorm + write Q and gate ---
        uint h = tg_idx;
        uint interleaved_base = h * 2 * head_dim;

        // Step 1: Load Q and compute RMSNorm sum-of-squares
        float q_val = 0.0f;
        float ss = 0.0f;
        for (uint i = tid; i < head_dim; i += tg_size) {
            q_val = qgate_interleaved[interleaved_base + i];
            ss += q_val * q_val;
            // Also write gate while we're reading the interleaved data
            gate_out[h * head_dim + i] = qgate_interleaved[interleaved_base + head_dim + i];
        }

        ss = simd_sum(ss);
        if (simd_lane == 0) {
            partial_sums[simd_group] = ss;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint num_sg = (tg_size + 31) / 32;
        if (simd_group == 0) {
            float val = (simd_lane < num_sg) ? partial_sums[simd_lane] : 0.0f;
            val = simd_sum(val);
            if (simd_lane == 0) {
                total_ss = val;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float scale = rsqrt(total_ss / float(head_dim) + eps);

        // Step 2: Write normalized Q to q_out (separate buffer to avoid aliasing)
        for (uint i = tid; i < head_dim; i += tg_size) {
            float qv = qgate_interleaved[interleaved_base + i];
            q_out[h * head_dim + i] = qv * scale * q_norm_w[i];
        }

        // Step 3: Copy V head (if this Q head index < num_kv_heads)
        if (h < num_kv_heads) {
            for (uint i = tid; i < head_dim; i += tg_size) {
                qkv_out[q_dim + kv_dim + h * head_dim + i] = v_data[h * head_dim + i];
            }
        }
    } else {
        // --- K head: RMSNorm + write K ---
        uint kh = tg_idx - num_heads;

        // Step 1: Load K and compute RMSNorm sum-of-squares
        float ss = 0.0f;
        for (uint i = tid; i < head_dim; i += tg_size) {
            float kv = k_data[kh * head_dim + i];
            ss += kv * kv;
        }

        ss = simd_sum(ss);
        if (simd_lane == 0) {
            partial_sums[simd_group] = ss;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint num_sg = (tg_size + 31) / 32;
        if (simd_group == 0) {
            float val = (simd_lane < num_sg) ? partial_sums[simd_lane] : 0.0f;
            val = simd_sum(val);
            if (simd_lane == 0) {
                total_ss = val;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float scale = rsqrt(total_ss / float(head_dim) + eps);

        // Step 2: Write normalized K to qkv_out at K offset [q_dim..q_dim+kv_dim)
        for (uint i = tid; i < head_dim; i += tg_size) {
            float kv = k_data[kh * head_dim + i];
            qkv_out[q_dim + kh * head_dim + i] = kv * scale * k_norm_w[i];
        }
    }
}

// ============================================================================
// deinterleave_qgate: Split interleaved [Q_h0, gate_h0, Q_h1, gate_h1, ...] into
// separate Q and gate buffers.
//
// In Qwen3.5 full-attention layers, attn_q.weight produces interleaved Q+gate:
//   [Q_head0(head_dim), gate_head0(head_dim), Q_head1(head_dim), gate_head1(head_dim), ...]
// Total elements = num_heads * 2 * head_dim.
// This kernel copies Q heads to contiguous Q buffer [num_heads * head_dim]
// and gate heads to contiguous gate buffer [num_heads * head_dim].
//
// buffer(0): qgate_interleaved [num_heads * 2 * head_dim] float
// buffer(1): Q     [num_heads * head_dim] float (output)
// buffer(2): gate  [num_heads * head_dim] float (output)
// buffer(3): head_dim (uint)
// buffer(4): num_heads (uint)
//
// grid: (ceil(num_heads * head_dim / 256), 1, 1), threadgroup: (256, 1, 1)
// ============================================================================

kernel void deinterleave_qgate(
    device const float* qgate  [[buffer(0)]],
    device       float* Q      [[buffer(1)]],
    device       float* gate   [[buffer(2)]],
    constant     uint&  head_dim  [[buffer(3)]],
    constant     uint&  num_heads [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint q_dim = num_heads * head_dim;
    if (gid >= q_dim) return;
    uint head = gid / head_dim;
    uint offset_in_head = gid % head_dim;
    // In interleaved layout: head h starts at h * 2 * head_dim
    // Q component at h * 2 * head_dim + offset
    // gate component at h * 2 * head_dim + head_dim + offset
    uint interleaved_base = head * 2 * head_dim;
    Q[gid]    = qgate[interleaved_base + offset_in_head];
    gate[gid] = qgate[interleaved_base + head_dim + offset_in_head];
}

// ============================================================================
// sigmoid_scale_buffer: Apply sigmoid of a scalar to scale an entire buffer.
//
// out[i] = out[i] * sigmoid(scalar[0])
//
// buffer(0): scalar   [1] float -- input scalar (e.g. dot product result)
// buffer(1): out      [dim] float -- R/W buffer to scale in-place
// buffer(2): dim      (uint)
//
// grid: (ceil(dim/256), 1, 1), threadgroup: (256, 1, 1)
// ============================================================================

kernel void sigmoid_scale_buffer(
    device const float* scalar  [[buffer(0)]],
    device float*       out     [[buffer(1)]],
    constant uint&      dim     [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= dim) return;
    float gate = 1.0f / (1.0f + exp(-scalar[0]));
    out[gid] *= gate;
}

// sigmoid_scale_add: Apply sigmoid-gated shared expert output and add to
// destination for decode (single token).
//
// dst[i] += sigmoid(scalar[0]) * src[i]
//
// Fuses sigmoid_scale_buffer + add_residual into 1 dispatch, eliminating
// the intermediate in-place multiply on src and the separate add_residual.
//
// buffer(0): scalar   [1] float -- gate scalar (e.g. dot product result)
// buffer(1): src      [dim] float -- shared expert FFN output (read-only)
// buffer(2): dst      [dim] float -- R/W (x_buf)
// buffer(3): dim      (uint)
//
// grid: (ceil(dim/256), 1, 1), threadgroup: (256, 1, 1)
// ============================================================================

kernel void sigmoid_scale_add(
    device const float* scalar  [[buffer(0)]],
    device const float* src     [[buffer(1)]],
    device float*       dst     [[buffer(2)]],
    constant uint&      dim     [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= dim) return;
    float gate = 1.0f / (1.0f + exp(-scalar[0]));
    dst[gid] += gate * src[gid];
}

// ============================================================================
// rmsnorm_per_head: Per-head RMS normalization for Q or K vectors.
//
// Applies RMSNorm to each head slice independently.
// weight[head_dim] is shared across all heads.
// ============================================================================
// sigmoid_scale_add_batched: For each token in a batch, apply sigmoid-gated
// shared expert output and add to destination.
//
// For token b at element t:
//   dst[b * hidden_dim + t] += sigmoid(gate_scalars[b]) * src[b * hidden_dim + t]
//
// buffer(0): gate_scalars  [batch_size] float -- one scalar per token
// buffer(1): src           [batch_size * hidden_dim] float -- shared expert FFN output
// buffer(2): dst           [batch_size * hidden_dim] float -- R/W (x_buf)
// buffer(3): hidden_dim    (uint)
// buffer(4): batch_size    (uint)
//
// grid: (ceil(batch_size * hidden_dim / 256), 1, 1), threadgroup: (256, 1, 1)
// ============================================================================

kernel void sigmoid_scale_add_batched(
    device const float* gate_scalars [[buffer(0)]],
    device const float* src          [[buffer(1)]],
    device float*       dst          [[buffer(2)]],
    constant uint&      hidden_dim   [[buffer(3)]],
    constant uint&      batch_size   [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = batch_size * hidden_dim;
    if (gid >= total) return;
    uint b = gid / hidden_dim;
    float g = 1.0f / (1.0f + exp(-gate_scalars[b]));
    dst[gid] += g * src[gid];
}

// ============================================================================
// rmsnorm_per_head: Per-head RMS normalization for Q or K vectors.
//
// Applies RMSNorm to each head slice independently.
// weight[head_dim] is shared across all heads.
// x[num_heads * head_dim] is the input/output vector.
//
// out[h * head_dim + i] = x[h * head_dim + i] * weight[i] / sqrt(mean(x_h^2) + eps)
//
// buffer(0): x        [num_heads * head_dim] float -- input
// buffer(1): weight   [head_dim] float -- per-element scale, shared across heads
// buffer(2): out      [num_heads * head_dim] float -- output (can alias x)
// buffer(3): head_dim (uint)
// buffer(4): eps      (float)
//
// Dispatch: threadgroups=(num_heads, 1, 1), threads_per_tg=(min(head_dim, 256), 1, 1)
// Each threadgroup handles one head.
// ============================================================================

kernel void rmsnorm_per_head(
    device const float* x       [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant uint&      head_dim [[buffer(3)]],
    constant float&     eps     [[buffer(4)]],
    uint tid                    [[thread_index_in_threadgroup]],
    uint tg_size                [[threads_per_threadgroup]],
    uint head_idx               [[threadgroup_position_in_grid]],
    uint simd_lane              [[thread_index_in_simdgroup]],
    uint simd_group             [[simdgroup_index_in_threadgroup]])
{
    uint base = head_idx * head_dim;

    // Pass 1: compute sum of squares for this head
    float ss = 0.0f;
    for (uint i = tid; i < head_dim; i += tg_size) {
        float v = x[base + i];
        ss += v * v;
    }

    ss = simd_sum(ss);

    threadgroup float partial_sums[32];
    if (simd_lane == 0) {
        partial_sums[simd_group] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simd_groups = (tg_size + 31) / 32;
    threadgroup float total_ss;
    if (simd_group == 0) {
        float val = (simd_lane < num_simd_groups) ? partial_sums[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            total_ss = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = rsqrt(total_ss / float(head_dim) + eps);

    // Pass 2: normalize and scale by weight
    for (uint i = tid; i < head_dim; i += tg_size) {
        out[base + i] = x[base + i] * scale * weight[i];
    }
}

// ============================================================================
// residual_add_copy: Fused residual addition + buffer copy
//
// dst[i] += src[i]; copy_dst[i] = dst[i]
//
// Replaces separate add_residual + copy_buffer dispatches.
//
// buffer(0): dst      [dim] float -- R/W (accumulator)
// buffer(1): src      [dim] float -- addend
// buffer(2): copy_dst [dim] float -- receives final value of dst
// buffer(3): dim (uint)
//
// grid: (ceil(dim/256), 1, 1), threadgroup: (256, 1, 1)
// ============================================================================

kernel void residual_add_copy(
    device       float* dst      [[buffer(0)]],
    device const float* src      [[buffer(1)]],
    device       float* copy_dst [[buffer(2)]],
    constant     uint&  dim      [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= dim) return;
    float val = dst[gid] + src[gid];
    dst[gid] = val;
    copy_dst[gid] = val;
}

// ============================================================================
// l2_normalize_qk: Fused L2 normalization for both Q and K head blocks
//
// Normalizes Q heads [0..n_q_heads) and K heads [0..n_k_heads) in a single
// dispatch. The two head arrays are at different buffer offsets.
// Each threadgroup handles one head (Q or K), distinguished by threadgroup index.
// Threadgroups 0..n_q_heads-1 handle Q heads, n_q_heads..n_q_heads+n_k_heads-1
// handle K heads.
//
// buffer(0): q_data     [n_q_heads * head_dim] float -- modified in-place
// buffer(1): k_data     [n_k_heads * head_dim] float -- modified in-place
// buffer(2): n_q_heads  (uint)
// buffer(3): n_k_heads  (uint)
// buffer(4): head_dim   (uint)
// buffer(5): eps        (float)
//
// grid: (n_q_heads + n_k_heads, 1, 1), threadgroup: (min(head_dim, 256), 1, 1)
// ============================================================================

kernel void l2_normalize_qk(
    device       float* q_data     [[buffer(0)]],
    device       float* k_data     [[buffer(1)]],
    constant     uint&  n_q_heads  [[buffer(2)]],
    constant     uint&  n_k_heads  [[buffer(3)]],
    constant     uint&  head_dim   [[buffer(4)]],
    constant     float& eps        [[buffer(5)]],
    uint head_idx                  [[threadgroup_position_in_grid]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_group                [[simdgroup_index_in_threadgroup]])
{
    uint total_heads = n_q_heads + n_k_heads;
    if (head_idx >= total_heads) return;

    // Select pointer: Q heads first, then K heads
    device float* head;
    if (head_idx < n_q_heads) {
        head = q_data + head_idx * head_dim;
    } else {
        head = k_data + (head_idx - n_q_heads) * head_dim;
    }

    // Pass 1: compute sum of squares
    float ss = 0.0f;
    for (uint i = tid; i < head_dim; i += tg_size) {
        float v = head[i];
        ss += v * v;
    }

    ss = simd_sum(ss);

    threadgroup float partial_sums[8];
    if (simd_lane == 0) {
        partial_sums[simd_group] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simd_groups = (tg_size + 31) / 32;
    threadgroup float total_ss;
    if (simd_group == 0) {
        float val = (simd_lane < num_simd_groups) ? partial_sums[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            total_ss = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float norm = sqrt(total_ss);
    float scale = (norm > eps) ? (1.0f / norm) : (1.0f / eps);

    // Pass 2: normalize
    for (uint i = tid; i < head_dim; i += tg_size) {
        head[i] *= scale;
    }
}

// ============================================================================
// Fused GDN state-update + output + RMS-norm-scale kernel (DELTA RULE)
//
// Implements the Gated Delta Net recurrence per head h and output column vj:
//   retrieval[vj] = sum_ki(h_state[h,ki,vj] * k[kv_head,ki])  // from ORIGINAL h
//   h_state[h,ki,vj] = alpha[h]*h_state[h,ki,vj] + k[kv_head,ki]*beta[h]*(v[h,vj]-retrieval[vj])
//   raw_out[vj]      = sum_ki(h_state[h,ki,vj] * q[kv_head,ki])   // from NEW h
//   output[vj]       = (raw_out[vj] / sqrt(mean(raw_out^2)+eps)) * scale[vj]
//
// KEY: retrieval uses ORIGINAL (undecayed) h; decay+update happen in one combined step.
// KEY: Q and K both use GQA (kv_head = h / (n_heads/n_kv_heads)).
// KEY: V uses direct head index h (n_heads V-heads, no GQA).
//
// buffer(0):  h_state    [n_heads * val_dim * key_dim] float -- R/W (transposed layout)
// buffer(1):  k_norm     [n_kv_heads * key_dim] float
// buffer(2):  v_tokens   [n_heads * val_dim] float  (n_heads V-heads, no GQA)
// buffer(3):  alpha      [n_heads] float -- decay factors
// buffer(4):  beta       [n_heads] float -- mixing rates
// buffer(5):  q_norm     [n_kv_heads * key_dim] float  (n_kv_heads Q-heads, GQA)
// buffer(6):  scale      [scale_n_heads * val_dim] float -- learned norm scale
// buffer(7):  output     [n_heads * val_dim] float -- OUTPUT
// buffer(8):  n_heads (uint)       -- state/V/output heads (= num_v_heads = 32)
// buffer(9):  key_dim (uint)
// buffer(10): val_dim (uint)
// buffer(11): n_kv_heads (uint)   -- Q and K original heads (= num_k_heads = 16)
// buffer(12): eps (float)
// buffer(13): scale_n_heads (uint)
//
// grid: (n_heads, 1, 1), threadgroup: (val_dim, 1, 1)  -- one thread per column
// ============================================================================

kernel void gdn_state_output_norm(
    device       float* h_state        [[buffer(0)]],
    device const float* k_norm         [[buffer(1)]],
    device const float* v_tokens       [[buffer(2)]],
    device const float* alpha          [[buffer(3)]],
    device const float* beta           [[buffer(4)]],
    device const float* q_norm         [[buffer(5)]],
    device const float* scale          [[buffer(6)]],
    device       float* output         [[buffer(7)]],
    constant     uint&  n_heads        [[buffer(8)]],
    constant     uint&  key_dim        [[buffer(9)]],
    constant     uint&  val_dim        [[buffer(10)]],
    constant     uint&  n_kv_heads     [[buffer(11)]],
    constant     float& eps            [[buffer(12)]],
    constant     uint&  scale_n_heads  [[buffer(13)]],
    uint head_idx                      [[threadgroup_position_in_grid]],
    uint tid                           [[thread_index_in_threadgroup]],
    uint tg_size                       [[threads_per_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_group                    [[simdgroup_index_in_threadgroup]])
{
    if (head_idx >= n_heads) return;

    uint h = head_idx;
    // V has n_heads=32 heads. Q and K have n_kv_heads=16 heads.
    // GGUF stores V-heads de-interleaved: our heads 0..15 = MLX even heads,
    // our heads 16..31 = MLX odd heads. Q/K heads are NOT de-interleaved.
    // Correct GQA mapping for de-interleaved V: kv_head = h % n_kv_heads
    // This maps our heads [0..15] to kv [0..15] and [16..31] to kv [0..15].
    uint kv_head = h % n_kv_heads;

    float a = alpha[h];
    float b = beta[h];
    uint vj = tid;

    // Q scaling factor: 1/sqrt(key_dim), matching llama.cpp build_delta_net_autoregressive
    float q_scale = rsqrt((float)key_dim);

    // GDN recurrence per column vj (h_state is [n_heads, val_dim, key_dim] transposed):
    //   Reference (llama.cpp build_delta_net_autoregressive):
    //     1. s_decayed = alpha * s_old          (decay FIRST)
    //     2. retrieval = s_decayed^T @ k        (retrieve from DECAYED state)
    //     3. delta = beta * (v - retrieval)
    //     4. s_new = s_decayed + outer(k, delta)
    //     5. output = s_new @ (q * scale)       (scale = 1/sqrt(key_dim))
    float my_out = 0.0f;
    if (vj < val_dim) {
        // V has n_heads heads (no GQA) — use h directly
        float v_val = v_tokens[h * val_dim + vj];
        device float* h_row = h_state + h * val_dim * key_dim + vj * key_dim;
        // Q and K both have n_kv_heads original heads (tile-style: kv_head = h * n_kv_heads / n_heads)
        device const float* q_head = q_norm + kv_head * key_dim;
        device const float* k_head = k_norm + kv_head * key_dim;

        // Phase 1: Decay state, then retrieve from DECAYED state
        float retrieval = 0.0f;
        for (uint ki = 0; ki < key_dim; ki++) {
            float h_decayed = a * h_row[ki];
            h_row[ki] = h_decayed;  // write decayed state back
            retrieval += h_decayed * k_head[ki];
        }

        // Phase 2: Delta rule update + output (on already-decayed state)
        float v_delta = b * (v_val - retrieval);
        for (uint ki = 0; ki < key_dim; ki++) {
            float h_updated = h_row[ki] + k_head[ki] * v_delta;
            h_row[ki] = h_updated;
            my_out += h_updated * q_head[ki] * q_scale;
        }
    }

    // Phase 4: RMSNorm + scale (cooperative reduction across threadgroup)
    // llama.cpp build_norm_gated uses ggml_rms_norm: x / sqrt(mean(x^2) + eps) * weight
    float ss = my_out * my_out;
    ss = simd_sum(ss);

    threadgroup float partial_sums[8];
    if (simd_lane == 0) {
        partial_sums[simd_group] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simd_groups = (tg_size + 31) / 32;
    threadgroup float total_ss;
    if (simd_group == 0) {
        float val = (simd_lane < num_simd_groups) ? partial_sums[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            total_ss = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms = sqrt(total_ss / (float)val_dim + eps);
    float inv_norm = 1.0f / rms;

    // Write normalized + scaled output
    if (vj < val_dim) {
        device const float* head_scale = scale + (h % scale_n_heads) * val_dim;
        output[h * val_dim + vj] = my_out * inv_norm * head_scale[vj];
    }
}

// ============================================================================
// gdn_state_output_norm_l2: Fused L2-normalize Q/K + state-update + output + RMSNorm
//
// Same as gdn_state_output_norm but takes UN-normalized Q and K and performs
// per-head L2 normalization inline using threadgroup memory. This eliminates
// the separate l2_normalize_qk dispatch and its preceding barrier.
//
// Each threadgroup (one per head) independently loads its Q/K head into
// threadgroup memory, computes L2 norm via simd_sum, normalizes, then
// proceeds with the standard GDN recurrence using the local normalized copy.
//
// buffer(0):  h_state    [n_heads * val_dim * key_dim] float -- R/W (transposed layout)
// buffer(1):  k_raw      [n_kv_heads * key_dim] float -- UN-normalized
// buffer(2):  v_tokens   [n_heads * val_dim] float  (n_heads V-heads, no GQA)
// buffer(3):  alpha      [n_heads] float -- decay factors
// buffer(4):  beta       [n_heads] float -- mixing rates
// buffer(5):  q_raw      [n_kv_heads * key_dim] float -- UN-normalized
// buffer(6):  scale      [scale_n_heads * val_dim] float -- learned norm scale
// buffer(7):  output     [n_heads * val_dim] float -- OUTPUT
// buffer(8):  n_heads (uint)       -- state/V/output heads (= 32)
// buffer(9):  key_dim (uint)       -- 128
// buffer(10): val_dim (uint)       -- 128
// buffer(11): n_kv_heads (uint)    -- Q and K original heads (= 16)
// buffer(12): eps (float)          -- RMSNorm eps
// buffer(13): scale_n_heads (uint)
// buffer(14): l2_eps (float)       -- L2 norm eps (typically 1e-12)
//
// grid: (n_heads, 1, 1), threadgroup: (val_dim, 1, 1)  -- one thread per column
// REQUIREMENT: key_dim == val_dim (both 128 for Qwen3.5)
// ============================================================================

kernel void gdn_state_output_norm_l2(
    device       float* h_state        [[buffer(0)]],
    device const float* k_raw          [[buffer(1)]],
    device const float* v_tokens       [[buffer(2)]],
    device const float* alpha          [[buffer(3)]],
    device const float* beta           [[buffer(4)]],
    device const float* q_raw          [[buffer(5)]],
    device const float* scale          [[buffer(6)]],
    device       float* output         [[buffer(7)]],
    constant     uint&  n_heads        [[buffer(8)]],
    constant     uint&  key_dim        [[buffer(9)]],
    constant     uint&  val_dim        [[buffer(10)]],
    constant     uint&  n_kv_heads     [[buffer(11)]],
    constant     float& eps            [[buffer(12)]],
    constant     uint&  scale_n_heads  [[buffer(13)]],
    constant     float& l2_eps         [[buffer(14)]],
    uint head_idx                      [[threadgroup_position_in_grid]],
    uint tid                           [[thread_index_in_threadgroup]],
    uint tg_size                       [[threads_per_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_group                    [[simdgroup_index_in_threadgroup]])
{
    if (head_idx >= n_heads) return;

    uint h = head_idx;
    uint kv_head = h % n_kv_heads;

    // --- Phase 0: L2 normalize Q and K in threadgroup memory ---
    // key_dim == val_dim == tg_size == 128, so thread tid loads element tid.
    threadgroup float q_local[128];
    threadgroup float k_local[128];

    float q_val = 0.0f;
    float k_val = 0.0f;
    if (tid < key_dim) {
        q_val = q_raw[kv_head * key_dim + tid];
        k_val = k_raw[kv_head * key_dim + tid];
    }

    // L2 norm for Q: compute sum of squares
    float q_ss = q_val * q_val;
    q_ss = simd_sum(q_ss);

    threadgroup float partial_sums_l2[8];
    if (simd_lane == 0) {
        partial_sums_l2[simd_group] = q_ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simd_groups = (tg_size + 31) / 32;
    threadgroup float total_l2;
    if (simd_group == 0) {
        float val = (simd_lane < num_simd_groups) ? partial_sums_l2[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            total_l2 = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float q_norm_val = sqrt(total_l2);
    float q_l2_scale = (q_norm_val > l2_eps) ? (1.0f / q_norm_val) : (1.0f / l2_eps);

    // L2 norm for K: compute sum of squares
    float k_ss = k_val * k_val;
    k_ss = simd_sum(k_ss);

    if (simd_lane == 0) {
        partial_sums_l2[simd_group] = k_ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        float val = (simd_lane < num_simd_groups) ? partial_sums_l2[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            total_l2 = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float k_norm_val = sqrt(total_l2);
    float k_l2_scale = (k_norm_val > l2_eps) ? (1.0f / k_norm_val) : (1.0f / l2_eps);

    // Store normalized Q and K in threadgroup memory
    if (tid < key_dim) {
        q_local[tid] = q_val * q_l2_scale;
        k_local[tid] = k_val * k_l2_scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Phase 1-3: GDN recurrence (same as gdn_state_output_norm, transposed layout) ---
    float a = alpha[h];
    float b = beta[h];
    uint vj = tid;
    float q_scale = rsqrt((float)key_dim);

    float my_out = 0.0f;
    if (vj < val_dim) {
        float v_val = v_tokens[h * val_dim + vj];
        device float* h_row = h_state + h * val_dim * key_dim + vj * key_dim;

        // Phase 1: Decay state, then retrieve from DECAYED state
        float retrieval = 0.0f;
        for (uint ki = 0; ki < key_dim; ki++) {
            float h_decayed = a * h_row[ki];
            h_row[ki] = h_decayed;
            retrieval += h_decayed * k_local[ki];
        }

        // Phase 2: Delta rule update + output
        float v_delta = b * (v_val - retrieval);
        for (uint ki = 0; ki < key_dim; ki++) {
            float h_updated = h_row[ki] + k_local[ki] * v_delta;
            h_row[ki] = h_updated;
            my_out += h_updated * q_local[ki] * q_scale;
        }
    }

    // --- Phase 4: RMSNorm + scale ---
    float ss = my_out * my_out;
    ss = simd_sum(ss);

    threadgroup float partial_sums[8];
    if (simd_lane == 0) {
        partial_sums[simd_group] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float total_ss;
    if (simd_group == 0) {
        float val = (simd_lane < num_simd_groups) ? partial_sums[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            total_ss = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms = sqrt(total_ss / (float)val_dim + eps);
    float inv_norm = 1.0f / rms;

    if (vj < val_dim) {
        device const float* head_scale = scale + (h % scale_n_heads) * val_dim;
        output[h * val_dim + vj] = my_out * inv_norm * head_scale[vj];
    }
}

// ============================================================================
// gdn_state_output_l2_sg: Simdgroup-parallel GDN state update
//
// Same algorithm as gdn_state_output_norm_l2 (L2 normalize Q/K + delta rule
// state update + output), but parallelized across (head, val_col) pairs using
// one simdgroup (32 threads) per pair. This increases occupancy from 32 TGs
// (0.53/core on M3 Ultra 60 cores) to 4096 TGs (68/core).
//
// Each simdgroup cooperates on key_dim=128 (4 elements per thread). All
// reductions use pure simd_sum with zero threadgroup barriers.
//
// Writes raw (un-normalized) output. RMSNorm is applied by a separate
// gdn_decode_norm_scale dispatch.
//
// buffer(0):  h_state    [n_heads * val_dim * key_dim] float -- R/W (transposed layout)
// buffer(1):  k_raw      [n_kv_heads * key_dim] float -- UN-normalized
// buffer(2):  v_tokens   [n_heads * val_dim] float  (n_heads V-heads, no GQA)
// buffer(3):  alpha      [n_heads] float -- decay factors
// buffer(4):  beta       [n_heads] float -- mixing rates
// buffer(5):  q_raw      [n_kv_heads * key_dim] float -- UN-normalized
// buffer(6):  raw_out    [n_heads * val_dim] float -- OUTPUT (raw, pre-norm)
// buffer(7):  n_heads (uint)
// buffer(8):  key_dim (uint)       -- 128
// buffer(9):  val_dim (uint)       -- 128
// buffer(10): n_kv_heads (uint)    -- 16
// buffer(11): l2_eps (float)       -- L2 norm eps (typically 1e-12)
//
// grid: (1, val_dim, n_heads), threadgroup: (32, 1, 1) -- 1 simdgroup per TG
// REQUIREMENT: key_dim == 128 (32 threads * 4 elements)
// ============================================================================

kernel void gdn_state_output_l2_sg(
    device       float* h_state        [[buffer(0)]],
    device const float* k_raw          [[buffer(1)]],
    device const float* v_tokens       [[buffer(2)]],
    device const float* alpha          [[buffer(3)]],
    device const float* beta           [[buffer(4)]],
    device const float* q_raw          [[buffer(5)]],
    device       float* raw_out        [[buffer(6)]],
    constant     uint&  n_heads        [[buffer(7)]],
    constant     uint&  key_dim        [[buffer(8)]],
    constant     uint&  val_dim        [[buffer(9)]],
    constant     uint&  n_kv_heads     [[buffer(10)]],
    constant     float& l2_eps         [[buffer(11)]],
    uint3 tgpig                        [[threadgroup_position_in_grid]],
    uint lane                          [[thread_index_in_simdgroup]])
{
    uint h  = tgpig.z;   // head index [0, n_heads)
    uint vi = tgpig.y;   // val_dim column [0, val_dim)
    if (h >= n_heads || vi >= val_dim) return;

    uint kv_head = h % n_kv_heads;
    float q_scale = rsqrt((float)key_dim);

    // --- Phase 0: L2 normalize Q and K using simd_sum ---
    // 32 threads * 4 elements = 128 = key_dim
    uint k_base = lane * 4;

    device const float* q_head = q_raw + kv_head * key_dim;
    device const float* k_head = k_raw + kv_head * key_dim;

    // Load 4 Q and K elements per thread
    float4 q_local = float4(q_head[k_base], q_head[k_base+1], q_head[k_base+2], q_head[k_base+3]);
    float4 k_local = float4(k_head[k_base], k_head[k_base+1], k_head[k_base+2], k_head[k_base+3]);

    // L2 norm Q
    float q_ss = q_local[0]*q_local[0] + q_local[1]*q_local[1] + q_local[2]*q_local[2] + q_local[3]*q_local[3];
    float q_total_ss = simd_sum(q_ss);
    float q_norm_val = sqrt(q_total_ss);
    float q_l2_scale = (q_norm_val > l2_eps) ? (1.0f / q_norm_val) : (1.0f / l2_eps);
    q_local *= q_l2_scale;

    // L2 norm K
    float k_ss = k_local[0]*k_local[0] + k_local[1]*k_local[1] + k_local[2]*k_local[2] + k_local[3]*k_local[3];
    float k_total_ss = simd_sum(k_ss);
    float k_norm_val = sqrt(k_total_ss);
    float k_l2_scale = (k_norm_val > l2_eps) ? (1.0f / k_norm_val) : (1.0f / l2_eps);
    k_local *= k_l2_scale;

    // --- Phase 1: Decay state + retrieval (from DECAYED state) ---
    float a = alpha[h];
    float b = beta[h];
    float v_val = v_tokens[h * val_dim + vi];

    // State row pointer: h_state[h, vi, ki] at h_row[ki] (transposed, coalesced)
    device float* h_row = h_state + h * val_dim * key_dim + vi * key_dim;

    // Load 4 state elements, apply decay (contiguous access!)
    float s0 = a * h_row[k_base + 0];
    float s1 = a * h_row[k_base + 1];
    float s2 = a * h_row[k_base + 2];
    float s3 = a * h_row[k_base + 3];

    // Write decayed state back
    h_row[k_base + 0] = s0;
    h_row[k_base + 1] = s1;
    h_row[k_base + 2] = s2;
    h_row[k_base + 3] = s3;

    // Compute retrieval = dot(h_decayed, k_normalized) across simd
    float ret_local = s0*k_local[0] + s1*k_local[1] + s2*k_local[2] + s3*k_local[3];
    float retrieval = simd_sum(ret_local);

    // --- Phase 2: Delta rule update + output ---
    float v_delta = b * (v_val - retrieval);

    s0 += k_local[0] * v_delta;
    s1 += k_local[1] * v_delta;
    s2 += k_local[2] * v_delta;
    s3 += k_local[3] * v_delta;

    // Write updated state back
    h_row[k_base + 0] = s0;
    h_row[k_base + 1] = s1;
    h_row[k_base + 2] = s2;
    h_row[k_base + 3] = s3;

    // Output = dot(h_updated, q_normalized * q_scale)
    float out_local = s0*q_local[0] + s1*q_local[1] + s2*q_local[2] + s3*q_local[3];
    float my_out = simd_sum(out_local) * q_scale;

    // Only lane 0 writes the output for this (head, val_col) pair
    if (lane == 0) {
        raw_out[h * val_dim + vi] = my_out;
    }
}

// ============================================================================
// gdn_decode_norm_scale: RMSNorm + learned scale on raw GDN decode output
//
// Applied after gdn_state_output_l2_sg to produce the final normed output.
// Each threadgroup handles one head: computes RMSNorm across val_dim elements,
// then applies learned scale weights.
//
// buffer(0): raw_out    [n_heads * val_dim] float -- INPUT
// buffer(1): scale      [scale_n_heads * val_dim] float -- learned norm scale
// buffer(2): output     [n_heads * val_dim] float -- OUTPUT
// buffer(3): n_heads (uint)
// buffer(4): val_dim (uint)
// buffer(5): eps (float)          -- RMSNorm eps
// buffer(6): scale_n_heads (uint)
//
// grid: (n_heads, 1, 1), threadgroup: (val_dim, 1, 1) where val_dim=128
// ============================================================================

kernel void gdn_decode_norm_scale(
    device const float* raw_out       [[buffer(0)]],
    device const float* scale         [[buffer(1)]],
    device       float* output        [[buffer(2)]],
    constant     uint&  n_heads       [[buffer(3)]],
    constant     uint&  val_dim       [[buffer(4)]],
    constant     float& eps           [[buffer(5)]],
    constant     uint&  scale_n_heads [[buffer(6)]],
    uint head_idx                     [[threadgroup_position_in_grid]],
    uint tid                          [[thread_index_in_threadgroup]],
    uint simd_lane                    [[thread_index_in_simdgroup]],
    uint simd_group                   [[simdgroup_index_in_threadgroup]])
{
    if (head_idx >= n_heads) return;

    uint h = head_idx;
    uint vj = tid;

    // Load raw output value for this (head, vj)
    float val = (vj < val_dim) ? raw_out[h * val_dim + vj] : 0.0f;

    // RMSNorm: compute sum of squares across val_dim
    float ss = val * val;
    ss = simd_sum(ss);

    // Cross-simdgroup reduction (val_dim=128 = 4 simdgroups of 32)
    threadgroup float partial[4];
    if (simd_lane == 0) {
        partial[simd_group] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_sg = (val_dim + 31) / 32;
    threadgroup float total_ss;
    if (simd_group == 0) {
        float v = (simd_lane < num_sg) ? partial[simd_lane] : 0.0f;
        v = simd_sum(v);
        if (simd_lane == 0) {
            total_ss = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (vj < val_dim) {
        float inv_norm = 1.0f / sqrt(total_ss / (float)val_dim + eps);
        device const float* head_scale = scale + (h % scale_n_heads) * val_dim;
        output[h * val_dim + vj] = val * inv_norm * head_scale[vj];
    }
}

// ============================================================================
// gdn_state_output_norm_l2_conv: Fused Conv1D+SiLU + L2-normalize Q/K +
//   state-update + output + RMSNorm
//
// Inlines the Conv1D+SiLU computation into the state kernel, eliminating the
// separate ssm_conv1d_silu_decode dispatch and one barrier per GDN layer.
// The kernel reads raw QKV (pre-conv) and conv_state, computes the convolution
// inline, applies SiLU, then proceeds with L2 normalization and GDN recurrence.
//
// Conv1D layout (matches ssm_conv1d_silu_decode exactly):
//   conv_state: [buf_slots * qkv_dim] float, indexed as [slot * qkv_dim + ch]
//   conv_weight: [qkv_dim * kernel_size] float, indexed as [ch * kernel_size + tap]
//   buf_slots = kernel_size - 1 = 3  (kernel_size = 4)
//   state_pos: current write position in circular buffer [0..buf_slots-1]
//   After conv, overwrites oldest: conv_state[state_pos * qkv_dim + ch] = input[ch]
//
// QKV channel layout (qkv_dim = 8192):
//   Q: [0 .. qk_dim)           = [0 .. 2048)    -- 16 kv_heads * 128
//   K: [qk_dim .. 2*qk_dim)    = [2048 .. 4096)  -- 16 kv_heads * 128
//   V: [2*qk_dim .. qkv_dim)   = [4096 .. 8192)  -- 32 heads * 128
//
// Thread mapping: 32 TGs (one per head), 128 threads per TG.
//   Thread tid in head h:
//     Q channel: (h % n_kv_heads) * key_dim + tid  (shared via GQA)
//     K channel: qk_dim + (h % n_kv_heads) * key_dim + tid  (shared via GQA)
//     V channel: 2*qk_dim + h * val_dim + tid  (unique per head)
//
// Conv_state updates: only heads with h < n_kv_heads write Q/K conv_state
// (avoiding duplicate writes from GQA-repeated heads). All heads write their
// unique V conv_state entries.
//
// buffer(0):  h_state      [n_heads * val_dim * key_dim] float -- R/W (transposed layout)
// buffer(1):  qkv_raw      [qkv_dim] float -- raw QKV from matvec (pre-conv)
// buffer(2):  conv_state   [buf_slots * qkv_dim] float -- circular buffer (R/W)
// buffer(3):  conv_weight  [qkv_dim * kernel_size] float -- conv weights
// buffer(4):  alpha        [n_heads] float -- decay factors
// buffer(5):  beta         [n_heads] float -- mixing rates
// buffer(6):  scale        [scale_n_heads * val_dim] float -- learned norm scale
// buffer(7):  output       [n_heads * val_dim] float -- OUTPUT
// buffer(8):  n_heads (uint)       -- state/V/output heads (= 32)
// buffer(9):  key_dim (uint)       -- 128
// buffer(10): val_dim (uint)       -- 128
// buffer(11): n_kv_heads (uint)    -- Q and K heads (= 16)
// buffer(12): eps (float)          -- RMSNorm eps
// buffer(13): scale_n_heads (uint)
// buffer(14): l2_eps (float)       -- L2 norm eps
// buffer(15): qk_dim (uint)       -- Q or K total dim (= n_kv_heads * key_dim = 2048)
// buffer(16): qkv_dim (uint)      -- total QKV dim (= 8192)
// buffer(17): kernel_size (uint)  -- conv kernel size (= 4)
// buffer(18): state_pos (uint)    -- current conv circular buffer position
//
// grid: (n_heads, 1, 1), threadgroup: (val_dim, 1, 1)  -- one thread per column
// REQUIREMENT: key_dim == val_dim (both 128 for Qwen3.5)
// ============================================================================

kernel void gdn_state_output_norm_l2_conv(
    device       float* h_state        [[buffer(0)]],
    device const float* qkv_raw        [[buffer(1)]],
    device       float* conv_state     [[buffer(2)]],
    device const float* conv_weight    [[buffer(3)]],
    device const float* alpha          [[buffer(4)]],
    device const float* beta           [[buffer(5)]],
    device const float* scale          [[buffer(6)]],
    device       float* output         [[buffer(7)]],
    constant     uint&  n_heads        [[buffer(8)]],
    constant     uint&  key_dim        [[buffer(9)]],
    constant     uint&  val_dim        [[buffer(10)]],
    constant     uint&  n_kv_heads     [[buffer(11)]],
    constant     float& eps            [[buffer(12)]],
    constant     uint&  scale_n_heads  [[buffer(13)]],
    constant     float& l2_eps         [[buffer(14)]],
    constant     uint&  qk_dim         [[buffer(15)]],
    constant     uint&  qkv_dim        [[buffer(16)]],
    constant     uint&  kernel_size    [[buffer(17)]],
    constant     uint&  state_pos      [[buffer(18)]],
    uint head_idx                      [[threadgroup_position_in_grid]],
    uint tid                           [[thread_index_in_threadgroup]],
    uint tg_size                       [[threads_per_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_group                    [[simdgroup_index_in_threadgroup]])
{
    if (head_idx >= n_heads) return;

    uint h = head_idx;
    uint kv_head = h % n_kv_heads;
    uint buf_slots = kernel_size - 1;

    // --- Phase -1: Inline Conv1D + SiLU for this thread's Q, K, V channels ---

    // Q channel conv1d + SiLU
    float q_val = 0.0f;
    if (tid < key_dim) {
        uint q_ch = kv_head * key_dim + tid;  // channel in [0..qk_dim)
        float q_sum = 0.0f;
        for (uint tap = 0; tap < buf_slots; tap++) {
            uint slot = (state_pos + tap) % buf_slots;
            q_sum += conv_weight[q_ch * kernel_size + tap] * conv_state[slot * qkv_dim + q_ch];
        }
        q_sum += conv_weight[q_ch * kernel_size + buf_slots] * qkv_raw[q_ch];
        q_val = q_sum / (1.0f + exp(-q_sum));  // SiLU

        // Only first n_kv_heads heads update Q conv_state (avoid GQA duplicate writes)
        if (h < n_kv_heads) {
            conv_state[state_pos * qkv_dim + q_ch] = qkv_raw[q_ch];
        }
    }

    // K channel conv1d + SiLU
    float k_val = 0.0f;
    if (tid < key_dim) {
        uint k_ch = qk_dim + kv_head * key_dim + tid;  // channel in [qk_dim..2*qk_dim)
        float k_sum = 0.0f;
        for (uint tap = 0; tap < buf_slots; tap++) {
            uint slot = (state_pos + tap) % buf_slots;
            k_sum += conv_weight[k_ch * kernel_size + tap] * conv_state[slot * qkv_dim + k_ch];
        }
        k_sum += conv_weight[k_ch * kernel_size + buf_slots] * qkv_raw[k_ch];
        k_val = k_sum / (1.0f + exp(-k_sum));  // SiLU

        // Only first n_kv_heads heads update K conv_state
        if (h < n_kv_heads) {
            conv_state[state_pos * qkv_dim + k_ch] = qkv_raw[k_ch];
        }
    }

    // V channel conv1d + SiLU (unique per head, no GQA sharing)
    float v_conv = 0.0f;
    if (tid < val_dim) {
        uint v_ch = 2 * qk_dim + h * val_dim + tid;  // channel in [2*qk_dim..qkv_dim)
        float v_sum = 0.0f;
        for (uint tap = 0; tap < buf_slots; tap++) {
            uint slot = (state_pos + tap) % buf_slots;
            v_sum += conv_weight[v_ch * kernel_size + tap] * conv_state[slot * qkv_dim + v_ch];
        }
        v_sum += conv_weight[v_ch * kernel_size + buf_slots] * qkv_raw[v_ch];
        v_conv = v_sum / (1.0f + exp(-v_sum));  // SiLU

        // All heads write their unique V conv_state entry
        conv_state[state_pos * qkv_dim + v_ch] = qkv_raw[v_ch];
    }

    // --- Phase 0: L2 normalize Q and K in threadgroup memory ---
    threadgroup float q_local[128];
    threadgroup float k_local[128];

    // L2 norm for Q: compute sum of squares
    float q_ss = q_val * q_val;
    q_ss = simd_sum(q_ss);

    threadgroup float partial_sums_l2[8];
    if (simd_lane == 0) {
        partial_sums_l2[simd_group] = q_ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simd_groups = (tg_size + 31) / 32;
    threadgroup float total_l2;
    if (simd_group == 0) {
        float val = (simd_lane < num_simd_groups) ? partial_sums_l2[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            total_l2 = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float q_norm_val = sqrt(total_l2);
    float q_l2_scale = (q_norm_val > l2_eps) ? (1.0f / q_norm_val) : (1.0f / l2_eps);

    // L2 norm for K: compute sum of squares
    float k_ss = k_val * k_val;
    k_ss = simd_sum(k_ss);

    if (simd_lane == 0) {
        partial_sums_l2[simd_group] = k_ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        float val = (simd_lane < num_simd_groups) ? partial_sums_l2[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            total_l2 = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float k_norm_val = sqrt(total_l2);
    float k_l2_scale = (k_norm_val > l2_eps) ? (1.0f / k_norm_val) : (1.0f / l2_eps);

    // Store normalized Q and K in threadgroup memory
    if (tid < key_dim) {
        q_local[tid] = q_val * q_l2_scale;
        k_local[tid] = k_val * k_l2_scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Phase 1-3: GDN recurrence (transposed layout) ---
    float a = alpha[h];
    float b = beta[h];
    uint vj = tid;
    float q_scale = rsqrt((float)key_dim);

    float my_out = 0.0f;
    if (vj < val_dim) {
        float v_val = v_conv;  // Use inline conv'd V value (already SiLU'd)
        device float* h_row = h_state + h * val_dim * key_dim + vj * key_dim;

        // Phase 1: Decay state, then retrieve from DECAYED state
        float retrieval = 0.0f;
        for (uint ki = 0; ki < key_dim; ki++) {
            float h_decayed = a * h_row[ki];
            h_row[ki] = h_decayed;
            retrieval += h_decayed * k_local[ki];
        }

        // Phase 2: Delta rule update + output
        float v_delta = b * (v_val - retrieval);
        for (uint ki = 0; ki < key_dim; ki++) {
            float h_updated = h_row[ki] + k_local[ki] * v_delta;
            h_row[ki] = h_updated;
            my_out += h_updated * q_local[ki] * q_scale;
        }
    }

    // --- Phase 4: RMSNorm + scale ---
    float ss = my_out * my_out;
    ss = simd_sum(ss);

    threadgroup float partial_sums[8];
    if (simd_lane == 0) {
        partial_sums[simd_group] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float total_ss;
    if (simd_group == 0) {
        float val = (simd_lane < num_simd_groups) ? partial_sums[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            total_ss = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms = sqrt(total_ss / (float)val_dim + eps);
    float inv_norm = 1.0f / rms;

    if (vj < val_dim) {
        device const float* head_scale = scale + (h % scale_n_heads) * val_dim;
        output[h * val_dim + vj] = my_out * inv_norm * head_scale[vj];
    }
}

// ============================================================================
// Batched GDN prefill kernels
//
// These kernels process T tokens in a single dispatch, eliminating per-token
// kernel launch overhead during prefill. All T tokens' Q, K, V, alpha, beta
// are pre-computed by batched GEMMs before these kernels run.
// ============================================================================

// ============================================================================
// gdn_prefill_state_output_norm: Multi-token GDN state update + output + RMSNorm
//
// Processes T tokens sequentially (recurrent dependency), but:
// - All T tokens' Q, K, V, alpha, beta are PRE-COMPUTED by batched GEMMs
// - Single kernel launch eliminates per-token dispatch overhead
// - RMSNorm cooperative reduction per token via threadgroup barrier
//
// buffer(0):  h_state      [n_heads * val_dim * key_dim] float -- R/W persistent (transposed)
// buffer(1):  k_norm_all   [T * n_kv_heads * key_dim] float
// buffer(2):  v_all        [T * n_heads * val_dim] float
// buffer(3):  alpha_all    [T * n_heads] float
// buffer(4):  beta_all     [T * n_heads] float
// buffer(5):  q_norm_all   [T * n_kv_heads * key_dim] float
// buffer(6):  scale        [scale_n_heads * val_dim] float
// buffer(7):  output_all   [T * n_heads * val_dim] float -- OUTPUT
// buffer(8):  n_heads (uint)       -- 32
// buffer(9):  key_dim (uint)       -- 128
// buffer(10): val_dim (uint)       -- 128
// buffer(11): n_kv_heads (uint)    -- 16
// buffer(12): eps (float)
// buffer(13): scale_n_heads (uint)
// buffer(14): T (uint)             -- number of tokens
//
// grid: (n_heads, 1, 1) = (32, 1, 1)
// threadgroup: (val_dim, 1, 1) = (128, 1, 1)
// ============================================================================

kernel void gdn_prefill_state_output_norm(
    device       float* h_state        [[buffer(0)]],
    device const float* k_norm_all     [[buffer(1)]],
    device const float* v_all          [[buffer(2)]],
    device const float* alpha_all      [[buffer(3)]],
    device const float* beta_all       [[buffer(4)]],
    device const float* q_norm_all     [[buffer(5)]],
    device const float* scale          [[buffer(6)]],
    device       float* output_all     [[buffer(7)]],
    constant     uint&  n_heads        [[buffer(8)]],
    constant     uint&  key_dim        [[buffer(9)]],
    constant     uint&  val_dim        [[buffer(10)]],
    constant     uint&  n_kv_heads     [[buffer(11)]],
    constant     float& eps            [[buffer(12)]],
    constant     uint&  scale_n_heads  [[buffer(13)]],
    constant     uint&  T              [[buffer(14)]],
    uint head_idx                      [[threadgroup_position_in_grid]],
    uint tid                           [[thread_index_in_threadgroup]],
    uint tg_size                       [[threads_per_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_group                    [[simdgroup_index_in_threadgroup]])
{
    if (head_idx >= n_heads) return;

    uint h = head_idx;
    // V has n_heads=32 heads. Q and K have n_kv_heads=16 heads.
    // GGUF stores V-heads de-interleaved: heads 0..15 = MLX even heads,
    // heads 16..31 = MLX odd heads. Correct GQA: kv_head = h % n_kv_heads
    uint kv_head = h % n_kv_heads;

    // Q scaling factor: 1/sqrt(key_dim)
    float q_scale = rsqrt((float)key_dim);

    uint vj = tid;

    // Pointer to this head's state row (contiguous key_dim, transposed layout)
    device float* h_row = (vj < val_dim) ? (h_state + h * val_dim * key_dim + vj * key_dim) : nullptr;

    for (uint t = 0; t < T; t++) {
        // Read per-token gating values
        float a = alpha_all[t * n_heads + h];
        float b = beta_all[t * n_heads + h];

        // Per-token K and Q pointers (GQA: use kv_head)
        device const float* k_head = k_norm_all + t * n_kv_heads * key_dim + kv_head * key_dim;
        device const float* q_head = q_norm_all + t * n_kv_heads * key_dim + kv_head * key_dim;

        // GDN recurrence per column vj (transposed layout):
        //   1. s_decayed = alpha * s_old          (decay FIRST)
        //   2. retrieval = s_decayed^T @ k        (retrieve from DECAYED state)
        //   3. delta = beta * (v - retrieval)
        //   4. s_new = s_decayed + outer(k, delta)
        //   5. output = s_new @ (q * scale)       (scale = 1/sqrt(key_dim))
        float my_out = 0.0f;
        if (vj < val_dim) {
            // V has n_heads heads (no GQA) -- use h directly
            float v_val = v_all[t * n_heads * val_dim + h * val_dim + vj];

            // Phase 1: Decay state, then retrieve from DECAYED state
            float retrieval = 0.0f;
            for (uint ki = 0; ki < key_dim; ki++) {
                float h_decayed = a * h_row[ki];
                h_row[ki] = h_decayed;  // write decayed state back
                retrieval += h_decayed * k_head[ki];
            }

            // Phase 2: Delta rule update + output (on already-decayed state)
            float v_delta = b * (v_val - retrieval);
            for (uint ki = 0; ki < key_dim; ki++) {
                float h_updated = h_row[ki] + k_head[ki] * v_delta;
                h_row[ki] = h_updated;
                my_out += h_updated * q_head[ki] * q_scale;
            }
        }

        // Phase 3: RMSNorm cooperative reduction across threadgroup
        // x / sqrt(mean(x^2) + eps) * weight
        float ss = my_out * my_out;
        ss = simd_sum(ss);

        threadgroup float partial_sums[8];
        if (simd_lane == 0) {
            partial_sums[simd_group] = ss;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint num_simd_groups = (tg_size + 31) / 32;
        threadgroup float total_ss;
        if (simd_group == 0) {
            float val = (simd_lane < num_simd_groups) ? partial_sums[simd_lane] : 0.0f;
            val = simd_sum(val);
            if (simd_lane == 0) {
                total_ss = val;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float rms = sqrt(total_ss / (float)val_dim + eps);
        float inv_norm = 1.0f / rms;

        // Write normalized + scaled output for this token
        if (vj < val_dim) {
            device const float* head_scale = scale + (h % scale_n_heads) * val_dim;
            output_all[t * n_heads * val_dim + h * val_dim + vj] = my_out * inv_norm * head_scale[vj];
        }
    }
}
// gdn_prefill_fused: Fused GDN prefill kernel with register-resident state
//
// Critical optimization: keeps the h_state matrix in registers instead of
// reading/writing from device memory for each token. This eliminates
// T * 2MB * 2 (R+W) = ~504MB of state memory traffic per layer (T=126).
//
// Also fuses: compute_gates (alpha/beta transformation) + L2 normalize Q/K
// + state update + output projection + RMSNorm + SiLU mul (output gating)
//
// Input layout (all pre-computed by batched matvec):
//   conv_out:  [T, qkv_dim] float  -- SiLU-activated conv1d output
//              Q at [0..qk_dim), K at [qk_dim..2*qk_dim), V at [2*qk_dim..2*qk_dim+value_dim)
//   alpha_raw: [T, n_heads] float  -- raw alpha projection (before compute_gates)
//   beta_raw:  [T, n_heads] float  -- raw beta projection (before compute_gates)
//   gate:      [T, q_dim] float    -- output gate for SiLU mul
//
// buffer(0):  h_state       [n_heads * val_dim * key_dim] float -- R/W persistent (transposed)
// buffer(1):  conv_out_all  [T * (qk_dim + qk_dim + value_dim)] float
// buffer(2):  alpha_raw_all [T * n_heads] float
// buffer(3):  beta_raw_all  [T * n_heads] float
// buffer(4):  gate_all      [T * q_dim] float (output gate)
// buffer(5):  dt_bias       [n_heads] float
// buffer(6):  A_log         [n_heads] float
// buffer(7):  norm_scale    [scale_n_heads * val_dim] float
// buffer(8):  ssm_out       [T * n_heads * val_dim] float -- OUTPUT (after RMSNorm + SiLU mul)
// buffer(9):  n_heads (uint)       -- 32
// buffer(10): key_dim (uint)       -- 128
// buffer(11): val_dim (uint)       -- 128
// buffer(12): n_kv_heads (uint)    -- 16
// buffer(13): eps (float)
// buffer(14): scale_n_heads (uint)
// buffer(15): T (uint)
// buffer(16): qk_dim (uint)        -- 2048
// buffer(17): qkv_dim (uint)       -- 8192 (total Q+K+V dim)
//
// grid: (n_heads, 1, 1) = (32, 1, 1)
// threadgroup: (val_dim, 1, 1) = (128, 1, 1)
// ============================================================================

kernel void gdn_prefill_fused(
    device       float* h_state        [[buffer(0)]],
    device const float* conv_out_all   [[buffer(1)]],
    device const float* alpha_raw_all  [[buffer(2)]],
    device const float* beta_raw_all   [[buffer(3)]],
    device const float* gate_all       [[buffer(4)]],
    device const float* dt_bias        [[buffer(5)]],
    device const float* A_log          [[buffer(6)]],
    device const float* norm_scale     [[buffer(7)]],
    device       float* ssm_out        [[buffer(8)]],
    constant     uint&  n_heads        [[buffer(9)]],
    constant     uint&  key_dim        [[buffer(10)]],
    constant     uint&  val_dim        [[buffer(11)]],
    constant     uint&  n_kv_heads     [[buffer(12)]],
    constant     float& eps            [[buffer(13)]],
    constant     uint&  scale_n_heads  [[buffer(14)]],
    constant     uint&  T              [[buffer(15)]],
    constant     uint&  qk_dim         [[buffer(16)]],
    constant     uint&  qkv_dim        [[buffer(17)]],
    uint head_idx                      [[threadgroup_position_in_grid]],
    uint tid                           [[thread_index_in_threadgroup]],
    uint tg_size                       [[threads_per_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_group                    [[simdgroup_index_in_threadgroup]])
{
    if (head_idx >= n_heads) return;

    uint h = head_idx;
    uint kv_head = h % n_kv_heads;
    float q_scale = rsqrt((float)key_dim);
    uint vj = tid;

    // Load state into registers (128 floats per thread, transposed layout)
    float state[128];  // key_dim=128 elements per val_dim row
    if (vj < val_dim) {
        device const float* h_row = h_state + h * val_dim * key_dim + vj * key_dim;
        for (uint ki = 0; ki < key_dim; ki++) {
            state[ki] = h_row[ki];
        }
    }

    // Pre-compute per-head gate transformation constants
    float dt_b = dt_bias[h];
    // GGUF stores ssm_a = -exp(A_log), already negated+exponentiated.
    // Typical values: ssm_a ~ -0.036 (slow decay) to -72 (fast decay).
    float ssm_a_val = A_log[h];

    for (uint t = 0; t < T; t++) {
        // --- Compute gates (fused, matching gdn_compute_gates kernel) ---
        float alpha_raw = alpha_raw_all[t * n_heads + h];
        float beta_raw = beta_raw_all[t * n_heads + h];

        // softplus(x) = log(1 + exp(x)), with numerical stability for large x
        float sp_input = alpha_raw + dt_b;
        float sp;
        if (sp_input > 20.0f) {
            sp = sp_input;
        } else {
            sp = log(1.0f + exp(sp_input));
        }
        // gate = ssm_a * softplus(alpha_raw + dt_bias)
        // ssm_a is negative, so gate is negative, alpha = exp(gate) in (0,1)
        float a = exp(ssm_a_val * sp);

        // beta = sigmoid(beta_raw)
        float b = 1.0f / (1.0f + exp(-beta_raw));

        // --- L2 normalize Q and K (fused, avoids separate dispatch) ---
        // Each thread contributes to the norm computation for its simdgroup's head.
        // Q is at conv_out_all[t * qkv_dim + 0..qk_dim)
        // K is at conv_out_all[t * qkv_dim + qk_dim..2*qk_dim)
        device const float* conv_t = conv_out_all + t * qkv_dim;
        device const float* q_raw = conv_t + kv_head * key_dim;
        device const float* k_raw = conv_t + qk_dim + kv_head * key_dim;

        // Compute L2 norms using simd reduction across key_dim
        // We need a threadgroup-level approach since key_dim=128 = 4 simdgroups
        // But each thread handles one vj, not one ki. We need to read Q and K
        // values in a cooperative manner.
        //
        // Actually, the current approach reads Q[ki] and K[ki] in the inner loop.
        // We can compute the L2 norm inline during the first pass.

        // First pass: compute L2 norms of Q and K heads (cooperative)
        // Each thread reads a portion of the Q/K vectors and contributes to the norm
        float q_val = (vj < key_dim) ? q_raw[vj] : 0.0f;
        float k_val = (vj < key_dim) ? k_raw[vj] : 0.0f;
        float q_sq = q_val * q_val;
        float k_sq = k_val * k_val;

        // Reduce across threadgroup to get full norms
        q_sq = simd_sum(q_sq);
        k_sq = simd_sum(k_sq);

        threadgroup float tg_q_sq[4];
        threadgroup float tg_k_sq[4];
        if (simd_lane == 0) {
            tg_q_sq[simd_group] = q_sq;
            tg_k_sq[simd_group] = k_sq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint num_sg = (tg_size + 31) / 32;
        threadgroup float tg_q_norm;
        threadgroup float tg_k_norm;
        if (simd_group == 0) {
            float qv = (simd_lane < num_sg) ? tg_q_sq[simd_lane] : 0.0f;
            float kv = (simd_lane < num_sg) ? tg_k_sq[simd_lane] : 0.0f;
            qv = simd_sum(qv);
            kv = simd_sum(kv);
            if (simd_lane == 0) {
                float l2_eps = 1e-12f;
                tg_q_norm = max(sqrt(qv), l2_eps);
                tg_k_norm = max(sqrt(kv), l2_eps);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float q_inv_norm = 1.0f / tg_q_norm;
        float k_inv_norm = 1.0f / tg_k_norm;

        // --- GDN recurrence with register-resident state ---
        float my_out = 0.0f;
        if (vj < val_dim) {
            // V value for this head and token
            float v_val = conv_t[qk_dim + qk_dim + h * val_dim + vj];

            // Phase 1: Decay state, then retrieve from DECAYED state
            float retrieval = 0.0f;
            for (uint ki = 0; ki < key_dim; ki++) {
                float h_decayed = a * state[ki];
                state[ki] = h_decayed;
                float k_norm_ki = k_raw[ki] * k_inv_norm;
                retrieval += h_decayed * k_norm_ki;
            }

            // Phase 2: Delta rule update + output projection
            float v_delta = b * (v_val - retrieval);
            for (uint ki = 0; ki < key_dim; ki++) {
                float k_norm_ki = k_raw[ki] * k_inv_norm;
                float h_updated = state[ki] + k_norm_ki * v_delta;
                state[ki] = h_updated;
                float q_norm_ki = q_raw[ki] * q_inv_norm;
                my_out += h_updated * q_norm_ki * q_scale;
            }
        }

        // Phase 3: RMSNorm cooperative reduction
        float ss = my_out * my_out;
        ss = simd_sum(ss);

        threadgroup float partial_sums[4];
        if (simd_lane == 0) {
            partial_sums[simd_group] = ss;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float total_ss;
        if (simd_group == 0) {
            float val = (simd_lane < num_sg) ? partial_sums[simd_lane] : 0.0f;
            val = simd_sum(val);
            if (simd_lane == 0) {
                total_ss = val;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float rms = sqrt(total_ss / (float)val_dim + eps);
        float inv_norm = 1.0f / rms;

        // Phase 4: Write SiLU-gated, normalized output
        // silu_mul: silu(gate) * normed_out = (gate * sigmoid(gate)) * (out * inv_norm * scale)
        if (vj < val_dim) {
            device const float* head_scale = norm_scale + (h % scale_n_heads) * val_dim;
            float normed = my_out * inv_norm * head_scale[vj];

            // Output gating: gate has q_dim=4096 elements, indexed by [h * val_dim + vj]
            float gate_val = gate_all[t * n_heads * val_dim + h * val_dim + vj];
            float silu_gate = gate_val / (1.0f + exp(-gate_val));  // silu(x) = x * sigmoid(x)

            ssm_out[t * n_heads * val_dim + h * val_dim + vj] = silu_gate * normed;
        }
    }

    // Write state back to device memory (transposed layout)
    if (vj < val_dim) {
        device float* h_row = h_state + h * val_dim * key_dim + vj * key_dim;
        for (uint ki = 0; ki < key_dim; ki++) {
            h_row[ki] = state[ki];
        }
    }
}

// ============================================================================
// gdn_prefill_fused_v2: Simdgroup-parallel GDN prefill kernel
//
// Key optimization: each thread holds only 4 state floats instead of 128.
// This reduces register pressure by 32x vs. gdn_prefill_fused.
//
// Thread mapping:
//   - Each threadgroup = 1 simdgroup = 32 threads
//   - Each threadgroup handles one (head, val_dim_index) pair
//   - 32 threads cooperate on key_dim=128 (4 elements each)
//   - simd_sum for all reductions across key_dim
//   - No threadgroup barriers (pure simd operations)
//
// Grid: (1, val_dim=128, n_heads=32) threadgroups
// Threadgroup: (32, 1, 1) threads
//
// Output: raw_out [T * n_heads * val_dim] float (before RMSNorm)
// RMSNorm + SiLU gating done by a separate gdn_prefill_norm_gate kernel.
//
// buffer(0):  h_state       [n_heads * val_dim * key_dim] float -- R/W persistent (transposed)
// buffer(1):  conv_out_all  [T * qkv_dim] float (SiLU-activated, Q/K L2-normalized)
// buffer(2):  alpha_all     [T * n_heads] float -- PRECOMPUTED: exp(gate)
// buffer(3):  beta_all      [T * n_heads] float -- PRECOMPUTED: sigmoid(beta_raw)
// buffer(4):  dt_bias       [n_heads] float     -- UNUSED (gates precomputed)
// buffer(5):  A_log         [n_heads] float     -- UNUSED (gates precomputed)
// buffer(6):  raw_out       [T * n_heads * val_dim] float -- OUTPUT (raw, before RMSNorm)
// buffer(7):  n_heads (uint)       -- 32
// buffer(8):  key_dim (uint)       -- 128
// buffer(9):  val_dim (uint)       -- 128
// buffer(10): n_kv_heads (uint)    -- 16
// buffer(11): T (uint)
// buffer(12): qk_dim (uint)        -- 2048
// buffer(13): qkv_dim (uint)       -- 8192
// ============================================================================

kernel void gdn_prefill_fused_v2(
    device       float* h_state        [[buffer(0)]],
    device const float* conv_out_all   [[buffer(1)]],
    device const float* alpha_all      [[buffer(2)]],
    device const float* beta_all       [[buffer(3)]],
    device const float* dt_bias        [[buffer(4)]],
    device const float* A_log          [[buffer(5)]],
    device       float* raw_out        [[buffer(6)]],
    constant     uint&  n_heads        [[buffer(7)]],
    constant     uint&  key_dim        [[buffer(8)]],
    constant     uint&  val_dim        [[buffer(9)]],
    constant     uint&  n_kv_heads     [[buffer(10)]],
    constant     uint&  T              [[buffer(11)]],
    constant     uint&  qk_dim         [[buffer(12)]],
    constant     uint&  qkv_dim        [[buffer(13)]],
    uint3 tg_pos                       [[threadgroup_position_in_grid]],
    uint lane                          [[thread_index_in_simdgroup]])
{
    // tg_pos.y = val_dim index (vj), tg_pos.z = head index (h)
    uint vj = tg_pos.y;
    uint h  = tg_pos.z;
    if (h >= n_heads || vj >= val_dim) return;

    uint kv_head = h % n_kv_heads;
    float q_scale = rsqrt((float)key_dim);

    // Each thread handles 4 key_dim elements: lane*4 .. lane*4+3
    uint k_base = lane * 4;

    // Load 4 state elements into registers (transposed layout, contiguous access)
    device float* h_row = h_state + h * val_dim * key_dim + vj * key_dim;
    float s0 = h_row[k_base + 0];
    float s1 = h_row[k_base + 1];
    float s2 = h_row[k_base + 2];
    float s3 = h_row[k_base + 3];

    for (uint t = 0; t < T; t++) {
        // --- Read precomputed gates ---
        float a = alpha_all[t * n_heads + h];
        float b = beta_all[t * n_heads + h];

        // --- Read pre-normalized Q and K (L2 normalization done by l2_normalize_qk_strided) ---
        device const float* conv_t = conv_out_all + t * qkv_dim;
        device const float* q_head = conv_t + kv_head * key_dim;
        device const float* k_head = conv_t + qk_dim + kv_head * key_dim;

        // Each thread loads 4 pre-normalized Q and K values
        float qn0 = q_head[k_base + 0];
        float qn1 = q_head[k_base + 1];
        float qn2 = q_head[k_base + 2];
        float qn3 = q_head[k_base + 3];

        float kn0 = k_head[k_base + 0];
        float kn1 = k_head[k_base + 1];
        float kn2 = k_head[k_base + 2];
        float kn3 = k_head[k_base + 3];

        // --- V value for this head and vj ---
        float v_val = conv_t[qk_dim + qk_dim + h * val_dim + vj];

        // --- Phase 1: Decay state + retrieval ---
        float d0 = a * s0;
        float d1 = a * s1;
        float d2 = a * s2;
        float d3 = a * s3;

        // retrieval = dot(h_decayed, k_norm) -- reduce across simd
        float ret_local = d0*kn0 + d1*kn1 + d2*kn2 + d3*kn3;
        float retrieval = simd_sum(ret_local);

        // --- Phase 2: Delta update + output projection ---
        float v_delta = b * (v_val - retrieval);

        s0 = d0 + kn0 * v_delta;
        s1 = d1 + kn1 * v_delta;
        s2 = d2 + kn2 * v_delta;
        s3 = d3 + kn3 * v_delta;

        // output = dot(h_updated, q_norm * q_scale) -- reduce across simd
        float out_local = s0*qn0 + s1*qn1 + s2*qn2 + s3*qn3;
        float my_out = simd_sum(out_local) * q_scale;

        // Write raw output (only lane 0 writes to avoid races)
        if (lane == 0) {
            raw_out[t * n_heads * val_dim + h * val_dim + vj] = my_out;
        }
    }

    // Write state back to device memory (transposed layout, contiguous)
    h_row[k_base + 0] = s0;
    h_row[k_base + 1] = s1;
    h_row[k_base + 2] = s2;
    h_row[k_base + 3] = s3;
}

// ============================================================================
// gdn_prefill_fused_v3_chunked: Loop-unrolled GDN prefill kernel
//
// Identical algorithm to gdn_prefill_fused_v2 but with C=4 loop unrolling.
// Prefetches alpha/beta/Q/K/V for 4 consecutive tokens into registers before
// processing the recurrence steps, enabling memory latency hiding and reducing
// loop overhead by 4x.
//
// Thread mapping and grid are identical to v2:
//   - Each threadgroup = 1 simdgroup = 32 threads
//   - Each threadgroup handles one (head, val_dim_index) pair
//   - 32 threads cooperate on key_dim=128 (4 elements each)
//   - simd_sum for all reductions across key_dim
//
// Grid: (1, val_dim=128, n_heads=32) threadgroups
// Threadgroup: (32, 1, 1) threads
//
// Buffers: alpha/beta are pre-computed gates (not raw projections)
// ============================================================================

kernel void gdn_prefill_fused_v3_chunked(
    device       float* h_state        [[buffer(0)]],
    device const float* conv_out_all   [[buffer(1)]],
    device const float* alpha_all      [[buffer(2)]],
    device const float* beta_all       [[buffer(3)]],
    device       float* raw_out        [[buffer(4)]],
    constant     uint&  n_heads        [[buffer(5)]],
    constant     uint&  key_dim        [[buffer(6)]],
    constant     uint&  val_dim        [[buffer(7)]],
    constant     uint&  n_kv_heads     [[buffer(8)]],
    constant     uint&  T              [[buffer(9)]],
    constant     uint&  qk_dim         [[buffer(10)]],
    constant     uint&  qkv_dim        [[buffer(11)]],
    uint3 tg_pos                       [[threadgroup_position_in_grid]],
    uint lane                          [[thread_index_in_simdgroup]])
{
    uint vj = tg_pos.y;
    uint h  = tg_pos.z;
    if (h >= n_heads || vj >= val_dim) return;

    uint kv_head = h % n_kv_heads;
    float q_scale = rsqrt((float)key_dim);

    // Each thread handles 4 key_dim elements: lane*4 .. lane*4+3
    uint k_base = lane * 4;

    // Load 4 state elements into registers (transposed layout, contiguous access)
    device float* h_row = h_state + h * val_dim * key_dim + vj * key_dim;
    float s0 = h_row[k_base + 0];
    float s1 = h_row[k_base + 1];
    float s2 = h_row[k_base + 2];
    float s3 = h_row[k_base + 3];

    // Precomputed offsets for Q/K/V indexing within conv_out_all
    uint q_head_off = kv_head * key_dim + k_base;
    uint k_head_off = qk_dim + kv_head * key_dim + k_base;
    uint v_head_off = qk_dim + qk_dim + h * val_dim + vj;
    uint out_stride = n_heads * val_dim;
    uint out_base = h * val_dim + vj;

    // Process T tokens in chunks of 4
    uint t = 0;
    uint T_aligned = T & ~3u;  // T rounded down to multiple of 4

    for (; t < T_aligned; t += 4) {
        // --- Prefetch all data for 4 tokens into registers ---
        device const float* c0 = conv_out_all + t * qkv_dim;
        device const float* c1 = c0 + qkv_dim;
        device const float* c2 = c1 + qkv_dim;
        device const float* c3 = c2 + qkv_dim;

        float a_0 = alpha_all[t * n_heads + h];
        float a_1 = alpha_all[(t+1) * n_heads + h];
        float a_2 = alpha_all[(t+2) * n_heads + h];
        float a_3 = alpha_all[(t+3) * n_heads + h];

        float b_0 = beta_all[t * n_heads + h];
        float b_1 = beta_all[(t+1) * n_heads + h];
        float b_2 = beta_all[(t+2) * n_heads + h];
        float b_3 = beta_all[(t+3) * n_heads + h];

        // Q values for 4 tokens (4 key elements each)
        float q0_0 = c0[q_head_off];     float q0_1 = c0[q_head_off+1];
        float q0_2 = c0[q_head_off+2];   float q0_3 = c0[q_head_off+3];
        float q1_0 = c1[q_head_off];     float q1_1 = c1[q_head_off+1];
        float q1_2 = c1[q_head_off+2];   float q1_3 = c1[q_head_off+3];
        float q2_0 = c2[q_head_off];     float q2_1 = c2[q_head_off+1];
        float q2_2 = c2[q_head_off+2];   float q2_3 = c2[q_head_off+3];
        float q3_0 = c3[q_head_off];     float q3_1 = c3[q_head_off+1];
        float q3_2 = c3[q_head_off+2];   float q3_3 = c3[q_head_off+3];

        // K values for 4 tokens
        float k0_0 = c0[k_head_off];     float k0_1 = c0[k_head_off+1];
        float k0_2 = c0[k_head_off+2];   float k0_3 = c0[k_head_off+3];
        float k1_0 = c1[k_head_off];     float k1_1 = c1[k_head_off+1];
        float k1_2 = c1[k_head_off+2];   float k1_3 = c1[k_head_off+3];
        float k2_0 = c2[k_head_off];     float k2_1 = c2[k_head_off+1];
        float k2_2 = c2[k_head_off+2];   float k2_3 = c2[k_head_off+3];
        float k3_0 = c3[k_head_off];     float k3_1 = c3[k_head_off+1];
        float k3_2 = c3[k_head_off+2];   float k3_3 = c3[k_head_off+3];

        // V values for 4 tokens
        float v0 = c0[v_head_off];
        float v1 = c1[v_head_off];
        float v2 = c2[v_head_off];
        float v3_v = c3[v_head_off];

        // --- Token 0: recurrence step ---
        {
            float d0 = a_0 * s0;  float d1 = a_0 * s1;
            float d2 = a_0 * s2;  float d3 = a_0 * s3;
            float retrieval = simd_sum(d0*k0_0 + d1*k0_1 + d2*k0_2 + d3*k0_3);
            float v_delta = b_0 * (v0 - retrieval);
            s0 = d0 + k0_0 * v_delta;  s1 = d1 + k0_1 * v_delta;
            s2 = d2 + k0_2 * v_delta;  s3 = d3 + k0_3 * v_delta;
            float my_out = simd_sum(s0*q0_0 + s1*q0_1 + s2*q0_2 + s3*q0_3) * q_scale;
            if (lane == 0) raw_out[t * out_stride + out_base] = my_out;
        }

        // --- Token 1: recurrence step ---
        {
            float d0 = a_1 * s0;  float d1 = a_1 * s1;
            float d2 = a_1 * s2;  float d3 = a_1 * s3;
            float retrieval = simd_sum(d0*k1_0 + d1*k1_1 + d2*k1_2 + d3*k1_3);
            float v_delta = b_1 * (v1 - retrieval);
            s0 = d0 + k1_0 * v_delta;  s1 = d1 + k1_1 * v_delta;
            s2 = d2 + k1_2 * v_delta;  s3 = d3 + k1_3 * v_delta;
            float my_out = simd_sum(s0*q1_0 + s1*q1_1 + s2*q1_2 + s3*q1_3) * q_scale;
            if (lane == 0) raw_out[(t+1) * out_stride + out_base] = my_out;
        }

        // --- Token 2: recurrence step ---
        {
            float d0 = a_2 * s0;  float d1 = a_2 * s1;
            float d2 = a_2 * s2;  float d3 = a_2 * s3;
            float retrieval = simd_sum(d0*k2_0 + d1*k2_1 + d2*k2_2 + d3*k2_3);
            float v_delta = b_2 * (v2 - retrieval);
            s0 = d0 + k2_0 * v_delta;  s1 = d1 + k2_1 * v_delta;
            s2 = d2 + k2_2 * v_delta;  s3 = d3 + k2_3 * v_delta;
            float my_out = simd_sum(s0*q2_0 + s1*q2_1 + s2*q2_2 + s3*q2_3) * q_scale;
            if (lane == 0) raw_out[(t+2) * out_stride + out_base] = my_out;
        }

        // --- Token 3: recurrence step ---
        {
            float d0 = a_3 * s0;  float d1 = a_3 * s1;
            float d2 = a_3 * s2;  float d3 = a_3 * s3;
            float retrieval = simd_sum(d0*k3_0 + d1*k3_1 + d2*k3_2 + d3*k3_3);
            float v_delta = b_3 * (v3_v - retrieval);
            s0 = d0 + k3_0 * v_delta;  s1 = d1 + k3_1 * v_delta;
            s2 = d2 + k3_2 * v_delta;  s3 = d3 + k3_3 * v_delta;
            float my_out = simd_sum(s0*q3_0 + s1*q3_1 + s2*q3_2 + s3*q3_3) * q_scale;
            if (lane == 0) raw_out[(t+3) * out_stride + out_base] = my_out;
        }
    }

    // --- Tail: process remaining 0-3 tokens one at a time ---
    for (; t < T; t++) {
        float a = alpha_all[t * n_heads + h];
        float b = beta_all[t * n_heads + h];

        device const float* conv_t = conv_out_all + t * qkv_dim;

        float qn0 = conv_t[q_head_off];     float qn1 = conv_t[q_head_off+1];
        float qn2 = conv_t[q_head_off+2];   float qn3 = conv_t[q_head_off+3];
        float kn0 = conv_t[k_head_off];     float kn1 = conv_t[k_head_off+1];
        float kn2 = conv_t[k_head_off+2];   float kn3 = conv_t[k_head_off+3];
        float v_val = conv_t[v_head_off];

        float d0 = a * s0;  float d1 = a * s1;
        float d2 = a * s2;  float d3 = a * s3;
        float retrieval = simd_sum(d0*kn0 + d1*kn1 + d2*kn2 + d3*kn3);
        float v_delta = b * (v_val - retrieval);
        s0 = d0 + kn0 * v_delta;  s1 = d1 + kn1 * v_delta;
        s2 = d2 + kn2 * v_delta;  s3 = d3 + kn3 * v_delta;
        float my_out = simd_sum(s0*qn0 + s1*qn1 + s2*qn2 + s3*qn3) * q_scale;
        if (lane == 0) raw_out[t * out_stride + out_base] = my_out;
    }

    // Write state back to device memory (transposed layout, contiguous)
    h_row[k_base + 0] = s0;
    h_row[k_base + 1] = s1;
    h_row[k_base + 2] = s2;
    h_row[k_base + 3] = s3;
}

// ============================================================================
// gdn_prefill_norm_gate: RMSNorm + SiLU gate on raw GDN output
//
// Applies per-head RMSNorm + scale + SiLU output gating.
// Designed as post-processing for gdn_prefill_fused_v3_chunked.
//
// This kernel CANNOT be fused into gdn_prefill_fused_v3_chunked.
// The state kernel is grid (1, val_dim=128, n_heads=32) -- each TG owns one vj.
// RMSNorm needs sum-of-squares across all 128 vj values for a given (head, token),
// which requires cross-TG reduction. Metal has no cross-TG sync within a dispatch.
// Restructuring to coalesce all vj into one TG would require 64KB state per TG,
// destroying occupancy. The separate dispatch + barrier is architecturally required.
//
// buffer(0): raw_out   [T * n_heads * val_dim] float -- INPUT (raw output)
// buffer(1): gate_all  [T * q_dim] float             -- SiLU gate
// buffer(2): norm_scale [scale_n_heads * val_dim] float
// buffer(3): ssm_out   [T * n_heads * val_dim] float -- OUTPUT
// buffer(4): n_heads (uint)
// buffer(5): val_dim (uint)
// buffer(6): eps (float)
// buffer(7): scale_n_heads (uint)
// buffer(8): T (uint)
//
// grid: (n_heads, T, 1), threadgroup: (128, 1, 1) where 128 = val_dim
// Each threadgroup handles one (head, token) pair
// ============================================================================

kernel void gdn_prefill_norm_gate(
    device const float* raw_out     [[buffer(0)]],
    device const float* gate_all    [[buffer(1)]],
    device const float* norm_scale  [[buffer(2)]],
    device       float* ssm_out     [[buffer(3)]],
    constant     uint&  n_heads     [[buffer(4)]],
    constant     uint&  val_dim     [[buffer(5)]],
    constant     float& eps         [[buffer(6)]],
    constant     uint&  scale_n_heads [[buffer(7)]],
    constant     uint&  T           [[buffer(8)]],
    uint2 tg_pos                    [[threadgroup_position_in_grid]],
    uint  tid                       [[thread_index_in_threadgroup]],
    uint  simd_lane                 [[thread_index_in_simdgroup]],
    uint  simd_group                [[simdgroup_index_in_threadgroup]])
{
    uint h = tg_pos.x;
    uint t = tg_pos.y;
    if (h >= n_heads || t >= T) return;

    uint vj = tid;
    uint idx = t * n_heads * val_dim + h * val_dim + vj;

    // Load raw output value
    float val = (vj < val_dim) ? raw_out[idx] : 0.0f;

    // RMSNorm: compute sum of squares across val_dim
    float ss = val * val;
    ss = simd_sum(ss);

    // Cross-simdgroup reduction (val_dim=128 = 4 simdgroups of 32)
    threadgroup float partial[4];
    if (simd_lane == 0) {
        partial[simd_group] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_sg = (val_dim + 31) / 32;
    threadgroup float total_ss;
    if (simd_group == 0) {
        float v = (simd_lane < num_sg) ? partial[simd_lane] : 0.0f;
        v = simd_sum(v);
        if (simd_lane == 0) {
            total_ss = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (vj < val_dim) {
        float inv_norm = rsqrt(total_ss / (float)val_dim + eps);

        // Apply scale
        device const float* head_scale = norm_scale + (h % scale_n_heads) * val_dim;
        float normed = val * inv_norm * head_scale[vj];

        // SiLU gate
        float gate_val = gate_all[idx];
        float silu_gate = gate_val / (1.0f + exp(-gate_val));

        ssm_out[idx] = silu_gate * normed;
    }
}

// ============================================================================

// ============================================================================
// ssm_conv1d_prefill: Batched causal conv1d for T tokens
//
// For each token t, channel d:
//   output[t][d] = sum_{tap=0}^{kernel_size-1} kernel_w[d*kernel_size+tap] * input_at_tap
// where input_at_tap reads from conv_state circular buffer (for taps before
// the batch) or from earlier tokens in the batch.
//
// After all T tokens, conv_state is updated with the last (kernel_size-1)
// tokens from the input batch.
//
// buffer(0): input      [T * dim] float
// buffer(1): conv_state [(kernel_size-1) * dim] float -- R/W circular buffer
// buffer(2): kernel_w   [dim * kernel_size] float     -- [dim, kernel_size]
// buffer(3): output     [T * dim] float
// buffer(4): dim (uint)
// buffer(5): kernel_size (uint)  -- 4
// buffer(6): state_pos (uint)    -- current write position in circular buffer
// buffer(7): T (uint)
//
// grid: (ceil(dim/256), 1, 1), threadgroup: (256, 1, 1)
// ============================================================================

kernel void ssm_conv1d_prefill(
    device const float* input      [[buffer(0)]],
    device       float* conv_state [[buffer(1)]],
    device const float* kernel_w   [[buffer(2)]],
    device       float* output     [[buffer(3)]],
    constant     uint&  dim        [[buffer(4)]],
    constant     uint&  kernel_size [[buffer(5)]],
    constant     uint&  state_pos  [[buffer(6)]],
    constant     uint&  T          [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= dim) return;

    uint d = gid;
    uint buf_slots = kernel_size - 1;  // e.g. 3 for kernel_size=4

    // Process each token in sequence
    for (uint t = 0; t < T; t++) {
        float sum = 0.0f;

        // For each tap position (0 = oldest, kernel_size-1 = newest/current)
        for (uint tap = 0; tap < kernel_size; tap++) {
            float input_val;

            // How many tokens back from current token t does this tap reach?
            // tap 0 = (kernel_size-1) tokens back, tap (kernel_size-1) = current token
            int tokens_back = (int)(kernel_size - 1 - tap);
            int source_t = (int)t - tokens_back;

            if (source_t >= 0) {
                // Read from the input batch
                input_val = input[(uint)source_t * dim + d];
            } else {
                // Read from conv_state circular buffer
                // source_t is negative, meaning we need to look into the state
                // The state holds the last buf_slots tokens before the batch.
                // state_pos is the NEXT write position (oldest entry).
                // Slot for this tap: (state_pos + buf_slots + source_t) % buf_slots
                uint slot = (state_pos + buf_slots + (uint)((int)buf_slots + source_t)) % buf_slots;
                input_val = conv_state[slot * dim + d];
            }

            sum += kernel_w[d * kernel_size + tap] * input_val;
        }

        output[t * dim + d] = sum;
    }

    // Update conv_state with the last buf_slots tokens from the input batch.
    // After processing T tokens, the conv_state should contain the last
    // (kernel_size-1) tokens so the next decode/prefill continues correctly.
    for (uint s = 0; s < buf_slots; s++) {
        int src_t = (int)T - (int)buf_slots + (int)s;
        if (src_t >= 0) {
            // This token came from the input batch
            uint new_slot = (state_pos + s) % buf_slots;
            conv_state[new_slot * dim + d] = input[(uint)src_t * dim + d];
        }
        // If src_t < 0, the state slot already has the correct value (unchanged)
    }
}

// ============================================================================
// ssm_conv1d_silu_prefill: Fused conv1d + SiLU for batched GDN prefill
//
// Same as ssm_conv1d_prefill but applies SiLU activation to the output.
// Saves 1 dispatch + 1 memory barrier vs. separate conv1d + SiLU.
//
// buffer(0): input      [T * dim] float
// buffer(1): conv_state [(kernel_size-1) * dim] float -- R/W circular buffer
// buffer(2): kernel_w   [dim * kernel_size] float     -- [dim, kernel_size]
// buffer(3): output     [T * dim] float
// buffer(4): dim (uint)
// buffer(5): kernel_size (uint)  -- 4
// buffer(6): state_pos (uint)    -- current write position in circular buffer
// buffer(7): T (uint)
//
// grid: (ceil(dim/256), 1, 1), threadgroup: (256, 1, 1)
// ============================================================================

kernel void ssm_conv1d_silu_prefill(
    device const float* input      [[buffer(0)]],
    device       float* conv_state [[buffer(1)]],
    device const float* kernel_w   [[buffer(2)]],
    device       float* output     [[buffer(3)]],
    constant     uint&  dim        [[buffer(4)]],
    constant     uint&  kernel_size [[buffer(5)]],
    constant     uint&  state_pos  [[buffer(6)]],
    constant     uint&  T          [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= dim) return;

    uint d = gid;
    uint buf_slots = kernel_size - 1;

    for (uint t = 0; t < T; t++) {
        float sum = 0.0f;

        for (uint tap = 0; tap < kernel_size; tap++) {
            float input_val;
            int tokens_back = (int)(kernel_size - 1 - tap);
            int source_t = (int)t - tokens_back;

            if (source_t >= 0) {
                input_val = input[(uint)source_t * dim + d];
            } else {
                uint slot = (state_pos + buf_slots + (uint)((int)buf_slots + source_t)) % buf_slots;
                input_val = conv_state[slot * dim + d];
            }

            sum += kernel_w[d * kernel_size + tap] * input_val;
        }

        // Fused SiLU: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
        output[t * dim + d] = sum / (1.0f + exp(-sum));
    }

    // Update conv_state
    for (uint s = 0; s < buf_slots; s++) {
        int src_t = (int)T - (int)buf_slots + (int)s;
        if (src_t >= 0) {
            uint new_slot = (state_pos + s) % buf_slots;
            conv_state[new_slot * dim + d] = input[(uint)src_t * dim + d];
        }
    }
}

// ============================================================================
// ssm_conv1d_silu_prefill_parallel: Token-parallel fused conv1d + SiLU
//
// Same computation as ssm_conv1d_silu_prefill but parallelized across tokens.
// The serial kernel dispatches ceil(dim/256) TGs, each looping over T tokens.
// This kernel dispatches (ceil(dim/TG_SIZE), T) TGs -- one per (channel_block, token).
//
// For T=128, dim=8192: 32 * 128 = 4096 TGs vs 32 serial TGs.
// Each thread computes conv1d + SiLU for one (channel, token) pair.
// Conv state update: only the last-token TGs (grid_y == T-1) update conv_state.
//
// buffer(0): input      [T * dim] float
// buffer(1): conv_state [(kernel_size-1) * dim] float -- R/W circular buffer
// buffer(2): kernel_w   [dim * kernel_size] float     -- [dim, kernel_size]
// buffer(3): output     [T * dim] float
// buffer(4): dim (uint)
// buffer(5): kernel_size (uint)  -- 4
// buffer(6): state_pos (uint)    -- current write position in circular buffer
// buffer(7): T (uint)
//
// grid: (ceil(dim/TG_SIZE), T, 1), threadgroup: (TG_SIZE, 1, 1)
// ============================================================================

kernel void ssm_conv1d_silu_prefill_parallel(
    device const float* input      [[buffer(0)]],
    device       float* conv_state [[buffer(1)]],
    device const float* kernel_w   [[buffer(2)]],
    device       float* output     [[buffer(3)]],
    constant     uint&  dim        [[buffer(4)]],
    constant     uint&  kernel_size [[buffer(5)]],
    constant     uint&  state_pos  [[buffer(6)]],
    constant     uint&  T          [[buffer(7)]],
    uint2  gid [[thread_position_in_grid]])
{
    uint d = gid.x;
    if (d >= dim) return;

    uint t = gid.y;
    uint buf_slots = kernel_size - 1;

    // Compute conv1d for this (d, t) pair
    float sum = 0.0f;
    for (uint tap = 0; tap < kernel_size; tap++) {
        float input_val;
        int tokens_back = (int)(kernel_size - 1 - tap);
        int source_t = (int)t - tokens_back;

        if (source_t >= 0) {
            input_val = input[(uint)source_t * dim + d];
        } else {
            uint slot = (state_pos + buf_slots + (uint)((int)buf_slots + source_t)) % buf_slots;
            input_val = conv_state[slot * dim + d];
        }

        sum += kernel_w[d * kernel_size + tap] * input_val;
    }

    // Fused SiLU: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    output[t * dim + d] = sum / (1.0f + exp(-sum));

    // Update conv_state: only threads handling the LAST token do this
    if (t == T - 1) {
        for (uint s = 0; s < buf_slots; s++) {
            int src_t = (int)T - (int)buf_slots + (int)s;
            if (src_t >= 0) {
                uint new_slot = (state_pos + s) % buf_slots;
                conv_state[new_slot * dim + d] = input[(uint)src_t * dim + d];
            }
        }
    }
}

// ============================================================================
// NOTE: silu_inplace_batched is NOT needed as a separate kernel.
// The existing silu_inplace kernel works for batched prefill by simply
// dispatching with dim = T * channels (total number of elements).
// ============================================================================

// ============================================================================
// l2_normalize_heads_batched: Per-head L2 normalization for T tokens
//
// For each token t and head h:
//   x[t*n_heads*head_dim + h*head_dim .. +head_dim] /= max(||x_head||, eps)
//
// buffer(0): x         [T * n_heads * head_dim] float -- modified in-place
// buffer(1): n_heads (uint)
// buffer(2): head_dim (uint)
// buffer(3): eps (float)
// buffer(4): T (uint)
//
// grid: (n_heads * T, 1, 1) -- one threadgroup per (token, head) pair
// threadgroup: (min(head_dim, 256), 1, 1)
// ============================================================================

kernel void l2_normalize_heads_batched(
    device       float* x        [[buffer(0)]],
    constant     uint&  n_heads  [[buffer(1)]],
    constant     uint&  head_dim [[buffer(2)]],
    constant     float& eps      [[buffer(3)]],
    constant     uint&  T        [[buffer(4)]],
    uint group_idx               [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]],
    uint tg_size                 [[threads_per_threadgroup]],
    uint simd_lane               [[thread_index_in_simdgroup]],
    uint simd_group              [[simdgroup_index_in_threadgroup]])
{
    // group_idx maps to (token, head) pair
    // Layout: group_idx = t * n_heads + h
    uint t = group_idx / n_heads;
    uint h = group_idx % n_heads;

    if (t >= T || h >= n_heads) return;

    device float* head = x + t * n_heads * head_dim + h * head_dim;

    // Pass 1: compute sum of squares
    float ss = 0.0f;
    for (uint i = tid; i < head_dim; i += tg_size) {
        float v = head[i];
        ss += v * v;
    }

    ss = simd_sum(ss);

    threadgroup float partial_sums[8];
    if (simd_lane == 0) {
        partial_sums[simd_group] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simd_groups = (tg_size + 31) / 32;
    threadgroup float total_ss;
    if (simd_group == 0) {
        float val = (simd_lane < num_simd_groups) ? partial_sums[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            total_ss = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float norm = sqrt(total_ss);
    float s = (norm > eps) ? (1.0f / norm) : (1.0f / eps);

    // Pass 2: normalize
    for (uint i = tid; i < head_dim; i += tg_size) {
        head[i] *= s;
    }
}

// ============================================================================
// l2_normalize_qk_strided: Per-head L2 normalization of Q and K within conv_out
//
// Q is at conv_out[t * stride + q_offset + h * head_dim .. + head_dim]
// K is at conv_out[t * stride + k_offset + h * head_dim .. + head_dim]
//
// buffer(0): data       [T * stride] float -- modified in-place
// buffer(1): n_heads (uint)       -- 16 (n_kv_heads)
// buffer(2): head_dim (uint)      -- 128
// buffer(3): T (uint)
// buffer(4): stride (uint)        -- qkv_dim = 8192
// buffer(5): q_offset (uint)      -- 0
// buffer(6): k_offset (uint)      -- qk_dim = 2048
//
// grid: (n_heads * T, 1, 1) -- one threadgroup per (token, head)
// threadgroup: (128, 1, 1)
// ============================================================================

kernel void l2_normalize_qk_strided(
    device       float* data      [[buffer(0)]],
    constant     uint&  n_heads   [[buffer(1)]],
    constant     uint&  head_dim  [[buffer(2)]],
    constant     uint&  T         [[buffer(3)]],
    constant     uint&  stride    [[buffer(4)]],
    constant     uint&  q_offset  [[buffer(5)]],
    constant     uint&  k_offset  [[buffer(6)]],
    uint group_idx               [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]],
    uint simd_lane               [[thread_index_in_simdgroup]],
    uint simd_group              [[simdgroup_index_in_threadgroup]])
{
    uint t = group_idx / n_heads;
    uint h = group_idx % n_heads;
    if (t >= T || h >= n_heads) return;

    device float* q_head = data + t * stride + q_offset + h * head_dim;
    device float* k_head = data + t * stride + k_offset + h * head_dim;

    // Compute L2 norms of Q and K heads (cooperative across threadgroup)
    float q_val = (tid < head_dim) ? q_head[tid] : 0.0f;
    float k_val = (tid < head_dim) ? k_head[tid] : 0.0f;
    float q_sq = q_val * q_val;
    float k_sq = k_val * k_val;

    q_sq = simd_sum(q_sq);
    k_sq = simd_sum(k_sq);

    uint num_sg = (head_dim + 31) / 32;
    threadgroup float tg_q[4];
    threadgroup float tg_k[4];
    if (simd_lane == 0) {
        tg_q[simd_group] = q_sq;
        tg_k[simd_group] = k_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float total_q_sq;
    threadgroup float total_k_sq;
    if (simd_group == 0) {
        float qv = (simd_lane < num_sg) ? tg_q[simd_lane] : 0.0f;
        float kv = (simd_lane < num_sg) ? tg_k[simd_lane] : 0.0f;
        qv = simd_sum(qv);
        kv = simd_sum(kv);
        if (simd_lane == 0) {
            float l2_eps = 1e-12f;
            total_q_sq = rsqrt(max(qv, l2_eps));
            total_k_sq = rsqrt(max(kv, l2_eps));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < head_dim) {
        q_head[tid] *= total_q_sq;
        k_head[tid] *= total_k_sq;
    }
}

// ============================================================================
// gdn_compute_gates_batched: Compute decay and mixing gates for T tokens
//
// Per-head computation for each token t (matching NVLabs GatedDeltaNet):
//   gate[h] = ssm_a[h] * softplus(alpha_weight[t*n_heads+h] + dt_bias[h])
//   alpha_out[t*n_heads+h] = exp(gate)     // decay factor in (0,1)
//   beta_out[t*n_heads+h]  = sigmoid(beta_weight[t*n_heads+h])
//
// buffer(0): ssm_dt_bias       [n_heads] float -- dt bias per head (shared across tokens)
// buffer(1): ssm_a             [n_heads] float -- -exp(A_log) (shared across tokens)
// buffer(2): ssm_beta_weight   [T * n_heads] float -- gn_proj per-token (pre-sigmoid)
// buffer(3): ssm_alpha_weight  [T * n_heads] float -- gk_proj per-token (from batched matvec)
// buffer(4): alpha_out         [T * n_heads] float -- OUTPUT: decay factors
// buffer(5): beta_out          [T * n_heads] float -- OUTPUT: mixing rates
// buffer(6): n_heads (uint)
// buffer(7): T (uint)
//
// grid: (ceil(n_heads * T / 256), 1, 1), threadgroup: (256, 1, 1)
// ============================================================================

kernel void gdn_compute_gates_batched(
    device const float* ssm_dt_bias       [[buffer(0)]],
    device const float* ssm_a             [[buffer(1)]],
    device const float* ssm_beta_weight   [[buffer(2)]],
    device const float* ssm_alpha_weight  [[buffer(3)]],
    device       float* alpha_out         [[buffer(4)]],
    device       float* beta_out          [[buffer(5)]],
    constant     uint&  n_heads           [[buffer(6)]],
    constant     uint&  T                 [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = n_heads * T;
    if (gid >= total) return;

    // gid = t * n_heads + h
    uint h = gid % n_heads;

    // softplus(x) = log(1 + exp(x)), numerically stable
    float sp_input = ssm_alpha_weight[gid] + ssm_dt_bias[h];
    float sp;
    if (sp_input > 20.0f) {
        sp = sp_input;  // softplus(x) ~= x for large x
    } else {
        sp = log(1.0f + exp(sp_input));
    }

    // gate = ssm_a * softplus(gk_proj + dt_bias)
    // ssm_a = -exp(A_log) (already negated+exponentiated in GGUF)
    // gate is negative => alpha = exp(gate) in (0, 1)
    float gate = ssm_a[h] * sp;
    alpha_out[gid] = exp(gate);

    // beta = sigmoid(gn_proj)
    float beta_raw = ssm_beta_weight[gid];
    beta_out[gid] = 1.0f / (1.0f + exp(-beta_raw));
}

// ============================================================================
// Deferred-reduction Q8_0 matvec + residual add + copy
//
// accum[row] += dot(w_q8_row, x); copy_dst[row] = accum[row]
//
// Combines the matvec output projection, residual accumulation, and buffer
// copy into a single dispatch. This fuses the SSMOut matvec + residual_add_copy
// pattern in GDN layers.
//
// buffer(0): weights   [out_dim * row_bytes] uchar -- Q8_0 weight matrix
// buffer(1): x         [in_dim] float -- input vector
// buffer(2): accum     [out_dim] float -- R/W accumulator (residual stream)
// buffer(3): in_dim    (uint)
// buffer(4): copy_dst  [out_dim] float -- receives final accumulated value
// buffer(5): out_dim   (uint)
//
// grid: (ceil(out_dim/4), 1, 1), threadgroup: (128, 1, 1)
// ============================================================================

kernel void dequant_matmul_q8_0_deferred_residual_copy(
    device const uchar* weights   [[buffer(0)]],
    device const float* x         [[buffer(1)]],
    device       float* accum     [[buffer(2)]],
    constant     uint&  in_dim    [[buffer(3)]],
    device       float* copy_dst  [[buffer(4)]],
    constant     uint&  out_dim   [[buffer(5)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint NR0 = 4;
    const uint NSG = 4;
    const uint NQ = 8;
    const uint NW = 32;
    const uint Q8_BLOCK_SIZE = 34;

    const uint nb = in_dim >> 5;
    const uint row_bytes = nb * Q8_BLOCK_SIZE;

    const uint r0 = tgpig * NR0;

    const uint ix = tiisg / (NW / NQ);
    const uint il = tiisg % (NW / NQ);

    const uint ib0 = sgitg * NQ + ix;

    device const uchar* ax[NR0];
    for (uint row = 0; row < NR0; ++row) {
        ax[row] = weights + (r0 + row) * row_bytes;
    }

    float sumf[NR0] = { 0.f, 0.f, 0.f, 0.f };

    float yl[NQ];

    device const float* yb = x + ib0 * 32 + il * NQ;

    for (uint ib = ib0; ib < nb; ib += NSG * NQ) {
        #pragma clang loop unroll(full)
        for (uint i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        for (uint row = 0; row < NR0; ++row) {
            if (r0 + row >= out_dim) break;

            device const uchar* bp = ax[row] + ib * Q8_BLOCK_SIZE;
            float scale = float(as_type<half>(*(device const ushort*)bp));
            device const char* qs = (device const char*)(bp + 2) + il * NQ;

            float sumq = 0.f;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < NQ; ++i) {
                sumq += float(qs[i]) * yl[i];
            }
            sumf[row] += sumq * scale;
        }

        yb += NSG * NQ * 32;
    }

    threadgroup float shmem[NR0 * NW];

    for (uint row = 0; row < NR0; ++row) {
        if (sgitg == 0) {
            shmem[row * NW + tiisg] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem[row * NW + sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0 && r0 + row < out_dim; ++row) {
        float tot = simd_sum(shmem[row * NW + tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            float val = tot + accum[r0 + row];
            accum[r0 + row] = val;
            copy_dst[r0 + row] = val;
        }
    }
}

// ============================================================================
// Deferred-reduction Q4_0 matvec + residual add + copy
//
// accum[row] += dot(w_q4_row, x); copy_dst[row] = accum[row]
//
// Q4_0 variant of the residual+copy fused matvec.
//
// buffer(0): weights   [out_dim * row_bytes] uchar -- Q4_0 weight matrix
// buffer(1): x         [in_dim] float -- input vector
// buffer(2): accum     [out_dim] float -- R/W accumulator (residual stream)
// buffer(3): in_dim    (uint)
// buffer(4): copy_dst  [out_dim] float -- receives final accumulated value
// buffer(5): out_dim   (uint)
//
// grid: (ceil(out_dim/4), 1, 1), threadgroup: (128, 1, 1)
// ============================================================================

kernel void dequant_matmul_q4_0_deferred_residual_copy(
    device const uchar* weights   [[buffer(0)]],
    device const float* x         [[buffer(1)]],
    device       float* accum     [[buffer(2)]],
    constant     uint&  in_dim    [[buffer(3)]],
    device       float* copy_dst  [[buffer(4)]],
    constant     uint&  out_dim   [[buffer(5)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint NR0 = 4;
    const uint NSG = 4;
    const uint NQ = 8;
    const uint NW = 32;

    const uint nb = in_dim >> 5;
    const uint row_bytes = nb * Q4_BLOCK_SIZE;

    const uint r0 = tgpig * NR0;

    const uint ix = tiisg / (NW / NQ);
    const uint il = tiisg % (NW / NQ);

    const uint ib0 = sgitg * NQ + ix;

    device const uchar* ax[NR0];
    for (uint row = 0; row < NR0; ++row) {
        ax[row] = weights + (r0 + row) * row_bytes;
    }

    float sumf[NR0] = { 0.f, 0.f, 0.f, 0.f };

    for (uint ib = ib0; ib < nb; ib += NSG * NQ) {
        uint block_base = ib * 32;
        float yl_lo[4], yl_hi[4];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            yl_lo[i] = x[block_base + il * 4 + i];
            yl_hi[i] = x[block_base + il * 4 + 16 + i];
        }

        for (uint row = 0; row < NR0; ++row) {
            if (r0 + row >= out_dim) break;

            device const uchar* bp = ax[row] + ib * Q4_BLOCK_SIZE;
            float scale = float(as_type<half>(*(device const ushort*)bp));
            device const uchar* qdata = (device const uchar*)(bp + 2) + il * 4;

            float sumq = 0.f;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; ++i) {
                uchar byte_val = qdata[i];
                float lo = float(byte_val & 0x0F) - 8.0f;
                float hi = float(byte_val >> 4) - 8.0f;
                sumq += lo * yl_lo[i] + hi * yl_hi[i];
            }
            sumf[row] += sumq * scale;
        }
    }

    threadgroup float shmem[NR0 * NW];

    for (uint row = 0; row < NR0; ++row) {
        if (sgitg == 0) {
            shmem[row * NW + tiisg] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem[row * NW + sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0 && r0 + row < out_dim; ++row) {
        float tot = simd_sum(shmem[row * NW + tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            float val = tot + accum[r0 + row];
            accum[r0 + row] = val;
            copy_dst[r0 + row] = val;
        }
    }
}

// ============================================================================
// dequant_matmul_q4_0_deferred_residual_copy_nr2: NR0=2 Q4_0 matvec + residual + copy
//
// accum[row] += dot(w_q4_row, x); copy_dst[row] = accum[row]
//
// NR0=2 variant for better GPU occupancy.
//
// Dispatch: threadgroups = ceil(out_dim/2), threads_per_threadgroup = 128
// ============================================================================

kernel void dequant_matmul_q4_0_deferred_residual_copy_nr2(
    device const uchar* weights   [[buffer(0)]],
    device const float* x         [[buffer(1)]],
    device       float* accum     [[buffer(2)]],
    constant     uint&  in_dim    [[buffer(3)]],
    device       float* copy_dst  [[buffer(4)]],
    constant     uint&  out_dim   [[buffer(5)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint NR0 = 2;
    const uint NSG = 4;
    const uint NQ = 8;
    const uint NW = 32;

    const uint nb = in_dim >> 5;
    const uint row_bytes = nb * Q4_BLOCK_SIZE;

    const uint r0 = tgpig * NR0;

    const uint ix = tiisg / (NW / NQ);
    const uint il = tiisg % (NW / NQ);

    const uint ib0 = sgitg * NQ + ix;

    device const uchar* ax[NR0];
    for (uint row = 0; row < NR0; ++row) {
        ax[row] = weights + (r0 + row) * row_bytes;
    }

    float sumf[NR0] = { 0.f, 0.f };

    for (uint ib = ib0; ib < nb; ib += NSG * NQ) {
        uint block_base = ib * 32;
        float yl_lo[4], yl_hi[4];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            yl_lo[i] = x[block_base + il * 4 + i];
            yl_hi[i] = x[block_base + il * 4 + 16 + i];
        }

        for (uint row = 0; row < NR0; ++row) {
            if (r0 + row >= out_dim) break;

            device const uchar* bp = ax[row] + ib * Q4_BLOCK_SIZE;
            float scale = float(as_type<half>(*(device const ushort*)bp));
            device const uchar* qdata = (device const uchar*)(bp + 2) + il * 4;

            float sumq = 0.f;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; ++i) {
                uchar byte_val = qdata[i];
                float lo = float(byte_val & 0x0F) - 8.0f;
                float hi = float(byte_val >> 4) - 8.0f;
                sumq += lo * yl_lo[i] + hi * yl_hi[i];
            }
            sumf[row] += sumq * scale;
        }
    }

    threadgroup float shmem[NR0 * NW];

    for (uint row = 0; row < NR0; ++row) {
        if (sgitg == 0) {
            shmem[row * NW + tiisg] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem[row * NW + sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0 && r0 + row < out_dim; ++row) {
        float tot = simd_sum(shmem[row * NW + tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            float val = tot + accum[r0 + row];
            accum[r0 + row] = val;
            copy_dst[r0 + row] = val;
        }
    }
}

// ============================================================================
// dequant_matmul_q4_0_silu_deferred_residual_copy_nr2:
// Same as dequant_matmul_q4_0_deferred_residual_copy_nr2 but applies
// silu(gate[i]) * x[i] inline during x-vector loading. This eliminates
// the separate silu_elementwise_mul dispatch and barrier.
//
// buffer(0): weights   [out_dim, in_dim] Q4_0
// buffer(1): x         [in_dim] float -- values to gate (normed GDN output)
// buffer(2): accum     [out_dim] float -- R/W (residual accumulator)
// buffer(3): in_dim (uint)
// buffer(4): copy_dst  [out_dim] float -- write copy of accum
// buffer(5): out_dim (uint)
// buffer(6): gate      [in_dim] float -- gate values (silu applied inline)
//
// Dispatch: threadgroups = ceil(out_dim/2), threads_per_threadgroup = 128
// ============================================================================

kernel void dequant_matmul_q4_0_silu_deferred_residual_copy_nr2(
    device const uchar* weights   [[buffer(0)]],
    device const float* x         [[buffer(1)]],
    device       float* accum     [[buffer(2)]],
    constant     uint&  in_dim    [[buffer(3)]],
    device       float* copy_dst  [[buffer(4)]],
    constant     uint&  out_dim   [[buffer(5)]],
    device const float* gate      [[buffer(6)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint NR0 = 2;
    const uint NSG = 4;
    const uint NQ = 8;
    const uint NW = 32;

    const uint nb = in_dim >> 5;
    const uint row_bytes = nb * Q4_BLOCK_SIZE;

    const uint r0 = tgpig * NR0;

    const uint ix = tiisg / (NW / NQ);
    const uint il = tiisg % (NW / NQ);

    const uint ib0 = sgitg * NQ + ix;

    device const uchar* ax[NR0];
    for (uint row = 0; row < NR0; ++row) {
        ax[row] = weights + (r0 + row) * row_bytes;
    }

    float sumf[NR0] = { 0.f, 0.f };

    for (uint ib = ib0; ib < nb; ib += NSG * NQ) {
        uint block_base = ib * 32;
        // Fused silu(gate) * x: load gate and x, compute inline
        // Q4_0 de-interleaved: elements 0-15 = lo nibbles, 16-31 = hi nibbles
        float yl_lo[4], yl_hi[4];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            uint idx_lo = block_base + il * 4 + i;
            uint idx_hi = block_base + il * 4 + 16 + i;
            float g_lo = gate[idx_lo];
            float g_hi = gate[idx_hi];
            float sig_lo = 1.0f / (1.0f + exp(-g_lo));
            float sig_hi = 1.0f / (1.0f + exp(-g_hi));
            yl_lo[i] = g_lo * sig_lo * x[idx_lo];
            yl_hi[i] = g_hi * sig_hi * x[idx_hi];
        }

        for (uint row = 0; row < NR0; ++row) {
            if (r0 + row >= out_dim) break;

            device const uchar* bp = ax[row] + ib * Q4_BLOCK_SIZE;
            float scale = float(as_type<half>(*(device const ushort*)bp));
            device const uchar* qdata = (device const uchar*)(bp + 2) + il * 4;

            float sumq = 0.f;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; ++i) {
                uchar byte_val = qdata[i];
                float lo = float(byte_val & 0x0F) - 8.0f;
                float hi = float(byte_val >> 4) - 8.0f;
                sumq += lo * yl_lo[i] + hi * yl_hi[i];
            }
            sumf[row] += sumq * scale;
        }
    }

    threadgroup float shmem[NR0 * NW];

    for (uint row = 0; row < NR0; ++row) {
        if (sgitg == 0) {
            shmem[row * NW + tiisg] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem[row * NW + sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < NR0 && r0 + row < out_dim; ++row) {
        float tot = simd_sum(shmem[row * NW + tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            float val = tot + accum[r0 + row];
            accum[r0 + row] = val;
            copy_dst[r0 + row] = val;
        }
    }
}

// ============================================================================
// dequant_batched_matvec_q8_0: Batched Q8_0 matvec for T input vectors
//
// For each output row r and token t:
//   out[t * out_dim + r] = dot(W[r, :], x[t * in_dim .. (t+1) * in_dim])
//
// Key optimization: weights for each row are loaded ONCE and reused across
// all T tokens, amortizing memory bandwidth by factor T.
//
// Uses the same 2-SG structure as dequant_matmul_q8_0_2sg but adds a
// token loop inside each threadgroup.
//
// buffer(0): weights  [out_dim * row_bytes] uchar -- Q8_0 weight matrix
// buffer(1): x        [T * in_dim] float -- input batch
// buffer(2): out      [T * out_dim] float -- output batch
// buffer(3): in_dim   (uint)
// buffer(4): out_dim  (uint)
// buffer(5): T        (uint) -- number of tokens in the batch
//
// grid: (ceil(out_dim/8), 1, 1), threadgroup: (64, 1, 1)
// ============================================================================

kernel void dequant_batched_matvec_q8_0(
    device const uchar* weights [[buffer(0)]],
    device const float* x       [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant uint&      in_dim  [[buffer(3)]],
    constant uint&      out_dim [[buffer(4)]],
    constant uint&      T       [[buffer(5)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint NR = 4;          // rows per simdgroup
    const uint NQ = 8;          // elements per thread per iteration
    const uint NW = 32;         // SIMD width
    const uint Q8_BLOCK_SIZE = 34;

    const uint nb = in_dim >> 5;
    const uint row_bytes = nb * Q8_BLOCK_SIZE;

    // Each SG owns 4 rows: SG 0 -> rows 0-3, SG 1 -> rows 4-7
    const uint r0 = tgpig * 8 + sgitg * NR;

    const uint ix = tiisg / (NW / NQ);
    const uint il = tiisg % (NW / NQ);

    // Pointers to weight rows (shared across all tokens)
    device const uchar* ax[NR];
    for (uint row = 0; row < NR; ++row) {
        ax[row] = weights + (r0 + row) * row_bytes;
    }

    // Process each token
    for (uint t = 0; t < T; t++) {
        device const float* xt = x + t * in_dim;
        float sumf[NR] = { 0.f, 0.f, 0.f, 0.f };
        float yl[NQ];

        device const float* yb = xt + ix * 32 + il * NQ;

        for (uint ib = ix; ib < nb; ib += NQ) {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < NQ; ++i) {
                yl[i] = yb[i];
            }

            for (uint row = 0; row < NR; ++row) {
                if (r0 + row >= out_dim) break;

                device const uchar* bp = ax[row] + ib * Q8_BLOCK_SIZE;
                float scale = float(as_type<half>(*(device const ushort*)bp));
                device const char* qs = (device const char*)(bp + 2) + il * NQ;

                float sumq = 0.f;
                #pragma clang loop unroll(full)
                for (uint i = 0; i < NQ; ++i) {
                    sumq += float(qs[i]) * yl[i];
                }
                sumf[row] += sumq * scale;
            }

            yb += NQ * 32;
        }

        for (uint row = 0; row < NR && r0 + row < out_dim; ++row) {
            sumf[row] = simd_sum(sumf[row]);
            if (tiisg == 0) {
                out[t * out_dim + r0 + row] = sumf[row];
            }
        }
    }
}

// ============================================================================
// dequant_batched_matvec_q8_0_dual: Fused dual alpha+beta GEMV + gate compute
//
// Performs TWO Q8_0 batched matrix-vector multiplications in a single dispatch
// AND applies gate transformations (softplus+exp for alpha, sigmoid for beta).
// This replaces 3 dispatches (alpha matvec + beta matvec + gate precompute)
// with a single fused kernel, eliminating 2 dispatches and 1 memory barrier.
//
// Gate computations (matching gdn_compute_gates_batched):
//   alpha_out = exp(-exp(A_log) * softplus(alpha_raw + dt_bias))
//   beta_out  = sigmoid(beta_raw)
//
// buffer(0): weights_a  [out_dim * row_bytes] uchar -- Q8_0 alpha weight matrix
// buffer(1): weights_b  [out_dim * row_bytes] uchar -- Q8_0 beta weight matrix
// buffer(2): x          [T * in_dim] float -- input batch (shared)
// buffer(3): out_a      [T * out_dim] float -- alpha output: exp(gate) decay factors
// buffer(4): out_b      [T * out_dim] float -- beta output: sigmoid mixing rates
// buffer(5): in_dim     (uint)
// buffer(6): out_dim    (uint)
// buffer(7): T          (uint) -- number of tokens in the batch
// buffer(8): dt_bias    [out_dim] float -- per-head dt bias
// buffer(9): A_log      [out_dim] float -- per-head A_log (ssm_a = -exp(A_log))
//
// grid: (ceil(out_dim/8), 1, 1), threadgroup: (64, 1, 1)
// ============================================================================

kernel void dequant_batched_matvec_q8_0_dual(
    device const uchar* weights_a [[buffer(0)]],
    device const uchar* weights_b [[buffer(1)]],
    device const float* x         [[buffer(2)]],
    device float*       out_a     [[buffer(3)]],
    device float*       out_b     [[buffer(4)]],
    constant uint&      in_dim    [[buffer(5)]],
    constant uint&      out_dim   [[buffer(6)]],
    constant uint&      T         [[buffer(7)]],
    device const float* dt_bias   [[buffer(8)]],
    device const float* A_log     [[buffer(9)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint NR = 4;          // rows per simdgroup
    const uint NQ = 8;          // elements per thread per iteration
    const uint NW = 32;         // SIMD width
    const uint Q8_BLOCK_SIZE = 34;

    const uint nb = in_dim >> 5;
    const uint row_bytes = nb * Q8_BLOCK_SIZE;

    // Each SG owns 4 rows: SG 0 -> rows 0-3, SG 1 -> rows 4-7
    const uint r0 = tgpig * 8 + sgitg * NR;

    const uint ix = tiisg / (NW / NQ);
    const uint il = tiisg % (NW / NQ);

    // Preload per-head constants (shared across all tokens)
    float head_dt_bias[NR];
    float head_ssm_a[NR];
    for (uint row = 0; row < NR; ++row) {
        if (r0 + row < out_dim) {
            head_dt_bias[row] = dt_bias[r0 + row];
            head_ssm_a[row] = -exp(A_log[r0 + row]); // ssm_a = -exp(A_log)
        }
    }

    // Pointers to weight rows for both matrices (shared across all tokens)
    device const uchar* ax_a[NR];
    device const uchar* ax_b[NR];
    for (uint row = 0; row < NR; ++row) {
        ax_a[row] = weights_a + (r0 + row) * row_bytes;
        ax_b[row] = weights_b + (r0 + row) * row_bytes;
    }

    // Process each token
    for (uint t = 0; t < T; t++) {
        device const float* xt = x + t * in_dim;
        float sumf_a[NR] = { 0.f, 0.f, 0.f, 0.f };
        float sumf_b[NR] = { 0.f, 0.f, 0.f, 0.f };
        float yl[NQ];

        device const float* yb = xt + ix * 32 + il * NQ;

        for (uint ib = ix; ib < nb; ib += NQ) {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < NQ; ++i) {
                yl[i] = yb[i];
            }

            for (uint row = 0; row < NR; ++row) {
                if (r0 + row >= out_dim) break;

                // Alpha weights
                {
                    device const uchar* bp = ax_a[row] + ib * Q8_BLOCK_SIZE;
                    float scale = float(as_type<half>(*(device const ushort*)bp));
                    device const char* qs = (device const char*)(bp + 2) + il * NQ;

                    float sumq = 0.f;
                    #pragma clang loop unroll(full)
                    for (uint i = 0; i < NQ; ++i) {
                        sumq += float(qs[i]) * yl[i];
                    }
                    sumf_a[row] += sumq * scale;
                }

                // Beta weights
                {
                    device const uchar* bp = ax_b[row] + ib * Q8_BLOCK_SIZE;
                    float scale = float(as_type<half>(*(device const ushort*)bp));
                    device const char* qs = (device const char*)(bp + 2) + il * NQ;

                    float sumq = 0.f;
                    #pragma clang loop unroll(full)
                    for (uint i = 0; i < NQ; ++i) {
                        sumq += float(qs[i]) * yl[i];
                    }
                    sumf_b[row] += sumq * scale;
                }
            }

            yb += NQ * 32;
        }

        for (uint row = 0; row < NR && r0 + row < out_dim; ++row) {
            sumf_a[row] = simd_sum(sumf_a[row]);
            sumf_b[row] = simd_sum(sumf_b[row]);
            if (tiisg == 0) {
                // Apply gate transformations inline:
                // alpha = exp(ssm_a * softplus(alpha_raw + dt_bias))
                float sp_input = sumf_a[row] + head_dt_bias[row];
                float sp;
                if (sp_input > 20.0f) {
                    sp = sp_input;
                } else {
                    sp = log(1.0f + exp(sp_input));
                }
                float gate = head_ssm_a[row] * sp;
                out_a[t * out_dim + r0 + row] = exp(gate);

                // beta = sigmoid(beta_raw)
                out_b[t * out_dim + r0 + row] = 1.0f / (1.0f + exp(-sumf_b[row]));
            }
        }
    }
}

"#;

