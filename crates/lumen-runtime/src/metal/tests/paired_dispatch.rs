//! Focused unit test comparing the paired BF16 GDN QKV+attn_gate kernel
//! against two separate `tiled_matmul_bf16_k64` dispatches. The paired kernel
//! consumes a concat-then-stripe BF16 weight buffer (see `repack_bf16.rs`);
//! the two-dispatch reference consumes the un-packed BF16 weights directly.
//!
//! Pass criterion: per-element max absolute difference <= 1e-5 across all
//! output elements for both Y_qkv and Y_gate.
//!
//! This test is the canonical correctness gate for the kernel. The unit
//! tests in `repack_bf16.rs` cover the byte-permutation layer; this test
//! covers the kernel-level interpretation.
//!
//! Run with:
//!   cargo test --release --lib -p lumen-runtime \
//!     metal::tests::paired_dispatch -- --nocapture

use crate::metal::MetalF32Backend;
use crate::metal::ffi::{MTLSize, MetalFunctionConstantValues};
use crate::metal::shaders::METAL_SHADER_SOURCE;

/// Convert a float to a BF16 bit pattern (round-to-nearest-even).
fn f32_to_bf16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7FFF + lsb);
    (rounded >> 16) as u16
}

fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn make_synthetic_bf16(n_rows: usize, k_cols: usize, seed: u32) -> Vec<u16> {
    let mut s: u64 = seed as u64;
    let mut out = Vec::with_capacity(n_rows * k_cols);
    for _ in 0..(n_rows * k_cols) {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (s >> 32) as u32;
        let f = ((u as f32) / (u32::MAX as f32) * 0.2 - 0.1) as f32;
        out.push(f32_to_bf16_bits(f));
    }
    out
}

fn make_random_f32_small(n: usize, seed: u32) -> Vec<f32> {
    let mut s: u64 = seed as u64;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (s >> 32) as u32;
        let f = (u as f32) / (u32::MAX as f32) - 0.5;
        v.push(f);
    }
    v
}

#[test]
fn paired_matches_two_dispatch() {
    paired_run(15, 64, 64, 32);
}

#[test]
fn paired_matches_two_dispatch_qwen35_shape() {
    // Production Qwen3.5-9B GDN shape: M=15 (prompt of 15), K=4096 (hidden_dim),
    // N_QKV=8192 (qkv-proj), N_GATE=4096 (attn-gate-proj).
    paired_run(15, 4096, 8192, 4096);
}

#[test]
fn paired_aligned_kernel_qwen35_shape() {
    // Force the aligned (FC_BC_*=false) variant to ensure that path is also
    // correct. M=64 (aligned to TILE_M=32) is the smallest M satisfying all
    // three alignment constraints with the production K=4096 and N totals.
    paired_run_aligned(64, 4096, 8192, 4096);
}

fn paired_run(m: usize, k: usize, n_qkv: usize, n_gate: usize) {
    let n_total = n_qkv + n_gate;
    let m_u32 = m as u32;
    let k_u32 = k as u32;
    let n_qkv_u32 = n_qkv as u32;
    let n_gate_u32 = n_gate as u32;
    paired_run_impl(m, k, n_qkv, n_gate, n_total, m_u32, k_u32, n_qkv_u32, n_gate_u32, true);
}

fn paired_run_aligned(m: usize, k: usize, n_qkv: usize, n_gate: usize) {
    let n_total = n_qkv + n_gate;
    let m_u32 = m as u32;
    let k_u32 = k as u32;
    let n_qkv_u32 = n_qkv as u32;
    let n_gate_u32 = n_gate as u32;
    paired_run_impl(m, k, n_qkv, n_gate, n_total, m_u32, k_u32, n_qkv_u32, n_gate_u32, false);
}

#[allow(non_snake_case)] // M/K = GEMM dims, N_* = output widths (math convention)
fn paired_run_impl(
    M: usize, K: usize, N_QKV: usize, N_GATE: usize, N_TOTAL: usize,
    m_u32: u32, k_u32: u32, n_qkv_u32: u32, n_gate_u32: u32,
    use_bc: bool,
) {

    let backend = MetalF32Backend::new().expect("Metal backend create");
    let lib = backend.device
        .new_library_with_source(METAL_SHADER_SOURCE)
        .expect("Metal lib compile");

    // Function constants: BC variant (true/true/true) is required when any
    // of M/N/K is not aligned to TILE_M=TILE_N=32 or TILE_K_64=64.
    // Aligned variant (false/false/false) is the production fast path for
    // aligned shapes.
    let fcv = MetalFunctionConstantValues::new();
    fcv.set_bool(use_bc, 10);
    fcv.set_bool(use_bc, 11);
    fcv.set_bool(use_bc, 12);

    // Reference: tiled_matmul_bf16_k64 (single output).
    let f_ref = lib.get_function_with_constants("tiled_matmul_bf16_k64", &fcv)
        .expect("ref kernel function");
    let pso_ref = backend.device.new_compute_pipeline_state(&f_ref)
        .expect("ref PSO");

    // Treatment: tiled_matmul_bf16_k64_qkv_gate_paired (dual output).
    let f_pair = lib.get_function_with_constants("tiled_matmul_bf16_k64_qkv_gate_paired", &fcv)
        .expect("paired kernel function");
    let pso_pair = backend.device.new_compute_pipeline_state(&f_pair)
        .expect("paired PSO");

    // Synthetic inputs.
    let w_qkv  = make_synthetic_bf16(N_QKV, K, 1);
    let w_gate = make_synthetic_bf16(N_GATE, K, 2);
    let x      = make_random_f32_small(M * K, 3);

    // ---- REF: two separate dispatches ----
    let w_qkv_buf = backend.device.new_buffer_with_bytes(unsafe {
        std::slice::from_raw_parts(w_qkv.as_ptr() as *const u8, w_qkv.len() * 2)
    }).expect("w_qkv buf");
    let w_gate_buf = backend.device.new_buffer_with_bytes(unsafe {
        std::slice::from_raw_parts(w_gate.as_ptr() as *const u8, w_gate.len() * 2)
    }).expect("w_gate buf");
    let x_buf = backend.device.new_buffer_with_bytes(unsafe {
        std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4)
    }).expect("x buf");

    let y_ref_qkv  = backend.device.new_buffer(M * N_QKV * 4).expect("y_ref_qkv");
    let y_ref_gate = backend.device.new_buffer(M * N_GATE * 4).expect("y_ref_gate");

    // Dispatch QKV
    let cmd = backend.queue.new_command_buffer().expect("cmd buf");
    let enc = cmd.new_compute_encoder().expect("enc");
    enc.set_pipeline_state(&pso_ref);
    enc.set_threadgroup_memory_length(8192, 0);
    enc.set_buffer(&w_qkv_buf, 0, 0);
    enc.set_buffer(&x_buf, 0, 1);
    enc.set_buffer(&y_ref_qkv, 0, 2);
    enc.set_bytes(&m_u32.to_le_bytes(), 3);
    enc.set_bytes(&n_qkv_u32.to_le_bytes(), 4);
    enc.set_bytes(&k_u32.to_le_bytes(), 5);
    enc.dispatch_threadgroups(
        MTLSize::new((N_QKV as u64).div_ceil(32), (M as u64).div_ceil(32), 1),
        MTLSize::new(128, 1, 1),
    );

    // Dispatch attn_gate
    enc.set_pipeline_state(&pso_ref);
    enc.set_threadgroup_memory_length(8192, 0);
    enc.set_buffer(&w_gate_buf, 0, 0);
    enc.set_buffer(&x_buf, 0, 1);
    enc.set_buffer(&y_ref_gate, 0, 2);
    enc.set_bytes(&m_u32.to_le_bytes(), 3);
    enc.set_bytes(&n_gate_u32.to_le_bytes(), 4);
    enc.set_bytes(&k_u32.to_le_bytes(), 5);
    enc.dispatch_threadgroups(
        MTLSize::new((N_GATE as u64).div_ceil(32), (M as u64).div_ceil(32), 1),
        MTLSize::new(128, 1, 1),
    );
    enc.end_encoding();
    cmd.commit_and_wait();

    let mut ref_qkv_out  = vec![0.0f32; M * N_QKV];
    let mut ref_gate_out = vec![0.0f32; M * N_GATE];
    y_ref_qkv.read_f32(&mut ref_qkv_out);
    y_ref_gate.read_f32(&mut ref_gate_out);

    // ---- TREATMENT: paired dispatch ----
    // Build the concat-then-stripe packed buffer via the repack helper.
    let w_qkv_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(w_qkv.as_ptr() as *const u8, w_qkv.len() * 2)
    };
    let w_gate_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(w_gate.as_ptr() as *const u8, w_gate.len() * 2)
    };
    let packed = crate::metal::repack_bf16::repack_bf16_qkv_gate_concat(
        w_qkv_bytes, w_gate_bytes, N_QKV, N_GATE, K
    ).expect("concat repack");
    let w_packed_buf = backend.device.new_buffer_with_bytes(&packed)
        .expect("packed buf");

    let y_pair_qkv  = backend.device.new_buffer(M * N_QKV * 4).expect("y_pair_qkv");
    let y_pair_gate = backend.device.new_buffer(M * N_GATE * 4).expect("y_pair_gate");

    let cmd2 = backend.queue.new_command_buffer().expect("cmd buf2");
    let enc2 = cmd2.new_compute_encoder().expect("enc2");
    enc2.set_pipeline_state(&pso_pair);
    enc2.set_threadgroup_memory_length(8192, 0);
    enc2.set_buffer(&w_packed_buf, 0, 0);
    enc2.set_buffer(&x_buf, 0, 1);
    enc2.set_buffer(&y_pair_qkv, 0, 2);
    enc2.set_buffer(&y_pair_gate, 0, 3);
    enc2.set_bytes(&m_u32.to_le_bytes(), 4);
    enc2.set_bytes(&n_qkv_u32.to_le_bytes(), 5);
    enc2.set_bytes(&n_gate_u32.to_le_bytes(), 6);
    enc2.set_bytes(&k_u32.to_le_bytes(), 7);
    enc2.dispatch_threadgroups(
        MTLSize::new((N_TOTAL as u64).div_ceil(32), (M as u64).div_ceil(32), 1),
        MTLSize::new(128, 1, 1),
    );
    enc2.end_encoding();
    cmd2.commit_and_wait();

    let mut pair_qkv_out  = vec![0.0f32; M * N_QKV];
    let mut pair_gate_out = vec![0.0f32; M * N_GATE];
    y_pair_qkv.read_f32(&mut pair_qkv_out);
    y_pair_gate.read_f32(&mut pair_gate_out);

    // ---- Compare ----
    let _ = (bf16_bits_to_f32(0u16),); // unused-import guard
    let max_qkv_diff = ref_qkv_out.iter().zip(pair_qkv_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_gate_diff = ref_gate_out.iter().zip(pair_gate_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // Print up to 8 mismatched (idx, ref, pair) for QKV.
    let mut qkv_mismatch = Vec::new();
    for (i, (a, b)) in ref_qkv_out.iter().zip(pair_qkv_out.iter()).enumerate() {
        if (a - b).abs() > 1e-5 {
            qkv_mismatch.push((i, *a, *b));
            if qkv_mismatch.len() >= 8 { break; }
        }
    }
    let mut gate_mismatch = Vec::new();
    for (i, (a, b)) in ref_gate_out.iter().zip(pair_gate_out.iter()).enumerate() {
        if (a - b).abs() > 1e-5 {
            gate_mismatch.push((i, *a, *b));
            if gate_mismatch.len() >= 8 { break; }
        }
    }

    eprintln!("max_qkv_diff  = {:.6e}", max_qkv_diff);
    eprintln!("max_gate_diff = {:.6e}", max_gate_diff);
    if !qkv_mismatch.is_empty() {
        eprintln!("qkv mismatches (first 8): {:?}", qkv_mismatch);
    }
    if !gate_mismatch.is_empty() {
        eprintln!("gate mismatches (first 8): {:?}", gate_mismatch);
    }

    // Show a sample of values from each.
    eprintln!("ref_qkv[0..8]  = {:?}", &ref_qkv_out[0..8]);
    eprintln!("pair_qkv[0..8] = {:?}", &pair_qkv_out[0..8]);
    eprintln!("ref_gate[0..8]  = {:?}", &ref_gate_out[0..8]);
    eprintln!("pair_gate[0..8] = {:?}", &pair_gate_out[0..8]);

    // Bit-identical comparison: identical FMA order, identical accumulators.
    // Allow a small floating-point tolerance for any reorder Apple may introduce.
    assert!(max_qkv_diff <= 1e-5, "QKV output diverges (max_diff = {})", max_qkv_diff);
    assert!(max_gate_diff <= 1e-5, "Gate output diverges (max_diff = {})", max_gate_diff);
}
