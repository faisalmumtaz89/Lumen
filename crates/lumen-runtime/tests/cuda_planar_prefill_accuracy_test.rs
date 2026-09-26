//! The PREFILL planar route for NVFP4 and FP8 against an independent host reference, at batch 1, 16 and
//! 2048, within `rel_l2 <= 1e-4`.
//!
//! Prefill serves the planar schemes through a different route from decode: the whole plane is dequantized
//! into a scratch by `dequant_{nvfp4,fp8}_to_f32` and then multiplied by the shared F32 SGEMM. A fault in a
//! dequant kernel would pass every decode matvec test and still corrupt the first token. This test launches
//! the dequant kernels and the SGEMM with its own configuration; the production launchers are checked by
//! `cuda::prefill::tests::planar_prefill_launchers_match_the_host_reference`.
//!
//! The reference decodes with `lumen_format::planar_dequant` and multiplies in f64, so it shares no
//! arithmetic with the GPU route.
//!
//! Batch 1 is the degenerate GEMM (one column), 16 is the tail of a short prompt, and 2048 spans many GEMM
//! tiles along the batch dimension.
//!
//! Requires a CUDA GPU:
//!   cargo test --release -p lumen-runtime --features cuda --test cuda_planar_prefill_accuracy_test
#![cfg(feature = "cuda")]

use cudarc::cublas::{sys as cublas_sys, Gemm, GemmConfig};
use cudarc::driver::{LaunchConfig, PushKernelArg};
use lumen_format::planar_dequant::{dequantize_fp8, dequantize_nvfp4};
use lumen_runtime::cuda::ffi::CudaDevice;
use lumen_runtime::cuda::shaders::{DEQUANT_FP8_KERNEL_SOURCE, DEQUANT_NVFP4_KERNEL_SOURCE};

/// A reduced 256 x 512 shape per scheme, carrying a realistic global scale for NVFP4 and
/// FP8 matrices. Small enough that the f64 host reference runs in seconds while still exercising a
/// non-square matrix.
const NVFP4_SHAPE: (usize, usize, f32) = (256, 512, 3.5e-4);
const FP8_SHAPE: (usize, usize, f32) = (256, 512, 3.3e-3);

/// The batch sizes. 2048 x 256 f64 outputs are 4 MB of reference — trivial, but the GPU path's scratch and
/// GEMM tiles are what the larger batch actually exercises.
const BATCHES: &[usize] = &[1, 16, 2048];

fn finite_e4m3(byte: u8) -> u8 {
    if byte == 0x7F || byte == 0xFF {
        0x38
    } else {
        byte
    }
}

struct Lcg(u64);
impl Lcg {
    fn next_u8(&mut self) -> u8 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u8
    }
}

/// Build a whole NVFP4 plane in the converter's order: weight | block_scale | global_scale(4B LE).
fn nvfp4_plane(o: usize, i: usize, global: f32, rng: &mut Lcg) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let packed: Vec<u8> = (0..o * i / 2).map(|_| rng.next_u8()).collect();
    let scales: Vec<u8> = (0..o * i / 16)
        .map(|_| finite_e4m3(rng.next_u8()))
        .collect();
    let mut plane = Vec::with_capacity(packed.len() + scales.len() + 4);
    plane.extend_from_slice(&packed);
    plane.extend_from_slice(&scales);
    plane.extend_from_slice(&global.to_le_bytes());
    (plane, packed, scales)
}

/// Build a whole FP8 plane: weight | global_scale(4B LE).
fn fp8_plane(o: usize, i: usize, scale: f32, rng: &mut Lcg) -> (Vec<u8>, Vec<u8>) {
    let weights: Vec<u8> = (0..o * i).map(|_| finite_e4m3(rng.next_u8())).collect();
    let mut plane = Vec::with_capacity(weights.len() + 4);
    plane.extend_from_slice(&weights);
    plane.extend_from_slice(&scale.to_le_bytes());
    (plane, weights)
}

/// The host reference: dequantize every row with the host decoder, then a f64 SGEMM in the layouts cuBLAS is
/// actually given. Both buffers are ROW-major `[batch, dim]` (one contiguous vector per token — the natural
/// activation layout), and cuBLAS reads a row-major `[n, k]` buffer as its column-major `[k, n]` operand
/// with `ld = k`. So `B[k][b] = x[b*in_dim + k]` and the result comes back as row-major `[batch, out_dim]`,
/// i.e. `out[b*out_dim + r]`. At batch 1 the two conventions coincide, which is why a transposed reference
/// passes every batch-1 test and disagrees only from batch 2 on.
fn reference_sgemm(
    dequant_rows: impl Fn(usize) -> Vec<f32>,
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
    batch: usize,
) -> Vec<f64> {
    let mut out = vec![0.0f64; out_dim * batch];
    for r in 0..out_dim {
        let w = dequant_rows(r);
        for b in 0..batch {
            let mut acc = 0.0f64;
            for k in 0..in_dim {
                acc += (w[k] as f64) * (x[b * in_dim + k] as f64);
            }
            out[b * out_dim + r] = acc;
        }
    }
    out
}

/// The prefill route rebuilt from its parts: the production dequant kernel into a scratch, then cuBLAS
/// SGEMM with the config `prefill.rs` uses (transa=T, transb=N, lda=ldb=in_dim, ldc=out_dim). The launch
/// here is its own (n threads for both schemes; the NVFP4 kernel writes four elements per thread and
/// its surplus threads return); the production launchers themselves are checked by
/// `planar_prefill_launchers_match_the_host_reference` in `cuda/prefill.rs`.
fn run_prefill_route(
    dev: &CudaDevice,
    plane: &[u8],
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
    batch: usize,
    fp8: bool,
) -> Vec<f32> {
    let d_plane = dev.htod_copy(plane).expect("htod plane");
    let d_x = dev.htod_copy(x).expect("htod x");
    let mut d_scratch = dev
        .alloc_zeros::<f32>(out_dim * in_dim)
        .expect("alloc scratch");
    let mut d_out = dev.alloc_zeros::<f32>(out_dim * batch).expect("alloc out");

    let (src, fname) = if fp8 {
        (DEQUANT_FP8_KERNEL_SOURCE, "dequant_fp8_to_f32")
    } else {
        (DEQUANT_NVFP4_KERNEL_SOURCE, "dequant_nvfp4_to_f32")
    };
    let m = dev.compile_and_load(src).expect("compile dequant");
    let f = m.load_function(fname).expect("dequant fn");
    let n = (out_dim * in_dim) as u32;
    let grid = (n + 255) / 256;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = dev.stream.launch_builder(&f);
    b.arg(&d_plane).arg(&mut d_scratch).arg(&n);
    unsafe { b.launch(cfg) }.expect("dequant launch");
    dev.stream.synchronize().expect("sync dequant");

    let gemm = GemmConfig {
        transa: cublas_sys::cublasOperation_t::CUBLAS_OP_T,
        transb: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
        m: out_dim as i32,
        n: batch as i32,
        k: in_dim as i32,
        alpha: 1.0f32,
        lda: in_dim as i32,
        ldb: in_dim as i32,
        beta: 0.0f32,
        ldc: out_dim as i32,
    };
    unsafe {
        dev.blas
            .gemm(gemm, &d_scratch, &d_x, &mut d_out)
            .expect("cublas sgemm")
    };
    dev.stream.synchronize().expect("sync gemm");
    dev.dtoh_copy(&d_out).expect("dtoh out")
}

fn check(got: &[f32], want: &[f64], out_dim: usize, in_dim: usize, batch: usize, tag: &str) {
    assert_eq!(got.len(), want.len());
    for (k, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(g.is_finite(), "{tag} b={batch}: device [{k}] is {g}");
        assert!(
            w.is_finite(),
            "{tag} b={batch}: reference [{k}] is {w} — bad test DATA"
        );
    }
    let mut max_abs = 0.0f64;
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (g, w) in got.iter().zip(want.iter()) {
        let d = (*g as f64 - *w).abs();
        if d > max_abs {
            max_abs = d;
        }
        num += d * d;
        den += w * w;
    }
    assert!(
        den > 0.0,
        "{tag} b={batch}: the reference is identically zero — proves nothing"
    );
    let rel_l2 = (num / den).sqrt();
    eprintln!(
        "  {tag} {out_dim}x{in_dim} batch={batch:<5} max_abs={max_abs:.3e} rel_l2={rel_l2:.3e} \
         (|out|max={:.4})",
        got.iter().fold(0.0f32, |m, v| m.max(v.abs()))
    );
    assert!(rel_l2 <= 1e-4, "{tag} b={batch}: rel_l2 {rel_l2:e} > 1e-4");
}

#[test]
fn nvfp4_prefill_matches_the_host_reference_at_every_batch() {
    let dev = CudaDevice::new(0).expect("CUDA device 0");
    let (o, i, global) = NVFP4_SHAPE;
    let mut rng = Lcg(0x2130_2130_2130_2130);
    let (plane, packed, scales) = nvfp4_plane(o, i, global, &mut rng);

    for &batch in BATCHES {
        let x: Vec<f32> = (0..batch * i)
            .map(|k| ((k % 211) as f32) * 0.004 - 0.42)
            .collect();
        let want = reference_sgemm(
            |r| {
                let rb = i / 2;
                let sb = i / 16;
                dequantize_nvfp4(
                    &packed[r * rb..(r + 1) * rb],
                    &scales[r * sb..(r + 1) * sb],
                    global,
                )
                .expect("oracle nvfp4")
            },
            &x,
            o,
            i,
            batch,
        );
        let got = run_prefill_route(&dev, &plane, &x, o, i, batch, false);
        check(&got, &want, o, i, batch, "nvfp4-prefill");
    }
}

#[test]
fn fp8_prefill_matches_the_host_reference_at_every_batch() {
    let dev = CudaDevice::new(0).expect("CUDA device 0");
    let (o, i, scale) = FP8_SHAPE;
    let mut rng = Lcg(0x2138_2138_2138_2138);
    let (plane, weights) = fp8_plane(o, i, scale, &mut rng);

    for &batch in BATCHES {
        let x: Vec<f32> = (0..batch * i)
            .map(|k| ((k % 197) as f32) * 0.005 - 0.49)
            .collect();
        let want = reference_sgemm(
            |r| dequantize_fp8(&weights[r * i..(r + 1) * i], scale),
            &x,
            o,
            i,
            batch,
        );
        let got = run_prefill_route(&dev, &plane, &x, o, i, batch, true);
        check(&got, &want, o, i, batch, "fp8-prefill");
    }
}

/// A hand-computed cell so a systematic fault in the transpose or the stride cannot pass by matching a
/// reference built from the same assumption. in_dim=16 (one NVFP4 block), batch=2, and an x whose two
/// columns differ, so a batch-major/row-major confusion moves a value that must not move.
#[test]
fn prefill_hand_computed_cell_pins_the_layout() {
    let dev = CudaDevice::new(0).expect("CUDA device 0");
    // One row, 16 columns. Codes 0x2..0x7, 0x9 and 0xA in the low nibbles of the 8 bytes; high nibbles +0.
    let packed: Vec<u8> = vec![0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x09, 0x0A];
    let scales = [0x38u8]; // E4M3 1.0
    let global = 1.0f32;
    let mut plane = packed.clone();
    plane.push(scales[0]);
    plane.extend_from_slice(&global.to_le_bytes());
    // Token 0 is all 1.0 -> row dot = 1.0+1.5+2+3+4+6-0.5-1 = 16.
    // Token 1 is all 2.0 -> exactly twice that. Built as row-major [batch, in_dim], the layout cuBLAS is
    // handed; a transposed x puts token 1's first element where token 0's second belongs and the two
    // results collapse to the same value, which is what this cell is here to catch.
    let mut x = vec![0.0f32; 2 * 16];
    for k in 0..16 {
        x[k] = 1.0;
        x[16 + k] = 2.0;
    }
    let got = run_prefill_route(&dev, &plane, &x, 1, 16, 2, false);
    assert_eq!(got.len(), 2);
    assert!((got[0] - 16.0).abs() < 1e-4, "token 0 = {}", got[0]);
    assert!(
        (got[1] - 32.0).abs() < 1e-4,
        "token 1 = {} (a layout swap shows here)",
        got[1]
    );
}
