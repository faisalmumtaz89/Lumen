//! The served NVFP4 matvec (`matvec_nvfp4_wide_f32`) against an independent host reference.
//!
//! The reference decodes the raw packed bytes with `lumen_format::planar_dequant` and multiplies in **f64**,
//! so it shares no code with the kernel's inner loop; a reference that reused the kernel's decode would
//! prove nothing. Every cell must be finite, have `max_abs < 1e-3` and `rel_l2 <= 1e-4`, over each distinct
//! representative projection shape (plus the GDN `in_proj_z` and `out_proj` geometries) at realistic per-tensor
//! global scales, spanning the range a ModelOpt export carries.
//!
//! Requires a CUDA GPU:
//!   cargo test --release -p lumen-runtime --features cuda --test cuda_nvfp4_matvec_test
#![cfg(feature = "cuda")]

use cudarc::driver::{LaunchConfig, PushKernelArg};
use lumen_format::planar_dequant::dequantize_nvfp4;
use lumen_runtime::cuda::ffi::CudaDevice;
use lumen_runtime::cuda::shaders::MATVEC_NVFP4_WIDE_KERNEL_SOURCE;

/// (logical out_dim, logical in_dim, global scale): representative projection shapes, each with a
/// realistic `weight_scale_2`.
///
/// The GDN `out_proj` (6144->5120) and `in_proj_z` (5120->6144) are FP8 in a mixed-precision export; their
/// geometries are included as NVFP4 shape cells because the matvec's row walk must hold over those shapes, not
/// because the tensors are NVFP4.
///
/// The global scales are realistic ones. A unit scale is far outside the range an export carries: outputs reach
/// |out| ~ 179,000, where one f32 ULP is 1.56e-2, so `max_abs < 1e-3` would be below a single ULP and no f32
/// implementation could meet it.
const CELLS: &[(u32, u32, f32)] = &[
    (5120, 17408, 3.5e-4),  // mlp.down_proj  (NVFP4)
    (17408, 5120, 1.5e-4),  // mlp.gate_proj / up_proj (NVFP4)
    (248320, 5120, 1.2e-4), // lm_head (NVFP4)
    (5120, 6144, 3.0e-3),   // linear_attn.out_proj geometry (FP8 in a mixed-precision export)
    (6144, 5120, 5.5e-4),   // linear_attn.in_proj_z geometry (FP8 in a mixed-precision export)
    (10240, 5120, 9.5e-4),
    (12288, 5120, 1.1e-3),
    (1024, 5120, 6.25e-4),
    (5120, 5120, 5.0e-5), // a small weight_scale_2
    (5120, 5120, 4.4e-4), // a large weight_scale_2
];

/// Map away E4M3's two NaN codes (0x7F/0xFF). 2 of 256 bytes are NaN, so a random stream puts one in every
/// 320-scale row of a 5120-wide matrix and the output is NaN by construction. A ModelOpt export carries no NaN
/// block scale, so a NaN-bearing stream models data the artifact cannot contain.
fn finite_e4m3(byte: u8) -> u8 {
    if byte == 0x7F || byte == 0xFF {
        0x38 // 1.0
    } else {
        byte
    }
}

/// A deterministic byte source: no RNG crate, and the same bytes on every run so a failure is reproducible.
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

/// Decode the whole matrix with the host decoder, then dot it in f64 — a reference that shares no code with
/// the kernel's loop.
fn reference_dot(
    packed: &[u8],
    scales: &[u8],
    global: f32,
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Vec<f64> {
    // One row at a time: the decoder takes a whole plane, so slice the row's bytes out.
    let row_bytes = in_dim / 2;
    let scale_bytes = in_dim / 16;
    let mut out = vec![0.0f64; out_dim];
    for row in 0..out_dim {
        let w = dequantize_nvfp4(
            &packed[row * row_bytes..(row + 1) * row_bytes],
            &scales[row * scale_bytes..(row + 1) * scale_bytes],
            global,
        )
        .expect("host dequant");
        let mut acc = 0.0f64;
        for (wi, xi) in w.iter().zip(x.iter()) {
            acc += (*wi as f64) * (*xi as f64);
        }
        out[row] = acc;
    }
    out
}

fn run_kernel(
    dev: &CudaDevice,
    packed: &[u8],
    scales: &[u8],
    global: f32,
    x: &[f32],
    out_dim: u32,
    in_dim: u32,
) -> Vec<f32> {
    // The kernel takes ONE plane: weight | block_scale | global_scale(F32 LE), the converter's order
    // (`convert_hf.rs::lower_nvfp4`). Building it here means the test exercises the real layout rather
    // than a simplified one, including the 4-byte scale after the block scales.
    let mut plane = Vec::with_capacity(packed.len() + scales.len() + 8);
    plane.extend_from_slice(packed);
    plane.extend_from_slice(scales);
    plane.extend_from_slice(&global.to_le_bytes());
    // The activation scale a converted slice may carry after its planes: NaN, so a kernel that read it
    // would fail every comparison here.
    plane.extend_from_slice(&f32::NAN.to_le_bytes());
    let d_w = dev.htod_copy(&plane).expect("htod plane");
    let d_x = dev.htod_copy(x).expect("htod x");
    let mut d_out = dev.alloc_zeros::<f32>(out_dim as usize).expect("alloc out");
    let m = dev
        .compile_and_load(MATVEC_NVFP4_WIDE_KERNEL_SOURCE)
        .expect("compile the NVFP4 matvec");
    let f = m
        .load_function("matvec_nvfp4_wide_f32")
        .expect("matvec_nvfp4_wide_f32");
    // The serving launch: 128 threads, one warp per row, no shared memory.
    const THREADS: u32 = 128;
    let warps = THREADS / 32;
    let grid = out_dim.div_ceil(warps).max(1);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = dev.stream.launch_builder(&f);
    b.arg(&d_w)
        .arg(&d_x)
        .arg(&mut d_out)
        .arg(&out_dim)
        .arg(&in_dim);
    unsafe { b.launch(cfg) }.expect("launch");
    dev.stream.synchronize().expect("sync");
    dev.dtoh_copy(&d_out).expect("dtoh")
}

#[test]
fn nvfp4_matvec_matches_the_host_reference_on_every_shape() {
    let dev = CudaDevice::new(0).expect("CUDA device 0 — this test needs a GPU");
    let mut rng = Lcg(0x5EED_2718_2818_2845);

    for &(out_dim, in_dim, global) in CELLS {
        let (o, i) = (out_dim as usize, in_dim as usize);
        assert_eq!(
            i % 16,
            0,
            "in_dim must be a multiple of the 16-weight group"
        );
        // Codes across the full E2M1 range including the zeros and the largest magnitude; scales across the
        // E4M3 range including 0x00 (zero), 0x7E (448.0) and the subnormal 0x01.
        let packed: Vec<u8> = (0..o * i / 2).map(|_| rng.next_u8()).collect();
        let scales: Vec<u8> = (0..o * i / 16)
            .map(|k| match k % 5 {
                0 => 0x00,
                1 => 0x7E,
                2 => 0x01,
                3 => 0x38, // 1.0
                _ => finite_e4m3(rng.next_u8()),
            })
            .collect();
        let x: Vec<f32> = (0..i).map(|k| ((k as f32) * 0.001 - 1.0) * 0.5).collect();

        let want = reference_dot(&packed, &scales, global, &x, o, i);
        let got = run_kernel(&dev, &packed, &scales, global, &x, out_dim, in_dim);
        assert_eq!(got.len(), want.len());

        // A NaN on either side would pass an unsigned-difference bound silently, so finiteness is asserted
        // first: an all-NaN kernel must fail here.
        for (k, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(g.is_finite(), "{out_dim}x{in_dim}: device [{k}] is {g}");
            assert!(
                w.is_finite(),
                "{out_dim}x{in_dim}: reference [{k}] is {w} — bad test DATA"
            );
        }
        let mut max_abs = 0.0f64;
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for (k, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let d = (*g as f64 - *w).abs();
            assert!(
                d < 1e-3,
                "{out_dim}x{in_dim} gs={global:e} [{k}]: device {g} vs reference {w} (d={d:e})"
            );
            max_abs = max_abs.max(d);
            num += d * d;
            den += w * w;
        }
        assert!(
            den > 0.0,
            "{out_dim}x{in_dim}: the reference is identically zero — proves nothing"
        );
        let rel_l2 = (num / den).sqrt();
        assert!(
            rel_l2 <= 1e-4,
            "{out_dim}x{in_dim} gs={global:e}: rel_l2 {rel_l2:e} > 1e-4 (max_abs {max_abs:e})"
        );
    }
}

/// A single tiny cell whose expected value can be worked out by hand, so a systematic error in the packing
/// order or the scale fold cannot pass by matching only a reference built from the same assumptions.
#[test]
fn nvfp4_matvec_hand_computed_cell_is_exact() {
    let dev = CudaDevice::new(0).expect("CUDA device 0 — this test needs a GPU");
    // in_dim = 16: one block. Codes 0x2,0x3,...,0x9 in the low nibbles; high nibbles zero.
    // E4M3 0x38 = 1.0, global 1.0, x all ones -> out = sum of the decoded codes.
    let codes: [u8; 8] = [0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x09, 0x0A];
    let packed = codes.to_vec();
    let scales = vec![0x38u8];
    let x = vec![1.0f32; 16];
    let got = run_kernel(&dev, &packed, &scales, 1.0, &x, 1, 16);
    // E2M1 values: 0x2=1.0 0x3=1.5 0x4=2.0 0x5=3.0 0x6=4.0 0x7=6.0 0x9=-0.5 0xA=-1.0  (high nibbles 0 -> +0)
    let expect = 1.0 + 1.5 + 2.0 + 3.0 + 4.0 + 6.0 + (-0.5) + (-1.0);
    let want = reference_dot(&packed, &scales, 1.0, &x, 1, 16)[0];
    assert!(
        (got[0] as f64 - want).abs() < 1e-6,
        "device {} vs reference {want}",
        got[0]
    );
    assert!(
        (got[0] as f64 - expect as f64).abs() < 1e-6,
        "device {} vs hand-computed {expect}",
        got[0]
    );
}
