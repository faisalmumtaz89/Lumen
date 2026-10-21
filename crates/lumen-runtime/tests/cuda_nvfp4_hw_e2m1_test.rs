//! The NVFP4 matvecs built with the hardware E2M1 conversion (`compute_120a`) against the default build.
//!
//! Serving loads the hardware build wherever the device admits the native FP4 target, so its output must equal
//! the default build's bit for bit: the plain matvec (projections and the output head), the fused gate/up/SwiGLU
//! and the fused down+residual. Covered: every weight byte 0..=255 (so both nibbles of every E2M1 code, the signed
//! zero 0x8 and the subnormal-path codes 1 and 9) under every finite E4M3 block scale, with a different activation
//! for each weight; and the 27B FFN and head shapes on deterministic bytes.
//!
//! Requires a CUDA GPU; on a device without the native FP4 target there is no hardware build to compare:
//!   cargo test --release -p lumen-runtime --features cuda --test cuda_nvfp4_hw_e2m1_test
#![cfg(feature = "cuda")]

use cudarc::driver::{CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
use lumen_runtime::cuda::ffi::CudaDevice;
use lumen_runtime::cuda::shaders::{matvec_nvfp4_wide_hw_source, MATVEC_NVFP4_WIDE_KERNEL_SOURCE};
use std::sync::Arc;

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

/// E4M3's two NaN codes become 1.0: an export carries no NaN block scale.
fn finite_e4m3(byte: u8) -> u8 {
    if byte == 0x7F || byte == 0xFF {
        0x38
    } else {
        byte
    }
}

/// One plane as the converter lays it out: weight nibbles | E4M3 block scales | F32 global scale.
fn plane(packed: &[u8], scales: &[u8], global: f32) -> Vec<u8> {
    let mut p = Vec::with_capacity(packed.len() + scales.len() + 4);
    p.extend_from_slice(packed);
    p.extend_from_slice(scales);
    p.extend_from_slice(&global.to_le_bytes());
    p
}

struct Builds {
    dev: CudaDevice,
    default: Arc<CudaModule>,
    hw: Arc<CudaModule>,
}

impl Builds {
    /// `None` when the device or the loaded NVRTC does not admit the native FP4 target.
    fn new() -> Option<Self> {
        let dev = CudaDevice::new(0).expect("CUDA device 0 — this test needs a GPU");
        let arch = dev.fp4_native_arch().expect("native FP4 target query")?;
        let default = dev
            .compile_and_load(MATVEC_NVFP4_WIDE_KERNEL_SOURCE)
            .expect("default build");
        let hw = dev
            .compile_and_load_with_arch(&matvec_nvfp4_wide_hw_source(), arch)
            .expect("hardware build on a device that admits the native FP4 target");
        Some(Builds { dev, default, hw })
    }

    fn function(module: &Arc<CudaModule>, name: &str) -> CudaFunction {
        module.load_function(name).expect(name)
    }

    fn cfg(out_dim: u32) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (out_dim.div_ceil(4).max(1), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// `matvec_nvfp4_wide_f32` from both builds on the same plane and input.
    fn plain(&self, p: &[u8], x: &[f32], out_dim: u32, in_dim: u32) -> [Vec<f32>; 2] {
        [&self.default, &self.hw].map(|m| {
            let f = Self::function(m, "matvec_nvfp4_wide_f32");
            let d_p = self.dev.htod_copy(p).expect("htod plane");
            let d_x = self.dev.htod_copy(x).expect("htod x");
            let mut d_out = self
                .dev
                .alloc_zeros::<f32>(out_dim as usize)
                .expect("alloc");
            let mut b = self.dev.stream.launch_builder(&f);
            b.arg(&d_p)
                .arg(&d_x)
                .arg(&mut d_out)
                .arg(&out_dim)
                .arg(&in_dim);
            unsafe { b.launch(Self::cfg(out_dim)) }.expect("launch");
            self.dev.stream.synchronize().expect("sync");
            self.dev.dtoh_copy(&d_out).expect("dtoh")
        })
    }

    /// `matvec_nvfp4_wide_glu_f32` (silu(gate) * up) from both builds.
    fn glu(&self, gate: &[u8], up: &[u8], x: &[f32], out_dim: u32, in_dim: u32) -> [Vec<f32>; 2] {
        [&self.default, &self.hw].map(|m| {
            let f = Self::function(m, "matvec_nvfp4_wide_glu_f32");
            let d_g = self.dev.htod_copy(gate).expect("htod gate");
            let d_u = self.dev.htod_copy(up).expect("htod up");
            let d_x = self.dev.htod_copy(x).expect("htod x");
            let mut d_out = self
                .dev
                .alloc_zeros::<f32>(out_dim as usize)
                .expect("alloc");
            let mut b = self.dev.stream.launch_builder(&f);
            b.arg(&d_g)
                .arg(&d_u)
                .arg(&d_x)
                .arg(&mut d_out)
                .arg(&out_dim)
                .arg(&in_dim);
            unsafe { b.launch(Self::cfg(out_dim)) }.expect("launch");
            self.dev.stream.synchronize().expect("sync");
            self.dev.dtoh_copy(&d_out).expect("dtoh")
        })
    }

    /// `matvec_nvfp4_wide_residual_f32` (residual + W x) from both builds.
    fn residual(
        &self,
        p: &[u8],
        x: &[f32],
        res: &[f32],
        out_dim: u32,
        in_dim: u32,
    ) -> [Vec<f32>; 2] {
        [&self.default, &self.hw].map(|m| {
            let f = Self::function(m, "matvec_nvfp4_wide_residual_f32");
            let d_p = self.dev.htod_copy(p).expect("htod plane");
            let d_x = self.dev.htod_copy(x).expect("htod x");
            let d_r = self.dev.htod_copy(res).expect("htod residual");
            let mut d_out = self
                .dev
                .alloc_zeros::<f32>(out_dim as usize)
                .expect("alloc");
            let mut b = self.dev.stream.launch_builder(&f);
            b.arg(&d_p)
                .arg(&d_x)
                .arg(&d_r)
                .arg(&mut d_out)
                .arg(&out_dim)
                .arg(&in_dim);
            unsafe { b.launch(Self::cfg(out_dim)) }.expect("launch");
            self.dev.stream.synchronize().expect("sync");
            self.dev.dtoh_copy(&d_out).expect("dtoh")
        })
    }
}

fn assert_identical(what: &str, [default, hw]: [Vec<f32>; 2]) {
    assert_eq!(default.len(), hw.len());
    for (k, (d, h)) in default.iter().zip(hw.iter()).enumerate() {
        assert!(d.is_finite(), "{what} [{k}]: default build gives {d}");
        assert_eq!(
            d.to_bits(),
            h.to_bits(),
            "{what} [{k}]: default build {d:e} ({:#010x}), hardware build {h:e} ({:#010x})",
            d.to_bits(),
            h.to_bits()
        );
    }
}

fn activations(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Lcg(seed);
    (0..n)
        .map(|_| (rng.next_u8() as f32 - 127.5) / 64.0 + 1.0e-3)
        .collect()
}

#[test]
fn every_weight_byte_and_block_scale_gives_the_same_bits_in_both_builds() {
    let Some(builds) = Builds::new() else {
        eprintln!(
            "this device or NVRTC does not admit the native FP4 target: there is no hardware build"
        );
        return;
    };
    // 16 rows of 512 weights: each row holds the 256 byte values once, in an order rotated per row, so every
    // code sits in both nibbles beside different neighbours; its 32 group scales walk all 256 E4M3 codes over
    // the 16 rows (NaN codes as 1.0). Every activation differs from its pair partner, so swapped nibbles change
    // the sum.
    let (out_dim, in_dim) = (16u32, 512u32);
    let packed: Vec<u8> = (0..out_dim as usize)
        .flat_map(|row| (0..256usize).map(move |k| ((k + 17 * row) % 256) as u8))
        .collect();
    let scales: Vec<u8> = (0..(out_dim * in_dim / 16) as usize)
        .map(|k| finite_e4m3(k as u8))
        .collect();
    let x = activations(in_dim as usize, 0x0E2_A1B2);
    let p = plane(&packed, &scales, 1.0 / 2688.0);
    assert_identical("exhaustive plane", builds.plain(&p, &x, out_dim, in_dim));
}

#[test]
fn the_27b_ffn_and_head_shapes_give_the_same_bits_in_both_builds() {
    let Some(builds) = Builds::new() else {
        eprintln!(
            "this device or NVRTC does not admit the native FP4 target: there is no hardware build"
        );
        return;
    };
    let mut rng = Lcg(0x5EED_2718_2818_2845);
    let mut make = |out_dim: u32, in_dim: u32, global: f32| {
        let n = (out_dim * in_dim) as usize;
        let packed: Vec<u8> = (0..n / 2).map(|_| rng.next_u8()).collect();
        let scales: Vec<u8> = (0..n / 16).map(|_| finite_e4m3(rng.next_u8())).collect();
        plane(&packed, &scales, global)
    };
    let (hidden, inter, vocab) = (5120u32, 17408u32, 248320u32);
    let gate = make(inter, hidden, 1.5e-4);
    let up = make(inter, hidden, 1.5e-4);
    let down = make(hidden, inter, 3.5e-4);
    let head = make(vocab, hidden, 1.2e-4);
    let x_hidden = activations(hidden as usize, 1);
    let x_inter = activations(inter as usize, 2);
    let res = activations(hidden as usize, 3);

    assert_identical("gate", builds.plain(&gate, &x_hidden, inter, hidden));
    assert_identical("down", builds.plain(&down, &x_inter, hidden, inter));
    assert_identical("head", builds.plain(&head, &x_hidden, vocab, hidden));
    assert_identical("glu", builds.glu(&gate, &up, &x_hidden, inter, hidden));
    assert_identical(
        "down+residual",
        builds.residual(&down, &x_inter, &res, hidden, inter),
    );
}
