//! The fused and restructured decode kernels of the planar (NVFP4 / FP8) path, each against the unfused
//! composition of served kernels it replaces, on full-size projection shapes.
//!
//! Every fused kernel must be BIT-identical to the launches it replaces: each one runs the same row code with
//! the same lane mapping, add order and reduction, so any differing bit is a defect, not rounding noise.
//! The one exception is the GDN input projections: the banked gates kernel they replace is built with
//! fast-math, which flushes denormals to zero, so a gate output agrees only while no denormal occurs (the
//! test data has none). Each test also asserts its reference is finite and (where the data allows) mostly
//! non-zero, so an all-NaN or all-zero pair cannot pass by agreeing.
//!
//! | test                                   | kernel under test                     | reference                                   |
//! |----------------------------------------|---------------------------------------|---------------------------------------------|
//! | `nvfp4_glu_...`                        | `matvec_nvfp4_wide_glu_f32`           | gate matvec, up matvec, `swiglu_inplace`    |
//! | `nvfp4_down_residual_...`              | `matvec_nvfp4_wide_residual_f32`      | matvec, `residual_add`                      |
//! | `gdn_input_projections_...`            | `gdn_input_projections_f32`           | FP8 qkv, `matvec_f32_gates_banked`, FP8 z   |
//! | `fp8_three_...`                        | `matvec_fp8_three_f32`                | `matvec_fp8_f32` three times                |
//! | `fp8_residual_rounded_...`             | `matvec_fp8_f32_residual_rounded`     | `matvec_fp8_f32`, `residual_add_copy`       |
//! | `rmsnorm_register_held_...`            | `rmsnorm`                             | `fused_residual_rmsnorm_f32` with b = 0     |
//! | `nvfp4_matvec_decoders_...`            | the NVFP4 matvec's E2M1/E4M3 decoders | `lumen_format::planar_dequant`              |
//! | `fp8_matvec_decode_table_...`          | the FP8 matvec's shared-memory table  | `lumen_format::planar_dequant`              |
//!
//! Requires a CUDA GPU:
//!   cargo test --release -p lumen-runtime --features cuda --test cuda_planar_fused_kernels_test
#![cfg(feature = "cuda")]

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use lumen_format::planar_dequant::{e2m1_to_f32, e4m3_to_f32};
use lumen_runtime::cuda::ffi::CudaDevice;
use lumen_runtime::cuda::shaders::{
    ACTIVATIONS_KERNEL_SOURCE, GDN_INPUT_PROJECTIONS_KERNEL_SOURCE, MATVEC_F32_GATES_KERNEL_SOURCE,
    MATVEC_FP8_KERNEL_SOURCE, MATVEC_NVFP4_WIDE_KERNEL_SOURCE, NORM_KERNEL_SOURCE,
};

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
    /// A byte with E4M3's two NaN codes (0x7F/0xFF) mapped to 1.0 (0x38). The checkpoint carries no NaN
    /// FP8 weight or block scale, and a NaN would make both sides NaN and the comparison vacuous.
    fn next_e4m3(&mut self) -> u8 {
        let b = self.next_u8();
        if b == 0x7F || b == 0xFF {
            0x38
        } else {
            b
        }
    }
}

fn device() -> CudaDevice {
    CudaDevice::new(0).expect("CUDA device 0 — this test needs a GPU")
}

/// One NVFP4 plane in the converter's layout: E2M1 weight nibbles | per-16 E4M3 block scales | global scale.
fn nvfp4_plane(
    dev: &CudaDevice,
    rng: &mut Lcg,
    out_dim: u32,
    in_dim: u32,
    global: f32,
) -> CudaSlice<u8> {
    let n = out_dim as usize * in_dim as usize;
    let mut p = Vec::with_capacity(n / 2 + n / 16 + 4);
    p.extend((0..n / 2).map(|_| rng.next_u8()));
    p.extend((0..n / 16).map(|_| rng.next_e4m3()));
    p.extend_from_slice(&global.to_le_bytes());
    dev.htod_copy(&p).expect("htod nvfp4 plane")
}

/// One FP8 plane in the converter's layout: E4M3 weights | per-tensor scale.
fn fp8_plane(
    dev: &CudaDevice,
    rng: &mut Lcg,
    out_dim: u32,
    in_dim: u32,
    scale: f32,
) -> CudaSlice<u8> {
    let n = out_dim as usize * in_dim as usize;
    let mut p = Vec::with_capacity(n + 4);
    p.extend((0..n).map(|_| rng.next_e4m3()));
    p.extend_from_slice(&scale.to_le_bytes());
    dev.htod_copy(&p).expect("htod fp8 plane")
}

/// The activation every matvec test uses: bounded, varied, both signs.
fn activation(dev: &CudaDevice, in_dim: u32) -> CudaSlice<f32> {
    let x: Vec<f32> = (0..in_dim)
        .map(|i| ((i % 211) as f32) * 0.001 - 0.1)
        .collect();
    dev.htod_copy(&x).expect("htod x")
}

/// One warp per row, four rows per 128-thread block: the planar matvecs' serving launch.
fn warp_per_row(rows: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (rows.div_ceil(4), 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// One thread per element: the elementwise activation kernels' launch.
fn elementwise(n: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Assert `got` equals `want` in every bit, after checking `want` is a meaningful reference.
fn assert_bit_identical(got: &[f32], want: &[f32], require_mostly_nonzero: bool, label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    assert!(
        want.iter().all(|v| v.is_finite()),
        "{label}: reference non-finite"
    );
    if require_mostly_nonzero {
        assert!(
            want.iter().filter(|v| v.abs() > 0.0).count() > want.len() / 2,
            "{label}: reference mostly zero"
        );
    }
    let bad = got
        .iter()
        .zip(want)
        .filter(|(g, w)| g.to_bits() != w.to_bits())
        .count();
    assert!(
        bad == 0,
        "{label}: {bad} of {} outputs differ in their bits",
        want.len()
    );
}

/// `matvec_nvfp4_wide_glu_f32` == gate matvec -> up matvec -> `swiglu_inplace`, at the FFN gate/up shape
/// (17408 x 5120), on four independent gate/up pairs.
#[test]
fn nvfp4_glu_is_bit_identical_to_gate_up_swiglu() {
    const OUT: u32 = 17408;
    const IN: u32 = 5120;
    const PAIRS: usize = 4;
    let dev = device();
    let m = dev
        .compile_and_load(MATVEC_NVFP4_WIDE_KERNEL_SOURCE)
        .expect("compile nvfp4");
    let mv = m.load_function("matvec_nvfp4_wide_f32").expect("matvec");
    let glu = m.load_function("matvec_nvfp4_wide_glu_f32").expect("glu");
    let ma = dev
        .compile_and_load(ACTIVATIONS_KERNEL_SOURCE)
        .expect("compile activations");
    let swiglu = ma.load_function("swiglu_inplace").expect("swiglu");

    let mut rng = Lcg(0x5EED_0061);
    let d_x = activation(&dev, IN);
    let mut d_gate = dev.alloc_zeros::<f32>(OUT as usize).expect("gate");
    let mut d_up = dev.alloc_zeros::<f32>(OUT as usize).expect("up");
    let mut d_fused = dev.alloc_zeros::<f32>(OUT as usize).expect("fused");
    for pair in 0..PAIRS {
        let gate = nvfp4_plane(&dev, &mut rng, OUT, IN, 3.6e-4);
        let up = nvfp4_plane(&dev, &mut rng, OUT, IN, 2.9e-4);

        let mut b = dev.stream.launch_builder(&mv);
        b.arg(&gate).arg(&d_x).arg(&mut d_gate).arg(&OUT).arg(&IN);
        unsafe { b.launch(warp_per_row(OUT)) }.expect("gate");
        let mut b = dev.stream.launch_builder(&mv);
        b.arg(&up).arg(&d_x).arg(&mut d_up).arg(&OUT).arg(&IN);
        unsafe { b.launch(warp_per_row(OUT)) }.expect("up");
        let mut b = dev.stream.launch_builder(&swiglu);
        b.arg(&mut d_gate).arg(&d_up).arg(&OUT);
        unsafe { b.launch(elementwise(OUT)) }.expect("swiglu");

        let mut b = dev.stream.launch_builder(&glu);
        b.arg(&gate)
            .arg(&up)
            .arg(&d_x)
            .arg(&mut d_fused)
            .arg(&OUT)
            .arg(&IN);
        unsafe { b.launch(warp_per_row(OUT)) }.expect("glu");
        dev.stream.synchronize().expect("sync");

        let want = dev.dtoh_copy(&d_gate).expect("dtoh");
        let got = dev.dtoh_copy(&d_fused).expect("dtoh");
        assert_bit_identical(&got, &want, true, &format!("glu pair {pair}"));
    }
}

/// `matvec_nvfp4_wide_residual_f32` == matvec -> `residual_add`, at the FFN down shape (5120 x 17408), on
/// eight independent planes.
#[test]
fn nvfp4_down_residual_is_bit_identical_to_matvec_then_add() {
    const OUT: u32 = 5120;
    const IN: u32 = 17408;
    const PLANES: usize = 8;
    let dev = device();
    let m = dev
        .compile_and_load(MATVEC_NVFP4_WIDE_KERNEL_SOURCE)
        .expect("compile nvfp4");
    let mv = m.load_function("matvec_nvfp4_wide_f32").expect("matvec");
    let mvr = m
        .load_function("matvec_nvfp4_wide_residual_f32")
        .expect("matvec residual");
    let ma = dev
        .compile_and_load(ACTIVATIONS_KERNEL_SOURCE)
        .expect("compile activations");
    let radd = ma.load_function("residual_add").expect("residual_add");

    let mut rng = Lcg(0x5EED_D0E5);
    let d_x = activation(&dev, IN);
    let resid: Vec<f32> = (0..OUT).map(|i| ((i % 97) as f32) * 0.37 - 17.0).collect();
    let d_res0 = dev.htod_copy(&resid).expect("residual");
    let mut d_down = dev.alloc_zeros::<f32>(OUT as usize).expect("down");
    let mut d_acc = dev.alloc_zeros::<f32>(OUT as usize).expect("acc");
    let mut d_out = dev.alloc_zeros::<f32>(OUT as usize).expect("out");
    for i in 0..PLANES {
        let plane = nvfp4_plane(&dev, &mut rng, OUT, IN, 3.5e-4);

        dev.stream
            .memcpy_dtod(&d_res0, &mut d_acc)
            .expect("reset acc");
        let mut b = dev.stream.launch_builder(&mv);
        b.arg(&plane).arg(&d_x).arg(&mut d_down).arg(&OUT).arg(&IN);
        unsafe { b.launch(warp_per_row(OUT)) }.expect("matvec");
        let mut b = dev.stream.launch_builder(&radd);
        b.arg(&mut d_acc).arg(&d_down).arg(&OUT);
        unsafe { b.launch(elementwise(OUT)) }.expect("residual_add");

        let mut b = dev.stream.launch_builder(&mvr);
        b.arg(&plane)
            .arg(&d_x)
            .arg(&d_res0)
            .arg(&mut d_out)
            .arg(&OUT)
            .arg(&IN);
        unsafe { b.launch(warp_per_row(OUT)) }.expect("matvec residual");
        dev.stream.synchronize().expect("sync");

        let want = dev.dtoh_copy(&d_acc).expect("dtoh");
        let got = dev.dtoh_copy(&d_out).expect("dtoh");
        assert_bit_identical(&got, &want, false, &format!("down residual plane {i}"));
    }
}

/// `gdn_input_projections_f32` == FP8 qkv matvec (10240 x 5120) + `matvec_f32_gates_banked` on the F32
/// alpha/beta rows (2 x 48 x 5120) + FP8 z matvec (6144 x 5120): one GDN layer's input projections, six independent layers.
/// All four outputs are compared; the data produces no denormal, where the fast-math gates kernel would flush
/// to zero and the fused kernel would not.
#[test]
fn gdn_input_projections_are_bit_identical_to_three_launches() {
    const IN: u32 = 5120;
    const QKV: u32 = 10240;
    const ZD: u32 = 6144;
    const HEADS: u32 = 48;
    const LAYERS: usize = 6;
    let dev = device();
    let m8 = dev
        .compile_and_load(MATVEC_FP8_KERNEL_SOURCE)
        .expect("compile fp8");
    let fp8 = m8.load_function("matvec_fp8_f32").expect("matvec_fp8_f32");
    // Each kernel is compiled as the engine compiles it: the banked gates kernel for compute_80 with
    // fast-math, the fused kernel with the default flags.
    let mg = dev
        .compile_and_load_with_arch_fast_math(MATVEC_F32_GATES_KERNEL_SOURCE, "compute_80")
        .expect("compile gates");
    let gates = mg
        .load_function("matvec_f32_gates_banked")
        .expect("matvec_f32_gates_banked");
    // The fused kernel is compiled after the two row-code sources it runs.
    let fused_src = format!(
        "{MATVEC_FP8_KERNEL_SOURCE}\n{MATVEC_F32_GATES_KERNEL_SOURCE}\n{GDN_INPUT_PROJECTIONS_KERNEL_SOURCE}"
    );
    let mf = dev.compile_and_load(&fused_src).expect("compile fused");
    let fused = mf
        .load_function("gdn_input_projections_f32")
        .expect("gdn_input_projections_f32");

    let mut rng = Lcg(0x5EED_6D11);
    let mut rng_gates = Lcg(0x5EED_6D12);
    let d_x = activation(&dev, IN);
    let mut r = [
        dev.alloc_zeros::<f32>(QKV as usize).unwrap(),
        dev.alloc_zeros::<f32>(ZD as usize).unwrap(),
        dev.alloc_zeros::<f32>(HEADS as usize).unwrap(),
        dev.alloc_zeros::<f32>(HEADS as usize).unwrap(),
    ];
    let mut f = [
        dev.alloc_zeros::<f32>(QKV as usize).unwrap(),
        dev.alloc_zeros::<f32>(ZD as usize).unwrap(),
        dev.alloc_zeros::<f32>(HEADS as usize).unwrap(),
        dev.alloc_zeros::<f32>(HEADS as usize).unwrap(),
    ];
    let fused_grid = 2 * HEADS + QKV.div_ceil(4) + ZD.div_ceil(4);
    for layer in 0..LAYERS {
        let wqkv = fp8_plane(&dev, &mut rng, QKV, IN, 9.7e-4);
        let wz = fp8_plane(&dev, &mut rng, ZD, IN, 5.3e-4);
        let mut gate_row = || (rng_gates.next_e4m3() as f32 - 127.5) * 1.0e-4;
        let a: Vec<f32> = (0..HEADS * IN).map(|_| gate_row()).collect();
        let b: Vec<f32> = (0..HEADS * IN).map(|_| gate_row()).collect();
        let wa = dev.htod_copy(&a).expect("wa");
        let wb = dev.htod_copy(&b).expect("wb");

        {
            let [qkv, z, alpha, beta] = &mut r;
            let mut k = dev.stream.launch_builder(&fp8);
            k.arg(&wqkv).arg(&d_x).arg(&mut *qkv).arg(&QKV).arg(&IN);
            unsafe { k.launch(warp_per_row(QKV)) }.unwrap();
            let mut k = dev.stream.launch_builder(&gates);
            k.arg(&wa)
                .arg(&wb)
                .arg(&d_x)
                .arg(&mut *alpha)
                .arg(&mut *beta)
                .arg(&HEADS)
                .arg(&IN);
            unsafe {
                k.launch(LaunchConfig {
                    grid_dim: (2 * HEADS, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .unwrap();
            let mut k = dev.stream.launch_builder(&fp8);
            k.arg(&wz).arg(&d_x).arg(&mut *z).arg(&ZD).arg(&IN);
            unsafe { k.launch(warp_per_row(ZD)) }.unwrap();
        }
        {
            let [qkv, z, alpha, beta] = &mut f;
            let mut k = dev.stream.launch_builder(&fused);
            k.arg(&wqkv)
                .arg(&wz)
                .arg(&wa)
                .arg(&wb)
                .arg(&d_x)
                .arg(&mut *qkv)
                .arg(&mut *z)
                .arg(&mut *alpha)
                .arg(&mut *beta)
                .arg(&QKV)
                .arg(&ZD)
                .arg(&HEADS)
                .arg(&IN);
            unsafe {
                k.launch(LaunchConfig {
                    grid_dim: (fused_grid, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .unwrap();
        }
        dev.stream.synchronize().unwrap();

        for (name, (want, got)) in ["qkv", "z", "alpha", "beta"]
            .iter()
            .zip(r.iter().zip(f.iter()))
        {
            let want = dev.dtoh_copy(want).unwrap();
            let got = dev.dtoh_copy(got).unwrap();
            assert_bit_identical(&got, &want, true, &format!("layer {layer} {name}"));
        }
    }
}

/// `matvec_fp8_three_f32` == `matvec_fp8_f32` on each of the attention q+gate (12288 x 5120), k and v
/// (1024 x 5120) planes, six independent layers.
#[test]
fn fp8_three_is_bit_identical_to_three_matvecs() {
    const IN: u32 = 5120;
    const DIMS: [u32; 3] = [12288, 1024, 1024];
    const LAYERS: usize = 6;
    let dev = device();
    let m = dev
        .compile_and_load(MATVEC_FP8_KERNEL_SOURCE)
        .expect("compile fp8");
    let one = m.load_function("matvec_fp8_f32").expect("matvec_fp8_f32");
    let three = m
        .load_function("matvec_fp8_three_f32")
        .expect("matvec_fp8_three_f32");

    let mut rng = Lcg(0x5EED_0A77);
    let d_x = activation(&dev, IN);
    let mut r = DIMS.map(|d| dev.alloc_zeros::<f32>(d as usize).unwrap());
    let mut f = DIMS.map(|d| dev.alloc_zeros::<f32>(d as usize).unwrap());
    let grid: u32 = DIMS.iter().map(|d| d.div_ceil(4)).sum();
    for layer in 0..LAYERS {
        let w = DIMS.map(|rows| fp8_plane(&dev, &mut rng, rows, IN, 1.1e-3));
        for j in 0..3 {
            let mut k = dev.stream.launch_builder(&one);
            k.arg(&w[j]).arg(&d_x).arg(&mut r[j]).arg(&DIMS[j]).arg(&IN);
            unsafe { k.launch(warp_per_row(DIMS[j])) }.unwrap();
        }
        let [o0, o1, o2] = &mut f;
        let mut k = dev.stream.launch_builder(&three);
        k.arg(&w[0])
            .arg(&w[1])
            .arg(&w[2])
            .arg(&d_x)
            .arg(&mut *o0)
            .arg(&mut *o1)
            .arg(&mut *o2)
            .arg(&DIMS[0])
            .arg(&DIMS[1])
            .arg(&DIMS[2])
            .arg(&IN);
        unsafe {
            k.launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .unwrap();
        dev.stream.synchronize().unwrap();
        for j in 0..3 {
            let want = dev.dtoh_copy(&r[j]).unwrap();
            let got = dev.dtoh_copy(&f[j]).unwrap();
            assert_bit_identical(&got, &want, false, &format!("layer {layer} out {j}"));
        }
    }
}

/// `matvec_fp8_f32_residual_rounded` == `matvec_fp8_f32` -> `residual_add_copy` on the GDN output projection
/// (5120 x 6144).
///
/// The contracting `matvec_fp8_f32_residual` (one FMA on the store) runs on the same inputs as a control: it
/// must differ from the two-launch reference somewhere, which shows the comparison can see a one-FMA store.
#[test]
fn fp8_residual_rounded_is_bit_identical_to_matvec_then_add() {
    const OUT: u32 = 5120;
    const IN: u32 = 6144;
    let dev = device();
    let fp8 = dev
        .compile_and_load(MATVEC_FP8_KERNEL_SOURCE)
        .expect("compile fp8");
    let act = dev
        .compile_and_load(ACTIVATIONS_KERNEL_SOURCE)
        .expect("compile activations");
    let matvec = fp8.load_function("matvec_fp8_f32").unwrap();
    let fused = fp8
        .load_function("matvec_fp8_f32_residual_rounded")
        .unwrap();
    let contracting = fp8.load_function("matvec_fp8_f32_residual").unwrap();
    let add = act.load_function("residual_add_copy").unwrap();

    let mut state = 0x00DD_BA11u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    let n = OUT as usize * IN as usize;
    let mut plane: Vec<u8> = (0..n)
        .map(|_| (next() & 0x7F) as u8 | if next() & 1 == 1 { 0x80 } else { 0 })
        .collect();
    for b in plane.iter_mut() {
        if *b & 0x7F == 0x7F {
            *b &= 0xFE; // skip the E4M3 NaN encodings
        }
    }
    plane.extend_from_slice(&0.0137f32.to_le_bytes());
    let x: Vec<f32> = (0..IN)
        .map(|_| (next() % 2001) as f32 * 1.0e-3 - 1.0)
        .collect();
    let residual: Vec<f32> = (0..OUT)
        .map(|_| (next() % 4001) as f32 * 7.3e-3 - 14.6)
        .collect();
    let d_plane = dev.htod_copy(&plane).unwrap();
    let d_x = dev.htod_copy(&x).unwrap();
    let d_res = dev.htod_copy(&residual).unwrap();
    let mut proj = dev.alloc_zeros::<f32>(OUT as usize).unwrap();
    let mut two_launch = dev.alloc_zeros::<f32>(OUT as usize).unwrap();
    let mut folded = dev.alloc_zeros::<f32>(OUT as usize).unwrap();
    let mut contracted = dev.alloc_zeros::<f32>(OUT as usize).unwrap();
    let (o, i) = (OUT, IN);
    unsafe {
        let mut b = dev.stream.launch_builder(&matvec);
        b.arg(&d_plane).arg(&d_x).arg(&mut proj).arg(&o).arg(&i);
        b.launch(warp_per_row(OUT)).unwrap();
        let mut b = dev.stream.launch_builder(&add);
        b.arg(&d_res).arg(&proj).arg(&mut two_launch).arg(&o);
        b.launch(elementwise(OUT)).unwrap();
        let mut b = dev.stream.launch_builder(&fused);
        b.arg(&d_plane)
            .arg(&d_x)
            .arg(&d_res)
            .arg(&mut folded)
            .arg(&o)
            .arg(&i);
        b.launch(warp_per_row(OUT)).unwrap();
        let mut b = dev.stream.launch_builder(&contracting);
        b.arg(&d_plane)
            .arg(&d_x)
            .arg(&d_res)
            .arg(&mut contracted)
            .arg(&o)
            .arg(&i);
        b.launch(warp_per_row(OUT)).unwrap();
    }
    let reference = dev.dtoh_copy(&two_launch).unwrap();
    let folded = dev.dtoh_copy(&folded).unwrap();
    let contracted = dev.dtoh_copy(&contracted).unwrap();
    assert!(
        reference.iter().all(|v| v.is_finite()),
        "reference non-finite"
    );
    let differ = |o: &[f32]| {
        reference
            .iter()
            .zip(o)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count()
    };
    assert!(
        differ(&contracted) > 0,
        "the contracting store matched everywhere, so this comparison cannot see a one-FMA store"
    );
    assert_eq!(
        differ(&folded),
        0,
        "rounded fold differs from matvec -> residual_add_copy"
    );
}

/// `rmsnorm` holds a thread's first values in registers between its two passes. Its output must equal the
/// two-pass form bit for bit, at a full hidden size and at sizes whose elements run past the held values
/// into the loop tail (20000 > 8 x 1024).
///
/// Reference: `fused_residual_rmsnorm_f32` with `b = 0`. It is the same per-thread strided sum, the same warp
/// and cross-warp reduction and the same `x * rms * weight` store, reading every value in both passes. Adding
/// `+0.0` returns every input's bits unchanged except `-0.0`, which the test data does not contain.
#[test]
fn rmsnorm_register_held_is_bit_identical_to_two_pass() {
    let dev = device();
    let m = dev
        .compile_and_load(NORM_KERNEL_SOURCE)
        .expect("compile norm");
    let rmsnorm = m.load_function("rmsnorm").expect("rmsnorm");
    let two_pass = m
        .load_function("fused_residual_rmsnorm_f32")
        .expect("fused_residual_rmsnorm_f32");
    let eps = 1.0e-6f32;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1024, 1, 1),
        shared_mem_bytes: 1024 / 32 * 4,
    };
    for dim in [5120u32, 4096, 9216, 20000] {
        let v: Vec<f32> = (0..dim).map(|i| ((i % 113) as f32) * 0.07 - 3.9).collect();
        assert!(
            v.iter().all(|x| x.to_bits() != (-0.0f32).to_bits()),
            "test data must not contain -0.0"
        );
        let w: Vec<f32> = (0..dim).map(|i| 0.8 + ((i % 29) as f32) * 0.01).collect();
        let d_v = dev.htod_copy(&v).unwrap();
        let d_w = dev.htod_copy(&w).unwrap();
        let d_zero = dev.alloc_zeros::<f32>(dim as usize).unwrap();
        let mut d_sum = dev.alloc_zeros::<f32>(dim as usize).unwrap();
        let mut o_ref = dev.alloc_zeros::<f32>(dim as usize).unwrap();
        let mut o_new = dev.alloc_zeros::<f32>(dim as usize).unwrap();

        let mut k = dev.stream.launch_builder(&two_pass);
        k.arg(&d_v)
            .arg(&d_zero)
            .arg(&mut d_sum)
            .arg(&d_w)
            .arg(&mut o_ref)
            .arg(&eps)
            .arg(&dim);
        unsafe { k.launch(cfg) }.unwrap();
        let mut k = dev.stream.launch_builder(&rmsnorm);
        k.arg(&d_v).arg(&d_w).arg(&mut o_new).arg(&eps).arg(&dim);
        unsafe { k.launch(cfg) }.unwrap();
        dev.stream.synchronize().unwrap();

        let sum = dev.dtoh_copy(&d_sum).unwrap();
        assert!(
            sum.iter().zip(&v).all(|(s, x)| s.to_bits() == x.to_bits()),
            "dim {dim}: x + 0 changed an input, so the reference is not a plain RMSNorm of x"
        );
        let want = dev.dtoh_copy(&o_ref).unwrap();
        let got = dev.dtoh_copy(&o_new).unwrap();
        assert_bit_identical(&got, &want, true, &format!("rmsnorm dim {dim}"));
    }
}

/// Test harness over the NVFP4 matvec's own device decoders: every byte through the E2M1 low/high-nibble
/// decoders and the E4M3 block-scale decoder.
const NVFP4_DECODE_HARNESS: &str = r#"
extern "C" __global__ void nvfp4_decode_every_byte(float* e2m1_lo, float* e2m1_hi, float* e4m3)
{
    const unsigned int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < 256u) {
        e2m1_lo[b] = nvfp4w_e2m1_lo(b);
        e2m1_hi[b] = nvfp4w_e2m1_hi(b);
        e4m3[b] = nvfp4w_e4m3_to_f32(b);
    }
}
"#;

/// The NVFP4 matvec builds E2M1 values arithmetically from the code bits and decodes E4M3 block scales
/// inline. Both decoders, as compiled into the served kernel source, against the host decoder on every byte.
///
/// Every E2M1 code is bit-identical to the table except the sign-only zero 0x8, which the bit construction
/// yields as `-0.0` where the table has `+0.0`. That cannot change a matvec output: each group sum starts at
/// `+0.0`, and adding a `-0.0` product to any accumulator leaves it unchanged. Code 0x1 is the subnormal
/// path and must be exactly 0.5, which fails if the kernel is ever built with flush-to-zero.
#[test]
fn nvfp4_matvec_decoders_match_the_host_decoder_on_every_byte() {
    let dev = device();
    let src = format!("{MATVEC_NVFP4_WIDE_KERNEL_SOURCE}\n{NVFP4_DECODE_HARNESS}");
    let m = dev.compile_and_load(&src).expect("compile harness");
    let f = m.load_function("nvfp4_decode_every_byte").expect("harness");
    let mut d_lo = dev.alloc_zeros::<f32>(256).unwrap();
    let mut d_hi = dev.alloc_zeros::<f32>(256).unwrap();
    let mut d_e4m3 = dev.alloc_zeros::<f32>(256).unwrap();
    let mut b = dev.stream.launch_builder(&f);
    b.arg(&mut d_lo).arg(&mut d_hi).arg(&mut d_e4m3);
    unsafe { b.launch(elementwise(256)) }.expect("launch harness");
    dev.stream.synchronize().unwrap();
    let lo = dev.dtoh_copy(&d_lo).unwrap();
    let hi = dev.dtoh_copy(&d_hi).unwrap();
    let e4m3 = dev.dtoh_copy(&d_e4m3).unwrap();

    for byte in 0u8..=255 {
        for (got, code, half) in [
            (lo[byte as usize], byte & 0x0F, "low"),
            (hi[byte as usize], byte >> 4, "high"),
        ] {
            let want = e2m1_to_f32(code);
            if code == 0x8 {
                assert_eq!(
                    got.to_bits(),
                    (-0.0f32).to_bits(),
                    "byte {byte:#04x} {half} nibble: code 0x8 must decode to -0.0, got {got:e}"
                );
                assert_eq!(want.to_bits(), 0, "the host table's code 0x8 is +0.0");
            } else {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "byte {byte:#04x} {half} nibble (code {code:#03x}): device {got:e} vs host {want:e}"
                );
            }
        }
        let got = e4m3[byte as usize];
        let want = e4m3_to_f32(byte);
        assert!(
            (got.is_nan() && want.is_nan()) || got.to_bits() == want.to_bits(),
            "E4M3 {byte:#04x}: device {:#010x} ({got:e}) vs host {:#010x} ({want:e})",
            got.to_bits(),
            want.to_bits()
        );
    }
    assert_eq!(lo[0x01], 0.5, "code 0x1 (subnormal) must be 0.5");
    assert!(e4m3[0x7F].is_nan() && e4m3[0xFF].is_nan(), "E4M3 NaNs");
}

/// Test harness over the FP8 matvec's own decode table: the block fills its shared-memory table exactly as
/// the matvec does and writes it out.
const FP8_TABLE_HARNESS: &str = r#"
extern "C" __global__ void fp8_dump_decode_table(float* out)
{
    __shared__ float lut[256];
    fp8_fill_lut(lut);
    for (unsigned int c = threadIdx.x; c < 256u; c += blockDim.x) {
        out[c] = lut[c];
    }
}
"#;

/// The FP8 matvec decodes E4M3 through a 256-entry shared-memory table filled by the block. The table, as
/// the served kernel source builds it at the serving block size, against the host decoder on all 256 codes:
/// bit-identical, `-0.0` preserved, 0x7F/0xFF NaN.
#[test]
fn fp8_matvec_decode_table_matches_the_host_decoder_on_every_code() {
    let dev = device();
    let src = format!("{MATVEC_FP8_KERNEL_SOURCE}\n{FP8_TABLE_HARNESS}");
    let m = dev.compile_and_load(&src).expect("compile harness");
    let f = m.load_function("fp8_dump_decode_table").expect("harness");
    let mut d_out = dev.alloc_zeros::<f32>(256).unwrap();
    let mut b = dev.stream.launch_builder(&f);
    b.arg(&mut d_out);
    unsafe {
        b.launch(LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .expect("launch harness");
    dev.stream.synchronize().unwrap();
    let table = dev.dtoh_copy(&d_out).unwrap();
    for code in 0u8..=255 {
        let got = table[code as usize];
        let want = e4m3_to_f32(code);
        assert!(
            (got.is_nan() && want.is_nan()) || got.to_bits() == want.to_bits(),
            "E4M3 {code:#04x}: table {:#010x} ({got:e}) vs host {:#010x} ({want:e})",
            got.to_bits(),
            want.to_bits()
        );
    }
    assert!(table[0x7F].is_nan() && table[0xFF].is_nan(), "E4M3 NaNs");
    assert_eq!(table[0x80].to_bits(), (-0.0f32).to_bits(), "0x80 is -0.0");
    assert_eq!(table[0x7E], 448.0, "0x7E is the largest finite value");
}
