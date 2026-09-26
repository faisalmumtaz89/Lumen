//! The native prefill's producer kernels (`native_prefill_kernels`), on their own, byte for byte
//! against a host implementation of their arithmetic.
//!
//! The host oracle (`native_oracle/producers.rs`) computes every IEEE step itself and takes only the
//! approximate hardware instructions (rcp/rsqrt/ex2 approximations, `div.full`, `div.approx`) from a
//! probe kernel run on exactly the operands the oracle computed; their results cannot be derived from
//! the PTX specification, which bounds them only to a few ulp.
//!
//! - `group_compiles_for_the_fp4_target_only`: `compute_120a` is selected, its PTX holds the E2M1
//!   conversion, the same source built for `compute_120` is refused by the driver, each kernel's
//!   dynamic shared memory attribute reads back its table value, and a launch with more dynamic
//!   shared memory than the default limit fails until the attribute allows it.
//! - `qualifying_launches_match_the_oracle`: the qualifying problem's outputs equal the oracle's,
//!   whose digests are the recorded ones, and the group loads.
//! - `producers_match_the_oracle`: every kernel at M = 1, 63, 64, 65, 127, 128, 131, 2047 and 2048
//!   rows of realistic activations, both GDN norm reduction shapes, both SwiGLU arithmetics.
//! - `quantizer_edge_cases_match_the_oracle`: exact E2M1 and E4M3 ties, zero blocks, underflow, BF16
//!   subnormals, clipping and saturation, infinities and NaNs.
//! - `corrupted_oracles_are_caught`: the comparison fails for a global scale x 1.01, a linear scale
//!   layout, an `rstd` one ulp off, and the GDN norm's other reduction shape.
//!
//! Requires a GPU of compute capability 12.0 and NVRTC 12.8 or newer:
//!
//!   cargo test --release -p lumen-runtime --features cuda --test cuda_native_producers_test \
//!     -- --ignored --test-threads=1
#![cfg(feature = "cuda")]

mod native_oracle;

use cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;
use cudarc::driver::{CudaFunction, CudaSlice, DevicePtr, LaunchConfig, PushKernelArg};
use lumen_runtime::cuda::ffi::CudaDevice;
use lumen_runtime::cuda::native_prefill_kernels::{
    checksum, fp4_scale_bytes, smoke, NativePrefillKernels, ADD_RMSNORM_FP4, ADD_RMSNORM_FP8,
    EMBED_GATHER_BF16, FINAL_ROW_F32, GDN_NORM_GATE_FP8, KERNELS, RMSNORM_FP8, SIGMOID_GATE_FP8,
    SILU_MUL_FP4_FAST, SILU_MUL_FP4_PRECISE,
};
use lumen_runtime::cuda::native_prefill_weights::swizzled_offset;
use lumen_runtime::cuda::shaders::NATIVE_PREFILL_KERNEL_SOURCE;
use native_oracle::producers::*;

const GDN_ROWS: usize = 48;
const AO: usize = 6144;
const ROWS: [usize; 9] = [1, 63, 64, 65, 127, 128, 131, 2047, 2048];

/// Representative activation scales: MLP gate/up and down (NVFP4), GDN qkv/z, GDN out_proj
/// and attention o_proj (FP8).
const GATE_UP_SCALE: f32 = 0.0014;
const DOWN_SCALE: f32 = 0.0025;
const QKV_SCALE: f32 = 0.11;
const OUT_SCALE: f32 = 0.023;
const O_SCALE: f32 = 0.0175;

// ---------------------------------------------------------------------------------------------
// Inputs.

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    fn uniform(&mut self) -> f64 {
        (self.next() as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

/// `rows x cols` BF16 activations: normal with standard deviation `sd`, every 97th column a
/// 16x outlier channel.
fn activations(rng: &mut Rng, rows: usize, cols: usize, sd: f64) -> Vec<u16> {
    (0..rows * cols)
        .map(|i| {
            let boost = if (i % cols) % 97 == 5 { 16.0 } else { 1.0 };
            to_bf((rng.normal() * sd * boost) as f32)
        })
        .collect()
}

/// A Gemma norm weight as the artifact stores it: F32 (w + 1) of a BF16 w.
fn gemma_weights(rng: &mut Rng) -> Vec<f32> {
    (0..H)
        .map(|_| bf(to_bf((rng.normal() * 0.3) as f32)) + 1.0)
        .collect()
}

/// A plain norm weight: F32 values of BF16 numbers.
fn plain_weights(rng: &mut Rng, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| bf(to_bf((1.0 + rng.normal() * 0.2) as f32)))
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Device plumbing.

fn dev() -> CudaDevice {
    CudaDevice::new(0).expect("CUDA device 0")
}

fn ptr<T>(d: &CudaDevice, s: &CudaSlice<T>) -> u64 {
    s.device_ptr(&d.stream).0
}

/// An output buffer filled with 0xA5, so a byte the kernel never writes shows.
fn sentinel(d: &CudaDevice, n: usize) -> CudaSlice<u8> {
    d.htod_copy(&vec![0xA5u8; n]).unwrap()
}

fn up16(d: &CudaDevice, v: &[u16]) -> CudaSlice<u8> {
    d.htod_copy(&bytes16(v)).unwrap()
}

/// Mismatching bytes between device output and oracle, and the first one.
fn mismatches(got: &[u8], want: &[u8]) -> (usize, Option<usize>) {
    assert_eq!(got.len(), want.len(), "length");
    let n = got.iter().zip(want).filter(|(a, b)| a != b).count();
    (n, got.iter().zip(want).position(|(a, b)| a != b))
}

fn assert_same(what: &str, got: &[u8], want: &[u8]) {
    let (n, first) = mismatches(got, want);
    assert_eq!(
        n,
        0,
        "{what}: {n} of {} bytes differ, first at {first:?}",
        got.len()
    );
    println!("PASS {what}: {} bytes equal", got.len());
}

/// One set of inputs for every producer, `m` tokens.
struct Inputs {
    m: usize,
    x: Vec<u16>,
    resid: Vec<u16>,
    w1: Vec<f32>,
    core: Vec<u16>,
    z: Vec<u16>,
    wn: Vec<f32>,
    gu: Vec<u16>,
    o: Vec<u16>,
    gate: Vec<u16>,
    s_gate: f32,
    s_down: f32,
}

impl Inputs {
    fn realistic(m: usize, seed: u64) -> Self {
        let mut rng = Rng(seed);
        Self {
            m,
            x: activations(&mut rng, m, H, 0.8),
            resid: activations(&mut rng, m, H, 2.0),
            w1: gemma_weights(&mut rng),
            core: activations(&mut rng, m * GDN_ROWS, 128, 0.4),
            z: activations(&mut rng, m * GDN_ROWS, 128, 1.5),
            wn: plain_weights(&mut rng, 128),
            gu: activations(&mut rng, m, 2 * I, 1.2),
            o: activations(&mut rng, m, AO, 0.3),
            gate: activations(&mut rng, m, AO, 2.0),
            s_gate: 1.0 / GATE_UP_SCALE,
            s_down: 1.0 / DOWN_SCALE,
        }
    }
}

/// Run every producer on `inp` and compare each output with the oracle.
fn check_all(
    d: &CudaDevice,
    k: &NativePrefillKernels,
    hw: &Hw,
    tab: &Tables,
    inp: &Inputs,
    tag: &str,
) {
    let m = inp.m;
    let mu = m as u32;
    let w1 = d.htod_copy(&inp.w1).unwrap();
    let x = up16(d, &inp.x);

    // Layer-0 norm, then the fused-add norms.
    for add in [false, true] {
        let resid = up16(d, &inp.resid);
        let q8 = sentinel(d, m * H);
        let normed = sentinel(d, m * H * 2);
        unsafe {
            if add {
                k.add_rmsnorm_fp8(
                    d,
                    ptr(d, &x),
                    ptr(d, &resid),
                    ptr(d, &w1),
                    EPS,
                    mu,
                    QKV_SCALE,
                    ptr(d, &q8),
                    ptr(d, &normed),
                )
            } else {
                k.rmsnorm_fp8(
                    d,
                    ptr(d, &x),
                    ptr(d, &w1),
                    EPS,
                    mu,
                    QKV_SCALE,
                    ptr(d, &q8),
                    ptr(d, &normed),
                )
            }
            .unwrap();
        }
        let (new_resid, out) = gemma_norm(hw, &inp.x, add.then_some(&inp.resid[..]), &inp.w1, m, 0);
        let name = if add {
            "add_rmsnorm_fp8"
        } else {
            "rmsnorm_fp8"
        };
        assert_same(
            &format!("{tag} {name} bf16"),
            &d.dtoh_copy(&normed).unwrap(),
            &bytes16(&out),
        );
        assert_same(
            &format!("{tag} {name} fp8"),
            &d.dtoh_copy(&q8).unwrap(),
            &fp8_all(&out, fp8_inv(hw, QKV_SCALE)),
        );
        let want_resid = if add { new_resid } else { inp.resid.clone() };
        assert_same(
            &format!("{tag} {name} residual"),
            &d.dtoh_copy(&resid).unwrap(),
            &bytes16(&want_resid),
        );

        // Without the BF16 output (`normed` 0): the same codes.
        let resid = up16(d, &inp.resid);
        let q8_only = sentinel(d, m * H);
        unsafe {
            if add {
                k.add_rmsnorm_fp8(
                    d,
                    ptr(d, &x),
                    ptr(d, &resid),
                    ptr(d, &w1),
                    EPS,
                    mu,
                    QKV_SCALE,
                    ptr(d, &q8_only),
                    0,
                )
            } else {
                k.rmsnorm_fp8(
                    d,
                    ptr(d, &x),
                    ptr(d, &w1),
                    EPS,
                    mu,
                    QKV_SCALE,
                    ptr(d, &q8_only),
                    0,
                )
            }
            .unwrap();
        }
        assert_same(
            &format!("{tag} {name} fp8 without the bf16 output"),
            &d.dtoh_copy(&q8_only).unwrap(),
            &d.dtoh_copy(&q8).unwrap(),
        );
    }

    // Fused-add norm to NVFP4.
    {
        let resid = up16(d, &inp.resid);
        let q = sentinel(d, m * H / 2);
        let sf = sentinel(d, fp4_scale_bytes(mu, H as u32));
        unsafe {
            k.add_rmsnorm_fp4(
                d,
                ptr(d, &x),
                ptr(d, &resid),
                ptr(d, &w1),
                EPS,
                mu,
                inp.s_gate,
                ptr(d, &q),
                ptr(d, &sf),
            )
            .unwrap();
        }
        let (new_resid, out) = gemma_norm(hw, &inp.x, Some(&inp.resid), &inp.w1, m, 0);
        let (wq, wsf) = Fp4::new(hw, inp.s_gate).quantize(&out, m, H);
        assert_same(
            &format!("{tag} add_rmsnorm_fp4 residual"),
            &d.dtoh_copy(&resid).unwrap(),
            &bytes16(&new_resid),
        );
        assert_same(
            &format!("{tag} add_rmsnorm_fp4 codes"),
            &d.dtoh_copy(&q).unwrap(),
            &wq,
        );
        assert_same(
            &format!("{tag} add_rmsnorm_fp4 scales"),
            &d.dtoh_copy(&sf).unwrap(),
            &wsf,
        );
    }

    // GDN gated norm, both reduction shapes.
    {
        let rows = m * GDN_ROWS;
        let core = up16(d, &inp.core);
        let z = up16(d, &inp.z);
        let wn = d.htod_copy(&inp.wn).unwrap();
        for lanes in [32u32, 16] {
            let q8 = sentinel(d, rows * 128);
            unsafe {
                k.gdn_norm_gate_fp8(
                    d,
                    ptr(d, &core),
                    ptr(d, &z),
                    ptr(d, &wn),
                    EPS,
                    rows as u32,
                    lanes,
                    OUT_SCALE,
                    ptr(d, &q8),
                )
                .unwrap();
            }
            let y = gated_norm(hw, tab, &inp.core, &inp.z, &inp.wn, rows, lanes as usize, 0);
            assert_same(
                &format!("{tag} gdn_norm_gate_fp8 lanes {lanes}"),
                &d.dtoh_copy(&q8).unwrap(),
                &fp8_all(&y, fp8_inv(hw, OUT_SCALE)),
            );
        }
    }

    // Final row: the last one.
    {
        let resid = up16(d, &inp.resid);
        let out = d.htod_copy(&vec![f32::NAN; H]).unwrap();
        unsafe {
            k.final_row_f32(d, ptr(d, &x), ptr(d, &resid), mu - 1, ptr(d, &out))
                .unwrap();
        }
        let want: Vec<u8> = (0..H)
            .flat_map(|c| {
                let i = (m - 1) * H + c;
                (bf(inp.x[i]) + bf(inp.resid[i])).to_le_bytes()
            })
            .collect();
        let got: Vec<u8> = d
            .dtoh_copy(&out)
            .unwrap()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_same(&format!("{tag} final_row_f32"), &got, &want);
    }

    // SwiGLU to NVFP4, both arithmetics.
    {
        let gu = up16(d, &inp.gu);
        let fp4 = Fp4::new(hw, inp.s_down);
        if m == 2048 {
            let (f, p) = (
                silu_bf16(tab, false, &inp.gu, m),
                silu_bf16(tab, true, &inp.gu, m),
            );
            let n = f.iter().zip(&p).filter(|(a, b)| a != b).count();
            assert!(
                n > 0,
                "the two SwiGLU arithmetics agree on every input: a swap would pass"
            );
            println!("INFO {tag}: the two SwiGLU arithmetics differ on {n} BF16 values");
        }
        for precise in [false, true] {
            let q = sentinel(d, m * I / 2);
            let sf = sentinel(d, fp4_scale_bytes(mu, I as u32));
            unsafe {
                k.silu_mul_fp4(
                    d,
                    precise,
                    ptr(d, &gu),
                    mu,
                    inp.s_down,
                    ptr(d, &q),
                    ptr(d, &sf),
                )
                .unwrap();
            }
            let (wq, wsf) = fp4.quantize(&silu_bf16(tab, precise, &inp.gu, m), m, I);
            let name = if precise { "precise" } else { "fast" };
            assert_same(
                &format!("{tag} silu_mul_fp4_{name} codes"),
                &d.dtoh_copy(&q).unwrap(),
                &wq,
            );
            assert_same(
                &format!("{tag} silu_mul_fp4_{name} scales"),
                &d.dtoh_copy(&sf).unwrap(),
                &wsf,
            );
        }
    }

    // Sigmoid gate to FP8.
    {
        let o = up16(d, &inp.o);
        let g = up16(d, &inp.gate);
        let q8 = sentinel(d, m * AO);
        unsafe {
            k.sigmoid_gate_fp8(d, ptr(d, &o), ptr(d, &g), mu, O_SCALE, ptr(d, &q8))
                .unwrap();
        }
        assert_same(
            &format!("{tag} sigmoid_gate_fp8"),
            &d.dtoh_copy(&q8).unwrap(),
            &fp8_all(&sigmoid_bf16(tab, &inp.o, &inp.gate), fp8_inv(hw, O_SCALE)),
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Tests.

#[test]
#[ignore = "needs a GPU of compute capability 12.0 with NVRTC >= 12.8"]
fn group_compiles_for_the_fp4_target_only() {
    let d = dev();
    let arch = d.fp4_native_arch().unwrap();
    assert_eq!(arch, Some("compute_120a"), "target");
    let opts = |arch: &'static str| cudarc::nvrtc::CompileOptions {
        arch: Some(arch),
        ..Default::default()
    };
    let ptx =
        cudarc::nvrtc::compile_ptx_with_opts(NATIVE_PREFILL_KERNEL_SOURCE, opts("compute_120a"))
            .expect("compute_120a compile");
    let text = ptx.to_src();
    let e2m1 = text.matches("cvt.rn.satfinite.e2m1x2.f32").count();
    assert!(e2m1 > 0, "the compute_120a PTX holds no E2M1 conversion");
    assert!(text.contains(".target sm_120a"), "PTX target line");
    println!(
        "PASS compute_120a PTX: {e2m1} E2M1 conversions, {} bytes",
        text.len()
    );
    let generic =
        cudarc::nvrtc::compile_ptx_with_opts(NATIVE_PREFILL_KERNEL_SOURCE, opts("compute_120"))
            .expect("NVRTC accepts the source for compute_120; the assembler must refuse it");
    match d.ctx.load_module(generic) {
        Ok(_) => panic!("the driver loaded compute_120 PTX holding an E2M1 conversion"),
        Err(e) => println!("PASS compute_120 PTX refused by the driver: {e}"),
    }

    let k = NativePrefillKernels::compile(&d).expect("compile");
    for spec in &KERNELS {
        let got = k
            .function(spec.name)
            .unwrap()
            .get_attribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES)
            .unwrap();
        assert_eq!(
            got, spec.dynamic_shared as i32,
            "{} dynamic shared memory",
            spec.name
        );
    }
    println!("PASS every kernel's dynamic shared memory attribute equals its table value");

    // The attribute is what admits a launch above the 48 KiB default: 101,376 bytes (the prefill
    // attention's tile budget) fail before it is raised and run after.
    let f = k.function("native_final_row_f32").unwrap();
    let big = 101_376u32;
    let src = d.htod_copy(&vec![0u16; 2 * H]).unwrap();
    let mut out = d.alloc_zeros::<f32>(H).unwrap();
    let row = 0u32;
    let launch = |f: &CudaFunction, out: &mut CudaSlice<f32>| unsafe {
        d.stream
            .launch_builder(f)
            .arg(&src)
            .arg(&src)
            .arg(&row)
            .arg(out)
            .launch(LaunchConfig {
                grid_dim: (20, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: big,
            })
            .map(|_| ())
            .and_then(|_| d.stream.synchronize())
    };
    let before = launch(f, &mut out);
    assert!(
        before.is_err(),
        "a {big}-byte launch ran without the attribute"
    );
    f.set_attribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, big as i32)
        .unwrap();
    launch(f, &mut out).expect("launch after raising the attribute");
    println!(
        "PASS {big} B of dynamic shared memory: refused ({}) before the attribute, run after",
        before.unwrap_err()
    );
}

#[test]
#[ignore = "needs a GPU of compute capability 12.0 with NVRTC >= 12.8"]
fn qualifying_launches_match_the_oracle() {
    let d = dev();
    let hw = Hw::new(&d);
    let tab = Tables::new(&hw);
    let k = NativePrefillKernels::compile(&d).expect("compile");
    let (h, m) = (H, smoke::ROWS);
    let w1 = smoke::weights(h, 2);
    let inv = fp8_inv(&hw, smoke::INPUT_SCALE);
    let fp4 = Fp4::new(&hw, smoke::S);
    let mut digests = Vec::new();
    for (i, spec) in KERNELS.iter().enumerate() {
        let want: Vec<u8> = match i {
            RMSNORM_FP8 => {
                let (_, out) = gemma_norm(&hw, &smoke::bf16(m * h, 1), None, &w1, m, 0);
                [fp8_all(&out, inv), bytes16(&out)].concat()
            }
            ADD_RMSNORM_FP8 => {
                let (x, r) = (smoke::bf16(m * h, 1), smoke::bf16(m * h, 3));
                let (r, out) = gemma_norm(&hw, &x, Some(&r), &w1, m, 0);
                [bytes16(&r), fp8_all(&out, inv), bytes16(&out)].concat()
            }
            ADD_RMSNORM_FP4 => {
                let (x, r) = (smoke::bf16(m * h, 1), smoke::bf16(m * h, 3));
                let (r, out) = gemma_norm(&hw, &x, Some(&r), &w1, m, 0);
                let (q, sf) = fp4.quantize(&out, m, H);
                [bytes16(&r), q, sf].concat()
            }
            GDN_NORM_GATE_FP8 => {
                let rows = m * GDN_ROWS;
                let (x, z) = (smoke::bf16(rows * 128, 4), smoke::bf16(rows * 128, 5));
                let w = smoke::weights(128, 6);
                [32, 16]
                    .iter()
                    .flat_map(|&l| fp8_all(&gated_norm(&hw, &tab, &x, &z, &w, rows, l, 0), inv))
                    .collect()
            }
            FINAL_ROW_F32 => {
                let (x, r) = (smoke::bf16(m * h, 7), smoke::bf16(m * h, 8));
                let last = (m - 1) * h;
                (0..h)
                    .flat_map(|c| (bf(x[last + c]) + bf(r[last + c])).to_le_bytes())
                    .collect()
            }
            SILU_MUL_FP4_FAST | SILU_MUL_FP4_PRECISE => {
                let (q, sf) = fp4.quantize(
                    &silu_bf16(&tab, i == SILU_MUL_FP4_PRECISE, &smoke::gu(), m),
                    m,
                    I,
                );
                [q, sf].concat()
            }
            SIGMOID_GATE_FP8 => fp8_all(
                &sigmoid_bf16(&tab, &smoke::bf16(m * AO, 10), &smoke::bf16(m * AO, 11)),
                inv,
            ),
            EMBED_GATHER_BF16 => {
                let table = smoke::table();
                smoke::IDS
                    .iter()
                    .flat_map(|&id| bytes16(&table[id as usize * h..(id as usize + 1) * h]))
                    .collect()
            }
            _ => unreachable!(),
        };
        let got = smoke::run(&k, &d, i).unwrap();
        assert_same(&format!("qualifying {}", spec.name), &got, &want);
        println!("SMOKE {} {}", spec.name, checksum(&want));
        digests.push(checksum(&want));
    }
    let loaded = NativePrefillKernels::load(&d);
    assert_ne!(
        digests[SILU_MUL_FP4_FAST], digests[SILU_MUL_FP4_PRECISE],
        "the qualifying problem tells the two SwiGLU arithmetics apart"
    );
    assert_eq!(digests, smoke::DIGESTS, "recorded digests");
    loaded.expect("load qualifies");
    println!("PASS the recorded digests are the oracle's, and the group loads");
}

#[test]
#[ignore = "needs a GPU of compute capability 12.0 with NVRTC >= 12.8"]
fn producers_match_the_oracle() {
    let d = dev();
    let hw = Hw::new(&d);
    let tab = Tables::new(&hw);
    let k = NativePrefillKernels::compile(&d).expect("compile");
    for (i, &m) in ROWS.iter().enumerate() {
        check_all(
            &d,
            &k,
            &hw,
            &tab,
            &Inputs::realistic(m, 1000 + i as u64),
            &format!("M={m}"),
        );
    }
}

/// Values that exercise a quantizer's edge cases once multiplied by a power of two.
fn fp4_edge_blocks(scale_exp: i32) -> Vec<[f32; 16]> {
    let p = |v: f32, e: i32| v * 2f32.powi(e);
    let mut blocks = vec![[0.0; 16], [-0.0; 16]];
    // Exact E2M1 midpoints (0.25, 0.75, ..., 5) at every block exponent, the block maximum 6.
    for e in scale_exp - 8..scale_exp {
        let mut b = [0.0; 16];
        b[0] = p(6.0, e);
        for (i, v) in [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0].iter().enumerate() {
            b[1 + 2 * i] = p(*v, e);
            b[2 + 2 * i] = -p(*v, e);
        }
        b[15] = p(0.5, e);
        blocks.push(b);
    }
    // Underflow next to a large value; clipping (scale saturation); infinities; NaNs; extremes.
    let mut under = [p(2.0, -24); 16];
    under[3] = 6.0;
    blocks.push(under);
    blocks.push([1.0e4; 16]);
    let mut clip = [-3.0e38; 16];
    clip[5] = 1.0;
    blocks.push(clip);
    let mut inf = [1.5; 16];
    inf[0] = f32::INFINITY;
    inf[9] = -2.0;
    blocks.push(inf);
    let mut ninf = [0.25; 16];
    ninf[7] = f32::NEG_INFINITY;
    blocks.push(ninf);
    let mut nan = [2.0; 16];
    nan[2] = f32::NAN;
    blocks.push(nan);
    blocks.push([f32::NAN; 16]);
    let mut nan_even = [3.0; 16];
    for i in (0..16).step_by(2) {
        nan_even[i] = f32::NAN;
    }
    blocks.push(nan_even);
    blocks.push([bf(0x7f7f); 16]);
    blocks
}

#[test]
#[ignore = "needs a GPU of compute capability 12.0 with NVRTC >= 12.8"]
fn quantizer_edge_cases_match_the_oracle() {
    let d = dev();
    let hw = Hw::new(&d);
    let tab = Tables::new(&hw);
    let k = NativePrefillKernels::compile(&d).expect("compile");

    // NVFP4: with g = 8192 both SwiGLU arithmetics give u * 8192 (1 + expf(-8192) = 1), so two rows
    // carry the edge values (up = value / 8192, exact for the tie blocks: their S = 512 scales are powers
    // of two); a third row of tiny operands gives BF16 subnormal products (flushed by the fast path).
    for s in [512.0f32, 1.0 / GATE_UP_SCALE] {
        let blocks = fp4_edge_blocks(-1);
        let mut rng = Rng(77);
        let rows = 3;
        let mut gu = vec![0u16; rows * 2 * I];
        for row in 0..2 {
            for c in 0..I {
                gu[row * 2 * I + c] = 0x4600;
                gu[row * 2 * I + I + c] =
                    to_bf(blocks[(c / 16 + row) % blocks.len()][c % 16] / 8192.0);
            }
        }
        for c in 0..I {
            gu[2 * 2 * I + c] = to_bf((rng.normal() * 1e-3) as f32);
            gu[2 * 2 * I + I + c] = to_bf((rng.normal() * 2f64.powi(-122)) as f32);
        }
        let dg = up16(&d, &gu);
        let fp4 = Fp4::new(&hw, s);
        for precise in [false, true] {
            let x = silu_bf16(&tab, precise, &gu, rows);
            let subnormal = x.iter().filter(|&&b| bf(b).is_subnormal()).count();
            let q = sentinel(&d, rows * I / 2);
            let sf = sentinel(&d, fp4_scale_bytes(rows as u32, I as u32));
            unsafe {
                k.silu_mul_fp4(
                    &d,
                    precise,
                    ptr(&d, &dg),
                    rows as u32,
                    s,
                    ptr(&d, &q),
                    ptr(&d, &sf),
                )
                .unwrap();
            }
            let (wq, wsf) = fp4.quantize(&x, rows, I);
            let exact_ties = (0..rows * I / 16)
                .map(|blk| {
                    let (row, b) = (blk / (I / 16), blk % (I / 16));
                    let code = wsf[swizzled_offset(row, b, I / 16)];
                    let scale = fp4.scale_of[code as usize];
                    (0..16)
                        .filter(|&i| {
                            let v = mul_ftz(bf(x[row * I + b * 16 + i]), scale);
                            is_tie(&E2M1, v)
                        })
                        .count()
                })
                .sum::<usize>();
            let name = if precise { "precise" } else { "fast" };
            println!(
                "INFO FP4 edge S={s} {name}: {exact_ties} exact E2M1 ties, {subnormal} BF16 subnormal inputs"
            );
            if s == 512.0 {
                assert!(
                    exact_ties >= 1000,
                    "too few E2M1 ties exercised: {exact_ties}"
                );
            }
            if precise {
                assert!(subnormal > 0, "no BF16 subnormal reached the quantizer");
            }
            assert_same(
                &format!("FP4 edge S={s} {name} codes"),
                &d.dtoh_copy(&q).unwrap(),
                &wq,
            );
            assert_same(
                &format!("FP4 edge S={s} {name} scales"),
                &d.dtoh_copy(&sf).unwrap(),
                &wsf,
            );
        }
    }

    // FP8: the sigmoid gate passes o through when gate = 16384 (sigmoid 1), so o carries E4M3
    // midpoints, saturation, subnormals and non-finite values; scale 1 keeps them exact.
    {
        let mut vals: Vec<f32> = Vec::new();
        for c in 0..126u8 {
            let (a, b) = (e4m3_value(c), e4m3_value(c + 1));
            vals.push((a + b) / 2.0);
            vals.push(-(a + b) / 2.0);
            vals.push(a);
        }
        vals.extend([
            448.0,
            464.0,
            449.0,
            480.0,
            1.0e5,
            -1.0e5,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.0,
            -0.0,
            bf(0x0001),
            bf(0x8001),
            bf(0x007f),
            2f32.powi(-10),
            3.0 * 2f32.powi(-11),
            bf(0x7f7f),
            bf(0xff7f),
        ]);
        let m = 2;
        let o: Vec<u16> = (0..m * AO).map(|i| to_bf(vals[i % vals.len()])).collect();
        let gate = vec![0x4680u16; m * AO];
        assert_eq!(tab.sigmoid[0x4680], 1.0, "sigmoid(16384)");
        for s in [1.0f32, O_SCALE] {
            let inv = fp8_inv(&hw, s);
            let x = sigmoid_bf16(&tab, &o, &gate);
            let ties = x
                .iter()
                .filter(|&&b| is_tie(e4m3_table(), bf(b) * inv))
                .count();
            let (dob, dgb) = (up16(&d, &o), up16(&d, &gate));
            let q8 = sentinel(&d, m * AO);
            unsafe {
                k.sigmoid_gate_fp8(&d, ptr(&d, &dob), ptr(&d, &dgb), m as u32, s, ptr(&d, &q8))
                    .unwrap();
            }
            println!("INFO FP8 edge scale {s}: {ties} exact E4M3 ties");
            if s == 1.0 {
                assert!(ties >= 252, "too few E4M3 ties exercised: {ties}");
            }
            assert_same(
                &format!("FP8 edge scale {s}"),
                &d.dtoh_copy(&q8).unwrap(),
                &fp8_all(&x, inv),
            );
        }
    }

    // Norms with shaped weights: zero columns (zero blocks), tiny (BF16 subnormal outputs), huge
    // (infinite outputs), NaN; rows of zeros and of huge values.
    {
        let m = 4;
        let mut inp = Inputs::realistic(m, 5);
        for (c, w) in inp.w1.iter_mut().enumerate() {
            *w = match (c / 16) % 6 {
                0 => 0.0,
                1 => 2f32.powi(-130),
                2 => 1.0e38,
                3 if c % 16 == 3 => f32::NAN,
                _ => *w,
            };
        }
        for c in 0..H {
            inp.x[c] = 0;
            inp.resid[c] = 0;
            inp.x[3 * H + c] = to_bf(3.0e38);
        }
        let (_, out) = gemma_norm(&hw, &inp.x, Some(&inp.resid), &inp.w1, m, 0);
        let subnormal = out.iter().filter(|&&b| bf(b).is_subnormal()).count();
        let nonfinite = out.iter().filter(|&&b| !bf(b).is_finite()).count();
        println!("INFO norm edge: {subnormal} subnormal and {nonfinite} non-finite BF16 outputs");
        assert!(
            subnormal > 0 && nonfinite > 0,
            "norm edge classes not reached"
        );
        check_all(&d, &k, &hw, &tab, &inp, "edge");
    }
}

#[test]
#[ignore = "needs a GPU of compute capability 12.0 with NVRTC >= 12.8"]
fn corrupted_oracles_are_caught() {
    let d = dev();
    let hw = Hw::new(&d);
    let tab = Tables::new(&hw);
    let k = NativePrefillKernels::compile(&d).expect("compile");
    let m = 131;
    let inp = Inputs::realistic(m, 9);
    let mu = m as u32;

    let gu = up16(&d, &inp.gu);
    let q = sentinel(&d, m * I / 2);
    let sf = sentinel(&d, fp4_scale_bytes(mu, I as u32));
    unsafe {
        k.silu_mul_fp4(
            &d,
            false,
            ptr(&d, &gu),
            mu,
            inp.s_down,
            ptr(&d, &q),
            ptr(&d, &sf),
        )
        .unwrap();
    }
    let x = silu_bf16(&tab, false, &inp.gu, m);
    let (gq, gsf) = (d.dtoh_copy(&q).unwrap(), d.dtoh_copy(&sf).unwrap());
    let (wq, wsf) = Fp4::new(&hw, inp.s_down).quantize(&x, m, I);
    assert_eq!(
        (mismatches(&gq, &wq).0, mismatches(&gsf, &wsf).0),
        (0, 0),
        "uncorrupted"
    );

    let (bq, bsf) = Fp4::new(&hw, inp.s_down * 1.01).quantize(&x, m, I);
    let n = mismatches(&gq, &bq).0 + mismatches(&gsf, &bsf).0;
    assert!(n > 0, "global scale x 1.01 not caught");
    println!("PASS global scale x 1.01: {n} bytes differ");

    let mut linear = Fp4::new(&hw, inp.s_down);
    linear.linear = true;
    let (_, lsf) = linear.quantize(&x, m, I);
    let n = mismatches(&gsf, &lsf).0;
    assert!(n > 0, "linear scale layout not caught");
    println!("PASS linear scale layout: {n} bytes differ");

    let w1 = d.htod_copy(&inp.w1).unwrap();
    let (xd, resid) = (up16(&d, &inp.x), up16(&d, &inp.resid));
    let q8 = sentinel(&d, m * H);
    let normed = sentinel(&d, m * H * 2);
    unsafe {
        k.add_rmsnorm_fp8(
            &d,
            ptr(&d, &xd),
            ptr(&d, &resid),
            ptr(&d, &w1),
            EPS,
            mu,
            QKV_SCALE,
            ptr(&d, &q8),
            ptr(&d, &normed),
        )
        .unwrap();
    }
    let got = d.dtoh_copy(&normed).unwrap();
    for ulps in [1, -1] {
        let (_, out) = gemma_norm(&hw, &inp.x, Some(&inp.resid), &inp.w1, m, ulps);
        let n = mismatches(&got, &bytes16(&out)).0;
        assert!(n > 0, "norm rstd {ulps:+} ulp not caught");
        println!("PASS norm rstd {ulps:+} ulp: {n} bytes differ");
    }

    // The gated norm's FP8 output hides most one-ulp changes of rstd, so its controls run over many
    // rows.
    let big = Inputs::realistic(2048, 10);
    let rows = 2048 * GDN_ROWS;
    let (core, z, wn) = (
        up16(&d, &big.core),
        up16(&d, &big.z),
        d.htod_copy(&big.wn).unwrap(),
    );
    let q8 = sentinel(&d, rows * 128);
    unsafe {
        k.gdn_norm_gate_fp8(
            &d,
            ptr(&d, &core),
            ptr(&d, &z),
            ptr(&d, &wn),
            EPS,
            rows as u32,
            16,
            OUT_SCALE,
            ptr(&d, &q8),
        )
        .unwrap();
    }
    let got = d.dtoh_copy(&q8).unwrap();
    let inv = fp8_inv(&hw, OUT_SCALE);
    for (ulps, what) in [(0, "uncorrupted"), (1, "rstd +1 ulp"), (-1, "rstd -1 ulp")] {
        let y = gated_norm(&hw, &tab, &big.core, &big.z, &big.wn, rows, 16, ulps);
        let n = mismatches(&got, &fp8_all(&y, inv)).0;
        assert_eq!(n == 0, ulps == 0, "gated norm {what}: {n} bytes differ");
        println!("PASS gated norm {what}: {n} bytes differ");
    }

    // Each reduction shape against both shapes' oracles over 8192 tokens of rows: the kernel matches
    // the shape it was launched with and differs from the other, so `lanes_per_row` is honoured.
    let t = 8192;
    let rows = t * GDN_ROWS;
    let mut rng = Rng(11);
    let x = activations(&mut rng, rows, 128, 0.4);
    let z = activations(&mut rng, rows, 128, 1.5);
    let (xd, zd) = (up16(&d, &x), up16(&d, &z));
    let oracle: Vec<Vec<u8>> = [32usize, 16]
        .iter()
        .map(|&l| fp8_all(&gated_norm(&hw, &tab, &x, &z, &big.wn, rows, l, 0), inv))
        .collect();
    for (s, lanes) in [32u32, 16].into_iter().enumerate() {
        let q8 = sentinel(&d, rows * 128);
        unsafe {
            k.gdn_norm_gate_fp8(
                &d,
                ptr(&d, &xd),
                ptr(&d, &zd),
                ptr(&d, &wn),
                EPS,
                rows as u32,
                lanes,
                OUT_SCALE,
                ptr(&d, &q8),
            )
            .unwrap();
        }
        let got = d.dtoh_copy(&q8).unwrap();
        let same = mismatches(&got, &oracle[s]).0;
        let other = mismatches(&got, &oracle[1 - s]).0;
        assert_eq!(same, 0, "gated norm {lanes} lanes vs its own oracle");
        assert!(
            other > 0,
            "gated norm {lanes} lanes: the other shape's oracle is not told apart"
        );
        println!("PASS gated norm launched with {lanes} lanes: equal to its shape, {other} bytes differ from the other");
    }
}
