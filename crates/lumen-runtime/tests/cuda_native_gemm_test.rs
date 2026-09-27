//! The native prefill's cuBLASLt route, tested on its own: the runtime-loaded
//! library, the plan table with its measured and verified algorithms and their cache, the prefill
//! weight views, and ownership.
//!
//! - `library_report`: the libraries found, with paths and versions (run once per library set).
//! - `selection_is_measured_recorded_and_cached`: a cold build selects and verifies every plan and
//!   writes the table; a second build reads it back. Needs `LUMEN_CACHE_DIR` naming an empty
//!   directory, and runs first.
//! - `every_plan_matches_the_f32_reference`: every shape at every bucket and at 1-7, 63-65, 127-131
//!   and 2047-2048 rows, against F32 SGEMMs of the decoded operands, plus f64 spot checks.
//! - `corrupted_operands_fail_the_check`: linear weight scales, alpha x 1.01, one activation scale
//!   doubled, one FP8 weight code changed.
//! - `fp8_cancellation_and_range`: every selected FP8 algorithm on cancelling, wide-range inputs.
//! - `real_artifact_admission_and_views`: the real artifact (`LUMEN_NATIVE_MODEL`) is admitted, its
//!   192 swizzled MLP scale copies hold the linear planes' bytes, its a/b copies are exact, and its
//!   real weights pass through the plans.
//! - `concurrent_backends_match_serial`, `drop_right_after_enqueue` (run under compute-sanitizer),
//!   and, with `test-fault-injection`, `injected_construction_failures_leak_nothing`.
//!
//! Requires a GPU of compute capability 12.0 and cuBLASLt 12.8 or newer:
//!
//!   LUMEN_CACHE_DIR=<empty dir> cargo test --release -p lumen-runtime --features cuda \
//!     --test cuda_native_gemm_test -- --ignored --test-threads=1 --exact <name>
#![cfg(feature = "cuda")]

use cudarc::driver::{CudaSlice, DevicePtr};
use lumen_format::planar_dequant::{dequantize_fp8, dequantize_nvfp4};
use lumen_runtime::cuda::cublaslt::{self, LtInput, LtOperands, ALGO_CONFIG_REDUCTION_SCHEME};
use lumen_runtime::cuda::cublaslt_algo_cache::{
    Activation, Check, Verdict, Verifier, WeightOperand,
};
use lumen_runtime::cuda::ffi::CudaDevice;
use lumen_runtime::cuda::native_prefill::{admit, ProviderSlices, SliceSource};
use lumen_runtime::cuda::native_prefill_gemm::{
    bucket, bucket_rows, GemmShape, NativeGemm, TableSource, ATTN_KV, BUCKETS, MAX_ROWS, SHAPES,
};
use lumen_runtime::cuda::native_prefill_weights::PrefillWeightViews;
use lumen_runtime::weight::provider_sync::SyncWeightProvider;

struct Lcg(u64);
impl Lcg {
    fn byte(&mut self) -> u8 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u8
    }
    /// An E4M3 code other than the two NaNs.
    fn e4m3(&mut self) -> u8 {
        match self.byte() {
            c if c & 0x7F == 0x7F => c - 1,
            c => c,
        }
    }
}

/// A weight's host planes (codes, then linear block scales and the global for NVFP4, or the scale
/// for FP8) and its device copies.
struct Weight {
    host: Vec<u8>,
    global: f32,
    planes: Vec<CudaSlice<u8>>,
    /// NVFP4: one swizzled block-scale copy per plane copy.
    swizzled: Vec<CudaSlice<u8>>,
}

impl Weight {
    fn upload(
        dev: &CudaDevice,
        shape: &GemmShape,
        host: Vec<u8>,
        global: f32,
        copies: usize,
    ) -> Self {
        let swizzled = match shape.input {
            LtInput::Nvfp4 => {
                let codes = shape.n * shape.k / 2;
                let linear = &host[codes..codes + shape.n * shape.k / 16];
                let s = lumen_runtime::cuda::native_prefill_weights::swizzle_block_scales(
                    linear,
                    shape.n,
                    shape.k / 16,
                );
                (0..copies).map(|_| dev.htod_copy(&s).unwrap()).collect()
            }
            LtInput::Fp8 => Vec::new(),
        };
        let planes = (0..copies).map(|_| dev.htod_copy(&host).unwrap()).collect();
        Self {
            host,
            global,
            planes,
            swizzled,
        }
    }

    /// Random codes, block scales between 2^-4 and 2^4 under a global of 2^-10, or an FP8 scale of
    /// 2^-9; enough copies that each is re-read only after 300 MB of other weights.
    fn synthetic(dev: &CudaDevice, shape: &GemmShape, seed: u64, rotate: bool) -> Self {
        let mut r = Lcg(seed);
        let (n, k) = (shape.n, shape.k);
        let mut host: Vec<u8>;
        let global: f32;
        match shape.input {
            LtInput::Nvfp4 => {
                host = (0..n * k / 2).map(|_| r.byte()).collect();
                host.extend((0..n * k / 16).map(|_| 0x18 + r.byte() % 0x41));
                global = 1.0 / 1024.0;
            }
            LtInput::Fp8 => {
                host = (0..n * k).map(|_| r.e4m3()).collect();
                global = 1.0 / 512.0;
            }
        }
        host.extend_from_slice(&global.to_le_bytes());
        let copies = if rotate {
            (300 << 20) / host.len() + 1
        } else {
            1
        };
        Self::upload(dev, shape, host, global, copies)
    }

    fn operand(&self, dev: &CudaDevice, shape: &GemmShape, copy: usize) -> WeightOperand {
        let plane = self.planes[copy].device_ptr(&dev.stream).0;
        match self.swizzled.get(copy) {
            Some(s) => WeightOperand {
                plane,
                scale: s.device_ptr(&dev.stream).0,
                unit_alpha: self.global,
            },
            None => WeightOperand {
                plane,
                scale: plane + (shape.n * shape.k) as u64,
                unit_alpha: 1.0,
            },
        }
    }

    fn operands(&self, dev: &CudaDevice, shape: &GemmShape) -> Vec<WeightOperand> {
        (0..self.planes.len())
            .map(|c| self.operand(dev, shape, c))
            .collect()
    }
}

fn weights(dev: &CudaDevice, rotate: bool) -> Vec<Weight> {
    SHAPES
        .iter()
        .enumerate()
        .map(|(i, s)| Weight::synthetic(dev, s, 1000 + i as u64, rotate))
        .collect()
}

fn operand_sets(dev: &CudaDevice, w: &[Weight]) -> [Vec<WeightOperand>; 7] {
    std::array::from_fn(|i| w[i].operands(dev, &SHAPES[i]))
}

fn build(dev: &CudaDevice, w: &[Weight]) -> (NativeGemm, TableSource, f64) {
    let (g, report) = unsafe { NativeGemm::build(dev, &operand_sets(dev, w)) }.expect("build");
    println!("build: {:.2} s, {}", report.seconds, report.library);
    (g, report.source, report.seconds)
}

/// Host activation codes and scales: `rows` rows of which the first `live` are random and the rest
/// zero.
struct HostAct {
    codes: Vec<u8>,
    scales: Vec<u8>,
    scale: f32,
}

fn host_act(input: LtInput, rows: usize, live: usize, k: usize, seed: u64) -> HostAct {
    let mut r = Lcg(seed);
    match input {
        LtInput::Nvfp4 => HostAct {
            codes: (0..rows * k / 2)
                .map(|i| if i < live * k / 2 { r.byte() } else { 0 })
                .collect(),
            scales: (0..rows * k / 16)
                .map(|i| {
                    if i < live * k / 16 {
                        0x28 + r.byte() % 0x21
                    } else {
                        0
                    }
                })
                .collect(),
            scale: 1.0,
        },
        LtInput::Fp8 => HostAct {
            codes: (0..rows * k)
                .map(|i| if i < live * k { r.e4m3() } else { 0 })
                .collect(),
            scales: Vec::new(),
            scale: 1.0 / 64.0,
        },
    }
}

fn upload_act(dev: &CudaDevice, input: LtInput, rows: usize, k: usize, a: &HostAct) -> Activation {
    Activation::upload(dev, input, rows, k, &a.codes, &a.scales, a.scale).unwrap()
}

fn fill_nan(dev: &CudaDevice, d: &CudaSlice<u16>) {
    dev.ctx.bind_to_thread().unwrap();
    let ptr = d.device_ptr(&dev.stream).0;
    unsafe {
        cudarc::driver::result::memset_d8_async(ptr, 0xFF, d.len() * 2, dev.stream.cu_stream())
    }
    .unwrap();
}

fn ops(dev: &CudaDevice, w: &WeightOperand, x: &Activation, d: &CudaSlice<u16>) -> LtOperands {
    LtOperands {
        w: w.plane,
        w_scale: w.scale,
        x: x.x(dev),
        x_scale: x.x_scale(dev),
        d: d.device_ptr(&dev.stream).0,
    }
}

/// Run `shape` at `rows` into a NaN-filled `d` and check it against `v`.
fn run_and_check(
    dev: &CudaDevice,
    g: &mut NativeGemm,
    s: usize,
    rows: usize,
    alpha: f32,
    o: &LtOperands,
    d: &CudaSlice<u16>,
    v: &mut Verifier,
) -> Check {
    fill_nan(dev, d);
    unsafe { g.run(s, rows, alpha, o) }.unwrap();
    let checked_rows = bucket_rows(bucket(rows).unwrap());
    unsafe { v.check(dev, checked_rows, o.d, SHAPES[s].ldd) }.unwrap()
}

#[test]
#[ignore = "needs a GPU; run once per library set"]
fn library_report() {
    let report = cublaslt::library_report();
    println!("LIBRARIES {report}");
    assert!(report.contains("NVRTC") && report.contains("cuBLASLt") && report.contains("driver"));
}

#[test]
#[ignore = "needs a GPU with cuBLASLt >= 12.8 and an empty LUMEN_CACHE_DIR"]
fn selection_is_measured_recorded_and_cached() {
    let dev = CudaDevice::new(0).unwrap();
    let w = weights(&dev, true);
    let (g, source, cold_s) = build(&dev, &w);
    let TableSource::Selected {
        reason,
        stored,
        plans,
    } = source
    else {
        panic!("the first build must select: LUMEN_CACHE_DIR must name an empty directory");
    };
    println!("cold: {cold_s:.2} s; cache unused because: {reason}");
    let stored = stored.expect("the table is written");
    println!("TABLE {}", stored.display());
    assert_eq!(plans.len(), SHAPES.len() * BUCKETS);
    let mut rejected_reduction = 0;
    let mut failed_verification = 0;
    for p in &plans {
        let red = p.selected.config[ALGO_CONFIG_REDUCTION_SCHEME];
        assert!(
            red == 0 || red == 2,
            "{} {}: reduction {red}",
            p.shape,
            p.rows
        );
        assert_eq!(
            p.outcomes
                .iter()
                .filter(|o| o.verdict == Verdict::Selected)
                .count(),
            1
        );
        for o in &p.outcomes {
            match o.verdict {
                Verdict::ReductionRejected(_) => rejected_reduction += 1,
                Verdict::Failed(_) => failed_verification += 1,
                _ => {}
            }
            println!(
                "CAND {} {} {:?} {:?} {:?}",
                p.shape, p.rows, o.config, o.median_us, o.verdict
            );
        }
        println!(
            "SEL {} {} config {:?} ws {} waves {} time_us {} worst {} candidates {}",
            p.shape,
            p.rows,
            p.selected.config,
            p.selected.workspace,
            p.selected.waves,
            p.selected.median_us,
            p.selected.worst,
            p.selected.candidates
        );
    }
    println!(
        "candidates rejected for their reduction: {rejected_reduction}; failing verification: \
         {failed_verification}"
    );
    for (s, shape) in SHAPES.iter().enumerate() {
        for rows in [128, 512, 2048] {
            let sel = g.selected(s, bucket(rows).unwrap());
            println!("TIME {} {rows} {:.2}", shape.name, sel.median_us);
        }
    }
    let before: Vec<_> = (0..SHAPES.len() * BUCKETS)
        .map(|i| *g.selected(i / BUCKETS, i % BUCKETS))
        .collect();
    drop(g);
    let (g, source, warm_s) = build(&dev, &w);
    let TableSource::Cached(path) = source else {
        panic!("the second build must read the table: {source:?}");
    };
    assert_eq!(path, stored);
    let after: Vec<_> = (0..SHAPES.len() * BUCKETS)
        .map(|i| *g.selected(i / BUCKETS, i % BUCKETS))
        .collect();
    assert_eq!(before, after, "the cache returns the selected table");
    println!("LOAD cold {cold_s:.2} s warm {warm_s:.2} s");
}

#[test]
#[ignore = "needs a GPU with cuBLASLt >= 12.8"]
fn every_plan_matches_the_f32_reference() {
    let dev = CudaDevice::new(0).unwrap();
    let w = weights(&dev, false);
    let (mut g, _, _) = build(&dev, &w);
    let mut worst = 0.0f32;
    for (s, shape) in SHAPES.iter().enumerate() {
        let w0 = w[s].operand(&dev, shape, 0);
        let d = unsafe { dev.alloc_uninit::<u16>(MAX_ROWS * shape.ldd) }.unwrap();
        // Every bucket, on an activation the selection never saw.
        let host = host_act(shape.input, MAX_ROWS, MAX_ROWS, shape.k, 77 + s as u64);
        let x = upload_act(&dev, shape.input, MAX_ROWS, shape.k, &host);
        let mut v = unsafe { Verifier::new(&dev, shape.n, shape.k, w0.plane, &x) }.unwrap();
        for b in 0..BUCKETS {
            let rows = bucket_rows(b);
            let c = run_and_check(
                &dev,
                &mut g,
                s,
                rows,
                w0.unit_alpha,
                &ops(&dev, &w0, &x, &d),
                &d,
                &mut v,
            );
            assert_eq!(c.violations, 0, "{} at {rows} rows: {c:?}", shape.name);
            worst = worst.max(c.worst);
        }
        // f64 spot checks of the 2048-row output (still in `d`): every column of rows 0 and 2047,
        // and 64 random elements.
        let dev_out = dev.dtoh_copy(&d).unwrap();
        let wh = decode_host(shape, &w[s].host, shape.n);
        let xh = decode_act_host(shape, &host);
        let mut r = Lcg(5);
        let mut samples: Vec<(usize, usize)> = (0..shape.n)
            .flat_map(|n| [(0, n), (MAX_ROWS - 1, n)])
            .collect();
        samples.extend((0..64).map(|_| {
            let m = (r.byte() as usize * 8 + r.byte() as usize % 8) % MAX_ROWS;
            let n = (r.byte() as usize * 251 + r.byte() as usize) % shape.n;
            (m, n)
        }));
        for (m, n) in samples {
            let (mut acc, mut mag) = (0.0f64, 0.0f64);
            for k in 0..shape.k {
                let t = xh[m * shape.k + k] as f64 * wh[n * shape.k + k] as f64;
                acc += t;
                mag += t.abs();
            }
            // The weight's decode includes its global scale, which is the GEMM's alpha.
            let reference = acc;
            let got = f32::from_bits((dev_out[m * shape.ldd + n] as u32) << 16) as f64;
            let bound =
                reference.abs().max(got.abs()) * 2f64.powi(-8) + 64.0 * 2f64.powi(-24) * mag;
            assert!(
                (got - reference).abs() <= bound,
                "{} f64 check at ({m}, {n}): got {got}, f64 {reference}, bound {bound}",
                shape.name
            );
        }
        // Lengths that are not a multiple of 16: the padded activation rows are zero, the output rows
        // past the length are zero, and the rows below it match.
        for t in (1..=7).chain(63..=65).chain(127..=131).chain(2047..=2048) {
            let rows = bucket_rows(bucket(t).unwrap());
            let host = host_act(shape.input, rows, t, shape.k, 900 + t as u64);
            let x = upload_act(&dev, shape.input, rows, shape.k, &host);
            let mut v = unsafe { Verifier::new(&dev, shape.n, shape.k, w0.plane, &x) }.unwrap();
            let c = run_and_check(
                &dev,
                &mut g,
                s,
                t,
                w0.unit_alpha,
                &ops(&dev, &w0, &x, &d),
                &d,
                &mut v,
            );
            assert_eq!(c.violations, 0, "{} at {t} rows: {c:?}", shape.name);
            let out = dev.dtoh_copy(&d).unwrap();
            let tail = &out[t * shape.ldd..rows * shape.ldd];
            assert!(
                tail.chunks(shape.ldd)
                    .all(|row| row[..shape.n].iter().all(|&v| v == 0 || v == 0x8000)),
                "{} at {t} rows: a padded row is not zero",
                shape.name
            );
        }
        println!("{}: every bucket and length verified", shape.name);
    }
    println!("worst error / bound over all plans: {worst}");
}

/// Host decode of the first `rows` rows of a weight's planes.
fn decode_host(shape: &GemmShape, planes: &[u8], rows: usize) -> Vec<f32> {
    let (n, k) = (shape.n, shape.k);
    match shape.input {
        LtInput::Nvfp4 => {
            let codes = &planes[..rows * k / 2];
            let scales = &planes[n * k / 2..n * k / 2 + rows * k / 16];
            let g = f32::from_le_bytes(planes[n * k / 2 + n * k / 16..][..4].try_into().unwrap());
            dequantize_nvfp4(codes, scales, g).unwrap()
        }
        LtInput::Fp8 => {
            let s = f32::from_le_bytes(planes[n * k..][..4].try_into().unwrap());
            dequantize_fp8(&planes[..rows * k], s)
        }
    }
}

fn decode_act_host(shape: &GemmShape, a: &HostAct) -> Vec<f32> {
    match shape.input {
        LtInput::Nvfp4 => dequantize_nvfp4(&a.codes, &a.scales, a.scale).unwrap(),
        LtInput::Fp8 => dequantize_fp8(&a.codes, a.scale),
    }
}

#[test]
#[ignore = "needs a GPU with cuBLASLt >= 12.8"]
fn corrupted_operands_fail_the_check() {
    let dev = CudaDevice::new(0).unwrap();
    let w = weights(&dev, false);
    let (mut g, _, _) = build(&dev, &w);
    let rows = 128;
    for s in [0usize, 2] {
        let shape = &SHAPES[s];
        let w0 = w[s].operand(&dev, shape, 0);
        let d = unsafe { dev.alloc_uninit::<u16>(rows * shape.ldd) }.unwrap();
        let host = host_act(shape.input, rows, rows, shape.k, 31);
        let x = upload_act(&dev, shape.input, rows, shape.k, &host);
        let mut v = unsafe { Verifier::new(&dev, shape.n, shape.k, w0.plane, &x) }.unwrap();
        let good = ops(&dev, &w0, &x, &d);
        let c = run_and_check(&dev, &mut g, s, rows, w0.unit_alpha, &good, &d, &mut v);
        assert_eq!(
            c.violations, 0,
            "{}: the uncorrupted run passes: {c:?}",
            shape.name
        );
        let mut controls: Vec<(&str, f32, LtOperands)> = vec![
            ("alpha x 1.01", w0.unit_alpha * 1.01, good),
            ("an overflowing alpha", 3.0e38, good),
        ];
        // Keep the corrupted copies alive until their GEMMs ran.
        let mut keep: Vec<CudaSlice<u8>> = Vec::new();
        let mut keep_x: Vec<Activation> = Vec::new();
        match shape.input {
            LtInput::Nvfp4 => {
                let linear = w0.plane + (shape.n * shape.k / 2) as u64;
                controls.push((
                    "linear weight scales",
                    w0.unit_alpha,
                    LtOperands {
                        w_scale: linear,
                        ..good
                    },
                ));
                let mut bad = HostAct {
                    codes: host.codes.clone(),
                    scales: host.scales.clone(),
                    scale: host.scale,
                };
                bad.scales[17 * shape.k / 16 + 5] += 8; // one UE4M3 scale doubled
                keep_x.push(upload_act(&dev, shape.input, rows, shape.k, &bad));
                let xb = keep_x.last().unwrap();
                controls.push((
                    "one activation scale doubled",
                    w0.unit_alpha,
                    LtOperands {
                        x_scale: xb.x_scale(&dev),
                        ..good
                    },
                ));
            }
            LtInput::Fp8 => {
                let mut bad = w[s].host.clone();
                bad[100 * shape.k + 7] = 0x7E; // one weight code set to 448
                keep.push(dev.htod_copy(&bad).unwrap());
                let p = keep.last().unwrap().device_ptr(&dev.stream).0;
                controls.push((
                    "one FP8 weight code changed",
                    1.0,
                    LtOperands {
                        w: p,
                        w_scale: p + (shape.n * shape.k) as u64,
                        ..good
                    },
                ));
            }
        }
        for (what, alpha, o) in controls {
            let c = run_and_check(&dev, &mut g, s, rows, alpha, &o, &d, &mut v);
            println!("CONTROL {} {what}: {} violations", shape.name, c.violations);
            assert!(
                c.violations > 0,
                "{} {what}: must fail the check",
                shape.name
            );
        }
        // Rows the GEMM never wrote: run 16 rows, check 32.
        fill_nan(&dev, &d);
        unsafe { g.run(s, 16, w0.unit_alpha, &good) }.unwrap();
        let c = unsafe { v.check(&dev, 32, good.d, shape.ldd) }.unwrap();
        println!(
            "CONTROL {} 16 rows unwritten: {} violations",
            shape.name, c.violations
        );
        assert_eq!(
            c.violations,
            16 * shape.n as u64,
            "{}: every unwritten element is a violation",
            shape.name
        );
    }
}

/// A positive E4M3 code (sign clear, not NaN), over every exponent including the subnormals.
fn positive_e4m3(r: &mut Lcg) -> u8 {
    r.byte() % 0x7F
}

/// `sum_k x[k] * w[k]` accumulated as a tensor-core GEMM does (exact 32-product chunks added to a
/// running sum), with the running sum rounded to `bits` significant bits after every chunk (24 is
/// F32), stored as BF16 bits.
fn emulated_dot(x: &[f32], w: &[f32], bits: i32) -> u16 {
    let mut s = 0.0f64;
    for c in (0..x.len()).step_by(32) {
        s += (c..c + 32).map(|k| x[k] as f64 * w[k] as f64).sum::<f64>();
        if s != 0.0 {
            let q = 2f64.powi(s.abs().log2().floor() as i32 - bits + 1);
            s = (s / q).round() * q;
        }
    }
    let u = (s as f32).to_bits();
    ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) as u16
}

#[test]
#[ignore = "needs a GPU with cuBLASLt >= 12.8"]
fn fp8_cancellation_and_range() {
    let dev = CudaDevice::new(0).unwrap();
    let base = weights(&dev, false);
    let (mut g, _, _) = build(&dev, &base);
    let mut r = Lcg(11);
    for (s, shape) in SHAPES.iter().enumerate() {
        if shape.input != LtInput::Fp8 {
            continue;
        }
        let (n, k) = (shape.n, shape.k);
        let h = k / 2;
        // Every product of the first half of k is positive, so partial sums grow over every exponent;
        // the second half negates them (the weight's sign flipped, the activation repeated) except for
        // one position in 64, so the sum cancels late down to a small residual.
        let mut host: Vec<u8> = Vec::with_capacity(n * k + 4);
        for _ in 0..n {
            let first: Vec<u8> = (0..h).map(|_| positive_e4m3(&mut r)).collect();
            host.extend_from_slice(&first);
            host.extend(first.iter().enumerate().map(|(j, &c)| {
                if j % 64 == 0 {
                    r.e4m3()
                } else {
                    c | 0x80
                }
            }));
        }
        host.extend_from_slice(&(1.0f32 / 256.0).to_le_bytes());
        let wt = Weight::upload(&dev, shape, host.clone(), 1.0, 1);
        let w0 = wt.operand(&dev, shape, 0);
        let codes: Vec<u8> = (0..MAX_ROWS)
            .flat_map(|_| {
                let first: Vec<u8> = (0..h).map(|_| positive_e4m3(&mut r)).collect();
                [first.clone(), first].concat()
            })
            .collect();
        let x_scale = 1.0f32 / 32.0;
        let x = Activation::upload(&dev, LtInput::Fp8, MAX_ROWS, k, &codes, &[], x_scale).unwrap();
        let d = unsafe { dev.alloc_uninit::<u16>(MAX_ROWS * shape.ldd) }.unwrap();
        let mut v = unsafe { Verifier::new(&dev, n, k, w0.plane, &x) }.unwrap();
        let mut worst = 0.0f32;
        for b in 0..BUCKETS {
            let rows = bucket_rows(b);
            let c = run_and_check(
                &dev,
                &mut g,
                s,
                rows,
                1.0,
                &ops(&dev, &w0, &x, &d),
                &d,
                &mut v,
            );
            assert_eq!(c.violations, 0, "{} at {rows} rows: {c:?}", shape.name);
            worst = worst.max(c.worst);
        }
        println!(
            "{}: every selected algorithm passes, worst error / bound {worst}",
            shape.name
        );
        // Control, on the smallest shape: the same check applied to emulated results of these inputs
        // passes an F32 accumulator and fails one of 14 bits.
        if s == ATTN_KV {
            let wd = decode_host(shape, &host, n);
            let xd = dequantize_fp8(&codes[..16 * k], x_scale);
            for (bits, must_pass) in [(24, true), (14, false)] {
                let out: Vec<u16> = (0..16)
                    .flat_map(|m| {
                        let (xd, wd) = (&xd, &wd);
                        (0..n).map(move |j| {
                            emulated_dot(&xd[m * k..(m + 1) * k], &wd[j * k..(j + 1) * k], bits)
                        })
                    })
                    .collect();
                let emulated = dev.htod_copy(&out).unwrap();
                let c =
                    unsafe { v.check(&dev, 16, emulated.device_ptr(&dev.stream).0, n) }.unwrap();
                println!(
                    "CONTROL {bits}-bit accumulator: {} violations of {}",
                    c.violations,
                    16 * n
                );
                assert_eq!(
                    c.violations == 0,
                    must_pass,
                    "{bits}-bit accumulator: {c:?}"
                );
            }
        }
    }
}

#[test]
#[ignore = "needs a GPU, cuBLASLt >= 12.8 and LUMEN_NATIVE_MODEL naming the real artifact"]
fn real_artifact_admission_and_views() {
    let path = std::env::var("LUMEN_NATIVE_MODEL").expect("LUMEN_NATIVE_MODEL");
    let provider = SyncWeightProvider::open(std::path::Path::new(&path)).unwrap();
    let lbc = provider.lbc();
    let hp = lbc.header.hyperparams;
    let layers: Vec<_> = lbc
        .layer_indices
        .iter()
        .map(|l| l.subtensors.clone())
        .collect();
    let slices = ProviderSlices::new(&provider);
    let src: &dyn SliceSource = &slices;
    admit(&hp, lbc.header.embedding.quant, &layers, src).expect("the real artifact is admitted");
    println!("admitted: {path}");

    let dev = CudaDevice::new(0).unwrap();
    let free0 = dev.free_memory().unwrap();
    let views = PrefillWeightViews::build(&dev, 5120, 17408, &layers, src).unwrap();
    dev.synchronize().unwrap();
    let used = free0 - dev.free_memory().unwrap();
    println!(
        "views: {} bytes held, {} bytes of device memory taken",
        views.device_bytes(),
        used
    );
    // Every MLP matrix's swizzled copy holds its linear plane's bytes where the layout puts them.
    let mut matrices = 0;
    for (l, layer) in layers.iter().enumerate() {
        for (i, (slice, n, k)) in [
            (&layer.w_gate, 17408usize, 5120usize),
            (&layer.w_up, 17408, 5120),
            (&layer.w_down, 5120, 17408),
        ]
        .into_iter()
        .enumerate()
        {
            let linear = src
                .read(l, slice, (n * k / 2) as u64, (n * k / 16) as u64)
                .unwrap();
            let got = dev.dtoh_copy(&views.mlp_scales[l][i]).unwrap();
            let blk_cols = k / 16;
            assert_eq!(got.len(), n * blk_cols);
            let tiles_c = blk_cols.div_ceil(4);
            let mut bad = 0usize;
            for row in 0..n {
                for blk in 0..blk_cols {
                    let off = ((row / 128) * tiles_c + blk / 4) * 512
                        + (row % 32) * 16
                        + (row % 128 / 32) * 4
                        + blk % 4;
                    bad += (got[off] != linear[row * blk_cols + blk]) as usize;
                }
            }
            assert_eq!(bad, 0, "layer {l} matrix {i}: {bad} scale bytes misplaced");
            if l == 0 && i == 0 {
                let linear_as_is = got.iter().zip(&linear).filter(|(a, b)| a != b).count();
                println!(
                    "control: read linearly, {linear_as_is} of {} bytes differ",
                    got.len()
                );
                assert!(
                    linear_as_is > 0,
                    "a linear reading must not match the swizzled copy"
                );
            }
            matrices += 1;
        }
        if let (Some(a), Some(b), Some(ab)) = (&layer.ssm_alpha, &layer.ssm_beta, &views.ab[l]) {
            let mut f = src.read(l, a, 0, a.length).unwrap();
            f.extend(src.read(l, b, 0, b.length).unwrap());
            let bits = dev.dtoh_copy(ab).unwrap();
            for (i, c) in f.chunks_exact(4).enumerate() {
                let v = f32::from_le_bytes(c.try_into().unwrap());
                assert_eq!(
                    f32::from_bits((bits[i] as u32) << 16).to_bits(),
                    v.to_bits(),
                    "layer {l} a/b {i}"
                );
            }
        }
    }
    assert_eq!(matrices, 192);
    println!("192 swizzled MLP scale copies byte-equal to their linear planes; a/b copies exact");

    // The real layer-0 gate, down and GDN qkv weights through their plans.
    let mut ws: Vec<Weight> = Vec::new();
    for (s, slice) in [
        (0usize, &layers[0].w_gate),
        (1, &layers[0].w_down),
        (2, &layers[0].wq),
    ] {
        let shape = &SHAPES[s];
        let bytes = src.read(0, slice, 0, slice.length).unwrap();
        let global = match shape.input {
            LtInput::Nvfp4 => {
                let at = shape.n * shape.k / 2 + shape.n * shape.k / 16;
                f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
            }
            LtInput::Fp8 => 1.0,
        };
        ws.push(Weight::upload(&dev, shape, bytes, global, 1));
    }
    let synth = weights(&dev, false);
    let (mut g, _, _) = build(&dev, &synth);
    for (i, s) in [0usize, 1, 2].into_iter().enumerate() {
        let shape = &SHAPES[s];
        let w0 = ws[i].operand(&dev, shape, 0);
        let rows = 128;
        let host = host_act(shape.input, rows, rows, shape.k, 3);
        let x = upload_act(&dev, shape.input, rows, shape.k, &host);
        let d = unsafe { dev.alloc_uninit::<u16>(rows * shape.ldd) }.unwrap();
        let mut v = unsafe { Verifier::new(&dev, shape.n, shape.k, w0.plane, &x) }.unwrap();
        let c = run_and_check(
            &dev,
            &mut g,
            s,
            rows,
            w0.unit_alpha,
            &ops(&dev, &w0, &x, &d),
            &d,
            &mut v,
        );
        assert_eq!(c.violations, 0, "real {} : {c:?}", shape.name);
        println!(
            "real layer-0 {} verified at {rows} rows: worst {}",
            shape.name, c.worst
        );
    }
}

/// One backend's GEMMs of every shape at 128 rows, as BF16 bits.
struct Backend {
    dev: CudaDevice,
    g: NativeGemm,
    _w: Vec<Weight>,
    xs: Vec<Activation>,
    ds: Vec<CudaSlice<u16>>,
    ops: Vec<(f32, LtOperands)>,
}

impl Backend {
    fn new() -> Self {
        let dev = CudaDevice::new(0).unwrap();
        let w = weights(&dev, false);
        let (g, _, _) = build(&dev, &w);
        let mut xs = Vec::new();
        let mut ds = Vec::new();
        let mut o = Vec::new();
        for (s, shape) in SHAPES.iter().enumerate() {
            let host = host_act(shape.input, 128, 128, shape.k, 50 + s as u64);
            xs.push(upload_act(&dev, shape.input, 128, shape.k, &host));
            ds.push(unsafe { dev.alloc_uninit::<u16>(128 * shape.ldd) }.unwrap());
            let w0 = w[s].operand(&dev, shape, 0);
            o.push((w0.unit_alpha, ops(&dev, &w0, &xs[s], &ds[s])));
        }
        Self {
            dev,
            g,
            _w: w,
            xs,
            ds,
            ops: o,
        }
    }

    fn run(&mut self) -> Vec<Vec<u16>> {
        for (s, (alpha, o)) in self.ops.iter().enumerate() {
            fill_nan(&self.dev, &self.ds[s]);
            unsafe { self.g.run(s, 128, *alpha, o) }.unwrap();
        }
        self.ds
            .iter()
            .map(|d| self.dev.dtoh_copy(d).unwrap())
            .collect()
    }
}

#[test]
#[ignore = "needs a GPU with cuBLASLt >= 12.8"]
fn concurrent_backends_match_serial() {
    let mut a = Backend::new();
    let mut b = Backend::new();
    let serial_a = a.run();
    let serial_b = b.run();
    assert_eq!(serial_a, serial_b, "two backends on one device agree");
    let ta = std::thread::spawn(move || {
        for _ in 0..20 {
            assert_eq!(a.run(), serial_a, "backend A, concurrent");
        }
        a.xs.len()
    });
    let tb = std::thread::spawn(move || {
        for _ in 0..20 {
            assert_eq!(b.run(), serial_b, "backend B, concurrent");
        }
        b.xs.len()
    });
    assert_eq!(ta.join().unwrap(), 7);
    assert_eq!(tb.join().unwrap(), 7);
    println!("two backends, 20 concurrent rounds each: every output equals the serial one");
}

#[test]
#[ignore = "needs a GPU with cuBLASLt >= 12.8; run under compute-sanitizer --tool memcheck"]
fn drop_right_after_enqueue() {
    let dev = CudaDevice::new(0).unwrap();
    let w = weights(&dev, false);
    let (mut g, _, _) = build(&dev, &w);
    let shape = &SHAPES[0];
    let host = host_act(shape.input, MAX_ROWS, MAX_ROWS, shape.k, 8);
    let x = upload_act(&dev, shape.input, MAX_ROWS, shape.k, &host);
    let d = unsafe { dev.alloc_uninit::<u16>(MAX_ROWS * shape.ldd) }.unwrap();
    let w0 = w[0].operand(&dev, shape, 0);
    let o = ops(&dev, &w0, &x, &d);
    for _ in 0..50 {
        unsafe { g.run(0, MAX_ROWS, w0.unit_alpha, &o) }.unwrap();
    }
    use cudarc::driver::sys::{cuStreamQuery, CUresult};
    let query = || unsafe { cuStreamQuery(dev.stream.cu_stream()) };
    assert_eq!(
        query(),
        CUresult::CUDA_ERROR_NOT_READY,
        "the GEMMs are still queued"
    );
    let t = std::time::Instant::now();
    drop(g);
    assert_eq!(
        query(),
        CUresult::CUDA_SUCCESS,
        "the drop waited for the stream"
    );
    println!(
        "dropped with 50 GEMMs enqueued; the drop waited {:?}",
        t.elapsed()
    );
    drop(d);
    drop(x);
    drop(w);
    dev.synchronize().unwrap();
}

#[cfg(feature = "test-fault-injection")]
#[test]
#[ignore = "needs a GPU with cuBLASLt >= 12.8 and a table in the cache"]
fn injected_construction_failures_leak_nothing() {
    use cublaslt::{fault, LtHandle, LtMatmul};
    use lumen_runtime::cuda::cublaslt_algo_cache::select_shape;
    let dev = CudaDevice::new(0).unwrap();
    let w = weights(&dev, false);
    let sets = operand_sets(&dev, &w);
    fault::fail_step(u64::MAX);
    let (g, report) = unsafe { NativeGemm::build(&dev, &sets) }.unwrap();
    assert!(
        matches!(report.source, TableSource::Cached(_)),
        "run after the selection test"
    );
    let steps = fault::steps();
    drop(g);
    assert_eq!(fault::live_objects(), 0);
    println!("a cached build takes {steps} construction steps");
    // Steps in order: the handle, the workspace, 8 per NVFP4 plan (descriptor, two transposes, two
    // scale modes, three layouts), 6 per FP8 plan, then 9 configuration reads per cached plan.
    let buckets = BUCKETS as u64;
    let fp8_first = 2 + 2 * buckets * 8;
    let reads_first = fp8_first + 5 * buckets * 6;
    assert_eq!(steps, reads_first + 7 * buckets * 9, "the step layout");
    // Every kind of step: both allocations, a whole NVFP4 plan, a whole FP8 plan, one plan's nine
    // reads, and the last step.
    let mut at: Vec<u64> = (0..10).collect();
    at.extend(fp8_first..fp8_first + 6);
    at.extend(reads_first..reads_first + 9);
    at.push(steps - 1);
    for &i in &at {
        let mut free_after_first = 0;
        let mut message = String::new();
        for rep in 0..100 {
            fault::fail_step(i);
            let err = match unsafe { NativeGemm::build(&dev, &sets) } {
                Ok(_) => panic!("step {i}: an injected failure must fail the build"),
                Err(e) => e.to_string(),
            };
            assert!(
                err.contains("injected construction failure"),
                "step {i}: {err}"
            );
            assert_eq!(fault::live_objects(), 0, "step {i}: objects left alive");
            dev.synchronize().unwrap();
            if rep == 0 {
                free_after_first = dev.free_memory().unwrap();
                message = err;
            }
        }
        let free = dev.free_memory().unwrap();
        assert!(
            free + (1 << 20) >= free_after_first,
            "step {i}: device memory fell from {free_after_first} to {free} over 100 failures"
        );
        println!("step {i}: 100 failures, none leaked; {message}");
    }
    // The selection path, on the smallest shape: its first bucket's preference, two preference
    // attributes and heuristic query, then the nine configuration reads of its first candidate.
    let handle = LtHandle::new(&dev.ctx).unwrap();
    let shape = &SHAPES[ATTN_KV];
    let mut plans: Vec<LtMatmul> = (0..BUCKETS)
        .map(|b| LtMatmul::new(&handle, shape.lt(bucket_rows(b))).unwrap())
        .collect();
    let ws = dev.alloc_zeros::<u8>(32 << 20).unwrap();
    let ws_ptr = ws.device_ptr(&dev.stream).0;
    let live = fault::live_objects();
    for i in 0..13 {
        for _ in 0..10 {
            fault::fail_step(i);
            let r = unsafe {
                select_shape(
                    &dev,
                    &handle,
                    shape,
                    &mut plans,
                    &sets[ATTN_KV],
                    ws_ptr,
                    32 << 20,
                )
            };
            let err = r
                .expect_err("an injected failure must fail the selection")
                .to_string();
            assert!(
                err.contains("injected construction failure"),
                "selection step {i}: {err}"
            );
            assert_eq!(fault::live_objects(), live, "selection step {i}");
        }
        println!("selection step {i}: 10 failures, none leaked");
    }
    fault::fail_step(u64::MAX);
    drop(plans);
    drop(handle);
    assert_eq!(fault::live_objects(), 0);
    // After all of it, a build works and its GEMMs verify.
    let (mut g, _) = unsafe { NativeGemm::build(&dev, &sets) }.unwrap();
    let shape = &SHAPES[0];
    let host = host_act(shape.input, 128, 128, shape.k, 4);
    let x = upload_act(&dev, shape.input, 128, shape.k, &host);
    let d = unsafe { dev.alloc_uninit::<u16>(128 * shape.ldd) }.unwrap();
    let w0 = sets[0][0];
    let mut v = unsafe { Verifier::new(&dev, shape.n, shape.k, w0.plane, &x) }.unwrap();
    let c = run_and_check(
        &dev,
        &mut g,
        0,
        128,
        w0.unit_alpha,
        &ops(&dev, &w0, &x, &d),
        &d,
        &mut v,
    );
    assert_eq!(c.violations, 0);
}
