//! The native prefill's GDN kernels (`native_prefill_gdn`), on their own, against f64 models of the
//! layer.
//!
//! Two host models, both in f64 (`native_oracle/gdn.rs`):
//! - the oracle: the formulas of Lumen's F32 prefill (conv, SiLU, L2 norm of q and k, gates, then
//!   one delta-rule step per token), evaluated in f64 on the unrounded inputs;
//! - the emulation: the 64-token chunked form of the same recurrence, rounding to BF16 at the
//!   points the native kernels hold BF16 (a and b, the conv output and normalized q/k, beta,
//!   (I + A)^-1, k beta e^G, v beta, W, U, the chunk's state snapshot, v_new, v_new e^(G_last - G),
//!   Aqk, the output) and computing everything else in f64. Its distance to the oracle is the error
//!   of that rounding.
//!
//! A layer passes when its output and final state are within 1.1x (relative L2) and 1.5x (max) of
//! the emulation's error against the oracle, and within 0.2x (output) and 0.15x (state) of that
//! error from the emulation itself, or 1.5x the distance between the emulation and the same
//! emulation with K K^T and the solve in F32 where that is larger (ill-conditioned solves, as real
//! activations have); its conv output within a per-element bound; its ring bit-exact;
//! its gate sums G and U within a small relative distance of the emulation's.
//!
//! - `host_models_agree_without_rounding` (host only): unrounded, the chunked emulation equals the
//!   sequential f64 oracle to f64 rounding, continuing a nonzero state; rounded, it is measurably
//!   farther away.
//! - `qualifying_problem_matches_the_models_and_its_digests`: the load-time problem's outputs pass
//!   the gate and hash to the recorded digests, the state kernel's scale expression (a probe copy)
//!   equals Lumen's runtime `rsqrtf(128)`, the module loads, and a changed kernel is refused.
//! - `layers_match_the_models`: T = 1, 2, 3, 63, 64, 65, 127, 128, 131, 2047, 2048, 2049, from a
//!   zero state and from the state and ring of a 96-, 97- or 98-token prefix (ring positions 0, 1, 2;
//!   all three for T <= 3), and T = 512 and 2048 ill-conditioned (correlated keys, beta near 1, slow
//!   decay).
//! - `slices_match_one_shot`: 2049 = 2048 + 1 and 131 = 64 + 64 + 3 equal the one-shot run bit for
//!   bit, every value written (the chunks are the same); 2049 = 1000 + 1049 passes the gate against
//!   an emulation sliced the same way.
//! - `layout_matches_the_f32_route`: from the same inputs and state, Lumen's F32 prefill kernels
//!   (`ssm_conv1d_silu_prefill`, `gdn_compute_gates_batched`, `l2_normalize_qk_strided`,
//!   `gdn_prefill_fused_v3`) match the oracle, and the native output, state and ring match theirs; a
//!   state indexed `[head][key][value]` does not.
//! - `negative_controls_are_caught`: kernels changed at one point each fail the check they must: the
//!   ring read or written ignoring its position, A without beta, no decay between chunks, the initial
//!   state ignored, the state rounded to BF16 every chunk, a or b not rounded, the state transposed;
//!   and every check but `ring` (covered by the position controls) rejects a copy of a good output
//!   changed past its bound, alone except `finite`, which a NaN output trips.
//! - `quick_layers_for_the_sanitizers`: two short layers and the qualifying problem, for
//!   compute-sanitizer.
//!
//! Requires a GPU of compute capability 12.0 and NVRTC 12.8 or newer:
//!
//!   cargo test --release -p lumen-runtime --features cuda --test cuda_native_gdn_test \
//!     -- --ignored --test-threads=1
#![cfg(feature = "cuda")]

mod native_oracle;

use cudarc::driver::{CudaSlice, DevicePtr, LaunchConfig, PushKernelArg};
use lumen_runtime::cuda::ffi::CudaDevice;
use lumen_runtime::cuda::native_prefill_gdn::{next_conv_position, smoke, NativeGdnKernels};
use lumen_runtime::cuda::native_prefill_kernels::checksum;
use lumen_runtime::cuda::shaders::{GDN_KERNEL_SOURCE, NATIVE_PREFILL_GDN_KERNEL_SOURCE};
use native_oracle::gdn::*;

const LENGTHS: [usize; 12] = [1, 2, 3, 63, 64, 65, 127, 128, 131, 2047, 2048, 2049];

// ---------------------------------------------------------------------------------------------
// GPU runs.

fn device() -> CudaDevice {
    CudaDevice::new(0).expect("a CUDA device")
}

fn ptr<T>(dev: &CudaDevice, s: &CudaSlice<T>) -> u64 {
    s.device_ptr(&dev.stream).0
}

fn widen(v: Vec<u16>) -> Vec<f32> {
    v.into_iter().map(bf).collect()
}

/// Run `inp` as consecutive slices ending at `cuts` (one slice: `[inp.t]`), the ring and the state
/// staying on the device between slices. Every output is filled with NaN first.
fn run_sliced(k: &NativeGdnKernels, dev: &CudaDevice, inp: &Inputs, cuts: &[usize]) -> Got {
    let ring = dev.htod_copy(&inp.ring).unwrap();
    let state = dev.htod_copy(&inp.s0).unwrap();
    let conv_w = dev.htod_copy(&inp.conv_w).unwrap();
    let dt_bias = dev.htod_copy(&inp.dt_bias).unwrap();
    let ssm_a = dev.htod_copy(&inp.ssm_a).unwrap();
    let mut got = Got {
        cv: Vec::new(),
        gc: Vec::new(),
        u: Vec::new(),
        out: Vec::new(),
        s: Vec::new(),
        ring: Vec::new(),
        state_pos: inp.state_pos,
    };
    let mut a = 0;
    for &b in cuts {
        let t = b - a;
        let qkv = dev.htod_copy(&inp.qkv[a * CONV..b * CONV]).unwrap();
        let ab = dev.htod_copy(&inp.ab[a * AB..b * AB]).unwrap();
        let nan16 = |n: usize| dev.htod_copy(&vec![0xFFFFu16; n]).unwrap();
        let cv = nan16(t * CONV);
        let gc = dev.htod_copy(&vec![f32::NAN; t * H]).unwrap();
        let (w, u, aqk, core) = (
            nan16(t * H * D),
            nan16(t * H * D),
            nan16(t * H * BT),
            nan16(t * H * D),
        );
        let tu = t as u32;
        unsafe {
            k.conv(
                dev,
                ptr(dev, &qkv),
                ptr(dev, &ring),
                ptr(dev, &conv_w),
                ptr(dev, &cv),
                tu,
                got.state_pos as u32,
            )
            .unwrap();
            k.chunk_intra(
                dev,
                ptr(dev, &cv),
                ptr(dev, &ab),
                ptr(dev, &dt_bias),
                ptr(dev, &ssm_a),
                ptr(dev, &gc),
                ptr(dev, &w),
                ptr(dev, &u),
                ptr(dev, &aqk),
                tu,
            )
            .unwrap();
            k.chunk_state(
                dev,
                ptr(dev, &cv),
                ptr(dev, &gc),
                ptr(dev, &w),
                ptr(dev, &u),
                ptr(dev, &aqk),
                ptr(dev, &state),
                ptr(dev, &core),
                tu,
            )
            .unwrap();
        }
        dev.synchronize().unwrap();
        got.cv.extend(widen(dev.dtoh_copy(&cv).unwrap()));
        got.gc.extend(dev.dtoh_copy(&gc).unwrap());
        got.u.extend(widen(dev.dtoh_copy(&u).unwrap()));
        got.out.extend(widen(dev.dtoh_copy(&core).unwrap()));
        got.state_pos = next_conv_position(got.state_pos as u32, tu) as usize;
        a = b;
    }
    got.s = dev.dtoh_copy(&state).unwrap();
    got.ring = dev.dtoh_copy(&ring).unwrap();
    got
}

fn run(k: &NativeGdnKernels, dev: &CudaDevice, inp: &Inputs) -> Got {
    run_sliced(k, dev, inp, &[inp.t])
}

/// Lumen's F32 prefill kernels on the same inputs (a and b unrounded, qkv widened), launched as its
/// F32 prefill launches them.
fn run_f32_route(dev: &CudaDevice, inp: &Inputs) -> Got {
    let module = dev.compile_and_load(GDN_KERNEL_SOURCE).unwrap();
    let f = |name: &str| module.load_function(name).unwrap();
    let (conv_fn, gates_fn, l2_fn, scan_fn) = (
        f("ssm_conv1d_silu_prefill"),
        f("gdn_compute_gates_batched"),
        f("l2_normalize_qk_strided"),
        f("gdn_prefill_fused_v3"),
    );
    let t = inp.t;
    let tu = t as u32;
    let qkv = dev.htod_copy(&widen(inp.qkv.clone())).unwrap();
    let ring = dev.htod_copy(&inp.ring).unwrap();
    let conv_w = dev.htod_copy(&inp.conv_w).unwrap();
    let conv_out = dev.alloc_zeros::<f32>(t * CONV).unwrap();
    let split = |off: usize| -> Vec<f32> {
        (0..t * H)
            .map(|i| inp.ab[(i / H) * AB + off + i % H])
            .collect()
    };
    let alpha_raw = dev.htod_copy(&split(0)).unwrap();
    let beta_raw = dev.htod_copy(&split(H)).unwrap();
    let alpha = dev.alloc_zeros::<f32>(t * H).unwrap();
    let beta = dev.alloc_zeros::<f32>(t * H).unwrap();
    let dt_bias = dev.htod_copy(&inp.dt_bias).unwrap();
    let ssm_a = dev.htod_copy(&inp.ssm_a).unwrap();
    let state = dev.htod_copy(&inp.s0).unwrap();
    let raw = dev.alloc_zeros::<f32>(t * H * D).unwrap();
    let (conv_u, ks, sp, h_u, hk_u, d_u, qk_u) = (
        CONV as u32,
        4u32,
        inp.state_pos as u32,
        H as u32,
        HK as u32,
        D as u32,
        QK as u32,
    );
    let cfg = |grid: (u32, u32), block: u32, shared: u32| LaunchConfig {
        grid_dim: (grid.0, grid.1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: shared,
    };
    let p = |s: &CudaSlice<f32>| ptr(dev, s);
    unsafe {
        dev.stream
            .launch_builder(&conv_fn)
            .arg(&p(&qkv))
            .arg(&p(&ring))
            .arg(&p(&conv_w))
            .arg(&p(&conv_out))
            .arg(&conv_u)
            .arg(&ks)
            .arg(&sp)
            .arg(&tu)
            .launch(cfg(((CONV as u32).div_ceil(256), 1), 256, 0))
            .unwrap();
        dev.stream
            .launch_builder(&gates_fn)
            .arg(&p(&dt_bias))
            .arg(&p(&ssm_a))
            .arg(&p(&beta_raw))
            .arg(&p(&alpha_raw))
            .arg(&p(&alpha))
            .arg(&p(&beta))
            .arg(&h_u)
            .arg(&tu)
            .launch(cfg((((t * H) as u32).div_ceil(256), 1), 256, 0))
            .unwrap();
        let (q_off, k_off) = (0u32, QK as u32);
        dev.stream
            .launch_builder(&l2_fn)
            .arg(&p(&conv_out))
            .arg(&hk_u)
            .arg(&d_u)
            .arg(&tu)
            .arg(&conv_u)
            .arg(&q_off)
            .arg(&k_off)
            .launch(cfg((hk_u * tu, 1), d_u, (d_u.div_ceil(32) + 1) * 4))
            .unwrap();
        dev.stream
            .launch_builder(&scan_fn)
            .arg(&p(&state))
            .arg(&p(&conv_out))
            .arg(&p(&alpha))
            .arg(&p(&beta))
            .arg(&p(&raw))
            .arg(&h_u)
            .arg(&d_u)
            .arg(&d_u)
            .arg(&hk_u)
            .arg(&tu)
            .arg(&qk_u)
            .arg(&conv_u)
            .launch(cfg((d_u, h_u), 32, 0))
            .unwrap();
    }
    dev.synchronize().unwrap();
    Got {
        cv: dev.dtoh_copy(&conv_out).unwrap(),
        gc: Vec::new(),
        u: Vec::new(),
        out: dev.dtoh_copy(&raw).unwrap(),
        s: dev.dtoh_copy(&state).unwrap(),
        ring: dev.dtoh_copy(&ring).unwrap(),
        state_pos: (inp.state_pos + t) % SLOTS,
    }
}

fn kernels(dev: &CudaDevice, source: &str) -> NativeGdnKernels {
    // SAFETY: every source here is the group's own or an `altered` one, whose change rewrites a value
    // or permutes an index within the same buffer.
    unsafe { NativeGdnKernels::compile_source(dev, source) }.unwrap_or_else(|e| panic!("{e}"))
}

/// `source` with `from` replaced by `to`; `from` must occur exactly `count` times.
fn altered(from: &str, to: &str, count: usize) -> String {
    let src = NATIVE_PREFILL_GDN_KERNEL_SOURCE;
    assert_eq!(src.matches(from).count(), count, "{from}");
    src.replace(from, to)
}

/// Collects the checks of a test; the test fails at the end if any failed.
#[derive(Default)]
struct Checks {
    failed: Vec<String>,
    passed: usize,
}

impl Checks {
    fn check(&mut self, ok: bool, what: &str, detail: &str) {
        println!("[{}] {what} :: {detail}", if ok { "PASS" } else { "FAIL" });
        if ok {
            self.passed += 1;
        } else {
            self.failed.push(what.to_string());
        }
    }

    /// A negative control passes when `check` is among the checks that rejected it.
    fn trips(&mut self, v: &Verdict, check: &str, what: &str) {
        self.check(
            v.failed.contains(&check),
            &format!("control {what}: rejected by {check}"),
            &v.msg,
        );
    }

    /// A control that `check` alone must reject.
    fn isolates(&mut self, v: &Verdict, check: &str, what: &str) {
        self.check(
            v.failed == [check],
            &format!("control {what}: rejected by {check} alone"),
            &v.msg,
        );
    }

    fn finish(self) {
        println!("SUMMARY pass={} fail={}", self.passed, self.failed.len());
        assert!(self.failed.is_empty(), "failed: {:?}", self.failed);
        assert!(self.passed > 0);
    }
}

/// The smoke problem's inputs as a layer.
fn smoke_inputs() -> Inputs {
    Inputs {
        t: smoke::T,
        state_pos: smoke::STATE_POS as usize,
        qkv: smoke::qkv(),
        ring: smoke::ring(),
        conv_w: smoke::conv_w(),
        ab: smoke::ab(),
        dt_bias: smoke::dt_bias(),
        ssm_a: smoke::ssm_a(),
        s0: smoke::state(),
    }
}

/// The qualifying problem's output bytes, as [`smoke::run`] returns them, read back as a layer.
fn smoke_got(outputs: &[Vec<u8>; 3]) -> Got {
    let (t, conv_b, hd_b) = (smoke::T, smoke::T * CONV * 2, smoke::T * H * D * 2);
    let u16s = |b: &[u8]| -> Vec<f32> {
        b.chunks_exact(2)
            .map(|c| bf(u16::from_le_bytes([c[0], c[1]])))
            .collect()
    };
    let f32s = |b: &[u8]| -> Vec<f32> {
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    let gc_b = t * H * 4;
    Got {
        cv: u16s(&outputs[0][..conv_b]),
        ring: f32s(&outputs[0][conv_b..]),
        gc: f32s(&outputs[1][..gc_b]),
        u: u16s(&outputs[1][gc_b + hd_b..gc_b + 2 * hd_b]),
        out: u16s(&outputs[2][..hd_b]),
        s: f32s(&outputs[2][hd_b..]),
        state_pos: (smoke::STATE_POS as usize + t) % SLOTS,
    }
}

/// Lumen's `rsqrtf` of a runtime operand; the state kernel's `rsqrt.approx` of its block size
/// (launched with 128 threads); the same of the constant 128, which the assembler folds.
const RSQRT_PROBE: &str = r#"
extern "C" __global__ void runtime_rsqrt(const float* x, float* y) { y[0] = rsqrtf(x[0]); }
extern "C" __global__ void block_rsqrt(float* y)
{
    float r;
    asm("rsqrt.approx.f32 %0, %1;" : "=f"(r) : "f"((float)blockDim.x));
    if (threadIdx.x == 0) {
        y[0] = r;
    }
}
extern "C" __global__ void constant_rsqrt(float* y)
{
    float r;
    asm("rsqrt.approx.f32 %0, %1;" : "=f"(r) : "f"((float)128));
    y[0] = r;
}
"#;

/// The continuation case the controls use: 131 tokens after a 97-token prefix (ring position 1).
fn control_case() -> (Inputs, Model, Model) {
    let inp = Inputs::continuation(131, 97, 1131);
    assert_eq!(inp.state_pos, 1);
    let (orc, emu) = (oracle(&inp), emulate(&inp, true));
    (inp, orc, emu)
}

// ---------------------------------------------------------------------------------------------
// Tests.

#[test]
#[ignore = "host-only check of the models the GPU tests use; run with them"]
fn host_models_agree_without_rounding() {
    // The chunked algebra is the recurrence: unrounded, the emulation equals the oracle to f64
    // rounding, including from a nonzero state; rounded, it is measurably farther away.
    let mut l = Checks::default();
    for (t, prefix) in [(131, 97), (200, 98)] {
        let inp = Inputs::continuation(t, prefix, 5);
        let (orc, exact, rounded) = (oracle(&inp), emulate(&inp, false), emulate(&inp, true));
        let (eo, es) = (err_of(&exact.out, &orc.out), err_of(&exact.s, &orc.s));
        let (ro, rs) = (err_of(&rounded.out, &orc.out), err_of(&rounded.s, &orc.s));
        l.check(
            eo.rel_l2 < 1e-12 && es.rel_l2 < 1e-12 && ro.rel_l2 > 1e-4 && rs.rel_l2 > 1e-4,
            &format!("T={t} after {prefix}: chunked == sequential in f64"),
            &format!(
                "unrounded out {:.2e} state {:.2e}; rounded out {:.2e} state {:.2e}",
                eo.rel_l2, es.rel_l2, ro.rel_l2, rs.rel_l2
            ),
        );
    }
    l.finish();
}

#[test]
#[ignore = "needs a GPU of compute capability 12.0 with NVRTC >= 12.8"]
fn qualifying_problem_matches_the_models_and_its_digests() {
    let dev = device();
    let mut l = Checks::default();
    let k = kernels(&dev, NATIVE_PREFILL_GDN_KERNEL_SOURCE);
    // The digests are of exactly the bytes this gate checks.
    let outputs = smoke::run(&k, &dev).unwrap();
    let inp = smoke_inputs();
    let v = gate(
        &inp,
        &smoke_got(&outputs),
        &oracle(&inp),
        &emulate(&inp, true),
        true,
    );
    l.check(v.ok(), "qualifying problem through the gate", &v.msg);
    for (i, (out, want)) in outputs.iter().zip(smoke::DIGESTS).enumerate() {
        let d = checksum(out);
        println!("SMOKE {i} {d}");
        l.check(d == want, &format!("digest {i}"), &d);
    }
    // The state kernel's scale is Lumen's runtime rsqrtf(128); the folded constant is not.
    let rt = dev.compile_and_load(RSQRT_PROBE).unwrap();
    let arch = dev.fp4_native_arch().unwrap().unwrap();
    let native = dev.compile_and_load_with_arch(RSQRT_PROBE, arch).unwrap();
    let y = dev.alloc_zeros::<f32>(1).unwrap();
    let runtime = |x: f32| -> u32 {
        let xd = dev.htod_copy(&[x]).unwrap();
        let f = rt.load_function("runtime_rsqrt").unwrap();
        unsafe {
            dev.stream
                .launch_builder(&f)
                .arg(&ptr(&dev, &xd))
                .arg(&ptr(&dev, &y))
                .launch(LaunchConfig::for_num_elems(1))
                .unwrap();
        }
        dev.dtoh_copy(&y).unwrap()[0].to_bits()
    };
    let at128 = runtime(128.0);
    let native_scale = |name: &str| -> u32 {
        let f = native.load_function(name).unwrap();
        unsafe {
            dev.stream
                .launch_builder(&f)
                .arg(&ptr(&dev, &y))
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
                .unwrap();
        }
        dev.dtoh_copy(&y).unwrap()[0].to_bits()
    };
    let (block, folded) = (native_scale("block_rsqrt"), native_scale("constant_rsqrt"));
    l.check(
        block == at128 && folded != at128,
        "the state kernel's scale == rsqrtf(128) at run time; the folded constant differs",
        &format!("block {block:#x}, runtime {at128:#x}, folded {folded:#x}"),
    );
    let loaded = NativeGdnKernels::load(&dev);
    l.check(
        loaded.is_ok(),
        "the module loads",
        &loaded.err().map(|e| e.to_string()).unwrap_or_default(),
    );
    let bad = kernels(&dev, &altered("] = a * s.b[i];", "] = a;", 1));
    let refused = bad.qualify(&dev);
    l.check(
        matches!(&refused, Err(e) if e.condition == "Q3" && e.reason.contains("native_gdn_chunk_intra")),
        "a changed in-chunk kernel is refused by qualification",
        &refused.err().map(|e| e.to_string()).unwrap_or_default(),
    );
    l.finish();
}

#[test]
#[ignore = "needs a GPU of compute capability 12.0 with NVRTC >= 12.8"]
fn layers_match_the_models() {
    let dev = device();
    let k = kernels(&dev, NATIVE_PREFILL_GDN_KERNEL_SOURCE);
    let mut l = Checks::default();
    // Continuations after 96, 97 or 98 tokens start at ring position 0, 1 or 2: every position for
    // T <= 3, where the ring keeps some of its old slots, and one per longer length.
    let (mut cases, mut cases_run) = (Vec::new(), Vec::new());
    for (i, &t) in LENGTHS.iter().enumerate() {
        cases.push((t, None));
        if t <= 3 {
            cases.extend([96, 97, 98].map(|p| (t, Some(p))));
        } else {
            cases.push((t, Some(96 + i % 3)));
        }
    }
    for (t, prefix) in cases {
        let inp = if let Some(prefix) = prefix {
            Inputs::continuation(t, prefix, 1000 + t as u64)
        } else {
            Inputs::fresh(t, 1000 + t as u64)
        };
        let what = match prefix {
            Some(p) => format!("T={t} after {p} tokens (ring position {})", inp.state_pos),
            None => format!("T={t} from zero state"),
        };
        cases_run.push((what, inp));
    }
    for t in [512, 2048] {
        let inp = Inputs::continuation(t, 97, 3000 + t as u64).ill_conditioned(t as u64);
        cases_run.push((format!("T={t} after 97 tokens, ill-conditioned"), inp));
    }
    for (what, inp) in cases_run {
        let got = run(&k, &dev, &inp);
        let v = gate(&inp, &got, &oracle(&inp), &emulate(&inp, true), true);
        l.check(v.ok(), &what, &v.msg);
    }
    l.finish();
}

/// Two runs equal bit for bit, with every value finite (outputs start NaN-filled, so a value neither
/// run wrote is refused).
fn same_run(a: &Got, b: &Got) -> bool {
    let finite = |g: &Got| {
        [&g.cv, &g.gc, &g.u, &g.out, &g.s, &g.ring]
            .iter()
            .all(|v| v.iter().all(|x| x.is_finite()))
    };
    got_bits(a) == got_bits(b) && finite(a) && finite(b)
}

#[test]
#[ignore = "needs a GPU of compute capability 12.0 with NVRTC >= 12.8"]
fn slices_match_one_shot() {
    let dev = device();
    let k = kernels(&dev, NATIVE_PREFILL_GDN_KERNEL_SOURCE);
    let mut l = Checks::default();
    for (t, cuts) in [(2049, vec![2048, 2049]), (131, vec![64, 128, 131])] {
        let inp = Inputs::continuation(t, 98, 77 + t as u64);
        let one = run(&k, &dev, &inp);
        let sliced = run_sliced(&k, &dev, &inp, &cuts);
        l.check(
            same_run(&one, &sliced),
            &format!("T={t} as {cuts:?} == one shot, bit for bit, every value written"),
            "",
        );
        // The comparison tells +0 from -0, which float equality does not, and refuses a value both
        // runs left unwritten (NaN-filled), which float equality also did.
        let (mut plus, mut minus) = (one.clone(), one.clone());
        (plus.out[0], minus.out[0]) = (0.0, -0.0);
        l.check(
            plus.out == minus.out && !same_run(&plus, &minus),
            &format!("control T={t}: +0 and -0 in one output differ"),
            "",
        );
        let mut unwritten = one.clone();
        unwritten.s[7] = f32::NAN;
        l.check(
            !same_run(&unwritten, &unwritten.clone()),
            &format!("control T={t}: a NaN left in both runs is refused"),
            "",
        );
    }
    let inp = Inputs::continuation(2049, 97, 2049);
    let cuts = [1000, 2049];
    let got = run_sliced(&k, &dev, &inp, &cuts);
    let v = gate(
        &inp,
        &got,
        &oracle(&inp),
        &emulate_sliced(&inp, &cuts),
        false,
    );
    l.check(
        v.ok(),
        "T=2049 as 1000 + 1049 against the emulation sliced alike",
        &v.msg,
    );
    l.finish();
}

#[test]
#[ignore = "needs a GPU of compute capability 12.0 with NVRTC >= 12.8"]
fn layout_matches_the_f32_route() {
    let dev = device();
    let k = kernels(&dev, NATIVE_PREFILL_GDN_KERNEL_SOURCE);
    let transposed = kernels(
        &dev,
        &altered(
            "((size_t)h * NATIVE_GDN_D + vj0 + vj) * NATIVE_GDN_D + ki]",
            "((size_t)h * NATIVE_GDN_D + ki) * NATIVE_GDN_D + vj0 + vj]",
            2,
        ),
    );
    let mut l = Checks::default();
    for (t, prefix) in [(131, 97), (2048, 98)] {
        let inp = Inputs::continuation(t, prefix, 31 + t as u64);
        let (orc, emu) = (oracle(&inp), emulate(&inp, true));
        let f32r = run_f32_route(&dev, &inp);
        let (fc, fo, fs) = (
            err_of(&f32r.cv, &orc.cv),
            err_of(&f32r.out, &orc.out),
            err_of(&f32r.s, &orc.s),
        );
        l.check(
            fc.rel_l2 < 1e-5
                && fo.rel_l2 < 1e-4
                && fs.rel_l2 < 1e-4
                && bits(&f32r.ring) == bits(&orc.ring),
            &format!("T={t}: the F32 route computes the oracle's recurrence"),
            &format!(
                "conv {:.2e} out {:.2e} state {:.2e} ring {}",
                fc.rel_l2,
                fo.rel_l2,
                fs.rel_l2,
                f32r.ring == orc.ring
            ),
        );
        let (mo, ms) = (err_of(&emu.out, &orc.out), err_of(&emu.s, &orc.s));
        let agree = |got: &Got| -> Verdict {
            let f64v = |v: &[f32]| v.iter().map(|&x| x as f64).collect::<Vec<_>>();
            let (o, s) = (
                err_of(&got.out, &f64v(&f32r.out)),
                err_of(&got.s, &f64v(&f32r.s)),
            );
            let checks = [
                ("finite", o.finite && s.finite),
                ("out", o.rel_l2 <= 1.1 * mo.rel_l2 + 1e-4),
                ("state", s.rel_l2 <= 1.1 * ms.rel_l2 + 1e-4),
                ("ring", bits(&got.ring) == bits(&f32r.ring)),
            ];
            Verdict {
                failed: checks.iter().filter(|c| !c.1).map(|c| c.0).collect(),
                msg: format!(
                    "out {:.3e} (emulation {:.3e}) state {:.3e} (emulation {:.3e}) ring {}",
                    o.rel_l2,
                    mo.rel_l2,
                    s.rel_l2,
                    ms.rel_l2,
                    got.ring == f32r.ring
                ),
            }
        };
        let v = agree(&run(&k, &dev, &inp));
        l.check(
            v.ok(),
            &format!("T={t}: native output, state and ring == the F32 route's"),
            &v.msg,
        );
        l.trips(
            &agree(&run(&transposed, &dev, &inp)),
            "state",
            &format!("T={t}: state read and written [head][key][value]"),
        );
    }
    l.finish();
}

#[test]
#[ignore = "needs a GPU of compute capability 12.0 with NVRTC >= 12.8"]
fn negative_controls_are_caught() {
    let dev = device();
    let (inp, orc, emu) = control_case();
    let mut l = Checks::default();
    let good_run = run(&kernels(&dev, NATIVE_PREFILL_GDN_KERNEL_SOURCE), &dev, &inp);
    let good = gate(&inp, &good_run, &orc, &emu, true);
    l.check(good.ok(), "unchanged kernels", &good.msg);
    // Kernels changed at one point, each with the check it must trip.
    let controls: [(&str, &str, &str, &str, usize); 9] = [
        (
            "ring read as if its position were 0",
            "conv",
            "ring[((state_pos + s + 3) % 3) * NATIVE_GDN_CONV + c]",
            "ring[((s + 3) % 3) * NATIVE_GDN_CONV + c]",
            1,
        ),
        (
            "ring written as if its position were 0",
            "ring",
            "ring[((state_pos + s) % 3) * NATIVE_GDN_CONV + c] =",
            "ring[(s % 3) * NATIVE_GDN_CONV + c] =",
            1,
        ),
        (
            "A built without beta",
            "out_oracle",
            "] = a * s.b[i];",
            "] = a;",
            1,
        ),
        (
            "no decay between chunks",
            "state_oracle",
            "const float egl = expf(glast);",
            "const float egl = 1.0f;",
            1,
        ),
        (
            "initial state ignored",
            "state_oracle",
            "S[mi][ni][e] = vj < vvalid ? state[",
            "S[mi][ni][e] = false ? state[",
            1,
        ),
        (
            "state rounded to BF16 every chunk",
            "state_emulation",
            "S[mi][ni][e] *= egl;",
            "S[mi][ni][e] = native_gdn_rbf(S[mi][ni][e]) * egl;",
            1,
        ),
        (
            "a not rounded to BF16",
            "G",
            "native_gdn_rbf(abrow[h])",
            "abrow[h]",
            1,
        ),
        (
            "b not rounded to BF16",
            "U",
            "native_gdn_rbf(abrow[NATIVE_GDN_H + h])",
            "abrow[NATIVE_GDN_H + h]",
            1,
        ),
        (
            "state read and written [head][key][value]",
            "state_oracle",
            "((size_t)h * NATIVE_GDN_D + vj0 + vj) * NATIVE_GDN_D + ki]",
            "((size_t)h * NATIVE_GDN_D + ki) * NATIVE_GDN_D + vj0 + vj]",
            2,
        ),
    ];
    for (what, check, from, to, count) in controls {
        let k = kernels(&dev, &altered(from, to, count));
        l.trips(
            &gate(&inp, &run(&k, &dev, &inp), &orc, &emu, true),
            check,
            what,
        );
    }
    // Ill-conditioned, the distance limits widen to 1.5x the spread of the two emulations (about 0.6x
    // on the recorded run); the unchanged kernels stay within them, and the widened output limit alone still rejects
    // an output 0.75x the emulation's error plus 0.62x of it alternating (0.67x from it). A state
    // rounded to BF16 every chunk moves this case less than the two emulations differ, so no distance
    // check can see it here; the well-conditioned case above and real activations catch it.
    let ill = Inputs::continuation(2048, 97, 5048).ill_conditioned(2048);
    let (ill_orc, ill_emu) = (oracle(&ill), emulate(&ill, true));
    let ill_run = run(&kernels(&dev, NATIVE_PREFILL_GDN_KERNEL_SOURCE), &dev, &ill);
    let good = gate(&ill, &ill_run, &ill_orc, &ill_emu, true);
    l.check(good.ok(), "unchanged kernels, ill-conditioned", &good.msg);
    // Each check alone rejects a copy of the good output changed just past its bound.
    let rms = |e: &[f64]| (e.iter().map(|v| v * v).sum::<f64>() / e.len() as f64).sqrt();
    let err = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x - y).collect::<Vec<f64>>();
    let (eo, es) = (err(&emu.out, &orc.out), err(&emu.s, &orc.s));
    let (mo, ms) = (err_of(&emu.out, &orc.out), err_of(&emu.s, &orc.s));
    // orc + f (emu - orc) + p (-1)^i, as F32.
    let blend = |orc: &[f64], e: &[f64], f: f64, p: f64| -> Vec<f32> {
        (0..orc.len())
            .map(|i| (orc[i] + f * e[i] + if i % 2 == 0 { p } else { -p }) as f32)
            .collect()
    };
    let bump = |v: &[f32], i: usize, to: f64| -> Vec<f32> {
        let mut v = v.to_vec();
        v[i] = to as f32;
        v
    };
    let mutations: Vec<(&str, &str, Got)> = vec![
        ("one q conv element moved to 2x + 1", "conv", {
            let mut g = good_run.clone();
            g.cv[5] = (2.0 * orc.cv[5] + 1.0) as f32;
            g
        }),
        (
            "output 1.15x the emulation's error",
            "out_oracle",
            Got {
                out: blend(&orc.out, &eo, 1.15, 0.0),
                ..good_run.clone()
            },
        ),
        (
            "one output element 1.6x the emulation's largest error",
            "out_max",
            Got {
                out: bump(&good_run.out, 7, orc.out[7] + 1.6 * mo.max_abs),
                ..good_run.clone()
            },
        ),
        (
            "output 0.9x the emulation's error plus 0.25x of it alternating",
            "out_emulation",
            Got {
                out: blend(&orc.out, &eo, 0.9, 0.25 * rms(&eo)),
                ..good_run.clone()
            },
        ),
        (
            "state 1.12x the emulation's error",
            "state_oracle",
            Got {
                s: blend(&orc.s, &es, 1.12, 0.0),
                ..good_run.clone()
            },
        ),
        (
            "one state element 1.6x the emulation's largest error",
            "state_max",
            Got {
                s: bump(&good_run.s, 7, orc.s[7] + 1.6 * ms.max_abs),
                ..good_run.clone()
            },
        ),
        (
            "state 0.9x the emulation's error plus 0.2x of it alternating",
            "state_emulation",
            Got {
                s: blend(&orc.s, &es, 0.9, 0.2 * rms(&es)),
                ..good_run.clone()
            },
        ),
        (
            "one output element NaN",
            "finite",
            Got {
                out: bump(&good_run.out, 3, f64::NAN),
                ..good_run.clone()
            },
        ),
        (
            "G times 1 + 1e-4",
            "G",
            Got {
                gc: good_run.gc.iter().map(|v| v * (1.0 + 1e-4)).collect(),
                ..good_run.clone()
            },
        ),
        (
            "U times 1 + 1e-3",
            "U",
            Got {
                u: good_run.u.iter().map(|v| v * (1.0 + 1e-3)).collect(),
                ..good_run.clone()
            },
        ),
    ];
    for (what, check, got) in mutations {
        let v = gate(&inp, &got, &orc, &emu, true);
        if check == "finite" {
            l.trips(&v, check, what);
        } else {
            l.isolates(&v, check, what);
        }
    }
    let ill_err = err(&ill_emu.out, &ill_orc.out);
    let widened = Got {
        out: blend(&ill_orc.out, &ill_err, 0.75, 0.62 * rms(&ill_err)),
        ..ill_run.clone()
    };
    l.isolates(
        &gate(&ill, &widened, &ill_orc, &ill_emu, true),
        "out_emulation",
        "ill-conditioned output 0.75x the emulation's error plus 0.62x of it alternating",
    );
    l.finish();
}

#[test]
#[ignore = "needs a GPU of compute capability 12.0 with NVRTC >= 12.8"]
fn quick_layers_for_the_sanitizers() {
    let dev = device();
    let k = kernels(&dev, NATIVE_PREFILL_GDN_KERNEL_SOURCE);
    let mut l = Checks::default();
    for (t, prefix) in [(3, 97), (131, 98)] {
        let inp = Inputs::continuation(t, prefix, 3 + t as u64);
        let v = gate(
            &inp,
            &run(&k, &dev, &inp),
            &oracle(&inp),
            &emulate(&inp, true),
            true,
        );
        l.check(v.ok(), &format!("T={t} after {prefix}"), &v.msg);
    }
    let outputs = smoke::run(&k, &dev).unwrap();
    l.check(
        outputs.iter().all(|o| !o.is_empty()),
        "qualifying problem runs",
        "",
    );
    l.finish();
}
