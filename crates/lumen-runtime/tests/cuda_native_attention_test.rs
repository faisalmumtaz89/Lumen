//! The native prefill's attention kernels (`native_prefill_attn`), on their own, through their own
//! launchers at the real geometry: 24 query heads, 4 KV heads, head size
//! 256, RoPE over 64 dimensions.
//!
//! The prep is checked byte for byte against a host implementation of its arithmetic
//! (`native_oracle/attn.rs`, with the attention models below), which takes
//! only the approximate hardware instructions (`rsqrt.approx.ftz`, `div.full`) from a probe kernel run
//! on exactly the operands the oracle computed. Attention accumulates on tensor cores, whose internal
//! order no host model reproduces bit for bit, so it is checked against an exact float64 attention:
//! its error must stay within 1.1x (relative L2) and 1.5x (largest) of the error of a float64
//! emulation that rounds where the kernel's format rounds (BF16 operands, BF16 probabilities with the
//! denominator summed from them, BF16 output). `tolerance_separates_the_kernel_model_from_truncated_keys`
//! shows, without a GPU, that this gate admits an F32 model of the kernel's algorithm and rejects the
//! same model with the keys staged by truncation.
//!
//! - `attention_group_compiles_for_the_fp4_target`: `compute_120a` is selected, the PTX holds the BF16
//!   tensor-core instructions, each kernel's dynamic shared memory attribute reads back its table
//!   value, and the attention cannot launch with its 101,376 bytes until the attribute allows it.
//! - `qualifying_launches_match_the_recorded_digests`: every output of the qualifying problem passes
//!   its oracle, its digest is the recorded one, and the group loads.
//! - `rope_table_matches_the_f32_route`: the RoPE table holds, bit for bit, the cosines and sines the
//!   F32 prefill's RoPE kernel applies, at positions 0 to 4096, 16383 and 65535.
//! - `prep_matches_the_oracle`: the norms, RoPE, gate split and new KV rows at T = 1 to 2049 tokens
//!   from p0 = 0 to 2048, with every byte outside the new KV rows unchanged; controls (an `rstd` one ulp
//!   off, the other contraction of the rotation, a kernel writing one row late) are caught.
//! - `attention_matches_the_f64_oracle`: prep, staging of the older rows and attention, T = 1 to 2049
//!   from p0 = 0 to 2048, older rows that BF16 cannot hold; controls (one future key visible, query
//!   head h reading KV head h % 4, keys staged by truncation) fail the gate.
//! - `attention_is_causal_at_the_real_geometry`: closed forms (exact zeros above the diagonal), rows
//!   before a perturbed future bit-identical, and NaN in the staging rows past p0 + T without effect
//!   (this shows the V rows there are not read; K rows there could not change the output even if read,
//!   since their scores are masked, and reads past p0 + T are left to memcheck); the future-key control
//!   fails both.
//! - `staging_rounds_to_nearest_even`: the staging of older rows holding exact BF16 ties, NaN,
//!   infinities, values that round up into the next binade, subnormals and signed zeros equals the F32
//!   caches rounded to nearest even, byte for byte; V staged by rounding half away from zero fails.
//!
//! Requires a GPU of compute capability 12.0 and NVRTC 12.8 or newer:
//!
//!   cargo test --release -p lumen-runtime --features cuda --test cuda_native_attention_test \
//!     -- --ignored --test-threads=1
#![cfg(feature = "cuda")]

mod native_oracle;

use cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;
use cudarc::driver::{CudaFunction, CudaSlice, DevicePtr, LaunchConfig, PushKernelArg};
use lumen_runtime::cuda::ffi::CudaDevice;
use lumen_runtime::cuda::native_prefill_attn::{
    scale_log2, smoke, NativeAttnKernels, ATTN_KERNELS, ATTN_PREFILL, ATTN_PREP, ATTN_SHARED,
    KV_TO_BF16, ROPE_TABLE,
};
use lumen_runtime::cuda::native_prefill_kernels::checksum;
use lumen_runtime::cuda::shaders::{NATIVE_PREFILL_ATTN_KERNEL_SOURCE, PREFILL_KERNEL_SOURCE};
use native_oracle::attn::*;

const THETA: f32 = 1.0e7;
const TS: [usize; 7] = [1, 16, 128, 131, 2047, 2048, 2049];
const P0S: [usize; 6] = [0, 37, 1000, 1536, 2047, 2048];

// The lines of the kernel source the controls alter, and what they become.
const CAUSAL: (&str, &str) = (
    "const bool allowed = key <= qp[r];",
    "const bool allowed = key <= qp[r] + 1;",
);
const HEAD_MAP: (&str, &str) = ("const int hk = h / 6;", "const int hk = h % 4;");
const STAGE_K: (&str, &str) = (
    "*reinterpret_cast<uint2*>(k_stage + off) = native_bf16x4_rn(a);",
    "{ uint2 tr; tr.x = (__float_as_uint(a.x) >> 16) | (__float_as_uint(a.y) & 0xFFFF0000u); \
     tr.y = (__float_as_uint(a.z) >> 16) | (__float_as_uint(a.w) & 0xFFFF0000u); \
     *reinterpret_cast<uint2*>(k_stage + off) = tr; }",
);
const STAGE_V_HALF_AWAY: (&str, &str) = (
    "*reinterpret_cast<uint2*>(v_stage + off) = native_bf16x4_rn(b);",
    "{ uint2 ha; ha.x = ((__float_as_uint(b.x) + 0x8000u) >> 16) | ((__float_as_uint(b.y) + 0x8000u) & 0xFFFF0000u); \
     ha.y = ((__float_as_uint(b.z) + 0x8000u) >> 16) | ((__float_as_uint(b.w) + 0x8000u) & 0xFFFF0000u); \
     *reinterpret_cast<uint2*>(v_stage + off) = ha; }",
);
const KV_ROW: (&str, &str) = (
    "const unsigned long long kv_row = (unsigned long long)hk * max_seq + p0 + t;",
    "const unsigned long long kv_row = (unsigned long long)hk * max_seq + p0 + t + 1;",
);

// ---------------------------------------------------------------------------------------------
// Device plumbing.

fn dev() -> CudaDevice {
    CudaDevice::new(0).expect("CUDA device 0")
}

fn ptr<T>(d: &CudaDevice, s: &CudaSlice<T>) -> u64 {
    s.device_ptr(&d.stream).0
}

const CACHE_FILL: u32 = 0xA5A5_A5A5;
const STAGE_FILL: u16 = 0xA5A5;

/// The GPU RoPE table of `n_pos` rows.
fn table(d: &CudaDevice, k: &NativeAttnKernels, n_pos: usize, theta: f32) -> CudaSlice<f32> {
    let cs = d.alloc_zeros::<f32>(n_pos * ROT).unwrap();
    unsafe { k.rope_table(d, ptr(d, &cs), n_pos as u32, theta) }.unwrap();
    cs
}

/// Device state of one layer's attention: the caches and staging [4][max_seq][256], and the prep's
/// outputs and the attention output for up to `tokens` tokens.
struct Layer {
    tokens: usize,
    max_seq: usize,
    kc: CudaSlice<f32>,
    vc: CudaSlice<f32>,
    ks: CudaSlice<u16>,
    vs: CudaSlice<u16>,
    q: CudaSlice<u16>,
    gate: CudaSlice<u16>,
    out: CudaSlice<u16>,
}

impl Layer {
    /// Caches holding `kc`/`vc`, staging and outputs for `t` tokens filled with the sentinels.
    fn new(d: &CudaDevice, t: usize, max_seq: usize, kc: &[f32], vc: &[f32]) -> Self {
        let n = HKV * max_seq * D;
        assert_eq!((kc.len(), vc.len()), (n, n));
        Self {
            tokens: t,
            max_seq,
            kc: d.htod_copy(kc).unwrap(),
            vc: d.htod_copy(vc).unwrap(),
            ks: d.htod_copy(&vec![STAGE_FILL; n]).unwrap(),
            vs: d.htod_copy(&vec![STAGE_FILL; n]).unwrap(),
            q: d.htod_copy(&vec![STAGE_FILL; t * QD]).unwrap(),
            gate: d.htod_copy(&vec![STAGE_FILL; t * QD]).unwrap(),
            out: d.htod_copy(&vec![STAGE_FILL; t * QD]).unwrap(),
        }
    }

    fn prep(
        &self,
        d: &CudaDevice,
        k: &NativeAttnKernels,
        inp: &PrepIn,
        cs: &CudaSlice<f32>,
        p0: usize,
    ) {
        check_capacity(self.tokens, inp.t);
        let qg = d.htod_copy(&inp.qg).unwrap();
        let kin = d.htod_copy(&inp.k).unwrap();
        let vin = d.htod_copy(&inp.v).unwrap();
        let qw = d.htod_copy(&inp.q_w1).unwrap();
        let kw = d.htod_copy(&inp.k_w1).unwrap();
        unsafe {
            k.prep(
                d,
                ptr(d, &qg),
                ptr(d, &kin),
                ptr(d, &vin),
                ptr(d, &qw),
                ptr(d, &kw),
                ptr(d, cs),
                EPS,
                inp.t as u32,
                p0 as u32,
                self.max_seq as u32,
                ptr(d, &self.q),
                ptr(d, &self.gate),
                ptr(d, &self.kc),
                ptr(d, &self.vc),
                ptr(d, &self.ks),
                ptr(d, &self.vs),
            )
        }
        .unwrap();
        d.synchronize().unwrap();
    }

    fn stage(&self, d: &CudaDevice, k: &NativeAttnKernels, len: usize) {
        unsafe {
            k.kv_to_bf16(
                d,
                ptr(d, &self.kc),
                ptr(d, &self.vc),
                ptr(d, &self.ks),
                ptr(d, &self.vs),
                len as u32,
                self.max_seq as u32,
            )
        }
        .unwrap();
    }

    fn attend(&self, d: &CudaDevice, k: &NativeAttnKernels, t: usize, p0: usize) -> Vec<u16> {
        unsafe {
            k.attention(
                d,
                ptr(d, &self.q),
                ptr(d, &self.ks),
                ptr(d, &self.vs),
                ptr(d, &self.out),
                t as u32,
                p0 as u32,
                self.max_seq as u32,
            )
        }
        .unwrap();
        d.dtoh_copy(&self.out).unwrap()
    }
}

/// The prep of `t` tokens writes q and gate rows [0, t): the buffers must hold them.
fn check_capacity(tokens: usize, t: usize) {
    assert!(
        t <= tokens,
        "a prep of {t} tokens into buffers of {tokens} tokens"
    );
}

/// Bytes of the staging rows [0, len) of every KV head that differ from the F32 caches rounded to
/// nearest even, K and V.
fn staging_mismatches(d: &CudaDevice, l: &Layer, len: usize) -> (usize, usize) {
    let (kc, vc) = (d.dtoh_copy(&l.kc).unwrap(), d.dtoh_copy(&l.vc).unwrap());
    let (ks, vs) = (d.dtoh_copy(&l.ks).unwrap(), d.dtoh_copy(&l.vs).unwrap());
    let rows = |c: &[f32], s: &[u16]| -> usize {
        (0..HKV)
            .map(|hk| {
                let r = hk * l.max_seq * D..(hk * l.max_seq + len) * D;
                let want: Vec<u16> = c[r.clone()].iter().map(|&x| to_bf(x)).collect();
                mismatches(&bytes16(&s[r]), &bytes16(&want)).0
            })
            .sum()
    };
    (rows(&kc, &ks), rows(&vc, &vs))
}

fn mismatches(got: &[u8], want: &[u8]) -> (usize, Option<usize>) {
    assert_eq!(got.len(), want.len(), "length");
    let n = got.iter().zip(want).filter(|(a, b)| a != b).count();
    (n, got.iter().zip(want).position(|(a, b)| a != b))
}

/// Differing bytes of every prep output against the oracle, the caches and staging against their
/// initial contents with only the rows [p0, p0 + t) of each head replaced.
fn prep_mismatches(
    d: &CudaDevice,
    l: &Layer,
    want: &PrepOut,
    inp: &PrepIn,
    kc0: &[f32],
    vc0: &[f32],
    p0: usize,
) -> usize {
    let (mut kc, mut vc) = (kc0.to_vec(), vc0.to_vec());
    let n = HKV * l.max_seq * D;
    let (mut ks, mut vs) = (vec![STAGE_FILL; n], vec![STAGE_FILL; n]);
    for t in 0..inp.t {
        for hk in 0..HKV {
            let at = (hk * l.max_seq + p0 + t) * D;
            let from = (t * HKV + hk) * D;
            for c in 0..D {
                ks[at + c] = want.k[from + c];
                vs[at + c] = inp.v[from + c];
                kc[at + c] = bf(want.k[from + c]);
                vc[at + c] = bf(inp.v[from + c]);
            }
        }
    }
    let checks: [(&str, Vec<u8>, Vec<u8>); 6] = [
        ("q", bytes16(&d.dtoh_copy(&l.q).unwrap()), bytes16(&want.q)),
        (
            "gate",
            bytes16(&d.dtoh_copy(&l.gate).unwrap()),
            bytes16(&want.gate),
        ),
        (
            "k cache",
            bytes32(&d.dtoh_copy(&l.kc).unwrap()),
            bytes32(&kc),
        ),
        (
            "v cache",
            bytes32(&d.dtoh_copy(&l.vc).unwrap()),
            bytes32(&vc),
        ),
        (
            "k staging",
            bytes16(&d.dtoh_copy(&l.ks).unwrap()),
            bytes16(&ks),
        ),
        (
            "v staging",
            bytes16(&d.dtoh_copy(&l.vs).unwrap()),
            bytes16(&vs),
        ),
    ];
    let mut total = 0;
    for (what, got, want) in &checks {
        let (n, first) = mismatches(got, want);
        if n > 0 {
            println!(
                "  {what}: {n} of {} bytes differ, first at {first:?}",
                got.len()
            );
        }
        total += n;
    }
    total
}

/// Caches of `max_seq` positions: rows [0, p0) of every head hold F32 values BF16 cannot hold (rows a
/// decode step writes), `k_sd` and 1.0 their standard deviations; the rest is the sentinel.
fn older_caches(rng: &mut Rng, p0: usize, max_seq: usize, k_sd: f64) -> (Vec<f32>, Vec<f32>) {
    let n = HKV * max_seq * D;
    let (mut kc, mut vc) = (
        vec![f32::from_bits(CACHE_FILL); n],
        vec![f32::from_bits(CACHE_FILL); n],
    );
    for hk in 0..HKV {
        for i in hk * max_seq * D..(hk * max_seq + p0) * D {
            kc[i] = (rng.normal() * k_sd) as f32;
            vc[i] = rng.normal() as f32;
        }
    }
    (kc, vc)
}

/// Run the chain prep -> staging of [0, p0) -> attention and return the oracle case built from the
/// kernel's inputs (the prep's q and the caches), the output, and the staging bytes of [0, p0) that
/// differ from the caches rounded to nearest even (K, V).
#[allow(clippy::too_many_arguments)]
fn run_chain(
    d: &CudaDevice,
    k: &NativeAttnKernels,
    cs: &CudaSlice<f32>,
    inp: &PrepIn,
    p0: usize,
    max_seq: usize,
    kc: &[f32],
    vc: &[f32],
) -> (AttnCase, Vec<u16>, (usize, usize)) {
    let l = Layer::new(d, inp.t, max_seq, kc, vc);
    l.prep(d, k, inp, cs, p0);
    l.stage(d, k, p0);
    let staged = staging_mismatches(d, &l, p0);
    let out = l.attend(d, k, inp.t, p0);
    let case = AttnCase {
        p0,
        max_seq,
        q: d.dtoh_copy(&l.q).unwrap().iter().map(|&b| bf(b)).collect(),
        k: d.dtoh_copy(&l.kc).unwrap(),
        v: d.dtoh_copy(&l.vc).unwrap(),
    };
    (case, out, staged)
}

fn altered(d: &CudaDevice, (from, to): (&str, &str)) -> NativeAttnKernels {
    assert_eq!(
        NATIVE_PREFILL_ATTN_KERNEL_SOURCE.matches(from).count(),
        1,
        "the source line `{from}`"
    );
    let source = NATIVE_PREFILL_ATTN_KERNEL_SOURCE.replacen(from, to, 1);
    // SAFETY: each change rewrites a value, a mask or a KV head below the group's count; the KV row
    // written one position late stays inside the cache, which `prep_case` sizes with 37 spare rows.
    unsafe { NativeAttnKernels::compile_source(d, &source) }.expect("the altered source compiles")
}

// ---------------------------------------------------------------------------------------------
// Tests.

#[test]
fn tolerance_separates_the_kernel_model_from_truncated_keys() {
    // Continuations over older rows BF16 cannot hold: the F32 model of the kernel passes the gate,
    // the same model with the keys truncated fails it.
    for (t, p0, w, seed) in [(16usize, 2047usize, 1.0f64, 1u64), (65, 1000, 2.0, 2)] {
        let mut rng = Rng(seed);
        let max_seq = p0 + t;
        let (mut k, mut v) = older_caches(&mut rng, p0, max_seq, w);
        for hk in 0..HKV {
            for i in (hk * max_seq + p0) * D..(hk * max_seq + p0 + t) * D {
                k[i] = rbf((rng.normal() * w) as f32);
                v[i] = rbf(rng.normal() as f32);
            }
        }
        let q = (0..t * QD)
            .map(|_| rbf((rng.normal() * w) as f32))
            .collect();
        let c = AttnCase {
            p0,
            max_seq,
            q,
            k,
            v,
        };
        let rows = check_rows(t);
        let (ex, em) = (attn_oracle(&c, &rows, false), attn_oracle(&c, &rows, true));
        let (ok, msg) = gate(&attn_model_f32(&c, &rows, false), &ex, &em);
        println!("T={t} p0={p0}: F32 model {msg}");
        assert!(ok, "the gate rejects the F32 model of the kernel: {msg}");
        let (ok, msg) = gate(&attn_model_f32(&c, &rows, true), &ex, &em);
        println!("T={t} p0={p0}: truncated keys {msg}");
        assert!(!ok, "the gate admits keys staged by truncation: {msg}");
    }
}

#[test]
#[ignore = "needs a GPU of compute capability 12.0 with NVRTC >= 12.8"]
fn attention_group_compiles_for_the_fp4_target() {
    let d = dev();
    assert_eq!(d.fp4_native_arch().unwrap(), Some("compute_120a"), "target");
    assert!(
        !NATIVE_PREFILL_ATTN_KERNEL_SOURCE.contains("#include"),
        "the group includes a header"
    );
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(
        NATIVE_PREFILL_ATTN_KERNEL_SOURCE,
        cudarc::nvrtc::CompileOptions {
            arch: Some("compute_120a"),
            ..Default::default()
        },
    )
    .expect("compute_120a compile");
    let text = ptx.to_src();
    assert!(text.contains(".target sm_120a"), "PTX target line");
    let mma = text
        .matches("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32")
        .count();
    assert!(mma > 0 && text.contains("ldmatrix") && text.contains("cp.async"));
    println!("PASS compute_120a PTX: {mma} BF16 MMA instructions, no header");

    let k = NativeAttnKernels::compile(&d).expect("compile");
    for spec in &ATTN_KERNELS {
        let got = k
            .function(spec.name)
            .unwrap()
            .get_attribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES)
            .unwrap();
        assert_eq!(got, spec.dynamic_shared as i32, "{}", spec.name);
    }
    println!("PASS every kernel's dynamic shared memory attribute equals its table value");

    // The same kernel from a module whose attribute was never raised cannot launch.
    let raw = d
        .compile_and_load_with_arch(NATIVE_PREFILL_ATTN_KERNEL_SOURCE, "compute_120a")
        .unwrap()
        .load_function("native_attn_prefill")
        .unwrap();
    let max_seq = 64;
    let zeros = d.htod_copy(&vec![0u16; HKV * max_seq * D]).unwrap();
    let out = d.htod_copy(&vec![0u16; QD]).unwrap();
    let (t, p0, ms, sl2) = (1u32, 0u32, max_seq as u32, scale_log2());
    let before = unsafe {
        d.stream
            .launch_builder(&raw)
            .arg(&zeros)
            .arg(&zeros)
            .arg(&zeros)
            .arg(&out)
            .arg(&t)
            .arg(&p0)
            .arg(&ms)
            .arg(&sl2)
            .launch(LaunchConfig {
                grid_dim: (HQ as u32, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: ATTN_SHARED,
            })
            .map(|_| ())
    }
    .and_then(|_| d.stream.synchronize());
    assert!(
        before.is_err(),
        "a {ATTN_SHARED}-byte launch ran without the attribute"
    );
    unsafe {
        k.attention(
            &d,
            ptr(&d, &zeros),
            ptr(&d, &zeros),
            ptr(&d, &zeros),
            ptr(&d, &out),
            1,
            0,
            max_seq as u32,
        )
    }
    .expect("launch with the attribute");
    d.synchronize().unwrap();
    println!(
        "PASS {ATTN_SHARED} B of dynamic shared memory: refused ({}) without the attribute, run with it",
        before.unwrap_err()
    );
}

/// Lumen's F32 prefill RoPE (`rope_apply_batched_neox`) of unit vectors: for positions
/// [pos_start, pos_start + n), row p holds cos then sin of the 32 pairs, bit for bit.
fn f32_route_rope(d: &CudaDevice, f: &CudaFunction, pos_start: u32, n: usize) -> Vec<f32> {
    let mut q = vec![0.0f32; n * D];
    for row in q.chunks_mut(D) {
        row[..ROT / 2].fill(1.0);
    }
    let mut dq = d.htod_copy(&q).unwrap();
    let mut dk = d.alloc_zeros::<f32>(n * D).unwrap();
    let (batch, heads, hd, rot) = (n as u32, 1u32, D as u32, ROT as u32);
    let work = batch * rot / 2;
    unsafe {
        d.stream
            .launch_builder(f)
            .arg(&mut dq)
            .arg(&mut dk)
            .arg(&pos_start)
            .arg(&batch)
            .arg(&heads)
            .arg(&heads)
            .arg(&hd)
            .arg(&THETA)
            .arg(&rot)
            .launch(LaunchConfig {
                grid_dim: (work.div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
            .unwrap();
    }
    let q = d.dtoh_copy(&dq).unwrap();
    q.chunks(D).flat_map(|r| r[..ROT].to_vec()).collect()
}

#[test]
#[ignore = "needs a GPU of compute capability 12.0 with NVRTC >= 12.8"]
fn rope_table_matches_the_f32_route() {
    let d = dev();
    let k = NativeAttnKernels::compile(&d).expect("compile");
    let f = d
        .compile_and_load(PREFILL_KERNEL_SOURCE)
        .unwrap()
        .load_function("rope_apply_batched_neox")
        .unwrap();
    let n = 65536;
    let tab = d.dtoh_copy(&table(&d, &k, n, THETA)).unwrap();
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let mut checked = 0;
    for (start, len) in [(0usize, 4097usize), (16383, 1), (65535, 1)] {
        let lumen = f32_route_rope(&d, &f, start as u32, len);
        let mine = &tab[start * ROT..(start + len) * ROT];
        let (bad, first) = mismatches(&bytes32(mine), &bytes32(&lumen));
        assert_eq!(
            bad, 0,
            "positions {start}+{len}: {bad} bytes differ, first at {first:?}"
        );
        checked += len;
        if start == 0 {
            let shifted = bits(&tab[ROT..(len + 1) * ROT]);
            let diff = shifted
                .iter()
                .zip(bits(&lumen))
                .filter(|(a, b)| *a != b)
                .count();
            assert!(diff > 0, "a table one position off matches");
            let other = d.dtoh_copy(&table(&d, &k, len, 1.0e6)).unwrap();
            let diff_theta = bits(&other)
                .iter()
                .zip(bits(&lumen))
                .filter(|(a, b)| *a != b)
                .count();
            assert!(diff_theta > 0, "a table of theta 1e6 matches");
            println!(
                "control: one position off {diff} of {} values differ, theta 1e6 {diff_theta}",
                lumen.len()
            );
        }
    }
    println!("PASS the table equals the F32 route's RoPE at {checked} positions, bit for bit");
}

/// Run the prep of `inp` at `p0` with `k` and count the bytes that differ from the oracle's.
#[allow(clippy::too_many_arguments)]
fn prep_case(
    d: &CudaDevice,
    k: &NativeAttnKernels,
    hw: &Hw,
    cs: &CudaSlice<f32>,
    cs_host: &[f32],
    inp: &PrepIn,
    p0: usize,
    variant: Variant,
) -> usize {
    let max_seq = p0 + inp.t + 37;
    let n = HKV * max_seq * D;
    let fill = vec![f32::from_bits(CACHE_FILL); n];
    let l = Layer::new(d, inp.t, max_seq, &fill, &fill);
    l.prep(d, k, inp, cs, p0);
    let want = prep_oracle(hw, inp, cs_host, p0, variant);
    prep_mismatches(d, &l, &want, inp, &fill, &fill, p0)
}

#[test]
#[ignore = "needs a GPU of compute capability 12.0 with NVRTC >= 12.8"]
fn prep_matches_the_oracle() {
    let d = dev();
    let hw = Hw::new(&d);
    let k = NativeAttnKernels::compile(&d).expect("compile");
    let n_pos = P0S[P0S.len() - 1] + TS[TS.len() - 1];
    let cs = table(&d, &k, n_pos, THETA);
    let cs_host = d.dtoh_copy(&cs).unwrap();
    let all = PrepIn::realistic(TS[TS.len() - 1], 11, 1.0);
    let mut cases = 0;
    for &t in &TS {
        let inp = all.prefix(t);
        for &p0 in &P0S {
            let bad = prep_case(&d, &k, &hw, &cs, &cs_host, &inp, p0, Variant::Kernel);
            assert_eq!(bad, 0, "prep T={t} p0={p0}: {bad} bytes differ");
            cases += 1;
        }
    }
    println!(
        "PASS prep: {cases} cases (T {TS:?} x p0 {P0S:?}), every output byte and every untouched cache byte"
    );

    let inp = all.prefix(131);
    for (what, v) in [
        ("rstd one ulp off", Variant::RstdUlp),
        (
            "the rotation's other contraction",
            Variant::OtherContraction,
        ),
    ] {
        let bad = prep_case(&d, &k, &hw, &cs, &cs_host, &inp, 37, v);
        assert!(bad > 0, "control {what}: no byte differs");
        println!("control {what}: {bad} bytes differ");
    }
    let late = altered(&d, KV_ROW);
    let bad = prep_case(&d, &late, &hw, &cs, &cs_host, &inp, 37, Variant::Kernel);
    assert!(bad > 0, "control KV rows written one late: no byte differs");
    println!("control KV rows written one late: {bad} bytes differ");
}

#[test]
#[ignore = "needs a GPU of compute capability 12.0 with NVRTC >= 12.8"]
fn qualifying_launches_match_the_recorded_digests() {
    use lumen_runtime::cuda::native_prefill_kernels::smoke::{bf16, weights};
    use smoke::{EPS as SEPS, MAX_SEQ, P0, T, THETA as STHETA};
    let d = dev();
    let hw = Hw::new(&d);
    let k = NativeAttnKernels::compile(&d).expect("compile");
    let got = smoke::run(&k, &d).expect("qualifying problem");
    let (t, p0, ms) = (T as usize, P0 as usize, MAX_SEQ as usize);
    assert_eq!((SEPS, STHETA), (EPS, THETA));

    // Table: the F32 route's RoPE at the same positions.
    let f = d
        .compile_and_load(PREFILL_KERNEL_SOURCE)
        .unwrap()
        .load_function("rope_apply_batched_neox")
        .unwrap();
    let cs: Vec<f32> = got[ROPE_TABLE]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(
        got[ROPE_TABLE],
        bytes32(&f32_route_rope(&d, &f, 0, ms)),
        "table"
    );

    // Prep: the oracle, with the caches' older rows and sentinels kept.
    let inp = PrepIn {
        t,
        qg: bf16(t * HQ * 2 * D, 21),
        k: bf16(t * KVD, 22),
        v: bf16(t * KVD, 23),
        q_w1: weights(D, 24),
        k_w1: weights(D, 25),
    };
    let want = prep_oracle(&hw, &inp, &cs, p0, Variant::Kernel);
    let (mut kc, mut vc) = (smoke::cache(26), smoke::cache(27));
    let n = HKV * ms * D;
    let (mut ks, mut vs) = (vec![0xA5A5u16; n], vec![0xA5A5u16; n]);
    for tok in 0..t {
        for hk in 0..HKV {
            let (at, from) = ((hk * ms + p0 + tok) * D, (tok * HKV + hk) * D);
            for c in 0..D {
                ks[at + c] = want.k[from + c];
                vs[at + c] = inp.v[from + c];
                kc[at + c] = bf(want.k[from + c]);
                vc[at + c] = bf(inp.v[from + c]);
            }
        }
    }
    let prep_want = [
        bytes16(&want.q),
        bytes16(&want.gate),
        bytes32(&kc),
        bytes32(&vc),
        bytes16(&ks),
        bytes16(&vs),
    ]
    .concat();
    assert_eq!(mismatches(&got[ATTN_PREP], &prep_want).0, 0, "prep");

    // Staging: the older rows rounded to nearest even.
    for hk in 0..HKV {
        for i in hk * ms * D..(hk * ms + p0) * D {
            ks[i] = to_bf(kc[i]);
            vs[i] = to_bf(vc[i]);
        }
    }
    assert_eq!(
        mismatches(&got[KV_TO_BF16], &[bytes16(&ks), bytes16(&vs)].concat()).0,
        0,
        "staging"
    );

    // Attention: the float64 gate on every row.
    let case = AttnCase {
        p0,
        max_seq: ms,
        q: want.q.iter().map(|&b| bf(b)).collect(),
        k: kc,
        v: vc,
    };
    let rows: Vec<usize> = (0..t).collect();
    let out: Vec<u16> = got[ATTN_PREFILL]
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    let (ok, msg) = gate(
        &gather(&out, &rows),
        &attn_oracle(&case, &rows, false),
        &attn_oracle(&case, &rows, true),
    );
    assert!(ok, "attention: {msg}");
    println!("PASS qualifying outputs: table, prep and staging exact; attention {msg}");

    let digests: Vec<String> = got.iter().map(|b| checksum(b)).collect();
    for (spec, dg) in ATTN_KERNELS.iter().zip(&digests) {
        println!("SMOKE {} {dg}", spec.name);
    }
    assert_eq!(digests, smoke::DIGESTS, "recorded digests");
    NativeAttnKernels::load(&d).expect("load qualifies");
    println!("PASS the recorded digests are the checked outputs', and the group loads");
}

#[test]
#[ignore = "needs a GPU of compute capability 12.0 with NVRTC >= 12.8"]
fn attention_matches_the_f64_oracle() {
    let d = dev();
    let k = NativeAttnKernels::compile(&d).expect("compile");
    let ts = [1usize, 16, 128, 2047, 2048, 2049];
    let n_pos = P0S[P0S.len() - 1] + ts[ts.len() - 1];
    let cs = table(&d, &k, n_pos, THETA);
    let all = PrepIn::realistic(ts[ts.len() - 1], 21, 1.0);
    let wide = PrepIn::realistic(2048, 22, 2.0);
    let mut cases: Vec<(usize, usize, bool)> = ts
        .iter()
        .flat_map(|&t| P0S.iter().map(move |&p0| (t, p0, false)))
        .collect();
    cases.push((2048, 0, true));
    let run = |kern: &NativeAttnKernels, t: usize, p0: usize, x2: bool| {
        let inp = if x2 { wide.prefix(t) } else { all.prefix(t) };
        let max_seq = p0 + t + 5;
        let mut rng = Rng(1000 + (t * 7 + p0) as u64);
        let (kc, vc) = older_caches(&mut rng, p0, max_seq, if x2 { 2.0 } else { 1.0 });
        let (case, out, staged) = run_chain(&d, kern, &cs, &inp, p0, max_seq, &kc, &vc);
        let rows = check_rows(t);
        let (ex, em) = (
            attn_oracle(&case, &rows, false),
            attn_oracle(&case, &rows, true),
        );
        let (ok, msg) = gate(&gather(&out, &rows), &ex, &em);
        (ok, msg, staged)
    };
    for &(t, p0, x2) in &cases {
        let (ok, msg, staged) = run(&k, t, p0, x2);
        println!(
            "T={t} p0={p0}{}: {msg}; staging bytes differing (K, V) {staged:?}",
            if x2 { " q/k x2" } else { "" }
        );
        assert_eq!(staged, (0, 0), "staging T={t} p0={p0}: bytes differ (K, V)");
        assert!(ok, "attention T={t} p0={p0}: {msg}");
    }
    println!("PASS attention: {} cases within the gate", cases.len());

    for (what, change, t, p0) in [
        ("one future key visible", CAUSAL, 2049, 0),
        ("query head h reading KV head h % 4", HEAD_MAP, 2049, 0),
        ("keys staged by truncation", STAGE_K, 16, 2047),
    ] {
        let (ok, msg, staged) = run(&altered(&d, change), t, p0, false);
        println!(
            "control {what} (T={t} p0={p0}): {msg}; staging bytes differing (K, V) {staged:?}"
        );
        assert!(!ok, "control {what} passes the gate");
        if change == STAGE_K {
            assert!(
                staged.0 > 0 && staged.1 == 0,
                "control {what}: the staging check sees {staged:?}"
            );
        }
    }
}

/// Closed form at Q = K = 0: every visible key has probability exactly 1; V at position p is a
/// one-hot at dimension p scaled by (KV head + 1). Returns the output values above the diagonal that
/// are not zero and the visible ones off (kv + 1) / (qp + 1) by more than one BF16 rounding.
fn closed_form(d: &CudaDevice, k: &NativeAttnKernels, t: usize, p0: usize) -> (usize, usize) {
    let max_seq = D;
    assert!(p0 + t <= max_seq);
    let l = Layer::new(
        d,
        t,
        max_seq,
        &vec![0.0; HKV * max_seq * D],
        &vec![0.0; HKV * max_seq * D],
    );
    let mut v = vec![0u16; HKV * max_seq * D];
    for hk in 0..HKV {
        for p in 0..p0 + t {
            v[(hk * max_seq + p) * D + p] = to_bf((hk + 1) as f32);
        }
    }
    let zq = d.htod_copy(&vec![0u16; t * QD]).unwrap();
    let zk = d.htod_copy(&vec![0u16; HKV * max_seq * D]).unwrap();
    let dv = d.htod_copy(&v).unwrap();
    unsafe {
        k.attention(
            d,
            ptr(d, &zq),
            ptr(d, &zk),
            ptr(d, &dv),
            ptr(d, &l.out),
            t as u32,
            p0 as u32,
            max_seq as u32,
        )
    }
    .unwrap();
    let o = d.dtoh_copy(&l.out).unwrap();
    let tol = 2f64.powi(-8) + 2f64.powi(-20);
    let (mut above, mut off) = (0, 0);
    for tok in 0..t {
        let qp = p0 + tok;
        for h in 0..HQ {
            let want = (h / (HQ / HKV) + 1) as f64 / (qp + 1) as f64;
            for dd in 0..D {
                let got = bf(o[tok * QD + h * D + dd]) as f64;
                if dd > qp {
                    above += (got != 0.0) as usize;
                } else {
                    off += ((got - want).abs() / want > tol) as usize;
                }
            }
        }
    }
    (above, off)
}

/// Staging of `lkv` positions (random BF16, K with standard deviation 1, V 1) in a cache of
/// `max_seq`, rows past `lkv` filled with `tail`.
fn random_stage(rng: &mut Rng, lkv: usize, max_seq: usize, tail: u16) -> (Vec<u16>, Vec<u16>) {
    let n = HKV * max_seq * D;
    let (mut ks, mut vs) = (vec![tail; n], vec![tail; n]);
    for hk in 0..HKV {
        for i in hk * max_seq * D..(hk * max_seq + lkv) * D {
            ks[i] = to_bf(rng.normal() as f32);
            vs[i] = to_bf(rng.normal() as f32);
        }
    }
    (ks, vs)
}

#[allow(clippy::too_many_arguments)]
fn attend_staged(
    d: &CudaDevice,
    k: &NativeAttnKernels,
    q: &[u16],
    ks: &[u16],
    vs: &[u16],
    t: usize,
    p0: usize,
    max_seq: usize,
) -> Vec<u16> {
    let (dq, dk, dv) = (
        d.htod_copy(q).unwrap(),
        d.htod_copy(ks).unwrap(),
        d.htod_copy(vs).unwrap(),
    );
    let out = d.htod_copy(&vec![STAGE_FILL; t * QD]).unwrap();
    unsafe {
        k.attention(
            d,
            ptr(d, &dq),
            ptr(d, &dk),
            ptr(d, &dv),
            ptr(d, &out),
            t as u32,
            p0 as u32,
            max_seq as u32,
        )
    }
    .unwrap();
    d.dtoh_copy(&out).unwrap()
}

/// Rows at positions before `cut` that changed and rows at or after it that changed, when the keys
/// and values at positions >= cut are replaced by large ones.
#[allow(clippy::too_many_arguments)]
fn perturbed(
    d: &CudaDevice,
    k: &NativeAttnKernels,
    q: &[u16],
    ks: &[u16],
    vs: &[u16],
    base: &[u16],
    t: usize,
    p0: usize,
    max_seq: usize,
    cut: usize,
) -> (usize, usize) {
    let (mut ks, mut vs) = (ks.to_vec(), vs.to_vec());
    let mut rng = Rng(99 + cut as u64);
    for hk in 0..HKV {
        for i in (hk * max_seq + cut) * D..(hk * max_seq + p0 + t) * D {
            ks[i] = to_bf((300.0 * rng.normal()) as f32);
            vs[i] = to_bf((1e4 * rng.normal()) as f32);
        }
    }
    let o = attend_staged(d, k, q, &ks, &vs, t, p0, max_seq);
    let (mut before, mut after) = (0, 0);
    for tok in 0..t {
        let changed = o[tok * QD..(tok + 1) * QD] != base[tok * QD..(tok + 1) * QD];
        if p0 + tok < cut {
            before += changed as usize;
        } else {
            after += changed as usize;
        }
    }
    (before, after)
}

#[test]
#[ignore = "needs a GPU of compute capability 12.0 with NVRTC >= 12.8"]
fn attention_is_causal_at_the_real_geometry() {
    let d = dev();
    let k = NativeAttnKernels::compile(&d).expect("compile");
    let closed = [
        (1usize, 0usize),
        (1, 200),
        (6, 0),
        (63, 0),
        (64, 0),
        (65, 0),
        (127, 0),
        (128, 0),
        (131, 0),
        (200, 40),
        (33, 200),
        (24, 230),
        (64, 1),
        (255, 1),
    ];
    for &(t, p0) in &closed {
        let (above, off) = closed_form(&d, &k, t, p0);
        assert_eq!(
            (above, off),
            (0, 0),
            "closed form T={t} p0={p0}: nonzero above the diagonal, visible values off"
        );
    }
    println!(
        "PASS closed form: {} cases, exact zeros above the diagonal, (kv + 1) / (qp + 1) within 2^-8",
        closed.len()
    );

    let mut cuts_checked = 0;
    for (t, p0) in [(2049usize, 0usize), (2048, 1000)] {
        let lkv = p0 + t;
        let max_seq = lkv + 100;
        let mut rng = Rng(7 + p0 as u64);
        let q: Vec<u16> = (0..t * QD).map(|_| to_bf(rng.normal() as f32)).collect();
        let (ks, vs) = random_stage(&mut rng, lkv, max_seq, 0);
        let base = attend_staged(&d, &k, &q, &ks, &vs, t, p0, max_seq);
        // NaN in the staging rows past p0 + T changes nothing: the V rows there are not read (a NaN
        // value would reach the output even with probability 0); the K rows there are masked, so this
        // cannot show whether they are read. Those rows lie inside the allocation here (max_seq =
        // p0 + T + 100); reads past the allocation are covered by memcheck on a case whose max_seq is
        // exactly p0 + T.
        let nan = |s: &[u16]| {
            let mut s = s.to_vec();
            for hk in 0..HKV {
                s[(hk * max_seq + lkv) * D..(hk + 1) * max_seq * D].fill(0x7FC0);
            }
            s
        };
        let poisoned = attend_staged(&d, &k, &q, &nan(&ks), &nan(&vs), t, p0, max_seq);
        assert!(
            poisoned == base,
            "T={t} p0={p0}: NaN V staging past p0 + T changed the output"
        );
        let mut cuts: Vec<usize> = [63, 64, 65, 1000, 2047]
            .into_iter()
            .chain([1, 63, 64, 65, t / 2, t - 1].map(|o| p0 + o))
            .filter(|&c| c > 0 && c < lkv)
            .collect();
        cuts.sort_unstable();
        cuts.dedup();
        for &cut in &cuts {
            let (before, after) = perturbed(&d, &k, &q, &ks, &vs, &base, t, p0, max_seq, cut);
            assert!(
                before == 0 && after > 0,
                "T={t} p0={p0} cut {cut}: rows before the cut changed {before}, after {after}"
            );
            cuts_checked += 1;
        }
        println!("PASS T={t} p0={p0}: NaN V past p0 + T not read; cuts {cuts:?} leave every earlier row bit-identical");
    }
    println!("PASS future perturbation: {cuts_checked} cuts");

    let leak = altered(&d, CAUSAL);
    let (above, _) = closed_form(&d, &leak, 131, 0);
    assert!(
        above > 0,
        "control: the future-key kernel passes the closed form"
    );
    let (t, p0) = (2049, 0);
    let max_seq = t + 100;
    let mut rng = Rng(7);
    let q: Vec<u16> = (0..t * QD).map(|_| to_bf(rng.normal() as f32)).collect();
    let (ks, vs) = random_stage(&mut rng, t, max_seq, 0);
    let base = attend_staged(&d, &leak, &q, &ks, &vs, t, p0, max_seq);
    // A cut on a 64 boundary cannot show the leak: the one extra key of row cut - 1 lies in a key
    // tile that its query tile never loads.
    let (before, _) = perturbed(&d, &leak, &q, &ks, &vs, &base, t, p0, max_seq, 1000);
    assert!(
        before > 0,
        "control: the future-key kernel passes the perturbation"
    );
    println!("control one future key visible: {above} nonzero values above the diagonal, {before} rows before cut 1000 changed");
}

/// Older F32 rows that exercise the rounding: exact ties with even and odd kept bits, NaN (quiet with
/// payload, signalling, negative), infinities, values that round up into the next binade (and the
/// largest finite values, which round to infinity), subnormals, signed zeros, and random bits.
fn rounding_edges(rng: &mut Rng, n: usize) -> Vec<f32> {
    const FIXED: [u32; 16] = [
        0x3F80_8000, // 1 + 2^-8: tie, kept bits even -> stays
        0x3F81_8000, // tie, kept bits odd -> rounds up
        0xBF80_8000,
        0xBF81_8000,
        0x7FC0_0001,
        0x7F80_0001,
        0xFFC1_2345,
        0x7F80_0000,
        0xFF80_0000,
        0x3FFF_FFFF, // just below 2 -> 2
        0x3F7F_8000, // tie into the next binade (odd kept bits) -> 1
        0x7F7F_FFFF, // largest finite -> infinity
        0x0000_8000, // subnormal tie
        0x0001_8001,
        0x8000_0000,
        0x0000_0000,
    ];
    (0..n)
        .map(|i| {
            f32::from_bits(match i % 4 {
                0 => FIXED[(i / 4) % FIXED.len()],
                1 => (rng.next() as u32 & 0xFFFF_0000) | 0x8000,
                _ => rng.next() as u32,
            })
        })
        .collect()
}

#[test]
#[ignore = "needs a GPU of compute capability 12.0 with NVRTC >= 12.8"]
fn staging_rounds_to_nearest_even() {
    let d = dev();
    let k = NativeAttnKernels::compile(&d).expect("compile");
    let run = |kern: &NativeAttnKernels, len: usize, max_seq: usize| {
        let mut rng = Rng(500 + len as u64);
        let n = HKV * max_seq * D;
        let (kc, vc) = (rounding_edges(&mut rng, n), rounding_edges(&mut rng, n));
        let l = Layer::new(&d, 1, max_seq, &kc, &vc);
        l.stage(&d, kern, len);
        // Rows past len keep the sentinel.
        let tail = |s: &CudaSlice<u16>| {
            let s = d.dtoh_copy(s).unwrap();
            (0..HKV)
                .flat_map(|hk| s[(hk * max_seq + len) * D..(hk + 1) * max_seq * D].to_vec())
                .filter(|&b| b != STAGE_FILL)
                .count()
        };
        let untouched = tail(&l.ks) + tail(&l.vs);
        (staging_mismatches(&d, &l, len), untouched)
    };
    for (len, max_seq) in [(1usize, 3usize), (37, 40), (2048, 2049), (4096, 4096)] {
        let (bad, tail) = run(&k, len, max_seq);
        assert_eq!(
            (bad, tail),
            ((0, 0), 0),
            "staging of {len} rows: differing bytes (K, V), written bytes past the rows"
        );
    }
    println!("PASS staging: ties, NaN, infinities, binade carries, subnormals and zeros round to nearest even");
    let (bad, _) = run(&altered(&d, STAGE_V_HALF_AWAY), 37, 40);
    println!("control V rounded half away from zero: {bad:?} bytes differ (K, V)");
    assert!(
        bad.0 == 0 && bad.1 > 0,
        "control: the check misses V rounded half away from zero"
    );
}
