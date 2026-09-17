//! GPU tests for the K-quant kernels (`matvec_q*_k_q8_1(_residual)`,
//! `dequant_q*_k_to_f16`, `embed_token_q*_k`, `embed_batch_q*_k`).
//!
//! Two families:
//!
//! * dequant identity — the device dequant (through the F16 tile and through
//!   the embedding gathers) is bit-identical to the host reference
//!   `lumen_runtime::weight::kquant::dequant_kquant_to_f32` on seeded random superblocks
//!   plus the edge blocks (zero scales, saturated 6-bit scales and mins,
//!   fp16-max and fp16-denormal `d`/`dmin`, zero mins, alternating nibbles,
//!   the Q6_K sub-scale sign extremes). No tolerance.
//! * matvec correctness — on every production shape of the Qwen3.8-27B
//!   files, the device matvec against Q8_1 activations vs a host reference
//!   over the same quantized activations (`max_abs < 1e-3`, `rel_l2 <= 1e-4`),
//!   and the activation-quantization error ratio against the Q4_0 dp4a path
//!   on the same activations (`E_kq <= 1.10 * E_q4`).
//!
//! Requires a CUDA GPU: SM 6.1+ for the dequant / edge gates, SM 8.0+ for the
//! `*_matvec_shapes` comparator (it loads the compute_80 Q4_0 dp4a kernel):
//!
//!   cargo test --release -p lumen-runtime --features cuda --test cuda_kquant_test -- --nocapture

#![cfg(feature = "cuda")]

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions, Ptx};
use lumen_format::quantization::QuantScheme;
use lumen_runtime::cuda::kernel_define;
use lumen_runtime::weight::kquant::{dequant_kquant_to_f32, host_f16_to_f32};
use std::sync::Arc;

const Q8_1_BLOCK_BYTES: usize = 36;

// ---------------------------------------------------------------------------
// Compile with the production loader's flags. The K-quant kernels ship through
// `load_fn_dp4a`: `compute_61` when the toolkit lists it, else the highest
// target the device runs, and no fast-math. The Q4_0 dp4a comparator ships
// through `load_fn_sm80_fast_math` (`compute_80`, `--use_fast_math`).
// ---------------------------------------------------------------------------

fn compile_dp4a_loader(ctx: &Arc<CudaContext>, src: &str) -> (Ptx, &'static str) {
    // `compute_61` when the toolkit lists it, else the highest target this device runs (a PTX
    // for a newer target compiles but cannot load).
    let (major, minor) = ctx.compute_capability().expect("compute capability");
    let cc = (major * 10 + minor) as u32;
    let mut last = String::new();
    // Every spelling `cuda::ffi::arch_name` knows at or above sm_61, `compute_61`
    // first and the rest descending, so the target this picks is the one
    // `dp4a_arch_for` picks on the same device and toolkit (that function is
    // crate-private, so an integration test cannot call it).
    let candidates: Vec<(&'static str, u32)> = vec![
        ("compute_61", 61),
        ("compute_121", 121),
        ("compute_120", 120),
        ("compute_110", 110),
        ("compute_103", 103),
        ("compute_100", 100),
        ("compute_90", 90),
        ("compute_89", 89),
        ("compute_88", 88),
        ("compute_87", 87),
        ("compute_86", 86),
        ("compute_80", 80),
        ("compute_75", 75),
        ("compute_72", 72),
        ("compute_70", 70),
        ("compute_62", 62),
    ];
    for (arch, arch_cc) in candidates {
        if arch_cc > cc {
            continue;
        }
        match compile_ptx_with_opts(
            src,
            CompileOptions {
                arch: Some(arch),
                ..Default::default()
            },
        ) {
            Ok(ptx) => return (ptx, arch),
            Err(e) => last = format!("{arch}: {e:?}"),
        }
    }
    panic!("NVRTC compile failed for every dp4a target up to cc {cc}; last: {last}");
}

fn compile_sm80_fast_math(src: &str) -> Ptx {
    compile_ptx_with_opts(
        src,
        CompileOptions {
            arch: Some("compute_80"),
            // the production loader passes the raw flag (cudarc's `use_fast_math`
            // field only adds `--fmad=true`)
            options: vec!["--use_fast_math".to_string()],
            ..Default::default()
        },
    )
    .unwrap_or_else(|e| panic!("NVRTC compile failed (compute_80 fast-math): {e:?}"))
}

fn create_context() -> (Arc<CudaContext>, Arc<CudaStream>) {
    let ctx = CudaContext::new(0).expect("No CUDA GPU available");
    let stream = ctx.default_stream();
    (ctx, stream)
}

// ---------------------------------------------------------------------------
// Deterministic RNG and exact float helpers.
// ---------------------------------------------------------------------------

fn rng_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 33
}

fn rng_f32(state: &mut u64, lo: f32, hi: f32) -> f32 {
    let u = (rng_next(state) % 1_000_000) as f32 / 1_000_000.0;
    lo + (hi - lo) * u
}

/// IEEE f32 -> f16 bits, round to nearest even, every class (zero, subnormal,
/// normal, overflow to inf, inf, NaN) as PTX `cvt.rn.f16.f32` produces it.
fn f32_to_f16_rne(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 31) as u16) << 15;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x7f_ffff;
    if exp == 0xff {
        return if frac != 0 { 0x7fff } else { sign | 0x7c00 };
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7c00; // overflow -> inf
    }
    if e <= 0 {
        // subnormal (or zero): value = (1.frac) * 2^(e-1) in units of 2^-24
        if e < -10 {
            return sign;
        }
        let mant = frac | 0x80_0000; // 24-bit significand
        let shift = (14 - e) as u32; // total right shift to reach 2^-24 units
        let mut h = (mant >> shift) as u16;
        let rem = mant & ((1u32 << shift) - 1);
        let half = 1u32 << (shift - 1);
        if rem > half || (rem == half && (h & 1) == 1) {
            h += 1;
        }
        return sign | h;
    }
    let mut h = ((e as u32) << 10) | (frac >> 13);
    let round_bits = frac & 0x1fff;
    if round_bits > 0x1000 || (round_bits == 0x1000 && (h & 1) == 1) {
        h += 1; // may carry into the exponent (correct: rounds up to the next binade / inf)
    }
    sign | h as u16
}

/// The engine's Q8_1 quantizer (`quantize_f32_to_q8_1`) mirrored on the host:
/// per 32-block `amax`, `scale = amax / 127`, `q = rn(v * (127 / amax))`
/// clamped to [-127, 127], header `f16 scale | f16 (scale_f16 * sum(q))`.
fn quantize_q8_1(x: &[f32]) -> Vec<u8> {
    assert_eq!(x.len() % 32, 0);
    let mut out = Vec::with_capacity(x.len() / 32 * Q8_1_BLOCK_BYTES);
    for block in x.chunks_exact(32) {
        let amax = block.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let scale = amax / 127.0;
        let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
        let q: Vec<i8> = block
            .iter()
            .map(|v| round_half_even(v * inv).clamp(-127.0, 127.0) as i8)
            .collect();
        let d_bits = f32_to_f16_rne(scale);
        let qsum: f32 = q.iter().map(|&v| v as f32).sum();
        let s_bits = f32_to_f16_rne(host_f16_to_f32(d_bits) * qsum);
        out.extend_from_slice(&d_bits.to_le_bytes());
        out.extend_from_slice(&s_bits.to_le_bytes());
        out.extend(q.iter().map(|&v| v as u8));
    }
    out
}

/// The activation the device kernel sees: `x_scale(f16) * q`, per element.
fn dequant_q8_1(q8_1: &[u8]) -> Vec<f32> {
    q8_1.chunks_exact(Q8_1_BLOCK_BYTES)
        .flat_map(|b| {
            let s = host_f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
            (0..32).map(move |i| s * (b[4 + i] as i8) as f32)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Scheme descriptors: block layout, kernel names, block generators.
// ---------------------------------------------------------------------------

struct Scheme {
    quant: QuantScheme,
    tag: &'static str,
    block_bytes: usize,
    source: &'static str,
}

const Q4_K: Scheme = Scheme {
    quant: QuantScheme::Q4_K,
    tag: "q4_k",
    block_bytes: 144,
    source: lumen_runtime::cuda::shaders::MATVEC_Q4_K_Q8_1_KERNEL_SOURCE,
};

const Q5_K: Scheme = Scheme {
    quant: QuantScheme::Q5_K,
    tag: "q5_k",
    block_bytes: 176,
    source: lumen_runtime::cuda::shaders::MATVEC_Q5_K_Q8_1_KERNEL_SOURCE,
};

const Q6_K: Scheme = Scheme {
    quant: QuantScheme::Q6_K,
    tag: "q6_k",
    block_bytes: 210,
    source: lumen_runtime::cuda::shaders::MATVEC_Q6_K_Q8_1_KERNEL_SOURCE,
};

impl Scheme {
    /// One superblock of realistic random content: `d`, `dmin` (`d` for Q6_K)
    /// as f16 in a small positive range, random packed scales, random quants.
    fn random_block(&self, s: &mut u64) -> Vec<u8> {
        let mut b = vec![0u8; self.block_bytes];
        match self.quant {
            QuantScheme::Q4_K | QuantScheme::Q5_K => {
                let d = f32_to_f16_rne(rng_f32(s, 0.001, 0.008));
                let dmin = f32_to_f16_rne(rng_f32(s, 0.0005, 0.004));
                b[0..2].copy_from_slice(&d.to_le_bytes());
                b[2..4].copy_from_slice(&dmin.to_le_bytes());
                for v in &mut b[4..] {
                    *v = rng_next(s) as u8;
                }
            }
            QuantScheme::Q6_K => {
                for v in &mut b[0..208] {
                    *v = rng_next(s) as u8;
                }
                let d = f32_to_f16_rne(rng_f32(s, 0.0005, 0.004));
                b[208..210].copy_from_slice(&d.to_le_bytes());
            }
            other => panic!("not a K-quant scheme: {other:?}"),
        }
        b
    }

    /// The edge superblocks of the dequant identity gate.
    fn edge_blocks(&self) -> Vec<(&'static str, Vec<u8>)> {
        let bb = self.block_bytes;
        let mut out = Vec::new();
        match self.quant {
            QuantScheme::Q4_K | QuantScheme::Q5_K => {
                let hdr = |d: u16, dmin: u16, scales: u8, qs: u8| -> Vec<u8> {
                    let mut b = vec![qs; bb];
                    b[0..2].copy_from_slice(&d.to_le_bytes());
                    b[2..4].copy_from_slice(&dmin.to_le_bytes());
                    for v in &mut b[4..16] {
                        *v = scales;
                    }
                    b
                };
                let d_norm = f32_to_f16_rne(0.0042);
                out.push(("all-zero scales", hdr(d_norm, d_norm, 0x00, 0xA5)));
                out.push(("max 6-bit scales/mins", hdr(d_norm, d_norm, 0xFF, 0xA5)));
                out.push(("d/dmin fp16 max", hdr(0x7BFF, 0x7BFF, 0xFF, 0xF0)));
                out.push(("d/dmin fp16 min denormal", hdr(0x0001, 0x0001, 0xFF, 0x5A)));
                out.push(("d/dmin fp16 max denormal", hdr(0x03FF, 0x03FF, 0x3F, 0xA5)));
                // mins = 0 with every scale 63. Scale field: bytes 0..4 hold
                // sc[j<4] (low 6) | sc[j+4] high bits; bytes 4..8 hold m[j<4]
                // (low 6) | m[j+4] high bits; bytes 8..12 hold sc[j+4] (low
                // nibble) | m[j+4] (high nibble).
                let mut zero_min = hdr(d_norm, d_norm, 0x00, 0x0F);
                for v in &mut zero_min[4..8] {
                    *v = 0xFF;
                }
                for v in &mut zero_min[8..12] {
                    *v = 0x00;
                }
                for v in &mut zero_min[12..16] {
                    *v = 0x0F;
                }
                out.push(("min = 0", zero_min));
                out.push(("alternating nibbles 0x5A", hdr(d_norm, d_norm, 0x91, 0x5A)));
                out.push(("alternating nibbles 0x0F", hdr(d_norm, d_norm, 0x91, 0x0F)));
                // negative super-block scales: the sign bit of the f16 d / dmin
                out.push(("negative d, negative dmin", hdr(0xC42D, 0xB800, 0x91, 0xA5)));
                out.push(("negative d, positive dmin", hdr(0xC42D, d_norm, 0x3F, 0x5A)));
            }
            QuantScheme::Q6_K => {
                let mk = |ql: u8, qh: u8, sc: i8, d: u16| -> Vec<u8> {
                    let mut b = vec![0u8; bb];
                    for v in &mut b[0..128] {
                        *v = ql;
                    }
                    for v in &mut b[128..192] {
                        *v = qh;
                    }
                    for v in &mut b[192..208] {
                        *v = sc as u8;
                    }
                    b[208..210].copy_from_slice(&d.to_le_bytes());
                    b
                };
                let d_norm = f32_to_f16_rne(0.0042);
                out.push(("all-zero scales", mk(0xA5, 0x5A, 0, d_norm)));
                out.push(("scale +127", mk(0xF0, 0xFF, 127, d_norm)));
                out.push(("scale -128", mk(0x0F, 0x00, -128, d_norm)));
                out.push(("d fp16 max, scale -128", mk(0xFF, 0xFF, -128, 0x7BFF)));
                out.push(("d fp16 min denormal", mk(0x5A, 0xA5, 127, 0x0001)));
                out.push(("d fp16 max denormal", mk(0xA5, 0x5A, -1, 0x03FF)));
                out.push(("alternating nibbles 0x5A", mk(0x5A, 0x00, 17, d_norm)));
                out.push(("alternating nibbles 0x0F", mk(0x0F, 0xC3, -17, d_norm)));
                out.push(("negative d, scale -128", mk(0xA5, 0x5A, -128, 0xC42D)));
                out.push(("negative d, scale +127", mk(0x5A, 0xA5, 127, 0xC42D)));
            }
            other => panic!("not a K-quant scheme: {other:?}"),
        }
        out
    }

    fn host_dequant(&self, raw: &[u8]) -> Vec<f32> {
        let n = raw.len() / self.block_bytes * 256;
        dequant_kquant_to_f32(raw, self.quant, n).expect("host reference")
    }
}

struct Kernels {
    _module: Arc<CudaModule>,
    matvec: CudaFunction,
    matvec_residual: CudaFunction,
    dequant_to_f16: CudaFunction,
    embed_token: CudaFunction,
    embed_batch: CudaFunction,
    arch: &'static str,
    /// the matvec's launch geometry as the source declares it (`#define <TAG>_THREADS`,
    /// `#define <TAG>_NR`): threads per CTA and rows per CTA
    threads: u32,
    rows_per_cta: u32,
}

impl Kernels {
    fn matvec_cfg(&self, out_dim: usize) -> LaunchConfig {
        LaunchConfig {
            grid_dim: ((out_dim as u32).div_ceil(self.rows_per_cta), 1, 1),
            block_dim: (self.threads, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

/// Round to nearest, ties to even — what the kernel's `__float2int_rn` does when it
/// quantizes the activation (`f32::round_ties_even` needs a newer MSRV than the crate declares).
fn round_half_even(x: f32) -> f32 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 {
        2.0 * (x / 2.0).round()
    } else {
        r
    }
}

fn load_scheme(ctx: &Arc<CudaContext>, sc: &Scheme) -> Kernels {
    let source = sc.source.to_string();
    let (ptx, arch) = compile_dp4a_loader(ctx, &source);
    let module = ctx.load_module(ptx).expect("load K-quant module");
    let f = |n: String| {
        module
            .load_function(&n)
            .unwrap_or_else(|e| panic!("{n}: {e}"))
    };
    // `kernel_define` is the runtime loader's own parser, so the geometry below is
    // exactly the geometry the runtime launches these kernels with.
    let prefix = sc.tag.replace('_', "").to_uppercase(); // q4_k -> Q4K
    let threads = kernel_define(&source, &format!("{prefix}_THREADS"))
        .unwrap_or_else(|e| panic!("{}: {e}", sc.tag));
    let rows_per_cta = kernel_define(&source, &format!("{prefix}_NR"))
        .unwrap_or_else(|e| panic!("{}: {e}", sc.tag));
    Kernels {
        matvec: f(format!("matvec_{}_q8_1", sc.tag)),
        matvec_residual: f(format!("matvec_{}_q8_1_residual", sc.tag)),
        dequant_to_f16: f(format!("dequant_{}_to_f16", sc.tag)),
        embed_token: f(format!("embed_token_{}", sc.tag)),
        embed_batch: f(format!("embed_batch_{}", sc.tag)),
        _module: module,
        arch,
        threads,
        rows_per_cta,
    }
}

fn cfg_elements(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

// ---------------------------------------------------------------------------
// Dequant identity: F16 tile and the two gathers vs the host reference, bit for bit.
// ---------------------------------------------------------------------------

fn dequant_identity(sc: &Scheme) {
    // one superblock per row (every superblock is a row of its own), then five
    // superblocks per row so the gathers' `e >> 8` superblock index is exercised
    // apart from the token id
    dequant_identity_width(sc, 256);
    dequant_identity_width(sc, 1280);
}

fn dequant_identity_width(sc: &Scheme, hidden: usize) {
    let (ctx, stream) = create_context();
    let k = load_scheme(&ctx, sc);
    println!(
        "[{}] dp4a loader target: {} (hidden {hidden})",
        sc.tag, k.arch
    );

    // 64 seeded random superblocks + every edge block, padded with random
    // superblocks to whole rows of `hidden / 256` superblocks each.
    let per_row = hidden / 256;
    let mut s = 0x5eed_0001u64 ^ (sc.block_bytes as u64) ^ (hidden as u64);
    let mut names: Vec<String> = Vec::new();
    let mut raw: Vec<u8> = Vec::new();
    for i in 0..64 {
        raw.extend(sc.random_block(&mut s));
        names.push(format!("random#{i}"));
    }
    for (name, b) in sc.edge_blocks() {
        assert_eq!(b.len(), sc.block_bytes);
        raw.extend(b);
        names.push(name.to_string());
    }
    while names.len() % per_row != 0 {
        raw.extend(sc.random_block(&mut s));
        names.push(format!("random#pad{}", names.len()));
    }
    let n_blocks = names.len();
    let rows = n_blocks / per_row;
    let n = rows * hidden;
    let expected = sc.host_dequant(&raw);
    assert_eq!(expected.len(), n);

    let w_gpu = stream.clone_htod(&raw).unwrap();

    // (1) F16 tile == RNE(host f32), bit for bit.
    let mut f16_gpu: CudaSlice<u16> = stream.alloc_zeros(n).unwrap();
    let n_u32 = n as u32;
    unsafe {
        stream
            .launch_builder(&k.dequant_to_f16)
            .arg(&w_gpu)
            .arg(&mut f16_gpu)
            .arg(&n_u32)
            .launch(cfg_elements(n))
            .unwrap();
    }
    let got_f16 = stream.clone_dtoh(&f16_gpu).unwrap();
    let mut f16_mismatch = 0usize;
    for (i, (&g, &e)) in got_f16.iter().zip(&expected).enumerate() {
        let want = f32_to_f16_rne(e);
        if g != want {
            f16_mismatch += 1;
            if f16_mismatch <= 5 {
                eprintln!(
                    "[{}] f16 mismatch elem {i} ({}, within {}): device {g:#06x} host {want:#06x} (f32 {e})",
                    sc.tag,
                    names[i / 256],
                    i % 256
                );
            }
        }
    }

    // (2) embed_token over every row (64 seeded random superblocks + 10 edge blocks,
    // padded to whole rows: 74 rows at hidden 256, 15 at hidden 1280) == host f32,
    // bit for bit.
    let hd = hidden as u32;
    let mut row_gpu: CudaSlice<f32> = stream.alloc_zeros(hidden).unwrap();
    let mut gather_mismatch = 0usize;
    for row in 0..rows {
        let tok = row as u32;
        unsafe {
            stream
                .launch_builder(&k.embed_token)
                .arg(&w_gpu)
                .arg(&mut row_gpu)
                .arg(&tok)
                .arg(&hd)
                .launch(cfg_elements(hidden))
                .unwrap();
        }
        let got = stream.clone_dtoh(&row_gpu).unwrap();
        for (i, (&g, &e)) in got
            .iter()
            .zip(&expected[row * hidden..(row + 1) * hidden])
            .enumerate()
        {
            if g.to_bits() != e.to_bits() {
                gather_mismatch += 1;
                if gather_mismatch <= 5 {
                    eprintln!(
                        "[{}] gather mismatch row {row} ({}) elem {i}: device {g} ({:#010x}) host {e} ({:#010x})",
                        sc.tag,
                        names[row * per_row + i / 256],
                        g.to_bits(),
                        e.to_bits()
                    );
                }
            }
        }
    }

    // (3) embed_batch over a seeded permutation of the rows, twice over.
    let mut ids: Vec<u32> = (0..rows as u32).chain(0..rows as u32).collect();
    for i in (1..ids.len()).rev() {
        let j = (rng_next(&mut s) as usize) % (i + 1);
        ids.swap(i, j);
    }
    let ids_gpu = stream.clone_htod(&ids).unwrap();
    let batch = ids.len();
    let mut out_gpu: CudaSlice<f32> = stream.alloc_zeros(batch * hidden).unwrap();
    let batch_u32 = batch as u32;
    unsafe {
        stream
            .launch_builder(&k.embed_batch)
            .arg(&w_gpu)
            .arg(&ids_gpu)
            .arg(&mut out_gpu)
            .arg(&batch_u32)
            .arg(&hd)
            .launch(cfg_elements(batch * hidden))
            .unwrap();
    }
    let got = stream.clone_dtoh(&out_gpu).unwrap();
    let mut batch_mismatch = 0usize;
    for (t, &tok) in ids.iter().enumerate() {
        let want = &expected[tok as usize * hidden..(tok as usize + 1) * hidden];
        for (i, (&g, &e)) in got[t * hidden..(t + 1) * hidden]
            .iter()
            .zip(want)
            .enumerate()
        {
            if g.to_bits() != e.to_bits() {
                batch_mismatch += 1;
                if batch_mismatch <= 5 {
                    eprintln!("[{}] batch gather mismatch slot {t} tok {tok} elem {i}: device {g} host {e}", sc.tag);
                }
            }
        }
    }

    println!(
        "[{}] dequant identity (hidden {hidden}, {rows} rows): {n_blocks} superblocks ({} random incl. padding + {} edge), {n} elements; \
         f16 tile mismatches={f16_mismatch}, gather mismatches={gather_mismatch}, batch gather mismatches={batch_mismatch}",
        sc.tag,
        n_blocks - sc.edge_blocks().len(),
        sc.edge_blocks().len()
    );
    assert_eq!(
        f16_mismatch, 0,
        "[{}] dequant_{}_to_f16 is not bit-identical to RNE(host)",
        sc.tag, sc.tag
    );
    assert_eq!(
        gather_mismatch, 0,
        "[{}] embed_token_{} is not bit-identical to the host",
        sc.tag, sc.tag
    );
    assert_eq!(
        batch_mismatch, 0,
        "[{}] embed_batch_{} is not bit-identical to the host",
        sc.tag, sc.tag
    );
}

#[test]
fn q4_k_dequant_identity() {
    dequant_identity(&Q4_K);
}

#[test]
fn q5_k_dequant_identity() {
    dequant_identity(&Q5_K);
}

#[test]
fn q6_k_dequant_identity() {
    dequant_identity(&Q6_K);
}

// ---------------------------------------------------------------------------
// Matvec correctness on the production shapes.
// ---------------------------------------------------------------------------

/// `(out_dim, in_dim, role)` of every K-quant plane role in the Qwen3.8-27B
/// Q4_K_M / Q5_K_M files, plus the head shape.
const SHAPES: &[(usize, usize, &str)] = &[
    (17408, 5120, "ffn_gate/ffn_up"),
    (5120, 17408, "ffn_down"),
    (10240, 5120, "attn_qkv (GDN in-proj)"),
    (6144, 5120, "attn_gate"),
    (12288, 5120, "attn_q (Q+gate)"),
    (1024, 5120, "attn_k/attn_v"),
    (5120, 6144, "ssm_out/attn_output"),
    (248320, 5120, "output head"),
    // shapes no file has: an odd superblock count and an output count that is not a
    // multiple of the rows per CTA (the guarded last CTA)
    (96, 1280, "odd superblock count"),
    (9, 5120, "partial row group"),
];

/// `max |a - b|` over the pair, NaN-propagating. Not a `f64::max` fold: that
/// returns the *other* argument for a NaN, so a NaN device output would reduce
/// to 0.0 and sail through the absolute bars below.
fn max_abs_err(a: &[f32], b: &[f64]) -> f64 {
    a.iter().zip(b).fold(0.0f64, |m, (&x, &y)| {
        let d = (x as f64 - y).abs();
        if d.is_nan() || d > m {
            d
        } else {
            m
        }
    })
}

fn rel_l2(a: &[f32], b: &[f64]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (&x, &y) in a.iter().zip(b) {
        num += (x as f64 - y).powi(2);
        den += y * y;
    }
    (num / den.max(1e-300)).sqrt()
}

/// One realistic real-valued weight row segment of 256 values: zero-mean,
/// std ~0.023 (a sum of four uniforms), deterministic from the seed.
fn real_block(s: &mut u64) -> [f32; 256] {
    let mut v = [0.0f32; 256];
    for x in &mut v {
        let mut acc = 0.0f32;
        for _ in 0..4 {
            acc += rng_f32(s, -1.0, 1.0);
        }
        *x = acc * 0.02;
    }
    v
}

/// A valid (not GGML's search-optimised) Q4_K quantization of 256 values:
/// per sub-block `scale_j = (max - min) / 15`, `min_j = max(-min, 0)`, the
/// super-block `d`/`dmin` sized so every 6-bit scale and min fits.
fn quantize_q4_k(v: &[f32; 256]) -> Vec<u8> {
    let mut scales = [0.0f32; 8];
    let mut mins = [0.0f32; 8];
    for j in 0..8 {
        let sub = &v[j * 32..(j + 1) * 32];
        let mn = sub.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = sub.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        scales[j] = (mx - mn) / 15.0;
        mins[j] = (-mn).max(0.0);
    }
    let d16 = f32_to_f16_rne(scales.iter().cloned().fold(0.0, f32::max) / 63.0);
    let dmin16 = f32_to_f16_rne(mins.iter().cloned().fold(0.0, f32::max) / 63.0);
    let d = host_f16_to_f32(d16);
    let dmin = host_f16_to_f32(dmin16);
    let mut sc = [0u8; 8];
    let mut m = [0u8; 8];
    for j in 0..8 {
        sc[j] = if d > 0.0 {
            (scales[j] / d).round().clamp(0.0, 63.0) as u8
        } else {
            0
        };
        m[j] = if dmin > 0.0 {
            (mins[j] / dmin).round().clamp(0.0, 63.0) as u8
        } else {
            0
        };
    }
    let mut out = vec![0u8; 144];
    out[0..2].copy_from_slice(&d16.to_le_bytes());
    out[2..4].copy_from_slice(&dmin16.to_le_bytes());
    // inverse of get_scale_min_k4
    for j in 0..4 {
        out[4 + j] = sc[j] | ((sc[j + 4] >> 4) << 6);
        out[8 + j] = m[j] | ((m[j + 4] >> 4) << 6);
        out[12 + j] = (sc[j + 4] & 0x0F) | ((m[j + 4] & 0x0F) << 4);
    }
    for j in 0..8 {
        let step = d * sc[j] as f32;
        let off = dmin * m[j] as f32;
        for l in 0..32 {
            let q = if step > 0.0 {
                ((v[j * 32 + l] + off) / step).round().clamp(0.0, 15.0) as u8
            } else {
                0
            };
            let byte = &mut out[16 + (j >> 1) * 32 + l];
            *byte |= if j & 1 == 1 { q << 4 } else { q };
        }
    }
    out
}

/// A valid Q5_K quantization of 256 values (the Q4_K scheme with 5-bit
/// quants: `scale_j = (max - min) / 31`, the 5th bit in the qh plane).
fn quantize_q5_k(v: &[f32; 256]) -> Vec<u8> {
    let mut scales = [0.0f32; 8];
    let mut mins = [0.0f32; 8];
    for j in 0..8 {
        let sub = &v[j * 32..(j + 1) * 32];
        let mn = sub.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = sub.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        scales[j] = (mx - mn) / 31.0;
        mins[j] = (-mn).max(0.0);
    }
    let d16 = f32_to_f16_rne(scales.iter().cloned().fold(0.0, f32::max) / 63.0);
    let dmin16 = f32_to_f16_rne(mins.iter().cloned().fold(0.0, f32::max) / 63.0);
    let d = host_f16_to_f32(d16);
    let dmin = host_f16_to_f32(dmin16);
    let mut sc = [0u8; 8];
    let mut m = [0u8; 8];
    for j in 0..8 {
        sc[j] = if d > 0.0 {
            (scales[j] / d).round().clamp(0.0, 63.0) as u8
        } else {
            0
        };
        m[j] = if dmin > 0.0 {
            (mins[j] / dmin).round().clamp(0.0, 63.0) as u8
        } else {
            0
        };
    }
    let mut out = vec![0u8; 176];
    out[0..2].copy_from_slice(&d16.to_le_bytes());
    out[2..4].copy_from_slice(&dmin16.to_le_bytes());
    for j in 0..4 {
        out[4 + j] = sc[j] | ((sc[j + 4] >> 4) << 6);
        out[8 + j] = m[j] | ((m[j + 4] >> 4) << 6);
        out[12 + j] = (sc[j + 4] & 0x0F) | ((m[j + 4] & 0x0F) << 4);
    }
    for j in 0..8 {
        let step = d * sc[j] as f32;
        let off = dmin * m[j] as f32;
        for l in 0..32 {
            let q = if step > 0.0 {
                ((v[j * 32 + l] + off) / step).round().clamp(0.0, 31.0) as u8
            } else {
                0
            };
            let lo = q & 0x0F;
            let byte = &mut out[48 + (j >> 1) * 32 + l];
            *byte |= if j & 1 == 1 { lo << 4 } else { lo };
            out[16 + l] |= (q >> 4) << j;
        }
    }
    out
}

/// A valid Q6_K quantization of 256 values: one int8 scale per 16 elements
/// (`amax / 31` aims `q - 32` at -31..31; rounding that scale to the stored
/// int8 can take it to the 6-bit -32..31 the clamp allows), `d` sized so every
/// scale fits in 0..127; the ql / qh planes packed in the GGML element order.
fn quantize_q6_k(v: &[f32; 256]) -> Vec<u8> {
    let mut scales = [0.0f32; 16];
    for (i, sc) in scales.iter_mut().enumerate() {
        let amax = v[i * 16..(i + 1) * 16]
            .iter()
            .fold(0.0f32, |a, x| a.max(x.abs()));
        *sc = amax / 31.0;
    }
    let d16 = f32_to_f16_rne(scales.iter().cloned().fold(0.0, f32::max) / 127.0);
    let d = host_f16_to_f32(d16);
    let mut out = vec![0u8; 210];
    let mut sc_i8 = [0i8; 16];
    for i in 0..16 {
        sc_i8[i] = if d > 0.0 {
            (scales[i] / d).round().clamp(0.0, 127.0) as i8
        } else {
            0
        };
        out[192 + i] = sc_i8[i] as u8;
    }
    out[208..210].copy_from_slice(&d16.to_le_bytes());
    for idx in 0..256 {
        let n = idx >> 7;
        let g = (idx >> 5) & 3;
        let j = idx & 31;
        let step = d * sc_i8[idx / 16] as f32;
        let q = if step > 0.0 {
            ((v[idx] / step).round() + 32.0).clamp(0.0, 63.0) as u8
        } else {
            32
        };
        let ql = &mut out[64 * n + 32 * (g & 1) + j];
        *ql |= if g >> 1 == 1 {
            (q & 0x0F) << 4
        } else {
            q & 0x0F
        };
        out[128 + 32 * n + j] |= (q >> 4) << (2 * g);
    }
    out
}

/// Q4_0 quantization of one 32-block in GGML's block layout: `d = max_signed / -8`
/// stored as f16, then `q = clamp(0, 15, trunc(v * (1 / stored_d) + 8.5))` against that
/// stored scale — the one `dequant_q4_0` reads back — nibbles de-interleaved.
fn quantize_q4_0(v: &[f32]) -> [u8; 18] {
    let mut amax = 0.0f32;
    let mut max_signed = 0.0f32;
    for &x in v {
        if x.abs() > amax {
            amax = x.abs();
            max_signed = x;
        }
    }
    let d = max_signed / -8.0;
    let d16 = f32_to_f16_rne(d);
    let id = if d != 0.0 {
        1.0 / host_f16_to_f32(d16)
    } else {
        0.0
    };
    let mut out = [0u8; 18];
    out[0..2].copy_from_slice(&d16.to_le_bytes());
    for j in 0..16 {
        let lo = ((v[j] * id + 8.5) as i32).clamp(0, 15) as u8;
        let hi = ((v[j + 16] * id + 8.5) as i32).clamp(0, 15) as u8;
        out[2 + j] = lo | (hi << 4);
    }
    out
}

fn dequant_q4_0(raw: &[u8]) -> Vec<f32> {
    let mut out = vec![0.0f32; raw.len() / 18 * 32];
    for (bi, b) in raw.chunks_exact(18).enumerate() {
        let d = host_f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
        for i in 0..16 {
            out[bi * 32 + i] = d * ((b[2 + i] & 0xF) as f32 - 8.0);
            out[bi * 32 + 16 + i] = d * ((b[2 + i] >> 4) as f32 - 8.0);
        }
    }
    out
}

fn dot(w: &[f32], x: &[f64]) -> f64 {
    w.iter().zip(x).map(|(&a, &b)| a as f64 * b).sum()
}

struct MatvecCase {
    max_abs: f64,
    rel_l2_q: f64,
    e_kq: f64,
    e_q4: f64,
    max_abs_res: f64,
    rel_l2_res: f64,
}

/// One shape: a realistic real-valued weight is quantized to the K-quant
/// scheme AND to Q4_0 (the same matrix, so the two paths differ only in the
/// weight format); the activations are shared. Host references stream per
/// row so the head shape needs no dense f32 copy.
fn matvec_case(
    sc: &Scheme,
    k: &Kernels,
    stream: &Arc<CudaStream>,
    q4_0: &(Arc<CudaModule>, CudaFunction),
    out_dim: usize,
    in_dim: usize,
    seed: u64,
) -> MatvecCase {
    let nsb = in_dim / 256;
    let mut s = seed;
    let x: Vec<f32> = (0..in_dim).map(|_| rng_f32(&mut s, -1.0, 1.0)).collect();
    let residual: Vec<f32> = (0..out_dim).map(|_| rng_f32(&mut s, -4.0, 4.0)).collect();
    let q8_1 = quantize_q8_1(&x);
    let x_q: Vec<f64> = dequant_q8_1(&q8_1).iter().map(|&v| v as f64).collect();
    let x_f: Vec<f64> = x.iter().map(|&v| v as f64).collect();

    let mut raw = Vec::with_capacity(out_dim * nsb * sc.block_bytes);
    let mut q4raw = Vec::with_capacity(out_dim * nsb * 8 * 18);
    let mut ref_q = vec![0.0f64; out_dim];
    let mut ref_f = vec![0.0f64; out_dim];
    let mut ref4_f = vec![0.0f64; out_dim];
    let mut row_raw = Vec::with_capacity(nsb * sc.block_bytes);
    let mut row_q4 = Vec::with_capacity(nsb * 8 * 18);
    for r in 0..out_dim {
        row_raw.clear();
        row_q4.clear();
        for _ in 0..nsb {
            let v = real_block(&mut s);
            row_raw.extend(match sc.quant {
                QuantScheme::Q4_K => quantize_q4_k(&v),
                QuantScheme::Q5_K => quantize_q5_k(&v),
                QuantScheme::Q6_K => quantize_q6_k(&v),
                other => panic!("no test quantizer for {other:?}"),
            });
            for blk in v.chunks_exact(32) {
                row_q4.extend_from_slice(&quantize_q4_0(blk));
            }
        }
        let w_row = sc.host_dequant(&row_raw);
        let w4_row = dequant_q4_0(&row_q4);
        ref_q[r] = dot(&w_row, &x_q);
        ref_f[r] = dot(&w_row, &x_f);
        ref4_f[r] = dot(&w4_row, &x_f);
        raw.extend_from_slice(&row_raw);
        q4raw.extend_from_slice(&row_q4);
    }

    let w_gpu = stream.clone_htod(&raw).unwrap();
    let x_gpu = stream.clone_htod(&q8_1).unwrap();
    let r_gpu = stream.clone_htod(&residual).unwrap();
    let mut out_gpu: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();
    let out_u32 = out_dim as u32;
    let in_u32 = in_dim as u32;
    let cfg = k.matvec_cfg(out_dim);
    unsafe {
        stream
            .launch_builder(&k.matvec)
            .arg(&w_gpu)
            .arg(&x_gpu)
            .arg(&mut out_gpu)
            .arg(&out_u32)
            .arg(&in_u32)
            .launch(cfg)
            .unwrap();
    }
    let got = stream.clone_dtoh(&out_gpu).unwrap();
    let max_abs = max_abs_err(&got, &ref_q);
    let rel_l2_q = rel_l2(&got, &ref_q);
    let e_kq = rel_l2(&got, &ref_f);

    // residual sibling
    let mut out_res: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();
    unsafe {
        stream
            .launch_builder(&k.matvec_residual)
            .arg(&w_gpu)
            .arg(&x_gpu)
            .arg(&r_gpu)
            .arg(&mut out_res)
            .arg(&out_u32)
            .arg(&in_u32)
            .launch(cfg)
            .unwrap();
    }
    let got_res = stream.clone_dtoh(&out_res).unwrap();
    let ref_res: Vec<f64> = ref_q
        .iter()
        .zip(&residual)
        .map(|(&e, &r)| e + r as f64)
        .collect();
    let max_abs_res = max_abs_err(&got_res, &ref_res);
    let rel_l2_res = rel_l2(&got_res, &ref_res);

    // Q4_0 dp4a comparator: the same real matrix in Q4_0, same activations.
    let w4_gpu = stream.clone_htod(&q4raw).unwrap();
    let mut out4: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();
    let cfg_q4_0 = LaunchConfig {
        grid_dim: (out_dim.div_ceil(4) as u32, 1, 1), // the Q4_0 dp4a kernel's own NR = 4
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&q4_0.1)
            .arg(&w4_gpu)
            .arg(&x_gpu)
            .arg(&mut out4)
            .arg(&out_u32)
            .arg(&in_u32)
            .launch(cfg_q4_0)
            .unwrap();
    }
    let got4 = stream.clone_dtoh(&out4).unwrap();
    let e_q4 = rel_l2(&got4, &ref4_f);

    MatvecCase {
        max_abs,
        rel_l2_q,
        e_kq,
        e_q4,
        max_abs_res,
        rel_l2_res,
    }
}

// The bars are written `!(x < bar)` on purpose: a NaN must fail them, and `x >= bar` would let it through.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn matvec_shapes(sc: &Scheme) {
    let (ctx, stream) = create_context();
    let k = load_scheme(&ctx, sc);
    let q4_module = ctx
        .load_module(compile_sm80_fast_math(
            lumen_runtime::cuda::shaders::MATVEC_Q4_0_DP4A_KERNEL_SOURCE,
        ))
        .expect("load Q4_0 dp4a module");
    let q4_fn = q4_module.load_function("matvec_q4_0_dp4a").unwrap();
    let q4_0 = (q4_module, q4_fn);
    println!(
        "[{}] dp4a loader target: {}; comparator matvec_q4_0_dp4a: compute_80 fast-math",
        sc.tag, k.arch
    );
    println!("[{}] shape                          role                     max_abs      rel_l2(q)    E_kq         E_q4         E_kq/E_q4  max_abs(res)", sc.tag);
    let mut failures = Vec::new();
    for (i, &(out_dim, in_dim, role)) in SHAPES.iter().enumerate() {
        let c = matvec_case(sc, &k, &stream, &q4_0, out_dim, in_dim, 0xC0FFEE + i as u64);
        let ratio = c.e_kq / c.e_q4;
        println!(
            "[{}] [{out_dim:>6} x {in_dim:>5}]  {role:<24} {:.3e}    {:.3e}    {:.3e}    {:.3e}    {ratio:.4}     {:.3e}",
            sc.tag, c.max_abs, c.rel_l2_q, c.e_kq, c.e_q4, c.max_abs_res
        );
        if !(c.max_abs < 1e-3) {
            failures.push(format!("{role}: max_abs {:.3e} >= 1e-3", c.max_abs));
        }
        if !(c.rel_l2_q <= 1e-4) {
            failures.push(format!("{role}: rel_l2 {:.3e} > 1e-4", c.rel_l2_q));
        }
        if !(c.max_abs_res < 1e-3) {
            failures.push(format!(
                "{role}: residual max_abs {:.3e} >= 1e-3",
                c.max_abs_res
            ));
        }
        if !(c.rel_l2_res <= 1e-4) {
            failures.push(format!(
                "{role}: residual rel_l2 {:.3e} > 1e-4",
                c.rel_l2_res
            ));
        }
        // The ratio bar is only as strong as its comparator: a positive-infinite E_q4
        // makes `1.10 * E_q4` infinite, which every finite E_kq satisfies, so the bar
        // passes vacuously; a NaN E_q4 makes the comparison false, so the bar fires
        // under the ratio's name instead of naming the comparator that broke. The
        // comparator is therefore checked on its own, on every shape, first.
        if !c.e_q4.is_finite() {
            failures.push(format!("{role}: E_q4 {:.3e} is not finite", c.e_q4));
        }
        // the quantisation-quality ratio is a statistic over a plane's many rows: on the
        // two small synthetic shapes only the bars above apply
        if out_dim * in_dim >= 1 << 20 && !(c.e_kq <= 1.10 * c.e_q4) {
            failures.push(format!(
                "{role}: E_kq {:.3e} > 1.10 x E_q4 {:.3e}",
                c.e_kq, c.e_q4
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "[{}] matvec bars failed:\n{}",
        sc.tag,
        failures.join("\n")
    );
}

#[test]
fn q4_k_matvec_shapes() {
    matvec_shapes(&Q4_K);
}

#[test]
fn q5_k_matvec_shapes() {
    matvec_shapes(&Q5_K);
}

#[test]
fn q6_k_matvec_shapes() {
    matvec_shapes(&Q6_K);
}

// ---------------------------------------------------------------------------
// Matvec on edge superblocks: the matvec decodes the scales on its own geometry
// (Q4_K / Q5_K unpack the header words in registers; Q6_K reads the two signed-byte
// sub-scales a lane needs and the halfword `d`), a different path from the per-element
// reads the dequant identity gate proves. Rows are built from every edge block
// except the two saturated-`d` ones (below) and from random-bytes superblocks
// (random signed Q6_K sub-scales, random 6-bit scales and mins),
// on a shape whose out_dim is odd — not a multiple of any NR the kernels use (2 or 4).
// ---------------------------------------------------------------------------

/// The edge fixtures whose `d` is fp16 max: their elements reach |d * sc * q| ~ 1e8, so a
/// row that mixes one with ordinary superblocks drives the kernel's f32 accumulator through
/// partial sums ~1e10 while the dot itself can cancel to far less, and the leftover f32
/// rounding then exceeds this gate's 1e-4 relative bar. Nothing overflows (the conservative
/// 512-element bound is ~1.4e11 against f32's ~3.4e38); the largest-subnormal `d` fixtures
/// (`0x03FF`) are ordinary here and stay in the pool. Every fixture, these included, is
/// covered bit-for-bit by the dequant identity gate above.
const SATURATED_D_FIXTURES: [&str; 2] = ["d/dmin fp16 max", "d fp16 max, scale -128"];

fn matvec_edge_case(sc: &Scheme) {
    let (ctx, stream) = create_context();
    let k = load_scheme(&ctx, sc);
    let out_dim = 71usize;
    let in_dim = 512usize;
    let nsb = in_dim / 256;
    let pool: Vec<Vec<u8>> = sc
        .edge_blocks()
        .into_iter()
        .filter(|(name, _)| !SATURATED_D_FIXTURES.contains(name))
        .map(|(_, b)| b)
        .collect();
    let mut s = 0xed6e_0001u64 ^ (sc.block_bytes as u64);
    let mut raw = Vec::with_capacity(out_dim * nsb * sc.block_bytes);
    for i in 0..out_dim * nsb {
        if i % 3 == 2 {
            raw.extend(sc.random_block(&mut s));
        } else {
            raw.extend_from_slice(&pool[(i / 3 + i) % pool.len()]);
        }
    }
    let w_deq = sc.host_dequant(&raw);
    let x: Vec<f32> = (0..in_dim).map(|_| rng_f32(&mut s, -1.0, 1.0)).collect();
    let residual: Vec<f32> = (0..out_dim).map(|_| rng_f32(&mut s, -4.0, 4.0)).collect();
    let q8_1 = quantize_q8_1(&x);
    let x_q: Vec<f64> = dequant_q8_1(&q8_1).iter().map(|&v| v as f64).collect();
    let ref_q: Vec<f64> = (0..out_dim)
        .map(|r| dot(&w_deq[r * in_dim..(r + 1) * in_dim], &x_q))
        .collect();
    let ref_res: Vec<f64> = ref_q
        .iter()
        .zip(&residual)
        .map(|(&e, &r)| e + r as f64)
        .collect();
    assert!(
        ref_q.iter().all(|v| v.is_finite()),
        "[{}] edge reference not finite",
        sc.tag
    );

    let w_gpu = stream.clone_htod(&raw).unwrap();
    let x_gpu = stream.clone_htod(&q8_1).unwrap();
    let r_gpu = stream.clone_htod(&residual).unwrap();
    let mut out_gpu: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();
    let mut out_res: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();
    let out_u32 = out_dim as u32;
    let in_u32 = in_dim as u32;
    let cfg = k.matvec_cfg(out_dim);
    unsafe {
        stream
            .launch_builder(&k.matvec)
            .arg(&w_gpu)
            .arg(&x_gpu)
            .arg(&mut out_gpu)
            .arg(&out_u32)
            .arg(&in_u32)
            .launch(cfg)
            .unwrap();
        stream
            .launch_builder(&k.matvec_residual)
            .arg(&w_gpu)
            .arg(&x_gpu)
            .arg(&r_gpu)
            .arg(&mut out_res)
            .arg(&out_u32)
            .arg(&in_u32)
            .launch(cfg)
            .unwrap();
    }
    let got = stream.clone_dtoh(&out_gpu).unwrap();
    let got_res = stream.clone_dtoh(&out_res).unwrap();
    let max_abs = max_abs_err(&got, &ref_q);
    let max_abs_res = max_abs_err(&got_res, &ref_res);
    let rl = rel_l2(&got, &ref_q);
    let rl_res = rel_l2(&got_res, &ref_res);
    let ref_max = ref_q.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    println!(
        "[{}] edge matvec [{out_dim} x {in_dim}] ({} edge kinds + random bytes): max|ref| {ref_max:.3e}, \
         max_abs {max_abs:.3e}, rel_l2 {rl:.3e}, residual max_abs {max_abs_res:.3e}, rel_l2 {rl_res:.3e}",
        sc.tag,
        pool.len()
    );
    // the edge rows carry values up to |d * sc * q| with saturated scales, so the absolute bar
    // is scaled by the reference magnitude; the relative bar is the gate's own
    let abs_bar = 1e-3 * ref_max.max(1.0);
    assert!(
        max_abs < abs_bar,
        "[{}] edge matvec max_abs {max_abs:.3e} >= {abs_bar:.3e}",
        sc.tag
    );
    assert!(
        rl <= 1e-4,
        "[{}] edge matvec rel_l2 {rl:.3e} > 1e-4",
        sc.tag
    );
    assert!(
        max_abs_res < abs_bar,
        "[{}] edge residual max_abs {max_abs_res:.3e} >= {abs_bar:.3e}",
        sc.tag
    );
    assert!(
        rl_res <= 1e-4,
        "[{}] edge residual rel_l2 {rl_res:.3e} > 1e-4",
        sc.tag
    );
}

#[test]
fn q4_k_matvec_edge_blocks() {
    matvec_edge_case(&Q4_K);
}

#[test]
fn q5_k_matvec_edge_blocks() {
    matvec_edge_case(&Q5_K);
}

#[test]
fn q6_k_matvec_edge_blocks() {
    matvec_edge_case(&Q6_K);
}
