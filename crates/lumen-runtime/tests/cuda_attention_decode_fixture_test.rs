//! The decode-attention reference fixture: the kernel's partial (m, l, o) and
//! merged output on deterministic inputs, pinned as SHA-256 digests so a
//! rewrite that must keep the arithmetic can be held to it bit for bit.
//!
//! Two modes, both on the production shape (24 query heads, 4 KV heads,
//! head_dim 256) at the shipped policy (one tile per CTA to 176 tiles, then
//! 128 CTAs per KV head), on both KV stores, over four input classes and the
//! contexts that cover every boundary of the tile and partition arithmetic:
//!
//! * `LUMEN_ATTN_FIXTURE_WRITE=<dir>`: run the kernel and write every raw
//!   array (little-endian F32) beside a `manifest.json` of their digests, the
//!   generator's identity and the launch geometry. Run once, on the reviewed
//!   kernel, before it changes.
//! * otherwise: if `tests/fixtures/attention_decode_reference.json` exists,
//!   regenerate the inputs, run the kernel and require every digest to match.
//!
//! The inputs are half-representable for both stores, so the two stores'
//! merged outputs must agree bit for bit as well.
//!
//! Requires the `cuda` feature; skipped where there is no device.

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use lumen_runtime::cuda::ffi::CudaDevice;
use lumen_runtime::cuda::{
    decode_attention_geometry_within, decode_attention_merge_shared_bytes, DecodeAttentionSpec,
    ATTN_DECODE_BLOCK_DIM as BLOCK_DIM, ATTN_DECODE_FIXTURE_SHAPE as SPEC,
};
use std::fmt::Write as _;

const NUM_HEADS: u32 = 24;
const NUM_KV_HEADS: u32 = 4;
const HEAD_DIM: u32 = SPEC.head_dim;
const DIM_TILES: u32 = SPEC.dim_tiles();
const MAX_SEQ_LEN: u32 = 32_768;
const SCALE: f32 = 0.0625;
const ONE_TILE: u32 = 176;
const TARGET: u32 = 128;
const PARTIAL_F32: &str = "attention_decode_partial_f32";
const PARTIAL_F16: &str = "attention_decode_partial_f16";
const MERGE: &str = "attention_decode_merge";
const MANIFEST: &str = "tests/fixtures/attention_decode_reference.json";

/// Every boundary of the tile (16), the one-tile bound (176 tiles = 2,816
/// keys), the whole-tile partition's target (128 CTAs), the merge's
/// eight-lane tail, the old kernels' reach (16,384) and the cache's end.
const LENGTHS: &[u32] = &[
    1, 15, 16, 17, 127, 128, 129, 330, 1100, 2600, 2816, 2817, 2832, 2833, 3072, 3968, 4096, 4097,
    6144, 6153, 8192, 12288, 16383, 16384, 16385, 24576, 32767, 32768,
];

/// (name, seed, query scale, V pattern)
const CLASSES: &[(&str, u64, f32, bool)] = &[
    ("unit", 0x0005_0911_F1A7_0001, 1.0, false),
    ("wide", 0x0005_0911_F1A7_0002, 8.0, false),
    ("zero_q", 0x0005_0911_F1A7_0003, 0.0, false),
    ("cancel", 0x0005_0911_F1A7_0004, 1.0, true),
];

fn try_device() -> Option<CudaDevice> {
    match CudaDevice::new(0) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("Skipping: no CUDA GPU available: {e}");
            None
        }
    }
}

// ---- deterministic generator (part of the fixture's identity) ----------

fn rng_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 33
}

/// Uniform in [-1, 1) on a 24-bit lattice, then rounded to the nearest half
/// so both stores see the same value.
fn rand_half(s: &mut u64) -> u16 {
    let x = ((rng_next(s) & 0xff_ffff) as f32 / 8_388_608.0) - 1.0;
    f32_to_f16_bits(x)
}

fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let mant = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = mant | 0x80_0000;
        let shift = (14 - e) as u32;
        let half_m = m >> shift;
        let rem = m & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        let round_up = rem > halfway || (rem == halfway && (half_m & 1) == 1);
        return sign | (half_m + u32::from(round_up)) as u16;
    }
    let half_m = mant >> 13;
    let rem = mant & 0x1fff;
    let round_up = rem > 0x1000 || (rem == 0x1000 && (half_m & 1) == 1);
    let mut out = ((e as u32) << 10) | half_m;
    if round_up {
        out += 1;
    }
    sign | out as u16
}

fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = u32::from(h & 0x8000) << 16;
    let exp = u32::from((h >> 10) & 0x1f);
    let mant = u32::from(h & 0x3ff);
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        let v = mant as f32 * (1.0f32 / 16_777_216.0f32);
        return if sign != 0 { -v } else { v };
    }
    if exp == 0x1f {
        return f32::from_bits(sign | 0x7f80_0000 | (mant << 13));
    }
    f32::from_bits(sign | ((exp + 127 - 15) << 23) | (mant << 13))
}

struct Inputs {
    q: Vec<f32>,
    k16: Vec<u16>,
    v16: Vec<u16>,
    k: Vec<f32>,
    v: Vec<f32>,
}

fn make_inputs(seed: u64, q_scale: f32, cancel: bool) -> Inputs {
    let cache = (NUM_KV_HEADS * MAX_SEQ_LEN * HEAD_DIM) as usize;
    let mut s = seed;
    let q: Vec<f32> = (0..(NUM_HEADS * HEAD_DIM) as usize)
        .map(|_| f16_bits_to_f32(rand_half(&mut s)) * q_scale)
        .collect();
    let k16: Vec<u16> = (0..cache).map(|_| rand_half(&mut s)).collect();
    let mut v16: Vec<u16> = (0..cache).map(|_| rand_half(&mut s)).collect();
    if cancel {
        // Large values of alternating sign at the first key of the first
        // four tiles of every KV head, on every dimension: the running-max
        // recurrence and the merge's rescale carry the cancellation.
        let hd = HEAD_DIM as usize;
        for kv in 0..NUM_KV_HEADS as usize {
            for (t, mag) in [
                (0usize, 4096.0f32),
                (16, -4096.0),
                (32, 6.1035e-5),
                (48, 4096.0),
            ] {
                let row = (kv * MAX_SEQ_LEN as usize + t) * hd;
                for d in 0..hd {
                    v16[row + d] = f32_to_f16_bits(mag);
                }
            }
        }
    }
    Inputs {
        q,
        k: k16.iter().map(|&h| f16_bits_to_f32(h)).collect(),
        v: v16.iter().map(|&h| f16_bits_to_f32(h)).collect(),
        k16,
        v16,
    }
}

// ---- SHA-256 (the digests are the fixture) ------------------------------

fn sha256(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut msg = data.to_vec();
    let bit_len = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in msg.chunks(64) {
        let mut w = [0u32; 64];
        for (i, word) in chunk.chunks(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (x, y) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *x = x.wrapping_add(y);
        }
    }
    let mut out = String::with_capacity(64);
    for x in h {
        let _ = write!(out, "{x:08x}");
    }
    out
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

// ---- the kernel -----------------------------------------------------------

struct Outputs {
    m: Vec<f32>,
    l: Vec<f32>,
    o: Vec<f32>,
    out: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
fn run<K: cudarc::driver::DeviceRepr>(
    dev: &CudaDevice,
    arch: Option<&'static str>,
    partial_name: &str,
    shared_bytes: u32,
    q: &CudaSlice<f32>,
    k: &CudaSlice<K>,
    v: &CudaSlice<K>,
    seq_len: u32,
    chunks: u32,
    partition: u32,
) -> Outputs {
    let module = match arch {
        Some(arch) => dev.compile_and_load_with_arch(&SPEC.source(), arch),
        None => dev.compile_and_load(&SPEC.source()),
    }
    .expect("compile");
    let partial = module.load_function(partial_name).expect("partial");
    let merge = module.load_function(MERGE).expect("merge");
    let n_part = (NUM_HEADS * chunks) as usize;
    let nan = |n: usize| dev.htod_copy(&vec![f32::NAN; n]).expect("alloc");
    let mut m_part = nan(n_part);
    let mut l_part = nan(n_part);
    let mut o_part = nan(n_part * HEAD_DIM as usize);
    let mut out = nan((NUM_HEADS * HEAD_DIM) as usize);
    let max_seq_len = MAX_SEQ_LEN;
    let scale = SCALE;
    unsafe {
        dev.stream
            .launch_builder(&partial)
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(&mut m_part)
            .arg(&mut l_part)
            .arg(&mut o_part)
            .arg(&seq_len)
            .arg(&max_seq_len)
            .arg(&scale)
            .arg(&chunks)
            .arg(&partition)
            .launch(LaunchConfig {
                grid_dim: (chunks, NUM_KV_HEADS, 1),
                block_dim: (BLOCK_DIM, 1, 1),
                shared_mem_bytes: shared_bytes,
            })
            .expect("partial launch");
        dev.stream
            .launch_builder(&merge)
            .arg(&m_part)
            .arg(&l_part)
            .arg(&o_part)
            .arg(&mut out)
            .arg(&chunks)
            .launch(LaunchConfig {
                grid_dim: (NUM_HEADS, DIM_TILES, 1),
                block_dim: (BLOCK_DIM, 1, 1),
                shared_mem_bytes: decode_attention_merge_shared_bytes(chunks),
            })
            .expect("merge launch");
    }
    dev.synchronize().expect("sync");
    Outputs {
        m: dev.dtoh_copy(&m_part).unwrap(),
        l: dev.dtoh_copy(&l_part).unwrap(),
        o: dev.dtoh_copy(&o_part).unwrap(),
        out: dev.dtoh_copy(&out).unwrap(),
    }
}

struct Entry {
    store: &'static str,
    class: &'static str,
    seq_len: u32,
    chunks: u32,
    partition: u32,
    m: String,
    l: String,
    o: String,
    out: String,
}

fn manifest_json(inputs: &[(String, String, String, String)], entries: &[Entry]) -> String {
    let mut s = String::new();
    s.push_str("{\n \"format\": \"attention-decode-reference@1\",\n");
    let _ = writeln!(
        s,
        " \"shape\": {{\"num_heads\": {NUM_HEADS}, \"num_kv_heads\": {NUM_KV_HEADS}, \"head_dim\": {HEAD_DIM}, \"max_seq_len\": {MAX_SEQ_LEN}}},"
    );
    let _ = writeln!(
        s,
        " \"policy\": {{\"one_tile\": {ONE_TILE}, \"target\": {TARGET}}},\n \"scale\": {SCALE:e},"
    );
    s.push_str(" \"generator\": \"lcg x*6364136223846793005+1442695040888963407, >>33; unit = ((x & 0xffffff) / 8388608) - 1, rounded to nearest-even half; q, then k, then v; q scaled after rounding\",\n");
    s.push_str(" \"inputs\": [\n");
    for (i, (class, q, k, v)) in inputs.iter().enumerate() {
        let _ = write!(
            s,
            "  {{\"class\": \"{class}\", \"q\": \"{q}\", \"k16\": \"{k}\", \"v16\": \"{v}\"}}"
        );
        s.push_str(if i + 1 < inputs.len() { ",\n" } else { "\n" });
    }
    s.push_str(" ],\n \"entries\": [\n");
    for (i, e) in entries.iter().enumerate() {
        let _ = write!(
            s,
            "  {{\"store\": \"{}\", \"class\": \"{}\", \"seq_len\": {}, \"chunks\": {}, \"partition\": {}, \"m\": \"{}\", \"l\": \"{}\", \"o\": \"{}\", \"out\": \"{}\"}}",
            e.store, e.class, e.seq_len, e.chunks, e.partition, e.m, e.l, e.o, e.out
        );
        s.push_str(if i + 1 < entries.len() { ",\n" } else { "\n" });
    }
    s.push_str(" ]\n}\n");
    s
}

/// A minimal reader for the manifest this file writes (no JSON crate in the
/// test dependencies): the digests keyed by (store, class, seq_len, field).
fn read_manifest(text: &str) -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        if !line.starts_with("{\"store\"") {
            continue;
        }
        let field = |name: &str| -> String {
            let key = format!("\"{name}\": ");
            let i = line
                .find(&key)
                .unwrap_or_else(|| panic!("manifest line lacks {name}"))
                + key.len();
            let rest = &line[i..];
            let rest = rest.strip_prefix('"').unwrap_or(rest);
            let end = rest
                .find(|c: char| c == '"' || c == ',' || c == '}')
                .unwrap_or(rest.len());
            rest[..end].to_string()
        };
        let store = field("store");
        let class = field("class");
        let seq_len = field("seq_len");
        for name in ["chunks", "partition", "m", "l", "o", "out"] {
            map.insert(format!("{store}/{class}/{seq_len}/{name}"), field(name));
        }
    }
    map
}

#[test]
fn the_kernel_reproduces_the_reference_fixture() {
    let write_dir = std::env::var("LUMEN_ATTN_FIXTURE_WRITE").ok();
    let manifest_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(MANIFEST);
    let expected = if write_dir.is_none() {
        match std::fs::read_to_string(&manifest_path) {
            Ok(t) => Some(read_manifest(&t)),
            Err(_) => {
                eprintln!(
                    "Skipping: no fixture at {} (set LUMEN_ATTN_FIXTURE_WRITE to capture one)",
                    manifest_path.display()
                );
                return;
            }
        }
    } else {
        None
    };
    assert_eq!(
        SPEC,
        DecodeAttentionSpec::for_shape(NUM_HEADS, NUM_KV_HEADS, HEAD_DIM).unwrap()
    );
    let Some(dev) = try_device() else { return };
    if let Some(d) = &write_dir {
        std::fs::create_dir_all(d).expect("fixture dir");
    }
    sweep(&dev, None, &write_dir, &expected);
}

/// The sweep: every class at every length on both stores, through a module
/// compiled at `arch` (NVRTC's default target when `None`), against the
/// manifest when one is given, and into `write_dir` when one is given.
fn sweep(
    dev: &CudaDevice,
    arch: Option<&'static str>,
    write_dir: &Option<String>,
    expected: &Option<std::collections::HashMap<String, String>>,
) {
    let mut input_digests = Vec::new();
    let mut entries = Vec::new();
    let mut mismatches = Vec::new();
    for &(class, seed, q_scale, cancel) in CLASSES {
        let inp = make_inputs(seed, q_scale, cancel);
        input_digests.push((
            class.to_string(),
            sha256(&f32_bytes(&inp.q)),
            sha256(&u16_bytes(&inp.k16)),
            sha256(&u16_bytes(&inp.v16)),
        ));
        let q = dev.htod_copy(&inp.q).unwrap();
        let k32 = dev.htod_copy(&inp.k).unwrap();
        let v32 = dev.htod_copy(&inp.v).unwrap();
        let k16 = dev.htod_copy(&inp.k16).unwrap();
        let v16 = dev.htod_copy(&inp.v16).unwrap();
        for &seq_len in LENGTHS {
            let (chunks, partition) = decode_attention_geometry_within(seq_len, ONE_TILE, TARGET);
            let a = run(
                dev,
                arch,
                PARTIAL_F32,
                SPEC.partial_shared_bytes(false),
                &q,
                &k32,
                &v32,
                seq_len,
                chunks,
                partition,
            );
            let b = run(
                dev,
                arch,
                PARTIAL_F16,
                SPEC.partial_shared_bytes(true),
                &q,
                &k16,
                &v16,
                seq_len,
                chunks,
                partition,
            );
            for (store, o) in [("f32", &a), ("f16", &b)] {
                assert!(
                    o.out.iter().all(|x| x.is_finite()),
                    "{store} {class} {seq_len}: non-finite output"
                );
                let e = Entry {
                    store,
                    class,
                    seq_len,
                    chunks,
                    partition,
                    m: sha256(&f32_bytes(&o.m)),
                    l: sha256(&f32_bytes(&o.l)),
                    o: sha256(&f32_bytes(&o.o)),
                    out: sha256(&f32_bytes(&o.out)),
                };
                if let Some(d) = &write_dir {
                    for (name, data) in [("m", &o.m), ("l", &o.l), ("o", &o.o), ("out", &o.out)] {
                        std::fs::write(
                            format!("{d}/{store}-{class}-{seq_len}.{name}.f32"),
                            f32_bytes(data),
                        )
                        .expect("write");
                    }
                }
                if let Some(exp) = &expected {
                    for (name, got) in [
                        ("chunks", chunks.to_string()),
                        ("partition", partition.to_string()),
                        ("m", e.m.clone()),
                        ("l", e.l.clone()),
                        ("o", e.o.clone()),
                        ("out", e.out.clone()),
                    ] {
                        let key = format!("{store}/{class}/{seq_len}/{name}");
                        match exp.get(&key) {
                            Some(want) if *want == got => {}
                            Some(want) => {
                                mismatches.push(format!("{key}: fixture {want}, kernel {got}"))
                            }
                            None => mismatches.push(format!("{key}: not in the fixture")),
                        }
                    }
                }
                entries.push(e);
            }
            // Half-representable inputs: the two stores agree bit for bit.
            assert!(
                a.out
                    .iter()
                    .zip(&b.out)
                    .all(|(x, y)| x.to_bits() == y.to_bits()),
                "{class} {seq_len}: the half store's output differs from the F32 store's"
            );
        }
        eprintln!("class {class}: {} contexts on both stores", LENGTHS.len());
    }
    if let Some(d) = &write_dir {
        let text = manifest_json(&input_digests, &entries);
        std::fs::write(format!("{d}/manifest.json"), &text).expect("manifest");
        eprintln!("wrote {} entries to {d}/manifest.json", entries.len());
    }
    if let Some(exp) = &expected {
        // Every fixture entry must have been visited, or a shortened sweep
        // would pass vacuously.
        let visited = entries.len() * 6;
        assert_eq!(
            visited,
            exp.len(),
            "the sweep visited {visited} digests but the fixture holds {}",
            exp.len()
        );
        assert!(
            mismatches.is_empty(),
            "{} digest(s) differ from the reference fixture:\n{}",
            mismatches.len(),
            mismatches.join("\n")
        );
    }
}

/// The knob `LUMEN_CUDA_ATTN_CODEGEN` compiles the same source for an
/// explicit virtual target. Each target the toolkit offers must reproduce the
/// fixture bit for bit too: a target that changed the bits would make the
/// knob a numerics change, and the fixture would say so here. A target the
/// toolkit refuses is skipped, named.
#[test]
fn every_nvrtc_target_reproduces_the_reference_fixture() {
    let manifest_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(MANIFEST);
    let expected = match std::fs::read_to_string(&manifest_path) {
        Ok(t) => Some(read_manifest(&t)),
        Err(_) => {
            eprintln!("Skipping: no fixture at {}", manifest_path.display());
            return;
        }
    };
    let Some(dev) = try_device() else { return };
    let mut ran = Vec::new();
    for arch in ["compute_80", "compute_120"] {
        if dev
            .compile_and_load_with_arch(&SPEC.source(), arch)
            .is_err()
        {
            eprintln!("Skipping {arch}: the toolkit or driver refused it");
            continue;
        }
        eprintln!("target {arch}");
        sweep(&dev, Some(arch), &None, &expected);
        ran.push(arch);
    }
    // A toolkit that refuses every explicit target has compared nothing;
    // say so rather than pass. (compute_80 is offered by every toolkit the
    // engine supports.)
    assert!(
        !ran.is_empty(),
        "no explicit NVRTC target could be compiled; the sweep compared nothing"
    );
    eprintln!("targets reproduced the fixture: {ran:?}");
}

/// The digest function is the fixture's identity; pin it to a published
/// vector so a wrong implementation cannot pin the wrong thing. Runs
/// without a GPU.
#[test]
fn the_digest_is_sha256() {
    assert_eq!(
        sha256(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        sha256(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    let million = vec![b'a'; 1_000_000];
    assert_eq!(
        sha256(&million),
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
    );
}
