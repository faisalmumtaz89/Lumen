//! Standalone NVRTC A/B harness for the decode-attention split-K pair.
//!
//! Nothing here touches the engine's dispatch: the example compiles the
//! kernels itself, drives them at the production geometry, and prints one
//! JSON record. Build/run:
//!
//! ```text
//! cargo build --release --features cuda --example attn_decode_ab -p lumen-runtime
//! ./target/release/examples/attn_decode_ab --out results.json
//! ```
//!
//! Requires the `cuda` feature (declared as `required-features` in
//! `Cargo.toml`, so a non-CUDA build skips the target entirely).
//!
//! ## Variants
//!
//! | id     | partial pass                                           | merge   |
//! |--------|--------------------------------------------------------|---------|
//! | `A`    | `attention_decode_splitk_partial` (shipping, `9e116ab`) | shipping |
//! | `B`    | `attention_decode_splitk_partial_warp` (`dd6b12f`)      | shipping |
//! | `C`    | `attention_decode_splitk_partial_gqa` (`4849eb4`)       | shipping |
//! | `D16`  | `attention_decode_splitk_partial_gqa6_f32`, chunk 16    | gqa6     |
//! | `D32`  | `attention_decode_splitk_partial_gqa6_f32`, chunk 32    | gqa6     |
//!
//! `D` compiles the SHIPPED pair out of
//! `lumen_runtime::cuda::shaders::ATTENTION_DECODE_SPLITK_GQA6_KERNEL_SOURCE`
//! — the same string the engine's kernel loader hands to NVRTC — and launches
//! it with the geometry `prefill::launch_attention_decode_splitk_gqa6` uses,
//! taking the tile count from the crate. So its timings describe the kernel
//! the engine runs, not a copy of it.
//! | `TILED`| `attention_decode_tiled` (single pass, reference only)  | —        |
//!
//! `A`/`B` reproduce the production launch geometry read out of
//! `cuda/prefill.rs::launch_attention_decode_splitk` at `9e116ab`: block
//! `ATTN_DECODE_TILED_BLOCK_DIM = 128`, shared
//! `attention_decode_tiled_shared_bytes(head_dim) = (8 + head_dim + 128) * 4`,
//! partial grid `num_heads * S`, merge grid `num_heads`, and
//! `S = attn_splitk_chunks(seq_len) = ceil(seq_len / 128)` clamped to
//! `[1, ATTN_SPLITK_S_MAX = 32]`. Those constants are mirrored below as
//! `PROD_*` with the source they came from.
//!
//! ## Measurement
//!
//! Two cache states. **warm** repeats on the same ~9 MB K/V working set,
//! which on a 5090 fits many times over in L2 — the optimistic bound.
//! **evicted** writes a buffer sized at 1.5x the device L2 between samples,
//! so K/V come from DRAM, mirroring real decode where the weight matvecs
//! stream through L2 between attention layers.
//!
//! Timing is CUDA events, one sample = one synchronised launch pair, 20
//! warm-up + 200 timed repetitions, variants round-robined inside each
//! repetition so drift hits every variant equally. Two passes:
//!   * `pair`  — two events around (partial, merge): the headline number.
//!   * `split` — three events, so the intervening `cuEventRecord` (~sub-µs)
//!     sits between the two kernels; used only for the per-kernel split.
//!
//! Correctness is against an F64 host reference at every length, for three
//! input sets (uniform, an 8x-scaled-Q large-score-spread stress case, and
//! an all-zero-Q case whose exact answer is the mean of V).

use std::collections::BTreeMap;

use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
use lumen_runtime::cuda::ffi::CudaDevice;
use lumen_runtime::cuda::shaders::{
    ATTENTION_DECODE_SPLITK_GQA6_KERNEL_SOURCE, ATTENTION_DECODE_SPLITK_KERNEL_SOURCE,
    ATTENTION_DECODE_TILED_KERNEL_SOURCE,
};
use lumen_runtime::cuda::ATTN_SPLITK_GQA6_DIM_TILES;

// --- Model geometry (Qwen3.8-27B full-attention layer) ---------------------
const NUM_HEADS: u32 = 24;
const NUM_KV_HEADS: u32 = 4;
const HEAD_DIM: u32 = 256;
const MAX_SEQ_LEN: u32 = 4096;
const SCALE: f32 = 0.0625; // 1 / sqrt(256)
const GQA_G: u32 = NUM_HEADS / NUM_KV_HEADS; // 6

// --- Production launch constants, mirrored from the engine -----------------
/// `cuda/decode.rs::ATTN_DECODE_TILED_BLOCK_DIM`.
const PROD_BLOCK_DIM: u32 = 128;
/// `cuda/decode.rs::ATTN_DECODE_TILED_T_C` (the `+ ATTN_DECODE_TILED_T_C`
/// term of `attention_decode_tiled_shared_bytes`).
const PROD_T_C: u32 = 128;
/// `runtime_defaults::ATTN_SPLITK_CHUNK_POSITIONS`.
const PROD_CHUNK_POSITIONS: u32 = 128;
/// `cuda/prefill.rs::ATTN_SPLITK_S_MAX`.
const PROD_S_MAX: u32 = 32;

/// `cuda/decode.rs::attention_decode_tiled_shared_bytes`.
const fn prod_shared_bytes(head_dim: u32) -> u32 {
    (8 + head_dim + PROD_T_C) * 4
}

/// `cuda/prefill.rs::attn_splitk_chunks` with the canonical defaults
/// (`LUMEN_CUDA_ATTN_SPLITK_SCALE` on, no `LUMEN_CUDA_ATTN_SPLITK_CHUNK`).
fn prod_splitk_chunks(seq_len: u32) -> u32 {
    seq_len.div_ceil(PROD_CHUNK_POSITIONS).clamp(1, PROD_S_MAX)
}

/// Candidate D's split count: one chunk per `c` positions, uncapped (the
/// scratch is sized for the resulting `S_MAX = ceil(4096 / 16) = 256`).
fn d_chunks(seq_len: u32, c: u32) -> u32 {
    seq_len.div_ceil(c).max(1)
}

/// The span the kernels actually walk: `ceil(seq_len / S)`. Identical
/// formula in every partial pass here.
fn span_of(seq_len: u32, chunks: u32) -> u32 {
    seq_len.div_ceil(chunks)
}

const D_S_MAX: u32 = 256;
const SCRATCH_S_MAX: u32 = D_S_MAX;

// --- Measurement plan ------------------------------------------------------
/// Every length gets a correctness check. Board shape is 1024-1152.
const ALL_LENGTHS: &[u32] = &[
    1, 15, 16, 17, 127, 128, 129, 330, 1024, 1100, 1152, 1300, 2600, 4096,
];
/// The timed positions: the board shape and the contexts either side of it.
const TIMED_LENGTHS: &[u32] = &[330, 1024, 1100, 1152, 1300, 2600];
const WARMUP_REPS: usize = 20;
const TIMED_REPS: usize = 200;

// ---------------------------------------------------------------------------
// Variants
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Variant {
    A,
    B,
    C,
    D { chunk: u32 },
    Tiled,
}

impl Variant {
    fn id(self) -> String {
        match self {
            Variant::A => "A".into(),
            Variant::B => "B".into(),
            Variant::C => "C".into(),
            Variant::D { chunk } => format!("D{chunk}"),
            Variant::Tiled => "TILED".into(),
        }
    }

    fn description(self) -> String {
        match self {
            Variant::A => "attention_decode_splitk_partial + _merge (shipping, 9e116ab)".into(),
            Variant::B => "attention_decode_splitk_partial_warp (dd6b12f) + shipping merge".into(),
            Variant::C => "attention_decode_splitk_partial_gqa (4849eb4) + shipping merge".into(),
            Variant::D { chunk } => format!(
                "attention_decode_splitk_partial_gqa6_f32 (C={chunk}) + \
                 _merge_gqa6_f32, both from src/cuda/shaders"
            ),
            Variant::Tiled => "attention_decode_tiled (single pass, no merge)".into(),
        }
    }

    /// Whether the variant can legally serve `seq_len` at this geometry.
    fn supports(self, seq_len: u32) -> bool {
        match self {
            Variant::A | Variant::B | Variant::Tiled => true,
            // 4849eb4 requires the chunk span to fit one T_C tile and the
            // GQA group to fit GQA_MAX = 8.
            Variant::C => span_of(seq_len, prod_splitk_chunks(seq_len)) <= PROD_T_C && GQA_G <= 8,
            // The gqa6 softmax gives one warp to a head, so a chunk must fit
            // 32 lanes; the scratch is sized for S <= 256.
            Variant::D { chunk, .. } => {
                let s = d_chunks(seq_len, chunk);
                chunk <= 32 && s <= D_S_MAX && span_of(seq_len, s) <= chunk
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel handles
// ---------------------------------------------------------------------------

struct Kernels {
    splitk_partial: CudaFunction,
    splitk_merge: CudaFunction,
    warp_partial: CudaFunction,
    gqa_partial: CudaFunction,
    gqa6_partial: CudaFunction,
    gqa6_merge: CudaFunction,
    tiled: CudaFunction,
    flush: CudaFunction,
}

struct Buffers {
    q: CudaSlice<f32>,
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
    m_part: CudaSlice<f32>,
    l_part: CudaSlice<f32>,
    o_part: CudaSlice<f32>,
    out: CudaSlice<f32>,
    flush: CudaSlice<f32>,
    flush_len: u32,
}

/// Launch the partial pass of `variant` at `seq_len`.
///
/// # Safety
/// Buffers must be sized as allocated in `main` and the variant must have
/// passed `Variant::supports`.
unsafe fn launch_partial(
    dev: &CudaDevice,
    k: &Kernels,
    b: &mut Buffers,
    variant: Variant,
    seq_len: u32,
) {
    let nh = NUM_HEADS;
    let nkv = NUM_KV_HEADS;
    let hd = HEAD_DIM;
    let msl = MAX_SEQ_LEN;
    let scale = SCALE;

    match variant {
        Variant::A | Variant::B => {
            let s = prod_splitk_chunks(seq_len);
            let func = if variant == Variant::A {
                &k.splitk_partial
            } else {
                &k.warp_partial
            };
            dev.stream
                .launch_builder(func)
                .arg(&b.q)
                .arg(&b.k)
                .arg(&b.v)
                .arg(&mut b.m_part)
                .arg(&mut b.l_part)
                .arg(&mut b.o_part)
                .arg(&nh)
                .arg(&nkv)
                .arg(&hd)
                .arg(&seq_len)
                .arg(&msl)
                .arg(&scale)
                .arg(&s)
                .launch(LaunchConfig {
                    grid_dim: (nh * s, 1, 1),
                    block_dim: (PROD_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: prod_shared_bytes(hd),
                })
                .expect("partial launch");
        }
        Variant::C => {
            let s = prod_splitk_chunks(seq_len);
            // 8 (reductions) + G*head_dim (q rows) + G*T_C (scores) floats.
            let shared = (8 + GQA_G * hd + GQA_G * PROD_T_C) * 4;
            dev.stream
                .launch_builder(&k.gqa_partial)
                .arg(&b.q)
                .arg(&b.k)
                .arg(&b.v)
                .arg(&mut b.m_part)
                .arg(&mut b.l_part)
                .arg(&mut b.o_part)
                .arg(&nh)
                .arg(&nkv)
                .arg(&hd)
                .arg(&seq_len)
                .arg(&msl)
                .arg(&scale)
                .arg(&s)
                .launch(LaunchConfig {
                    grid_dim: (nkv * s, 1, 1),
                    block_dim: (PROD_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: shared,
                })
                .expect("gqa partial launch");
        }
        Variant::D { chunk, .. } => {
            let s = d_chunks(seq_len, chunk);
            // 6*256 (Q) + C*256 (V) + 6*C (scores) + 12 (m, l) floats.
            let shared = (GQA_G * hd + chunk * hd + GQA_G * chunk + 12) * 4;
            dev.stream
                .launch_builder(&k.gqa6_partial)
                .arg(&b.q)
                .arg(&b.k)
                .arg(&b.v)
                .arg(&mut b.m_part)
                .arg(&mut b.l_part)
                .arg(&mut b.o_part)
                .arg(&seq_len)
                .arg(&msl)
                .arg(&scale)
                .arg(&s)
                .arg(&chunk)
                .launch(LaunchConfig {
                    grid_dim: (s, nkv, 1),
                    block_dim: (PROD_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: shared,
                })
                .expect("gqa6 partial launch");
        }
        Variant::Tiled => {
            dev.stream
                .launch_builder(&k.tiled)
                .arg(&b.q)
                .arg(&b.k)
                .arg(&b.v)
                .arg(&mut b.out)
                .arg(&nh)
                .arg(&nkv)
                .arg(&hd)
                .arg(&seq_len)
                .arg(&msl)
                .arg(&scale)
                .launch(LaunchConfig {
                    grid_dim: (nh, 1, 1),
                    block_dim: (PROD_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: prod_shared_bytes(hd),
                })
                .expect("tiled launch");
        }
    }
}

/// Launch the merge pass of `variant`. `Variant::Tiled` has none.
///
/// # Safety
/// Same contract as [`launch_partial`].
unsafe fn launch_merge(
    dev: &CudaDevice,
    k: &Kernels,
    b: &mut Buffers,
    variant: Variant,
    seq_len: u32,
) {
    let nh = NUM_HEADS;
    let hd = HEAD_DIM;
    match variant {
        Variant::Tiled => {}
        Variant::A | Variant::B | Variant::C => {
            let s = prod_splitk_chunks(seq_len);
            dev.stream
                .launch_builder(&k.splitk_merge)
                .arg(&b.m_part)
                .arg(&b.l_part)
                .arg(&b.o_part)
                .arg(&mut b.out)
                .arg(&nh)
                .arg(&hd)
                .arg(&s)
                .launch(LaunchConfig {
                    grid_dim: (nh, 1, 1),
                    block_dim: (PROD_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                })
                .expect("merge launch");
        }
        Variant::D { chunk } => {
            let s = d_chunks(seq_len, chunk);
            let shared = (s + 4) * 4;
            dev.stream
                .launch_builder(&k.gqa6_merge)
                .arg(&b.m_part)
                .arg(&b.l_part)
                .arg(&b.o_part)
                .arg(&mut b.out)
                .arg(&s)
                .launch(LaunchConfig {
                    grid_dim: (nh, ATTN_SPLITK_GQA6_DIM_TILES, 1),
                    block_dim: (PROD_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: shared,
                })
                .expect("gqa6 merge launch");
        }
    }
}

/// # Safety
/// `b.flush` must hold `b.flush_len` floats.
unsafe fn launch_flush(dev: &CudaDevice, k: &Kernels, b: &mut Buffers, val: f32) {
    let n = b.flush_len;
    dev.stream
        .launch_builder(&k.flush)
        .arg(&mut b.flush)
        .arg(&n)
        .arg(&val)
        .launch(LaunchConfig {
            grid_dim: (2048, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })
        .expect("flush launch");
}

// ---------------------------------------------------------------------------
// Host reference and inputs
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [-1, 1).
    fn next_f32(&mut self) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32;
        u * 2.0 - 1.0
    }
}

struct Inputs {
    name: &'static str,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
}

fn make_inputs(seed: u64) -> Vec<Inputs> {
    let kv_len = (NUM_KV_HEADS * MAX_SEQ_LEN * HEAD_DIM) as usize;
    let q_len = (NUM_HEADS * HEAD_DIM) as usize;
    let mut rng = Rng(seed);
    let q: Vec<f32> = (0..q_len).map(|_| rng.next_f32()).collect();
    let k: Vec<f32> = (0..kv_len).map(|_| rng.next_f32()).collect();
    let v: Vec<f32> = (0..kv_len).map(|_| rng.next_f32()).collect();
    vec![
        Inputs {
            name: "uniform",
            q: q.clone(),
            k: k.clone(),
            v: v.clone(),
        },
        // Large score spread: |q| up to 8 lifts the pre-softmax dot products
        // by 8x, so chunk-local maxima diverge and the merge's rescale is
        // exercised hard.
        Inputs {
            name: "stress_q8",
            q: q.iter().map(|x| x * 8.0).collect(),
            k: k.clone(),
            v: v.clone(),
        },
        // Exact answer: the softmax is uniform, so out = mean of V.
        Inputs {
            name: "zero_q",
            q: vec![0.0; q_len],
            k,
            v,
        },
    ]
}

/// F64 single-pass reference for the whole decode-attention output.
fn reference(inp: &Inputs, seq_len: u32) -> Vec<f64> {
    let hd = HEAD_DIM as usize;
    let seq = seq_len as usize;
    let mut out = vec![0.0f64; (NUM_HEADS * HEAD_DIM) as usize];
    let mut s = vec![0.0f64; seq];
    for h in 0..NUM_HEADS as usize {
        let kv = h / GQA_G as usize;
        let base = kv * MAX_SEQ_LEN as usize * hd;
        let mut m = f64::NEG_INFINITY;
        for (p, sp) in s.iter_mut().enumerate() {
            let mut dot = 0.0f64;
            for i in 0..hd {
                dot += f64::from(inp.q[h * hd + i]) * f64::from(inp.k[base + p * hd + i]);
            }
            *sp = dot * f64::from(SCALE);
            m = m.max(*sp);
        }
        let mut l = 0.0f64;
        for sp in s.iter_mut() {
            *sp = (*sp - m).exp();
            l += *sp;
        }
        for i in 0..hd {
            let mut acc = 0.0f64;
            for (p, sp) in s.iter().enumerate() {
                acc += *sp * f64::from(inp.v[base + p * hd + i]);
            }
            out[h * hd + i] = acc / l;
        }
    }
    out
}

/// (max abs error, relative L2 error) of `got` against the F64 `refv`.
fn errors(got: &[f32], refv: &[f64]) -> (f64, f64) {
    let mut max_abs = 0.0f64;
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (g, r) in got.iter().zip(refv.iter()) {
        let d = f64::from(*g) - r;
        max_abs = max_abs.max(d.abs());
        num += d * d;
        den += r * r;
    }
    (
        max_abs,
        if den > 0.0 {
            (num / den).sqrt()
        } else {
            num.sqrt()
        },
    )
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx]
}

#[derive(Default, Clone)]
struct Stat {
    median: f64,
    p10: f64,
    p90: f64,
}

fn stat(mut v: Vec<f64>) -> Stat {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Stat {
        median: pct(&v, 0.5),
        p10: pct(&v, 0.10),
        p90: pct(&v, 0.90),
    }
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn main() {
    let mut out_path = "attn_decode_ab.json".to_string();
    let mut modes: Vec<&str> = vec!["warm", "evicted"];
    let mut only: Option<String> = None;
    let mut only_len: Option<u32> = None;
    let mut skip_correctness = false;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                out_path = args[i + 1].clone();
                i += 1;
            }
            "--mode" => {
                modes = if args[i + 1] == "warm" {
                    vec!["warm"]
                } else if args[i + 1] == "evicted" {
                    vec!["evicted"]
                } else {
                    vec!["warm", "evicted"]
                };
                i += 1;
            }
            // NCU driver: one launch of one variant at one length, nothing else.
            "--only" => {
                only = Some(args[i + 1].clone());
                i += 1;
            }
            "--len" => {
                only_len = Some(args[i + 1].parse().expect("--len"));
                i += 1;
            }
            "--no-correctness" => skip_correctness = true,
            other => panic!("unknown argument {other}"),
        }
        i += 1;
    }

    let dev = CudaDevice::new(0).expect("CUDA device 0");
    let name = dev.name().unwrap_or_default();
    let (cc_major, cc_minor) = dev.compute_capability().unwrap_or((0, 0));
    use cudarc::driver::sys::CUdevice_attribute;
    let l2_bytes = dev
        .ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
        .unwrap_or(0) as i64;
    let sm_count = dev
        .ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        .unwrap_or(0);

    // Match main's NVRTC settings: `compile_and_load` (default target, no
    // fast math), exactly what `cuda/decode.rs` uses for this pair.
    let m_ship = dev
        .compile_and_load(ATTENTION_DECODE_SPLITK_KERNEL_SOURCE)
        .expect("compile shipping split-K");
    let m_tiled = dev
        .compile_and_load(ATTENTION_DECODE_TILED_KERNEL_SOURCE)
        .expect("compile tiled");
    let m_warp = dev
        .compile_and_load(include_str!("attn_decode_ab_cu/warp_dd6b12f.cu"))
        .expect("compile dd6b12f");
    let m_gqa = dev
        .compile_and_load(include_str!("attn_decode_ab_cu/gqa_4849eb4.cu"))
        .expect("compile 4849eb4");
    // Candidate D is the SHIPPED shader, compiled the way the engine's kernel
    // loader compiles it — not a copy of it.
    let m_gqa6 = dev
        .compile_and_load(ATTENTION_DECODE_SPLITK_GQA6_KERNEL_SOURCE)
        .expect("compile gqa6");
    let m_flush = dev
        .compile_and_load(include_str!("attn_decode_ab_cu/flush.cu"))
        .expect("compile flush");

    let kern = Kernels {
        splitk_partial: m_ship
            .load_function("attention_decode_splitk_partial")
            .unwrap(),
        splitk_merge: m_ship
            .load_function("attention_decode_splitk_merge")
            .unwrap(),
        warp_partial: m_warp
            .load_function("attention_decode_splitk_partial_warp")
            .unwrap(),
        gqa_partial: m_gqa
            .load_function("attention_decode_splitk_partial_gqa")
            .unwrap(),
        gqa6_partial: m_gqa6
            .load_function("attention_decode_splitk_partial_gqa6_f32")
            .unwrap(),
        gqa6_merge: m_gqa6
            .load_function("attention_decode_splitk_merge_gqa6_f32")
            .unwrap(),
        tiled: m_tiled.load_function("attention_decode_tiled").unwrap(),
        flush: m_flush.load_function("gqa6_l2_flush").unwrap(),
    };

    // Fixed seed: every run of this harness sees the same Q/K/V.
    let inputs = make_inputs(0x0005_0904_A11E_0001_u64);
    let flush_floats = ((l2_bytes as f64 * 1.5) as usize / 4).max(1 << 20);

    let mut bufs = Buffers {
        q: dev.htod_copy(&inputs[0].q).expect("q"),
        k: dev.htod_copy(&inputs[0].k).expect("k"),
        v: dev.htod_copy(&inputs[0].v).expect("v"),
        m_part: dev
            .alloc_zeros::<f32>((NUM_HEADS * SCRATCH_S_MAX) as usize)
            .expect("m_part"),
        l_part: dev
            .alloc_zeros::<f32>((NUM_HEADS * SCRATCH_S_MAX) as usize)
            .expect("l_part"),
        o_part: dev
            .alloc_zeros::<f32>((NUM_HEADS * SCRATCH_S_MAX * HEAD_DIM) as usize)
            .expect("o_part"),
        out: dev
            .alloc_zeros::<f32>((NUM_HEADS * HEAD_DIM) as usize)
            .expect("out"),
        flush: dev.alloc_zeros::<f32>(flush_floats).expect("flush"),
        flush_len: flush_floats as u32,
    };

    let all_variants = [
        Variant::A,
        Variant::B,
        Variant::C,
        Variant::D { chunk: 16 },
        Variant::D { chunk: 32 },
    ];

    // --- NCU single-shot mode ---------------------------------------------
    if let Some(sel) = &only {
        let seq = only_len.expect("--only requires --len");
        let v = all_variants
            .iter()
            .copied()
            .chain(std::iter::once(Variant::Tiled))
            .find(|v| &v.id() == sel)
            .expect("unknown variant id");
        assert!(v.supports(seq), "{sel} does not support seq_len {seq}");
        unsafe {
            // One warm-up outside the profiled region is not possible with a
            // single-shot ncu filter, so profile exactly what runs: two
            // launches, the second of which ncu's `--launch-skip 1` keeps.
            for _ in 0..2 {
                launch_partial(&dev, &kern, &mut bufs, v, seq);
                launch_merge(&dev, &kern, &mut bufs, v, seq);
            }
        }
        dev.synchronize().expect("sync");
        println!("ncu single-shot: {sel} at seq_len {seq} done");
        return;
    }

    // --- Correctness -------------------------------------------------------
    let mut correctness: Vec<String> = Vec::new();
    if !skip_correctness {
        for inp in &inputs {
            dev.htod_copy_into(&inp.q, &mut bufs.q).expect("q up");
            dev.htod_copy_into(&inp.k, &mut bufs.k).expect("k up");
            dev.htod_copy_into(&inp.v, &mut bufs.v).expect("v up");
            for &seq in ALL_LENGTHS {
                let refv = reference(inp, seq);
                for v in all_variants.iter().copied().chain([Variant::Tiled]) {
                    if !v.supports(seq) {
                        continue;
                    }
                    dev.stream.memset_zeros(&mut bufs.out).expect("zero out");
                    unsafe {
                        launch_partial(&dev, &kern, &mut bufs, v, seq);
                        launch_merge(&dev, &kern, &mut bufs, v, seq);
                    }
                    dev.synchronize().expect("sync");
                    let got = dev.dtoh_copy(&bufs.out).expect("dtoh");
                    let (max_abs, rel_l2) = errors(&got, &refv);
                    correctness.push(format!(
                        "{{\"input\":\"{}\",\"seq_len\":{},\"variant\":\"{}\",\
                         \"max_abs_err\":{:.6e},\"rel_l2_err\":{:.6e}}}",
                        inp.name,
                        seq,
                        v.id(),
                        max_abs,
                        rel_l2
                    ));
                }
            }
        }
        // Restore the uniform input for the timing phase.
        dev.htod_copy_into(&inputs[0].q, &mut bufs.q).expect("q");
        dev.htod_copy_into(&inputs[0].k, &mut bufs.k).expect("k");
        dev.htod_copy_into(&inputs[0].v, &mut bufs.v).expect("v");
        dev.synchronize().expect("sync");
        eprintln!("correctness: {} records", correctness.len());
    }

    // --- Timing ------------------------------------------------------------
    use cudarc::driver::sys::CUevent_flags;
    let ev: Vec<_> = (0..3)
        .map(|_| {
            dev.ctx
                .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .expect("event")
        })
        .collect();

    let mut timing: Vec<String> = Vec::new();
    for &mode in &modes {
        let evict = mode == "evicted";
        for &seq in TIMED_LENGTHS {
            let live: Vec<Variant> = all_variants
                .iter()
                .copied()
                .chain(if seq == 330 {
                    vec![Variant::Tiled]
                } else {
                    vec![]
                })
                .filter(|v| v.supports(seq))
                .collect();

            let mut pair: BTreeMap<String, Vec<f64>> = BTreeMap::new();
            let mut part: BTreeMap<String, Vec<f64>> = BTreeMap::new();
            let mut merg: BTreeMap<String, Vec<f64>> = BTreeMap::new();

            // Pass 1: two events around the pair (headline).
            for rep in 0..(WARMUP_REPS + TIMED_REPS) {
                for &v in &live {
                    unsafe {
                        if evict {
                            launch_flush(&dev, &kern, &mut bufs, rep as f32);
                        }
                        ev[0].record(&dev.stream).expect("ev0");
                        launch_partial(&dev, &kern, &mut bufs, v, seq);
                        launch_merge(&dev, &kern, &mut bufs, v, seq);
                        ev[2].record(&dev.stream).expect("ev2");
                    }
                    dev.synchronize().expect("sync");
                    if rep >= WARMUP_REPS {
                        let us = f64::from(ev[0].elapsed_ms(&ev[2]).expect("elapsed")) * 1000.0;
                        pair.entry(v.id()).or_default().push(us);
                    }
                }
            }
            // Pass 2: three events, per-kernel split.
            for rep in 0..(WARMUP_REPS + TIMED_REPS) {
                for &v in &live {
                    unsafe {
                        if evict {
                            launch_flush(&dev, &kern, &mut bufs, rep as f32);
                        }
                        ev[0].record(&dev.stream).expect("ev0");
                        launch_partial(&dev, &kern, &mut bufs, v, seq);
                        ev[1].record(&dev.stream).expect("ev1");
                        launch_merge(&dev, &kern, &mut bufs, v, seq);
                        ev[2].record(&dev.stream).expect("ev2");
                    }
                    dev.synchronize().expect("sync");
                    if rep >= WARMUP_REPS {
                        let p = f64::from(ev[0].elapsed_ms(&ev[1]).expect("e01")) * 1000.0;
                        let m = f64::from(ev[1].elapsed_ms(&ev[2]).expect("e12")) * 1000.0;
                        part.entry(v.id()).or_default().push(p);
                        merg.entry(v.id()).or_default().push(m);
                    }
                }
            }

            let kv_bytes = f64::from(seq) * 8192.0; // 2 * seq * 4 kv heads * 256 * 4 B
            for v in &live {
                let id = v.id();
                let sp = stat(pair.remove(&id).unwrap_or_default());
                let ss = stat(part.remove(&id).unwrap_or_default());
                let sm = stat(merg.remove(&id).unwrap_or_default());
                let bw = kv_bytes / (sp.median * 1e-6) / 1e12; // TB/s
                let floor_us = kv_bytes / 1.79e12 * 1e6;
                timing.push(format!(
                    "{{\"mode\":\"{}\",\"seq_len\":{},\"variant\":\"{}\",\
                     \"pair_us\":{{\"median\":{:.3},\"p10\":{:.3},\"p90\":{:.3}}},\
                     \"partial_us\":{{\"median\":{:.3},\"p10\":{:.3},\"p90\":{:.3}}},\
                     \"merge_us\":{{\"median\":{:.3},\"p10\":{:.3},\"p90\":{:.3}}},\
                     \"kv_bytes\":{},\"achieved_tbs\":{:.4},\
                     \"f32_floor_us_at_1.79TBs\":{:.3},\"ratio_to_floor\":{:.3},\
                     \"grid\":\"{}\"}}",
                    mode,
                    seq,
                    id,
                    sp.median,
                    sp.p10,
                    sp.p90,
                    ss.median,
                    ss.p10,
                    ss.p90,
                    sm.median,
                    sm.p10,
                    sm.p90,
                    kv_bytes as u64,
                    bw,
                    floor_us,
                    sp.median / floor_us,
                    json_escape(&grid_note(*v, seq)),
                ));
            }
            eprintln!("timed mode={mode} seq_len={seq}");
        }
    }

    let json = format!(
        "{{\n  \"harness\": \"attn_decode_ab\",\n  \"device\": \"{}\",\n  \
         \"compute_capability\": \"{}.{}\",\n  \"sm_count\": {},\n  \
         \"l2_bytes\": {},\n  \"flush_bytes\": {},\n  \
         \"geometry\": {{\"num_heads\":{},\"num_kv_heads\":{},\"head_dim\":{},\
         \"max_seq_len\":{},\"scale\":{}}},\n  \
         \"reps\": {{\"warmup\":{},\"timed\":{}}},\n  \
         \"variants\": [{}],\n  \"correctness\": [\n    {}\n  ],\n  \
         \"timing\": [\n    {}\n  ]\n}}\n",
        json_escape(&name),
        cc_major,
        cc_minor,
        sm_count,
        l2_bytes,
        flush_floats * 4,
        NUM_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        MAX_SEQ_LEN,
        SCALE,
        WARMUP_REPS,
        TIMED_REPS,
        all_variants
            .iter()
            .copied()
            .chain([Variant::Tiled])
            .map(|v| format!(
                "{{\"id\":\"{}\",\"desc\":\"{}\"}}",
                v.id(),
                json_escape(&v.description())
            ))
            .collect::<Vec<_>>()
            .join(","),
        correctness.join(",\n    "),
        timing.join(",\n    "),
    );
    std::fs::write(&out_path, &json).expect("write json");
    println!("{json}");
    eprintln!("wrote {out_path}");
}

/// Human-readable launch geometry for the record.
fn grid_note(v: Variant, seq_len: u32) -> String {
    match v {
        Variant::A | Variant::B => {
            let s = prod_splitk_chunks(seq_len);
            format!(
                "partial grid={}x1 (heads*S, S={}, span={}), block={}, smem={}B; merge grid={}",
                NUM_HEADS * s,
                s,
                span_of(seq_len, s),
                PROD_BLOCK_DIM,
                prod_shared_bytes(HEAD_DIM),
                NUM_HEADS
            )
        }
        Variant::C => {
            let s = prod_splitk_chunks(seq_len);
            format!(
                "partial grid={}x1 (kv_heads*S, S={}, span={}), block={}, smem={}B; merge grid={}",
                NUM_KV_HEADS * s,
                s,
                span_of(seq_len, s),
                PROD_BLOCK_DIM,
                (8 + GQA_G * HEAD_DIM + GQA_G * PROD_T_C) * 4,
                NUM_HEADS
            )
        }
        Variant::D { chunk } => {
            let s = d_chunks(seq_len, chunk);
            format!(
                "partial grid=({},{}) (S={}, C={}, span={}), block={}, smem={}B; \
                 merge grid=({},{})",
                s,
                NUM_KV_HEADS,
                s,
                chunk,
                span_of(seq_len, s),
                PROD_BLOCK_DIM,
                (GQA_G * HEAD_DIM + chunk * HEAD_DIM + GQA_G * chunk + 12) * 4,
                NUM_HEADS,
                ATTN_SPLITK_GQA6_DIM_TILES
            )
        }
        Variant::Tiled => format!(
            "grid={}, block={}, smem={}B; no merge",
            NUM_HEADS,
            PROD_BLOCK_DIM,
            prod_shared_bytes(HEAD_DIM)
        ),
    }
}
