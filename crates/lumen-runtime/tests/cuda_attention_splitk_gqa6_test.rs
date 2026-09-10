//! Correctness suite for the GQA-shared split-K decode-attention pair
//! (`attention_decode_splitk_partial_gqa6_f32` +
//! `attention_decode_splitk_merge_gqa6_f32`, `LUMEN_CUDA_ATTN_SPLITK_GQA6`).
//!
//! Requires the `cuda` feature; the GPU cases are skipped where there is no
//! device, and `the_error_helper_rejects_a_poisoned_result` runs anywhere.
//! Run:
//!     cargo test --release -p lumen-runtime --features cuda \
//!         --test cuda_attention_splitk_gqa6_test
//!
//! Every case is scored two ways: against an F64 host reference computed in
//! one pass (the ground truth), and against the shipping
//! `attention_decode_splitk` pair on the same inputs (the route this one
//! replaces). The two kernels reassociate their sums differently, so the
//! second comparison is a near-tie bound, not an equality.
//!
//! Both 6:1 geometries the dispatcher admits are covered — 24 query heads
//! over 4 KV heads and 12 over 2 — because the kernels index by group, so
//! the KV-head count is grid height and nothing else.
//!
//! LAUNCH GEOMETRY: `cuda::prefill` is `pub(crate)`, so an integration test
//! cannot call `launch_attention_decode_splitk_gqa6`. The geometry below
//! MIRRORS it and must be kept in step with it:
//!
//! | | shipping pair | GQA-shared pair |
//! |---|---|---|
//! | partial grid | `(num_heads * S, 1, 1)` | `(S, num_kv_heads, 1)` |
//! | merge grid | `(num_heads, 1, 1)` | `(num_heads, head_dim / 128, 1)` |
//! | block | 128 | 128 |
//! | partial shared | `(8 + head_dim + 128) * 4` | `attn_splitk_gqa6_partial_shared_bytes()` |
//! | merge shared | 0 | `attn_splitk_gqa6_merge_shared_bytes(S)` |
//! | S | `ceil(seq_len / 128)`, capped at 32 | `ceil(seq_len / C)`, C = 16 |
//!
//! Both are compiled the way the kernel loader compiles them: NVRTC's default
//! target, through `CudaDevice::compile_and_load`.

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use lumen_runtime::cuda::ffi::CudaDevice;
use lumen_runtime::cuda::shaders::{
    ATTENTION_DECODE_SPLITK_GQA6_KERNEL_SOURCE, ATTENTION_DECODE_SPLITK_KERNEL_SOURCE,
};
// The launch geometry comes from the crate, so a retune of any of these
// cannot leave this mirror describing a kernel production no longer runs.
use lumen_runtime::cuda::{
    attn_splitk_chunks, attn_splitk_gqa6_chunks, attn_splitk_gqa6_max_seq_len,
    attn_splitk_gqa6_merge_shared_bytes, attn_splitk_gqa6_partial_shared_bytes,
    ATTN_DECODE_TILED_BLOCK_DIM as BLOCK_DIM, ATTN_DECODE_TILED_T_C as T_C,
    ATTN_SPLITK_GQA6_CHUNK as GQA6_CHUNK, ATTN_SPLITK_GQA6_DIM_TILES as GQA6_DIM_TILES,
    ATTN_SPLITK_GQA6_HEAD_DIM as HEAD_DIM,
};

const MAX_SEQ_LEN: u32 = 4096;
const SCALE: f32 = 0.0625; // 1 / sqrt(256)

/// Roughly five times the largest error the pair produced against F64 over
/// this length set (4.33e-7, bound by the wide-score-spread inputs at 4,096
/// keys), so a real regression trips it and F32 noise does not.
const MAX_ABS_ERR_VS_F64: f64 = 2e-6;
/// The two routes are each within ~4e-7 of F64, so they agree with each other
/// to about 1e-6.
const MAX_ABS_ERR_VS_SHIPPING: f64 = 2e-6;

/// Contexts covering both boundaries of every tile and chunk arithmetic in
/// play: the single-position case, either side of one GQA-shared chunk (16),
/// either side of one shipping tile (128), the board shape, and the cap.
const LENGTHS: &[u32] = &[
    1, 15, 16, 17, 127, 128, 129, 330, 1100, 1300, 2600, 4095, 4096,
];

/// A query-head / KV-head pair the dispatcher admits: any 6:1 group.
#[derive(Clone, Copy)]
struct Geom {
    num_heads: u32,
    num_kv_heads: u32,
}

/// The 27B full-attention shape.
const G24_4: Geom = Geom {
    num_heads: 24,
    num_kv_heads: 4,
};
/// The same group size at a different grid height, which is the only thing
/// the KV-head count changes.
const G12_2: Geom = Geom {
    num_heads: 12,
    num_kv_heads: 2,
};
const GEOMETRIES: &[Geom] = &[G24_4, G12_2];

impl Geom {
    fn cache_floats(self) -> usize {
        (self.num_kv_heads * MAX_SEQ_LEN * HEAD_DIM) as usize
    }
    fn out_floats(self) -> usize {
        (self.num_heads * HEAD_DIM) as usize
    }
    fn label(self) -> String {
        format!("{}q/{}kv", self.num_heads, self.num_kv_heads)
    }
}

fn try_device() -> Option<CudaDevice> {
    match CudaDevice::new(0) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("Skipping: no CUDA GPU available: {e}");
            None
        }
    }
}

fn rng_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 33
}

/// Uniform in [-1, 1) with a full 24-bit mantissa, so products of two such
/// values are not exactly representable and a changed summation order shows.
fn rand_unit(s: &mut u64) -> f32 {
    ((rng_next(s) & 0xff_ffff) as f32 / 8_388_608.0) - 1.0
}

struct Inputs {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
}

fn make_inputs(g: Geom, seed: u64, q_scale: f32) -> Inputs {
    let mut s = seed;
    Inputs {
        q: (0..g.out_floats())
            .map(|_| rand_unit(&mut s) * q_scale)
            .collect(),
        k: (0..g.cache_floats()).map(|_| rand_unit(&mut s)).collect(),
        v: (0..g.cache_floats()).map(|_| rand_unit(&mut s)).collect(),
    }
}

/// Single-pass F64 reference for the whole decode-attention output.
fn reference(g: Geom, inp: &Inputs, seq_len: u32) -> Vec<f64> {
    let hd = HEAD_DIM as usize;
    let seq = seq_len as usize;
    let group = (g.num_heads / g.num_kv_heads) as usize;
    let mut out = vec![0.0f64; g.out_floats()];
    let mut s = vec![0.0f64; seq];
    for h in 0..g.num_heads as usize {
        let base = (h / group) * MAX_SEQ_LEN as usize * hd;
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

/// Largest absolute difference between two results, and where.
///
/// Rejects a length mismatch and any non-finite element of EITHER side before
/// comparing. A running maximum alone cannot do this: `(NaN - r).abs() > worst`
/// is false for every element, so one all-NaN side reports zero error and
/// passes every assertion in this file — including the ones whose scratch is
/// NaN-poisoned precisely to catch an element the kernel never wrote. `zip`
/// hides a short result the same way, by stopping at the shorter side.
///
/// Both sides need the check because both are computed: `want` is the F64
/// host reference in most callers, but it is the shipping kernel's own output
/// in `gqa6_agrees_with_the_shipping_pair_at_every_length`, where a NaN would
/// otherwise make the two routes agree perfectly.
fn max_abs_err(got: &[f32], want: &[f64]) -> Result<(f64, usize), String> {
    if got.len() != want.len() {
        return Err(format!(
            "length mismatch: {} values against {} reference values",
            got.len(),
            want.len()
        ));
    }
    if let Some((i, v)) = got.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(format!("element {i} is {v}, not a finite number"));
    }
    if let Some((i, v)) = want.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(format!("reference element {i} is {v}, not a finite number"));
    }
    let mut worst = 0.0f64;
    let mut at = 0usize;
    for (i, (g, r)) in got.iter().zip(want.iter()).enumerate() {
        let d = (f64::from(*g) - r).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    Ok((worst, at))
}

/// Device-side buffers, uploaded once and shared by every case.
struct Gpu {
    q: CudaSlice<f32>,
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
}

fn upload(dev: &CudaDevice, inp: &Inputs) -> Gpu {
    Gpu {
        q: dev.htod_copy(&inp.q).expect("Q upload"),
        k: dev.htod_copy(&inp.k).expect("K upload"),
        v: dev.htod_copy(&inp.v).expect("V upload"),
    }
}

/// Run the GQA-shared pair at `chunks` splits and return the merged output.
///
/// The scratch is NaN-filled first: an element the partial pass fails to
/// write reaches the merge as NaN, and `max_abs_err` rejects a non-finite
/// result rather than scoring it zero.
fn run_gqa6(dev: &CudaDevice, g: Geom, gpu: &Gpu, seq_len: u32, chunks: u32) -> Vec<f32> {
    let module = dev
        .compile_and_load(ATTENTION_DECODE_SPLITK_GQA6_KERNEL_SOURCE)
        .expect("compile GQA-shared pair");
    let partial = module
        .load_function("attention_decode_splitk_partial_gqa6_f32")
        .expect("partial");
    let merge = module
        .load_function("attention_decode_splitk_merge_gqa6_f32")
        .expect("merge");

    let n_part = (g.num_heads * chunks) as usize;
    let mut m_part = dev.htod_copy(&vec![f32::NAN; n_part]).unwrap();
    let mut l_part = dev.htod_copy(&vec![f32::NAN; n_part]).unwrap();
    let mut o_part = dev
        .htod_copy(&vec![f32::NAN; n_part * HEAD_DIM as usize])
        .unwrap();
    let mut out = dev.htod_copy(&vec![f32::NAN; g.out_floats()]).unwrap();

    let chunk = GQA6_CHUNK;
    let max_seq_len = MAX_SEQ_LEN;
    let scale = SCALE;

    unsafe {
        dev.stream
            .launch_builder(&partial)
            .arg(&gpu.q)
            .arg(&gpu.k)
            .arg(&gpu.v)
            .arg(&mut m_part)
            .arg(&mut l_part)
            .arg(&mut o_part)
            .arg(&seq_len)
            .arg(&max_seq_len)
            .arg(&scale)
            .arg(&chunks)
            .arg(&chunk)
            .launch(LaunchConfig {
                grid_dim: (chunks, g.num_kv_heads, 1),
                block_dim: (BLOCK_DIM, 1, 1),
                shared_mem_bytes: attn_splitk_gqa6_partial_shared_bytes(),
            })
            .expect("gqa6 partial launch");
        dev.stream
            .launch_builder(&merge)
            .arg(&m_part)
            .arg(&l_part)
            .arg(&o_part)
            .arg(&mut out)
            .arg(&chunks)
            .launch(LaunchConfig {
                grid_dim: (g.num_heads, GQA6_DIM_TILES, 1),
                block_dim: (BLOCK_DIM, 1, 1),
                shared_mem_bytes: attn_splitk_gqa6_merge_shared_bytes(chunks),
            })
            .expect("gqa6 merge launch");
    }
    dev.synchronize().expect("sync");
    dev.dtoh_copy(&out).expect("out")
}

/// Run the shipping pair at the split count production would pick.
fn run_shipping(dev: &CudaDevice, g: Geom, gpu: &Gpu, seq_len: u32) -> Vec<f32> {
    let module = dev
        .compile_and_load(ATTENTION_DECODE_SPLITK_KERNEL_SOURCE)
        .expect("compile shipping pair");
    let partial = module
        .load_function("attention_decode_splitk_partial")
        .expect("partial");
    let merge = module
        .load_function("attention_decode_splitk_merge")
        .expect("merge");

    let chunks = attn_splitk_chunks(seq_len);
    let n_part = (g.num_heads * chunks) as usize;
    let mut m_part = dev.htod_copy(&vec![f32::NAN; n_part]).unwrap();
    let mut l_part = dev.htod_copy(&vec![f32::NAN; n_part]).unwrap();
    let mut o_part = dev
        .htod_copy(&vec![f32::NAN; n_part * HEAD_DIM as usize])
        .unwrap();
    let mut out = dev.htod_copy(&vec![f32::NAN; g.out_floats()]).unwrap();

    let (num_heads, num_kv_heads, head_dim) = (g.num_heads, g.num_kv_heads, HEAD_DIM);
    let max_seq_len = MAX_SEQ_LEN;
    let scale = SCALE;
    unsafe {
        dev.stream
            .launch_builder(&partial)
            .arg(&gpu.q)
            .arg(&gpu.k)
            .arg(&gpu.v)
            .arg(&mut m_part)
            .arg(&mut l_part)
            .arg(&mut o_part)
            .arg(&num_heads)
            .arg(&num_kv_heads)
            .arg(&head_dim)
            .arg(&seq_len)
            .arg(&max_seq_len)
            .arg(&scale)
            .arg(&chunks)
            .launch(LaunchConfig {
                grid_dim: (num_heads * chunks, 1, 1),
                block_dim: (BLOCK_DIM, 1, 1),
                shared_mem_bytes: (8 + head_dim + T_C) * 4,
            })
            .expect("shipping partial launch");
        dev.stream
            .launch_builder(&merge)
            .arg(&m_part)
            .arg(&l_part)
            .arg(&o_part)
            .arg(&mut out)
            .arg(&num_heads)
            .arg(&head_dim)
            .arg(&chunks)
            .launch(LaunchConfig {
                grid_dim: (num_heads, 1, 1),
                block_dim: (BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            })
            .expect("shipping merge launch");
    }
    dev.synchronize().expect("sync");
    dev.dtoh_copy(&out).expect("out")
}

/// The sweep has to reach the boundary the dispatcher stops at, or a retune
/// of the chunk length would quietly leave the longest eligible context
/// untested. Runs with or without a GPU.
#[test]
fn the_sweep_reaches_the_longest_eligible_context() {
    let cap = attn_splitk_gqa6_max_seq_len();
    assert_eq!(
        LENGTHS.iter().copied().max(),
        Some(cap),
        "the length sweep stops short of the {cap}-position bound"
    );
    assert!(
        LENGTHS.contains(&(cap - 1)),
        "the sweep skips the length just inside the bound"
    );
}

/// The guard that makes every other assertion in this file mean something.
/// Runs with or without a GPU.
#[test]
fn the_error_helper_rejects_a_poisoned_result() {
    let want = vec![0.5f64; 8];
    let mut one_nan = vec![0.5f32; 8];
    one_nan[3] = f32::NAN;

    for (what, got) in [
        ("an all-NaN result", vec![f32::NAN; 8]),
        ("a single NaN element", one_nan),
        ("an infinite element", vec![f32::INFINITY; 8]),
        ("a truncated result", vec![0.5f32; 7]),
        ("an empty result", Vec::new()),
    ] {
        assert!(max_abs_err(&got, &want).is_err(), "{what} scored as a pass");
    }

    // The same poisoning on the reference side, which is another kernel's
    // output whenever the two routes are compared against each other.
    let mut want_one_nan = vec![0.5f64; 8];
    want_one_nan[5] = f64::NAN;
    for (what, bad_want) in [
        ("an all-NaN reference", vec![f64::NAN; 8]),
        ("a single NaN reference element", want_one_nan),
        ("an infinite reference element", vec![f64::INFINITY; 8]),
    ] {
        assert!(
            max_abs_err(&[0.5f32; 8], &bad_want).is_err(),
            "{what} scored as a pass"
        );
    }

    let (err, at) = max_abs_err(&[0.5f32; 8], &want).expect("a finite exact result scores");
    assert_eq!((err, at), (0.0, 0));
    let (err, at) = max_abs_err(&[0.5, 0.5, 0.25, 0.5, 0.5, 0.5, 0.5, 0.5], &want)
        .expect("a finite result scores");
    assert_eq!((err, at), (0.25, 2));
}

#[test]
fn gqa6_matches_the_f64_reference_at_every_length() {
    let Some(dev) = try_device() else { return };
    for &g in GEOMETRIES {
        let inp = make_inputs(g, 0x5150, 1.0);
        let gpu = upload(&dev, &inp);
        for &seq_len in LENGTHS {
            let want = reference(g, &inp, seq_len);
            let got = run_gqa6(&dev, g, &gpu, seq_len, attn_splitk_gqa6_chunks(seq_len));
            let (err, at) = max_abs_err(&got, &want)
                .unwrap_or_else(|e| panic!("{} seq_len {seq_len}: {e}", g.label()));
            assert!(
                err <= MAX_ABS_ERR_VS_F64,
                "{} seq_len {seq_len}: max abs error {err:.3e} at element {at} \
                 (got {}, want {})",
                g.label(),
                got[at],
                want[at]
            );
        }
    }
}

/// The same sweep with the query scaled by 8, which spreads the pre-softmax
/// scores eightfold: chunk-local maxima diverge and the merge's cross-chunk
/// rescale carries the result instead of being a near-no-op.
#[test]
fn gqa6_matches_the_f64_reference_under_a_wide_score_spread() {
    let Some(dev) = try_device() else { return };
    for &g in GEOMETRIES {
        let inp = make_inputs(g, 0x5151, 8.0);
        let gpu = upload(&dev, &inp);
        for &seq_len in LENGTHS {
            let want = reference(g, &inp, seq_len);
            let got = run_gqa6(&dev, g, &gpu, seq_len, attn_splitk_gqa6_chunks(seq_len));
            let (err, at) = max_abs_err(&got, &want)
                .unwrap_or_else(|e| panic!("{} seq_len {seq_len}: {e}", g.label()));
            assert!(
                err <= MAX_ABS_ERR_VS_F64,
                "{} seq_len {seq_len}: max abs error {err:.3e} at element {at}",
                g.label()
            );
        }
    }
}

/// A zero query makes every score zero, so the softmax is uniform and the
/// exact answer is the mean of the V rows in range — a case whose reference
/// does not depend on the exponential at all.
#[test]
fn gqa6_zero_query_averages_the_values() {
    let Some(dev) = try_device() else { return };
    let g = G24_4;
    let mut inp = make_inputs(g, 0x5152, 1.0);
    inp.q.iter_mut().for_each(|x| *x = 0.0);
    let gpu = upload(&dev, &inp);
    for &seq_len in LENGTHS {
        let want = reference(g, &inp, seq_len);
        let got = run_gqa6(&dev, g, &gpu, seq_len, attn_splitk_gqa6_chunks(seq_len));
        let (err, at) =
            max_abs_err(&got, &want).unwrap_or_else(|e| panic!("seq_len {seq_len}: {e}"));
        assert!(
            err <= MAX_ABS_ERR_VS_F64,
            "seq_len {seq_len}: max abs error {err:.3e} at element {at}"
        );
    }
}

#[test]
fn gqa6_agrees_with_the_shipping_pair_at_every_length() {
    let Some(dev) = try_device() else { return };
    for &g in GEOMETRIES {
        let inp = make_inputs(g, 0x5153, 1.0);
        let gpu = upload(&dev, &inp);
        for &seq_len in LENGTHS {
            let mine = run_gqa6(&dev, g, &gpu, seq_len, attn_splitk_gqa6_chunks(seq_len));
            let theirs = run_shipping(&dev, g, &gpu, seq_len);
            let as_f64: Vec<f64> = theirs.iter().map(|x| f64::from(*x)).collect();
            let (err, at) = max_abs_err(&mine, &as_f64)
                .unwrap_or_else(|e| panic!("{} seq_len {seq_len}: {e}", g.label()));
            assert!(
                err <= MAX_ABS_ERR_VS_SHIPPING,
                "{} seq_len {seq_len}: the two routes differ by {err:.3e} at element {at} \
                 (gqa6 {}, shipping {})",
                g.label(),
                mine[at],
                theirs[at]
            );
        }
    }
}

/// The headline accuracy claim: over the sweep, the GQA-shared pair's worst
/// departure from the F64 reference is no larger than the pair it replaces.
/// Compared as maxima over the whole sweep, which is the shape of the claim —
/// a per-length comparison would be a stricter statement than anything
/// measured, and F32 noise would decide it at some lengths.
#[test]
fn gqa6_is_no_further_from_the_reference_than_the_shipping_pair() {
    let Some(dev) = try_device() else { return };
    let mut worst_gqa6 = 0.0f64;
    let mut worst_shipping = 0.0f64;
    // Both distributions, because the wide-spread one is what binds.
    for (seed, q_scale) in [(0x5155u64, 1.0f32), (0x5156, 8.0)] {
        for &g in GEOMETRIES {
            let inp = make_inputs(g, seed, q_scale);
            let gpu = upload(&dev, &inp);
            for &seq_len in LENGTHS {
                let want = reference(g, &inp, seq_len);
                let mine = run_gqa6(&dev, g, &gpu, seq_len, attn_splitk_gqa6_chunks(seq_len));
                let theirs = run_shipping(&dev, g, &gpu, seq_len);
                let (a, _) = max_abs_err(&mine, &want)
                    .unwrap_or_else(|e| panic!("gqa6 {} seq_len {seq_len}: {e}", g.label()));
                let (b, _) = max_abs_err(&theirs, &want)
                    .unwrap_or_else(|e| panic!("shipping {} seq_len {seq_len}: {e}", g.label()));
                worst_gqa6 = worst_gqa6.max(a);
                worst_shipping = worst_shipping.max(b);
            }
        }
    }
    eprintln!("worst abs error vs F64: gqa6 {worst_gqa6:.3e}, shipping {worst_shipping:.3e}");
    assert!(
        worst_gqa6 <= worst_shipping,
        "the GQA-shared pair is further from the reference than the pair it \
         replaces: {worst_gqa6:.3e} against {worst_shipping:.3e}"
    );
}

/// Deterministic for a fixed split count: the same input twice must give the
/// same bits, not merely the same value to a tolerance.
#[test]
fn gqa6_is_bit_reproducible() {
    let Some(dev) = try_device() else { return };
    let g = G24_4;
    let inp = make_inputs(g, 0x5157, 1.0);
    let gpu = upload(&dev, &inp);
    for &seq_len in LENGTHS {
        let chunks = attn_splitk_gqa6_chunks(seq_len);
        let first = run_gqa6(&dev, g, &gpu, seq_len, chunks);
        let second = run_gqa6(&dev, g, &gpu, seq_len, chunks);
        let differing = first
            .iter()
            .zip(second.iter())
            .position(|(a, b)| a.to_bits() != b.to_bits());
        assert!(
            differing.is_none(),
            "seq_len {seq_len}: two runs of the same input differ at element {}",
            differing.unwrap()
        );
    }
}

/// Chunks past the end of the context are exercised directly: with a split
/// count above the natural one, the trailing chunks have no positions at all
/// and must contribute nothing to the merge. The dispatcher never asks for a
/// count this high, but the kernel's empty-chunk arm is what keeps a count
/// that overshoots the context exact rather than merely lucky.
#[test]
fn gqa6_empty_chunks_contribute_nothing() {
    let Some(dev) = try_device() else { return };
    let g = G24_4;
    let inp = make_inputs(g, 0x5154, 1.0);
    let gpu = upload(&dev, &inp);
    for &seq_len in &[1u32, 15, 16, 17, 127, 330] {
        // The kernels divide the context evenly into the count, so a count
        // above the natural one only empties a chunk when the rounding leaves
        // the last one starting past the end. Take the smallest such count.
        let natural = attn_splitk_gqa6_chunks(seq_len);
        let padded = ((natural + 1)..=(natural + 64))
            .find(|&c| seq_len.div_ceil(c) * (c - 1) >= seq_len)
            .unwrap_or_else(|| panic!("seq_len {seq_len}: no split count empties a chunk"));
        let span = seq_len.div_ceil(padded);
        assert!(
            span <= GQA6_CHUNK,
            "seq_len {seq_len}: span {span} outgrows the chunk length"
        );
        let want = reference(g, &inp, seq_len);
        let got = run_gqa6(&dev, g, &gpu, seq_len, padded);
        let (err, at) = max_abs_err(&got, &want)
            .unwrap_or_else(|e| panic!("seq_len {seq_len} with {padded} chunks: {e}"));
        assert!(
            err <= MAX_ABS_ERR_VS_F64,
            "seq_len {seq_len} with {padded} chunks: max abs error {err:.3e} at element {at} \
             (got {})",
            got[at]
        );
    }
}
