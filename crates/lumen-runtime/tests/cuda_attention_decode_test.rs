//! Correctness suite for the decode-attention kernel pair
//! (`attention_decode_partial_f32` + `attention_decode_merge`) at the
//! (6, 256) shape, over every context from one key to the cache's end.
//!
//! Requires the `cuda` feature; the GPU cases are skipped where there is no
//! device, and `the_error_helper_rejects_a_poisoned_result` runs anywhere.
//! Run:
//!     cargo test --release -p lumen-runtime --features cuda \
//!         --test cuda_attention_decode_test
//!
//! Every case is scored against an F64 host reference computed in one pass
//! (the ground truth); the whole-tile partition is also scored against the
//! one-tile partition on the same inputs, a near-tie bound rather than an
//! equality because the two reassociate the sum differently.
//!
//! Both 6:1 grids are covered — 24 query heads over 4 KV heads and 12 over
//! 2 — because the kernel indexes by group, so the KV-head count is grid
//! height and nothing else. The other (group, head_dim) shapes in the
//! kernel's domain are covered by `cuda_attention_decode_shapes_test.rs`.
//!
//! LAUNCH GEOMETRY: the launcher is `pub(crate)`, so an integration test
//! cannot call it. The geometry below comes from the crate's own helpers
//! and mirrors the launcher:
//!
//! | partial grid | `(S, num_kv_heads, 1)` |
//! |---|---|
//! | merge grid | `(num_heads, head_dim / 128, 1)` |
//! | block | 128 |
//! | partial shared | `SPEC.partial_shared_bytes(false)` |
//! | merge shared | `decode_attention_merge_shared_bytes(S)` |
//! | S | `decode_attention_geometry_within(seq_len, one_tile, target).0` |
//!
//! Compiled the way the kernel loader compiles it: NVRTC's default target,
//! through `CudaDevice::compile_and_load`.

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use lumen_runtime::cuda::ffi::CudaDevice;
// The launch geometry comes from the crate, so a retune of any of these
// cannot leave this mirror describing a kernel production no longer runs.
use lumen_runtime::cuda::{
    decode_attention_geometry_within, decode_attention_merge_shared_bytes,
    ATTN_DECODE_BLOCK_DIM as BLOCK_DIM, ATTN_DECODE_REVIEWED_SHAPE as SPEC, ATTN_DECODE_S_MAX,
    ATTN_DECODE_TILE as DECODE_CHUNK,
};
const HEAD_DIM: u32 = SPEC.head_dim;
const DECODE_DIM_TILES: u32 = SPEC.dim_tiles();
use lumen_runtime::runtime_defaults::ATTN_ONE_TILE_DEFAULT;

const MAX_SEQ_LEN: u32 = 32_768;
const SCALE: f32 = 0.0625; // 1 / sqrt(256)

/// Absolute tolerance against the float64 reference. Every run prints the
/// observed maximum per length (`f64-reference …` lines) so the headroom is a
/// recorded number, read off every run's output rather than assumed. Never
/// widen this to absorb a storage change; give a
/// changed input format its own reference instead.
const MAX_ABS_ERR_VS_F64: f64 = 2e-6;
/// The whole-tile and one-tile partitions are each within ~4e-7 of F64, so
/// they agree with each other to about 1e-6.
const MAX_ABS_ERR_BETWEEN_PARTITIONS: f64 = 2e-6;

/// Contexts covering both boundaries of every tile and chunk arithmetic in
/// play: the single-position case, either side of one 16-key tile, either
/// side of 128 keys, the board shapes, either side of the one-tile
/// partition's compile-time reach (16,384 keys, the merge's split ceiling),
/// and the contexts only the whole-tile partition can serve, to the cache's
/// end.
const LENGTHS: &[u32] = &[
    1, 15, 16, 17, 127, 128, 129, 330, 1100, 1300, 2600, 4095, 4096, 4097, 6144, 8192, 12288,
    16383, 16384, 16385, 24576, 32767, 32768,
];

/// A query-head / KV-head pair at the suite's group size of 6.
#[derive(Clone, Copy)]
struct Geom {
    num_heads: u32,
    num_kv_heads: u32,
}

/// The one-tile split count: one CTA per 16 keys, the partition the policy
/// keeps below its one-tile bound.
fn one_tile_chunks(seq_len: u32) -> u32 {
    decode_attention_geometry_within(seq_len, u32::MAX, 1).0
}

/// The geometry the sweep runs at a length: the one-tile partition wherever
/// it can serve (its split count within the merge's compile-time ceiling),
/// the whole-tile partition at the default target beyond that — the only
/// form that reaches 16,385 keys and more.
fn sweep_geometry(seq_len: u32) -> (u32, u32) {
    decode_attention_geometry_within(seq_len, ATTN_DECODE_S_MAX, 128)
}

/// The geometry the 4-KV-head default policy launches at a length, which
/// differs from the sweep's between the one-tile default and the one-tile
/// partition's ceiling. The distribution sweeps run both wherever they differ.
fn production_geometry(seq_len: u32) -> (u32, u32) {
    decode_attention_geometry_within(seq_len, ATTN_ONE_TILE_DEFAULT, 128)
}

fn geometries_at(seq_len: u32) -> Vec<(u32, u32)> {
    let mut g = vec![sweep_geometry(seq_len)];
    if production_geometry(seq_len) != g[0] {
        g.push(production_geometry(seq_len));
    }
    g
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
/// host reference in most callers, but it is the one-tile partition's own output
/// in `decode_whole_tile_partition_matches_the_f64_reference`, where a NaN would
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
fn run(dev: &CudaDevice, g: Geom, gpu: &Gpu, seq_len: u32, geometry: (u32, u32)) -> Vec<f32> {
    run_at(dev, g, gpu, seq_len, geometry.0, geometry.1)
}

/// The pair at an explicit `(chunks, partition)`: `0` the one-tile form,
/// `1` the whole-tile balanced partition at `chunks` CTAs.
fn run_at(
    dev: &CudaDevice,
    g: Geom,
    gpu: &Gpu,
    seq_len: u32,
    chunks: u32,
    partition: u32,
) -> Vec<f32> {
    let module = dev
        .compile_and_load(&SPEC.source())
        .expect("compile GQA-shared pair");
    let partial = module
        .load_function("attention_decode_partial_f32")
        .expect("partial");
    let merge = module
        .load_function("attention_decode_merge")
        .expect("merge");

    let n_part = (g.num_heads * chunks) as usize;
    let mut m_part = dev.htod_copy(&vec![f32::NAN; n_part]).unwrap();
    let mut l_part = dev.htod_copy(&vec![f32::NAN; n_part]).unwrap();
    let mut o_part = dev
        .htod_copy(&vec![f32::NAN; n_part * HEAD_DIM as usize])
        .unwrap();
    let mut out = dev.htod_copy(&vec![f32::NAN; g.out_floats()]).unwrap();

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
            .arg(&partition)
            .launch(LaunchConfig {
                grid_dim: (chunks, g.num_kv_heads, 1),
                block_dim: (BLOCK_DIM, 1, 1),
                shared_mem_bytes: SPEC.partial_shared_bytes(false),
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
                grid_dim: (g.num_heads, DECODE_DIM_TILES, 1),
                block_dim: (BLOCK_DIM, 1, 1),
                shared_mem_bytes: decode_attention_merge_shared_bytes(chunks),
            })
            .expect("merge launch");
    }
    dev.synchronize().expect("sync");
    dev.dtoh_copy(&out).expect("out")
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
fn decode_matches_the_f64_reference_at_every_length() {
    let Some(dev) = try_device() else { return };
    for &g in GEOMETRIES {
        let inp = make_inputs(g, 0x5150, 1.0);
        let gpu = upload(&dev, &inp);
        for &seq_len in LENGTHS {
            let want = reference(g, &inp, seq_len);
            let got = run(&dev, g, &gpu, seq_len, sweep_geometry(seq_len));
            let (err, at) = max_abs_err(&got, &want)
                .unwrap_or_else(|e| panic!("{} seq_len {seq_len}: {e}", g.label()));
            // The observed maximum is part of the record: the tolerance's
            // headroom is read off these lines, never assumed.
            eprintln!(
                "f64-reference {} seq_len {seq_len}: max abs error {err:.3e}",
                g.label()
            );
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
fn decode_matches_the_f64_reference_under_a_wide_score_spread() {
    let Some(dev) = try_device() else { return };
    for &g in GEOMETRIES {
        let inp = make_inputs(g, 0x5151, 8.0);
        let gpu = upload(&dev, &inp);
        for &seq_len in LENGTHS {
            let want = reference(g, &inp, seq_len);
            for geometry in geometries_at(seq_len) {
                let got = run(&dev, g, &gpu, seq_len, geometry);
                let (err, at) = max_abs_err(&got, &want).unwrap_or_else(|e| {
                    panic!("{} seq_len {seq_len} {geometry:?}: {e}", g.label())
                });
                assert!(
                    err <= MAX_ABS_ERR_VS_F64,
                    "{} seq_len {seq_len} {geometry:?}: max abs error {err:.3e} at element {at}",
                    g.label()
                );
            }
        }
    }
}

/// A zero query makes every score zero, so the softmax is uniform and the
/// exact answer is the mean of the V rows in range — a case whose reference
/// does not depend on the exponential at all.
#[test]
fn decode_zero_query_averages_the_values() {
    let Some(dev) = try_device() else { return };
    let g = G24_4;
    let mut inp = make_inputs(g, 0x5152, 1.0);
    inp.q.iter_mut().for_each(|x| *x = 0.0);
    let gpu = upload(&dev, &inp);
    for &seq_len in LENGTHS {
        let want = reference(g, &inp, seq_len);
        for geometry in geometries_at(seq_len) {
            let got = run(&dev, g, &gpu, seq_len, geometry);
            let (err, at) = max_abs_err(&got, &want)
                .unwrap_or_else(|e| panic!("seq_len {seq_len} {geometry:?}: {e}"));
            assert!(
                err <= MAX_ABS_ERR_VS_F64,
                "seq_len {seq_len} {geometry:?}: max abs error {err:.3e} at element {at}"
            );
        }
    }
}

/// Deterministic for a fixed split count: the same input twice must give the
/// same bits, not merely the same value to a tolerance.
#[test]
fn decode_is_bit_reproducible() {
    let Some(dev) = try_device() else { return };
    let g = G24_4;
    let inp = make_inputs(g, 0x5157, 1.0);
    let gpu = upload(&dev, &inp);
    for &seq_len in LENGTHS {
        let geometry = sweep_geometry(seq_len);
        let first = run(&dev, g, &gpu, seq_len, geometry);
        let second = run(&dev, g, &gpu, seq_len, geometry);
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
/// and must contribute nothing to the merge. The launcher never asks for a
/// count this high, but the kernel's empty-chunk arm is what keeps a count
/// that overshoots the context exact rather than merely lucky.
#[test]
fn decode_empty_chunks_contribute_nothing() {
    let Some(dev) = try_device() else { return };
    let g = G24_4;
    let inp = make_inputs(g, 0x5154, 1.0);
    let gpu = upload(&dev, &inp);
    for &seq_len in &[1u32, 15, 16, 17, 127, 330] {
        // The kernels divide the context evenly into the count, so a count
        // above the natural one only empties a chunk when the rounding leaves
        // the last one starting past the end. Take the smallest such count.
        let natural = one_tile_chunks(seq_len);
        let padded = ((natural + 1)..=(natural + 64))
            .find(|&c| seq_len.div_ceil(c) * (c - 1) >= seq_len)
            .unwrap_or_else(|| panic!("seq_len {seq_len}: no split count empties a chunk"));
        let span = seq_len.div_ceil(padded);
        assert!(
            span <= DECODE_CHUNK,
            "seq_len {seq_len}: span {span} outgrows the chunk length"
        );
        let want = reference(g, &inp, seq_len);
        let got = run(&dev, g, &gpu, seq_len, (padded, 0));
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

/// The whole-tile partition at the shipped target: every context from the
/// first multi-tile CTA to the cache's end, against the F64 reference and
/// against the one-tile form, with the balanced partition's tile-boundary
/// contexts (16 * 176 +- 1, 16 * 128 * k +- 1) covered.
#[test]
fn decode_whole_tile_partition_matches_the_f64_reference() {
    let Some(dev) = try_device() else { return };
    for &g in GEOMETRIES {
        let inp = make_inputs(g, 0x5158, 1.0);
        let gpu = upload(&dev, &inp);
        for &seq_len in &[
            2817u32, 2831, 2832, 2833, 3968, 4096, 4097, 6144, 8192, 12288, 16383, 16384, 16385,
            24576, 32767, 32768,
        ] {
            let want = reference(g, &inp, seq_len);
            let got = run_at(&dev, g, &gpu, seq_len, 128, 1);
            let (err, at) = max_abs_err(&got, &want)
                .unwrap_or_else(|e| panic!("{} whole-tile seq_len {seq_len}: {e}", g.label()));
            eprintln!(
                "f64-reference whole-tile {} seq_len {seq_len}: max abs error {err:.3e}",
                g.label()
            );
            assert!(
                err <= MAX_ABS_ERR_VS_F64,
                "{} whole-tile seq_len {seq_len}: max abs error {err:.3e} at element {at}",
                g.label()
            );
            // The one-tile form is a comparator only where it can serve.
            if one_tile_chunks(seq_len) > ATTN_DECODE_S_MAX {
                continue;
            }
            let one_tile = run(&dev, g, &gpu, seq_len, (one_tile_chunks(seq_len), 0));
            let (d, _) = max_abs_err(
                &got,
                &one_tile.iter().map(|&x| f64::from(x)).collect::<Vec<_>>(),
            )
            .unwrap_or_else(|e| panic!("{} seq_len {seq_len}: {e}", g.label()));
            assert!(
                d <= MAX_ABS_ERR_BETWEEN_PARTITIONS,
                "{} seq_len {seq_len}: whole-tile vs one-tile differ by {d:.3e}",
                g.label()
            );
        }
    }
}

/// The whole-tile partition is deterministic run to run, like the one-tile form.
#[test]
fn decode_whole_tile_partition_is_bit_reproducible() {
    let Some(dev) = try_device() else { return };
    let g = G24_4;
    let inp = make_inputs(g, 0x5159, 1.0);
    let gpu = upload(&dev, &inp);
    for &seq_len in &[2817u32, 6144, 16384, 32768] {
        let a = run_at(&dev, g, &gpu, seq_len, 128, 1);
        let b = run_at(&dev, g, &gpu, seq_len, 128, 1);
        assert!(
            a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
            "seq_len {seq_len}"
        );
    }
}
