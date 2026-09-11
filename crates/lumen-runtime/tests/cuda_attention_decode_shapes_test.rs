//! The decode-attention kernel over its whole shape domain: every group size
//! 1..=8 at head_dim 128 and 256, on both partitions and both stores, against
//! a float64 reference. The shipped models are (4, 256), (6, 256) and
//! (8, 256); the other groups exercise the partly filled softmax rounds
//! (four warps over G heads) and head_dim 128 the one-float4 lane.
//!
//! Requires the `cuda` feature; skipped where there is no device.

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use lumen_runtime::cuda::ffi::CudaDevice;
use lumen_runtime::cuda::{
    decode_attention_geometry_within, decode_attention_merge_shared_bytes, DecodeAttentionSpec,
    ATTN_DECODE_BLOCK_DIM as BLOCK_DIM,
};

const NUM_KV_HEADS: u32 = 2;
const MAX_SEQ_LEN: u32 = 16_384;
const ONE_TILE: u32 = 176;
const TARGET: u32 = 128;
/// Absolute tolerance against float64, the same as the production-shape suite's.
const MAX_ABS_ERR: f64 = 2e-6;
/// One-tile boundaries, the whole-tile partition just above the one-tile
/// bound and at the cache's end, and the merge's eight-lane tail.
const LENGTHS: &[u32] = &[1, 16, 17, 129, 2816, 2817, 2833, 4097, 16_384];

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

/// Half-representable inputs (so both stores see the same values), Q on a
/// wider scale so the running-max recurrence and the merge's rescale carry
/// real work.
struct Inputs {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    k16: Vec<u16>,
    v16: Vec<u16>,
}

fn make_inputs(spec: DecodeAttentionSpec, seed: u64) -> Inputs {
    let hd = spec.head_dim as usize;
    let num_heads = (NUM_KV_HEADS * spec.group) as usize;
    let cache = NUM_KV_HEADS as usize * MAX_SEQ_LEN as usize * hd;
    let mut s = seed;
    let mut unit = || ((rng_next(&mut s) & 0xff_ffff) as f32 / 8_388_608.0) - 1.0;
    let q: Vec<f32> = (0..num_heads * hd)
        .map(|_| f16_bits_to_f32(f32_to_f16_bits(unit())) * 4.0)
        .collect();
    let k16: Vec<u16> = (0..cache).map(|_| f32_to_f16_bits(unit())).collect();
    let v16: Vec<u16> = (0..cache).map(|_| f32_to_f16_bits(unit())).collect();
    Inputs {
        q,
        k: k16.iter().map(|&h| f16_bits_to_f32(h)).collect(),
        v: v16.iter().map(|&h| f16_bits_to_f32(h)).collect(),
        k16,
        v16,
    }
}

fn reference(spec: DecodeAttentionSpec, inp: &Inputs, seq_len: u32, scale: f32) -> Vec<f64> {
    let hd = spec.head_dim as usize;
    let group = spec.group as usize;
    let num_heads = NUM_KV_HEADS as usize * group;
    let seq = seq_len as usize;
    let mut out = vec![0.0f64; num_heads * hd];
    let mut s = vec![0.0f64; seq];
    for h in 0..num_heads {
        let base = (h / group) * MAX_SEQ_LEN as usize * hd;
        let mut m = f64::NEG_INFINITY;
        for (p, sp) in s.iter_mut().enumerate() {
            let mut dot = 0.0f64;
            for i in 0..hd {
                dot += f64::from(inp.q[h * hd + i]) * f64::from(inp.k[base + p * hd + i]);
            }
            *sp = dot * f64::from(scale);
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

#[allow(clippy::too_many_arguments)]
fn run<K: cudarc::driver::DeviceRepr>(
    dev: &CudaDevice,
    spec: DecodeAttentionSpec,
    partial_name: &str,
    half: bool,
    q: &CudaSlice<f32>,
    k: &CudaSlice<K>,
    v: &CudaSlice<K>,
    seq_len: u32,
    chunks: u32,
    partition: u32,
    scale: f32,
) -> Vec<f32> {
    let module = dev.compile_and_load(&spec.source()).expect("compile");
    let partial = module.load_function(partial_name).expect("partial");
    let merge = module
        .load_function("attention_decode_merge")
        .expect("merge");
    let num_heads = NUM_KV_HEADS * spec.group;
    let n_part = (num_heads * chunks) as usize;
    let nan = |n: usize| dev.htod_copy(&vec![f32::NAN; n]).expect("alloc");
    let mut m_part = nan(n_part);
    let mut l_part = nan(n_part);
    let mut o_part = nan(n_part * spec.head_dim as usize);
    let mut out = nan((num_heads * spec.head_dim) as usize);
    let max_seq_len = MAX_SEQ_LEN;
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
                shared_mem_bytes: spec.partial_shared_bytes(half),
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
                grid_dim: (num_heads, spec.dim_tiles(), 1),
                block_dim: (BLOCK_DIM, 1, 1),
                shared_mem_bytes: decode_attention_merge_shared_bytes(chunks),
            })
            .expect("merge launch");
    }
    dev.synchronize().expect("sync");
    dev.dtoh_copy(&out).unwrap()
}

fn max_abs_err(got: &[f32], want: &[f64]) -> (f64, usize) {
    assert_eq!(got.len(), want.len());
    let mut worst = (0.0f64, 0usize);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(g.is_finite(), "element {i} is {g}");
        let d = (f64::from(*g) - w).abs();
        if d > worst.0 {
            worst = (d, i);
        }
    }
    worst
}

#[test]
fn every_admitted_shape_matches_the_f64_reference_on_both_stores() {
    let Some(dev) = try_device() else { return };
    let mut seed = 0x0005_0911_5AA9_0000u64;
    for hd in DecodeAttentionSpec::HEAD_DIMS {
        for group in DecodeAttentionSpec::GROUPS {
            let spec = DecodeAttentionSpec::for_shape(NUM_KV_HEADS * group, NUM_KV_HEADS, hd)
                .expect("in the domain");
            seed += 1;
            let inp = make_inputs(spec, seed);
            let scale = 1.0 / (hd as f32).sqrt();
            let q = dev.htod_copy(&inp.q).unwrap();
            let k32 = dev.htod_copy(&inp.k).unwrap();
            let v32 = dev.htod_copy(&inp.v).unwrap();
            let k16 = dev.htod_copy(&inp.k16).unwrap();
            let v16 = dev.htod_copy(&inp.v16).unwrap();
            let mut worst = 0.0f64;
            for &seq_len in LENGTHS {
                let want = reference(spec, &inp, seq_len, scale);
                // Both partitions at every length: the policy's, and the
                // other one at the shipped target.
                let policy = decode_attention_geometry_within(seq_len, ONE_TILE, TARGET);
                let other = if policy.1 == 0 {
                    (TARGET.min(seq_len.div_ceil(16)), 1)
                } else {
                    (seq_len.div_ceil(16), 0)
                };
                for (chunks, partition) in [policy, other] {
                    let a = run(
                        &dev,
                        spec,
                        "attention_decode_partial_f32",
                        false,
                        &q,
                        &k32,
                        &v32,
                        seq_len,
                        chunks,
                        partition,
                        scale,
                    );
                    let b = run(
                        &dev,
                        spec,
                        "attention_decode_partial_f16",
                        true,
                        &q,
                        &k16,
                        &v16,
                        seq_len,
                        chunks,
                        partition,
                        scale,
                    );
                    let (err, at) = max_abs_err(&a, &want);
                    worst = worst.max(err);
                    assert!(
                        err <= MAX_ABS_ERR,
                        "(G={group}, HD={hd}) seq_len {seq_len} chunks {chunks} partition {partition}: F32 store error {err:.3e} at {at}"
                    );
                    assert!(
                        a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
                        "(G={group}, HD={hd}) seq_len {seq_len} chunks {chunks} partition {partition}: the half store differs from the F32 store"
                    );
                }
            }
            eprintln!("f64-reference (G={group}, HD={hd}): worst abs error {worst:.3e} over {} lengths, both partitions, both stores", LENGTHS.len());
        }
    }
}
