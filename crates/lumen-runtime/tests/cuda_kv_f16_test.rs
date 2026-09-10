//! The 16-bit KV cache's kernels against their F32 originals.
//!
//! Requires the `cuda` feature; every case is skipped where there is no
//! device. Run:
//!     cargo test --release -p lumen-runtime --features cuda --test cuda_kv_f16_test
//!
//! The half store changes exactly one thing, the rounding on store; the
//! readers widen exactly and keep the F32 kernels' arithmetic and order. So
//! the contract is bit-level, not a tolerance:
//!   * the writers produce the IEEE round-to-nearest-even bits a host
//!     conversion produces, over the special values that distinguish RNE
//!     from truncation, and count exactly the inputs that do not fit;
//!   * on half-representable inputs the half partial reproduces the F32
//!     partial's (m, l, o) bit for bit and the half tiled kernel the F32
//!     tiled kernel's output, at every context that crosses a tile or chunk
//!     boundary;
//!   * the widening read reproduces the host widening bit for bit.
#![cfg(feature = "cuda")]

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use lumen_runtime::cuda::ffi::CudaDevice;
use lumen_runtime::cuda::shaders::{
    ATTENTION_DECODE_SPLITK_GQA6_KERNEL_SOURCE, ATTENTION_DECODE_TILED_KERNEL_SOURCE,
    KV_CACHE_F16_KERNEL_SOURCE,
};
use lumen_runtime::cuda::{
    attn_splitk_gqa6_chunks, attn_splitk_gqa6_merge_shared_bytes,
    attn_splitk_gqa6_partial_shared_bytes, attn_splitk_gqa6_partial_shared_bytes_f16,
    ATTN_DECODE_TILED_BLOCK_DIM as BLOCK_DIM, ATTN_DECODE_TILED_T_C as T_C,
    ATTN_SPLITK_GQA6_CHUNK as GQA6_CHUNK, ATTN_SPLITK_GQA6_DIM_TILES as GQA6_DIM_TILES,
    ATTN_SPLITK_GQA6_HEAD_DIM as HEAD_DIM,
};

const NUM_HEADS: u32 = 24;
const NUM_KV_HEADS: u32 = 4;
const MAX_SEQ_LEN: u32 = 8_192;
const SCALE: f32 = 0.0625;

/// Either side of one GQA-shared chunk (16), one tiled tile (128), the
/// board shape and a multi-tile context.
const LENGTHS: &[u32] = &[
    1, 15, 16, 17, 127, 128, 129, 330, 1100, 1300, 2600, 4096, 4097, 8192,
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

// ---- host half conversion (round to nearest even), the oracle ---------

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

fn rng_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn rand_unit(s: &mut u64) -> f32 {
    let u = (rng_next(s) >> 40) as f32 / (1u32 << 24) as f32;
    u * 2.0 - 1.0
}

struct Inputs {
    q: Vec<f32>,
    /// Half-representable K and V, as the floats they denote and as bits.
    k: Vec<f32>,
    v: Vec<f32>,
    k16: Vec<u16>,
    v16: Vec<u16>,
}

fn make_inputs(seed: u64) -> Inputs {
    let cache = (NUM_KV_HEADS * MAX_SEQ_LEN * HEAD_DIM) as usize;
    let mut s = seed;
    let q: Vec<f32> = (0..(NUM_HEADS * HEAD_DIM) as usize)
        .map(|_| rand_unit(&mut s) * 4.0)
        .collect();
    let k16: Vec<u16> = (0..cache)
        .map(|_| f32_to_f16_bits(rand_unit(&mut s)))
        .collect();
    let v16: Vec<u16> = (0..cache)
        .map(|_| f32_to_f16_bits(rand_unit(&mut s)))
        .collect();
    Inputs {
        q,
        k: k16.iter().map(|&h| f16_bits_to_f32(h)).collect(),
        v: v16.iter().map(|&h| f16_bits_to_f32(h)).collect(),
        k16,
        v16,
    }
}

fn nan_filled(dev: &CudaDevice, n: usize) -> CudaSlice<f32> {
    dev.htod_copy(&vec![f32::NAN; n]).expect("alloc")
}

/// The GQA-shared partial's (m, l, o) and merged output from the named
/// partial kernel over the given caches.
#[allow(clippy::too_many_arguments)]
fn run_gqa6_partial<K: cudarc::driver::DeviceRepr>(
    dev: &CudaDevice,
    partial_name: &str,
    shared_bytes: u32,
    q: &CudaSlice<f32>,
    k: &CudaSlice<K>,
    v: &CudaSlice<K>,
    seq_len: u32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let module = dev
        .compile_and_load(ATTENTION_DECODE_SPLITK_GQA6_KERNEL_SOURCE)
        .expect("compile GQA-shared pair");
    let partial = module.load_function(partial_name).expect("partial");
    let merge = module
        .load_function("attention_decode_splitk_merge_gqa6_f32")
        .expect("merge");
    let chunks = attn_splitk_gqa6_chunks(seq_len);
    let n_part = (NUM_HEADS * chunks) as usize;
    let mut m_part = nan_filled(dev, n_part);
    let mut l_part = nan_filled(dev, n_part);
    let mut o_part = nan_filled(dev, n_part * HEAD_DIM as usize);
    let mut out = nan_filled(dev, (NUM_HEADS * HEAD_DIM) as usize);
    let chunk = GQA6_CHUNK;
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
            .arg(&chunk)
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
                grid_dim: (NUM_HEADS, GQA6_DIM_TILES, 1),
                block_dim: (BLOCK_DIM, 1, 1),
                shared_mem_bytes: attn_splitk_gqa6_merge_shared_bytes(chunks),
            })
            .expect("merge launch");
    }
    dev.synchronize().expect("sync");
    (
        dev.dtoh_copy(&m_part).unwrap(),
        dev.dtoh_copy(&l_part).unwrap(),
        dev.dtoh_copy(&o_part).unwrap(),
        dev.dtoh_copy(&out).unwrap(),
    )
}

fn run_tiled<K: cudarc::driver::DeviceRepr>(
    dev: &CudaDevice,
    name: &str,
    q: &CudaSlice<f32>,
    k: &CudaSlice<K>,
    v: &CudaSlice<K>,
    seq_len: u32,
) -> Vec<f32> {
    let module = dev
        .compile_and_load(ATTENTION_DECODE_TILED_KERNEL_SOURCE)
        .expect("compile tiled");
    let kernel = module.load_function(name).expect("tiled kernel");
    let mut out = nan_filled(dev, (NUM_HEADS * HEAD_DIM) as usize);
    let (nh, nkv, hd, msl, scale) = (NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, MAX_SEQ_LEN, SCALE);
    unsafe {
        dev.stream
            .launch_builder(&kernel)
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(&mut out)
            .arg(&nh)
            .arg(&nkv)
            .arg(&hd)
            .arg(&seq_len)
            .arg(&msl)
            .arg(&scale)
            .launch(LaunchConfig {
                grid_dim: (NUM_HEADS, 1, 1),
                block_dim: (BLOCK_DIM, 1, 1),
                shared_mem_bytes: (8 + HEAD_DIM + T_C) * 4,
            })
            .expect("tiled launch");
    }
    dev.synchronize().expect("sync");
    dev.dtoh_copy(&out).unwrap()
}

fn bits(xs: &[f32]) -> Vec<u32> {
    xs.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn the_half_partial_reproduces_the_f32_partial_bit_for_bit() {
    let Some(dev) = try_device() else { return };
    let inp = make_inputs(0x0005_0910_F16A_0001);
    let q = dev.htod_copy(&inp.q).unwrap();
    let k32 = dev.htod_copy(&inp.k).unwrap();
    let v32 = dev.htod_copy(&inp.v).unwrap();
    let k16 = dev.htod_copy(&inp.k16).unwrap();
    let v16 = dev.htod_copy(&inp.v16).unwrap();
    for &seq_len in LENGTHS {
        let a = run_gqa6_partial(
            &dev,
            "attention_decode_splitk_partial_gqa6_f32",
            attn_splitk_gqa6_partial_shared_bytes(),
            &q,
            &k32,
            &v32,
            seq_len,
        );
        let b = run_gqa6_partial(
            &dev,
            "attention_decode_splitk_partial_gqa6_f16",
            attn_splitk_gqa6_partial_shared_bytes_f16(),
            &q,
            &k16,
            &v16,
            seq_len,
        );
        assert!(
            a.3.iter().all(|x| x.is_finite()),
            "F32 output not finite at {seq_len}"
        );
        assert_eq!(bits(&a.0), bits(&b.0), "m differs at seq_len {seq_len}");
        assert_eq!(bits(&a.1), bits(&b.1), "l differs at seq_len {seq_len}");
        assert_eq!(bits(&a.2), bits(&b.2), "o differs at seq_len {seq_len}");
        assert_eq!(
            bits(&a.3),
            bits(&b.3),
            "merged output differs at seq_len {seq_len}"
        );
    }
}

#[test]
fn the_half_tiled_kernel_reproduces_the_f32_tiled_kernel_bit_for_bit() {
    let Some(dev) = try_device() else { return };
    let inp = make_inputs(0x0005_0910_F16A_0002);
    let q = dev.htod_copy(&inp.q).unwrap();
    let k32 = dev.htod_copy(&inp.k).unwrap();
    let v32 = dev.htod_copy(&inp.v).unwrap();
    let k16 = dev.htod_copy(&inp.k16).unwrap();
    let v16 = dev.htod_copy(&inp.v16).unwrap();
    for &seq_len in LENGTHS {
        let a = run_tiled(&dev, "attention_decode_tiled", &q, &k32, &v32, seq_len);
        let b = run_tiled(&dev, "attention_decode_tiled_f16", &q, &k16, &v16, seq_len);
        assert!(
            a.iter().all(|x| x.is_finite()),
            "F32 output not finite at {seq_len}"
        );
        assert_eq!(
            bits(&a),
            bits(&b),
            "tiled output differs at seq_len {seq_len}"
        );
    }
}

/// The values that separate round-to-nearest-even from truncation and from
/// a saturating store, and the ones that must be counted.
fn special_values() -> Vec<(f32, bool)> {
    let two = |e: i32| 2f32.powi(e);
    vec![
        (0.0, false),
        (-0.0, false),
        (two(-24), false),                  // smallest half subnormal
        (two(-25), false),                  // tie between 0 and the smallest subnormal -> even (0)
        (1.5 * two(-24), false),            // tie -> even (2^-23)
        (two(-14), false),                  // smallest half normal
        (two(-14) - two(-25), false),       // largest subnormal region
        (1.0 + two(-11), false),            // tie -> 1.0 (even)
        (1.0 + 3.0 * two(-11), false),      // tie -> 1 + 2^-9 (even)
        (1.0 + two(-11) + two(-20), false), // above the tie -> rounds up
        (-3.14159, false),
        (1e-3, false),
        (123.456, false),
        (65_504.0, false), // largest half
        (65_519.0, false), // below the midpoint: rounds to 65,504, not counted
        (65_520.0, true),  // the midpoint: rounds to infinity, counted
        (1e5, true),
        (f32::INFINITY, true),
        (f32::NEG_INFINITY, true),
        (f32::NAN, true),
        (-70_000.0, true),
    ]
}

#[test]
fn the_writers_round_to_nearest_even_and_count_what_does_not_fit() {
    let Some(dev) = try_device() else { return };
    let module = dev
        .compile_and_load(KV_CACHE_F16_KERNEL_SOURCE)
        .expect("compile half writers");
    let write = module.load_function("kv_cache_write_f16").expect("write");
    let write_batch = module
        .load_function("kv_cache_write_batch_f16")
        .expect("write batch");
    let specials = special_values();
    let head_dim: u32 = 32;
    assert!(specials.len() <= head_dim as usize);
    let num_kv_heads: u32 = 1;
    let max_seq_len: u32 = 4;
    let mut data = vec![0.5f32; head_dim as usize];
    for (i, (x, _)) in specials.iter().enumerate() {
        data[i] = *x;
    }
    let expected_count = specials.iter().filter(|(_, counted)| *counted).count() as u32;
    let data_gpu = dev.htod_copy(&data).unwrap();

    // Single-token writer at position 2.
    let mut cache = dev
        .alloc_zeros::<u16>((num_kv_heads * max_seq_len * head_dim) as usize)
        .unwrap();
    let mut overflow = dev.alloc_zeros::<u32>(1).unwrap();
    let pos: u32 = 2;
    unsafe {
        dev.stream
            .launch_builder(&write)
            .arg(&mut cache)
            .arg(&data_gpu)
            .arg(&mut overflow)
            .arg(&pos)
            .arg(&num_kv_heads)
            .arg(&max_seq_len)
            .arg(&head_dim)
            .launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (head_dim, 1, 1),
                shared_mem_bytes: 0,
            })
            .expect("write launch");
    }
    dev.synchronize().unwrap();
    let got: Vec<u16> = dev.dtoh_copy(&cache).unwrap();
    let count: Vec<u32> = dev.dtoh_copy(&overflow).unwrap();
    for (i, (x, _)) in specials.iter().enumerate() {
        let device_bits = got[(pos * head_dim) as usize + i];
        let host_bits = f32_to_f16_bits(*x);
        if x.is_nan() {
            assert!(
                (device_bits & 0x7c00) == 0x7c00 && (device_bits & 0x3ff) != 0,
                "NaN must store as a NaN, got {device_bits:#06x}"
            );
        } else {
            assert_eq!(
                device_bits, host_bits,
                "value {x:e} (index {i}): device {device_bits:#06x}, host RNE {host_bits:#06x}"
            );
        }
    }
    assert_eq!(count[0], expected_count, "overflow count");
    // Untouched positions stay zero.
    assert!(got[..(pos * head_dim) as usize].iter().all(|&h| h == 0));

    // Batched writer: the same row at positions 0 and 1, counted again.
    let two_rows: Vec<f32> = data.iter().chain(data.iter()).copied().collect();
    let two_rows_gpu = dev.htod_copy(&two_rows).unwrap();
    let pos_start: u32 = 0;
    let batch: u32 = 2;
    unsafe {
        dev.stream
            .launch_builder(&write_batch)
            .arg(&mut cache)
            .arg(&two_rows_gpu)
            .arg(&mut overflow)
            .arg(&pos_start)
            .arg(&batch)
            .arg(&num_kv_heads)
            .arg(&max_seq_len)
            .arg(&head_dim)
            .launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (batch * head_dim, 1, 1),
                shared_mem_bytes: 0,
            })
            .expect("batch write launch");
    }
    dev.synchronize().unwrap();
    let got: Vec<u16> = dev.dtoh_copy(&cache).unwrap();
    let count: Vec<u32> = dev.dtoh_copy(&overflow).unwrap();
    for row in 0..batch {
        for (i, (x, _)) in specials.iter().enumerate() {
            let device_bits = got[(row * head_dim) as usize + i];
            if x.is_nan() {
                assert!((device_bits & 0x7c00) == 0x7c00 && (device_bits & 0x3ff) != 0);
            } else {
                assert_eq!(
                    device_bits,
                    f32_to_f16_bits(*x),
                    "batch row {row} index {i}"
                );
            }
        }
    }
    assert_eq!(
        count[0],
        3 * expected_count,
        "overflow count after the batch"
    );
}

#[test]
fn the_widening_read_is_exact() {
    let Some(dev) = try_device() else { return };
    let module = dev
        .compile_and_load(KV_CACHE_F16_KERNEL_SOURCE)
        .expect("compile half kernels");
    let widen = module.load_function("kv_cache_widen_f16").expect("widen");
    let num_kv_heads: u32 = 4;
    let max_seq_len: u32 = 300;
    let head_dim: u32 = 256;
    let count: u32 = 173;
    let mut s = 0x0005_0910_F16A_0003u64;
    let total = (num_kv_heads * max_seq_len * head_dim) as usize;
    let cache_host: Vec<u16> = (0..total)
        .map(|_| f32_to_f16_bits(rand_unit(&mut s) * 100.0))
        .collect();
    let cache = dev.htod_copy(&cache_host).unwrap();
    let out_len = (num_kv_heads * count * head_dim) as usize;
    let mut out = nan_filled(&dev, out_len);
    unsafe {
        dev.stream
            .launch_builder(&widen)
            .arg(&cache)
            .arg(&mut out)
            .arg(&num_kv_heads)
            .arg(&count)
            .arg(&max_seq_len)
            .arg(&head_dim)
            .launch(LaunchConfig {
                grid_dim: (out_len.div_ceil(256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
            .expect("widen launch");
    }
    dev.synchronize().unwrap();
    let got: Vec<f32> = dev.dtoh_copy(&out).unwrap();
    for h in 0..num_kv_heads as usize {
        for p in 0..count as usize {
            for d in 0..head_dim as usize {
                let src = cache_host[(h * max_seq_len as usize + p) * head_dim as usize + d];
                let want = f16_bits_to_f32(src);
                let have = got[(h * count as usize + p) * head_dim as usize + d];
                assert_eq!(have.to_bits(), want.to_bits(), "head {h} pos {p} dim {d}");
            }
        }
    }
}
