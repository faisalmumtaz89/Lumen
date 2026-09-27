//! The bfloat16 KV cache's writers and widening read against their oracles.
//!
//! Requires the `cuda` feature; every case is skipped where there is no
//! device. Run:
//!     cargo test --release -p lumen-runtime --features cuda --test cuda_kv_bf16_test
//!
//! The bfloat16 store changes exactly one thing, the rounding on store, so the
//! contract is bit-level:
//!   * the store's integer rounding is the device's own `cvt.rn.bf16.f32` on
//!     every non-NaN input, NaN staying NaN;
//!   * the writers produce the round-to-nearest-even bits a host conversion
//!     produces and count exactly the values stored as ±Inf or NaN;
//!   * the widening read reproduces the host widening bit for bit;
//!   * the fused prep's bfloat16 twin produces the F32 kernel's Q, gate and K
//!     bit for bit and stores the rounded values the F32 kernel stores.
//!
//! The decode partial over a bfloat16 store is held to the F32 store bit for
//! bit in `prefill_attention.rs`
//! (`bf16_store_dispatch_reproduces_the_f32_store_bit_for_bit`).
#![cfg(feature = "cuda")]

use cudarc::driver::{LaunchConfig, PushKernelArg};
use lumen_runtime::cuda::ffi::CudaDevice;
use lumen_runtime::cuda::shaders::{KV_CACHE_16_KERNEL_SOURCE, QGATE_FUSION_KERNEL_SOURCE};

fn try_device() -> Option<CudaDevice> {
    match CudaDevice::new(0) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("Skipping: no CUDA GPU available: {e}");
            None
        }
    }
}

/// Host bfloat16 conversion, round to nearest even: the oracle. NaN is
/// reported by `is_nan`, never compared bit for bit.
fn f32_to_bf16_bits(x: f32) -> u16 {
    assert!(!x.is_nan());
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn bf16_bits_to_f32(h: u16) -> f32 {
    f32::from_bits(u32::from(h) << 16)
}

fn stored_non_finite(h: u16) -> bool {
    (h & 0x7f80) == 0x7f80
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

fn bits(xs: &[f32]) -> Vec<u32> {
    xs.iter().map(|x| x.to_bits()).collect()
}

/// The values that separate round-to-nearest-even from truncation, and the
/// ones that must be counted (stored as ±Inf or NaN).
#[allow(clippy::approx_constant)] // -3.14159 is an ordinary sample value, not π
fn special_values() -> Vec<(f32, bool)> {
    let two = |e: i32| 2f32.powi(e);
    vec![
        (0.0, false),
        (-0.0, false),
        (f32::from_bits(1), false),      // smallest subnormal: rounds to 0
        (f32::from_bits(0x8000), false), // tie between 0 and the smallest bf16 subnormal -> 0
        (f32::from_bits(0x18000), false), // tie -> even (2 * 2^-133)
        (f32::MIN_POSITIVE, false),      // smallest normal, exact
        (1.0 + two(-8), false),          // tie -> 1.0 (even)
        (1.0 + 3.0 * two(-8), false),    // tie -> 1 + 2^-6 (even)
        (1.0 + two(-8) + two(-20), false), // above the tie -> rounds up
        (-3.141_59, false),
        (1e-3, false),
        (123.456, false),
        (65_520.0, false), // fits bfloat16 (half would overflow)
        (1e30, false),
        (f32::from_bits(0x7f7f_7fff), false), // below the largest float's midpoint: finite
        (f32::from_bits(0x7f7f_8000), true),  // the midpoint above bf16's largest: rounds to Inf
        (f32::MAX, true),
        (f32::MIN, true),
        (f32::INFINITY, true),
        (f32::NEG_INFINITY, true),
        (f32::NAN, true),
    ]
}

#[test]
fn the_integer_rounding_is_the_device_conversion() {
    let Some(dev) = try_device() else { return };
    if dev.compute_capability().expect("compute capability") < (8, 0) {
        eprintln!("Skipping: cvt.rn.bf16.f32 needs compute capability 8.0");
        return;
    }
    let src = format!(
        "{KV_CACHE_16_KERNEL_SOURCE}\n\
         extern \"C\" __global__ void bf16_pair(const unsigned int* in, unsigned short* ours,\n\
             unsigned short* hw, unsigned int n) {{\n\
             unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;\n\
             if (i >= n) return;\n\
             float x = __uint_as_float(in[i]);\n\
             ours[i] = kvbf16_f32_to_bits(x);\n\
             unsigned short h;\n\
             asm(\"cvt.rn.bf16.f32 %0, %1;\" : \"=h\"(h) : \"f\"(x));\n\
             hw[i] = h;\n\
         }}\n"
    );
    // `cvt.rn.bf16.f32` needs compute_80; the store itself uses integer
    // arithmetic precisely so that it does not.
    let module = dev
        .compile_and_load_with_arch(&src, "compute_80")
        .expect("compile rounding probe");
    let f = module.load_function("bf16_pair").expect("probe");
    // Every exponent with the mantissas around the rounding point, the
    // specials, and random bit patterns.
    let mut inputs: Vec<u32> = special_values().iter().map(|(x, _)| x.to_bits()).collect();
    for sign in [0u32, 0x8000_0000] {
        for exp in 0u32..=0xff {
            for low in [
                0u32, 1, 0x7fff, 0x8000, 0x8001, 0xffff, 0x1_7fff, 0x1_8000, 0x7f_ffff,
            ] {
                inputs.push(sign | (exp << 23) | low);
            }
        }
    }
    let mut s = 0xB16F_0001u64;
    inputs.extend((0..1_000_000).map(|_| rng_next(&mut s) as u32));
    let n = inputs.len() as u32;
    let in_g = dev.htod_copy(&inputs).unwrap();
    let mut ours = dev.alloc_zeros::<u16>(inputs.len()).unwrap();
    let mut hw = dev.alloc_zeros::<u16>(inputs.len()).unwrap();
    unsafe {
        dev.stream
            .launch_builder(&f)
            .arg(&in_g)
            .arg(&mut ours)
            .arg(&mut hw)
            .arg(&n)
            .launch(LaunchConfig {
                grid_dim: (n.div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
            .expect("probe launch");
    }
    dev.synchronize().unwrap();
    let ours: Vec<u16> = dev.dtoh_copy(&ours).unwrap();
    let hw: Vec<u16> = dev.dtoh_copy(&hw).unwrap();
    let mut nans = 0usize;
    for (i, &x) in inputs.iter().enumerate() {
        if f32::from_bits(x).is_nan() {
            assert!(
                bf16_bits_to_f32(ours[i]).is_nan(),
                "NaN {x:#010x} stored as {:#06x}",
                ours[i]
            );
            nans += 1;
        } else {
            assert_eq!(
                ours[i], hw[i],
                "input {x:#010x}: integer {:#06x}, device {:#06x}",
                ours[i], hw[i]
            );
            assert_eq!(
                ours[i],
                f32_to_bf16_bits(f32::from_bits(x)),
                "input {x:#010x}: host oracle"
            );
        }
    }
    assert!(nans > 0 && nans < inputs.len());
}

#[test]
fn the_writers_round_to_nearest_even_and_count_what_is_not_finite() {
    let Some(dev) = try_device() else { return };
    let module = dev
        .compile_and_load(KV_CACHE_16_KERNEL_SOURCE)
        .expect("compile 16-bit writers");
    let write = module.load_function("kv_cache_write_bf16").expect("write");
    let write_batch = module
        .load_function("kv_cache_write_batch_bf16")
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
    let check_row = |got: &[u16], row: usize| {
        for (i, (x, counted)) in specials.iter().enumerate() {
            let h = got[row * head_dim as usize + i];
            assert_eq!(
                stored_non_finite(h),
                *counted,
                "value {x:e} stored as {h:#06x}"
            );
            if x.is_nan() {
                assert!(bf16_bits_to_f32(h).is_nan());
            } else {
                assert_eq!(h, f32_to_bf16_bits(*x), "value {x:e} (index {i})");
            }
        }
    };

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
    check_row(&got, pos as usize);
    assert_eq!(
        dev.dtoh_copy(&overflow).unwrap()[0],
        expected_count,
        "overflow count"
    );
    assert!(got[..(pos * head_dim) as usize].iter().all(|&h| h == 0));

    // Batched writer: the same row at positions 0 and 1, counted again.
    let two_rows: Vec<f32> = data.iter().chain(data.iter()).copied().collect();
    let two_rows_gpu = dev.htod_copy(&two_rows).unwrap();
    let (pos_start, batch) = (0u32, 2u32);
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
    check_row(&got, 0);
    check_row(&got, 1);
    assert_eq!(
        dev.dtoh_copy(&overflow).unwrap()[0],
        3 * expected_count,
        "overflow count after the batch"
    );
}

#[test]
fn the_widening_read_is_exact() {
    let Some(dev) = try_device() else { return };
    let module = dev
        .compile_and_load(KV_CACHE_16_KERNEL_SOURCE)
        .expect("compile 16-bit kernels");
    let widen = module.load_function("kv_cache_widen_bf16").expect("widen");
    let (num_kv_heads, max_seq_len, head_dim, count) = (4u32, 300u32, 256u32, 173u32);
    let mut s = 0xB16F_0003u64;
    let total = (num_kv_heads * max_seq_len * head_dim) as usize;
    let cache_host: Vec<u16> = (0..total)
        .map(|_| f32_to_bf16_bits(rand_unit(&mut s) * 1e6))
        .collect();
    let cache = dev.htod_copy(&cache_host).unwrap();
    let out_len = (num_kv_heads * count * head_dim) as usize;
    let mut out = dev.htod_copy(&vec![f32::NAN; out_len]).unwrap();
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
                let have = got[(h * count as usize + p) * head_dim as usize + d];
                assert_eq!(
                    have.to_bits(),
                    bf16_bits_to_f32(src).to_bits(),
                    "head {h} pos {p} dim {d}"
                );
            }
        }
    }
}

/// The fused Q/K/V prep writer: its bfloat16 twin must produce the same Q,
/// gate and K outputs bit for bit (the store type touches only the cache
/// stores), write the round-to-nearest-even bfloat16s of exactly the values
/// the F32 kernel stores, and count exactly the values stored as ±Inf or NaN.
#[test]
fn the_fused_prep_bf16_twin_matches_the_f32_kernel_and_rounds_its_stores() {
    let Some(dev) = try_device() else { return };
    let module = dev
        .compile_and_load(QGATE_FUSION_KERNEL_SOURCE)
        .expect("compile fused prep");
    let f32_fn = module.load_function("attn_prep_fused").expect("f32 fused");
    let bf16_fn = module
        .load_function("attn_prep_fused_kvbf16")
        .expect("bf16 fused");
    let (nqh, nkv, hd, msl, pos) = (6u32, 1u32, 256u32, 8u32, 3u32);
    let (eps, theta, rotary_dim) = (1e-6f32, 10_000.0f32, 64u32);
    let mut s = 0xB16F_0004u64;
    let qgate: Vec<f32> = (0..(nqh * hd * 2) as usize)
        .map(|_| rand_unit(&mut s))
        .collect();
    let q_norm: Vec<f32> = (0..hd as usize)
        .map(|_| 0.5 + rand_unit(&mut s) * 0.25)
        .collect();
    let k_norm: Vec<f32> = (0..hd as usize)
        .map(|_| 0.5 + rand_unit(&mut s) * 0.25)
        .collect();
    let k_in: Vec<f32> = (0..(nkv * hd) as usize)
        .map(|_| rand_unit(&mut s))
        .collect();
    // A V with values far outside half's range and a few that are stored as
    // ±Inf or NaN, so the count is exercised.
    let mut v_in: Vec<f32> = (0..(nkv * hd) as usize)
        .map(|i| {
            if i % 5 == 0 {
                rand_unit(&mut s) * 1e30
            } else {
                rand_unit(&mut s)
            }
        })
        .collect();
    v_in[7] = f32::MAX;
    v_in[11] = f32::NAN;
    v_in[13] = f32::NEG_INFINITY;
    // A NaN that truncation would turn into +Inf, and the ties that separate
    // round-to-nearest-even from round-half-up and from truncation.
    v_in[17] = f32::from_bits(0x7f80_0001);
    v_in[19] = 1.0 + 2f32.powi(-8);
    v_in[23] = 1.0 + 3.0 * 2f32.powi(-8);
    v_in[29] = f32::from_bits(0x0001_8000);
    let expected_count = 4u32;

    let qgate_g = dev.htod_copy(&qgate).unwrap();
    let qn_g = dev.htod_copy(&q_norm).unwrap();
    let kn_g = dev.htod_copy(&k_norm).unwrap();
    let v_g = dev.htod_copy(&v_in).unwrap();
    let cache_len = (nkv * msl * hd) as usize;
    let cfg = LaunchConfig {
        grid_dim: (nqh + 2 * nkv, 1, 1),
        block_dim: (hd, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut q32 = dev.alloc_zeros::<f32>((nqh * hd) as usize).unwrap();
    let mut gate32 = dev.alloc_zeros::<f32>((nqh * hd) as usize).unwrap();
    let mut k32 = dev.htod_copy(&k_in).unwrap();
    let mut kc32 = dev.alloc_zeros::<f32>(cache_len).unwrap();
    let mut vc32 = dev.alloc_zeros::<f32>(cache_len).unwrap();
    let mut q16 = dev.alloc_zeros::<f32>((nqh * hd) as usize).unwrap();
    let mut gate16 = dev.alloc_zeros::<f32>((nqh * hd) as usize).unwrap();
    let mut k16 = dev.htod_copy(&k_in).unwrap();
    let mut kc16 = dev.alloc_zeros::<u16>(cache_len).unwrap();
    let mut vc16 = dev.alloc_zeros::<u16>(cache_len).unwrap();
    let mut overflow = dev.alloc_zeros::<u32>(1).unwrap();
    unsafe {
        dev.stream
            .launch_builder(&f32_fn)
            .arg(&qgate_g)
            .arg(&mut q32)
            .arg(&mut gate32)
            .arg(&mut k32)
            .arg(&v_g)
            .arg(&qn_g)
            .arg(&kn_g)
            .arg(&mut kc32)
            .arg(&mut vc32)
            .arg(&pos)
            .arg(&msl)
            .arg(&nqh)
            .arg(&nkv)
            .arg(&hd)
            .arg(&eps)
            .arg(&theta)
            .arg(&rotary_dim)
            .launch(cfg)
            .expect("f32 fused launch");
        dev.stream
            .launch_builder(&bf16_fn)
            .arg(&qgate_g)
            .arg(&mut q16)
            .arg(&mut gate16)
            .arg(&mut k16)
            .arg(&v_g)
            .arg(&qn_g)
            .arg(&kn_g)
            .arg(&mut kc16)
            .arg(&mut vc16)
            .arg(&mut overflow)
            .arg(&pos)
            .arg(&msl)
            .arg(&nqh)
            .arg(&nkv)
            .arg(&hd)
            .arg(&eps)
            .arg(&theta)
            .arg(&rotary_dim)
            .launch(cfg)
            .expect("bf16 fused launch");
    }
    dev.synchronize().unwrap();
    let get = |x: &cudarc::driver::CudaSlice<f32>| -> Vec<f32> { dev.dtoh_copy(x).unwrap() };
    assert_eq!(bits(&get(&q32)), bits(&get(&q16)), "Q differs");
    assert_eq!(bits(&get(&gate32)), bits(&get(&gate16)), "gate differs");
    assert_eq!(bits(&get(&k32)), bits(&get(&k16)), "K differs");
    let (kc32, vc32) = (get(&kc32), get(&vc32));
    let kc16: Vec<u16> = dev.dtoh_copy(&kc16).unwrap();
    let vc16: Vec<u16> = dev.dtoh_copy(&vc16).unwrap();
    let slot = (pos * hd) as usize..((pos + 1) * hd) as usize;
    for i in slot.clone() {
        assert_eq!(
            kc16[i],
            f32_to_bf16_bits(kc32[i]),
            "K cache {i}: {:e}",
            kc32[i]
        );
        if vc32[i].is_nan() {
            assert!(bf16_bits_to_f32(vc16[i]).is_nan(), "V cache {i}: NaN");
        } else {
            assert_eq!(
                vc16[i],
                f32_to_bf16_bits(vc32[i]),
                "V cache {i}: {:e}",
                vc32[i]
            );
        }
    }
    assert!(
        kc16[..slot.start].iter().all(|&h| h == 0) && vc16[..slot.start].iter().all(|&h| h == 0)
    );
    assert_eq!(
        dev.dtoh_copy(&overflow).unwrap()[0],
        expected_count,
        "overflow count from the fused writer"
    );
}
