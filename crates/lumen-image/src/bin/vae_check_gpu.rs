//! The VAE decoder's GPU forward, against the reference decodes `oracle_dump.py`
//! writes — and against the CPU decoder on the same input.
//!
//! The GPU decoder runs in bf16, the CPU decoder (`vae.rs`) in f32. The GPU
//! decodes `vae_decode_input_bf16`, the final latents denormalised in bf16 as
//! Lumen's pipeline does, and is held to the image's own bars
//! (`vae_decode_bars`): the largest distance, from the reference's bf16 decode
//! and from its f32 one, of a bf16 decoder that differs from the reference only
//! in summation order, attention kernel or norm reduction, plus the spread of
//! those distances. On the reference
//! images the f32 decoder lands outside the first bar, so it tells a bf16
//! decoder that rounds where the reference does from one that does not. The
//! CPU decoder is held to the reference's f32 decode within 1e-3, and its
//! distance from the GPU decoder is reported.
//!
//! The run fails when any distance is above its bar, or when a decode split
//! into bands differs from the one-pass decode in any bit — on the oracle
//! latents and on wide, tall and large pseudo-random ones. Every comparison runs
//! before the result is reported.
//!
//! Usage: `vae-check-gpu <lbi-dir> <oracle-dir> [tag]`

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use lumen_image::cuda::vae_gpu::VaeGpu;
use lumen_image::npy;
use lumen_image::vae::VaeDecoder;
use lumen_runtime::cuda::ffi::CudaDevice;

/// PSNR against a peak of 2.0, the range the decoder clamps to.
fn psnr(got: &[f32], want: &[f32]) -> f32 {
    let mse: f64 = got
        .iter()
        .zip(want)
        .map(|(g, w)| {
            let d = (*g as f64) - (*w as f64);
            d * d
        })
        .sum::<f64>()
        / got.len() as f64;
    if mse == 0.0 {
        f32::INFINITY
    } else {
        (10.0 * (4.0f64 / mse).log10()) as f32
    }
}

fn rel_l2(got: &[f32], want: &[f32]) -> f32 {
    let mut num = 0f64;
    let mut den = 0f64;
    for (g, w) in got.iter().zip(want) {
        let g = *g as f64;
        let w = *w as f64;
        num += (g - w) * (g - w);
        den += w * w;
    }
    (num.sqrt() / den.sqrt().max(1e-30)) as f32
}

fn max_abs(got: &[f32], want: &[f32]) -> f32 {
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0f32, f32::max)
}

fn gib(bytes: usize) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

/// Fail unless `got` and `want` hold the same bits in every position.
fn same_bits(label: &str, got: &[f32], want: &[f32]) -> Result<(), String> {
    let differ = got
        .iter()
        .zip(want)
        .filter(|(g, w)| g.to_bits() != w.to_bits())
        .count();
    println!("  {label:<44} {differ} of {} values differ", want.len());
    if got.len() != want.len() || differ > 0 {
        return Err(format!("{label}: not the same bits"));
    }
    Ok(())
}

/// Deterministic latents in [-2, 2), so a run is reproducible without a dump.
fn pseudo(n: usize) -> Vec<f32> {
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
        })
        .collect()
}

/// Print the three distances and return the relative L2.
fn distance(label: &str, got: &[f32], want: &[f32]) -> Result<f32, String> {
    if got.len() != want.len() {
        return Err(format!(
            "{label}: {got_len} values against {want_len}",
            got_len = got.len(),
            want_len = want.len()
        ));
    }
    let r = rel_l2(got, want);
    println!(
        "  {label:<22} rel-L2={r:.3e}  max_abs={:.3e}  PSNR={:.2} dB",
        max_abs(got, want),
        psnr(got, want)
    );
    Ok(r)
}

/// Print the three distances and fail when the relative L2 is above `bar`.
fn report(label: &str, got: &[f32], want: &[f32], bar: f32) -> Result<(), String> {
    let r = distance(label, got, want)?;
    if r.is_nan() || r > bar {
        return Err(format!(
            "{label}: rel-L2 {r:.3e} is not within the {bar:.3e} bar"
        ));
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let usage = || {
        eprintln!("usage: vae-check-gpu <lbi-dir> <oracle-dir> [tag]");
        std::process::exit(2)
    };
    let lbi_dir = PathBuf::from(args.next().unwrap_or_else(usage));
    let oracle = PathBuf::from(args.next().unwrap_or_else(usage));
    let tag = args.next().unwrap_or_else(|| "smoke".to_string());

    let load = |name: &str| -> Result<npy::Npy, String> {
        let p = oracle.join(format!("{name}_{tag}.npy"));
        npy::load(&p).map_err(|e| format!("{}: {e}", p.display()))
    };

    // The oracle's own inputs, read before the device is touched so a bad
    // oracle directory fails without a 1 GiB upload in the way.
    let z = load("vae_decode_input")?;
    let want = load("vae_decode_output")?;
    let z16 = load("vae_decode_input_bf16")?;
    let want16 = load("vae_decode_output_bf16")?;
    let bars = load("vae_decode_bars")?;
    let &[bar16, bar32] = bars.data.as_slice() else {
        return Err(format!(
            "vae_decode_bars holds {} values, not 2",
            bars.data.len()
        ));
    };
    if !(bar16.is_finite() && bar32.is_finite() && bar16 > 0.0 && bar32 > 0.0) {
        return Err(format!(
            "vae_decode_bars [{bar16}, {bar32}] are not positive and finite"
        ));
    }

    // The oracle's VAE input is [batch, 64, 1, H, W] with H,W the latent size.
    if z.shape.len() != 5 || z16.shape != z.shape {
        return Err(format!(
            "vae_decode_input shapes {:?} and {:?} are not one 5-D shape",
            z.shape, z16.shape
        ));
    }
    let (batch, channels, frames, h, w) =
        (z.shape[0], z.shape[1], z.shape[2], z.shape[3], z.shape[4]);
    if frames != 1 {
        return Err(format!("expected a single frame, got {frames}"));
    }
    println!(
        "decoding [{batch}, {channels}, {frames}, {h}, {w}] -> [_, 4, 1, {}, {}]",
        h * 16,
        w * 16
    );

    let lbi = lbi_dir.join("vae.lbi");

    // --- the CPU decoder, in f32 ----------------------------------------------
    let cpu_decoder = VaeDecoder::load(&lbi).map_err(|e| format!("{e}"))?;
    let t0 = Instant::now();
    let cpu = cpu_decoder
        .decode(&z.data, batch, h, w)
        .map_err(|e| format!("cpu: {e}"))?;
    let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("cpu decoder   {cpu_ms:>9.1} ms");

    // --- the GPU path -------------------------------------------------------
    let dev = CudaDevice::new(0).map_err(|e| format!("no CUDA device: {e}"))?;
    println!(
        "device: {}  free {:.2} GiB of {:.2} GiB",
        dev.name().unwrap_or_else(|_| "<unknown>".into()),
        gib(dev.free_memory().map_err(|e| e.to_string())?),
        gib(dev.total_memory().map_err(|e| e.to_string())?),
    );

    let free_before = dev.free_memory().map_err(|e| e.to_string())?;
    let mut gpu_decoder = VaeGpu::load(&lbi, &dev).map_err(|e| format!("{e}"))?;
    let free_after = dev.free_memory().map_err(|e| e.to_string())?;
    println!(
        "resident weights: {:.3} GiB (free {:.2} GiB after load)",
        (free_before as f64 - free_after as f64) / (1u64 << 30) as f64,
        gib(free_after),
    );

    let t1 = Instant::now();
    let got16 = gpu_decoder
        .decode(&z16.data, batch, h, w)
        .map_err(|e| format!("gpu: {e}"))?;
    let gpu_ms = t1.elapsed().as_secs_f64() * 1000.0;
    println!(
        "gpu decoder   {gpu_ms:>9.1} ms  ({:.2}x the CPU)",
        cpu_ms / gpu_ms
    );
    let got = gpu_decoder
        .decode(&z.data, batch, h, w)
        .map_err(|e| format!("gpu: {e}"))?;

    let mut failures = Vec::new();
    let mut check = |result: Result<(), String>| {
        if let Err(e) = result {
            println!("  FAIL: {e}");
            failures.push(e);
        }
    };

    // --- the comparisons ----------------------------------------------------
    println!("comparisons (bars {bar16:.3e} and {bar32:.3e}):");
    check(report("gpu vs bf16 oracle", &got16, &want16.data, bar16));
    check(report("gpu vs f32 oracle", &got16, &want.data, bar32));
    check(distance("gpu vs cpu", &got, &cpu).map(|_| ()));
    // The CPU path should reproduce `vae-check`'s own number. Reporting it makes
    // a bad oracle dump indistinguishable from a bad GPU path only if this line
    // is also wrong, which is the point of printing it.
    check(report("cpu vs oracle", &cpu, &want.data, 1e-3));

    // A banded decode keeps only values computed from the whole-image inputs,
    // so it must reproduce the one-pass decode bit for bit: two bands, an
    // uneven split, and one latent row per band, where every band leans on its
    // context rows. Then latents too wide and too tall for the oracle cases,
    // whose convolutions run at the shapes large images use.
    println!(
        "banded decodes ({} context rows per side):",
        gpu_decoder.halo_rows()
    );
    for bands in [2, 5, h] {
        match gpu_decoder.decode_in_bands(&z16.data, batch, h, w, bands) {
            Ok(banded) => check(same_bits(
                &format!("{bands} bands vs one pass"),
                &banded,
                &got16,
            )),
            Err(e) => check(Err(format!("gpu, {bands} bands: {e}"))),
        }
    }
    // Widths of an even number of latent columns that are not a multiple of
    // four (4064 pixels is 254), where a band's rows are not 16-byte aligned
    // the way the whole image's are, with band heights odd and even.
    for (lh, lw, splits) in [
        (32usize, 256usize, [2, 5, 17]),
        (256, 32, [2, 5, 17]),
        (128, 128, [2, 5, 17]),
        (16, 254, [3, 5, 8]),
        (254, 66, [3, 5, 8]),
        (66, 66, [3, 5, 8]),
    ] {
        let latents = pseudo(batch * channels * lh * lw);
        let one = gpu_decoder
            .decode_in_bands(&latents, batch, lh, lw, 1)
            .map_err(|e| format!("gpu, {lh}x{lw} latent: {e}"))?;
        for bands in splits {
            match gpu_decoder.decode_in_bands(&latents, batch, lh, lw, bands) {
                Ok(banded) => check(same_bits(
                    &format!("{lh}x{lw} latent, {bands} bands vs one pass"),
                    &banded,
                    &one,
                )),
                Err(e) => check(Err(format!("gpu, {lh}x{lw} latent, {bands} bands: {e}"))),
            }
        }
    }

    // The mid block's attention in chunks of query rows, about 23 of them
    // with a short last one, against the one chunk these latents take by
    // default. The chunks' products have another shape, so their sums round
    // differently, and the decoder carries that to the output at the scale of
    // a bf16 rounding here and there, a summation-order difference the bar
    // allows; a chunk that read or wrote the wrong rows moves the output by
    // order one.
    let seq = h * w;
    gpu_decoder.set_attention_score_budget(seq.div_ceil(23) * seq * 4);
    let chunked = gpu_decoder
        .decode(&z16.data, batch, h, w)
        .map_err(|e| format!("gpu, chunked attention: {e}"))?;
    gpu_decoder.set_attention_score_budget(usize::MAX);
    let differ = chunked
        .iter()
        .zip(&got16)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    println!(
        "attention in chunks: {differ} of {} values differ",
        got16.len()
    );
    check(report("chunked vs whole", &chunked, &got16, bar16));
    check(report(
        "chunked vs bf16 oracle",
        &chunked,
        &want16.data,
        bar16,
    ));

    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("; "))
    }
}
