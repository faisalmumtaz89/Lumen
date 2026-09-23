//! The VAE decoder's GPU forward, against the same oracle tensors `vae-check`
//! compares — and against the CPU decoder on the same input.
//!
//! Three numbers per pair, so the check answers two questions at once:
//!   - GPU vs the reference's own output: does the GPU path decode as the
//!     reference does?
//!   - GPU vs the CPU decoder on the same latents: does the GPU path compute
//!     the same function as the independent f32 implementation?
//!
//! The reference runs its convolutions in TF32 (torch's default for cuDNN),
//! as the GPU path does; `vae.rs` runs them in full f32. So the first
//! comparison is the sharper one: on the `main` oracle case the GPU decode is
//! rel-L2 5.7e-5 from the reference (the reference itself lands 5.6e-5 from its
//! own output when cuDNN picks another algorithm), where full-f32 convolutions
//! land 3.6e-4 away, and its 1.5e-4 bar tells the two apart. The GPU decode is
//! 3.6e-4 from the CPU decoder, the TF32 rounding, and that bar of 1e-3 catches
//! a decoder that computes something else. The run fails when either distance is above its
//! bar.
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

/// Print the three distances and fail when the relative L2 is above `bar`.
fn report(label: &str, got: &[f32], want: &[f32], bar: f32) -> Result<(), String> {
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
    if r.is_nan() || r > bar {
        return Err(format!(
            "{label}: rel-L2 {r:.3e} is not within the {bar:.0e} bar"
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

    // The oracle's own input, read before the device is touched so a bad oracle
    // directory fails without a 1 GiB upload in the way.
    let z = load("vae_decode_input")?;
    let want = load("vae_decode_output")?;

    // The oracle's VAE input is [batch, 64, 1, H, W] with H,W the latent size.
    if z.shape.len() != 5 {
        return Err(format!("vae_decode_input shape {:?} is not 5-D", z.shape));
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

    // --- the CPU reference, which is what the GPU path must reproduce --------
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
    let gpu_decoder = VaeGpu::load(&lbi, &dev).map_err(|e| format!("{e}"))?;
    let free_after = dev.free_memory().map_err(|e| e.to_string())?;
    println!(
        "resident weights: {:.3} GiB (free {:.2} GiB after load)",
        (free_before as f64 - free_after as f64) / (1u64 << 30) as f64,
        gib(free_after),
    );

    let free_pre = dev.free_memory().map_err(|e| e.to_string())?;
    let t1 = Instant::now();
    let got = gpu_decoder
        .decode(&z.data, batch, h, w)
        .map_err(|e| format!("gpu: {e}"))?;
    let gpu_ms = t1.elapsed().as_secs_f64() * 1000.0;
    let free_post = dev.free_memory().map_err(|e| e.to_string())?;
    println!(
        "gpu decoder   {gpu_ms:>9.1} ms  ({:.2}x the CPU, peak activation {:.2} GiB)",
        cpu_ms / gpu_ms,
        (free_pre as f64 - free_post as f64) / (1u64 << 30) as f64,
    );

    // --- the comparisons ----------------------------------------------------
    println!("comparisons:");
    report("gpu vs oracle", &got, &want.data, 1.5e-4)?;
    report("gpu vs cpu", &got, &cpu, 1e-3)?;
    // The CPU path should reproduce `vae-check`'s own number. Reporting it makes
    // a bad oracle dump indistinguishable from a bad GPU path only if this line
    // is also wrong, which is the point of printing it.
    report("cpu vs oracle", &cpu, &want.data, 1e-3)?;

    if got.len() != want.data.len() {
        return Err(format!(
            "gpu output has {} values, the reference has {} (shape {:?})",
            got.len(),
            want.data.len(),
            want.shape
        ));
    }
    Ok(())
}
