//! Decode the oracle's own VAE input and compare against its output.
//!
//! The input is injected, so the decoder cannot hide its error behind the
//! pipeline that produced the latents.
//!
//! Usage: `vae-check <lbi-dir> <oracle-dir> [tag]`

use std::path::PathBuf;
use std::process::ExitCode;

use lumen_image::npy;
use lumen_image::vae::VaeDecoder;

fn psnr(got: &[f32], want: &[f32], peak: f32) -> f32 {
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
        (10.0 * ((peak as f64 * peak as f64) / mse).log10()) as f32
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

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let usage = || {
        eprintln!("usage: vae-check <lbi-dir> <oracle-dir> [tag]");
        std::process::exit(2)
    };
    let lbi_dir = PathBuf::from(args.next().unwrap_or_else(usage));
    let oracle = PathBuf::from(args.next().unwrap_or_else(usage));
    let tag = args.next().unwrap_or_else(|| "smoke".to_string());

    let load = |name: &str| -> Result<npy::Npy, String> {
        let p = oracle.join(format!("{name}_{tag}.npy"));
        npy::load(&p).map_err(|e| format!("{}: {e}", p.display()))
    };

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
        "decoding [{batch}, {channels}, {frames}, {h}, {w}] with {} values",
        z.data.len()
    );

    let decoder = VaeDecoder::load(&lbi_dir.join("vae.lbi")).map_err(|e| format!("{e}"))?;
    let got = decoder
        .decode(&z.data, batch, h, w)
        .map_err(|e| format!("{e}"))?;

    if got.len() != want.data.len() {
        return Err(format!(
            "output has {} values, the reference has {} (shape {:?})",
            got.len(),
            want.data.len(),
            want.shape
        ));
    }
    // Compare in [-1, 1], which is the range the decoder clamps to.
    println!(
        "vae decode    rel-L2={:.3e}  max_abs={:.3e}  PSNR={:.2} dB",
        rel_l2(&got, &want.data),
        max_abs(&got, &want.data),
        psnr(&got, &want.data, 2.0)
    );
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
