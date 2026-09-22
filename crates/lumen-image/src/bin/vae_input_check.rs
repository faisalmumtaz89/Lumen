//! Does the pipeline's latent transform reproduce the oracle's decoder input?
//!
//! The components each pass their own check, but the transform that bridges the
//! transformer's output layout to the decoder's input layout is the pipeline's,
//! and a wrong layout there still decodes to a finite image of the right shape.
//! So it is checked directly against the tensor the decoder was actually given.
//!
//! Usage: `vae-input-check <oracle-dir> [tag]`

use std::path::PathBuf;
use std::process::ExitCode;

use lumen_image::npy;
use lumen_image::vae::{denormalize_latents_with, VaeConfig};

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

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let oracle = PathBuf::from(args.next().unwrap_or_else(|| {
        eprintln!("usage: vae-input-check <oracle-dir> [tag]");
        std::process::exit(2)
    }));
    let tag = args.next().unwrap_or_else(|| "smoke".to_string());

    let latents = npy::load(&oracle.join(format!("latents_{tag}.npy")))
        .map_err(|e| format!("latents: {e}"))?;
    let want = npy::load(&oracle.join(format!("vae_decode_input_{tag}.npy")))
        .map_err(|e| format!("vae_decode_input: {e}"))?;

    // The final denoising step's latent, [batch, tokens, channels].
    if latents.shape.len() != 4 {
        return Err(format!("latents shape {:?}", latents.shape));
    }
    let (seq, ch) = (latents.shape[2], latents.shape[3]);
    let last = &latents.data[(latents.shape[0] - 1) * seq * ch..];

    // seq = h*w; the oracle used a square latent.
    let side = (seq as f64).sqrt() as usize;
    if side * side != seq {
        return Err(format!("latent token count {seq} is not a square"));
    }
    let cfg = VaeConfig::qwen_image_2_1();
    let ours = denormalize_latents_with(last, 1, side, side, &cfg.latents_mean, &cfg.latents_std)
        .map_err(|e| format!("{e}"))?;

    if ours.len() != want.data.len() {
        return Err(format!(
            "transform gives {} values, the oracle's decoder input has {} (shape {:?})",
            ours.len(),
            want.data.len(),
            want.shape
        ));
    }
    let r = rel_l2(&ours, &want.data);
    let max = ours
        .iter()
        .zip(&want.data)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("decoder input  rel-L2={r:.3e}  max_abs={max:.3e}");
    if r > 1e-5 {
        return Err(format!(
            "the pipeline's latent transform does not reproduce the decoder input (rel-L2 {r:.3e})"
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
