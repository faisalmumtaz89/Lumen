//! Generate an image end to end: prompt in, PNG out.
//!
//! This is the pipeline driver's front door. It runs the whole sequence —
//! tokenize, encode, denoise, decode, encode PNG — and writes the result, so a
//! check can compare it against the reference image rather than against an
//! intermediate tensor.
//!
//! Usage: `generate <lbi-dir> <checkpoint-dir> <out.png> [options]`

use std::process::ExitCode;

use lumen_image::pipeline::{generate_cpu, GenerationRequest, PipelinePaths};
use lumen_image::png;

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
        eprintln!(
            "usage: generate <lbi-dir> <checkpoint-dir> <out.png> [--prompt P] [--size N] [--steps N] [--seed N] [--init-latents FILE.npy] [--gpu]"
        );
        std::process::exit(2)
    };
    let lbi = std::path::PathBuf::from(args.next().unwrap_or_else(usage));
    let ckpt = std::path::PathBuf::from(args.next().unwrap_or_else(usage));
    let out = std::path::PathBuf::from(args.next().unwrap_or_else(usage));

    let mut prompt = "A red apple on a wooden table, studio lighting".to_string();
    let mut size = 512usize;
    let mut steps = 8usize;
    let mut seed = 42u64;
    let mut gpu = false;
    let mut init_path: Option<std::path::PathBuf> = None;
    while let Some(flag) = args.next() {
        let mut val = || args.next().unwrap_or_else(&usage);
        match flag.as_str() {
            "--prompt" => prompt = val(),
            "--size" => size = val().parse().map_err(|e| format!("--size: {e}"))?,
            "--steps" => steps = val().parse().map_err(|e| format!("--steps: {e}"))?,
            "--seed" => seed = val().parse().map_err(|e| format!("--seed: {e}"))?,
            "--init-latents" => init_path = Some(val().into()),
            "--gpu" => gpu = true,
            other => return Err(format!("unknown flag {other}")),
        }
    }

    let paths = PipelinePaths::from_roots(&lbi, &ckpt);
    // The reference's own noise, when asked for: see `GenerationRequest::init_latents`.
    let init = match &init_path {
        Some(p) => {
            let npy = lumen_image::npy::load(p)
                .map_err(|e| format!("--init-latents {}: {e}", p.display()))?;
            // Token-major `[seq, channels]`, with any leading unit axes: a
            // channel-major dump has the same element count and would be
            // read as permuted noise, silently, which defeats the flag.
            let seq = lumen_image::pipeline::latent_side(size).pow(2);
            let channels = lumen_image::dit::DitConfig::qwen_image_2_1().in_channels;
            let trailing: Vec<usize> = npy.shape.iter().copied().skip_while(|&d| d == 1).collect();
            if trailing != [seq, channels] {
                return Err(format!(
                    "--init-latents {}: shape {:?} is not [{seq}, {channels}] for {size}x{size}",
                    p.display(),
                    npy.shape
                ));
            }
            Some(npy.data)
        }
        None => None,
    };
    let req = GenerationRequest {
        prompt: &prompt,
        height: size,
        width: size,
        steps,
        seed,
        init_latents: init.as_deref(),
    };
    println!("generating {size}x{size} in {steps} steps: {prompt:?}");
    let started = std::time::Instant::now();
    let mut last = 0usize;
    let mut report = |done: usize, total: usize| {
        if done > 0 && (done == total || done % 5 == 0) {
            println!("  step {done}/{total}");
        }
        last = done;
        std::ops::ControlFlow::Continue(())
    };
    let image = if gpu {
        #[cfg(feature = "cuda")]
        {
            println!("  running on the GPU");
            lumen_image::pipeline::generate_gpu(&paths, &req, &mut report)
                .map_err(|e| format!("{e}"))?
        }
        #[cfg(not(feature = "cuda"))]
        {
            return Err("built without the cuda feature".to_string());
        }
    } else {
        generate_cpu(&paths, &req, &mut report).map_err(|e| format!("{e}"))?
    };
    println!(
        "  {} step(s) in {:.1}s",
        last,
        started.elapsed().as_secs_f32()
    );

    let png = png::encode(&image);
    std::fs::write(&out, &png).map_err(|e| format!("{}: {e}", out.display()))?;
    println!(
        "wrote {} ({}x{}, {} bytes)",
        out.display(),
        image.width,
        image.height,
        png.len()
    );
    Ok(())
}
