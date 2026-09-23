//! The whole denoising trajectory against the oracle, not just its first step.
//!
//! `dit-check` compares one forward pass with the oracle's inputs injected. This
//! runs the *loop*: the oracle's own initial noise goes in, and every
//! intermediate latent the oracle recorded must come back out. That is what
//! catches an error that only shows up once the model is fed its own output.
//!
//! It also reports the first step at which the two trajectories diverge, which
//! is what makes a failure actionable rather than just "they differ".
//!
//! Usage: `e2e-check <lbi-dir> <oracle-dir> [tag]`

use std::path::PathBuf;
use std::process::ExitCode;

use lumen_image::dit::{Dit, DitForwardArgs};
use lumen_image::npy;
use lumen_image::scheduler::{SchedulerConfig, SigmaSchedule};
use lumen_image::tensor::Matrix;

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
    let usage = || {
        eprintln!("usage: e2e-check <lbi-dir> <oracle-dir> [tag]");
        std::process::exit(2)
    };
    let lbi_dir = PathBuf::from(args.next().unwrap_or_else(usage));
    let oracle = PathBuf::from(args.next().unwrap_or_else(usage));
    let tag = args.next().unwrap_or_else(|| "smoke".to_string());

    let load = |name: &str| -> Result<npy::Npy, String> {
        let p = oracle.join(format!("{name}_{tag}.npy"));
        npy::load(&p).map_err(|e| format!("{}: {e}", p.display()))
    };

    // Everything the loop needs, from the oracle's own run.
    let init = load("init_noise")?;
    let ehs = load("encoder_hidden_states")?;
    let img_mask = load("img_mask")?;
    let timesteps = load("dit_timestep")?;
    let want_latents = load("latents")?;
    let meta = std::fs::read_to_string(oracle.join(format!("meta_{tag}.json")))
        .map_err(|e| format!("meta: {e}"))?;
    let meta: serde_json::Value =
        serde_json::from_str(&meta).map_err(|e| format!("meta parse: {e}"))?;
    let steps = meta["steps"].as_u64().ok_or("meta has no steps")? as usize;
    let shapes: Vec<(u64, u64, u64)> = meta["img_shapes"][0]
        .as_array()
        .ok_or("meta has no img_shapes")?
        .iter()
        .map(|s| {
            let a = s.as_array().expect("shape entry");
            (
                a[0].as_u64().unwrap(),
                a[1].as_u64().unwrap(),
                a[2].as_u64().unwrap(),
            )
        })
        .collect();

    // Latent geometry: [steps, 1, seq, ch].
    if want_latents.shape.len() != 4 {
        return Err(format!(
            "latents shape {:?} is not [steps,1,seq,ch]",
            want_latents.shape
        ));
    }
    let (seq, ch) = (want_latents.shape[2], want_latents.shape[3]);
    if init.data.len() != seq * ch {
        return Err(format!(
            "init noise has {} values, expected {seq}x{ch}",
            init.data.len()
        ));
    }

    let use_gpu = std::env::args().any(|a| a == "--gpu");
    // The GPU path needs the cuda feature; without it `--gpu` is a clear error
    // rather than a silent fallback to CPU, which would make a run's identity
    // ambiguous.
    #[cfg(not(feature = "cuda"))]
    if use_gpu {
        return Err("built without the cuda feature; rebuild with --features cuda".to_string());
    }
    #[cfg(not(feature = "cuda"))]
    let _ = use_gpu;

    #[cfg(feature = "cuda")]
    let (dit_gpu, dit_cpu) = if use_gpu {
        let dev = lumen_runtime::cuda::ffi::CudaDevice::new(0).map_err(|e| format!("cuda: {e}"))?;
        let g = lumen_image::cuda::dit_gpu::DitGpu::load(&lbi_dir.join("transformer.lbi"), &dev)
            .map_err(|e| format!("gpu dit: {e}"))?;
        println!("running on GPU");
        (Some(g), None)
    } else {
        println!("running on CPU");
        (
            None,
            Some(Dit::load(&lbi_dir.join("transformer.lbi")).map_err(|e| format!("{e}"))?),
        )
    };
    #[cfg(not(feature = "cuda"))]
    let dit_cpu = Some(Dit::load(&lbi_dir.join("transformer.lbi")).map_err(|e| format!("{e}"))?);
    #[cfg(not(feature = "cuda"))]
    println!("running on CPU");
    let encoder = Matrix::new(ehs.shape[1], ehs.shape[2], ehs.data.clone());
    let mask: Vec<bool> = img_mask.data.iter().map(|&v| v != 0.0).collect();

    // The schedule the pipeline used.
    let seq_len: usize = shapes.iter().map(|(_, h, w)| (h * w) as usize).sum();
    let cfg = SchedulerConfig::qwen_image_2_1();
    let sched = SigmaSchedule::new(steps, seq_len, &cfg);

    let mut latents = init.data.clone();
    println!("denoising {steps} steps at {seq}x{ch}, {seq_len} image tokens");

    // The GPU forward is bf16, and its loop keeps bf16 latents as the
    // pipeline's does; the CPU forward is the f32 check path. The bf16 path
    // differs from the oracle by the reference's own kernel choices: 6.5e-4
    // after one step and 7.5e-3 at the end on the `main` oracle case, where an
    // image lands as close to the reference as the reference with another
    // attention kernel; a forward that keeps its intermediates in f32 ends at
    // 2.2e-2.
    #[cfg(feature = "cuda")]
    let bf16 = dit_gpu.is_some();
    #[cfg(not(feature = "cuda"))]
    let bf16 = false;
    let (step_bar, final_bar) = if bf16 { (1e-3, 1e-2) } else { (1e-5, 1e-3) };

    // The pipeline derives each step's timestep from its own schedule; the
    // oracle recorded what the reference passed.
    for (step, &want) in timesteps.data.iter().enumerate().take(steps) {
        let got = sched.model_timestep(step);
        if got.to_bits() != want.to_bits() {
            return Err(format!(
                "step {step}: the pipeline's timestep {got} is not the oracle's {want}"
            ));
        }
    }

    let mut first_divergence: Option<usize> = None;
    for step in 0..steps {
        let hidden = Matrix::new(seq, ch, latents.clone());
        #[cfg(feature = "cuda")]
        let out = if let Some(g) = &dit_gpu {
            g.forward(DitForwardArgs {
                hidden_states: &hidden,
                encoder_hidden_states: &encoder,
                timestep: timesteps.data[step],
                img_shapes: &shapes,
                img_mask: &mask,
            })
            .map_err(|e| format!("step {step}: {e}"))?
        } else {
            dit_cpu
                .as_ref()
                .expect("cpu dit when no device")
                .forward(DitForwardArgs {
                    hidden_states: &hidden,
                    encoder_hidden_states: &encoder,
                    timestep: timesteps.data[step],
                    img_shapes: &shapes,
                    img_mask: &mask,
                })
                .map_err(|e| format!("step {step}: {e}"))?
        };
        #[cfg(not(feature = "cuda"))]
        let out = dit_cpu
            .as_ref()
            .expect("cpu dit")
            .forward(DitForwardArgs {
                hidden_states: &hidden,
                encoder_hidden_states: &encoder,
                // The oracle records the timestep the pipeline passed, already
                // divided by 1000; the DiT API wants sigma and rescales itself.
                timestep: timesteps.data[step],
                img_shapes: &shapes,
                img_mask: &mask,
            })
            .map_err(|e| format!("step {step}: {e}"))?;

        // Step 0 returns the whole joint sequence; later steps return only the
        // target tokens, which the pipeline slices off.
        let pred = if out.rows > seq {
            out.data[(out.rows - seq) * ch..].to_vec()
        } else {
            out.data.clone()
        };

        let stepped = sched.step(step, &latents, &pred, bf16);
        latents = stepped;

        let want_off = step * seq * ch;
        let want = &want_latents.data[want_off..want_off + seq * ch];
        let r = rel_l2(&latents, want);
        if r > step_bar && first_divergence.is_none() {
            first_divergence = Some(step);
        }
        if step == 0 || step + 1 == steps || step % 10 == 0 || r > step_bar {
            println!("  step {step:3}  rel-L2 vs oracle = {r:.3e}");
        }
    }

    // The final latent is what the VAE would decode.
    let want_off = (steps - 1) * seq * ch;
    let want = &want_latents.data[want_off..want_off + seq * ch];
    let final_r = rel_l2(&latents, want);
    println!("final latent rel-L2 vs oracle = {final_r:.3e}");
    match first_divergence {
        Some(s) => println!("first step above {step_bar:e}: {s}"),
        None => println!("no step exceeded {step_bar:e} at any point"),
    }
    if final_r > final_bar {
        return Err(format!(
            "final latent rel-L2 {final_r:.3e} is above {final_bar:e}"
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
