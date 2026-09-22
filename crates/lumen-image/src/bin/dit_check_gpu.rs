//! The DiT's GPU forward against the same reference tensor `dit-check` compares.
//!
//! Same inputs, same oracle dump, same three numbers: this is the GPU path on
//! the identical per-stage check, so the two can be read side by side. It also
//! reports the device memory the model occupies, so the residency claim is a
//! measurement rather than an estimate.
//!
//! Usage: `dit-check-gpu <lbi-dir> <oracle-dir> [tag]`

use std::path::PathBuf;
use std::process::ExitCode;

use lumen_image::cuda::dit_gpu::DitGpu;
use lumen_image::dit::DitForwardArgs;
use lumen_image::npy;
use lumen_image::tensor::Matrix;
use lumen_runtime::cuda::ffi::CudaDevice;

/// Relative L2 error between two equal-length slices.
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

fn cosine(got: &[f32], want: &[f32]) -> f32 {
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for (g, w) in got.iter().zip(want) {
        let g = *g as f64;
        let w = *w as f64;
        dot += g * w;
        na += g * g;
        nb += w * w;
    }
    (dot / (na.sqrt() * nb.sqrt()).max(1e-30)) as f32
}

fn gib(bytes: usize) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
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
        eprintln!("usage: dit-check-gpu <lbi-dir> <oracle-dir> [tag]");
        std::process::exit(2)
    };
    let lbi_dir = PathBuf::from(args.next().unwrap_or_else(usage));
    let oracle = PathBuf::from(args.next().unwrap_or_else(usage));
    let tag = args.next().unwrap_or_else(|| "smoke".to_string());

    let load = |name: &str| -> Result<npy::Npy, String> {
        let p: PathBuf = oracle.join(format!("{name}_{tag}.npy"));
        npy::load(&p).map_err(|e| format!("{}: {e}", p.display()))
    };

    // The oracle's own inputs, read before the device is touched so a bad
    // oracle directory fails without a 13 GiB upload in the way.
    let hs = load("dit_inputs")?;
    let ehs = load("encoder_hidden_states")?;
    let img_mask = load("img_mask")?;
    let timesteps = load("dit_timestep")?;
    let want = load("dit_out_prefill")?;
    let meta = std::fs::read_to_string(oracle.join(format!("meta_{tag}.json")))
        .map_err(|e| format!("meta: {e}"))?;
    let meta: serde_json::Value =
        serde_json::from_str(&meta).map_err(|e| format!("meta parse: {e}"))?;

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

    // `dit_inputs` is [steps, batch, seq, channels]; step 0 is the prefill, and
    // for text-to-image its input is the packed target latents alone.
    if hs.shape.len() != 4 || hs.shape[1] != 1 {
        return Err(format!(
            "dit_inputs shape {:?} is not [steps, 1, seq, ch]",
            hs.shape
        ));
    }
    let (seq, ch) = (hs.shape[2], hs.shape[3]);
    let hidden = Matrix::new(seq, ch, hs.data[..seq * ch].to_vec());
    if ehs.shape.len() != 3 || ehs.shape[0] != 1 {
        return Err(format!(
            "encoder_hidden_states shape {:?} is not [1, seq, ch]",
            ehs.shape
        ));
    }
    let encoder = Matrix::new(ehs.shape[1], ehs.shape[2], ehs.data.clone());
    let mask: Vec<bool> = img_mask.data.iter().map(|&v| v != 0.0).collect();

    // Step 0 runs with the whole joint sequence and its timestep. The oracle
    // stores the timestep already divided by 1000 by the pipeline; the DiT API
    // takes sigma and scales by 1000 itself, so undo the pipeline's division.
    let t = timesteps.data[0] * 1000.0;

    let dev = CudaDevice::new(0).map_err(|e| format!("no CUDA device: {e}"))?;
    println!(
        "device: {}  free {:.2} GiB of {:.2} GiB",
        dev.name().unwrap_or_else(|_| "<unknown>".into()),
        gib(dev.free_memory().map_err(|e| e.to_string())?),
        gib(dev.total_memory().map_err(|e| e.to_string())?),
    );

    println!("loading the transformer from {}", lbi_dir.display());
    let free_before = dev.free_memory().map_err(|e| e.to_string())?;
    let dit = DitGpu::load(&lbi_dir.join("transformer.lbi"), &dev).map_err(|e| format!("{e}"))?;
    let free_after = dev.free_memory().map_err(|e| e.to_string())?;
    println!(
        "resident weights: {:.2} GiB (free {:.2} GiB after load)",
        (free_before as f64 - free_after as f64) / (1u64 << 30) as f64,
        gib(free_after),
    );

    println!(
        "forward: hidden [{seq} x {ch}] encoder [{:?}] mask {} slots images {:?} t={t}",
        vec![ehs.shape[1], ehs.shape[2]],
        mask.len(),
        shapes
    );
    let free_pre_forward = dev.free_memory().map_err(|e| e.to_string())?;
    let got = dit
        .forward(DitForwardArgs {
            hidden_states: &hidden,
            encoder_hidden_states: &encoder,
            timestep: t / 1000.0,
            img_shapes: &shapes,
            img_mask: &mask,
        })
        .map_err(|e| format!("{e}"))?;
    let free_post_forward = dev.free_memory().map_err(|e| e.to_string())?;
    println!(
        "activation peak during the forward: {:.2} GiB (free {:.2} GiB after)",
        (free_pre_forward as f64 - free_post_forward as f64) / (1u64 << 30) as f64,
        gib(free_post_forward),
    );

    let want_n = want.data.len();
    if got.data.len() != want_n {
        return Err(format!(
            "output has {} values, the reference has {want_n} (shape {:?} vs {:?})",
            got.data.len(),
            vec![got.rows, got.cols],
            want.shape
        ));
    }
    println!(
        "dit prefill gpu  rel-L2={:.3e}  max_abs={:.3e}  cosine={:.8}",
        rel_l2(&got.data, &want.data),
        max_abs(&got.data, &want.data),
        cosine(&got.data, &want.data)
    );
    Ok(())
}
