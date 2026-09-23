//! Bisect the GPU DiT against the CPU one by block count.
//!
//! `dit-check-gpu` says whether the whole forward agrees; when it does not, the
//! next question is *where* it stops agreeing. Both models take a
//! `DitConfig`, and both loaders honour `num_layers`, so truncating the block
//! stack isolates the arithmetic around it:
//!
//!   `--layers 0`  the text projection, the joint assembly, the timestep
//!                 embedding, the modulation projection, `norm_out` and
//!                 `proj_out` — everything except the blocks
//!   `--layers 1`  adds exactly one block
//!   ...
//!
//! The GPU forward is bf16, as the reference runs it, and the CPU forward f32,
//! so the two differ by bf16 rounding that grows with depth: on these inputs
//! rel-L2 3.0e-3 with no blocks to 5.5e-2 with all 32. A mislaid operand moves
//! the output by order one (a dropped `1 +` in the AdaLN scale reads 1.0 to
//! 2.3 from the first block on), so a count above 0.2 fails, and the first
//! count whose rel-L2 jumps a hundredfold is the block that introduces the
//! error; if `--layers 0` already fails, no attention or MLP code is involved
//! at all. This needs no oracle: it compares the two implementations on the
//! same synthetic inputs, so it works wherever the `.lbi` does.
//!
//! Usage: `dit-stage-check <lbi-dir> [--layers N]`

use std::path::PathBuf;
use std::process::ExitCode;

use lumen_image::cuda::dit_gpu::DitGpu;
use lumen_image::dit::{Dit, DitConfig, DitForwardArgs};
use lumen_image::tensor::Matrix;
use lumen_image::LbiFile;
use lumen_runtime::cuda::ffi::CudaDevice;

/// The rel-L2 above which a layer count fails; see the module doc.
const BAR: f32 = 0.2;

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

/// Deterministic inputs, so a run is reproducible without an oracle dump.
fn pseudo(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

fn main() -> ExitCode {
    match run() {
        Ok(failures) => {
            if failures > 0 {
                ExitCode::FAILURE
            } else {
                ExitCode::SUCCESS
            }
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<usize, String> {
    let mut args = std::env::args().skip(1);
    let usage = || {
        eprintln!("usage: dit-stage-check <lbi-dir> [--layers N]");
        std::process::exit(2)
    };
    let lbi_dir = PathBuf::from(args.next().unwrap_or_else(usage));
    let mut layers: Vec<usize> = vec![0, 1, 2, 4, 8, 16, 32];
    let mut rest: Vec<String> = args.collect();
    if let Some(i) = rest.iter().position(|a| a == "--layers") {
        if i + 1 >= rest.len() {
            usage();
        }
        layers = vec![rest[i + 1].parse().map_err(|e| format!("--layers: {e}"))?];
        rest.drain(i..i + 2);
    }
    if !rest.is_empty() {
        usage();
    }

    // One 1x8x8 target image: 64 latent tokens, so 16 slots on top of the text.
    // Small enough that a wrong block shows up immediately and cheap enough to
    // run seven times.
    let text_seq = 4usize;
    let (f, h, w) = (1u64, 8u64, 8u64);
    let hidden_states = Matrix::new((h * w) as usize, 64, pseudo((h * w) as usize * 64, 1));
    let encoder = Matrix::new(text_seq, 4096, pseudo(text_seq * 4096, 2));
    let mut img_mask = vec![false; text_seq];
    img_mask.extend(std::iter::repeat(true).take(((h * w) / 4) as usize));
    let shapes = [(f, h, w)];
    let args = DitForwardArgs {
        hidden_states: &hidden_states,
        encoder_hidden_states: &encoder,
        timestep: 0.7,
        img_shapes: &shapes,
        img_mask: &img_mask,
    };

    let dev = CudaDevice::new(0).map_err(|e| format!("no CUDA device: {e}"))?;
    let lbi = lbi_dir.join("transformer.lbi");

    let mut failures = 0usize;
    let mut previous: Option<f32> = None;
    for n in layers {
        let cfg = DitConfig {
            num_layers: n,
            ..DitConfig::qwen_image_2_1()
        };
        let cpu = Dit::load_with(&lbi, cfg.clone()).map_err(|e| format!("cpu load {n}: {e}"))?;
        let want = cpu
            .forward(args)
            .map_err(|e| format!("cpu forward {n}: {e}"))?;

        // Scoped so the device memory is released before the next load.
        let got = {
            let file = LbiFile::open(&lbi).map_err(|e| format!("open {}: {e}", lbi.display()))?;
            let gpu =
                DitGpu::load_with(&file, &dev, cfg).map_err(|e| format!("gpu load {n}: {e}"))?;
            gpu.forward(args)
                .map_err(|e| format!("gpu forward {n}: {e}"))?
        };

        if got.data.len() != want.data.len() {
            eprintln!(
                "layers {n:2}: shape {:?} vs {:?}",
                vec![got.rows, got.cols],
                vec![want.rows, want.cols]
            );
            failures += 1;
            continue;
        }
        let r = rel_l2(&got.data, &want.data);
        let m = max_abs(&got.data, &want.data);
        // The first count that leaves the rounding floor is the block that
        // introduces the error, so the jump is what the reader looks for.
        let jump = previous.is_some_and(|p| r > (p * 100.0).max(1e-5));
        println!(
            "layers {n:2}  rel-L2={r:.3e}  max_abs={m:.3e}{}",
            if jump { "   <- error appears here" } else { "" }
        );
        if r > BAR {
            failures += 1;
        }
        previous = Some(r);
    }

    if failures > 0 {
        eprintln!("{failures} layer count(s) disagreed above {BAR}");
    } else {
        println!("every truncated forward matched the CPU reference");
    }
    Ok(failures)
}
