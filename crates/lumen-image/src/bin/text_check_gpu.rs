//! The text tower's GPU forward against the same oracle `text-check` compares —
//! and against the CPU reference on the same prompt.
//!
//! Three numbers per pair, so the check answers two questions at once:
//!   - GPU vs the real checkpoint's hidden states: does the GPU path encode
//!     the way the reference does?
//!   - GPU vs the CPU module on the same ids: does the GPU path compute the
//!     same function as the independent f32 implementation?
//!
//! The two paths round differently, so neither comparison is exact. The GPU
//! path rounds to bf16 where the reference does and lands closer to the oracle
//! (rel-L2 2.9e-02 on the `main` oracle tag) than the f32 CPU module
//! (7.3e-02); the two differ from each other by the same order (5.5e-02 to
//! 8.0e-02). The failure this is written to catch is a correctly-shaped,
//! finite tensor from a mislaid operand — the DiT had three such bugs — which
//! moves both numbers far past that rounding spread instead of by a fraction
//! of it.
//!
//! Usage: `text-check-gpu <lbi-dir> <oracle-dir> [tag] [drop_idx]`

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use lumen_image::cuda::text_gpu::TextGpu;
use lumen_image::npy;
use lumen_image::text_encoder::TextEncoder;
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

fn report(what: &str, got: &[f32], want: &[f32]) {
    println!(
        "  {what:<16} rel-L2={:.3e}  max_abs={:.3e}  cosine={:.8}",
        rel_l2(got, want),
        max_abs(got, want),
        cosine(got, want)
    );
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
        eprintln!("usage: text-check-gpu <lbi-dir> <oracle-dir> [tag] [drop_idx]");
        std::process::exit(2)
    };
    let lbi_dir = PathBuf::from(args.next().unwrap_or_else(usage));
    let oracle = PathBuf::from(args.next().unwrap_or_else(usage));
    let tag = args.next().unwrap_or_else(|| "smoke".to_string());
    // The pipeline strips the leading system tokens before the DiT sees them;
    // a fixed property of the prompt template, not of this run. Taken from the
    // argument list rather than the meta file, because the current dump schema
    // does not record it and defaulting to zero would compare the wrong rows.
    let drop_idx: usize = args
        .next()
        .map(|s| s.parse().map_err(|_| format!("bad drop_idx {s:?}")))
        .transpose()?
        .unwrap_or(14);

    let load = |name: &str| -> Result<npy::Npy, String> {
        let p = oracle.join(format!("{name}_{tag}.npy"));
        npy::load(&p).map_err(|e| format!("{}: {e}", p.display()))
    };

    // The oracle's own inputs, read before the device is touched so a bad
    // oracle directory fails without a 14 GiB upload in the way.
    let ids = load("prompt_input_ids")?;
    let mask = load("prompt_attention_mask")?;
    let want = load("encoder_hidden_states")?;

    // Only the unpadded tokens are encoded: left padding means the valid ids are
    // the suffix once the mask is applied. Same selection `text-check` makes.
    let valid: Vec<u32> = ids
        .data
        .iter()
        .zip(&mask.data)
        .filter(|(_, &m)| m != 0.0)
        .map(|(&t, _)| t as u32)
        .collect();
    println!(
        "tokens: {} of {} valid, drop_idx {drop_idx}",
        valid.len(),
        ids.data.len()
    );

    let dev = CudaDevice::new(0).map_err(|e| format!("no CUDA device: {e}"))?;
    println!(
        "device: {}  free {:.2} GiB of {:.2} GiB",
        dev.name().unwrap_or_else(|_| "<unknown>".into()),
        gib(dev.free_memory().map_err(|e| e.to_string())?),
        gib(dev.total_memory().map_err(|e| e.to_string())?),
    );

    let lbi = lbi_dir.join("text_encoder.lbi");

    // The CPU module first: it needs no device, and its output is the
    // independent f32 path the GPU result is compared against.
    let t0 = Instant::now();
    let cpu = TextEncoder::load(&lbi).map_err(|e| format!("{e}"))?;
    let cpu_hidden = cpu.forward(&valid).map_err(|e| format!("{e}"))?;
    println!(
        "cpu forward: [{}, {}] in {:.1?}",
        cpu_hidden.rows,
        cpu_hidden.cols,
        t0.elapsed()
    );

    println!("loading the text tower from {}", lbi.display());
    let free_before = dev.free_memory().map_err(|e| e.to_string())?;
    let t0 = Instant::now();
    let gpu = TextGpu::load(&lbi, &dev).map_err(|e| format!("{e}"))?;
    let free_after = dev.free_memory().map_err(|e| e.to_string())?;
    println!(
        "resident weights: {:.2} GiB (free {:.2} GiB after load, {:.1?} to load)",
        (free_before as f64 - free_after as f64) / (1u64 << 30) as f64,
        gib(free_after),
        t0.elapsed()
    );

    let free_pre_forward = dev.free_memory().map_err(|e| e.to_string())?;
    let t0 = Instant::now();
    let got = gpu.forward(&valid).map_err(|e| format!("{e}"))?;
    let elapsed = t0.elapsed();
    let free_post_forward = dev.free_memory().map_err(|e| e.to_string())?;
    println!(
        "activation peak during the forward: {:.2} GiB (free {:.2} GiB after)",
        (free_pre_forward as f64 - free_post_forward as f64) / (1u64 << 30) as f64,
        gib(free_post_forward),
    );

    // Both paths return every row; drop the system prefix the pipeline drops.
    let (rows, cols) = (got.rows, got.cols);
    if rows < drop_idx || cpu_hidden.rows != rows || cpu_hidden.cols != cols {
        return Err(format!(
            "gpu [{rows}, {cols}] vs cpu [{}, {}], drop_idx {drop_idx}",
            cpu_hidden.rows, cpu_hidden.cols
        ));
    }
    let gpu_kept: Vec<f32> = got.data[drop_idx * cols..].to_vec();
    let cpu_kept: Vec<f32> = cpu_hidden.data[drop_idx * cols..].to_vec();

    println!(
        "text tower gpu  {rows} rows, {} kept, {elapsed:.1?}",
        rows - drop_idx
    );
    report("gpu vs cpu", &gpu_kept, &cpu_kept);

    // The reference may carry more rows than the encoder produces only if the
    // oracle kept padding; report the shape difference rather than guessing.
    if gpu_kept.len() != want.data.len() {
        return Err(format!(
            "kept {} values ({rows} rows - {drop_idx} dropped x {cols}), reference has {} (shape {:?})",
            gpu_kept.len(),
            want.data.len(),
            want.shape
        ));
    }
    report("gpu vs oracle", &gpu_kept, &want.data);
    Ok(())
}
