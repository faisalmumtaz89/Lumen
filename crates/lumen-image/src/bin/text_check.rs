//! Run the text encoder over the oracle's own token ids and compare hidden states.
//!
//! The reference stores the hidden states the DiT consumes — after the leading
//! system tokens are dropped — so the comparison applies the same drop.
//!
//! Usage: `text-check <lbi-dir> <oracle-dir> [tag]`

use std::path::PathBuf;
use std::process::ExitCode;

use lumen_image::npy;
use lumen_image::text_encoder::TextEncoder;

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

fn max_abs(got: &[f32], want: &[f32]) -> f32 {
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0f32, f32::max)
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let usage = || {
        eprintln!("usage: text-check <lbi-dir> <oracle-dir> [tag]");
        std::process::exit(2)
    };
    let lbi_dir = PathBuf::from(args.next().unwrap_or_else(usage));
    let oracle = PathBuf::from(args.next().unwrap_or_else(usage));
    let tag = args.next().unwrap_or_else(|| "smoke".to_string());

    let load = |name: &str| -> Result<npy::Npy, String> {
        let p = oracle.join(format!("{name}_{tag}.npy"));
        npy::load(&p).map_err(|e| format!("{}: {e}", p.display()))
    };

    let ids = load("prompt_input_ids")?;
    let mask = load("prompt_attention_mask")?;
    let want = load("encoder_hidden_states")?;

    // The pipeline strips the leading system tokens before the DiT sees them.
    // It is a fixed property of the prompt template, not of this run: the
    // rendered system message tokenizes to 14 tokens. Taken from the argument
    // list rather than the meta file, because the current dump schema does not
    // record it and defaulting to zero would silently compare the wrong rows.
    let drop_idx: usize = args
        .next()
        .map(|s| s.parse().map_err(|_| format!("bad drop_idx {s:?}")))
        .transpose()?
        .unwrap_or(14);

    // Only the unpadded tokens are encoded: left padding means the valid ids are
    // the suffix once the mask is applied.
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

    let enc = TextEncoder::load(&lbi_dir.join("text_encoder.lbi")).map_err(|e| format!("{e}"))?;
    let hidden = enc.forward(&valid).map_err(|e| format!("{e}"))?;

    // Hidden states are [seq, 4096]; drop the system prefix the pipeline drops.
    let seq = hidden.rows;
    if seq < drop_idx {
        return Err(format!("encoded {seq} tokens but drop_idx is {drop_idx}"));
    }
    let kept: Vec<f32> = hidden.data[drop_idx * hidden.cols..].to_vec();
    let want_n = want.data.len();

    // The reference may carry more rows than the encoder produces only if the
    // oracle kept padding; report the shape difference rather than guessing.
    if kept.len() != want_n {
        return Err(format!(
            "kept {} values ({seq} rows - {drop_idx} dropped x {}), reference has {want_n} (shape {:?})",
            kept.len(),
            hidden.cols,
            want.shape
        ));
    }
    println!(
        "text encoder  rel-L2={:.3e}  max_abs={:.3e}  cosine={:.8}",
        rel_l2(&kept, &want.data),
        max_abs(&kept, &want.data),
        cosine(&kept, &want.data)
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
