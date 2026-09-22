//! Convert a Qwen-Image-2.1 checkpoint directory into three `.lbi` files.

use std::path::PathBuf;
use std::process::ExitCode;

use lumen_image::convert::{convert_component, convert_single_file, has_shard_index, single_shard};
use lumen_image::lbi::LbiFile;

fn usage() -> ! {
    eprintln!(
        "usage: lbi-convert <checkpoint-dir> <out-dir>\n\
         \n\
         Reads <checkpoint-dir>/{{transformer,vae,text_encoder}} and writes\n\
         <out-dir>/{{transformer,vae,text_encoder}}.lbi, then re-opens each\n\
         output and checks it reads back."
    );
    std::process::exit(2)
}

fn component_config(dir: &std::path::Path) -> Result<serde_json::Value, String> {
    let p = dir.join("config.json");
    let bytes = std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", p.display()))
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let ckpt = PathBuf::from(args.next().unwrap_or_else(|| usage()));
    let out = PathBuf::from(args.next().unwrap_or_else(|| usage()));
    std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;

    for component in ["transformer", "vae", "text_encoder"] {
        let comp_dir = ckpt.join(component);
        let config = component_config(&comp_dir)?;
        let target = out.join(format!("{component}.lbi"));

        // Sharded checkpoints carry an index; single-file ones do not.
        let report = if has_shard_index(&comp_dir) {
            convert_component(&ckpt, component, &target, config)
                .map_err(|e| format!("convert {component}: {e}"))?
        } else {
            let shard = single_shard(&comp_dir).map_err(|e| format!("convert {component}: {e}"))?;
            convert_single_file(&shard, component, &target, config)
                .map_err(|e| format!("convert {component}: {e}"))?
        };

        // Re-open what was written: a file that fails to read is not a result.
        let f = LbiFile::open(&target).map_err(|e| format!("reopen {component}: {e}"))?;
        if f.len() != report.tensor_count {
            return Err(format!(
                "{component}: wrote {} tensors but read back {}",
                report.tensor_count,
                f.len()
            ));
        }
        let bytes = std::fs::metadata(&target)
            .map_err(|e| format!("{component}: {e}"))?
            .len();
        let dtypes: Vec<String> = report
            .dtypes
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect();
        println!(
            "{component:14} {:4} tensors  {}  {:.2} GiB written, {:.2} GiB blob  ({})",
            report.tensor_count,
            dtypes.join(" "),
            bytes as f64 / (1u64 << 30) as f64,
            report.total_bytes as f64 / (1u64 << 30) as f64,
            target.display()
        );
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
