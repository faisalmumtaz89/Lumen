//! Convert a Qwen-Image-2.1 checkpoint directory into three `.lbi` files.

use std::path::PathBuf;
use std::process::ExitCode;

use lumen_image::convert::convert_checkpoint;

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

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let ckpt = PathBuf::from(args.next().unwrap_or_else(|| usage()));
    let out = PathBuf::from(args.next().unwrap_or_else(|| usage()));
    for report in convert_checkpoint(&ckpt, &out).map_err(|e| e.to_string())? {
        let target = out.join(format!("{}.lbi", report.component));
        let bytes = std::fs::metadata(&target)
            .map_err(|e| format!("{}: {e}", report.component))?
            .len();
        let dtypes: Vec<String> = report
            .dtypes
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect();
        println!(
            "{:14} {:4} tensors  {}  {:.2} GiB written, {:.2} GiB blob  ({})",
            report.component,
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
