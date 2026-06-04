use crate::help::print_convert_usage;

use lumen_convert::convert::ConvertTarget;
use lumen_format::quantization::QuantScheme;
use std::path::Path;

pub(crate) fn convert_cmd(args: &[String]) {
    use lumen_convert::convert::{convert_gguf_to_lbc, ConvertOptions};

    let mut input_path: Option<String> = None;
    let mut output_path: Option<String> = None;
    let mut dequantize = false;
    let mut requant: Option<QuantScheme> = None;
    // Default convert target follows the host OS: on macOS we want Metal
    // K-quant upcast so any Q6_K layer tensor in the source GGUF lands as
    // Q8_0 in the LBC (Metal backend has no K-quant kernels). On Linux
    // (CUDA host) we keep the legacy Generic behaviour.
    let mut target: ConvertTarget = default_target_for_host();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input" | "-i" => {
                i += 1;
                input_path = Some(args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --input requires a path");
                    std::process::exit(1);
                }).clone());
            }
            "--output" | "-o" => {
                i += 1;
                output_path = Some(args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --output requires a path");
                    std::process::exit(1);
                }).clone());
            }
            "--dequantize" => {
                dequantize = true;
            }
            "--requant" => {
                i += 1;
                let val = args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --requant requires a value (e.g. q4_0)");
                    std::process::exit(1);
                });
                requant = match val.to_lowercase().as_str() {
                    "q4_0" | "q4" => Some(QuantScheme::Q4_0),
                    "q8_0" | "q8" => Some(QuantScheme::Q8_0),
                    other => {
                        eprintln!("Error: unsupported requant target: {other} (supported: q4_0, q8_0)");
                        std::process::exit(1);
                    }
                };
            }
            "--target" => {
                i += 1;
                let val = args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --target requires a value (metal | generic)");
                    std::process::exit(1);
                });
                target = match val.to_lowercase().as_str() {
                    "metal" => ConvertTarget::Metal,
                    "generic" | "cuda" => ConvertTarget::Generic,
                    other => {
                        eprintln!("Error: unsupported target: {other} (supported: metal, generic)");
                        std::process::exit(1);
                    }
                };
            }
            "--help" | "-h" => {
                print_convert_usage();
                return;
            }
            other => {
                eprintln!("Unknown option: {other}");
                print_convert_usage();
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let input_path = input_path.unwrap_or_else(|| {
        eprintln!("Error: --input is required");
        print_convert_usage();
        std::process::exit(1);
    });

    let output_path = output_path.unwrap_or_else(|| {
        // Default: same name with .lbc extension
        let p = Path::new(&input_path);
        p.with_extension("lbc")
            .to_string_lossy()
            .into_owned()
    });

    let input = Path::new(&input_path);
    if !input.exists() {
        eprintln!("Error: input file not found: {input_path}");
        std::process::exit(1);
    }

    let opts = ConvertOptions {
        alignment: 128 * 1024,
        dequantize_to_f32: dequantize,
        requant_to: requant,
        target,
    };

    println!("Converting: {input_path} -> {output_path} (target={target:?})");

    match convert_gguf_to_lbc(input, Path::new(&output_path), &opts) {
        Ok(stats) => {
            println!("{stats}");
            println!("Done.");
        }
        Err(e) => {
            eprintln!("Conversion error: {e}");
            std::process::exit(1);
        }
    }
}

/// Pick the default `ConvertTarget` for the host the converter is running on.
///
/// On macOS the only available GPU backend is Metal, and Metal has no
/// K-quant dispatch kernels. So we default to `Metal` to ensure any K-quant
/// layer tensor (e.g. the Q6_K `attn_q` in the Q4 MoE-30B GGUF) gets upcast to Q8_0 at convert
/// time -- matches what CUDA's K-quant dequant kernels do implicitly.
///
/// On Linux/Windows the host is presumed to be CUDA-capable, so we keep
/// the legacy `Generic` behaviour (K-quant layer tensors pass through).
pub(crate) fn default_target_for_host() -> ConvertTarget {
    #[cfg(target_os = "macos")]
    { ConvertTarget::Metal }
    #[cfg(not(target_os = "macos"))]
    { ConvertTarget::Generic }
}
