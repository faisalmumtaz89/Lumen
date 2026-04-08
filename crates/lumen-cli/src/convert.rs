use crate::help::print_convert_usage;

use lumen_format::quantization::QuantScheme;
use std::path::Path;

pub(crate) fn convert_cmd(args: &[String]) {
    use lumen_convert::convert::{convert_gguf_to_lbc, ConvertOptions};

    let mut input_path: Option<String> = None;
    let mut output_path: Option<String> = None;
    let mut dequantize = false;
    let mut requant: Option<QuantScheme> = None;

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
    };

    println!("Converting: {input_path} -> {output_path}");

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
