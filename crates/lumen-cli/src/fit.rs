//! A text model the device it would run on cannot run is refused before the
//! first byte, each rule from the facts it needs alone: a quantization whose
//! CUDA route needs a newer compute capability than the device has (the
//! K-quants need 8.0); one whose weights alone are bigger than the memory it
//! would run in, the CUDA device's or, for a CPU run, the memory this machine
//! has available; and a BF16 model on a CUDA device that holds BF16 layer
//! weights as F32, whose weights take at least [`BF16_AS_F32`]'s share of its
//! download there. The CUDA backend holds the converted weights on the device
//! (all but one norm vector per dense layer, at most 1.3 MB per model), a CPU
//! backend reads all of them from host memory, and each registry
//! quantization's converted weights differ from its download by the amount
//! [`CONVERSION`] records (the MTP layer the converter drops, the tensors it
//! re-quantizes, and the file's metadata), so wherever a model converts as it
//! does for CUDA (everywhere but macOS, whose conversion targets Metal) a
//! download is held to its quantization's own amount, and any other download
//! to the most conversion removes from any of them, [`CONVERSION_SHRINK`].
//! The tests below hold the censuses. These are bounds on the weights only:
//! what serving needs beside them is checked when the model loads, and once it
//! is converted
//! `lumen_runtime::compute::check_cpu_memory` checks a CPU run's whole need.
#![cfg_attr(not(all(feature = "download", feature = "cuda")), allow(dead_code))]

use std::collections::BTreeMap;

use lumen_format::quantization::QuantScheme;
use lumen_format::serving_rules::{self, ComputeCapability};

use crate::registry::ModelEntry;

/// Download minus converted weight bytes of every registry quantization, in
/// thousandths of a GiB, for the download of the bytes given: each file's real
/// header (metadata, tokenizer, tensor table) at its full length, converted
/// for the CUDA target, and the weights of the result summed (every layer
/// tensor, the embedding, the final norm and the output head).
const CONVERSION: [(&str, &str, u64, i64); 11] = [
    ("qwen3-5-9b", "Q8_0", 9_804_541_984, 268),
    ("qwen3-5-9b", "Q4_0", 5_741_391_904, -43),
    ("qwen3-5-9b", "BF16", 18_407_321_344, 820),
    ("qwen3-5-moe-35b-a3b", "Q8_0", 37_812_647_744, 227),
    ("qwen3-5-moe-35b-a3b", "Q4_0", 20_836_243_264, 106),
    ("qwen3-5-moe-35b-a3b", "BF16", 71_065_941_760, 1287),
    ("qwen3-8-27b", "Q8_0", 29_047_086_048, 431),
    ("qwen3-8-27b", "Q4_0", 16_056_478_688, -451),
    ("qwen3-8-27b", "Q4_K_M", 17_442_399_968, 233),
    ("qwen3-8-27b", "Q5_K_M", 20_923_877_088, 233),
    ("qwen3-8-27b", "BF16", 54_657_735_616, 2140),
];

/// What a [`CONVERSION`] row is widened by, in thousandths of a GiB, so the
/// weights it bounds are never overstated: the half thousandth its rounding
/// hides and the norm vectors the CUDA backend keeps on the host.
const CONVERSION_MARGIN: i64 = 10;

/// The most a registry model's converted weights fall short of its download,
/// 2.2 GiB: the [`CONVERSION`] maximum, 2.140 GiB for qwen3.8-27b BF16,
/// rounded up to a tenth of a GiB.
const CONVERSION_SHRINK: u64 = 2_362_232_013;

/// Whether a download of `size` bytes of `quant` of `entry` may have weights
/// that fit `memory` bytes on `target`: not when, short by the most its
/// conversion may remove, it is still bigger.
fn may_fit(entry: &ModelEntry, quant: &str, target: Target, size: u64, memory: u64) -> bool {
    let converts_for_cuda = match target {
        Target::Cuda(_) => true,
        Target::Host => !cfg!(target_os = "macos"),
    };
    let row = CONVERSION
        .iter()
        .filter(|_| converts_for_cuda)
        .find(|(key, q, measured, _)| *key == entry.key && *q == quant && *measured == size);
    let least = match row {
        Some(&(_, _, _, shrink)) => {
            i128::from(size) - i128::from(shrink + CONVERSION_MARGIN) * (1 << 30) / 1000
        }
        None => i128::from(size.saturating_sub(CONVERSION_SHRINK)),
    };
    least <= i128::from(memory)
}

/// Where a model would run: a CUDA device, or this machine's memory for a CPU
/// backend.
#[derive(Clone, Copy)]
pub(crate) enum Target {
    Cuda(usize),
    Host,
}

impl Target {
    /// Who has the memory, as the refusal names it.
    fn holder(self) -> String {
        match self {
            Target::Cuda(ordinal) => format!("CUDA device {ordinal} has"),
            Target::Host => "this machine has available".into(),
        }
    }

    fn there(self) -> &'static str {
        match self {
            Target::Cuda(_) => "on that GPU",
            Target::Host => "on this machine",
        }
    }

    /// The `lumen run` flag that runs on this target, when it is not the
    /// default device.
    fn flag(self) -> String {
        match self {
            Target::Cuda(0) => String::new(),
            Target::Cuda(ordinal) => format!(" --cuda-device {ordinal}"),
            Target::Host => " --simd".into(),
        }
    }

    /// What would run the model instead when none of its quantizations can.
    fn instead(self) -> &'static str {
        match self {
            Target::Cuda(_) => "a smaller model or a larger GPU is needed",
            Target::Host => "a smaller model or more memory is needed",
        }
    }
}

/// The least each BF16 registry model's weights take on a CUDA device that
/// holds BF16 layer weights as F32 (compute capability below 9.0), in
/// hundredths of its download, rounded down, without the MTP layer the
/// converter drops. A dense model: every 2-D BF16 layer tensor at 4 bytes an
/// element, except `ssm_alpha`, `ssm_beta` and `ssm_out`, which the converter
/// re-quantizes to Q8_0, and everything else at its stored size (the
/// embedding and output head stay BF16, norms F32). The MoE: each layer as it
/// is stored, uploaded whole for its experts, which are read as BF16 in place,
/// and its attention projections again at 4 bytes an element beside it.
/// Measured from each file's real header for the download of the bytes given,
/// and used only when the download is that size.
const BF16_AS_F32: [(&str, u64, u64); 3] = [
    ("qwen3-5-9b", 18_407_321_344, 165),
    ("qwen3-5-moe-35b-a3b", 71_065_941_760, 103),
    ("qwen3-8-27b", 54_657_735_616, 179),
];

/// The scheme a registry quantization's weights mostly use, which decides the
/// CUDA route they need.
fn primary_scheme(quant: &str) -> Option<QuantScheme> {
    match quant {
        "BF16" => Some(QuantScheme::Bf16),
        "Q8_0" => Some(QuantScheme::Q8_0),
        "Q4_0" => Some(QuantScheme::Q4_0),
        "Q4_K_M" => Some(QuantScheme::Q4_K),
        "Q5_K_M" => Some(QuantScheme::Q5_K),
        _ => None,
    }
}

/// How many of a quantization's files are downloaded already.
#[derive(Clone, Copy)]
pub(crate) enum Held {
    Nothing,
    Part,
    All,
}

/// What is known of where a model would run before anything is downloaded:
/// its memory, and for a CUDA device its compute capability.
#[derive(Clone, Copy)]
pub(crate) struct Device {
    pub(crate) memory: Option<u64>,
    pub(crate) compute_capability: Option<ComputeCapability>,
}

/// Why `quant` cannot run on the device, each from the facts it needs alone.
enum Ruling {
    /// Its CUDA route needs a newer compute capability than the device has.
    Capability {
        need: ComputeCapability,
        have: ComputeCapability,
    },
    /// Its weights alone are more than the device's memory.
    Weights { size: u64, memory: u64 },
    /// The device holds BF16 layer weights as F32, at which they take at least
    /// `at_least`, more than its memory.
    BF16AsF32 {
        size: u64,
        at_least: u64,
        memory: u64,
        have: ComputeCapability,
    },
}

fn ruling(
    entry: &ModelEntry,
    quant: &str,
    target: Target,
    device: Device,
    size: Option<u64>,
) -> Option<Ruling> {
    let cuda_cc = match target {
        Target::Cuda(_) => device.compute_capability,
        Target::Host => None,
    };
    if let (Some(have), Some(need)) = (
        cuda_cc,
        primary_scheme(quant).and_then(serving_rules::cuda_minimum_compute_capability),
    ) {
        if have < need {
            return Some(Ruling::Capability { need, have });
        }
    }
    let (size, memory) = (size?, device.memory?);
    if !may_fit(entry, quant, target, size, memory) {
        return Some(Ruling::Weights { size, memory });
    }
    let have = cuda_cc?;
    if quant != "BF16" || !serving_rules::cuda_bf16_layers_as_f32(have) {
        return None;
    }
    let &(_, _, hundredths) = BF16_AS_F32
        .iter()
        .find(|(key, measured, _)| *key == entry.key && *measured == size)?;
    let at_least = (u128::from(size) * u128::from(hundredths) / 100) as u64;
    (at_least > memory).then_some(Ruling::BF16AsF32 {
        size,
        at_least,
        memory,
        have,
    })
}

/// The refusal for `quant` of `entry`, which the user named `name`, when it
/// cannot run on `device`, `target`'s; `None` when it is not ruled out.
/// `sizes` holds the download size of each of the model's quants known,
/// `cached` the quants already downloaded and converted, and `held` how many
/// of `quant`'s files are downloaded already.
pub(crate) fn refusal(
    entry: &ModelEntry,
    name: &str,
    quant: &str,
    target: Target,
    device: Device,
    sizes: &BTreeMap<String, u64>,
    cached: &[String],
    held: Held,
) -> Option<String> {
    let outcome = match held {
        Held::Nothing => "nothing was downloaded",
        Held::Part => "nothing more was downloaded",
        Held::All => "its downloaded files were not converted",
    };
    let gib = |bytes: u64| format!("{:.2} GiB", bytes as f64 / (1u64 << 30) as f64);
    let model = &entry.display_name;
    let mut message = match ruling(entry, quant, target, device, sizes.get(quant).copied())? {
        Ruling::Capability { need, have } => format!(
            "{model} {quant} needs a CUDA GPU of compute capability {}.{} or newer and {} \
             compute capability {}.{}, so it cannot run there; {outcome}.",
            need.0,
            need.1,
            target.holder(),
            have.0,
            have.1
        ),
        Ruling::Weights { size, memory } => format!(
            "{model} {quant} is a {} download whose weights alone are more than the {} of \
             memory {}, so it cannot run there; {outcome}.",
            gib(size),
            gib(memory),
            target.holder()
        ),
        Ruling::BF16AsF32 {
            size,
            at_least,
            memory,
            have,
        } => format!(
            "{model} {quant} is a {} download whose weights take at least {} on a GPU of \
             compute capability {}.{}, which holds BF16 weights as F32 (below 9.0), more than \
             the {} of memory {}, so it cannot run there; {outcome}.",
            gib(size),
            gib(at_least),
            have.0,
            have.1,
            gib(memory),
            target.holder()
        ),
    };
    let mut open: Vec<(&String, Option<u64>)> = entry
        .gguf_files
        .keys()
        .filter(|q| q.as_str() != quant)
        .map(|q| (q, sizes.get(q).copied()))
        .filter(|(q, size)| ruling(entry, q, target, device, *size).is_none())
        .collect();
    open.sort_by(|a, b| b.1.cmp(&a.1));
    if open.is_empty() {
        message.push_str(&format!(
            "\nEvery quantization of {model} is ruled out {}: {}.",
            target.there(),
            target.instead()
        ));
    } else {
        message.push_str(&format!("\nNot ruled out {}:", target.there()));
        for (q, size) in open {
            let size = size.map_or_else(|| "size unknown".to_owned(), gib);
            let default = if entry.default_quant.as_deref() == Some(q.as_str()) {
                ", the default"
            } else {
                ""
            };
            let downloaded = if cached.contains(q) {
                ", already downloaded"
            } else {
                ""
            };
            message.push_str(&format!(
                "\n  lumen run {name}:{}{} \"…\"   ({size}{default}{downloaded})",
                q.to_lowercase(),
                target.flag(),
            ));
        }
    }
    Some(message)
}

/// Refuse to download or convert `quant` of `entry` when CUDA device
/// `ordinal` could not run it. Nothing is checked when this machine has no
/// such device; a rule whose facts (the device's memory or compute capability,
/// a download size, from disk for a file already downloaded) cannot be learnt
/// is skipped, and the download then goes ahead as it would have.
#[cfg(all(feature = "download", feature = "cuda"))]
pub(crate) fn check(
    entry: &ModelEntry,
    name: &str,
    quant: &str,
    ordinal: usize,
) -> Result<(), String> {
    check_against(entry, name, quant, Target::Cuda(ordinal), || Device {
        memory: lumen_runtime::cuda::ffi::device_total_memory(ordinal),
        compute_capability: lumen_runtime::cuda::ffi::device_compute_capability(ordinal),
    })
}

/// Refuse to download or convert `quant` of `entry` for a CPU run when its
/// weights alone could not fit the memory this machine has available. Nothing
/// is checked when a size or the available memory cannot be learnt.
#[cfg(feature = "download")]
pub(crate) fn check_host(entry: &ModelEntry, name: &str, quant: &str) -> Result<(), String> {
    check_against(entry, name, quant, Target::Host, || Device {
        memory: lumen_runtime::compute::host_available_bytes(),
        compute_capability: None,
    })
}

/// The download check against `target`, whose facts `device` reports.
#[cfg(feature = "download")]
fn check_against(
    entry: &ModelEntry,
    name: &str,
    quant: &str,
    target: Target,
    device: impl FnOnce() -> Device,
) -> Result<(), String> {
    let Some(src) = entry.gguf_files.get(quant) else {
        return Ok(());
    };
    let device = device();
    let mut sizes = BTreeMap::new();
    if let Some(size) = download_size(entry, quant) {
        sizes.insert(quant.to_owned(), size);
    }
    if ruling(entry, quant, target, device, sizes.get(quant).copied()).is_none() {
        return Ok(());
    }
    for q in entry.gguf_files.keys().filter(|q| *q != quant) {
        if let Some(s) = download_size(entry, q) {
            sizes.insert(q.clone(), s);
        }
    }
    let cached: Vec<String> = entry
        .gguf_files
        .keys()
        .filter(|q| crate::cache::cached_lbc(&entry.key, q).is_some())
        .cloned()
        .collect();
    let present = src.files.iter().filter(|f| downloaded(f).is_some()).count();
    let held = match present {
        0 => Held::Nothing,
        n if n == src.files.len() => Held::All,
        _ => Held::Part,
    };
    match refusal(entry, name, quant, target, device, &sizes, &cached, held) {
        Some(message) => Err(message),
        None => Ok(()),
    }
}

/// The bytes `quant` of `entry` downloads: each file's size on disk when it
/// is downloaded already, else the size Hugging Face reports; `None` when one
/// is unknown.
#[cfg(feature = "download")]
fn download_size(entry: &ModelEntry, quant: &str) -> Option<u64> {
    let src = entry.gguf_files.get(quant)?;
    src.files
        .iter()
        .map(|f| match downloaded(f) {
            Some(path) => std::fs::metadata(path).ok().map(|m| m.len()),
            None => crate::download::remote_size(&src.repo, f),
        })
        .sum()
}

/// The registry's file `f` in the cache, where the downloader keeps it: under
/// its base name, whatever folder of the repository it comes from.
#[cfg(feature = "download")]
fn downloaded(f: &str) -> Option<std::path::PathBuf> {
    let (_, local) = crate::download::split_repo_path(f).ok()?;
    crate::cache::cached_gguf(&local)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    fn entry() -> ModelEntry {
        crate::registry::load_registry()
            .resolve("qwen3.8-27b")
            .expect("registry has qwen3.8-27b")
            .clone()
    }

    fn sizes(pairs: &[(&str, u64)]) -> BTreeMap<String, u64> {
        pairs.iter().map(|(q, s)| (q.to_string(), *s)).collect()
    }

    #[test]
    fn the_shrink_is_the_census_maximum_rounded_up_and_covers_every_registry_model() {
        let max = CONVERSION.iter().map(|c| c.3).max().unwrap();
        let rounded = (max as f64 / 100.0).ceil() / 10.0;
        assert_eq!(CONVERSION_SHRINK, (rounded * GIB as f64).ceil() as u64);
        let reg = crate::registry::load_registry();
        let mut cells = 0;
        for entry in reg.list().iter().filter(|e| e.checkpoint.is_none()) {
            for quant in entry.gguf_files.keys() {
                cells += 1;
                assert!(
                    CONVERSION.iter().any(|c| c.0 == entry.key && c.1 == quant),
                    "{} {quant} is not in the census",
                    entry.key
                );
            }
        }
        assert_eq!(cells, CONVERSION.len());
    }

    /// A device at compute capability 12.0, where no rule but the weights
    /// one can refuse a registry quantization that is not BF16 below 9.0.
    const BLACKWELL: ComputeCapability = (12, 0);

    #[test]
    fn each_quantization_is_held_to_its_own_conversion_with_the_margin() {
        for &(key, quant, size, shrink) in &CONVERSION {
            let e = crate::registry::load_registry()
                .resolve(key)
                .unwrap()
                .clone();
            let s = sizes(&[(quant, size)]);
            let least = (size as i128 - (shrink + 10) as i128 * GIB as i128 / 1000) as u64;
            let refused = |memory: u64| {
                refusal(
                    &e,
                    key,
                    quant,
                    Target::Cuda(0),
                    gpu(memory, BLACKWELL),
                    &s,
                    &[],
                    Held::Nothing,
                )
                .is_some()
            };
            assert!(
                !refused(least),
                "{key} {quant} refused with room for its least weights"
            );
            assert!(
                refused(least - 1),
                "{key} {quant} admitted one byte short of them"
            );
        }
    }

    #[test]
    fn no_bound_exceeds_the_file_a_real_conversion_wrote() {
        // `.lbc` files converted for CUDA from these very downloads, which hold
        // the weights and more (header, tokenizer, the host-kept norms).
        let converted = [
            ("qwen3-5-9b", "Q8_0", 9_528_454_404u64),
            ("qwen3-5-9b", "Q4_0", 5_799_718_148),
            ("qwen3-5-9b", "BF16", 17_538_264_324),
            ("qwen3-5-moe-35b-a3b", "Q4_0", 20_733_275_396),
            ("qwen3-8-27b", "Q8_0", 28_600_219_073),
            ("qwen3-8-27b", "Q4_0", 16_556_012_993),
            ("qwen3-8-27b", "Q4_K_M", 17_203_507_632),
            ("qwen3-8-27b", "Q5_K_M", 20_689_367_472),
        ];
        for (key, quant, file) in converted {
            let &(_, _, size, _) = CONVERSION
                .iter()
                .find(|c| c.0 == key && c.1 == quant)
                .unwrap();
            let e = crate::registry::load_registry()
                .resolve(key)
                .unwrap()
                .clone();
            assert!(
                may_fit(&e, quant, Target::Cuda(0), size, file),
                "{key} {quant}: the bound exceeds the {file}-byte conversion"
            );
        }
    }

    #[test]
    fn a_download_of_another_size_and_a_cpu_run_on_macos_keep_the_widest_bound() {
        let e = entry();
        let other = 16_056_478_688 + 1;
        let s = sizes(&[("Q4_0", other)]);
        let at = other - CONVERSION_SHRINK;
        let refused = |target: Target, device: Device, s: &BTreeMap<String, u64>| {
            refusal(
                &e,
                "qwen3.8-27b",
                "Q4_0",
                target,
                device,
                s,
                &[],
                Held::Nothing,
            )
            .is_some()
        };
        assert!(!refused(Target::Cuda(0), gpu(at, BLACKWELL), &s));
        assert!(refused(Target::Cuda(0), gpu(at - 1, BLACKWELL), &s));
        // a CPU run converts as CUDA does except on macOS, whose conversion targets Metal
        let measured = sizes_27b();
        let host_at = if cfg!(target_os = "macos") {
            16_056_478_688 - CONVERSION_SHRINK
        } else {
            (16_056_478_688i128 + 441 * GIB as i128 / 1000) as u64
        };
        assert!(!refused(Target::Host, memory(host_at), &measured));
        assert!(refused(Target::Host, memory(host_at - 1), &measured));
    }

    /// A device whose memory is known and whose compute capability is not.
    fn memory(bytes: u64) -> Device {
        Device {
            memory: Some(bytes),
            compute_capability: None,
        }
    }

    fn gpu(bytes: u64, cc: ComputeCapability) -> Device {
        Device {
            memory: Some(bytes),
            compute_capability: Some(cc),
        }
    }

    /// The 27B's quants at their download sizes on 2026-10-20, BF16's the one
    /// its census row was measured on.
    fn sizes_27b() -> BTreeMap<String, u64> {
        sizes(&[
            ("BF16", 54_657_735_616),
            ("Q8_0", 29_047_086_048),
            ("Q5_K_M", 20_923_877_088),
            ("Q4_K_M", 17_442_399_968),
            ("Q4_0", 16_056_478_688),
        ])
    }

    #[test]
    fn the_27b_q4_0_whose_conversion_outgrows_a_t4_is_refused() {
        let message = refusal(
            &entry(),
            "qwen3.8-27b",
            "Q4_0",
            Target::Cuda(0),
            gpu(15_360 << 20, (7, 5)),
            &sizes_27b(),
            &[],
            Held::Nothing,
        )
        .expect("refused");
        assert!(
            message.starts_with(
                "Qwen3.8 27B Q4_0 is a 14.95 GiB download whose weights alone are more than \
                 the 15.00 GiB of memory CUDA device 0 has, so it cannot run there; nothing was \
                 downloaded."
            ),
            "{message}"
        );
        assert!(
            message.ends_with(
                "\nEvery quantization of Qwen3.8 27B is ruled out on that GPU: a smaller model or \
                 a larger GPU is needed."
            ),
            "{message}"
        );
    }

    #[test]
    fn weights_that_may_fit_are_not_refused() {
        let at = 24 * GIB + CONVERSION_SHRINK;
        let s = sizes(&[("Q8_0", at)]);
        let e = entry();
        assert_eq!(
            refusal(
                &e,
                "qwen3.8-27b",
                "Q8_0",
                Target::Cuda(0),
                memory(24 * GIB),
                &s,
                &[],
                Held::Nothing
            ),
            None
        );
        assert_eq!(
            refusal(
                &e,
                "qwen3.8-27b",
                "Q8_0",
                Target::Cuda(0),
                memory(24 * GIB),
                &BTreeMap::new(),
                &[],
                Held::Nothing
            ),
            None,
            "an unknown size refuses nothing"
        );
    }

    #[test]
    fn weights_one_byte_too_many_are_refused_naming_what_is_not_ruled_out() {
        let s = sizes(&[
            ("Q8_0", 24 * GIB + CONVERSION_SHRINK + 1),
            ("Q4_0", 15 * GIB),
            ("Q4_K_M", 24 * GIB + CONVERSION_SHRINK),
            ("Q5_K_M", 25 * GIB + CONVERSION_SHRINK),
            ("BF16", 51 * GIB),
        ]);
        let cached = vec!["Q4_0".to_string()];
        let message = refusal(
            &entry(),
            "qwen3.8-27b",
            "Q8_0",
            Target::Cuda(1),
            memory(24 * GIB),
            &s,
            &cached,
            Held::Nothing,
        )
        .unwrap();
        assert!(
            message.starts_with(
                "Qwen3.8 27B Q8_0 is a 26.20 GiB download whose weights alone are more than \
                 the 24.00 GiB of memory CUDA device 1 has"
            ),
            "{message}"
        );
        let k = message
            .find("lumen run qwen3.8-27b:q4_k_m --cuda-device 1")
            .expect(&message);
        let q4 = message
            .find(
                "lumen run qwen3.8-27b:q4_0 --cuda-device 1 \"…\"   (15.00 GiB, the default, already downloaded)",
            )
            .expect(&message);
        assert!(k < q4, "the biggest not ruled out comes first: {message}");
        assert!(!message.contains(":bf16"), "{message}");
        assert!(!message.contains(":q5_k_m"), "{message}");
        assert!(!message.contains("--simd"), "{message}");
    }

    #[test]
    fn a_gpu_every_quant_is_ruled_out_on_is_told_so_and_what_is_needed() {
        let s = sizes(&[
            ("Q8_0", 27 * GIB),
            ("Q4_0", 15 * GIB),
            ("Q4_K_M", 16 * GIB),
            ("Q5_K_M", 19 * GIB),
            ("BF16", 51 * GIB),
        ]);
        let message = refusal(
            &entry(),
            "qwen3.8-27b",
            "Q8_0",
            Target::Cuda(0),
            memory(12 * GIB),
            &s,
            &[],
            Held::Nothing,
        )
        .unwrap();
        assert!(
            message.contains(
                "\nEvery quantization of Qwen3.8 27B is ruled out on that GPU: a smaller model \
                 or a larger GPU is needed."
            ),
            "{message}"
        );
        assert!(!message.contains("Not ruled out"), "{message}");
    }

    #[test]
    fn a_quant_of_unknown_size_is_not_ruled_out_by_its_size() {
        let s = sizes(&[("Q8_0", 27 * GIB), ("Q4_0", 15 * GIB)]);
        let message = refusal(
            &entry(),
            "qwen3.8-27b",
            "Q8_0",
            Target::Cuda(0),
            memory(12 * GIB),
            &s,
            &[],
            Held::Nothing,
        )
        .unwrap();
        assert!(
            message.contains("lumen run qwen3.8-27b:q4_k_m \"…\"   (size unknown)"),
            "{message}"
        );
        assert!(!message.contains(":q4_0"), "{message}");
    }

    #[test]
    fn a_cpu_run_names_the_memory_this_machine_has_available() {
        let s = sizes(&[
            ("Q8_0", 27 * GIB),
            ("Q4_0", 15 * GIB),
            ("Q4_K_M", 17 * GIB),
            ("Q5_K_M", 20 * GIB),
            ("BF16", 51 * GIB),
        ]);
        let message = refusal(
            &entry(),
            "qwen3.8-27b",
            "Q8_0",
            Target::Host,
            memory(16 * GIB),
            &s,
            &[],
            Held::Nothing,
        )
        .unwrap();
        assert!(
            message.starts_with(
                "Qwen3.8 27B Q8_0 is a 27.00 GiB download whose weights alone are more than \
                 the 16.00 GiB of memory this machine has available"
            ),
            "{message}"
        );
        assert!(
            message.contains("\nNot ruled out on this machine:"),
            "{message}"
        );
        assert!(
            message.contains("lumen run qwen3.8-27b:q4_0 --simd \"…\""),
            "a CPU suggestion runs on the CPU: {message}"
        );
    }

    #[test]
    fn a_refusal_of_downloaded_files_says_they_were_not_converted() {
        let t4 = gpu(15_645_278_208, (7, 5));
        let message = refusal(
            &entry(),
            "qwen3.8-27b",
            "Q4_K_M",
            Target::Cuda(0),
            t4,
            &sizes_27b(),
            &[],
            Held::All,
        )
        .unwrap();
        assert!(
            message.contains("so it cannot run there; its downloaded files were not converted."),
            "{message}"
        );
        assert!(!message.contains("nothing was downloaded"), "{message}");
        let part = refusal(
            &entry(),
            "qwen3.8-27b",
            "Q4_K_M",
            Target::Cuda(0),
            t4,
            &sizes_27b(),
            &[],
            Held::Part,
        )
        .unwrap();
        assert!(
            part.contains("so it cannot run there; nothing more was downloaded."),
            "{part}"
        );
    }

    #[test]
    fn a_kquant_is_refused_below_compute_capability_8_naming_both_and_offering_nothing_that_cannot_run(
    ) {
        let t4 = gpu(15_645_278_208, (7, 5));
        let message = refusal(
            &entry(),
            "qwen3.8-27b",
            "Q4_K_M",
            Target::Cuda(0),
            t4,
            &sizes_27b(),
            &[],
            Held::Nothing,
        )
        .unwrap();
        assert!(
            message.starts_with(
                "Qwen3.8 27B Q4_K_M needs a CUDA GPU of compute capability 8.0 or newer and \
                 CUDA device 0 has compute capability 7.5, so it cannot run there; nothing was \
                 downloaded."
            ),
            "{message}"
        );
        // the Q4_0's conversion outgrows a T4, so no quantization of the 27B is offered
        let none = "\nEvery quantization of Qwen3.8 27B is ruled out on that GPU: a smaller \
                    model or a larger GPU is needed.";
        assert!(message.ends_with(none), "{message}");
        // nor by a size refusal on the same GPU
        let q8 = refusal(
            &entry(),
            "qwen3.8-27b",
            "Q8_0",
            Target::Cuda(0),
            t4,
            &sizes_27b(),
            &[],
            Held::Nothing,
        )
        .unwrap();
        assert!(q8.ends_with(none), "{q8}");
    }

    #[test]
    fn the_capability_rule_needs_no_download_size_and_no_memory() {
        let device = Device {
            memory: None,
            compute_capability: Some((7, 5)),
        };
        let message = refusal(
            &entry(),
            "qwen3.8-27b",
            "Q5_K_M",
            Target::Cuda(0),
            device,
            &BTreeMap::new(),
            &[],
            Held::Nothing,
        )
        .unwrap();
        assert!(
            message.contains("compute capability 8.0 or newer"),
            "{message}"
        );
        assert!(
            message.contains("lumen run qwen3.8-27b:q4_0 \"…\"   (size unknown, the default)"),
            "{message}"
        );
        assert!(!message.contains(":q4_k_m"), "{message}");
    }

    #[test]
    fn compute_capability_8_0_and_an_unknown_one_admit_a_kquant() {
        for device in [gpu(24 * GIB, (8, 0)), memory(24 * GIB)] {
            assert_eq!(
                refusal(
                    &entry(),
                    "qwen3.8-27b",
                    "Q4_K_M",
                    Target::Cuda(0),
                    device,
                    &sizes_27b(),
                    &[],
                    Held::Nothing
                ),
                None
            );
        }
        assert!(refusal(
            &entry(),
            "qwen3.8-27b",
            "Q4_K_M",
            Target::Cuda(0),
            gpu(24 * GIB, (7, 9)),
            &sizes_27b(),
            &[],
            Held::Nothing
        )
        .is_some());
    }

    /// The bytes the census row says the 27B BF16 takes at the least on F32.
    fn bf16_27b_at_least() -> u64 {
        (54_657_735_616u128 * 179 / 100) as u64
    }

    #[test]
    fn bf16_held_as_f32_is_refused_from_one_byte_short_of_its_least() {
        let at_least = bf16_27b_at_least();
        let message = refusal(
            &entry(),
            "qwen3.8-27b",
            "BF16",
            Target::Cuda(0),
            gpu(at_least - 1, (8, 0)),
            &sizes_27b(),
            &[],
            Held::Nothing,
        )
        .unwrap();
        assert!(
            message.starts_with(
                "Qwen3.8 27B BF16 is a 50.90 GiB download whose weights take at least 91.12 GiB \
                 on a GPU of compute capability 8.0, which holds BF16 weights as F32 (below 9.0)"
            ),
            "{message}"
        );
        assert!(message.contains(":q8_0"), "{message}");
        assert_eq!(
            refusal(
                &entry(),
                "qwen3.8-27b",
                "BF16",
                Target::Cuda(0),
                gpu(at_least, (8, 0)),
                &sizes_27b(),
                &[],
                Held::Nothing
            ),
            None
        );
    }

    #[test]
    fn bf16_on_compute_capability_9_or_on_the_cpu_is_judged_by_its_download() {
        let a100 = 80 * GIB;
        for (target, device) in [
            (Target::Cuda(0), gpu(a100, (9, 0))),
            (Target::Cuda(0), gpu(a100, (12, 0))),
            (Target::Cuda(0), memory(a100)),
            (Target::Host, gpu(a100, (8, 0))),
        ] {
            assert_eq!(
                refusal(
                    &entry(),
                    "qwen3.8-27b",
                    "BF16",
                    target,
                    device,
                    &sizes_27b(),
                    &[],
                    Held::Nothing
                ),
                None
            );
        }
    }

    #[test]
    fn a_bf16_download_of_another_size_than_measured_skips_the_f32_bound() {
        let mut s = sizes_27b();
        s.insert("BF16".into(), 54_657_735_617);
        assert_eq!(
            refusal(
                &entry(),
                "qwen3.8-27b",
                "BF16",
                Target::Cuda(0),
                gpu(80 * GIB, (8, 0)),
                &s,
                &[],
                Held::Nothing
            ),
            None
        );
    }

    #[test]
    fn the_9b_bf16_does_not_fit_24_gib_below_compute_capability_9() {
        let reg = crate::registry::load_registry();
        let nine = reg.resolve("qwen3.5-9b").unwrap();
        let s = sizes(&[
            ("BF16", 18_407_321_344),
            ("Q8_0", 9_804_541_984),
            ("Q4_0", 5_741_391_904),
        ]);
        let message = refusal(
            nine,
            "qwen3.5-9b",
            "BF16",
            Target::Cuda(0),
            gpu(24 * GIB, (8, 6)),
            &s,
            &[],
            Held::Nothing,
        )
        .unwrap();
        assert!(message.contains("take at least 28.29 GiB"), "{message}");
        assert_eq!(
            refusal(
                nine,
                "qwen3.5-9b",
                "BF16",
                Target::Cuda(0),
                gpu(32 * GIB, (8, 6)),
                &s,
                &[],
                Held::Nothing
            ),
            None
        );
    }

    #[test]
    fn the_moe_bf16_on_f32_is_judged_by_its_own_row() {
        let reg = crate::registry::load_registry();
        let moe = reg.resolve("qwen3.5-moe").unwrap();
        let download = 71_065_941_760u64;
        let at_least = (u128::from(download) * 103 / 100) as u64;
        let s = sizes(&[("BF16", download)]);
        let rule = |memory| {
            refusal(
                moe,
                "qwen3.5-moe",
                "BF16",
                Target::Cuda(0),
                gpu(memory, (8, 0)),
                &s,
                &[],
                Held::Nothing,
            )
        };
        assert!(rule(at_least - 1).is_some());
        assert_eq!(rule(at_least), None);
        assert_eq!(rule(80 * GIB), None, "an A100-80GB is not ruled out");
    }

    #[test]
    fn every_bf16_registry_model_has_a_row() {
        let reg = crate::registry::load_registry();
        for entry in reg
            .list()
            .iter()
            .filter(|e| e.gguf_files.contains_key("BF16"))
        {
            assert!(
                BF16_AS_F32.iter().any(|(key, _, _)| *key == entry.key),
                "{} BF16 has no census row",
                entry.key
            );
        }
        for (key, _, hundredths) in BF16_AS_F32 {
            assert!(reg.list().iter().any(|e| e.key == key), "{key}");
            assert!((100..=200).contains(&hundredths), "{key}: {hundredths}");
        }
    }

    #[test]
    fn a_suggested_command_runs_on_the_device_that_was_checked() {
        let message = refusal(
            &entry(),
            "qwen3.8-27b",
            "BF16",
            Target::Cuda(1),
            gpu(80 * GIB, (8, 0)),
            &sizes_27b(),
            &[],
            Held::Nothing,
        )
        .unwrap();
        assert!(
            message.contains("lumen run qwen3.8-27b:q8_0 --cuda-device 1 \"…\""),
            "{message}"
        );
        let on_0 = refusal(
            &entry(),
            "qwen3.8-27b",
            "BF16",
            Target::Cuda(0),
            gpu(80 * GIB, (8, 0)),
            &sizes_27b(),
            &[],
            Held::Nothing,
        )
        .unwrap();
        assert!(!on_0.contains("--cuda-device"), "{on_0}");
    }

    #[cfg(feature = "download")]
    #[test]
    fn a_shard_is_found_downloaded_under_its_base_name() {
        let _env = crate::cache::tests::SERIAL
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let dir = std::env::temp_dir().join(format!("lumen-fit-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("Qwen3.8-27B-BF16-00001-of-00002.gguf"), b"x").unwrap();
        let prior = std::env::var_os("LUMEN_CACHE_DIR");
        std::env::set_var("LUMEN_CACHE_DIR", &dir);
        let first = downloaded("BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf");
        let second = downloaded("BF16/Qwen3.8-27B-BF16-00002-of-00002.gguf");
        match prior {
            Some(v) => std::env::set_var("LUMEN_CACHE_DIR", v),
            None => std::env::remove_var("LUMEN_CACHE_DIR"),
        }
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(
            first,
            Some(dir.join("Qwen3.8-27B-BF16-00001-of-00002.gguf"))
        );
        assert_eq!(second, None);
    }
}
