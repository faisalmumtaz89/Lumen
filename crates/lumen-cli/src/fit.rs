//! A text model whose download shows its weights alone are bigger than the
//! memory of the GPU it would run on is refused before the first byte. The
//! CUDA backend holds the converted weights on the device (all but one norm
//! vector per dense layer, at most 1.3 MB per model), and for every model in
//! the registry the converted weights fall short of the download by at most
//! [`CONVERSION_SHRINK`] (the MTP layer the converter drops, the tensors it
//! re-quantizes, and the file's metadata). The tests below hold that census.
#![cfg_attr(not(all(feature = "download", feature = "cuda")), allow(dead_code))]

use std::collections::BTreeMap;

use crate::registry::ModelEntry;

/// The most a registry model's converted weights fall short of its download,
/// 2.2 GiB: the census maximum, 2.140 GiB for qwen3.8-27b BF16, rounded up to
/// a tenth of a GiB.
const CONVERSION_SHRINK: u64 = 2_362_232_013;

/// Whether a download of `size` bytes may have weights that fit `memory`
/// bytes: not when, short by the most conversion removes, it is still bigger.
fn may_fit(size: u64, memory: u64) -> bool {
    size.saturating_sub(CONVERSION_SHRINK) <= memory
}

/// The refusal for `quant` of `entry`, which the user named `name`, when its
/// weights are more than `memory`, the bytes CUDA device `ordinal` has;
/// `None` when they may not be. `sizes` holds the download size of each of the
/// model's quants known, including `quant`'s, and `cached` the quants already
/// downloaded and converted.
pub(crate) fn refusal(
    entry: &ModelEntry,
    name: &str,
    quant: &str,
    ordinal: usize,
    memory: u64,
    sizes: &BTreeMap<String, u64>,
    cached: &[String],
) -> Option<String> {
    let size = *sizes.get(quant)?;
    if may_fit(size, memory) {
        return None;
    }
    let gib = |bytes: u64| format!("{:.2} GiB", bytes as f64 / (1u64 << 30) as f64);
    let mut message = format!(
        "{} {quant} is a {} download whose weights alone are more than the {} of \
         memory CUDA device {ordinal} has, so it cannot run there; nothing was downloaded.",
        entry.display_name,
        gib(size),
        gib(memory)
    );
    let mut fit: Vec<(&String, u64)> = sizes
        .iter()
        .filter(|(q, s)| q.as_str() != quant && may_fit(**s, memory))
        .map(|(q, s)| (q, *s))
        .collect();
    fit.sort_by(|a, b| b.1.cmp(&a.1));
    if fit.is_empty() {
        message.push_str(&format!(
            "\nEvery quantization of {} is ruled out on that GPU.",
            entry.display_name
        ));
    } else {
        message.push_str("\nNot ruled out on that GPU:");
        for (q, s) in fit {
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
                "\n  lumen run {name}:{} \"…\"   ({}{default}{downloaded})",
                q.to_lowercase(),
                gib(s)
            ));
        }
    }
    Some(message)
}

/// Refuse a download of `quant` of `entry` that CUDA device `ordinal` could
/// not hold. Nothing is checked when no file is left to download, when this
/// machine has no such device, or when a size cannot be learnt; the download
/// then goes ahead as it would have.
#[cfg(all(feature = "download", feature = "cuda"))]
pub(crate) fn check(
    entry: &ModelEntry,
    name: &str,
    quant: &str,
    ordinal: usize,
) -> Result<(), String> {
    let Some(src) = entry.gguf_files.get(quant) else {
        return Ok(());
    };
    if src.files.iter().all(|f| downloaded(f).is_some()) {
        return Ok(());
    }
    let Some(memory) = lumen_runtime::cuda::ffi::device_total_memory(ordinal) else {
        return Ok(());
    };
    let Some(size) = download_size(entry, quant) else {
        return Ok(());
    };
    if may_fit(size, memory) {
        return Ok(());
    }
    let mut sizes = BTreeMap::from([(quant.to_owned(), size)]);
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
    match refusal(entry, name, quant, ordinal, memory, &sizes, &cached) {
        Some(message) => Err(message),
        None => Ok(()),
    }
}

/// The bytes `quant` of `entry` downloads: each file's size on disk when it
/// is downloaded already, else the size Hugging Face reports; `None` when one
/// is unknown.
#[cfg(all(feature = "download", feature = "cuda"))]
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

    /// Download minus converted weight bytes of every registry model, in
    /// GiB: each file's real header (metadata, tokenizer, tensor table) at
    /// its full length, converted for the CUDA target, and the weights of the
    /// result summed (every layer tensor, the embedding, the final norm and
    /// the output head).
    const CENSUS: [(&str, &str, f64); 11] = [
        ("qwen3.5-9b", "Q8_0", 0.268),
        ("qwen3.5-9b", "Q4_0", -0.043),
        ("qwen3.5-9b", "BF16", 0.820),
        ("qwen3.5-moe", "Q8_0", 0.227),
        ("qwen3.5-moe", "Q4_0", 0.106),
        ("qwen3.5-moe", "BF16", 1.287),
        ("qwen3.8-27b", "Q8_0", 0.431),
        ("qwen3.8-27b", "Q4_0", -0.451),
        ("qwen3.8-27b", "Q4_K_M", 0.233),
        ("qwen3.8-27b", "Q5_K_M", 0.233),
        ("qwen3.8-27b", "BF16", 2.140),
    ];

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
        let max = CENSUS.iter().map(|c| c.2).fold(f64::MIN, f64::max);
        let rounded = (max * 10.0).ceil() / 10.0;
        assert_eq!(CONVERSION_SHRINK, (rounded * GIB as f64).ceil() as u64);
        let reg = crate::registry::load_registry();
        let mut cells = 0;
        for entry in reg.list().iter().filter(|e| e.checkpoint.is_none()) {
            for quant in entry.gguf_files.keys() {
                cells += 1;
                assert!(
                    CENSUS
                        .iter()
                        .any(|c| reg.resolve(c.0).unwrap().key == entry.key && c.1 == quant),
                    "{} {quant} is not in the census",
                    entry.key
                );
            }
        }
        assert_eq!(cells, CENSUS.len());
    }

    #[test]
    fn weights_that_may_fit_are_not_refused() {
        let at = 24 * GIB + CONVERSION_SHRINK;
        let s = sizes(&[("Q8_0", at)]);
        assert_eq!(
            refusal(&entry(), "qwen3.8-27b", "Q8_0", 0, 24 * GIB, &s, &[]),
            None
        );
        assert_eq!(
            refusal(
                &entry(),
                "qwen3.8-27b",
                "Q8_0",
                0,
                24 * GIB,
                &BTreeMap::new(),
                &[]
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
            ("BF16", 51 * GIB),
        ]);
        let cached = vec!["Q4_0".to_string()];
        let message = refusal(&entry(), "qwen3.8-27b", "Q8_0", 1, 24 * GIB, &s, &cached).unwrap();
        assert!(
            message.starts_with(
                "Qwen3.8 27B Q8_0 is a 26.20 GiB download whose weights alone are more than \
                 the 24.00 GiB of memory CUDA device 1 has"
            ),
            "{message}"
        );
        let k = message
            .find("lumen run qwen3.8-27b:q4_k_m")
            .expect(&message);
        let q4 = message
            .find("lumen run qwen3.8-27b:q4_0 \"…\"   (15.00 GiB, the default, already downloaded)")
            .expect(&message);
        assert!(k < q4, "the biggest not ruled out comes first: {message}");
        assert!(!message.contains(":bf16"), "{message}");
        assert!(!message.contains("--simd"), "{message}");
    }

    #[test]
    fn a_gpu_every_quant_is_ruled_out_on_is_told_so() {
        let s = sizes(&[("Q8_0", 27 * GIB), ("Q4_0", 15 * GIB)]);
        let message = refusal(&entry(), "qwen3.8-27b", "Q8_0", 0, 12 * GIB, &s, &[]).unwrap();
        assert!(
            message.contains("\nEvery quantization of Qwen3.8 27B is ruled out on that GPU."),
            "{message}"
        );
        assert!(!message.contains("Not ruled out"), "{message}");
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
