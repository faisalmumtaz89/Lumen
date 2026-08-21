//! HF safetensors checkpoint reader for compressed-tensors "pack-quantized"
//! INT4 group-32 asymmetric models (the [`QuantScheme::CtInt4G32`] source
//! dialect).
//!
//! Reads a checkpoint directory (`config.json` +
//! `model.safetensors.index.json` + shards), enforces a strict one-dialect
//! preflight, and serves exact tensor bytes by name. Anything outside the
//! supported dialect is rejected at open time with a clear error — never
//! silently reinterpreted.
//!
//! [`QuantScheme::CtInt4G32`]: lumen_format::QuantScheme::CtInt4G32

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::convert::ConvertError;

/// Tensor element types the supported checkpoints contain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HfDtype {
    F32,
    F16,
    Bf16,
    I32,
    I64,
}

impl HfDtype {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "F32" => Some(Self::F32),
            "F16" => Some(Self::F16),
            "BF16" => Some(Self::Bf16),
            "I32" => Some(Self::I32),
            "I64" => Some(Self::I64),
            _ => None,
        }
    }

    pub fn byte_size(&self) -> u64 {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::I64 => 8,
        }
    }
}

/// Location + typing of one tensor inside a shard.
#[derive(Debug, Clone)]
pub struct HfTensorInfo {
    pub dtype: HfDtype,
    pub shape: Vec<u64>,
    shard: usize,
    /// Absolute byte range within the shard file (data section already
    /// offset by the header).
    begin: u64,
    end: u64,
}

impl HfTensorInfo {
    pub fn byte_len(&self) -> u64 {
        self.end - self.begin
    }
}

/// The quantization dialect this importer supports, validated at open time.
/// One accepted configuration; every field is checked against the checkpoint.
#[derive(Debug, Clone)]
pub struct CtQuantConfig {
    /// Module prefixes excluded from quantization (kept in floating point).
    pub ignore: Vec<String>,
}

/// An open checkpoint: validated shard set + tensor index + quant config.
#[derive(Debug)]
pub struct HfCtCheckpoint {
    shard_paths: Vec<PathBuf>,
    tensors: HashMap<String, HfTensorInfo>,
    pub quant: CtQuantConfig,
    pub config: serde_json::Value,
}

fn bad(msg: String) -> ConvertError {
    ConvertError::UnsupportedArchitecture(msg)
}

fn read_json(path: &Path) -> Result<serde_json::Value, ConvertError> {
    let bytes = std::fs::read(path).map_err(ConvertError::Io)?;
    serde_json::from_slice(&bytes)
        .map_err(|e| bad(format!("{}: invalid JSON: {e}", path.display())))
}

/// Validate `quantization_config` against the one supported dialect:
/// compressed-tensors "pack-quantized", a single group targeting `Linear`
/// with int4, group-32, asymmetric, group-strategy weights.
fn preflight(config: &serde_json::Value) -> Result<CtQuantConfig, ConvertError> {
    let qc = config
        .get("quantization_config")
        .ok_or_else(|| bad("checkpoint has no quantization_config".into()))?;
    let fmt = qc.get("format").and_then(|v| v.as_str()).unwrap_or("");
    if fmt != "pack-quantized" {
        return Err(bad(format!(
            "unsupported quantization format {fmt:?} (need \"pack-quantized\")"
        )));
    }
    let groups = qc
        .get("config_groups")
        .and_then(|v| v.as_object())
        .ok_or_else(|| bad("quantization_config has no config_groups".into()))?;
    if groups.len() != 1 {
        return Err(bad(format!(
            "need exactly one quantization config group, got {}",
            groups.len()
        )));
    }
    let g = groups.values().next().unwrap();
    let targets: Vec<Option<&str>> = g
        .get("targets")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().map(|t| t.as_str()).collect())
        .unwrap_or_default();
    if targets != [Some("Linear")] {
        return Err(bad(format!(
            "unsupported quantization targets {targets:?} (need [\"Linear\"])"
        )));
    }
    // Activation-quantization specs live on the config GROUP, not inside
    // `weights` — checking the wrong level would wave through checkpoints
    // that require quantized activations.
    for k in ["input_activations", "output_activations"] {
        if g.get(k).is_some_and(|v| !v.is_null()) {
            return Err(bad(format!(
                "activation quantization ({k}) is not supported"
            )));
        }
    }
    let w = g
        .get("weights")
        .ok_or_else(|| bad("quantization group has no weights spec".into()))?;
    let field = |k: &str| w.get(k).cloned().unwrap_or(serde_json::Value::Null);
    let checks = [
        ("type", serde_json::json!("int")),
        ("num_bits", serde_json::json!(4)),
        ("group_size", serde_json::json!(32)),
        ("symmetric", serde_json::json!(false)),
        ("strategy", serde_json::json!("group")),
        ("dynamic", serde_json::json!(false)),
    ];
    for (k, want) in checks {
        let got = field(k);
        if got != want {
            return Err(bad(format!(
                "unsupported weights.{k} = {got} (need {want})"
            )));
        }
    }
    // Activation ordering permutes quantization groups via a `weight_g_idx`
    // tensor; the importer applies scales to CONTIGUOUS k/32 groups, so an
    // ordered checkpoint would convert silently wrong. The schema puts
    // `actorder` inside `weights`; check the group level too so a
    // nonstandard producer cannot slip it past.
    for (loc, v) in [
        ("weights", field("actorder")),
        (
            "group",
            g.get("actorder")
                .cloned()
                .unwrap_or(serde_json::Value::Null),
        ),
    ] {
        if !v.is_null() && v != serde_json::json!(false) {
            return Err(bad(format!(
                "unsupported {loc} actorder = {v} (activation ordering is not supported)"
            )));
        }
    }
    if !field("input_activations").is_null() || !field("output_activations").is_null() {
        return Err(bad("activation quantization is not supported".into()));
    }
    let mut ignore = Vec::new();
    match qc.get("ignore") {
        None | Some(serde_json::Value::Null) => {}
        Some(serde_json::Value::Array(arr)) => {
            for entry in arr {
                let s = entry.as_str().ok_or_else(|| {
                    bad(format!("quantization ignore entry {entry} is not a string"))
                })?;
                ignore.push(s.to_owned());
            }
        }
        Some(other) => return Err(bad(format!("quantization ignore is not an array: {other}"))),
    }
    Ok(CtQuantConfig { ignore })
}

/// Parse one safetensors header: `u64 LE header length` + JSON map of
/// `name -> {dtype, shape, data_offsets}`. Returns entries with ranges made
/// absolute (offset by the data-section start) and bounds-checked against
/// the file length.
fn parse_shard_header(
    path: &Path,
    shard_idx: usize,
) -> Result<Vec<(String, HfTensorInfo)>, ConvertError> {
    let mut f = File::open(path).map_err(ConvertError::Io)?;
    let file_len = f.metadata().map_err(ConvertError::Io)?.len();
    let mut len_buf = [0u8; 8];
    f.read_exact(&mut len_buf).map_err(ConvertError::Io)?;
    let header_len = u64::from_le_bytes(len_buf);
    if header_len == 0 || header_len > 100 * 1024 * 1024 || header_len + 8 > file_len {
        return Err(bad(format!(
            "{}: implausible safetensors header length {header_len}",
            path.display()
        )));
    }
    let mut header_bytes = vec![0u8; header_len as usize];
    f.read_exact(&mut header_bytes).map_err(ConvertError::Io)?;
    let header: serde_json::Value = serde_json::from_slice(&header_bytes).map_err(|e| {
        bad(format!(
            "{}: invalid safetensors header: {e}",
            path.display()
        ))
    })?;
    let map = header.as_object().ok_or_else(|| {
        bad(format!(
            "{}: safetensors header is not a map",
            path.display()
        ))
    })?;

    let data_start = 8 + header_len;
    let data_len = file_len - data_start;
    let mut out = Vec::with_capacity(map.len());
    for (name, meta) in map {
        if name == "__metadata__" {
            continue;
        }
        let dtype_str = meta.get("dtype").and_then(|v| v.as_str()).unwrap_or("");
        let dtype = HfDtype::parse(dtype_str).ok_or_else(|| {
            bad(format!(
                "tensor {name}: unsupported safetensors dtype {dtype_str:?}"
            ))
        })?;
        // Reject (never drop) non-integer members: a silently skipped
        // dimension would shift every downstream size check.
        let u64_list = |key: &str| -> Result<Vec<u64>, ConvertError> {
            let arr = meta
                .get(key)
                .and_then(|v| v.as_array())
                .ok_or_else(|| bad(format!("tensor {name}: {key} is not an array")))?;
            arr.iter()
                .map(|d| {
                    d.as_u64()
                        .ok_or_else(|| bad(format!("tensor {name}: {key} member {d} is not u64")))
                })
                .collect()
        };
        let shape = u64_list("shape")?;
        let offs = u64_list("data_offsets")?;
        if offs.len() != 2 || offs[1] < offs[0] || offs[1] > data_len {
            return Err(bad(format!(
                "tensor {name}: bad data_offsets {offs:?} (shard data length {data_len})"
            )));
        }
        let n_bytes: u64 = shape
            .iter()
            .try_fold(1u64, |acc, &d| acc.checked_mul(d))
            .and_then(|n| n.checked_mul(dtype.byte_size()))
            .ok_or_else(|| bad(format!("tensor {name}: shape overflow {shape:?}")))?;
        if offs[1] - offs[0] != n_bytes {
            return Err(bad(format!(
                "tensor {name}: byte range {} != shape {:?} x {dtype:?}",
                offs[1] - offs[0],
                shape
            )));
        }
        out.push((
            name.clone(),
            HfTensorInfo {
                dtype,
                shape,
                shard: shard_idx,
                begin: data_start + offs[0],
                end: data_start + offs[1],
            },
        ));
    }
    // Overlapping ranges would let two names alias the same bytes; gaps or
    // trailing unindexed bytes mean the file is not a well-formed safetensors
    // shard (the format requires the data section to be exactly covered).
    let mut ranges: Vec<(u64, u64, &str)> = out
        .iter()
        .map(|(n, i)| (i.begin - data_start, i.end - data_start, n.as_str()))
        .collect();
    ranges.sort_unstable();
    let mut covered = 0u64;
    for (begin, end, name) in &ranges {
        if *begin > covered {
            return Err(bad(format!(
                "{}: {} bytes of unindexed data before tensor {name}",
                path.display(),
                begin - covered
            )));
        }
        if *begin < covered {
            return Err(bad(format!(
                "{}: tensor {name} overlaps the preceding tensor",
                path.display()
            )));
        }
        covered = *end;
    }
    if covered != data_len {
        return Err(bad(format!(
            "{}: {} trailing bytes not covered by the tensor index",
            path.display(),
            data_len - covered
        )));
    }
    Ok(out)
}

impl HfCtCheckpoint {
    /// Open and fully validate a checkpoint directory.
    pub fn open(dir: &Path) -> Result<Self, ConvertError> {
        let config = read_json(&dir.join("config.json"))?;
        let quant = preflight(&config)?;

        let index = read_json(&dir.join("model.safetensors.index.json"))?;
        let weight_map = index
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| bad("index has no weight_map".into()))?;

        let mut shard_paths: Vec<PathBuf> = Vec::new();
        let mut shard_of: HashMap<String, usize> = HashMap::new();
        for shard_name in weight_map.values() {
            let s = shard_name
                .as_str()
                .ok_or_else(|| bad("weight_map value is not a string".into()))?;
            // Shard names must be plain filenames — an index carrying path
            // separators or `..` could otherwise read files outside the
            // checkpoint directory.
            if s.is_empty() || Path::new(s).components().count() != 1 || s.contains(['/', '\\']) {
                return Err(bad(format!(
                    "weight_map shard name {s:?} is not a plain filename"
                )));
            }
            if !shard_of.contains_key(s) {
                shard_of.insert(s.to_owned(), shard_paths.len());
                shard_paths.push(dir.join(s));
            }
        }

        let mut tensors: HashMap<String, HfTensorInfo> = HashMap::new();
        for (shard_name, &idx) in &shard_of {
            for (name, info) in parse_shard_header(&shard_paths[idx], idx)? {
                let expected_shard = weight_map.get(&name).and_then(|v| v.as_str());
                if expected_shard != Some(shard_name.as_str()) {
                    return Err(bad(format!(
                        "tensor {name} found in shard {shard_name} but the index maps it to {expected_shard:?}"
                    )));
                }
                if tensors.insert(name.clone(), info).is_some() {
                    return Err(bad(format!("duplicate tensor {name} across shards")));
                }
            }
        }
        for name in weight_map.keys() {
            if !tensors.contains_key(name) {
                return Err(ConvertError::MissingTensor(format!(
                    "{name} (in index but absent from its shard)"
                )));
            }
        }
        // Reject group-permutation index tensors outright (see the actorder
        // preflight check) — their presence means the packed groups are not
        // laid out contiguously.
        if let Some(name) = tensors.keys().find(|n| n.ends_with(".weight_g_idx")) {
            return Err(bad(format!(
                "checkpoint contains {name}: activation-ordered quantization is not supported"
            )));
        }
        Ok(Self {
            shard_paths,
            tensors,
            quant,
            config,
        })
    }

    pub fn tensor_info(&self, name: &str) -> Option<&HfTensorInfo> {
        self.tensors.get(name)
    }

    /// Total on-disk size of the checkpoint's shard files.
    pub fn total_shard_bytes(&self) -> Result<u64, ConvertError> {
        let mut total = 0u64;
        for p in &self.shard_paths {
            total += std::fs::metadata(p).map_err(ConvertError::Io)?.len();
        }
        Ok(total)
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }

    /// Read a tensor's exact bytes from its shard.
    pub fn tensor_bytes(&self, name: &str) -> Result<Vec<u8>, ConvertError> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| ConvertError::MissingTensor(name.to_owned()))?;
        let mut f = File::open(&self.shard_paths[info.shard]).map_err(ConvertError::Io)?;
        f.seek(SeekFrom::Start(info.begin))
            .map_err(ConvertError::Io)?;
        let mut buf = vec![0u8; info.byte_len() as usize];
        f.read_exact(&mut buf).map_err(ConvertError::Io)?;
        Ok(buf)
    }
}

#[cfg(test)]
pub(crate) mod test_fixture {
    use std::path::Path;

    /// Serialize one synthetic safetensors shard.
    pub(crate) fn shard_bytes(entries: &[(&str, &str, &[u64], &[u8])]) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        let mut data: Vec<u8> = Vec::new();
        for (name, dtype, shape, bytes) in entries {
            let begin = data.len() as u64;
            data.extend_from_slice(bytes);
            header.insert(
                (*name).to_owned(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [begin, begin + bytes.len() as u64],
                }),
            );
        }
        let hjson = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut out = (hjson.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&hjson);
        out.extend_from_slice(&data);
        out
    }

    /// A config.json matching the one supported dialect.
    pub(crate) fn dialect_config() -> serde_json::Value {
        serde_json::json!({
            "quantization_config": {
                "format": "pack-quantized",
                "ignore": ["lm_head"],
                "config_groups": { "group_0": {
                    "targets": ["Linear"],
                    "input_activations": null,
                    "output_activations": null,
                    "weights": {
                        "type": "int", "num_bits": 4, "group_size": 32,
                        "symmetric": false, "strategy": "group", "dynamic": false
                    }
                }}
            }
        })
    }

    /// Write a full synthetic checkpoint directory.
    pub(crate) fn write_checkpoint(
        dir: &Path,
        config: &serde_json::Value,
        shards: &[(&str, Vec<u8>)],
        weight_map: &[(&str, &str)],
    ) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("config.json"), serde_json::to_vec(config).unwrap()).unwrap();
        let wm: serde_json::Map<String, serde_json::Value> = weight_map
            .iter()
            .map(|(t, s)| ((*t).to_owned(), serde_json::json!(s)))
            .collect();
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({"weight_map": wm})).unwrap(),
        )
        .unwrap();
        for (name, bytes) in shards {
            std::fs::write(dir.join(name), bytes).unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_fixture::{shard_bytes, write_checkpoint};

    fn good_config() -> serde_json::Value {
        test_fixture::dialect_config()
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("lumen-hfct-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn open_and_fetch_exact_bytes() {
        let dir = temp_dir("ok");
        let payload: Vec<u8> = (0u8..32).collect();
        let shard = shard_bytes(&[("a.weight_packed", "I32", &[2, 4], &payload)]);
        write_checkpoint(
            &dir,
            &good_config(),
            &[("model-00001.safetensors", shard)],
            &[("a.weight_packed", "model-00001.safetensors")],
        );
        let ckpt = HfCtCheckpoint::open(&dir).unwrap();
        assert_eq!(ckpt.quant.ignore, vec!["lm_head".to_owned()]);
        let info = ckpt.tensor_info("a.weight_packed").unwrap();
        assert_eq!(info.dtype, HfDtype::I32);
        assert_eq!(info.shape, vec![2, 4]);
        assert_eq!(ckpt.tensor_bytes("a.weight_packed").unwrap(), payload);
    }

    #[test]
    fn rejects_wrong_dialect() {
        for (patch, needle) in [
            ("/quantization_config/format", "pack-quantized"),
            (
                "/quantization_config/config_groups/group_0/weights/group_size",
                "group_size",
            ),
            (
                "/quantization_config/config_groups/group_0/weights/symmetric",
                "symmetric",
            ),
            (
                "/quantization_config/config_groups/group_0/weights/num_bits",
                "num_bits",
            ),
        ] {
            let mut cfg = good_config();
            let bad_val = match needle {
                "pack-quantized" => serde_json::json!("naive-quantized"),
                "group_size" => serde_json::json!(128),
                "symmetric" => serde_json::json!(true),
                _ => serde_json::json!(8),
            };
            *cfg.pointer_mut(patch).unwrap() = bad_val;
            let dir = temp_dir(needle);
            write_checkpoint(&dir, &cfg, &[], &[]);
            let err = HfCtCheckpoint::open(&dir).unwrap_err();
            assert!(
                err.to_string().contains(needle) || err.to_string().contains("format"),
                "dialect violation not rejected: {err}"
            );
        }
    }

    #[test]
    fn rejects_out_of_bounds_offsets() {
        let dir = temp_dir("oob");
        let mut shard = shard_bytes(&[("a.weight_scale", "BF16", &[4], &[0u8; 8])]);
        shard.truncate(shard.len() - 4);
        write_checkpoint(
            &dir,
            &good_config(),
            &[("model-00001.safetensors", shard)],
            &[("a.weight_scale", "model-00001.safetensors")],
        );
        assert!(HfCtCheckpoint::open(&dir).is_err());
    }

    #[test]
    fn rejects_shape_byte_mismatch() {
        let dir = temp_dir("mismatch");
        let shard = shard_bytes(&[("a.weight_scale", "BF16", &[5], &[0u8; 8])]);
        write_checkpoint(
            &dir,
            &good_config(),
            &[("model-00001.safetensors", shard)],
            &[("a.weight_scale", "model-00001.safetensors")],
        );
        assert!(HfCtCheckpoint::open(&dir).is_err());
    }

    #[test]
    fn rejects_index_shard_disagreement() {
        let dir = temp_dir("disagree");
        let shard = shard_bytes(&[("a.weight_packed", "I32", &[1], &[0u8; 4])]);
        write_checkpoint(
            &dir,
            &good_config(),
            &[("model-00001.safetensors", shard)],
            &[("b.weight_packed", "model-00001.safetensors")],
        );
        assert!(HfCtCheckpoint::open(&dir).is_err());
    }

    #[test]
    fn rejects_activation_ordering() {
        // actorder in the weights spec (its schema location)...
        let mut cfg = good_config();
        cfg.pointer_mut("/quantization_config/config_groups/group_0/weights")
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert("actorder".into(), serde_json::json!("group"));
        let dir = temp_dir("actorder");
        write_checkpoint(&dir, &cfg, &[], &[]);
        assert!(HfCtCheckpoint::open(&dir)
            .unwrap_err()
            .to_string()
            .contains("actorder"));
        // ...and a weight_g_idx tensor with a clean config.
        let dir = temp_dir("gidx");
        let shard = shard_bytes(&[("a.weight_g_idx", "I32", &[2], &[0u8; 8])]);
        write_checkpoint(
            &dir,
            &good_config(),
            &[("model-00001.safetensors", shard)],
            &[("a.weight_g_idx", "model-00001.safetensors")],
        );
        assert!(HfCtCheckpoint::open(&dir)
            .unwrap_err()
            .to_string()
            .contains("weight_g_idx"));
    }

    #[test]
    fn rejects_path_traversal_shard_names() {
        let dir = temp_dir("traversal");
        write_checkpoint(
            &dir,
            &good_config(),
            &[],
            &[("a.weight_packed", "../outside.safetensors")],
        );
        assert!(HfCtCheckpoint::open(&dir)
            .unwrap_err()
            .to_string()
            .contains("plain filename"));
    }

    #[test]
    fn rejects_uncovered_shard_bytes() {
        let dir = temp_dir("holes");
        // Valid entry + 4 trailing bytes no tensor indexes.
        let mut shard = shard_bytes(&[("a.weight_scale", "BF16", &[4], &[0u8; 8])]);
        shard.extend_from_slice(&[0u8; 4]);
        write_checkpoint(
            &dir,
            &good_config(),
            &[("model-00001.safetensors", shard)],
            &[("a.weight_scale", "model-00001.safetensors")],
        );
        assert!(HfCtCheckpoint::open(&dir)
            .unwrap_err()
            .to_string()
            .contains("trailing"));
    }

    #[test]
    fn rejects_unsupported_dtype() {
        let dir = temp_dir("dtype");
        let shard = shard_bytes(&[("a.weight_packed", "U8", &[4], &[0u8; 4])]);
        write_checkpoint(
            &dir,
            &good_config(),
            &[("model-00001.safetensors", shard)],
            &[("a.weight_packed", "model-00001.safetensors")],
        );
        assert!(HfCtCheckpoint::open(&dir).is_err());
    }
}
