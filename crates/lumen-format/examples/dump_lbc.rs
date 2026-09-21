//! LBC structure dumper: prints an artifact's layout and its streamed bytes per token.
//!
//! Pure file-structure parse: opens an .lbc via `LbcFile::open`, which reads
//! and CRC-checks the header, the layer index and the tokenizer section; its
//! index read is sized `layer_index_offset + num_layers *
//! MAX_LAYER_INDEX_ENTRY_SIZE`, which on a large model reaches past the
//! payload offset and pulls the first bytes of the payload into its buffer,
//! uninterpreted; it does not parse weight data. No GPU, and it prints every
//! top-level tensor and every per-layer subtensor with name/length/quant,
//! plus the derived overhead and per-pool streamed-bytes decomposition.
//!
//! Run: `cargo run -p lumen-format --example dump_lbc -- <path-to.lbc>`

use std::collections::BTreeMap;
use std::path::Path;

use lumen_format::index::TensorSlice;
use lumen_format::reader::LbcFile;
use lumen_format::{LbcHeader, ModelHyperparams, QuantScheme};

/// One verbose sub-tensor line: name, absolute file offset, length, scheme.
/// The columns are fixed — a separate instrument parses this exact format.
fn slice_line(name: &str, off: u64, len: u64, quant: QuantScheme) -> String {
    format!("        {name:<22} off={off:>12} len={len:>12} quant={quant:?}")
}

/// The per-layer section, line by line: for every layer, its own line and
/// then one line per sub-tensor it carries. A slice's offset is relative to
/// the layer blob, so the layer's offset is added to say where the bytes are
/// in the file. Zero-length slices are the format's absence sentinels and
/// have no bytes to point at, so they are left out.
fn per_layer_lines(lbc: &LbcFile) -> Vec<String> {
    let mut out = Vec::new();
    for (li, layer) in lbc.layer_indices.iter().enumerate() {
        let slices: Vec<(String, &TensorSlice)> = layer.subtensors.named_slices();
        let layer_sub_total: u64 = slices.iter().map(|(_, s)| s.length).sum();
        let type_str = if layer.subtensors.layer_type.unwrap_or(0) == 1 {
            "GDN "
        } else {
            "FULL"
        };
        out.push(format!(
            "layer {li:>3} [{type_str}] blob_len={:>12} sub_total={:>12} (off={})",
            layer.layer_length_bytes, layer_sub_total, layer.layer_offset_bytes
        ));
        out.extend(
            slices
                .iter()
                .filter(|(_, s)| s.length != 0)
                .map(|(name, s)| {
                    slice_line(name, layer.layer_offset_bytes + s.offset, s.length, s.quant)
                }),
        );
    }
    out
}

/// The `=== HEADER ===` body, line by line. `index_end` is where the index
/// the reader parsed ends: entries are variable-length, so it comes from the
/// reader rather than from any sum taken here.
///
/// Every field of `LbcHeader` is named in the destructuring below — the ones
/// printed here, the ones a later section prints, and the structural ones —
/// so a field added to the header stops it compiling until it is placed.
fn header_lines(lbc: &LbcFile) -> Vec<String> {
    let LbcHeader {
        version,
        num_layers,
        alignment,
        layer_index_offset,
        payload_offset,
        weight_tying,
        tokenizer_section_offset,
        tokenizer_section_length,
        quantization,
        // Printed by a later section of the dump, not this one.
        hyperparams: _,
        embedding: _,
        final_norm: _,
        output_proj: _,
        // Structural: the reader has already used or checked these, and a
        // dump of the layout has nothing to point at for them.
        magic: _,
        endianness: _,
        header_checksum: _,
        has_expert_index: _,
        expert_index_offset: _,
        tokenizer_section_crc32: _,
    } = &lbc.header;
    vec![
        field("version", version),
        field("num_layers", num_layers),
        field("alignment", alignment),
        field("layer_index_off", layer_index_offset),
        field("index_end", lbc.layer_index_end),
        field("payload_offset", payload_offset),
        field("weight_tying", weight_tying),
        field("tokenizer_off", tokenizer_section_offset),
        field("tokenizer_len", tokenizer_section_length),
        field("primary_quant", format!("{:?}", quantization.scheme)),
    ]
}

/// The `=== HYPERPARAMS ===` body: every field of the struct, one per line,
/// so an instrument can be pointed at any of them by name. A field added to
/// `ModelHyperparams` is added here too — the destructuring below stops
/// compiling until it is.
fn hyperparams_lines(hp: &ModelHyperparams) -> Vec<String> {
    let ModelHyperparams {
        num_layers,
        num_heads,
        num_kv_heads,
        head_dim,
        hidden_dim,
        intermediate_dim,
        vocab_size,
        max_seq_len,
        rope_params,
        num_experts,
        num_active_experts,
        norm_eps,
        rotary_dim,
        rope_neox,
        gdn,
    } = hp;
    vec![
        field("num_layers", num_layers),
        field("num_heads", num_heads),
        field("num_kv_heads", num_kv_heads),
        field("head_dim", head_dim),
        field("hidden_dim", hidden_dim),
        field("intermediate_dim", intermediate_dim),
        field("vocab_size", vocab_size),
        field("max_seq_len", max_seq_len),
        field("rope_params", format!("{rope_params:?}")),
        field("num_experts", format!("{num_experts:?}")),
        field("num_active_experts", format!("{num_active_experts:?}")),
        field("norm_eps", norm_eps),
        field("rotary_dim", format!("{rotary_dim:?}")),
        field("rope_neox", rope_neox),
        field("gdn", format!("{gdn:?}")),
    ]
}

/// One `<name> = <value>` line in the shared column width.
fn field(name: &str, value: impl std::fmt::Display) -> String {
    format!("{name:<15} = {value}")
}

fn main() {
    let path_arg = std::env::args().nth(1).expect("usage: dump_lbc <path.lbc>");
    let path = Path::new(&path_arg);

    let file_total = std::fs::metadata(path).expect("stat file").len();

    let lbc = LbcFile::open(path).expect("open lbc");
    let h = &lbc.header;
    let hp = &h.hyperparams;

    println!("=== FILE ===");
    println!("path            = {}", path.display());
    println!("file_total      = {file_total} bytes");
    println!();
    println!("=== HEADER ===");
    for line in header_lines(&lbc) {
        println!("{line}");
    }
    println!();
    println!("=== HYPERPARAMS ===");
    for line in hyperparams_lines(hp) {
        println!("{line}");
    }
    println!();

    // -- Top-level (global) tensors --
    println!("=== TOP-LEVEL TENSORS ===");
    let emb = h.embedding;
    let fnorm = h.final_norm;
    let outp = h.output_proj;
    println!(
        "embedding    off={:>14} len={:>14} quant={:?}",
        emb.offset, emb.length, emb.quant
    );
    println!(
        "final_norm   off={:>14} len={:>14} quant={:?}",
        fnorm.offset, fnorm.length, fnorm.quant
    );
    println!(
        "output_proj  off={:>14} len={:>14} quant={:?}",
        outp.offset, outp.length, outp.quant
    );
    let top_level_sum = emb.length + fnorm.length + outp.length;
    println!("top_level_sum = {top_level_sum}");
    println!();

    // -- Per-layer dump --
    // Pool accumulators (in bytes).
    let mut pool_dense_ffn: u64 = 0; // w_gate + w_up + w_down over all layers
    let mut pool_full_attn: u64 = 0; // wq/wk/wv/wo (+q/k norm, biases) over full-attn layers
    let mut pool_gdn_inproj: u64 = 0; // GDN in-proj (qkv+gate): wq/wk/wv/wo on GDN layers + ssm_a/conv/dt/beta/alpha
    let mut pool_ssm_out: u64 = 0; // ssm_out over GDN layers
    let mut pool_norms: u64 = 0; // attn_norm/ffn_norm/ssm_norm/attn_post_norm/q/k norm
    let mut pool_other: u64 = 0; // anything uncategorized (router/experts/shared etc.)

    let mut sum_layer_blob_lengths: u64 = 0; // sum of layer_length_bytes (may include intra-blob padding)
    let mut sum_subtensor_lengths: u64 = 0; // sum of every named slice length across all layers

    // quant tally across all subtensors
    let mut quant_tally: BTreeMap<String, (u64, u64)> = BTreeMap::new(); // quant -> (count, bytes)

    let mut n_gdn = 0usize;
    let mut n_full = 0usize;

    let is_norm = |name: &str| -> bool {
        name.ends_with("norm")
            || name == "attn_norm"
            || name == "ffn_norm"
            || name == "ssm_norm"
            || name == "attn_post_norm"
            || name == "attn_q_norm"
            || name == "attn_k_norm"
    };

    println!("=== PER-LAYER ===");
    for line in per_layer_lines(&lbc) {
        println!("{line}");
    }

    for layer in &lbc.layer_indices {
        sum_layer_blob_lengths += layer.layer_length_bytes;
        let is_gdn = layer.subtensors.layer_type.unwrap_or(0) == 1;
        if is_gdn {
            n_gdn += 1;
        } else {
            n_full += 1;
        }
        let slices: Vec<(String, &TensorSlice)> = layer.subtensors.named_slices();
        sum_subtensor_lengths += slices.iter().map(|(_, s)| s.length).sum::<u64>();

        for (name, s) in &slices {
            if s.length == 0 {
                continue;
            }
            let q = format!("{:?}", s.quant);
            let e = quant_tally.entry(q).or_insert((0, 0));
            e.0 += 1;
            e.1 += s.length;

            // Categorize into pools.
            let n = name.as_str();
            if n == "w_gate" || n == "w_up" || n == "w_down" {
                pool_dense_ffn += s.length;
            } else if n == "ssm_out" {
                pool_ssm_out += s.length;
            } else if is_norm(n) {
                pool_norms += s.length;
            } else if n == "wq"
                || n == "wk"
                || n == "wv"
                || n == "wo"
                || n == "bq"
                || n == "bk"
                || n == "bv"
                || n == "attn_gate"
            {
                if is_gdn {
                    pool_gdn_inproj += s.length;
                } else {
                    pool_full_attn += s.length;
                }
            } else if n.starts_with("ssm_") {
                // ssm_a / ssm_conv1d / ssm_dt / ssm_beta / ssm_alpha => GDN in-proj family
                pool_gdn_inproj += s.length;
            } else {
                pool_other += s.length;
            }
        }
    }
    println!();
    println!("n_gdn_layers  = {n_gdn}");
    println!("n_full_layers = {n_full}");
    println!();

    println!("=== QUANT TALLY (all subtensors) ===");
    for (q, (cnt, bytes)) in &quant_tally {
        println!("{q:<10} count={cnt:>6} bytes={bytes:>14}");
    }
    println!();

    // -- Aggregate accounting --
    println!("=== ACCOUNTING ===");
    println!("sum_layer_blob_lengths = {sum_layer_blob_lengths}");
    println!("sum_subtensor_lengths  = {sum_subtensor_lengths}");
    println!("top_level_sum          = {top_level_sum}");

    let all_content = sum_layer_blob_lengths + top_level_sum;
    let overhead_via_bloblen = file_total as i128 - all_content as i128;
    println!("file_total - (blob_lengths + top_level) = {overhead_via_bloblen}  (= header+index+tokenizer+alignment_padding)");

    let all_content_sub = sum_subtensor_lengths + top_level_sum;
    let overhead_via_subs = file_total as i128 - all_content_sub as i128;
    println!("file_total - (subtensor_lengths + top_level) = {overhead_via_subs}  (= header+index+tokenizer + intra-blob padding)");
    println!();

    // -- BYTES_PER_TOKEN --
    // Streamed = every subtensor of every layer (weights + tiny norms) + output_proj (lm_head)
    //            + final_norm. Embedding is GATHERED, excluded.
    let streamed_direct = sum_subtensor_lengths + outp.length + fnorm.length;
    println!("=== BYTES_PER_TOKEN (direct streamed sum) ===");
    println!("streamed = sum_subtensor_lengths + output_proj + final_norm");
    println!(
        "         = {sum_subtensor_lengths} + {} + {}",
        outp.length, fnorm.length
    );
    println!("         = {streamed_direct}");
    println!("         = {:.4} GB/tok", streamed_direct as f64 / 1e9);
    println!();

    // Formula variant: file_total - embedding - overhead(header+index+tokenizer).
    // Compute header+index+tokenizer WITHOUT alignment padding by summing the
    // three structural regions explicitly if derivable; otherwise report the
    // blob-length overhead which already excludes intra-payload gaps only via padding note.
    println!("=== BYTES_PER_TOKEN (formula: file_total - embedding - overhead) ===");
    // overhead_structural = header+index+tokenizer. We approximate via:
    //   file_total - embedding - (all streamed weights) - (alignment padding)
    // The cleanest structural overhead = file_total - (sum_layer_blob_lengths + top_level_sum),
    // MINUS any alignment padding folded into blob lengths. Use the blob-length overhead as the
    // structural+padding overhead and also the subtensor-based one.
    let bpt_formula_bloblen = file_total as i128 - emb.length as i128 - overhead_via_bloblen;
    println!(
        "using blob-length overhead {overhead_via_bloblen}: BPT = {bpt_formula_bloblen} = {:.4} GB/tok",
        bpt_formula_bloblen as f64 / 1e9
    );
    // This should equal sum_layer_blob_lengths + top_level_sum - embedding
    //   = sum_layer_blob_lengths + final_norm + output_proj  (embedding cancels)
    let bpt_bloblen_check = sum_layer_blob_lengths + fnorm.length + outp.length;
    println!("cross-check (blob_lengths + final_norm + output_proj) = {bpt_bloblen_check}");
    println!();

    // -- Per-pool decomposition (GB/tok) --
    println!("=== PER-POOL DECOMPOSITION (streamed, GB/tok) ===");
    let g = |b: u64| b as f64 / 1e9;
    println!(
        "dense_ffn (gate+up+down, all layers) = {:>14}  {:.4} GB",
        pool_dense_ffn,
        g(pool_dense_ffn)
    );
    println!(
        "gdn qkv+gate (in-proj+ssm a/conv/dt)  = {:>14}  {:.4} GB",
        pool_gdn_inproj,
        g(pool_gdn_inproj)
    );
    println!(
        "full_attn (wq/wk/wv/wo full layers)   = {:>14}  {:.4} GB",
        pool_full_attn,
        g(pool_full_attn)
    );
    println!(
        "ssm_out                               = {:>14}  {:.4} GB",
        pool_ssm_out,
        g(pool_ssm_out)
    );
    println!(
        "lm_head (output_proj)                 = {:>14}  {:.4} GB",
        outp.length,
        g(outp.length)
    );
    println!(
        "norms (attn/ffn/ssm/q/k, all layers)  = {:>14}  {:.4} GB",
        pool_norms,
        g(pool_norms)
    );
    println!(
        "final_norm                            = {:>14}  {:.4} GB",
        fnorm.length,
        g(fnorm.length)
    );
    println!(
        "other (router/experts/shared/uncat)   = {:>14}  {:.4} GB",
        pool_other,
        g(pool_other)
    );

    let pool_sum = pool_dense_ffn
        + pool_gdn_inproj
        + pool_full_attn
        + pool_ssm_out
        + outp.length
        + pool_norms
        + fnorm.length
        + pool_other;
    println!(
        "POOL SUM                              = {:>14}  {:.4} GB",
        pool_sum,
        g(pool_sum)
    );
    println!(
        "streamed_direct                       = {:>14}  {:.4} GB",
        streamed_direct,
        g(streamed_direct)
    );
    let recon = (pool_sum as i128 - streamed_direct as i128).abs();
    println!("reconciliation |pool_sum - streamed|  = {recon} bytes");
    println!();
    println!(
        "(embedding EXCLUDED from all pools: len={} quant={:?} = {:.4} GB, gathered not streamed)",
        emb.length,
        emb.quant,
        g(emb.length)
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_verbose_slice_line_has_fixed_columns() {
        assert_eq!(
            slice_line("w_gate", 4096, 2304, QuantScheme::Q4_0),
            "        w_gate                 off=        4096 len=        2304 quant=Q4_0"
        );
    }

    #[test]
    fn the_header_prints_where_the_index_ends_and_hyperparams_print_every_field() {
        let bytes = lumen_format::test_model::generate_test_model_q8_0_gdn(
            &lumen_format::test_model::TestModelQ8Config {
                num_layers: 3,
                ..Default::default()
            },
        );
        let path = std::env::temp_dir().join(format!("lumen-dump-hdr-{}.lbc", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        let lbc = LbcFile::open(&path).unwrap();

        // The printed offset is the reader's own, and it is where the index
        // ends: the bytes up to it parse, one byte fewer does not.
        assert!(header_lines(&lbc).contains(&field("index_end", lbc.layer_index_end)));
        let end = lbc.layer_index_end as usize;
        assert!(end > lbc.header.layer_index_offset as usize);
        assert!(end <= lbc.header.payload_offset as usize);
        LbcFile::from_bytes(&bytes[..end], path.clone())
            .expect("the index does not end where index_end says");
        assert!(
            LbcFile::from_bytes(&bytes[..end - 1], path.clone()).is_err(),
            "the index ends before index_end says"
        );

        // Every field of the struct, once each, in the shared column width.
        let lines = hyperparams_lines(&lbc.header.hyperparams);
        for name in [
            "num_layers",
            "num_heads",
            "num_kv_heads",
            "head_dim",
            "hidden_dim",
            "intermediate_dim",
            "vocab_size",
            "max_seq_len",
            "rope_params",
            "num_experts",
            "num_active_experts",
            "norm_eps",
            "rotary_dim",
            "rope_neox",
            "gdn",
        ] {
            let prefix = format!("{name:<15} = ");
            assert_eq!(
                lines.iter().filter(|l| l.starts_with(&prefix)).count(),
                1,
                "{name} is not printed exactly once: {lines:?}"
            );
        }
        assert_eq!(lines.len(), 15, "a field is printed that is not named here");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn the_header_section_prints_every_field_it_names_once() {
        // The test models embed no tokenizer, so one is added to set the
        // header's tokenizer offset and length.
        let bytes = lumen_format::test_model::rewrite(
            &lumen_format::test_model::generate_test_model_q8_0_gdn(
                &lumen_format::test_model::TestModelQ8Config {
                    num_layers: 3,
                    ..Default::default()
                },
            ),
            lumen_format::test_model::Tokenizer::Embedded,
            |_| {},
        );
        let path = std::env::temp_dir().join(format!("lumen-dump-flds-{}.lbc", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        let lbc = LbcFile::open(&path).unwrap();
        let h = &lbc.header;
        // The tokenizer fields are set and differ, so the two lines cannot
        // trade values unseen.
        assert!(lbc.tokenizer.is_some());
        assert!(h.tokenizer_section_offset != 0 && h.tokenizer_section_length != 0);
        assert_ne!(h.tokenizer_section_offset, h.tokenizer_section_length);

        // Every name the section prints, against the value the reader holds
        // for it — so a line that is dropped, renamed, or paired with the
        // wrong field is a failure here.
        let named: [(&str, String); 10] = [
            ("version", h.version.to_string()),
            ("num_layers", h.num_layers.to_string()),
            ("alignment", h.alignment.to_string()),
            ("layer_index_off", h.layer_index_offset.to_string()),
            ("index_end", lbc.layer_index_end.to_string()),
            ("payload_offset", h.payload_offset.to_string()),
            ("weight_tying", h.weight_tying.to_string()),
            ("tokenizer_off", h.tokenizer_section_offset.to_string()),
            ("tokenizer_len", h.tokenizer_section_length.to_string()),
            ("primary_quant", format!("{:?}", h.quantization.scheme)),
        ];
        let lines = header_lines(&lbc);
        for (name, value) in &named {
            let prefix = format!("{name:<15} = ");
            assert_eq!(
                lines.iter().filter(|l| l.starts_with(&prefix)).count(),
                1,
                "{name} is not printed exactly once: {lines:?}"
            );
            assert!(
                lines.contains(&field(name, value)),
                "{name} is not printed with the reader's value {value}: {lines:?}"
            );
        }
        assert_eq!(
            lines.len(),
            named.len(),
            "a header field is printed that is not named here: {lines:?}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn every_layer_prints_its_slices_at_their_file_offsets() {
        // A hybrid model: layer 0 is GDN, the other three full attention,
        // so a dump that stopped after the first layer of each kind would
        // leave two out.
        let bytes = lumen_format::test_model::generate_test_model_q8_0_gdn(
            &lumen_format::test_model::TestModelQ8Config {
                num_layers: 4,
                ..Default::default()
            },
        );
        let path = std::env::temp_dir().join(format!("lumen-dump-lbc-{}.lbc", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        let lbc = LbcFile::open(&path).unwrap();
        assert_eq!(lbc.layer_indices.len(), 4);
        // Every layer but the first sits at a non-zero file offset, so a
        // line printing the blob-relative offset cannot match by accident.
        assert!(lbc.layer_indices[1..]
            .iter()
            .all(|l| l.layer_offset_bytes > 0));

        // Split the output back into one block per layer.
        let lines = per_layer_lines(&lbc);
        let mut blocks: Vec<Vec<String>> = Vec::new();
        for line in lines {
            if line.starts_with("layer ") {
                blocks.push(Vec::new());
            }
            blocks
                .last_mut()
                .expect("a slice line before any layer line")
                .push(line);
        }
        assert_eq!(blocks.len(), lbc.layer_indices.len(), "one block per layer");

        for (li, layer) in lbc.layer_indices.iter().enumerate() {
            let expected: Vec<String> = layer
                .subtensors
                .named_slices()
                .iter()
                .filter(|(_, s)| s.length != 0)
                .map(|(name, s)| {
                    slice_line(name, layer.layer_offset_bytes + s.offset, s.length, s.quant)
                })
                .collect();
            assert!(!expected.is_empty(), "layer {li} carries no slices");
            assert_eq!(blocks[li][1..], expected[..], "layer {li}");
        }
        let _ = std::fs::remove_file(&path);
    }
}
