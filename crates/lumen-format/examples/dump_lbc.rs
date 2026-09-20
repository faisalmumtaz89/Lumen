//! LBC structure dumper: prints an artifact's layout and its streamed bytes per token.
//!
//! Pure file-structure parse: opens an .lbc via `LbcFile::open` (header + index
//! only, no weight payload load, no GPU) and prints every top-level tensor and
//! every per-layer subtensor with name/length/quant, plus the derived overhead
//! and per-pool streamed-bytes decomposition.
//!
//! Run: `cargo run -p lumen-format --example dump_lbc -- <path-to.lbc>`

use std::collections::BTreeMap;
use std::path::Path;

use lumen_format::index::TensorSlice;
use lumen_format::reader::LbcFile;
use lumen_format::QuantScheme;

/// One verbose sub-tensor line: name, absolute file offset, length, scheme.
/// The columns are fixed — a separate instrument parses this exact format.
fn slice_line(name: &str, off: u64, len: u64, quant: QuantScheme) -> String {
    format!("        {name:<22} off={off:>12} len={len:>12} quant={quant:?}")
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
    println!("version         = {}", h.version);
    println!("num_layers      = {}", h.num_layers);
    println!("alignment       = {}", h.alignment);
    println!("layer_index_off = {}", h.layer_index_offset);
    println!("payload_offset  = {}", h.payload_offset);
    println!("weight_tying    = {}", h.weight_tying);
    println!("tokenizer_off   = {}", h.tokenizer_section_offset);
    println!("tokenizer_len   = {}", h.tokenizer_section_length);
    println!("primary_quant   = {:?}", h.quantization.scheme);
    println!();
    println!("=== HYPERPARAMS ===");
    println!("vocab_size      = {}", hp.vocab_size);
    println!("hidden_dim      = {}", hp.hidden_dim);
    println!("intermediate    = {}", hp.intermediate_dim);
    println!("num_heads       = {}", hp.num_heads);
    println!("num_kv_heads    = {}", hp.num_kv_heads);
    println!("head_dim        = {}", hp.head_dim);
    println!("num_experts     = {:?}", hp.num_experts);
    println!("gdn             = {:?}", hp.gdn);
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
    for (li, layer) in lbc.layer_indices.iter().enumerate() {
        sum_layer_blob_lengths += layer.layer_length_bytes;
        let lt = layer.subtensors.layer_type.unwrap_or(0);
        let is_gdn = lt == 1;
        if is_gdn {
            n_gdn += 1;
        } else {
            n_full += 1;
        }
        let slices: Vec<(String, &TensorSlice)> = layer.subtensors.named_slices();

        // Compact per-layer line: type + total subtensor bytes.
        let layer_sub_total: u64 = slices.iter().map(|(_, s)| s.length).sum();
        sum_subtensor_lengths += layer_sub_total;

        let type_str = if is_gdn { "GDN " } else { "FULL" };
        println!(
            "layer {li:>3} [{type_str}] blob_len={:>12} sub_total={:>12} (off={})",
            layer.layer_length_bytes, layer_sub_total, layer.layer_offset_bytes
        );
        for (name, s) in &slices {
            if s.length == 0 {
                continue;
            }
            // The slice offset is relative to the layer blob; print where
            // the bytes are in the file.
            println!(
                "{}",
                slice_line(name, layer.layer_offset_bytes + s.offset, s.length, s.quant)
            );
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
}
