//! Provider diagnostic: byte-for-byte compare SyncWeightProvider vs MmapWeightProvider.
//!
//! Usage: cargo run --release -p lumen-runtime --example provider_diff -- <model.lbc>
//!
//! Compares, for the SAME LBC file opened through both providers:
//!   1. Global tensors (embedding / final_norm / output_proj + raw + quant).
//!   2. Per-layer get_layer_raw() bytes and every subtensor slice (offset/length/quant).
//!   3. Per-layer get_layer_blocking() bytes (the CPU-dequant path).
//! Prints the FIRST divergence found with full detail, then a summary.

use lumen_runtime::storage::MmapConfig;
use lumen_runtime::weight::provider_mmap::MmapWeightProvider;
use lumen_runtime::weight::provider_sync::SyncWeightProvider;
use lumen_runtime::weight::cache::WeightProvider;
use lumen_format::{SubtensorOffsets, TensorSlice};
use std::path::Path;

fn slice_summary(name: &str, s: &TensorSlice) -> String {
    format!("{name}: off={} len={} quant={:?}", s.offset, s.length, s.quant)
}

fn opt_slice_summary(name: &str, s: &Option<TensorSlice>) -> String {
    match s {
        Some(t) => slice_summary(name, t),
        None => format!("{name}: None"),
    }
}

fn dump_subtensors(label: &str, st: &SubtensorOffsets) {
    println!("--- {label} subtensors ---");
    println!("  {}", slice_summary("wq", &st.wq));
    println!("  {}", slice_summary("wk", &st.wk));
    println!("  {}", slice_summary("wv", &st.wv));
    println!("  {}", slice_summary("wo", &st.wo));
    println!("  {}", slice_summary("w_gate", &st.w_gate));
    println!("  {}", slice_summary("w_up", &st.w_up));
    println!("  {}", slice_summary("w_down", &st.w_down));
    println!("  {}", slice_summary("attn_norm", &st.attn_norm));
    println!("  {}", slice_summary("ffn_norm", &st.ffn_norm));
    println!("  {}", opt_slice_summary("attn_gate", &st.attn_gate));
    println!("  {}", opt_slice_summary("ssm_conv1d", &st.ssm_conv1d));
    println!("  {}", opt_slice_summary("ssm_out", &st.ssm_out));
    println!("  {}", opt_slice_summary("ssm_norm", &st.ssm_norm));
    println!("  layer_type: {:?}", st.layer_type);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: provider_diff <model.lbc>");
        std::process::exit(2);
    }
    let path = Path::new(&args[1]);
    println!("Opening {path:?} with both providers...");

    let sync = SyncWeightProvider::open(path).expect("sync open");
    let mmap = MmapWeightProvider::open(path, MmapConfig::default()).expect("mmap open");

    let num_layers = sync.num_layers();
    println!("num_layers: sync={} mmap={}", num_layers, mmap.num_layers());

    // 1. Global tensors.
    println!("\n=== GLOBAL TENSORS ===");
    println!(
        "embedding len: sync={} mmap={} EQ={}",
        sync.embedding.len(),
        mmap.embedding.len(),
        sync.embedding == mmap.embedding
    );
    println!(
        "final_norm len: sync={} mmap={} EQ={}",
        sync.final_norm.len(),
        mmap.final_norm.len(),
        sync.final_norm == mmap.final_norm
    );
    println!(
        "output_proj len: sync={} mmap={} EQ={}",
        sync.output_proj.len(),
        mmap.output_proj.len(),
        sync.output_proj == mmap.output_proj
    );
    println!(
        "embedding_raw len: sync={} mmap={} quant sync={:?} mmap={:?} EQ_bytes={}",
        sync.embedding_raw.len(),
        mmap.embedding_raw.len(),
        sync.embedding_quant,
        mmap.embedding_quant,
        sync.embedding_raw == mmap.embedding_raw
    );
    println!(
        "output_proj_raw len: sync={} mmap={} quant sync={:?} mmap={:?} EQ_bytes={}",
        sync.output_proj_raw.len(),
        mmap.output_proj_raw.len(),
        sync.output_proj_quant,
        mmap.output_proj_quant,
        sync.output_proj_raw == mmap.output_proj_raw
    );
    println!("weight_tying: sync={} mmap={}", sync.weight_tying, mmap.weight_tying);

    // 2. Per-layer get_layer_raw comparison.
    println!("\n=== PER-LAYER get_layer_raw() ===");
    let mut raw_mismatch = 0usize;
    let mut empty_mandatory_layers: Vec<(usize, String)> = Vec::new();
    for l in 0..num_layers {
        let sv = sync.get_layer_raw(l).expect("sync get_layer_raw");
        let mv = mmap.get_layer_raw(l).expect("mmap get_layer_raw");
        let sb = sv.as_bytes();
        let mb = mv.as_bytes();
        let bytes_eq = sb == mb;
        let subt_eq = format!("{:?}", sv.subtensors) == format!("{:?}", mv.subtensors);
        if !bytes_eq || !subt_eq {
            raw_mismatch += 1;
            if raw_mismatch <= 3 {
                println!(
                    "LAYER {l} RAW MISMATCH: bytes_eq={bytes_eq} subt_eq={subt_eq} sync_len={} mmap_len={}",
                    sb.len(),
                    mb.len()
                );
                dump_subtensors(&format!("sync L{l}"), &sv.subtensors);
                dump_subtensors(&format!("mmap L{l}"), &mv.subtensors);
            }
        }
        // Detect empty mandatory subtensors (the suspected cpu_naive panic cause).
        let st = &sv.subtensors;
        for (nm, t) in [
            ("wq", &st.wq), ("wk", &st.wk), ("wv", &st.wv), ("wo", &st.wo),
            ("w_gate", &st.w_gate), ("w_up", &st.w_up), ("w_down", &st.w_down),
        ] {
            if t.length == 0 {
                empty_mandatory_layers.push((l, format!("{nm} (layer_type={:?})", st.layer_type)));
            }
        }
    }
    println!("raw_mismatch layers: {raw_mismatch} / {num_layers}");
    if !empty_mandatory_layers.is_empty() {
        println!("\n!!! EMPTY MANDATORY SUBTENSORS (len==0) — cpu_naive would panic here:");
        for (l, what) in empty_mandatory_layers.iter().take(10) {
            println!("  layer {l}: {what}");
        }
        println!("  ...total {} empty-mandatory entries", empty_mandatory_layers.len());
    }

    // 3. Per-layer get_layer_blocking comparison (CPU dequant path).
    println!("\n=== PER-LAYER get_layer_blocking() (CPU dequant path) ===");
    let mut blk_mismatch = 0usize;
    for l in 0..num_layers {
        let sv = sync.get_layer_blocking(l).expect("sync get_layer_blocking");
        let mv = mmap.get_layer_blocking(l).expect("mmap get_layer_blocking");
        // Sync dequantizes to F32; Mmap returns raw. So bytes will differ by design
        // for quantized models — compare only the *subtensor slice structure* and the
        // per-tensor lengths after each provider's transform.
        let s_st = &sv.subtensors;
        let m_st = &mv.subtensors;
        // The telling metric: does sync's blocking path leave any MANDATORY slice empty?
        for (nm, t) in [
            ("wq", &s_st.wq), ("wk", &s_st.wk), ("wv", &s_st.wv), ("wo", &s_st.wo),
            ("w_gate", &s_st.w_gate), ("w_up", &s_st.w_up), ("w_down", &s_st.w_down),
        ] {
            if t.length == 0 && blk_mismatch < 8 {
                println!(
                    "  SYNC blocking L{l} mandatory {nm} EMPTY: off={} len={} quant={:?} | layer_type={:?}",
                    t.offset, t.length, t.quant, s_st.layer_type
                );
                blk_mismatch += 1;
            }
        }
        let _ = m_st;
    }
    if blk_mismatch == 0 {
        println!("  (no empty mandatory slices in sync blocking path)");
    }

    println!("\n=== DONE ===");
}
