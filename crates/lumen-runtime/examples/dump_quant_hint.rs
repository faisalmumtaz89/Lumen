//! Diagnostic: print exactly what the runtime-default resolvers see for a given
//! LBC — the `output_proj_quant` fed to `set_model_dense_quant`, the primary
//! quant scheme, the layer count fed to `set_model_block_count`, the MoE flag,
//! and the resolved defaults that depend on them.
//!
//! Usage: cargo run --release --example dump_quant_hint -- <path-to.lbc>

use lumen_runtime::runtime_defaults as rd;
use lumen_runtime::weight::provider_sync::SyncWeightProvider;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: dump_quant_hint <lbc>");
    let provider = SyncWeightProvider::open(std::path::Path::new(&path)).expect("open LBC");
    let opq = provider.output_proj_quant;
    let num_layers = provider.lbc().header.hyperparams.num_layers;
    let num_experts = provider.lbc().header.hyperparams.num_experts.unwrap_or(0);
    let is_moe = num_experts > 0;

    let primary = provider.lbc().header.quantization.scheme;

    // Replicate the binary's setter calls (same order as lumen-server main).
    rd::set_model_dense_quant(opq);
    rd::set_model_primary_quant(primary);
    rd::set_model_block_count(num_layers);
    rd::set_model_is_moe(is_moe);

    println!("LBC: {path}");
    println!("  header.quantization.scheme (PRIMARY/bulk)     = {primary:?}");
    println!("  output_proj_quant (-> set_model_dense_quant) = {opq:?}");
    println!("  num_layers        (-> set_model_block_count)  = {num_layers}");
    println!("  num_experts                                   = {num_experts} (is_moe={is_moe})");
    println!(
        "  model_dense_quant()                           = {:?}",
        rd::model_dense_quant_pub()
    );
    println!(
        "  gdn_decode_via_prefill_default()              = {}",
        rd::gdn_decode_via_prefill_default()
    );
}
