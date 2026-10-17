//! A prompt longer than one prefill slice runs through the layers slice by slice, so the
//! batch-sized scratch is allocated for one slice, not for the whole prompt.

use crate::compute::ComputeBackend;
use crate::kv::{KvCache, KvCacheConfig, KvPrecision};
use crate::metal::prefill::PREFILL_SLICE_TOKENS;
use crate::metal::MetalF32Backend;
use crate::weight::provider_sync::SyncWeightProvider;
use lumen_format::test_model::{generate_test_model, TestModelConfig};

#[test]
fn a_prompt_longer_than_a_slice_sizes_the_batch_scratch_for_one_slice() {
    let total = 2 * PREFILL_SLICE_TOKENS + 5;
    let data = generate_test_model(&TestModelConfig {
        max_seq_len: total as u32,
        ..TestModelConfig::default()
    });
    let dir =
        std::env::temp_dir().join(format!("lumen_metal_prefill_slices_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.lbc");
    std::fs::write(&path, &data).unwrap();
    let provider = SyncWeightProvider::open(&path).unwrap();
    let hp = provider.lbc().header.hyperparams;

    let mut backend = MetalF32Backend::new().unwrap();
    backend.set_global_tensors(
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
    );
    backend.init(&hp).unwrap();
    let mut kv = KvCache::new(KvCacheConfig {
        max_seq_len: total,
        num_layers: hp.num_layers as usize,
        num_kv_heads: hp.num_kv_heads as usize,
        head_dim: hp.head_dim as usize,
        precision: KvPrecision::F32,
    })
    .unwrap();
    let prompt: Vec<u32> = (0..total)
        .map(|i| ((i * 7919 + 13) % hp.vocab_size as usize) as u32)
        .collect();

    backend.prefill(&prompt, &provider, &mut kv).unwrap();
    assert_eq!(kv.seq_len(), total);
    let max_batch = backend
        .scratch
        .lock()
        .unwrap()
        .as_ref()
        .unwrap()
        .current_max_batch;
    assert_eq!(max_batch, PREFILL_SLICE_TOKENS);
    std::fs::remove_dir_all(&dir).unwrap();
}
