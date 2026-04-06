//! Metal reference values for CUDA correctness debugging.
//!
//! Runs Qwen2.5-3B Q8_0 inference on the Metal backend with a fixed prompt,
//! captures intermediate tensor values at key checkpoints. These values are
//! the GROUND TRUTH that the CUDA backend must match.
//!
//! Run with:
//! ```
//! cargo test --release -p lumen-runtime --test cuda_correctness_reference -- --nocapture metal_reference_layer0
//! ```

#[cfg(target_os = "macos")]
mod metal_reference {
    use lumen_runtime::compute::ComputeBackend;
    use lumen_runtime::config::RuntimeConfig;
    use lumen_runtime::engine::{InferenceEngine, SamplingParams, StopCondition};
    use lumen_runtime::kv::{KvCache, KvCacheConfig, KvPrecision};
    use lumen_runtime::metal::MetalF32Backend;
    use lumen_runtime::pipeline::PipelineMode;
    use lumen_runtime::storage::MmapConfig;
    use lumen_runtime::weight::provider_mmap::MmapWeightProvider;
    use lumen_runtime::WeightProvider;
    use lumen_format::quantization::QuantScheme;
    use std::path::Path;

    const MODEL_PATH: &str = "/tmp/lumen-bench/qwen2.5-3b-q8_0.lbc";
    // "<|im_start|>system\nYou are" -- Qwen2.5 chat template tokens
    const PROMPT_TOKENS: &[u32] = &[151644, 8948, 198, 2610, 525];

    /// Helper: print first N values of an f32 slice with 6-digit precision.
    fn print_values(label: &str, vals: &[f32], n: usize) {
        let n = n.min(vals.len());
        eprint!("{label} [");
        for i in 0..n {
            if i > 0 { eprint!(", "); }
            eprint!("{:.6}", vals[i]);
        }
        eprintln!("]");
    }

    /// Set up Metal backend with MmapWeightProvider for a real LBC model.
    /// Returns (provider, backend, hyperparams) with the backend fully initialized.
    fn setup_metal(model_path: &str) -> (
        MmapWeightProvider,
        MetalF32Backend,
        lumen_format::ModelHyperparams,
    ) {
        let path = Path::new(model_path);
        assert!(path.exists(), "Model file not found: {model_path}");

        let mmap_config = MmapConfig {
            prefetch_window: 2,
            advise_sequential: true,
            release_with_dontneed: false,
        };
        let provider = MmapWeightProvider::open(path, mmap_config)
            .expect("Failed to open model");

        let hyperparams = provider.lbc().header.hyperparams;
        let mut hp = hyperparams;
        hp.max_seq_len = 64;

        let mut metal = MetalF32Backend::new()
            .expect("Metal backend not available");

        metal.set_global_tensors(
            provider.embedding.clone(),
            provider.final_norm.clone(),
            provider.output_proj.clone(),
        );
        if matches!(provider.output_proj_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16)
            && !provider.output_proj_raw.is_empty()
        {
            metal.set_output_proj_raw(provider.output_proj_raw.clone(), provider.output_proj_quant);
        }
        if matches!(provider.embedding_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16)
            && !provider.embedding_raw.is_empty()
        {
            metal.set_embedding_raw(provider.embedding_raw.clone(), provider.embedding_quant);
        }
        if provider.weight_tying {
            metal.set_weight_tying(true);
        }

        metal.init(&hp).expect("Metal init failed");

        (provider, metal, hp)
    }

    /// PHASE 1: Capture Metal reference values at layer boundaries.
    ///
    /// Uses the streaming (non-GPU-resident) path with compute_layer per layer.
    /// After each compute_layer, reads back the GPU x_buf via readback_x_buf()
    /// to capture the actual hidden state (the ActivationBuffer returned by
    /// embed_token/compute_layer is a CPU placeholder -- real data stays on GPU).
    #[test]
    fn metal_reference_layer0() {
        let (provider, metal, hp) = setup_metal(MODEL_PATH);

        let num_layers = hp.num_layers as usize;
        let hidden_dim = hp.hidden_dim as usize;

        eprintln!("=== Metal Reference Values for CUDA Correctness ===");
        eprintln!("Model: {MODEL_PATH}");
        eprintln!("Layers: {num_layers}, hidden_dim: {hidden_dim}");
        eprintln!("Heads: {}, KV heads: {}, head_dim: {}",
            hp.num_heads, hp.num_kv_heads, hp.head_dim);
        eprintln!("Vocab: {}, inter_dim: {}", hp.vocab_size, hp.intermediate_dim);
        eprintln!("Prompt tokens: {PROMPT_TOKENS:?}");
        eprintln!();

        // Initialize KV cache (F32 precision for maximum clarity)
        let mut kv = KvCache::new(KvCacheConfig {
            max_seq_len: 64,
            num_layers,
            num_kv_heads: hp.num_kv_heads as usize,
            head_dim: hp.head_dim as usize,
            precision: KvPrecision::F32,
        }).expect("KV cache init failed");

        metal.reset_recurrent_state();

        // Process each prompt token through all layers.
        // Use readback_x_buf() to read GPU-side hidden state after each step.
        let mut last_x = None;
        for (tok_idx, &token_id) in PROMPT_TOKENS.iter().enumerate() {
            let seq_pos = kv.seq_len();
            eprintln!("--- Token {tok_idx}: id={token_id}, seq_pos={seq_pos} ---");

            // Step 1: Embedding lookup (GPU-side)
            let mut x = metal.embed_token(token_id)
                .expect("embed_token failed");

            // Read back the actual GPU embedding via x_buf
            let gpu_embed = metal.readback_x_buf();
            print_values(&format!("CHECKPOINT embed_token_{tok_idx} (id={token_id}):"), &gpu_embed, 10);
            let l2: f32 = gpu_embed.iter().map(|v| v * v).sum::<f32>().sqrt();
            eprintln!("  L2 norm: {l2:.6}");

            // Step 2: Run through each layer
            for layer in 0..num_layers {
                let (layer_view, _hit) = match provider.try_get_layer(layer) {
                    Some(view) => (view, true),
                    None => (provider.get_layer_blocking(layer)
                        .expect("get_layer_blocking failed"), false),
                };

                let mut kv_view = kv.view_mut(layer)
                    .expect("kv view_mut failed");

                metal.compute_layer(layer, &mut x, &layer_view, Some(&mut kv_view), seq_pos)
                    .expect("compute_layer failed");

                kv.commit_view(kv_view).expect("commit_view failed");

                // Read back GPU x_buf after this layer
                let gpu_vals = metal.readback_x_buf();

                // Print checkpoints for layers 0, 1, and last
                if layer == 0 {
                    print_values(&format!("CHECKPOINT tok{tok_idx}_post_layer0:"), &gpu_vals, 10);
                    let l2: f32 = gpu_vals.iter().map(|v| v * v).sum::<f32>().sqrt();
                    eprintln!("  L2 norm: {l2:.6}");
                }
                if layer == 1 {
                    print_values(&format!("CHECKPOINT tok{tok_idx}_post_layer1:"), &gpu_vals, 10);
                }
                if layer == num_layers - 1 {
                    print_values(&format!("CHECKPOINT tok{tok_idx}_post_layer{layer} (final):"), &gpu_vals, 10);
                    let l2: f32 = gpu_vals.iter().map(|v| v * v).sum::<f32>().sqrt();
                    eprintln!("  L2 norm: {l2:.6}");
                }
            }

            kv.advance_seq_len().expect("advance_seq_len failed");
            last_x = Some(x);
        }

        // Step 3: Compute final logits from the last token's hidden state
        eprintln!();
        eprintln!("--- Final logits (after all {} prompt tokens) ---", PROMPT_TOKENS.len());

        let final_x = last_x.unwrap();
        let logits = metal.compute_final(&final_x)
            .expect("compute_final failed");

        eprintln!("Vocab size: {}", logits.vocab_size());

        // Top-5 logits
        let mut indexed: Vec<(usize, f32)> = logits.data.iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));

        eprintln!("CHECKPOINT top5_logits:");
        for i in 0..5.min(indexed.len()) {
            let (token_id, logit_val) = indexed[i];
            eprintln!("  rank {}: token_id={token_id}, logit={logit_val:.6}", i + 1);
        }

        let argmax_token = logits.argmax();
        eprintln!("CHECKPOINT argmax: token_id={argmax_token}, logit={:.6}",
            logits.data[argmax_token]);

        // Bottom-5 for sanity
        eprintln!("CHECKPOINT bottom5_logits:");
        let n = indexed.len();
        for i in (n.saturating_sub(5))..n {
            let (token_id, logit_val) = indexed[i];
            eprintln!("  rank {}: token_id={token_id}, logit={logit_val:.6}", i + 1);
        }

        eprintln!();
        eprintln!("=== End of Metal Reference Values ===");
    }

    /// Verify that running generate() with greedy sampling produces consistent results.
    /// Uses the GPU-resident prefill path (batched_prefill=true).
    #[test]
    fn metal_reference_generate_greedy() {
        let (provider, metal, hp) = setup_metal(MODEL_PATH);

        metal.preload_weights_gpu_resident(&provider)
            .expect("GPU-resident preload failed");

        let config = RuntimeConfig {
            pipeline_mode: PipelineMode::MinMem,
            prefetch_distance: 2,
            kv_precision: KvPrecision::F32,
            max_seq_len: 64,
            collect_per_layer_timings: false,
        };

        let engine = InferenceEngine::new(config, hp);

        let stop = StopCondition::MaxTokens(10);
        let sampling = SamplingParams {
            temperature: 0.0,
            seed: Some(42),
        };

        let result = engine.generate(
            PROMPT_TOKENS,
            &provider,
            &metal as &dyn ComputeBackend,
            &stop,
            &sampling,
        ).expect("generate failed");

        eprintln!("=== Metal Reference: Greedy Generation ===");
        eprintln!("Prompt: {PROMPT_TOKENS:?}");
        eprintln!("Generated {} tokens: {:?}", result.tokens.len(), result.tokens);
        eprintln!("Prefill: {:.1} tok/s", result.metrics.prefill_tokens_per_sec);
        eprintln!("Decode: {:.1} tok/s", result.metrics.decode_tokens_per_sec);
        eprintln!("=== End ===");

        assert_eq!(result.tokens.len(), 10, "expected 10 generated tokens");
        let vocab_size = hp.vocab_size;
        for &tok in &result.tokens {
            assert!(tok < vocab_size,
                "token {tok} >= vocab_size {vocab_size}");
        }
    }
}

/// CPU naive backend full generation — compare against Metal reference.
/// If CPU produces same tokens as Metal → CUDA has a unique bug.
/// If CPU differs from Metal → precision difference between Q8_0 paths.
#[test]
#[ignore]
fn cpu_naive_full_generation() {
    use lumen_runtime::compute::ComputeBackend;
    use lumen_runtime::compute::cpu_naive::NaiveF32Backend;
    use lumen_runtime::engine::{InferenceEngine, SamplingParams, StopCondition};
    use lumen_runtime::kv::{KvCache, KvCacheConfig, KvPrecision};
    use lumen_runtime::weight::provider_mmap::MmapWeightProvider;
    use lumen_runtime::storage::MmapConfig;
    use lumen_runtime::WeightProvider;
    use lumen_format::quantization::QuantScheme;
    use std::path::Path;

    let lbc_path = "/tmp/lumen-bench/qwen2.5-3b-q8_0.lbc";
    if !Path::new(lbc_path).exists() {
        eprintln!("SKIP: model not found at {lbc_path}");
        return;
    }

    let mmap_config = MmapConfig {
        prefetch_window: 0,
        advise_sequential: true,
        release_with_dontneed: false,
    };
    let provider = MmapWeightProvider::open(Path::new(lbc_path), mmap_config)
        .expect("open LBC");
    let hp = provider.lbc().header.hyperparams;

    let mut backend = NaiveF32Backend::new();
    backend.set_global_tensors(
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
    );
    if matches!(provider.output_proj_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16)
        && !provider.output_proj_raw.is_empty()
    {
        backend.set_output_proj_raw(provider.output_proj_raw.clone(), provider.output_proj_quant);
    }
    if matches!(provider.embedding_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16)
        && !provider.embedding_raw.is_empty()
    {
        backend.set_embedding_raw(provider.embedding_raw.clone(), provider.embedding_quant);
    }
    if provider.weight_tying {
        backend.set_weight_tying(true);
    }
    backend.init(&hp).expect("init");

    use lumen_runtime::config::RuntimeConfig;
    use lumen_runtime::pipeline::PipelineMode;
    let config = RuntimeConfig { pipeline_mode: PipelineMode::default(), prefetch_distance: 0, kv_precision: KvPrecision::F32, max_seq_len: 64, collect_per_layer_timings: false };
    let hp_copy = hp;
    let engine = InferenceEngine::new(config, hp_copy);
    let prompt = vec![151644u32, 8948, 198, 2610, 525];
    let stop = StopCondition::MaxTokens(10);
    let sampling = SamplingParams { temperature: 0.0, seed: Some(42) };

    eprintln!("Running CPU naive full generation (5 prompt + 10 greedy)...");
    let result = engine.generate(
        &prompt, &provider, &backend, &stop, &sampling,
    ).expect("generate");

    eprintln!("CPU naive generated tokens: {:?}", result.tokens);
    eprintln!("Metal reference tokens:     [458, 6203, 304, 1667, 279, 54364, 14817, 17458, 13, 358]");

    let metal_ref = [458u32, 6203, 304, 1667, 279, 54364, 14817, 17458, 13, 358];
    let mut all_match = true;
    for (i, (&cpu, &metal)) in result.tokens.iter().zip(metal_ref.iter()).enumerate() {
        let status = if cpu == metal { "MATCH" } else { all_match = false; "MISMATCH" };
        eprintln!("  [{}]: CPU={}, Metal={} ({})", i, cpu, metal, status);
    }
    if all_match {
        eprintln!("CPU naive matches Metal reference PERFECTLY");
    } else {
        eprintln!("CPU naive DIFFERS from Metal reference");
    }
}
