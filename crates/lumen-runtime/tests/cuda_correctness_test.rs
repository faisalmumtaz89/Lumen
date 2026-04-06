//! CUDA vs Metal correctness test for Qwen2.5-3B Q8_0.
//!
//! Loads a real Qwen2.5-3B Q8_0 LBC model, runs inference token-by-token through
//! all 36 layers on the CUDA backend, and compares intermediate activations and
//! final logits against Metal ground-truth reference values.
//!
//! The Metal reference values were captured on Apple M3 Ultra using the
//! `cuda_correctness_reference.rs` test with F32 KV precision and streaming
//! (non-GPU-resident) path.
//!
//! ## Metal Reference Values (Ground Truth)
//! ```text
//! embed token 0 (151644): [-0.007917, -0.000546, -0.001911, ...]
//! post layer 0, token 0:  [-0.823990, 0.137963, 0.185958, ...]
//! post layer 35, token 4: [1.277048, -1.153568, -1.833645, ...]
//! argmax after full forward: token 458, logit 24.437672
//! greedy 10 tokens: [458, 6203, 304, 1667, 279, 54364, 14817, 17458, 13, 358]
//! ```
//!
//! Run on Modal A100:
//! ```
//! modal run modal/run_cuda_correctness.py
//! ```
//!
//! Or directly on a CUDA machine with the LBC at /tmp/models/qwen2.5-3b-instruct-q8_0.lbc:
//! ```
//! cargo test -p lumen-runtime --features cuda --release \
//!     --test cuda_correctness_test -- --nocapture cuda_vs_metal_layer0
//! ```

#![cfg(feature = "cuda")]

mod cuda_correctness {
    use lumen_format::quantization::QuantScheme;
    use lumen_runtime::compute::ComputeBackend;
    use lumen_runtime::cuda::CudaBackend;
    use lumen_runtime::kv::{KvCache, KvCacheConfig, KvPrecision};
    use lumen_runtime::weight::provider_sync::SyncWeightProvider;
    use lumen_runtime::WeightProvider;
    use std::path::Path;

    /// LBC model path. On Modal, the run_cuda_correctness.py script downloads
    /// the GGUF and converts it to LBC at this path.
    const MODEL_PATH: &str = "/tmp/models/qwen2.5-3b-instruct-q8_0.lbc";

    /// The same prompt tokens used in the Metal reference test.
    /// "<|im_start|>system\nYou are" in Qwen2.5 tokenizer.
    const PROMPT_TOKENS: &[u32] = &[151644, 8948, 198, 2610, 525];

    /// Tolerance for comparing CUDA vs Metal activations.
    /// Q8_0 dequantization introduces per-element noise; 1e-3 is generous but
    /// catches gross errors (wrong weights, transposed dims, broken kernels).
    const TOLERANCE: f32 = 1e-3;

    // -----------------------------------------------------------------------
    // Metal reference checkpoints (ground truth from cuda_correctness_reference)
    // -----------------------------------------------------------------------

    /// First 10 values of embedding lookup for token 151644.
    const REF_EMBED_TOK0: &[f32] = &[
        -0.007917, -0.000546, -0.001911, // only 3 provided in spec
    ];

    /// First 3 values after layer 0, token 0.
    const REF_POST_LAYER0_TOK0: &[f32] = &[
        -0.823990, 0.137963, 0.185958,
    ];

    /// First 3 values after layer 35 (final), token 4.
    const REF_POST_LAYER35_TOK4: &[f32] = &[
        1.277048, -1.153568, -1.833645,
    ];

    /// Expected argmax token after processing all 5 prompt tokens.
    const REF_ARGMAX_TOKEN: u32 = 458;
    const REF_ARGMAX_LOGIT: f32 = 24.437672;

    /// Expected greedy generation sequence (10 tokens).
    const REF_GREEDY_TOKENS: &[u32] = &[458, 6203, 304, 1667, 279, 54364, 14817, 17458, 13, 358];

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Print first N values of an f32 slice.
    fn print_values(label: &str, vals: &[f32], n: usize) {
        let n = n.min(vals.len());
        eprint!("{label} [");
        for i in 0..n {
            if i > 0 {
                eprint!(", ");
            }
            eprint!("{:.6}", vals[i]);
        }
        eprintln!("]");
    }

    /// Compare CUDA values against Metal reference, returning (matches, max_diff).
    /// Only compares up to `reference.len()` elements.
    fn check_reference(
        label: &str,
        cuda_vals: &[f32],
        reference: &[f32],
        tol: f32,
    ) -> (bool, f32) {
        let n = reference.len().min(cuda_vals.len());
        let mut max_diff: f32 = 0.0;
        let mut all_match = true;
        for i in 0..n {
            let diff = (cuda_vals[i] - reference[i]).abs();
            max_diff = max_diff.max(diff);
            if diff > tol {
                all_match = false;
            }
        }
        if all_match {
            eprintln!("  {label}: MATCH (max_diff={max_diff:.2e}, tol={tol:.1e})");
        } else {
            eprintln!("  {label}: MISMATCH (max_diff={max_diff:.2e}, tol={tol:.1e})");
            for i in 0..n {
                let diff = (cuda_vals[i] - reference[i]).abs();
                if diff > tol {
                    eprintln!(
                        "    [{i}]: CUDA={:.6}, Metal={:.6}, diff={diff:.2e}",
                        cuda_vals[i], reference[i]
                    );
                }
            }
        }
        (all_match, max_diff)
    }

    /// Set up the CUDA backend from a real LBC file using SyncWeightProvider.
    fn setup_cuda(model_path: &str) -> (SyncWeightProvider, CudaBackend, lumen_format::ModelHyperparams) {
        let path = Path::new(model_path);
        assert!(path.exists(), "Model file not found: {model_path}");

        let provider = SyncWeightProvider::open(path)
            .expect("Failed to open LBC model");

        let mut hp = provider.lbc().header.hyperparams;
        hp.max_seq_len = 64; // small for correctness testing

        let mut cuda = CudaBackend::new(0)
            .expect("CUDA backend not available");

        cuda.set_global_tensors(
            provider.embedding.clone(),
            provider.final_norm.clone(),
            provider.output_proj.clone(),
        );

        // Set raw quantized tensors so CUDA uses native Q8_0 kernels.
        if matches!(provider.output_proj_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16)
            && !provider.output_proj_raw.is_empty()
        {
            cuda.set_output_proj_raw(provider.output_proj_raw.clone(), provider.output_proj_quant);
        }
        if matches!(provider.embedding_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16)
            && !provider.embedding_raw.is_empty()
        {
            cuda.set_embedding_raw(provider.embedding_raw.clone(), provider.embedding_quant);
        }
        if provider.weight_tying {
            cuda.set_weight_tying(true);
        }

        cuda.init(&hp).expect("CUDA init failed");

        (provider, cuda, hp)
    }

    // -----------------------------------------------------------------------
    // Main test: layer-by-layer CUDA vs Metal comparison
    // -----------------------------------------------------------------------

    /// Process all 5 prompt tokens through all 36 layers on CUDA, comparing
    /// intermediate hidden states against Metal reference values at each checkpoint.
    ///
    /// CUDA compute_layer writes results back to the host-side ActivationBuffer
    /// (unlike Metal which keeps data on GPU), so x.as_f32_slice() gives us the
    /// correct hidden state after each layer without needing a special readback.
    #[test]
    fn cuda_vs_metal_layer0() {
        let (provider, cuda, hp) = setup_cuda(MODEL_PATH);

        let num_layers = hp.num_layers as usize;
        let hidden_dim = hp.hidden_dim as usize;

        eprintln!("=== CUDA vs Metal Correctness Test ===");
        eprintln!("Model: {MODEL_PATH}");
        eprintln!("Layers: {num_layers}, hidden_dim: {hidden_dim}");
        eprintln!(
            "Heads: {}, KV heads: {}, head_dim: {}",
            hp.num_heads, hp.num_kv_heads, hp.head_dim
        );
        eprintln!("Vocab: {}, inter_dim: {}", hp.vocab_size, hp.intermediate_dim);
        eprintln!("Prompt tokens: {PROMPT_TOKENS:?}");
        eprintln!();

        // KV cache with F32 precision (same as Metal reference)
        let mut kv = KvCache::new(KvCacheConfig {
            max_seq_len: 64,
            num_layers,
            num_kv_heads: hp.num_kv_heads as usize,
            head_dim: hp.head_dim as usize,
            precision: KvPrecision::F32,
        })
        .expect("KV cache init failed");

        cuda.reset_recurrent_state();

        let mut first_divergence: Option<String> = None;
        let mut all_match = true;

        // Track last activation for compute_final
        let mut last_x = None;

        for (tok_idx, &token_id) in PROMPT_TOKENS.iter().enumerate() {
            let seq_pos = kv.seq_len();
            eprintln!("--- Token {tok_idx}: id={token_id}, seq_pos={seq_pos} ---");

            // 1. Embedding lookup
            let mut x = cuda.embed_token(token_id).expect("embed_token failed");
            let embed_vals = x.as_f32_slice().to_vec();
            print_values(
                &format!("CHECKPOINT embed_token_{tok_idx} (id={token_id}):"),
                &embed_vals,
                10,
            );
            let l2: f32 = embed_vals.iter().map(|v| v * v).sum::<f32>().sqrt();
            eprintln!("  L2 norm: {l2:.6}");

            // Check embedding for token 0 against Metal reference
            if tok_idx == 0 {
                let (ok, _) = check_reference(
                    "embed_tok0 vs Metal",
                    &embed_vals,
                    REF_EMBED_TOK0,
                    TOLERANCE,
                );
                if !ok && first_divergence.is_none() {
                    first_divergence = Some(format!(
                        "embed_token_0: CUDA embedding diverges from Metal"
                    ));
                    all_match = false;
                }
            }

            // 2. Process through each layer
            for layer in 0..num_layers {
                let layer_view = provider
                    .get_layer_blocking(layer)
                    .expect("get_layer_blocking failed");

                let mut kv_view = kv.view_mut(layer).expect("kv view_mut failed");

                cuda.compute_layer(layer, &mut x, &layer_view, Some(&mut kv_view), seq_pos)
                    .expect(&format!("compute_layer {layer} failed"));

                kv.commit_view(kv_view).expect("commit_view failed");

                // CUDA writes results back to host x, so we can read directly.
                let layer_vals = x.as_f32_slice().to_vec();

                // Print checkpoints for layers 0, 1, and last
                if layer == 0 {
                    print_values(
                        &format!("CHECKPOINT tok{tok_idx}_post_layer0:"),
                        &layer_vals,
                        10,
                    );
                    let l2: f32 = layer_vals.iter().map(|v| v * v).sum::<f32>().sqrt();
                    eprintln!("  L2 norm: {l2:.6}");

                    // Check layer 0, token 0 against Metal reference
                    if tok_idx == 0 {
                        let (ok, _) = check_reference(
                            "tok0_post_layer0 vs Metal",
                            &layer_vals,
                            REF_POST_LAYER0_TOK0,
                            TOLERANCE,
                        );
                        if !ok && first_divergence.is_none() {
                            first_divergence = Some(format!(
                                "token 0, post layer 0: CUDA diverges from Metal"
                            ));
                            all_match = false;
                        }
                    }
                }
                if layer == 1 {
                    print_values(
                        &format!("CHECKPOINT tok{tok_idx}_post_layer1:"),
                        &layer_vals,
                        10,
                    );
                }
                if layer == num_layers - 1 {
                    print_values(
                        &format!("CHECKPOINT tok{tok_idx}_post_layer{layer} (final):"),
                        &layer_vals,
                        10,
                    );
                    let l2: f32 = layer_vals.iter().map(|v| v * v).sum::<f32>().sqrt();
                    eprintln!("  L2 norm: {l2:.6}");

                    // Check final layer, token 4 against Metal reference
                    if tok_idx == 4 {
                        let (ok, _) = check_reference(
                            "tok4_post_layer35 vs Metal",
                            &layer_vals,
                            REF_POST_LAYER35_TOK4,
                            TOLERANCE,
                        );
                        if !ok && first_divergence.is_none() {
                            first_divergence = Some(format!(
                                "token 4, post layer {layer}: CUDA diverges from Metal"
                            ));
                            all_match = false;
                        }
                    }
                }
            }

            kv.advance_seq_len().expect("advance_seq_len failed");
            last_x = Some(x);
        }

        // 3. Compute final logits from the last token's hidden state
        eprintln!();
        eprintln!(
            "--- Final logits (after all {} prompt tokens) ---",
            PROMPT_TOKENS.len()
        );

        let final_x = last_x.unwrap();
        let logits = cuda.compute_final(&final_x).expect("compute_final failed");

        eprintln!("Vocab size: {}", logits.vocab_size());

        // Top-5 logits
        let mut indexed: Vec<(usize, f32)> = logits
            .data
            .iter()
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
        let argmax_logit = logits.data[argmax_token];
        eprintln!("CHECKPOINT argmax: token_id={argmax_token}, logit={argmax_logit:.6}");

        // Bottom-5 for sanity
        eprintln!("CHECKPOINT bottom5_logits:");
        let n = indexed.len();
        for i in (n.saturating_sub(5))..n {
            let (token_id, logit_val) = indexed[i];
            eprintln!("  rank {}: token_id={token_id}, logit={logit_val:.6}", i + 1);
        }

        // Check argmax against Metal reference
        eprintln!();
        if argmax_token as u32 == REF_ARGMAX_TOKEN {
            eprintln!(
                "ARGMAX: MATCH (token {argmax_token}, logit {argmax_logit:.6} vs Metal {REF_ARGMAX_LOGIT:.6})"
            );
            let logit_diff = (argmax_logit - REF_ARGMAX_LOGIT).abs();
            eprintln!("  Logit diff: {logit_diff:.6}");
        } else {
            eprintln!(
                "ARGMAX: MISMATCH! CUDA={argmax_token} (logit {argmax_logit:.6}), Metal={REF_ARGMAX_TOKEN} (logit {REF_ARGMAX_LOGIT:.6})"
            );
            // Show where Metal's expected token ranks in CUDA output
            if let Some(pos) = indexed.iter().position(|(tid, _)| *tid == REF_ARGMAX_TOKEN as usize)
            {
                let (_, logit_at_ref) = indexed[pos];
                eprintln!(
                    "  Metal's token {REF_ARGMAX_TOKEN} is at CUDA rank {} with logit {logit_at_ref:.6}",
                    pos + 1
                );
            }
            if first_divergence.is_none() {
                first_divergence = Some(format!(
                    "argmax mismatch: CUDA={argmax_token}, Metal={REF_ARGMAX_TOKEN}"
                ));
                all_match = false;
            }
        }

        // 4. Greedy generation: continue from the prefill state
        eprintln!();
        eprintln!("--- Greedy generation (10 tokens from prefill state) ---");

        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut _gen_x = None;

        // First generated token is argmax of the logits we just computed
        let first_gen = argmax_token as u32;
        generated_tokens.push(first_gen);
        eprintln!("  gen[0] = {first_gen}");

        // Generate 9 more tokens autoregressively
        for gen_idx in 1..10 {
            let prev_token = generated_tokens[gen_idx - 1];
            let seq_pos = kv.seq_len();

            let mut gx = cuda.embed_token(prev_token).expect("embed_token failed");

            for layer in 0..num_layers {
                let layer_view = provider
                    .get_layer_blocking(layer)
                    .expect("get_layer_blocking failed");
                let mut kv_view = kv.view_mut(layer).expect("kv view_mut failed");

                cuda.compute_layer(layer, &mut gx, &layer_view, Some(&mut kv_view), seq_pos)
                    .expect(&format!("compute_layer {layer} failed (gen {gen_idx})"));

                kv.commit_view(kv_view).expect("commit_view failed");
            }

            kv.advance_seq_len().expect("advance_seq_len failed");

            let gen_logits = cuda.compute_final(&gx).expect("compute_final failed");
            let gen_argmax = gen_logits.argmax() as u32;
            generated_tokens.push(gen_argmax);
            eprintln!("  gen[{gen_idx}] = {gen_argmax}");

            _gen_x = Some(gx);
        }

        eprintln!();
        eprintln!("CHECKPOINT greedy_tokens: {:?}", generated_tokens);
        eprintln!("Metal reference tokens:   {:?}", REF_GREEDY_TOKENS);

        let tokens_match = generated_tokens == REF_GREEDY_TOKENS;
        if tokens_match {
            eprintln!("GREEDY GENERATION: MATCH (all 10 tokens identical)");
        } else {
            eprintln!("GREEDY GENERATION: MISMATCH");
            for i in 0..10.min(generated_tokens.len()) {
                let cuda_tok = generated_tokens[i];
                let metal_tok = REF_GREEDY_TOKENS[i];
                if cuda_tok == metal_tok {
                    eprintln!("  [{i}]: {cuda_tok} == {metal_tok} (match)");
                } else {
                    eprintln!("  [{i}]: CUDA={cuda_tok}, Metal={metal_tok} (MISMATCH)");
                    if first_divergence.is_none() {
                        first_divergence = Some(format!(
                            "greedy token [{i}]: CUDA={cuda_tok}, Metal={metal_tok}"
                        ));
                        all_match = false;
                    }
                }
            }
        }

        // Summary
        eprintln!();
        eprintln!("=== SUMMARY ===");
        if all_match {
            eprintln!("ALL CHECKPOINTS MATCH within tolerance {TOLERANCE:.1e}");
        } else {
            eprintln!("DIVERGENCE DETECTED");
            if let Some(ref div) = first_divergence {
                eprintln!("First divergence: {div}");
            }
        }
        eprintln!("=== End of CUDA vs Metal Correctness Test ===");

        // Do NOT assert -- we want to see ALL output even when there's divergence.
        // The test prints MATCH/MISMATCH at each checkpoint. The user reads the output
        // to find exactly where CUDA diverges from Metal. This is a diagnostic test.
    }

    // -----------------------------------------------------------------------
    // CPU naive reference: produces intra-layer checkpoint values for layer 0.
    //
    // Set LUMEN_CPU_DEBUG_LAYER=0 to print intermediate values. These are the
    // ground truth to compare against CUDA debug output (LUMEN_CUDA_DEBUG_LAYER=0).
    //
    // The CPU naive backend uses the same Q8_0 dequantization as CUDA, so
    // differences in checkpoint values pinpoint CUDA kernel bugs.
    // -----------------------------------------------------------------------

    /// Run token 0 through layer 0 on the CPU naive backend, printing
    /// intra-layer debug values when LUMEN_CPU_DEBUG_LAYER=0 is set.
    #[test]
    fn cpu_reference_layer0() {
        use lumen_runtime::NaiveF32Backend;

        let path = Path::new(MODEL_PATH);
        if !path.exists() {
            eprintln!("Model file not found: {MODEL_PATH} -- skipping cpu_reference_layer0");
            return;
        }

        let provider = SyncWeightProvider::open(path)
            .expect("Failed to open LBC model");

        let mut hp = provider.lbc().header.hyperparams;
        hp.max_seq_len = 64;

        let mut cpu = NaiveF32Backend::new();
        cpu.set_global_tensors(
            provider.embedding.clone(),
            provider.final_norm.clone(),
            provider.output_proj.clone(),
        );
        cpu.init(&hp).expect("CPU naive init failed");

        let hidden_dim = hp.hidden_dim as usize;
        let num_layers = hp.num_layers as usize;

        eprintln!("=== CPU Naive Reference (layer 0 intra-layer debug) ===");
        eprintln!("Model: {MODEL_PATH}");
        eprintln!("hidden_dim: {hidden_dim}, heads: {}, kv_heads: {}, head_dim: {}",
            hp.num_heads, hp.num_kv_heads, hp.head_dim);
        eprintln!("Set LUMEN_CPU_DEBUG_LAYER=0 to see intermediate values");
        eprintln!();

        let mut kv = KvCache::new(KvCacheConfig {
            max_seq_len: 64,
            num_layers,
            num_kv_heads: hp.num_kv_heads as usize,
            head_dim: hp.head_dim as usize,
            precision: KvPrecision::F32,
        }).expect("KV cache init failed");

        // Embed token 0
        let token_id = PROMPT_TOKENS[0];
        let mut x = cpu.embed_token(token_id).expect("embed_token failed");
        let embed_vals = x.as_f32_slice().to_vec();
        print_values(&format!("CPU embed (id={token_id}):"), &embed_vals, 10);

        // Run layer 0 only (with debug prints from LUMEN_CPU_DEBUG_LAYER)
        let layer_view = provider.get_layer_blocking(0).expect("get_layer_blocking(0) failed");
        let mut kv_view = kv.view_mut(0).expect("kv view_mut(0) failed");

        cpu.compute_layer(0, &mut x, &layer_view, Some(&mut kv_view), 0)
            .expect("compute_layer(0) failed");

        kv.commit_view(kv_view).expect("commit_view failed");

        let layer0_vals = x.as_f32_slice().to_vec();
        print_values("CPU post_layer0:", &layer0_vals, 10);
        let l2: f32 = layer0_vals.iter().map(|v| v * v).sum::<f32>().sqrt();
        eprintln!("  L2 norm: {l2:.6}");

        eprintln!("=== End CPU Naive Reference ===");
    }
}
