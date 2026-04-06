//! Integration test for CUDA embed_token kernel.
//!
//! This test requires a CUDA-capable GPU and is gated behind the `cuda` feature.
//! It will fail on macOS (no NVIDIA GPU). Run on Modal or a CUDA machine:
//!
//!   cargo test --release -p lumen-runtime --features cuda --test cuda_embed_test
//!
//! Test plan:
//! 1. Create a 4x3 embedding table: [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
//! 2. Initialize CudaBackend with vocab_size=4, hidden_dim=3
//! 3. Call embed_token(2) and verify output is [7,8,9]
//! 4. Call embed_token(0) and verify output is [1,2,3]
//! 5. Test out-of-range token_id returns an error

#![cfg(feature = "cuda")]

use lumen_runtime::compute::{ComputeBackend, ComputeDtype};
use lumen_runtime::cuda::CudaBackend;
use lumen_format::hyperparams::{ModelHyperparams, RopeParams, RopeScalingType};

/// Build minimal hyperparams for embed_token testing.
fn test_hyperparams(vocab_size: u32, hidden_dim: u32) -> ModelHyperparams {
    ModelHyperparams {
        num_layers: 1,
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: hidden_dim,
        hidden_dim,
        intermediate_dim: hidden_dim,
        vocab_size,
        max_seq_len: 32,
        rope_params: Some(RopeParams {
            theta: 10000.0,
            scaling_factor: 1.0,
            scaling_type: RopeScalingType::None,
        }),
        num_experts: None,
        num_active_experts: None,
        norm_eps: 1e-5,
        rotary_dim: None, rope_neox: false,
    }
}

#[test]
fn test_cuda_embed_token_f32() {
    // 4 tokens, hidden_dim=3: [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
    let embedding: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let hp = test_hyperparams(4, 3);

    let mut backend = CudaBackend::new(0).expect("Failed to create CUDA backend");
    backend.set_global_tensors(embedding, vec![1.0; 3], vec![0.0; 12]);
    backend.init(&hp).expect("Failed to init CUDA backend");

    // Embed token 2 -> should return [7, 8, 9]
    let result = backend.embed_token(2).expect("embed_token(2) failed");
    assert_eq!(result.num_elements, 3);
    assert_eq!(result.dtype, ComputeDtype::F32);

    let values = result.as_f32_slice();
    assert_eq!(values.len(), 3);
    assert!(
        (values[0] - 7.0).abs() < 1e-6,
        "expected 7.0, got {}",
        values[0]
    );
    assert!(
        (values[1] - 8.0).abs() < 1e-6,
        "expected 8.0, got {}",
        values[1]
    );
    assert!(
        (values[2] - 9.0).abs() < 1e-6,
        "expected 9.0, got {}",
        values[2]
    );

    // Embed token 0 -> should return [1, 2, 3]
    let result = backend.embed_token(0).expect("embed_token(0) failed");
    let values = result.as_f32_slice();
    assert!(
        (values[0] - 1.0).abs() < 1e-6,
        "expected 1.0, got {}",
        values[0]
    );
    assert!(
        (values[1] - 2.0).abs() < 1e-6,
        "expected 2.0, got {}",
        values[1]
    );
    assert!(
        (values[2] - 3.0).abs() < 1e-6,
        "expected 3.0, got {}",
        values[2]
    );

    // Embed token 3 (last valid) -> should return [10, 11, 12]
    let result = backend.embed_token(3).expect("embed_token(3) failed");
    let values = result.as_f32_slice();
    assert!(
        (values[0] - 10.0).abs() < 1e-6,
        "expected 10.0, got {}",
        values[0]
    );
    assert!(
        (values[1] - 11.0).abs() < 1e-6,
        "expected 11.0, got {}",
        values[1]
    );
    assert!(
        (values[2] - 12.0).abs() < 1e-6,
        "expected 12.0, got {}",
        values[2]
    );
}

#[test]
fn test_cuda_embed_token_out_of_range() {
    let embedding: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let hp = test_hyperparams(2, 3);

    let mut backend = CudaBackend::new(0).expect("Failed to create CUDA backend");
    backend.set_global_tensors(embedding, vec![1.0; 3], vec![0.0; 6]);
    backend.init(&hp).expect("Failed to init CUDA backend");

    // Token ID 2 is out of range for vocab_size=2
    let result = backend.embed_token(2);
    assert!(result.is_err(), "embed_token(2) should fail for vocab_size=2");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("out of range"),
        "error should mention out of range: {err_msg}"
    );
}

#[test]
fn test_cuda_embed_token_large_hidden_dim() {
    // Test with a realistic hidden_dim to exercise multi-block kernel launch.
    let hidden_dim = 4096;
    let vocab_size = 8;
    let embedding: Vec<f32> = (0..vocab_size * hidden_dim)
        .map(|i| (i as f32) * 0.001)
        .collect();
    let hp = test_hyperparams(vocab_size as u32, hidden_dim as u32);

    let mut backend = CudaBackend::new(0).expect("Failed to create CUDA backend");
    backend.set_global_tensors(
        embedding.clone(),
        vec![1.0; hidden_dim],
        vec![0.0; vocab_size * hidden_dim],
    );
    backend.init(&hp).expect("Failed to init CUDA backend");

    // Embed token 5
    let result = backend.embed_token(5).expect("embed_token(5) failed");
    assert_eq!(result.num_elements, hidden_dim);
    let values = result.as_f32_slice();

    // Verify every element matches the expected row
    let expected_start = 5 * hidden_dim;
    for i in 0..hidden_dim {
        let expected = embedding[expected_start + i];
        assert!(
            (values[i] - expected).abs() < 1e-5,
            "mismatch at index {i}: expected {expected}, got {}",
            values[i]
        );
    }
}
