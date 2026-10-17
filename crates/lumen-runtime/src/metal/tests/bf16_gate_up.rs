//! The BF16 fused gate+up+SwiGLU kernel chosen for a batch writes exactly the
//! batch's rows, for every NR microtile, including batches that fill only part of
//! the last tile.

use crate::compute::ComputeBackend;
use crate::metal::ffi::MTLSize;
use crate::metal::MetalF32Backend;
use crate::weight::provider_sync::SyncWeightProvider;
use lumen_format::test_model::{generate_test_model, TestModelConfig};

#[test]
fn bf16_gate_up_swiglu_writes_only_the_batch_rows() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.lbc");
    std::fs::write(&path, generate_test_model(&TestModelConfig::default())).unwrap();
    let provider = SyncWeightProvider::open(&path).unwrap();
    let mut backend = MetalF32Backend::new().unwrap();
    backend.init(&provider.lbc().header.hyperparams).unwrap();
    let pipelines = backend.pipelines.as_ref().unwrap();

    // Unit weights and inputs: every output is silu(K) * K.
    let (inter_dim, hidden_dim) = (32usize, 64usize);
    let expected = {
        let g = hidden_dim as f32;
        g / (1.0 + (-g).exp()) * g
    };
    let sentinel = -7.0f32;
    let weights: Vec<u8> = (0..inter_dim * hidden_dim)
        .flat_map(|_| 0x3F80u16.to_le_bytes())
        .collect();
    let w_gate = backend.device.new_buffer_with_bytes(&weights).unwrap();
    let w_up = backend.device.new_buffer_with_bytes(&weights).unwrap();

    for nr in [1u8, 2, 4] {
        for batch_size in [16usize, 32, 48, 64, 96, 128] {
            // Room for a whole extra tile past the batch, so a kernel that runs
            // past it writes there instead of outside the buffer.
            let rows = batch_size + 64;
            let (pso, tile_m, shmem_bytes) = MetalF32Backend::bf16_gate_up_swiglu_pipeline(
                pipelines, nr, batch_size, inter_dim, hidden_dim,
            );
            let x = backend
                .upload_f32(&vec![1.0f32; rows * hidden_dim])
                .unwrap();
            let y = backend
                .upload_f32(&vec![sentinel; rows * inter_dim])
                .unwrap();
            let cmd = backend.queue.new_command_buffer().unwrap();
            let enc = cmd.new_compute_encoder().unwrap();
            enc.set_pipeline_state(pso);
            enc.set_threadgroup_memory_length(shmem_bytes, 0);
            enc.set_buffer(&w_gate, 0, 0);
            enc.set_buffer(&x, 0, 1);
            enc.set_buffer(&y, 0, 2);
            enc.set_bytes(&(batch_size as u32).to_le_bytes(), 3);
            enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 4);
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 5);
            enc.set_buffer(&w_up, 0, 6);
            enc.dispatch_threadgroups(
                MTLSize::new(
                    (inter_dim as u64).div_ceil(32),
                    (batch_size as u64).div_ceil(tile_m),
                    1,
                ),
                MTLSize::new(128, 1, 1),
            );
            enc.end_encoding();
            cmd.commit_and_wait();

            let mut out = vec![0.0f32; rows * inter_dim];
            y.read_f32(&mut out);
            let (batch, past) = out.split_at(batch_size * inter_dim);
            assert!(
                batch
                    .iter()
                    .all(|&v| (v - expected).abs() <= 1e-3 * expected),
                "NR={nr}, batch {batch_size}: a batch row is not silu({hidden_dim}) * {hidden_dim}"
            );
            assert!(
                past.iter().all(|&v| v == sentinel),
                "NR={nr}, batch {batch_size}: {} values written past the batch",
                past.iter().filter(|&&v| v != sentinel).count()
            );
        }
    }
}
