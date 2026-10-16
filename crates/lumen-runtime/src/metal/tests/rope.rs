// Batched prefill RoPE kernel tests

use crate::metal::prefill_encode::BatchedRope;
use crate::metal::*;

/// The batched RoPE kernels, launched as the prefill launches them, rotate exactly the batch's
/// rows: Qwen3.5's K has 4 heads of 32 rotated pairs, 128 threads per token, so an odd batch
/// fills only half of its last 256-thread group. Each rotated pair matches the rotation by its
/// position's angle, and the floats past the batch keep their values.
#[test]
fn batched_rope_rotates_the_batch_and_nothing_past_it() {
    let backend = MetalF32Backend::new().unwrap();
    let pipelines = backend.compile_pipelines().unwrap();
    let (num_heads, head_dim, rope_half_dim) = (4usize, 256usize, 32usize);
    let row = num_heads * head_dim;
    let start_pos = 5usize;
    let positions = 64usize;
    let angle = |pos: usize, i: usize| pos as f32 * 0.01 * (i + 1) as f32;
    let cos: Vec<f32> = (0..positions * rope_half_dim)
        .map(|j| angle(j / rope_half_dim, j % rope_half_dim).cos())
        .collect();
    let sin: Vec<f32> = (0..positions * rope_half_dim)
        .map(|j| angle(j / rope_half_dim, j % rope_half_dim).sin())
        .collect();
    let cos_buf = backend.upload_f32(&cos).unwrap();
    let sin_buf = backend.upload_f32(&sin).unwrap();
    let neox = pipelines.rope_batched_neox.as_ref().unwrap();

    for (pipeline, neox_pairs) in [(&pipelines.rope_batched, false), (neox, true)] {
        for batch_size in [1usize, 3, 7] {
            // Two rows past the batch, as a buffer sized for a larger batch has.
            let before: Vec<f32> = (0..(batch_size + 2) * row)
                .map(|i| (i % 97) as f32 * 0.25 - 12.0)
                .collect();
            let rows = backend.upload_f32(&before).unwrap();
            let cmd = backend.queue.new_command_buffer().unwrap();
            let enc = cmd.new_compute_encoder().unwrap();
            BatchedRope {
                pipeline,
                cos: &cos_buf,
                sin: &sin_buf,
                head_dim,
                rope_half_dim,
                start_pos,
                batch_size,
            }
            .encode(&enc, &rows, num_heads);
            enc.end_encoding();
            cmd.commit_and_wait();
            let mut after = vec![0f32; before.len()];
            rows.read_f32(&mut after);

            let mut want = before.clone();
            for t in 0..batch_size {
                for h in 0..num_heads {
                    let base = t * row + h * head_dim;
                    for i in 0..rope_half_dim {
                        let (a, b) = if neox_pairs {
                            (base + i, base + i + rope_half_dim)
                        } else {
                            (base + 2 * i, base + 2 * i + 1)
                        };
                        let j = (start_pos + t) * rope_half_dim + i;
                        let (v0, v1) = (before[a], before[b]);
                        want[a] = v0 * cos[j] - v1 * sin[j];
                        want[b] = v0 * sin[j] + v1 * cos[j];
                    }
                }
            }
            for (k, (got, want)) in after.iter().zip(&want).enumerate() {
                assert!(
                    (got - want).abs() <= 1e-5 * want.abs().max(1.0),
                    "neox {neox_pairs}, batch {batch_size}: float {k} (token {}): {got} != {want}",
                    k / row
                );
            }
        }
    }
}
