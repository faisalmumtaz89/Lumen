// Batched prefill attention kernel tests

use crate::metal::ffi::MTLSize;
use crate::metal::repack_q4::{f16_to_f32, f32_to_f16_bits_rne};
use crate::metal::shaders::METAL_SHADER_SOURCE;
use crate::metal::*;

/// `attention_scores_tiled` gives every query its causal scores whatever position the batch
/// starts at, a cached prefix whose length is not a multiple of the 32-row tile included:
/// each kept score matches Q·K scaled (with Q and K rounded to half as the kernel loads them)
/// and every score past a query's position is zero.
#[test]
fn tiled_attention_scores_are_causal_at_any_start_position() {
    let backend = MetalF32Backend::new().unwrap();
    let lib = backend
        .device
        .new_library_with_source(METAL_SHADER_SOURCE)
        .unwrap();
    let func = lib.get_function("attention_scores_tiled").unwrap();
    let pso = backend.device.new_compute_pipeline_state(&func).unwrap();

    let (head_dim, num_q_heads, num_kv_heads) = (64usize, 2usize, 1usize);
    let kv_dim = num_kv_heads * head_dim;
    let scale = 0.125f32;
    let half = |v: f32| f16_to_f32(f32_to_f16_bits_rne(v));
    for (start_pos, batch) in [(0usize, 40usize), (32, 40), (8, 40), (600, 33), (31, 2)] {
        let attend = start_pos + batch;
        let stride = (attend + 7) & !7;
        let q: Vec<f32> = (0..batch * num_q_heads * head_dim)
            .map(|i| 0.25 + (i % 7) as f32 * 0.125)
            .collect();
        let k: Vec<f32> = (0..attend * kv_dim)
            .map(|i| 0.5 + (i % 5) as f32 * 0.25)
            .collect();
        let k_bytes: Vec<u8> = k
            .iter()
            .flat_map(|&v| f32_to_f16_bits_rne(v).to_le_bytes())
            .collect();
        let q_buf = backend.upload_f32(&q).unwrap();
        let k_buf = backend.device.new_buffer_with_bytes(&k_bytes).unwrap();
        // Filled with ones, so a score the kernel never writes shows up.
        let scores_buf = backend
            .device
            .new_buffer_with_bytes(&[0x00u8, 0x3c].repeat(batch * num_q_heads * stride))
            .unwrap();

        let cmd = backend.queue.new_command_buffer().unwrap();
        let enc = cmd.new_compute_encoder().unwrap();
        enc.set_pipeline_state(&pso);
        enc.set_buffer(&q_buf, 0, 0);
        enc.set_buffer(&k_buf, 0, 1);
        enc.set_buffer(&scores_buf, 0, 2);
        enc.set_bytes(&(head_dim as u32).to_le_bytes(), 3);
        enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 4);
        enc.set_bytes(&(num_q_heads as u32).to_le_bytes(), 5);
        enc.set_bytes(&(num_kv_heads as u32).to_le_bytes(), 6);
        enc.set_bytes(&scale.to_le_bytes(), 7);
        enc.set_bytes(&(start_pos as u32).to_le_bytes(), 8);
        enc.set_bytes(&(stride as u32).to_le_bytes(), 9);
        enc.set_bytes(&(batch as u32).to_le_bytes(), 10);
        enc.dispatch_threadgroups(
            MTLSize::new(
                (attend as u64).div_ceil(32),
                (batch as u64).div_ceil(32),
                num_q_heads as u64,
            ),
            MTLSize::new(128, 1, 1),
        );
        enc.end_encoding();
        cmd.commit_and_wait();

        let mut bits = vec![0u16; batch * num_q_heads * stride];
        scores_buf.read_u16(&mut bits);
        for m in 0..batch {
            for h in 0..num_q_heads {
                let kv_h = h / (num_q_heads / num_kv_heads);
                for n in 0..attend {
                    let got = f16_to_f32(bits[(m * num_q_heads + h) * stride + n]);
                    let want = if n <= start_pos + m {
                        (0..head_dim)
                            .map(|d| {
                                half(q[(m * num_q_heads + h) * head_dim + d] * scale)
                                    * half(k[n * kv_dim + kv_h * head_dim + d])
                            })
                            .sum::<f32>()
                    } else {
                        0.0
                    };
                    assert!(
                        (got - want).abs() <= 1e-2 * want.abs().max(1.0),
                        "start_pos {start_pos}, batch {batch}: query {m}, head {h}, key {n}: \
                         {got} != {want}"
                    );
                }
            }
        }
    }
}

/// The batched prefill's attention in slices of query rows gives each query the same output,
/// bit for bit, as one pass over the whole batch: 4,097 queries from position 3 over 16 heads
/// run as slices of 4,064 and 33 rows, both starting at positions that are not multiples of 32.
#[test]
fn sliced_batched_attention_equals_one_pass_over_the_batch() {
    use crate::metal::prefill_encode::{attention_slice_rows, BatchedAttention};

    let backend = MetalF32Backend::new().unwrap();
    let pipelines = backend.compile_pipelines().unwrap();
    let (head_dim, num_heads, num_kv_heads) = (64usize, 16usize, 2usize);
    let (start_pos, batch) = (3usize, 4097usize);
    let (q_dim, kv_dim) = (num_heads * head_dim, num_kv_heads * head_dim);
    let attend = start_pos + batch;
    let stride = (attend + 7) & !7;
    let max_seq_len = stride;
    let scale = 0.125f32;
    assert_eq!(attention_slice_rows(batch, num_heads, stride), 4064);

    let half = |v: f32| f32_to_f16_bits_rne(v).to_le_bytes();
    let q: Vec<f32> = (0..batch * q_dim)
        .map(|i| ((17 * i % 127) as f32 - 63.0) / 64.0)
        .collect();
    let k: Vec<u8> = (0..max_seq_len * kv_dim)
        .flat_map(|i| half(((13 * i % 113) as f32 - 56.0) / 64.0))
        .collect();
    let v: Vec<u8> = (0..kv_dim * max_seq_len)
        .flat_map(|i| {
            let (d, t) = (i / max_seq_len, i % max_seq_len);
            half(((19 * d + 7 * t) % 101) as f32 / 64.0 - 50.0 / 64.0)
        })
        .collect();
    let q_buf = backend.upload_f32(&q).unwrap();
    let k_buf = backend.device.new_buffer_with_bytes(&k).unwrap();
    let v_buf = backend.device.new_buffer_with_bytes(&v).unwrap();
    let out_bytes = batch * q_dim * 4;
    let sliced_out = backend.device.new_buffer(out_bytes).unwrap();
    let whole_out = backend.device.new_buffer(out_bytes).unwrap();
    let sliced_scores = backend
        .device
        .new_buffer(4064 * num_heads * stride * 2)
        .unwrap();
    let whole_scores = backend
        .device
        .new_buffer(batch * num_heads * stride * 2)
        .unwrap();

    let cmd = backend.queue.new_command_buffer().unwrap();
    let enc = cmd.new_compute_encoder().unwrap();
    BatchedAttention {
        q: &q_buf,
        k_cache: &k_buf,
        v_cache: &v_buf,
        scores: &sliced_scores,
        out: &sliced_out,
        head_dim,
        kv_dim,
        num_heads,
        num_kv_heads,
        scale,
        start_pos,
        batch_size: batch,
        max_seq_len,
    }
    .encode(&enc, &pipelines);
    enc.end_encoding();

    // The whole batch in one pass of each kernel.
    let u = |x: usize| (x as u32).to_le_bytes();
    let enc = cmd.new_compute_encoder().unwrap();
    enc.set_pipeline_state(&pipelines.attention_scores_tiled);
    enc.set_buffer(&q_buf, 0, 0);
    enc.set_buffer(&k_buf, 0, 1);
    enc.set_buffer(&whole_scores, 0, 2);
    enc.set_bytes(&u(head_dim), 3);
    enc.set_bytes(&u(kv_dim), 4);
    enc.set_bytes(&u(num_heads), 5);
    enc.set_bytes(&u(num_kv_heads), 6);
    enc.set_bytes(&scale.to_le_bytes(), 7);
    enc.set_bytes(&u(start_pos), 8);
    enc.set_bytes(&u(stride), 9);
    enc.set_bytes(&u(batch), 10);
    enc.dispatch_threadgroups(
        MTLSize::new(
            (attend as u64).div_ceil(32),
            (batch as u64).div_ceil(32),
            num_heads as u64,
        ),
        MTLSize::new(128, 1, 1),
    );
    enc.memory_barrier_with_scope(1);
    enc.set_pipeline_state(&pipelines.softmax_batched);
    enc.set_buffer(&whole_scores, 0, 0);
    enc.set_bytes(&u(stride), 1);
    enc.set_bytes(&u(num_heads), 2);
    enc.set_bytes(&u(start_pos), 3);
    enc.set_bytes(&u(batch), 4);
    enc.dispatch_threadgroups(
        MTLSize::new(num_heads as u64, batch as u64, 1),
        MTLSize::new(256, 1, 1),
    );
    enc.memory_barrier_with_scope(1);
    enc.set_pipeline_state(&pipelines.attention_output_tiled);
    enc.set_buffer(&whole_scores, 0, 0);
    enc.set_buffer(&v_buf, 0, 1);
    enc.set_buffer(&whole_out, 0, 2);
    enc.set_bytes(&u(head_dim), 3);
    enc.set_bytes(&u(kv_dim), 4);
    enc.set_bytes(&u(num_heads), 5);
    enc.set_bytes(&u(num_kv_heads), 6);
    enc.set_bytes(&u(start_pos), 7);
    enc.set_bytes(&u(stride), 8);
    enc.set_bytes(&u(batch), 9);
    enc.set_bytes(&u(max_seq_len), 10);
    enc.dispatch_threadgroups(
        MTLSize::new(
            (head_dim as u64).div_ceil(32),
            (batch as u64).div_ceil(32),
            num_heads as u64,
        ),
        MTLSize::new(128, 1, 1),
    );
    enc.end_encoding();
    cmd.commit_and_wait();

    let mut sliced = vec![0f32; batch * q_dim];
    let mut whole = vec![0f32; batch * q_dim];
    sliced_out.read_f32(&mut sliced);
    whole_out.read_f32(&mut whole);
    assert!(whole.iter().all(|x| x.is_finite()) && whole.iter().any(|&x| x != 0.0));
    let first_difference = sliced
        .iter()
        .zip(&whole)
        .position(|(a, b)| a.to_bits() != b.to_bits());
    assert_eq!(first_difference, None);
}
