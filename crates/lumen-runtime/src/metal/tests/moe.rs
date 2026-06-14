// MoE (Mixture of Experts) kernel and routing tests.
// Extracted from mod.rs for modularity.

use crate::metal::*;
use crate::metal::shaders::METAL_SHADER_SOURCE;
use crate::metal::ffi::MTLSize;
use crate::compute::ComputeBackend;
use lumen_format::hyperparams::ModelHyperparams;

/// Test moe_expert_accum kernel correctness (decode path).
///
/// Creates known expert outputs for 8 experts with hidden_dim=256.
/// Selects experts [2, 5] with weights [0.7, 0.3].
/// Verifies: output[i] = 0.0 + 0.7 * 3.0 + 0.3 * 6.0 = 3.9 for all i.
#[test]
fn test_moe_expert_accum_correctness() {
    let backend = MetalF32Backend::new().unwrap();

    let lib = backend.device
        .new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let func = lib.get_function("moe_expert_accum").unwrap();
    let pso = backend.device.new_compute_pipeline_state(&func).unwrap();

    let num_experts: usize = 8;
    let hidden_dim: usize = 256;
    let top_k: usize = 2;

    // Expert e has all values = (e + 1) as f32.
    // So expert 0 = all 1.0, expert 1 = all 2.0, ..., expert 7 = all 8.0.
    let mut expert_outputs = vec![0.0f32; num_experts * hidden_dim];
    for e in 0..num_experts {
        let val = (e + 1) as f32;
        for i in 0..hidden_dim {
            expert_outputs[e * hidden_dim + i] = val;
        }
    }
    let expert_outputs_buf = backend.upload_f32(&expert_outputs).unwrap();

    // Top-2 routing: experts 2 and 5 selected, weights [0.7, 0.3].
    let expert_weights = vec![0.7f32, 0.3f32];
    let expert_weights_buf = backend.upload_f32(&expert_weights).unwrap();

    // Expert IDs: [2, 5]
    let expert_ids: Vec<u32> = vec![2, 5];
    let expert_ids_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(expert_ids.as_ptr() as *const u8, expert_ids.len() * 4)
    };
    let expert_ids_buf = backend.device.new_buffer_with_bytes(expert_ids_bytes).unwrap();

    // Residual = all 0.0 (isolates the accumulation logic)
    let residual = vec![0.0f32; hidden_dim];
    let residual_buf = backend.upload_f32(&residual).unwrap();

    // Output buffer
    let output_buf = backend.device.new_buffer(hidden_dim * 4).unwrap();

    let hidden_dim_u32 = hidden_dim as u32;
    let top_k_u32 = top_k as u32;

    let cmd = backend.queue.new_command_buffer().unwrap();
    let enc = cmd.new_compute_encoder().unwrap();
    enc.set_pipeline_state(&pso);
    enc.set_buffer(&expert_outputs_buf, 0, 0);     // expert_outputs [num_experts * hidden_dim]
    enc.set_buffer(&expert_weights_buf, 0, 1);      // expert_weights [top_k]
    enc.set_buffer(&expert_ids_buf, 0, 2);           // expert_ids [top_k] u32
    enc.set_buffer(&output_buf, 0, 3);               // output [hidden_dim]
    enc.set_buffer(&residual_buf, 0, 4);             // residual [hidden_dim]
    enc.set_bytes(&hidden_dim_u32.to_le_bytes(), 5);
    enc.set_bytes(&top_k_u32.to_le_bytes(), 6);
    let tg_count = ((hidden_dim as u64) + 255) / 256;
    enc.dispatch_threadgroups(
        MTLSize::new(tg_count, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
    cmd.commit_and_wait();

    let mut result = vec![0.0f32; hidden_dim];
    output_buf.read_f32(&mut result);

    // Expected: 0.7 * expert_2 + 0.3 * expert_5 = 0.7 * 3.0 + 0.3 * 6.0 = 2.1 + 1.8 = 3.9
    let expected = 0.7f32 * 3.0f32 + 0.3f32 * 6.0f32;
    for (i, &v) in result.iter().enumerate() {
        assert!(
            (v - expected).abs() < 1e-4,
            "moe_expert_accum[{i}]: GPU={v}, expected={expected}"
        );
    }
    eprintln!(
        "moe_expert_accum: all {} values = {:.4} (expected {:.4}) -- PASS",
        hidden_dim, result[0], expected,
    );
}

/// Test moe_expert_accum with non-zero residual to verify residual addition.
#[test]
fn test_moe_expert_accum_with_residual() {
    let backend = MetalF32Backend::new().unwrap();

    let lib = backend.device
        .new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let func = lib.get_function("moe_expert_accum").unwrap();
    let pso = backend.device.new_compute_pipeline_state(&func).unwrap();

    let num_experts: usize = 8;
    let hidden_dim: usize = 64;
    let top_k: usize = 2;

    // Expert e has all values = (e + 1) as f32.
    let mut expert_outputs = vec![0.0f32; num_experts * hidden_dim];
    for e in 0..num_experts {
        let val = (e + 1) as f32;
        for i in 0..hidden_dim {
            expert_outputs[e * hidden_dim + i] = val;
        }
    }
    let expert_outputs_buf = backend.upload_f32(&expert_outputs).unwrap();

    // Select experts [0, 7] with weights [0.4, 0.6]
    let expert_weights = vec![0.4f32, 0.6f32];
    let expert_weights_buf = backend.upload_f32(&expert_weights).unwrap();

    let expert_ids: Vec<u32> = vec![0, 7];
    let expert_ids_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(expert_ids.as_ptr() as *const u8, expert_ids.len() * 4)
    };
    let expert_ids_buf = backend.device.new_buffer_with_bytes(expert_ids_bytes).unwrap();

    // Residual = all 10.0
    let residual = vec![10.0f32; hidden_dim];
    let residual_buf = backend.upload_f32(&residual).unwrap();

    let output_buf = backend.device.new_buffer(hidden_dim * 4).unwrap();

    let cmd = backend.queue.new_command_buffer().unwrap();
    let enc = cmd.new_compute_encoder().unwrap();
    enc.set_pipeline_state(&pso);
    enc.set_buffer(&expert_outputs_buf, 0, 0);
    enc.set_buffer(&expert_weights_buf, 0, 1);
    enc.set_buffer(&expert_ids_buf, 0, 2);
    enc.set_buffer(&output_buf, 0, 3);
    enc.set_buffer(&residual_buf, 0, 4);
    enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 5);
    enc.set_bytes(&(top_k as u32).to_le_bytes(), 6);
    let tg_count = ((hidden_dim as u64) + 255) / 256;
    enc.dispatch_threadgroups(
        MTLSize::new(tg_count, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
    cmd.commit_and_wait();

    let mut result = vec![0.0f32; hidden_dim];
    output_buf.read_f32(&mut result);

    // Expected: 10.0 + 0.4 * 1.0 + 0.6 * 8.0 = 10.0 + 0.4 + 4.8 = 15.2
    let expected = 10.0f32 + 0.4f32 * 1.0f32 + 0.6f32 * 8.0f32;
    for (i, &v) in result.iter().enumerate() {
        assert!(
            (v - expected).abs() < 1e-4,
            "moe_expert_accum_residual[{i}]: GPU={v}, expected={expected}"
        );
    }
    eprintln!(
        "moe_expert_accum (with residual): all {} values = {:.4} (expected {:.4}) -- PASS",
        hidden_dim, result[0], expected,
    );
}

/// Test moe_expert_accum_batched kernel correctness (prefill path).
///
/// Creates known expert outputs for 4 experts, batch_size=3, hidden_dim=32.
/// Each batch item selects different top-2 experts.
/// Verifies correct [num_experts, batch_size, hidden_dim] indexing.
#[test]
fn test_moe_expert_accum_batched_correctness() {
    let backend = MetalF32Backend::new().unwrap();

    let lib = backend.device
        .new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let func = lib.get_function("moe_expert_accum_batched").unwrap();
    let pso = backend.device.new_compute_pipeline_state(&func).unwrap();

    let num_experts: usize = 4;
    let batch_size: usize = 3;
    let hidden_dim: usize = 32;
    let top_k: usize = 2;

    // expert_outputs layout: [num_experts, batch_size, hidden_dim]
    // Expert e, batch b, element t => value = (e+1)*100 + (b+1)*10 + t
    // This produces unique values that let us verify correct indexing.
    let mut expert_outputs = vec![0.0f32; num_experts * batch_size * hidden_dim];
    for e in 0..num_experts {
        for b in 0..batch_size {
            for t in 0..hidden_dim {
                expert_outputs[e * batch_size * hidden_dim + b * hidden_dim + t] =
                    (e + 1) as f32 * 100.0 + (b + 1) as f32 * 10.0 + t as f32;
            }
        }
    }
    let expert_outputs_buf = backend.upload_f32(&expert_outputs).unwrap();

    // Per-batch routing:
    // batch 0: experts [1, 3], weights [0.6, 0.4]
    // batch 1: experts [0, 2], weights [0.5, 0.5]
    // batch 2: experts [2, 3], weights [0.8, 0.2]
    let expert_ids: Vec<u32> = vec![1, 3,  0, 2,  2, 3];
    let expert_weights: Vec<f32> = vec![0.6, 0.4,  0.5, 0.5,  0.8, 0.2];

    let expert_ids_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(expert_ids.as_ptr() as *const u8, expert_ids.len() * 4)
    };
    let expert_ids_buf = backend.device.new_buffer_with_bytes(expert_ids_bytes).unwrap();
    let expert_weights_buf = backend.upload_f32(&expert_weights).unwrap();

    let residual = vec![0.0f32; batch_size * hidden_dim];
    let residual_buf = backend.upload_f32(&residual).unwrap();
    let output_buf = backend.device.new_buffer(batch_size * hidden_dim * 4).unwrap();

    let cmd = backend.queue.new_command_buffer().unwrap();
    let enc = cmd.new_compute_encoder().unwrap();
    enc.set_pipeline_state(&pso);
    enc.set_buffer(&expert_outputs_buf, 0, 0);
    enc.set_buffer(&expert_weights_buf, 0, 1);
    enc.set_buffer(&expert_ids_buf, 0, 2);
    enc.set_buffer(&output_buf, 0, 3);
    enc.set_buffer(&residual_buf, 0, 4);
    enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 5);
    enc.set_bytes(&(top_k as u32).to_le_bytes(), 6);
    enc.set_bytes(&(batch_size as u32).to_le_bytes(), 7);
    let total_elems = (batch_size * hidden_dim) as u64;
    let tg_count = total_elems.div_ceil(256);
    enc.dispatch_threadgroups(
        MTLSize::new(tg_count, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
    cmd.commit_and_wait();

    let mut result = vec![0.0f32; batch_size * hidden_dim];
    output_buf.read_f32(&mut result);

    // Verify each batch item
    // Helper: expert_val(e, b, t) = (e+1)*100 + (b+1)*10 + t
    let expert_val = |e: usize, b: usize, t: usize| -> f32 {
        (e + 1) as f32 * 100.0 + (b + 1) as f32 * 10.0 + t as f32
    };

    let mut max_err = 0.0f32;
    for b in 0..batch_size {
        let ids = [expert_ids[b * top_k] as usize, expert_ids[b * top_k + 1] as usize];
        let weights = [expert_weights[b * top_k], expert_weights[b * top_k + 1]];
        for t in 0..hidden_dim {
            let expected = weights[0] * expert_val(ids[0], b, t)
                         + weights[1] * expert_val(ids[1], b, t);
            let actual = result[b * hidden_dim + t];
            let err = (actual - expected).abs();
            if err > max_err { max_err = err; }
            assert!(
                err < 1e-2,
                "moe_expert_accum_batched[b={b}, t={t}]: GPU={actual}, expected={expected}, err={err}"
            );
        }
    }
    eprintln!(
        "moe_expert_accum_batched: batch_size={}, hidden_dim={}, max_err={:.6} -- PASS",
        batch_size, hidden_dim, max_err,
    );
}

/// Verify moe_prefill_route_sort produces correct per-expert
/// segments (counting sort) and grouped (tok, slot) lists.
#[test]
fn test_moe_prefill_route_sort() {
    let backend = MetalF32Backend::new().unwrap();
    let lib = backend.device.new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let func = lib.get_function("moe_prefill_route_sort").unwrap();
    let pso = backend.device.new_compute_pipeline_state(&func).unwrap();

    let num_experts: usize = 6;
    let batch_size: usize = 4;
    let top_k: usize = 2;
    // token 0 -> [3,1]; token 1 -> [1,5]; token 2 -> [3,0]; token 3 -> [1,3]
    let expert_ids: Vec<u32> = vec![3,1, 1,5, 3,0, 1,3];
    let a = batch_size * top_k;

    let eid_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(expert_ids.as_ptr() as *const u8, expert_ids.len()*4) };
    let eid_buf = backend.device.new_buffer_with_bytes(eid_bytes).unwrap();
    let seg_off_buf = backend.device.new_buffer((num_experts+1)*4).unwrap();
    let tok_buf = backend.device.new_buffer(a*4).unwrap();
    let slot_buf = backend.device.new_buffer(a*4).unwrap();

    let cmd = backend.queue.new_command_buffer().unwrap();
    let enc = cmd.new_compute_encoder().unwrap();
    enc.set_pipeline_state(&pso);
    enc.set_buffer(&eid_buf, 0, 0);
    enc.set_buffer(&seg_off_buf, 0, 1);
    enc.set_buffer(&tok_buf, 0, 2);
    enc.set_buffer(&slot_buf, 0, 3);
    enc.set_bytes(&(batch_size as u32).to_le_bytes(), 4);
    enc.set_bytes(&(top_k as u32).to_le_bytes(), 5);
    enc.set_bytes(&(num_experts as u32).to_le_bytes(), 6);
    enc.dispatch_threadgroups(MTLSize::new(1,1,1), MTLSize::new(num_experts as u64,1,1));
    enc.end_encoding();
    cmd.commit_and_wait();

    let mut seg = vec![0u32; num_experts+1];
    seg_off_buf.read_u32(&mut seg);
    let mut tok = vec![0u32; a];
    tok_buf.read_u32(&mut tok);
    let mut slot = vec![0u32; a];
    slot_buf.read_u32(&mut slot);

    // CPU counting-sort reference.
    let mut counts = vec![0u32; num_experts];
    for &e in &expert_ids { counts[e as usize] += 1; }
    let mut exp_seg = vec![0u32; num_experts+1];
    for e in 0..num_experts { exp_seg[e+1] = exp_seg[e] + counts[e]; }
    assert_eq!(seg, exp_seg, "seg_off mismatch: got {seg:?} exp {exp_seg:?}");

    // For each expert segment, the (tok,slot) entries must all route to e.
    for e in 0..num_experts {
        for p in seg[e]..seg[e+1] {
            let t = tok[p as usize] as usize;
            let k = slot[p as usize] as usize;
            assert_eq!(expert_ids[t*top_k + k] as usize, e,
                "expert {e} seg pos {p}: tok {t} slot {k} routes to {} not {e}",
                expert_ids[t*top_k+k]);
        }
    }
    eprintln!("moe_prefill_route_sort: seg={seg:?} -- PASS");
}

/// Verify moe_grouped_gemm_q8_0 computes Y_seg = X_seg @ W_e^T
/// over per-expert segments, matching a CPU reference.
#[test]
fn test_moe_grouped_gemm_q8_0() {
    let backend = MetalF32Backend::new().unwrap();
    let lib = backend.device.new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let func = lib.get_function("moe_grouped_gemm_q8_0").unwrap();
    let pso = backend.device.new_compute_pipeline_state(&func).unwrap();

    // 2 experts. seg_off = [0, 3, 5] => expert0 has rows 0..3 (3 rows),
    // expert1 has rows 3..5 (2 rows). total_assign=5.
    let num_experts: usize = 2;
    let n: usize = 64;   // output dim (inter)
    let k: usize = 64;   // input dim (hidden)
    let total_assign: usize = 5;
    let seg_off: Vec<u32> = vec![0, 3, 5];

    // Weights: expert e gate matrix [N, K] = constant (e+1)*0.1
    let blocks_per_row = k / 32;
    let q8b = 34usize;
    let ebytes = n * blocks_per_row * q8b;
    let mut layer = vec![0u8; num_experts * ebytes];
    let mut woff = vec![0u64; num_experts];
    let evals = [0.1f32, 0.2f32];
    for e in 0..num_experts {
        woff[e] = (e * ebytes) as u64;
        let w = vec![evals[e]; n * k];
        let enc = encode_q8_0_matrix(&w, n, k);
        layer[e*ebytes..(e+1)*ebytes].copy_from_slice(&enc);
    }

    // Input: row r has constant value (r+1)*1.0
    let mut x = vec![0.0f32; total_assign * k];
    for r in 0..total_assign {
        for c in 0..k { x[r*k+c] = (r+1) as f32; }
    }

    let layer_buf = backend.device.new_buffer_with_bytes(&layer).unwrap();
    let x_buf = backend.upload_f32(&x).unwrap();
    let y_buf = backend.device.new_buffer(total_assign * n * 4).unwrap();
    let seg_bytes: Vec<u8> = seg_off.iter().flat_map(|v| v.to_le_bytes()).collect();
    let seg_buf = backend.device.new_buffer_with_bytes(&seg_bytes).unwrap();
    let woff_bytes: Vec<u8> = woff.iter().flat_map(|v| v.to_le_bytes()).collect();
    let woff_buf = backend.device.new_buffer_with_bytes(&woff_bytes).unwrap();

    let cmd = backend.queue.new_command_buffer().unwrap();
    let enc = cmd.new_compute_encoder().unwrap();
    enc.set_pipeline_state(&pso);
    enc.set_buffer(&layer_buf, 0, 0);
    enc.set_buffer(&x_buf, 0, 1);
    enc.set_buffer(&y_buf, 0, 2);
    enc.set_bytes(&(n as u32).to_le_bytes(), 3);
    enc.set_bytes(&(k as u32).to_le_bytes(), 4);
    enc.set_buffer(&seg_buf, 0, 5);
    enc.set_buffer(&woff_buf, 0, 6);
    enc.set_bytes(&(num_experts as u32).to_le_bytes(), 7);
    let max_m_tiles = ((total_assign as u64) + 31) / 32;
    enc.dispatch_threadgroups(
        MTLSize::new(((n as u64)+31)/32, max_m_tiles.max(1), num_experts as u64),
        MTLSize::new(128, 1, 1));
    enc.end_encoding();
    cmd.commit_and_wait();

    let mut y = vec![0.0f32; total_assign * n];
    y_buf.read_f32(&mut y);

    // CPU ref: row r in expert e => Y[r,j] = sum_c x[r,c]*w_e = K * (r+1) * evals[e]
    let mut max_err = 0.0f32;
    for e in 0..num_experts {
        for r in seg_off[e]..seg_off[e+1] {
            let r = r as usize;
            let expected = (k as f32) * (r+1) as f32 * evals[e];
            for j in 0..n {
                let got = y[r*n+j];
                let err = (got-expected).abs() / expected.max(1e-3);
                if err > max_err { max_err = err; }
                assert!(err < 0.02,
                    "grouped_gemm e={e} r={r} j={j}: got {got} exp {expected} rel_err {err}");
            }
        }
    }
    eprintln!("moe_grouped_gemm_q8_0: total_assign={total_assign} max_rel_err={max_err:.5} -- PASS");
}

/// Gather + scatter roundtrip + assign_expert.
/// Verifies the full index plumbing reproduces the [num_experts,batch,hidden]
/// dense layout that moe_expert_accum_batched expects.
#[test]
fn test_moe_prefill_gather_scatter() {
    let backend = MetalF32Backend::new().unwrap();
    let lib = backend.device.new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let sort = backend.device.new_compute_pipeline_state(
        &lib.get_function("moe_prefill_route_sort").unwrap()).unwrap();
    let assign = backend.device.new_compute_pipeline_state(
        &lib.get_function("moe_prefill_assign_expert").unwrap()).unwrap();
    let gather = backend.device.new_compute_pipeline_state(
        &lib.get_function("moe_prefill_gather").unwrap()).unwrap();
    let scatter = backend.device.new_compute_pipeline_state(
        &lib.get_function("moe_prefill_scatter").unwrap()).unwrap();

    let num_experts: usize = 4;
    let batch_size: usize = 3;
    let hidden: usize = 8;
    let top_k: usize = 2;
    let a = batch_size * top_k;
    // tok0->[2,0], tok1->[1,2], tok2->[0,3]
    let expert_ids: Vec<u32> = vec![2,0, 1,2, 0,3];

    // normed[tok][h] = tok*100 + h
    let mut normed = vec![0.0f32; batch_size*hidden];
    for t in 0..batch_size { for h in 0..hidden { normed[t*hidden+h] = (t*100+h) as f32; } }

    let eid_bytes: &[u8] = unsafe { std::slice::from_raw_parts(expert_ids.as_ptr() as *const u8, a*4) };
    let eid_buf = backend.device.new_buffer_with_bytes(eid_bytes).unwrap();
    let seg_buf = backend.device.new_buffer((num_experts+1)*4).unwrap();
    let tok_buf = backend.device.new_buffer(a*4).unwrap();
    let slot_buf = backend.device.new_buffer(a*4).unwrap();
    let ae_buf = backend.device.new_buffer(a*4).unwrap();
    let normed_buf = backend.upload_f32(&normed).unwrap();
    let grp_in_buf = backend.device.new_buffer(a*hidden*4).unwrap();
    // grouped "down output" = the gathered input itself (identity), to test scatter roundtrip.
    let eout_buf = backend.upload_f32(&vec![-1.0f32; num_experts*batch_size*hidden]).unwrap();

    let cmd = backend.queue.new_command_buffer().unwrap();
    let enc = cmd.new_compute_encoder().unwrap();
    enc.set_pipeline_state(&sort);
    enc.set_buffer(&eid_buf,0,0); enc.set_buffer(&seg_buf,0,1);
    enc.set_buffer(&tok_buf,0,2); enc.set_buffer(&slot_buf,0,3);
    enc.set_bytes(&(batch_size as u32).to_le_bytes(),4);
    enc.set_bytes(&(top_k as u32).to_le_bytes(),5);
    enc.set_bytes(&(num_experts as u32).to_le_bytes(),6);
    enc.dispatch_threadgroups(MTLSize::new(1,1,1), MTLSize::new(num_experts as u64,1,1));
    enc.memory_barrier_with_scope(1);
    enc.set_pipeline_state(&assign);
    enc.set_buffer(&seg_buf,0,0); enc.set_buffer(&ae_buf,0,1);
    enc.set_bytes(&(num_experts as u32).to_le_bytes(),2);
    enc.dispatch_threadgroups(MTLSize::new(1,1,1), MTLSize::new(num_experts as u64,1,1));
    enc.memory_barrier_with_scope(1);
    enc.set_pipeline_state(&gather);
    enc.set_buffer(&normed_buf,0,0); enc.set_buffer(&tok_buf,0,1); enc.set_buffer(&grp_in_buf,0,2);
    enc.set_bytes(&(hidden as u32).to_le_bytes(),3);
    enc.set_bytes(&(a as u32).to_le_bytes(),4);
    enc.dispatch_threadgroups(MTLSize::new(((a*hidden) as u64+255)/256,1,1), MTLSize::new(256,1,1));
    enc.memory_barrier_with_scope(1);
    // scatter grp_in (identity "output") into eout
    enc.set_pipeline_state(&scatter);
    enc.set_buffer(&grp_in_buf,0,0); enc.set_buffer(&tok_buf,0,1); enc.set_buffer(&ae_buf,0,2);
    enc.set_buffer(&eout_buf,0,3);
    enc.set_bytes(&(hidden as u32).to_le_bytes(),4);
    enc.set_bytes(&(batch_size as u32).to_le_bytes(),5);
    enc.set_bytes(&(a as u32).to_le_bytes(),6);
    enc.dispatch_threadgroups(MTLSize::new(((a*hidden) as u64+255)/256,1,1), MTLSize::new(256,1,1));
    enc.end_encoding();
    cmd.commit_and_wait();

    let mut eout = vec![0.0f32; num_experts*batch_size*hidden];
    eout_buf.read_f32(&mut eout);

    // For each routed (token, expert), eout[e*batch*hidden + tok*hidden + h] == normed[tok][h].
    for t in 0..batch_size {
        for k in 0..top_k {
            let e = expert_ids[t*top_k+k] as usize;
            for h in 0..hidden {
                let got = eout[e*batch_size*hidden + t*hidden + h];
                let exp = normed[t*hidden+h];
                assert!((got-exp).abs() < 1e-4,
                    "scatter (t={t},e={e},h={h}): got {got} exp {exp}");
            }
        }
    }
    eprintln!("moe_prefill_gather_scatter: roundtrip OK -- PASS");
}

/// FULL grouped prefill pipeline (sort->assign->gather->gate->
/// up->swiglu->down->scatter->accum) vs a CPU reference computing the same MoE
/// FFN. This mirrors the exact dispatch sequence in encode_moe_prefill_grouped_q8_0
/// and validates the routed result is correct end-to-end.
#[test]
fn test_moe_prefill_grouped_end_to_end() {
    let backend = MetalF32Backend::new().unwrap();
    let lib = backend.device.new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let p = |n: &str| backend.device.new_compute_pipeline_state(&lib.get_function(n).unwrap()).unwrap();
    let sort = p("moe_prefill_route_sort");
    let assign = p("moe_prefill_assign_expert");
    let gather = p("moe_prefill_gather");
    let ggemm = p("moe_grouped_gemm_q8_0");
    let swiglu = p("swiglu_batched");
    let scatter = p("moe_prefill_scatter");
    let accum = p("moe_expert_accum_batched");

    // EXACT Qwen3.5-MoE server scenario: ne=256, top_k=8, hidden=2048, inter=512,
    // small batch=21 (most of the 256 experts get ZERO tokens — the empty-segment
    // early-exit path). This reproduces the real prefill the server runs.
    let ne: usize = 256;
    let batch: usize = 21;
    let hidden: usize = 2048;
    let inter: usize = 512;
    let top_k: usize = 8;
    let a = batch * top_k;

    // Routing: deterministic distinct top_k experts per token, spread over 256.
    let mut expert_ids: Vec<u32> = Vec::with_capacity(a);
    let mut expert_weights: Vec<f32> = Vec::with_capacity(a);
    for t in 0..batch {
        let mut ws = 0.0f32;
        let mut chosen = Vec::new();
        for k in 0..top_k {
            let mut e = ((t*13 + k*29 + 1) % ne) as u32;
            while chosen.contains(&e) { e = (e + 1) % ne as u32; }
            chosen.push(e);
            let w = ((t + k + 1) % 7 + 1) as f32;
            ws += w;
            expert_ids.push(e);
            expert_weights.push(w);
        }
        for k in 0..top_k { expert_weights[t*top_k+k] /= ws; }
    }

    // normed[token][h] in [-1,1] deterministic.
    let mut normed = vec![0.0f32; batch*hidden];
    for t in 0..batch { for h in 0..hidden {
        normed[t*hidden+h] = (((t*7 + h*3) % 13) as f32 / 13.0) - 0.5;
    }}
    // Per-expert gate/up/down weights, [N,K] each, small deterministic values.
    let wval = |e: usize, kind: usize, r: usize, c: usize| -> f32 {
        (((e*31 + kind*17 + r*5 + c*2) % 19) as f32 / 19.0) - 0.5
    };
    let bpr = hidden/32; let q8b = 34usize;
    let gate_bytes = inter*bpr*q8b;       // [inter, hidden]
    let down_bytes = hidden*(inter/32)*q8b; // [hidden, inter]
    // Pack layer buffer: for each expert: gate | up | down.
    let estride = gate_bytes*2 + down_bytes;
    let mut layer = vec![0u8; ne*estride];
    let mut gate_woff = vec![0u64; ne];
    let mut up_woff = vec![0u64; ne];
    let mut down_woff = vec![0u64; ne];
    for e in 0..ne {
        let base = e*estride;
        gate_woff[e] = base as u64;
        up_woff[e] = (base+gate_bytes) as u64;
        down_woff[e] = (base+gate_bytes*2) as u64;
        let g: Vec<f32> = (0..inter*hidden).map(|i| wval(e,0,i/hidden,i%hidden)).collect();
        let u: Vec<f32> = (0..inter*hidden).map(|i| wval(e,1,i/hidden,i%hidden)).collect();
        let d: Vec<f32> = (0..hidden*inter).map(|i| wval(e,2,i/inter,i%inter)).collect();
        layer[base..base+gate_bytes].copy_from_slice(&encode_q8_0_matrix(&g, inter, hidden));
        layer[base+gate_bytes..base+gate_bytes*2].copy_from_slice(&encode_q8_0_matrix(&u, inter, hidden));
        layer[base+gate_bytes*2..base+estride].copy_from_slice(&encode_q8_0_matrix(&d, hidden, inter));
    }

    // ---- GPU buffers ----
    let mkbuf = |n: usize| backend.device.new_buffer(n*4).unwrap();
    let eid_bytes: &[u8] = unsafe { std::slice::from_raw_parts(expert_ids.as_ptr() as *const u8, a*4) };
    let eid_buf = backend.device.new_buffer_with_bytes(eid_bytes).unwrap();
    let ew_buf = backend.upload_f32(&expert_weights).unwrap();
    let normed_buf = backend.upload_f32(&normed).unwrap();
    let layer_buf = backend.device.new_buffer_with_bytes(&layer).unwrap();
    let seg_buf = mkbuf(ne+1);
    let tok_buf = mkbuf(a); let slot_buf = mkbuf(a); let ae_buf = mkbuf(a);
    let grp_in = mkbuf(a*hidden); let grp_sw = mkbuf(a*inter); let grp_dn = mkbuf(a*hidden);
    let gate_woff_buf = backend.device.new_buffer_with_bytes(
        &gate_woff.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    let up_woff_buf = backend.device.new_buffer_with_bytes(
        &up_woff.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    let down_woff_buf = backend.device.new_buffer_with_bytes(
        &down_woff.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    let eout = backend.upload_f32(&vec![0.0f32; ne*batch*hidden]).unwrap();
    let residual = backend.upload_f32(&vec![0.0f32; batch*hidden]).unwrap();
    let final_out = mkbuf(batch*hidden);

    let cmd = backend.queue.new_command_buffer().unwrap();
    let enc = cmd.new_compute_encoder().unwrap();
    let u32b = |v: usize| (v as u32).to_le_bytes();
    let mm = |v: usize| ((v as u64)+31)/32;
    // sort + assign
    enc.set_pipeline_state(&sort);
    enc.set_buffer(&eid_buf,0,0); enc.set_buffer(&seg_buf,0,1);
    enc.set_buffer(&tok_buf,0,2); enc.set_buffer(&slot_buf,0,3);
    enc.set_bytes(&u32b(batch),4); enc.set_bytes(&u32b(top_k),5); enc.set_bytes(&u32b(ne),6);
    enc.dispatch_threadgroups(MTLSize::new(1,1,1), MTLSize::new(ne as u64,1,1));
    enc.memory_barrier_with_scope(1);
    enc.set_pipeline_state(&assign);
    enc.set_buffer(&seg_buf,0,0); enc.set_buffer(&ae_buf,0,1); enc.set_bytes(&u32b(ne),2);
    enc.dispatch_threadgroups(MTLSize::new(1,1,1), MTLSize::new(ne as u64,1,1));
    enc.memory_barrier_with_scope(1);
    // gather
    enc.set_pipeline_state(&gather);
    enc.set_buffer(&normed_buf,0,0); enc.set_buffer(&tok_buf,0,1); enc.set_buffer(&grp_in,0,2);
    enc.set_bytes(&u32b(hidden),3); enc.set_bytes(&u32b(a),4);
    enc.dispatch_threadgroups(MTLSize::new(((a*hidden) as u64+255)/256,1,1), MTLSize::new(256,1,1));
    enc.memory_barrier_with_scope(1);
    // gate -> grp_sw ; up -> grp_dn
    for (woff, out) in [(&gate_woff_buf,&grp_sw),(&up_woff_buf,&grp_dn)] {
        enc.set_pipeline_state(&ggemm);
        enc.set_buffer(&layer_buf,0,0); enc.set_buffer(&grp_in,0,1); enc.set_buffer(out,0,2);
        enc.set_bytes(&u32b(inter),3); enc.set_bytes(&u32b(hidden),4);
        enc.set_buffer(&seg_buf,0,5); enc.set_buffer(woff,0,6); enc.set_bytes(&u32b(ne),7);
        enc.dispatch_threadgroups(MTLSize::new(mm(inter),mm(batch),ne as u64), MTLSize::new(128,1,1));
    }
    enc.memory_barrier_with_scope(1);
    // swiglu grp_sw = silu(grp_sw)*grp_dn
    enc.set_pipeline_state(&swiglu);
    enc.set_buffer(&grp_sw,0,0); enc.set_buffer(&grp_dn,0,1); enc.set_bytes(&u32b(a*inter),2);
    enc.dispatch_threadgroups(MTLSize::new(((a*inter) as u64+255)/256,1,1), MTLSize::new(256,1,1));
    enc.memory_barrier_with_scope(1);
    // down grp_sw -> grp_dn
    enc.set_pipeline_state(&ggemm);
    enc.set_buffer(&layer_buf,0,0); enc.set_buffer(&grp_sw,0,1); enc.set_buffer(&grp_dn,0,2);
    enc.set_bytes(&u32b(hidden),3); enc.set_bytes(&u32b(inter),4);
    enc.set_buffer(&seg_buf,0,5); enc.set_buffer(&down_woff_buf,0,6); enc.set_bytes(&u32b(ne),7);
    enc.dispatch_threadgroups(MTLSize::new(mm(hidden),mm(batch),ne as u64), MTLSize::new(128,1,1));
    enc.memory_barrier_with_scope(1);
    // scatter -> eout
    enc.set_pipeline_state(&scatter);
    enc.set_buffer(&grp_dn,0,0); enc.set_buffer(&tok_buf,0,1); enc.set_buffer(&ae_buf,0,2);
    enc.set_buffer(&eout,0,3); enc.set_bytes(&u32b(hidden),4); enc.set_bytes(&u32b(batch),5);
    enc.set_bytes(&u32b(a),6);
    enc.dispatch_threadgroups(MTLSize::new(((a*hidden) as u64+255)/256,1,1), MTLSize::new(256,1,1));
    enc.memory_barrier_with_scope(1);
    // accum -> final_out
    enc.set_pipeline_state(&accum);
    enc.set_buffer(&eout,0,0); enc.set_buffer(&ew_buf,0,1); enc.set_buffer(&eid_buf,0,2);
    enc.set_buffer(&final_out,0,3); enc.set_buffer(&residual,0,4);
    enc.set_bytes(&u32b(hidden),5); enc.set_bytes(&u32b(top_k),6); enc.set_bytes(&u32b(batch),7);
    enc.dispatch_threadgroups(MTLSize::new(((batch*hidden) as u64+255)/256,1,1), MTLSize::new(256,1,1));
    enc.end_encoding();
    cmd.commit_and_wait();

    let mut got = vec![0.0f32; batch*hidden];
    final_out.read_f32(&mut got);

    // CPU reference using the DEQUANTIZED Q8 weights (exact values the GPU
    // reads), so Q8 quantization error is removed from the comparison and only
    // fp-order / f16-scale differences (~1e-3) remain. This isolates kernel
    // correctness from quantization noise (which catastrophic cancellation in
    // this synthetic test would otherwise amplify).
    let deq = |bytes: &[u8], rows: usize, cols: usize| -> Vec<f32> {
        let bpr = cols/32; let mut out = vec![0.0f32; rows*cols];
        for r in 0..rows { for b in 0..bpr {
            let bs = (r*bpr+b)*34;
            let scale = f16_bits_to_f32_moe(u16::from_le_bytes([bytes[bs], bytes[bs+1]]));
            for j in 0..32 {
                let q = bytes[bs+2+j] as i8;
                out[r*cols + b*32 + j] = scale * q as f32;
            }
        }}
        out
    };
    // Dequantize each expert's gate/up/down from the packed layer buffer.
    let mut dq_gate = vec![Vec::new(); ne];
    let mut dq_up = vec![Vec::new(); ne];
    let mut dq_down = vec![Vec::new(); ne];
    for e in 0..ne {
        let base = e*estride;
        dq_gate[e] = deq(&layer[base..base+gate_bytes], inter, hidden);
        dq_up[e]   = deq(&layer[base+gate_bytes..base+gate_bytes*2], inter, hidden);
        dq_down[e] = deq(&layer[base+gate_bytes*2..base+estride], hidden, inter);
    }
    let silu = |x: f32| x / (1.0 + (-x).exp());
    let mut max_rel = 0.0f32;
    let mut nfail = 0;
    for t in 0..batch {
        let mut out = vec![0.0f32; hidden];
        for k in 0..top_k {
            let e = expert_ids[t*top_k+k] as usize;
            let w = expert_weights[t*top_k+k];
            let mut sw = vec![0.0f32; inter];
            for r in 0..inter {
                let mut g = 0.0; let mut u = 0.0;
                for c in 0..hidden {
                    g += dq_gate[e][r*hidden+c] * normed[t*hidden+c];
                    u += dq_up[e][r*hidden+c] * normed[t*hidden+c];
                }
                sw[r] = silu(g) * u;
            }
            for r in 0..hidden {
                let mut d = 0.0;
                for c in 0..inter { d += dq_down[e][r*inter+c] * sw[c]; }
                out[r] += w * d;
            }
        }
        for h in 0..hidden {
            let g = got[t*hidden+h];
            // Mixed tolerance: kernel correctness is judged on max(rel, abs).
            // Down outputs are O(100) and the weighted sum cancels to O(0.01..3),
            // so fp-order differences between tiled-GPU and sequential-CPU give
            // ~1e-2 absolute deltas on heavily-cancelled elements. Accept if
            // EITHER abs<0.05 OR rel<6%.
            let abs = (g - out[h]).abs();
            let rel = abs / out[h].abs().max(1e-6);
            let ok = abs < 0.05 || rel < 0.06;
            if rel > max_rel && abs >= 0.05 { max_rel = rel; }
            if !ok && nfail < 12 {
                eprintln!("  FAIL t={t} h={h}: got {g:.5} exp {:.5} abs {abs:.5} rel {rel:.3}", out[h]);
                nfail += 1;
            }
        }
    }
    eprintln!("moe_prefill_grouped_end_to_end: batch={batch} max_rel={max_rel:.4} nfail={nfail}");
    assert!(nfail == 0, "grouped e2e had {nfail}+ failures, max_rel={max_rel:.4}");
}

// ====================================================================
// Partial layer loading tests
// ====================================================================

#[test]
fn test_non_expert_byte_end_dense_layer() {
    // Dense layer: no experts, non_expert_end should cover all non-expert tensors.
    use lumen_format::index::{SubtensorOffsets, TensorSlice};
    use lumen_format::quantization::QuantScheme;

    let make = |off: u64, len: u64| TensorSlice {
        offset: off, length: len, quant: QuantScheme::Q8_0,
    };

    let st = SubtensorOffsets {
        wq: make(0, 1024),
        wk: make(1024, 512),
        wv: make(1536, 512),
        wo: make(2048, 1024),
        bq: None, bk: None, bv: None,
        w_gate: make(3072, 2048),
        w_up: make(5120, 2048),
        w_down: make(7168, 2048),
        attn_norm: make(9216, 64),
        ffn_norm: make(9280, 64),
        router_weight: None,
        experts: None,
        shared_expert_gate: None, shared_expert_up: None, shared_expert_down: None,
        attn_gate: None, attn_post_norm: None,
        ssm_a: None, ssm_conv1d: None, ssm_dt: None,
        ssm_beta: None, ssm_alpha: None, ssm_norm: None, ssm_out: None,
        attn_q_norm: None, attn_k_norm: None, ffn_gate_inp_shexp: None,
        layer_type: None,
    };

    let end = MetalF32Backend::non_expert_byte_end(&st);
    // Last tensor ends at 9280 + 64 = 9344
    assert_eq!(end, 9344, "non_expert_byte_end should cover all tensors for dense layer");
}

#[test]
fn test_non_expert_byte_end_moe_layer() {
    // MoE layer: non_expert_end should stop before expert data.
    use lumen_format::index::{SubtensorOffsets, TensorSlice, ExpertSlice};
    use lumen_format::quantization::QuantScheme;

    let make = |off: u64, len: u64| TensorSlice {
        offset: off, length: len, quant: QuantScheme::Q8_0,
    };

    // Layout: [wq=0..1024][wk=1024..1536][wv=1536..2048][wo=2048..3072]
    //         [attn_norm=3072..3136][ffn_norm=3136..3200]
    //         [router=3200..3456]
    //         [expert0_gate=3456..5504][expert0_up=5504..7552][expert0_down=7552..9600]
    //         [expert1_gate=9600..11648]...
    let st = SubtensorOffsets {
        wq: make(0, 1024),
        wk: make(1024, 512),
        wv: make(1536, 512),
        wo: make(2048, 1024),
        bq: None, bk: None, bv: None,
        w_gate: make(0, 0),  // sentinel for MoE
        w_up: make(0, 0),
        w_down: make(0, 0),
        attn_norm: make(3072, 64),
        ffn_norm: make(3136, 64),
        router_weight: Some(make(3200, 256)),
        experts: Some(vec![
            ExpertSlice {
                gate: make(3456, 2048),
                up: make(5504, 2048),
                down: make(7552, 2048),
            },
            ExpertSlice {
                gate: make(9600, 2048),
                up: make(11648, 2048),
                down: make(13696, 2048),
            },
        ]),
        shared_expert_gate: None, shared_expert_up: None, shared_expert_down: None,
        attn_gate: None, attn_post_norm: None,
        ssm_a: None, ssm_conv1d: None, ssm_dt: None,
        ssm_beta: None, ssm_alpha: None, ssm_norm: None, ssm_out: None,
        attn_q_norm: None, attn_k_norm: None, ffn_gate_inp_shexp: None,
        layer_type: None,
    };

    let end = MetalF32Backend::non_expert_byte_end(&st);
    // Router ends at 3200 + 256 = 3456, which is exactly where experts start.
    assert_eq!(end, 3456, "non_expert_byte_end should stop at router end (before experts)");
}

#[test]
fn test_non_expert_byte_end_with_biases() {
    // MoE layer with biases: non_expert_end should include bias data.
    use lumen_format::index::{SubtensorOffsets, TensorSlice, ExpertSlice};
    use lumen_format::quantization::QuantScheme;

    let make = |off: u64, len: u64| TensorSlice {
        offset: off, length: len, quant: QuantScheme::Q8_0,
    };

    let st = SubtensorOffsets {
        wq: make(0, 1024),
        wk: make(1024, 512),
        wv: make(1536, 512),
        wo: make(2048, 1024),
        bq: Some(make(3072, 128)),
        bk: Some(make(3200, 64)),
        bv: Some(make(3264, 64)),
        w_gate: make(0, 0),
        w_up: make(0, 0),
        w_down: make(0, 0),
        attn_norm: make(3328, 64),
        ffn_norm: make(3392, 64),
        router_weight: Some(make(3456, 256)),
        experts: Some(vec![
            ExpertSlice {
                gate: make(3712, 2048),
                up: make(5760, 2048),
                down: make(7808, 2048),
            },
        ]),
        shared_expert_gate: None, shared_expert_up: None, shared_expert_down: None,
        attn_gate: None, attn_post_norm: None,
        ssm_a: None, ssm_conv1d: None, ssm_dt: None,
        ssm_beta: None, ssm_alpha: None, ssm_norm: None, ssm_out: None,
        attn_q_norm: None, attn_k_norm: None, ffn_gate_inp_shexp: None,
        layer_type: None,
    };

    let end = MetalF32Backend::non_expert_byte_end(&st);
    // Router ends at 3456 + 256 = 3712, which is where experts start.
    assert_eq!(end, 3712, "non_expert_byte_end should include biases and router");
}

/// Verify non_expert_byte_end includes shared expert tensors.
#[test]
fn test_non_expert_byte_end_with_shared_expert() {
    use lumen_format::index::{SubtensorOffsets, TensorSlice, ExpertSlice};
    use lumen_format::quantization::QuantScheme;

    let make = |off: u64, len: u64| TensorSlice {
        offset: off, length: len, quant: QuantScheme::Q4_0,
    };

    // Layout simulating Qwen3.5-MoE:
    // [attn=0..3072][norms=3072..3200][router=3200..3456]
    // [shared_expert_gate=3456..4480][shared_expert_up=4480..5504][shared_expert_down=5504..6528]
    // [experts start at 6528...]
    let st = SubtensorOffsets {
        wq: make(0, 1024),
        wk: make(1024, 512),
        wv: make(1536, 512),
        wo: make(2048, 1024),
        bq: None, bk: None, bv: None,
        w_gate: make(0, 0),
        w_up: make(0, 0),
        w_down: make(0, 0),
        attn_norm: make(3072, 64),
        ffn_norm: make(3136, 64),
        router_weight: Some(make(3200, 256)),
        experts: Some(vec![
            ExpertSlice {
                gate: make(6528, 2048),
                up: make(8576, 2048),
                down: make(10624, 2048),
            },
        ]),
        shared_expert_gate: Some(make(3456, 1024)),
        shared_expert_up: Some(make(4480, 1024)),
        shared_expert_down: Some(make(5504, 1024)),
        attn_gate: None, attn_post_norm: None,
        ssm_a: None, ssm_conv1d: None, ssm_dt: None,
        ssm_beta: None, ssm_alpha: None, ssm_norm: None, ssm_out: None,
        attn_q_norm: None, attn_k_norm: None, ffn_gate_inp_shexp: None,
        layer_type: Some(0),
    };

    let end = MetalF32Backend::non_expert_byte_end(&st);
    // Shared expert down ends at 5504 + 1024 = 6528, which is where experts start.
    assert_eq!(end, 6528,
        "non_expert_byte_end should include shared expert tensors (got {})", end);
}

/// Verify non_expert_byte_end includes SSM tensors for GDN layers.
#[test]
fn test_non_expert_byte_end_with_ssm_tensors() {
    use lumen_format::index::{SubtensorOffsets, TensorSlice};
    use lumen_format::quantization::QuantScheme;

    let make = |off: u64, len: u64| TensorSlice {
        offset: off, length: len, quant: QuantScheme::Q8_0,
    };

    let st = SubtensorOffsets {
        wq: make(0, 1024),
        wk: make(1024, 512),
        wv: make(1536, 512),
        wo: make(2048, 1024),
        bq: None, bk: None, bv: None,
        w_gate: make(0, 0),
        w_up: make(0, 0),
        w_down: make(0, 0),
        attn_norm: make(3072, 64),
        ffn_norm: make(3136, 64),
        router_weight: None,
        experts: None,
        shared_expert_gate: None, shared_expert_up: None, shared_expert_down: None,
        attn_gate: Some(make(3200, 128)),
        attn_post_norm: Some(make(3328, 64)),
        ssm_a: Some(make(3392, 256)),
        ssm_conv1d: Some(make(3648, 128)),
        ssm_dt: Some(make(3776, 256)),
        ssm_beta: Some(make(4032, 64)),
        ssm_alpha: Some(make(4096, 64)),
        ssm_norm: Some(make(4160, 64)),
        ssm_out: Some(make(4224, 512)),
        attn_q_norm: None, attn_k_norm: None, ffn_gate_inp_shexp: None,
        layer_type: Some(1),
    };

    let end = MetalF32Backend::non_expert_byte_end(&st);
    // ssm_out ends at 4224 + 512 = 4736 (the last tensor).
    assert_eq!(end, 4736,
        "non_expert_byte_end should include SSM tensors (got {})", end);
}

#[test]
fn test_streaming_moe_cache_assembly_simulation() {
    // Simulate the mixed cache/reader assembly pattern.
    // This tests the data flow without requiring Metal GPU dispatch.
    use lumen_format::index::{ExpertSlice, TensorSlice};
    use lumen_format::quantization::QuantScheme;

    let make_slice = |off: u64, len: u64| TensorSlice {
        offset: off, length: len, quant: QuantScheme::F32,
    };

    let num_experts = 4;
    let expert_tensor_size: u64 = 64;

    // Build simulated expert data and slices.
    let mut expert_data: Vec<Vec<u8>> = Vec::new();
    let mut expert_slices: Vec<ExpertSlice> = Vec::new();

    for eid in 0..num_experts {
        let gate = vec![(eid * 3) as u8; expert_tensor_size as usize];
        let up = vec![(eid * 3 + 1) as u8; expert_tensor_size as usize];
        let down = vec![(eid * 3 + 2) as u8; expert_tensor_size as usize];

        let mut data = Vec::new();
        data.extend_from_slice(&gate);
        data.extend_from_slice(&up);
        data.extend_from_slice(&down);

        let slice = ExpertSlice {
            gate: make_slice(0, expert_tensor_size),
            up: make_slice(expert_tensor_size, expert_tensor_size),
            down: make_slice(2 * expert_tensor_size, expert_tensor_size),
        };

        expert_data.push(data);
        expert_slices.push(slice);
    }

    // Simulate cache: experts 0 and 2 are cached.
    let cached_experts: Vec<usize> = vec![0, 2];
    let uncached_experts: Vec<usize> = vec![1, 3];

    // Assemble buffer: cached from "cache", uncached from "reader".
    let mut assembled = Vec::new();
    let mut gate_offs: Vec<u64> = Vec::new();
    let mut up_offs: Vec<u64> = Vec::new();
    let mut down_offs: Vec<u64> = Vec::new();

    for eid in 0..num_experts {
        let (data, slices) = if cached_experts.contains(&eid) {
            // From cache
            (expert_data[eid].clone(), expert_slices[eid].clone())
        } else {
            // From reader (same data in this simulation)
            (expert_data[eid].clone(), expert_slices[eid].clone())
        };

        let base = assembled.len() as u64;
        gate_offs.push(base + slices.gate.offset);
        up_offs.push(base + slices.up.offset);
        down_offs.push(base + slices.down.offset);
        assembled.extend_from_slice(&data);
    }

    // Verify assembled buffer integrity.
    assert_eq!(
        assembled.len(),
        num_experts * 3 * expert_tensor_size as usize,
        "assembled buffer should contain all expert data"
    );

    // Verify each expert's data is at the correct offset.
    for eid in 0..num_experts {
        let gate_start = gate_offs[eid] as usize;
        let up_start = up_offs[eid] as usize;
        let down_start = down_offs[eid] as usize;

        assert!(
            assembled[gate_start..gate_start + expert_tensor_size as usize]
                .iter().all(|&b| b == (eid * 3) as u8),
            "expert {eid} gate data mismatch"
        );
        assert!(
            assembled[up_start..up_start + expert_tensor_size as usize]
                .iter().all(|&b| b == (eid * 3 + 1) as u8),
            "expert {eid} up data mismatch"
        );
        assert!(
            assembled[down_start..down_start + expert_tensor_size as usize]
                .iter().all(|&b| b == (eid * 3 + 2) as u8),
            "expert {eid} down data mismatch"
        );
    }

    eprintln!(
        "streaming_moe_cache_assembly: {} experts, {} cached / {} from reader, \
         assembled {} bytes -- PASS",
        num_experts, cached_experts.len(), uncached_experts.len(), assembled.len()
    );
}

#[test]
fn test_streaming_moe_cache_hit_rate_simulation() {
    // Simulate N tokens of streaming MoE decode with an LFU cache.
    // Measures cache hit rate and bytes saved vs full-blob loading.
    use crate::expert::cache::ExpertLfuCache;
    use lumen_format::index::{ExpertSlice, TensorSlice};
    use lumen_format::quantization::QuantScheme;

    let make_slice = |off: u64, len: u64| TensorSlice {
        offset: off, length: len, quant: QuantScheme::F32,
    };

    let num_layers = 4;
    let num_experts_per_layer = 8;
    let top_k = 2;
    let expert_size_bytes = 3 * 2048; // gate + up + down
    let non_expert_size = 4096; // attention + norms + router
    let full_blob_size = non_expert_size + num_experts_per_layer * expert_size_bytes;
    let num_tokens = 50;
    let cache_capacity = num_layers * 4; // Cache top-4 per layer

    let mut cache = ExpertLfuCache::new(cache_capacity);
    let mut total_bytes_loaded_from_disk = 0usize;
    let mut total_bytes_if_no_cache = 0usize;
    let mut total_cache_hits = 0usize;
    let mut total_cache_misses = 0usize;

    // Simulate a skewed expert activation pattern.
    // Experts 0 and 1 are "hot" (selected 70% of the time).
    let mut rng_state = 42u64;
    let mut next_rng = || -> u64 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        rng_state
    };

    for _token in 0..num_tokens {
        for layer in 0..num_layers {
            // Select top_k experts with skewed distribution.
            let mut selected = Vec::new();
            while selected.len() < top_k {
                let r = next_rng() % 100;
                let eid = if r < 35 {
                    0
                } else if r < 70 {
                    1
                } else {
                    (2 + (next_rng() % (num_experts_per_layer as u64 - 2))) as u32
                };
                if !selected.contains(&eid) {
                    selected.push(eid);
                }
            }

            // Without cache: always load full blob.
            total_bytes_if_no_cache += full_blob_size;

            // With cache: load non-expert always + only uncached experts.
            total_bytes_loaded_from_disk += non_expert_size;

            for &eid in &selected {
                let key = (layer, eid);
                if cache.get(&key).is_some() {
                    total_cache_hits += 1;
                } else {
                    total_cache_misses += 1;
                    total_bytes_loaded_from_disk += expert_size_bytes;

                    // Insert into cache.
                    let data = vec![0u8; expert_size_bytes];
                    let slices = ExpertSlice {
                        gate: make_slice(0, 2048),
                        up: make_slice(2048, 2048),
                        down: make_slice(4096, 2048),
                    };
                    cache.insert(key, data, slices);
                }
            }
        }
    }

    let hit_rate = if total_cache_hits + total_cache_misses > 0 {
        total_cache_hits as f64 / (total_cache_hits + total_cache_misses) as f64
    } else {
        0.0
    };

    let bytes_saved = total_bytes_if_no_cache.saturating_sub(total_bytes_loaded_from_disk);
    let savings_pct = if total_bytes_if_no_cache > 0 {
        bytes_saved as f64 / total_bytes_if_no_cache as f64 * 100.0
    } else {
        0.0
    };

    eprintln!(
        "\nStreaming MoE cache simulation ({} tokens, {} layers, {} experts/layer, top-{}):",
        num_tokens, num_layers, num_experts_per_layer, top_k,
    );
    eprintln!(
        "  Cache: capacity={}, final_entries={}, hit_rate={:.1}%",
        cache_capacity, cache.len(), hit_rate * 100.0,
    );
    eprintln!(
        "  Bytes: no_cache={:.1} MB, with_cache={:.1} MB, saved={:.1} MB ({:.1}%)",
        total_bytes_if_no_cache as f64 / 1e6,
        total_bytes_loaded_from_disk as f64 / 1e6,
        bytes_saved as f64 / 1e6,
        savings_pct,
    );

    // Verify cache hit rate is reasonable with skewed distribution.
    assert!(
        hit_rate > 0.3,
        "Expected cache hit rate > 30% with skewed distribution, got {:.1}%",
        hit_rate * 100.0,
    );

    // Verify significant byte savings.
    assert!(
        savings_pct > 50.0,
        "Expected >50% byte savings from caching, got {:.1}%",
        savings_pct,
    );
}

// ====================================================================
// Biased router kernel tests
// ====================================================================

#[test]
fn test_moe_router_softmax_biased_zero_lambda_matches_unbiased() {
    // When cache_bias_lambda=0 and all is_cached=0, the biased kernel
    // must produce identical output to the unbiased kernel.
    let backend = MetalF32Backend::new().unwrap();

    let lib = backend.device
        .new_library_with_source(METAL_SHADER_SOURCE).unwrap();

    let unbiased_fn = lib.get_function("moe_router_softmax").unwrap();
    let biased_fn = lib.get_function("moe_router_softmax_biased").unwrap();
    let unbiased_pso = backend.device.new_compute_pipeline_state(&unbiased_fn).unwrap();
    let biased_pso = backend.device.new_compute_pipeline_state(&biased_fn).unwrap();

    let hidden_dim: usize = 64;
    let num_experts: usize = 8;
    let top_k: usize = 2;

    // Deterministic hidden state and gate weights.
    let hidden: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 * 0.1).sin()).collect();
    let gate: Vec<f32> = (0..num_experts * hidden_dim)
        .map(|i| (i as f32 * 0.037).cos())
        .collect();
    let is_cached = vec![0u8; num_experts];

    let hidden_buf = backend.upload_f32(&hidden).unwrap();
    let gate_buf = backend.upload_f32(&gate).unwrap();
    let is_cached_buf = backend.device.new_buffer_with_bytes(&is_cached).unwrap();

    // Run unbiased kernel.
    let ids_buf_u = backend.device.new_buffer(top_k * 4).unwrap();
    let weights_buf_u = backend.device.new_buffer(top_k * 4).unwrap();
    {
        let cmd = backend.queue.new_command_buffer().unwrap();
        let enc = cmd.new_compute_encoder().unwrap();
        enc.set_pipeline_state(&unbiased_pso);
        enc.set_buffer(&hidden_buf, 0, 0);
        enc.set_buffer(&gate_buf, 0, 1);
        enc.set_buffer(&ids_buf_u, 0, 2);
        enc.set_buffer(&weights_buf_u, 0, 3);
        enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 4);
        enc.set_bytes(&(num_experts as u32).to_le_bytes(), 5);
        enc.set_bytes(&(top_k as u32).to_le_bytes(), 6);
        let tg = 256u64.min(hidden_dim as u64).max(1);
        enc.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Run biased kernel with lambda=0.
    let ids_buf_b = backend.device.new_buffer(top_k * 4).unwrap();
    let weights_buf_b = backend.device.new_buffer(top_k * 4).unwrap();
    {
        let cmd = backend.queue.new_command_buffer().unwrap();
        let enc = cmd.new_compute_encoder().unwrap();
        enc.set_pipeline_state(&biased_pso);
        enc.set_buffer(&hidden_buf, 0, 0);
        enc.set_buffer(&gate_buf, 0, 1);
        enc.set_buffer(&ids_buf_b, 0, 2);
        enc.set_buffer(&weights_buf_b, 0, 3);
        enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 4);
        enc.set_bytes(&(num_experts as u32).to_le_bytes(), 5);
        enc.set_bytes(&(top_k as u32).to_le_bytes(), 6);
        enc.set_buffer(&is_cached_buf, 0, 7);
        enc.set_bytes(&0.0f32.to_le_bytes(), 8);
        let tg = 256u64.min(hidden_dim as u64).max(1);
        enc.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Compare outputs.
    let mut ids_u = vec![0u32; top_k];
    let mut ids_b = vec![0u32; top_k];
    let mut wts_u = vec![0.0f32; top_k];
    let mut wts_b = vec![0.0f32; top_k];
    ids_buf_u.read_u32(&mut ids_u);
    ids_buf_b.read_u32(&mut ids_b);
    weights_buf_u.read_f32(&mut wts_u);
    weights_buf_b.read_f32(&mut wts_b);

    assert_eq!(ids_u, ids_b, "expert_ids should match when lambda=0");
    for k in 0..top_k {
        assert!(
            (wts_u[k] - wts_b[k]).abs() < 1e-5,
            "expert_weights[{k}] differ: unbiased={}, biased={}",
            wts_u[k], wts_b[k],
        );
    }
    eprintln!(
        "moe_router_softmax_biased (lambda=0): ids={:?}, weights={:?} -- MATCH",
        ids_b, wts_b,
    );
}

#[test]
fn test_moe_router_softmax_biased_nudges_cached_expert() {
    // With a non-zero lambda and close expert scores, biased routing
    // should prefer a cached expert over a non-cached one.
    let backend = MetalF32Backend::new().unwrap();

    let lib = backend.device
        .new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let biased_fn = lib.get_function("moe_router_softmax_biased").unwrap();
    let biased_pso = backend.device.new_compute_pipeline_state(&biased_fn).unwrap();

    let hidden_dim: usize = 4;
    let num_experts: usize = 4;
    let top_k: usize = 2;

    // Craft hidden state and gate weights so that:
    // Expert 0: logit = 2.0 (highest)
    // Expert 1: logit = 1.9 (close second)
    // Expert 2: logit = 0.5
    // Expert 3: logit = 0.1
    // Without bias: top-2 = [0, 1]
    // With bias lambda=0.5, is_cached=[0,0,0,1]:
    // Expert 3 logit becomes 0.1 + 0.5 = 0.6, still below 1.9.
    // With is_cached=[0,1,0,0] and lambda=0: top-2 = [0, 1] (unchanged)

    // Use simple identity-like weights: gate_weight[e] = [target_logit, 0, 0, 0] / hidden[0]
    // hidden = [1.0, 0.0, 0.0, 0.0] so logit = gate_weight[e][0]
    let hidden = vec![1.0f32, 0.0, 0.0, 0.0];
    let target_logits = [2.0f32, 1.9, 0.5, 0.1];
    let mut gate = vec![0.0f32; num_experts * hidden_dim];
    for e in 0..num_experts {
        gate[e * hidden_dim] = target_logits[e];
    }

    let hidden_buf = backend.upload_f32(&hidden).unwrap();
    let gate_buf = backend.upload_f32(&gate).unwrap();

    // Run with large lambda and expert 2 cached.
    // Logits become: [2.0, 1.9, 0.5+2.0=2.5, 0.1]
    // Top-2 should now be [2, 0] (expert 2 jumps to highest).
    let is_cached = vec![0u8, 0, 1, 0];
    let lambda: f32 = 2.0;

    let is_cached_buf = backend.device.new_buffer_with_bytes(&is_cached).unwrap();
    let ids_buf = backend.device.new_buffer(top_k * 4).unwrap();
    let weights_buf = backend.device.new_buffer(top_k * 4).unwrap();

    {
        let cmd = backend.queue.new_command_buffer().unwrap();
        let enc = cmd.new_compute_encoder().unwrap();
        enc.set_pipeline_state(&biased_pso);
        enc.set_buffer(&hidden_buf, 0, 0);
        enc.set_buffer(&gate_buf, 0, 1);
        enc.set_buffer(&ids_buf, 0, 2);
        enc.set_buffer(&weights_buf, 0, 3);
        enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 4);
        enc.set_bytes(&(num_experts as u32).to_le_bytes(), 5);
        enc.set_bytes(&(top_k as u32).to_le_bytes(), 6);
        enc.set_buffer(&is_cached_buf, 0, 7);
        enc.set_bytes(&lambda.to_le_bytes(), 8);
        let tg = 256u64.min(hidden_dim as u64).max(1);
        enc.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    let mut ids = vec![0u32; top_k];
    let mut wts = vec![0.0f32; top_k];
    ids_buf.read_u32(&mut ids);
    weights_buf.read_f32(&mut wts);

    // Expert 2 (biased logit 2.5) should be top-1.
    assert_eq!(ids[0], 2, "biased expert 2 should be top-1, got {:?}", ids);
    // Expert 0 (logit 2.0) should be top-2.
    assert_eq!(ids[1], 0, "expert 0 should be top-2, got {:?}", ids);

    // Verify renormalization: weights should sum to ~1.0.
    let sum: f32 = wts.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "weights should sum to 1.0, got {sum:.6} (weights={:?})", wts,
    );

    eprintln!(
        "moe_router_softmax_biased (nudge): ids={:?}, weights={:?} -- PASS",
        ids, wts,
    );
}

#[test]
fn test_expert_io_stats_initial_zero() {
    // Verify expert_io_stats returns all zeros on a fresh backend.
    let backend = MetalF32Backend::new().unwrap();
    let (disk, cache, blob) = backend.expert_io_stats();
    assert_eq!(disk, 0, "initial disk bytes should be 0");
    assert_eq!(cache, 0, "initial cache bytes should be 0");
    assert_eq!(blob, 0, "initial blob bytes should be 0");
}

/// Test moe_expert_accum_option_a kernel correctness.
///
/// Creates known expert outputs for 2 selected experts (from 8 total)
/// with hidden_dim=256. Dense layout: expert_outputs[k * hidden_dim + t].
/// Selects experts with weights [0.7, 0.3].
/// Verifies: output[i] = 1.0 + 0.7 * 3.0 + 0.3 * 6.0 = 3.9 for all i
/// (with residual = 1.0).
#[test]
fn test_moe_expert_accum_option_a_correctness() {
    let backend = MetalF32Backend::new().unwrap();

    let lib = backend.device
        .new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let func = lib.get_function("moe_expert_accum_option_a").unwrap();
    let pso = backend.device.new_compute_pipeline_state(&func).unwrap();

    let top_k: usize = 2;
    let hidden_dim: usize = 256;

    // Dense expert outputs: [top_k, hidden_dim]
    // Slot 0 (selected expert): all 3.0
    // Slot 1 (selected expert): all 6.0
    let mut expert_outputs = vec![0.0f32; top_k * hidden_dim];
    for t in 0..hidden_dim {
        expert_outputs[0 * hidden_dim + t] = 3.0;
        expert_outputs[1 * hidden_dim + t] = 6.0;
    }
    let expert_weights = vec![0.7f32, 0.3];
    let residual = vec![1.0f32; hidden_dim];

    // Upload to GPU
    let expert_outputs_buf = backend.upload_f32(&expert_outputs).unwrap();
    let expert_weights_buf = backend.upload_f32(&expert_weights).unwrap();
    let residual_buf = backend.upload_f32(&residual).unwrap();
    let output_buf = backend.device.new_buffer(hidden_dim * 4).unwrap();

    // Dispatch
    let cmd = backend.queue.new_command_buffer().unwrap();
    let enc = cmd.new_compute_encoder().unwrap();
    enc.set_pipeline_state(&pso);
    enc.set_buffer(&expert_outputs_buf, 0, 0);
    enc.set_buffer(&expert_weights_buf, 0, 1);
    enc.set_buffer(&output_buf, 0, 2);
    enc.set_buffer(&residual_buf, 0, 3);
    enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 4);
    enc.set_bytes(&(top_k as u32).to_le_bytes(), 5);
    let tg_count = ((hidden_dim as u64) + 255) / 256;
    enc.dispatch_threadgroups(
        MTLSize::new(tg_count, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // Verify: output[i] = 1.0 + 0.7 * 3.0 + 0.3 * 6.0 = 3.9
    let expected = 1.0 + 0.7 * 3.0 + 0.3 * 6.0;
    let mut result = vec![0.0f32; hidden_dim];
    output_buf.read_f32(&mut result);
    for (i, &v) in result.iter().enumerate() {
        assert!(
            (v - expected).abs() < 1e-4,
            "moe_expert_accum_option_a[{i}]: GPU={v}, expected={expected}"
        );
    }
    eprintln!(
        "moe_expert_accum_option_a: all {} values = {:.4} (expected {:.4}) -- PASS",
        hidden_dim, result[0], expected,
    );
}

/// Test configure_option_a defaults to false.
#[test]
fn test_option_a_default_disabled() {
    let backend = MetalF32Backend::new().unwrap();
    assert!(!backend.use_option_a, "Option A should default to false");
}

/// Test configure_option_a toggle.
#[test]
fn test_option_a_configure() {
    let mut backend = MetalF32Backend::new().unwrap();
    backend.configure_option_a(true);
    assert!(backend.use_option_a, "Option A should be true after configure_option_a(true)");
    backend.configure_option_a(false);
    assert!(!backend.use_option_a, "Option A should be false after configure_option_a(false)");
}

// Minimal WeightProvider stub for tests that need a &dyn WeightProvider
// but never actually load weights (GPU-resident path ignores it).
struct StubWeightProvider;
impl crate::weight::cache::WeightProvider for StubWeightProvider {
    fn prefetch_layer(&self, _l: usize, _p: crate::weight::cache::PrefetchPriority) -> Result<crate::weight::cache::PrefetchHandle, RuntimeError> {
        Err(RuntimeError::Compute("stub".into()))
    }
    fn get_layer_blocking(&self, _l: usize) -> Result<crate::weight::cache::LayerView, RuntimeError> {
        Err(RuntimeError::Compute("stub".into()))
    }
    fn try_get_layer(&self, _l: usize) -> Option<crate::weight::cache::LayerView> { None }
    fn release_layer_hint(&self, _l: usize) {}
    fn stats(&self) -> crate::weight::cache::CacheStats {
        crate::weight::cache::CacheStats {
            layers_cached: 0, bytes_cached: 0, capacity_bytes: 0,
            hits: 0, misses: 0, evictions: 0,
            prefetch_hits: 0, prefetch_misses: 0, inflight_prefetches: 0,
        }
    }
    fn num_layers(&self) -> usize { 0 }
}

// Helper: create minimal ModelHyperparams for tests (dense, non-MoE).
fn test_hyperparams() -> ModelHyperparams {
    ModelHyperparams {
        hidden_dim: 64,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 2,
        vocab_size: 32,
        max_seq_len: 64,
        intermediate_dim: 128,
        head_dim: 16,
        norm_eps: 1e-5,
        rope_params: Some(lumen_format::hyperparams::RopeParams {
            theta: 10000.0,
            scaling_factor: 1.0,
            scaling_type: lumen_format::hyperparams::RopeScalingType::None,
        }),
        num_experts: None,
        num_active_experts: None,
        rotary_dim: None, rope_neox: false,
        gdn: None,
    }
}

/// Test that decode_token_option_a_gpu_resident rejects
/// non-GPU-resident mode (no weights loaded).
#[test]
fn test_option_a_gpu_resident_requires_weights() {
    let mut backend = MetalF32Backend::new().unwrap();
    backend.configure_option_a(true);

    let hp = test_hyperparams();
    backend.init(&hp).unwrap();

    let mut kv = crate::kv::KvCache::new(crate::kv::KvCacheConfig {
        max_seq_len: hp.max_seq_len as usize,
        num_layers: hp.num_layers as usize,
        num_kv_heads: hp.num_kv_heads as usize,
        head_dim: hp.head_dim as usize,
        precision: crate::kv::KvPrecision::F32,
    }).unwrap();
    let weights = StubWeightProvider;

    // Without GPU-resident weights loaded, should get an error.
    let result = backend.decode_token_option_a_gpu_resident(1, &weights, &mut kv);
    assert!(result.is_err(), "Option A GPU-resident should fail without GPU-resident weights");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("GPU-resident"),
        "Error should mention GPU-resident: {err_msg}"
    );
}

/// Test that decode_token_greedy dispatches to Option A
/// when use_option_a is true (validates the routing logic).
#[test]
fn test_option_a_greedy_dispatch_routing() {
    let mut backend = MetalF32Backend::new().unwrap();
    assert!(!backend.use_option_a);

    // Enable Option A: decode_token_greedy should now route to the
    // Option A path. We verify by checking the error references GPU-resident
    // (since no GPU-resident weights are loaded, the Option A path fails).
    backend.configure_option_a(true);

    let hp = test_hyperparams();
    backend.init(&hp).unwrap();

    let mut kv = crate::kv::KvCache::new(crate::kv::KvCacheConfig {
        max_seq_len: hp.max_seq_len as usize,
        num_layers: hp.num_layers as usize,
        num_kv_heads: hp.num_kv_heads as usize,
        head_dim: hp.head_dim as usize,
        precision: crate::kv::KvPrecision::F32,
    }).unwrap();
    let weights = StubWeightProvider;
    let result = backend.decode_token_greedy(1, &weights, &mut kv);
    assert!(result.is_err());
    // The error should come from decode_token_option_a_gpu_resident,
    // confirming the dispatch routing works.
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("option_a_gpu_resident") || err_msg.contains("GPU-resident"),
        "Greedy decode with Option A enabled should route to Option A path: {err_msg}"
    );
}

/// Real hardware MoE I/O benchmark test.
///
/// Opens a MoE LBC file, benchmarks ExpertReader I/O latency for individual
/// and parallel expert loads, exercises the expert cache, and prints I/O stats.
/// Gated behind #[ignore] because it requires a real model file on disk.
///
/// Run with: cargo test -p lumen-runtime -- --ignored test_moe_hw_benchmark
#[ignore = "requires external model file at runtime"]
#[test]
fn test_moe_hw_benchmark() {
    use crate::expert::reader::ExpertReader;
    use crate::expert::cache::ExpertLfuCache;
    use crate::expert::profiler::ExpertActivationProfiler;
    use std::path::Path;
    use std::time::Instant;

    let lbc_path = Path::new("/tmp/lumen-bench/qwen3-5-moe-35b-a3b-Q4_0.lbc");
    if !lbc_path.exists() {
        eprintln!(
            "SKIP: MoE benchmark file not found at {}",
            lbc_path.display()
        );
        return;
    }

    eprintln!("=== MoE Hardware I/O Benchmark ===");
    eprintln!("Model: {}", lbc_path.display());

    // Open model to read hyperparams.
    let lbc = lumen_format::reader::LbcFile::open(lbc_path).unwrap();
    let hp = &lbc.header.hyperparams;
    let num_experts = hp.num_experts.unwrap_or(0) as usize;
    let top_k = hp.num_active_experts.unwrap_or(0) as usize;
    let num_layers = hp.num_layers as usize;
    eprintln!(
        "Architecture: {} layers, hidden_dim={}, experts={}, top-k={}",
        num_layers, hp.hidden_dim, num_experts, top_k,
    );

    if num_experts == 0 {
        eprintln!("SKIP: Model is not MoE (num_experts=0).");
        return;
    }

    // --- Phase 1: Sequential ExpertReader I/O latency ---
    eprintln!("\n--- Phase 1: Sequential ExpertReader I/O ---");
    let mut reader = ExpertReader::open(lbc_path).unwrap();
    let mut total_bytes: u64 = 0;
    let mut load_count: u64 = 0;
    let start_seq = Instant::now();
    // Load top_k experts from layer 0 and layer 1 sequentially.
    for layer in 0..2.min(num_layers) {
        for expert_id in 0..(top_k as u32).min(num_experts as u32) {
            match reader.load_expert(layer, expert_id) {
                Ok((data, slice)) => {
                    total_bytes += data.len() as u64;
                    load_count += 1;
                    eprintln!(
                        "  Layer {} expert {}: {} bytes ({:.2} MB), gate+up+down",
                        layer, expert_id, data.len(), data.len() as f64 / 1e6
                    );
                    let _ = slice; // used for offset info
                }
                Err(e) => {
                    eprintln!("  Layer {} expert {}: FAILED: {}", layer, expert_id, e);
                }
            }
        }
    }
    let elapsed_seq = start_seq.elapsed();
    if load_count > 0 {
        eprintln!(
            "Sequential: {} experts, {:.2} MB total, {:.2?} ({:.2} MB/s)",
            load_count,
            total_bytes as f64 / 1e6,
            elapsed_seq,
            total_bytes as f64 / 1e6 / elapsed_seq.as_secs_f64()
        );
    }

    // --- Phase 2: Parallel ExpertReader I/O latency ---
    eprintln!("\n--- Phase 2: Parallel ExpertReader I/O ---");
    let requests: Vec<(usize, u32)> = (0..2.min(num_layers))
        .flat_map(|l| (0..(top_k as u32).min(num_experts as u32)).map(move |e| (l, e)))
        .collect();
    let start_par = Instant::now();
    let par_reader = ExpertReader::open(lbc_path).unwrap();
    let results = par_reader.load_experts_parallel(&requests);
    let elapsed_par = start_par.elapsed();
    let par_bytes: u64 = results.iter()
        .filter_map(|r| r.as_ref().ok().map(|(d, _)| d.len() as u64))
        .sum();
    let par_ok = results.iter().filter(|r| r.is_ok()).count();
    eprintln!(
        "Parallel: {}/{} succeeded, {:.2} MB, {:.2?} ({:.2} MB/s)",
        par_ok, requests.len(),
        par_bytes as f64 / 1e6,
        elapsed_par,
        par_bytes as f64 / 1e6 / elapsed_par.as_secs_f64()
    );
    if elapsed_seq.as_nanos() > 0 {
        eprintln!(
            "Parallel speedup vs sequential: {:.2}x",
            elapsed_seq.as_secs_f64() / elapsed_par.as_secs_f64()
        );
    }

    // --- Phase 3: Expert cache hit/miss simulation ---
    eprintln!("\n--- Phase 3: Expert Cache Simulation (50 tokens) ---");
    let mut cache = ExpertLfuCache::new(64);
    let mut profiler = ExpertActivationProfiler::new(num_layers, num_experts);
    let mut rng_state: u64 = 42;
    let mut cache_hits: u64 = 0;
    let mut cache_misses: u64 = 0;
    // Dummy ExpertSlice for cache insertion (we only measure cache behavior).
    let dummy_ts = lumen_format::TensorSlice { offset: 0, length: 0, quant: lumen_format::QuantScheme::F32 };
    let dummy_slice = lumen_format::ExpertSlice {
        gate: dummy_ts, up: dummy_ts, down: dummy_ts,
    };
    for _token in 0..50 {
        for layer in 0..num_layers {
            // Simulate router: pick top_k random experts (deterministic xorshift).
            let mut selected = Vec::with_capacity(top_k);
            for _ in 0..top_k {
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                let eid = (rng_state % num_experts as u64) as u32;
                selected.push(eid);
            }
            profiler.record(layer, &selected);
            for &eid in &selected {
                let key: crate::expert::cache::ExpertKey = (layer, eid);
                if cache.get(&key).is_some() {
                    cache_hits += 1;
                } else {
                    cache_misses += 1;
                    cache.insert(key, vec![0u8; 1024], dummy_slice.clone());
                }
            }
        }
    }
    let total_lookups = cache_hits + cache_misses;
    eprintln!(
        "Cache: {} hits, {} misses, {:.1}% hit rate",
        cache_hits, cache_misses,
        if total_lookups > 0 { cache_hits as f64 / total_lookups as f64 * 100.0 } else { 0.0 }
    );
    let stats = cache.stats();
    eprintln!(
        "Cache state: {}/{} entries ({:.2} MB cached)",
        stats.cached_experts, stats.capacity,
        stats.cached_bytes as f64 / 1e6,
    );

    let summary = profiler.summary();
    eprintln!("\nProfiler: {} total tokens", summary.total_tokens);
    for (i, entropy) in summary.per_layer_entropy.iter().enumerate() {
        if i < 4 || i >= num_layers.saturating_sub(2) {
            eprintln!("  Layer {:2}: entropy={:.3}", i, entropy);
        } else if i == 4 {
            eprintln!("  ...");
        }
    }
    if !summary.global_top_experts.is_empty() {
        eprintln!("Top 5 hottest experts:");
        for &(layer, eid, freq) in summary.global_top_experts.iter().take(5) {
            eprintln!("  Layer {} expert {}: freq={:.4}", layer, eid, freq);
        }
    }

    eprintln!("\n=== Benchmark PASS ===");
}

// ====================================================================
// Router kernel correctness + diagnostics tests
// ====================================================================

#[test]
fn test_moe_router_correct_topk() {
    // Verify that `moe_router_softmax` returns the correct top-K experts
    // for a known input where the answer is analytically determined.
    //
    // Setup: hidden_dim=4, num_experts=4, top_k=2.
    // hidden_state = [1.0, 0.0, 0.0, 0.0] -- only the first dimension is active.
    // gate_weight row e = [target_logit_e, 0, 0, 0], so logit = target_logit_e.
    // target logits: expert 0 = 1.0, expert 1 = 3.0, expert 2 = 2.0, expert 3 = 0.5.
    // Expected top-2: [1, 2] (expert 1 highest, expert 2 second).
    let backend = MetalF32Backend::new().unwrap();

    let lib = backend.device
        .new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let router_fn = lib.get_function("moe_router_softmax").unwrap();
    let router_pso = backend.device.new_compute_pipeline_state(&router_fn).unwrap();

    let hidden_dim: usize = 4;
    let num_experts: usize = 4;
    let top_k: usize = 2;

    let hidden = vec![1.0f32, 0.0, 0.0, 0.0];
    let target_logits = [1.0f32, 3.0, 2.0, 0.5];
    let mut gate = vec![0.0f32; num_experts * hidden_dim];
    for e in 0..num_experts {
        gate[e * hidden_dim] = target_logits[e];
    }

    let hidden_buf = backend.upload_f32(&hidden).unwrap();
    let gate_buf = backend.upload_f32(&gate).unwrap();
    let ids_buf = backend.device.new_buffer(top_k * 4).unwrap();
    let weights_buf = backend.device.new_buffer(top_k * 4).unwrap();

    {
        let cmd = backend.queue.new_command_buffer().unwrap();
        let enc = cmd.new_compute_encoder().unwrap();
        enc.set_pipeline_state(&router_pso);
        enc.set_buffer(&hidden_buf, 0, 0);
        enc.set_buffer(&gate_buf, 0, 1);
        enc.set_buffer(&ids_buf, 0, 2);
        enc.set_buffer(&weights_buf, 0, 3);
        enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 4);
        enc.set_bytes(&(num_experts as u32).to_le_bytes(), 5);
        enc.set_bytes(&(top_k as u32).to_le_bytes(), 6);
        let tg = 256u64.min(hidden_dim as u64).max(1);
        enc.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    let mut ids = vec![0u32; top_k];
    let mut wts = vec![0.0f32; top_k];
    ids_buf.read_u32(&mut ids);
    weights_buf.read_f32(&mut wts);

    // (a) Top-K experts are returned: expert 1 (logit 3.0) first, expert 2 (logit 2.0) second.
    assert_eq!(ids[0], 1, "top-1 should be expert 1 (logit 3.0), got expert {}", ids[0]);
    assert_eq!(ids[1], 2, "top-2 should be expert 2 (logit 2.0), got expert {}", ids[1]);

    // (b) Weights sum to 1.0 (renormalized).
    let sum: f32 = wts.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "weights should sum to 1.0, got {sum:.6} (weights={wts:?})"
    );

    // (c) The correct argmax is selected: weight[0] > weight[1]
    // because softmax(3.0) > softmax(2.0).
    assert!(
        wts[0] > wts[1],
        "weight for expert 1 ({:.4}) should be > weight for expert 2 ({:.4})",
        wts[0], wts[1]
    );

    // Verify the weights approximately match the analytical softmax.
    // softmax(3.0) / (softmax(3.0) + softmax(2.0)) = e^3 / (e^3 + e^2) = e / (e + 1)
    let expected_w0 = std::f32::consts::E / (std::f32::consts::E + 1.0);
    let expected_w1 = 1.0 / (std::f32::consts::E + 1.0);
    assert!(
        (wts[0] - expected_w0).abs() < 1e-4,
        "weight[0]={:.6}, expected {:.6}", wts[0], expected_w0
    );
    assert!(
        (wts[1] - expected_w1).abs() < 1e-4,
        "weight[1]={:.6}, expected {:.6}", wts[1], expected_w1
    );

    eprintln!(
        "moe_router_correct_topk: ids={:?}, weights={:?} -- PASS (expected w0={:.4}, w1={:.4})",
        ids, wts, expected_w0, expected_w1,
    );
}

#[test]
fn test_moe_router_parallel_matches_serial() {
    // The parallel two-kernel router (moe_router_logits_f32 +
    // moe_router_topk_softmax) must select the SAME top-k experts and weights as
    // the legacy single-threadgroup moe_router_softmax, for a 256-expert config
    // mirroring Qwen3.5-35B-A3B. This locks in the runtime byte-identity result.
    let backend = MetalF32Backend::new().unwrap();
    let lib = backend.device.new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let serial = backend.device
        .new_compute_pipeline_state(&lib.get_function("moe_router_softmax").unwrap()).unwrap();
    let logits_pso = backend.device
        .new_compute_pipeline_state(&lib.get_function("moe_router_logits_f32").unwrap()).unwrap();
    let topk_pso = backend.device
        .new_compute_pipeline_state(&lib.get_function("moe_router_topk_softmax").unwrap()).unwrap();

    let hidden_dim: usize = 2048;
    let num_experts: usize = 256;
    let top_k: usize = 8;

    // Deterministic pseudo-random hidden + gate weights.
    let mut seed: u32 = 0x1234_5678;
    let mut rng = || { seed = seed.wrapping_mul(1664525).wrapping_add(1013904223); (seed >> 8) as f32 / 16_777_216.0 - 0.5 };
    let hidden: Vec<f32> = (0..hidden_dim).map(|_| rng()).collect();
    let gate: Vec<f32> = (0..num_experts * hidden_dim).map(|_| rng()).collect();

    let hidden_buf = backend.upload_f32(&hidden).unwrap();
    let gate_buf = backend.upload_f32(&gate).unwrap();

    let run = |parallel: bool| -> (Vec<u32>, Vec<f32>) {
        let ids_buf = backend.device.new_buffer(top_k * 4).unwrap();
        let wts_buf = backend.device.new_buffer(top_k * 4).unwrap();
        let cmd = backend.queue.new_command_buffer().unwrap();
        let enc = cmd.new_compute_encoder().unwrap();
        if parallel {
            let logits_buf = backend.device.new_buffer(num_experts * 4).unwrap();
            enc.set_pipeline_state(&logits_pso);
            enc.set_buffer(&hidden_buf, 0, 0);
            enc.set_buffer(&gate_buf, 0, 1);
            enc.set_buffer(&logits_buf, 0, 2);
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
            enc.set_bytes(&(num_experts as u32).to_le_bytes(), 4);
            enc.dispatch_threadgroups(MTLSize::new(num_experts as u64, 1, 1), MTLSize::new(256, 1, 1));
            enc.memory_barrier_with_scope(1);
            enc.set_pipeline_state(&topk_pso);
            enc.set_buffer(&logits_buf, 0, 0);
            enc.set_buffer(&ids_buf, 0, 1);
            enc.set_buffer(&wts_buf, 0, 2);
            enc.set_bytes(&(num_experts as u32).to_le_bytes(), 3);
            enc.set_bytes(&(top_k as u32).to_le_bytes(), 4);
            enc.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        } else {
            enc.set_pipeline_state(&serial);
            enc.set_buffer(&hidden_buf, 0, 0);
            enc.set_buffer(&gate_buf, 0, 1);
            enc.set_buffer(&ids_buf, 0, 2);
            enc.set_buffer(&wts_buf, 0, 3);
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 4);
            enc.set_bytes(&(num_experts as u32).to_le_bytes(), 5);
            enc.set_bytes(&(top_k as u32).to_le_bytes(), 6);
            enc.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let mut ids = vec![0u32; top_k];
        let mut wts = vec![0.0f32; top_k];
        ids_buf.read_u32(&mut ids);
        wts_buf.read_f32(&mut wts);
        (ids, wts)
    };

    let (ids_s, wts_s) = run(false);
    let (ids_p, wts_p) = run(true);

    assert_eq!(ids_s, ids_p, "parallel router selected different experts: serial={ids_s:?} parallel={ids_p:?}");
    for k in 0..top_k {
        assert!(
            (wts_s[k] - wts_p[k]).abs() < 1e-5,
            "weight mismatch at k={k}: serial={} parallel={}", wts_s[k], wts_p[k]
        );
    }
    eprintln!("router_parallel_matches_serial: ids={ids_p:?} weights match within 1e-5 -- PASS");
}

#[test]
fn test_moe_router_diversity_non_degenerate() {
    // Verify that different hidden states produce different expert selections.
    // This proves the router kernel is sensitive to input, ruling out
    // a wiring bug where the router always selects the same expert.
    let backend = MetalF32Backend::new().unwrap();

    let lib = backend.device
        .new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let router_fn = lib.get_function("moe_router_softmax").unwrap();
    let router_pso = backend.device.new_compute_pipeline_state(&router_fn).unwrap();

    let hidden_dim: usize = 8;
    let num_experts: usize = 8;
    let top_k: usize = 2;

    // Gate weights: identity-like -- expert e's gating vector is a one-hot
    // in dimension e. So logit[e] = hidden_state[e].
    let mut gate = vec![0.0f32; num_experts * hidden_dim];
    for e in 0..num_experts {
        gate[e * hidden_dim + e] = 1.0;
    }
    let gate_buf = backend.upload_f32(&gate).unwrap();

    // Run several hidden states that should route to different experts.
    let test_cases: Vec<(Vec<f32>, u32)> = vec![
        // hidden_state = [5, 0, 0, 0, 0, 0, 0, 0] -> logit[0]=5, expect top-1 = expert 0
        ({let mut v = vec![0.0f32; hidden_dim]; v[0] = 5.0; v}, 0),
        // hidden_state = [0, 0, 0, 5, 0, 0, 0, 0] -> logit[3]=5, expect top-1 = expert 3
        ({let mut v = vec![0.0f32; hidden_dim]; v[3] = 5.0; v}, 3),
        // hidden_state = [0, 0, 0, 0, 0, 0, 0, 5] -> logit[7]=5, expect top-1 = expert 7
        ({let mut v = vec![0.0f32; hidden_dim]; v[7] = 5.0; v}, 7),
        // hidden_state = [0, 0, 5, 0, 0, 0, 0, 0] -> logit[2]=5, expect top-1 = expert 2
        ({let mut v = vec![0.0f32; hidden_dim]; v[2] = 5.0; v}, 2),
    ];

    let mut selected_experts: Vec<u32> = Vec::new();

    for (hidden, expected_top1) in &test_cases {
        let hidden_buf = backend.upload_f32(hidden).unwrap();
        let ids_buf = backend.device.new_buffer(top_k * 4).unwrap();
        let weights_buf = backend.device.new_buffer(top_k * 4).unwrap();

        {
            let cmd = backend.queue.new_command_buffer().unwrap();
            let enc = cmd.new_compute_encoder().unwrap();
            enc.set_pipeline_state(&router_pso);
            enc.set_buffer(&hidden_buf, 0, 0);
            enc.set_buffer(&gate_buf, 0, 1);
            enc.set_buffer(&ids_buf, 0, 2);
            enc.set_buffer(&weights_buf, 0, 3);
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 4);
            enc.set_bytes(&(num_experts as u32).to_le_bytes(), 5);
            enc.set_bytes(&(top_k as u32).to_le_bytes(), 6);
            let tg = 256u64.min(hidden_dim as u64).max(1);
            enc.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(tg, 1, 1));
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        let mut ids = vec![0u32; top_k];
        let mut wts = vec![0.0f32; top_k];
        ids_buf.read_u32(&mut ids);
        weights_buf.read_f32(&mut wts);

        assert_eq!(
            ids[0], *expected_top1,
            "Expected top-1 = expert {}, got expert {} (hidden={:?})",
            expected_top1, ids[0], hidden,
        );

        // Verify renormalization.
        let sum: f32 = wts.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "weights should sum to 1.0, got {sum:.6}"
        );

        selected_experts.push(ids[0]);
        eprintln!(
            "  hidden dim{} dominant -> top-1=expert {}, weight={:.4} -- OK",
            hidden.iter().position(|&x| x > 1.0).unwrap_or(0), ids[0], wts[0],
        );
    }

    // Verify that we got diversity: at least 3 different experts were selected.
    let unique: std::collections::HashSet<u32> = selected_experts.iter().copied().collect();
    assert!(
        unique.len() >= 3,
        "Expected >= 3 unique expert selections, got {} ({:?})",
        unique.len(), selected_experts,
    );

    eprintln!(
        "moe_router_diversity: {} unique experts across {} inputs -- PASS",
        unique.len(), test_cases.len(),
    );
}

#[test]
fn test_moe_router_large_hidden_dim() {
    // Verify correctness with a larger hidden_dim (256) to exercise
    // the multi-thread cooperative reduction code path in the kernel.
    let backend = MetalF32Backend::new().unwrap();

    let lib = backend.device
        .new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let router_fn = lib.get_function("moe_router_softmax").unwrap();
    let router_pso = backend.device.new_compute_pipeline_state(&router_fn).unwrap();

    let hidden_dim: usize = 256;
    let num_experts: usize = 8;
    let top_k: usize = 2;

    // hidden_state: all 1.0.
    // gate_weight: row e has all values = (e+1) / hidden_dim.
    // logit[e] = sum over hidden_dim of (e+1)/hidden_dim * 1.0 = (e+1).
    // So logit[0]=1, logit[1]=2, ..., logit[7]=8.
    // Top-2: expert 7 (logit 8), expert 6 (logit 7).
    let hidden = vec![1.0f32; hidden_dim];
    let mut gate = vec![0.0f32; num_experts * hidden_dim];
    for e in 0..num_experts {
        let val = (e + 1) as f32 / hidden_dim as f32;
        for j in 0..hidden_dim {
            gate[e * hidden_dim + j] = val;
        }
    }

    let hidden_buf = backend.upload_f32(&hidden).unwrap();
    let gate_buf = backend.upload_f32(&gate).unwrap();
    let ids_buf = backend.device.new_buffer(top_k * 4).unwrap();
    let weights_buf = backend.device.new_buffer(top_k * 4).unwrap();

    {
        let cmd = backend.queue.new_command_buffer().unwrap();
        let enc = cmd.new_compute_encoder().unwrap();
        enc.set_pipeline_state(&router_pso);
        enc.set_buffer(&hidden_buf, 0, 0);
        enc.set_buffer(&gate_buf, 0, 1);
        enc.set_buffer(&ids_buf, 0, 2);
        enc.set_buffer(&weights_buf, 0, 3);
        enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 4);
        enc.set_bytes(&(num_experts as u32).to_le_bytes(), 5);
        enc.set_bytes(&(top_k as u32).to_le_bytes(), 6);
        let tg = 256u64.min(hidden_dim as u64).max(1);
        enc.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    let mut ids = vec![0u32; top_k];
    let mut wts = vec![0.0f32; top_k];
    ids_buf.read_u32(&mut ids);
    weights_buf.read_f32(&mut wts);

    assert_eq!(ids[0], 7, "top-1 should be expert 7, got {}", ids[0]);
    assert_eq!(ids[1], 6, "top-2 should be expert 6, got {}", ids[1]);

    let sum: f32 = wts.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "weights should sum to 1.0, got {sum:.6}"
    );
    assert!(wts[0] > wts[1], "w0={:.4} should be > w1={:.4}", wts[0], wts[1]);

    eprintln!(
        "moe_router_large_hidden_dim: ids={:?}, weights={:?} -- PASS",
        ids, wts,
    );
}

/// Verify that perfectly uniform logits produce expert 0 (tiebreaker).
/// This confirms the strict `>` argmax tiebreaker always picks the lowest index.
#[test]
fn test_moe_router_uniform_logits_selects_expert_zero() {
    let backend = MetalF32Backend::new().unwrap();

    let lib = backend.device
        .new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let router_fn = lib.get_function("moe_router_softmax").unwrap();
    let router_pso = backend.device.new_compute_pipeline_state(&router_fn).unwrap();

    let hidden_dim: usize = 8;
    let num_experts: usize = 8;
    let top_k: usize = 2;

    // All-ones hidden state + uniform gate weights -> all logits identical.
    let hidden = vec![1.0f32; hidden_dim];
    // Every expert row is identical: [1/hidden_dim, 1/hidden_dim, ...].
    // logit[e] = sum(1.0 * 1/8) = 1.0 for all e.
    let gate = vec![1.0f32 / hidden_dim as f32; num_experts * hidden_dim];

    let hidden_buf = backend.upload_f32(&hidden).unwrap();
    let gate_buf = backend.upload_f32(&gate).unwrap();
    let ids_buf = backend.device.new_buffer(top_k * 4).unwrap();
    let weights_buf = backend.device.new_buffer(top_k * 4).unwrap();

    {
        let cmd = backend.queue.new_command_buffer().unwrap();
        let enc = cmd.new_compute_encoder().unwrap();
        enc.set_pipeline_state(&router_pso);
        enc.set_buffer(&hidden_buf, 0, 0);
        enc.set_buffer(&gate_buf, 0, 1);
        enc.set_buffer(&ids_buf, 0, 2);
        enc.set_buffer(&weights_buf, 0, 3);
        enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 4);
        enc.set_bytes(&(num_experts as u32).to_le_bytes(), 5);
        enc.set_bytes(&(top_k as u32).to_le_bytes(), 6);
        let tg = 256u64.min(hidden_dim as u64).max(1);
        enc.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    let mut ids = vec![0u32; top_k];
    let mut wts = vec![0.0f32; top_k];
    ids_buf.read_u32(&mut ids);
    weights_buf.read_f32(&mut wts);

    // With uniform logits, strict `>` tiebreaker picks expert 0 first, expert 1 second.
    assert_eq!(ids[0], 0, "uniform logits: top-1 should be expert 0 (tiebreaker), got {}", ids[0]);
    assert_eq!(ids[1], 1, "uniform logits: top-2 should be expert 1 (tiebreaker), got {}", ids[1]);

    // Weights should be equal (0.5 each after renormalization of 2 equal probs).
    let spread = (wts[0] - wts[1]).abs();
    assert!(spread < 1e-5, "uniform logits: weight spread should be ~0, got {spread:.6}");

    eprintln!(
        "moe_router_uniform_logits: ids={:?}, weights={:?}, spread={:.6} -- PASS",
        ids, wts, spread,
    );
}

/// Verify that a tiny logit difference (epsilon) correctly routes
/// to the higher-logit expert, not the tiebreaker expert 0.
#[test]
fn test_moe_router_tiny_logit_difference_routes_correctly() {
    let backend = MetalF32Backend::new().unwrap();

    let lib = backend.device
        .new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let router_fn = lib.get_function("moe_router_softmax").unwrap();
    let router_pso = backend.device.new_compute_pipeline_state(&router_fn).unwrap();

    let hidden_dim: usize = 8;
    let num_experts: usize = 8;
    let top_k: usize = 2;

    // Hidden state: all ones.
    let hidden = vec![1.0f32; hidden_dim];
    // Gate: mostly uniform, but expert 5 has a tiny extra bias.
    let mut gate = vec![1.0f32 / hidden_dim as f32; num_experts * hidden_dim];
    // Add a small epsilon to expert 5's row to make logit[5] slightly larger.
    let epsilon = 0.01f32;
    for j in 0..hidden_dim {
        gate[5 * hidden_dim + j] += epsilon / hidden_dim as f32;
    }

    let hidden_buf = backend.upload_f32(&hidden).unwrap();
    let gate_buf = backend.upload_f32(&gate).unwrap();
    let ids_buf = backend.device.new_buffer(top_k * 4).unwrap();
    let weights_buf = backend.device.new_buffer(top_k * 4).unwrap();

    {
        let cmd = backend.queue.new_command_buffer().unwrap();
        let enc = cmd.new_compute_encoder().unwrap();
        enc.set_pipeline_state(&router_pso);
        enc.set_buffer(&hidden_buf, 0, 0);
        enc.set_buffer(&gate_buf, 0, 1);
        enc.set_buffer(&ids_buf, 0, 2);
        enc.set_buffer(&weights_buf, 0, 3);
        enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 4);
        enc.set_bytes(&(num_experts as u32).to_le_bytes(), 5);
        enc.set_bytes(&(top_k as u32).to_le_bytes(), 6);
        let tg = 256u64.min(hidden_dim as u64).max(1);
        enc.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    let mut ids = vec![0u32; top_k];
    let mut wts = vec![0.0f32; top_k];
    ids_buf.read_u32(&mut ids);
    weights_buf.read_f32(&mut wts);

    // Expert 5 should be top-1 because it has the highest logit.
    assert_eq!(ids[0], 5, "tiny diff: top-1 should be expert 5, got {}", ids[0]);
    // Weight spread should be tiny but nonzero.
    let spread = wts[0] - wts[1];
    assert!(spread > 0.0, "tiny diff: weight spread should be > 0, got {spread:.6}");

    eprintln!(
        "moe_router_tiny_logit_diff: ids={:?}, weights={:?}, spread={:.6} -- PASS",
        ids, wts, spread,
    );
}

/// Verify RouterLayerStats.weight_spread captures top1-top2 difference.
#[test]
fn test_router_layer_stats_weight_spread() {
    let stats = RouterLayerStats {
        layer: 0,
        expert_ids: vec![3, 1],
        expert_weights: vec![0.75, 0.25],
        weight_spread: 0.5,
    };
    assert_eq!(stats.weight_spread, 0.5);

    // Verify spread computation matches what readback would produce.
    let computed = stats.expert_weights[0] - stats.expert_weights[1];
    assert!((computed - stats.weight_spread).abs() < 1e-7,
        "weight_spread={} should match w0-w1={}", stats.weight_spread, computed);
}

/// Verify router_debug_summary diagnoses near-zero spread as Q4_0 issue.
#[test]
fn test_router_debug_summary_degenerate_near_zero_spread() {
    let backend = MetalF32Backend::new().unwrap();

    // Simulate degenerate routing with near-zero weight spread (uniform softmax).
    {
        let mut log = backend.router_debug_log.lock().unwrap();
        for _token in 0..5 {
            log.push(RouterLayerStats {
                layer: 0,
                expert_ids: vec![0, 1],
                expert_weights: vec![0.1251, 0.1249],
                weight_spread: 0.0002,
            });
            log.push(RouterLayerStats {
                layer: 1,
                expert_ids: vec![0, 1],
                expert_weights: vec![0.1253, 0.1247],
                weight_spread: 0.0006,
            });
        }
    }

    let summary = backend.router_debug_summary().unwrap();

    // Should detect degenerate routing.
    assert!(
        summary.contains("entropy=0.000"),
        "near-zero spread should have entropy=0: {summary}"
    );

    // Should contain the weight spread analysis.
    assert!(
        summary.contains("Weight spread analysis"),
        "summary should contain weight spread analysis: {summary}"
    );

    // Should diagnose near-zero spread as model quality issue.
    assert!(
        summary.contains("Near-zero weight spread"),
        "summary should diagnose near-zero spread: {summary}"
    );
    assert!(
        summary.contains("model quality issue"),
        "summary should identify model quality issue: {summary}"
    );

    eprintln!("router_debug_summary (near-zero spread):\n{summary}");
}

#[test]
fn test_configure_router_debug() {
    let mut backend = MetalF32Backend::new().unwrap();
    assert!(!backend.router_debug_enabled, "router debug should default to false");
    backend.configure_router_debug(true);
    assert!(backend.router_debug_enabled, "router debug should be true after configure");
    backend.configure_router_debug(false);
    assert!(!backend.router_debug_enabled, "router debug should be false after disable");
}

#[test]
fn test_router_debug_log_empty() {
    let backend = MetalF32Backend::new().unwrap();
    let log = backend.get_router_debug_log();
    assert!(log.is_empty(), "router debug log should be empty initially");
    assert!(
        backend.router_debug_summary().is_none(),
        "router debug summary should be None when no data collected"
    );
}

#[test]
fn test_router_debug_summary_format() {
    // Manually populate the router debug log and verify the summary output.
    let backend = MetalF32Backend::new().unwrap();

    // Simulate 3 tokens across 2 layers with degenerate routing.
    {
        let mut log = backend.router_debug_log.lock().unwrap();
        for _token in 0..3 {
            log.push(RouterLayerStats {
                layer: 0,
                expert_ids: vec![0, 1],
                expert_weights: vec![0.8, 0.2],
                weight_spread: 0.6,
            });
            log.push(RouterLayerStats {
                layer: 1,
                expert_ids: vec![0, 1],
                expert_weights: vec![0.9, 0.1],
                weight_spread: 0.8,
            });
        }
    }

    let summary = backend.router_debug_summary().unwrap();
    assert!(
        summary.contains("Router Diagnostics"),
        "summary should contain header"
    );
    assert!(
        summary.contains("entropy=0.000"),
        "degenerate routing should have entropy=0: {summary}"
    );
    assert!(
        summary.contains("always same expert"),
        "summary should flag degenerate routing: {summary}"
    );

    eprintln!("router_debug_summary (degenerate):\n{summary}");
}

#[test]
fn test_router_debug_summary_diverse() {
    // Populate with diverse routing and verify non-degenerate summary.
    let backend = MetalF32Backend::new().unwrap();

    {
        let mut log = backend.router_debug_log.lock().unwrap();
        // Token 0: layer 0 selects expert 0, layer 1 selects expert 3
        log.push(RouterLayerStats { layer: 0, expert_ids: vec![0, 1], expert_weights: vec![0.7, 0.3], weight_spread: 0.4 });
        log.push(RouterLayerStats { layer: 1, expert_ids: vec![3, 2], expert_weights: vec![0.6, 0.4], weight_spread: 0.2 });
        // Token 1: layer 0 selects expert 2, layer 1 selects expert 1
        log.push(RouterLayerStats { layer: 0, expert_ids: vec![2, 0], expert_weights: vec![0.5, 0.5], weight_spread: 0.0 });
        log.push(RouterLayerStats { layer: 1, expert_ids: vec![1, 0], expert_weights: vec![0.6, 0.4], weight_spread: 0.2 });
        // Token 2: layer 0 selects expert 1, layer 1 selects expert 0
        log.push(RouterLayerStats { layer: 0, expert_ids: vec![1, 3], expert_weights: vec![0.55, 0.45], weight_spread: 0.1 });
        log.push(RouterLayerStats { layer: 1, expert_ids: vec![0, 2], expert_weights: vec![0.7, 0.3], weight_spread: 0.4 });
    }

    let summary = backend.router_debug_summary().unwrap();
    assert!(
        summary.contains("non-degenerate"),
        "diverse routing should be flagged as non-degenerate: {summary}"
    );
    // Entropy should be > 0 for both layers.
    assert!(
        !summary.contains("entropy=0.000"),
        "diverse routing should not have entropy=0: {summary}"
    );

    eprintln!("router_debug_summary (diverse):\n{summary}");
}

/// Integration test: run moe_router_softmax on the supported MoE model and check routing entropy.
/// Requires the LBC model file at the expected path.
/// Run with: cargo test -p lumen-runtime -- --ignored test_moe_routing_entropy
#[ignore = "requires external model file at runtime"]
#[test]
fn test_moe_routing_entropy() {
    use std::path::Path;

    let lbc_path = Path::new("/tmp/lumen-bench/qwen3-5-moe-35b-a3b-Q4_0.lbc");
    if !lbc_path.exists() {
        eprintln!(
            "SKIP: MoE LBC file not found at {}",
            lbc_path.display()
        );
        return;
    }

    eprintln!("=== MoE Routing Entropy Test ===");

    // Open the model.
    let mmap_config = crate::storage::MmapConfig {
        prefetch_window: 2,
        advise_sequential: true,
        release_with_dontneed: true,
    };
    let provider = crate::weight::provider_mmap::MmapWeightProvider::open(lbc_path, mmap_config).unwrap();
    let hp = &provider.lbc().header.hyperparams;

    let num_experts = hp.num_experts.unwrap_or(0) as usize;
    let top_k = hp.num_active_experts.unwrap_or(0) as usize;
    let num_layers = hp.num_layers as usize;

    if num_experts == 0 {
        eprintln!("SKIP: Model is not MoE (num_experts=0).");
        return;
    }
    eprintln!(
        "Model: {} layers, hidden_dim={}, experts={}, top_k={}",
        num_layers, hp.hidden_dim, num_experts, top_k,
    );

    // Create Metal backend with router debug.
    let mut metal = MetalF32Backend::new().unwrap();
    metal.set_global_tensors(
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
    );
    if !provider.output_proj_raw.is_empty() {
        metal.set_output_proj_q8(provider.output_proj_raw.clone(), provider.output_proj_quant);
    }
    if !provider.embedding_raw.is_empty() {
        metal.set_embedding_raw(provider.embedding_raw.clone(), provider.embedding_quant);
    }
    if provider.weight_tying {
        metal.set_weight_tying(true);
    }
    metal.configure_router_debug(true);

    // Use capped context length.
    let mut hp_capped = *hp;
    hp_capped.max_seq_len = hp.max_seq_len.min(512);
    metal.init(&hp_capped).unwrap();

    // Pre-load GPU-resident weights.
    metal.preload_weights_gpu_resident(&provider).unwrap();

    // Run 20-token decode with a simple prompt.
    let prompt_tokens: Vec<u32> = vec![1, 15043, 29892, 1128, 338]; // "Hello, how are"
    let config = crate::config::RuntimeConfig {
        pipeline_mode: crate::pipeline::PipelineMode::MinMem,
        prefetch_distance: 2,
        kv_precision: crate::kv::KvPrecision::F32,
        max_seq_len: hp_capped.max_seq_len as usize,
        collect_per_layer_timings: false,
    };
    let engine = crate::engine::InferenceEngine::new(config, hp_capped);
    let sampling = crate::engine::SamplingParams {
        temperature: 0.0,
        seed: Some(42),
        ..Default::default()
    };
    let stop = crate::engine::StopCondition::MaxTokens(20);

    match engine.generate(
        &prompt_tokens, &provider, &metal as &dyn ComputeBackend, &stop, &sampling,
    ) {
        Ok(result) => {
            eprintln!("Generated {} tokens: {:?}", result.tokens.len(), result.tokens);
        }
        Err(e) => {
            eprintln!("Inference failed: {e}");
            eprintln!("(This may be expected if model file is incomplete.)");
        }
    }

    // Analyze the router debug log.
    let log = metal.get_router_debug_log();
    eprintln!("\nRouter debug log: {} entries", log.len());

    if log.is_empty() {
        eprintln!("WARNING: No router debug entries collected.");
        eprintln!("This may indicate the model did not run MoE decode tokens.");
        return;
    }

    // Check if routing entropy across tokens > 0.
    let max_layer = log.iter().map(|s| s.layer).max().unwrap_or(0);
    let mut any_diverse = false;

    for layer in 0..=max_layer {
        let entries: Vec<&RouterLayerStats> = log.iter()
            .filter(|s| s.layer == layer)
            .collect();
        if entries.is_empty() { continue; }

        let top1_experts: Vec<u32> = entries.iter()
            .filter_map(|e| e.expert_ids.first().copied())
            .collect();
        let unique: std::collections::HashSet<u32> = top1_experts.iter().copied().collect();
        let avg_w0: f32 = entries.iter()
            .filter_map(|e| e.expert_weights.first().copied())
            .sum::<f32>() / entries.len() as f32;

        if unique.len() > 1 { any_diverse = true; }

        eprintln!(
            "  Layer {:2}: {} tokens, {} unique top-1 experts {:?}, avg_top1_weight={:.4}",
            layer, entries.len(), unique.len(), unique, avg_w0,
        );
    }

    if !any_diverse {
        eprintln!("\nFINDING: ALL layers always select the same top-1 expert.");
        eprintln!("Routing entropy = 0 across all {} tokens.", log.len() / (max_layer + 1).max(1));
        eprintln!("Possible causes:");
        eprintln!("  1. Router weights collapsed due to Q4_0 quantization");
        eprintln!("  2. Short prompt does not provide enough diversity");
        eprintln!("  3. Bug in router weight offsets or hidden state wiring");
    } else {
        eprintln!("\nRouting shows diversity (at least one layer selects different experts).");
    }

    eprintln!("\n=== MoE Routing Entropy Test COMPLETE ===");
}

// ============================================================================
// Q8_0 Batched MoE kernel tests
// ============================================================================

/// Helper: convert f32 to f16 bit pattern (simplified, matches basic.rs)
fn f16_bits_to_f32_moe(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;
    let out = if exp == 0 {
        if frac == 0 { sign << 31 } else {
            // subnormal
            let mut e = -1i32; let mut f = frac;
            while (f & 0x400) == 0 { f <<= 1; e -= 1; }
            f &= 0x3FF;
            let new_exp = (e + 127 + 1 - 15) as u32;
            (sign << 31) | (new_exp << 23) | (f << 13)
        }
    } else if exp == 0x1F {
        (sign << 31) | 0x7F80_0000 | (frac << 13)
    } else {
        let new_exp = exp + 127 - 15;
        (sign << 31) | (new_exp << 23) | (frac << 13)
    };
    f32::from_bits(out)
}

fn f32_to_f16_bits_moe(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7FFFFF;

    if exp == 0 {
        return (sign << 15) as u16;
    }
    if exp == 0xFF {
        let f16_frac = if frac != 0 { 0x200u16 } else { 0 };
        return ((sign << 15) | 0x7C00 | f16_frac as u32) as u16;
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return ((sign << 15) | 0x7C00) as u16;
    }
    if new_exp <= 0 {
        return (sign << 15) as u16;
    }
    let f16_frac = (frac >> 13) as u16;
    ((sign << 15) as u16) | ((new_exp as u16) << 10) | f16_frac
}

/// Helper: encode a float matrix as Q8_0 blocks.
///
/// Input: `rows` x `cols` matrix (row-major). `cols` must be multiple of 32.
/// Output: byte vector with Q8_0 encoding (2-byte f16 scale + 32 int8 values per block).
fn encode_q8_0_matrix(data: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(cols % 32, 0, "cols must be multiple of 32");
    assert_eq!(data.len(), rows * cols);

    let blocks_per_row = cols / 32;
    let block_size = 34usize; // 2 bytes scale + 32 bytes data
    let mut out = vec![0u8; rows * blocks_per_row * block_size];

    for r in 0..rows {
        for b in 0..blocks_per_row {
            let block_start = (r * blocks_per_row + b) * block_size;
            // Find max absolute value in this block of 32 elements
            let data_start = r * cols + b * 32;
            let block_data = &data[data_start..data_start + 32];
            let max_abs = block_data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

            // Scale: max_abs / 127.0
            let scale = if max_abs == 0.0 { 0.0 } else { max_abs / 127.0 };
            let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };

            let scale_bits = f32_to_f16_bits_moe(scale);
            out[block_start] = (scale_bits & 0xFF) as u8;
            out[block_start + 1] = (scale_bits >> 8) as u8;

            // Quantize each value: round(value / scale), clamped to [-128, 127]
            for j in 0..32 {
                let q = (block_data[j] * inv_scale).round().clamp(-128.0, 127.0) as i8;
                out[block_start + 2 + j] = q as u8;
            }
        }
    }
    out
}

/// Test moe_batched_gate_up_swiglu_q8_0 kernel correctness.
///
/// Setup: 4 experts with inter_dim=64, hidden_dim=64.
/// Top-2 selection: experts [1, 3].
/// Gate and up weight matrices are simple scalars (all values = scale * 1).
/// Input x = [1.0; 64].
///
/// For each expert k, the gate+up+swiglu output at row `row` is:
///   gate_dot = sum_i gate_w[row, i] * x[i]
///   up_dot   = sum_i up_w[row, i] * x[i]
///   swiglu   = gate_dot * sigmoid(gate_dot) * up_dot
#[test]
fn test_moe_batched_gate_up_swiglu_q8_0_correctness() {
    let backend = MetalF32Backend::new().unwrap();

    let lib = backend.device
        .new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let func = lib.get_function("moe_batched_gate_up_swiglu_q8_0").unwrap();
    let pso = backend.device.new_compute_pipeline_state(&func).unwrap();

    let num_experts: usize = 4;
    let hidden_dim: usize = 64;
    let inter_dim: usize = 64;
    let top_k: usize = 2;

    // Expert weight matrices: each expert has gate [inter_dim, hidden_dim] and up [inter_dim, hidden_dim].
    // Expert 0: gate all 0.5, up all 0.5
    // Expert 1: gate all 1.0, up all 0.5
    // Expert 2: gate all 0.5, up all 1.0
    // Expert 3: gate all 2.0, up all 1.0
    let expert_gate_vals = [0.5f32, 1.0, 0.5, 2.0];
    let expert_up_vals = [0.5f32, 0.5, 1.0, 1.0];

    // Build a "layer buffer" with all expert gate and up weights packed contiguously.
    // Layout: [expert0_gate | expert0_up | expert1_gate | expert1_up | ...]
    let blocks_per_row = hidden_dim / 32;
    let q8_block_size = 34usize;
    let expert_weight_bytes = inter_dim * blocks_per_row * q8_block_size;

    let mut layer_buf_data = vec![0u8; num_experts * 2 * expert_weight_bytes];

    // Build offset table
    let mut expert_offsets = vec![0u64; num_experts * 2];
    for e in 0..num_experts {
        let gate_off = e * 2 * expert_weight_bytes;
        let up_off = gate_off + expert_weight_bytes;
        expert_offsets[e * 2] = gate_off as u64;
        expert_offsets[e * 2 + 1] = up_off as u64;

        // Fill gate weights: all rows have constant value
        let gate_data = vec![expert_gate_vals[e]; inter_dim * hidden_dim];
        let gate_encoded = encode_q8_0_matrix(&gate_data, inter_dim, hidden_dim);
        layer_buf_data[gate_off..gate_off + expert_weight_bytes].copy_from_slice(&gate_encoded);

        // Fill up weights: all rows have constant value
        let up_data = vec![expert_up_vals[e]; inter_dim * hidden_dim];
        let up_encoded = encode_q8_0_matrix(&up_data, inter_dim, hidden_dim);
        layer_buf_data[up_off..up_off + expert_weight_bytes].copy_from_slice(&up_encoded);
    }

    // Input x = all ones
    let x = vec![1.0f32; hidden_dim];

    // Expert IDs: select experts 1 and 3
    let expert_ids: Vec<u32> = vec![1, 3];

    // Allocate GPU buffers
    let layer_buf = backend.device.new_buffer_with_bytes(&layer_buf_data).unwrap();
    let x_buf = backend.upload_f32(&x).unwrap();
    let swiglu_out_buf = backend.device.new_buffer(top_k * inter_dim * 4).unwrap();
    let expert_ids_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(expert_ids.as_ptr() as *const u8, expert_ids.len() * 4)
    };
    let expert_ids_buf = backend.device.new_buffer_with_bytes(expert_ids_bytes).unwrap();
    let offsets_bytes: Vec<u8> = expert_offsets.iter().flat_map(|v| v.to_le_bytes()).collect();
    let offsets_buf = backend.device.new_buffer_with_bytes(&offsets_bytes).unwrap();

    // Dispatch
    let cmd = backend.queue.new_command_buffer().unwrap();
    let enc = cmd.new_compute_encoder().unwrap();
    enc.set_pipeline_state(&pso);
    enc.set_buffer(&layer_buf, 0, 0);
    enc.set_buffer(&x_buf, 0, 1);
    enc.set_buffer(&swiglu_out_buf, 0, 2);
    enc.set_buffer(&expert_ids_buf, 0, 3);
    enc.set_buffer(&offsets_buf, 0, 4);
    enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 5);
    enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 6);
    enc.set_bytes(&(top_k as u32).to_le_bytes(), 7);
    let n_tg = (top_k * inter_dim) as u64;
    enc.dispatch_threadgroups(
        MTLSize::new(n_tg, 1, 1),
        MTLSize::new(128, 1, 1),
    );
    enc.end_encoding();
    cmd.commit_and_wait();

    // Read results
    let mut result = vec![0.0f32; top_k * inter_dim];
    swiglu_out_buf.read_f32(&mut result);

    // Compute expected values for each expert slot.
    // Expert 1 (slot 0): gate_val=1.0, up_val=0.5
    //   gate_dot = sum(1.0 * 1.0) * scale_dequant
    //   Due to Q8_0 quantization: 1.0 gets encoded as (scale=1.0/127, q=127),
    //   so dequant = (1.0/127) * 127 = 1.0 per element, sum over 64 = 64.0
    //   up_dot = 64 * 0.5 = 32.0 (approximately, Q8_0 introduces small error)
    //   swiglu = gate_dot * sigmoid(gate_dot) * up_dot
    for slot in 0..top_k {
        let eid = expert_ids[slot] as usize;
        let gate_val = expert_gate_vals[eid];
        let up_val = expert_up_vals[eid];

        // Expected dot products (approximate due to Q8_0 quantization)
        let gate_dot = gate_val * (hidden_dim as f32);
        let up_dot = up_val * (hidden_dim as f32);
        let sigmoid_gate = 1.0 / (1.0 + (-gate_dot).exp());
        let expected = gate_dot * sigmoid_gate * up_dot;

        // Check a few rows
        for row in [0, inter_dim / 2, inter_dim - 1] {
            let actual = result[slot * inter_dim + row];
            let rel_err = if expected.abs() > 1e-6 {
                ((actual - expected) / expected).abs()
            } else {
                (actual - expected).abs()
            };
            assert!(
                rel_err < 0.05,
                "Q8_0 batched gate+up+swiglu: slot={slot}, row={row}, expert={eid}: \
                 actual={actual}, expected={expected}, rel_err={rel_err}"
            );
        }
        eprintln!(
            "Expert {eid} (slot {slot}): gate_dot={gate_dot:.1}, up_dot={up_dot:.1}, \
             expected={expected:.4}, actual[0]={:.4}",
            result[slot * inter_dim]
        );
    }

    eprintln!("test_moe_batched_gate_up_swiglu_q8_0_correctness PASSED");
}

/// Test moe_batched_down_accum_q8_0 kernel correctness.
///
/// Setup: 4 experts with inter_dim=64, hidden_dim=32, top_k=2.
/// Expert IDs: [1, 3], weights: [0.7, 0.3].
/// SwiGLU input: expert k's inter_dim values are all (k+1).0.
/// Down weights: all 1.0 for each expert (so dot product = sum of swiglu input = inter_dim * val).
/// Expected output: residual + sum_k w_k * (inter_dim * swiglu_k_val)
#[test]
fn test_moe_batched_down_accum_q8_0_correctness() {
    let backend = MetalF32Backend::new().unwrap();

    let lib = backend.device
        .new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let func = lib.get_function("moe_batched_down_accum_q8_0").unwrap();
    let pso = backend.device.new_compute_pipeline_state(&func).unwrap();

    let num_experts: usize = 4;
    let hidden_dim: usize = 32;
    let inter_dim: usize = 64;
    let top_k: usize = 2;

    // SwiGLU input: [top_k * inter_dim] float
    // Slot 0 (expert 1): all 2.0
    // Slot 1 (expert 3): all 4.0
    let mut swiglu_in = vec![0.0f32; top_k * inter_dim];
    for i in 0..inter_dim { swiglu_in[i] = 2.0; }          // slot 0
    for i in 0..inter_dim { swiglu_in[inter_dim + i] = 4.0; } // slot 1

    // Down weight matrices: all 1.0 for each expert -> dot = sum(swiglu_k)
    let blocks_per_row = inter_dim / 32;
    let q8_block_size = 34usize;
    let down_weight_bytes = hidden_dim * blocks_per_row * q8_block_size;

    let mut layer_buf_data = vec![0u8; num_experts * down_weight_bytes];
    let mut down_offsets = vec![0u64; num_experts];
    for e in 0..num_experts {
        let off = e * down_weight_bytes;
        down_offsets[e] = off as u64;
        let down_data = vec![1.0f32; hidden_dim * inter_dim];
        let down_encoded = encode_q8_0_matrix(&down_data, hidden_dim, inter_dim);
        layer_buf_data[off..off + down_weight_bytes].copy_from_slice(&down_encoded);
    }

    // Expert IDs: [1, 3], weights: [0.7, 0.3]
    let expert_ids: Vec<u32> = vec![1, 3];
    let expert_weights = vec![0.7f32, 0.3f32];

    // Residual: all 10.0
    let residual = vec![10.0f32; hidden_dim];

    // Allocate GPU buffers
    let layer_buf = backend.device.new_buffer_with_bytes(&layer_buf_data).unwrap();
    let swiglu_buf = backend.upload_f32(&swiglu_in).unwrap();
    let output_buf = backend.device.new_buffer(hidden_dim * 4).unwrap();
    let residual_buf = backend.upload_f32(&residual).unwrap();
    let expert_ids_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(expert_ids.as_ptr() as *const u8, expert_ids.len() * 4)
    };
    let expert_ids_buf = backend.device.new_buffer_with_bytes(expert_ids_bytes).unwrap();
    let expert_weights_buf = backend.upload_f32(&expert_weights).unwrap();
    let offsets_bytes: Vec<u8> = down_offsets.iter().flat_map(|v| v.to_le_bytes()).collect();
    let offsets_buf = backend.device.new_buffer_with_bytes(&offsets_bytes).unwrap();

    // Dispatch
    let cmd = backend.queue.new_command_buffer().unwrap();
    let enc = cmd.new_compute_encoder().unwrap();
    enc.set_pipeline_state(&pso);
    enc.set_buffer(&layer_buf, 0, 0);
    enc.set_buffer(&swiglu_buf, 0, 1);
    enc.set_buffer(&output_buf, 0, 2);
    enc.set_buffer(&residual_buf, 0, 3);
    enc.set_buffer(&expert_ids_buf, 0, 4);
    enc.set_buffer(&expert_weights_buf, 0, 5);
    enc.set_buffer(&offsets_buf, 0, 6);
    enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 7);
    enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 8);
    enc.set_bytes(&(top_k as u32).to_le_bytes(), 9);
    let n_tg = ((hidden_dim as u64) + 3) / 4;
    enc.dispatch_threadgroups(
        MTLSize::new(n_tg, 1, 1),
        MTLSize::new(128, 1, 1),
    );
    enc.end_encoding();
    cmd.commit_and_wait();

    // Read results
    let mut result = vec![0.0f32; hidden_dim];
    output_buf.read_f32(&mut result);

    // Expected:
    // Down weights are all 1.0 -> Q8_0 dequant gives ~1.0 per element (small quant error).
    // dot(down_row, swiglu_slot0) ~ sum(1.0 * 2.0 * 64) = 128.0
    // dot(down_row, swiglu_slot1) ~ sum(1.0 * 4.0 * 64) = 256.0
    // output[d] = residual[d] + 0.7 * 128.0 + 0.3 * 256.0 = 10.0 + 89.6 + 76.8 = 176.4
    let expected_dot_0 = 2.0 * inter_dim as f32; // ~128.0
    let expected_dot_1 = 4.0 * inter_dim as f32; // ~256.0
    let expected = 10.0 + 0.7 * expected_dot_0 + 0.3 * expected_dot_1;

    for d in 0..hidden_dim {
        let actual = result[d];
        let rel_err = ((actual - expected) / expected).abs();
        assert!(
            rel_err < 0.05,
            "Q8_0 batched down+accum: d={d}, actual={actual}, expected={expected}, rel_err={rel_err}"
        );
    }

    eprintln!(
        "test_moe_batched_down_accum_q8_0_correctness PASSED: expected={expected:.1}, actual[0]={:.4}",
        result[0]
    );
}

/// Test sigmoid_scale_add kernel correctness.
///
/// Verifies: dst[i] += sigmoid(scalar[0]) * src[i]
#[test]
fn test_sigmoid_scale_add_correctness() {
    let backend = MetalF32Backend::new().unwrap();

    let lib = backend.device
        .new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let func = lib.get_function("sigmoid_scale_add").unwrap();
    let pso = backend.device.new_compute_pipeline_state(&func).unwrap();

    let dim: usize = 256;

    // scalar = 2.0 => sigmoid(2.0) = 1/(1+exp(-2)) ~ 0.8808
    let scalar = vec![2.0f32];
    let src = vec![3.0f32; dim]; // src = all 3.0
    let dst_init = vec![10.0f32; dim]; // dst starts at 10.0

    let scalar_buf = backend.upload_f32(&scalar).unwrap();
    let src_buf = backend.upload_f32(&src).unwrap();
    let dst_buf = backend.upload_f32(&dst_init).unwrap();

    let cmd = backend.queue.new_command_buffer().unwrap();
    let enc = cmd.new_compute_encoder().unwrap();
    enc.set_pipeline_state(&pso);
    enc.set_buffer(&scalar_buf, 0, 0);
    enc.set_buffer(&src_buf, 0, 1);
    enc.set_buffer(&dst_buf, 0, 2);
    enc.set_bytes(&(dim as u32).to_le_bytes(), 3);
    let tg = 256u64;
    enc.dispatch_threadgroups(
        MTLSize::new((dim as u64).div_ceil(tg), 1, 1),
        MTLSize::new(tg, 1, 1),
    );
    enc.end_encoding();
    cmd.commit_and_wait();

    let mut result = vec![0.0f32; dim];
    dst_buf.read_f32(&mut result);

    // Expected: 10.0 + sigmoid(2.0) * 3.0 = 10.0 + 0.8808 * 3.0 = 12.6424
    let sigmoid_val = 1.0 / (1.0 + (-2.0f32).exp());
    let expected = 10.0 + sigmoid_val * 3.0;

    for d in 0..dim {
        let err = (result[d] - expected).abs();
        assert!(
            err < 1e-4,
            "sigmoid_scale_add: d={d}, actual={}, expected={expected}, err={err}",
            result[d]
        );
    }

    eprintln!(
        "test_sigmoid_scale_add_correctness PASSED: sigmoid(2.0)={sigmoid_val:.6}, expected={expected:.4}, actual[0]={:.4}",
        result[0]
    );
}
