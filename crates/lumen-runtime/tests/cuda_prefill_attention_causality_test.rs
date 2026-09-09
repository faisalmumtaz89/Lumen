//! The scalar prefill attention kernel, `flash_attention_causal_br4`, against
//! closed forms. Causality: with Q = K = 0 every allowed key gets the same
//! weight, and with V's position `p` a one-hot at dimension `p` scaled by its
//! KV head, query `q` must come out as `(kv + 1) / (q + 1)` on dimensions
//! `0..=q` and exactly zero above: any mass above `q` is a key from the
//! future. Reference: random inputs spanning several key tiles against a
//! float64 softmax attention. Requires a CUDA GPU:
//!
//!   cargo test --release -p lumen-runtime --features cuda --test cuda_prefill_attention_causality_test
#![cfg(feature = "cuda")]

use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
use lumen_runtime::cuda::ffi::CudaDevice;
use lumen_runtime::cuda::shaders::FLASH_ATTENTION_KERNEL_SOURCE;

const NUM_HEADS: u32 = 4;
const NUM_KV_HEADS: u32 = 2;
const HEAD_DIM: u32 = 128;
const MAX_SEQ_LEN: u32 = 96;
const FA_BC: u32 = 32;
const FA_BR: u32 = 4;

/// (batch, pos_start): tails of one, two and three query rows in a block,
/// whole blocks, a nonzero start, and a span across key tiles.
const CASES: [(u32, u32); 7] = [
    (1, 0),
    (6, 0),
    (16, 0),
    (16, 16),
    (7, 40),
    (33, 0),
    (24, 70),
];

const _: () = {
    assert!(
        MAX_SEQ_LEN <= HEAD_DIM,
        "every key has its own marker dimension"
    );
    let mut i = 0;
    while i < CASES.len() {
        assert!(
            CASES[i].0 + CASES[i].1 <= MAX_SEQ_LEN,
            "a case reads past the KV cache"
        );
        i += 1;
    }
};

fn device() -> Option<CudaDevice> {
    match CudaDevice::new(0) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("skipping: no CUDA GPU: {e}");
            None
        }
    }
}

fn kernel(device: &CudaDevice) -> CudaFunction {
    device
        .compile_and_load(FLASH_ATTENTION_KERNEL_SOURCE)
        .unwrap()
        .load_function("flash_attention_causal_br4")
        .unwrap()
}

/// Run the kernel over `batch` queries at `pos_start`, returning
/// `[batch, NUM_HEADS * HEAD_DIM]`. `k` and `v` are
/// `[NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM]`.
fn run(
    device: &CudaDevice,
    kernel: &CudaFunction,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: u32,
    pos_start: u32,
) -> Vec<f32> {
    assert_eq!(q.len(), (batch * NUM_HEADS * HEAD_DIM) as usize);
    assert_eq!(k.len(), (NUM_KV_HEADS * MAX_SEQ_LEN * HEAD_DIM) as usize);
    assert_eq!(v.len(), k.len());
    let q_gpu = device.htod_copy(q).unwrap();
    let k_gpu = device.htod_copy(k).unwrap();
    let v_gpu = device.htod_copy(v).unwrap();
    let mut out = device.alloc_zeros::<f32>(q.len()).unwrap();
    let cfg = LaunchConfig {
        grid_dim: (NUM_HEADS, batch.div_ceil(FA_BR), 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: FA_BR * (HEAD_DIM + FA_BC) * 4,
    };
    let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
    unsafe {
        device
            .stream
            .launch_builder(kernel)
            .arg(&q_gpu)
            .arg(&k_gpu)
            .arg(&v_gpu)
            .arg(&mut out)
            .arg(&batch)
            .arg(&NUM_HEADS)
            .arg(&NUM_KV_HEADS)
            .arg(&HEAD_DIM)
            .arg(&pos_start)
            .arg(&MAX_SEQ_LEN)
            .arg(&scale)
            .launch(cfg)
    }
    .unwrap();
    device.synchronize().unwrap();
    device.dtoh_copy(&out).unwrap()
}

#[test]
fn scalar_kernel_is_causal() {
    let Some(device) = device() else { return };
    let kernel = kernel(&device);
    let kv_len = (MAX_SEQ_LEN * HEAD_DIM) as usize;
    let k = vec![0.0f32; NUM_KV_HEADS as usize * kv_len];
    let mut v = vec![0.0f32; k.len()];
    for kv in 0..NUM_KV_HEADS {
        for p in 0..MAX_SEQ_LEN {
            v[kv as usize * kv_len + (p * HEAD_DIM + p) as usize] = (kv + 1) as f32;
        }
    }
    let q_dim = (NUM_HEADS * HEAD_DIM) as usize;
    for (batch, pos_start) in CASES {
        let q = vec![0.0f32; batch as usize * q_dim];
        let out = run(&device, &kernel, &q, &k, &v, batch, pos_start);
        let mut worst = 0.0f32;
        for row in 0..batch {
            let last = pos_start + row;
            for h in 0..NUM_HEADS {
                let kv = h / (NUM_HEADS / NUM_KV_HEADS);
                for d in 0..HEAD_DIM {
                    let got = out[row as usize * q_dim + (h * HEAD_DIM + d) as usize];
                    let want = if d <= last {
                        (kv + 1) as f32 / (last + 1) as f32
                    } else {
                        0.0
                    };
                    let err = (got - want).abs();
                    assert!(
                        if d <= last { err <= 1e-6 } else { got == 0.0 },
                        "batch {batch} at position {pos_start}: query {row} head {h} \
                         dimension {d}: got {got}, want {want}"
                    );
                    worst = worst.max(err);
                }
            }
        }
        println!("causal: batch {batch} at position {pos_start}: max error {worst:.2e}");
    }
}

#[test]
fn scalar_kernel_matches_reference_across_key_tiles() {
    let Some(device) = device() else { return };
    let kernel = kernel(&device);
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    let kv_len = (MAX_SEQ_LEN * HEAD_DIM) as usize;
    let k: Vec<f32> = (0..NUM_KV_HEADS as usize * kv_len)
        .map(|_| next())
        .collect();
    let v: Vec<f32> = (0..k.len()).map(|_| next()).collect();
    let q_dim = (NUM_HEADS * HEAD_DIM) as usize;
    let hd = HEAD_DIM as usize;
    let scale = 1.0 / (HEAD_DIM as f64).sqrt();
    for (batch, pos_start) in CASES {
        let q: Vec<f32> = (0..batch as usize * q_dim).map(|_| next()).collect();
        let out = run(&device, &kernel, &q, &k, &v, batch, pos_start);
        let mut worst = 0.0f64;
        for row in 0..batch as usize {
            let seq_len = pos_start as usize + row + 1;
            for h in 0..NUM_HEADS as usize {
                let kv = h / (NUM_HEADS / NUM_KV_HEADS) as usize;
                let q_head = &q[row * q_dim + h * hd..][..hd];
                let key = |t: usize| &k[kv * kv_len + t * hd..][..hd];
                let val = |t: usize| &v[kv * kv_len + t * hd..][..hd];
                let scores: Vec<f64> = (0..seq_len)
                    .map(|t| {
                        q_head
                            .iter()
                            .zip(key(t))
                            .map(|(&a, &b)| a as f64 * b as f64)
                            .sum::<f64>()
                            * scale
                    })
                    .collect();
                let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let total: f64 = weights.iter().sum();
                for d in 0..hd {
                    let want = (0..seq_len)
                        .map(|t| weights[t] * val(t)[d] as f64)
                        .sum::<f64>()
                        / total;
                    let got = out[row * q_dim + h * hd + d] as f64;
                    let err = (got - want).abs();
                    assert!(
                        err <= 1e-5,
                        "batch {batch} at position {pos_start}: query {row} head {h} \
                         dimension {d}: got {got}, want {want}"
                    );
                    worst = worst.max(err);
                }
            }
        }
        println!("reference: batch {batch} at position {pos_start}: max error {worst:.2e}");
    }
}
