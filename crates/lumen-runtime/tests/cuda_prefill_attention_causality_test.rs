//! A prefill attention kernel must not let a query see a later key. With
//! Q = K = 0 every allowed key gets the same weight, and with V's position
//! `p` a one-hot at dimension `p`, query `q` must come out as `1 / (q + 1)`
//! on dimensions `0..=q` and exactly zero above: any mass above `q` is a
//! key from the future. Requires a CUDA GPU:
//!
//!   cargo test --release -p lumen-runtime --features cuda --test cuda_prefill_attention_causality_test
#![cfg(feature = "cuda")]

use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
use lumen_runtime::cuda::ffi::CudaDevice;
use lumen_runtime::cuda::shaders::FLASH_ATTENTION_KERNEL_SOURCE;

const NUM_HEADS: u32 = 2;
const NUM_KV_HEADS: u32 = 1;
const HEAD_DIM: u32 = 64;
const MAX_SEQ_LEN: u32 = 64;
const FA_BC: u32 = 32;
const FA_BR: u32 = 4;

fn device() -> Option<CudaDevice> {
    match CudaDevice::new(0) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("skipping: no CUDA GPU: {e}");
            None
        }
    }
}

/// Run the scalar kernel over `batch` queries at `pos_start`, returning
/// `[batch, NUM_HEADS * HEAD_DIM]`.
fn run(device: &CudaDevice, kernel: &CudaFunction, batch: u32, pos_start: u32) -> Vec<f32> {
    let q_dim = NUM_HEADS * HEAD_DIM;
    let q = vec![0.0f32; (batch * q_dim) as usize];
    let k = vec![0.0f32; (NUM_KV_HEADS * MAX_SEQ_LEN * HEAD_DIM) as usize];
    let mut v = vec![0.0f32; k.len()];
    for p in 0..MAX_SEQ_LEN.min(HEAD_DIM) {
        v[(p * HEAD_DIM + p) as usize] = 1.0;
    }
    let q_gpu = device.htod_copy(&q).unwrap();
    let k_gpu = device.htod_copy(&k).unwrap();
    let v_gpu = device.htod_copy(&v).unwrap();
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

/// Every query, head and dimension against the closed form: allowed keys
/// within the kernel's own arithmetic, keys from the future exactly zero.
fn check(out: &[f32], batch: u32, pos_start: u32) {
    let q_dim = (NUM_HEADS * HEAD_DIM) as usize;
    let mut worst = 0.0f32;
    for q in 0..batch {
        let last = pos_start + q;
        for h in 0..NUM_HEADS {
            for d in 0..HEAD_DIM {
                let got = out[q as usize * q_dim + (h * HEAD_DIM + d) as usize];
                let want = if d <= last {
                    1.0 / (last + 1) as f32
                } else {
                    0.0
                };
                let err = (got - want).abs();
                assert!(
                    if d <= last { err <= 1e-6 } else { got == 0.0 },
                    "batch {batch} at position {pos_start}: query {q} head {h} \
                     dimension {d}: got {got}, want {want}"
                );
                worst = worst.max(err);
            }
        }
    }
    println!("batch {batch} at position {pos_start}: max error {worst:.2e}");
}

const CASES: [(u32, u32); 6] = [(1, 0), (6, 0), (16, 0), (16, 16), (7, 40), (33, 0)];

#[test]
fn scalar_kernel_is_causal() {
    let Some(device) = device() else { return };
    let kernel = device
        .compile_and_load(FLASH_ATTENTION_KERNEL_SOURCE)
        .unwrap()
        .load_function("flash_attention_causal_br4")
        .unwrap();
    for (batch, pos_start) in CASES {
        check(&run(&device, &kernel, batch, pos_start), batch, pos_start);
    }
}
