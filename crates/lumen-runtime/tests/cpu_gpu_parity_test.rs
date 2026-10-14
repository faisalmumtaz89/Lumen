//! The SIMD CPU backend against a GPU backend on a real Qwen3.5 LBC (`#[ignore]`).
//!
//! ```text
//! LUMEN_CPU_PARITY_LBC=/path/to/qwen3-5-9b-Q8_0.lbc cargo test --release -p lumen-runtime \
//!   --features cuda --test cpu_gpu_parity_test -- --ignored --nocapture
//! LUMEN_CPU_PARITY_LBC=$HOME/Library/Caches/lumen/qwen3-5-9b-Q8_0.lbc \
//!   cargo test --release -p lumen-runtime --test cpu_gpu_parity_test -- --ignored --nocapture
//! ```
//!
//! - CUDA (a build with the `cuda` feature): the GPU (weights resident) decodes `STEPS` greedy
//!   tokens; the CPU backend reads the same prompt and tokens and gives its logits at every one
//!   of those positions. Its first token must be the GPU's, its logits within cosine 0.999 of
//!   the GPU's at every position, and its argmax the GPU's at 98% of them or more. Metal is not
//!   held to this: it re-quantizes a Q8_0 model's output head and GDN `ssm_out` to Q4_0 at load.
//! - Metal: a session saved right after prefill, loaded into the CPU backend and saved again
//!   carries Metal's recurrent state byte for byte; the CPU's own prefill of the prompt saves
//!   the same conv ring positions, and h and conv states within 0.05 relative L2 of Metal's in
//!   every GDN layer (the same ring read one slot off is about 0.7 away).
//!
//! The artifact's SHA-256, header scheme and the backend settings are printed with the results.

#![cfg(any(target_os = "macos", feature = "cuda"))]

use lumen_runtime::compute::cpu_simd::SimdF32Backend;
use lumen_runtime::storage::MmapConfig;
use lumen_runtime::weight::provider_mmap::MmapWeightProvider;
use lumen_runtime::ComputeBackend;
use std::path::{Path, PathBuf};

/// The 9B's chat template around "Explain in one paragraph why the sky is blue.", thinking off.
const PROMPT: &[u32] = &[
    248045, 846, 198, 814, 20139, 303, 799, 13901, 3069, 279, 12515, 369, 6105, 13, 248046, 198,
    248045, 74455, 198, 248068, 271, 248069, 271,
];
const CONTEXT: usize = 512;

fn lbc() -> PathBuf {
    PathBuf::from(
        std::env::var("LUMEN_CPU_PARITY_LBC").expect("set LUMEN_CPU_PARITY_LBC to an LBC"),
    )
}

fn describe(path: &Path, provider: &MmapWeightProvider) {
    let sha = std::process::Command::new("/usr/bin/shasum")
        .args(["-a", "256"])
        .arg(path)
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_default();
    let h = &provider.lbc().header;
    eprintln!(
        "artifact {sha}\n  scheme {:?}, output_proj {:?}, embedding {:?}, {} layers",
        h.quantization.scheme,
        provider.output_proj_quant,
        provider.embedding_quant,
        h.hyperparams.num_layers
    );
}

fn cpu(provider: &MmapWeightProvider, max_seq_len: usize) -> SimdF32Backend {
    let mut c = SimdF32Backend::new();
    c.set_global_tensors(
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
    );
    let mut hp = provider.lbc().header.hyperparams;
    hp.max_seq_len = max_seq_len as u32;
    c.init(&hp).unwrap();
    c.preload_weights(provider).unwrap();
    eprintln!("  CPU: SIMD, {} threads", c.thread_count());
    c
}

#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
mod cuda {
    use super::*;
    use lumen_format::QuantScheme;
    use lumen_runtime::kv::{KvCache, KvCacheConfig, KvPrecision};
    use lumen_runtime::WeightProvider;

    const STEPS: usize = 128;

    /// CUDA device 0 with the weights resident, ready to decode, and its KV cache.
    fn gpu(provider: &MmapWeightProvider) -> (Box<dyn ComputeBackend>, KvCache) {
        let mut c = lumen_runtime::CudaBackend::new(0).expect("CUDA device 0");
        c.set_kv_precision(KvPrecision::F32).unwrap();
        c.set_global_tensors(
            provider.embedding.clone(),
            provider.final_norm.clone(),
            provider.output_proj.clone(),
        );
        if [
            QuantScheme::Q8_0,
            QuantScheme::Q4_0,
            QuantScheme::F16,
            QuantScheme::Bf16,
            QuantScheme::Q6_K,
        ]
        .contains(&provider.output_proj_quant)
            && !provider.output_proj_raw.is_empty()
        {
            c.set_output_proj_raw(provider.output_proj_raw.clone(), provider.output_proj_quant);
        }
        if [
            QuantScheme::Q8_0,
            QuantScheme::Q4_0,
            QuantScheme::F16,
            QuantScheme::Bf16,
            QuantScheme::Q4_K,
            QuantScheme::Q5_K,
            QuantScheme::Q6_K,
        ]
        .contains(&provider.embedding_quant)
            && !provider.embedding_raw.is_empty()
        {
            c.set_embedding_raw(provider.embedding_raw.clone(), provider.embedding_quant);
        }
        if provider.weight_tying {
            c.set_weight_tying(true);
        }
        let mut hp = provider.lbc().header.hyperparams;
        hp.max_seq_len = CONTEXT as u32;
        c.init(&hp).unwrap();
        c.preload_weights(provider).unwrap();
        eprintln!("  GPU: CUDA device 0, weights resident, F32 KV");
        let hp = provider.lbc().header.hyperparams;
        let kv = KvCache::new_without_host_mirror(KvCacheConfig {
            max_seq_len: CONTEXT,
            num_layers: hp.num_layers as usize,
            num_kv_heads: hp.num_kv_heads as usize,
            head_dim: hp.head_dim as usize,
            precision: KvPrecision::F32,
        })
        .unwrap();
        (Box::new(c), kv)
    }

    /// The CPU backend's F32 KV cache.
    fn cpu_kv(provider: &MmapWeightProvider) -> KvCache {
        let hp = provider.lbc().header.hyperparams;
        KvCache::new(KvCacheConfig {
            max_seq_len: CONTEXT,
            num_layers: hp.num_layers as usize,
            num_kv_heads: hp.num_kv_heads as usize,
            head_dim: hp.head_dim as usize,
            precision: KvPrecision::F32,
        })
        .unwrap()
    }

    fn argmax(v: &[f32]) -> u32 {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap()
    }

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
        let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        dot / (na * nb)
    }

    /// One token through every layer on the CPU backend; the logits.
    fn step(
        backend: &dyn ComputeBackend,
        w: &dyn WeightProvider,
        kv: &mut KvCache,
        token: u32,
    ) -> Vec<f32> {
        let mut x = backend.embed_token(token).unwrap();
        let pos = kv.seq_len();
        for layer in 0..w.num_layers() {
            let view = w.get_layer_raw(layer).unwrap();
            let mut v = kv.view_mut(layer).unwrap();
            backend
                .compute_layer(layer, &mut x, &view, Some(&mut v), pos)
                .unwrap();
            kv.commit_view(v).unwrap();
        }
        kv.advance_seq_len().unwrap();
        backend.compute_final(&x).unwrap().data
    }

    #[test]
    #[ignore = "needs a GPU and a real Qwen3.5 LBC in LUMEN_CPU_PARITY_LBC"]
    fn cpu_logits_track_the_gpu_over_one_history() {
        let path = lbc();
        let provider = MmapWeightProvider::open(&path, MmapConfig::default()).unwrap();
        describe(&path, &provider);

        let (m, mut mkv) = gpu(&provider);
        m.reset_recurrent_state();
        let hidden = m.prefill(PROMPT, &provider, &mut mkv).unwrap();
        let mut x = lumen_runtime::compute::ActivationBuffer::zeros(
            hidden.len(),
            lumen_runtime::compute::ComputeDtype::F32,
        );
        x.write_f32_from(&hidden);
        let mut logits = m.compute_final(&x).unwrap().data;
        let (mut tokens, mut gpu_logits) = (Vec::new(), Vec::new());
        for _ in 0..STEPS {
            let t = argmax(&logits);
            tokens.push(t);
            gpu_logits.push(logits);
            logits = m.decode_token(t, &provider, &mut mkv).unwrap().data;
        }
        drop(m);

        let c = cpu(&provider, CONTEXT);
        let mut ckv = cpu_kv(&provider);
        let mut cpu_logits = Vec::new();
        let mut last = Vec::new();
        for &t in PROMPT {
            last = step(&c, &provider, &mut ckv, t);
        }
        for &t in &tokens {
            cpu_logits.push(std::mem::take(&mut last));
            last = step(&c, &provider, &mut ckv, t);
        }

        let cos: Vec<f64> = cpu_logits
            .iter()
            .zip(&gpu_logits)
            .map(|(a, b)| cosine(a, b))
            .collect();
        let agree = cpu_logits
            .iter()
            .zip(&tokens)
            .filter(|(l, &t)| argmax(l) == t)
            .count();
        let min_cos = cos.iter().cloned().fold(f64::MAX, f64::min);
        let mean_cos = cos.iter().sum::<f64>() / cos.len() as f64;
        eprintln!(
            "  {STEPS} positions: top-1 agreement {agree}/{STEPS}, cosine min {min_cos:.6} mean \
             {mean_cos:.6}, first mismatch {:?}",
            cpu_logits
                .iter()
                .zip(&tokens)
                .position(|(l, &t)| argmax(l) != t)
        );
        assert_eq!(argmax(&cpu_logits[0]), tokens[0], "first token");
        assert!(min_cos >= 0.999, "cosine {min_cos:.6}");
        assert!(agree * 100 >= STEPS * 98, "agreement {agree}/{STEPS}");
    }
}

#[cfg(target_os = "macos")]
mod metal {
    use super::*;
    use lumen_format::QuantScheme;
    use lumen_runtime::kv::disk::{
        load_into, serialize_hyperparams_le, ModelFingerprint, RecurrentState,
    };
    use lumen_runtime::kv::KvPrecision;
    use lumen_runtime::{RuntimeConfig, SamplingParams, Session};

    fn metal(provider: &MmapWeightProvider, max_seq_len: usize) -> lumen_runtime::MetalF32Backend {
        if std::env::var_os("LUMEN_METAL_MMAP_ONLY").is_none() {
            std::env::set_var("LUMEN_METAL_MMAP_ONLY", "1");
        }
        let mut m = lumen_runtime::MetalF32Backend::new().expect("Metal");
        m.set_global_tensors(
            provider.embedding.clone(),
            provider.final_norm.clone(),
            provider.output_proj.clone(),
        );
        let raw = [
            QuantScheme::Q8_0,
            QuantScheme::Q4_0,
            QuantScheme::F16,
            QuantScheme::Bf16,
        ];
        if raw.contains(&provider.output_proj_quant) && !provider.output_proj_raw.is_empty() {
            m.set_output_proj_raw(provider.output_proj_raw.clone(), provider.output_proj_quant);
        }
        if raw[..3].contains(&provider.embedding_quant) && !provider.embedding_raw.is_empty() {
            m.set_embedding_raw(provider.embedding_raw.clone(), provider.embedding_quant);
        }
        if provider.weight_tying {
            m.set_weight_tying(true);
        }
        let mut hp = provider.lbc().header.hyperparams;
        hp.max_seq_len = max_seq_len as u32;
        m.init(&hp).unwrap();
        m.preload_weights(provider).unwrap();
        m
    }

    fn greedy() -> SamplingParams {
        SamplingParams {
            temperature: 0.0,
            repetition_penalty: Some(1.0),
            seed: Some(1),
            ..Default::default()
        }
    }

    fn config() -> RuntimeConfig {
        RuntimeConfig {
            pipeline_mode: lumen_runtime::pipeline::PipelineMode::MinMem,
            prefetch_distance: 1,
            kv_precision: KvPrecision::F16,
            max_seq_len: CONTEXT,
            collect_per_layer_timings: false,
        }
    }

    /// The recurrent section of the session file at `path`.
    fn recurrent(path: &Path) -> RecurrentState {
        load_into(path, None, None, None, None)
            .unwrap()
            .recurrent
            .expect("a GDN model saves its recurrent state")
    }

    /// Relative L2 distance of the f32 buffer `a` from `b`.
    fn rel_l2(a: &[u8], b: &[u8]) -> f64 {
        let f = |v: &[u8]| -> Vec<f64> {
            v.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
                .collect()
        };
        let (a, b) = (f(a), f(b));
        let diff: f64 = a.iter().zip(&b).map(|(x, y)| (x - y).powi(2)).sum();
        let norm: f64 = b.iter().map(|y| y.powi(2)).sum();
        (diff / norm).sqrt()
    }

    #[test]
    #[ignore = "needs Apple Silicon and a real Qwen3.5 LBC in LUMEN_CPU_PARITY_LBC"]
    fn the_cpu_restores_and_saves_metals_recurrent_state() {
        let path = lbc();
        let provider = MmapWeightProvider::open(&path, MmapConfig::default()).unwrap();
        describe(&path, &provider);
        let mut hp = provider.lbc().header.hyperparams;
        hp.max_seq_len = CONTEXT as u32;
        let fp = ModelFingerprint::from_live_model(
            &serialize_hyperparams_le(&provider.lbc().header.hyperparams),
            &[],
            provider.output_proj_quant.to_u8() as u32,
            lumen_format::LBC_VERSION,
        );
        let dir = tempfile::tempdir().unwrap();
        let (metal_file, resaved_file, cpu_file) = (
            dir.path().join("metal.kv"),
            dir.path().join("resaved.kv"),
            dir.path().join("cpu.kv"),
        );

        let m = metal(&provider, CONTEXT);
        let mut s = Session::new(config(), hp, greedy()).unwrap();
        s.extend(PROMPT, &m, &provider).unwrap();
        s.save_to_disk(&metal_file, &m, &provider, &fp).unwrap();
        drop(s);
        drop(m);

        let c = cpu(&provider, CONTEXT);
        let mut restored =
            Session::load_from_disk(&metal_file, config(), hp, greedy(), &c, &fp).unwrap();
        restored
            .save_to_disk(&resaved_file, &c, &provider, &fp)
            .unwrap();
        let mut cold = Session::new(config(), hp, greedy()).unwrap();
        cold.extend(PROMPT, &c, &provider).unwrap();
        cold.save_to_disk(&cpu_file, &c, &provider, &fp).unwrap();

        let (from_metal, resaved, from_cpu) = (
            recurrent(&metal_file),
            recurrent(&resaved_file),
            recurrent(&cpu_file),
        );
        assert_eq!(resaved.conv_positions, from_metal.conv_positions);
        assert!(
            resaved.h_states == from_metal.h_states,
            "h changed through the CPU"
        );
        assert!(
            resaved.conv_states == from_metal.conv_states,
            "conv changed through the CPU"
        );
        assert_eq!(from_cpu.conv_positions, from_metal.conv_positions);
        let (mut worst_h, mut worst_conv) = (0f64, 0f64);
        for i in 0..from_metal.h_states.len() {
            let h = rel_l2(&from_cpu.h_states[i], &from_metal.h_states[i]);
            let conv = rel_l2(&from_cpu.conv_states[i], &from_metal.conv_states[i]);
            assert!(
                h <= 0.05 && conv <= 0.05,
                "GDN layer {i}: h {h:.3e}, conv {conv:.3e}"
            );
            worst_h = worst_h.max(h);
            worst_conv = worst_conv.max(conv);
        }
        eprintln!(
            "  CPU prefill vs Metal: worst relative L2 h {worst_h:.3e}, conv {worst_conv:.3e}"
        );
    }
}
