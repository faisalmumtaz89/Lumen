//! A planar slice may carry its module's F32 activation scale after its planes, and the CUDA route reads
//! the same weights either way. On the GDN/attention hybrid with NVFP4 MLP and head and FP8 attention and
//! GDN projections, an artifact with no scales, one with every scale and one with half of them pass
//! admission, load, and give bit-identical logits at the prompt's last token and at every greedy decode
//! step after it. A weight byte changed in the first artifact changes those logits, so the comparison can
//! see the weights. Slices one byte long, eight bytes long or four bytes short, in a layer or in the head,
//! are refused at load, naming the tensor, its length and the lengths it may have. Requires a CUDA GPU:
//!
//!   cargo test --release -p lumen-runtime --features cuda --test cuda_planar_input_scale_test
#![cfg(feature = "cuda")]

mod common;

use common::gdn_hybrid::{build_gdn_hybrid_lbc, gdn_model_hyperparams};
use lumen_format::header::LbcHeader;
use lumen_format::index::TensorSlice;
use lumen_format::quantization::{QuantGroupSize, QuantizationDescriptor};
use lumen_format::serving_rules::{unservable_scheme, ServingBackend};
use lumen_format::writer::{write_lbc, GlobalTensors};
use lumen_format::{Fp8Planes, LbcFile, Nvfp4Planes, QuantScheme};
use lumen_runtime::compute::{ActivationBuffer, ComputeBackend, ComputeDtype};
use lumen_runtime::cuda::CudaBackend;
use lumen_runtime::kv::{KvCache, KvCacheConfig, KvPrecision};
use lumen_runtime::weight::provider_sync::SyncWeightProvider;

const PROMPT: usize = 24;
const STEPS: usize = 8;

/// What follows a slice's planes, or how they are cut.
#[derive(Clone, Copy)]
enum Tail {
    None,
    /// The module's activation scale.
    Scale(f32),
    /// Bytes that are no part of the format.
    Extra(usize),
    /// The planes' last bytes dropped.
    Cut(usize),
}

fn lcg(seed: &mut u64) -> u8 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*seed >> 33) as u8
}

/// E4M3 codes other than the two NaN codes.
fn e4m3(seed: &mut u64) -> u8 {
    let c = lcg(seed);
    if c & 0x7F == 0x7F {
        c - 1
    } else {
        c
    }
}

/// One matrix's planes, NVFP4 or FP8, from `seed`, then its tail.
fn planes(quant: QuantScheme, n: usize, k: usize, seed: u64, tail: Tail) -> Vec<u8> {
    let mut seed = seed;
    let mut out = match quant {
        QuantScheme::Nvfp4 => {
            let p = Nvfp4Planes::for_shape(n as u64, k as u64).unwrap();
            let mut out: Vec<u8> = (0..p.weight_bytes).map(|_| lcg(&mut seed)).collect();
            // Block scales 0.5..=1.0 and a small global scale keep every weight well inside the range.
            out.extend((0..p.block_scale_bytes).map(|_| 0x30 + lcg(&mut seed) % 9));
            out.extend_from_slice(&0.02f32.to_le_bytes());
            assert_eq!(out.len() as u64, p.total_bytes());
            out
        }
        QuantScheme::Fp8E4M3 => {
            let p = Fp8Planes::for_shape(n as u64, k as u64).unwrap();
            let mut out: Vec<u8> = (0..p.weight_bytes).map(|_| e4m3(&mut seed)).collect();
            out.extend_from_slice(&1.0e-3f32.to_le_bytes());
            assert_eq!(out.len() as u64, p.total_bytes());
            out
        }
        other => unreachable!("{other:?} is not planar"),
    };
    match tail {
        Tail::None => {}
        Tail::Scale(s) => out.extend_from_slice(&s.to_le_bytes()),
        Tail::Extra(n) => out.resize(out.len() + n, 0x5A),
        Tail::Cut(n) => out.truncate(out.len() - n),
    }
    out
}

/// The hybrid of `common::gdn_hybrid` (layer 0 GDN, layer 1 full attention) with its projections and head
/// replaced by planar planes: NVFP4 MLP and head, FP8 attention and GDN in/out projections. The F32 slices
/// it replaces stay behind as unread bytes. `tail` names what follows each planar slice, keyed
/// `"<layer>.<role>"` or `"head"`; `flip` changes one weight byte of that tensor.
fn planar_hybrid(tail: impl Fn(&str) -> Tail, flip: Option<&str>) -> Vec<u8> {
    let hp = gdn_model_hyperparams();
    let source = build_gdn_hybrid_lbc();
    let lbc = LbcFile::from_bytes(&source, "source.lbc".into()).unwrap();
    let (hidden, inter) = (hp.hidden_dim as usize, hp.intermediate_dim as usize);
    let q_dim = hp.num_heads as usize * hp.head_dim as usize;
    let kv_dim = hp.num_kv_heads as usize * hp.head_dim as usize;
    let gd = hp.gdn_dims();
    let value_dim = gd.num_v_heads as usize * gd.head_dim as usize;
    let qkv_dim = 2 * gd.num_k_heads as usize * gd.head_dim as usize + value_dim;
    let (nvfp4, fp8) = (QuantScheme::Nvfp4, QuantScheme::Fp8E4M3);

    let mut seed = 1u64;
    let mut make = |key: &str, quant: QuantScheme, n: usize, k: usize| {
        seed += 1;
        let mut bytes = planes(quant, n, k, seed, tail(key));
        if flip == Some(key) {
            bytes[3] ^= 0x11;
        }
        bytes
    };

    let mut indices = lbc.layer_indices.clone();
    let mut blobs = Vec::new();
    for (layer, index) in indices.iter_mut().enumerate() {
        let begin = index.layer_offset_bytes as usize;
        let mut blob = source[begin..begin + index.layer_length_bytes as usize].to_vec();
        let st = &mut index.subtensors;
        let mut roles: Vec<(&str, &mut TensorSlice, QuantScheme, usize, usize)> = vec![
            ("w_gate", &mut st.w_gate, nvfp4, inter, hidden),
            ("w_up", &mut st.w_up, nvfp4, inter, hidden),
            ("w_down", &mut st.w_down, nvfp4, hidden, inter),
        ];
        if layer == 0 {
            roles.push(("wq", &mut st.wq, fp8, qkv_dim, hidden));
            roles.push((
                "attn_gate",
                st.attn_gate.as_mut().unwrap(),
                fp8,
                value_dim,
                hidden,
            ));
            roles.push((
                "ssm_out",
                st.ssm_out.as_mut().unwrap(),
                fp8,
                hidden,
                value_dim,
            ));
        } else {
            roles.push(("wq", &mut st.wq, fp8, q_dim, hidden));
            roles.push(("wk", &mut st.wk, fp8, kv_dim, hidden));
            roles.push(("wv", &mut st.wv, fp8, kv_dim, hidden));
            roles.push(("wo", &mut st.wo, fp8, hidden, q_dim));
        }
        for (role, slice, quant, n, k) in roles {
            let bytes = make(&format!("{layer}.{role}"), quant, n, k);
            *slice = TensorSlice {
                offset: blob.len() as u64,
                length: bytes.len() as u64,
                quant,
            };
            blob.extend_from_slice(&bytes);
        }
        index.layer_length_bytes = blob.len() as u64;
        blobs.push(blob);
    }

    let mut header = LbcHeader::new(
        hp,
        QuantizationDescriptor {
            scheme: nvfp4,
            group_size: QuantGroupSize::Group(16),
            block_byte_size: 0,
            scale_offset_in_block: None,
        },
    );
    header.output_proj.quant = nvfp4;
    let at = |s: &lumen_format::header::GlobalTensorRange| {
        source[s.offset as usize..(s.offset + s.length) as usize].to_vec()
    };
    let globals = GlobalTensors {
        embedding: at(&lbc.header.embedding),
        final_norm: at(&lbc.header.final_norm),
        output_proj: make("head", nvfp4, hp.vocab_size as usize, hidden),
    };
    let blob_refs: Vec<&[u8]> = blobs.iter().map(Vec::as_slice).collect();
    let mut out = Vec::new();
    write_lbc(&mut out, &header, &indices, &globals, &blob_refs, None).unwrap();
    out
}

/// Admission, then the weight provider and the CUDA backend, as the binaries build them. `Err` carries the
/// first refusal.
fn load(lbc: &[u8], dir: &std::path::Path) -> Result<(SyncWeightProvider, CudaBackend), String> {
    let path = dir.join("model.lbc");
    std::fs::write(&path, lbc).unwrap();
    let parsed = LbcFile::open(&path).map_err(|e| e.to_string())?;
    assert_eq!(
        unservable_scheme(&parsed, ServingBackend::Cuda),
        None,
        "admission refused the artifact"
    );
    let provider = SyncWeightProvider::open(&path).map_err(|e| e.to_string())?;
    let mut cuda = CudaBackend::new(0).expect("this test needs a CUDA GPU");
    cuda.set_global_tensors(
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
    );
    cuda.set_output_proj_raw(provider.output_proj_raw.clone(), provider.output_proj_quant);
    cuda.init(&provider.lbc().header.hyperparams)
        .map_err(|e| e.to_string())?;
    cuda.preload_weights(&provider).map_err(|e| e.to_string())?;
    Ok((provider, cuda))
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

/// The logits' bits at the prompt's last token and at each greedy decode step after it.
fn serve(provider: &SyncWeightProvider, cuda: &CudaBackend) -> Vec<Vec<u32>> {
    let hp = provider.lbc().header.hyperparams;
    let mut kv = KvCache::new(KvCacheConfig {
        max_seq_len: hp.max_seq_len as usize,
        num_layers: hp.num_layers as usize,
        num_kv_heads: hp.num_kv_heads as usize,
        head_dim: hp.head_dim as usize,
        precision: KvPrecision::F32,
    })
    .unwrap();
    let prompt: Vec<u32> = (0..PROMPT)
        .map(|i| ((i * 7919 + 3) % hp.vocab_size as usize) as u32)
        .collect();
    cuda.reset_recurrent_state();
    let x = cuda.prefill(&prompt, provider, &mut kv).expect("prefill");
    let mut buf = ActivationBuffer::zeros(x.len(), ComputeDtype::F32);
    buf.write_f32_from(&x);
    let mut logits = cuda.compute_final(&buf).expect("compute_final").data;
    let mut steps = Vec::new();
    for step in 0..=STEPS {
        assert!(
            logits.iter().all(|v| v.is_finite()) && logits.iter().any(|&v| v != logits[0]),
            "step {step}: degenerate logits"
        );
        steps.push(logits.iter().map(|v| v.to_bits()).collect());
        if step < STEPS {
            logits = cuda
                .decode_token(argmax(&logits), provider, &mut kv)
                .expect("decode")
                .data;
        }
    }
    steps
}

fn served(lbc: &[u8]) -> Vec<Vec<u32>> {
    let dir = tempfile::tempdir().unwrap();
    let (provider, cuda) = load(lbc, dir.path()).unwrap_or_else(|e| panic!("refused: {e}"));
    serve(&provider, &cuda)
}

/// Every planar tensor of the hybrid, by key.
const KEYS: [&str; 14] = [
    "0.w_gate",
    "0.w_up",
    "0.w_down",
    "0.wq",
    "0.attn_gate",
    "0.ssm_out",
    "1.w_gate",
    "1.w_up",
    "1.w_down",
    "1.wq",
    "1.wk",
    "1.wv",
    "1.wo",
    "head",
];

fn scale_of(key: &str) -> f32 {
    0.01 + KEYS.iter().position(|k| *k == key).unwrap() as f32 * 1e-3
}

#[test]
fn artifacts_with_no_all_or_some_input_scales_serve_identically() {
    let old = served(&planar_hybrid(|_| Tail::None, None));
    let extended = served(&planar_hybrid(|k| Tail::Scale(scale_of(k)), None));
    let mixed = served(&planar_hybrid(
        |k| {
            // Each role present in both layers is scaled in one and not in the other, and each fused
            // gate/up pair holds one of each.
            let i = KEYS.iter().position(|x| *x == k).unwrap();
            if (i + i / 6) % 2 == 0 {
                Tail::Scale(scale_of(k))
            } else {
                Tail::None
            }
        },
        None,
    ));
    assert_eq!(old.len(), STEPS + 1);
    for (name, other) in [("extended", &extended), ("mixed", &mixed)] {
        for (step, (a, b)) in old.iter().zip(other).enumerate() {
            assert!(
                a == b,
                "{name}: logits differ from the old artifact's at step {step}"
            );
        }
    }
    // The control: one weight byte of one tensor changed, and the logits see it.
    for key in ["1.w_down", "head"] {
        let flipped = served(&planar_hybrid(|_| Tail::None, Some(key)));
        assert_ne!(flipped, old, "{key}: a changed weight went unseen");
    }
}

#[test]
fn malformed_planar_lengths_are_refused_by_name_at_load() {
    let hp = gdn_model_hyperparams();
    for (key, role) in [
        ("0.wq", "wq"),
        ("0.ssm_out", "ssm_out"),
        ("1.wo", "wo"),
        ("1.w_down", "w_down"),
        ("head", "head"),
    ] {
        for bad in [Tail::Extra(1), Tail::Extra(8), Tail::Cut(4)] {
            let lbc = planar_hybrid(|k| if k == key { bad } else { Tail::None }, None);
            let dir = tempfile::tempdir().unwrap();
            let err = match load(&lbc, dir.path()) {
                Ok(_) => panic!("{key}: a malformed length was loaded"),
                Err(e) => e,
            };
            // The good length and the malformed one this case wrote, from the file itself.
            let parsed = LbcFile::from_bytes(&lbc, "bad.lbc".into()).unwrap();
            let got = if key == "head" {
                parsed.header.output_proj.length
            } else {
                let layer: usize = key[..1].parse().unwrap();
                let st = &parsed.layer_indices[layer].subtensors;
                match role {
                    "wq" => st.wq.length,
                    "ssm_out" => st.ssm_out.unwrap().length,
                    "wo" => st.wo.length,
                    _ => st.w_down.length,
                }
            };
            let planes = match role {
                "head" => Nvfp4Planes::for_shape(hp.vocab_size as u64, hp.hidden_dim as u64)
                    .unwrap()
                    .total_bytes(),
                "w_down" => {
                    Nvfp4Planes::for_shape(hp.hidden_dim as u64, hp.intermediate_dim as u64)
                        .unwrap()
                        .total_bytes()
                }
                _ => {
                    let (n, k) = match role {
                        "wq" => (1024u64, 64u64),
                        "ssm_out" => (64, 512),
                        _ => (64, 512),
                    };
                    Fp8Planes::for_shape(n, k).unwrap().total_bytes()
                }
            };
            let named = if role == "head" {
                err.contains(&format!("output head plane is {got} bytes"))
                    && err.contains(&format!("exactly {planes}, or {}", planes + 4))
            } else {
                err.contains(&format!("layer {}: {role} is {got} bytes", &key[..1]))
                    && err.contains(&format!(
                        "[{planes}] bytes, each optionally followed by a 4-byte input scale"
                    ))
            };
            assert!(
                named,
                "{key} ({got} bytes): not refused by name and lengths: {err}"
            );
        }
    }
}
