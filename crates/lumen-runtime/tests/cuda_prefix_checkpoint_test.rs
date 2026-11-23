//! CUDA on a real Qwen3.x model (`#[ignore]`): the prefix checkpoint puts a sequence back to
//! the end of its leading block. A block is prefilled and kept, a rest prefilled and decoded
//! from it; the backend then serves another rest of the same block and decodes; the restore
//! for the first rest must leave every GDN layer's recurrent state, conv ring and ring
//! position and every attention layer's rows of the block as they were when the block was
//! kept (compared byte for byte on an F32 KV store, where the comparison before the restore
//! must differ), and the rest's logits and 64 greedy tokens must be the first run's bit for
//! bit. Nothing but a longer prompt that starts with the kept tokens restores, and a refused
//! rest leaves the checkpoint usable. The split prefill is compared with one call prefilling
//! the whole prompt: its logits lie within 0.25 (relative L2) of the one call's, the bound
//! `cuda_native_prefill_test` holds two prefill routes to, and on an F32 KV store its recurrent
//! states within 0.05, the bound `cuda_warm_append_test` holds a continued prefill to. Each
//! one's winner and runner-up and the time each takes are printed. Requires a CUDA GPU:
//!
//!   LUMEN_PREFIX_CHECKPOINT_MODEL=/path/to/model.lbc [LUMEN_PREFIX_CHECKPOINT_KV=f32|f16|bf16] \
//!     cargo test --release -p lumen-runtime --features test-state-snapshot \
//!     --test cuda_prefix_checkpoint_test -- --ignored --nocapture
#![cfg(feature = "test-state-snapshot")]

use lumen_format::quantization::QuantScheme;
use lumen_runtime::compute::{ActivationBuffer, ComputeBackend, ComputeDtype};
use lumen_runtime::cuda::{CudaBackend, StateSnapshot};
use lumen_runtime::kv::{KvCache, KvCacheConfig, KvPrecision};
use lumen_runtime::weight::provider_sync::SyncWeightProvider;
use std::path::PathBuf;
use std::time::Instant;

const CONTEXT: usize = 8192;
const GREEDY_STEPS: usize = 64;

fn precision() -> KvPrecision {
    match std::env::var("LUMEN_PREFIX_CHECKPOINT_KV").as_deref() {
        Ok("f16") => KvPrecision::F16,
        Ok("bf16") => KvPrecision::Bf16,
        _ => KvPrecision::F32,
    }
}

/// The backend as `lumen-server` builds it for CUDA, with the context capped at `CONTEXT`.
fn backend(provider: &SyncWeightProvider) -> CudaBackend {
    let mut hp = provider.lbc().header.hyperparams;
    hp.max_seq_len = hp.max_seq_len.min(CONTEXT as u32);
    let mut cuda = CudaBackend::new(0).expect("CUDA device 0");
    cuda.set_kv_precision(precision()).expect("KV precision");
    cuda.set_global_tensors(
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
    );
    let embedding_raw = matches!(
        provider.embedding_quant,
        QuantScheme::Q8_0
            | QuantScheme::Q4_0
            | QuantScheme::F16
            | QuantScheme::Bf16
            | QuantScheme::Q4_K
            | QuantScheme::Q5_K
            | QuantScheme::Q6_K
    ) && !provider.embedding_raw.is_empty();
    if embedding_raw {
        cuda.set_embedding_raw(provider.embedding_raw.clone(), provider.embedding_quant);
    }
    let head_raw = matches!(
        provider.output_proj_quant,
        QuantScheme::Q8_0
            | QuantScheme::Q4_0
            | QuantScheme::F16
            | QuantScheme::Bf16
            | QuantScheme::Q6_K
            | QuantScheme::Nvfp4
    ) && !provider.output_proj_raw.is_empty();
    if head_raw {
        cuda.set_output_proj_raw(provider.output_proj_raw.clone(), provider.output_proj_quant);
    }
    if provider.weight_tying {
        cuda.set_weight_tying(true);
    }
    cuda.init(&hp).expect("init");
    cuda.preload_weights(provider).expect("preload");
    cuda
}

fn kv_cache(provider: &SyncWeightProvider) -> KvCache {
    let hp = provider.lbc().header.hyperparams;
    KvCache::new(KvCacheConfig {
        max_seq_len: CONTEXT,
        num_layers: hp.num_layers as usize,
        num_kv_heads: hp.num_kv_heads as usize,
        head_dim: hp.head_dim as usize,
        precision: precision(),
    })
    .unwrap()
}

/// Token ids of an ordinary vocabulary range; `seed` selects the sequence.
fn tokens(len: usize, seed: usize) -> Vec<u32> {
    (0..len)
        .map(|i| (1000 + ((i + 31 * seed) * 7919 + 104_729 * (seed + 1)) % 90_000) as u32)
        .collect()
}

fn logits_of(cuda: &CudaBackend, hidden: &[f32]) -> Vec<f32> {
    let mut x = ActivationBuffer::zeros(hidden.len(), ComputeDtype::F32);
    x.write_f32_from(hidden);
    cuda.compute_final(&x).expect("logits").data
}

/// The largest logit's token and the runner-up's.
fn top2(logits: &[f32]) -> (u32, u32) {
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    (order[0] as u32, order[1] as u32)
}

/// Prefill `rest` from the backend's position and decode greedily: the rest's logits and the
/// tokens chosen.
fn run_rest(
    cuda: &CudaBackend,
    provider: &SyncWeightProvider,
    kv: &mut KvCache,
    rest: &[u32],
    steps: usize,
) -> (Vec<f32>, Vec<u32>) {
    let hidden = cuda
        .prefill_from(kv.seq_len(), rest, provider, kv)
        .expect("prefill of the rest");
    let first = logits_of(cuda, &hidden);
    let mut chosen = Vec::with_capacity(steps);
    let mut next = top2(&first).0;
    for _ in 0..steps {
        chosen.push(next);
        next = top2(&cuda.decode_token(next, provider, kv).expect("decode").data).0;
    }
    (first, chosen)
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// Which parts of the state at the end of the block `got` differs from `want` in: recurrent
/// states, conv rings, ring positions, sequence length, the block's KV rows.
fn differs(got: &StateSnapshot, want: &StateSnapshot, block: usize, head_dim: usize) -> [bool; 5] {
    let rows = |snap: &StateSnapshot| -> Vec<u32> {
        let per_head = snap.seq_len * head_dim;
        snap.kv
            .iter()
            .flat_map(|(k, v)| [k, v])
            .flat_map(|buf| {
                buf.chunks(per_head)
                    .flat_map(|head| &head[..block * head_dim])
            })
            .map(|v| v.to_bits())
            .collect()
    };
    let flat = |all: &[Vec<f32>]| -> Vec<u32> { all.iter().flat_map(|v| bits(v)).collect() };
    [
        flat(&got.h_states) != flat(&want.h_states),
        flat(&got.conv_states) != flat(&want.conv_states),
        got.conv_positions != want.conv_positions,
        got.seq_len != want.seq_len,
        rows(got) != rows(want),
    ]
}

/// Relative L2 distance of `got` from `want`.
fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let (mut d, mut n) = (0f64, 0f64);
    for (&x, &y) in got.iter().zip(want) {
        d += f64::from(x - y).powi(2);
        n += f64::from(y).powi(2);
    }
    (d / n).sqrt()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

#[test]
#[ignore]
fn a_restore_puts_the_sequence_back_to_the_end_of_its_block() {
    let path = PathBuf::from(
        std::env::var("LUMEN_PREFIX_CHECKPOINT_MODEL")
            .expect("set LUMEN_PREFIX_CHECKPOINT_MODEL to an LBC"),
    );
    let provider = SyncWeightProvider::open(&path).expect("open the model");
    let cuda = backend(&provider);
    let head_dim = provider.lbc().header.hyperparams.head_dim as usize;
    let f32_store = precision() == KvPrecision::F32;
    println!("KV {:?}", precision());

    // Block lengths cover every ring position (length mod 3), none a multiple of 64, one past
    // the 2,048-token prefill slice; the rests are below and above the session's 32-token
    // hand-off, and the other rest ends at another ring position.
    for (block_len, rest_len, other_len) in [
        (395, 1, 41),
        (397, 40, 18),
        (2501, 17, 300),
        (129, 40, 2100),
    ] {
        let tag = format!("block {block_len}, rest {rest_len}");
        let block = tokens(block_len, 0);
        let prompt: Vec<u32> = [block.clone(), tokens(rest_len, 1)].concat();
        let other: Vec<u32> = [block.clone(), tokens(other_len, 2)].concat();

        let mut kv = kv_cache(&provider);
        assert!(cuda.keeps_prefix_checkpoint());
        cuda.prefill_and_checkpoint(&block, &provider, &mut kv)
            .expect("keep the block");
        assert_eq!(kv.seq_len(), block_len, "{tag}");
        let kept = f32_store.then(|| cuda.snapshot_state(&kv).expect("snapshot"));
        let (want_logits, want_tokens) = run_rest(
            &cuda,
            &provider,
            &mut kv,
            &prompt[block_len..],
            GREEDY_STEPS,
        );

        // Another rest of the same block, served from the checkpoint and decoded.
        let mut kv = kv_cache(&provider);
        assert_eq!(
            cuda.restore_prefix_checkpoint(&other, &mut kv).unwrap(),
            Some(block_len),
            "{tag}"
        );
        let (other_logits, _) = run_rest(&cuda, &provider, &mut kv, &other[block_len..], 5);
        assert_ne!(bits(&other_logits), bits(&want_logits), "{tag}");
        if let Some(kept) = &kept {
            let moved = differs(
                &cuda.snapshot_state(&kv).expect("snapshot"),
                kept,
                block_len,
                head_dim,
            );
            assert_eq!(
                moved,
                [true, true, true, true, false],
                "{tag}: before the restore the recurrent states, conv rings, ring positions and \
                 length must differ from the kept ones and the block's KV rows must not"
            );
        }

        let mut kv = kv_cache(&provider);
        assert_eq!(
            cuda.restore_prefix_checkpoint(&prompt, &mut kv).unwrap(),
            Some(block_len),
            "{tag}"
        );
        assert_eq!(kv.seq_len(), block_len, "{tag}");
        if let Some(kept) = &kept {
            let restored = cuda.snapshot_state(&kv).expect("snapshot");
            assert_eq!(
                differs(&restored, kept, block_len, head_dim),
                [false; 5],
                "{tag}: state after the restore"
            );
            assert_eq!(restored.decode_token_count, 0, "{tag}");
        }
        let (got_logits, got_tokens) = run_rest(
            &cuda,
            &provider,
            &mut kv,
            &prompt[block_len..],
            GREEDY_STEPS,
        );
        assert_eq!(kv.seq_len(), prompt.len() + GREEDY_STEPS, "{tag}");
        assert_eq!(
            bits(&got_logits),
            bits(&want_logits),
            "{tag}: the rest's logits"
        );
        assert_eq!(got_tokens, want_tokens, "{tag}: greedy tokens");

        // Only a longer prompt that starts with the kept tokens restores.
        let mut changed = vec![
            prompt[..block_len].to_vec(),
            prompt[..block_len - 1].to_vec(),
        ];
        for at in [0, block_len / 2, block_len - 1] {
            let mut p = prompt.clone();
            p[at] += 1;
            changed.push(p);
        }
        for p in &changed {
            let mut kv = kv_cache(&provider);
            assert_eq!(
                cuda.restore_prefix_checkpoint(p, &mut kv).unwrap(),
                None,
                "{tag}"
            );
            assert_eq!(kv.seq_len(), 0, "{tag}");
        }

        // A rest the backend refuses leaves the checkpoint as it was.
        let mut kv = kv_cache(&provider);
        cuda.restore_prefix_checkpoint(&prompt, &mut kv).unwrap();
        assert!(cuda
            .prefill_from(block_len, &[u32::MAX], &provider, &mut kv)
            .is_err());
        let mut kv = kv_cache(&provider);
        assert_eq!(
            cuda.restore_prefix_checkpoint(&prompt, &mut kv).unwrap(),
            Some(block_len),
            "{tag}"
        );
        let (again, _) = run_rest(&cuda, &provider, &mut kv, &prompt[block_len..], 0);
        assert_eq!(
            bits(&again),
            bits(&want_logits),
            "{tag}: after a refused rest"
        );
        let split_state = f32_store.then(|| cuda.snapshot_state(&kv).expect("snapshot"));

        // The whole prompt in one call, beside the split one.
        cuda.reset_recurrent_state();
        let mut kv = kv_cache(&provider);
        assert_eq!(
            cuda.restore_prefix_checkpoint(&prompt, &mut kv).unwrap(),
            None,
            "{tag}: a reset ends the checkpoint"
        );
        let hidden = cuda.prefill(&prompt, &provider, &mut kv).expect("one call");
        let one_call = logits_of(&cuda, &hidden);
        let (split, whole) = (top2(&want_logits), top2(&one_call));
        let logits_apart = rel_l2(&want_logits, &one_call);
        let states_apart = split_state.map(|split| {
            let whole = cuda.snapshot_state(&kv).expect("snapshot");
            rel_l2(&split.h_states.concat(), &whole.h_states.concat())
        });
        println!(
            "{tag}: one call chooses {} (runner-up {}), split {} (runner-up {}), which one call \
             holds {:.3} logits lower; logits {logits_apart:.3e} apart, recurrent states {}",
            whole.0,
            whole.1,
            split.0,
            split.1,
            one_call[whole.0 as usize] - one_call[split.0 as usize],
            states_apart.map_or("not read".into(), |apart| format!("{apart:.3e}"))
        );
        assert!(
            logits_apart <= 0.25,
            "{tag}: the split prefill's logits are {logits_apart:.3e} from one call's"
        );
        if let Some(apart) = states_apart {
            assert!(
                apart <= 0.05,
                "{tag}: the split prefill's recurrent states are {apart:.3e} from one call's"
            );
        }
    }

    // Milliseconds from a request's first token to its logits, median of 7 after one unmeasured
    // pass: the prompt in one call, the block kept on the way, and the block restored.
    println!("block rest one_call_ms kept_ms restored_ms");
    for (block_len, rest_len) in [(395, 15), (395, 100), (2501, 15), (2501, 100)] {
        let block = tokens(block_len, 3);
        let prompt: Vec<u32> = [block.clone(), tokens(rest_len, 4)].concat();
        let (mut one_call, mut kept, mut restored) = (Vec::new(), Vec::new(), Vec::new());
        for pass in 0..8 {
            cuda.reset_recurrent_state();
            let mut kv = kv_cache(&provider);
            let t = Instant::now();
            let hidden = cuda.prefill(&prompt, &provider, &mut kv).unwrap();
            logits_of(&cuda, &hidden);
            let a = t.elapsed().as_secs_f64() * 1e3;

            let mut kv = kv_cache(&provider);
            let t = Instant::now();
            cuda.prefill_and_checkpoint(&block, &provider, &mut kv)
                .unwrap();
            let hidden = cuda
                .prefill_from(block_len, &prompt[block_len..], &provider, &mut kv)
                .unwrap();
            logits_of(&cuda, &hidden);
            let b = t.elapsed().as_secs_f64() * 1e3;

            let mut kv = kv_cache(&provider);
            let t = Instant::now();
            cuda.restore_prefix_checkpoint(&prompt, &mut kv).unwrap();
            let hidden = cuda
                .prefill_from(block_len, &prompt[block_len..], &provider, &mut kv)
                .unwrap();
            logits_of(&cuda, &hidden);
            let c = t.elapsed().as_secs_f64() * 1e3;
            if pass > 0 {
                one_call.push(a);
                kept.push(b);
                restored.push(c);
            }
        }
        println!(
            "{block_len} {rest_len} {:.2} {:.2} {:.2}",
            median(one_call),
            median(kept),
            median(restored)
        );
    }
}
