//! Decode attention on CUDA: one kernel family for every model and context.
//!
//! The kernels (`attention_decode_partial_f32` / `_f16` and
//! `attention_decode_merge`, `shaders/attention_decode.cu`) are the split-K
//! flash-decoding structure: one CTA per (KV head, chunk) walks 16-key tiles
//! with a running-max recurrence and writes an unnormalised numerator with
//! its running max and sum; the merge combines the chunks per head. Below a
//! one-tile bound every CTA takes one tile; above it the split count is held
//! at a fixed target and each CTA walks a balanced run of whole tiles, so the
//! scratch is fixed at init whatever the context and a generation never
//! changes route as it grows. The module is compiled once per backend for the
//! model's shape ([`DecodeAttentionSpec`]); the policy is derived per model
//! ([`DecodeAttentionPolicy`]); every launch is decided once
//! ([`DecodeAttentionLaunch`]) and the same receipt reaches the launcher, the
//! route line and the dump.

use cudarc::driver::{CudaSlice, DeviceRepr, LaunchConfig as CudarcLaunchConfig, PushKernelArg};

use crate::error::RuntimeError;

use super::decode::{KernelSet, ATTN_DECODE_BLOCK_DIM};
use super::ffi::CudaDevice;
use super::kv_cache::KvRef;

/// KV positions per tile: the whole tile's scores for one query head live in
/// one warp's lanes, so a tile cannot exceed 32; 16 is what the geometry
/// measured fastest at.
pub const ATTN_DECODE_TILE: u32 = 16;

/// Ceiling on the split count: the merge's shared block is sized for it and
/// no policy asks for more.
pub const ATTN_DECODE_S_MAX: u32 = 1024;

/// The shape the decode-attention module is compiled for: the
/// group size (query heads per KV head) and the head dimension. The kernel
/// forms all `group` scores of a KV head from one register-resident K row
/// and owns `head_dim` across its 128 threads, so both are compile-time
/// constants of the module: the host prepends them to the source as
/// `#define DECODE_G` / `#define DECODE_HD` (the shader refuses to compile
/// without them), and a backend compiles the module once, for the model it
/// serves. Every shipped model has head_dim 256 — Qwen3.5-9B at a group of 4
/// (16 query / 4 KV heads), Qwen3.6-27B and Qwen3.8-27B at 6 (24 / 4),
/// Qwen3.5-MoE-35B-A3B at 8 (16 / 2); head_dim 128 serves the generated test
/// models. Register budget at `__launch_bounds__(128, 4)`
/// through the engine's NVRTC path on sm_120: 72 / 96 / 128 registers at
/// groups 4 / 6 / 8 with head_dim 256, no spills at any admitted shape.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct DecodeAttentionSpec {
    /// Query heads per KV head, 1..=8.
    pub group: u32,
    /// Head dimension: 128 or 256.
    pub head_dim: u32,
}

impl DecodeAttentionSpec {
    pub const GROUPS: std::ops::RangeInclusive<u32> = 1..=8;
    pub const HEAD_DIMS: [u32; 2] = [128, 256];

    /// The specification for a model's full-attention shape, or the reason
    /// the kernel cannot serve it (the message names the shape and the
    /// domain).
    pub fn for_shape(num_heads: u32, num_kv_heads: u32, head_dim: u32) -> Result<Self, String> {
        if num_kv_heads == 0 || num_heads == 0 {
            return Err(format!(
                "decode attention: {num_heads} query heads over {num_kv_heads} KV heads; both must be positive"
            ));
        }
        if num_heads % num_kv_heads != 0 {
            return Err(format!(
                "decode attention: {num_heads} query heads are not a whole number of groups over {num_kv_heads} KV heads"
            ));
        }
        let group = num_heads / num_kv_heads;
        if !Self::GROUPS.contains(&group) {
            return Err(format!(
                "decode attention: {group} query heads per KV head ({num_heads} / {num_kv_heads}); the kernel serves 1 to 8"
            ));
        }
        if !Self::HEAD_DIMS.contains(&head_dim) {
            return Err(format!(
                "decode attention: head_dim {head_dim}; the kernel serves 128 or 256"
            ));
        }
        Ok(Self { group, head_dim })
    }

    /// The module source for this shape: the two defines, then the shader.
    pub fn source(&self) -> String {
        format!(
            "#define DECODE_G {}u\n#define DECODE_HD {}u\n{}",
            self.group,
            self.head_dim,
            crate::cuda::shaders::ATTENTION_DECODE_KERNEL_SOURCE
        )
    }

    /// Whether a dispatch's shape is the one this module was compiled for.
    pub const fn matches(&self, num_heads: u32, num_kv_heads: u32, head_dim: u32) -> bool {
        num_kv_heads != 0 && num_heads == num_kv_heads * self.group && head_dim == self.head_dim
    }

    /// Dynamic shared bytes of the partial pass: Q for the group, one 16-key
    /// V tile (halves on a half store), the tile's scores, and the running
    /// max, sum and rescale per head.
    pub const fn partial_shared_bytes(&self, half_store: bool) -> u32 {
        let v = if half_store {
            ATTN_DECODE_TILE * self.head_dim / 2
        } else {
            ATTN_DECODE_TILE * self.head_dim
        };
        (self.group * self.head_dim + v + self.group * ATTN_DECODE_TILE + 3 * self.group) * 4
    }

    /// CTAs per query head in the merge, 128 dimensions each.
    pub const fn dim_tiles(&self) -> u32 {
        self.head_dim / ATTN_DECODE_BLOCK_DIM
    }
}

/// The reviewed shape of the kernel (Qwen3.8-27B: 24 query heads over 4 KV
/// heads at head_dim 256), which the reference fixture pins bit for bit.
pub const ATTN_DECODE_REVIEWED_SHAPE: DecodeAttentionSpec = DecodeAttentionSpec {
    group: 6,
    head_dim: 256,
};

const _: () = assert!(ATTN_DECODE_TILE <= 32);
const _: () = assert!(ATTN_DECODE_REVIEWED_SHAPE.partial_shared_bytes(false) == 22_984);
const _: () = assert!(ATTN_DECODE_REVIEWED_SHAPE.partial_shared_bytes(true) == 14_792);
// The largest admitted shape stays under the 48 KiB default dynamic-shared
// cap, so no opt-in is ever needed.
const _: () = assert!(
    DecodeAttentionSpec {
        group: 8,
        head_dim: 256
    }
    .partial_shared_bytes(false)
        <= 49_152
);
const _: () = assert!(decode_attention_merge_shared_bytes(ATTN_DECODE_S_MAX) <= 49_152);

/// Which partition a launch takes: `0` one tile per CTA on the span
/// partition (S = ceil(keys / 16)), `1` the balanced whole-tile partition.
pub type DecodePartition = u32;

/// The split policy for one model: the one-tile bound and the whole-tile
/// target, from the knobs when set, else derived from the model's KV-head
/// count ([`crate::runtime_defaults::attn_one_tile_default`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct DecodeAttentionPolicy {
    /// One tile per CTA up to this many tiles, within the compile-time ceiling.
    pub one_tile_max: u32,
    /// The split count above the one-tile bound.
    pub target: u32,
}

impl DecodeAttentionPolicy {
    pub fn from_env(num_kv_heads: u32) -> Self {
        Self {
            one_tile_max: crate::runtime_defaults::attn_one_tile_max(num_kv_heads)
                .clamp(1, ATTN_DECODE_S_MAX),
            target: crate::runtime_defaults::attn_target(num_kv_heads).clamp(1, ATTN_DECODE_S_MAX),
        }
    }

    pub const fn geometry(&self, seq_len: u32) -> (u32, DecodePartition) {
        decode_attention_geometry_within(seq_len, self.one_tile_max, self.target)
    }

    pub const fn scratch_chunks(&self, max_seq_len: u32) -> u32 {
        decode_attention_scratch_chunks_within(max_seq_len, self.one_tile_max, self.target)
    }
}

/// The launch geometry for a context under an explicit policy: the split
/// count and the partition (0 = one tile per CTA, 1 = whole tiles).
pub const fn decode_attention_geometry_within(
    seq_len: u32,
    one_tile_max: u32,
    target: u32,
) -> (u32, DecodePartition) {
    let n = seq_len.div_ceil(ATTN_DECODE_TILE);
    let n = if n == 0 { 1 } else { n };
    if n <= one_tile_max {
        (n, 0)
    } else {
        // Never more CTAs than tiles: a target above the tile count would
        // leave CTAs empty and ask for scratch a short cache never sized.
        (if target < n { target } else { n }, 1)
    }
}

/// The scratch count for an explicit policy (pure): the largest split count
/// [`decode_attention_geometry_within`] returns for any context up to
/// `max_seq_len` under `(one_tile_max, target)`. The unit test
/// `the_scratch_holds_every_context_the_policy_can_produce` sweeps the pair.
pub const fn decode_attention_scratch_chunks_within(
    max_seq_len: u32,
    one_tile_max: u32,
    target: u32,
) -> u32 {
    let n_max = max_seq_len.div_ceil(ATTN_DECODE_TILE);
    let n_max = if n_max == 0 { 1 } else { n_max };
    let policy_max = if one_tile_max >= target {
        one_tile_max
    } else {
        target
    };
    let s = if n_max < policy_max {
        n_max
    } else {
        policy_max
    };
    if s == 0 {
        1
    } else {
        s
    }
}

/// Dynamic shared bytes for the merge: one rescale factor per
/// chunk plus the four-warp reduction scratch.
pub const fn decode_attention_merge_shared_bytes(chunks: u32) -> u32 {
    (chunks + 4) * 4
}

/// The split count the scratch must hold for a cache of `max_seq_len` under
/// the policy in force for a model with `num_kv_heads` KV heads.
pub fn decode_attention_scratch_chunks(max_seq_len: u32, num_kv_heads: u32) -> u32 {
    DecodeAttentionPolicy::from_env(num_kv_heads).scratch_chunks(max_seq_len)
}

/// What one decode-attention call launches, decided once from a policy
/// snapshot and the compiled module: the launcher launches exactly this, and
/// the route line and the dump record it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct DecodeAttentionLaunch {
    pub chunks: u32,
    pub partition: DecodePartition,
    pub policy: DecodeAttentionPolicy,
    /// The NVRTC target the module was compiled for (`LUMEN_CUDA_ATTN_CODEGEN`).
    pub codegen: &'static str,
}

impl DecodeAttentionLaunch {
    /// The route name a census sees: the kernel, suffixed with the codegen
    /// target when it is not NVRTC's default (the CUDA symbol is the same).
    pub fn route_name(&self, half_store: bool) -> &'static str {
        match (half_store, self.codegen) {
            (false, "ptx120") => "attention_decode_partial_f32_ptx120",
            (false, "ptx80") => "attention_decode_partial_f32_ptx80",
            (false, _) => "attention_decode_partial_f32",
            (true, "ptx120") => "attention_decode_partial_f16_ptx120",
            (true, "ptx80") => "attention_decode_partial_f16_ptx80",
            (true, _) => "attention_decode_partial_f16",
        }
    }
}

/// The admission: the dispatch's shape must be the module's, the scratch
/// must hold the split count the policy picks. Both hold by construction
/// (the backend refused any other shape at init and sized the scratch from
/// the same policy), so a failure here is an engine defect and is an error,
/// never a fallback.
fn admit(
    kernels: &KernelSet,
    scratch_o_floats: usize,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
) -> Result<DecodeAttentionLaunch, RuntimeError> {
    let spec = kernels.attn_spec;
    if !spec.matches(num_heads, num_kv_heads, head_dim) {
        return Err(RuntimeError::Compute(format!(
            "decode attention: dispatch shape {num_heads} query heads / {num_kv_heads} KV heads / head_dim {head_dim} is not the compiled shape (group {}, head_dim {})",
            spec.group, spec.head_dim
        )));
    }
    if seq_len == 0 {
        return Err(RuntimeError::Compute(
            "decode attention: a call with no KV positions".into(),
        ));
    }
    let policy = DecodeAttentionPolicy::from_env(num_kv_heads);
    let (chunks, partition) = policy.geometry(seq_len);
    let need = (num_heads as usize) * (chunks as usize) * (head_dim as usize);
    if scratch_o_floats < need {
        return Err(RuntimeError::Compute(format!(
            "decode attention: the split-K scratch holds {scratch_o_floats} floats but {chunks} chunks at seq_len {seq_len} need {need}"
        )));
    }
    Ok(DecodeAttentionLaunch {
        chunks,
        partition,
        policy,
        codegen: kernels.attn_codegen,
    })
}

/// The partial pass and the merge for one store type.
#[allow(clippy::too_many_arguments)]
unsafe fn launch_pair<K: DeviceRepr>(
    device: &CudaDevice,
    kernels: &KernelSet,
    partial_fn: &cudarc::driver::CudaFunction,
    half_store: bool,
    q: &CudaSlice<f32>,
    k_cache: &CudaSlice<K>,
    v_cache: &CudaSlice<K>,
    scratch: &mut (CudaSlice<f32>, CudaSlice<f32>, CudaSlice<f32>),
    attn_out: &mut CudaSlice<f32>,
    num_heads: u32,
    num_kv_heads: u32,
    seq_len: u32,
    max_seq_len: u32,
    scale: f32,
    launch: DecodeAttentionLaunch,
) -> Result<(), RuntimeError> {
    let spec = kernels.attn_spec;
    let s = launch.chunks;
    let partition = launch.partition;
    let (m_part, l_part, o_part) = scratch;
    device
        .stream
        .launch_builder(partial_fn)
        .arg(q)
        .arg(k_cache)
        .arg(v_cache)
        .arg(&mut *m_part)
        .arg(&mut *l_part)
        .arg(&mut *o_part)
        .arg(&seq_len)
        .arg(&max_seq_len)
        .arg(&scale)
        .arg(&s)
        .arg(&partition)
        .launch(CudarcLaunchConfig {
            grid_dim: (s, num_kv_heads, 1),
            block_dim: (ATTN_DECODE_BLOCK_DIM, 1, 1),
            shared_mem_bytes: spec.partial_shared_bytes(half_store),
        })
        .map_err(|e| RuntimeError::Compute(format!("{}: {e}", launch.route_name(half_store))))?;
    device
        .stream
        .launch_builder(&kernels.attention_decode_merge)
        .arg(&*m_part)
        .arg(&*l_part)
        .arg(&*o_part)
        .arg(attn_out)
        .arg(&s)
        .launch(CudarcLaunchConfig {
            grid_dim: (num_heads, spec.dim_tiles(), 1),
            block_dim: (ATTN_DECODE_BLOCK_DIM, 1, 1),
            shared_mem_bytes: decode_attention_merge_shared_bytes(s),
        })
        .map_err(|e| RuntimeError::Compute(format!("attention_decode_merge: {e}")))?;
    Ok(())
}

/// Name the route on its first dispatch, with the geometry that dispatch
/// took and the policy in force.
fn announce_route(
    half_store: bool,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    launch: DecodeAttentionLaunch,
) {
    static SEEN_F32: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    static SEEN_F16: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    let seen = if half_store { &SEEN_F16 } else { &SEEN_F32 };
    super::decode::announce_route_once(seen, || {
        format!(
            "[CUDA] {}: ACTIVE (kv={}, q_heads={num_heads}, kv_heads={num_kv_heads}, head_dim={head_dim}, \
             seq_len={seq_len}, chunks={}, partition={}, tile={}, one_tile_max={}, target={}, \
             block={}, merge=attention_decode_merge)",
            launch.route_name(half_store),
            if half_store { "f16" } else { "f32" },
            launch.chunks,
            if launch.partition == 0 {
                "one-tile"
            } else {
                "whole-tile"
            },
            ATTN_DECODE_TILE,
            launch.policy.one_tile_max,
            launch.policy.target,
            ATTN_DECODE_BLOCK_DIM,
        )
    });
}

/// Decode attention for one token over the store, whichever type it holds:
/// the admission, the pair, the route line and, when asked for, the dump.
/// The store's type picks the partial kernel; the merge is shared.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn launch_attention_decode(
    device: &CudaDevice,
    kernels: &KernelSet,
    q: &CudaSlice<f32>,
    kv: KvRef<'_>,
    scratch: &mut (CudaSlice<f32>, CudaSlice<f32>, CudaSlice<f32>),
    attn_out: &mut CudaSlice<f32>,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    max_seq_len: u32,
    scale: f32,
) -> Result<DecodeAttentionLaunch, RuntimeError> {
    let launch = admit(
        kernels,
        scratch.2.len(),
        num_heads,
        num_kv_heads,
        head_dim,
        seq_len,
    )?;
    let half_store = matches!(kv, KvRef::F16 { .. });
    match kv {
        KvRef::F32 { k, v } => launch_pair(
            device,
            kernels,
            &kernels.attention_decode_partial,
            false,
            q,
            k,
            v,
            scratch,
            attn_out,
            num_heads,
            num_kv_heads,
            seq_len,
            max_seq_len,
            scale,
            launch,
        )?,
        KvRef::F16 { k, v } => {
            let f16 = kernels.kv_f16.as_ref().ok_or_else(|| {
                RuntimeError::Compute("16-bit KV cache dispatched without its kernels".into())
            })?;
            launch_pair(
                device,
                kernels,
                &f16.attention_decode_partial,
                true,
                q,
                k,
                v,
                scratch,
                attn_out,
                num_heads,
                num_kv_heads,
                seq_len,
                max_seq_len,
                scale,
                launch,
            )?
        }
    }
    announce_route(
        half_store,
        num_heads,
        num_kv_heads,
        head_dim,
        seq_len,
        launch,
    );
    if let Some((dir, lengths)) = attention_dump_config() {
        if lengths.contains(&seq_len) {
            dump_attention_call(
                device,
                dir,
                q,
                kv,
                attn_out,
                num_heads,
                num_kv_heads,
                head_dim,
                seq_len,
                max_seq_len,
                scale,
                launch,
            )?;
        }
    }
    Ok(launch)
}

/// `LUMEN_CUDA_ATTN_DUMP=<dir>:<seq_len>[,<seq_len>...]`, parsed once: the
/// directory the decode-attention dump writes into and the sequence lengths
/// it writes at. `None` when unset or malformed (a malformed value is said
/// once and ignored, so a typo can never stall a decode).
fn attention_dump_config() -> Option<&'static (std::path::PathBuf, Vec<u32>)> {
    static CONFIG: std::sync::OnceLock<Option<(std::path::PathBuf, Vec<u32>)>> =
        std::sync::OnceLock::new();
    CONFIG
        .get_or_init(|| {
            let raw = std::env::var("LUMEN_CUDA_ATTN_DUMP").ok()?;
            let parsed = raw.rsplit_once(':').and_then(|(dir, lengths)| {
                let lengths: Vec<u32> = lengths
                    .split(',')
                    .map(|n| n.trim().parse::<u32>().ok())
                    .collect::<Option<_>>()?;
                (!dir.is_empty() && !lengths.is_empty())
                    .then(|| (std::path::PathBuf::from(dir), lengths))
            });
            if parsed.is_none() {
                eprintln!(
                    "[CUDA] LUMEN_CUDA_ATTN_DUMP={raw:?}: want <dir>:<seq_len>[,<seq_len>...]; ignored"
                );
            }
            parsed
        })
        .as_ref()
}

/// Write one decode-attention call to `dir`: `attn-<seq_len>-<call>.json`
/// (shape, scale, the kernel and launch geometry that served) beside the raw little-endian F32
/// files `.q.f32` (`[num_heads, head_dim]`), `.k.f32` and `.v.f32` — `.k.f16`
/// and `.v.f16` on a half store, with `kv_dtype` in the header — (the live
/// `[num_kv_heads, seq_len, head_dim]` region of the cache) and `.out.f32`
/// (the route's output, `[num_heads, head_dim]`). The call counter runs over
/// the process, so the attention layers of one token appear in order. A
/// diagnostic for replaying real activations through a reference; it copies
/// the cache to the host and is not for a measured run.
#[allow(clippy::too_many_arguments)]
fn dump_attention_call(
    device: &CudaDevice,
    dir: &std::path::Path,
    q: &CudaSlice<f32>,
    kv: KvRef<'_>,
    attn_out: &CudaSlice<f32>,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    max_seq_len: u32,
    scale: f32,
    launch: DecodeAttentionLaunch,
) -> Result<(), RuntimeError> {
    use std::sync::atomic::{AtomicU32, Ordering};
    static CALL: AtomicU32 = AtomicU32::new(0);
    let call = CALL.fetch_add(1, Ordering::Relaxed);
    let io = |e: std::io::Error| RuntimeError::Compute(format!("attention dump: {e}"));
    std::fs::create_dir_all(dir).map_err(io)?;
    let stem = dir.join(format!("attn-{seq_len}-{call:04}"));
    let (nh, nkv, hd, sl, msl) = (
        num_heads as usize,
        num_kv_heads as usize,
        head_dim as usize,
        seq_len as usize,
        max_seq_len as usize,
    );
    let write_f32 = |suffix: &str, data: &[f32]| -> Result<(), RuntimeError> {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for x in data {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        std::fs::write(stem.with_extension(suffix), bytes).map_err(io)
    };
    let q_host: Vec<f32> = device.dtoh_copy_view(&q.slice(0..nh * hd))?;
    write_f32("q.f32", &q_host)?;
    // The cache region as stored: F32 words, or the half bit patterns the
    // kernel read (`.k.f16` / `.v.f16`, 16-bit little-endian). A replay of a
    // half dump must widen exactly; the storage rounding already happened.
    let kv_dtype = match kv {
        KvRef::F32 { k, v } => {
            for (name, cache) in [("k.f32", k), ("v.f32", v)] {
                let mut host = Vec::with_capacity(nkv * sl * hd);
                for kv_h in 0..nkv {
                    let base = kv_h * msl * hd;
                    let region: Vec<f32> =
                        device.dtoh_copy_view(&cache.slice(base..base + sl * hd))?;
                    host.extend_from_slice(&region);
                }
                write_f32(name, &host)?;
            }
            "f32"
        }
        KvRef::F16 { k, v } => {
            for (name, cache) in [("k.f16", k), ("v.f16", v)] {
                let mut bytes = Vec::with_capacity(nkv * sl * hd * 2);
                for kv_h in 0..nkv {
                    let base = kv_h * msl * hd;
                    let region: Vec<u16> =
                        device.dtoh_copy_view(&cache.slice(base..base + sl * hd))?;
                    for x in &region {
                        bytes.extend_from_slice(&x.to_le_bytes());
                    }
                }
                std::fs::write(stem.with_extension(name), bytes).map_err(io)?;
            }
            "f16"
        }
    };
    let out_host: Vec<f32> = device.dtoh_copy_view(&attn_out.slice(0..nh * hd))?;
    write_f32("out.f32", &out_host)?;
    // The receipt is the launch's own, not a fresh reading of the policy:
    // the dump records what ran, with the geometry a replay needs to
    // reproduce the reduction order.
    let route = launch.route_name(kv_dtype == "f16");
    let geometry = format!(
        ",\n \"chunks\": {},\n \"partition\": \"{}\",\n \"one_tile_max\": {},\n \"target\": {},\n \"codegen\": \"{}\"",
        launch.chunks,
        if launch.partition == 0 { "one-tile" } else { "whole-tile" },
        launch.policy.one_tile_max,
        launch.policy.target,
        launch.codegen,
    );
    let engine = crate::runtime_defaults::build_identity();
    let meta = format!(
        "{{\n \"format\": \"lumen-attn-dump@1\",\n \"engine\": \"{engine}\",\n \"kv_dtype\": \"{kv_dtype}\",\n \"call\": {call},\n \"route\": \"{route}\",\n \"num_heads\": {num_heads},\n \"num_kv_heads\": {num_kv_heads},\n \"head_dim\": {head_dim},\n \"seq_len\": {seq_len},\n \"max_seq_len\": {max_seq_len},\n \"scale\": {scale:e}{geometry}\n}}\n"
    );
    std::fs::write(stem.with_extension("json"), meta).map_err(io)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shader carries no shape of its own: the host prepends the group
    /// size and head dimension, and the source refuses to compile without
    /// them.
    #[test]
    fn the_shader_takes_its_shape_from_the_host() {
        let src = crate::cuda::shaders::ATTENTION_DECODE_KERNEL_SOURCE;
        for name in ["DECODE_G", "DECODE_HD"] {
            assert!(
                !src.lines()
                    .any(|l| l.starts_with(&format!("#define {name} "))),
                "the shader defines {name} itself"
            );
            assert!(
                src.contains(&format!("#ifndef {name}\n#error")),
                "the shader does not refuse to compile without {name}"
            );
        }
        let block = src
            .lines()
            .find(|l| l.starts_with("#define DECODE_BLOCK "))
            .expect("DECODE_BLOCK");
        assert_eq!(
            block
                .split_whitespace()
                .nth(2)
                .map(|v| v.trim_end_matches('u')),
            Some("128")
        );
        let tile = src
            .lines()
            .find(|l| l.starts_with("#define DECODE_TILE "))
            .expect("DECODE_TILE");
        assert_eq!(
            tile.split_whitespace()
                .nth(2)
                .map(|v| v.trim_end_matches('u')),
            Some("16")
        );
        let text = ATTN_DECODE_REVIEWED_SHAPE.source();
        assert!(text.starts_with("#define DECODE_G 6u\n#define DECODE_HD 256u\n"));
        assert!(text.ends_with(src));
    }

    /// The shape domain: every shipped model, the test shape, and the
    /// refusals with their reasons.
    #[test]
    fn the_spec_admits_the_shipped_models_and_names_what_it_refuses() {
        assert_eq!(
            DecodeAttentionSpec::for_shape(16, 4, 256),
            Ok(DecodeAttentionSpec {
                group: 4,
                head_dim: 256
            })
        );
        assert_eq!(
            DecodeAttentionSpec::for_shape(24, 4, 256),
            Ok(ATTN_DECODE_REVIEWED_SHAPE)
        );
        assert_eq!(
            DecodeAttentionSpec::for_shape(16, 2, 256),
            Ok(DecodeAttentionSpec {
                group: 8,
                head_dim: 256
            })
        );
        assert_eq!(
            DecodeAttentionSpec::for_shape(2, 2, 128),
            Ok(DecodeAttentionSpec {
                group: 1,
                head_dim: 128
            })
        );
        for (nh, nkv, hd, word) in [
            (0, 4, 256, "positive"),
            (24, 0, 256, "positive"),
            (25, 4, 256, "whole number"),
            (36, 4, 256, "1 to 8"),
            (24, 4, 64, "128 or 256"),
            (24, 4, 96, "128 or 256"),
            (24, 4, 512, "128 or 256"),
        ] {
            let err = DecodeAttentionSpec::for_shape(nh, nkv, hd).unwrap_err();
            assert!(err.contains(word), "({nh}, {nkv}, {hd}): {err}");
        }
    }

    /// The merge's CTAs cover the head exactly, one dimension per thread;
    /// shared bytes grow with the group and shrink on the half store.
    #[test]
    fn the_merge_tiles_cover_the_head() {
        for hd in DecodeAttentionSpec::HEAD_DIMS {
            let spec = DecodeAttentionSpec {
                group: 1,
                head_dim: hd,
            };
            assert_eq!(spec.dim_tiles() * ATTN_DECODE_BLOCK_DIM, hd);
        }
        let g8 = DecodeAttentionSpec {
            group: 8,
            head_dim: 256,
        };
        assert_eq!(g8.partial_shared_bytes(false), 25_184);
        assert_eq!(g8.partial_shared_bytes(true), 16_992);
    }

    /// No context makes a tile outgrow its warp, under any policy.
    #[test]
    fn no_context_makes_a_tile_outgrow_its_warp() {
        for (one_tile, target) in [(176, 128), (352, 256), (1024, 128), (64, 340), (1, 1)] {
            for seq_len in (1..=70_000)
                .step_by(7)
                .chain([1u32, 16, 17, 2816, 2817, 4096, 65_536])
            {
                let n = seq_len.div_ceil(ATTN_DECODE_TILE).max(1);
                let (s, partition) = decode_attention_geometry_within(seq_len, one_tile, target);
                assert!(s >= 1 && s <= ATTN_DECODE_S_MAX, "seq_len={seq_len}: S={s}");
                if partition == 0 {
                    assert_eq!(s, n, "one-tile partition keeps S = N at seq_len={seq_len}");
                    assert!(n <= one_tile);
                    assert!(seq_len.div_ceil(s) <= ATTN_DECODE_TILE);
                } else {
                    assert_eq!(
                        s,
                        target.min(n),
                        "whole-tile: the target, never more CTAs than tiles"
                    );
                    assert!(n > one_tile);
                }
            }
        }
    }

    /// The scratch sized at init holds the split count of every context the
    /// cache can reach, under every policy the knobs can express.
    #[test]
    fn the_scratch_holds_every_context_the_policy_can_produce() {
        for (one_tile, target) in [
            (176, 128),
            (352, 256),
            (1024, 128),
            (64, 340),
            (1, 1),
            (1, 1024),
            (1024, 1024),
            (2, 3),
        ] {
            for max_seq_len in [
                1u32,
                15,
                16,
                17,
                1_000,
                2_816,
                2_817,
                4_096,
                16_384,
                32_768,
                1 << 20,
            ] {
                let scratch = decode_attention_scratch_chunks_within(max_seq_len, one_tile, target);
                assert!((1..=ATTN_DECODE_S_MAX).contains(&scratch));
                for seq_len in (1..=max_seq_len).step_by(13).chain([
                    1,
                    max_seq_len,
                    max_seq_len.saturating_sub(1).max(1),
                ]) {
                    let (s, _) = decode_attention_geometry_within(seq_len, one_tile, target);
                    assert!(
                        s <= scratch,
                        "policy ({one_tile}, {target}): seq_len {seq_len} takes {s} chunks but the scratch for max_seq_len {max_seq_len} holds {scratch}"
                    );
                }
            }
        }
    }

    /// The split count is bounded by the policy, never by the context, and
    /// the per-model defaults reproduce the measured 4-KV-head values.
    #[test]
    fn the_policy_is_per_model() {
        let _guard = crate::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        std::env::remove_var("LUMEN_CUDA_ATTN_ONE_TILE");
        std::env::remove_var("LUMEN_CUDA_ATTN_TARGET");
        let p4 = DecodeAttentionPolicy::from_env(4);
        assert_eq!((p4.one_tile_max, p4.target), (176, 128));
        assert_eq!(p4.geometry(2_816), (176, 0));
        assert_eq!(p4.geometry(2_817), (128, 1));
        assert_eq!(p4.geometry(1 << 20), (128, 1));
        assert_eq!(p4.scratch_chunks(1 << 20), 176);
        assert_eq!(p4.scratch_chunks(1_000), 63);
        let p2 = DecodeAttentionPolicy::from_env(2);
        assert_eq!((p2.one_tile_max, p2.target), (352, 256));
        assert_eq!(p2.geometry(352 * 16), (352, 0));
        assert_eq!(p2.geometry(352 * 16 + 1), (256, 1));
        assert_eq!(decode_attention_geometry_within(0, 176, 128), (1, 0));
        assert_eq!(decode_attention_geometry_within(2817, 176, 340), (177, 1));
        std::env::set_var("LUMEN_CUDA_ATTN_ONE_TILE", "64");
        std::env::set_var("LUMEN_CUDA_ATTN_TARGET", "96");
        let p = DecodeAttentionPolicy::from_env(4);
        assert_eq!((p.one_tile_max, p.target), (64, 96));
        std::env::remove_var("LUMEN_CUDA_ATTN_ONE_TILE");
        std::env::remove_var("LUMEN_CUDA_ATTN_TARGET");
    }

    #[test]
    fn the_route_name_carries_the_codegen_target() {
        let l = |codegen| DecodeAttentionLaunch {
            chunks: 1,
            partition: 0,
            policy: DecodeAttentionPolicy {
                one_tile_max: 176,
                target: 128,
            },
            codegen,
        };
        assert_eq!(
            l("default").route_name(false),
            "attention_decode_partial_f32"
        );
        assert_eq!(
            l("ptx120").route_name(false),
            "attention_decode_partial_f32_ptx120"
        );
        assert_eq!(
            l("default").route_name(true),
            "attention_decode_partial_f16"
        );
        assert_eq!(
            l("ptx80").route_name(true),
            "attention_decode_partial_f16_ptx80"
        );
    }
}
