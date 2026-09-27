//! The native prefill's attention kernels for the full-attention layers
//! (`shaders/native_prefill_attn.cu`): the RoPE table, the per-head query/key norm with RoPE and
//! the new KV rows, the BF16 staging of older KV rows, and causal GQA attention on BF16 tensor cores
//! with F32 accumulation.
//!
//! The group is its own NVRTC module, built like the producers' group
//! ([`super::native_prefill_kernels`]) for `compute_120a` with no header, under the same admission
//! conditions (Q1 to Q3). Its kernels have `native_` names and are reachable only through
//! [`NativeAttnKernels`], never through the decode kernel set. Attention reads BF16 K and V in the
//! cache layout (`[4][max_seq][256]` per layer): an F32 cache's BF16 staging copy, or a BF16 cache
//! itself, which the prep writes alone.
//!
//! Loading launches every kernel once on a small fixed problem whose outputs must match recorded
//! digests. The GPU suite (`tests/cuda_native_attention_test.rs`) checks T = 1 to 2049 tokens
//! from p0 = 0 to 2048, including the 64-row tile edges, against host oracles.

use super::ffi::CudaDevice;
use super::native_prefill::{Refusal, HEADS, HEAD_DIM, KV_HEADS, ROTARY_DIM};
use super::native_prefill_kernels::{checksum, compile_group, KernelSpec};
use super::shaders::NATIVE_PREFILL_ATTN_KERNEL_SOURCE;
use crate::error::RuntimeError;
use cudarc::driver::{CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
use std::sync::Arc;

/// Query rows and keys per attention tile.
pub const ATTN_TILE: u32 = 64;
/// Row stride of a shared attention tile in BF16 elements: the head size plus 8 of padding.
const ATTN_LD: u32 = HEAD_DIM + 8;
/// Dynamic shared memory of the attention kernel: its query, key and value tiles.
pub const ATTN_SHARED: u32 = 3 * ATTN_TILE * ATTN_LD * 2;
const _: () = assert!(
    ATTN_SHARED <= 101_376,
    "the attention tiles exceed the shared memory a compute capability 12.0 block can opt into"
);

pub const ROPE_TABLE: usize = 0;
pub const ATTN_PREP: usize = 1;
pub const KV_TO_BF16: usize = 2;
pub const ATTN_PREFILL: usize = 3;

/// Every kernel of the group.
pub const ATTN_KERNELS: [KernelSpec; 4] = [
    KernelSpec {
        name: "native_rope_table",
        block: 256,
        dynamic_shared: 0,
    },
    KernelSpec {
        name: "native_attn_prep",
        block: 128,
        dynamic_shared: 0,
    },
    KernelSpec {
        name: "native_kv_to_bf16",
        block: 256,
        dynamic_shared: 0,
    },
    KernelSpec {
        name: "native_attn_prefill",
        block: 128,
        dynamic_shared: ATTN_SHARED,
    },
];

/// `scale * log2(e)` of the softmax, with scale = 1 / sqrt(256) = 1 / 16.
pub fn scale_log2() -> f32 {
    (1.0 / 16.0) * std::f32::consts::LOG2_E
}

/// The compiled, qualified attention kernels.
pub struct NativeAttnKernels {
    _module: Arc<CudaModule>,
    functions: Vec<CudaFunction>,
}

fn refusal(condition: &'static str, reason: String) -> Refusal {
    Refusal { condition, reason }
}

fn launch_err(name: &str) -> impl FnOnce(cudarc::driver::DriverError) -> RuntimeError + '_ {
    move |e| RuntimeError::Compute(format!("{name}: {e}"))
}

impl NativeAttnKernels {
    /// Compile the group, set each kernel's attributes and run the qualifying launches.
    pub fn load(device: &CudaDevice) -> Result<Self, Refusal> {
        let kernels = Self::compile(device)?;
        kernels.qualify(device)?;
        Ok(kernels)
    }

    /// Compile the group and set each kernel's attributes, without the qualifying launches.
    pub fn compile(device: &CudaDevice) -> Result<Self, Refusal> {
        // SAFETY: the group's own source.
        unsafe { Self::compile_source(device, NATIVE_PREFILL_ATTN_KERNEL_SOURCE) }
    }

    /// [`Self::compile`] of `source` in place of the group's own: the suites build altered variants
    /// through the same target selection and attribute setup.
    ///
    /// # Safety
    /// [`Self::qualify`] and the launchers run these kernels as the group's own: `source` must define
    /// every kernel in [`ATTN_KERNELS`] with the group's parameters, and each must access only the
    /// memory those launches give it.
    pub unsafe fn compile_source(device: &CudaDevice, source: &str) -> Result<Self, Refusal> {
        let (module, functions) = compile_group(device, source, &ATTN_KERNELS)?;
        Ok(Self {
            _module: module,
            functions,
        })
    }

    /// Run the fixed problem of [`smoke`] and compare each kernel's output with its recorded digest.
    pub fn qualify(&self, device: &CudaDevice) -> Result<(), Refusal> {
        let outputs = smoke::run(self, device).map_err(|e| refusal("Q3", e.to_string()))?;
        for (i, (bytes, want)) in outputs.iter().zip(smoke::DIGESTS).enumerate() {
            let got = checksum(bytes);
            if got != want {
                return Err(refusal(
                    "Q3",
                    format!(
                        "{} launched but its output digest {got} differs from the recorded {want}",
                        ATTN_KERNELS[i].name
                    ),
                ));
            }
        }
        Ok(())
    }

    /// The loaded function of kernel `name`.
    pub fn function(&self, name: &str) -> Option<&CudaFunction> {
        ATTN_KERNELS
            .iter()
            .position(|k| k.name == name)
            .map(|i| &self.functions[i])
    }

    /// RoPE table rows [0, `n_pos`) into `cs` F32 [n_pos][64]: per position the cosines of the 32
    /// rotated pairs, then their sines, computed as the F32 prefill's RoPE computes them.
    ///
    /// # Safety
    /// `cs` holds `n_pos * 64` F32 values of device memory that stays allocated until the device
    /// stream has run the launch.
    pub unsafe fn rope_table(
        &self,
        device: &CudaDevice,
        cs: u64,
        n_pos: u32,
        theta: f32,
    ) -> Result<(), RuntimeError> {
        let k = ROPE_TABLE;
        let threads = n_pos * (ROTARY_DIM / 2);
        device
            .stream
            .launch_builder(&self.functions[k])
            .arg(&cs)
            .arg(&n_pos)
            .arg(&theta)
            .launch(LaunchConfig {
                grid_dim: (threads.div_ceil(ATTN_KERNELS[k].block), 1, 1),
                block_dim: (ATTN_KERNELS[k].block, 1, 1),
                shared_mem_bytes: 0,
            })
            .map(|_| ())
            .map_err(launch_err(ATTN_KERNELS[k].name))
    }

    /// The norms and RoPE of `t` tokens at positions [p0, p0 + t): `qg` [t][24][512] BF16 (per head
    /// the query, then its gate), `k` and `v` [t][4][256] BF16, `q_w1`/`k_w1` the stored F32 (w + 1)
    /// [256], `cs` the RoPE table. Writes `q_out` and `gate_out` [t][24][256] BF16, and for each KV
    /// head the rows p0..p0 + t of `k_stage`/`v_stage` (BF16) and, unless both are 0 (a BF16
    /// cache, which is then the staging), of `k_cache`/`v_cache` (F32), all [4][max_seq][256];
    /// nothing else.
    ///
    /// # Safety
    /// `t >= 1`, `p0 + t <= max_seq`, `cs` holds at least `p0 + t` rows; `k_cache` and `v_cache` are
    /// both 0 or both not; every other pointer, and those two when not 0, is 16-byte aligned and
    /// addresses device memory of the stated size that stays allocated until the device stream has
    /// run the launch, no buffer the launch writes overlaps another pointer argument, and the launch
    /// is ordered after the work that writes the inputs (the device stream).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn prep(
        &self,
        device: &CudaDevice,
        qg: u64,
        k_in: u64,
        v_in: u64,
        q_w1: u64,
        k_w1: u64,
        cs: u64,
        eps: f32,
        t: u32,
        p0: u32,
        max_seq: u32,
        q_out: u64,
        gate_out: u64,
        k_cache: u64,
        v_cache: u64,
        k_stage: u64,
        v_stage: u64,
    ) -> Result<(), RuntimeError> {
        let k = ATTN_PREP;
        device
            .stream
            .launch_builder(&self.functions[k])
            .arg(&qg)
            .arg(&k_in)
            .arg(&v_in)
            .arg(&q_w1)
            .arg(&k_w1)
            .arg(&cs)
            .arg(&eps)
            .arg(&p0)
            .arg(&max_seq)
            .arg(&q_out)
            .arg(&gate_out)
            .arg(&k_cache)
            .arg(&v_cache)
            .arg(&k_stage)
            .arg(&v_stage)
            .launch(LaunchConfig {
                grid_dim: (t, HEADS + KV_HEADS, 1),
                block_dim: (ATTN_KERNELS[k].block, 1, 1),
                shared_mem_bytes: 0,
            })
            .map(|_| ())
            .map_err(launch_err(ATTN_KERNELS[k].name))
    }

    /// The BF16 staging of cache positions [0, `len`) of every KV head, rounded to nearest even.
    /// Nothing is launched for `len` 0.
    ///
    /// # Safety
    /// `len <= max_seq`; the four buffers are [4][max_seq][256], 16-byte aligned, distinct, allocated
    /// until the device stream has run the launch, and ordered after the work that writes the caches.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn kv_to_bf16(
        &self,
        device: &CudaDevice,
        k_cache: u64,
        v_cache: u64,
        k_stage: u64,
        v_stage: u64,
        len: u32,
        max_seq: u32,
    ) -> Result<(), RuntimeError> {
        if len == 0 {
            return Ok(());
        }
        let k = KV_TO_BF16;
        // One thread per 4 values: 4 heads x 64 threads per position = one block per position.
        device
            .stream
            .launch_builder(&self.functions[k])
            .arg(&k_cache)
            .arg(&v_cache)
            .arg(&k_stage)
            .arg(&v_stage)
            .arg(&len)
            .arg(&max_seq)
            .launch(LaunchConfig {
                grid_dim: (
                    len * KV_HEADS * (HEAD_DIM / 4) / ATTN_KERNELS[k].block,
                    1,
                    1,
                ),
                block_dim: (ATTN_KERNELS[k].block, 1, 1),
                shared_mem_bytes: 0,
            })
            .map(|_| ())
            .map_err(launch_err(ATTN_KERNELS[k].name))
    }

    /// Causal attention of the `t` queries at positions [p0, p0 + t) (`q` [t][24][256] BF16) over the
    /// staged keys and values [0, p0 + t), into `out` [t][24][256] BF16.
    ///
    /// # Safety
    /// `t >= 1`, `p0 + t <= max_seq`, the staging rows [0, p0 + t) are written; every pointer is
    /// 16-byte aligned, allocated until the device stream has run the launch and ordered after the
    /// work that writes it, and `out` overlaps no other pointer argument.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn attention(
        &self,
        device: &CudaDevice,
        q: u64,
        k_stage: u64,
        v_stage: u64,
        out: u64,
        t: u32,
        p0: u32,
        max_seq: u32,
    ) -> Result<(), RuntimeError> {
        let k = ATTN_PREFILL;
        let scale = scale_log2();
        device
            .stream
            .launch_builder(&self.functions[k])
            .arg(&q)
            .arg(&k_stage)
            .arg(&v_stage)
            .arg(&out)
            .arg(&t)
            .arg(&p0)
            .arg(&max_seq)
            .arg(&scale)
            .launch(LaunchConfig {
                grid_dim: (HEADS, t.div_ceil(ATTN_TILE), 1),
                block_dim: (ATTN_KERNELS[k].block, 1, 1),
                shared_mem_bytes: ATTN_KERNELS[k].dynamic_shared,
            })
            .map(|_| ())
            .map_err(launch_err(ATTN_KERNELS[k].name))
    }
}

/// The qualifying problem: 3 tokens continuing 62 cached positions in a cache of 72, through the table,
/// the prep, the staging of the 62 older positions and the attention. The keys span two 64-key tiles
/// (the query at position 64 reads the second), so the second tile's load, masking and accumulation
/// take part; the running maximum does not change there (the GPU suite covers the rescale over many
/// tiles). The older rows are finite F32 values whose low 16 bits cycle over 0x8000 (an exact tie),
/// 0xC35A (above half) and 0x5A5A (below half) under upper halves of both parities, so rounding to
/// nearest even differs from truncation and from rounding half away from zero. Every output is
/// pre-filled with 0xA5 so a byte a kernel fails to write changes its digest. The digests were recorded
/// on an RTX 5090 (NVRTC 13.3, driver 610) from a run whose outputs the GPU suite
/// (`tests/cuda_native_attention_test.rs`) checks against its oracles; the machine code comes from the
/// driver's PTX compiler, so a different NVRTC or driver may compile it differently and is then
/// refused until the digests are recorded for it.
pub mod smoke {
    use super::*;
    use crate::cuda::native_prefill_kernels::smoke::{bf16, weights};
    use cudarc::driver::{CudaSlice, DevicePtr};

    pub const T: u32 = 3;
    pub const P0: u32 = 62;
    pub const MAX_SEQ: u32 = 72;
    pub const THETA: f32 = 1.0e7;
    pub const EPS: f32 = 1e-6;

    /// Expected digests, in [`ATTN_KERNELS`] order: the table; the prep's q, gate, caches and
    /// staging; the staging after the older rows; the attention output.
    pub const DIGESTS: [&str; 4] = [
        "80d03d87fe72eca002abe6a56f17a47ec559513cb2e51bfaf430da667ccf8eeb",
        "0f1fe31776c68d3e9f335523bf01eeae769a281c995eb8d0a8f2ed15945884a0",
        "7a24a964febc1adab3ee617314d3feafabafdd38d1d04721b6edc1d64ee7c5c1",
        "c95a5b7c997a4dbd3e1fc6e59afa13d85c212e93ecab5c29b177f7d13639cbfc",
    ];

    fn ptr<T>(device: &CudaDevice, s: &CudaSlice<T>) -> u64 {
        s.device_ptr(&device.stream).0
    }

    fn bytes_u16(v: &[u16]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn bytes_f32(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    /// The F32 cache before the prep: 0xA5 bytes, rows [0, P0) of every head hold the BF16 values of
    /// `bf16(.., seed)` as upper halves with low halves cycling over 0x8000, 0xC35A and 0x5A5A.
    pub fn cache(seed: u32) -> Vec<f32> {
        let d = HEAD_DIM as usize;
        let mut c = vec![f32::from_bits(0xA5A5_A5A5); KV_HEADS as usize * MAX_SEQ as usize * d];
        let v = bf16(KV_HEADS as usize * P0 as usize * d, seed);
        for hk in 0..KV_HEADS as usize {
            for i in 0..P0 as usize * d {
                let b = v[hk * P0 as usize * d + i];
                let low = [0x8000, 0xC35A, 0x5A5A][i % 3];
                c[hk * MAX_SEQ as usize * d + i] = f32::from_bits(((b as u32) << 16) | low);
            }
        }
        c
    }

    /// Run the problem and return each kernel's output bytes, in [`ATTN_KERNELS`] order.
    pub fn run(k: &NativeAttnKernels, device: &CudaDevice) -> Result<[Vec<u8>; 4], RuntimeError> {
        let (t, d) = (T as usize, HEAD_DIM as usize);
        let (h, hkv) = (HEADS as usize, KV_HEADS as usize);
        let kv = hkv * MAX_SEQ as usize * d;
        let cs = device.htod_copy(&vec![0xA5u8; MAX_SEQ as usize * ROTARY_DIM as usize * 4])?;
        let qg = device.htod_copy(&bf16(t * h * 2 * d, 21))?;
        let kin = device.htod_copy(&bf16(t * hkv * d, 22))?;
        let vin = device.htod_copy(&bf16(t * hkv * d, 23))?;
        let qw = device.htod_copy(&weights(d, 24))?;
        let kw = device.htod_copy(&weights(d, 25))?;
        let q = device.htod_copy(&vec![0xA5A5u16; t * h * d])?;
        let gate = device.htod_copy(&vec![0xA5A5u16; t * h * d])?;
        let kc = device.htod_copy(&cache(26))?;
        let vc = device.htod_copy(&cache(27))?;
        let ks = device.htod_copy(&vec![0xA5A5u16; kv])?;
        let vs = device.htod_copy(&vec![0xA5A5u16; kv])?;
        let out = device.htod_copy(&vec![0xA5A5u16; t * h * d])?;
        let p = |s: &CudaSlice<u16>| ptr(device, s);
        let mut outputs: [Vec<u8>; 4] = Default::default();
        // SAFETY: every buffer is allocated at the size its launch reads or writes (T tokens,
        // positions [0, MAX_SEQ), a table of MAX_SEQ rows), and each read-back follows its launch on
        // the device stream.
        unsafe {
            k.rope_table(device, ptr(device, &cs), MAX_SEQ, THETA)?;
            outputs[ROPE_TABLE] = device.dtoh_copy(&cs)?;
            k.prep(
                device,
                p(&qg),
                p(&kin),
                p(&vin),
                ptr(device, &qw),
                ptr(device, &kw),
                ptr(device, &cs),
                EPS,
                T,
                P0,
                MAX_SEQ,
                p(&q),
                p(&gate),
                ptr(device, &kc),
                ptr(device, &vc),
                p(&ks),
                p(&vs),
            )?;
            let o = &mut outputs[ATTN_PREP];
            o.extend(bytes_u16(&device.dtoh_copy(&q)?));
            o.extend(bytes_u16(&device.dtoh_copy(&gate)?));
            o.extend(bytes_f32(&device.dtoh_copy(&kc)?));
            o.extend(bytes_f32(&device.dtoh_copy(&vc)?));
            o.extend(bytes_u16(&device.dtoh_copy(&ks)?));
            o.extend(bytes_u16(&device.dtoh_copy(&vs)?));
            k.kv_to_bf16(
                device,
                ptr(device, &kc),
                ptr(device, &vc),
                p(&ks),
                p(&vs),
                P0,
                MAX_SEQ,
            )?;
            let o = &mut outputs[KV_TO_BF16];
            o.extend(bytes_u16(&device.dtoh_copy(&ks)?));
            o.extend(bytes_u16(&device.dtoh_copy(&vs)?));
            k.attention(device, p(&q), p(&ks), p(&vs), p(&out), T, P0, MAX_SEQ)?;
            outputs[ATTN_PREFILL] = bytes_u16(&device.dtoh_copy(&out)?);
        }
        Ok(outputs)
    }
}
