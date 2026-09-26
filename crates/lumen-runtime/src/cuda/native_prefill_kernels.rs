//! The native prefill's own NVRTC kernel group: the producers that normalize, gate and quantize
//! activations for its GEMMs (`shaders/native_prefill_*.cu`).
//!
//! The group is compiled as one module for `compute_120a` ([`CudaDevice::fp4_native_arch`]): the
//! FP4 conversion `cvt.rn.satfinite.e2m1x2` exists only on the architecture-specific target. Its
//! sources include no header, so the compile needs nothing but `libnvrtc`. Every kernel has a
//! `native_` name and lives in this module alone, apart from the decode kernels.
//!
//! Loading qualifies the group: the device and NVRTC must allow the target (admission conditions
//! Q1 and Q2), and every kernel must compile and then launch once on a small fixed problem whose
//! output matches a recorded digest (Q3). A group that compiles but does not reproduce those outputs
//! is not served. The GPU suite covers what the small problem cannot (every row count and edge case).

use super::ffi::CudaDevice;
use super::native_prefill::{Refusal, HIDDEN, INTERMEDIATE};
use super::shaders::NATIVE_PREFILL_KERNEL_SOURCE;
use crate::error::RuntimeError;
use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;
use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, LaunchConfig, PushKernelArg};
use std::sync::Arc;

/// Launch metadata of one kernel: its block size and dynamic shared memory, which loading sets as
/// the function's maximum.
#[derive(Clone, Copy, Debug)]
pub struct KernelSpec {
    pub name: &'static str,
    pub block: u32,
    pub dynamic_shared: u32,
}

const fn spec(name: &'static str, block: u32) -> KernelSpec {
    KernelSpec {
        name,
        block,
        dynamic_shared: 0,
    }
}

pub const RMSNORM_FP8: usize = 0;
pub const ADD_RMSNORM_FP8: usize = 1;
pub const ADD_RMSNORM_FP4: usize = 2;
pub const GDN_NORM_GATE_FP8: usize = 3;
pub const FINAL_ROW_F32: usize = 4;
pub const SILU_MUL_FP4_FAST: usize = 5;
pub const SILU_MUL_FP4_PRECISE: usize = 6;
pub const SIGMOID_GATE_FP8: usize = 7;
pub const EMBED_GATHER_BF16: usize = 8;

/// Every kernel of the group.
pub const KERNELS: [KernelSpec; 9] = [
    spec("native_rmsnorm_fp8", 128),
    spec("native_add_rmsnorm_fp8", 128),
    spec("native_add_rmsnorm_fp4", 128),
    spec("native_gdn_norm_gate_fp8", 128),
    spec("native_final_row_f32", 256),
    spec("native_silu_mul_fp4_fast", 256),
    spec("native_silu_mul_fp4_precise", 256),
    spec("native_sigmoid_gate_fp8", 256),
    spec("native_embed_gather_bf16", 256),
];

/// Rows of the GDN gated norm per token: one per value head.
pub const GDN_NORM_ROWS_PER_TOKEN: u32 = 48;
/// Columns of the attention output and its gate.
pub const ATTN_OUT: u32 = 6144;

/// The reduction shape of the GDN gated norm for `rows` rows on a device with `sm_count`
/// multiprocessors: one row per warp (32 lanes) while `rows <= 2 * sm_count`, two rows per warp
/// (16 lanes each) above. The two shapes sum a row's squares in different orders, so the rule fixes
/// the order for `rows` rows on a device of `sm_count` multiprocessors.
pub fn gdn_norm_lanes_per_row(rows: u32, sm_count: u32) -> u32 {
    if rows as u64 <= 2 * sm_count as u64 {
        32
    } else {
        16
    }
}

/// The compiled, qualified kernel group.
pub struct NativePrefillKernels {
    _module: Arc<CudaModule>,
    functions: Vec<CudaFunction>,
    sm_count: u32,
}

fn refusal(condition: &'static str, reason: String) -> Refusal {
    Refusal { condition, reason }
}

fn launch_err(name: &str) -> impl FnOnce(cudarc::driver::DriverError) -> RuntimeError + '_ {
    move |e| RuntimeError::Compute(format!("{name}: {e}"))
}

impl NativePrefillKernels {
    /// Compile the group, set each kernel's attributes and run the qualifying launches.
    pub fn load(device: &CudaDevice) -> Result<Self, Refusal> {
        let kernels = Self::compile(device)?;
        kernels.qualify(device)?;
        Ok(kernels)
    }

    /// Compile the group and set each kernel's attributes, without the qualifying launches: for the
    /// suites that record and check the digests. The route uses [`Self::load`].
    pub fn compile(device: &CudaDevice) -> Result<Self, Refusal> {
        let (module, functions) = compile_group(device, NATIVE_PREFILL_KERNEL_SOURCE, &KERNELS)?;
        let sm_count = device
            .ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .map_err(|e| refusal("Q3", format!("multiprocessor count: {e}")))?
            as u32;
        Ok(Self {
            _module: module,
            functions,
            sm_count,
        })
    }

    /// Launch every kernel once on the fixed problem of [`smoke`] and compare each output with its
    /// recorded digest.
    pub fn qualify(&self, device: &CudaDevice) -> Result<(), Refusal> {
        for (i, want) in smoke::DIGESTS.iter().enumerate() {
            let got = smoke::run(self, device, i)
                .map_err(|e| refusal("Q3", format!("{}: {e}", KERNELS[i].name)))?;
            let got = checksum(&got);
            if got != *want {
                return Err(refusal(
                    "Q3",
                    format!(
                        "{} launched but its output digest {got} differs from the recorded {want}",
                        KERNELS[i].name
                    ),
                ));
            }
        }
        Ok(())
    }

    /// The loaded function of kernel `name`.
    pub fn function(&self, name: &str) -> Option<&CudaFunction> {
        KERNELS
            .iter()
            .position(|k| k.name == name)
            .map(|i| &self.functions[i])
    }

    /// The device's multiprocessor count.
    pub fn sm_count(&self) -> u32 {
        self.sm_count
    }

    fn cfg(kernel: usize, grid: u32) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (KERNELS[kernel].block, 1, 1),
            shared_mem_bytes: KERNELS[kernel].dynamic_shared,
        }
    }

    /// Layer 0's input norm: `x` [m][5120] BF16 (the embedded rows), `w1` the F32 (w + 1); writes the
    /// FP8 codes `q8` [m][5120] and, unless `normed` is 0, the BF16 output [m][5120].
    ///
    /// # Safety
    /// Every pointer is 16-byte aligned and addresses device memory of the stated size that stays
    /// allocated until the device stream has run the launch, no buffer the launch writes overlaps
    /// another pointer argument, and the launch is ordered after the work that writes the inputs
    /// (the device stream).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rmsnorm_fp8(
        &self,
        device: &CudaDevice,
        x: u64,
        w1: u64,
        eps: f32,
        m: u32,
        input_scale: f32,
        q8: u64,
        normed: u64,
    ) -> Result<(), RuntimeError> {
        let k = RMSNORM_FP8;
        device
            .stream
            .launch_builder(&self.functions[k])
            .arg(&x)
            .arg(&w1)
            .arg(&eps)
            .arg(&m)
            .arg(&input_scale)
            .arg(&q8)
            .arg(&normed)
            .launch(Self::cfg(k, m.div_ceil(2)))
            .map(|_| ())
            .map_err(launch_err(KERNELS[k].name))
    }

    /// `resid <- bf16(x + resid)` and the norm of the sum, as [`Self::rmsnorm_fp8`].
    ///
    /// # Safety
    /// As [`Self::rmsnorm_fp8`]; `resid` [m][5120] BF16 is read and written.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn add_rmsnorm_fp8(
        &self,
        device: &CudaDevice,
        x: u64,
        resid: u64,
        w1: u64,
        eps: f32,
        m: u32,
        input_scale: f32,
        q8: u64,
        normed: u64,
    ) -> Result<(), RuntimeError> {
        let k = ADD_RMSNORM_FP8;
        device
            .stream
            .launch_builder(&self.functions[k])
            .arg(&x)
            .arg(&resid)
            .arg(&w1)
            .arg(&eps)
            .arg(&m)
            .arg(&input_scale)
            .arg(&q8)
            .arg(&normed)
            .launch(Self::cfg(k, m.div_ceil(2)))
            .map(|_| ())
            .map_err(launch_err(KERNELS[k].name))
    }

    /// `resid <- bf16(x + resid)` and the norm of the sum to NVFP4 with global scale `s`: codes `q`
    /// [m][2560] and swizzled block scales `sf` ([`fp4_scale_bytes`]`(m, 5120)`).
    ///
    /// # Safety
    /// As [`Self::add_rmsnorm_fp8`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn add_rmsnorm_fp4(
        &self,
        device: &CudaDevice,
        x: u64,
        resid: u64,
        w1: u64,
        eps: f32,
        m: u32,
        s: f32,
        q: u64,
        sf: u64,
    ) -> Result<(), RuntimeError> {
        let k = ADD_RMSNORM_FP4;
        device
            .stream
            .launch_builder(&self.functions[k])
            .arg(&x)
            .arg(&resid)
            .arg(&w1)
            .arg(&eps)
            .arg(&m)
            .arg(&s)
            .arg(&q)
            .arg(&sf)
            .launch(Self::cfg(k, pad128(m) / 2))
            .map(|_| ())
            .map_err(launch_err(KERNELS[k].name))
    }

    /// The GDN gated norm of `rows` rows of 128 (`x` the core output, `z` the gate, both BF16; `w` the
    /// F32 weight [128]) to FP8 codes `q8` [rows][128], with the reduction shape `lanes_per_row`
    /// (32 or 16, [`gdn_norm_lanes_per_row`]).
    ///
    /// # Safety
    /// As [`Self::rmsnorm_fp8`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gdn_norm_gate_fp8(
        &self,
        device: &CudaDevice,
        x: u64,
        z: u64,
        w: u64,
        eps: f32,
        rows: u32,
        lanes_per_row: u32,
        input_scale: f32,
        q8: u64,
    ) -> Result<(), RuntimeError> {
        let k = GDN_NORM_GATE_FP8;
        let rows_per_block = if lanes_per_row == 32 { 4 } else { 8 };
        device
            .stream
            .launch_builder(&self.functions[k])
            .arg(&x)
            .arg(&z)
            .arg(&w)
            .arg(&eps)
            .arg(&rows)
            .arg(&lanes_per_row)
            .arg(&input_scale)
            .arg(&q8)
            .launch(Self::cfg(k, rows.div_ceil(rows_per_block)))
            .map(|_| ())
            .map_err(launch_err(KERNELS[k].name))
    }

    /// `out[i] = f32(x[row][i]) + f32(resid[row][i])` over the hidden size, into `out` F32 [5120].
    ///
    /// # Safety
    /// As [`Self::rmsnorm_fp8`]; `row` is a row of `x` and of `resid`.
    pub unsafe fn final_row_f32(
        &self,
        device: &CudaDevice,
        x: u64,
        resid: u64,
        row: u32,
        out: u64,
    ) -> Result<(), RuntimeError> {
        let k = FINAL_ROW_F32;
        device
            .stream
            .launch_builder(&self.functions[k])
            .arg(&x)
            .arg(&resid)
            .arg(&row)
            .arg(&out)
            .launch(Self::cfg(k, HIDDEN.div_ceil(KERNELS[k].block)))
            .map(|_| ())
            .map_err(launch_err(KERNELS[k].name))
    }

    /// SwiGLU of `gu` [m][2 * 17408] BF16 (gate, then up) to NVFP4 with global scale `s`: codes `q`
    /// [m][8704] and swizzled block scales `sf` ([`fp4_scale_bytes`]`(m, 17408)`). `precise` selects
    /// the IEEE arithmetic of the attention layers' MLPs, otherwise the fast-math arithmetic of the
    /// GDN layers' MLPs.
    ///
    /// # Safety
    /// As [`Self::rmsnorm_fp8`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn silu_mul_fp4(
        &self,
        device: &CudaDevice,
        precise: bool,
        gu: u64,
        m: u32,
        s: f32,
        q: u64,
        sf: u64,
    ) -> Result<(), RuntimeError> {
        let k = if precise {
            SILU_MUL_FP4_PRECISE
        } else {
            SILU_MUL_FP4_FAST
        };
        let threads = pad128(m) * (INTERMEDIATE / 16);
        device
            .stream
            .launch_builder(&self.functions[k])
            .arg(&gu)
            .arg(&m)
            .arg(&s)
            .arg(&q)
            .arg(&sf)
            .launch(Self::cfg(k, threads.div_ceil(KERNELS[k].block)))
            .map(|_| ())
            .map_err(launch_err(KERNELS[k].name))
    }

    /// `bf16(sigmoid(gate) * o)` of the attention output `o` and its gate, both [m][6144] BF16, to FP8
    /// codes `q8` [m][6144].
    ///
    /// # Safety
    /// As [`Self::rmsnorm_fp8`].
    pub unsafe fn sigmoid_gate_fp8(
        &self,
        device: &CudaDevice,
        o: u64,
        gate: u64,
        m: u32,
        input_scale: f32,
        q8: u64,
    ) -> Result<(), RuntimeError> {
        let k = SIGMOID_GATE_FP8;
        let threads = m * (ATTN_OUT / 8);
        device
            .stream
            .launch_builder(&self.functions[k])
            .arg(&o)
            .arg(&gate)
            .arg(&m)
            .arg(&input_scale)
            .arg(&q8)
            .launch(Self::cfg(k, threads.div_ceil(KERNELS[k].block)))
            .map(|_| ())
            .map_err(launch_err(KERNELS[k].name))
    }

    /// The BF16 rows of `table` ([vocab][5120]) for the `m` token ids `ids` (u32, each below the
    /// vocabulary size) into `out` [m][5120].
    ///
    /// # Safety
    /// As [`Self::rmsnorm_fp8`]; every id addresses a row of `table`.
    pub unsafe fn embed_gather_bf16(
        &self,
        device: &CudaDevice,
        table: u64,
        ids: u64,
        m: u32,
        out: u64,
    ) -> Result<(), RuntimeError> {
        let k = EMBED_GATHER_BF16;
        let threads = m * (HIDDEN / 8);
        device
            .stream
            .launch_builder(&self.functions[k])
            .arg(&table)
            .arg(&ids)
            .arg(&m)
            .arg(&out)
            .launch(Self::cfg(k, threads.div_ceil(KERNELS[k].block)))
            .map(|_| ())
            .map_err(launch_err(KERNELS[k].name))
    }
}

/// Compile `source`, a native kernel group, for `compute_120a` under admission conditions Q1 to Q3,
/// load each of `kernels` and set its dynamic shared memory attribute to its table value.
pub(crate) fn compile_group(
    device: &CudaDevice,
    source: &str,
    kernels: &[KernelSpec],
) -> Result<(Arc<CudaModule>, Vec<CudaFunction>), Refusal> {
    let q = |condition: &'static str| move |e: RuntimeError| refusal(condition, e.to_string());
    let cc = device.compute_capability().map_err(q("Q1"))?;
    if cc != (12, 0) {
        return Err(refusal(
            "Q1",
            format!("compute capability {}.{}; the route needs 12.0", cc.0, cc.1),
        ));
    }
    let arch = device.fp4_native_arch().map_err(q("Q2"))?.ok_or_else(|| {
        let v = super::ffi::nvrtc_version()
            .map(|(major, minor)| format!("{major}.{minor}"))
            .unwrap_or_else(|e| e.to_string());
        refusal(
            "Q2",
            format!(
                "NVRTC {v} cannot build compute_120a: it needs 12.8 or newer listing target 120"
            ),
        )
    })?;
    let module = device
        .compile_and_load_with_arch(source, arch)
        .map_err(q("Q3"))?;
    let mut functions = Vec::with_capacity(kernels.len());
    for k in kernels {
        let f = module
            .load_function(k.name)
            .map_err(|e| refusal("Q3", format!("load {}: {e}", k.name)))?;
        f.set_attribute(
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            k.dynamic_shared as i32,
        )
        .map_err(|e| {
            refusal(
                "Q3",
                format!("{} dynamic shared memory {}: {e}", k.name, k.dynamic_shared),
            )
        })?;
        functions.push(f);
    }
    Ok((module, functions))
}

fn pad128(m: u32) -> u32 {
    m.div_ceil(128) * 128
}

/// Bytes of the swizzled block scales of an `[m][k]` NVFP4 activation: whole 128-row tiles.
pub fn fp4_scale_bytes(m: u32, k: u32) -> usize {
    super::native_prefill_weights::swizzled_len(pad128(m) as usize, k as usize / 16)
}

/// The digest [`NativePrefillKernels::qualify`] compares: SHA-256 of the output bytes, in hex.
pub fn checksum(bytes: &[u8]) -> String {
    super::ptx_cache::sha256_hex(bytes)
}

/// The qualifying problem: two tokens through each kernel, inputs from a fixed generator, outputs
/// pre-filled with 0xA5 so a byte a kernel fails to write changes the digest. The first SwiGLU
/// block has gate -88 and up 2^100, where the two arithmetics differ: the fast division returns 0
/// for a divisor above 2^126, the IEEE one does not. The outputs' digests were recorded from a host
/// implementation of the kernels' arithmetic, which the GPU suite
/// (`tests/cuda_native_producers_test.rs`) checks byte for byte against the kernels.
pub mod smoke {
    use super::*;

    pub const EPS: f32 = 1e-6;
    /// FP8 activation scale.
    pub const INPUT_SCALE: f32 = 0.0625;
    /// NVFP4 global scale (1 / activation scale).
    pub const S: f32 = 24.0;
    /// Tokens of the problem.
    pub const ROWS: usize = 2;

    /// Expected digests, in [`KERNELS`] order. The GDN gated norm's covers both reduction shapes,
    /// 32 lanes then 16. The embedding gather's is that of its expected output, [`gathered`], which
    /// a copy reproduces exactly (and a write past the last row does not).
    pub const DIGESTS: [&str; 9] = [
        "da95171dda3f128d992037eece54a49b53d64966e20b0285adfe743ca90f9c8a",
        "a494b6904958bbb89a5037565629fddde4a184f5233ddc7928d5f35e3120e5dd",
        "e1e5a1da5952bc7acda0fa354fc4b8a16bce733726d9146ba258ab8893989159",
        "e8e3db85c29d647552197dfcc5b8c3ad9bd46793b216c5b30b226e37df6b2247",
        "b25f1b2623ce648896e365b84b8018e62518cff392f858c967ade3eb233e4dc1",
        "5f500ca7b94561748f040598f580b6fb7dfe0d8737dd92b0a80c3b8d1a3f88d0",
        "d00c4d33fa2845a5e8fe1e05613ee22e9cfbef3199b9538de17f7b1a78ab2b41",
        "d30a70b3dbb40792247158b69fb4f9f80a92121ead7a1bdbf02942b5cb5cfbc9",
        "90a328c8e491c6b8f80d4933e95a92d014d85f912aa2ba6aaf1b53c70517756f",
    ];

    /// Rows of the gather's table.
    pub const TABLE_ROWS: usize = 5;
    /// Token ids of the gather: an odd count, so its last block of 256 threads is half past the last
    /// row (3 x 640 threads = 7.5 blocks) and a missing bound writes into the guard band. The high
    /// bits of wide ids are checked on the real embedding when the route is published.
    pub const IDS: [u32; 3] = [3, 0, 4];
    /// Sentinel values after the gather's last row, part of its digest.
    pub const GUARD: usize = 256;

    /// The gather's table [TABLE_ROWS][5120].
    pub fn table() -> Vec<u16> {
        bf16(TABLE_ROWS * HIDDEN as usize, 12)
    }

    /// The bytes the gather's output must hold: the rows of [`table`] named by [`IDS`], in order,
    /// then the [`GUARD`] sentinel values it must leave.
    pub fn gathered() -> Vec<u8> {
        let h = HIDDEN as usize;
        let table = table();
        let mut out: Vec<u8> = IDS
            .iter()
            .flat_map(|&id| bytes_u16(&table[id as usize * h..(id as usize + 1) * h]))
            .collect();
        out.resize(out.len() + 2 * GUARD, 0xA5);
        out
    }

    /// `n` BF16 values in [-4, 4) from a linear congruential sequence seeded with `seed`.
    pub fn bf16(n: usize, seed: u32) -> Vec<u16> {
        let mut state = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                let v = (state >> 8) as f32 / (1u32 << 24) as f32 * 8.0 - 4.0;
                (v.to_bits() >> 16) as u16
            })
            .collect()
    }

    /// `n` F32 norm weights in [0.5, 1.5).
    pub fn weights(n: usize, seed: u32) -> Vec<f32> {
        bf16(n, seed)
            .into_iter()
            .map(|b| 1.0 + f32::from_bits((b as u32) << 16) / 8.0)
            .collect()
    }

    /// The SwiGLU input [ROWS][2 * 17408]: generated, then gate -88 (0xC2B0) and up 2^100 (0x7180) over
    /// the first 16 columns of row 0.
    pub fn gu() -> Vec<u16> {
        let i = INTERMEDIATE as usize;
        let mut v = bf16(ROWS * 2 * i, 9);
        for c in 0..16 {
            v[c] = 0xC2B0;
            v[i + c] = 0x7180;
        }
        v
    }

    fn bytes_u16(v: &[u16]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn up<T: cudarc::driver::DeviceRepr>(
        device: &CudaDevice,
        v: &[T],
    ) -> Result<CudaSlice<T>, RuntimeError> {
        device.htod_copy(v)
    }

    /// `bytes` bytes of 0xA5.
    fn sentinel(device: &CudaDevice, bytes: usize) -> Result<CudaSlice<u8>, RuntimeError> {
        device.htod_copy(&vec![0xA5u8; bytes])
    }

    fn ptr<T>(device: &CudaDevice, s: &CudaSlice<T>) -> u64 {
        use cudarc::driver::DevicePtr;
        s.device_ptr(&device.stream).0
    }

    /// Run kernel `kernel` of [`KERNELS`] on its qualifying problem and return its output bytes.
    pub fn run(
        k: &NativePrefillKernels,
        device: &CudaDevice,
        kernel: usize,
    ) -> Result<Vec<u8>, RuntimeError> {
        let (m, mu) = (ROWS, ROWS as u32);
        let h = HIDDEN as usize;
        let i = INTERMEDIATE as usize;
        let w1 = up(device, &weights(h, 2))?;
        let mut out = Vec::new();
        // SAFETY: every buffer below is allocated at the size its launch reads or writes, and each
        // read-back follows its launch on the device stream.
        unsafe {
            match kernel {
                RMSNORM_FP8 | ADD_RMSNORM_FP8 => {
                    let x = up(device, &bf16(m * h, 1))?;
                    let q8 = sentinel(device, m * h)?;
                    let normed = sentinel(device, 2 * m * h)?;
                    let (xp, wp, qp, np) = (
                        ptr(device, &x),
                        ptr(device, &w1),
                        ptr(device, &q8),
                        ptr(device, &normed),
                    );
                    if kernel == RMSNORM_FP8 {
                        k.rmsnorm_fp8(device, xp, wp, EPS, mu, INPUT_SCALE, qp, np)?;
                    } else {
                        let resid = up(device, &bf16(m * h, 3))?;
                        let rp = ptr(device, &resid);
                        k.add_rmsnorm_fp8(device, xp, rp, wp, EPS, mu, INPUT_SCALE, qp, np)?;
                        out.extend(bytes_u16(&device.dtoh_copy(&resid)?));
                    }
                    out.extend(device.dtoh_copy(&q8)?);
                    out.extend(device.dtoh_copy(&normed)?);
                }
                ADD_RMSNORM_FP4 => {
                    let x = up(device, &bf16(m * h, 1))?;
                    let resid = up(device, &bf16(m * h, 3))?;
                    let q = sentinel(device, m * h / 2)?;
                    let sf = sentinel(device, fp4_scale_bytes(mu, HIDDEN))?;
                    k.add_rmsnorm_fp4(
                        device,
                        ptr(device, &x),
                        ptr(device, &resid),
                        ptr(device, &w1),
                        EPS,
                        mu,
                        S,
                        ptr(device, &q),
                        ptr(device, &sf),
                    )?;
                    out.extend(bytes_u16(&device.dtoh_copy(&resid)?));
                    out.extend(device.dtoh_copy(&q)?);
                    out.extend(device.dtoh_copy(&sf)?);
                }
                GDN_NORM_GATE_FP8 => {
                    let rows = m * GDN_NORM_ROWS_PER_TOKEN as usize;
                    let x = up(device, &bf16(rows * 128, 4))?;
                    let z = up(device, &bf16(rows * 128, 5))?;
                    let w = up(device, &weights(128, 6))?;
                    for lanes in [32, 16] {
                        let q8 = sentinel(device, rows * 128)?;
                        k.gdn_norm_gate_fp8(
                            device,
                            ptr(device, &x),
                            ptr(device, &z),
                            ptr(device, &w),
                            EPS,
                            rows as u32,
                            lanes,
                            INPUT_SCALE,
                            ptr(device, &q8),
                        )?;
                        out.extend(device.dtoh_copy(&q8)?);
                    }
                }
                FINAL_ROW_F32 => {
                    let x = up(device, &bf16(m * h, 7))?;
                    let resid = up(device, &bf16(m * h, 8))?;
                    let o = sentinel(device, 4 * h)?;
                    k.final_row_f32(
                        device,
                        ptr(device, &x),
                        ptr(device, &resid),
                        mu - 1,
                        ptr(device, &o),
                    )?;
                    out.extend(device.dtoh_copy(&o)?);
                }
                SILU_MUL_FP4_FAST | SILU_MUL_FP4_PRECISE => {
                    let gu = up(device, &gu())?;
                    let q = sentinel(device, m * i / 2)?;
                    let sf = sentinel(device, fp4_scale_bytes(mu, INTERMEDIATE))?;
                    k.silu_mul_fp4(
                        device,
                        kernel == SILU_MUL_FP4_PRECISE,
                        ptr(device, &gu),
                        mu,
                        S,
                        ptr(device, &q),
                        ptr(device, &sf),
                    )?;
                    out.extend(device.dtoh_copy(&q)?);
                    out.extend(device.dtoh_copy(&sf)?);
                }
                SIGMOID_GATE_FP8 => {
                    let n = m * ATTN_OUT as usize;
                    let o = up(device, &bf16(n, 10))?;
                    let g = up(device, &bf16(n, 11))?;
                    let q8 = sentinel(device, n)?;
                    k.sigmoid_gate_fp8(
                        device,
                        ptr(device, &o),
                        ptr(device, &g),
                        mu,
                        INPUT_SCALE,
                        ptr(device, &q8),
                    )?;
                    out.extend(device.dtoh_copy(&q8)?);
                }
                EMBED_GATHER_BF16 => {
                    let table = up(device, &table())?;
                    let ids = up(device, &IDS)?;
                    let o = sentinel(device, 2 * (IDS.len() * h + GUARD))?;
                    k.embed_gather_bf16(
                        device,
                        ptr(device, &table),
                        ptr(device, &ids),
                        IDS.len() as u32,
                        ptr(device, &o),
                    )?;
                    out.extend(device.dtoh_copy(&o)?);
                }
                _ => unreachable!("kernel {kernel} is not in KERNELS"),
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_gather_digest_is_that_of_its_expected_rows() {
        let want = smoke::gathered();
        assert_eq!(
            want.len(),
            2 * (smoke::IDS.len() * HIDDEN as usize + smoke::GUARD)
        );
        assert_eq!(checksum(&want), smoke::DIGESTS[EMBED_GATHER_BF16]);
    }

    #[test]
    fn the_gather_problem_tells_a_write_past_the_last_row_apart() {
        let want = smoke::gathered();
        assert_eq!(
            (smoke::IDS.len() * HIDDEN as usize / 8) % 256,
            128,
            "the last block is half past the last row"
        );
        let mut stray = want.clone();
        stray[2 * smoke::IDS.len() * HIDDEN as usize] = 0;
        assert_ne!(checksum(&stray), smoke::DIGESTS[EMBED_GATHER_BF16]);
    }
}
