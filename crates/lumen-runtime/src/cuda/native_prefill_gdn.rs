//! The native prefill's GDN kernels (`shaders/native_prefill_gdn.cu`): the causal conv1d of every
//! token at once, and the gated delta rule over 64-token chunks on BF16 tensor cores.
//!
//! One GDN layer is three launches in order, [`NativeGdnKernels::conv`],
//! [`NativeGdnKernels::chunk_intra`] and [`NativeGdnKernels::chunk_state`]. They update the layer's
//! state where decode keeps it, in place and in decode's layouts: the conv ring `[3][10240]` F32,
//! whose slot `(state_pos + s) % 3` holds token `s`, and the recurrent state `[48][128][128]` F32,
//! `[head][value][key]`. After the three launches the layer's ring position is
//! [`next_conv_position`].
//!
//! The kernels compile as their own NVRTC module, with no header and for the producers' target
//! ([`CudaDevice::fp4_native_arch`]), under `native_gdn_` names that no other module uses, so the
//! decode route cannot reach them. Loading qualifies the module the way the producers are qualified
//! ([`super::native_prefill_kernels`]): a fixed problem through the three kernels must reproduce
//! recorded output digests. The GPU suite (`tests/cuda_native_gdn_test.rs`) checks the kernels
//! against an f64 model of the recurrence.

use super::ffi::CudaDevice;
use super::native_prefill::Refusal;
use super::native_prefill_kernels::{checksum, KernelSpec};
use super::shaders::NATIVE_PREFILL_GDN_KERNEL_SOURCE;
use crate::error::RuntimeError;
use cudarc::driver::sys::CUfunction_attribute::{
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
    CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
};
use cudarc::driver::{CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
use std::sync::Arc;

/// Value heads.
pub const HEADS: u32 = 48;
/// Channels of the conv: q 2048, k 2048, v 6144.
pub const CONV_DIM: u32 = 10240;
/// Head size of keys and values.
pub const HEAD_DIM: u32 = 128;
/// Tokens per chunk.
pub const CHUNK: u32 = 64;
/// Slots of the conv ring (conv kernel 4).
pub const RING_SLOTS: u32 = 3;

pub const CONV_SILU_L2: usize = 0;
pub const CHUNK_INTRA: usize = 1;
pub const CHUNK_STATE: usize = 2;

/// The kernels; their dynamic shared memory is the size of each kernel's shared-memory struct,
/// which the source asserts.
pub const KERNELS: [KernelSpec; 3] = [
    KernelSpec {
        name: "native_gdn_conv_silu_l2",
        block: 128,
        dynamic_shared: 0,
    },
    KernelSpec {
        name: "native_gdn_chunk_intra",
        block: 128,
        dynamic_shared: 44_800,
    },
    KernelSpec {
        name: "native_gdn_chunk_state_bv48",
        block: 128,
        dynamic_shared: 96_256,
    },
];

/// The conv ring position after a prefill of `t` tokens from position `state_pos`: decode's rule.
pub fn next_conv_position(state_pos: u32, t: u32) -> u32 {
    (state_pos + t) % RING_SLOTS
}

/// The compiled, qualified GDN kernels.
pub struct NativeGdnKernels {
    _module: Arc<CudaModule>,
    functions: Vec<CudaFunction>,
}

fn refusal(condition: &'static str, reason: String) -> Refusal {
    Refusal { condition, reason }
}

impl NativeGdnKernels {
    /// Compile the module, set each kernel's attributes and run the qualifying launches.
    pub fn load(device: &CudaDevice) -> Result<Self, Refusal> {
        // SAFETY: the group's own source.
        let kernels = unsafe { Self::compile_source(device, NATIVE_PREFILL_GDN_KERNEL_SOURCE) }?;
        kernels.qualify(device)?;
        Ok(kernels)
    }

    /// Compile `source`, which defines [`KERNELS`], and set each kernel's attributes, without the
    /// qualifying launches: for the suite, which also compiles altered sources as negative controls.
    /// The route uses [`Self::load`].
    ///
    /// # Safety
    /// [`Self::qualify`] and the launchers run these kernels as the group's own: `source` must define
    /// every kernel in [`KERNELS`] with the group's parameters, and each must access only the memory
    /// those launches give it.
    pub unsafe fn compile_source(device: &CudaDevice, source: &str) -> Result<Self, Refusal> {
        let arch = device
            .fp4_native_arch()
            .map_err(|e| refusal("Q2", e.to_string()))?
            .ok_or_else(|| {
                refusal(
                    "Q2",
                    "the device or NVRTC cannot build compute_120a".to_string(),
                )
            })?;
        let module = device
            .compile_and_load_with_arch(source, arch)
            .map_err(|e| refusal("Q3", e.to_string()))?;
        let mut functions = Vec::with_capacity(KERNELS.len());
        for k in &KERNELS {
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
        // Two in-chunk blocks per multiprocessor need the largest shared-memory carveout.
        functions[CHUNK_INTRA]
            .set_attribute(CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 100)
            .map_err(|e| {
                refusal(
                    "Q3",
                    format!("{} shared memory carveout: {e}", KERNELS[CHUNK_INTRA].name),
                )
            })?;
        Ok(Self {
            _module: module,
            functions,
        })
    }

    /// Run [`smoke`]'s problem and compare each kernel's output with its recorded digest.
    pub fn qualify(&self, device: &CudaDevice) -> Result<(), Refusal> {
        let outputs = smoke::run(self, device).map_err(|e| refusal("Q3", e.to_string()))?;
        for (i, (out, want)) in outputs.iter().zip(smoke::DIGESTS).enumerate() {
            let got = checksum(out);
            if got != want {
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

    fn launch_err(kernel: usize) -> impl FnOnce(cudarc::driver::DriverError) -> RuntimeError {
        move |e| RuntimeError::Compute(format!("{}: {e}", KERNELS[kernel].name))
    }

    fn cfg(kernel: usize, grid: (u32, u32)) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (grid.0, grid.1, 1),
            block_dim: (KERNELS[kernel].block, 1, 1),
            shared_mem_bytes: KERNELS[kernel].dynamic_shared,
        }
    }

    /// Conv + SiLU of `t` tokens of `qkv` [t][10240] BF16 with `conv_w` [10240][4] F32, the q/k
    /// heads L2-normalized: `cv` [t][10240] BF16. `ring` [3][10240] F32 at position `state_pos` is
    /// read and then holds the last three tokens' inputs.
    ///
    /// # Safety
    /// `t >= 1` and `state_pos < 3`; every pointer is 16-byte aligned and addresses device memory of
    /// the stated size that stays allocated until the device stream has run the launch, no buffer the
    /// launch writes overlaps another pointer argument, and the launch is ordered after the work that
    /// writes the inputs (the device stream).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn conv(
        &self,
        device: &CudaDevice,
        qkv: u64,
        ring: u64,
        conv_w: u64,
        cv: u64,
        t: u32,
        state_pos: u32,
    ) -> Result<(), RuntimeError> {
        let k = CONV_SILU_L2;
        let (t, state_pos) = (t as i32, state_pos as i32);
        device
            .stream
            .launch_builder(&self.functions[k])
            .arg(&qkv)
            .arg(&ring)
            .arg(&conv_w)
            .arg(&cv)
            .arg(&t)
            .arg(&state_pos)
            .launch(Self::cfg(k, (CONV_DIM / HEAD_DIM, (t as u32).div_ceil(32))))
            .map(|_| ())
            .map_err(Self::launch_err(k))
    }

    /// The in-chunk products of `t` tokens: from `cv`, the `a` and `b` rows `ab` [t][96] F32 (a in
    /// columns 0-47), `dt_bias` and `ssm_a` [48] F32, writes the gate sums `gc` [t][48] F32 and `w`,
    /// `u` [t][48][128] and `aqk` [t][48][64] BF16.
    ///
    /// # Safety
    /// `t >= 1`, and every pointer as [`Self::conv`] requires.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn chunk_intra(
        &self,
        device: &CudaDevice,
        cv: u64,
        ab: u64,
        dt_bias: u64,
        ssm_a: u64,
        gc: u64,
        w: u64,
        u: u64,
        aqk: u64,
        t: u32,
    ) -> Result<(), RuntimeError> {
        let k = CHUNK_INTRA;
        let t = t as i32;
        device
            .stream
            .launch_builder(&self.functions[k])
            .arg(&cv)
            .arg(&ab)
            .arg(&dt_bias)
            .arg(&ssm_a)
            .arg(&gc)
            .arg(&w)
            .arg(&u)
            .arg(&aqk)
            .arg(&t)
            .launch(Self::cfg(k, ((t as u32).div_ceil(CHUNK), HEADS)))
            .map(|_| ())
            .map_err(Self::launch_err(k))
    }

    /// The recurrence over the chunks of `t` tokens and the layer output `core` [t][48][128]
    /// BF16, from [`Self::chunk_intra`]'s outputs and `cv`; `state` [48][128][128] F32 is read and
    /// then holds the state after the last token.
    ///
    /// # Safety
    /// `t >= 1`, and every pointer as [`Self::conv`] requires.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn chunk_state(
        &self,
        device: &CudaDevice,
        cv: u64,
        gc: u64,
        w: u64,
        u: u64,
        aqk: u64,
        state: u64,
        core: u64,
        t: u32,
    ) -> Result<(), RuntimeError> {
        let k = CHUNK_STATE;
        let t = t as i32;
        device
            .stream
            .launch_builder(&self.functions[k])
            .arg(&cv)
            .arg(&gc)
            .arg(&w)
            .arg(&u)
            .arg(&aqk)
            .arg(&state)
            .arg(&core)
            .arg(&t)
            .launch(Self::cfg(k, (HEAD_DIM.div_ceil(48), HEADS)))
            .map(|_| ())
            .map_err(Self::launch_err(k))
    }
}

/// The qualifying problem: one layer of 67 tokens (a whole chunk and three tokens more) from ring
/// position 2 and a nonzero state, inputs from the producers' generator (a and b off the BF16 grid,
/// gates small enough that every head carries its state across the chunk boundary), every output
/// pre-filled with 0xA5 so an unwritten byte changes the digest. The digests were recorded from the
/// kernels on an RTX 5090 (NVRTC 13.3, driver 610), after the GPU suite had checked the same outputs
/// against its f64 models; the machine code comes from the driver's PTX compiler (from NVRTC when it
/// is a newer minor version than the driver), so a different NVRTC or driver may compile it
/// differently and is then refused until the digests are recorded for it.
pub mod smoke {
    use super::*;
    use crate::cuda::native_prefill_kernels::smoke::bf16;
    use cudarc::driver::{CudaSlice, DevicePtr};

    /// Tokens of the problem.
    pub const T: usize = 67;
    /// Ring position before it.
    pub const STATE_POS: u32 = 2;

    /// Expected digests of the conv (output, then the ring), the in-chunk kernel (G, W, U, Aqk) and
    /// the state kernel (output, then the state).
    pub const DIGESTS: [&str; 3] = [
        "0bcb2293d8a3fbdbf3179cbc097272975878bf4d15a45c046a880d6509dd6f41",
        "66e5d95ebb411e89dbbc784baa3e8526524eefb02d67bf93199e13142840ac70",
        "971a5d56841fb1f691867e77cf332dda36c6075038a24ff179ca8ef302ed0365",
    ];

    fn widen(v: Vec<u16>, scale: f32) -> Vec<f32> {
        v.into_iter()
            .map(|b| f32::from_bits((b as u32) << 16) * scale)
            .collect()
    }

    /// `qkv` [T][10240] BF16 in [-4, 4).
    pub fn qkv() -> Vec<u16> {
        bf16(T * CONV_DIM as usize, 21)
    }

    /// The ring before the problem [3][10240], BF16 values in [-4, 4).
    pub fn ring() -> Vec<f32> {
        widen(bf16((RING_SLOTS * CONV_DIM) as usize, 22), 1.0)
    }

    /// Conv weights [10240][4] in [-0.5, 0.5).
    pub fn conv_w() -> Vec<f32> {
        widen(bf16(CONV_DIM as usize * 4, 23), 0.125)
    }

    /// `a` and `b` rows [T][96] in (-4.004, 4.004), BF16 values times 1 + 2^-10, so their rounding
    /// to BF16 is not the identity.
    pub fn ab() -> Vec<f32> {
        widen(bf16(T * 2 * HEADS as usize, 24), 1.0 + 1.0 / 1024.0)
    }

    /// `dt_bias` [48] in [-1, 1).
    pub fn dt_bias() -> Vec<f32> {
        widen(bf16(HEADS as usize, 25), 0.25)
    }

    /// `ssm_a` [48] in (-0.13, -0.005].
    pub fn ssm_a() -> Vec<f32> {
        widen(bf16(HEADS as usize, 26), 1.0 / 32.0)
            .into_iter()
            .map(|v| -(v.abs() + 0.005))
            .collect()
    }

    /// The state before the problem [48][128][128] in [-0.5, 0.5).
    pub fn state() -> Vec<f32> {
        widen(bf16((HEADS * HEAD_DIM * HEAD_DIM) as usize, 27), 0.125)
    }

    /// `bytes` bytes of 0xA5.
    fn sentinel(device: &CudaDevice, bytes: usize) -> Result<CudaSlice<u8>, RuntimeError> {
        device.htod_copy(&vec![0xA5u8; bytes])
    }

    fn ptr<T>(device: &CudaDevice, s: &CudaSlice<T>) -> u64 {
        s.device_ptr(&device.stream).0
    }

    /// Run the problem through the three kernels and return each one's output bytes, as
    /// [`DIGESTS`] orders them.
    pub fn run(k: &NativeGdnKernels, device: &CudaDevice) -> Result<[Vec<u8>; 3], RuntimeError> {
        let (h, d, conv) = (HEADS as usize, HEAD_DIM as usize, CONV_DIM as usize);
        let qkv = device.htod_copy(&qkv())?;
        let ring = device.htod_copy(&ring())?;
        let conv_w = device.htod_copy(&conv_w())?;
        let ab = device.htod_copy(&ab())?;
        let dt_bias = device.htod_copy(&dt_bias())?;
        let ssm_a = device.htod_copy(&ssm_a())?;
        let state = device.htod_copy(&state())?;
        let cv = sentinel(device, 2 * T * conv)?;
        let gc = sentinel(device, 4 * T * h)?;
        let w = sentinel(device, 2 * T * h * d)?;
        let u = sentinel(device, 2 * T * h * d)?;
        let aqk = sentinel(device, 2 * T * h * CHUNK as usize)?;
        let core = sentinel(device, 2 * T * h * d)?;
        let tu = T as u32;
        // SAFETY: every buffer is allocated at the size its launch reads or writes, the outputs
        // overlap no input, and the read-backs follow the launches on the device stream.
        unsafe {
            k.conv(
                device,
                ptr(device, &qkv),
                ptr(device, &ring),
                ptr(device, &conv_w),
                ptr(device, &cv),
                tu,
                STATE_POS,
            )?;
            k.chunk_intra(
                device,
                ptr(device, &cv),
                ptr(device, &ab),
                ptr(device, &dt_bias),
                ptr(device, &ssm_a),
                ptr(device, &gc),
                ptr(device, &w),
                ptr(device, &u),
                ptr(device, &aqk),
                tu,
            )?;
            k.chunk_state(
                device,
                ptr(device, &cv),
                ptr(device, &gc),
                ptr(device, &w),
                ptr(device, &u),
                ptr(device, &aqk),
                ptr(device, &state),
                ptr(device, &core),
                tu,
            )?;
        }
        let bytes = |v: Vec<f32>| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let mut conv_out = device.dtoh_copy(&cv)?;
        conv_out.extend(bytes(device.dtoh_copy(&ring)?));
        let mut intra = device.dtoh_copy(&gc)?;
        intra.extend(device.dtoh_copy(&w)?);
        intra.extend(device.dtoh_copy(&u)?);
        intra.extend(device.dtoh_copy(&aqk)?);
        let mut state_out = device.dtoh_copy(&core)?;
        state_out.extend(bytes(device.dtoh_copy(&state)?));
        Ok([conv_out, intra, state_out])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::shaders;

    /// The `extern "C"` kernel names a CUDA source defines.
    fn entry_points(source: &str) -> Vec<String> {
        source
            .split("extern \"C\"")
            .skip(1)
            .filter_map(|rest| {
                let rest = match rest.find("__launch_bounds__(") {
                    Some(i) if i < rest.find('(')? => &rest[i + rest[i..].find(')')? + 1..],
                    _ => rest,
                };
                let head = &rest[..rest.find('(')?];
                head.split_whitespace().last().map(str::to_string)
            })
            .collect()
    }

    /// Every other module's kernel sources, including the producers'.
    fn other_sources() -> Vec<&'static str> {
        vec![
            shaders::GDN_KERNEL_SOURCE,
            shaders::GDN_MEGAKERNEL_SOURCE,
            shaders::GDN_REGISTER_RESIDENT_KERNEL_SOURCE,
            shaders::GDN_F64ACCUM_KERNEL_SOURCE,
            shaders::GDN_INPUT_PROJECTIONS_KERNEL_SOURCE,
            shaders::NATIVE_PREFILL_KERNEL_SOURCE,
            shaders::NATIVE_PREFILL_ATTN_KERNEL_SOURCE,
        ]
    }

    #[test]
    fn the_module_defines_exactly_its_kernels_under_names_no_other_module_uses() {
        let names = entry_points(NATIVE_PREFILL_GDN_KERNEL_SOURCE);
        let table: Vec<String> = KERNELS.iter().map(|k| k.name.to_string()).collect();
        assert_eq!(names, table);
        for name in &names {
            assert!(name.starts_with("native_gdn_"), "{name}");
            for other in other_sources() {
                assert!(
                    !other.contains(name.as_str()),
                    "{name} appears in another module"
                );
            }
        }
    }
}
