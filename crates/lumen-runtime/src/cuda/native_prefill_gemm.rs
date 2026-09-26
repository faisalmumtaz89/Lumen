//! The native prefill's GEMM plan table: every quantized projection shape of the admitted model at
//! every row bucket, each a cuBLASLt plan with a measured, verified algorithm
//! (`cublaslt_algo_cache`), and the per-call launch.
//!
//! A prefill of `T` rows runs each GEMM at `T` rounded up to a multiple of [`ROW_STEP`]; the rows
//! past `T` of every activation are zero, and GEMM rows are independent, so the padding changes no
//! row below `T`.
//!
//! `NativeGemm` owns its handle, plans and workspace, and is used only on the backend's single
//! stream. Dropping it first waits for that stream, so no enqueued GEMM still reads the workspace,
//! then destroys the plans, then the handle, then frees the workspace.

use super::cublaslt::{construction_step, LtGemmShape, LtHandle, LtInput, LtMatmul, LtOperands};
use super::cublaslt_algo_cache::{
    self as algo_cache, Identity, PlanSelection, Selected, WeightOperand,
};
use super::ffi::CudaDevice;
use crate::error::RuntimeError;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

/// One projection shape: `D[T][n] (row stride ldd) = X[T][k] * W[n][k]^T`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemmShape {
    pub name: &'static str,
    pub input: LtInput,
    pub n: usize,
    pub k: usize,
    pub ldd: usize,
}

impl GemmShape {
    /// The cuBLASLt shape at `rows` rows.
    pub fn lt(&self, rows: usize) -> LtGemmShape {
        LtGemmShape {
            input: self.input,
            n: self.n as u64,
            k: self.k as u64,
            m: rows as u64,
            ldd: self.ldd as u64,
        }
    }
}

pub const GATE_UP: usize = 0;
pub const DOWN: usize = 1;
pub const GDN_QKV: usize = 2;
pub const GDN_Z: usize = 3;
pub const OUT: usize = 4;
pub const ATTN_Q: usize = 5;
pub const ATTN_KV: usize = 6;

/// The admitted model's projections. Gate and up write the two halves of one `[T][2 * 17408]`
/// buffer; the GDN output projection and the attention output projection share a shape.
pub const SHAPES: [GemmShape; 7] = [
    GemmShape {
        name: "gate_up",
        input: LtInput::Nvfp4,
        n: 17408,
        k: 5120,
        ldd: 2 * 17408,
    },
    GemmShape {
        name: "down",
        input: LtInput::Nvfp4,
        n: 5120,
        k: 17408,
        ldd: 5120,
    },
    GemmShape {
        name: "gdn_qkv",
        input: LtInput::Fp8,
        n: 10240,
        k: 5120,
        ldd: 10240,
    },
    GemmShape {
        name: "gdn_z",
        input: LtInput::Fp8,
        n: 6144,
        k: 5120,
        ldd: 6144,
    },
    GemmShape {
        name: "out",
        input: LtInput::Fp8,
        n: 5120,
        k: 6144,
        ldd: 5120,
    },
    GemmShape {
        name: "attn_q",
        input: LtInput::Fp8,
        n: 12288,
        k: 5120,
        ldd: 12288,
    },
    GemmShape {
        name: "attn_kv",
        input: LtInput::Fp8,
        n: 1024,
        k: 5120,
        ldd: 1024,
    },
];

// The block-scaled layout of the NVFP4 weights has no partial tile: rows a multiple of 128 and
// scale columns (k / 16) a multiple of 4.
const _: () = {
    let mut i = 0;
    while i < SHAPES.len() {
        let s = SHAPES[i];
        if let LtInput::Nvfp4 = s.input {
            assert!(s.n % 128 == 0 && (s.k / 16) % 4 == 0 && s.k % 16 == 0);
        }
        i += 1;
    }
};

pub const ROW_STEP: usize = 16;
pub const MAX_ROWS: usize = 2048;
pub const BUCKETS: usize = MAX_ROWS / ROW_STEP;
/// cuBLASLt workspace.
pub const WORKSPACE_BYTES: usize = 32 << 20;

/// Rows a GEMM runs at for bucket `b`.
pub fn bucket_rows(b: usize) -> usize {
    (b + 1) * ROW_STEP
}

/// The bucket of a prefill of `rows` rows, `None` outside `1..=MAX_ROWS`.
pub fn bucket(rows: usize) -> Option<usize> {
    (1..=MAX_ROWS)
        .contains(&rows)
        .then(|| (rows - 1) / ROW_STEP)
}

/// How a build got its table.
#[derive(Debug)]
pub enum TableSource {
    /// Read from the cache file.
    Cached(PathBuf),
    /// Selected now, because the cache was not usable for the reason given; written to the path
    /// when it could be.
    Selected {
        reason: String,
        stored: Option<PathBuf>,
        plans: Vec<PlanSelection>,
    },
}

/// What a build did and how long it took.
#[derive(Debug)]
pub struct BuildReport {
    pub source: TableSource,
    pub seconds: f64,
    pub library: String,
}

struct Plan {
    matmul: LtMatmul,
    selected: Selected,
}

/// The plan table and what it runs with.
pub struct NativeGemm {
    plans: Vec<Plan>,
    handle: LtHandle,
    workspace: CudaSlice<u8>,
    stream: Arc<CudaStream>,
}

impl NativeGemm {
    /// Build every plan and give each its algorithm: from the cache when it holds a table for this
    /// library and device, otherwise by selection over `weights` (per shape, in [`SHAPES`] order,
    /// the copies the timing rotates over; the first is also the one verified).
    ///
    /// # Safety
    /// Every address in `weights` must be a live device operand of its shape as [`WeightOperand`]
    /// describes; they are read only when the cache cannot be used.
    pub unsafe fn build(
        device: &CudaDevice,
        weights: &[Vec<WeightOperand>; SHAPES.len()],
    ) -> Result<(Self, BuildReport), RuntimeError> {
        let t0 = Instant::now();
        let handle = LtHandle::new(&device.ctx)?;
        construction_step("workspace allocation")?;
        let workspace = device.alloc_zeros::<u8>(WORKSPACE_BYTES)?;
        let mut matmuls = Vec::with_capacity(SHAPES.len() * BUCKETS);
        for shape in &SHAPES {
            for b in 0..BUCKETS {
                matmuls.push(LtMatmul::new(&handle, shape.lt(bucket_rows(b)))?);
            }
        }
        let lib = handle.library();
        let identity = Identity::new(device, lib, WORKSPACE_BYTES)?;
        let (table, source) = match algo_cache::load(&identity, lib)? {
            Ok((table, path)) => (table, TableSource::Cached(path)),
            Err(reason) => {
                let ws = workspace.device_ptr(&device.stream).0;
                let mut plans = Vec::with_capacity(matmuls.len());
                for (s, shape) in SHAPES.iter().enumerate() {
                    plans.extend(algo_cache::select_shape(
                        device,
                        &handle,
                        shape,
                        &mut matmuls[s * BUCKETS..(s + 1) * BUCKETS],
                        &weights[s],
                        ws,
                        WORKSPACE_BYTES,
                    )?);
                }
                let table: Vec<Selected> = plans.iter().map(|p| p.selected).collect();
                let stored = algo_cache::store(&identity, &table);
                (
                    table,
                    TableSource::Selected {
                        reason,
                        stored,
                        plans,
                    },
                )
            }
        };
        let plans = matmuls
            .into_iter()
            .zip(table)
            .map(|(matmul, selected)| Plan { matmul, selected })
            .collect();
        Ok((
            Self {
                plans,
                handle,
                workspace,
                stream: device.stream.clone(),
            },
            BuildReport {
                source,
                seconds: t0.elapsed().as_secs_f64(),
                library: lib.describe(),
            },
        ))
    }

    /// The selected algorithm of `shape` (an index into [`SHAPES`]) at bucket `b`.
    pub fn selected(&self, shape: usize, b: usize) -> &Selected {
        &self.plans[shape * BUCKETS + b].selected
    }

    /// Enqueue `shape` for a prefill of `rows` rows (run at its bucket's row count) on the backend
    /// stream. `alpha` is the NVFP4 GEMM's `activation scale x weight global scale`, or 1 for FP8.
    ///
    /// # Safety
    /// The operand pointers must address live device memory sized for the bucket's rows, the
    /// activation rows past `rows` must be zero, and the memory must stay live until the stream
    /// reaches this GEMM.
    pub unsafe fn run(
        &mut self,
        shape: usize,
        rows: usize,
        alpha: f32,
        ops: &LtOperands,
    ) -> Result<(), RuntimeError> {
        let b = bucket(rows).ok_or_else(|| {
            RuntimeError::Compute(format!(
                "a native prefill GEMM of {rows} rows is outside 1..={MAX_ROWS}"
            ))
        })?;
        let ws = self.workspace.device_ptr(&self.stream).0;
        let plan = &mut self.plans[shape * BUCKETS + b];
        plan.matmul.launch(
            &self.handle,
            &self.stream,
            &plan.selected.algo,
            alpha,
            ops,
            ws,
            WORKSPACE_BYTES,
        )
    }
}

impl Drop for NativeGemm {
    fn drop(&mut self) {
        // Wait for every enqueued GEMM before the plans, the handle and the workspace go.
        let _ = self.stream.synchronize();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_cover_every_length_once() {
        assert_eq!(BUCKETS, 128);
        assert_eq!(bucket(0), None);
        assert_eq!(bucket(MAX_ROWS + 1), None);
        for rows in 1..=MAX_ROWS {
            let b = bucket(rows).unwrap();
            assert!(
                bucket_rows(b) >= rows && bucket_rows(b) - rows < ROW_STEP,
                "{rows}"
            );
        }
        assert_eq!(bucket(16), Some(0));
        assert_eq!(bucket(17), Some(1));
        assert_eq!(bucket_rows(BUCKETS - 1), MAX_ROWS);
    }
}
