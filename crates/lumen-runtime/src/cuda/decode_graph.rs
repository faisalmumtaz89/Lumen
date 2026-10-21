//! Decode as one CUDA graph: a token's launches and device copies, captured once when the model
//! loads and replayed for every token.
//!
//! The capture records the eager decode's own enqueue, so the graph holds the same kernels with the
//! same arguments in the same order. While it runs, the device refuses every host copy, allocation
//! and synchronization ([`refuse_in_capture`]), and each launch whose arguments change
//! from token to token names its node and those argument slots ([`register`]); a per-token launch
//! on a route the graph does not replay refuses the capture at its site. A refused capture leaves
//! the backend eager for its lifetime.
//!
//! Per token, a [`DecodePlan`] gives every per-token value. A registered node whose values differ
//! from the ones it holds gets all of its slots, and the launch geometry that follows the attention
//! split count, rewritten in its own argument storage and applied with one
//! `cuGraphExecKernelNodeSetParams`; one `cuGraphLaunch` then runs the token.

use std::cell::{Cell, RefCell};
use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{sys, CudaStream};

use super::attention_decode::decode_attention_merge_shared_bytes;
use super::ffi::CudaDevice;
use crate::error::RuntimeError;

/// A per-token value a kernel argument takes. Every one is a `u32` argument.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Slot {
    /// The token being decoded.
    Token,
    /// The token's position: the KV length before it is appended.
    Position,
    /// The keys attention covers: the position plus one.
    Keys,
    /// The attention split count.
    Splits,
    /// The attention partition.
    Partition,
    /// The conv ring position of the GDN layer with this GDN index.
    Ring(usize),
}

/// The launch geometry of a registered node.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Geometry {
    /// As captured.
    Fixed,
    /// The attention partial pass: one CTA column per split.
    SplitGrid,
    /// The attention merge: shared memory sized for the split count.
    MergeShared,
}

/// One token's per-token values.
pub(crate) struct DecodePlan<'a> {
    pub token: u32,
    pub position: u32,
    pub splits: u32,
    pub partition: u32,
    /// Each GDN layer's conv ring position, by GDN index.
    pub rings: &'a [u32],
}

impl DecodePlan<'_> {
    fn value(&self, slot: Slot) -> u32 {
        match slot {
            Slot::Token => self.token,
            Slot::Position => self.position,
            Slot::Keys => self.position + 1,
            Slot::Splits => self.splits,
            Slot::Partition => self.partition,
            Slot::Ring(gdn_idx) => self.rings[gdn_idx],
        }
    }
}

/// A launch with per-token arguments, as the capture recorded it.
struct Registered {
    node: sys::CUgraphNode,
    /// The kernel's argument count.
    args: usize,
    slots: Vec<(usize, Slot)>,
    geometry: Geometry,
}

/// What the capture on this thread has recorded so far.
#[derive(Default)]
struct Capture {
    registered: Vec<Registered>,
    /// The first call or route that refused the capture.
    refused: Option<String>,
}

thread_local! {
    /// The capture in progress on this thread (CUDA captures in thread-local mode).
    static CAPTURE: RefCell<Option<Capture>> = const { RefCell::new(None) };
}

/// The error of a capture `what` refused.
fn refusal(what: &str) -> RuntimeError {
    RuntimeError::Compute(format!("decode graph capture refused by {what}"))
}

/// Refuse `what` while this thread captures the decode graph: a host copy, an allocation, a
/// synchronization, or a per-token launch the graph does not replay. Outside a capture, `Ok`.
pub(crate) fn refuse_in_capture(what: &str) -> Result<(), RuntimeError> {
    CAPTURE.with(|c| match c.borrow_mut().as_mut() {
        None => Ok(()),
        Some(capture) => {
            capture.refused.get_or_insert_with(|| what.to_string());
            Err(refusal(what))
        }
    })
}

/// Record the kernel just launched on the capturing stream as a node whose argument slots `slots`
/// (index, value) take per-token values; `args` is the kernel's argument count. Outside a capture
/// this does nothing.
pub(crate) fn register(
    device: &CudaDevice,
    args: usize,
    slots: &[(usize, Slot)],
    geometry: Geometry,
) -> Result<(), RuntimeError> {
    CAPTURE.with(|c| {
        let mut c = c.borrow_mut();
        let Some(capture) = c.as_mut() else {
            return Ok(());
        };
        let mut status = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        let mut deps: *const sys::CUgraphNode = std::ptr::null();
        let mut count = 0usize;
        // SAFETY: the stream is this thread's capturing stream and every out-parameter is valid or
        // null (the id and graph are optional).
        let queried = unsafe {
            sys::cuStreamGetCaptureInfo_v2(
                device.stream.cu_stream(),
                &mut status,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut deps,
                &mut count,
            )
        };
        if queried != sys::CUresult::CUDA_SUCCESS
            || status != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE
            || count != 1
        {
            let what =
                format!("cuStreamGetCaptureInfo ({queried:?}, {status:?}, {count} dependencies)");
            let error = refusal(&what);
            capture.refused.get_or_insert(what);
            return Err(error);
        }
        // SAFETY: the driver returned one dependency, the node the launch just added.
        let node = unsafe { *deps };
        capture.registered.push(Registered {
            node,
            args,
            slots: slots.to_vec(),
            geometry,
        });
        Ok(())
    })
}

/// A registered node of the executable graph, with the argument array it is set from.
struct Node {
    handle: sys::CUgraphNode,
    /// The captured parameters with `_args` as their argument array, holding the grid and shared
    /// size last applied.
    params: sys::CUDA_KERNEL_NODE_PARAMS,
    /// The argument array `params` points at, one pointer per kernel argument: the captured
    /// node's own values, except the per-token slots, which point into `values`. Held here only
    /// so it lives as long as `params`; nothing reads it by name.
    _args: Vec<*mut c_void>,
    slots: Vec<(usize, Slot)>,
    /// The per-token values last written, one per slot.
    values: Box<[Cell<u32>]>,
    geometry: Geometry,
    /// Whether the executable holds what `values` and `params` hold: false
    /// while an update is applied, and after one that failed.
    applied: bool,
}

/// The captured decode graph and its executable.
pub(crate) struct DecodeGraph {
    graph: sys::CUgraph,
    exec: sys::CUgraphExec,
    nodes: Vec<Node>,
    /// Tokens this graph has run.
    launches: u64,
    /// Keeps the context the graph was made in alive until it is destroyed.
    _stream: Arc<CudaStream>,
}

// SAFETY: the graph, its executable and the argument storage are used only through `&mut self`
// under the backend's state mutex, which serialises every call on them as CUDA requires of graph
// objects; the raw pointers point into memory this value or its captured graph owns.
unsafe impl Send for DecodeGraph {}

impl Drop for DecodeGraph {
    fn drop(&mut self) {
        // SAFETY: each handle is null or was created by `capture` and is destroyed once, the
        // executable before the graph its nodes belong to.
        unsafe {
            if !self.exec.is_null() {
                sys::cuGraphExecDestroy(self.exec);
            }
            if !self.graph.is_null() {
                sys::cuGraphDestroy(self.graph);
            }
        }
    }
}

/// `call`'s result as a `RuntimeError` naming it.
fn check(result: sys::CUresult, call: &str) -> Result<(), RuntimeError> {
    result
        .result()
        .map_err(|e| RuntimeError::Compute(format!("{call}: {e}")))
}

impl DecodeGraph {
    /// Capture what `enqueue` puts on the device's stream, instantiate it and upload it. The
    /// capture executes nothing. Fails with the call that refused the capture, or the driver's
    /// error.
    pub(crate) fn capture(
        device: &CudaDevice,
        enqueue: impl FnOnce() -> Result<(), RuntimeError>,
    ) -> Result<Self, RuntimeError> {
        let stream = device.stream.cu_stream();
        let mut built = DecodeGraph {
            graph: std::ptr::null_mut(),
            exec: std::ptr::null_mut(),
            nodes: Vec::new(),
            launches: 0,
            _stream: device.stream.clone(),
        };
        device
            .stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| RuntimeError::Compute(format!("cuStreamBeginCapture: {e}")))?;
        CAPTURE.with(|c| *c.borrow_mut() = Some(Capture::default()));
        let enqueued = enqueue();
        let capture = CAPTURE.with(|c| c.borrow_mut().take()).unwrap_or_default();
        // SAFETY: ends the capture begun above on this thread; `graph` is a valid out-parameter.
        let ended = unsafe { sys::cuStreamEndCapture(stream, &mut built.graph) };
        if let Some(what) = capture.refused {
            return Err(refusal(&what));
        }
        enqueued?;
        check(ended, "cuStreamEndCapture")?;
        // SAFETY: `graph` is the captured graph; `exec` is a valid out-parameter.
        check(
            unsafe { sys::cuGraphInstantiateWithFlags(&mut built.exec, built.graph, 0) },
            "cuGraphInstantiate",
        )?;
        for r in capture.registered {
            // SAFETY: zeroed is a valid parameter block, which the call fills.
            let mut params: sys::CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
            // SAFETY: `r.node` is a kernel node of the captured graph.
            check(
                unsafe { sys::cuGraphKernelNodeGetParams_v2(r.node, &mut params) },
                "cuGraphKernelNodeGetParams",
            )?;
            // SAFETY: the node's argument array holds one pointer per kernel argument (`r.args`,
            // the launch site's count) and, with the values they point to, stays valid while the
            // captured graph, which is never modified, is alive.
            let mut args =
                unsafe { std::slice::from_raw_parts(params.kernelParams, r.args) }.to_vec();
            let values: Box<[Cell<u32>]> = r
                .slots
                .iter()
                // SAFETY: every per-token slot is a `u32` argument of the captured launch.
                .map(|&(i, _)| Cell::new(unsafe { *(args[i] as *const u32) }))
                .collect();
            for (value, &(i, _)) in values.iter().zip(&r.slots) {
                args[i] = value.as_ptr().cast();
            }
            params.kernelParams = args.as_mut_ptr();
            built.nodes.push(Node {
                handle: r.node,
                params,
                _args: args,
                slots: r.slots,
                values,
                geometry: r.geometry,
                applied: true,
            });
        }
        // SAFETY: `exec` was instantiated above; the stream is the device's.
        check(
            unsafe { sys::cuGraphUpload(built.exec, stream) },
            "cuGraphUpload",
        )?;
        Ok(built)
    }

    /// Nodes in the graph, and those with per-token arguments.
    pub(crate) fn node_counts(&self) -> Result<(usize, usize), RuntimeError> {
        let mut count = 0usize;
        // SAFETY: a null node array asks for the count only.
        check(
            unsafe { sys::cuGraphGetNodes(self.graph, std::ptr::null_mut(), &mut count) },
            "cuGraphGetNodes",
        )?;
        Ok((count, self.nodes.len()))
    }

    /// Tokens this graph has run.
    pub(crate) fn launches(&self) -> u64 {
        self.launches
    }

    /// Write `plan` into every registered node whose values or geometry it changes, then run the
    /// graph on the device's stream.
    pub(crate) fn launch(
        &mut self,
        device: &CudaDevice,
        plan: &DecodePlan<'_>,
    ) -> Result<(), RuntimeError> {
        device
            .ctx
            .bind_to_thread()
            .map_err(|e| RuntimeError::Compute(format!("decode graph: {e}")))?;
        for node in &mut self.nodes {
            let (grid_x, shared) = match node.geometry {
                Geometry::Fixed => (node.params.gridDimX, node.params.sharedMemBytes),
                Geometry::SplitGrid => (plan.splits, node.params.sharedMemBytes),
                Geometry::MergeShared => (
                    node.params.gridDimX,
                    decode_attention_merge_shared_bytes(plan.splits),
                ),
            };
            let changed = !node.applied
                || grid_x != node.params.gridDimX
                || shared != node.params.sharedMemBytes
                || node
                    .slots
                    .iter()
                    .zip(node.values.iter())
                    .any(|(&(_, slot), value)| value.get() != plan.value(slot));
            if !changed {
                continue;
            }
            for (&(_, slot), value) in node.slots.iter().zip(node.values.iter()) {
                value.set(plan.value(slot));
            }
            node.params.gridDimX = grid_x;
            node.params.sharedMemBytes = shared;
            node.applied = false;
            // SAFETY: the node belongs to this executable; `params` is its captured block with the
            // argument array this node owns, every entry pointing at a live value.
            check(
                unsafe {
                    sys::cuGraphExecKernelNodeSetParams_v2(self.exec, node.handle, &node.params)
                },
                "cuGraphExecKernelNodeSetParams",
            )?;
            node.applied = true;
        }
        // SAFETY: `exec` is this graph's executable; the stream is the device's.
        check(
            unsafe { sys::cuGraphLaunch(self.exec, device.stream.cu_stream()) },
            "cuGraphLaunch",
        )?;
        self.launches += 1;
        Ok(())
    }
}
