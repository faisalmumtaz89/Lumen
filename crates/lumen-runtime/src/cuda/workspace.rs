//! The batched F32 prefill's workspace: one device allocation holding every
//! value the prefill keeps between kernels, each value (a [`Role`]) in a
//! region of it.
//!
//! The workspace is sized for the longest slice a request can run, allocated
//! once at load and reused across every request and layer. Every region is a
//! `[rows, dim]` matrix (or a scratch) of which a slice uses its first rows. A
//! caller borrows the regions one operation needs with
//! [`Workspace::regions`], which refuses two regions that share a byte, so
//! the operands of one call never alias.
//!
//! [`liveness`] lists the operations of a slice and the values each reads,
//! writes or uses up inside itself; a value is live from the operation that
//! writes it to the last one that reads it. [`compact`] lets two values share
//! bytes only when no operation has both live, so a value keeps its bytes
//! from its writer to its last reader.

use cudarc::driver::{CudaSlice, CudaViewMut};

use crate::error::RuntimeError;

use super::ffi::CudaDevice;
use super::prefill::PrefillWorkspaceLayout;

/// The element type of a role's region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Elem {
    F32,
    U32,
    I32,
    U8,
    I8,
}

impl Elem {
    fn size(self) -> usize {
        match self {
            Elem::F32 | Elem::U32 | Elem::I32 => 4,
            Elem::U8 | Elem::I8 => 1,
        }
    }
}

/// A value the F32 prefill keeps in the workspace.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Role {
    /// Activation matrix: [batch, hidden_dim].
    X,
    /// Normalized activation: [batch, hidden_dim].
    Normed,
    /// Q projection: [batch, q_dim].
    Q,
    /// K projection: [batch, kv_dim].
    K,
    /// V projection: [batch, kv_dim].
    V,
    /// Attention output: [batch, q_dim].
    AttnOut,
    /// The fused Q+gate projection `[batch, q_dim * 2]`. Present when a layer
    /// fuses Q and gate.
    QGateFused,
    /// The deinterleaved attention gate `[batch, q_dim]`. Present when a layer
    /// fuses Q and gate.
    QGateGate,
    /// Output projection + residual: [batch, hidden_dim].
    AttnProj,
    /// Gate FFN: [batch, inter_dim].
    Gate,
    /// Up FFN: [batch, inter_dim].
    Up,
    /// Down projection: [batch, hidden_dim].
    Down,
    /// Token IDs: [batch].
    TokenIds,
    /// Single-token Q: [q_dim]. Never read; only the unshared layout holds it.
    QSingle,
    /// Single-token attention output: [q_dim]. Never read; only the unshared
    /// layout holds it.
    AttnOutSingle,
    /// F32 scratch for a dequantized projection weight, sized to the largest
    /// one a GEMM's route decodes into it; the operand of the SGEMM routes.
    DequantF32,
    /// F16 (or BF16) activation scratch for the HGEMM input conversion, in
    /// bytes: [batch * in_dim * 2] for the widest row a route converts.
    ActivationF16,
    /// F16 scratch for a dequantized projection weight, in bytes, sized to
    /// the largest one a GEMM's route dequantizes into it; the operand of the
    /// dequant-to-F16 HGEMM routes.
    DequantF16,
    /// Score block of the tiled SGEMM prefill attention: `group × min(rows,
    /// 512) × context` F32, reused across every KV head, layer and request.
    /// Present only when the tiled route is taken (switch on, kernel loaded, a
    /// fused Q+gate attention layer); the dispatcher otherwise takes the
    /// scalar attention, which needs no score scratch.
    Scores,
    /// GDN batched QKV projection: [batch, qkv_dim].
    GdnQkv,
    /// GDN alpha raw projection: [batch, num_heads].
    GdnAlphaRaw,
    /// GDN beta raw projection: [batch, num_heads].
    GdnBetaRaw,
    /// GDN gate projection: [batch, value_dim].
    GdnGate,
    /// GDN output (per-token gated output): [batch, value_dim].
    GdnOut,
    /// GDN conv1d + SiLU output: [batch, qkv_dim].
    GdnConvOut,
    /// Raw output of the fused GDN state update: [batch, num_heads * val_dim].
    GdnRawOut,
    /// Computed alpha gates: [batch, num_heads].
    GdnAlphaOut,
    /// Computed beta gates: [batch, num_heads].
    GdnBetaOut,
    /// Grouped MoE: batched router logits, [batch, num_experts].
    MoeRouterLogits,
    /// Grouped MoE: top-K expert ids, [batch, num_experts] (the first top_k of
    /// each row valid).
    MoeExpertIds,
    /// Grouped MoE: top-K expert weights, [batch, top_k].
    MoeExpertWeights,
    /// Grouped MoE: each compact column's expert, [batch * top_k].
    MoeColExpert,
    /// Grouped MoE: each compact column's token, [batch * top_k].
    MoeColSrcTok,
    /// Grouped MoE: each (token, slot)'s compact column, [batch * top_k].
    MoeDstToCol,
    /// Grouped MoE: each expert's first compact column, [num_experts + 1].
    MoeExpertBounds,
    /// Grouped MoE: the tiled gate+up kernel's tile list, 4 per tile.
    MoeGateUpTiles,
    /// Grouped MoE: per-column SwiGLU output, [batch * top_k, inter_dim].
    MoeSwiglu,
    /// Grouped MoE: per-column down output, [batch * top_k, hidden_dim].
    MoeDown,
    /// Batched shared expert: SwiGLU output, [batch, inter_dim].
    MoeSharedSwiglu,
    /// Batched shared expert: sigmoid-gate logits, [batch].
    MoeSharedLogit,
    /// W10 gate+up: the prequantized activation, int8 by compact column.
    MoeW10Activation,
    /// W10 gate+up: the prequantized activation's scales.
    MoeW10Scales,
    /// W10 gate+up: the bucketed tile list.
    MoeW10Tiles,
}

/// The number of roles.
pub(crate) const ROLES: usize = 43;

impl Role {
    /// Every role, in the order the unshared layout places them.
    pub(crate) const ALL: [Role; ROLES] = [
        Role::X,
        Role::Normed,
        Role::Q,
        Role::K,
        Role::V,
        Role::AttnOut,
        Role::QGateFused,
        Role::QGateGate,
        Role::AttnProj,
        Role::Gate,
        Role::Up,
        Role::Down,
        Role::TokenIds,
        Role::QSingle,
        Role::AttnOutSingle,
        Role::DequantF32,
        Role::ActivationF16,
        Role::DequantF16,
        Role::Scores,
        Role::GdnQkv,
        Role::GdnAlphaRaw,
        Role::GdnBetaRaw,
        Role::GdnGate,
        Role::GdnOut,
        Role::GdnConvOut,
        Role::GdnRawOut,
        Role::GdnAlphaOut,
        Role::GdnBetaOut,
        Role::MoeRouterLogits,
        Role::MoeExpertIds,
        Role::MoeExpertWeights,
        Role::MoeColExpert,
        Role::MoeColSrcTok,
        Role::MoeDstToCol,
        Role::MoeExpertBounds,
        Role::MoeGateUpTiles,
        Role::MoeSwiglu,
        Role::MoeDown,
        Role::MoeSharedSwiglu,
        Role::MoeSharedLogit,
        Role::MoeW10Activation,
        Role::MoeW10Scales,
        Role::MoeW10Tiles,
    ];

    /// The grouped MoE prefill's routed-expert values, which one MoE FFN
    /// writes and uses up.
    pub(crate) const MOE_ROUTED: [Role; 13] = [
        Role::MoeRouterLogits,
        Role::MoeExpertIds,
        Role::MoeExpertWeights,
        Role::MoeColExpert,
        Role::MoeColSrcTok,
        Role::MoeDstToCol,
        Role::MoeExpertBounds,
        Role::MoeGateUpTiles,
        Role::MoeSwiglu,
        Role::MoeDown,
        Role::MoeW10Activation,
        Role::MoeW10Scales,
        Role::MoeW10Tiles,
    ];

    pub(crate) fn elem(self) -> Elem {
        match self {
            Role::TokenIds | Role::MoeExpertIds => Elem::U32,
            Role::MoeColExpert
            | Role::MoeColSrcTok
            | Role::MoeDstToCol
            | Role::MoeExpertBounds
            | Role::MoeGateUpTiles
            | Role::MoeW10Tiles => Elem::I32,
            Role::ActivationF16 | Role::DequantF16 => Elem::U8,
            Role::MoeW10Activation => Elem::I8,
            _ => Elem::F32,
        }
    }
}

/// Byte alignment of every region's offset: what `cudaMalloc` guarantees a
/// buffer of its own.
pub(crate) const REGION_ALIGN: usize = 256;

/// Where a role lives in the arena: `offset` bytes from its start (a multiple
/// of [`REGION_ALIGN`]) and `len` elements of the role's type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Region {
    pub offset: usize,
    pub len: usize,
}

impl Region {
    fn end(self, role: Role) -> usize {
        self.offset + self.len * role.elem().size()
    }
}

/// The region of every role a workspace holds; `None` for a role this load's
/// layout has no value for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RegionTable {
    regions: [Option<Region>; ROLES],
}

impl Default for RegionTable {
    fn default() -> Self {
        Self {
            regions: [None; ROLES],
        }
    }
}

impl RegionTable {
    fn get(&self, role: Role) -> Option<Region> {
        self.regions[role as usize]
    }

    /// The roles of `entries`, each with its element count, back to back at
    /// aligned offsets.
    #[cfg(test)]
    pub(crate) fn of(entries: &[(Role, usize)]) -> Self {
        let mut table = Self::default();
        let mut end = 0usize;
        for &(role, len) in entries {
            let region = Region {
                offset: end.next_multiple_of(REGION_ALIGN),
                len,
            };
            end = region.end(role);
            table.regions[role as usize] = Some(region);
        }
        table
    }

    /// Bytes of the arena the regions lie in.
    pub(crate) fn bytes(&self) -> usize {
        Role::ALL
            .iter()
            .filter_map(|&role| self.get(role).map(|r| r.end(role)))
            .max()
            .unwrap_or(0)
    }

    /// The regions of `roles`, in order. Panics, naming the roles, when one
    /// has no region, one is requested twice, or two share a byte, as
    /// `<[T]>::get_disjoint_mut` refuses overlapping indices.
    fn disjoint<const N: usize>(&self, roles: &[Role; N]) -> [Region; N] {
        self.disjoint_from(&[], roles)
    }

    /// [`RegionTable::disjoint`] of `roles`, which must also be apart from
    /// every one of `lent`.
    fn disjoint_from<const M: usize>(
        &self,
        lent: &[(Role, Region)],
        roles: &[Role; M],
    ) -> [Region; M] {
        let regions = self.present_from(lent, roles);
        std::array::from_fn(|i| {
            regions[i].unwrap_or_else(|| panic!("workspace: {:?} has no region", roles[i]))
        })
    }

    /// The regions of `roles` the table has, in order (`None` for a role it
    /// has none for), refused as [`RegionTable::disjoint_from`] refuses them.
    fn present_from<const M: usize>(
        &self,
        lent: &[(Role, Region)],
        roles: &[Role; M],
    ) -> [Option<Region>; M] {
        let regions = roles.map(|role| self.get(role));
        for i in 0..M {
            for j in 0..i {
                let (a, b) = (roles[j], roles[i]);
                assert!(a != b, "workspace: {a:?} requested twice");
                if let (Some(ra), Some(rb)) = (regions[j], regions[i]) {
                    assert_apart((a, ra), (b, rb));
                }
            }
            // A lent role has a region, so only a present one can be it.
            if let Some(region) = regions[i] {
                for &lent in lent {
                    assert_apart(lent, (roles[i], region));
                }
            }
        }
        regions
    }
}

/// Panics, naming the roles, when `a` and `b` are one role or share a byte.
fn assert_apart((a, ra): (Role, Region), (b, rb): (Role, Region)) {
    assert!(a != b, "workspace: {a:?} requested twice");
    assert!(
        ra.end(a) <= rb.offset || rb.end(b) <= ra.offset,
        "workspace: {a:?} and {b:?} overlap"
    );
}

/// The element count of `role`'s value in `layout` with the conversion
/// scratch sized to the model's largest weight and widest row; `None` when
/// the model has no such value. The Q+gate pair is present only when a layer
/// fuses them, the score block only when the tiled attention is the route,
/// the GDN values only for a model with GDN layers.
fn unshared_len(layout: &PrefillWorkspaceLayout, role: Role) -> Option<usize> {
    let s = &layout.scratch;
    let gdn = layout.gdn;
    match role {
        Role::X | Role::Normed | Role::AttnProj | Role::Down => Some(s.hidden),
        Role::Q | Role::AttnOut => Some(s.q),
        Role::K | Role::V => Some(s.kv),
        Role::QGateFused => s.q_gate.map(|(fused, _)| fused),
        Role::QGateGate => s.q_gate.map(|(_, gate)| gate),
        Role::Gate | Role::Up => Some(s.inter),
        Role::TokenIds => Some(s.token_ids),
        Role::QSingle | Role::AttnOutSingle => Some(s.q_single),
        Role::DequantF32 => Some(s.dequant_f32),
        Role::ActivationF16 => Some(s.activation_f16),
        Role::DequantF16 => Some(s.dequant_f16),
        Role::Scores => layout.score_elems,
        Role::GdnQkv | Role::GdnConvOut => gdn.map(|g| g.qkv),
        Role::GdnAlphaRaw | Role::GdnBetaRaw | Role::GdnAlphaOut | Role::GdnBetaOut => {
            gdn.map(|g| g.heads)
        }
        Role::GdnGate | Role::GdnOut | Role::GdnRawOut => gdn.map(|g| g.value),
        Role::MoeRouterLogits
        | Role::MoeExpertIds
        | Role::MoeExpertWeights
        | Role::MoeColExpert
        | Role::MoeColSrcTok
        | Role::MoeDstToCol
        | Role::MoeExpertBounds
        | Role::MoeGateUpTiles
        | Role::MoeSwiglu
        | Role::MoeDown
        | Role::MoeSharedSwiglu
        | Role::MoeSharedLogit
        | Role::MoeW10Activation
        | Role::MoeW10Scales
        | Role::MoeW10Tiles => layout.grouped.and_then(|g| g.len(role)),
    }
}

/// The layout that gives every role `layout` has a value for its own region,
/// in [`Role::ALL`]'s order: the bytes of one buffer per value, each offset
/// rounded up to [`REGION_ALIGN`], the conversion scratch sized to the
/// model's largest weight and widest row.
#[cfg(any(test, feature = "test-state-snapshot"))]
pub(crate) fn unshared(layout: &PrefillWorkspaceLayout) -> RegionTable {
    let mut table = RegionTable::default();
    let mut end = 0usize;
    for role in Role::ALL {
        if let Some(len) = unshared_len(layout, role) {
            let region = Region {
                offset: end.next_multiple_of(REGION_ALIGN),
                len,
            };
            end = region.end(role);
            table.regions[role as usize] = Some(region);
        }
    }
    table
}

/// One operation of a prefill slice and the values it touches: `reads`
/// values an earlier operation wrote, `writes` values a later one reads (an
/// in-place operation reads and writes its value), `scratch` values it
/// writes and uses up itself. An operation on a path the load cannot take
/// touches no value.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Step {
    pub name: &'static str,
    pub reads: &'static [Role],
    pub writes: &'static [Role],
    pub scratch: &'static [Role],
}

struct Steps(Vec<Step>);

impl Steps {
    fn op(
        &mut self,
        on: bool,
        name: &'static str,
        reads: &'static [Role],
        writes: &'static [Role],
        scratch: &'static [Role],
    ) {
        let (reads, writes, scratch) = if on {
            (reads, writes, scratch)
        } else {
            (&[][..], &[][..], &[][..])
        };
        self.0.push(Step {
            name,
            reads,
            writes,
            scratch,
        });
    }

    /// A projection or residual GEMM, as its two conversion arms: the F32
    /// dequant scratch, or the F16 activation and weight pair (an arm takes
    /// one of the two or neither, never both).
    fn gemm(
        &mut self,
        on: bool,
        name: &'static str,
        reads: &'static [Role],
        writes: &'static [Role],
    ) {
        self.op(on, name, reads, writes, &[Role::DequantF32]);
        self.op(
            on,
            name,
            reads,
            writes,
            &[Role::ActivationF16, Role::DequantF16],
        );
    }

    /// The FFN that follows a layer's attention or GDN block, from its
    /// output in `AttnProj` to the layer's output in `X`.
    fn ffn(&mut self, dense: bool, moe: bool) {
        use Role::*;
        self.op(dense, "FFN norm", &[AttnProj], &[Normed], &[]);
        self.gemm(dense, "FFN gate", &[Normed], &[Gate]);
        self.gemm(dense, "FFN up", &[Normed], &[Up]);
        self.op(dense, "SwiGLU", &[Gate, Up], &[Gate], &[]);
        self.gemm(dense, "FFN down", &[Gate], &[Down]);
        self.op(dense, "FFN residual", &[AttnProj, Down], &[AttnProj], &[]);
        self.op(dense, "FFN output", &[AttnProj], &[X], &[]);
        // `prefill_moe_ffn_layer`: the normed dump, the routed experts (grouped
        // or per token, the residual in AttnProj, the output in X), the shared
        // expert accumulating into X, the probes and AttnProj made X's copy.
        self.op(moe, "MoE norm", &[AttnProj], &[Normed], &[]);
        self.op(moe, "MoE normed dump", &[Normed], &[], &[]);
        self.op(
            moe,
            "MoE routed experts",
            &[Normed, AttnProj],
            &[X],
            &Role::MOE_ROUTED,
        );
        self.op(
            moe,
            "MoE shared expert",
            &[Normed, X],
            &[X],
            &[MoeSharedSwiglu, MoeSharedLogit],
        );
        self.op(moe, "MoE probe", &[X, AttnProj], &[], &[]);
        self.op(moe, "MoE output", &[X], &[AttnProj], &[]);
    }
}

/// The operations of one prefill slice for `layout`'s model, in order: the
/// embedding, an attention layer of each kind (with the fused Q+gate
/// projection, without it), a GDN layer (fused state update, per-token
/// loop), each followed by its FFN (dense, MoE), and the slice's last row.
/// No value but `X` outlives a layer, so one layer of each kind stands for
/// all of them. Diagnostic reads (probes, dumps) are operations of their own
/// at the point they run.
pub(crate) fn liveness(layout: &PrefillWorkspaceLayout) -> Vec<Step> {
    use Role::*;
    let r = layout.routes;
    let fused = layout.scratch.q_gate.is_some();
    let tiled = fused && layout.score_elems.is_some();
    let gdn = layout.gdn.is_some();
    let (gdn_fused, gdn_loop) = (gdn && r.gdn_fused, gdn && !r.gdn_fused);
    let mut t = Steps(Vec::new());
    t.op(true, "token upload", &[], &[TokenIds], &[]);
    t.op(true, "embed", &[TokenIds], &[X], &[]);

    t.op(fused, "attention norm", &[X], &[Normed], &[]);
    t.gemm(fused, "wq_qgate", &[Normed], &[QGateFused]);
    t.gemm(fused, "wk", &[Normed], &[K]);
    t.gemm(fused, "wv", &[Normed], &[V]);
    t.op(
        fused,
        "Q+gate deinterleave",
        &[QGateFused],
        &[Q, QGateGate],
        &[],
    );
    t.op(fused, "Q and K norms", &[Q, K], &[Q, K], &[]);
    t.op(fused, "RoPE", &[Q, K], &[Q, K], &[]);
    t.op(fused, "RoPE probe", &[Q, K, V], &[], &[]);
    t.op(fused, "KV write", &[K, V], &[], &[]);
    t.op(tiled, "tiled attention", &[Q], &[AttnOut], &[Scores]);
    t.op(fused, "scalar attention", &[Q], &[AttnOut], &[]);
    t.op(fused, "attention probe", &[AttnOut], &[], &[]);
    // The gated output goes through Q, free after the attention, and back.
    t.op(fused, "sigmoid gate", &[QGateGate, AttnOut], &[Q], &[]);
    t.op(fused, "gated output copy", &[Q], &[AttnOut], &[]);
    t.gemm(fused, "wo + residual", &[AttnOut, X], &[AttnProj]);
    t.ffn(fused && r.dense_ffn, fused && r.moe_ffn);

    let plain = r.plain_attention;
    t.op(plain, "attention norm", &[X], &[Normed], &[]);
    t.gemm(plain, "wq", &[Normed], &[Q]);
    t.gemm(plain, "wk", &[Normed], &[K]);
    t.gemm(plain, "wv", &[Normed], &[V]);
    t.op(plain, "QKV bias", &[Q, K, V], &[Q, K, V], &[]);
    t.op(plain, "RoPE", &[Q, K], &[Q, K], &[]);
    t.op(plain, "KV write", &[K, V], &[], &[]);
    t.op(plain, "scalar attention", &[Q], &[AttnOut], &[]);
    t.gemm(plain, "wo + residual", &[AttnOut, X], &[AttnProj]);
    t.ffn(plain && r.dense_ffn, plain && r.moe_ffn);

    t.op(gdn, "GDN norm", &[X], &[Normed], &[]);
    t.gemm(gdn, "gdn_qkv", &[Normed], &[GdnQkv]);
    t.gemm(gdn, "gdn_gate", &[Normed], &[GdnGate]);
    t.gemm(gdn, "gdn_alpha", &[Normed], &[GdnAlphaRaw]);
    t.gemm(gdn, "gdn_beta", &[Normed], &[GdnBetaRaw]);
    t.op(
        gdn,
        "GDN projection probe",
        &[GdnQkv, GdnAlphaRaw, GdnBetaRaw, GdnGate],
        &[],
        &[],
    );
    t.op(gdn_fused, "conv1d + SiLU", &[GdnQkv], &[GdnConvOut], &[]);
    // The raw gates' dumps run after the gates kernel, in this operation.
    t.op(
        gdn_fused,
        "gates",
        &[GdnAlphaRaw, GdnBetaRaw],
        &[GdnAlphaOut, GdnBetaOut],
        &[],
    );
    t.op(gdn_fused, "Q/K L2 norm", &[GdnConvOut], &[GdnConvOut], &[]);
    t.op(
        gdn_fused,
        "state scan",
        &[GdnConvOut, GdnAlphaOut, GdnBetaOut],
        &[GdnRawOut],
        &[],
    );
    t.op(
        gdn_fused,
        "norm + gate",
        &[GdnRawOut, GdnGate],
        &[GdnOut],
        &[],
    );
    // Each token reads its row of the projections and writes its output row.
    t.op(
        gdn_loop,
        "per-token state update",
        &[GdnQkv, GdnAlphaRaw, GdnBetaRaw, GdnGate],
        &[GdnOut],
        &[],
    );
    t.gemm(gdn, "gdn_ssm_out + residual", &[GdnOut, X], &[AttnProj]);
    t.op(gdn, "GDN output dump", &[AttnProj], &[], &[]);
    t.ffn(gdn && r.dense_ffn, gdn && r.moe_ffn);

    t.op(true, "last row", &[X], &[], &[]);
    t.0
}

/// For each role, whether its value is live at each operation of `table`:
/// from an operation that writes it to the last that reads what it wrote,
/// and at each operation it is scratch of. `X` is live throughout: every
/// layer carries it to the next, and the slice's last row is read from it
/// after the last.
fn live(table: &[Step]) -> [Vec<bool>; ROLES] {
    std::array::from_fn(|i| {
        let role = Role::ALL[i];
        let mut live = vec![role == Role::X; table.len()];
        let mut written = None;
        for (s, step) in table.iter().enumerate() {
            if step.reads.contains(&role) {
                let w = written.unwrap_or_else(|| {
                    panic!("workspace: {} reads {role:?} before a write", step.name)
                });
                live[w..=s].fill(true);
            }
            if step.writes.contains(&role) {
                written = Some(s);
                live[s] = true;
            }
            if step.scratch.contains(&role) {
                live[s] = true;
            }
        }
        live
    })
}

/// The element count of each role's region in the compact layout: as in the
/// unshared one, but the conversion scratch sized per GEMM route
/// (`layout.conversion`), and none for a value no path of this load writes:
/// the single-token pair (never read), the attention values of a model with
/// no attention layer, the dense FFN values of a model with no dense FFN,
/// the fused state update's values when the GDN layers take the per-token
/// loop.
fn compact_len(layout: &PrefillWorkspaceLayout, role: Role) -> Option<usize> {
    let r = layout.routes;
    let attention = layout.scratch.q_gate.is_some() || r.plain_attention;
    match role {
        Role::QSingle | Role::AttnOutSingle => None,
        Role::Q | Role::K | Role::V | Role::AttnOut if !attention => None,
        Role::Gate | Role::Up | Role::Down if !r.dense_ffn => None,
        Role::GdnConvOut | Role::GdnAlphaOut | Role::GdnBetaOut | Role::GdnRawOut
            if !r.gdn_fused =>
        {
            None
        }
        Role::DequantF32 => layout.conversion.dequant_f32,
        Role::ActivationF16 => layout.conversion.activation_f16,
        Role::DequantF16 => layout.conversion.dequant_f16,
        _ => unshared_len(layout, role),
    }
}

/// The layout that lets values share bytes: each role `layout` has a value
/// for gets a region apart from the region of every role live at an
/// operation of `table` it is live at. Roles are placed largest first (a
/// role live throughout before all, as it is apart from every other), each
/// at the lowest offset, a multiple of [`REGION_ALIGN`], clear of the
/// regions already placed for the roles live with it.
pub(crate) fn compact(layout: &PrefillWorkspaceLayout, table: &[Step]) -> RegionTable {
    let live = live(table);
    let together = |a: Role, b: Role| {
        live[a as usize]
            .iter()
            .zip(&live[b as usize])
            .any(|(&x, &y)| x && y)
    };
    let mut order: Vec<(Role, usize)> = Role::ALL
        .into_iter()
        .filter_map(|role| compact_len(layout, role).map(|len| (role, len)))
        .collect();
    order.sort_by_key(|&(role, len)| {
        let throughout = live[role as usize].iter().all(|&l| l);
        (!throughout, std::cmp::Reverse(len * role.elem().size()))
    });
    let mut table = RegionTable::default();
    let mut placed: Vec<(Role, Region)> = Vec::new();
    for (role, len) in order {
        let bytes = len * role.elem().size();
        let clash: Vec<(Role, Region)> = placed
            .iter()
            .copied()
            .filter(|&(other, _)| together(role, other))
            .collect();
        let offset = std::iter::once(0)
            .chain(
                clash
                    .iter()
                    .map(|&(other, r)| r.end(other).next_multiple_of(REGION_ALIGN)),
            )
            .filter(|&offset| {
                clash
                    .iter()
                    .all(|&(other, r)| offset + bytes <= r.offset || r.end(other) <= offset)
            })
            .min()
            .expect("the end of the last clashing region is clear");
        let region = Region { offset, len };
        table.regions[role as usize] = Some(region);
        placed.push((role, region));
    }
    table
}

/// The prefill workspace: one zeroed allocation and the region of each role
/// in it.
pub(crate) struct Workspace {
    arena: CudaSlice<u8>,
    regions: RegionTable,
}

impl Workspace {
    /// Allocate the arena `regions` lie in.
    pub(crate) fn alloc(device: &CudaDevice, regions: RegionTable) -> Result<Self, RuntimeError> {
        Ok(Self {
            arena: device.alloc_zeros(regions.bytes())?,
            regions,
        })
    }

    /// Whether this load's layout has a region for `role`.
    pub(crate) fn has(&self, role: Role) -> bool {
        self.regions.get(role).is_some()
    }

    /// Views of the regions of `roles`, in order, for one operation. Panics
    /// when a role has no region, is requested twice, or shares a byte with
    /// another requested role. Every view borrows the whole workspace, so no
    /// other borrow of it coexists with them.
    pub(crate) fn regions<const N: usize>(&mut self, roles: [Role; N]) -> [RoleView<'_>; N] {
        self.lend(roles).0
    }

    /// [`Workspace::regions`] of `roles`, and the rest of the workspace, from
    /// which the same operation borrows the regions it picks only once it has
    /// started: a GEMM's conversion scratch, which its weight's route selects.
    pub(crate) fn lend<const N: usize>(
        &mut self,
        roles: [Role; N],
    ) -> ([RoleView<'_>; N], Rest<'_, N>) {
        let regions = self.regions.disjoint(&roles);
        let mut whole = self.arena.as_view_mut();
        // SAFETY: `disjoint` has checked that no two of the regions share a
        // byte, and the rest lends none of them; see `view`.
        let views = std::array::from_fn(|i| unsafe { view(&mut whole, roles[i], regions[i]) });
        let rest = Rest {
            whole,
            table: &self.regions,
            lent: std::array::from_fn(|i| (roles[i], regions[i])),
        };
        (views, rest)
    }
}

/// The view of `role`'s `region` of the arena `whole` views, for as long as
/// the arena is borrowed.
///
/// # Safety
///
/// No other view of a byte of `region` may exist while the returned one
/// does, as the results of `<[T]>::get_disjoint_mut` alias no element. Each
/// region's offset is a multiple of `REGION_ALIGN` and its byte length is
/// `len` elements of the role's type, so the typed view is aligned and in
/// bounds. A view carries only the arena's device pointer, length and event
/// and stream references, which live as long as the arena's borrow `'a`; the
/// shorter lifetime `slice_mut` and `transmute_mut` give it only records the
/// borrow of `whole`, which the view does not use.
unsafe fn view<'a>(whole: &mut CudaViewMut<'a, u8>, role: Role, region: Region) -> RoleView<'a> {
    let mut bytes = whole.slice_mut(region.offset..region.end(role));
    let view = match role.elem() {
        Elem::F32 => Typed::F32(std::mem::transmute::<
            CudaViewMut<'_, f32>,
            CudaViewMut<'a, f32>,
        >(
            bytes
                .transmute_mut::<f32>(region.len)
                .expect("a region holds its elements"),
        )),
        Elem::U32 => Typed::U32(std::mem::transmute::<
            CudaViewMut<'_, u32>,
            CudaViewMut<'a, u32>,
        >(
            bytes
                .transmute_mut::<u32>(region.len)
                .expect("a region holds its elements"),
        )),
        Elem::I32 => Typed::I32(std::mem::transmute::<
            CudaViewMut<'_, i32>,
            CudaViewMut<'a, i32>,
        >(
            bytes
                .transmute_mut::<i32>(region.len)
                .expect("a region holds its elements"),
        )),
        Elem::U8 => Typed::U8(std::mem::transmute::<
            CudaViewMut<'_, u8>,
            CudaViewMut<'a, u8>,
        >(bytes)),
        Elem::I8 => Typed::I8(std::mem::transmute::<
            CudaViewMut<'_, i8>,
            CudaViewMut<'a, i8>,
        >(
            bytes
                .transmute_mut::<i8>(region.len)
                .expect("a region holds its elements"),
        )),
    };
    RoleView { role, view }
}

/// The regions of a [`Workspace`] that [`Workspace::lend`] did not lend, for
/// the operation it lent them to.
pub(crate) struct Rest<'a, const N: usize> {
    whole: CudaViewMut<'a, u8>,
    table: &'a RegionTable,
    lent: [(Role, Region); N],
}

impl<const N: usize> Rest<'_, N> {
    /// Elements of `role`'s region; 0 when this load's layout has none.
    pub(crate) fn len(&self, role: Role) -> usize {
        self.table.get(role).map_or(0, |r| r.len)
    }

    /// Views of the regions of `roles`, refused as [`Workspace::regions`]
    /// refuses them, and also when one is a lent role or shares a byte with
    /// one.
    pub(crate) fn regions<const M: usize>(&mut self, roles: [Role; M]) -> [RoleView<'_>; M] {
        let regions = self.table.disjoint_from(&self.lent, &roles);
        // SAFETY: the regions share no byte with each other or with a lent
        // one (`disjoint_from`), and every view borrows `self`; see `view`.
        std::array::from_fn(|i| unsafe { view(&mut self.whole, roles[i], regions[i]) })
    }

    /// Whether this load's layout has a region for `role`.
    pub(crate) fn has(&self, role: Role) -> bool {
        self.table.get(role).is_some()
    }

    /// [`Rest::regions`] of the roles of `roles` this load's layout has a
    /// region for; `None` for the others.
    pub(crate) fn present<const M: usize>(
        &mut self,
        roles: [Role; M],
    ) -> [Option<RoleView<'_>>; M] {
        let regions = self.table.present_from(&self.lent, &roles);
        // SAFETY: as in `regions`; see `view`.
        std::array::from_fn(|i| {
            regions[i].map(|region| unsafe { view(&mut self.whole, roles[i], region) })
        })
    }
}

/// A role's region, typed as the role's elements.
enum Typed<'a> {
    F32(CudaViewMut<'a, f32>),
    U32(CudaViewMut<'a, u32>),
    I32(CudaViewMut<'a, i32>),
    U8(CudaViewMut<'a, u8>),
    I8(CudaViewMut<'a, i8>),
}

/// One role's region, borrowed from a [`Workspace`] by
/// [`Workspace::regions`].
pub(crate) struct RoleView<'a> {
    role: Role,
    view: Typed<'a>,
}

impl RoleView<'_> {
    /// The region as F32 elements. Panics when the role holds another type.
    pub(crate) fn f32(&mut self) -> CudaViewMut<'_, f32> {
        match &mut self.view {
            Typed::F32(v) => v.slice_mut(..),
            _ => elem_mismatch(self.role, Elem::F32),
        }
    }

    /// The region as U32 elements. Panics when the role holds another type.
    pub(crate) fn u32(&mut self) -> CudaViewMut<'_, u32> {
        match &mut self.view {
            Typed::U32(v) => v.slice_mut(..),
            _ => elem_mismatch(self.role, Elem::U32),
        }
    }

    /// The region as I32 elements. Panics when the role holds another type.
    pub(crate) fn i32(&mut self) -> CudaViewMut<'_, i32> {
        match &mut self.view {
            Typed::I32(v) => v.slice_mut(..),
            _ => elem_mismatch(self.role, Elem::I32),
        }
    }

    /// The region as bytes. Panics when the role holds another type.
    pub(crate) fn u8(&mut self) -> CudaViewMut<'_, u8> {
        match &mut self.view {
            Typed::U8(v) => v.slice_mut(..),
            _ => elem_mismatch(self.role, Elem::U8),
        }
    }

    /// The region as I8 elements. Panics when the role holds another type.
    pub(crate) fn i8(&mut self) -> CudaViewMut<'_, i8> {
        match &mut self.view {
            Typed::I8(v) => v.slice_mut(..),
            _ => elem_mismatch(self.role, Elem::I8),
        }
    }
}

impl<'a> RoleView<'a> {
    /// [`RoleView::f32`] for the whole borrow.
    pub(crate) fn into_f32(self) -> CudaViewMut<'a, f32> {
        match self.view {
            Typed::F32(v) => v,
            _ => elem_mismatch(self.role, Elem::F32),
        }
    }

    /// [`RoleView::u32`] for the whole borrow.
    pub(crate) fn into_u32(self) -> CudaViewMut<'a, u32> {
        match self.view {
            Typed::U32(v) => v,
            _ => elem_mismatch(self.role, Elem::U32),
        }
    }

    /// [`RoleView::i32`] for the whole borrow.
    pub(crate) fn into_i32(self) -> CudaViewMut<'a, i32> {
        match self.view {
            Typed::I32(v) => v,
            _ => elem_mismatch(self.role, Elem::I32),
        }
    }

    /// [`RoleView::i8`] for the whole borrow.
    pub(crate) fn into_i8(self) -> CudaViewMut<'a, i8> {
        match self.view {
            Typed::I8(v) => v,
            _ => elem_mismatch(self.role, Elem::I8),
        }
    }
}

fn elem_mismatch(role: Role, wanted: Elem) -> ! {
    panic!(
        "workspace: {role:?} holds {:?}, not {wanted:?}",
        role.elem()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::moe::PrefillGroupedLayout;
    use crate::cuda::prefill::test_models::{
        dense_switches, moe_switches, Model, MOE, Q27B, Q9B, ROWS,
    };
    use crate::cuda::prefill::{
        ConversionScratch, GdnPrefillScratchLayout, GemmSwitches, PrefillRoutes,
        PrefillScratchLayout, WeightClass,
    };

    fn table(entries: &[(Role, usize, usize)]) -> RegionTable {
        let mut t = RegionTable::default();
        for &(role, offset, len) in entries {
            t.regions[role as usize] = Some(Region { offset, len });
        }
        t
    }

    #[test]
    fn roles_are_listed_once_in_declaration_order() {
        for (i, role) in Role::ALL.iter().enumerate() {
            assert_eq!(*role as usize, i, "{role:?}");
        }
    }

    #[test]
    fn each_role_holds_the_element_type_its_kernels_take() {
        for role in Role::ALL {
            let want = match role {
                Role::TokenIds | Role::MoeExpertIds => Elem::U32,
                Role::MoeColExpert
                | Role::MoeColSrcTok
                | Role::MoeDstToCol
                | Role::MoeExpertBounds
                | Role::MoeGateUpTiles
                | Role::MoeW10Tiles => Elem::I32,
                Role::ActivationF16 | Role::DequantF16 => Elem::U8,
                Role::MoeW10Activation => Elem::I8,
                _ => Elem::F32,
            };
            assert_eq!(role.elem(), want, "{role:?}");
        }
    }

    #[test]
    #[should_panic(expected = "workspace: TokenIds holds U32, not F32")]
    fn a_view_of_the_wrong_type_is_refused() {
        elem_mismatch(Role::TokenIds, Elem::F32);
    }

    #[test]
    fn disjoint_regions_are_lent_in_request_order() {
        let t = table(&[
            (Role::X, 0, 64),
            (Role::Normed, 256, 64),
            (Role::TokenIds, 512, 3),
            (Role::DequantF16, 768, 256),
        ]);
        let got = t.disjoint(&[Role::DequantF16, Role::X, Role::TokenIds]);
        assert_eq!(
            got,
            [
                Region {
                    offset: 768,
                    len: 256
                },
                Region { offset: 0, len: 64 },
                Region {
                    offset: 512,
                    len: 3
                },
            ]
        );
    }

    #[test]
    fn regions_that_only_touch_are_disjoint() {
        // X ends at byte 256 exactly where Normed starts.
        let t = table(&[(Role::X, 0, 64), (Role::Normed, 256, 64)]);
        t.disjoint(&[Role::Normed, Role::X]);
    }

    #[test]
    #[should_panic(expected = "workspace: X and Normed overlap")]
    fn overlapping_regions_are_refused() {
        // X's 64 floats end at byte 256; Normed starts at byte 252.
        let t = table(&[(Role::X, 0, 64), (Role::Normed, 252, 1)]);
        t.disjoint(&[Role::X, Role::Normed]);
    }

    #[test]
    #[should_panic(expected = "workspace: Normed and X overlap")]
    fn a_region_inside_another_is_refused() {
        let t = table(&[(Role::Normed, 0, 1024), (Role::X, 256, 1)]);
        t.disjoint(&[Role::Normed, Role::X]);
    }

    #[test]
    #[should_panic(expected = "workspace: Q requested twice")]
    fn a_role_requested_twice_is_refused() {
        let t = table(&[(Role::Q, 0, 64), (Role::K, 256, 64)]);
        t.disjoint(&[Role::Q, Role::K, Role::Q]);
    }

    #[test]
    fn the_rest_lends_regions_apart_from_the_lent_ones() {
        let t = table(&[
            (Role::Normed, 0, 64),
            (Role::Q, 256, 64),
            (Role::DequantF16, 512, 256),
            (Role::ActivationF16, 768, 256),
        ]);
        let lent = [(Role::Normed, t.get(Role::Normed).unwrap())];
        let got = t.disjoint_from(&lent, &[Role::DequantF16, Role::ActivationF16]);
        assert_eq!(got[1], t.get(Role::ActivationF16).unwrap());
    }

    #[test]
    #[should_panic(expected = "workspace: Normed requested twice")]
    fn the_rest_refuses_a_lent_role() {
        let t = table(&[(Role::Normed, 0, 64)]);
        let lent = [(Role::Normed, t.get(Role::Normed).unwrap())];
        t.disjoint_from(&lent, &[Role::Normed]);
    }

    #[test]
    #[should_panic(expected = "workspace: Q and DequantF32 overlap")]
    fn the_rest_refuses_a_region_a_lent_one_shares() {
        let t = table(&[(Role::Q, 0, 64), (Role::DequantF32, 128, 64)]);
        let lent = [(Role::Q, t.get(Role::Q).unwrap())];
        t.disjoint_from(&lent, &[Role::DequantF32]);
    }

    #[test]
    #[should_panic(expected = "workspace: Scores has no region")]
    fn an_absent_role_is_refused() {
        let t = table(&[(Role::Q, 0, 64)]);
        t.disjoint(&[Role::Q, Role::Scores]);
    }

    /// A small layout with every optional part: each role gets its own
    /// aligned region of its element count, back to back in `Role::ALL`'s
    /// order, and the arena ends at the last one.
    #[test]
    fn the_unshared_layout_gives_every_value_its_own_aligned_region() {
        let scratch = PrefillScratchLayout::new(3, 40, 24, 8, 72, Some(56), Some(32), true);
        let layout = PrefillWorkspaceLayout::of(
            scratch,
            Some(GdnPrefillScratchLayout::new(3, 56, 4, 32)),
            Some(77),
            Some(PrefillGroupedLayout {
                w10: true,
                ..PrefillGroupedLayout::new(3, 2, 4, 40, 128)
            }),
            PrefillRoutes::default(),
            ConversionScratch::default(),
        );
        let t = unshared(&layout);
        let mut end = 0;
        for role in Role::ALL {
            let r = t.get(role).unwrap_or_else(|| panic!("{role:?} absent"));
            assert_eq!(r.offset % REGION_ALIGN, 0, "{role:?}");
            assert!(r.offset >= end && r.offset < end + REGION_ALIGN, "{role:?}");
            end = r.end(role);
        }
        assert_eq!(t.bytes(), end);
        assert_eq!(t.get(Role::X).unwrap().len, 3 * 40);
        assert_eq!(t.get(Role::QGateFused).unwrap().len, 3 * 24 * 2);
        assert_eq!(t.get(Role::TokenIds).unwrap().len, 3);
        assert_eq!(
            t.get(Role::ActivationF16).unwrap().len,
            scratch.activation_f16
        );
        assert_eq!(t.get(Role::Scores).unwrap().len, 77);
        assert_eq!(t.get(Role::GdnRawOut).unwrap().len, 3 * 32);
        t.disjoint(&Role::ALL);
    }

    #[test]
    fn the_unshared_layout_leaves_out_the_values_the_model_has_none_of() {
        let layout = PrefillWorkspaceLayout::of(
            PrefillScratchLayout::new(3, 40, 24, 8, 72, None, None, false),
            None,
            None,
            None,
            PrefillRoutes::default(),
            ConversionScratch::default(),
        );
        let t = unshared(&layout);
        for role in Role::ALL {
            let absent = matches!(
                role,
                Role::QGateFused
                    | Role::QGateGate
                    | Role::Scores
                    | Role::GdnQkv
                    | Role::GdnAlphaRaw
                    | Role::GdnBetaRaw
                    | Role::GdnGate
                    | Role::GdnOut
                    | Role::GdnConvOut
                    | Role::GdnRawOut
                    | Role::GdnAlphaOut
                    | Role::GdnBetaOut
                    | Role::MoeSharedSwiglu
                    | Role::MoeSharedLogit
            ) || Role::MOE_ROUTED.contains(&role);
            assert_eq!(t.get(role).is_none(), absent, "{role:?}");
        }
    }

    /// The layout of `model` with every weight `weight` and the GDN gates
    /// `alpha_beta` under `switches`.
    fn layout_of(
        model: &Model,
        weight: WeightClass,
        alpha_beta: WeightClass,
        switches: &GemmSwitches,
    ) -> PrefillWorkspaceLayout {
        let ab_cache = model.moe && alpha_beta == WeightClass::Q8Raw;
        model.layout(ConversionScratch::plan(
            &model.gemms(weight, alpha_beta, ab_cache),
            switches,
            ROWS,
            &model.scratch(),
        ))
    }

    /// The default route set of each served model and scheme, at 2048 rows
    /// and 32768 keys.
    fn route_sets() -> Vec<(String, PrefillWorkspaceLayout)> {
        use WeightClass::*;
        let mut sets = Vec::new();
        for model in [Q27B, Q9B] {
            for (scheme, weight, alpha_beta) in [
                ("Q8_0", Q8Raw, Q8Raw),
                ("BF16", Bf16Raw, Q8Raw),
                ("NVFP4", Planar, F32),
                ("F32", F32, F32),
            ] {
                sets.push((
                    format!("{} {scheme}", model.name),
                    layout_of(&model, weight, alpha_beta, &dense_switches()),
                ));
            }
        }
        sets.push((
            "35B-A3B Q8_0".into(),
            layout_of(&MOE, Q8Raw, Q8Raw, &moe_switches(false)),
        ));
        sets.push((
            "35B-A3B BF16".into(),
            layout_of(&MOE, Bf16Raw, Q8Raw, &moe_switches(true)),
        ));
        sets
    }

    /// Each route set and its variants on the paths a load can also take:
    /// the plain attention, the per-token GDN loop, no score block, a dense
    /// and a MoE FFN in one model.
    fn every_route_set() -> Vec<(String, PrefillWorkspaceLayout)> {
        let mut sets = route_sets();
        for (name, layout) in route_sets() {
            let mut plain = layout;
            plain.routes.plain_attention = true;
            let mut per_token = layout;
            per_token.routes.gdn_fused = false;
            let mut scalar = layout;
            scalar.score_elems = None;
            let mut mixed = layout;
            mixed.routes.dense_ffn = true;
            mixed.routes.moe_ffn = true;
            sets.extend([
                (format!("{name} + plain attention"), plain),
                (format!("{name} + per-token GDN"), per_token),
                (format!("{name} + scalar attention"), scalar),
                (format!("{name} + dense and MoE FFN"), mixed),
            ]);
        }
        sets
    }

    /// A route's scratch: a GEMM's conversion scratch, the grouped MoE
    /// FFN's, held only when some layer takes the route.
    fn route_scratch(role: Role) -> bool {
        matches!(
            role,
            Role::DequantF32
                | Role::ActivationF16
                | Role::DequantF16
                | Role::MoeSharedSwiglu
                | Role::MoeSharedLogit
        ) || Role::MOE_ROUTED.contains(&role)
    }

    /// At every operation of a slice, the compact regions of the values live
    /// there share no byte; every value an operation on a taken path touches
    /// has a region (a route's scratch only where some layer takes the
    /// route); and every value is written before it is read.
    #[test]
    fn the_values_live_at_each_operation_lie_apart() {
        for (name, layout) in every_route_set() {
            let table = liveness(&layout);
            let regions = compact(&layout, &table);
            let live = live(&table);
            for (s, step) in table.iter().enumerate() {
                let here: Vec<(Role, Region)> = Role::ALL
                    .into_iter()
                    .filter(|&role| live[role as usize][s])
                    .filter_map(|role| regions.get(role).map(|r| (role, r)))
                    .collect();
                for i in 0..here.len() {
                    for j in 0..i {
                        let ((a, ra), (b, rb)) = (here[j], here[i]);
                        assert!(
                            ra.end(a) <= rb.offset || rb.end(b) <= ra.offset,
                            "{name}: {}: {a:?} and {b:?} share bytes",
                            step.name
                        );
                    }
                }
                for &role in step.reads.iter().chain(step.writes).chain(step.scratch) {
                    if !route_scratch(role) {
                        assert!(
                            regions.get(role).is_some(),
                            "{name}: {} touches {role:?}, which has no region",
                            step.name
                        );
                    }
                }
            }
            for role in [Role::DequantF32, Role::ActivationF16, Role::DequantF16] {
                assert_eq!(
                    regions.get(role).map(|r| r.len),
                    compact_len(&layout, role),
                    "{name}: {role:?}"
                );
            }
            for role in Role::ALL {
                if let Some(r) = regions.get(role) {
                    assert_eq!(r.offset % REGION_ALIGN, 0, "{name}: {role:?}");
                }
            }
        }
    }

    /// The compact layout never takes more than the unshared one, and holds
    /// a value exactly where the unshared one does, except the values no path
    /// writes and the conversion scratch no route takes.
    #[test]
    fn the_compact_layout_is_never_larger_than_the_unshared_one() {
        for (name, layout) in every_route_set() {
            let compact = compact(&layout, &liveness(&layout));
            let unshared = unshared(&layout);
            assert!(compact.bytes() <= unshared.bytes(), "{name}");
            for role in Role::ALL {
                if compact.get(role).is_some() {
                    assert!(unshared.get(role).is_some(), "{name}: {role:?}");
                }
            }
            assert_eq!(
                compact.get(Role::Scores).is_some(),
                layout.score_elems.is_some(),
                "{name}"
            );
            assert_eq!(
                compact.get(Role::QGateFused).is_some(),
                layout.scratch.q_gate.is_some(),
                "{name}"
            );
            assert_eq!(compact.get(Role::QSingle), None, "{name}");
            for role in Role::MOE_ROUTED
                .into_iter()
                .chain([Role::MoeSharedSwiglu, Role::MoeSharedLogit])
            {
                assert_eq!(
                    compact.get(role).map(|r| r.len),
                    layout.grouped.and_then(|g| g.len(role)),
                    "{name}: {role:?}"
                );
            }
        }
    }

    #[cfg(feature = "test-state-snapshot")]
    #[test]
    fn the_test_switch_selects_the_unshared_layout() {
        let (_, mut layout) = route_sets().remove(0);
        assert_eq!(layout.regions(), compact(&layout, &liveness(&layout)));
        layout.unshared = true;
        assert_eq!(layout.regions(), unshared(&layout));
    }

    /// The compact arena of each served model's default route set at 2048
    /// rows and 32768 keys, and the values whose bytes stack to its end.
    /// `f32(dim)` is a [2048, dim] F32 value; every byte count is a multiple
    /// of the region alignment, so the stacks hold no padding. The unshared
    /// arena holds what one buffer per value takes: 2_050_547_712 B on the
    /// 27B and 1_376_821_248 B on the 9B; on the 35B-A3B 1_141_940_224 B and
    /// the grouped MoE scratch's 215_512_068 B (W10 on), with 252 B of
    /// padding after the expert bounds' 257 x 4 bytes.
    #[test]
    fn the_compact_sizes_of_the_served_models() {
        let f32 = |dim: usize| ROWS * dim * 4;
        let want = [
            // The up GEMM's F16 arm (X, Normed, AttnProj, Gate, Up, the F16
            // pair) holds 660_602_880 B; placed largest first, AttnOut (live
            // with the pair in `wo`) lands above the pair and AttnProj above
            // AttnOut: X + DequantF16 (17408 x 5120 two-byte elements) + Gate
            // + Up + ActivationF16 (2048 x 17408 two-byte elements) + AttnOut
            // + AttnProj.
            (
                "27B Q8_0",
                f32(5120)
                    + 17408 * 5120 * 2
                    + 2 * f32(17408)
                    + ROWS * 17408 * 2
                    + f32(6144)
                    + f32(5120),
                668_991_488,
            ),
            // The tiled attention: X, the score block (6 x 512 x 32768), Q,
            // AttnOut and the gate.
            (
                "27B BF16",
                f32(5120) + 6 * 512 * 32768 * 4 + 3 * f32(6144),
                595_591_168,
            ),
            // The up GEMM's F32 arm: X, the F32 scratch (17408 x 5120),
            // Normed, Gate, Up, AttnProj.
            (
                "27B NVFP4",
                f32(5120) + 17408 * 5120 * 4 + 2 * f32(17408) + 2 * f32(5120),
                767_557_632,
            ),
            (
                "27B F32",
                f32(5120) + 6 * 512 * 32768 * 4 + 3 * f32(6144),
                595_591_168,
            ),
            // The up GEMM's F16 arm with AttnProj: X, Normed, AttnProj, Gate,
            // Up, the F16 pair (12288 x 4096 and 2048 x 12288 two-byte
            // elements).
            (
                "9B Q8_0",
                3 * f32(4096) + 2 * f32(12288) + 12288 * 4096 * 2 + ROWS * 12288 * 2,
                452_984_832,
            ),
            // The tiled attention: X, the score block (4 x 512 x 32768), Q,
            // AttnOut and the gate.
            (
                "9B BF16",
                f32(4096) + 4 * 512 * 32768 * 4 + 3 * f32(4096),
                402_653_184,
            ),
            // The up GEMM's F32 arm: X, the F32 scratch (12288 x 4096),
            // Normed, Gate, Up, AttnProj.
            (
                "9B NVFP4",
                f32(4096) + 12288 * 4096 * 4 + 2 * f32(12288) + 2 * f32(4096),
                503_316_480,
            ),
            (
                "9B F32",
                f32(4096) + 4 * 512 * 32768 * 4 + 3 * f32(4096),
                402_653_184,
            ),
            // The tiled attention: X, the score block (8 x 512 x 32768), Q,
            // AttnOut and the gate (q_dim 4096).
            (
                "35B-A3B Q8_0",
                f32(2048) + 8 * 512 * 32768 * 4 + 3 * f32(4096),
                654_311_424,
            ),
            (
                "35B-A3B BF16",
                f32(2048) + 8 * 512 * 32768 * 4 + 3 * f32(4096),
                654_311_424,
            ),
        ];
        let sets = route_sets();
        assert_eq!(sets.len(), want.len());
        for ((name, layout), (want_name, arithmetic, bytes)) in sets.iter().zip(want) {
            assert_eq!(name, want_name);
            assert_eq!(arithmetic, bytes, "{name}");
            assert_eq!(layout.bytes(), bytes as u64, "{name}");
            let one_buffer_each = match &name[..3] {
                "27B" => 2_050_547_712,
                "9B " => 1_376_821_248,
                _ => 1_141_940_224 + 215_512_068 + 252,
            };
            assert_eq!(unshared(layout).bytes(), one_buffer_each, "{name}");
        }
    }
}
