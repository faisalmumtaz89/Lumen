//! The batched F32 prefill's workspace: one device allocation holding every
//! value the prefill keeps between kernels, each value (a [`Role`]) in its own
//! region of it.
//!
//! The workspace is sized for the longest slice a request can run, allocated
//! once at load and reused across every request and layer. Every region is a
//! `[rows, dim]` matrix (or a scratch) of which a slice uses its first rows. A
//! caller borrows the regions one operation needs with
//! [`Workspace::regions`], which refuses two regions that share a byte, so
//! the operands of one call never alias.

use cudarc::driver::{CudaSlice, CudaViewMut};

use crate::error::RuntimeError;

use super::ffi::CudaDevice;
use super::prefill::PrefillWorkspaceLayout;

/// The element type of a role's region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Elem {
    F32,
    U32,
    U8,
}

impl Elem {
    fn size(self) -> usize {
        match self {
            Elem::F32 | Elem::U32 => 4,
            Elem::U8 => 1,
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
    /// Single-token Q: [q_dim]. Never read.
    QSingle,
    /// Single-token attention output: [q_dim]. Never read.
    AttnOutSingle,
    /// F32 scratch for a dequantized projection weight, sized to the largest
    /// one; the operand of the SGEMM routes.
    DequantF32,
    /// F16 (or BF16) activation scratch for the HGEMM input conversion, in
    /// bytes: [batch * max_in_dim * 2].
    ActivationF16,
    /// F16 scratch for a dequantized projection weight, in bytes, sized to
    /// the largest one; the operand of the dequant-to-F16 HGEMM routes.
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
}

/// The number of roles.
pub(crate) const ROLES: usize = 28;

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
    ];

    pub(crate) fn elem(self) -> Elem {
        match self {
            Role::TokenIds => Elem::U32,
            Role::ActivationF16 | Role::DequantF16 => Elem::U8,
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RegionTable {
    regions: [Option<Region>; ROLES],
}

impl RegionTable {
    fn get(&self, role: Role) -> Option<Region> {
        self.regions[role as usize]
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
        let regions = roles.map(|role| {
            self.get(role)
                .unwrap_or_else(|| panic!("workspace: {role:?} has no region"))
        });
        for i in 0..N {
            for j in 0..i {
                let (a, b) = (roles[j], roles[i]);
                assert!(a != b, "workspace: {a:?} requested twice");
                let (ra, rb) = (regions[j], regions[i]);
                assert!(
                    ra.end(a) <= rb.offset || rb.end(b) <= ra.offset,
                    "workspace: {a:?} and {b:?} overlap"
                );
            }
        }
        regions
    }
}

/// The layout that gives every role `layout` has a value for its own region,
/// in [`Role::ALL`]'s order: the bytes of one buffer per value, each offset
/// rounded up to [`REGION_ALIGN`]. The Q+gate pair is present only when a
/// layer fuses them, the score block only when the tiled attention is the
/// route, the GDN values only for a model with GDN layers.
pub(crate) fn unshared(layout: &PrefillWorkspaceLayout) -> RegionTable {
    let s = &layout.scratch;
    let gdn = layout.gdn;
    let mut table = RegionTable::default();
    let mut end = 0usize;
    for role in Role::ALL {
        let len = match role {
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
        };
        if let Some(len) = len {
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
    pub(crate) fn regions<'a, const N: usize>(&'a mut self, roles: [Role; N]) -> [RoleView<'a>; N] {
        let regions = self.regions.disjoint(&roles);
        let mut whole = self.arena.as_view_mut();
        std::array::from_fn(|i| {
            let (role, region) = (roles[i], regions[i]);
            let mut bytes = whole.slice_mut(region.offset..region.end(role));
            // SAFETY: `disjoint` has checked that no two requested regions
            // share a byte, so the views alias no device memory, as the
            // results of `<[T]>::get_disjoint_mut` alias no element. Each
            // region's offset is a multiple of `REGION_ALIGN` and its byte
            // length is `len` elements of the role's type, so the typed view
            // is aligned and in bounds. A view carries only the arena's
            // device pointer, length and event and stream references, which
            // live as long as the borrow of `self`; the shorter lifetime
            // `slice_mut` and `transmute_mut` give it only records the
            // borrow of `whole`, which nothing uses after this closure.
            let view = unsafe {
                match role.elem() {
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
                    Elem::U8 => Typed::U8(std::mem::transmute::<
                        CudaViewMut<'_, u8>,
                        CudaViewMut<'a, u8>,
                    >(bytes)),
                }
            };
            RoleView { role, view }
        })
    }
}

/// A role's region, typed as the role's elements.
enum Typed<'a> {
    F32(CudaViewMut<'a, f32>),
    U32(CudaViewMut<'a, u32>),
    U8(CudaViewMut<'a, u8>),
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

    /// The region as bytes. Panics when the role holds another type.
    pub(crate) fn u8(&mut self) -> CudaViewMut<'_, u8> {
        match &mut self.view {
            Typed::U8(v) => v.slice_mut(..),
            _ => elem_mismatch(self.role, Elem::U8),
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
    use crate::cuda::prefill::{GdnPrefillScratchLayout, PrefillScratchLayout};

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
    fn token_ids_are_u32_the_conversion_scratch_bytes_the_rest_f32() {
        for role in Role::ALL {
            let want = match role {
                Role::TokenIds => Elem::U32,
                Role::ActivationF16 | Role::DequantF16 => Elem::U8,
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
        let layout = PrefillWorkspaceLayout {
            scratch,
            gdn: Some(GdnPrefillScratchLayout::new(3, 56, 4, 32)),
            score_elems: Some(77),
        };
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
        let layout = PrefillWorkspaceLayout {
            scratch: PrefillScratchLayout::new(3, 40, 24, 8, 72, None, None, false),
            gdn: None,
            score_elems: None,
        };
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
            );
            assert_eq!(t.get(role).is_none(), absent, "{role:?}");
        }
    }
}
