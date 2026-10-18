//! The CUDA load's memory plan: the device memory a model needs to serve a
//! context, part by part, sized from the allocators' own layouts and checked
//! before any of it is allocated. A driver that backs device memory with host
//! memory once the card is full (Windows under WSL) lets an allocation past
//! the card succeed, and every token then pages over PCIe; checking the plan
//! first turns that into a refusal that says what fits.

use crate::kv::KvPrecision;

/// The bytes one load allocates to serve its context, beyond the weights.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ServingPlan {
    /// The attention layers' KV caches.
    pub kv: u64,
    /// The F32 widening pair a 16-bit KV store's prefill readers take.
    pub widening: u64,
    /// The batched prefill's workspace for the longest slice and the context.
    pub workspace: u64,
    /// The GDN layers' recurrent and conv state and their decode scratch.
    pub gdn_state: u64,
    /// The grouped MoE prefill scratch.
    pub moe_grouped: u64,
    /// The F16 dequant caches (the full-attention projections' with their
    /// sizing margin, and the GDN alpha/beta copies when they are made).
    pub f16_caches: u64,
    /// The decode Q+gate pair of a model that fuses Q and gate.
    pub decode_scratch: u64,
    /// [`SERVING_ALLOWANCE_BYTES`].
    pub allowance: u64,
}

/// Device memory the plan keeps free beyond the buffers it counts: the
/// allocations the CUDA libraries and driver make for themselves and the
/// allocator's rounding of each buffer.
pub(crate) const SERVING_ALLOWANCE_BYTES: u64 = 128 * 1024 * 1024;

impl ServingPlan {
    pub(crate) fn total(&self) -> u64 {
        self.kv
            + self.widening
            + self.workspace
            + self.gdn_state
            + self.moe_grouped
            + self.f16_caches
            + self.decode_scratch
            + self.allowance
    }
}

/// Contexts the refusal suggests are multiples of this.
const CONTEXT_STEP: usize = 1024;

/// The refusal when serving `context` tokens at `precision` does not fit in
/// `budget` bytes, or `None` when it does. `plan` sizes the serving memory for
/// any context and precision; `budget` is the free device memory plus the part
/// of the plan already allocated (the widening pair `init` made). The message
/// names each part, the largest context that fits (a multiple of
/// [`CONTEXT_STEP`]), and, for an F32 store, whether BF16 KV fits.
pub(crate) fn refusal(
    plan: impl Fn(usize, KvPrecision) -> ServingPlan,
    context: usize,
    precision: KvPrecision,
    budget: u64,
) -> Option<String> {
    let need = plan(context, precision);
    if need.total() <= budget {
        return None;
    }
    let gb = |b: u64| b as f64 / 1.0e9;
    let mut parts = vec![
        format!("KV cache {:.2} GB", gb(need.kv + need.widening)),
        format!("prompt workspace {:.2} GB", gb(need.workspace)),
    ];
    for (label, bytes) in [
        ("linear-attention state", need.gdn_state),
        ("MoE prompt workspace", need.moe_grouped),
        ("F16 weight caches", need.f16_caches),
        ("decode scratch", need.decode_scratch),
        ("headroom", need.allowance),
    ] {
        if bytes > 0 {
            parts.push(format!("{label} {:.2} GB", gb(bytes)));
        }
    }
    let fits = |c: usize, p: KvPrecision| plan(c, p).total() <= budget;
    let largest = |p: KvPrecision| {
        (1..=context.saturating_sub(1) / CONTEXT_STEP)
            .rev()
            .map(|k| k * CONTEXT_STEP)
            .find(|&c| fits(c, p))
    };
    let mut advice = match largest(precision) {
        Some(c) => format!("The largest context that fits is {c} tokens (--context-len {c})"),
        None => format!(
            "No context of {CONTEXT_STEP} tokens or more fits beside these weights; \
             use a smaller quantization"
        ),
    };
    if precision == KvPrecision::F32 {
        if fits(context, KvPrecision::Bf16) {
            advice.push_str(&format!(
                "; with --kv-precision bf16 the {context}-token context fits"
            ));
        } else if let Some(c) = largest(KvPrecision::Bf16) {
            advice.push_str(&format!("; with --kv-precision bf16, {c} tokens fit"));
        }
    }
    Some(format!(
        "Not enough GPU memory to serve a {context}-token context: it needs {:.2} GB ({}) \
         and {:.2} GB is available after the weights. {advice}.",
        gb(need.total()),
        parts.join(", "),
        gb(budget),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A plan whose context-dependent parts grow by `per_token` bytes per
    /// token (half for a 16-bit store) over a fixed `base`.
    fn linear(base: u64, per_token: u64) -> impl Fn(usize, KvPrecision) -> ServingPlan {
        move |c, p| ServingPlan {
            kv: c as u64
                * if p == KvPrecision::F32 {
                    per_token
                } else {
                    per_token / 2
                },
            workspace: base,
            ..Default::default()
        }
    }

    #[test]
    fn a_plan_that_fits_is_admitted_to_the_byte() {
        let plan = linear(1000, 10);
        assert_eq!(refusal(&plan, 4096, KvPrecision::F32, 1000 + 40960), None);
        assert!(refusal(&plan, 4096, KvPrecision::F32, 1000 + 40959).is_some());
    }

    #[test]
    fn the_refusal_names_the_largest_context_that_fits_and_the_bf16_option() {
        // 32768 tokens need 2 GB of KV + 1 GB of workspace; 2.2 GB is available.
        let plan = linear(1_000_000_000, 65_536);
        let msg = refusal(&plan, 32768, KvPrecision::F32, 2_200_000_000).expect("refused");
        // 1e9 + c * 65536 <= 2.2e9  =>  c <= 18310, so 17408 is the largest step.
        assert!(
            msg.contains("The largest context that fits is 17408 tokens"),
            "{msg}"
        );
        assert!(msg.contains("--context-len 17408"), "{msg}");
        // BF16 halves the KV: 1e9 + 32768 * 32768 = 2.07e9 fits.
        assert!(
            msg.contains("with --kv-precision bf16 the 32768-token context fits"),
            "{msg}"
        );
        assert!(msg.contains("KV cache 2.15 GB"), "{msg}");
        assert!(msg.contains("prompt workspace 1.00 GB"), "{msg}");
        assert!(!msg.contains("linear-attention"), "{msg}");
    }

    #[test]
    fn a_16_bit_store_is_not_told_about_bf16_and_nothing_fitting_is_said() {
        let plan = linear(5_000_000_000, 65_536);
        let msg = refusal(&plan, 8192, KvPrecision::Bf16, 1_000_000_000).expect("refused");
        assert!(
            msg.contains("No context of 1024 tokens or more fits"),
            "{msg}"
        );
        assert!(!msg.contains("--kv-precision"), "{msg}");
    }
}
