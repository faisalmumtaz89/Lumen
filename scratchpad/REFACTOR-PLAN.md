# Fold-in refactor — design of record

Adopted after codex-sol consult #17 (fold PR 2 + PR 3 into one release-ready
CUDA PR) and verification of its four defect claims, all TRUE.

## The default activation policy — topology-derived, no env

`q4_decode_f32_act = has_gdn && gdn_dims().num_v_heads == 32` is ALREADY derived
from model topology (backend_impl.rs:17281). It pins ALL SIX Q4 projection
families to F32 on the narrow-GDN class because ONE of them is fragile. The
campaign showed the quality cliff is concentrated in `wo`.

Recorded evidence in that comment block:
- 9B-Q4  (v_heads 32, dense): whole-model int8 => GQ 7/15. llama.cpp degenerates
  identically on the same GGUF, so this is engine-agnostic, not a Lumen defect.
- MoE-Q4 (v_heads 32, MoE):   whole-model int8 => GQ-001 13/15.
- 27B-Q4 (v_heads 48):        certified clean on int8, keeps dp4a.

So the narrow-GDN class contains BOTH 9B and MoE. The campaign validated ONLY
9B. Applying five-family Q8_1 to the whole class would silently change MoE,
where int8 was measured to fail.

Discriminant: `hp.num_experts.is_none()` (None = dense).

```
narrow_gdn && dense   -> AttnWo = F32, other five = Q8_1   [NEW - 9B-Q4]
narrow_gdn && !dense  -> all six F32                       [unchanged - MoE-Q4]
otherwise             -> all six Q8_1                      [unchanged - 27B, non-GDN]
```

Every branch has recorded evidence. Nothing generalises past what was measured.

## Structural changes

1. `Q4ActPlan::for_model(narrow_gdn, dense)` replaces `resolve_q4_act_plan()`.
   Delete `parse_zone`, `zone_contains`, the `q4_act_plan()` OnceLock,
   `LUMEN_CUDA_PRECISION_ZONE`, `LUMEN_CUDA_Q4_F16_ZONE`, and `is_default`.
2. Delete `Q4ActMode::F16` — nothing selects it; measured at parity with F32 on
   9B-Q4 with worse quality, so it is dead weight.
3. `KernelSet.q4_decode_f32_act: bool` -> `KernelSet.q4_act_plan: Q4ActPlan`.
   The bool's readers become `plan.mode_for(family) == F32`.
4. Levers become unconditional on typed prerequisites (no feature flags):
   - lane F32 kernel: plan says F32 for the family AND a sibling exists
   - Q8_1 split kernel: plan says Q8_1 AND prerequisites exist
   - direct residual: residual twin + down sibling exist
   Delete `LUMEN_CUDA_Q4_SPLIT`, `_ATTN`, `_F32`, `LUMEN_CUDA_FFN_DIRECT_RESIDUAL`.
5. The `ffn_down` Q8_1 exception DISSOLVES: FfnDown's plan mode is Q8_1, so the
   shortcut just reads the plan. This is the honest fix for the inconsistency
   whose "correction" cost 12% - model corrected to match reality, not reverse.
6. Split-clone budget = the one retained resource control:
   - unset/`auto` -> resource-aware cap
   - `0`          -> no siblings  (currently means "unset": `.filter(|gb| *gb > 0.0)`)
   - `>0`         -> requested cap, clamped to safe available
   - invalid/neg  -> startup error, never silent auto
   Remove the KV double-reserve: KV is allocated in `init()` (:13703) and the
   resolver runs inside `preload_weights` (:16978), which ERRORS unless init
   ran. `free_memory()` already excludes KV. The "before KV alloc" comment is
   false. Also purge the stale 5.1 GB floor from the docs (:12137).
7. `route_census_verify` must verify ALL SIX families incl. F32 ones; today
   `if want != Q4ActMode::F32` skips them (runtime_defaults.rs:2949).

## Interaction that must be MEASURED, not assumed

With the plan on, five families use Q8_1 split kernels, so the lane F32 kernel's
scope shrinks to `wo` only. The folded default is therefore NOT
"ship stack + plan" - it is a different configuration. Also, the campaign peak
157.97 included four levers dropped for falling under the repo's +1.5% gate
(GDN fused conv, K-quant requant, GDN T=1 W4, attn prep), so the folded default
will land below 157.97. Re-measure the exact no-env branch. Do not compose
marginal percentages.

## Evidence required before default-on (codex-sol, AH-11)

>= 100 DISTINCT prompts (not 100 reps of 3), paired across three arms:
candidate / all-F32 oracle / current production default, plus llama.cpp for the
fidelity gates. Mix: 25 short exact-answer, 20 reasoning+code, 10
structured/tool-call, 15 long-form (>=1200 tok, several 3072), 15 multi-turn
incl. growing prefix, 15 long-context/multilingual/degeneration-stress.

Pass criteria: every applicable GQ gate passes on every clean-process run, no
regression vs either control; GQ-005/006/007/013 must be REAL results, not
DEFERRED (run_suite.py currently defers them); zero candidate-only hard
failures; one-sided 95% bound on candidate-minus-reference quality above a
-5pp non-inferiority margin, with no prompt class showing an observed
regression; identical token-stream hashes across three fresh-process reruns;
route evidence for all six families incl. AttnWo's F32 route.

Resource cells: full siblings on A100, forced `budget=0`, auto-budget
small-VRAM card that fits the base model. No startup OOM, no numerical
dependence on sibling availability.

## Known pre-existing issue — NOT to be "fixed" here

`wo` runs through dp4a int8 on 27B/non-GDN models today. If wo's outliers
defeat per-32 block scaling structurally, that is a latent quality question on
those models. It is pre-existing shipped behaviour, outside this campaign's
measured scope. Record it; do not change it. (Changing shipped behaviour on the
strength of a plausible argument is exactly what cost 12% earlier.)
