# Ship evidence — CUDA 9B-Q4 activation policy + split layout

Branch `prep/cuda-9bq4-ship` @ 46368f2, off `origin/main` = 2c5ffea (v0.5.0).
**Local `main` is STALE (8291693, v0.4.0) — always diff against `origin/main`.**

## ROOT CAUSE — two deleted GDN recurrence launches (FIXED)

The branch generated garbage on 9B-Q4 under every configuration tested. It was
not the split layout and not the activation policy: `gdn_prefill_fused_v3
[_f64accum]` had been deleted from BOTH GDN sites, leaving only the step-4
comment. The norm-gate consumed buffers nothing had written and the recurrence
state never advanced.

Introduced by `b10a9d2 "wip: cleaned campaign branch (pre-scope-reduction)"`,
so it predates the activation-plan work. Found by codex-sol reading the diff.

Neither guard could have caught it: 605 tests pass with the launches missing
(none execute a CUDA GDN layer), and the route census certifies matvec
dispatch — the recurrence is not a matvec.

**Every throughput number measured on this branch before 8b89007 is void, and
was inflated**: 100.402 and 150.311 were fast partly because the model was
skipping its own recurrence. The v0.5.0 baselines are unaffected.

## CONFIRMED — on the repaired build (8b89007+)

### Decode throughput — paired A/B, coherence-gated, no env vars

Two builds, one container, one A100-80GB, interleaved. Every arm's generated
text is anchor-checked; an incoherent arm contributes no number.

| arm | a | b | c | mean | coherent |
|---|---:|---:|---:|---:|:--:|
| base (v0.5.0) | 87.399 | 88.091 | 87.600 | **87.696** | 3/3 |
| ship (folded default) | 147.235 | 147.109 | 147.047 | **147.130** | 3/3 |

**x1.6777 vs v0.5.0. x0.9598 vs llama.cpp 153.3** (that reference predates this
build; a same-container re-measure is still owed).

### Per-family int8 tolerance — every arm coherent

Each arm starts from `ffn_down` (always-shipped int8) and adds one family:

| arm | coherent | tok/s |
|---|:--:|---:|
| PROD-reference (v0.5.0) | yes | 88.87 |
| none (all F32) | yes | 98.52 |
| ffn_down | yes | 106.08 |
| ffn_down+attn_qkv | yes | 107.84 |
| ffn_down+ffn_gate_up | yes | **124.50** |
| ffn_down+gdn_qkv | yes | 109.50 |
| ffn_down+gdn_attn_gate | yes | 107.45 |
| all_but_wo (the shipped policy) | yes | 150.04 |

No family produces garbage. `ffn_gate_up` is the dominant one. NOTE: this gate
is 3 prompts — it proves NOT-garbage, not quality. The 101-prompt campaign is
what answers the quality question.

### Resource degradation

| cell | result |
|---|---|
| A100 auto (71.79 GB budget) | 32 layers / 172 jobs / 3.59 GB |
| A100 `budget=0` | 0 layers, 0.00 GB — runs on base kernels |
| A100 `budget=1GB` | 12 layers / 0.97 GB — partial, no OOM |
| L4 24 GB auto | 32 layers / 3.59 GB, no OOM |
| `budget=banana` / `budget=-4` | exit 30, no silent fallback |

### Route census

All six families verified on their planned mode, `wo` on F32 via
`F32_SPLIT_SOA_LANE_RES`. The verifier now also rejects silence on any family
the loaded model has Q4 weights for, and clears observations per model load.

### Tests / hygiene

606 runtime tests green. `cargo fmt --all -- --check` clean (branch had added
41 violations over origin/main's 4). `git diff --check` clean. Five dead-code
warnings, byte-identical to origin/main.

## QUALITY VERDICT — the five-family policy REGRESSES long-form

AH-11 campaign, 101 distinct prompts, 5 arms, candidate run 3x in fresh
processes. All three candidate runs byte-identical to each other; prod and
oracle byte-identical to each other.

| gate | prod (v0.5.0) | oracle (branch, all-F32) | candidate x3 |
|---|---|---|---|
| GQ-001 short | 24/25 | 24/25 | 24/25 |
| GQ-002 medium | 18/20 | 18/20 | 18/20 |
| GQ-004 verylong | 14/15 | 14/15 | **12/15** |
| GQ-004b degeneration | 2/3 | 2/3 | **1/3** |
| GQ-007/008/009/010/012 | pass | pass | pass |
| GQ-014 multi-turn | 15/15 | 15/15 | 15/15, determinism 15/15 |

CANDIDATE-ONLY failures, all three reruns: `vlong-explain-01` [DD-SPAM],
`vlong-story-02` [DD-REP], `stress-rep-01` [DD-REP], `med-explain-04` [DD-REP].

The oracle arm is the same branch — same kernels, same split layout, same lane
kernel — differing ONLY in activation precision. So this is not a kernel
defect: five-family int8 activations degrade long-form generation into
repetition. The whole-model F32 pin was protecting against exactly this.

**The three-prompt coherence gate could not have caught it**, and neither could
the first per-family bisect, which used the same gate and cleared every family.
That gate proves output is not garbage; this failure is fluent text collapsing
after a thousand tokens.

**The layout work is clean.** The oracle arm scores identically to v0.5.0 on
every gate while measuring 98.52 tok/s against prod's 88.87 (+10.9%).

Two corpus items fail on the SHIPPED baseline too — `vlong-howto-01` [DD-REP]
and `stress-rep-02` [DD-SPAM]. The second is my defect: it asks for 1..200 in
words, whose correct answer is inherently repetitive. Kept, and handled by
scoring arm-vs-arm on candidate-only failures rather than absolute thresholds.

## CO-LOCATED vs llama.cpp b10032 — same GPU, same container

| cell | Lumen | llama.cpp | vLLM | vs llama.cpp | vs vLLM |
|---|---:|---:|---:|---:|---:|
| 9b-q4 | 150.20 | 153.50 | 136.56 | **0.979** | **1.100** |
| 9b-q8 | 119.41 | 124.05 | 111.16 | 0.963 | 1.074 |

9b-q4 was 0.574x llama.cpp at campaign start. 9b-q8 was 0.955x on the v0.5.0
board, so no regression there. NOTE: this ran the candidate build, whose
activation policy is now known to regress long-form quality — the number is
real but it is not a shippable configuration.

## PER-FAMILY LONG-FORM BISECT — the instrument cannot rank families

Seven arms, each adding one family to `ffn_down`, gated on GQ-002/004/004b
(15 verylong prompts to 3072 tokens plus degeneration traps) rather than the
three short prompts the first bisect used.

| arm | GQ-002 | GQ-004 |
|---|---|---|
| none (all-F32) | 18/20 | 14/15 |
| ffn_down | 18/20 | 13/15 |
| + ffn_gate_up | 18/20 | 12/15 |
| + attn_qkv | 18/20 | **15/15** |
| + gdn_qkv | 18/20 | 13/15 |
| + gdn_attn_gate | **16/20** | 14/15 |
| all_but_wo | 18/20 | 12/15 |

**GQ-004 is non-monotonic.** Adding `attn_qkv` to `ffn_down` REPAIRED both of
`ffn_down`'s failures and beat the all-F32 control. That is irreconcilable with
int8 damaging long-form quality monotonically. Threshold-based repetition
detectors on 3000-token generations flip individual items in BOTH directions
under small numeric perturbation, so 15 binary items cannot separate these
configurations.

Consequently the earlier claim — "the five-family policy regresses long-form
quality" — is **RETRACTED as unproven**. The three byte-identical candidate
reruns established DETERMINISM, not statistical independence: three copies of
one sample, not three samples.

**Instrument validation:** `all_but_wo` reproduces the campaign candidate
exactly — 12/15, same three items, same GQ-002 failures as all three `cand`
runs. The probe build is faithful.

**The one non-noise signal:** `gdn_attn_gate` fails `med-reason-02` with
`ans✗` — a WRONG ANSWER, not a detector threshold flip. That is a different
class of evidence and the only per-family result worth acting on.

**What this needs to be answered properly:** a much larger long-form corpus, or
a graded degeneration score rather than a binary threshold. Neither exists yet.

## RECOMMENDATION

Ship the all-F32 configuration: **98.52 vs 88.87 tok/s (+10.9%)**, identical to
v0.5.0 on all ten gates, no open quality question. It carries the split layout,
the lane kernel, the direct-residual store, the restored GDN recurrence, the
budget-semantics fixes and the census fixes.

Hold the per-family activation policy. Its measured ceiling is real — 150.20
co-located, 0.979x llama.cpp, 1.100x vLLM — but resolving whether it costs
quality needs an instrument that does not yet exist.

## DEFERRED, EXPLICITLY

- Full 3-family CUDA matrix (27b, moe). Gated on the quality result: the
  change touches shared dispatch, so a policy revert would invalidate it.
- GQ-005/006 (llama fidelity) and GQ-013 (coherence judge) remain DEFERRED in
  `run_suite.py`; GQ-007/008/010 no longer are.

## KNOWN, NOT FIXED HERE

`wo` runs through dp4a int8 on 27B and non-GDN models today. If its
sigmoid-gated outliers defeat per-32 block scaling structurally, that is a
latent question on those models. It is pre-existing shipped behaviour outside
this campaign's measured scope. Changing shipped behaviour on the strength of
a plausible argument is what cost 12% of decode earlier on this branch.
