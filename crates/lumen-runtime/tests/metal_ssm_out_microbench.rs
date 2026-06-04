//! Standalone microbench: is `gdn/ssm_out_gemm` REALLY 9.8 ms/layer in the
//! GDN chain, or is the deep-profile number an artifact of the per-CB split?
//!
//! Background:
//!   The Metal deep-profile (`LUMEN_METAL_PROFILE_GDN=1`) reported
//!   `gdn/ssm_out_gemm` at ~9.8 ms/layer (82% of the GDN megakernel total)
//!   while the IDENTICAL-shape `gdn/attn_gate_gemm` ran at ~0.43 ms/layer
//!   with the plain k64-aligned kernel.
//!
//!   Both dispatches process [M=128, N=4096, K=4096] Q8_0 weights through
//!   the SAME TILE_K=64 family. The structural differences are:
//!     - ssm_out uses the `_residual_batched` variant (fused R add at writeback).
//!     - ssm_out is the LAST GEMM in the GDN chain (its weight tensor is cold).
//!     - ssm_out has a true RAW dependency on the immediately-prior phase2b kernel.
//!
//! What this bench does:
//!   Runs both kernel variants over the same shape/buffer pattern OUTSIDE the
//!   GDN megakernel chain, in two timing modes:
//!     - Single CB with N iterations  (matches production: one commit_and_wait)
//!     - Per-iteration CB             (matches the deep-profile split)
//!
//! Verdict logic:
//!   - If both kernels finish near 0.5 ms in single-CB mode (production path):
//!     the deep-profile 9.8 ms is an artifact (likely H3 / H4 — the per-CB sync
//!     overhead or the RAW dependency, not the kernel itself).
//!   - If the residual kernel is genuinely ~20x slower in single-CB mode:
//!     H1 is real (the residual variant has a structural bug at this shape).
//!
//! Run:
//!   cargo test --release --test metal_ssm_out_microbench -- --ignored --nocapture
//!
//! Output is also saved to `target/metal/ssm-out-verify.md`.

#![cfg(target_os = "macos")]

use lumen_runtime::metal::MetalF32Backend;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

const WARMUP_ITERS: usize = 20;
const TIMED_ITERS: usize = 100;

#[ignore = "metal ssm_out microbench: opt-in only"]
#[test]
fn metal_ssm_out_microbench() {
    let mut backend = MetalF32Backend::new().expect("Metal backend must be available on macOS");

    let report = backend
        .run_ssm_out_microbench(WARMUP_ITERS, TIMED_ITERS)
        .expect("ssm_out microbench failed");

    // ------------------- formatted console report -------------------
    let mut s = String::new();
    s.push_str("\n# Metal ssm_out microbench: is ssm_out_gemm really 9.8 ms/layer, or a profile artifact?\n\n");
    s.push_str(&format!("Device:       {}\n", report.device_name));
    s.push_str(&format!(
        "Shape:        M={} N={} K={} (matches Qwen3.5-9B ssm_out)\n",
        report.m, report.n, report.k
    ));
    s.push_str(&format!(
        "Q8_0 weight:  {:.2} MB (= N*K/32*34)\n",
        report.weight_mb
    ));
    s.push_str(&format!("Warmup:       {WARMUP_ITERS} iter (discarded)\n"));
    s.push_str(&format!("Timed:        {TIMED_ITERS} iter / scenario\n\n"));

    s.push_str("## Scenario summary (single-CB == production path)\n\n");
    s.push_str(&format!(
        "{:<28}  {:>16}  {:>16}  {:>16}\n",
        "scenario", "1CB total (ms)", "1CB per call ms", "per-CB med ms"
    ));
    s.push_str(&format!(
        "{:-<28}  {:->16}  {:->16}  {:->16}\n",
        "", "", "", ""
    ));
    for sc in &report.scenarios {
        s.push_str(&format!(
            "{:<28}  {:>16.3}  {:>16.4}  {:>16.4}\n",
            sc.label, sc.single_cb_total_ms, sc.single_cb_per_call_ms, sc.per_cb_median_ms
        ));
    }
    s.push('\n');

    // Per-trial detail (per-CB only — single-CB has only one trial)
    s.push_str("## Per-CB per-trial detail (ms) — first 10 of each scenario\n\n");
    for sc in &report.scenarios {
        s.push_str(&format!("- {} (median = {:.4} ms):\n  ", sc.label, sc.per_cb_median_ms));
        let take = sc.per_cb_trials_ms.iter().take(10).copied().collect::<Vec<_>>();
        s.push_str(
            &take
                .iter()
                .map(|v| format!("{:.4}", v))
                .collect::<Vec<_>>()
                .join("  "),
        );
        s.push_str("\n\n");
    }

    // ------------------- verdict -------------------
    let plain = &report.scenarios[0];
    let resid_rand = &report.scenarios[1];
    let resid_zero = &report.scenarios[2];

    let plain_pc = plain.single_cb_per_call_ms;
    let resid_pc = resid_rand.single_cb_per_call_ms;
    let zeros_pc = resid_zero.single_cb_per_call_ms;
    let perdiff_resid_vs_plain = resid_pc / plain_pc.max(1e-6);
    let perdiff_zeros_vs_resid = (zeros_pc / resid_pc.max(1e-6) - 1.0).abs();
    let percb_resid_vs_plain = resid_rand.per_cb_median_ms / plain.per_cb_median_ms.max(1e-6);

    s.push_str("## Verdict\n\n");
    s.push_str(&format!(
        "Single-CB ratio (residual / plain):     {:.2}x\n",
        perdiff_resid_vs_plain
    ));
    s.push_str(&format!(
        "Single-CB ratio (zeros R / random R):   {:.2}x ({} from random)\n",
        zeros_pc / resid_pc.max(1e-6),
        if perdiff_zeros_vs_resid < 0.10 { "<10% delta" } else { ">=10% delta" }
    ));
    s.push_str(&format!(
        "Per-CB med  ratio (residual / plain):   {:.2}x\n\n",
        percb_resid_vs_plain
    ));

    // Decision tree based on the verdict thresholds in the task description:
    //   - "If both kernels finish in ~0.5 ms each: artifact (H4/H3)"
    //   - "If the residual kernel is genuinely ~20x slower: H1 is real"
    let verdict = if perdiff_resid_vs_plain >= 5.0 {
        // Residual variant is structurally slow at the kernel level.
        "H1 SUPPORTED — residual variant is structurally slower than the plain sibling \
         on this exact shape, even outside the GDN chain."
    } else if perdiff_resid_vs_plain <= 2.0 && resid_pc < 2.0 {
        // Kernel itself is fast in production-like mode. The deep-profile
        // 9.8 ms came from the split path, not from this kernel.
        "ARTIFACT — both kernels finish in <2 ms / call in the production single-CB path. \
         The deep-profile 9.8 ms attribution is an artifact of the per-CB split (H3/H4): \
         either the RAW dependency on phase2b (the split serialises what was hidden by \
         Apple's GPU scheduler in production), or the per-CB sync drain cost being \
         attributed to the LAST dispatch."
    } else {
        // Mid-range: residual is slower but not dramatically so. The slowdown
        // is partly real (likely H2 / H5) but not the dominant factor.
        "INCONCLUSIVE — residual variant runs slower than the plain sibling but \
         not dramatically so. Likely H2 (cold weight cache) or H5 (one-time pipeline \
         state binding) accounts for part of the gap; the rest is profile artifact."
    };
    s.push_str(&format!("**{verdict}**\n\n"));

    // Headroom estimate: if the artifact hypothesis holds, what is the
    // expected prefill speedup once the deep-profile split overhead is
    // removed? Per the deep-profile: 24 layers * 9.8 ms = 235 ms of
    // "ssm_out time". If real kernel is ~0.5 ms, the true ssm_out cost
    // is ~12 ms.
    if perdiff_resid_vs_plain <= 2.0 {
        let true_ssm_ms = resid_pc * 24.0;
        let saved_ms = 235.0 - true_ssm_ms;
        s.push_str(&format!(
            "Headroom: per the deep-profile the 9.8 ms × 24 = 235 ms attribution. \
             If the true kernel cost is the measured {:.4} ms × 24 = {:.1} ms, that \
             REDUCES the reported prefill share by ~{:.1} ms — but **this saving is \
             not realised in production** because the production prefill already \
             pays only the true kernel cost (single CB). The 9.8 ms was \
             profile-only inflation.\n\n",
            resid_pc, true_ssm_ms, saved_ms
        ));
    } else {
        s.push_str(
            "Headroom: kernel-level slowdown is real. Investigate the residual \
             variant's writeback path (`gemm_q8_0.msl` lines 696-723) and weight-cache \
             warming as next steps.\n\n",
        );
    }

    // ------------------- emit -------------------
    print!("{s}");
    let _ = std::io::stdout().flush();

    let artifact_path = artifact_path();
    if let Some(parent) = artifact_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    fs::write(&artifact_path, &s).expect("failed to write microbench report");
    eprintln!("[microbench] full report saved to {}", artifact_path.display());

    // Sanity: the timings should be positive and the kernels should not panic.
    for sc in &report.scenarios {
        assert!(sc.single_cb_total_ms > 0.0);
        assert_eq!(sc.iterations, TIMED_ITERS);
        assert_eq!(sc.per_cb_trials_ms.len(), TIMED_ITERS);
    }
}

fn artifact_path() -> PathBuf {
    let cwd = std::env::current_dir().expect("cwd");
    // Tests run from crates/lumen-runtime; walk up to find the repo root.
    let mut p = cwd.clone();
    while !p.join("target").exists() && p.parent().is_some() {
        p = p.parent().unwrap().to_path_buf();
    }
    if !p.join("target").exists() {
        // Fall back to current working directory.
        return std::env::current_dir()
            .unwrap_or_default()
            .join("target")
            .join("metal")
            .join("ssm-out-verify.md");
    }
    p.join("target").join("metal").join("ssm-out-verify.md")
}
