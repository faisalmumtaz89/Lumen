//! Semantic coherence detector.
//!
//! Formalises the ad-hoc gibberish detector ("does the output
//! collapse into single-token repetition or non-printable garbage?") into
//! a small reusable scoring function.  This catches the 跟跟跟… /
//! 心里心里… patterns that the 16-token reference-match gate misses
//! when the gibberish only starts after position 16.
//!
//! The detector is intentionally simple and language-agnostic:
//! 1. **Repetition score** = unique_tokens / total_tokens. A perfectly
//!    diverse output scores 1.0; the `跟跟跟` collapse scores
//!    roughly 1/N which is near-zero for N≥10.
//! 2. **Printable-text ratio** = fraction of decoded bytes that fall in
//!    the "printable" class (letters, digits, whitespace, common
//!    punctuation, and any byte ≥ 0x80 since multi-byte UTF-8 is
//!    accepted as text by Unicode-aware downstream parsers).
//!
//! The default gate (`pass`) is:
//! `repetition_score ≥ 0.30  AND  printable_text_ratio ≥ 0.70`
//!
//! Tuning rationale: pathological `跟跟跟…` 128-token outputs hit
//! repetition_score ≈ 0.01 (well below 0.30), and a coherent Qwen3.5
//! completion (mixed CJK + ASCII) hits repetition_score ≈ 0.6-0.95.
//! 0.30 is a permissive lower bound that catches collapse without
//! flagging legitimately repetitive content (e.g. numbered lists).
//!
//! Cost: O(N) over tokens + O(B) over decoded bytes.  Safe to call
//! end-of-generation; MUST NOT be invoked inside the per-token decode
//! loop.

/// Outcome of running the coherence detector on one completion.
///
/// All fields are present (no Option) so the row can be JSON-serialised
/// without conditional branches.  `pass` collapses the two sub-scores
/// against the documented default gate, which lets callers route the
/// row into the results matrix without re-implementing the
/// threshold logic.
#[derive(Debug, Clone, PartialEq)]
pub struct CoherenceVerdict {
    /// `unique_tokens / total_tokens` in [0.0, 1.0].  Zero when
    /// `tokens.is_empty()`.
    pub repetition_score: f64,

    /// Fraction of decoded bytes that look like printable text in
    /// [0.0, 1.0].  Zero when `decoded.is_empty()`.
    pub printable_text_ratio: f64,

    /// Number of input tokens that the verdict was computed over.
    /// Exposed so harnesses can record the sample size next to the
    /// score (a 5-token sample at 0.30 repetition is noise; a 128-token
    /// sample at 0.30 is a real signal).
    pub n_tokens: usize,

    /// Number of decoded bytes the printable-text ratio was computed
    /// over.  Same rationale as `n_tokens`.
    pub n_decoded_bytes: usize,

    /// Pass/fail against the default gate: `repetition_score ≥ 0.30
    /// AND printable_text_ratio ≥ 0.70`.
    pub pass: bool,
}

/// Default repetition lower bound.  See module docs for tuning rationale.
pub const DEFAULT_REPETITION_MIN: f64 = 0.30;

/// Default printable-text lower bound.  See module docs for tuning
/// rationale.
pub const DEFAULT_PRINTABLE_MIN: f64 = 0.70;

/// Run the coherence detector against one generated completion.
///
/// `tokens` is the generated token-ID slice (NOT including the prompt).
/// `decoded` is the detokenised text (caller-supplied so this module
/// stays decoder-agnostic; the original `tokenizer: &Tokenizer` arg is
/// expressed here as the already-decoded string).
///
/// Returns a [`CoherenceVerdict`] using the default gate thresholds.
/// Callers needing custom thresholds can inspect the sub-scores and
/// re-apply their own gate.
pub fn coherence_score(tokens: &[u32], decoded: &str) -> CoherenceVerdict {
    let repetition_score = repetition(tokens);
    let printable_text_ratio = printable_ratio(decoded);
    let pass = repetition_score >= DEFAULT_REPETITION_MIN
        && printable_text_ratio >= DEFAULT_PRINTABLE_MIN;
    CoherenceVerdict {
        repetition_score,
        printable_text_ratio,
        n_tokens: tokens.len(),
        n_decoded_bytes: decoded.len(),
        pass,
    }
}

/// Fraction of unique token IDs over total tokens.  Returns 0.0 on
/// empty input so the verdict's `pass` short-circuits to false (a
/// zero-token completion is not "coherent", it is"absent".
fn repetition(tokens: &[u32]) -> f64 {
    if tokens.is_empty() {
        return 0.0;
    }
    // Linear-time unique count via a small HashSet.  For typical
    // N=128 decode windows this allocates <2 KB and runs in microseconds.
    use std::collections::HashSet;
    let mut seen = HashSet::with_capacity(tokens.len());
    for &t in tokens {
        seen.insert(t);
    }
    seen.len() as f64 / tokens.len() as f64
}

/// Fraction of decoded bytes that fall in the printable set:
/// - any ASCII letter / digit / common-punctuation byte, OR
/// - any byte ≥ 0x80 (multi-byte UTF-8 lead/continuation).
///
/// Newlines (`\n`), carriage returns (`\r`), tabs (`\t`), and the
/// space character (` `) all count as printable.  Other ASCII control
/// bytes (0x00-0x1F minus the above, plus 0x7F) count as non-printable.
fn printable_ratio(decoded: &str) -> f64 {
    let bytes = decoded.as_bytes();
    if bytes.is_empty() {
        return 0.0;
    }
    let mut printable = 0usize;
    for &b in bytes {
        if is_printable_byte(b) {
            printable += 1;
        }
    }
    printable as f64 / bytes.len() as f64
}

/// True iff `b` is part of the printable set described in
/// `printable_ratio`.  Inlined for the byte-counting hot loop.
#[inline]
fn is_printable_byte(b: u8) -> bool {
    // ASCII printable range (space..tilde inclusive).
    if (0x20..=0x7E).contains(&b) {
        return true;
    }
    // Common whitespace controls.
    if b == b'\n' || b == b'\r' || b == b'\t' {
        return true;
    }
    // Multi-byte UTF-8 lead / continuation bytes.  We do NOT validate
    // codepoints here — the caller passed a `&str` so the bytes are
    // already valid UTF-8 by construction.
    if b >= 0x80 {
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_fails() {
        // No tokens, no decoded text => zero scores, gate fails.
        let v = coherence_score(&[], "");
        assert_eq!(v.repetition_score, 0.0);
        assert_eq!(v.printable_text_ratio, 0.0);
        assert_eq!(v.n_tokens, 0);
        assert_eq!(v.n_decoded_bytes, 0);
        assert!(!v.pass);
    }

    #[test]
    fn perfectly_unique_tokens_pass_repetition() {
        // 10 distinct token IDs => repetition_score = 1.0.
        let tokens: Vec<u32> = (0u32..10u32).collect();
        let v = coherence_score(&tokens, "The quick brown fox jumps over the lazy dog.");
        assert!(
            (v.repetition_score - 1.0).abs() < 1e-9,
            "repetition_score={}",
            v.repetition_score
        );
        assert!(v.printable_text_ratio >= 0.99);
        assert!(v.pass, "well-formed completion must pass: {v:?}");
    }

    #[test]
    fn single_token_repeated_fails_repetition() {
        // FIX-3 跟跟跟… pattern: 100 copies of one token.
        let tokens = vec![42u32; 100];
        let v = coherence_score(&tokens, "跟跟跟跟跟跟跟跟跟跟");
        // repetition_score = 1 / 100 = 0.01, well below 0.30.
        assert!(
            v.repetition_score < 0.05,
            "expected ≪0.05 for monoculture, got {}",
            v.repetition_score
        );
        // Decoded text IS printable (multi-byte UTF-8 ≥0x80) — so the
        // failure mode is purely the repetition gate.
        assert!(v.printable_text_ratio >= 0.99);
        assert!(!v.pass, "monoculture must fail: {v:?}");
    }

    #[test]
    fn non_printable_garbage_fails_printable() {
        // 50 distinct tokens (passes repetition) but decoded text is
        // pure non-printable bytes (NUL / SOH / etc.).
        let tokens: Vec<u32> = (0u32..50u32).collect();
        // 100 NUL bytes.  Wrap in a String safely via from_utf8.
        let bad = String::from_utf8(vec![0u8; 100]).unwrap();
        let v = coherence_score(&tokens, &bad);
        assert!(v.repetition_score >= 0.99, "{v:?}");
        assert_eq!(v.printable_text_ratio, 0.0);
        assert!(!v.pass);
    }

    #[test]
    fn realistic_cjk_repetition_gibberish_fails() {
        // Synthesise a realistic post-corruption failure mode: the output is
        // mostly one repeated CJK token.  We model it as 120 copies of `跟`
        // token-id 5396 + 8 distinct lead-in tokens.
        let mut tokens = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
        tokens.extend(std::iter::repeat(5396u32).take(120));
        let decoded = "好的，我来回答这个问题。".to_string()
            + &"跟".repeat(120);
        let v = coherence_score(&tokens, &decoded);
        // Unique = 9; total = 128 → 9/128 ≈ 0.07, below 0.30.
        assert!(
            v.repetition_score < 0.30,
            "CJK-repetition gibberish must trip repetition gate: {v:?}"
        );
        assert!(!v.pass, "CJK-repetition gibberish must FAIL: {v:?}");
    }

    #[test]
    fn realistic_good_completion_passes() {
        // Diverse multi-byte UTF-8 + ASCII.  Tokens model a ~50-token
        // decode window with ~40 distinct IDs (realistic for English
        // narrative text).
        let tokens: Vec<u32> = (1u32..=50u32).chain(std::iter::repeat(7u32).take(10)).collect();
        let decoded = "The answer is 42 because computational complexity \
                       theory predicts that exhaustive search over a finite \
                       state space terminates.";
        let v = coherence_score(&tokens, decoded);
        // 50 unique / 60 total ≈ 0.83 ≥ 0.30 ✓
        assert!(v.repetition_score >= 0.30, "{v:?}");
        assert!(v.printable_text_ratio >= 0.99, "{v:?}");
        assert!(v.pass, "well-formed mixed text must pass: {v:?}");
    }

    #[test]
    fn whitespace_controls_are_printable() {
        // Newlines, tabs, CR must NOT be treated as garbage.
        let tokens = vec![1u32, 2, 3, 4, 5];
        let decoded = "line1\nline2\tcol2\rline3";
        let v = coherence_score(&tokens, decoded);
        assert!(
            v.printable_text_ratio >= 0.99,
            "whitespace controls must be printable: ratio={}",
            v.printable_text_ratio
        );
    }

    #[test]
    fn boundary_at_default_thresholds() {
        // Exactly at the gate: 3/10 = 0.30 repetition + 0.70 printable.
        // Construct 10 tokens with 3 unique.
        let tokens = vec![1u32, 2, 3, 1, 1, 1, 1, 1, 1, 1];
        // 10 bytes: 7 printable ('a') + 3 NUL.
        let mut s = "aaaaaaa".to_string();
        s.push_str(&String::from_utf8(vec![0u8; 3]).unwrap());
        let v = coherence_score(&tokens, &s);
        assert!(
            (v.repetition_score - 0.30).abs() < 1e-9,
            "exact-threshold repetition: {}",
            v.repetition_score
        );
        assert!(
            (v.printable_text_ratio - 0.70).abs() < 1e-9,
            "exact-threshold printable: {}",
            v.printable_text_ratio
        );
        assert!(v.pass, "≥ thresholds must pass: {v:?}");
    }

    #[test]
    fn sub_threshold_repetition_fails_with_printable_ok() {
        // Repetition below threshold even though printable text is
        // perfect — single-axis failure.
        let tokens = vec![1u32; 100]; // repetition = 0.01
        let v = coherence_score(&tokens, "perfectly printable English text.");
        assert!(v.printable_text_ratio >= 0.99);
        assert!(!v.pass);
    }
}
