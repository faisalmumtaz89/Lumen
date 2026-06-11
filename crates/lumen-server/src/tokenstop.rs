//! Stop-sequence detection over decoded text.
//!
//! OpenAI-style APIs let callers pass a `stop` array. Generation halts when
//! the cumulative decoded text ends with any of those sequences. This is
//! pure substring matching on the emitter's output; tool-call and EOS
//! detection are handled elsewhere.

/// Rolling matcher for one or more textual stop sequences.
///
/// Conceptually a single byte stream rolls through the matcher. Bytes are
/// emitted to the client as soon as we are sure they cannot be part of a
/// future-completing stop sequence; the rest is held in `window` (up to
/// `max_stop_len - 1` bytes) so a stop sequence straddling two pushes is
/// still detected.
///
/// Returned text on a hit is the prefix BEFORE the matched sequence; the
/// matched bytes themselves are dropped (clients do not see the stop
/// string per OpenAI semantics).
pub struct StopMatcher {
    sequences: Vec<Vec<u8>>,
    max_len: usize,
    /// Held-back tail of the cumulative emit stream. Always shorter than
    /// `max_len`. Bytes in `window` have NOT yet been emitted to the user.
    window: Vec<u8>,
}

impl StopMatcher {
    /// Build a matcher from textual stop sequences.
    ///
    /// PRECONDITION: each sequence is valid UTF-8 (it is — both the JSON
    /// `stop` field, a `serde` `Value::String`, and the CLI `--stop` arg are
    /// guaranteed-valid Rust `String`s). UTF-8 self-synchronization then
    /// guarantees a valid stop string can never match aligned on a
    /// continuation byte, so the safe-prefix slices in `push`/`finish` always
    /// land on a char boundary. Those decodes use `from_utf8_lossy` anyway, so
    /// a future non-UTF-8 caller degrades gracefully rather than silently
    /// dropping a whole prefix.
    pub fn new(sequences: Vec<String>) -> Self {
        let bytes: Vec<Vec<u8>> = sequences.into_iter().map(|s| s.into_bytes()).collect();
        let max_len = bytes.iter().map(|s| s.len()).max().unwrap_or(0);
        Self {
            sequences: bytes,
            max_len,
            window: Vec::with_capacity(max_len.saturating_sub(1)),
        }
    }

    /// Push the next text fragment. Returns the prefix of the cumulative
    /// stream that is now safe to emit, and a flag indicating whether a
    /// stop was matched. Once `true` is returned the caller MUST stop.
    pub fn push(&mut self, fragment: &str) -> (String, bool) {
        if self.sequences.is_empty() {
            return (fragment.to_string(), false);
        }
        // Combine window + fragment. Bytes 0..window_len in `combined` were
        // held back from prior pushes (not yet emitted). Bytes window_len..
        // are fresh from this fragment. All of `combined` is currently
        // unflushed.
        let window_len = self.window.len();
        let mut combined = std::mem::take(&mut self.window);
        combined.extend_from_slice(fragment.as_bytes());

        // Find the earliest stop-sequence occurrence in `combined`.
        let mut best: Option<(usize, usize)> = None; // (match_start, match_end)
        for seq in &self.sequences {
            if seq.is_empty() {
                continue;
            }
            if let Some(pos) = find_subslice(&combined, seq) {
                let end = pos + seq.len();
                if best.map_or(true, |(_, prev_end)| end < prev_end) {
                    best = Some((pos, end));
                }
            }
        }

        if let Some((match_start, _)) = best {
            // Emit everything BEFORE the match. The matched bytes and any
            // trailing data are dropped per stop semantics. `from_utf8_lossy`
            // rather than `from_utf8(..).unwrap_or_default()`: for the
            // guaranteed-valid-UTF-8 stop strings this matcher is built from
            // (see `StopMatcher::new`) the prefix is always a clean boundary,
            // but lossy decoding degrades gracefully (it never silently drops
            // the WHOLE prefix) should a future caller ever feed bytes that
            // split a codepoint.
            let safe = String::from_utf8_lossy(&combined[..match_start]).into_owned();
            self.window.clear();
            // Suppress unused warning when window_len happens not to bind.
            let _ = window_len;
            return (safe, true);
        }

        // No match. Determine the largest suffix that COULD still complete
        // a stop sequence. That is the longest prefix of any stop sequence
        // that matches a suffix of `combined`. The held window must cover
        // at least that many bytes so the next push can complete it.
        let mut hold_len = 0usize;
        for seq in &self.sequences {
            let k = longest_suffix_prefix(&combined, seq);
            if k > hold_len {
                hold_len = k;
            }
        }
        // Belt-and-suspenders cap at max_len - 1 so the window never grows
        // indefinitely.
        hold_len = hold_len.min(self.max_len.saturating_sub(1));
        hold_len = hold_len.min(combined.len());

        let safe_end = combined.len() - hold_len;
        // Lossy decode (see the match-hit branch): graceful on a split
        // codepoint instead of dropping the whole safe prefix.
        let safe = String::from_utf8_lossy(&combined[..safe_end]).into_owned();
        self.window.extend_from_slice(&combined[safe_end..]);
        (safe, false)
    }

    /// Whether any stop sequences are configured.
    pub fn is_active(&self) -> bool {
        !self.sequences.is_empty()
    }

    /// Drain any held-back bytes as a final flush. Callers invoke this when
    /// the stream ends naturally (EOS or end-of-tokens) without having
    /// matched a stop sequence -- the held bytes are then safe to emit.
    pub fn finish(self) -> String {
        // Lossy decode (see `push`): the held window is byte-identical to the
        // input slices, so for the valid-UTF-8 stop strings this matcher is
        // built from it round-trips exactly; lossy only matters as graceful
        // degradation for a hypothetical non-UTF-8 caller.
        String::from_utf8_lossy(&self.window).into_owned()
    }
}

/// Naive substring search. Marker lengths are tiny so a memmem crate
/// would be overkill.
fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > hay.len() {
        return None;
    }
    let n = needle.len();
    let last = hay.len() - n;
    for i in 0..=last {
        if &hay[i..i + n] == needle {
            return Some(i);
        }
    }
    None
}

/// Longest k such that `tail` ends with `needle[..k]`. Used to decide how
/// many trailing bytes to hold back so a stop sequence straddling two
/// pushes is still detected.
fn longest_suffix_prefix(tail: &[u8], needle: &[u8]) -> usize {
    let max = tail.len().min(needle.len().saturating_sub(1));
    for k in (1..=max).rev() {
        if &tail[tail.len() - k..] == &needle[..k] {
            return k;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_stops_pass_through() {
        let mut m = StopMatcher::new(vec![]);
        let (t, stop) = m.push("hello");
        assert_eq!(t, "hello");
        assert!(!stop);
    }

    #[test]
    fn detects_complete_stop_in_one_fragment() {
        let mut m = StopMatcher::new(vec!["END".into()]);
        let (t, stop) = m.push("hi there END trailing");
        assert!(stop);
        assert_eq!(t, "hi there ");
    }

    #[test]
    fn detects_stop_split_across_fragments() {
        let mut m = StopMatcher::new(vec!["STOPHERE".into()]);
        let (t1, s1) = m.push("we are STOP");
        assert!(!s1);
        let (t2, s2) = m.push("HERE more");
        assert!(s2);
        let combined = format!("{t1}{t2}");
        assert_eq!(combined, "we are ");
    }

    #[test]
    fn longest_stop_dominates_window() {
        let mut m = StopMatcher::new(vec!["short".into(), "longerthing".into()]);
        let (t1, _s1) = m.push("ab");
        let (t2, s2) = m.push("c shortz");
        // "short" was matched.
        assert!(s2);
        let combined = format!("{t1}{t2}");
        assert_eq!(combined, "abc ");
    }

    #[test]
    fn no_stop_emits_safe_prefix_holds_ambiguous_tail() {
        let mut m = StopMatcher::new(vec!["BANG".into()]);
        // "hello BA" -> "hello " emitted, "BA" held.
        let (t, s) = m.push("hello BA");
        assert!(!s);
        assert_eq!(t, "hello ");
        // Next push that does NOT complete -> emit the held bytes.
        let (t2, s2) = m.push("X done");
        assert!(!s2);
        assert_eq!(t2, "BAX done");
    }

    #[test]
    fn finish_drains_held_bytes() {
        let mut m = StopMatcher::new(vec!["STOP".into()]);
        let (t, _) = m.push("data ST");
        assert_eq!(t, "data ");
        let residual = m.finish();
        assert_eq!(residual, "ST");
    }

    #[test]
    fn longest_suffix_prefix_helper() {
        assert_eq!(longest_suffix_prefix(b"hello ST", b"STOP"), 2);
        assert_eq!(longest_suffix_prefix(b"helloS", b"STOP"), 1);
        assert_eq!(longest_suffix_prefix(b"hello", b"STOP"), 0);
        assert_eq!(longest_suffix_prefix(b"xxxSTO", b"STOP"), 3);
    }

    /// A multi-byte (3-byte) Unicode stop sequence is matched on its byte
    /// boundary and the preceding multi-byte content round-trips intact. The
    /// `€` (E2 82 AC) before the stop spans the emit, and the stop itself is a
    /// multi-byte string — proving the byte-level windowing + `from_utf8_lossy`
    /// decode never corrupt a real Unicode prefix into U+FFFD.
    #[test]
    fn multibyte_unicode_stop_and_prefix_roundtrip() {
        let mut m = StopMatcher::new(vec!["。".into()]); // U+3002, 3 bytes
        let (t, stop) = m.push("価格は€です。あと");
        assert!(stop, "the multi-byte stop must be detected");
        assert_eq!(
            t, "価格は€です",
            "the multi-byte prefix (incl. €) is emitted intact up-to-but-excluding the stop"
        );
    }

    /// A multi-byte char held in the window (because its leading bytes look
    /// like the start of a longer stop sequence) is reassembled and flushed
    /// without corruption when the next push fails to complete the stop.
    #[test]
    fn multibyte_char_in_window_reassembles() {
        // Stop is "€END" (€ = E2 82 AC). Pushing "a€" holds the full € (its
        // 3 bytes are a prefix of the stop); "more" then fails to complete it.
        let mut m = StopMatcher::new(vec!["€END".into()]);
        let (t1, s1) = m.push("a€");
        assert!(!s1);
        assert_eq!(t1, "a", "the € is held back as a possible stop prefix");
        let (t2, s2) = m.push("more");
        assert!(!s2);
        assert_eq!(t2, "€more", "the held € is flushed intact, not corrupted");
    }

    /// `finish()` flushing a held multi-byte char yields it intact.
    #[test]
    fn finish_drains_multibyte_char_intact() {
        let mut m = StopMatcher::new(vec!["€END".into()]);
        let (t, _) = m.push("x€");
        assert_eq!(t, "x");
        assert_eq!(m.finish(), "€", "held multi-byte tail flushed intact on finish()");
    }
}
