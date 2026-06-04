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
            // trailing data are dropped per stop semantics.
            let safe = String::from_utf8(combined[..match_start].to_vec())
                .unwrap_or_default();
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
        let safe = String::from_utf8(combined[..safe_end].to_vec()).unwrap_or_default();
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
        String::from_utf8(self.window).unwrap_or_default()
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
}
