pub(crate) mod properties;
pub(crate) mod transitions;

use crate::uax29::Action;
use crate::uax29::swar::{SWAR_HIGH, WORD_BYTES, load_chunk, swar_in_range};
use properties::{ASCII_SENTENCE_BREAK_PROP, lookup_sentence_break_property};
use transitions::{State, TRANSITION_TABLE, Transition};

#[derive(Default)]
#[non_exhaustive]
pub struct Options {}

pub fn tokenize(text: &str, _options: Options, mut on_breakpoint: impl FnMut(usize) -> bool) {
    if text.is_empty() {
        return;
    }
    let bytes = text.as_bytes();
    let mut state = State::StartOfText;
    let mut deferred_break_pos = None;
    let mut pos = 0;
    while pos < text.len() {
        // Fast path: inside a sentence (states Any/Upper/Lower), every ASCII byte except the
        // sentence-relevant ones (`. ! ? \n \r`) is a guaranteed NoBreak that lands back in one of
        // these same three states (letters keep Upper/Lower for SB7; everything else → Any). So we
        // can skip the per-byte property+transition table loads and scan a machine word at a time,
        // stopping at the first byte that the DFA must actually inspect. Sentences are long, so
        // this run covers the overwhelming majority of the input.
        if matches!(state, State::Any | State::Upper | State::Lower) {
            let end = scan_sentence_interior(bytes, pos);
            if end > pos {
                // Resulting state is determined solely by the last consumed byte (the DFA is
                // Markovian within this run): an ASCII letter sets Upper/Lower (needed so a
                // following ATerm enters LetterATerm for SB7), anything else resets to Any.
                let last = bytes[end - 1];
                state = if last.is_ascii_uppercase() {
                    State::Upper
                } else if last.is_ascii_lowercase() {
                    State::Lower
                } else {
                    State::Any
                };
                pos = end;
                continue;
            }
        }

        let b = bytes[pos];
        let (prop, char_len) = if b < 0x80 {
            (ASCII_SENTENCE_BREAK_PROP[b as usize], 1usize)
        } else {
            let c = text[pos..].chars().next().unwrap();
            (lookup_sentence_break_property(c), c.len_utf8())
        };
        let Transition(next_state, action) = TRANSITION_TABLE[state as usize][prop as usize];
        match action {
            Action::Break => {
                state = next_state;
                if !on_breakpoint(pos) {
                    return;
                }
                pos += char_len;
                continue;
            }
            Action::NoBreak => {
                if next_state.is_deferred() {
                    if deferred_break_pos.is_none() {
                        deferred_break_pos = Some(pos);
                    }
                } else {
                    deferred_break_pos = None;
                }
                state = next_state;
                pos += char_len;
            }
            Action::Transparent => {
                // State doesn't change, but we still consume the character.
                pos += char_len;
            }
            Action::DeferredBreak => {
                let boundary = deferred_break_pos.take().unwrap();
                state = next_state;
                // Don't advance pos — re-examine current char in new state.
                if !on_breakpoint(boundary) {
                    return;
                }
                continue;
            }
        }
    }

    // Deferred state at EOT — defer failed, confirm break
    if state.is_deferred() {
        if !on_breakpoint(deferred_break_pos.unwrap()) {
            return;
        }
    }

    // SB2: Any	÷ eot (break at end of text)
    _ = on_breakpoint(text.len());
}

/// Scans the "sentence interior" run starting at `start`, returning the index of the first byte
/// that the sentence DFA must inspect — i.e. the first byte that is *not* a plain ASCII interior
/// byte. A byte stops the scan iff it is non-ASCII (>= 0x80) or one of the sentence-relevant ASCII
/// bytes: `\n` `\r` `!` `.` `?` (ParaSep / STerm / ATerm).
///
/// Only valid to call from states `Any`/`Upper`/`Lower`, where every other ASCII byte is a NoBreak
/// that stays within those three states (see `tokenize` and the `Any`/`Upper`/`Lower` rows in
/// `transitions.rs` — if those rows change, revisit `is_sentence_stop_byte`). We may stop *early*
/// without affecting correctness (the DFA just resumes), so the rare control bytes `\x0B`/`\x0C`
/// are folded into the `\n..=\r` test for free.
#[inline]
fn scan_sentence_interior(bytes: &[u8], start: usize) -> usize {
    let mut pos = start;

    // SWAR fast lane: classify `WORD_BYTES` bytes per iteration with no per-byte table loads. This
    // mirrors `is_sentence_stop_byte`, evaluated a machine word at a time.
    while pos + WORD_BYTES <= bytes.len() {
        // SAFETY: the `while` condition guarantees `pos + WORD_BYTES <= bytes.len()`.
        let chunk = unsafe { load_chunk(bytes, pos) };
        // High bit set in every lane the DFA must inspect: non-ASCII, or a sentence-relevant byte.
        let stop = (chunk & SWAR_HIGH)            // >= 0x80
            | swar_in_range(chunk, 0x0A, 0x0D)    // \n \r (and rare \x0B \x0C, harmlessly)
            | swar_in_range(chunk, b'!', b'!')    // STerm
            | swar_in_range(chunk, b'.', b'.')    // ATerm
            | swar_in_range(chunk, b'?', b'?'); // STerm
        if stop == 0 {
            pos += WORD_BYTES;
        } else {
            // First stop byte is at this lane offset within the chunk.
            return pos + (stop.trailing_zeros() / 8) as usize;
        }
    }

    // Scalar tail (and small inputs): same classification, one byte at a time.
    while pos < bytes.len() && !is_sentence_stop_byte(bytes[pos]) {
        pos += 1;
    }
    pos
}

/// A byte the sentence DFA must inspect from an interior state: non-ASCII (>= 0x80) or one of the
/// sentence-relevant ASCII bytes `\n` `\r` `!` `.` `?` (ParaSep / STerm / ATerm). The rare control
/// bytes `\x0B`/`\x0C` are included for free (stopping early is harmless). Kept in lockstep with
/// the SWAR classification in `scan_sentence_interior`.
#[inline(always)]
fn is_sentence_stop_byte(b: u8) -> bool {
    b >= 0x80 || matches!(b, b'\n' | 0x0B | 0x0C | b'\r' | b'!' | b'.' | b'?')
}

#[cfg(test)]
mod tests {
    use super::{Options, is_sentence_stop_byte, scan_sentence_interior, tokenize};
    use crate::uax29::test_helpers::test_against_uax29_break_tests;

    /// Trivially-correct reference: byte-at-a-time scan that stops at the first byte the sentence
    /// DFA must inspect.
    fn scan_reference(bytes: &[u8], start: usize) -> usize {
        let mut pos = start;
        while pos < bytes.len() && !is_sentence_stop_byte(bytes[pos]) {
            pos += 1;
        }
        pos
    }

    #[test]
    fn scan_sentence_interior_matches_reference() {
        // Exhaustive: every single byte value, at every alignment offset within a chunk, with an
        // interior-byte prefix so the SWAR lane containing the byte varies.
        for prefix in 0..=16usize {
            for b in 0..=255u8 {
                let mut buf = vec![b'a'; prefix];
                buf.push(b);
                buf.extend_from_slice(b"bc de.");
                assert_eq!(
                    scan_sentence_interior(&buf, 0),
                    scan_reference(&buf, 0),
                    "prefix={prefix} byte={b:#04x}"
                );
            }
        }

        // Randomized multi-byte inputs across the full byte range (biased toward interior bytes so
        // we exercise long runs and boundaries at every offset).
        let mut state: u64 = 0x243F6A8885A308D3;
        let mut rng = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..200_000 {
            let len = (rng() % 40) as usize;
            let buf: Vec<u8> = (0..len)
                .map(|_| {
                    let r = rng();
                    if r % 4 == 0 {
                        (r >> 8) as u8 // full range, includes non-ASCII & stop bytes
                    } else {
                        let alphabet = b"abcXYZ0189 ,.!?\n";
                        alphabet[(r >> 8) as usize % alphabet.len()]
                    }
                })
                .collect();
            let start = if len == 0 {
                0
            } else {
                (rng() as usize) % (len + 1)
            };
            assert_eq!(
                scan_sentence_interior(&buf, start),
                scan_reference(&buf, start),
                "buf={buf:?} start={start}"
            );
        }
    }

    #[test]
    fn test_sentence_break_against_uax29_tests() {
        let (passed, failed) =
            test_against_uax29_break_tests("testdata/SentenceBreakTest.txt", |s, breakpoints| {
                tokenize(s, Options::default(), |bp| {
                    breakpoints.push(bp);
                    true
                });
            });
        assert_eq!(
            (512, 0),
            (passed, failed),
            "{} / {} tests passed",
            passed,
            passed + failed
        );
    }

    #[test]
    fn tokenizer_sanity() {
        fn assert_breaks(s: &str, expected: Vec<usize>) {
            let mut breakpoints = Vec::new();
            tokenize(s, Options::default(), |bp| {
                breakpoints.push(bp);
                true
            });
            assert_eq!(breakpoints, expected, "input: {:?}", s);
        }

        // Empty string yields no breakpoints.
        assert_breaks("", vec![]);

        // Non-empty strings break at the start & end.
        assert_breaks("a", vec![0, 1]);
        assert_breaks(".", vec![0, 1]);

        // SB998: don't break within a sentence.
        assert_breaks("Hello world", vec![0, 11]);

        // SB3: CR × LF (don't break between CR and LF)
        assert_breaks("\r\n", vec![0, 2]);

        // SB4: Break after paragraph separators (Sep, CR, LF).
        assert_breaks("a\nb", vec![0, 2, 3]);
        assert_breaks("a\r\nb", vec![0, 3, 4]);
        assert_breaks("a\rb", vec![0, 2, 3]);

        // SB5: Extend and Format are transparent.
        assert_breaks("a\u{0308}b", vec![0, 4]); // a + combining diaeresis + b

        // SB6: ATerm × Numeric — don't break between "." and a digit.
        assert_breaks("3.4", vec![0, 3]);

        // SB7: (Upper | Lower) ATerm × Upper — abbreviations like U.S.A.
        assert_breaks("U.S.A.", vec![0, 6]);
        assert_breaks("U.S.", vec![0, 4]);
        assert_breaks("c.D", vec![0, 3]);

        // SB8: ATerm Close* Sp* × (¬(OLetter|Upper|Lower|ParaSep|SATerm))* Lower
        // Don't break after "." when eventually followed by a lowercase letter.
        assert_breaks("c.d", vec![0, 3]);
        assert_breaks("etc. the", vec![0, 8]);
        assert_breaks("the resp. leaders are", vec![0, 21]);

        // SB8: with Close and Sp between ATerm and Lower.
        assert_breaks("etc.)'\u{a0}the", vec![0, 11]);

        // SB8a: SATerm Close* Sp* × (SContinue | SATerm)
        // Don't break before continuation punctuation after sentence terminators.
        assert_breaks(".,", vec![0, 2]); // ATerm × SContinue
        assert_breaks("..", vec![0, 2]); // ATerm × ATerm
        assert_breaks("!,", vec![0, 2]); // STerm × SContinue
        assert_breaks("!.", vec![0, 2]); // STerm × ATerm

        // SB9/SB10/SB11: Break after sentence terminators,
        // but include trailing Close, Sp, and ParaSep in the sentence.
        assert_breaks("Hello. World", vec![0, 7, 12]);
        assert_breaks("Hello!) World", vec![0, 8, 13]);
        assert_breaks("Hello.  World", vec![0, 8, 13]);
        assert_breaks("Hello.\nWorld", vec![0, 7, 12]);

        // SB11: STerm breaks even when followed by lowercase.
        assert_breaks("Hello! world", vec![0, 7, 12]);

        // SB8 vs SB11: ATerm followed by OLetter or Upper DOES break (SB8 fails).
        assert_breaks("Hello. World", vec![0, 7, 12]);

        // Figures 3 & 4 from the spec:
        // Figure 3: Forbidden breaks on "." (should NOT break)
        assert_breaks("c.d", vec![0, 3]);
        assert_breaks("3.4", vec![0, 3]);
        assert_breaks("U.S.", vec![0, 4]);
        assert_breaks("the resp. leaders are", vec![0, 21]);
        assert_breaks("etc.)\u{2019}\u{a0}\u{2018}(the", vec![0, 17]);

        // Figure 4: Allowed breaks on "." (SHOULD break)
        assert_breaks(
            "She said \"See spot run.\" John shook his head.",
            vec![0, 25, 45],
        );
    }
}
